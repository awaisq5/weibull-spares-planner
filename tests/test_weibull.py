import io

import numpy as np
import pytest
from scipy import stats

from weibull_spares import (
    Weibull,
    fit_mle,
    fit_rank_regression,
    lead_time_demand,
    median_ranks,
    optimal_replacement_age,
    poisson_binomial_pmf,
    simulate_failures,
    stock_for_service_level,
    stock_options,
)
from weibull_spares.data import DataError, load_history


def censored_sample(beta, eta, n, censor_at, seed=1):
    rng = np.random.default_rng(seed)
    life = eta * rng.weibull(beta, n)
    # Random removal ages mimic units still running at different ages.
    removal = rng.uniform(0.2, 1.0, n) * censor_at
    failed = life <= removal
    return np.where(failed, life, removal), failed


class TestModel:
    def test_b10_and_mean_match_closed_forms(self):
        m = Weibull(2.0, 1000.0)
        assert m.b_life(0.10) == pytest.approx(1000 * (-np.log(0.9)) ** 0.5)
        assert m.mean_life() == pytest.approx(1000 * np.sqrt(np.pi) / 2)

    def test_conditional_failure_matches_survival_ratio(self):
        m = Weibull(2.5, 800.0)
        expected = 1 - m.reliability(700) / m.reliability(500)
        assert m.conditional_failure(500, 200) == pytest.approx(expected)

    def test_exponential_case_is_memoryless(self):
        m = Weibull(1.0, 500.0)
        assert m.conditional_failure(0, 100) == pytest.approx(m.conditional_failure(10_000, 100))

    def test_describes_failure_mode(self):
        assert "wear-out" in Weibull(2.4, 1).failure_mode()
        assert "early-life" in Weibull(0.6, 1).failure_mode()


class TestFitting:
    def test_mle_agrees_with_scipy_on_censored_data(self):
        t, d = censored_sample(2.3, 1450, 300, censor_at=2500)
        ours = fit_mle(t, d).model
        data = stats.CensoredData(uncensored=t[d], right=t[~d])
        shape, _, scale = stats.weibull_min.fit(data, floc=0)
        assert ours.beta == pytest.approx(shape, rel=1e-4)
        assert ours.eta == pytest.approx(scale, rel=1e-4)

    def test_mle_recovers_true_parameters(self):
        t, d = censored_sample(2.3, 1450, 4000, censor_at=3000, seed=11)
        fit = fit_mle(t, d)
        assert fit.model.beta == pytest.approx(2.3, rel=0.06)
        assert fit.model.eta == pytest.approx(1450, rel=0.04)
        lo, hi = fit.bounds["beta"]
        assert lo < fit.model.beta < hi

    def test_ignoring_suspensions_underestimates_life(self):
        t, d = censored_sample(2.3, 1450, 400, censor_at=1800, seed=3)
        with_susp = fit_mle(t, d).model
        failures_only = fit_mle(t[d], np.ones(d.sum(), dtype=bool)).model
        assert failures_only.eta < with_susp.eta * 0.8

    def test_rank_regression_is_close_to_mle(self):
        t, d = censored_sample(2.0, 1000, 600, censor_at=1600, seed=5)
        rr = fit_rank_regression(t, d).model
        ml = fit_mle(t, d).model
        assert rr.beta == pytest.approx(ml.beta, rel=0.15)
        assert rr.eta == pytest.approx(ml.eta, rel=0.08)

    def test_median_ranks_without_suspensions_are_bernard(self):
        _, f = median_ranks([10, 20, 30, 40], [1, 1, 1, 1])
        assert f == pytest.approx([(i - 0.3) / 4.4 for i in (1, 2, 3, 4)])

    def test_suspensions_raise_later_ranks(self):
        # Textbook case: suspension between two failures adds a fractional rank.
        _, f = median_ranks([10, 15, 20], [1, 0, 1])
        adjusted_second = 1 + (3 + 1 - 1) / (1 + 1)
        assert f[1] == pytest.approx((adjusted_second - 0.3) / 3.4)

    def test_needs_two_failures(self):
        with pytest.raises(ValueError, match="two failures"):
            fit_mle([10, 20, 30], [1, 0, 0])


class TestForecast:
    def test_poisson_binomial_matches_binomial_for_equal_p(self):
        pmf = poisson_binomial_pmf([0.2] * 10)
        assert pmf == pytest.approx(stats.binom.pmf(np.arange(11), 10, 0.2))
        assert pmf.sum() == pytest.approx(1)

    def test_poisson_binomial_handles_certain_events(self):
        assert poisson_binomial_pmf([1.0, 0.0, 1.0]) == pytest.approx([0, 0, 1, 0])

    def test_older_units_drive_more_demand(self):
        m = Weibull(2.5, 1500)
        young = lead_time_demand(m, [100] * 50, 300).expected
        old = lead_time_demand(m, [1400] * 50, 300).expected
        assert old > 4 * young

    def test_simulation_agrees_with_exact_lead_time_demand(self):
        m = Weibull(2.5, 1500)
        ages = np.linspace(50, 1800, 120)
        exact = lead_time_demand(m, ages, 300).expected
        sim = simulate_failures(m, ages, period_hours=300, periods=1, runs=20_000).mean[0]
        assert sim == pytest.approx(exact, rel=0.04)


class TestStocking:
    def test_service_level_stock_is_the_first_level_that_meets_target(self):
        demand = lead_time_demand(Weibull(2.0, 1000), np.full(40, 600.0), 200)
        s = stock_for_service_level(demand, 0.95)
        assert demand.cdf(s) >= 0.95
        assert s == 0 or demand.cdf(s - 1) < 0.95

    def test_higher_downtime_cost_means_more_stock(self):
        demand = lead_time_demand(Weibull(2.0, 1000), np.full(40, 600.0), 200)
        cheap = min(stock_options(demand, 1000, 0.05, 500), key=lambda o: o.total_cost).stock
        costly = min(stock_options(demand, 1000, 0.05, 50_000), key=lambda o: o.total_cost).stock
        assert costly > cheap

    def test_preventive_replacement_pays_off_for_wear_out(self):
        policy = optimal_replacement_age(Weibull(3.0, 1000), preventive_cost=1_000, failure_cost=10_000)
        assert policy.age is not None
        assert 200 < policy.age < 1000
        assert policy.saving_share > 0.2

    def test_no_preventive_replacement_for_random_failures(self):
        assert optimal_replacement_age(Weibull(1.0, 1000), 1_000, 10_000).age is None


class TestData:
    def test_reads_status_words(self):
        h = load_history(io.StringIO("unit_id,hours,status\nA,100,failed\nB,200,Suspended\n"))
        assert h.failed.tolist() == [True, False]

    def test_points_at_the_bad_row(self):
        with pytest.raises(DataError, match="row 3"):
            load_history(io.StringIO("unit_id,hours,status\nA,100,failed\nB,-5,failed\n"))

    def test_names_missing_columns(self):
        with pytest.raises(DataError, match="status"):
            load_history(io.StringIO("unit_id,hours\nA,100\n"))
