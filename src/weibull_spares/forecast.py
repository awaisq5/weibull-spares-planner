"""Failure forecasts for a fleet of units with different ages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .weibull import Weibull


def poisson_binomial_pmf(p: ArrayLike) -> np.ndarray:
    """Distribution of the number of failures when unit ``i`` fails with probability ``p[i]``.

    Units of different ages have different failure probabilities, so the count is
    Poisson-binomial, not binomial. Computed exactly by dynamic programming.
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for k, pk in enumerate(p, start=1):
        pmf[1 : k + 1] = pmf[1 : k + 1] * (1 - pk) + pmf[0:k] * pk
        pmf[0] *= 1 - pk
    return pmf


@dataclass(frozen=True)
class LeadTimeDemand:
    """Failures expected while a replacement part is on its way."""

    horizon_hours: float
    probabilities: np.ndarray
    pmf: np.ndarray

    @property
    def expected(self) -> float:
        return float(self.probabilities.sum())

    def cdf(self, k: int) -> float:
        return float(self.pmf[: k + 1].sum())

    def expected_shortage(self, stock: int) -> float:
        """Expected number of failures that find no spare on the shelf."""
        k = np.arange(len(self.pmf))
        return float((np.maximum(k - stock, 0) * self.pmf).sum())


def lead_time_demand(model: Weibull, ages: ArrayLike, horizon_hours: float) -> LeadTimeDemand:
    """Failures among installed units within ``horizon_hours`` of operation.

    Assumes a failed unit's replacement does not itself fail inside the same
    horizon, which holds when the horizon is short against the B10 life.
    ``renewal_share`` quantifies the assumption.
    """
    p = model.conditional_failure(np.asarray(ages, dtype=float), horizon_hours)
    return LeadTimeDemand(horizon_hours, p, poisson_binomial_pmf(p))


def renewal_share(model: Weibull, horizon_hours: float) -> float:
    """Probability that a brand-new replacement fails within the horizon.

    If this is small, ignoring second failures in the lead-time window is safe.
    """
    return float(model.cdf(horizon_hours))


@dataclass(frozen=True)
class PeriodForecast:
    period_hours: float
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray


def simulate_failures(
    model: Weibull,
    ages: ArrayLike,
    period_hours: float,
    periods: int,
    runs: int = 4000,
    seed: int = 7,
    band: float = 0.80,
) -> PeriodForecast:
    """Failures per period over a longer horizon, by Monte Carlo with renewal.

    Every failed unit is replaced by a new one that can fail again, which the
    closed-form lead-time calculation deliberately ignores.
    """
    rng = np.random.default_rng(seed)
    ages = np.asarray(ages, dtype=float)
    horizon = period_hours * periods
    counts = np.zeros((runs, periods), dtype=np.int32)

    # First failure of the installed unit, sampled from its conditional life.
    u = rng.random((runs, len(ages)))
    current_age_term = (ages / model.eta) ** model.beta
    remaining = model.eta * (current_age_term - np.log(u)) ** (1 / model.beta) - ages
    clock = remaining
    while True:
        alive = clock < horizon
        if not alive.any():
            break
        idx = np.nonzero(alive)
        period = (clock[idx] // period_hours).astype(int)
        np.add.at(counts, (idx[0], period), 1)
        # Replacement starts new: draw a fresh life.
        fresh = model.eta * (-np.log(rng.random(len(idx[0])))) ** (1 / model.beta)
        clock[idx] = clock[idx] + fresh
        clock[~alive] = np.inf

    q = (1 - band) / 2
    return PeriodForecast(
        period_hours=period_hours,
        mean=counts.mean(axis=0),
        low=np.quantile(counts, q, axis=0),
        high=np.quantile(counts, 1 - q, axis=0),
    )
