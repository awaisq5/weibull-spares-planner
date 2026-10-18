"""Weibull model and fitting from field data with suspensions.

Field data is almost always censored: most units have not failed yet, they have
only been running for a while ("suspensions"). Ignoring them biases life
estimates downward, so both fitting methods here use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, special, stats

Method = Literal["mle", "rank_regression"]


@dataclass(frozen=True)
class Weibull:
    """Weibull life distribution.

    ``beta`` is the **shape**: below 1 the failure rate falls with age, around 1
    it is constant, above 1 parts wear out. ``eta`` is the **scale** or
    characteristic life: the age by which 63.2% of parts have failed.
    MATLAB's ``wblfit`` returns the same two numbers in the order [scale, shape].
    """

    beta: float
    eta: float

    def __post_init__(self) -> None:
        if not (self.beta > 0 and self.eta > 0):
            raise ValueError("beta and eta must be positive")

    def cdf(self, t: ArrayLike) -> np.ndarray:
        """Probability of failure by age ``t``."""
        t = np.maximum(np.asarray(t, dtype=float), 0.0)
        return -np.expm1(-((t / self.eta) ** self.beta))

    def reliability(self, t: ArrayLike) -> np.ndarray:
        """Probability of surviving to age ``t``."""
        t = np.maximum(np.asarray(t, dtype=float), 0.0)
        return np.exp(-((t / self.eta) ** self.beta))

    def pdf(self, t: ArrayLike) -> np.ndarray:
        t = np.maximum(np.asarray(t, dtype=float), 0.0)
        return (self.beta / self.eta) * (t / self.eta) ** (self.beta - 1) * self.reliability(t)

    def hazard(self, t: ArrayLike) -> np.ndarray:
        """Instantaneous failure rate at age ``t``."""
        t = np.maximum(np.asarray(t, dtype=float), 0.0)
        return (self.beta / self.eta) * (t / self.eta) ** (self.beta - 1)

    def b_life(self, fraction: float = 0.10) -> float:
        """Age by which ``fraction`` of units have failed (B10 life for 0.10)."""
        if not 0 < fraction < 1:
            raise ValueError("fraction must be between 0 and 1")
        return float(self.eta * (-np.log1p(-fraction)) ** (1 / self.beta))

    def mean_life(self) -> float:
        """Mean time to failure."""
        return float(self.eta * special.gamma(1 + 1 / self.beta))

    def conditional_failure(self, age: ArrayLike, horizon: float) -> np.ndarray:
        """Probability that a unit which has survived to ``age`` fails within the next ``horizon``."""
        age = np.asarray(age, dtype=float)
        # Ratio of survival functions, computed in log space for old units.
        log_r = -(((age + horizon) / self.eta) ** self.beta) + (age / self.eta) ** self.beta
        return -np.expm1(log_r)

    def failure_mode(self) -> str:
        """Plain-language reading of the shape parameter."""
        if self.beta < 0.95:
            return "early-life failures: the failure rate falls with age (infant mortality)"
        if self.beta <= 1.05:
            return "random failures: the failure rate does not depend on age"
        if self.beta < 3.5:
            return "wear-out: the failure rate rises with age"
        return "rapid wear-out: failures cluster tightly around one age"


@dataclass(frozen=True)
class Fit:
    model: Weibull
    method: Method
    failures: int
    suspensions: int
    log_likelihood: float
    #: Two-sided confidence bounds (95% by default), keyed by "beta" and "eta". Only for MLE.
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Points for a Weibull probability plot: time and median-rank unreliability per failure.
    plot_points: tuple[np.ndarray, np.ndarray] | None = None


def _validate(times: ArrayLike, failed: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    d = np.asarray(failed, dtype=bool)
    if t.shape != d.shape or t.ndim != 1:
        raise ValueError("times and failed must be one-dimensional and the same length")
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("every time must be a positive number")
    if d.sum() < 2:
        raise ValueError("at least two failures are needed to fit a Weibull distribution")
    return t, d


def log_likelihood(model: Weibull, times: np.ndarray, failed: np.ndarray) -> float:
    z = times / model.eta
    ll_fail = np.log(model.beta / model.eta) + (model.beta - 1) * np.log(z[failed])
    return float(ll_fail.sum() - (z**model.beta).sum())


def fit_mle(times: ArrayLike, failed: ArrayLike, confidence: float = 0.95) -> Fit:
    """Maximum likelihood fit with right-censored data.

    Uses the profile likelihood: for a given beta the best eta has a closed form,
    so only one equation in beta needs solving. Times are scaled by their maximum
    so ``t ** beta`` cannot overflow.
    """
    t, d = _validate(times, failed)
    r = d.sum()
    scale = t.max()
    u = t / scale
    log_u = np.log(u)
    mean_log_fail = log_u[d].mean()

    def profile(beta: float) -> float:
        w = u**beta
        return (w * log_u).sum() / w.sum() - 1 / beta - mean_log_fail

    beta = optimize.brentq(profile, 1e-3, 100.0, xtol=1e-12)
    eta = float(scale * ((u**beta).sum() / r) ** (1 / beta))
    model = Weibull(beta, eta)

    bounds = _fisher_bounds(model, t, d, confidence)
    return Fit(
        model=model,
        method="mle",
        failures=int(r),
        suspensions=int((~d).sum()),
        log_likelihood=log_likelihood(model, t, d),
        bounds=bounds,
        plot_points=median_ranks(t, d),
    )


def _fisher_bounds(model: Weibull, t: np.ndarray, d: np.ndarray, confidence: float) -> dict[str, tuple[float, float]]:
    """Fisher-matrix bounds, computed on the log scale so they stay positive."""

    def nll(params: np.ndarray) -> float:
        return -log_likelihood(Weibull(float(np.exp(params[0])), float(np.exp(params[1]))), t, d)

    x0 = np.log([model.beta, model.eta])
    h = 1e-4
    hess = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i] * h
            ej = np.eye(2)[j] * h
            hess[i, j] = (nll(x0 + ei + ej) - nll(x0 + ei - ej) - nll(x0 - ei + ej) + nll(x0 - ei - ej)) / (4 * h * h)
    cov = np.linalg.inv(hess)
    z = stats.norm.ppf(0.5 + confidence / 2)
    se = np.sqrt(np.diag(cov))
    return {
        "beta": (float(np.exp(x0[0] - z * se[0])), float(np.exp(x0[0] + z * se[0]))),
        "eta": (float(np.exp(x0[1] - z * se[1])), float(np.exp(x0[1] + z * se[1]))),
    }


def median_ranks(times: ArrayLike, failed: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Failure times and their median-rank unreliability.

    Suspensions shift the ranks of later failures (Johnson's adjusted ranks);
    Bernard's approximation turns adjusted ranks into median ranks.
    """
    t = np.asarray(times, dtype=float)
    d = np.asarray(failed, dtype=bool)
    order = np.lexsort((~d, t))  # by time; at ties, failures before suspensions
    t, d = t[order], d[order]
    n = len(t)
    adjusted = 0.0
    fail_times, ranks = [], []
    for position, (time, is_failure) in enumerate(zip(t, d, strict=True), start=1):
        if not is_failure:
            continue
        reverse_rank = n - position + 1
        adjusted += (n + 1 - adjusted) / (1 + reverse_rank)
        fail_times.append(time)
        ranks.append(adjusted)
    unreliability = (np.array(ranks) - 0.3) / (n + 0.4)
    return np.array(fail_times), unreliability


def fit_rank_regression(times: ArrayLike, failed: ArrayLike) -> Fit:
    """Median-rank regression of X on Y, the method behind a classic Weibull plot."""
    t, d = _validate(times, failed)
    ft, f = median_ranks(t, d)
    x = np.log(ft)
    y = np.log(-np.log1p(-f))
    slope, intercept = np.polyfit(y, x, 1)  # x = intercept + slope * y
    model = Weibull(1 / slope, float(np.exp(intercept)))
    return Fit(
        model=model,
        method="rank_regression",
        failures=int(d.sum()),
        suspensions=int((~d).sum()),
        log_likelihood=log_likelihood(model, t, d),
        plot_points=(ft, f),
    )


def fit(times: ArrayLike, failed: ArrayLike, method: Method = "mle") -> Fit:
    if method == "mle":
        return fit_mle(times, failed)
    if method == "rank_regression":
        return fit_rank_regression(times, failed)
    raise ValueError(f"unknown method {method!r}")


@dataclass(frozen=True)
class ThreeParameterFit:
    """Weibull with a failure-free period: no part fails before ``gamma`` hours.

    ``fit`` describes the hours *after* ``gamma``, so a part's probability of
    failing by age ``t`` is ``fit.model.cdf(t - gamma)``.
    """

    gamma: float
    fit: Fit
    two_parameter: Fit
    #: 2 · (three-parameter − two-parameter log-likelihood)
    lr_statistic: float
    #: Against "no failure-free period". gamma = 0 lies on the boundary, so the
    #: statistic follows a 50:50 mixture of 0 and chi-square(1), which halves the p-value.
    p_value: float
    #: False when every gamma above 0 needs a shape below 1, the degenerate case
    #: where the likelihood has no maximum. gamma is then 0.
    supported: bool

    @property
    def significant(self) -> bool:
        return self.supported and self.p_value < 0.05

    def cdf(self, t: ArrayLike) -> np.ndarray:
        return self.fit.model.cdf(np.asarray(t, dtype=float) - self.gamma)

    def b_life(self, fraction: float = 0.10) -> float:
        return self.gamma + self.fit.model.b_life(fraction)


def _shifted_fit(t: np.ndarray, d: np.ndarray, gamma: float) -> Fit:
    """Two-parameter fit of the hours after ``gamma``.

    Suspensions at or before ``gamma`` survived the failure-free period, which
    has probability 1, so they add nothing to the likelihood and are dropped.
    """
    keep = t > gamma
    return fit_mle(t[keep] - gamma, d[keep])


def fit_mle_3p(times: ArrayLike, failed: ArrayLike, grid: int = 60) -> ThreeParameterFit:
    """Maximum likelihood with a failure-free period ``gamma``.

    For a fixed gamma the best shape and scale come from the two-parameter fit
    of the shifted times, so the likelihood is profiled over gamma alone, on
    [0, first failure). A coarse grid finds the region and a bounded search
    refines it, since the profile can have more than one bump.

    With a shape below 1 the density is infinite at gamma, so the likelihood
    can be made arbitrarily large by moving gamma up to the first failure. That
    is not evidence of a failure-free period, so only gamma > 0 with a shape of
    at least 1 is considered.
    """
    t, d = _validate(times, failed)
    two = fit_mle(t, d)
    first_failure = float(t[d].min())
    upper = first_failure * (1 - 1e-6)

    def nll(gamma: float) -> float:
        f = _shifted_fit(t, d, gamma)
        if gamma > 0 and f.model.beta < 1:
            return np.inf
        return -f.log_likelihood

    gammas = np.linspace(0.0, upper, grid)
    values = np.array([nll(g) for g in gammas])
    supported = bool(np.isfinite(values[1:]).any())
    i = int(np.argmin(values))
    gamma = float(gammas[i])
    if 0 < i:
        lo, hi = gammas[i - 1], gammas[min(i + 1, grid - 1)]
        best = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": first_failure * 1e-6})
        if np.isfinite(best.fun) and best.fun <= values[i]:
            gamma = float(best.x)
    three = _shifted_fit(t, d, gamma) if gamma > 0 else two
    lr = max(0.0, 2 * (three.log_likelihood - two.log_likelihood))
    p = 1.0 if lr == 0 else 0.5 * float(stats.chi2.sf(lr, 1))
    return ThreeParameterFit(gamma, three, two, lr, p, supported)
