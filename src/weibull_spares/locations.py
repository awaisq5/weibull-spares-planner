"""Do parts last differently at different locations?

The same part number can wear out faster where conditions are harsher. Fitting
one model per location shows the trend; a likelihood ratio test says whether
the difference is real or just noise from small samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .weibull import Fit, fit_mle


@dataclass(frozen=True)
class LocationFit:
    location: str
    fit: Fit | None
    parts: int
    failures: int


@dataclass(frozen=True)
class LocationComparison:
    pooled: Fit
    locations: list[LocationFit]
    #: 2 · (sum of separate log-likelihoods − pooled log-likelihood)
    lr_statistic: float
    degrees_of_freedom: int
    p_value: float

    @property
    def differs(self) -> bool:
        return self.p_value < 0.05


def compare_locations(
    hours: np.ndarray, failed: np.ndarray, location: np.ndarray, min_failures: int = 3
) -> LocationComparison:
    """Fit each location separately and test against one pooled model.

    Locations with fewer than ``min_failures`` failures are shown but left out
    of the test, since their fit would be mostly guesswork.
    """
    hours = np.asarray(hours, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    location = np.asarray(location)

    fits: list[LocationFit] = []
    tested = np.zeros(len(hours), dtype=bool)
    for loc in sorted(set(location.tolist())):
        mask = location == loc
        r = int(failed[mask].sum())
        f = fit_mle(hours[mask], failed[mask]) if r >= min_failures else None
        if f:
            tested |= mask
        fits.append(LocationFit(loc, f, int(mask.sum()), r))

    usable = [f for f in fits if f.fit]
    if len(usable) < 2:
        pooled = fit_mle(hours, failed)
        return LocationComparison(pooled, fits, 0.0, 0, 1.0)

    pooled = fit_mle(hours[tested], failed[tested])
    separate = sum(f.fit.log_likelihood for f in usable)  # type: ignore[union-attr]
    lr = max(0.0, 2 * (separate - pooled.log_likelihood))
    dof = 2 * (len(usable) - 1)
    return LocationComparison(pooled, fits, lr, dof, float(stats.chi2.sf(lr, dof)))
