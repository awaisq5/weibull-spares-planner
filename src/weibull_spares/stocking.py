"""Turning a failure forecast into stocking and replacement decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize

from .forecast import LeadTimeDemand
from .weibull import Weibull


@dataclass(frozen=True)
class StockOption:
    stock: int
    #: Probability that every failure during the lead time finds a spare.
    service_level: float
    #: Failures expected to find the shelf empty.
    expected_shortage: float
    holding_cost: float
    shortage_cost: float

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.shortage_cost


def stock_options(
    demand: LeadTimeDemand,
    unit_cost: float,
    holding_rate: float,
    downtime_cost: float,
    max_stock: int | None = None,
) -> list[StockOption]:
    """Cost and service of every stock level from zero up to a level that covers ~all demand.

    ``holding_rate`` is the carrying cost per lead-time period as a fraction of
    unit cost; ``downtime_cost`` is the cost of one failure waiting for a part.
    """
    if max_stock is None:
        max_stock = int(np.searchsorted(np.cumsum(demand.pmf), 0.9999)) + 1
    options = []
    for s in range(max_stock + 1):
        shortage = demand.expected_shortage(s)
        options.append(
            StockOption(
                stock=s,
                service_level=demand.cdf(s),
                expected_shortage=shortage,
                holding_cost=s * unit_cost * holding_rate,
                shortage_cost=shortage * downtime_cost,
            )
        )
    return options


def stock_for_service_level(demand: LeadTimeDemand, target: float) -> int:
    """Smallest stock with at least ``target`` probability of covering lead-time failures."""
    if not 0 < target < 1:
        raise ValueError("target service level must be between 0 and 1")
    cdf = np.cumsum(demand.pmf)
    return int(np.searchsorted(cdf, target - 1e-12))


def cost_optimal_stock(options: list[StockOption]) -> StockOption:
    return min(options, key=lambda o: o.total_cost)


@dataclass(frozen=True)
class ReplacementPolicy:
    #: Planned replacement age; ``None`` means run to failure is cheapest.
    age: float | None
    cost_per_hour: float
    run_to_failure_cost_per_hour: float

    @property
    def saving_share(self) -> float:
        return 1 - self.cost_per_hour / self.run_to_failure_cost_per_hour


def optimal_replacement_age(model: Weibull, preventive_cost: float, failure_cost: float) -> ReplacementPolicy:
    """Age replacement policy: replace at age T or at failure, whichever comes first.

    Long-run cost per operating hour::

        C(T) = (Cp * R(T) + Cf * F(T)) / integral_0^T R(t) dt

    Preventive replacement only pays when the failure rate rises with age
    (beta > 1) and a planned replacement is cheaper than a failure.
    """
    run_to_failure = failure_cost / model.mean_life()
    if model.beta <= 1 or preventive_cost >= failure_cost:
        return ReplacementPolicy(None, run_to_failure, run_to_failure)

    def cost_rate(age: float) -> float:
        expected_cycle_length, _ = integrate.quad(lambda t: float(model.reliability(t)), 0, age, limit=200)
        r = float(model.reliability(age))
        return (preventive_cost * r + failure_cost * (1 - r)) / expected_cycle_length

    upper = model.eta * 4
    res = optimize.minimize_scalar(cost_rate, bounds=(model.eta * 1e-3, upper), method="bounded")
    if res.fun >= run_to_failure * 0.999 or res.x > upper * 0.99:
        return ReplacementPolicy(None, run_to_failure, run_to_failure)
    return ReplacementPolicy(float(res.x), float(res.fun), run_to_failure)
