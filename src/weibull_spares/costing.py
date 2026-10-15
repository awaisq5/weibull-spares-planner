"""Lifetime costing: what a part costs per circulating hour, and what the fleet will cost.

The life model feeds the cost model. Two numbers matter to a cost forecast:

* the **cost per circulating hour** of a part, which spreads its purchase and
  replacement cost over its expected life;
* the **expected scrap events** in the fleet over the coming hours, which turn
  into replacement spend.

Both depend on getting the life right, so the module also shows what the
forecast would have been with a naive model fitted to scrapped parts only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .forecast import simulate_failures
from .weibull import Weibull


@dataclass(frozen=True)
class CostInputs:
    #: Purchase price of a new part.
    unit_cost: float
    #: Labour, logistics and handling to swap a part, per event.
    replacement_cost: float = 0.0
    #: Value recovered when a part is scrapped (scrap metal, core credit).
    salvage_value: float = 0.0

    @property
    def cost_per_event(self) -> float:
        return self.unit_cost + self.replacement_cost - self.salvage_value


@dataclass(frozen=True)
class LifetimeCost:
    mean_life_hours: float
    cost_per_hour: float
    #: Expected scrap events among the installed parts over the horizon.
    expected_events: float
    expected_cost: float
    #: Events per 1,000 circulating hours of fleet operation.
    events_per_1000_hours: float


def lifetime_cost(
    model: Weibull, costs: CostInputs, fleet_ages: ArrayLike, horizon_hours: float, runs: int = 4000
) -> LifetimeCost:
    """Cost per hour of one part, and expected events and spend for the fleet over ``horizon_hours``.

    Expected events come from the Monte Carlo simulation with renewal, so a
    replacement part that also wears out within the horizon is counted too.
    """
    ages = np.asarray(fleet_ages, dtype=float)
    events = float(simulate_failures(model, ages, period_hours=horizon_hours, periods=1, runs=runs).mean[0])
    fleet_hours = len(ages) * horizon_hours
    return LifetimeCost(
        mean_life_hours=model.mean_life(),
        cost_per_hour=costs.cost_per_event / model.mean_life(),
        expected_events=events,
        expected_cost=events * costs.cost_per_event,
        events_per_1000_hours=1000 * events / fleet_hours if fleet_hours else 0.0,
    )


@dataclass(frozen=True)
class EventPlan:
    """Event-based maintenance: when parts will reach the end of their life."""

    #: Expected events in each period.
    per_period: np.ndarray
    period_hours: float
    #: Parts most likely to need replacing in the first period, by index.
    next_up: np.ndarray


def event_plan(model: Weibull, fleet_ages: ArrayLike, period_hours: float, periods: int, top: int = 10) -> EventPlan:
    """Expected end-of-life events per period for the installed parts (no renewal), and who is next."""
    ages = np.asarray(fleet_ages, dtype=float)
    edges = np.arange(periods + 1) * period_hours
    cum = np.array([model.conditional_failure(ages, e).sum() for e in edges])
    first_period = model.conditional_failure(ages, period_hours)
    return EventPlan(per_period=np.diff(cum), period_hours=period_hours, next_up=np.argsort(-first_period)[:top])
