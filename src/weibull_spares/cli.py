"""Command line: fit a life model, then plan spares for a fleet.

weibull-spares fit data/bearing_pack_history.csv
weibull-spares plan data/bearing_pack_history.csv data/fleet.csv --lead-time-days 21
"""

from __future__ import annotations

import argparse
import sys

from .data import DataError, load_fleet, load_history
from .forecast import lead_time_demand, renewal_share
from .stocking import cost_optimal_stock, optimal_replacement_age, stock_for_service_level, stock_options
from .weibull import Fit, fit


def _print_fit(f: Fit) -> None:
    m = f.model
    print(f"Fitted with {'maximum likelihood' if f.method == 'mle' else 'median-rank regression'}")
    print(f"  data             {f.failures} failures, {f.suspensions} suspensions")
    beta = f"{m.beta:.2f}"
    eta = f"{m.eta:,.0f} h"
    if f.bounds:
        b_lo, b_hi = f.bounds["beta"]
        e_lo, e_hi = f.bounds["eta"]
        beta += f"   (90% bounds {b_lo:.2f} to {b_hi:.2f})"
        eta += f"   (90% bounds {e_lo:,.0f} to {e_hi:,.0f} h)"
    print(f"  shape beta       {beta}")
    print(f"  scale eta        {eta}")
    print(f"  B10 life         {m.b_life(0.10):,.0f} h")
    print(f"  mean life        {m.mean_life():,.0f} h")
    print(f"  reading          {m.failure_mode()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="weibull-spares", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="fit a Weibull model to a life history")
    p_fit.add_argument("history", help="CSV with unit_id, hours, status")
    p_fit.add_argument("--method", choices=["mle", "rank_regression"], default="mle")

    p_plan = sub.add_parser("plan", help="forecast failures and size spare stock")
    p_plan.add_argument("history", help="CSV with unit_id, hours, status")
    p_plan.add_argument("fleet", help="CSV with unit_id, age_hours")
    p_plan.add_argument("--lead-time-days", type=float, required=True, help="days to get a replacement part")
    p_plan.add_argument("--hours-per-day", type=float, default=6, help="average operating hours per unit per day")
    p_plan.add_argument("--service-level", type=float, default=0.95)
    p_plan.add_argument("--unit-cost", type=float, default=3950)
    p_plan.add_argument("--holding-rate", type=float, default=0.25, help="annual carrying cost as share of unit cost")
    p_plan.add_argument("--downtime-cost", type=float, default=18000, help="cost of one failure waiting for a part")
    p_plan.add_argument("--preventive-cost", type=float, default=4600, help="cost of a planned replacement")
    p_plan.add_argument("--failure-cost", type=float, default=21000, help="cost of an in-service failure")

    args = parser.parse_args(argv)
    try:
        history = load_history(args.history)
        result = fit(history.hours, history.failed, args.method if args.command == "fit" else "mle")
        _print_fit(result)
        if args.command == "fit":
            return 0

        _, ages = load_fleet(args.fleet)
        model = result.model
        horizon = args.lead_time_days * args.hours_per_day
        demand = lead_time_demand(model, ages, horizon)
        holding = args.holding_rate * args.lead_time_days / 365
        options = stock_options(demand, args.unit_cost, holding, args.downtime_cost)
        target = stock_for_service_level(demand, args.service_level)
        best = cost_optimal_stock(options)
        policy = optimal_replacement_age(model, args.preventive_cost, args.failure_cost)

        print()
        print(f"Fleet of {len(ages)} units, lead time {args.lead_time_days:g} days ({horizon:,.0f} operating hours)")
        print(f"  expected failures in lead time   {demand.expected:.1f}")
        print(f"  stock for {args.service_level:.0%} service level      {target}")
        print(f"  cost-optimal stock               {best.stock}  ({best.service_level:.1%} service level)")
        if renewal_share(model, horizon) > 0.05:
            print("  note: lead time is long against unit life; replacements may fail again within it.")
        print()
        print("  stock  service  short/lead time  total cost")
        for o in options:
            if min(target, best.stock) - 3 <= o.stock <= max(target, best.stock) + 2:
                print(f"  {o.stock:>5}  {o.service_level:>7.1%}  {o.expected_shortage:>15.2f}  {o.total_cost:>10,.0f}")
        print()
        if policy.age is None:
            print("Preventive replacement does not pay; run to failure.")
        else:
            print(
                f"Replace at {policy.age:,.0f} h: {policy.cost_per_hour:.2f} per operating hour against "
                f"{policy.run_to_failure_cost_per_hour:.2f} run-to-failure ({policy.saving_share:.0%} lower)."
            )
        return 0
    except (DataError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
