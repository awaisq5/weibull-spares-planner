"""Command line for the lifetime analysis tool.

weibull-spares fit data/parts_register.csv
weibull-spares locations data/parts_register.csv
weibull-spares cost data/parts_register.csv --unit-cost 3950 --horizon 1000
weibull-spares spares data/parts_register.csv --lead-time-days 21

Add --json to any command for one JSON object instead of text.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from .costing import CostInputs, event_plan, lifetime_cost
from .data import DataError, PartsRegister, load_parts
from .forecast import lead_time_demand, renewal_share
from .locations import compare_locations
from .stocking import cost_optimal_stock, optimal_replacement_age, stock_for_service_level, stock_options
from .weibull import Fit, ThreeParameterFit, fit, fit_mle, fit_mle_3p


def _print_fit(f: Fit, label: str = "") -> None:
    m = f.model
    print(f"{label}Fitted with {'maximum likelihood' if f.method == 'mle' else 'median-rank regression'}")
    print(f"  data             {f.failures} worn out (failures), {f.suspensions} censored")
    beta = f"{m.beta:.2f}"
    eta = f"{m.eta:,.0f} CH"
    if f.bounds:
        b_lo, b_hi = f.bounds["beta"]
        e_lo, e_hi = f.bounds["eta"]
        beta += f"   (95% bounds {b_lo:.2f} to {b_hi:.2f})"
        eta += f"   (95% bounds {e_lo:,.0f} to {e_hi:,.0f} CH)"
    print(f"  shape beta       {beta}")
    print(f"  scale eta        {eta}")
    print(f"  B10 life         {m.b_life(0.10):,.0f} CH")
    print(f"  mean life        {m.mean_life():,.0f} CH")
    print(f"  reading          {m.failure_mode()}")


def _fit_dict(f: Fit) -> dict[str, Any]:
    m = f.model
    out: dict[str, Any] = {
        "method": f.method,
        "failures": f.failures,
        "suspensions": f.suspensions,
        "beta": m.beta,
        "eta_hours": m.eta,
        "b10_hours": m.b_life(0.10),
        "mean_life_hours": m.mean_life(),
        "failure_mode": m.failure_mode(),
        "log_likelihood": f.log_likelihood,
    }
    if f.bounds:
        out["bounds_95"] = {
            "beta": list(f.bounds["beta"]),
            "eta_hours": list(f.bounds["eta"]),
        }
    return out


def _three_dict(t: ThreeParameterFit) -> dict[str, Any]:
    return {
        "supported": t.supported,
        "gamma_hours": t.gamma,
        "beta": t.fit.model.beta,
        "eta_hours": t.fit.model.eta,
        "b10_hours": t.b_life(0.10),
        "lr_statistic": t.lr_statistic,
        "p_value": t.p_value,
        "significant": t.significant,
    }


def _print_three(t: ThreeParameterFit) -> None:
    print("\nFailure-free period (three-parameter Weibull)")
    if not t.supported:
        print("  not supported    the shape is below 1, so parts fail early rather than after a minimum age")
        return
    print(f"  gamma            {t.gamma:,.0f} CH, no part fails before this")
    print(f"  after gamma      shape beta {t.fit.model.beta:.2f}, scale eta {t.fit.model.eta:,.0f} CH")
    print(f"  B10 life         {t.b_life(0.10):,.0f} CH")
    print(f"  test             LR {t.lr_statistic:.1f}, p = {t.p_value:.3g} against no failure-free period")
    if t.significant:
        print("  reading          the failure-free period fits clearly better. Mixed locations or failure modes")
        print("                   can look like one too: check `locations` before relying on it.")
    else:
        print("  reading          no evidence of a failure-free period; keep the two-parameter model.")


def _emit(data: dict[str, Any]) -> None:
    # NumPy scalars are not JSON types; .item() turns them into plain Python numbers.
    print(json.dumps(data, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _naive(p: PartsRegister) -> Fit:
    """What you get if every scrapped part counts as a failure and parts in service are ignored."""
    scrapped = p.where(~p.in_service)
    return fit_mle(scrapped.hours, np.ones(len(scrapped.hours), dtype=bool))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weibull-spares", description="Lifetime analysis for parts with censored field data."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="fit a Weibull model to the parts register")
    p_fit.add_argument("parts", help="CSV with part_id, part_number, location, circulating_hours, scrap_reason")
    p_fit.add_argument("--method", choices=["mle", "rank_regression"], default="mle")
    p_fit.add_argument(
        "--failure-free", action="store_true", help="also fit a failure-free period (three-parameter Weibull)"
    )

    p_loc = sub.add_parser("locations", help="compare part life between locations")
    p_loc.add_argument("parts")

    p_cost = sub.add_parser("cost", help="lifetime cost and expected scrap events for the parts in service")
    p_cost.add_argument("parts")
    p_cost.add_argument("--unit-cost", type=float, required=True, help="price of a new part")
    p_cost.add_argument("--replacement-cost", type=float, default=0, help="labour and logistics per replacement")
    p_cost.add_argument("--salvage", type=float, default=0, help="value recovered per scrapped part")
    p_cost.add_argument("--horizon", type=float, default=1000, help="circulating hours to forecast")

    p_sp = sub.add_parser("spares", help="spare stock for the replacement lead time, and replacement age")
    p_sp.add_argument("parts")
    p_sp.add_argument("--lead-time-days", type=float, required=True, help="days to get a replacement part")
    p_sp.add_argument("--hours-per-day", type=float, default=6, help="average circulating hours per part per day")
    p_sp.add_argument("--service-level", type=float, default=0.95)
    p_sp.add_argument("--unit-cost", type=float, default=3950)
    p_sp.add_argument("--holding-rate", type=float, default=0.25, help="annual carrying cost as share of unit cost")
    p_sp.add_argument("--downtime-cost", type=float, default=18000, help="cost of one failure waiting for a part")
    p_sp.add_argument("--preventive-cost", type=float, default=4600, help="cost of a planned replacement")
    p_sp.add_argument("--failure-cost", type=float, default=21000, help="cost of an in-service failure")

    for p in (p_fit, p_loc, p_cost, p_sp):
        p.add_argument("--json", action="store_true", help="print one JSON object instead of text")

    args = parser.parse_args(argv)
    try:
        parts = load_parts(args.parts)

        if args.command == "fit":
            fitted = fit(parts.hours, parts.failed, args.method)
            naive = _naive(parts).model if args.method == "mle" else None
            three = fit_mle_3p(parts.hours, parts.failed) if args.failure_free else None
            if args.json:
                result = _fit_dict(fitted)
                if naive:
                    result["naive"] = {"beta": naive.beta, "eta_hours": naive.eta}
                if three:
                    result["three_parameter"] = _three_dict(three)
                _emit(result)
                return 0
            _print_fit(fitted)
            if naive:
                print("\nIf every scrapped part counted as a failure and parts in service were ignored:")
                print(f"  scale eta        {naive.eta:,.0f} CH, shape beta {naive.beta:.2f}")
            if three:
                _print_three(three)
            return 0

        model = fit_mle(parts.hours, parts.failed).model
        fleet = parts.hours[parts.in_service]

        if args.command == "locations":
            c = compare_locations(parts.hours, parts.failed, parts.location)
            if args.json:
                _emit(
                    {
                        "locations": [
                            {
                                "location": loc.location,
                                "parts": loc.parts,
                                "failures": loc.failures,
                                "fit": _fit_dict(loc.fit) if loc.fit else None,
                            }
                            for loc in c.locations
                        ],
                        "pooled": _fit_dict(c.pooled),
                        "lr_statistic": c.lr_statistic,
                        "degrees_of_freedom": c.degrees_of_freedom,
                        "p_value": c.p_value,
                        "differs": c.differs,
                    }
                )
                return 0
            print("Location               parts  worn out   beta   eta (CH)   B10 (CH)")
            for loc in c.locations:
                if loc.fit:
                    m = loc.fit.model
                    row = f"{loc.location:<22} {loc.parts:>5} {loc.failures:>9}"
                    print(f"{row} {m.beta:>6.2f} {m.eta:>10,.0f} {m.b_life():>10,.0f}")
                else:
                    print(f"{loc.location:<22} {loc.parts:>5} {loc.failures:>9}   too few failures to fit")
            verdict = "differs between locations" if c.differs else "is consistent across locations"
            test = f"{c.lr_statistic:.1f} on {c.degrees_of_freedom} df, p = {c.p_value:.3g}"
            print(f"\nLikelihood ratio test: {test}. Part life {verdict}.")
            return 0

        if args.command == "cost":
            costs = CostInputs(args.unit_cost, args.replacement_cost, args.salvage)
            lc = lifetime_cost(model, costs, fleet, args.horizon)
            naive = lifetime_cost(_naive(parts).model, costs, fleet, args.horizon)
            plan = event_plan(model, fleet, args.horizon / 4, 4)
            if args.json:
                _emit(
                    {
                        "parts_in_service": len(fleet),
                        "horizon_hours": args.horizon,
                        "cost_per_hour": lc.cost_per_hour,
                        "expected_events": lc.expected_events,
                        "expected_cost": lc.expected_cost,
                        "events_per_1000_hours": lc.events_per_1000_hours,
                        "first_life_events_per_period": [float(x) for x in plan.per_period],
                        "period_hours": plan.period_hours,
                        "naive_expected_cost": naive.expected_cost,
                    }
                )
                return 0
            print(f"{len(fleet)} parts in service, forecast over the next {args.horizon:,.0f} circulating hours")
            print(f"  cost per circulating hour   {lc.cost_per_hour:,.2f}")
            print(f"  expected scrap events       {lc.expected_events:,.1f}")
            print(f"  expected replacement spend  {lc.expected_cost:,.0f}")
            print(f"  events per 1,000 fleet CH   {lc.events_per_1000_hours:.3f}")
            print(
                f"  first-life events per {plan.period_hours:,.0f} CH  {', '.join(f'{x:.1f}' for x in plan.per_period)}"
                "  (parts installed today)"
            )
            diff = naive.expected_cost / lc.expected_cost - 1 if lc.expected_cost else 0
            print(f"\nCounting every scrap as a failure would forecast {naive.expected_cost:,.0f} ({diff:+.0%}).")
            return 0

        horizon = args.lead_time_days * args.hours_per_day
        demand = lead_time_demand(model, fleet, horizon)
        holding = args.holding_rate * args.lead_time_days / 365
        options = stock_options(demand, args.unit_cost, holding, args.downtime_cost)
        target = stock_for_service_level(demand, args.service_level)
        best = cost_optimal_stock(options)
        policy = optimal_replacement_age(model, args.preventive_cost, args.failure_cost)
        if args.json:
            _emit(
                {
                    "parts_in_service": len(fleet),
                    "lead_time_days": args.lead_time_days,
                    "lead_time_hours": horizon,
                    "expected_events_in_lead_time": demand.expected,
                    "service_level_target": args.service_level,
                    "stock_for_service_level": target,
                    "cost_optimal_stock": best.stock,
                    "cost_optimal_service_level": best.service_level,
                    "renewal_share": renewal_share(model, horizon),
                    "replacement_age_hours": policy.age,
                    "replacement_cost_per_hour": policy.cost_per_hour,
                    "run_to_failure_cost_per_hour": policy.run_to_failure_cost_per_hour,
                }
            )
            return 0
        print(
            f"{len(fleet)} parts in service, lead time {args.lead_time_days:g} days ({horizon:,.0f} circulating hours)"
        )
        print(f"  expected end-of-life events in lead time  {demand.expected:.1f}")
        print(f"  stock for {args.service_level:.0%} service level               {target}")
        print(f"  cost-optimal stock                        {best.stock}  ({best.service_level:.1%} service level)")
        if renewal_share(model, horizon) > 0.05:
            print("  note: lead time is long against part life; replacements may wear out within it.")
        if policy.age is None:
            print("Planned replacement does not pay; run to end of life.")
        else:
            print(
                f"Replace at {policy.age:,.0f} CH: {policy.cost_per_hour:.2f} per circulating hour against "
                f"{policy.run_to_failure_cost_per_hour:.2f} when run to end of life ({policy.saving_share:.0%} lower)."
            )
        return 0
    except (DataError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
