"""Streamlit front end for the lifetime analysis tool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import integrate

from weibull_spares import (
    CostInputs,
    compare_locations,
    event_plan,
    fit,
    fit_mle,
    lead_time_demand,
    lifetime_cost,
    optimal_replacement_age,
    renewal_share,
    simulate_failures,
    stock_for_service_level,
    stock_options,
)
from weibull_spares.data import DataError, load_parts

DATA = Path(__file__).resolve().parent.parent / "data" / "parts_register.csv"
INK = "#1B2A24"
PAPER = "#F6F7F2"
GRID = "#C9D6CC"
BLUE = "#2A4D8F"
MUTED = "#5E6B64"
ALARM = "#A3322A"
CENSORED = "#9BB0A3"
LOCATION_COLOURS = ["#2A4D8F", "#A3322A", "#3E7D5A", "#B07A1E", "#6B4E8A", "#2F8A8A"]

st.set_page_config(page_title="Lifetime analysis tool", layout="wide")
st.markdown(
    f"""
    <style>
      .block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
      h1 {{ font-weight: 700; letter-spacing: -0.01em; }}
      .verdict {{ font-size: 1.4rem; line-height: 1.45; max-width: 46em; margin: 0.4rem 0 1.4rem; color: {INK}; }}
      .verdict b {{ color: {BLUE}; }}
      [data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def layout(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=36 if kw.get("title") else 10, b=10),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=INK, size=13),
        showlegend=kw.pop("showlegend", False),
        legend=dict(orientation="h", y=1.1, x=0),
        bargap=0.04,
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=INK, zeroline=False, ticks="outside")
    fig.update_yaxes(gridcolor=GRID, linecolor=INK, zeroline=False, ticks="outside")
    return fig


# Inputs -------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Parts register")
    source = st.radio("Data", ["Sample: mud motor bearing packs", "Upload a CSV file"], label_visibility="collapsed")
    upload = (
        st.file_uploader("part_id, part_number, location, circulating_hours, scrap_reason", type="csv")
        if source.startswith("Upload")
        else None
    )
    method = st.selectbox(
        "Fitting method",
        ["mle", "rank_regression"],
        format_func=lambda m: "Maximum likelihood" if m == "mle" else "Median-rank regression",
    )

    st.header("Costs per part")
    unit_cost = st.number_input("Price of a new part, €", 1.0, value=3950.0, step=50.0)
    replacement_cost = st.number_input("Replacement labour and logistics, €", 0.0, value=1200.0, step=50.0)
    salvage = st.number_input("Value recovered when scrapped, €", 0.0, value=150.0, step=10.0)

    st.header("Planning")
    horizon = st.number_input("Forecast horizon, circulating hours", 50.0, 10000.0, 1000.0, 50.0)
    hours_per_day = st.number_input("Circulating hours per part per day", 0.5, 24.0, 6.0, 0.5)
    lead_days = st.number_input("Replacement lead time, days", 1.0, 365.0, 21.0, 1.0)
    service = st.slider("Target service level", 0.80, 0.995, 0.95, 0.005, format="%.3f")
    downtime_cost = st.number_input("Cost of a failure waiting for a part, €", 0.0, value=18000.0, step=500.0)
    failure_cost = st.number_input("Cost of an in-service failure, €", 1.0, value=21000.0, step=500.0)

st.title("Lifetime analysis tool")
st.caption(
    "Independent personal project with synthetic sample data; it contains no information from any former employer."
)

if source.startswith("Upload") and upload is None:
    st.info(
        "Upload a parts register: one row per part with its circulating hours and scrap reason "
        "(blank if still in service). A MATLAB-style `censored` column (1 = censored, 0 = observed) also works."
    )
    st.stop()

try:
    register = load_parts(upload if upload is not None else DATA)
except DataError as exc:
    st.error(str(exc))
    st.stop()

part_numbers = sorted(set(register.part_number.tolist()))
if len(part_numbers) > 1:
    pn = st.selectbox("Part number", part_numbers)
    register = register.where(register.part_number == pn)

try:
    result = fit(register.hours, register.failed, method)
    scrapped = register.where(~register.in_service)
    naive = fit_mle(scrapped.hours, np.ones(len(scrapped.hours), dtype=bool))
except ValueError as exc:
    st.error(f"Cannot fit a model: {exc}")
    st.stop()

model = result.model
fleet = register.hours[register.in_service]
costs = CostInputs(unit_cost, replacement_cost, salvage)

# Verdict ------------------------------------------------------------------------------------

mode = model.failure_mode().split(":")[0]
n_life = int(register.failed.sum())
n_cens = len(register.hours) - n_life
st.markdown(
    f"""<p class="verdict">{len(register.hours)} parts, {n_life} worn out and {n_cens} censored, show
    <b>{mode}</b> (β {model.beta:.2f}). Characteristic life is <b>{model.eta:,.0f} circulating hours</b>;
    counting every scrapped part as a failure would have put it at {naive.model.eta:,.0f}.</p>""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scale η (characteristic life)", f"{model.eta:,.0f} CH", help="Age by which 63.2% of parts have worn out.")
c2.metric("Shape β", f"{model.beta:.2f}", help="Below 1: early failures. Around 1: random. Above 1: wear-out.")
c3.metric("B10 life", f"{model.b_life(0.10):,.0f} CH", help="Age by which 10% of parts have worn out.")
c4.metric("Cost per circulating hour", f"€{costs.cost_per_event / model.mean_life():.2f}")

tabs = st.tabs(["Lifetime model", "Locations", "Lifetime cost", "Maintenance events", "Spares and replacement age"])

# Lifetime model -----------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Why censoring matters")
    st.caption(
        "Left: what the register contains. Right: the life distribution fitted two ways. Counting only scrapped parts "
        "ignores everything still running and every part scrapped for a reason unrelated to wear, so parts look weaker "
        "than they are."
    )
    hi = float(np.percentile(register.hours, 99.5)) * 1.05
    bins = dict(start=0, end=hi, size=hi / 20)
    other = ~register.failed & ~register.in_service
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for mask, name, colour in [
            (register.failed, "Worn out (failure)", BLUE),
            (other, "Scrapped, other reason (censored)", "#7C8F84"),
            (register.in_service, "In service (censored)", CENSORED),
        ]:
            fig.add_histogram(x=register.hours[mask], xbins=bins, marker_color=colour, name=name)
        layout(fig, title="Parts by circulating hours and outcome", showlegend=True, barmode="stack")
        fig.update_layout(legend=dict(orientation="h", y=-0.28, x=0))
        fig.update_xaxes(title="Circulating hours")
        fig.update_yaxes(title="Parts")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        tt = np.linspace(1, max(model.b_life(0.995), naive.model.b_life(0.995)), 300)
        fig = go.Figure()
        fig.add_scatter(
            x=tt,
            y=naive.model.pdf(tt),
            mode="lines",
            line=dict(color=MUTED, width=2.5, dash="dash"),
            name=f"Scrapped parts only: η {naive.model.eta:,.0f} CH",
        )
        fig.add_scatter(
            x=tt,
            y=model.pdf(tt),
            mode="lines",
            line=dict(color=ALARM, width=3),
            name=f"All parts, with censoring: η {model.eta:,.0f} CH",
        )
        for m, colour in [(naive.model, MUTED), (model, ALARM)]:
            fig.add_vline(x=m.eta, line=dict(color=colour, width=1, dash="dot"))
        layout(fig, title="Fitted life distribution", showlegend=True)
        fig.update_layout(legend=dict(orientation="h", y=-0.28, x=0))
        fig.update_xaxes(title="Circulating hours")
        fig.update_yaxes(title="Probability density", showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("How each part is counted")
        sample = pd.DataFrame(
            {
                "part": register.part_id,
                "hours": register.hours,
                "failed": register.failed,
                "in_service": register.in_service,
            }
        )
        picks = pd.concat(
            [
                sample[sample.failed].nlargest(2, "hours"),
                sample[~sample.failed & ~sample.in_service].nlargest(2, "hours"),
                sample[sample.in_service].nlargest(2, "hours"),
            ]
        ).iloc[::-1]
        fig = go.Figure()
        for i, r in enumerate(picks.itertuples()):
            colour = BLUE if r.failed else CENSORED
            fig.add_scatter(x=[0, r.hours], y=[i, i], mode="lines", line=dict(color=colour, width=4), hoverinfo="skip")
            fig.add_scatter(
                x=[r.hours],
                y=[i],
                mode="markers",
                marker=dict(
                    symbol="x-thin" if r.failed else "triangle-right",
                    size=14,
                    color=colour,
                    line=dict(width=3, color=colour),
                ),
                hovertemplate=f"{r.part}: {r.hours:,.0f} CH<extra></extra>",
            )
        labels = [
            f"{r.part}  " + ("worn out" if r.failed else "in service" if r.in_service else "scrapped, other reason")
            for r in picks.itertuples()
        ]
        layout(fig, height=300)
        fig.update_yaxes(tickvals=list(range(len(picks))), ticktext=labels, gridcolor=PAPER)
        fig.update_xaxes(title="Circulating hours", rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "✕ the life ended: fully observed (censoring flag 0). ▸ the part was still going when we stopped "
            "watching: right-censored (flag 1). Both kinds carry information; dropping the second kind makes parts "
            "look weaker than they are."
        )
    with right:
        st.subheader("Parameters")
        if result.bounds:
            st.table(
                pd.DataFrame(
                    [
                        (
                            "Shape β",
                            f"{model.beta:.3f}",
                            f"{result.bounds['beta'][0]:.2f} to {result.bounds['beta'][1]:.2f}",
                        ),
                        (
                            "Scale η",
                            f"{model.eta:,.0f} CH",
                            f"{result.bounds['eta'][0]:,.0f} to {result.bounds['eta'][1]:,.0f} CH",
                        ),
                    ],
                    columns=["", "Estimate", "95% bounds"],
                ).set_index("")
            )
        else:
            st.table(
                pd.DataFrame(
                    [("Shape β", f"{model.beta:.3f}"), ("Scale η", f"{model.eta:,.0f} CH")], columns=["", "Estimate"]
                ).set_index("")
            )
        st.markdown(f"**Reading:** {model.failure_mode()}.")
        st.table(register.summary().set_index("Outcome"))

# Locations ----------------------------------------------------------------------------------

with tabs[1]:
    comparison = compare_locations(register.hours, register.failed, register.location)
    fitted = [loc for loc in comparison.locations if loc.fit]
    if len(fitted) < 2:
        st.info("At least two locations with three or more worn-out parts are needed to compare.")
    else:
        verdict = (
            f"Part life **differs between locations** (likelihood ratio {comparison.lr_statistic:.1f} on "
            f"{comparison.degrees_of_freedom} degrees of freedom, p = {comparison.p_value:.2g})."
            if comparison.differs
            else f"The data does not show a real difference between locations (p = {comparison.p_value:.2f})."
        )
        st.markdown(verdict)
        left, right = st.columns([3, 2])
        with left:
            tt = np.linspace(0, max(loc.fit.model.b_life(0.95) for loc in fitted), 300)
            fig = go.Figure()
            for i, loc in enumerate(fitted):
                fig.add_scatter(
                    x=tt,
                    y=loc.fit.model.reliability(tt),
                    mode="lines",
                    name=loc.location,
                    line=dict(color=LOCATION_COLOURS[i % len(LOCATION_COLOURS)], width=2.5),
                    hovertemplate=f"{loc.location}<br>%{{x:,.0f}} CH: %{{y:.0%}} still running<extra></extra>",
                )
            layout(fig, height=380, showlegend=True)
            fig.update_xaxes(title="Circulating hours")
            fig.update_yaxes(title="Share of parts still running", tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.table(
                pd.DataFrame(
                    [
                        (
                            loc.location,
                            loc.parts,
                            loc.failures,
                            f"{loc.fit.model.eta:,.0f}" if loc.fit else "–",
                            f"{loc.fit.model.beta:.2f}" if loc.fit else "–",
                            f"{loc.fit.model.b_life():,.0f}" if loc.fit else "–",
                        )
                        for loc in comparison.locations
                    ],
                    columns=["Location", "Parts", "Worn out", "η (CH)", "β", "B10 (CH)"],
                ).set_index("Location")
            )
            st.caption(
                "The test compares one pooled model with a separate model per location. A small p-value means the "
                "separate models explain the data better than chance would."
            )

# Lifetime cost ------------------------------------------------------------------------------

with tabs[2]:
    lc = lifetime_cost(model, costs, fleet, horizon)
    lc_naive = lifetime_cost(naive.model, costs, fleet, horizon)
    st.markdown(
        f"The **{len(fleet)} parts in service** are expected to reach the end of their life "
        f"**{lc.expected_events:.0f} times** over the next {horizon:,.0f} circulating hours, replacements included: "
        f"**€{lc.expected_cost:,.0f}** of replacement spend."
    )
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_bar(
            y=["Scrapped parts only", "All parts, with censoring"],
            x=[lc_naive.expected_cost, lc.expected_cost],
            orientation="h",
            marker_color=[CENSORED, BLUE],
            text=[f"€{lc_naive.expected_cost:,.0f}", f"€{lc.expected_cost:,.0f}"],
            textposition="inside",
            insidetextanchor="end",
            hoverinfo="skip",
        )
        layout(fig, height=200)
        fig.update_xaxes(title=f"Expected replacement spend, next {horizon:,.0f} CH", tickprefix="€")
        st.plotly_chart(fig, use_container_width=True)
        gap = lc_naive.expected_cost / lc.expected_cost - 1 if lc.expected_cost else 0
        st.caption(f"Counting every scrapped part as a failure overstates the forecast by {gap:.0%}.")
    with right:
        st.table(
            pd.DataFrame(
                [
                    ("Cost per replacement event", f"€{costs.cost_per_event:,.0f}"),
                    ("Mean life", f"{lc.mean_life_hours:,.0f} CH"),
                    ("Cost per circulating hour", f"€{lc.cost_per_hour:.2f}"),
                    ("Events per 1,000 fleet CH", f"{lc.events_per_1000_hours:.3f}"),
                ],
                columns=["", "Value"],
            ).set_index("")
        )

# Maintenance events -------------------------------------------------------------------------

with tabs[3]:
    periods = 8
    period = horizon / periods
    plan = event_plan(model, fleet, period, periods)
    sim = simulate_failures(model, fleet, period_hours=period, periods=periods)
    labels = [f"{int(period * i):,}–{int(period * (i + 1)):,}" for i in range(periods)]
    st.markdown(
        "How many parts will reach the end of their life in each block of circulating hours: the event frequency that "
        "event-based maintenance plans around."
    )
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_bar(x=labels, y=sim.mean, marker_color=BLUE, name="Expected, replacements included")
        fig.add_scatter(
            x=labels,
            y=plan.per_period,
            mode="markers",
            marker=dict(color=INK, size=9, symbol="diamond"),
            name="Parts installed today only",
        )
        fig.add_scatter(
            x=labels,
            y=sim.high,
            mode="markers",
            marker=dict(symbol="line-ew-open", size=18, color=ALARM, line=dict(width=2)),
            name="Busy block (90th percentile)",
        )
        layout(fig, height=380, showlegend=True)
        fig.update_xaxes(title="Circulating hours from now")
        fig.update_yaxes(title="End-of-life events")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown(f"**Next up in the first {period:,.0f} CH**")
        ids = register.part_id[register.in_service]
        locs = register.location[register.in_service]
        p_next = model.conditional_failure(fleet, period)
        st.dataframe(
            pd.DataFrame(
                {
                    "Part": ids[plan.next_up],
                    "Location": locs[plan.next_up],
                    "Hours so far": fleet[plan.next_up],
                    "Chance": p_next[plan.next_up],
                }
            ).style.format({"Hours so far": "{:,.0f}", "Chance": "{:.0%}"}),
            hide_index=True,
            use_container_width=True,
        )

# Spares and replacement age -----------------------------------------------------------------

with tabs[4]:
    lt_hours = lead_days * hours_per_day
    demand = lead_time_demand(model, fleet, lt_hours)
    options = stock_options(demand, unit_cost, 0.25 * lead_days / 365, downtime_cost)
    target = stock_for_service_level(demand, service)
    best = min(options, key=lambda o: o.total_cost)
    policy = optimal_replacement_age(model, costs.cost_per_event, failure_cost)
    st.markdown(
        f"In a {lead_days:g}-day replacement lead time ({lt_hours:,.0f} CH), expect **{demand.expected:.1f}** "
        f"end-of-life events. **{target} spares** give a {service:.1%} chance that none waits for a part; "
        f"**{best.stock}** is cheapest once downtime is priced in."
    )
    if renewal_share(model, lt_hours) > 0.05:
        st.warning("The lead time is long compared with part life, so treat the stock figure as a lower bound.")
    left, right = st.columns(2)
    with left:
        view = [o for o in options if max(0, min(target, best.stock) - 8) <= o.stock <= max(target, best.stock) + 4]
        fig = go.Figure()
        fig.add_scatter(
            x=[o.stock for o in view],
            y=[o.service_level for o in view],
            mode="lines+markers",
            line=dict(color=BLUE, shape="hv"),
            hovertemplate="%{x} spares: %{y:.1%}<extra></extra>",
        )
        fig.add_hline(y=service, line=dict(color=MUTED, dash="dot"))
        fig.add_vline(x=target, line=dict(color=INK, width=1.5))
        layout(fig, height=320)
        fig.update_xaxes(title="Spares on the shelf", dtick=1)
        fig.update_yaxes(title="Chance no event waits for a part", tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        tt = np.linspace(model.eta * 0.08, model.eta * 2.5, 160)
        rates = []
        for t in tt:
            cycle, _ = integrate.quad(lambda x: float(model.reliability(x)), 0, t)
            r = float(model.reliability(t))
            rates.append((costs.cost_per_event * r + failure_cost * (1 - r)) / cycle)
        fig = go.Figure()
        fig.add_scatter(
            x=tt,
            y=rates,
            mode="lines",
            line=dict(color=BLUE, width=2.5),
            hovertemplate="replace at %{x:,.0f} CH: €%{y:.2f}/CH<extra></extra>",
        )
        fig.add_hline(y=policy.run_to_failure_cost_per_hour, line=dict(color=ALARM, dash="dash"))
        if policy.age:
            fig.add_vline(x=policy.age, line=dict(color=INK, width=1.5))
        layout(fig, height=320)
        fig.update_xaxes(title="Planned replacement age, CH")
        fig.update_yaxes(title="Cost per circulating hour, €", range=[0, max(rates[len(rates) // 6 :]) * 1.15])
        st.plotly_chart(fig, use_container_width=True)
        if policy.age:
            st.caption(
                f"Replacing at {policy.age:,.0f} CH costs €{policy.cost_per_hour:.2f} per circulating hour against "
                f"€{policy.run_to_failure_cost_per_hour:.2f} when running parts to the end of their life (red line)."
            )
        else:
            st.caption("Planned replacement does not lower the cost per circulating hour for this part.")
