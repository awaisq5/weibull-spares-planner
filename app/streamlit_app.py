"""Streamlit front end: fit a life model, forecast fleet failures, size spare stock."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import integrate

from weibull_spares import (
    fit,
    lead_time_demand,
    optimal_replacement_age,
    renewal_share,
    simulate_failures,
    stock_for_service_level,
    stock_options,
)
from weibull_spares.data import DataError, load_fleet, load_history

DATA = Path(__file__).resolve().parent.parent / "data"
INK = "#1B2A24"
PAPER = "#F6F7F2"
GRID = "#A9C7B0"
BLUE = "#2A4D8F"
MUTED = "#5E6B64"
ALARM = "#A3322A"

st.set_page_config(page_title="Weibull spares planner", layout="wide")
st.markdown(
    f"""
    <style>
      .block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
      h1 {{ font-weight: 700; letter-spacing: -0.01em; }}
      .verdict {{ font-size: 1.45rem; line-height: 1.4; max-width: 46em; margin: 0.4rem 0 1.4rem; color: {INK}; }}
      .verdict b {{ color: {BLUE}; }}
      [data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=INK, size=13),
        showlegend=kw.pop("showlegend", False),
        legend=dict(orientation="h", y=1.08, x=0),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=INK, zeroline=False, ticks="outside")
    fig.update_yaxes(gridcolor=GRID, linecolor=INK, zeroline=False, ticks="outside")
    return fig


# Inputs ---------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Data")
    source = st.radio(
        "Life history", ["Sample: mud motor bearing packs", "Upload CSV files"], label_visibility="collapsed"
    )
    if source.startswith("Upload"):
        history_file = st.file_uploader("Life history (unit_id, hours, status)", type="csv")
        fleet_file = st.file_uploader("Installed fleet (unit_id, age_hours)", type="csv")
    else:
        history_file = DATA / "bearing_pack_history.csv"
        fleet_file = DATA / "fleet.csv"
    method = st.selectbox(
        "Fitting method",
        ["mle", "rank_regression"],
        format_func=lambda m: "Maximum likelihood" if m == "mle" else "Median-rank regression",
    )

    st.header("Planning")
    lead_days = st.number_input("Replacement lead time, days", 1.0, 365.0, 21.0, 1.0)
    hours_per_day = st.number_input("Operating hours per unit per day", 0.5, 24.0, 6.0, 0.5)
    service = st.slider("Target service level", 0.80, 0.995, 0.95, 0.005, format="%.3f")
    unit_cost = st.number_input("Unit cost, €", 1.0, value=3950.0, step=50.0)
    holding_rate = st.number_input("Carrying cost per year, % of unit cost", 0.0, 100.0, 25.0, 1.0) / 100
    downtime_cost = st.number_input("Cost of a failure waiting for a part, €", 0.0, value=18000.0, step=500.0)

    st.header("Replacement policy")
    preventive_cost = st.number_input("Planned replacement, €", 1.0, value=4600.0, step=100.0)
    failure_cost = st.number_input("In-service failure, €", 1.0, value=21000.0, step=500.0)

if history_file is None or fleet_file is None:
    st.title("Weibull spares planner")
    st.info("Upload a life history and an installed-fleet file to start, or switch to the sample data.")
    st.stop()

try:
    history = load_history(history_file)
    _, ages = load_fleet(fleet_file)
    result = fit(history.hours, history.failed, method)
except (DataError, ValueError) as exc:
    st.title("Weibull spares planner")
    st.error(str(exc))
    st.stop()

model = result.model
horizon = lead_days * hours_per_day
demand = lead_time_demand(model, ages, horizon)
options = stock_options(demand, unit_cost, holding_rate * lead_days / 365, downtime_cost)
target_stock = stock_for_service_level(demand, service)
best = min(options, key=lambda o: o.total_cost)
policy = optimal_replacement_age(model, preventive_cost, failure_cost)

# Verdict -------------------------------------------------------------------------------------

st.title("Weibull spares planner")
mode = model.failure_mode().split(":")[0]
st.markdown(
    f"""<p class="verdict">The data shows <b>{mode}</b> (β {model.beta:.2f}). With {len(ages)} units in service
    and a {lead_days:g}-day replacement lead time, expect <b>{demand.expected:.1f} failures</b> while a part is
    on its way. Keep <b>{target_stock} spares</b> for a {service:.1%} chance of never waiting on a part.</p>""",
    unsafe_allow_html=True,
)
if renewal_share(model, horizon) > 0.05:
    st.warning(
        "The lead time is long compared with unit life, so replacements may fail again before the next part "
        "arrives. Treat the stock figure as a lower bound and use the 12-month simulation below."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Characteristic life η", f"{model.eta:,.0f} h")
c2.metric("B10 life", f"{model.b_life(0.10):,.0f} h", help="Age by which 10% of units have failed.")
c3.metric("Mean life", f"{model.mean_life():,.0f} h")
c4.metric("Cost-optimal stock", f"{best.stock}", help=f"{best.service_level:.1%} service level")

tab_life, tab_fleet, tab_stock, tab_policy = st.tabs(["Life model", "Fleet forecast", "Stock level", "Replacement age"])

# Life model: Weibull probability paper --------------------------------------------------------

with tab_life:
    left, right = st.columns([3, 2])
    with left:
        ft, f = result.plot_points
        lin = lambda p: np.log(-np.log1p(-np.asarray(p)))  # noqa: E731
        probs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.632, 0.8, 0.9, 0.95, 0.99])
        t_lo = min(ft.min(), model.b_life(0.01)) * 0.7
        t_hi = max(ft.max(), model.b_life(0.99)) * 1.3
        tt = np.geomspace(t_lo, t_hi, 200)
        fig = go.Figure()
        fig.add_scatter(
            x=tt,
            y=lin(model.cdf(tt)),
            mode="lines",
            line=dict(color=BLUE, width=2.5),
            name="Fitted model",
            hovertemplate="%{x:,.0f} h<extra>fitted</extra>",
        )
        fig.add_scatter(
            x=ft,
            y=lin(f),
            mode="markers",
            marker=dict(color=INK, size=6, symbol="circle-open", line=dict(width=1.6)),
            name="Failures (median ranks)",
            hovertemplate="%{x:,.0f} h<extra>failure</extra>",
        )
        fig.add_hline(y=lin(0.632), line=dict(color=MUTED, dash="dot", width=1))
        fig.add_annotation(
            x=np.log10(model.eta),
            y=lin(0.632),
            text=f"η = {model.eta:,.0f} h",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            xshift=-10,
            font=dict(color=MUTED),
        )
        layout(fig, height=440, showlegend=True)
        fig.update_xaxes(type="log", title="Operating hours")
        fig.update_yaxes(
            title="Share failed",
            tickvals=lin(probs),
            ticktext=[f"{p:.0%}" if p != 0.632 else "63.2%" for p in probs],
            range=[lin(0.005), lin(0.995)],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Weibull probability paper: if the points follow a straight line, the Weibull model fits. "
            f"{result.failures} failures plotted; {result.suspensions} suspensions shift their ranks but are not drawn."
        )
    with right:
        st.subheader("Parameters")
        rows = [("Shape β", f"{model.beta:.3f}"), ("Scale η", f"{model.eta:,.0f} h")]
        if result.bounds:
            rows = [
                ("Shape β", f"{model.beta:.3f}", f"{result.bounds['beta'][0]:.2f} to {result.bounds['beta'][1]:.2f}"),
                (
                    "Scale η",
                    f"{model.eta:,.0f} h",
                    f"{result.bounds['eta'][0]:,.0f} to {result.bounds['eta'][1]:,.0f} h",
                ),
            ]
            st.table(pd.DataFrame(rows, columns=["", "Estimate", "90% bounds"]).set_index(""))
        else:
            st.table(pd.DataFrame(rows, columns=["", "Estimate"]).set_index(""))
        st.markdown(f"**Reading:** {model.failure_mode()}.")
        st.markdown(
            f"**Data:** {result.failures} failures and {result.suspensions} suspensions. Suspensions are units "
            "that were still running or removed for another reason. Leaving them out would make the part look "
            "less reliable than it is."
        )

# Fleet forecast ------------------------------------------------------------------------------

with tab_fleet:
    left, right = st.columns(2)
    with left:
        st.subheader(f"Who fails in the next {lead_days:g} days")
        fig = go.Figure()
        fig.add_scatter(
            x=ages,
            y=demand.probabilities,
            mode="markers",
            marker=dict(color=demand.probabilities, colorscale=[[0, GRID], [1, ALARM]], size=8, line=dict(width=0)),
            hovertemplate="age %{x:,.0f} h<br>%{y:.0%} chance<extra></extra>",
        )
        layout(fig)
        fig.update_xaxes(title="Current age, operating hours")
        fig.update_yaxes(title="Chance of failing in lead time", tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each dot is an installed unit. Older units carry most of the risk because the part wears out.")
    with right:
        st.subheader("Failures per month, next 12 months")
        period = 30 * hours_per_day
        sim = simulate_failures(model, ages, period_hours=period, periods=12)
        months = [f"M{i}" for i in range(1, 13)]
        fig = go.Figure()
        fig.add_bar(
            x=months, y=sim.mean, marker_color=BLUE, name="Expected", hovertemplate="%{y:.1f} expected<extra></extra>"
        )
        fig.add_scatter(
            x=months,
            y=sim.high,
            mode="markers",
            marker=dict(symbol="line-ew-open", size=18, color=INK, line=dict(width=2)),
            name="Busy month (90th percentile)",
            hovertemplate="up to %{y:.0f}<extra></extra>",
        )
        layout(fig, showlegend=True)
        fig.update_yaxes(title="Failures")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Monte Carlo simulation with renewal: failed units are replaced by new ones that can fail again. "
            f"{sim.mean.sum():.0f} failures expected over the year."
        )

# Stock level ---------------------------------------------------------------------------------

with tab_stock:
    table = pd.DataFrame(
        {
            "Spares": [o.stock for o in options],
            "Service level": [o.service_level for o in options],
            "Short per lead time": [o.expected_shortage for o in options],
            "Holding cost €": [o.holding_cost for o in options],
            "Shortage cost €": [o.shortage_cost for o in options],
            "Total €": [o.total_cost for o in options],
        }
    )
    lo = max(0, min(target_stock, best.stock) - 8)
    view = table[(table.Spares >= lo) & (table.Spares <= max(target_stock, best.stock) + 4)]
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_scatter(
            x=view.Spares,
            y=view["Service level"],
            mode="lines+markers",
            line=dict(color=BLUE, shape="hv"),
            marker=dict(size=6),
            hovertemplate="%{x} spares: %{y:.1%}<extra></extra>",
        )
        fig.add_hline(y=service, line=dict(color=MUTED, dash="dot"))
        fig.add_vline(x=target_stock, line=dict(color=INK, width=1.5))
        fig.add_annotation(
            x=target_stock,
            y=service,
            text=f"{target_stock} spares meet {service:.1%}",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=6,
            yshift=-4,
        )
        layout(fig, height=360)
        fig.update_xaxes(title="Spares on the shelf", dtick=1)
        fig.update_yaxes(title="Chance no failure waits for a part", tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown(
            f"**Service-level answer:** {target_stock} spares give a {options[target_stock].service_level:.1%} "
            f"chance that every failure in the lead time finds a part."
        )
        st.markdown(
            f"**Cost answer:** {best.stock} spares minimise holding plus downtime cost "
            f"(€{best.total_cost:,.0f} per lead time). "
            + (
                "Downtime is expensive enough to justify more stock than the service target."
                if best.stock > target_stock
                else "The service target already carries enough stock."
            )
        )
        st.caption(
            "Failures in the lead time follow a Poisson-binomial distribution: every unit has its own probability "
            "based on its age. It is computed exactly rather than approximated."
        )
    st.dataframe(
        view.style.format(
            {
                "Service level": "{:.1%}",
                "Short per lead time": "{:.2f}",
                "Holding cost €": "{:,.0f}",
                "Shortage cost €": "{:,.0f}",
                "Total €": "{:,.0f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

# Replacement age -----------------------------------------------------------------------------

with tab_policy:
    tt = np.linspace(model.eta * 0.08, model.eta * 2.5, 160)
    cost = []
    for t in tt:
        cycle, _ = integrate.quad(lambda x: float(model.reliability(x)), 0, t)
        r = float(model.reliability(t))
        cost.append((preventive_cost * r + failure_cost * (1 - r)) / cycle)
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_scatter(
            x=tt,
            y=cost,
            mode="lines",
            line=dict(color=BLUE, width=2.5),
            hovertemplate="replace at %{x:,.0f} h: €%{y:.2f}/h<extra></extra>",
        )
        fig.add_hline(y=policy.run_to_failure_cost_per_hour, line=dict(color=ALARM, dash="dash"))
        fig.add_annotation(
            x=tt[-1],
            y=policy.run_to_failure_cost_per_hour,
            text="run to failure",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color=ALARM),
        )
        if policy.age:
            fig.add_vline(x=policy.age, line=dict(color=INK, width=1.5))
        layout(fig, height=360)
        fig.update_xaxes(title="Planned replacement age, operating hours")
        fig.update_yaxes(title="Cost per operating hour, €", range=[0, max(cost[len(cost) // 6 :]) * 1.15])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if policy.age is None:
            st.markdown("**Run to failure.** Planned replacement does not lower the cost per operating hour.")
        else:
            st.markdown(
                f"**Replace at {policy.age:,.0f} hours.** That costs €{policy.cost_per_hour:.2f} per operating hour "
                f"against €{policy.run_to_failure_cost_per_hour:.2f} when running to failure, "
                f"{policy.saving_share:.0%} lower."
            )
        st.caption(
            "Age replacement policy: replace at a fixed age or at failure, whichever comes first. It only pays when "
            "the part wears out (β above 1) and a planned job is cheaper than a failure."
        )
