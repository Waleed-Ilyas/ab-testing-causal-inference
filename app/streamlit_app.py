"""A/B testing & causal inference on a real randomized ad-incrementality experiment: ATE with proper
inference, health-check diagnostics, CUPED variance reduction, uplift modeling, and a peeking simulation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abtest import config  # noqa: E402
from abtest import peeking as PK  # noqa: E402

st.set_page_config(page_title="A/B testing & causal inference", page_icon="🧪", layout="wide")
GOLD, CYAN, RED, GREEN, GREY, VIOLET, IVORY = "#D9B26A", "#4FD1C5", "#E0625A", "#5fcf80", "#8a8578", "#b08fe8", "#F2EDE4"


@st.cache_data
def load():
    res = json.loads((config.ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    qini = {n: pd.read_parquet(config.ARTIFACTS / f"qini_{n}.parquet") for n in ("t", "x", "s", "random")}
    return res, qini


res, qini = load()
ds, ate, cuped, up, srm, bal = res["dataset"], res["ate"], res["cuped"], res["uplift"], res["srm"], res["balance"]

with st.sidebar:
    st.header("🧪 A/B testing & causal inference")
    st.caption(f"A real randomized incrementality test (Criteo AI Lab, Diemert et al. 2018): "
               f"{ds['n_sample']:,} users sampled from {ds['n_total_source']:,}, {ds['treat_ratio']:.1%} "
               "shown ads (treatment), the rest held out (control).")
    st.divider()
    with st.expander("How to read this"):
        st.markdown(
            "- **SRM** (Sample Ratio Mismatch): checks the treatment/control split matches the intended "
            "randomization ratio. If it doesn't, something is broken upstream and nothing else can be trusted.\n"
            "- **CUPED** uses a pre-treatment covariate to shrink the confidence interval without changing "
            "what's being estimated.\n"
            "- **Qini curve**: how many extra conversions you get by targeting the top X% ranked by "
            "predicted uplift, vs. targeting X% at random.\n"
            "- Features `f0`-`f11` are anonymised by the data provider (privacy), so uplift explanations "
            "name feature indices, not real attributes.")

st.title("A/B testing & causal inference")
st.caption("Did the ad campaign work, for whom, and how would smarter targeting have done?")

c = st.columns(4)
c[0].metric("Sample analysed", f"{ds['n_sample']:,}")
c[1].metric("Conversion lift", f"+{ate['conversion']['relative_lift']:.0%}", "p<0.001", delta_arrow="off")
c[2].metric("SRM check", "passed" if not srm["flagged"] else "FAILED", f"p={srm['p_value']:.2f}", delta_arrow="off")
c[3].metric("Covariates flagged imbalanced", f"{sum(b['flagged'] for b in bal)} of {len(bal)}")

tab_ate, tab_cuped, tab_uplift, tab_peek = st.tabs(
    ["Is there an effect?", "CUPED", "Who should we target?", "The peeking problem"])

# ------------------------------------------------------------------ ATE + diagnostics
with tab_ate:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Treatment effect")
        rows = [{"Outcome": o, "Control rate": ate[o]["p_control"], "Treatment rate": ate[o]["p_treat"],
                "Absolute effect": ate[o]["ate"], "Relative lift": ate[o]["relative_lift"],
                "95% CI": f"{ate[o]['ci'][0]:.5f} to {ate[o]['ci'][1]:.5f}", "p-value": ate[o]["p_value"]}
               for o in ("visit", "conversion")]
        st.dataframe(pd.DataFrame(rows).style.format({"Control rate": "{:.4%}", "Treatment rate": "{:.4%}",
                                                       "Absolute effect": "{:.5f}", "Relative lift": "{:+.0%}",
                                                       "p-value": "{:.2e}"}), hide_index=True, use_container_width=True)
        st.caption(f"A relative lift this small (10%) at this baseline conversion rate needs roughly "
                   f"{res['required_n_per_arm_10pct_mde']['conversion']:,} users per arm to detect reliably — "
                   f"this experiment had {ate['conversion']['n_control']:,} in control, enough to detect the "
                   f"much larger effect ({ate['conversion']['relative_lift']:.0%}) that was actually there.")
    with right:
        st.subheader("Is the experiment itself trustworthy?")
        st.markdown(f"**Sample ratio mismatch**: observed {srm['observed']['control']:,} control / "
                   f"{srm['observed']['treatment']:,} treatment vs. an expected {srm['expected_ratio']:.0%} "
                   f"treated ratio. Chi-square p={srm['p_value']:.3f} — "
                   f"{'no mismatch detected.' if not srm['flagged'] else '**mismatch detected.**'}")
        bdf = pd.DataFrame(bal)
        st.markdown("**Covariate balance** (standardized mean difference, treatment vs. control):")
        fig = go.Figure(go.Bar(x=bdf.smd, y=bdf.feature, orientation="h",
                               marker_color=[RED if f else CYAN for f in bdf.flagged]))
        fig.add_vline(x=0.1, line_dash="dot", line_color=GREY)
        fig.add_vline(x=-0.1, line_dash="dot", line_color=GREY)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="standardized mean difference",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=IVORY)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("All 12 covariates fall inside +/-0.1 — consistent with clean randomization at this scale.")

# ------------------------------------------------------------------ CUPED
with tab_cuped:
    st.subheader("Variance reduction with CUPED")
    st.caption(f"Covariate: `{cuped['covariate']}` (the feature most correlated with conversion in the "
               "control arm, chosen automatically, not treatment-related).")
    cuped_rows = [("Raw", cuped["raw_ate"], cuped["raw_ci"], CYAN),
                 ("CUPED-adjusted", cuped["cuped_ate"], cuped["cuped_ci"], GOLD)]
    fig = go.Figure()
    for label, a, ci, color in cuped_rows:
        fig.add_trace(go.Scatter(x=[a], y=[label],
                                 error_x=dict(type="data", array=[ci[1] - a], arrayminus=[a - ci[0]]),
                                 mode="markers", marker=dict(size=16, color=color), name=label))
    fig.update_layout(height=280, xaxis_title="conversion treatment effect", showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=IVORY,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Variance reduction", f"{cuped['variance_reduction']:.1%}")
    c2.metric("CI narrower by", f"{cuped['ci_width_reduction']:.1%}")
    c3.metric("Point estimate moved by", f"{abs(cuped['cuped_ate'] - cuped['raw_ate']):.6f}",
             "should be ~0 (unbiased)", delta_arrow="off")
    st.caption("A modest result here: the best available anonymised covariate explains only "
              f"{cuped['variance_reduction']:.0%} of the outcome's variance, because a rare binary outcome "
              "(0.3% conversion) leaves little variance for any covariate to explain. CUPED works best on "
              "continuous, high-variance metrics (revenue, session length) with a strong pre-period covariate.")

# ------------------------------------------------------------------ uplift
with tab_uplift:
    st.subheader(f"Best model: {up['best_model']}")
    fig = go.Figure()
    colors = {"t": CYAN, "x": VIOLET, "s": GOLD}
    names = {"t": "T-learner", "x": "X-learner", "s": "S-learner"}
    for k, col in colors.items():
        q = qini[k]
        fig.add_trace(go.Scatter(x=q.frac, y=q.gain, mode="lines", name=names[k],
                                 line=dict(color=col, width=3 if names[k] == up["best_model"] else 1.4)))
    fig.add_trace(go.Scatter(x=qini["random"].frac, y=qini["random"].gain, mode="lines", name="random",
                             line=dict(color=GREY, dash="dash")))
    fig.update_layout(height=400, xaxis_title="fraction targeted (by predicted uplift)",
                      yaxis_title="cumulative incremental conversions", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color=IVORY, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Targeting policy value** (out-of-sample holdout, unbiased IPW estimator):")
    fracs = [10, 20, 30, 50]
    pv = up["policy_values"]
    rows = [{"Coverage": f"{f}%", "Top-uplift targeting": pv[f"top_{f}pct_uplift"]["value"],
            "Random targeting": pv[f"random_{f}pct"]["value"],
            "Lift over random": pv[f"top_{f}pct_uplift"]["value"] - pv[f"random_{f}pct"]["value"]} for f in fracs]
    rows.append({"Coverage": "100% (treat everyone)", "Top-uplift targeting": pv["treat_all"]["value"],
                "Random targeting": pv["treat_all"]["value"], "Lift over random": 0.0})
    st.dataframe(pd.DataFrame(rows).style.format({"Top-uplift targeting": "{:.4%}", "Random targeting": "{:.4%}",
                                                  "Lift over random": "{:+.4%}"}), hide_index=True,
                use_container_width=True)
    d = pd.DataFrame(up["decile_table"])
    fig2 = go.Figure(go.Bar(x=d.decile, y=d.observed_uplift * 1000,
                            marker_color=[GOLD if v > 0 else RED for v in d.observed_uplift]))
    fig2.add_hline(y=ate["conversion"]["ate"] * 1000, line_dash="dot", line_color=IVORY,
                  annotation_text="overall ATE")
    fig2.update_layout(height=320, xaxis_title="decile (1 = highest predicted uplift)",
                       yaxis_title="observed uplift (per 1,000 users)", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font_color=IVORY, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Almost all of the measurable lift concentrates in decile 1; deciles 2-9 are noisy around zero "
              "because conversion is rare (a few thousand events spread across ~60,000 users per decile).")

# ------------------------------------------------------------------ peeking
with tab_peek:
    st.subheader("Why checking your A/B test dashboard every day is dangerous")
    st.caption("Illustrative simulation (not the Criteo data above): 2,000 simulated A/A tests where BOTH "
              "arms have the exact same true conversion rate, so any 'significant' result is, by "
              "construction, a false positive.")
    naive, corr = res["peeking"]["naive"], res["peeking"]["corrected"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Look once at the end", f"{naive['false_positive_rate_final_only']:.1%}", "target: 5%", delta_arrow="off")
    c2.metric("Peek 20 times, stop at first p<0.05", f"{naive['false_positive_rate_with_peeking']:.1%}",
             "target: 5%", delta_color="inverse", delta_arrow="off")
    c3.metric("Peek 20 times, O'Brien-Fleming bounds", f"{corr['false_positive_rate_corrected']:.1%}",
             "target: 5%", delta_arrow="off")
    fig = go.Figure(go.Bar(x=["Look once", "Peek + stop early", "Peek + spending function"],
                           y=[naive["false_positive_rate_final_only"], naive["false_positive_rate_with_peeking"],
                             corr["false_positive_rate_corrected"]],
                           marker_color=[CYAN, RED, GREEN]))
    fig.add_hline(y=0.05, line_dash="dot", line_color=IVORY, annotation_text="nominal alpha = 0.05")
    fig.update_layout(height=360, yaxis_title="false-positive rate", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color=IVORY, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Run your own simulation:")
    n_exp = st.slider("Simulated A/A tests", 200, 3000, 800, 100)
    n_looks = st.slider("Number of peeks", 2, 30, 10)
    if st.button("Run simulation"):
        with st.spinner("Simulating..."):
            live = PK.simulate_fixed_horizon_peeking(n_exp, 3000, n_looks=n_looks)
        st.write(f"Peeking {n_looks} times inflated the false-positive rate to "
                f"**{live['false_positive_rate_with_peeking']:.1%}** (vs. the nominal 5%).")
