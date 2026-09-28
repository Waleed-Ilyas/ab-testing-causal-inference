"""README figures from artifacts/. Run: python -m abtest.report"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

GOLD, CYAN, IVORY, BG, GREY, RED, VIOLET, GREEN = ("#D9B26A", "#4FD1C5", "#F2EDE4", "#0A0A0C", "#8a8578",
                                                   "#E0625A", "#b08fe8", "#5fcf80")
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": IVORY,
                     "axes.labelcolor": IVORY, "xtick.color": IVORY, "ytick.color": IVORY,
                     "axes.edgecolor": GREY, "font.size": 11})


def fig_ate(res) -> None:
    ate = res["ate"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, out in enumerate(("visit", "conversion")):
        a = ate[out]
        ax.errorbar(a["ate"], i, xerr=[[a["ate"] - a["ci"][0]], [a["ci"][1] - a["ate"]]], fmt="o",
                   color=GOLD if out == "conversion" else CYAN, capsize=5, markersize=10)
        p_txt = "p<0.001" if a["p_value"] < 0.001 else f"p={a['p_value']:.3f}"
        ax.text(a["ci"][1], i + 0.18, f"+{a['relative_lift']:.0%}  ({p_txt})", va="center", ha="left", fontsize=9)
    ax.axvline(0, color=GREY, lw=1, ls=":")
    ax.set_yticks([0, 1], ["visit", "conversion"])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlim(right=ate["visit"]["ci"][1] * 1.35)
    ax.set_xlabel("treatment effect (absolute, 95% CI)")
    ax.set_title("Treatment effect on visits and conversions", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "01_ate.png", dpi=150)


def fig_cuped(res) -> None:
    c = res["cuped"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (_label, ate, ci, color) in enumerate([("Raw", c["raw_ate"], c["raw_ci"], CYAN),
                                                 ("CUPED-adjusted", c["cuped_ate"], c["cuped_ci"], GOLD)]):
        ax.errorbar(ate, i, xerr=[[ate - ci[0]], [ci[1] - ate]], fmt="o", color=color, capsize=6, markersize=11)
    ax.axvline(0, color=GREY, lw=1, ls=":")
    ax.set_yticks([0, 1], ["Raw ATE", f"CUPED (covariate {c['covariate']})"])
    ax.set_xlabel("conversion treatment effect")
    ax.set_title(f"CUPED: {c['variance_reduction']:.0%} variance reduction, "
                 f"CI {c['ci_width_reduction']:.0%} narrower", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "02_cuped.png", dpi=150)


def fig_qini(res) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"t": CYAN, "x": VIOLET, "s": GOLD}
    for name, color in colors.items():
        q = pd.read_parquet(config.ARTIFACTS / f"qini_{name}.parquet")
        label = {"t": "T-learner", "x": "X-learner", "s": "S-learner"}[name]
        ax.plot(q.frac, q.gain, color=color, lw=2.4 if label == res["uplift"]["best_model"] else 1.4, label=label)
    rand = pd.read_parquet(config.ARTIFACTS / "qini_random.parquet")
    ax.plot(rand.frac, rand.gain, color=GREY, ls="--", lw=1.4, label="random targeting")
    ax.set_xlabel("fraction of holdout targeted (sorted by predicted uplift, descending)")
    ax.set_ylabel("cumulative incremental conversions")
    ax.set_title("Qini curves: how much does targeting by predicted uplift beat random targeting?", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "03_qini.png", dpi=150)


def fig_decile(res) -> None:
    d = pd.DataFrame(res["uplift"]["decile_table"])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(d.decile, d.observed_uplift * 1000, color=[GOLD if v > 0 else RED for v in d.observed_uplift])
    ax.axhline(res["ate"]["conversion"]["ate"] * 1000, color=IVORY, ls=":", lw=1.2, label="overall ATE")
    ax.set_xticks(d.decile)
    ax.set_xlabel("decile (1 = highest predicted uplift)")
    ax.set_ylabel("observed uplift (conversions per 1,000 users)")
    ax.set_title(f"{res['uplift']['best_model']}: does the ranking match reality?", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "04_deciles.png", dpi=150)


def fig_policy_value(res) -> None:
    pv = res["uplift"]["policy_values"]
    fracs = [10, 20, 30, 50]
    top = [pv[f"top_{f}pct_uplift"]["value"] for f in fracs]
    rand = [pv[f"random_{f}pct"]["value"] for f in fracs]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(fracs))
    ax.bar(x - 0.18, top, 0.36, color=GOLD, label="top-k% by predicted uplift")
    ax.bar(x + 0.18, rand, 0.36, color=GREY, label="random k%")
    ax.axhline(pv["treat_none"]["value"], color=CYAN, ls=":", lw=1.2, label="treat no one")
    ax.axhline(pv["treat_all"]["value"], color=RED, ls="--", lw=1.2, label="treat everyone")
    ax.set_xticks(x, [f"{f}%" for f in fracs])
    ax.set_xlabel("share of users targeted")
    ax.set_ylabel("conversion rate under this policy (IPW-estimated)")
    ax.set_title("Targeting policy value: smart targeting vs random, at equal coverage", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "05_policy_value.png", dpi=150)


def fig_peeking(res) -> None:
    naive, corr = res["peeking"]["naive"], res["peeking"]["corrected"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    vals = [naive["false_positive_rate_final_only"], naive["false_positive_rate_with_peeking"],
           corr["false_positive_rate_corrected"]]
    labels = ["Look once\n(final sample only)", "Peek 20 times,\nstop at first p<0.05", "Peek 20 times,\nO'Brien-Fleming bounds"]
    ax.bar(labels, vals, color=[CYAN, RED, GREEN])
    ax.axhline(0.05, color=IVORY, ls=":", lw=1.2, label="nominal alpha = 0.05")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=10)
    ax.set_ylabel("false-positive rate (2,000 simulated A/A tests, true effect = 0)")
    ax.set_title("Peeking inflates false positives; a spending-function boundary controls it", fontsize=10.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "06_peeking.png", dpi=150)


if __name__ == "__main__":
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    r = json.loads((config.ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    fig_ate(r)
    fig_cuped(r)
    fig_qini(r)
    fig_decile(r)
    fig_policy_value(r)
    fig_peeking(r)
    print("figures written")
