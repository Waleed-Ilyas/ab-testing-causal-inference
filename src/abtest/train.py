"""Run every analysis in this project once and write artifacts/results.json.

    python -m abtest.train
"""

from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd

from . import config, data, peeking, stats
from . import uplift as U


def _find_cuped_covariate(df: pd.DataFrame, outcome: str) -> str:
    """The feature most correlated with the outcome (within the control arm, so treatment can't leak in)
    makes the best CUPED covariate — pick it data-drivenly rather than guessing."""
    ctrl = df[df[config.TREATMENT] == 0]
    corr = {f: abs(np.corrcoef(ctrl[f], ctrl[outcome])[0, 1]) for f in config.FEATURES}
    return max(corr, key=corr.get)


def main():
    t0 = time.time()
    config.ARTIFACTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("abtest")

    df = data.load_sample()
    print(f"sample: {len(df):,} rows ({df[config.TREATMENT].mean():.1%} treated)", flush=True)
    nominal_ratio = 0.85  # stated in the source dataset's README

    with mlflow.start_run(run_name="pipeline"):
        # --- 1. sample ratio mismatch --------------------------------------------------------------
        srm = stats.srm_check(df[config.TREATMENT], nominal_ratio)
        print("SRM:", srm, flush=True)

        # --- 2. covariate balance --------------------------------------------------------------------
        bal = stats.balance_table(df, config.FEATURES, config.TREATMENT)
        print("balance flagged:", bal[bal.flagged].feature.tolist(), flush=True)

        # --- 3. ATE for each outcome, with bootstrap cross-check ------------------------------------
        ate = {}
        for out in config.OUTCOMES:
            res = stats.two_proportion_test(df[out], df[config.TREATMENT])
            boot_ci = stats.bootstrap_ate_ci(df[out].to_numpy(float), df[config.TREATMENT].to_numpy(int))
            ate[out] = {"p_treat": res.p_treat, "p_control": res.p_control, "ate": res.ate,
                       "relative_lift": res.relative_lift, "ci": res.ci, "bootstrap_ci": boot_ci,
                       "z": res.z, "p_value": res.p_value, "n_treat": res.n_treat, "n_control": res.n_control}
            mlflow.log_metrics({f"{out}_ate": res.ate, f"{out}_p_value": res.p_value})
            print(out, ate[out], flush=True)

        # --- 4. required sample size, for context -----------------------------------------------------
        n_needed = {out: stats.required_sample_size(ate[out]["p_control"], 0.10) for out in config.OUTCOMES}

        # --- 5. CUPED on the outcome with the biggest realistic effect (conversion) -------------------
        covariate = _find_cuped_covariate(df, "conversion")
        cuped = stats.cuped_ate(df, "conversion", config.TREATMENT, covariate)
        print("CUPED covariate:", covariate, "variance reduction:", cuped["variance_reduction"], flush=True)

        # --- 6. uplift modeling on conversion (rare; visit is used for a quick model sanity check) ----
        train, holdout = data.train_holdout_split(df)
        p_hat = float(train[config.TREATMENT].mean())
        outcome = "conversion"
        mu1, mu0 = U.fit_t_learner(train, config.FEATURES, config.TREATMENT, outcome)
        t_up_hold = U.t_learner_uplift(mu1, mu0, holdout, config.FEATURES)
        x_model = U.fit_x_learner(train, config.FEATURES, config.TREATMENT, outcome, p_hat)
        x_up_hold = U.x_learner_uplift(x_model, holdout, config.FEATURES)
        s_model = U.fit_s_learner(train, config.FEATURES, config.TREATMENT, outcome)
        s_up_hold = U.s_learner_uplift(s_model, holdout, config.FEATURES, config.TREATMENT)

        y_h, t_h = holdout[outcome].to_numpy(float), holdout[config.TREATMENT].to_numpy(int)
        qinis, aucs = {}, {}
        for name, score in (("T-learner", t_up_hold), ("X-learner", x_up_hold), ("S-learner", s_up_hold)):
            q = U.qini_curve(score, y_h, t_h)
            qinis[name] = q
            aucs[name] = U.auuc(q)
            mlflow.log_metric(f"auuc_{name.split('-')[0].lower()}", aucs[name])
        rand_line = U.random_targeting_line(y_h, t_h)
        aucs["random"] = U.auuc(rand_line)
        best_name = max(("T-learner", "X-learner", "S-learner"), key=lambda k: aucs[k])
        best_score = {"T-learner": t_up_hold, "X-learner": x_up_hold, "S-learner": s_up_hold}[best_name]
        print("AUUC:", {k: round(v, 4) for k, v in aucs.items()}, "best:", best_name, flush=True)

        policy_values = U.evaluate_targeting_policies(y_h, t_h, best_score, p_hat)
        print("policy values:", {k: round(v["value"], 5) for k, v in policy_values.items()}, flush=True)

        # decile uplift table (a standard uplift-model diagnostic: does predicted rank order real effect?)
        order = np.argsort(-best_score)
        y_o, t_o = y_h[order], t_h[order]
        deciles = []
        for d in range(10):
            lo, hi = int(d * len(y_o) / 10), int((d + 1) * len(y_o) / 10)
            yt, tt = y_o[lo:hi], t_o[lo:hi]
            n_t, n_c = int(tt.sum()), int((1 - tt).sum())
            uplift = (yt[tt == 1].mean() if n_t else np.nan) - (yt[tt == 0].mean() if n_c else np.nan)
            deciles.append({"decile": d + 1, "n": hi - lo, "n_treat": n_t, "n_control": n_c,
                            "observed_uplift": float(uplift), "mean_predicted_uplift": float(best_score[order[lo:hi]].mean())})

        # --- 7. peeking simulation (synthetic, clearly separate from the real data above) --------------
        peek_naive = peeking.simulate_fixed_horizon_peeking(2000, 4000)
        peek_corrected = peeking.simulate_sequential_correction(2000, 4000)
        print("peeking:", peek_naive, peek_corrected, flush=True)

        results = {
            "dataset": {"n_sample": len(df), "n_total_source": 13_979_592, "treat_ratio": float(df[config.TREATMENT].mean()),
                       "nominal_treat_ratio": nominal_ratio, "conversion_rate": float(df.conversion.mean()),
                       "visit_rate": float(df.visit.mean())},
            "srm": {"observed": srm.observed, "expected_ratio": srm.expected_ratio, "chi2": srm.chi2,
                   "p_value": srm.p_value, "flagged": srm.flagged},
            "balance": bal.to_dict(orient="records"),
            "ate": ate, "required_n_per_arm_10pct_mde": n_needed,
            "cuped": {**cuped, "covariate": covariate},
            "uplift": {"n_train": len(train), "n_holdout": len(holdout), "propensity": p_hat,
                      "auuc": aucs, "best_model": best_name, "decile_table": deciles,
                      "policy_values": policy_values},
            "peeking": {"naive": peek_naive, "corrected": peek_corrected},
        }
        (config.ARTIFACTS / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
        for name, q in qinis.items():
            q.assign(model=name).to_parquet(config.ARTIFACTS / f"qini_{name.split('-')[0].lower()}.parquet")
        rand_line.to_parquet(config.ARTIFACTS / "qini_random.parquet")
        holdout.assign(uplift_score=best_score).sample(min(20000, len(holdout)), random_state=config.SEED) \
            .to_parquet(config.ARTIFACTS / "holdout_scored_sample.parquet")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
