"""Heterogeneous treatment effects: S/T/X-learners, the Qini curve, and unbiased off-policy value
estimation for a "treat the top-uplift users" targeting policy."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config

LGB_PARAMS = {"n_estimators": 200, "num_leaves": 31, "min_child_samples": 100, "learning_rate": 0.05,
             "random_state": config.SEED, "verbose": -1, "n_jobs": 4}


# ------------------------------------------------------------------ learners
def fit_s_learner(df: pd.DataFrame, features: list[str], treat_col: str, outcome: str) -> lgb.LGBMClassifier:
    """One model, treatment as a feature. Uplift = f(x, treat=1) - f(x, treat=0)."""
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(df[[*features, treat_col]], df[outcome])
    return m


def s_learner_uplift(model, df: pd.DataFrame, features: list[str], treat_col: str) -> np.ndarray:
    x1 = df[features].assign(**{treat_col: 1})[[*features, treat_col]]
    x0 = df[features].assign(**{treat_col: 0})[[*features, treat_col]]
    return model.predict_proba(x1)[:, 1] - model.predict_proba(x0)[:, 1]


def fit_t_learner(df: pd.DataFrame, features: list[str], treat_col: str, outcome: str):
    """Two independent models, one per arm. Uplift = mu1(x) - mu0(x)."""
    mu1 = lgb.LGBMClassifier(**LGB_PARAMS).fit(df.loc[df[treat_col] == 1, features], df.loc[df[treat_col] == 1, outcome])
    mu0 = lgb.LGBMClassifier(**LGB_PARAMS).fit(df.loc[df[treat_col] == 0, features], df.loc[df[treat_col] == 0, outcome])
    return mu1, mu0


def t_learner_uplift(mu1, mu0, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return mu1.predict_proba(df[features])[:, 1] - mu0.predict_proba(df[features])[:, 1]


def fit_x_learner(df: pd.DataFrame, features: list[str], treat_col: str, outcome: str, propensity: float):
    """Künzel et al. (2019). Stage 1 = a T-learner. Stage 2: impute each unit's individual effect using the
    OTHER arm's model, then regress those imputed effects on X within each arm. Stage 3: blend the two
    stage-2 models by the (known, since randomized) propensity score."""
    mu1, mu0 = fit_t_learner(df, features, treat_col, outcome)
    treated, control = df[df[treat_col] == 1], df[df[treat_col] == 0]
    d1 = treated[outcome].to_numpy(float) - mu0.predict_proba(treated[features])[:, 1]
    d0 = mu1.predict_proba(control[features])[:, 1] - control[outcome].to_numpy(float)
    tau1 = lgb.LGBMRegressor(**LGB_PARAMS).fit(treated[features], d1)
    tau0 = lgb.LGBMRegressor(**LGB_PARAMS).fit(control[features], d0)
    return {"tau1": tau1, "tau0": tau0, "propensity": propensity}


def x_learner_uplift(x_model: dict, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    e = x_model["propensity"]
    t1 = x_model["tau1"].predict(df[features])
    t0 = x_model["tau0"].predict(df[features])
    return e * t0 + (1 - e) * t1  # weight each arm's estimate by the OTHER arm's propensity


# ------------------------------------------------------------------ Qini curve / AUUC
def qini_curve(uplift_score: np.ndarray, y: np.ndarray, treat: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    """Sort by predicted uplift, descending. At each fraction of the population targeted, the incremental
    gain is `treated_conversions_in_top_k - control_conversions_in_top_k * (n_treated_in_top_k / n_control_in_top_k)`
    — the standard Qini-curve construction (Radcliffe, 2007), which corrects for arm-size imbalance within
    the slice so it is valid even when treat/control are not 50/50."""
    order = np.argsort(-uplift_score)
    y, treat = y[order], treat[order]
    n = len(y)
    fracs = np.linspace(1 / n_points, 1.0, n_points)
    rows = []
    for frac in fracs:
        k = max(1, int(round(frac * n)))
        yt, tt = y[:k], treat[:k]
        n_t, n_c = int(tt.sum()), int((1 - tt).sum())
        y_t, y_c = float(yt[tt == 1].sum()), float(yt[tt == 0].sum())
        gain = y_t - y_c * (n_t / n_c) if n_c > 0 else np.nan
        rows.append({"frac": frac, "n_targeted": k, "gain": gain})
    return pd.DataFrame(rows)


def auuc(qini: pd.DataFrame) -> float:
    """Area under the Qini curve, trapezoidal, in units of (fraction targeted) x (incremental conversions)."""
    return float(np.trapezoid(qini.gain, qini.frac))


def random_targeting_line(y: np.ndarray, treat: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    """The Qini curve a model with zero information would produce in expectation: a straight line from
    (0, 0) to (1, total incremental gain)."""
    n_t, n_c = int(treat.sum()), int((1 - treat).sum())
    total = float(y[treat == 1].sum() - y[treat == 0].sum() * (n_t / n_c)) if n_c > 0 else np.nan
    fracs = np.linspace(1 / n_points, 1.0, n_points)
    return pd.DataFrame({"frac": fracs, "gain": fracs * total})


# ------------------------------------------------------------------ off-policy value
def ipw_policy_value(y: np.ndarray, treat: np.ndarray, propensity: float, policy_treat: np.ndarray) -> dict:
    """Unbiased estimate of E[outcome] if everyone were treated according to `policy_treat` (a boolean array
    of who the policy would target), using inverse-propensity weighting on the REAL randomized outcomes:
    each unit whose actual assignment matches the policy contributes y / P(that assignment); units where the
    policy disagrees with what was actually assigned contribute nothing. Valid because propensity is known
    exactly (randomization), not estimated."""
    assign_prob = np.where(policy_treat, propensity, 1 - propensity)
    matches = (treat == policy_treat.astype(int))
    contrib = np.where(matches, y / assign_prob, 0.0)
    value = float(contrib.mean())
    se = float(contrib.std(ddof=1) / np.sqrt(len(contrib)))
    return {"value": value, "se": se, "ci": (value - 1.96 * se, value + 1.96 * se), "n": len(y),
            "n_targeted": int(policy_treat.sum())}


def evaluate_targeting_policies(y: np.ndarray, treat: np.ndarray, uplift_score: np.ndarray, propensity: float,
                                fractions=(0.1, 0.2, 0.3, 0.5)) -> dict:
    """Compare 'treat everyone', 'treat no one', 'treat a random X%', and 'treat the top-X%-by-predicted-uplift'
    — all evaluated on the SAME held-out data with the SAME unbiased estimator, so they are directly comparable."""
    out = {"treat_all": ipw_policy_value(y, treat, propensity, np.ones(len(y), bool)),
          "treat_none": ipw_policy_value(y, treat, propensity, np.zeros(len(y), bool))}
    order = np.argsort(-uplift_score)
    rng = np.random.default_rng(config.SEED)
    for frac in fractions:
        k = int(round(frac * len(y)))
        top = np.zeros(len(y), bool)
        top[order[:k]] = True
        out[f"top_{int(frac * 100)}pct_uplift"] = ipw_policy_value(y, treat, propensity, top)
        rand_mask = np.zeros(len(y), bool)
        rand_mask[rng.choice(len(y), k, replace=False)] = True
        out[f"random_{int(frac * 100)}pct"] = ipw_policy_value(y, treat, propensity, rand_mask)
    return out
