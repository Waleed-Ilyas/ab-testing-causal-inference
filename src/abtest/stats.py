"""Core A/B-test inference: sample-ratio mismatch, covariate balance, the treatment effect with a proper
interval, and CUPED/regression-adjustment variance reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ sample ratio mismatch
@dataclass
class SRMResult:
    observed: dict[str, int]
    expected_ratio: float
    chi2: float
    p_value: float
    flagged: bool


def srm_check(treatment: pd.Series, expected_treat_ratio: float, alpha: float = 0.001) -> SRMResult:
    """Chi-square goodness-of-fit test: does the observed split match the nominal randomization ratio?
    A very low alpha (0.001, the standard SRM convention) is used because false alarms here are expensive:
    every metric below is meaningless if the randomization itself is broken."""
    n = len(treatment)
    obs_t = int(treatment.sum())
    obs_c = n - obs_t
    exp_t, exp_c = n * expected_treat_ratio, n * (1 - expected_treat_ratio)
    chi2, p = stats.chisquare([obs_c, obs_t], [exp_c, exp_t])
    return SRMResult({"control": obs_c, "treatment": obs_t}, expected_treat_ratio, float(chi2), float(p), p < alpha)


# ------------------------------------------------------------------ covariate balance
def standardized_mean_diff(x: pd.Series, treat: pd.Series) -> float:
    """(mean_treated - mean_control) / pooled std. |SMD| > 0.1 is the usual "worth a look" convention."""
    a, b = x[treat == 1], x[treat == 0]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def balance_table(df: pd.DataFrame, features: list[str], treat_col: str) -> pd.DataFrame:
    rows = []
    for f in features:
        smd = standardized_mean_diff(df[f], df[treat_col])
        t_stat, p = stats.ttest_ind(df.loc[df[treat_col] == 1, f], df.loc[df[treat_col] == 0, f], equal_var=False)
        rows.append({"feature": f, "smd": smd, "t_p_value": float(p), "flagged": abs(smd) > 0.1})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ two-proportion test
@dataclass
class ATEResult:
    p_treat: float
    p_control: float
    ate: float          # absolute difference, treatment - control
    relative_lift: float
    se: float
    ci: tuple[float, float]
    z: float
    p_value: float
    n_treat: int
    n_control: int


def two_proportion_test(y: pd.Series, treat: pd.Series, alpha: float = 0.05) -> ATEResult:
    """Standard two-proportion z-test with a Wald CI on the difference (the textbook A/B-test calculation)."""
    t, c = y[treat == 1], y[treat == 0]
    n_t, n_c = len(t), len(c)
    p_t, p_c = float(t.mean()), float(c.mean())
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ate = p_t - p_c
    z = ate / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return ATEResult(p_t, p_c, ate, ate / p_c if p_c > 0 else float("nan"), float(se),
                     (ate - z_crit * se, ate + z_crit * se), float(z), float(p), n_t, n_c)


def required_sample_size(baseline_rate: float, mde_relative: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size for detecting a relative lift of `mde_relative` over `baseline_rate`
    (standard two-proportion power formula, equal arm sizes)."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde_relative)
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


# ------------------------------------------------------------------ CUPED / regression adjustment
def cuped_adjust(y: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED (Deng et al., 2013): remove the part of the outcome linearly predictable from a covariate
    that is independent of treatment assignment, so its expectation is unchanged but its variance shrinks.
    Returns (adjusted outcome, variance-reduction fraction)."""
    theta = np.cov(covariate, y, ddof=1)[0, 1] / np.var(covariate, ddof=1)
    adjusted = y - theta * (covariate - covariate.mean())
    reduction = 1 - adjusted.var(ddof=1) / y.var(ddof=1)
    return adjusted, float(reduction)


def cuped_ate(df: pd.DataFrame, outcome: str, treat_col: str, covariate: str, alpha: float = 0.05) -> dict:
    """ATE on the CUPED-adjusted outcome, with the variance-reduction fraction and how much narrower the
    CI got as a result. The point estimate should barely move (CUPED is unbiased); only its precision changes."""
    y, x, t = df[outcome].to_numpy(float), df[covariate].to_numpy(float), df[treat_col].to_numpy(int)
    adj, reduction = cuped_adjust(y, x)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    def _se_ci(v: np.ndarray) -> tuple[float, float, tuple[float, float]]:
        """Two-sample SE from actual sample variance per arm — correct for a binary OR continuous
        outcome, unlike a Bernoulli p(1-p)/n formula, which is wrong once CUPED makes the outcome
        continuous."""
        a, c = v[t == 1], v[t == 0]
        se = float(np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c)))
        d = float(a.mean() - c.mean())
        return d, se, (d - z_crit * se, d + z_crit * se)

    raw_ate, raw_se, raw_ci = _se_ci(y)
    ate, se, ci = _se_ci(adj)
    return {"raw_ate": raw_ate, "raw_ci": raw_ci, "raw_se": raw_se, "cuped_ate": ate, "cuped_ci": ci,
            "cuped_se": se, "variance_reduction": reduction, "ci_width_reduction": 1 - (2 * z_crit * se) / (raw_ci[1] - raw_ci[0])}


def bootstrap_ate_ci(y: np.ndarray, treat: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the ATE, resampling within each arm — a distribution-free cross-check
    on the z-test's Wald interval."""
    rng = np.random.default_rng(seed)
    t_idx, c_idx = np.flatnonzero(treat == 1), np.flatnonzero(treat == 0)
    ests = np.empty(n_boot)
    for i in range(n_boot):
        ests[i] = y[rng.choice(t_idx, len(t_idx))].mean() - y[rng.choice(c_idx, len(c_idx))].mean()
    return float(np.percentile(ests, 100 * alpha / 2)), float(np.percentile(ests, 100 * (1 - alpha / 2)))
