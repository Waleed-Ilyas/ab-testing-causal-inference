"""Offline tests on small synthetic data: no dataset download, no LightGBM training beyond tiny fits."""

import numpy as np
import pandas as pd
import pytest

from abtest import peeking, stats
from abtest import uplift as U


def _rct(n=4000, treat_ratio=0.5, base_rate=0.05, true_lift=0.02, seed=0, confound_feature=False):
    rng = np.random.default_rng(seed)
    treat = (rng.random(n) < treat_ratio).astype(int)
    f0 = rng.normal(0, 1, n)
    p = base_rate + true_lift * treat + (0.03 * f0 if confound_feature else 0)
    y = (rng.random(n) < np.clip(p, 0, 1)).astype(int)
    return pd.DataFrame({"treatment": treat, "conversion": y, "f0": f0,
                         "f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})


# ---------------------------------------------------------------- SRM
def test_srm_flags_a_broken_split_but_not_a_clean_one():
    rng = np.random.default_rng(1)
    clean = pd.Series((rng.random(20000) < 0.85).astype(int))
    assert not stats.srm_check(clean, 0.85).flagged
    broken = pd.Series((rng.random(20000) < 0.70).astype(int))
    assert stats.srm_check(broken, 0.85).flagged


# ---------------------------------------------------------------- balance
def test_balance_flags_a_confounded_feature_not_a_clean_one():
    rng = np.random.default_rng(2)
    n = 6000
    treat = (rng.random(n) < 0.5).astype(int)
    clean = rng.normal(0, 1, n)
    confounded = rng.normal(0, 1, n) + 0.5 * treat  # correlated with assignment: broken randomization
    df = pd.DataFrame({"treatment": treat, "clean": clean, "confounded": confounded})
    bal = stats.balance_table(df, ["clean", "confounded"], "treatment").set_index("feature")
    assert not bal.loc["clean", "flagged"]
    assert bal.loc["confounded", "flagged"]


def test_smd_zero_for_identical_distributions():
    x = pd.Series(np.arange(100.0))
    t = pd.Series([1] * 50 + [0] * 50)
    x_shuffled = x.sample(frac=1, random_state=3).reset_index(drop=True)
    assert abs(stats.standardized_mean_diff(x_shuffled, t)) < 0.3  # random split, no systematic difference


# ---------------------------------------------------------------- two-proportion test
def test_two_proportion_test_recovers_a_known_effect():
    df = _rct(n=40000, base_rate=0.05, true_lift=0.02, seed=4)
    res = stats.two_proportion_test(df.conversion, df.treatment)
    assert res.ate == pytest.approx(0.02, abs=0.01)
    assert res.ci[0] < 0.02 < res.ci[1]
    assert res.p_value < 0.01  # a real 40% relative lift at n=40k should be easily significant


def test_two_proportion_test_null_effect_is_usually_not_significant():
    df = _rct(n=3000, true_lift=0.0, seed=5)
    res = stats.two_proportion_test(df.conversion, df.treatment)
    assert res.ci[0] < 0 < res.ci[1]  # the true zero effect should be inside a 95% CI


def test_required_sample_size_shrinks_for_a_bigger_effect():
    small_mde = stats.required_sample_size(0.05, 0.05)
    big_mde = stats.required_sample_size(0.05, 0.30)
    assert big_mde < small_mde


# ---------------------------------------------------------------- CUPED
def test_cuped_is_unbiased_and_reduces_variance_with_a_correlated_covariate():
    rng = np.random.default_rng(6)
    n = 5000
    treat = (rng.random(n) < 0.5).astype(int)
    covariate = rng.normal(0, 1, n)
    y = 0.1 + 0.02 * treat + 0.5 * covariate + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"treatment": treat, "y": y, "cov": covariate})
    out = stats.cuped_ate(df, "y", "treatment", "cov")
    assert out["cuped_ate"] == pytest.approx(out["raw_ate"], abs=0.02)  # unbiased: point estimate barely moves
    assert out["variance_reduction"] > 0.5  # the covariate explains most of the outcome's variance
    assert out["cuped_se"] < out["raw_se"]


def test_cuped_with_useless_covariate_barely_changes_anything():
    df = _rct(n=4000, seed=7)
    out = stats.cuped_ate(df, "conversion", "treatment", "f1")  # f1 is unrelated to the outcome by construction
    assert out["variance_reduction"] < 0.05


# ---------------------------------------------------------------- uplift
def test_t_learner_ranks_a_planted_heterogeneous_effect():
    rng = np.random.default_rng(8)
    n = 8000
    treat = (rng.random(n) < 0.5).astype(int)
    segment = rng.random(n) < 0.5  # segment True has a real, bigger treatment effect
    f0 = segment.astype(float) + rng.normal(0, 0.1, n)
    p = 0.05 + treat * np.where(segment, 0.15, 0.0)
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"treatment": treat, "conversion": y, "f0": f0, "f1": rng.normal(0, 1, n)})
    mu1, mu0 = U.fit_t_learner(df, ["f0", "f1"], "treatment", "conversion")
    up = U.t_learner_uplift(mu1, mu0, df, ["f0", "f1"])
    assert up[segment].mean() > up[~segment].mean() + 0.05  # the model finds the planted segment


def test_qini_curve_monotone_frac_and_matches_full_population_formula():
    rng = np.random.default_rng(9)
    n = 3000
    y = (rng.random(n) < 0.05).astype(float)
    t = (rng.random(n) < 0.5).astype(int)
    score = rng.normal(0, 1, n)
    q = U.qini_curve(score, y, t, n_points=10)
    assert list(q.frac) == sorted(q.frac)
    n_t, n_c = t.sum(), (1 - t).sum()
    expected_full = y[t == 1].sum() - y[t == 0].sum() * (n_t / n_c)
    assert q.gain.iloc[-1] == pytest.approx(expected_full)


def test_auuc_of_random_line_is_half_its_endpoint():
    line = pd.DataFrame({"frac": np.linspace(0.01, 1, 100), "gain": np.linspace(0.01, 1, 100) * 10})
    assert U.auuc(line) == pytest.approx(5.0, rel=0.05)


def test_ipw_policy_value_recovers_arm_means_for_all_or_none():
    rng = np.random.default_rng(10)
    n = 5000
    t = (rng.random(n) < 0.5).astype(int)
    y = rng.random(n) < (0.05 + 0.03 * t)
    v_all = U.ipw_policy_value(y.astype(float), t, 0.5, np.ones(n, bool))
    v_none = U.ipw_policy_value(y.astype(float), t, 0.5, np.zeros(n, bool))
    assert v_all["value"] == pytest.approx(y[t == 1].mean(), abs=0.01)
    assert v_none["value"] == pytest.approx(y[t == 0].mean(), abs=0.01)


def test_evaluate_targeting_policies_top_uplift_beats_random_with_real_heterogeneity():
    rng = np.random.default_rng(11)
    n = 6000
    t = (rng.random(n) < 0.5).astype(int)
    segment = rng.random(n) < 0.2  # a small high-uplift segment
    p = 0.05 + t * np.where(segment, 0.3, 0.0)
    y = (rng.random(n) < p).astype(float)
    perfect_score = segment.astype(float)  # a perfect (oracle) uplift score
    out = U.evaluate_targeting_policies(y, t, perfect_score, 0.5, fractions=(0.2,))
    assert out["top_20pct_uplift"]["value"] > out["random_20pct"]["value"]


# ---------------------------------------------------------------- peeking
def test_peeking_inflates_false_positive_rate_above_nominal_alpha():
    out = peeking.simulate_fixed_horizon_peeking(600, 3000, true_p=0.05, n_looks=15, alpha=0.05, seed=1)
    assert out["false_positive_rate_with_peeking"] > out["false_positive_rate_final_only"]
    assert out["false_positive_rate_with_peeking"] > 0.08  # meaningfully above the nominal 5%


def test_final_only_false_positive_rate_is_near_nominal_alpha():
    out = peeking.simulate_fixed_horizon_peeking(2500, 3000, true_p=0.05, n_looks=10, alpha=0.05, seed=2)
    assert out["false_positive_rate_final_only"] == pytest.approx(0.05, abs=0.03)


def test_sequential_correction_controls_the_false_positive_rate():
    out = peeking.simulate_sequential_correction(1500, 3000, true_p=0.05, n_looks=15, alpha=0.05, seed=3)
    assert out["false_positive_rate_corrected"] < 0.08
