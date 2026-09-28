"""Illustrative simulation: peeking at a fixed-horizon z-test repeatedly inflates the false-positive rate,
and a sequential (always-valid) alternative controls it. Every number in this module comes from simulated
data with a KNOWN true null effect, generated and clearly labelled here — it stands apart from the real
Criteo experiment analysed elsewhere in this project, which is never touched by this module."""

from __future__ import annotations

import numpy as np
from scipy import stats


def simulate_fixed_horizon_peeking(n_experiments: int, n_per_arm: int, true_p: float = 0.05,
                                   n_looks: int = 20, alpha: float = 0.05, seed: int = 42) -> dict:
    """Run `n_experiments` A/A tests (both arms have the SAME true conversion rate, so any significant
    result is a false positive by construction). Each experiment is "peeked at" `n_looks` times as data
    accumulates, stopping and declaring a "win" the first time the naive two-sided z-test crosses alpha.
    Reports the false-positive rate from (a) looking only at the final sample and (b) stopping at the
    first significant peek."""
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_arm / n_looks, n_per_arm, n_looks).astype(int)
    final_reject, ever_reject, stop_at = 0, 0, []
    for _ in range(n_experiments):
        t = rng.random(n_per_arm) < true_p
        c = rng.random(n_per_arm) < true_p
        stopped = None
        for n in checkpoints:
            pt, pc = t[:n].mean(), c[:n].mean()
            se = np.sqrt(pt * (1 - pt) / n + pc * (1 - pc) / n)
            z = (pt - pc) / se if se > 0 else 0.0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            if p < alpha and stopped is None:
                stopped = int(n)
        if stopped is not None:
            ever_reject += 1
            stop_at.append(stopped)
        pt, pc = t.mean(), c.mean()
        se = np.sqrt(pt * (1 - pt) / n_per_arm + pc * (1 - pc) / n_per_arm)
        z = (pt - pc) / se if se > 0 else 0.0
        if 2 * (1 - stats.norm.cdf(abs(z))) < alpha:
            final_reject += 1
    return {"n_experiments": n_experiments, "n_looks": n_looks, "nominal_alpha": alpha,
            "false_positive_rate_final_only": final_reject / n_experiments,
            "false_positive_rate_with_peeking": ever_reject / n_experiments,
            "mean_stop_fraction": float(np.mean(stop_at) / n_per_arm) if stop_at else None}


def obrien_fleming_boundary(look: int, n_looks: int, alpha: float = 0.05) -> float:
    """A simple O'Brien-Fleming-style alpha-spending boundary: much stricter early, relaxing toward the
    nominal alpha at the final look, so the total false-positive rate across all looks stays at alpha."""
    frac = look / n_looks
    z_final = stats.norm.ppf(1 - alpha / 2)
    z_boundary = z_final / np.sqrt(frac)
    return float(2 * (1 - stats.norm.cdf(z_boundary)))


def simulate_sequential_correction(n_experiments: int, n_per_arm: int, true_p: float = 0.05,
                                   n_looks: int = 20, alpha: float = 0.05, seed: int = 43) -> dict:
    """The same A/A peeking experiment, but each look is tested against an O'Brien-Fleming boundary instead
    of the fixed alpha, showing the false-positive rate drop back toward the nominal level."""
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_arm / n_looks, n_per_arm, n_looks).astype(int)
    boundaries = [obrien_fleming_boundary(i + 1, n_looks, alpha) for i in range(n_looks)]
    ever_reject = 0
    for _ in range(n_experiments):
        t = rng.random(n_per_arm) < true_p
        c = rng.random(n_per_arm) < true_p
        for n, b in zip(checkpoints, boundaries, strict=True):
            pt, pc = t[:n].mean(), c[:n].mean()
            se = np.sqrt(pt * (1 - pt) / n + pc * (1 - pc) / n)
            z = (pt - pc) / se if se > 0 else 0.0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            if p < b:
                ever_reject += 1
                break
    return {"n_experiments": n_experiments, "n_looks": n_looks, "nominal_alpha": alpha,
            "false_positive_rate_corrected": ever_reject / n_experiments}
