"""Criteo uplift-modeling benchmark (Diemert et al., 2018): a real randomized incrementality test,
13,979,592 rows. Downloaded on demand; a reproducible random subsample is cached as parquet."""

from __future__ import annotations

import gzip
import shutil
import urllib.request

import numpy as np
import pandas as pd

from . import config


def download() -> None:
    if config.RAW_GZ.exists():
        return
    config.DATA.mkdir(parents=True, exist_ok=True)
    tmp = config.RAW_GZ.with_suffix(".tmp")
    with urllib.request.urlopen(config.SOURCE_URL, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(config.RAW_GZ)


def sample_path(n: int, seed: int):
    return config.DATA / f"sample_{n}_{seed}.parquet"


def load_sample(n: int = config.N_SAMPLE, seed: int = config.SEED) -> pd.DataFrame:
    """A reproducible random subsample of the full 13.98M-row file, cached as parquet (keyed by n and seed)
    after the first call.

    Reservoir-free approach: stream the gzip file in chunks, keep every row with independent probability
    n / N_total (N_total is the file's known row count), which gives an unbiased simple random sample
    without loading 14M rows into memory."""
    path = sample_path(n, seed)
    if path.exists():
        return pd.read_parquet(path)
    download()
    rng = np.random.default_rng(seed)
    n_total = 13_979_592
    p_keep = n / n_total
    chunks = []
    with gzip.open(config.RAW_GZ, "rt") as f:
        reader = pd.read_csv(f, chunksize=200_000)
        for chunk in reader:
            mask = rng.random(len(chunk)) < p_keep
            if mask.any():
                chunks.append(chunk[mask])
    out = pd.concat(chunks, ignore_index=True)
    out["exposure"] = out["exposure"].astype(np.int8)
    out[config.TREATMENT] = out[config.TREATMENT].astype(np.int8)
    for c in config.OUTCOMES:
        out[c] = out[c].astype(np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def train_holdout_split(df: pd.DataFrame, frac: float = config.UPLIFT_HOLDOUT_FRAC, seed: int = config.SEED):
    """Split for uplift modeling: `train` fits the model, `holdout` is untouched until policy evaluation.
    ATE / SRM / balance diagnostics in this project always run on the FULL sample, not this split — the
    split exists only so the targeting policy is graded on data its own model never saw."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < frac
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
