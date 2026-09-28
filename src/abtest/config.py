"""Paths and experiment constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
SOURCE_URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
RAW_GZ = DATA / "criteo-uplift.csv.gz"
SAMPLE_PARQUET = None  # unused; see data.sample_path(n)

SEED = 42
FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOMES = ["visit", "conversion"]
N_SAMPLE = 1_500_000   # rows kept from the 13.98M-row source (compute-bounded, stratified by treatment)
ALPHA = 0.05           # significance level used throughout

# Uplift modeling: hold out a slice never used for ATE inference, so targeting-policy value is out-of-sample
UPLIFT_HOLDOUT_FRAC = 0.4
