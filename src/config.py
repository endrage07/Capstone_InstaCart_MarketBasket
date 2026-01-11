from pathlib import Path

# -----------------------------
# Reproducibility
# -----------------------------
RANDOM_SEED = 42

# -----------------------------
# Dataset configuration
# -----------------------------
KAGGLE_DATASET = "psparks/instacart-market-basket-analysis"

# Optional local fallback (not committed)
RAW_DIR = Path("data/raw")

# -----------------------------
# Feature engineering params
# -----------------------------
CAP_DAYS = 30

# -----------------------------
# Modeling controls
# -----------------------------
MAX_ROWS = 500_000        # cap for full training table
TUNE_ROWS = 50_000        # subsample for hyperparameter tuning
