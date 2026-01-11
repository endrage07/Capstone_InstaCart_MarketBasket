from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Train/validation split with reproducibility."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )
