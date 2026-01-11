import pandas as pd

from .config import KAGGLE_DATASET, RAW_DIR


def _load_from_kagglehub():
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)

    return {
        "orders": pd.read_csv(f"{path}/orders.csv"),
        "order_products_prior": pd.read_csv(f"{path}/order_products__prior.csv"),
        "order_products_train": pd.read_csv(f"{path}/order_products__train.csv"),
        "products": pd.read_csv(f"{path}/products.csv"),
        "aisles": pd.read_csv(f"{path}/aisles.csv"),
        "departments": pd.read_csv(f"{path}/departments.csv"),
    }


def _load_from_local():
    return {
        "orders": pd.read_csv(RAW_DIR / "orders.csv"),
        "order_products_prior": pd.read_csv(RAW_DIR / "order_products__prior.csv"),
        "order_products_train": pd.read_csv(RAW_DIR / "order_products__train.csv"),
        "products": pd.read_csv(RAW_DIR / "products.csv"),
        "aisles": pd.read_csv(RAW_DIR / "aisles.csv"),
        "departments": pd.read_csv(RAW_DIR / "departments.csv"),
    }


def load_instacart():
    """
    Load Instacart dataset using KaggleHub (preferred).
    Falls back to local CSVs if KaggleHub is unavailable.
    """
    try:
        return _load_from_kagglehub()
    except Exception:
        return _load_from_local()
