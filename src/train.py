import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED


def build_training_table(
    order_products_train: pd.DataFrame,
    orders: pd.DataFrame,
    product_features: pd.DataFrame,
    user_features: pd.DataFrame,
) -> pd.DataFrame:
    train = order_products_train.merge(orders, on="order_id", how="left")
    train = train.merge(product_features, on="product_id", how="left")
    train = train.merge(user_features, on="user_id", how="left")
    return train


def split_xy(train: pd.DataFrame):
    y = train["reordered"]

    X = train.drop(
        columns=[
            "reordered",
            "order_id",
            "user_id",
            "product_id",
        ],
        errors="ignore",
    )

    return X, y


def train_baseline_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test
