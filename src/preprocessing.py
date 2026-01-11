import pandas as pd

from .config import CAP_DAYS


# -----------------------------
# Orders preprocessing
# -----------------------------
def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()

    orders["days_since_prior_order"] = (
        orders["days_since_prior_order"]
        .fillna(0)
        .clip(upper=CAP_DAYS)
    )

    orders["is_first_order"] = (orders["order_number"] == 1).astype(int)
    return orders


# -----------------------------
# Prior merged table
# -----------------------------
def build_prior_merged(
    orders: pd.DataFrame,
    order_products_prior: pd.DataFrame,
    products: pd.DataFrame,
    aisles: pd.DataFrame,
    departments: pd.DataFrame,
) -> pd.DataFrame:
    prior = order_products_prior.merge(
        orders, on="order_id", how="left"
    )

    prior = prior.merge(products, on="product_id", how="left")
    prior = prior.merge(aisles, on="aisle_id", how="left")
    prior = prior.merge(departments, on="department_id", how="left")

    return prior


# -----------------------------
# Feature engineering
# -----------------------------
def build_product_features(prior: pd.DataFrame) -> pd.DataFrame:
    return (
        prior.groupby("product_id")
        .agg(
            prod_orders=("order_id", "count"),
            prod_reorder_rate=("reordered", "mean"),
        )
        .reset_index()
    )


def build_user_features(prior: pd.DataFrame) -> pd.DataFrame:
    return (
        prior.groupby("user_id")
        .agg(
            user_orders=("order_id", "nunique"),
            user_avg_basket=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )
