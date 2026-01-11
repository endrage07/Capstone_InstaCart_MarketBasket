import pandas as pd
from .config import RAW_DIR

def load_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    return pd.read_csv(path)

def load_instacart():
    orders = load_csv("orders.csv")
    order_products_prior = load_csv("order_products__prior.csv")
    products = load_csv("products.csv")
    aisles = load_csv("aisles.csv")
    departments = load_csv("departments.csv")
    return orders, order_products_prior, products, aisles, departments
