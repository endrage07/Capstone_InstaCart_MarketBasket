from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from .config import MODELS_DIR, RANDOM_SEED

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, name="model.pkl"):
    MODELS_DIR.mkdir(exist_ok=True)
    dump(model, MODELS_DIR / name)
