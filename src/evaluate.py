from sklearn.metrics import roc_auc_score, classification_report

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y, y_proba),
        "report": classification_report(y, y_pred)
    }
    return metrics
