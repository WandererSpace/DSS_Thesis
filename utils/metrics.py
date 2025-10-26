import json
import numpy as np
from pathlib import Path
from sklearn import metrics as skm

def classification_metrics(y_true, y_pred, y_proba=None, sample_weight=None):
    res = {
        "accuracy": skm.accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "precision": skm.precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
        "recall": skm.recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
        "f1": skm.f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            res["roc_auc"] = skm.roc_auc_score(y_true, y_proba, sample_weight=sample_weight)
        except Exception:
            pass
    return res

def regression_metrics(y_true, y_pred, sample_weight=None):
    mse = skm.mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    return {
        "mae": skm.mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
        "rmse": float(np.sqrt(mse)),
        "r2": skm.r2_score(y_true, y_pred, sample_weight=sample_weight),
    }

def save_json(d: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
