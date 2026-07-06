from __future__ import annotations

from typing import Dict

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob),
    }

