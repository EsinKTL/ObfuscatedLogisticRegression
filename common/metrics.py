from enum import Enum

import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)


class Metric(Enum):
    RECALL            = "recall"
    PRECISION         = "precision"
    F_MEASURE         = "f_measure"
    BALANCED_ACCURACY = "balanced_accuracy"
    AUC_ROC           = "auc_roc"
    AUC_PR            = "auc_pr"


# TODO: threshold=0.5 is not correct for every metric, we need to revisit notes from the labs of AML.
def evaluate(y_true, y_prob, metric=None, threshold=0.5):
    if metric is None:
        metric = Metric.F_MEASURE

    y_pred = (y_prob >= threshold).astype(int)

    table = {
        Metric.RECALL:            lambda: recall_score(y_true, y_pred, zero_division=0),
        Metric.PRECISION:         lambda: precision_score(y_true, y_pred, zero_division=0),
        Metric.F_MEASURE:         lambda: f1_score(y_true, y_pred, zero_division=0),
        Metric.BALANCED_ACCURACY: lambda: balanced_accuracy_score(y_true, y_pred),
        Metric.AUC_ROC:           lambda: roc_auc_score(y_true, y_prob),
        Metric.AUC_PR:            lambda: average_precision_score(y_true, y_prob),
    }

    return table[metric]()


METRIC_LABELS = {
    Metric.RECALL:            "Recall",
    Metric.PRECISION:         "Precision",
    Metric.F_MEASURE:         "F1",
    Metric.BALANCED_ACCURACY: "Balanced Accuracy",
    Metric.AUC_ROC:           "ROC AUC",
    Metric.AUC_PR:            "PR AUC",
}


def print_evaluation(name, y_true, y_proba, threshold=0.5):
    print(f"[{name}]")
    for metric in Metric:
        score = evaluate(y_true, y_proba, metric, threshold)
        print(f"  {METRIC_LABELS[metric]:<22}: {score:.4f}")
    print()
