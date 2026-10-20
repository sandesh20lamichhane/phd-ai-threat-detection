"""Detection metrics required by the experimental protocol.

Accuracy is deliberately absent: on CICIDS2017 a model that always predicts "benign"
already scores above 80%. Report these instead (docs/protocol/experimental_protocol.md).
"""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_recall(y_true, y_pred, labels=None):
    labels = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
    scores = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {str(k): float(v) for k, v in zip(labels, scores)}


def pr_auc(y_true, scores, classes=None):
    """Average precision. Binary: `scores` is P(attack). Multiclass: (n, k) matrix, macro one-vs-rest."""
    scores = np.asarray(scores)
    if scores.ndim == 1:
        return float(average_precision_score(y_true, scores))
    classes = classes if classes is not None else np.unique(y_true)
    y_bin = label_binarize(y_true, classes=classes)
    return float(average_precision_score(y_bin, scores, average="macro"))


def fpr_at_detection_rate(y_true, attack_scores, target_tpr=0.95):
    """False-positive rate at the first threshold reaching `target_tpr` (binary, attack = 1)."""
    fpr, tpr, _ = roc_curve(y_true, attack_scores)
    idx = np.searchsorted(tpr, target_tpr, side="left")
    return float(fpr[min(idx, len(fpr) - 1)])


def detection_lead_time(alert_times, compromise_times):
    """Lead time per incident: compromise time minus the first alert at or before it.

    alert_times: dict incident -> sorted list of alert timestamps (seconds)
    compromise_times: dict incident -> timestamp of the red-team event
    Returns dict incident -> lead time in seconds (positive = early warning; None = missed).
    """
    out = {}
    for incident, t_c in compromise_times.items():
        early = [t for t in alert_times.get(incident, []) if t <= t_c]
        out[incident] = float(t_c - early[0]) if early else None
    return out


def detection_report(y_true, y_pred, attack_scores=None, benign_label="BENIGN", target_tpr=0.95):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rep = {"macro_f1": macro_f1(y_true, y_pred), "per_class_recall": per_class_recall(y_true, y_pred)}
    if attack_scores is not None:
        y_bin = (y_true != benign_label).astype(int)
        rep["pr_auc"] = pr_auc(y_bin, attack_scores)
        rep[f"fpr_at_tpr_{target_tpr}"] = fpr_at_detection_rate(y_bin, attack_scores, target_tpr)
    return rep
