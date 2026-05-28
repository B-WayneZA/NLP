"""

Metrics computation for multi-label emotion classification.
Designed for use with HuggingFace Trainer's compute_metrics callback.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    jaccard_score,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def make_compute_metrics(threshold: float = 0.5, zero_division: int = 0):
    """
    Factory that returns a compute_metrics function for HuggingFace Trainer.
    Closure captures threshold and zero_division settings.

    Args:
        threshold    : Probability threshold for binary prediction (default 0.5)
        zero_division: Value when denominator is zero (default 0)

    Returns:
        compute_metrics: callable suitable for TrainingArguments
    """

    def compute_metrics(eval_pred) -> dict:
        """
        Called by Trainer at evaluation time.

        Args:
            eval_pred: EvalPrediction namedtuple with .predictions and .label_ids

        Returns:
            Dictionary of metric name → float value
        """
        logits, labels = eval_pred.predictions, eval_pred.label_ids

        # Logits → probabilities → binary predictions
        probs = sigmoid(logits)
        y_pred = (probs >= threshold).astype(int)
        y_true = labels.astype(int)

        return {
            "f1_macro":        f1_score(y_true, y_pred, average="macro",  zero_division=zero_division),
            "f1_micro":        f1_score(y_true, y_pred, average="micro",  zero_division=zero_division),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=zero_division),
            "recall_macro":    recall_score(y_true, y_pred, average="macro",    zero_division=zero_division),
            "hamming_loss":    hamming_loss(y_true, y_pred),
            "jaccard_macro":   jaccard_score(y_true, y_pred, average="macro",   zero_division=zero_division),
        }

    return compute_metrics


def compute_metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: int = 0,
) -> dict:
    """
    Compute metrics directly from prediction arrays.
    Used for final test-set evaluation outside the Trainer loop.

    Args:
        y_true: Ground truth binary matrix (n_samples, n_labels)
        y_pred: Predicted binary matrix  (n_samples, n_labels)
        zero_division: Value when denominator is zero

    Returns:
        Dictionary of metric name → float value
    """
    return {
        "f1_macro":        f1_score(y_true, y_pred, average="macro",  zero_division=zero_division),
        "f1_micro":        f1_score(y_true, y_pred, average="micro",  zero_division=zero_division),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=zero_division),
        "recall_macro":    recall_score(y_true, y_pred, average="macro",    zero_division=zero_division),
        "hamming_loss":    hamming_loss(y_true, y_pred),
        "jaccard_macro":   jaccard_score(y_true, y_pred, average="macro",   zero_division=zero_division),
    }


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list,
    zero_division: int = 0,
) -> dict:
    """
    Per-label F1, precision, recall, support.

    Args:
        y_true      : Ground truth binary matrix
        y_pred      : Predicted binary matrix
        label_names : List of emotion label strings
        zero_division: Value when denominator is zero

    Returns:
        {label_name: {"f1", "precision", "recall", "support"}}
    """
    per_label = {}
    for i, label in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        per_label[label] = {
            "f1":        f1_score(yt, yp, zero_division=zero_division),
            "precision": precision_score(yt, yp, zero_division=zero_division),
            "recall":    recall_score(yt, yp, zero_division=zero_division),
            "support":   int(yt.sum()),
        }
    return per_label


def threshold_search(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: list = None,
    zero_division: int = 0,
) -> dict:
    """
    Stretch goal: evaluate F1 macro at multiple thresholds.

    Args:
        logits    : Raw model logits (n_samples, n_labels)
        labels    : Ground truth binary matrix
        thresholds: List of floats to evaluate (default: [0.3, 0.4, 0.5, 0.6, 0.7])

    Returns:
        {threshold: f1_macro}
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    probs = sigmoid(logits)
    results = {}
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        results[t] = f1_score(labels.astype(int), y_pred, average="macro", zero_division=zero_division)

    return results