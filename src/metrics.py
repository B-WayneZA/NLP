"""

Multi-label classification evaluation metrics.
Supports macro/micro F1, precision, recall with zero_division handling.
"""

import numpy as np
from sklearn.metrics import (
   f1_score,
   precision_score,
   recall_score,
   classification_report,
   hamming_loss,
   jaccard_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, zero_division: int = 0) -> dict:
   """
   Compute standard multi-label classification metrics.

   Args:
      y_true: Ground truth binary matrix (n_samples, n_labels)
      y_pred: Predicted binary matrix (n_samples, n_labels)
      zero_division: Value to return when there is a zero division (default: 0)

   Returns:
      Dictionary with macro/micro F1, precision, recall, hamming loss, jaccard
   """
   return {
      "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=zero_division),
      "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=zero_division),
      "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=zero_division),
      "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=zero_division),
      "hamming_loss": hamming_loss(y_true, y_pred),
      "jaccard_macro": jaccard_score(y_true, y_pred, average="macro", zero_division=zero_division),
   }


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list,
    zero_division: int = 0,
) -> dict:
   """
   Compute per-label F1, precision, recall.

   Args:
      y_true: Ground truth binary matrix (n_samples, n_labels)
      y_pred: Predicted binary matrix (n_samples, n_labels)
      label_names: List of emotion label names
      zero_division: Value to return when there is a zero division

   Returns:
      Dictionary mapping label names to their metrics
   """
   per_label = {}
   n_labels = y_true.shape[1]

   for i, label in enumerate(label_names):
      y_true_label = y_true[:, i]
      y_pred_label = y_pred[:, i]

      per_label[label] = {
         "f1": f1_score(y_true_label, y_pred_label, zero_division=zero_division),
         "precision": precision_score(y_true_label, y_pred_label, zero_division=zero_division),
         "recall": recall_score(y_true_label, y_pred_label, zero_division=zero_division),
         "support": int(y_true_label.sum()),
      }

   return per_label


def print_metrics_table(metrics_dict: dict, title: str = "Metrics"):
   """
   Pretty-print metrics as a table.

   Args:
      metrics_dict: Dictionary from compute_metrics()
      title: Table title
   """
   print(f"\n{'='*60}")
   print(f"  {title}")
   print(f"{'='*60}")
   for metric, value in metrics_dict.items():
      print(f"  {metric:<20} {value:>8.4f}")
   print("="*60)


def print_per_label_table(per_label_dict: dict, title: str = "Per-Label Metrics"):
    """
    Pretty-print per-label metrics as a table.

    Args:
        per_label_dict: Dictionary from compute_per_label_metrics()
        title: Table title
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Label':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Support':>8}")
    print("-"*70)
    for label, scores in per_label_dict.items():
        print(
            f"  {label:<20} {scores['f1']:>8.4f} {scores['precision']:>10.4f} "
            f"{scores['recall']:>8.4f} {scores['support']:>8}"
        )
    print("="*70)