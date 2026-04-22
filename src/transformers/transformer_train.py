"""

Reusable fine-tuning pipeline for mBERT and XLM-R.
Handles multi-label emotion classification via BCEWithLogitsLoss.

Key design decisions:
  - problem_type="multi_label_classification" enables BCE loss natively
  - sigmoid + 0.5 threshold at inference time
  - Identical hyperparameters across models and languages for fair comparison
  - Fallback train logic mirrors Week 2 baseline (validation → train)
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from transformer_dataset import build_datasets
from transformer_metrics import (
    make_compute_metrics,
    compute_metrics_from_arrays,
    compute_per_label_metrics,
    threshold_search,
    sigmoid,
)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr":  "xlm-roberta-base",
}

# Fixed hyperparameters — identical across all runs for fair comparison
# NOTE: transformers >=5.0 uses eval_strategy (not evaluation_strategy)
#CPU. If that's too long, reduce num_train_epochs=1 or per_device_train_batch_size=4 in TRAINING_DEFAULTS inside transformer_train.py.
TRAINING_DEFAULTS = {
    "num_train_epochs":              1,
    "learning_rate":                 2e-5,
    "per_device_train_batch_size":   8,
    "per_device_eval_batch_size":    16,
    "weight_decay":                  0.01,
    "eval_strategy":                 "epoch",  # transformers 5.x (was evaluation_strategy)
    "save_strategy":                 "no",     # Saves disk I/O on CPU
    "load_best_model_at_end":        False,    # Requires save_strategy != "no"
    "logging_strategy":              "epoch",
    "seed":                          42,
    "dataloader_num_workers":        0,        # Safer on CPU
    "fp16":                          False,    # CPU doesn't support fp16
    "use_cpu":                       True,     # transformers 5.x (was no_cuda)
    "report_to":                     "none",   # Disable wandb/tensorboard
}

MAX_LENGTH   = 128
THRESHOLD    = 0.5
RESULTS_DIR  = Path("results")
CHECKPOINTS  = Path("checkpoints")
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def load_model_and_tokenizer(
    model_key: str,
    num_labels: int,
    label_names: list,
):
    """
    Load pretrained model and tokenizer for multi-label classification.

    Args:
        model_key  : "mbert" or "xlmr"
        num_labels : Number of emotion labels
        label_names: List of emotion label strings

    Returns:
        (model, tokenizer)
    """
    model_name = MODELS[model_key]
    print(f"\n  → Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # problem_type="multi_label_classification" configures:
    #   - BCEWithLogitsLoss as loss function
    #   - sigmoid output activation
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✔ Model loaded — {trainable_params:,} / {total_params:,} params trainable")

    return model, tokenizer


def build_training_args(
    model_key: str,
    lang_code: str,
    output_dir: Path,
    overrides: dict = None,
) -> TrainingArguments:
   """
   Build TrainingArguments with fixed defaults + optional overrides.

   Args:
      model_key  : For output directory naming
      lang_code  : For output directory naming
      output_dir : Base checkpoints directory
      overrides  : Optional dict of hyperparameter overrides

   Returns:
      TrainingArguments
   """
   run_output_dir = output_dir / f"{model_key}_{lang_code}"

   args = dict(TRAINING_DEFAULTS)

   if overrides:
      args.update(overrides)

   args["output_dir"] = str(run_output_dir)

   return TrainingArguments(**args)


def run_single_experiment(
   model_key: str,
   lang_code: str,
   datasets: dict,          # raw DataFrames from dataset_loader
   emotion_index: list,
   overrides: dict = None,
) -> dict:

   """
   Complete training + evaluation pipeline for one model × language.

   Args:
      model_key    : "mbert" or "xlmr"
      lang_code    : "zul" | "xho" | "swa"
      datasets     : Output of dataset_loader.run_pipeline()
      emotion_index: List of emotion label names
      overrides    : Optional hyperparameter overrides

   Returns:
      Result dict with metrics, timing, and prediction arrays
   """
   print(f"\n{'='*70}")
   print(f"  {model_key.upper()} — {lang_code.upper()}")
   print(f"{'='*70}")

   set_seed(RANDOM_STATE)
   num_labels = len(emotion_index)
   start_time = time.time()

   # Load model + tokenizer fresh for each run (prevents bleed-over)
   model, tokenizer = load_model_and_tokenizer(model_key, num_labels, emotion_index)

   # Build datasets
   lang_splits = datasets[lang_code]
   split_data = build_datasets(
      splits=lang_splits,
      emotion_index=emotion_index,
      tokenizer=tokenizer,
      lang_code=lang_code,
      max_length=MAX_LENGTH,
   )

   if split_data["train"] is None:
      elapsed = time.time() - start_time
      print(f"  ✗ No training data available for {lang_code} — skipping")
      return {
         "lang": lang_code,
         "model": model_key,
         "status": "no_data",
         "metrics": None,
         "elapsed_seconds": elapsed,
      }

   # Training arguments
   training_args = build_training_args(
      model_key=model_key,
      lang_code=lang_code,
      output_dir=CHECKPOINTS,
      overrides=overrides,
   )


   # Compute metrics callback
   compute_metrics_fn = make_compute_metrics(threshold=THRESHOLD)

   has_eval = (
    "validation" in split_data
    and split_data["validation"] is not None
    and len(split_data["validation"]) > 0
   )

   strategy = "epoch" if has_eval else "no"

   if hasattr(training_args, "eval_strategy"):
      training_args.eval_strategy = strategy

   if hasattr(training_args, "evaluation_strategy"):
      training_args.evaluation_strategy = strategy

   # Build Trainer
   trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=split_data["train"],
      eval_dataset=split_data["validation"] if has_eval else None,
      compute_metrics=compute_metrics_fn,
      processing_class=tokenizer,
   )

   # Train
   print(f"\n  Training...")
   print(f"  Epochs        : {training_args.num_train_epochs}")
   print(f"  Batch size    : {training_args.per_device_train_batch_size}")
   print(f"  Learning rate : {training_args.learning_rate}")
   print(f"  Train source  : {split_data['train_source']}")
   print(f"  Train samples : {len(split_data['train'])}")
   if split_data["validation"]:
      print(f"  Val samples   : {len(split_data['validation'])}")

   trainer.train()
   train_elapsed = time.time() - start_time

   # Evaluate on test set
   if split_data["test"] is None:
      print(f"  ✗ No test data — cannot evaluate {lang_code}")
      return {
         "lang": lang_code,
         "model": model_key,
         "status": "no_test",
         "metrics": None,
         "elapsed_seconds": train_elapsed,
      }

   print(f"\n  Evaluating on test set ({len(split_data['test'])} samples)...")

   # Get raw logits from test set
   test_preds = trainer.predict(split_data["test"])
   logits = test_preds.predictions
   y_true = test_preds.label_ids.astype(int)

   # Apply threshold
   probs = sigmoid(logits)
   y_pred = (probs >= THRESHOLD).astype(int)

   # Compute final metrics on test set
   metrics = compute_metrics_from_arrays(y_true, y_pred, zero_division=0)

   # Per-label metrics (stretch goal)
   per_label = compute_per_label_metrics(y_true, y_pred, emotion_index, zero_division=0)

   # Threshold sensitivity analysis (stretch goal)
   threshold_results = threshold_search(logits, y_true)

   total_elapsed = time.time() - start_time

   # Print results
   print(f"\n  Test Metrics:")
   for k, v in metrics.items():
      print(f"    {k:<20} {v:>8.4f}")

   print(f"\n  Threshold sensitivity:")
   for t, f1 in threshold_results.items():
      bar = "█" * int(f1 * 30)
      print(f"    t={t:.1f}  F1={f1:.4f}  {bar}")

   print(f"\n  ⏱ Total elapsed: {total_elapsed:.1f}s")

   return {
      "lang":           lang_code,
      "model":          model_key,
      "status":         "success",
      "train_source":   split_data["train_source"],
      "train_size":     len(split_data["train"]),
      "test_size":      len(split_data["test"]),
      "metrics":        metrics,
      "per_label":      per_label,
      "threshold_results": threshold_results,
      "elapsed_seconds": total_elapsed,
      "predictions": {
         "y_true": y_true,
         "y_pred": y_pred,
         "logits": logits,
      },
   }


# ─────────────────────────────────────────────
# MULTI-RUN ORCHESTRATION
# ─────────────────────────────────────────────

def run_model_all_languages(
    model_key: str,
    datasets: dict,
    emotion_index: list,
) -> dict:
    """
    Run one model across all three languages.

    Returns:
        {lang_code: result_dict}
    """
    print(f"\n{'#'*70}")
    print(f"  MODEL: {model_key.upper()}  ({MODELS[model_key]})")
    print(f"{'#'*70}")

    results = {}
    for lang_code in ["zul", "xho", "swa"]:
        print(f"\nRunning {model_key.upper()} on {lang_code}...")
        result = run_single_experiment(
            model_key=model_key,
            lang_code=lang_code,
            datasets=datasets,
            emotion_index=emotion_index,
        )
        results[lang_code] = result

    return results


def results_to_dataframe(all_results: dict, model_key: str) -> pd.DataFrame:
    """
    Convert result dicts to a flat DataFrame for saving.

    Args:
        all_results: {lang_code: result_dict}
        model_key  : "mbert" or "xlmr"

    Returns:
        DataFrame with one row per language
    """
    rows = []
    for lang_code, result in all_results.items():
        row = {
            "language": lang_code,
            "model": model_key,
        }

        if result["status"] == "success":
            m = result["metrics"]
            row.update({
                "train_source":    result.get("train_source"),
                "train_size":      result.get("train_size"),
                "test_size":       result.get("test_size"),
                "f1_macro":        m["f1_macro"],
                "f1_micro":        m["f1_micro"],
                "precision_macro": m["precision_macro"],
                "recall_macro":    m["recall_macro"],
                "hamming_loss":    m["hamming_loss"],
                "jaccard_macro":   m["jaccard_macro"],
                "elapsed_seconds": result.get("elapsed_seconds"),
                "status":          "success",
            })
        else:
            row.update({
                "train_source":    None,
                "train_size":      None,
                "test_size":       None,
                "f1_macro":        None,
                "f1_micro":        None,
                "precision_macro": None,
                "recall_macro":    None,
                "hamming_loss":    None,
                "jaccard_macro":   None,
                "elapsed_seconds": result.get("elapsed_seconds"),
                "status":          result["status"],
            })

        rows.append(row)

    return pd.DataFrame(rows)


def save_per_label_metrics(all_results: dict, model_key: str, output_dir: Path):
    """
    Save per-label metrics to CSV (stretch goal).

    Args:
        all_results: {lang_code: result_dict}
        model_key  : "mbert" or "xlmr"
        output_dir : Where to save
    """
    rows = []
    for lang_code, result in all_results.items():
        if result["status"] != "success" or not result.get("per_label"):
            continue
        for label, scores in result["per_label"].items():
            rows.append({
                "language":  lang_code,
                "model":     model_key,
                "label":     label,
                "f1":        scores["f1"],
                "precision": scores["precision"],
                "recall":    scores["recall"],
                "support":   scores["support"],
            })

    if rows:
        df = pd.DataFrame(rows)
        path = output_dir / f"{model_key}_per_label_metrics.csv"
        df.to_csv(path, index=False)
        print(f"  ✔ Per-label metrics saved to {path}")