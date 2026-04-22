"""

Execute baseline models for all languages with fallback training logic.
Handles cases where train.parquet is missing (zul, xho).

Fallback strategy:
  1. Use train split if available
  2. Otherwise use validation as train, test as validation
  3. Report all metrics on test split
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_loader import run_pipeline
from baseline import BaselineModel, TFIDF_CONFIG, LOGREG_CONFIG
from metrics import compute_metrics, compute_per_label_metrics

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models/baseline")
RANDOM_STATE = 42



def prepare_splits_with_fallback(splits: dict, lang_code: str) -> dict:
    """
    Prepare train/test splits with fallback logic.
    
    Strategy:
      - If train exists: use train → test
      - If no train: use validation → test
      
    Returns:
        {
            "train_df": DataFrame,
            "test_df": DataFrame,
            "train_source": "train" | "validation",
            "status": "success" | "no_data"
        }
    """
    if "train" in splits and splits["train"] is not None and len(splits["train"]) > 0:
        return {
            "train_df": splits["train"],
            "test_df": splits.get("test"),
            "train_source": "train",
            "status": "success",
        }
    
    # Fallback: use validation as train
    if "validation" in splits and splits["validation"] is not None and len(splits["validation"]) > 0:
        print(f"  ⚠ [{lang_code}] No train split — using validation as train")
        return {
            "train_df": splits["validation"],
            "test_df": splits.get("test"),
            "train_source": "validation",
            "status": "success",
        }
    
    return {
        "train_df": None,
        "test_df": None,
        "train_source": None,
        "status": "no_data",
    }


def train_and_evaluate_with_fallback(
    lang_code: str,
    datasets: dict,
    emotion_index: list,
) -> dict:
    """
    Train and evaluate baseline model with fallback split logic.
    
    Returns comprehensive results including:
      - overall metrics
      - per-label metrics
      - predictions for error analysis
      - model status
    """
    print(f"\n{'='*70}")
    print(f"  LANGUAGE: {lang_code.upper()}")
    print(f"{'='*70}")
    
    splits = datasets[lang_code]
    split_config = prepare_splits_with_fallback(splits, lang_code)
    
    if split_config["status"] == "no_data":
        print(f"  ✗ No data available for {lang_code}")
        return {
            "lang": lang_code,
            "status": "no_data",
            "metrics": None,
            "per_label_metrics": None,
        }
    
    train_df = split_config["train_df"]
    test_df = split_config["test_df"]
    train_source = split_config["train_source"]
    
    if test_df is None or len(test_df) == 0:
        print(f"  ✗ No test data available for {lang_code}")
        return {
            "lang": lang_code,
            "status": "no_test",
            "metrics": None,
            "per_label_metrics": None,
        }
    
    # Prepare data
    train_texts = train_df["text"].tolist()
    train_labels = np.array(train_df["labels"].tolist())
    test_texts = test_df["text"].tolist()
    test_labels = np.array(test_df["labels"].tolist())
    
    print(f"\n[1/3] Training")
    print(f"  Train source : {train_source}")
    print(f"  Train samples: {len(train_texts)}")
    print(f"  Test samples : {len(test_texts)}")
    print(f"  Label dims   : {train_labels.shape[1]}")
    
    # Train model
    model = BaselineModel(TFIDF_CONFIG, LOGREG_CONFIG)
    model.fit(train_texts, train_labels)
    
    # Predict
    print(f"\n[2/3] Predicting")
    y_pred = model.predict(test_texts)
    
    # Evaluate
    print(f"\n[3/3] Evaluating")
    metrics = compute_metrics(test_labels, y_pred, zero_division=0)
    per_label = compute_per_label_metrics(test_labels, y_pred, emotion_index, zero_division=0)
    
    # Print summary
    print(f"\n  Overall Metrics:")
    for k, v in metrics.items():
        print(f"    {k:<20} {v:>8.4f}")
    
    # Save model
    model_path = MODELS_DIR / f"{lang_code}_baseline.pkl"
    model.save(model_path)
    
    return {
        "lang": lang_code,
        "status": "success",
        "train_source": train_source,
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "metrics": metrics,
        "per_label_metrics": per_label,
        "predictions": {
            "y_true": test_labels,
            "y_pred": y_pred,
            "texts": test_texts,
        },
        "model_path": str(model_path),
    }

def main():
    """Run baseline training and evaluation for all languages."""
    print("\n" + "="*70)
    print("  WEEK 2 — ISSUE #3: RUN BASELINE PER LANGUAGE")
    print("="*70)
    
    # Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[STEP 1] Loading datasets...")
    datasets, emotion_index = run_pipeline()
    
    print(f"\n  Emotion index: {emotion_index}")
    print(f"  Number of labels: {len(emotion_index)}")
    
    # Train models
    print("\n[STEP 2] Training baseline models with fallback logic...")
    all_results = {}
    
    for lang_code in ["zul", "xho", "swa"]:
        result = train_and_evaluate_with_fallback(lang_code, datasets, emotion_index)
        all_results[lang_code] = result
    
    # Save comprehensive results
    print("\n[STEP 3] Saving results...")
    
    # Main metrics table
    metrics_data = []
    for lang_code, result in all_results.items():
        if result["status"] == "success":
            m = result["metrics"]
            metrics_data.append({
                "language": lang_code,
                "model": "LogisticRegression",
                "train_source": result["train_source"],
                "train_size": result["train_size"],
                "test_size": result["test_size"],
                "f1_macro": m["f1_macro"],
                "f1_micro": m["f1_micro"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "hamming_loss": m["hamming_loss"],
                "jaccard_macro": m["jaccard_macro"],
                "status": "success",
            })
        else:
            metrics_data.append({
                "language": lang_code,
                "model": "LogisticRegression",
                "train_source": None,
                "train_size": None,
                "test_size": None,
                "f1_macro": None,
                "f1_micro": None,
                "precision_macro": None,
                "recall_macro": None,
                "hamming_loss": None,
                "jaccard_macro": None,
                "status": result["status"],
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = RESULTS_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  ✔ Metrics saved to {metrics_path}")
    
    # Print summary tables
    print("\n" + "="*70)
    print("  SUMMARY: BASELINE METRICS")
    print("="*70)
    print(f"\n{metrics_df.to_string(index=False)}")
    
    # Quick comparison
    print("\n" + "="*70)
    print("  QUICK COMPARISON")
    print("="*70)
    summary_cols = ["language", "model", "train_source", "f1_macro", "precision_macro"]
    if all(col in metrics_df.columns for col in summary_cols):
        print(f"\n{metrics_df[summary_cols].to_string(index=False)}")
    
    print("\n✅ Baseline execution complete.\n")
    
    # Return for use in exploratory analysis
    return all_results, emotion_index


if __name__ == "__main__":
    results, emotion_index = main()