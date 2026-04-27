"""
: Fine-tune mBERT (bert-base-multilingual-cased) for all languages.

Outputs:
  results/mbert_metrics.csv
  results/mbert_per_label_metrics.csv   [stretch goal]
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import run_pipeline
from src.transformers_pipeline.transformer_train import (
    run_model_all_languages,
    results_to_dataframe,
    save_per_label_metrics,
    RESULTS_DIR,
)


def main():
    print("\n" + "="*70)
    print("  ISSUE #6: FINE-TUNE mBERT")
    print("="*70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\n[STEP 1] Loading datasets...")
    datasets, emotion_index = run_pipeline()
    print(f"\n  Emotion index ({len(emotion_index)} classes): {emotion_index}")

    # Train mBERT on all languages
    print("\n[STEP 2] Fine-tuning mBERT...")
    mbert_results = run_model_all_languages("mbert", datasets, emotion_index)

    # Convert to DataFrame
    print("\n[STEP 3] Saving results...")
    mbert_df = results_to_dataframe(mbert_results, "mbert")

    csv_path = RESULTS_DIR / "mbert_metrics.csv"
    mbert_df.to_csv(csv_path, index=False)
    print(f"  ✔ Saved to {csv_path}")

    # Save per-label metrics (stretch goal)
    save_per_label_metrics(mbert_results, "mbert", RESULTS_DIR)

    # Print summary
    print("\n" + "="*70)
    print("  mBERT RESULTS SUMMARY")
    print("="*70)
    display_cols = ["language", "model", "train_source", "f1_macro", "f1_micro",
                    "precision_macro", "recall_macro", "elapsed_seconds", "status"]
    print(f"\n{mbert_df[display_cols].to_string(index=False)}")

    print("\n✅ mBERT fine-tuning complete.\n")
    return mbert_results, emotion_index


if __name__ == "__main__":
    main()