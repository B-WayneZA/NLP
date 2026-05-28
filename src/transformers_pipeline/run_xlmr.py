"""
Fine-tune XLM-R (xlm-roberta-base) for all languages.

Outputs:
  results/xlmr_metrics.csv
  results/xlmr_per_label_metrics.csv   [stretch goal]
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import run_pipeline
from src.transformers_pipeline.transformer_train import (
    run_single_experiment,
    results_to_dataframe,
    save_per_label_metrics,
    RESULTS_DIR,
    MODELS,
)


def main():
    print("\n" + "="*70)
    print(" FINE-TUNE XLM-R")
    print("="*70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\n[STEP 1] Loading datasets...")
    datasets, emotion_index = run_pipeline()
    print(f"\n  Emotion index ({len(emotion_index)} classes): {emotion_index}")

    # Train XLM-R on all languages — save after each language to survive crashes
    print("\n[STEP 2] Fine-tuning XLM-R...")
    print(f"\n{'#'*70}")
    print(f"  MODEL: XLM-R  ({MODELS['xlmr']})")
    print(f"{'#'*70}")

    xlmr_results = {}
    csv_path = RESULTS_DIR / "xlmr_metrics.csv"

    for lang_code in ["zul", "xho", "swa"]:
        print(f"\nRunning XLM-R on {lang_code}...")
        result = run_single_experiment(
            model_key="xlmr",
            lang_code=lang_code,
            datasets=datasets,
            emotion_index=emotion_index,
        )
        xlmr_results[lang_code] = result

        # Save incrementally after each language so partial results survive
        partial_df = results_to_dataframe(xlmr_results, "xlmr")
        partial_df.to_csv(csv_path, index=False)
        print(f"  ✔ Partial results saved → {csv_path} ({lang_code} done)")

    # Final consolidated save
    print("\n[STEP 3] Saving final results...")
    xlmr_df = results_to_dataframe(xlmr_results, "xlmr")

    xlmr_df.to_csv(csv_path, index=False)
    print(f"  ✔ Saved to {csv_path}")

    # Save per-label metrics (stretch goal)
    save_per_label_metrics(xlmr_results, "xlmr", RESULTS_DIR)

    # Print summary
    print("\n" + "="*70)
    print("  XLM-R RESULTS SUMMARY")
    print("="*70)
    display_cols = ["language", "model", "train_source", "f1_macro", "f1_micro",
                    "precision_macro", "recall_macro", "elapsed_seconds", "status"]
    print(f"\n{xlmr_df[display_cols].to_string(index=False)}")

    print("\n✅ XLM-R fine-tuning complete.\n")
    return xlmr_results, emotion_index


if __name__ == "__main__":
    main()