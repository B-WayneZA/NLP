"""
— Transformer Fine-Tuning Pipeline

Runs mBERT and XLM-R on all languages and generates:
  results/mbert_metrics.csv
  results/xlmr_metrics.csv
  results/transformer_summary.csv   ← combined with baseline for comparison

Usage:
    # Run everything
    python3 run_transformer_pipeline.py

    # Run one model only
    python3 run_transformer_pipeline.py --model mbert
    python3 run_transformer_pipeline.py --model xlmr
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset_loader import run_pipeline
from src.transformers_pipeline.transformer_train import (
    run_model_all_languages,
    results_to_dataframe,
    save_per_label_metrics,
    RESULTS_DIR,
    MODELS,
    TRAINING_DEFAULTS,
    MAX_LENGTH,
    THRESHOLD,
)


# ─────────────────────────────────────────────
# COMBINED SUMMARY
# ─────────────────────────────────────────────

def build_transformer_summary(mbert_df: pd.DataFrame, xlmr_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate mBERT and XLM-R results into one summary table."""
    return pd.concat([mbert_df, xlmr_df], ignore_index=True)


def load_baseline_if_exists() -> pd.DataFrame:
    """Load baseline metrics CSV if it exists."""
    baseline_path = RESULTS_DIR / "baseline_metrics.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        # Harmonise column name
        if "model" not in df.columns:
            df["model"] = "LogisticRegression"
        return df
    return None


def print_full_comparison(summary_df: pd.DataFrame, baseline_df: pd.DataFrame = None):
    """Print a clean comparison table across all models and languages."""
    print("\n" + "="*80)
    print("  FULL MODEL COMPARISON")
    print("="*80)

    all_dfs = []

    if baseline_df is not None:
        cols = ["language", "model", "f1_macro", "f1_micro",
                "precision_macro", "recall_macro", "hamming_loss", "jaccard_macro"]
        available_cols = [c for c in cols if c in baseline_df.columns]
        all_dfs.append(baseline_df[available_cols].copy())

    trans_cols = ["language", "model", "f1_macro", "f1_micro",
                  "precision_macro", "recall_macro", "hamming_loss", "jaccard_macro"]
    available_trans_cols = [c for c in trans_cols if c in summary_df.columns]
    all_dfs.append(summary_df[available_trans_cols].copy())

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["language", "f1_macro"], ascending=[True, False])

    print(f"\n{combined.to_string(index=False)}")

    # Per-language winner
    print("\n" + "="*80)
    print("  BEST MODEL PER LANGUAGE")
    print("="*80)
    for lang in ["zul", "xho", "swa"]:
        lang_df = combined[combined["language"] == lang].dropna(subset=["f1_macro"])
        if len(lang_df) > 0:
            best_row = lang_df.loc[lang_df["f1_macro"].idxmax()]
            print(f"\n  {lang.upper()}:")
            print(f"    Best model    : {best_row['model']}")
            print(f"    F1 Macro      : {best_row['f1_macro']:.4f}")
            if baseline_df is not None:
                baseline_row = combined[(combined["language"] == lang) &
                                       (combined["model"] == "LogisticRegression")]
                if len(baseline_row) > 0 and pd.notna(baseline_row.iloc[0]["f1_macro"]):
                    baseline_f1 = baseline_row.iloc[0]["f1_macro"]
                    improvement = best_row["f1_macro"] - baseline_f1
                    print(f"    vs Baseline   : {baseline_f1:.4f}  ({improvement:+.4f})")

    print("\n" + "="*80)


def log_experiment_config():
    """Print a clear record of the experiment configuration for reproducibility."""
    print("\n" + "="*80)
    print("  EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"\n  Models:")
    for key, name in MODELS.items():
        print(f"    {key:<8} {name}")

    print(f"\n  Hyperparameters (fixed across all runs — transformers 5.x API):")
    for k, v in TRAINING_DEFAULTS.items():
        print(f"    {k:<35} {v}")

    print(f"\n  Tokenization:")
    print(f"    max_length   : {MAX_LENGTH}")
    print(f"    padding      : max_length")
    print(f"    truncation   : True")

    print(f"\n  Inference:")
    print(f"    threshold    : {THRESHOLD}")
    print(f"    loss         : BCEWithLogitsLoss")
    print(f"    problem_type : multi_label_classification")
    print("="*80)



def main(run_mbert: bool = True, run_xlmr: bool = True):
    print("\n" + "="*80)
    print("  WEEK 3 — TRANSFORMER FINE-TUNING (Issues #5-#7)")
    print("  mBERT vs XLM-R for Multi-Label Emotion Classification")
    print("="*80)

    # Log configuration upfront
    log_experiment_config()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets once — shared across both models
    print("\n[STEP 1] Loading datasets...")
    datasets, emotion_index = run_pipeline()
    print(f"\n  Emotion index ({len(emotion_index)} classes): {emotion_index}")

    mbert_df = None
    xlmr_df  = None

    # ── mBERT ──────────────────────────────────────────────────
    if run_mbert:
        print("\n[STEP 2a] Fine-tuning mBERT...")
        print("  Note: This may take 30-90 min per language on CPU")
        mbert_results = run_model_all_languages("mbert", datasets, emotion_index)
        mbert_df = results_to_dataframe(mbert_results, "mbert")

        mbert_path = RESULTS_DIR / "mbert_metrics.csv"
        mbert_df.to_csv(mbert_path, index=False)
        print(f"\n  ✔ mBERT metrics saved to {mbert_path}")

        save_per_label_metrics(mbert_results, "mbert", RESULTS_DIR)

    elif (RESULTS_DIR / "mbert_metrics.csv").exists():
        print("\n[STEP 2a] Loading existing mBERT results...")
        mbert_df = pd.read_csv(RESULTS_DIR / "mbert_metrics.csv")
        print(f"  ✔ Loaded {len(mbert_df)} rows")

    # ── XLM-R ──────────────────────────────────────────────────
    if run_xlmr:
        print("\n[STEP 2b] Fine-tuning XLM-R...")
        print("  Note: This may take 30-90 min per language on CPU")
        xlmr_results = run_model_all_languages("xlmr", datasets, emotion_index)
        xlmr_df = results_to_dataframe(xlmr_results, "xlmr")

        xlmr_path = RESULTS_DIR / "xlmr_metrics.csv"
        xlmr_df.to_csv(xlmr_path, index=False)
        print(f"\n  ✔ XLM-R metrics saved to {xlmr_path}")

        save_per_label_metrics(xlmr_results, "xlmr", RESULTS_DIR)

    elif (RESULTS_DIR / "xlmr_metrics.csv").exists():
        print("\n[STEP 2b] Loading existing XLM-R results...")
        xlmr_df = pd.read_csv(RESULTS_DIR / "xlmr_metrics.csv")
        print(f"  ✔ Loaded {len(xlmr_df)} rows")

    # ── Combined Summary ────────────────────────────────────────
    if mbert_df is not None and xlmr_df is not None:
        print("\n[STEP 3] Building combined summary...")
        summary_df = build_transformer_summary(mbert_df, xlmr_df)

        summary_path = RESULTS_DIR / "transformer_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  ✔ Transformer summary saved to {summary_path}")

        # Load baseline for comparison
        baseline_df = load_baseline_if_exists()
        if baseline_df is not None:
            print("  ✔ Baseline results loaded for comparison")
        else:
            print("  ⚠ No baseline results found — run baseline first for full comparison")

        # Print final comparison
        print_full_comparison(summary_df, baseline_df)

    print("\n" + "="*80)
    print("="*80)
    print("\nGenerated files:")
    if run_mbert:
        print("  results/mbert_metrics.csv")
        print("  results/mbert_per_label_metrics.csv")
    if run_xlmr:
        print("  results/xlmr_metrics.csv")
        print("  results/xlmr_per_label_metrics.csv")
    if mbert_df is not None and xlmr_df is not None:
        print("  results/transformer_summary.csv")
    print("\nNext steps:")
    print("  → Review results/transformer_summary.csv")
    print("  → Run Week 4 analysis and comparison")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 3: Transformer fine-tuning")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mbert", "xlmr", "both"],
        default="both",
        help="Which model to run (default: both)",
    )
    args = parser.parse_args()

    run_mbert = args.model in ("mbert", "both")
    run_xlmr  = args.model in ("xlmr",  "both")

    main(run_mbert=run_mbert, run_xlmr=run_xlmr)