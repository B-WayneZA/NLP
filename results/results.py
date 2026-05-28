"""

Central utility for managing and accessing experiment results.
Provides convenience functions for loading, viewing, and comparing outputs.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

RESULTS_DIR = Path("results")

def load_baseline_metrics() -> pd.DataFrame:
    """Load baseline metrics table."""
    path = RESULTS_DIR / "baseline_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return pd.read_csv(path)


def load_mbert_metrics() -> pd.DataFrame:
    """Load mBERT fine-tuning metrics."""
    path = RESULTS_DIR / "mbert_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"mBERT metrics file not found: {path}")
    return pd.read_csv(path)


def load_xlmr_metrics() -> pd.DataFrame:
    """Load XLM-R fine-tuning metrics."""
    path = RESULTS_DIR / "xlmr_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"XLM-R metrics file not found: {path}")
    return pd.read_csv(path)


def load_weak_labels() -> pd.DataFrame:
    """Load weak labels analysis."""
    path = RESULTS_DIR / "weak_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Weak labels file not found: {path}")
    return pd.read_csv(path)


def load_class_distribution() -> pd.DataFrame:
    """Load class distribution analysis."""
    path = RESULTS_DIR / "class_distribution.csv"
    if not path.exists():
        raise FileNotFoundError(f"Distribution file not found: {path}")
    return pd.read_csv(path)


def load_error_examples() -> pd.DataFrame:
    """Load error examples."""
    path = RESULTS_DIR / "error_examples.csv"
    if not path.exists():
        raise FileNotFoundError(f"Error examples file not found: {path}")
    return pd.read_csv(path)


def load_observations() -> str:
    """Load observation report text."""
    path = RESULTS_DIR / "baseline_observations.md"
    if not path.exists():
        raise FileNotFoundError(f"Observations file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()



def compare_languages(metric: str = "f1_macro") -> pd.DataFrame:
    """
    Compare all three models across languages on a specific metric.

    Args:
        metric: One of f1_macro, f1_micro, precision_macro, recall_macro

    Returns:
        DataFrame with languages as rows, models as columns.
    """
    frames = []
    for loader, label in [
        (load_baseline_metrics, "baseline"),
        (load_mbert_metrics,    "mbert"),
        (load_xlmr_metrics,     "xlmr"),
    ]:
        try:
            df = loader()
            if metric not in df.columns:
                raise ValueError(f"Metric '{metric}' not found in {label} results")
            frames.append(df[["language", metric]].rename(columns={metric: label}))
        except FileNotFoundError:
            pass  # model not yet run — omit column

    if not frames:
        raise FileNotFoundError("No result files found — run at least one model first.")

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="language", how="outer")

    # Add best-model column
    model_cols = [c for c in result.columns if c != "language"]
    result["best"] = result[model_cols].idxmax(axis=1)
    return result.sort_values("language").reset_index(drop=True)


def get_language_summary(lang_code: str) -> Dict:
    """
    Get comprehensive summary for a single language.
    
    Returns dictionary with:
      - metrics
      - weak labels
      - class distribution
      - error examples
    """
    metrics_df = load_baseline_metrics()
    lang_metrics = metrics_df[metrics_df["language"] == lang_code]
    
    if len(lang_metrics) == 0:
        raise ValueError(f"No results found for language: {lang_code}")
    
    weak_df = load_weak_labels()
    lang_weak = weak_df[weak_df["language"] == lang_code]
    
    dist_df = load_class_distribution()
    lang_dist = dist_df[dist_df["language"] == lang_code]
    
    error_df = load_error_examples()
    lang_errors = error_df[error_df["language"] == lang_code]
    
    return {
        "metrics": lang_metrics.to_dict("records")[0],
        "weak_labels": lang_weak.to_dict("records"),
        "class_distribution": lang_dist.to_dict("records"),
        "error_examples": lang_errors.to_dict("records"),
    }


def get_worst_performing_labels(top_k: int = 5) -> pd.DataFrame:
    """Get the worst performing labels across all languages."""
    weak_df = load_weak_labels()
    worst = weak_df.nsmallest(top_k, "f1")
    return worst[["language", "label", "f1", "precision", "recall", "issue"]]


def get_most_imbalanced_labels(top_k: int = 5) -> pd.DataFrame:
    """Get the most imbalanced labels based on frequency variance."""
    dist_df = load_class_distribution()
    
    # Compute coefficient of variation per label
    label_stats = dist_df.groupby("label").agg({
        "count": ["mean", "std", "min", "max"]
    }).reset_index()
    label_stats.columns = ["label", "mean_count", "std_count", "min_count", "max_count"]
    label_stats["cv"] = label_stats["std_count"] / label_stats["mean_count"]
    label_stats["range_ratio"] = label_stats["max_count"] / (label_stats["min_count"] + 1)
    
    most_imbalanced = label_stats.nlargest(top_k, "range_ratio")
    return most_imbalanced


def print_summary():
    """Print a quick summary of all results."""
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    
    # Baseline metrics
    print("\n[1] Baseline Performance (TF-IDF + Logistic Regression):")
    metrics_df = load_baseline_metrics()
    print(metrics_df[["language", "model", "f1_macro", "f1_micro", "status"]].to_string(index=False))
    
    # mBERT metrics
    print("\n[2] mBERT Performance:")
    try:
        mbert_df = load_mbert_metrics()
        print(mbert_df[["language", "f1_macro", "f1_micro", "precision_macro", "recall_macro"]].to_string(index=False))
    except FileNotFoundError:
        print("  (not yet run — execute src/transformers_pipeline/run_mbert.py)")

    # XLM-R metrics
    print("\n[3] XLM-R Performance:")
    try:
        xlmr_df = load_xlmr_metrics()
        print(xlmr_df[["language", "f1_macro", "f1_micro", "precision_macro", "recall_macro"]].to_string(index=False))
    except FileNotFoundError:
        print("  (not yet run — execute src/transformers_pipeline/run_xlmr.py)")

    # Weak labels count
    print("\n[4] Weak Labels:")
    try:
        weak_df = load_weak_labels()
        if len(weak_df) > 0:
            weak_counts = weak_df.groupby("language").size()
            for lang, count in weak_counts.items():
                print(f"  {lang}: {count} weak labels")
        else:
            print("  None detected")
    except FileNotFoundError:
        print("  (not yet generated)")
    
    # Class imbalance
    print("\n[5] Class Imbalance:")
    try:
        dist_df = load_class_distribution()
        for lang in dist_df["language"].unique():
            lang_dist = dist_df[dist_df["language"] == lang]
            max_count = lang_dist["count"].max()
            min_count = lang_dist[lang_dist["count"] > 0]["count"].min() if (lang_dist["count"] > 0).any() else 1
            ratio = max_count / min_count
            print(f"  {lang}: {ratio:.1f}:1 imbalance ratio")
    except FileNotFoundError:
        print("  (not yet generated)")
    
    # Error examples
    print("\n[6] Error Examples:")
    try:
        error_df = load_error_examples()
        error_counts = error_df.groupby(["language", "error_type"]).size()
        for (lang, etype), count in error_counts.items():
            print(f"  {lang} - {etype}: {count}")
    except FileNotFoundError:
        print("  (not yet generated)")
    
    print("\n" + "="*70 + "\n")

def export_for_paper(output_path: str = "results/paper_tables.txt"):
    """
    Export key tables in LaTeX-friendly format for paper inclusion.
    """
    output = []
    
    # Table 1: Baseline metrics
    output.append("% Table 1: Baseline Model Performance\n")
    metrics_df = load_baseline_metrics()
    latex_metrics = metrics_df[["language", "f1_macro", "f1_micro", "precision_macro", "recall_macro"]]
    output.append(latex_metrics.to_latex(index=False, float_format="%.3f"))
    
    # Table 2: mBERT metrics
    try:
        output.append("\n% Table 2: mBERT Performance\n")
        mbert_df = load_mbert_metrics()
        output.append(mbert_df[["language", "f1_macro", "f1_micro", "precision_macro", "recall_macro"]].to_latex(index=False, float_format="%.3f"))
    except FileNotFoundError:
        output.append("% mBERT results not yet available\n")
    
    # Table 3: XLM-R metrics
    try:
        output.append("\n% Table 3: XLM-R Performance\n")
        xlmr_df = load_xlmr_metrics()
        output.append(xlmr_df[["language", "f1_macro", "f1_micro", "precision_macro", "recall_macro"]].to_latex(index=False, float_format="%.3f"))
    except FileNotFoundError:
        output.append("% XLM-R results not yet available\n")
    
    # Table 4: Weak labels summary
    try:
        output.append("\n% Table 4: Weak Labels Per Language\n")
        weak_df = load_weak_labels()
        weak_summary = weak_df.groupby("language").agg({
            "label": "count",
            "f1": "mean"
        }).reset_index()
        weak_summary.columns = ["Language", "Weak Label Count", "Avg F1"]
        output.append(weak_summary.to_latex(index=False, float_format="%.3f"))
    except FileNotFoundError:
        output.append("% Weak labels not yet available\n")
    
    with open(output_path, "w") as f:
        f.writelines(output)
    
    print(f"✔ Exported paper tables to {output_path}")



if __name__ == "__main__":
    print("\nTesting results.py utilities...\n")
    
    try:
        print_summary()
        
        print("\n[TEST] Language comparison on F1 macro:")
        print(compare_languages("f1_macro"))
        
        print("\n[TEST] Worst performing labels:")
        print(get_worst_performing_labels(3))
        
    except FileNotFoundError as e:
        print(f"\n⚠ Results files not found. Run the experiments first:")
        print(f"  python run_baseline.py")
        print(f"  python exploratory_analysis.py")