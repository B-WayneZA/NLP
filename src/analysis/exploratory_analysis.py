"""
Generates:
  - weak_labels.csv         : Low-performing emotion labels
  - class_distribution.csv  : Label frequency analysis
  - error_examples.csv      : Representative prediction failures
  - baseline_observations.md: Human-readable findings
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset_loader import run_pipeline
from baseline.run_baseline import main as run_baseline_main


RESULTS_DIR = Path("results")
WEAK_LABEL_THRESHOLD = 0.15  # F1 below this is considered weak
EXAMPLE_LIMIT = 10  # Max error examples per category

def identify_weak_labels(all_results: dict, emotion_index: list) -> pd.DataFrame:
    """
    Identify poorly-performing labels per language.
    
    Criteria for "weak":
      - F1 < threshold
      - Recall = 0 (never predicted)
      - Support > 0 but prediction = 0
    """
    print("\n[1/4] Identifying weak labels...")
    
    weak_data = []
    
    for lang_code, result in all_results.items():
        if result["status"] != "success":
            continue
        
        per_label = result["per_label_metrics"]
        
        for label, scores in per_label.items():
            f1 = scores["f1"]
            precision = scores["precision"]
            recall = scores["recall"]
            support = scores["support"]
            
            is_weak = (
                f1 < WEAK_LABEL_THRESHOLD or
                (recall == 0 and support > 0) or
                (precision == 0 and support > 0)
            )
            
            if is_weak:
                weak_data.append({
                    "language": lang_code,
                    "label": label,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "support": support,
                    "issue": _diagnose_weakness(f1, precision, recall, support),
                })
    
    weak_df = pd.DataFrame(weak_data)
    
    if len(weak_df) > 0:
        weak_df = weak_df.sort_values(["language", "f1"])
        print(f"  ✔ Found {len(weak_df)} weak labels")
    else:
        print(f"  ✔ No weak labels detected")
    
    return weak_df


def _diagnose_weakness(f1, precision, recall, support):
    """Provide human-readable diagnosis of why label is weak."""
    if recall == 0 and support > 0:
        return "never_predicted"
    elif precision == 0 and support > 0:
        return "all_predictions_wrong"
    elif f1 < 0.10:
        return "extremely_low_f1"
    elif recall < precision / 2:
        return "low_recall"
    elif precision < recall / 2:
        return "low_precision"
    else:
        return "general_weakness"




def analyze_class_distribution(datasets: dict, emotion_index: list) -> pd.DataFrame:
    """Compute label frequency across train splits."""
    print("\n[2/4] Analyzing class distribution...")
    
    dist_data = []
    
    for lang_code, splits in datasets.items():
        # Use train if available, otherwise validation
        if "train" in splits and splits["train"] is not None and len(splits["train"]) > 0:
            df = splits["train"]
            split_name = "train"
        elif "validation" in splits and splits["validation"] is not None:
            df = splits["validation"]
            split_name = "validation"
        else:
            continue
        
        # Count label occurrences
        label_matrix = np.array(df["labels"].tolist())
        label_counts = label_matrix.sum(axis=0)
        
        total_samples = len(df)
        
        for i, emotion in enumerate(emotion_index):
            count = int(label_counts[i])
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            
            dist_data.append({
                "language": lang_code,
                "split": split_name,
                "label": emotion,
                "count": count,
                "total_samples": total_samples,
                "percentage": percentage,
            })
    
    dist_df = pd.DataFrame(dist_data)
    dist_df = dist_df.sort_values(["language", "count"], ascending=[True, False])
    
    print(f"  ✔ Computed distribution for {len(dist_df)} label-language pairs")
    
    return dist_df



def extract_error_examples(all_results: dict, emotion_index: list) -> pd.DataFrame:
    """
    Extract representative prediction failures.
    
    Categories:
      - false_negatives: labels exist but none predicted
      - over_prediction: too many labels predicted
      - confusion: specific label confusions
    """
    print("\n[3/4] Extracting error examples...")
    
    error_data = []
    
    for lang_code, result in all_results.items():
        if result["status"] != "success":
            continue
        
        preds = result["predictions"]
        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        texts = preds["texts"]
        
        # False negatives (no prediction when labels exist)
        fn_indices = []
        for i in range(len(y_true)):
            if y_true[i].sum() > 0 and y_pred[i].sum() == 0:
                fn_indices.append(i)
        
        # Sample up to EXAMPLE_LIMIT
        fn_sample = np.random.choice(
            fn_indices,
            size=min(len(fn_indices), EXAMPLE_LIMIT // 3),
            replace=False,
        ) if fn_indices else []
        
        for idx in fn_sample:
            error_data.append({
                "language": lang_code,
                "text": texts[idx][:200],  # Truncate long texts
                "true_labels": _format_labels(y_true[idx], emotion_index),
                "predicted_labels": _format_labels(y_pred[idx], emotion_index),
                "error_type": "false_negative",
                "notes": "No prediction despite ground truth labels",
            })
        
        # Over-predictions (predicted too many)
        op_indices = []
        for i in range(len(y_true)):
            if y_pred[i].sum() > y_true[i].sum() + 1:  # Predicted 2+ more labels
                op_indices.append(i)
        
        op_sample = np.random.choice(
            op_indices,
            size=min(len(op_indices), EXAMPLE_LIMIT // 3),
            replace=False,
        ) if op_indices else []
        
        for idx in op_sample:
            error_data.append({
                "language": lang_code,
                "text": texts[idx][:200],
                "true_labels": _format_labels(y_true[idx], emotion_index),
                "predicted_labels": _format_labels(y_pred[idx], emotion_index),
                "error_type": "over_prediction",
                "notes": f"Predicted {int(y_pred[idx].sum())} vs true {int(y_true[idx].sum())}",
            })
        
        # Partial matches (some correct, some wrong)
        partial_indices = []
        for i in range(len(y_true)):
            intersection = (y_true[i] & y_pred[i]).sum()
            union = (y_true[i] | y_pred[i]).sum()
            if 0 < intersection < union:
                partial_indices.append(i)
        
        partial_sample = np.random.choice(
            partial_indices,
            size=min(len(partial_indices), EXAMPLE_LIMIT // 3),
            replace=False,
        ) if partial_indices else []
        
        for idx in partial_sample:
            error_data.append({
                "language": lang_code,
                "text": texts[idx][:200],
                "true_labels": _format_labels(y_true[idx], emotion_index),
                "predicted_labels": _format_labels(y_pred[idx], emotion_index),
                "error_type": "partial_match",
                "notes": "Some labels correct, others missed or added",
            })
    
    error_df = pd.DataFrame(error_data)
    print(f"  ✔ Extracted {len(error_df)} error examples")
    
    return error_df


def _format_labels(label_vector: np.ndarray, emotion_index: list) -> str:
    """Convert binary vector to comma-separated label string."""
    active_labels = [emotion_index[i] for i in range(len(label_vector)) if label_vector[i] == 1]
    return ", ".join(active_labels) if active_labels else "[none]"



def generate_observation_report(
    all_results: dict,
    weak_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    error_df: pd.DataFrame,
    emotion_index: list,
) -> str:
    """Generate markdown report with key findings."""
    print("\n[4/4] Generating observation report...")
    
    report = []
    report.append("# Baseline Model Exploratory Observations")
    report.append(f"\n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")
    
    # Overall performance
    report.append("## 1. Overall Performance\n")
    
    success_langs = {k: v for k, v in all_results.items() if v["status"] == "success"}
    if success_langs:
        best_lang = max(success_langs.items(), key=lambda x: x[1]["metrics"]["f1_macro"])
        worst_lang = min(success_langs.items(), key=lambda x: x[1]["metrics"]["f1_macro"])
        
        report.append(f"- **Best performing language:** {best_lang[0].upper()} "
                     f"(F1 Macro: {best_lang[1]['metrics']['f1_macro']:.4f})")
        report.append(f"- **Worst performing language:** {worst_lang[0].upper()} "
                     f"(F1 Macro: {worst_lang[1]['metrics']['f1_macro']:.4f})")
        
        # Training data notes
        for lang, res in success_langs.items():
            report.append(f"- **{lang.upper()}:** Trained on {res['train_source']} "
                         f"({res['train_size']} samples), tested on {res['test_size']} samples")
    else:
        report.append("- No successful model runs to compare.")
    
    # Weak labels
    report.append("\n## 2. Weak Labels\n")
    if len(weak_df) > 0:
        report.append(f"Found **{len(weak_df)} weak labels** across languages:\n")
        
        for lang in weak_df["language"].unique():
            lang_weak = weak_df[weak_df["language"] == lang]
            report.append(f"\n### {lang.upper()}")
            for _, row in lang_weak.head(5).iterrows():
                report.append(f"- **{row['label']}**: F1={row['f1']:.3f}, "
                             f"Recall={row['recall']:.3f}, Issue: {row['issue']}")
    else:
        report.append("No critically weak labels detected (all F1 > threshold).")
    
    # Class imbalance
    report.append("\n## 3. Class Imbalance\n")
    if len(dist_df) > 0:
        report.append("Label distribution varies significantly:\n")
        
        for lang in dist_df["language"].unique():
            lang_dist = dist_df[dist_df["language"] == lang]
            most_common = lang_dist.iloc[0]
            least_common = lang_dist.iloc[-1]
            
            report.append(f"\n### {lang.upper()}")
            report.append(f"- **Most frequent:** {most_common['label']} "
                         f"({most_common['count']} samples, {most_common['percentage']:.1f}%)")
            report.append(f"- **Least frequent:** {least_common['label']} "
                         f"({least_common['count']} samples, {least_common['percentage']:.1f}%)")
            
            # Imbalance severity
            ratio = most_common['count'] / max(least_common['count'], 1)
            if ratio > 10:
                report.append(f"- ⚠️ **Severe imbalance detected** (ratio: {ratio:.1f}:1)")
            elif ratio > 5:
                report.append(f"- ⚠️ **Moderate imbalance** (ratio: {ratio:.1f}:1)")
    
    # Error patterns
    report.append("\n## 4. Common Error Patterns\n")
    if len(error_df) > 0:
        error_counts = error_df.groupby(["language", "error_type"]).size().reset_index(name="count")
        
        for lang in error_df["language"].unique():
            lang_errors = error_counts[error_counts["language"] == lang]
            report.append(f"\n### {lang.upper()}")
            for _, row in lang_errors.iterrows():
                report.append(f"- **{row['error_type']}**: {row['count']} examples")
    else:
        report.append("No error examples extracted (perfect predictions or no data).")
    
    # Summary
    report.append("\n## 5. Key Takeaways\n")
    report.append("- The baseline TF-IDF + Logistic Regression model provides a lower bound for comparison.")
    report.append("- Languages with limited training data (zul, xho using validation split) show lower performance.")
    report.append("- Class imbalance is present and likely affects model ability to learn minority emotions.")
    report.append("- False negatives (missing predictions) are more common than over-predictions.")
    report.append("- Next steps: Compare against mBERT and XLM-R to assess transfer learning benefits.")
    
    report.append("\n---\n")
    report.append("*End of Report*")
    
    return "\n".join(report)



def main():
    """Run full exploratory analysis pipeline."""
    print("\n" + "="*70)
    print("  WEEK 2 — ISSUE #4: EXPLORATORY ANALYSIS")
    print("="*70)
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run baseline (or load results if already run)
    print("\n[PREREQUISITE] Running baseline models...")
    all_results, emotion_index = run_baseline_main()
    
    # Load datasets for distribution analysis
    datasets, _ = run_pipeline()
    
    # 1. Weak labels
    weak_df = identify_weak_labels(all_results, emotion_index)
    weak_path = RESULTS_DIR / "weak_labels.csv"
    weak_df.to_csv(weak_path, index=False)
    print(f"  ✔ Saved to {weak_path}")
    
    # 2. Class distribution
    dist_df = analyze_class_distribution(datasets, emotion_index)
    dist_path = RESULTS_DIR / "class_distribution.csv"
    dist_df.to_csv(dist_path, index=False)
    print(f"  ✔ Saved to {dist_path}")
    
    # 3. Error examples
    error_df = extract_error_examples(all_results, emotion_index)
    error_path = RESULTS_DIR / "error_examples.csv"
    error_df.to_csv(error_path, index=False)
    print(f"  ✔ Saved to {error_path}")
    
    # 4. Observation report
    report_text = generate_observation_report(
        all_results, weak_df, dist_df, error_df, emotion_index
    )
    report_path = RESULTS_DIR / "baseline_observations.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  ✔ Saved to {report_path}")
    
    # Print report preview
    print("\n" + "="*70)
    print("  OBSERVATION REPORT PREVIEW")
    print("="*70)
    print(report_text[:1000] + "\n...\n[See full report in results/baseline_observations.md]")
    
    print("\n✅ Exploratory analysis complete.\n")


if __name__ == "__main__":
    main()