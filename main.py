"""

Central control centre for the emotion classification pipeline.


All imports are deferred inside functions so a missing dependency
(e.g. torch not installed) only breaks the stage that needs it,
not the entire script.

"""

import sys
from pathlib import Path

# Ensure project root is on the path so all sub-packages resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────
# TOGGLES — set to True to enable each stage
# ─────────────────────────────────────────────
RUN_WEEK2 = True   # data loading + baseline + exploratory analysis
RUN_WEEK3 = False  # mBERT + XLM-R fine-tuning
RUN_WEEK4 = False  # cross-lingual analysis + final evaluation


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ─────────────────────────────────────────────
# WEEK 2
# ─────────────────────────────────────────────

def run_week2():
    _banner("WEEK 2: Baseline + Exploratory Analysis")

    # Deferred imports — torch not required here
    from src.baseline.run_baseline import main as run_baseline

    # We call run_baseline once here — it returns datasets to avoid a second
    # run_pipeline() call inside the exploratory analysis step.
    from src.analysis.exploratory_analysis import (
        identify_weak_labels,
        analyze_class_distribution,
        extract_error_examples,
        generate_observation_report,
        RESULTS_DIR,
    )

    # Step 1 — baseline (single pass; returns datasets for reuse below)
    print("\n[STEP 1/3] Running baseline...")
    all_results, emotion_index, datasets = run_baseline()

    # Step 2 — exploratory analysis using already-loaded datasets
    print("\n[STEP 2/3] Running exploratory analysis...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    weak_df  = identify_weak_labels(all_results, emotion_index)
    weak_df.to_csv(RESULTS_DIR / "weak_labels.csv", index=False)

    dist_df  = analyze_class_distribution(datasets, emotion_index)
    dist_df.to_csv(RESULTS_DIR / "class_distribution.csv", index=False)

    error_df = extract_error_examples(all_results, emotion_index)
    error_df.to_csv(RESULTS_DIR / "error_examples.csv", index=False)

    report   = generate_observation_report(
        all_results, weak_df, dist_df, error_df, emotion_index
    )
    (RESULTS_DIR / "baseline_observations.md").write_text(report, encoding="utf-8")
    print("  ✔ Observation report saved")

    # Step 3 — summary
    print("\n[STEP 3/3] Generating summary...")
    from results.results import print_summary
    print_summary()

    print("\n✔ Week 2 complete.")


# ─────────────────────────────────────────────
# WEEK 3
# ─────────────────────────────────────────────

def run_week3():
    _banner("WEEK 3: Transformer Fine-Tuning (mBERT + XLM-R)")

    from src.transformers_pipeline.run_mbert import main as run_mbert
    from src.transformers_pipeline.run_xlmr import main as run_xlmr

    print("\n[1/2] Fine-tuning mBERT...")
    run_mbert()

    print("\n[2/2] Fine-tuning XLM-R...")
    run_xlmr()

    # Print summary after transformers complete
    print("\n[POST-TRAINING] Generating summary...")
    from results.results import print_summary
    print_summary()

    print("\n✔ Week 3 complete.")


# ─────────────────────────────────────────────
# WEEK 4 (stubs for now)
# ─────────────────────────────────────────────

def run_week4():
    _banner("WEEK 4: Evaluation + Cross-Lingual Analysis")

    from results.results import (
        compare_languages,
        load_weak_labels,
        load_class_distribution,
        print_summary,
        RESULTS_DIR,
    )
    import pandas as pd

    print("\n[1/4] Cross-lingual model comparison...")
    for metric in ("f1_macro", "f1_micro", "precision_macro", "recall_macro"):
        try:
            cmp = compare_languages(metric)
            print(f"\n  {metric}:")
            print(cmp.to_string(index=False))
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")

    # Save cross-lingual comparison CSV for all key metrics
    try:
        rows = []
        for metric in ("f1_macro", "f1_micro", "precision_macro", "recall_macro"):
            row = compare_languages(metric)
            row.insert(0, "metric", metric)
            rows.append(row)
        cross_df = pd.concat(rows, ignore_index=True)
        cross_df.to_csv(RESULTS_DIR / "cross_lingual_comparison.csv", index=False)
        print("\n  ✔ Saved results/cross_lingual_comparison.csv")
    except FileNotFoundError:
        pass

    print("\n[2/4] Per-label performance breakdown...")
    for fname, label in [
        ("mbert_per_label_metrics.csv",  "mBERT"),
        ("xlmr_per_label_metrics.csv",   "XLM-R"),
    ]:
        path = RESULTS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            print(f"\n  {label} — worst 5 labels by F1:")
            print(df.nsmallest(5, "f1")[["language", "label", "f1", "recall", "support"]].to_string(index=False))
        else:
            print(f"  ⚠ {fname} not found")

    print("\n[3/4] Weak label summary (baseline)...")
    try:
        weak_df = load_weak_labels()
        print(weak_df[["language", "label", "f1", "recall", "support", "issue"]].to_string(index=False))
    except FileNotFoundError:
        print("  ⚠ weak_labels.csv not found — run Week 2 first")

    print("\n[4/4] Final results summary...")
    print_summary()

    print("\n✔ Week 4 complete.")




def main():
    #  at least one stage must be enabled
    if not any([RUN_WEEK2, RUN_WEEK3, RUN_WEEK4]):
        print("⚠ All stages are disabled. Set at least one RUN_WEEK* = True.")
        return

    try:
        if RUN_WEEK2:
            run_week2()

        if RUN_WEEK3:
            run_week3()

        if RUN_WEEK4:
            run_week4()

        print("\n✔ Pipeline completed successfully.\n")

    except Exception as e:
        print(f"\n✗ Pipeline failed — {type(e).__name__}: {e}")
        raise  # Preserve full traceback


if __name__ == "__main__":
    main()