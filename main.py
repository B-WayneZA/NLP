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
RUN_WEEK2 = False   # data loading + baseline + exploratory analysis
RUN_WEEK3 = True  # mBERT + XLM-R fine-tuning
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
    from src.data.dataset_loader import run_pipeline
    from src.baseline.run_baseline import main as run_baseline


    # We call run_baseline once here, capture the results, then pass them
    # directly into the analysis functions to avoid running baseline twice.
    from analysis.exploratory_analysis import (
        identify_weak_labels,
        analyze_class_distribution,
        extract_error_examples,
        generate_observation_report,
        RESULTS_DIR,
    )

    # Step 1 — load data explicitly so failures surface here, not inside baseline
    print("\n[STEP 1/4] Loading datasets...")
    datasets, emotion_index = run_pipeline()

    # Step 2 — baseline (single run, results reused in step 3)
    print("\n[STEP 2/4] Running baseline...")
    all_results, emotion_index = run_baseline()

    # Step 3 — exploratory analysis using already-computed results
    print("\n[STEP 3/4] Running exploratory analysis...")
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

    # Step 4 — summary + paper export
    print("\n[STEP 4/4] Generating summary...")
    from results.results import print_summary, export_for_paper
    print_summary()
    export_for_paper()

    print("\n✔ Week 2 complete.")


# ─────────────────────────────────────────────
# WEEK 3
# ─────────────────────────────────────────────

def run_week3():
    _banner("WEEK 3: Transformer Fine-Tuning (mBERT + XLM-R)")

    from src.transformers_pipeline.run_transformer_pipeline import main as run_transformers

    run_transformers(run_mbert=True, run_xlmr=True)

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

    # from analysis.crosslingual import main as run_crosslingual
    # run_crosslingual()

    # from analysis.final_evaluation import main as run_final_eval
    # run_final_eval()

    from results.results import print_summary, export_for_paper
    print_summary()
    export_for_paper()

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