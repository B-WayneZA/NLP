"""
run_week2_full.py
=================
Master script to execute all Week 2 analyses:
  - Issue #3: Run baseline per language
  - Issue #4: Exploratory analysis

Usage:
    python run_week2_full.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from run_baseline import main as run_baseline
from exploratory_analysis import main as run_exploratory
from results.results import print_summary, export_for_paper


def main():
    """Run complete Week 2 pipeline."""
    print("\n" + "="*70)
    print("  WEEK 2 — COMPLETE PIPELINE")
    print("  Issues #3 and #4: Baseline + Exploratory Analysis")
    print("="*70)
    
    try:
        # Issue #3: Run baseline
        print("\n[PHASE 1] Running baseline models...")
        print("-"*70)
        all_results, emotion_index = run_baseline()
        
        # Issue #4: Exploratory analysis
        print("\n[PHASE 2] Running exploratory analysis...")
        print("-"*70)
        run_exploratory()
        
        # Generate summary
        print("\n[PHASE 3] Generating final summary...")
        print("-"*70)
        print_summary()
        
        # Export for paper
        print("\n[PHASE 4] Exporting tables for paper...")
        export_for_paper()
        
        print("\n" + "="*70)
        print("  ✅ WEEK 2 COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("  results/baseline_metrics.csv")
        print("  results/weak_labels.csv")
        print("  results/class_distribution.csv")
        print("  results/error_examples.csv")
        print("  results/baseline_observations.md")
        print("  results/paper_tables.txt")
        print("\nNext steps:")
        print("  → Review baseline_observations.md")
        print("  → Proceed to Week 3: mBERT and XLM-R comparison")
        print()
        
    except Exception as e:
        print(f"\n✗ Error during pipeline execution:")
        print(f"  {type(e).__name__}: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Ensure data/ folder exists with correct structure")
        print(f"  2. Check that dataset_loader.py is in parent directory")
        print(f"  3. Verify all dependencies installed: pandas, numpy, scikit-learn")
        raise


if __name__ == "__main__":
    main()