

## Overview

This implements a multi-label emotion classification baseline using:
- **TF-IDF** features (5000 max features, 1-2 ngrams)
- **One-vs-Rest Logistic Regression** (balanced class weights)

Trained and evaluated separately for **isiZulu (zul)**, **isiXhosa (xho)**, and **Swahili (swa)**.

---

## File Structure

```
.
├── data/                          # Dataset directory (BRIGHTER parquet files)
│   ├── BRIGHTER/
│   │   ├── zul/
│   │   │   ├── dev.parquet
│   │   │   └── test.parquet
│   │   ├── xho/
│   │   │   ├── dev.parquet
│   │   │   └── test.parquet
│   │   └── swa/
│   │       ├── train.parquet
│   │       ├── dev.parquet
│   │       └── test.parquet
│
├── src/
│   ├── data/
│   │   └── dataset_loader.py      # Data validation & standardisation
│   ├── baseline/
│   │   ├── model.py               # BaselineModel class (TF-IDF + OvR LogReg)
│   │   └── run_baseline.py        # Training entry point
│   └── metrics.py             # Multi-label evaluation metrics
│
├── results/
│   └── baseline_metrics.csv   # Output: per-language metrics
│
└── models/
    └── baseline/
        ├── zul_baseline.pkl
        ├── xho_baseline.pkl
        └── swa_baseline.pkl
```

---

## Installation

```bash
pip install pandas numpy scikit-learn pyarrow
```

---

## Usage

### 1. Run the full pipeline

```bash
cd /path/to/project
python src/baseline/run_baseline.py
```

This will:
1. Load and standardise datasets using `src/data/dataset_loader.py`
2. Train TF-IDF + Logistic Regression for each language
3. Evaluate on test sets
4. Save results to `results/baseline_metrics.csv`
5. Save models to `models/baseline/`

### 2. Expected output

**Console:**
```
BASELINE: TF-IDF + LOGISTIC REGRESSION
============================================================

[STEP 1] Loading datasets...
  ✔ Loaded  [swa] train      —  1234 rows
  ⚠ Skipping [zul] train     — file not found
  ✔ Loaded  [zul] validation —   456 rows
  ...

[STEP 2] Training baseline models...

============================================================
  LANGUAGE: SWA
============================================================

[1/3] Training
  Train samples: 1234
  → Fitting TF-IDF on 1234 samples...
    TF-IDF shape: (1234, 5000)
  → Training OneVsRest Logistic Regression...
    ✔ Model fitted

[2/3] Predicting
  Test samples: 300

[3/3] Evaluating
============================================================
  SWA Test Metrics
============================================================
  f1_macro              0.6123
  f1_micro              0.6845
  precision_macro       0.5987
  recall_macro          0.6301
============================================================

  ✔ Model saved to models/baseline/swa_baseline.pkl

...

[STEP 3] Compiling results...
  ✔ Results saved to results/baseline_metrics.csv

============================================================
  SUMMARY
============================================================

language            model  f1_macro  f1_micro  precision_macro  recall_macro
     zul  LogisticRegression       NaN       NaN              NaN           NaN
     xho  LogisticRegression       NaN       NaN              NaN           NaN
     swa  LogisticRegression    0.6123    0.6845           0.5987        0.6301

============================================================
  QUICK COMPARISON
============================================================

language            model  f1_macro
     zul  LogisticRegression       NaN
     xho  LogisticRegression       NaN
     swa  LogisticRegression    0.6123

✅ Baseline pipeline complete.
```

**CSV output (`results/baseline_metrics.csv`):**
```csv
language,model,f1_macro,f1_micro,precision_macro,recall_macro,hamming_loss,jaccard_macro
zul,LogisticRegression,,,,,
xho,LogisticRegression,,,,,
swa,LogisticRegression,0.6123,0.6845,0.5987,0.6301,0.1234,0.4567
```

---

## Notes

### Missing Training Data

- **zul** and **xho** have no `train.parquet` → models cannot be trained
- Pipeline logs `⚠ No training data available` and skips gracefully
- **swa** has full train/dev/test → model trained normally

This is expected and aligns with your zero-shot evaluation design for isiZulu and isiXhosa.

### Reproducibility

- All random seeds set to `42`
- TF-IDF settings frozen in `TFIDF_CONFIG`
- Logistic Regression config frozen in `LOGREG_CONFIG`

### Performance Notes (CPU)

- Training time: ~1-3 minutes per language on CPU
- TF-IDF vectorization: ~10-30 seconds depending on corpus size
- Prediction: near-instant (<1 second)

---

## Next Steps

Once baseline is complete, you'll compare against:
- **mBERT** (multilingual BERT)
- **XLM-R** (cross-lingual RoBERTa)

This baseline serves as the **lower bound** for your research questions.

---

## Troubleshooting

**Error: `FileNotFoundError`**
- Ensure `data/` folder is in the project root
- Check that parquet files exist for each language

**Error: `ModuleNotFoundError: No module named 'dataset_loader'`**
- Run from project root (not inside `src/`)
- Or adjust `sys.path.insert(0, ...)` in `baseline.py`

**Warning: `No training data available`**
- Expected for `zul` and `xho` given your dataset structure
- Only `swa` will produce trained models and metrics