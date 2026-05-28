# Emotion Analysis for African Languages

### A Comparative Study of Multilingual Transformer Models for Multi-Label Emotion Classification in Bantu Languages

## Overview

This project investigates **multi-label emotion classification** for underrepresented African languages using modern NLP methods. The focus is on comparing traditional machine learning baselines against multilingual transformer models across selected Bantu languages.

Languages currently included:

* isiZulu (`zul`)
* isiXhosa (`xho`)
* Swahili (`swa`)

The project evaluates whether multilingual pretrained models such as mBERT and XLM-R outperform classical approaches in low-resource multilingual settings.

---

## Research Questions

This project aims to answer:

1. How do multilingual transformer models compare to baseline methods for multi-label emotion classification?
2. How does performance vary across linguistically related African languages?
3. What common prediction errors occur in multilingual emotion classification systems?

---

## Current Project Phases

## Week 2 — Baseline Modelling + Exploratory Analysis

Implemented:

* TF-IDF + Logistic Regression baseline
* Per-language training (`zul`, `xho`, `swa`)
* Automatic fallback training split logic where official train data is missing
* Standard evaluation metrics
* Exploratory label analysis
* Weak label detection
* Error examples

Outputs generated in `results/` include baseline metrics and observations.

---

## Week 3 — Transformer Fine-Tuning

Implemented:

* mBERT training pipeline
* XLM-R training pipeline
* Multi-label classification setup
* Shared evaluation pipeline
* Per-language experiments
* CSV metric exports

This phase compares transformer models directly against the baseline.

---

## Week 4 — Evaluation + Cross-Lingual Analysis

Implemented:

* Cross-lingual model comparison table across all three models and four metrics
* Per-label breakdown for mBERT and XLM-R (worst-performing labels)
* Weak label summary from baseline
* Outputs saved to `results/cross_lingual_comparison.csv`

Full analysis and discussion are in the written report (`documentation/report.tex`).

---

# Installation

Create and activate a virtual environment, then install dependencies:

```bash id="97r7i7"
pip install -r requirements.txt
```

Recommended Python version:

```text id="0o2n1m"
Python 3.9+
```

---

# Dataset Setup

Place datasets inside the `data/` directory using language codes:

```text id="efb2co"
zul/
xho/
swa/
```

Each language folder should contain available parquet splits such as:

```text id="kq5t5n"
train.parquet
dev.parquet
test.parquet
```

If no training split exists, the pipeline can automatically use validation/dev data as fallback training data.

---

# Running the Project

Main execution is controlled from:

```text id="9jlw8u"
main.py
```

Open `main.py` and configure the phase toggles:

```python id="m8b4ol"
RUN_WEEK2 = True
RUN_WEEK3 = False
RUN_WEEK4 = False
```

---

## Run Baseline Only

```python id="c9psbo"
RUN_WEEK2 = True
RUN_WEEK3 = False
RUN_WEEK4 = False
```

---

## Run Transformer Experiments Only

```python id="m2x6te"
RUN_WEEK2 = False
RUN_WEEK3 = True
RUN_WEEK4 = False
```

---

## Run Final Evaluation Only

```python id="n7r39l"
RUN_WEEK2 = False
RUN_WEEK3 = False
RUN_WEEK4 = True
```

Then execute:

```bash id="qthp8q"
python main.py
```

---

# Configuration Notes

## Baseline Configuration

Baseline settings are defined inside the baseline module. Typical parameters include:

* TF-IDF max features
* n-gram range
* Logistic Regression max iterations
* class balancing

---

## Transformer Configuration

Common settings inside the transformer pipeline:

```text id="kbgwya"
epochs
batch_size
learning_rate
max_length
threshold
```

Recommended starting values:

```text id="m7jby2"
epochs = 3
batch_size = 8
learning_rate = 2e-5
max_length = 128
threshold = 0.3
```

If GPU memory errors occur, reduce batch size first.

---

# Results

Outputs are written to `results/`.

Typical files include:

* baseline metrics
* transformer metrics
* combined summaries
* weak label reports
* class distribution tables
* qualitative error samples

These outputs can be used directly in the final report.

---

# Developer Notes

## Add a New Language

1. Add a dataset folder using language code.
2. Ensure parquet files exist.
3. Add language code to experiment loops.

---

## Add a New Model

Use the transformer pipeline and register a new Hugging Face model name.

---

## Reproducibility

Use fixed seeds where available:

```text id="ggcb1j"
random_state = 42
seed = 42
```

---

# Troubleshooting

## Dataset Not Found

Check folder names and parquet filenames.

## No Training Data

The pipeline will attempt fallback training using validation/dev split.

## CUDA / Memory Errors

Lower:

* batch size
* max sequence length

## Import Errors

Ensure dependencies are installed and run from project root.

---

# Project Goal

Beyond model performance, this project aims to contribute practical insights for NLP development in low-resource African languages and multilingual emotion understanding.
