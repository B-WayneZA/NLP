# Baseline Model Exploratory Observations

**Generated:** 2026-05-28 10:31:55

---

## 1. Overall Performance

- **Best performing language:** XHO (F1 Macro: 0.2650)
- **Worst performing language:** SWA (F1 Macro: 0.2504)
- **ZUL:** Trained on validation (875 samples), tested on 2047 samples
- **XHO:** Trained on validation (682 samples), tested on 1594 samples
- **SWA:** Trained on train (3307 samples), tested on 3312 samples

## 2. Weak Labels

Found **5 weak labels** across languages:


### SWA
- **fear**: F1=0.067, Recall=0.064, Issue: extremely_low_f1

### XHO
- **disgust**: F1=0.017, Recall=0.125, Issue: extremely_low_f1
- **fear**: F1=0.044, Recall=0.065, Issue: extremely_low_f1
- **anger**: F1=0.069, Recall=0.091, Issue: extremely_low_f1

### ZUL
- **disgust**: F1=0.075, Recall=0.119, Issue: extremely_low_f1

## 3. Class Imbalance

Label distribution varies significantly:


### SWA
- **Most frequent:** surprise (536 samples, 16.2%)
- **Least frequent:** fear (93 samples, 2.8%)
- ⚠️ **Moderate imbalance** (ratio: 5.8:1)

### XHO
- **Most frequent:** sadness (281 samples, 41.2%)
- **Least frequent:** disgust (3 samples, 0.4%)
- ⚠️ **Severe imbalance detected** (ratio: 93.7:1)

### ZUL
- **Most frequent:** sadness (179 samples, 20.5%)
- **Least frequent:** disgust (25 samples, 2.9%)
- ⚠️ **Moderate imbalance** (ratio: 7.2:1)

## 4. Common Error Patterns


### ZUL
- **false_negative**: 3 examples
- **over_prediction**: 3 examples
- **partial_match**: 3 examples

### XHO
- **false_negative**: 3 examples
- **over_prediction**: 3 examples
- **partial_match**: 3 examples

### SWA
- **false_negative**: 3 examples
- **over_prediction**: 3 examples
- **partial_match**: 3 examples

## 5. Key Takeaways

- The baseline TF-IDF + Logistic Regression model provides a lower bound for comparison.
- Languages with limited training data (zul, xho using validation split) show lower performance.
- Class imbalance is present and likely affects model ability to learn minority emotions.
- False negatives (missing predictions) are more common than over-predictions.
- Next steps: Compare against mBERT and XLM-R to assess transfer learning benefits.

---

*End of Report*