"""
baseline.py
===========
TF-IDF + One-vs-Rest Logistic Regression baseline for multi-label emotion classification.

Usage:
    python3 baseline.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Add parent directory to path to import dataset_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_loader import run_pipeline

# Import metrics module
from metrics import compute_metrics, compute_per_label_metrics, print_metrics_table


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RANDOM_STATE = 42
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models/baseline")

TFIDF_CONFIG = {
   "max_features": 5000,
   "ngram_range": (1, 2),
   "min_df": 2,
   "lowercase": True,
   "strip_accents": None,  # Keep original accents for Bantu languages
}

LOGREG_CONFIG = {
   "max_iter": 1000,
   "class_weight": "balanced",
   "random_state": RANDOM_STATE,
   "n_jobs": -1,  # Use all CPU cores
}


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

class BaselineModel:
   """TF-IDF + Logistic Regression multi-label classifier."""

   def __init__(
      self,
      tfidf_config: dict = None,
      logreg_config: dict = None,
   ):
      self.tfidf_config = tfidf_config or TFIDF_CONFIG
      self.logreg_config = logreg_config or LOGREG_CONFIG

      self.vectorizer = TfidfVectorizer(**self.tfidf_config)
      self.classifier = OneVsRestClassifier(
         LogisticRegression(**self.logreg_config)
      )

      self.is_fitted = False

   def fit(self, texts: list, labels: np.ndarray):
      """
      Fit the model.

      Args:
         texts: List of text strings
         labels: Binary label matrix (n_samples, n_labels)
      """
      print(f"  → Fitting TF-IDF on {len(texts)} samples...")
      X = self.vectorizer.fit_transform(texts)
      print(f"    TF-IDF shape: {X.shape}")

      print(f"  → Training OneVsRest Logistic Regression...")
      self.classifier.fit(X, labels)
      self.is_fitted = True
      print(f"    ✔ Model fitted")

   def predict(self, texts: list) -> np.ndarray:
      """
      Predict binary label matrix.

      Args:
         texts: List of text strings

      Returns:
         Binary label matrix (n_samples, n_labels)
      """
      if not self.is_fitted:
         raise RuntimeError("Model must be fitted before prediction")

      X = self.vectorizer.transform(texts)
      return self.classifier.predict(X)

   def save(self, path: Path):
      """Save model to disk."""
      path.parent.mkdir(parents=True, exist_ok=True)
      with open(path, "wb") as f:
         pickle.dump(
               {
                  "vectorizer": self.vectorizer,
                  "classifier": self.classifier,
                  "tfidf_config": self.tfidf_config,
                  "logreg_config": self.logreg_config,
               },
               f,
         )
      print(f"  ✔ Model saved to {path}")

   @classmethod
   def load(cls, path: Path):
      """Load model from disk."""
      with open(path, "rb") as f:
         data = pickle.load(f)
      model = cls(
         tfidf_config=data["tfidf_config"],
         logreg_config=data["logreg_config"],
      )
      model.vectorizer = data["vectorizer"]
      model.classifier = data["classifier"]
      model.is_fitted = True
      return model


def train_and_evaluate_language(
    lang_code: str,
    datasets: dict,
    emotion_index: list,
) -> dict:
   """
   Train and evaluate baseline model for a single language.

   Args:
      lang_code: Language code (zul, xho, swa)
      datasets: Dataset dictionary from dataset_loader
      emotion_index: List of emotion labels

   Returns:
      Dictionary containing metrics and model
   """
   print(f"\n{'='*60}")
   print(f"  LANGUAGE: {lang_code.upper()}")
   print(f"{'='*60}")

   splits = datasets[lang_code]

      # Check if train split exists
      # Determine training split using fallback logic
   if "train" in splits and splits["train"] is not None and len(splits["train"]) > 0:
      train_df = splits["train"]
      train_source = "train"

   elif "validation" in splits and splits["validation"] is not None and len(splits["validation"]) > 0:
      train_df = splits["validation"]
      train_source = "validation (fallback as train)"

   else:
      print(f"  ⚠ No training or validation data available for {lang_code}")
      return {
         "lang": lang_code,
         "model_type": "LogisticRegression",
         "status": "no_train_data",
         "metrics": None,
      }

   # Prepare training data
   train_texts = train_df["text"].tolist()
   train_labels = np.array(train_df["labels"].tolist())

   print(f"\n[1/3] Training")
   print(f"  Train source: {train_source}")
   print(f"  Train samples: {len(train_texts)}")
   print(f"  Label dimensions: {train_labels.shape}")

    # Initialize and train model
   model = BaselineModel(
      tfidf_config=TFIDF_CONFIG,
      logreg_config=LOGREG_CONFIG,
   )
   model.fit(train_texts, train_labels)

   # Prepare test data
   test_df = splits.get("test")
   if test_df is None or len(test_df) == 0:
      print(f"  ⚠ No test data available for {lang_code}")
      return {
         "lang": lang_code,
         "model_type": "LogisticRegression",
         "status": "no_test_data",
         "metrics": None,
      }

   test_texts = test_df["text"].tolist()
   test_labels = np.array(test_df["labels"].tolist())

   print(f"\n[2/3] Predicting")
   print(f"  Test samples: {len(test_texts)}")
   y_pred = model.predict(test_texts)

   # Compute metrics
   print(f"\n[3/3] Evaluating")
   metrics = compute_metrics(test_labels, y_pred, zero_division=0)
   print_metrics_table(metrics, title=f"{lang_code.upper()} Test Metrics")

   # Save model
   model_path = MODELS_DIR / f"{lang_code}_baseline.pkl"
   model.save(model_path)

   return {
      "lang": lang_code,
      "model_type": "LogisticRegression",
      "status": "success",
      "metrics": metrics,
      "model_path": str(model_path),
   }


def extract_top_features(
    model: BaselineModel,
    emotion_index: list,
    top_k: int = 10,
) -> dict:
   """
   Extract top TF-IDF features per emotion label.

   Args:
      model: Fitted BaselineModel
      emotion_index: List of emotion labels
      top_k: Number of top features to extract per label

   Returns:
      Dictionary mapping emotion to list of (feature, weight) tuples
   """
   if not model.is_fitted:
      raise RuntimeError("Model must be fitted first")

   feature_names = model.vectorizer.get_feature_names_out()
   top_features = {}

   for i, emotion in enumerate(emotion_index):
      # Get coefficients for this emotion's binary classifier
      coef = model.classifier.estimators_[i].coef_[0]

      # Get top k features by absolute weight
      top_indices = np.argsort(np.abs(coef))[-top_k:][::-1]
      top_features[emotion] = [
         (feature_names[idx], float(coef[idx]))
         for idx in top_indices
      ]

   return top_features


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
   """Run full baseline training and evaluation pipeline."""
   print("\n" + "="*60)
   print("  BASELINE: TF-IDF + LOGISTIC REGRESSION")
   print("="*60)

   # Create output directories
   RESULTS_DIR.mkdir(parents=True, exist_ok=True)
   MODELS_DIR.mkdir(parents=True, exist_ok=True)

   # Load datasets
   print("\n[STEP 1] Loading datasets...")
   datasets, emotion_index = run_pipeline()

   print(f"\n  Emotion index: {emotion_index}")
   print(f"  Number of labels: {len(emotion_index)}")

   # Train and evaluate per language
   print("\n[STEP 2] Training baseline models...")
   all_results = []

   for lang_code in ["zul", "xho", "swa"]:
      result = train_and_evaluate_language(lang_code, datasets, emotion_index)
      all_results.append(result)

   # Compile results
   print("\n[STEP 3] Compiling results...")
   results_data = []
   for result in all_results:
      if result["status"] == "success" and result["metrics"]:
         metrics = result["metrics"]
         results_data.append({
               "language": result["lang"],
               "model": result["model_type"],
               "f1_macro": metrics["f1_macro"],
               "f1_micro": metrics["f1_micro"],
               "precision_macro": metrics["precision_macro"],
               "recall_macro": metrics["recall_macro"],
               "hamming_loss": metrics["hamming_loss"],
               "jaccard_macro": metrics["jaccard_macro"],
         })
      else:
         results_data.append({
               "language": result["lang"],
               "model": result["model_type"],
               "f1_macro": None,
               "f1_micro": None,
               "precision_macro": None,
               "recall_macro": None,
               "hamming_loss": None,
               "jaccard_macro": None,
               "status": result["status"],
         })

   # Save to CSV
   results_df = pd.DataFrame(results_data)
   csv_path = RESULTS_DIR / "baseline_metrics.csv"
   results_df.to_csv(csv_path, index=False)
   print(f"\n  ✔ Results saved to {csv_path}")

   # Print summary table
   print("\n" + "="*60)
   print("  SUMMARY")
   print("="*60)
   print(f"\n{results_df.to_string(index=False)}")

   # Print simplified view
   print("\n" + "="*60)
   print("  QUICK COMPARISON")
   print("="*60)
   summary_cols = ["language", "model", "f1_macro"]
   if all(col in results_df.columns for col in summary_cols):
      print(f"\n{results_df[summary_cols].to_string(index=False)}")

   print("\n✅ Baseline pipeline complete.\n")


if __name__ == "__main__":
   main()