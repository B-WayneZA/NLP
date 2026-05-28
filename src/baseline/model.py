"""

TF-IDF + One-vs-Rest Logistic Regression baseline for multi-label emotion classification.

"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


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

