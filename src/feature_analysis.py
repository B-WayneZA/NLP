"""
feature_analysis.py
===================
Extract and visualize top TF-IDF features per emotion label (stretch goal).

Usage:
    python src/feature_analysis.py --lang swa --top_k 15
"""

import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Import from dataset loader to get emotion index
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_loader import run_pipeline


def load_model(lang_code: str, models_dir: Path = Path("models/baseline")):
    """Load a saved baseline model."""
    model_path = models_dir / f"{lang_code}_baseline.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    return data["vectorizer"], data["classifier"]


def extract_top_features(vectorizer, classifier, emotion_index: list, top_k: int = 10):
    """
    Extract top TF-IDF features per emotion.
    
    Args:
        vectorizer: Fitted TfidfVectorizer
        classifier: Fitted OneVsRestClassifier
        emotion_index: List of emotion labels
        top_k: Number of top features per label
    
    Returns:
        Dictionary mapping emotion -> [(feature, weight), ...]
    """
    feature_names = vectorizer.get_feature_names_out()
    top_features = {}
    
    for i, emotion in enumerate(emotion_index):
        # Get coefficients for this emotion's classifier
        coef = classifier.estimators_[i].coef_[0]
        
        # Get top k by absolute magnitude
        top_positive_idx = np.argsort(coef)[-top_k:][::-1]
        top_negative_idx = np.argsort(coef)[:top_k]
        
        top_features[emotion] = {
            "positive": [
                (feature_names[idx], float(coef[idx]))
                for idx in top_positive_idx
            ],
            "negative": [
                (feature_names[idx], float(coef[idx]))
                for idx in top_negative_idx
            ],
        }
    
    return top_features


def print_feature_table(features_dict: dict, lang_code: str, top_k: int):
    """Pretty-print top features per emotion."""
    print(f"\n{'='*80}")
    print(f"  TOP {top_k} TF-IDF FEATURES — {lang_code.upper()}")
    print(f"{'='*80}")
    
    for emotion, feature_data in features_dict.items():
        print(f"\n  📊 {emotion.upper()}")
        print(f"  {'-'*78}")
        
        print(f"\n    Positive indicators (strong association):")
        for feature, weight in feature_data["positive"]:
            bar = "█" * int((weight / 2) * 20)  # Scale for visualization
            print(f"      {feature:<30} {weight:>8.4f}  {bar}")
        
        print(f"\n    Negative indicators (strong dissociation):")
        for feature, weight in feature_data["negative"]:
            bar = "▓" * int((abs(weight) / 2) * 20)
            print(f"      {feature:<30} {weight:>8.4f}  {bar}")
    
    print(f"\n{'='*80}\n")


def save_features_json(features_dict: dict, lang_code: str, output_dir: Path):
    """Save features to JSON for later analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{lang_code}_top_features.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  ✔ Features saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract top TF-IDF features per emotion")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["zul", "xho", "swa"],
        help="Language code",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top features to extract per emotion",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/features",
        help="Directory to save feature JSONs",
    )
    
    args = parser.parse_args()
    
    # Load emotion index
    print("\n[1/3] Loading emotion index...")
    _, emotion_index = run_pipeline()
    print(f"  ✔ Found {len(emotion_index)} emotion labels")
    
    # Load model
    print(f"\n[2/3] Loading {args.lang} baseline model...")
    try:
        vectorizer, classifier = load_model(args.lang)
        print(f"  ✔ Model loaded")
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        print(f"  → Train the baseline first: python src/baseline.py")
        return
    
    # Extract features
    print(f"\n[3/3] Extracting top {args.top_k} features per emotion...")
    features = extract_top_features(vectorizer, classifier, emotion_index, args.top_k)
    
    # Print table
    print_feature_table(features, args.lang, args.top_k)
    
    # Save JSON
    save_features_json(features, args.lang, Path(args.output_dir))
    
    print("✅ Feature extraction complete.\n")


if __name__ == "__main__":
    main()