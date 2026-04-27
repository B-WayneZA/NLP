"""

PyTorch Dataset wrapper for multi-label emotion classification.
Handles tokenization for both mBERT and XLM-R.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for multi-label emotion classification.
    Converts text + binary label vectors into tokenized tensors.
    """

    def __init__(
        self,
        texts: list,
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            texts      : List of raw text strings
            labels     : Binary label matrix (n_samples, n_labels)
            tokenizer  : HuggingFace tokenizer
            max_length : Max token length (128 keeps CPU memory sane)
        """
        self.texts = texts
        self.labels = labels.astype(np.float32)  # BCEWithLogitsLoss needs float
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.float32),
        }


def build_datasets(
    splits: dict,
    emotion_index: list,
    tokenizer: AutoTokenizer,
    lang_code: str,
    max_length: int = 128,
) -> dict:
    """
    Build train/validation/test EmotionDatasets with fallback logic.
    Mirrors Week 2 baseline fallback: if no train, use validation.

    Args:
        splits       : dict of DataFrames {"train", "validation", "test"}
        emotion_index: List of emotion label names
        tokenizer    : Fitted HuggingFace tokenizer
        lang_code    : Language code for logging
        max_length   : Max token length

    Returns:
        {
            "train":      EmotionDataset | None
            "validation": EmotionDataset | None
            "test":       EmotionDataset | None
            "train_source": "train" | "validation"
        }
    """
    def df_to_arrays(df):
        texts  = df["text"].tolist()
        labels = np.array(df["labels"].tolist(), dtype=np.float32)
        return texts, labels

    # Determine train source (mirror baseline fallback)
    if "train" in splits and splits["train"] is not None and len(splits["train"]) > 0:
        train_texts, train_labels = df_to_arrays(splits["train"])
        train_source = "train"
    elif "validation" in splits and splits["validation"] is not None and len(splits["validation"]) > 0:
        print(f"  ⚠ [{lang_code}] No train split — using validation as train")
        train_texts, train_labels = df_to_arrays(splits["validation"])
        train_source = "validation"
    else:
        print(f"  ✗ [{lang_code}] No training data available")
        return {
            "train":        None,
            "validation":   None,
            "test":         None,
            "train_source": None,
        }

    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)

    # Validation split (optional — used for eval_strategy="epoch")
    val_dataset = None
    if "validation" in splits and splits["validation"] is not None and train_source == "train":
        val_texts, val_labels = df_to_arrays(splits["validation"])
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length)

    # Test split
    test_dataset = None
    if "test" in splits and splits["test"] is not None and len(splits["test"]) > 0:
        test_texts, test_labels = df_to_arrays(splits["test"])
        test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length)

    print(f"  ✔ [{lang_code}] Datasets built — "
          f"train={len(train_dataset)} "
          f"val={len(val_dataset) if val_dataset else 'N/A'} "
          f"test={len(test_dataset) if test_dataset else 'N/A'}")

    return {
        "train":        train_dataset,
        "validation":   val_dataset,
        "test":         test_dataset,
        "train_source": train_source,
    }