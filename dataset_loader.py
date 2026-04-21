"""
W2-01 to W2-03: Dataset Validation & Standardisation
=====================================================
Loads, validates, and standardises BRIGHTER parquet datasets for:
  - isiZulu  (zul)
  - isiXhosa (xho)
  - Swahili  (swa)

Output schema per split:
  text     : str
  labels   : list[int]   (multi-label binary vector)
  language : str
"""

import os
import ast
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_ROOT = Path("data/BRIGHTER")  # Adjust if your data is in a different location

LANGUAGES = {
    "zul": "isiZulu",
    "xho": "isiXhosa",
    "swa": "Swahili",
}

SPLITS = {
    "train":      "train.parquet",
    "validation": "dev.parquet",
    "test":       "test.parquet",
}

# Expected output columns (unified schema)
SCHEMA_COLUMNS = ["text", "labels", "language"]


# ─────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────

def load_raw_datasets(data_root: Path, languages: dict, splits: dict) -> dict:
    """
    Load all parquet files from disk.
    Missing splits are skipped with a warning (e.g. zul/xho have no train.parquet).

    Returns:
        raw_data[lang][split] = DataFrame (unmodified)
        Missing splits are simply absent from the inner dict.
    """
    raw_data = {}

    for lang_code in languages:
        raw_data[lang_code] = {}
        for split_name, filename in splits.items():
            filepath = data_root / lang_code / filename
            if not filepath.exists():
                print(f"  ⚠ Skipping [{lang_code}] {split_name:>10} — file not found: {filepath}")
                continue
            df = pd.read_parquet(filepath)
            raw_data[lang_code][split_name] = df
            print(f"  ✔ Loaded  [{lang_code}] {split_name:>10} — {len(df):>5} rows")

    return raw_data


# ─────────────────────────────────────────────
# STEP 2 — INSPECT SCHEMA
# ─────────────────────────────────────────────

def inspect_schema(raw_data: dict, languages: dict, n_rows: int = 3) -> dict:
    """
    Print schema summary for each language/split.
    Detects text column, label column(s), and label format.

    Returns:
        schema_info[lang] = {
            "text_col":    str,
            "label_cols":  list[str],
            "label_format": str     # "multi_label_list" | "binary_columns" | "single_label"
        }
    """
    schema_info = {}

    for lang_code, lang_name in languages.items():
        print(f"\n{'='*60}")
        print(f"  LANGUAGE: {lang_name} ({lang_code})")
        print(f"{'='*60}")

        # Use first available split as reference for schema detection
        available_splits = list(raw_data[lang_code].keys())
        print(f"\n  Available splits : {available_splits}")
        df = raw_data[lang_code][available_splits[0]]

        print(f"\n  Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"\n  First {n_rows} rows:")
        print(df.head(n_rows).to_string(index=False))
        print(f"\n  dtypes:\n{df.dtypes.to_string()}")
        print(f"\n  Null counts:\n{df.isnull().sum().to_string()}")

        # ── Detect text column ──────────────────────────
        text_col = _detect_text_column(df)
        print(f"\n  ✔ Detected text column  : '{text_col}'")

        # ── Detect label columns & format ──────────────
        label_cols, label_format = _detect_label_columns(df, text_col)
        print(f"  ✔ Detected label column(s): {label_cols}")
        print(f"  ✔ Label format            : {label_format}")

        schema_info[lang_code] = {
            "text_col":    text_col,
            "label_cols":  label_cols,
            "label_format": label_format,
        }

    return schema_info


def _detect_text_column(df: pd.DataFrame) -> str:
    """Heuristic: find the main text column."""
    preferred = ["text", "sentence", "tweet", "content", "utterance"]
    for name in preferred:
        if name in df.columns:
            return name
    # Fall back: first string column
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError("Could not detect a text column.")


def _detect_label_columns(df: pd.DataFrame, text_col: str):
    """
    Heuristic: identify label columns and their format.

    Possible formats:
      "multi_label_list"  — one column containing a list of emotion strings
      "binary_columns"    — multiple 0/1 columns, one per emotion
      "single_label"      — one column with a single string/int label
    """
    non_text_cols = [c for c in df.columns if c != text_col]

    # Case 1: binary emotion columns (dtype int/float, values in {0,1})
    binary_cols = []
    for col in non_text_cols:
        if df[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0}):
                binary_cols.append(col)
    if len(binary_cols) > 1:
        return binary_cols, "binary_columns"

    # Case 2: one column with list/str of multiple labels
    for col in non_text_cols:
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if isinstance(sample, list):
            return [col], "multi_label_list"
        if isinstance(sample, str):
            try:
                parsed = ast.literal_eval(sample)
                if isinstance(parsed, list):
                    return [col], "multi_label_list"
            except (ValueError, SyntaxError):
                pass

    # Case 3: single label column
    if len(non_text_cols) == 1:
        return non_text_cols, "single_label"
    if non_text_cols:
        return [non_text_cols[0]], "single_label"

    raise ValueError("Could not detect label columns.")


# ─────────────────────────────────────────────
# STEP 3 — STANDARDISE
# ─────────────────────────────────────────────

def build_emotion_index(raw_data: dict, schema_info: dict) -> list:
    """
    Build a globally consistent, sorted list of all emotion labels
    found across all languages and splits.

    Returns:
        emotion_index : list[str]  — sorted, deduplicated emotion names
    """
    all_emotions = set()

    for lang_code, info in schema_info.items():
        fmt = info["label_format"]
        label_cols = info["label_cols"]

        if fmt == "binary_columns":
            all_emotions.update(label_cols)

        elif fmt == "multi_label_list":
            col = label_cols[0]
            for split_df in raw_data[lang_code].values():
                for val in split_df[col].dropna():
                    labels = val if isinstance(val, list) else ast.literal_eval(val)
                    all_emotions.update(labels)

        elif fmt == "single_label":
            col = label_cols[0]
            for split_df in raw_data[lang_code].values():
                all_emotions.update(split_df[col].dropna().unique())

    emotion_index = sorted(all_emotions)
    print(f"\n  ✔ Global emotion index ({len(emotion_index)} classes): {emotion_index}")
    return emotion_index


def _to_binary_vector(labels_input, emotion_index: list, label_format: str) -> list:
    """
    Convert a raw label value to a binary vector aligned with emotion_index.
    Handles all three label formats.
    """
    n = len(emotion_index)
    idx_map = {e: i for i, e in enumerate(emotion_index)}
    vec = [0] * n

    if label_format == "binary_columns":
        # labels_input is a dict {emotion: 0/1}
        for emotion, val in labels_input.items():
            if emotion in idx_map and val == 1:
                vec[idx_map[emotion]] = 1

    elif label_format == "multi_label_list":
        labels = (
            label_format and
            (labels_input if isinstance(labels_input, list)
             else ast.literal_eval(labels_input))
        )
        # re-parse safely
        if isinstance(labels_input, list):
            labels = labels_input
        else:
            try:
                labels = ast.literal_eval(labels_input)
            except (ValueError, SyntaxError):
                labels = []
        for emotion in labels:
            if emotion in idx_map:
                vec[idx_map[emotion]] = 1

    elif label_format == "single_label":
        label = str(labels_input)
        if label in idx_map:
            vec[idx_map[label]] = 1

    return vec


def standardise_split(
    df: pd.DataFrame,
    lang_code: str,
    schema: dict,
    emotion_index: list,
) -> pd.DataFrame:
    """
    Transform a single split DataFrame into the unified schema:
        text | labels | language

    Steps:
      1. Drop rows with null text or null label columns
      2. Build binary label vector per row
      3. Add language column
      4. Return clean DataFrame
    """
    text_col    = schema["text_col"]
    label_cols  = schema["label_cols"]
    label_format = schema["label_format"]

    # ── 1. Drop malformed rows ──────────────────────────
    required_cols = [text_col] + label_cols
    df_clean = df.dropna(subset=required_cols).copy()
    dropped = len(df) - len(df_clean)
    if dropped:
        print(f"    ⚠ Dropped {dropped} rows with nulls in [{lang_code}]")

    # ── 2. Build binary vectors ─────────────────────────
    def row_to_vector(row):
        if label_format == "binary_columns":
            label_dict = {col: int(row[col]) for col in label_cols}
            return _to_binary_vector(label_dict, emotion_index, label_format)
        else:
            return _to_binary_vector(row[label_cols[0]], emotion_index, label_format)

    df_clean["labels"] = df_clean.apply(row_to_vector, axis=1)

    # ── 3. Rename text column & add language ───────────
    df_clean = df_clean.rename(columns={text_col: "text"})
    df_clean["language"] = lang_code

    # ── 4. Return only unified schema columns ──────────
    return df_clean[SCHEMA_COLUMNS].reset_index(drop=True)


def standardise_all(
    raw_data: dict,
    schema_info: dict,
    emotion_index: list,
) -> dict:
    """
    Standardise all languages and splits.

    Returns:
        datasets[lang][split] = standardised DataFrame
    """
    datasets = {}

    for lang_code in raw_data:
        datasets[lang_code] = {}
        schema = schema_info[lang_code]

        for split_name, df in raw_data[lang_code].items():
            std_df = standardise_split(df, lang_code, schema, emotion_index)
            datasets[lang_code][split_name] = std_df
            print(
                f"  ✔ Standardised [{lang_code}] {split_name:>10} "
                f"— {len(std_df):>5} rows  "
                f"| label_dim={len(emotion_index)}"
            )

    return datasets


# ─────────────────────────────────────────────
# STEP 4 — VALIDATE OUTPUT
# ─────────────────────────────────────────────

def validate_datasets(datasets: dict, emotion_index: list) -> bool:
    """
    Run post-standardisation checks.
    Asserts: schema, label vector length, no nulls, language column.
    """
    print(f"\n{'='*60}")
    print("  VALIDATION")
    print(f"{'='*60}")
    all_passed = True
    n = len(emotion_index)

    for lang_code, splits in datasets.items():
        for split_name, df in splits.items():
            if df is None:
                print(f"  ⚠ [{lang_code}][{split_name}] skipped — not available")
                continue
            tag = f"[{lang_code}][{split_name}]"

            # Check columns
            assert list(df.columns) == SCHEMA_COLUMNS, \
                f"{tag} Column mismatch: {list(df.columns)}"

            # Check no nulls
            assert df.isnull().sum().sum() == 0, \
                f"{tag} Unexpected nulls found"

            # Check label vector length
            bad_len = df["labels"].apply(lambda v: len(v) != n).sum()
            assert bad_len == 0, \
                f"{tag} {bad_len} rows have incorrect label vector length"

            # Check language column
            assert (df["language"] == lang_code).all(), \
                f"{tag} Incorrect language values"

            print(f"  ✔ {tag} passed all checks — {len(df)} rows")

    return all_passed


# ─────────────────────────────────────────────
# STEP 5 — SUMMARY STATS
# ─────────────────────────────────────────────

def print_summary(datasets: dict, emotion_index: list):
    """Print label distribution summary per language/split."""
    print(f"\n{'='*60}")
    print("  LABEL DISTRIBUTION SUMMARY")
    print(f"{'='*60}")

    for lang_code, splits in datasets.items():
        print(f"\n  [{lang_code}]")
        for split_name, df in splits.items():
            label_matrix = np.array(df["labels"].tolist())
            counts = label_matrix.sum(axis=0)
            print(f"\n    {split_name} (n={len(df)}):")
            for emotion, count in zip(emotion_index, counts):
                bar = "█" * int(count / max(counts) * 20)
                print(f"      {emotion:<20} {int(count):>5}  {bar}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(data_root: Path = DATA_ROOT) -> tuple:
    """
    Full W2 pipeline. Returns (datasets, emotion_index).

    datasets : dict[lang][split] -> DataFrame
    emotion_index : list[str]
    """
    print("\n" + "="*60)
    print("  W2 — DATASET VALIDATION & STANDARDISATION")
    print("="*60)

    # 1. Load
    print("\n[1/5] Loading raw datasets...")
    raw_data = load_raw_datasets(data_root, LANGUAGES, SPLITS)

    # 2. Inspect
    print("\n[2/5] Inspecting schemas...")
    schema_info = inspect_schema(raw_data, LANGUAGES)

    # 3. Build global emotion index
    print("\n[3/5] Building global emotion index...")
    emotion_index = build_emotion_index(raw_data, schema_info)

    # 4. Standardise
    print("\n[4/5] Standardising all datasets...")
    datasets = standardise_all(raw_data, schema_info, emotion_index)

    # 5. Validate
    print("\n[5/5] Validating output...")
    validate_datasets(datasets, emotion_index)

    # Summary
    print_summary(datasets, emotion_index)

    print("\n✅ Pipeline complete. Datasets ready for modelling.\n")
    return datasets, emotion_index


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    datasets, emotion_index = run_pipeline()

    # ── Example output ──────────────────────────────────
    print("\n── Example: datasets['zul']['train'].head() ──")
    print(datasets["zul"]["test"].head())

    print("\n── Example: datasets['xho']['validation'].head() ──")
    print(datasets["xho"]["validation"].head())

    print("\n── Example: datasets['swa']['test'].head() ──")
    print(datasets["swa"]["test"].head())

    print(f"\n── Emotion index ──\n{emotion_index}")