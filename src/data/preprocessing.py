# preprocessing.py — intentionally minimal
#
# Text preprocessing for this project is handled at two levels:
#
#   1. Baseline (TF-IDF): tokenisation and normalisation are performed
#      internally by sklearn's TfidfVectorizer (see src/baseline/model.py).
#
#   2. Transformers (mBERT / XLM-R): text is passed directly to the
#      HuggingFace AutoTokenizer, which handles all subword tokenisation
#      and encoding (see src/transformers_pipeline/transformer_train.py).
#
# Any future corpus-level cleaning (e.g. HTML stripping, language
# detection, deduplication) should be added here.
