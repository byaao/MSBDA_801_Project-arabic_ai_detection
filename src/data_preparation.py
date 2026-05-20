"""
data_preparation.py
===================
Functions for:
  1. Downloading the Hugging Face dataset
  2. Creating a binary-labelled Spark DataFrame (human=0, AI=1)
  3. Arabic-specific text preprocessing (Spark UDFs) — using CAMeL Tools
     to match the approach in preprocessing v2.5
  4. MapReduce-style corpus statistics (word count, n-gram frequency)
  5. Stratified train / validation / test split
"""

from __future__ import annotations

import re
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType

from src.utils import (
    logger,
    AI_MODEL_COLS, LABEL_COL, TEXT_COL, SOURCE_COL, SPLIT_COL,
    DATA_RAW,
)

# ---------------------------------------------------------------------------
# 1. Dataset download
# ---------------------------------------------------------------------------

DATASET_ID = "KFUPM-JRCAI/arabic-generated-abstracts"
HF_SPLITS  = ["by_polishing", "from_title", "from_title_and_content"]


def download_dataset(cache_dir: Optional[str] = None) -> dict:
    """
    Download the KFUPM Arabic-generated-abstracts dataset from Hugging Face.

    The three generation-method subsets are treated as Hugging Face *splits*
    (not configs), so we use split=<name> when calling load_dataset.

    Returns
    -------
    dict  {split_name: pandas.DataFrame}

    Side effects
    ------------
    Saves Parquet copies to data/raw/<split>.parquet
    """
    from datasets import load_dataset  # type: ignore

    logger.info("Downloading dataset '%s' …", DATASET_ID)
    result = {}

    for split in HF_SPLITS:
        logger.info("  ↓ split: %s", split)
        hf_ds = load_dataset(
            DATASET_ID,
            split=split,
            cache_dir=cache_dir or str(DATA_RAW / "hf_cache"),
        )
        pdf = hf_ds.to_pandas()
        out_path = DATA_RAW / f"{split}.parquet"
        pdf.to_parquet(out_path, index=False)
        result[split] = pdf
        logger.info("    → %d rows saved to %s", len(pdf), out_path)

    return result


# ---------------------------------------------------------------------------
# 2. Build binary-labelled Spark DataFrame
# ---------------------------------------------------------------------------

def build_labelled_dataframe(spark: SparkSession) -> DataFrame:
    """
    Load the three raw Parquet files and melt them into a single
    binary-labelled DataFrame.

    Result schema
    -------------
    text               StringType   — abstract text
    label              IntegerType  — 0 = human, 1 = AI
    source_model       StringType   — "human"|"allam"|"jais"|"llama"|"openai"
    generation_method  StringType   — by_polishing|from_title|from_title_and_content
    """
    all_frames: List[DataFrame] = []

    for split in HF_SPLITS:
        path = str(DATA_RAW / f"{split}.parquet")
        logger.info("Reading raw parquet: %s", path)
        sdf = spark.read.parquet(path).withColumn(SPLIT_COL, F.lit(split))

        # Human rows
        human_df = (
            sdf.select(
                F.col("original_abstract").alias(TEXT_COL),
                F.lit(0).cast(IntegerType()).alias(LABEL_COL),
                F.lit("human").alias(SOURCE_COL),
                F.col(SPLIT_COL),
            )
            .filter(F.col(TEXT_COL).isNotNull() & (F.length(TEXT_COL) > 10))
        )
        all_frames.append(human_df)

        # AI rows — one row per model column
        for model_col in AI_MODEL_COLS:
            if model_col not in sdf.columns:
                continue
            model_name = model_col.replace("_generated_abstract", "")
            ai_df = (
                sdf.select(
                    F.col(model_col).alias(TEXT_COL),
                    F.lit(1).cast(IntegerType()).alias(LABEL_COL),
                    F.lit(model_name).alias(SOURCE_COL),
                    F.col(SPLIT_COL),
                )
                .filter(F.col(TEXT_COL).isNotNull() & (F.length(TEXT_COL) > 10))
            )
            all_frames.append(ai_df)

    combined = all_frames[0]
    for df in all_frames[1:]:
        combined = combined.union(df)

    total = combined.count()
    logger.info("Labelled dataset built — %d total rows", total)
    return combined


# ---------------------------------------------------------------------------
# 3. Arabic text preprocessing using CAMeL Tools
# ---------------------------------------------------------------------------
# Normalization pipeline:
#   1. normalize_unicode (NFKC)    — CAMeL
#   2. dediac_ar                   — CAMeL: remove tashkeel
#   3. normalize_alef_ar           — CAMeL: أ/إ/آ/ٱ → ا
#   4. normalize_alef_maksura_ar   — CAMeL: ى → ي
#   5. Remove non-Arabic chars (keep spaces)
#   6. Remove stop words (NLTK Arabic + custom domain stops)
#   7. Lemmatize via MLEDisambiguator (lazy-loaded, MSA model)

# Patterns
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ARABIC_PATTERN = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]")
ARABIC_TOKEN_PATTERN = re.compile(r"^[\u0600-\u06FF\u0750-\u077F]+$")
DIACRITIC_PATTERN = re.compile(r"[\u064B-\u065F\u0670]")

# Stopwords
ARABIC_CUSTOM_STOPWORDS = {
    "فقد", "ابن", "وعلي", "اليه", "انها", "عليها", "فهو", "وهذا", "وفي",
    "لهذا", "لهذه", "التي", "اليها", "بن", "وغيرها", "وذلك", "وهي", "لان",
    "اخري", "ومع", "وكيف", "لذلك", "وكذا", "والتي", "فهي", "كانت", "وعليه",
    "وقد", "عنها",
}

# Cache
_CACHED_STOPWORDS: Optional[Set[str]] = None
_CACHED_MLE = None


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text: unicode, diacritics, alef variants."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if not text:
        return ""

    try:
        from camel_tools.utils.normalize import (
            normalize_unicode, normalize_alef_ar, normalize_alef_maksura_ar,
        )
        from camel_tools.utils.dediac import dediac_ar

        text = normalize_unicode(text, compatibility=True)
        text = dediac_ar(text)
        text = normalize_alef_ar(text)
        text = normalize_alef_maksura_ar(text)
    except ImportError:
        text = DIACRITIC_PATTERN.sub("", text)
        text = re.sub(r"[أإآٱ]", "ا", text)
        text = text.replace("ى", "ي").replace("ة", "ه")

    return WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_non_arabic_noise(text: str) -> str:
    """Remove non-Arabic characters."""
    if not text:
        return ""
    text = NON_ARABIC_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize_arabic_text(text: str) -> List[str]:
    """Tokenize Arabic text and filter non-Arabic tokens."""
    if not text:
        return []

    try:
        from camel_tools.tokenizers.word import simple_word_tokenize
        tokens = simple_word_tokenize(text)
    except ImportError:
        tokens = text.split()

    return [t for t in tokens if ARABIC_TOKEN_PATTERN.match(t)]


def get_arabic_stopwords() -> Set[str]:
    """Load and cache Arabic stopwords."""
    global _CACHED_STOPWORDS

    if _CACHED_STOPWORDS is None:
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                raw = set(stopwords.words("arabic"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                raw = set(stopwords.words("arabic"))
            nltk_stops = {normalize_arabic_text(w) for w in raw}
        except ImportError:
            nltk_stops = set()

        custom_stops = {normalize_arabic_text(w) for w in ARABIC_CUSTOM_STOPWORDS}
        _CACHED_STOPWORDS = nltk_stops | custom_stops

    return _CACHED_STOPWORDS


def remove_arabic_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords and short tokens."""
    if not tokens:
        return []
    stops = get_arabic_stopwords()
    return [t for t in tokens if t not in stops and len(t) > 1]


def _get_mle_disambiguator():
    """Lazy-load MLEDisambiguator."""
    global _CACHED_MLE

    if _CACHED_MLE is None:
        try:
            from camel_tools.disambig.mle import MLEDisambiguator
            _CACHED_MLE = MLEDisambiguator.pretrained()
        except (ImportError, Exception) as e:
            logger.warning(f"MLEDisambiguator unavailable: {e}")
            _CACHED_MLE = False

    return _CACHED_MLE if _CACHED_MLE is not False else None


def lemmatize_arabic_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using MLEDisambiguator."""
    if not tokens:
        return []

    mle = _get_mle_disambiguator()
    if mle is None:
        return tokens

    try:
        disamb = mle.disambiguate(tokens)
        result = []
        for d in disamb:
            lemma = d.analyses[0].analysis.get("lemma", d.word) if hasattr(d, "analyses") and d.analyses else d.word
            lemma = normalize_arabic_text(lemma or d.word)
            if lemma:
                result.append(lemma)
        return result
    except Exception:
        return tokens


def preprocess_arabic(text: str) -> List[str]:
    """Complete preprocessing pipeline."""
    if not isinstance(text, str) or not text.strip():
        return []

    text = normalize_arabic_text(text)
    text = remove_non_arabic_noise(text)
    tokens = tokenize_arabic_text(text)
    tokens = remove_arabic_stopwords(tokens)
    tokens = lemmatize_arabic_tokens(tokens)
    
    return " ".join(tokens)


# Register as Spark UDF (used in notebooks and streaming pipeline)
preprocess_udf = F.udf(preprocess_arabic, StringType())


def apply_preprocessing(df: DataFrame) -> DataFrame:
    """
    Add a 'clean_text' column (preprocessed) while keeping the original
    'text' column intact.
    """
    logger.info("Applying Arabic preprocessing pipeline (CAMeL approach) …")
    return df.withColumn("clean_text", preprocess_udf(F.col(TEXT_COL)))


# ---------------------------------------------------------------------------
# 4. MapReduce-style corpus statistics
# ---------------------------------------------------------------------------

def mapreduce_word_count(df: DataFrame, text_col: str = "clean_text") -> DataFrame:
    """
    MapReduce word count implemented with Spark transformations.

    Map stage  : explode words → (word, 1)
    Reduce stage: groupBy word → sum counts

    Returns
    -------
    DataFrame   columns: word, count  (ordered by count descending)
    """
    logger.info("Running MapReduce word count on '%s' …", text_col)
    return (
        df.select(F.explode(F.split(F.col(text_col), r"\s+")).alias("word"))
        .filter(F.col("word") != "")
        .groupBy("word")
        .agg(F.count("*").alias("count"))
        .orderBy(F.col("count").desc())
    )


def mapreduce_ngram_frequency(
    df: DataFrame,
    n: int = 2,
    text_col: str = "clean_text",
) -> DataFrame:
    """
    MapReduce n-gram frequency using Spark MLlib NGram transformer.

    Returns
    -------
    DataFrame   columns: ngram, count  (ordered by count descending)
    """
    from pyspark.ml.feature import Tokenizer, NGram

    logger.info("Running MapReduce %d-gram frequency on '%s' …", n, text_col)
    tokenizer = Tokenizer(inputCol=text_col, outputCol="_tokens")
    ngram_tf  = NGram(n=n, inputCol="_tokens", outputCol="_ngrams")

    tokenized = tokenizer.transform(df)
    ngrammed  = ngram_tf.transform(tokenized)

    return (
        ngrammed
        .select(F.explode(F.col("_ngrams")).alias("ngram"))
        .groupBy("ngram")
        .agg(F.count("*").alias("count"))
        .orderBy(F.col("count").desc())
    )


# ---------------------------------------------------------------------------
# 5. Stratified train / validation / test split
# ---------------------------------------------------------------------------

def stratified_split(
    df: DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Per-class stratified random split.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    assert train_ratio + val_ratio + test_ratio <= 1.001

    train_parts, val_parts, test_parts = [], [], []
    for label_val in [0, 1]:
        subset = df.filter(F.col(LABEL_COL) == label_val)
        tr, va, te = subset.randomSplit([train_ratio, val_ratio, test_ratio], seed=seed)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

    train_df = train_parts[0].union(train_parts[1])
    val_df   = val_parts[0].union(val_parts[1])
    test_df  = test_parts[0].union(test_parts[1])

    logger.info(
        "Stratified split — train: %d | val: %d | test: %d",
        train_df.count(), val_df.count(), test_df.count(),
    )
    return train_df, val_df, test_df


