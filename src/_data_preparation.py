"""
data_preparation.py
===================
Functions for:
  1. Downloading the Hugging Face dataset
  2. Creating a binary-labelled Spark DataFrame  (human=0, AI=1)
  3. Arabic-specific text preprocessing (Spark UDFs)
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

    Returns
    -------
    dict  {split_name: pandas.DataFrame}
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
# 3. Arabic text preprocessing  (pure Python + Spark UDFs)
# ---------------------------------------------------------------------------

_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
_TATWEEL     = re.compile(r"\u0640+")
_EXTRA_SPACES = re.compile(r"\s+")
_NON_ARABIC  = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]")

# Core Arabic stop-words (normalised form)
_RAW_STOPWORDS = {
    "من", "في", "على", "علي", "إلى", "الى", "الي", "عن", "مع",
    "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "الذين",
    "كان", "كانت", "يكون", "تكون", "كما", "أن", "ان", "إن",
    "لا", "ما", "هو", "هي", "هم", "هن", "أنا", "انا", "أنت", "انت",
    "أو", "او", "إذا", "اذا", "حيث", "عند", "بعد", "قبل", "بين",
    "حتى", "لكن", "إلا", "الا", "ثم", "لقد", "قد", "لم", "لن",
    "هل", "أم", "ام", "منذ", "خلال", "وقد", "وكان", "وأن", "وان",
    "وفي", "وعلى", "وعلي", "وإلى", "والي", "ولا", "فإن", "فان", "فقد",
    "و", "،", ".", ":", "؛", "؟", "!", "-",
    "ومن", "كل", "بعض", "مما", "ايضا", "اي", "غير",
    # Domain-specific additions
    "لدى", "لدي", "وهو", "يمكن", "تم", "يتم", "فيه", "فيها",
    "بها", "لها", "انه", "أنّه", "إنه", "والتي", "مثل", "مدى", "مدي",
}


def _normalize_arabic(text: str) -> str:
    """Remove diacritics, tatweel, and unify alef/teh-marbuta/alef-maqsura forms."""
    if not text:
        return ""
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    return _EXTRA_SPACES.sub(" ", text).strip()


# Normalise stop-words so matching works after normalisation
ARABIC_STOPWORDS = {_normalize_arabic(w) for w in _RAW_STOPWORDS}


def _remove_noise(text: str) -> str:
    """Keep only Arabic Unicode characters and spaces."""
    return _EXTRA_SPACES.sub(" ", _NON_ARABIC.sub(" ", text)).strip()


def _remove_stopwords(text: str) -> str:
    """Remove normalised Arabic stop-words (min length > 1)."""
    return " ".join(
        w for w in text.split()
        if w not in ARABIC_STOPWORDS and len(w) > 1
    )


def _light_stem(word: str) -> str:
    """Light Arabic stemmer — strips common prefixes and suffixes."""
    for p in ["وال", "بال", "كال", "فال", "لل", "ال"]:
        if word.startswith(p) and len(word) > len(p) + 2:
            word = word[len(p):]
            break
    for s in ["ات", "ون", "ين", "ان", "ها", "هم", "هن", "كم", "كن", "نا", "تم"]:
        if word.endswith(s) and len(word) > len(s) + 2:
            word = word[: -len(s)]
            break
    return word


def preprocess_arabic(text: str) -> str:
    """
    Full Arabic preprocessing pipeline:
        normalize → strip noise → remove stop-words → light stemming.

    Designed to run inside a Spark UDF (no global Spark state used).
    """
    if not text or not isinstance(text, str):
        return ""
    text = _normalize_arabic(text)
    text = _remove_noise(text)
    text = _remove_stopwords(text)
    tokens = [_light_stem(w) for w in text.split() if len(w) > 1]
    return " ".join(tokens)


# Register as Spark UDF (used in notebooks and streaming pipeline)
preprocess_udf = F.udf(preprocess_arabic, StringType())


def apply_preprocessing(df: DataFrame) -> DataFrame:
    """
    Add a 'clean_text' column (preprocessed) while keeping the original
    'text' column intact.
    """
    logger.info("Applying Arabic preprocessing pipeline …")
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

    Map stage  : tokenize → generate n-grams → explode
    Reduce stage: groupBy n-gram → sum counts

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

    Spark's randomSplit is not truly stratified; we split each class
    separately and union the results to approximate stratification.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    assert train_ratio + val_ratio + test_ratio <= 1.001, "Ratios must sum to 1"

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
