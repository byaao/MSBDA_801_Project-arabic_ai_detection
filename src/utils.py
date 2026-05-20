"""
utils.py
========
Shared utility functions for the Arabic AI Text Detection pipeline.
All helpers used across notebooks and src modules live here.
"""

import os
import re
import time
import logging
import shutil
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arabic_ai_detection")


# ---------------------------------------------------------------------------
# Project paths  (Colab GDrive mount or local fallback)
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root — GDrive mount on Colab, CWD elsewhere."""
    gdrive = Path("/content/drive/MyDrive/MSBDA-801-Project/arabic_ai_detection")
    if gdrive.parent.parent.exists():          # /content/drive exists → Colab
        gdrive.mkdir(parents=True, exist_ok=True)
        return gdrive
    return Path(os.environ.get("PROJECT_ROOT", Path.cwd()))


PROJECT_ROOT   = get_project_root()
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
REPORTS_DIR    = PROJECT_ROOT / "reports"
FIGURES_DIR    = REPORTS_DIR / "figures"

# CAMeL Tools data cache — set the env var here so every module picks it up
CAMEL_CACHE = PROJECT_ROOT / ".camel_cache"

for _d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, CAMEL_CACHE]:
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("CAMELTOOLS_DATA", str(CAMEL_CACHE))
logger.info("CAMELTOOLS_DATA → %s", os.environ["CAMELTOOLS_DATA"])


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------

def create_spark_session(
    app_name: str = "ArabicAIDetection",
    executor_memory: str = "4g",
    driver_memory: str = "4g",
    shuffle_partitions: int = 8,
) -> SparkSession:
    """
    Create (or retrieve) a SparkSession optimised for Google Colab.

    Parameters
    ----------
    app_name           : Spark application name (shown in the web UI).
    executor_memory    : Executor heap size (Colab has ~12 GB total RAM).
    driver_memory      : Driver heap size.
    shuffle_partitions : Low value reduces shuffle overhead on a single-node
                         cluster (default Spark value of 200 is too high).

    Returns
    -------
    SparkSession
    """
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created — version %s", spark.version)
    return spark


# ---------------------------------------------------------------------------
# Spark worker packaging helper
# ---------------------------------------------------------------------------

def add_src_to_spark(spark: SparkSession) -> str:
    """
    Zip the src/ package and register it with Spark workers via addPyFile.

    Spark Python workers run in separate processes and cannot import project
    modules unless this is called.  Call once per session after creating the
    SparkSession.

    Returns
    -------
    str  Path to the generated zip file.
    """
    zip_base = str(PROJECT_ROOT / "src_package")
    zip_path = shutil.make_archive(
        zip_base, "zip", root_dir=PROJECT_ROOT, base_dir="src"
    )
    spark.sparkContext.addPyFile(zip_path)
    logger.info("src package added to Spark workers: %s", zip_path)
    return zip_path


# ---------------------------------------------------------------------------
# Parquet checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(df: DataFrame, name: str, overwrite: bool = True) -> str:
    """
    Persist a Spark DataFrame as a Parquet checkpoint to the processed dir.

    Parameters
    ----------
    df        : DataFrame to persist.
    name      : Sub-directory name inside data/processed/.
    overwrite : Overwrite existing checkpoint (default True).

    Returns
    -------
    str  Absolute path where data was written.
    """
    path = str(DATA_PROCESSED / name)
    mode = "overwrite" if overwrite else "error"
    t0 = time.time()
    df.write.mode(mode).parquet(path)
    elapsed = time.time() - t0
    logger.info("Checkpoint '%s' saved in %.1f s → %s", name, elapsed, path)
    return path


def load_checkpoint(spark: SparkSession, name: str) -> DataFrame:
    """Load a Parquet checkpoint from the processed directory."""
    path = str(DATA_PROCESSED / name)
    logger.info("Loading checkpoint '%s' ← %s", name, path)
    return spark.read.parquet(path)


def checkpoint_exists(name: str) -> bool:
    """
    Return True only if the checkpoint directory exists and contains
    at least one valid Parquet part file.
    """
    path = DATA_PROCESSED / name
    return path.exists() and any(path.rglob("*.parquet"))


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer for benchmarking pipeline steps."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self._start
        msg = f"{self.label}: {self.elapsed:.2f}s" if self.label else f"{self.elapsed:.2f}s"
        logger.info("⏱  %s", msg)

    def __repr__(self):
        return f"Timer(elapsed={self.elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# Dataset schema constants
# ---------------------------------------------------------------------------

AI_MODEL_COLS = [
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract",
]

LABEL_COL  = "label"            # 0 = human, 1 = AI
TEXT_COL   = "text"             # unified text column after melting
SOURCE_COL = "source_model"     # "human" | "allam" | "jais" | "llama" | "openai"
SPLIT_COL  = "generation_method"
FEAT_PREFIX = "feat_"           # prefix applied to all stylometric feature columns


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_class_distribution(df: DataFrame, label_col: str = LABEL_COL) -> None:
    """Print class counts and their percentage share."""
    total = df.count()
    df.groupBy(label_col).count().withColumn(
        "pct", F.round(F.col("count") / total * 100, 2)
    ).orderBy(label_col).show()


def sample_texts(df: DataFrame, n: int = 5, label: Optional[int] = None) -> None:
    """Print n sample texts, optionally filtered by label."""
    sub = df.filter(F.col(LABEL_COL) == label) if label is not None else df
    rows = sub.select(TEXT_COL, LABEL_COL, SOURCE_COL).limit(n).collect()
    for i, row in enumerate(rows, 1):
        print(f"─── Sample {i} [label={row[LABEL_COL]}, src={row[SOURCE_COL]}] ───")
        print(row[TEXT_COL][:300])
        print()


def count_parquet_rows(spark: SparkSession, name: str) -> int:
    """Quick row count on a checkpoint without loading the full schema."""
    path = str(DATA_PROCESSED / name)
    return spark.read.parquet(path).count()
