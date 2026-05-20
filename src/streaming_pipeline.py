"""
streaming_pipeline.py
=====================
Real-time Arabic AI text detection using Spark Structured Streaming
with a file-based stream source (Colab-compatible Kafka substitute).
"""

from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import PipelineModel

from src.utils import logger, DATA_PROCESSED, PROJECT_ROOT, TEXT_COL, LABEL_COL
from src.data_preparation import preprocess_udf
from src.feature_engineering import extract_all_features, assemble_feature_vector

# ---------------------------------------------------------------------------
# Stream directories
# ---------------------------------------------------------------------------

STREAM_INPUT_DIR  = PROJECT_ROOT / "stream" / "input"
STREAM_OUTPUT_DIR = PROJECT_ROOT / "stream" / "output"
STREAM_CHECKPOINT = PROJECT_ROOT / "stream" / "checkpoint"

for _d in [STREAM_INPUT_DIR, STREAM_OUTPUT_DIR, STREAM_CHECKPOINT]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Stream schema
# ---------------------------------------------------------------------------

STREAM_SCHEMA = StructType([
    StructField("text",              StringType(),  True),
    StructField("label",             IntegerType(), True),
    StructField("source_model",      StringType(),  True),
    StructField("generation_method", StringType(),  True),
])


# ---------------------------------------------------------------------------
# StreamWriter — simulates a Kafka producer
# ---------------------------------------------------------------------------

class StreamWriter:
    """
    Writes JSON records to STREAM_INPUT_DIR at a configurable rate to
    simulate a live stream of Arabic abstracts.

    Usage
    -----
        writer = StreamWriter(test_rows, rate=2.0)
        writer.start()
        # ... streaming query runs ...
        writer.stop()
    """

    def __init__(self, rows: list, rate: float = 1.0, loop: bool = False):
        """
        Parameters
        ----------
        rows  : List of dicts with at least a 'text' key.
        rate  : Records per second to emit.
        loop  : If True, restart from beginning when rows are exhausted.
        """
        self.rows        = rows
        self.rate        = rate
        self.loop        = loop
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._counter    = 0

    def _write_loop(self):
        rows = self.rows[:]
        idx  = 0
        while not self._stop_event.is_set():
            if idx >= len(rows):
                if self.loop:
                    idx = 0
                else:
                    break
            row  = rows[idx]
            path = STREAM_INPUT_DIR / f"record_{self._counter:06d}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False)
            self._counter += 1
            idx           += 1
            time.sleep(1.0 / self.rate)

        logger.info("StreamWriter finished — %d records written.", self._counter)

    def start(self):
        logger.info("StreamWriter starting (rate=%.1f rec/s) …", self.rate)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("StreamWriter stopped.")


# ---------------------------------------------------------------------------
# Streaming pipeline builder
# ---------------------------------------------------------------------------

def build_streaming_pipeline(
    spark: SparkSession,
    model: PipelineModel,
    tfidf_model: PipelineModel,
    trigger_seconds: float = 5.0,
) -> "StreamingQuery":
    """
    Build and start a Spark Structured Streaming query.

    Steps per micro-batch (via foreachBatch):
      1. Arabic preprocessing UDF  →  clean_text
      2. TF-IDF PipelineModel      →  tfidf_features
      3. Stylometric features      →  feat_* columns
      4. VectorAssembler           →  features
      5. Best classifier           →  prediction
      6. Write Parquet results     →  stream/output/batch_NNNNNN/

    Parameters
    ----------
    spark           : Active SparkSession.
    model           : Best trained classifier PipelineModel (from Phase 3).
    tfidf_model     : Fitted TF-IDF PipelineModel (saved in Phase 4).
    trigger_seconds : Micro-batch interval in seconds.

    Returns
    -------
    StreamingQuery  — call .awaitTermination(timeout) or .stop()
    """
    logger.info("Building Structured Streaming pipeline …")
    logger.info("  Input dir : %s", STREAM_INPUT_DIR)
    logger.info("  Output dir: %s", STREAM_OUTPUT_DIR)

    # ── Source ────────────────────────────────────────────────────────────
    raw_stream = (
        spark.readStream
        .schema(STREAM_SCHEMA)
        .option("maxFilesPerTrigger", 10)
        .option("cleanSource", "delete")        # delete input files after reading
        .json(str(STREAM_INPUT_DIR))
    )

    # ── Preprocessing ─────────────────────────────────────────────────────
    preprocessed = (
        raw_stream
        .withColumn("clean_text", preprocess_udf(F.col(TEXT_COL)))
        .withColumn("ingest_timestamp", F.current_timestamp())
        .filter(F.col("clean_text").isNotNull() & (F.length("clean_text") > 5))
    )

    # ── Batch processing function (MLlib requires foreachBatch for transforms)
    def foreach_batch_fn(batch_df: DataFrame, batch_id: int):
        if batch_df.isEmpty():
            return

        n = batch_df.count()
        logger.info("Batch %d — %d records received", batch_id, n)

        # Apply the full feature + inference pipeline
        tfidf_ready = tfidf_model.transform(batch_df)
        featured    = extract_all_features(tfidf_ready, input_col="clean_text")
        assembled   = assemble_feature_vector(featured)
        predictions = model.transform(assembled)

        output = predictions.select(
            F.col(TEXT_COL),
            F.col("clean_text"),
            F.col("prediction").cast(IntegerType()).alias("predicted_label"),
            F.col(LABEL_COL).alias("true_label"),
            F.col("source_model"),
            F.col("generation_method"),
            F.col("ingest_timestamp"),
            F.current_timestamp().alias("output_timestamp"),
            F.lit(batch_id).alias("batch_id"),
        )

        out_path = str(STREAM_OUTPUT_DIR / f"batch_{batch_id:06d}")
        output.write.mode("overwrite").parquet(out_path)
        logger.info("→ Batch %d: %d predictions written to %s", batch_id, n, out_path)

    # ── Streaming query ────────────────────────────────────────────────────
    query = (
        preprocessed.writeStream
        .foreachBatch(foreach_batch_fn)
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .option("checkpointLocation", str(STREAM_CHECKPOINT))
        .start()
    )

    logger.info("Streaming query started — ID: %s", query.id)
    return query


# ---------------------------------------------------------------------------
# Latency & throughput benchmark  (Task 4.4)
# ---------------------------------------------------------------------------

def measure_stream_latency(
    output_dir: Path = STREAM_OUTPUT_DIR,
    spark: Optional[SparkSession] = None,
) -> dict:
    """
    Read all output Parquet batches and compute stream performance metrics.

    Computes end-to-end latency from ingest_timestamp → output_timestamp,
    and throughput in records per second.

    Returns
    -------
    dict  keys: total_records, total_batches, avg_batch_size,
                total_time_s, throughput_rps
    """
    if spark is None:
        return {}

    batch_paths = [str(p) for p in sorted(Path(output_dir).glob("batch_*"))]
    if not batch_paths:
        return {
            "total_records": 0, "total_batches": 0, "avg_batch_size": 0,
            "total_time_s": 0.0, "throughput_rps": 0.0,
        }

    all_outputs = spark.read.parquet(*batch_paths)

    total_records = all_outputs.count()
    total_batches = all_outputs.select("batch_id").distinct().count()
    avg_batch     = round(total_records / max(total_batches, 1), 1)

    # Compute total wall-clock time from first ingest to last output
    time_stats = all_outputs.agg(
        F.min("ingest_timestamp").alias("t_start"),
        F.max("output_timestamp").alias("t_end"),
    ).collect()[0]

    total_time_s = 0.0
    throughput   = 0.0
    if time_stats["t_start"] and time_stats["t_end"]:
        delta        = (time_stats["t_end"] - time_stats["t_start"]).total_seconds()
        total_time_s = round(max(delta, 1e-6), 2)
        throughput   = round(total_records / total_time_s, 1)

    return {
        "total_records":  total_records,
        "total_batches":  total_batches,
        "avg_batch_size": avg_batch,
        "total_time_s":   total_time_s,
        "throughput_rps": throughput,
    }


# ---------------------------------------------------------------------------
# Demo helper
# ---------------------------------------------------------------------------

def run_stream_demo(
    spark: SparkSession,
    model: PipelineModel,
    tfidf_model: PipelineModel,
    sample_texts: list,
    duration_seconds: int = 60,
    rate: float = 2.0,
) -> dict:
    """
    Run a complete end-to-end stream demo for `duration_seconds`.

    Parameters
    ----------
    sample_texts     : List of dicts from the test set.
    duration_seconds : How long to let the stream run.
    rate             : Emission rate (records per second).

    Returns
    -------
    dict  Stream latency metrics from measure_stream_latency().
    """
    writer = StreamWriter(sample_texts, rate=rate, loop=True)
    query  = build_streaming_pipeline(
        spark, model, tfidf_model, trigger_seconds=5.0
    )

    writer.start()
    logger.info("Stream demo running for %d seconds …", duration_seconds)

    try:
        query.awaitTermination(timeout=duration_seconds)
    except Exception as e:
        logger.warning("Stream query terminated: %s", e)
    finally:
        writer.stop()
        if query.isActive:
            query.stop()
        logger.info("Stream demo complete.")

    return measure_stream_latency(output_dir=STREAM_OUTPUT_DIR, spark=spark)
