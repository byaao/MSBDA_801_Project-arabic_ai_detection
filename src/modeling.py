"""
modeling.py
===========
Spark MLlib model training, hyperparameter tuning, and evaluation.

Models implemented:
  1. Baseline  : Logistic Regression 
  2. Advanced A: Random Forest       
  3. Advanced B: Linear SVM          

Evaluation: Accuracy, F1 (macro), ROC-AUC, Confusion Matrix
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    LinearSVC,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from src.utils import logger, LABEL_COL, MODELS_DIR


# ---------------------------------------------------------------------------
# 1. Model factories
# ---------------------------------------------------------------------------

def get_logistic_regression(features_col: str = "features") -> LogisticRegression:
    """Return a configured Logistic Regression estimator."""
    return LogisticRegression(
        featuresCol=features_col,
        labelCol=LABEL_COL,
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
        family="binomial",
    )


def get_random_forest(features_col: str = "features") -> RandomForestClassifier:
    """Return a configured Random Forest estimator."""
    return RandomForestClassifier(
        featuresCol=features_col,
        labelCol=LABEL_COL,
        numTrees=100,
        maxDepth=8,
        seed=42,
        featureSubsetStrategy="sqrt",
    )


def get_linear_svm(features_col: str = "features") -> LinearSVC:
    """Return a configured Linear SVM estimator."""
    return LinearSVC(
        featuresCol=features_col,
        labelCol=LABEL_COL,
        maxIter=100,
        regParam=0.1,
        standardization=True,
    )


# ---------------------------------------------------------------------------
# 2. Training with k-fold cross-validation
# ---------------------------------------------------------------------------

def train_baseline(
    train_df: DataFrame,
    features_col: str = "features",
    num_folds: int = 3,
) -> Tuple[PipelineModel, float]:
    """
    Train Logistic Regression baseline with CV over regParam.

    Returns
    -------
    (best_model, train_time_seconds)
    """
    logger.info("Training baseline — Logistic Regression (CV=%d folds) …", num_folds)
    lr       = get_logistic_regression(features_col)
    pipeline = Pipeline(stages=[lr])
    grid     = ParamGridBuilder().addGrid(lr.regParam, [0.001, 0.01, 0.1]).build()
    evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, metricName="f1")

    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,
                        evaluator=evaluator, numFolds=num_folds, seed=42)
    t0 = time.time()
    cv_model = cv.fit(train_df)
    elapsed  = time.time() - t0

    logger.info("Logistic Regression trained in %.1f s | best CV F1=%.4f",
                elapsed, max(cv_model.avgMetrics))
    return cv_model.bestModel, elapsed


def train_random_forest(
    train_df: DataFrame,
    features_col: str = "features",
    num_folds: int = 3,
) -> Tuple[PipelineModel, float]:
    """
    Train Random Forest with CV over numTrees and maxDepth.

    Returns
    -------
    (best_model, train_time_seconds)
    """
    logger.info("Training Random Forest (CV=%d folds) …", num_folds)
    rf       = get_random_forest(features_col)
    pipeline = Pipeline(stages=[rf])
    grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100])
        .addGrid(rf.maxDepth, [6, 10])
        .build()
    )
    evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, metricName="f1")

    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,
                        evaluator=evaluator, numFolds=num_folds, seed=42)
    t0 = time.time()
    cv_model = cv.fit(train_df)
    elapsed  = time.time() - t0

    logger.info("Random Forest trained in %.1f s | best CV F1=%.4f",
                elapsed, max(cv_model.avgMetrics))
    return cv_model.bestModel, elapsed


def train_linear_svm(
    train_df: DataFrame,
    features_col: str = "features",
    num_folds: int = 3,
) -> Tuple[PipelineModel, float]:
    """
    Train Linear SVM with CV over regParam.

    Returns
    -------
    (best_model, train_time_seconds)
    """
    logger.info("Training Linear SVM (CV=%d folds) …", num_folds)
    svm      = get_linear_svm(features_col)
    pipeline = Pipeline(stages=[svm])
    grid     = ParamGridBuilder().addGrid(svm.regParam, [0.01, 0.1, 1.0]).build()
    evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, metricName="f1")

    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,
                        evaluator=evaluator, numFolds=num_folds, seed=42)
    t0 = time.time()
    cv_model = cv.fit(train_df)
    elapsed  = time.time() - t0

    logger.info("Linear SVM trained in %.1f s | best CV F1=%.4f",
                elapsed, max(cv_model.avgMetrics))
    return cv_model.bestModel, elapsed


# ---------------------------------------------------------------------------
# 3. Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_model(
    model: PipelineModel,
    test_df: DataFrame,
    model_name: str = "model",
) -> Dict:
    """
    Evaluate a trained model on a held-out test set.

    Returns
    -------
    dict  keys: model, accuracy, f1, precision, recall, roc_auc
    """
    logger.info("Evaluating '%s' …", model_name)
    predictions = model.transform(test_df)
    mc_eval     = MulticlassClassificationEvaluator(labelCol=LABEL_COL)

    accuracy  = mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"})
    f1        = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})
    precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
    recall    = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})

    roc_auc = 0.0
    try:
        bin_eval = BinaryClassificationEvaluator(
            labelCol=LABEL_COL, rawPredictionCol="rawPrediction"
        )
        roc_auc = bin_eval.evaluate(predictions)
    except Exception as e:
        logger.warning("ROC-AUC not computable for '%s': %s", model_name, e)

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy,  4),
        "f1":        round(f1,        4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "roc_auc":   round(roc_auc,   4),
    }
    logger.info("%s | Acc=%.4f | F1=%.4f | AUC=%.4f",
                model_name, accuracy, f1, roc_auc)
    return metrics


def confusion_matrix_spark(model: PipelineModel, test_df: DataFrame) -> DataFrame:
    """
    Compute a confusion matrix as a Spark DataFrame.

    Returns
    -------
    DataFrame  columns: actual, predicted, count
    """
    preds = model.transform(test_df)
    return (
        preds
        .groupBy(F.col(LABEL_COL).alias("actual"), F.col("prediction").alias("predicted"))
        .count()
        .orderBy("actual", "predicted")
    )


def extract_feature_importances(
    rf_model,
    feature_col_names: List[str],
    top_n: int = 20,
) -> List[Tuple[str, float]]:
    """
    Extract the top N feature importances from a fitted RandomForest model.

    Parameters
    ----------
    rf_model          : Fitted RandomForestClassificationModel stage.
    feature_col_names : Feature column names in the same order as VectorAssembler.
    top_n             : Number of top features to return.

    Returns
    -------
    List of (feature_name, importance) tuples sorted descending.
    """
    importances = rf_model.featureImportances.toArray()
    paired      = sorted(zip(feature_col_names, importances), key=lambda x: x[1], reverse=True)
    return paired[:top_n]


# ---------------------------------------------------------------------------
# 4. Model persistence
# ---------------------------------------------------------------------------

def save_model(model: PipelineModel, name: str) -> str:
    """Save a fitted PipelineModel to the models directory."""
    path = str(MODELS_DIR / name)
    model.write().overwrite().save(path)
    logger.info("Model saved → %s", path)
    return path


def load_model(spark: SparkSession, name: str) -> PipelineModel:
    """Load a saved PipelineModel from the models directory."""
    path = str(MODELS_DIR / name)
    logger.info("Loading model ← %s", path)
    return PipelineModel.load(path)


# ---------------------------------------------------------------------------
# 5. Scalability benchmark  (Task 4.4)
# ---------------------------------------------------------------------------

def benchmark_inference(
    model: PipelineModel,
    test_df: DataFrame,
    parallelism_levels: List[int],
    spark: SparkSession,
) -> List[Dict]:
    """
    Benchmark batch inference throughput at different Spark parallelism levels.

    Repartitions the test DataFrame to simulate different executor counts and
    forces a full action (count) to measure wall-clock time.

    Parameters
    ----------
    parallelism_levels : Partition counts to test, e.g. [1, 2, 4, 8].

    Returns
    -------
    List of dicts: {partitions, rows, elapsed_s, rows_per_sec}
    """
    results = []
    n_rows  = test_df.count()

    for n_parts in parallelism_levels:
        logger.info("Benchmarking %d partition(s) …", n_parts)
        df_repartitioned = test_df.repartition(n_parts)
        t0 = time.time()
        model.transform(df_repartitioned).count()   # force full execution
        elapsed = time.time() - t0
        tput    = n_rows / max(elapsed, 1e-6)
        results.append({
            "partitions":   n_parts,
            "rows":         n_rows,
            "elapsed_s":    round(elapsed, 2),
            "rows_per_sec": round(tput, 1),
        })
        logger.info("  %d parts → %.2f s (%.0f rows/s)", n_parts, elapsed, tput)

    return results
