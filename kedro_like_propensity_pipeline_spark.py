"""Kedro-Style ML Flow (Spark / PySpark)

Spark-native reimplementation of the pure-Python Kedro-inspired pipeline.

This script preserves the same *shape* as `kedro_like_propensity_pipeline.py`:
- Node: function with declared inputs/outputs
- Pipeline: ordered list of nodes
- Catalog: named datasets / objects shared between nodes
- Runner: executes nodes, materializes outputs back into the catalog

Differences vs the sklearn/CatBoost version:
- Uses PySpark DataFrames and Spark MLlib models
- Uses LogisticRegression for baseline and GBTClassifier for the final model
- “Explainability” columns are a lightweight heuristic (not SHAP)

Run:
  python kedro_like_propensity_pipeline_spark.py --list-pipelines
  python kedro_like_propensity_pipeline_spark.py --pipeline full

Notes:
- Requires Java + PySpark.
- The default master is local[*] for laptop execution.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

Catalog = Dict[str, Any]


def _as_list(x: Union[None, str, Sequence[str]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


@dataclass(frozen=True)
class Node:
    func: Callable[..., Any]
    inputs: Union[None, str, Sequence[str]] = None
    outputs: Union[None, str, Sequence[str]] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", self.func.__name__)


@dataclass(frozen=True)
class Pipeline:
    nodes: List[Node]
    name: str = "pipeline"


class DataSet:
    def load(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    def save(self, data: Any) -> None:  # pragma: no cover
        raise NotImplementedError


class MemoryDataSet(DataSet):
    def __init__(self, data: Any = None):
        self._data = data

    def load(self) -> Any:
        return self._data

    def save(self, data: Any) -> None:
        self._data = data


class SparkParquetDataSet(DataSet):
    def __init__(self, spark, path: str, mode: str = "overwrite"):
        self.spark = spark
        self.path = str(path)
        self.mode = str(mode)

    def load(self):
        return self.spark.read.parquet(self.path)

    def save(self, df) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        df.write.mode(self.mode).parquet(self.path)


class SparkCSVDataSet(DataSet):
    def __init__(self, spark, path: str, header: bool = True, mode: str = "overwrite"):
        self.spark = spark
        self.path = str(path)
        self.header = bool(header)
        self.mode = str(mode)

    def load(self):
        return self.spark.read.option("header", self.header).csv(self.path, inferSchema=True)

    def save(self, df) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # Spark writes a folder for CSV; keep that behavior explicit.
        df.coalesce(1).write.mode(self.mode).option("header", self.header).csv(self.path)


class SparkMLlibModelDataSet(DataSet):
    def __init__(self, path: str, mode: str = "overwrite"):
        self.path = str(path)
        self.mode = str(mode)

    def load(self):
        from pyspark.ml.pipeline import PipelineModel

        return PipelineModel.load(self.path)

    def save(self, model) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "overwrite":
            model.write().overwrite().save(self.path)
        else:
            model.write().save(self.path)


class DataCatalog:
    """Very small Kedro-like catalog.

    - Named entries can be pure in-memory objects (implicit) or backed by DataSet instances.
    - Runner will `save()` node outputs into registered datasets when available.
    """

    def __init__(self, initial: Optional[Dict[str, Any]] = None, datasets: Optional[Dict[str, DataSet]] = None):
        self._data: Dict[str, Any] = dict(initial or {})
        self._datasets: Dict[str, DataSet] = dict(datasets or {})

    def keys(self):
        return self._data.keys()

    def register(self, name: str, dataset: DataSet) -> None:
        self._datasets[str(name)] = dataset

    def __getitem__(self, name: str) -> Any:
        key = str(name)
        if key in self._data:
            return self._data[key]
        if key in self._datasets:
            obj = self._datasets[key].load()
            self._data[key] = obj
            return obj
        raise KeyError(key)

    def __setitem__(self, name: str, value: Any) -> None:
        key = str(name)
        self._data[key] = value
        if key in self._datasets:
            self._datasets[key].save(value)

    def as_dict(self) -> Catalog:
        return dict(self._data)


def run_pipeline(
    pipeline: Pipeline,
    catalog: DataCatalog,
    verbose: bool = True,
    on_node_start: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_node_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_node_error: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> DataCatalog:
    if verbose:
        print("Running pipeline:", pipeline.name)

    node_timings: List[Dict[str, Any]] = []
    pipeline_start = time.time()
    total_nodes = int(len(pipeline.nodes))

    for node_idx, node in enumerate(pipeline.nodes):
        in_names = _as_list(node.inputs)
        out_names = _as_list(node.outputs)
        args = [catalog[n] for n in in_names]

        if on_node_start is not None:
            on_node_start(
                {
                    "pipeline": pipeline.name,
                    "node": str(node.name),
                    "node_index": int(node_idx),
                    "node_count": int(total_nodes),
                    "inputs": list(in_names),
                    "outputs": list(out_names),
                    "ts": float(time.time()),
                }
            )

        if verbose:
            print("")
            print("NODE:", node.name)
            if in_names:
                print("  inputs :", in_names)
            if out_names:
                print("  outputs:", out_names)

        start = time.time()
        try:
            result = node.func(*args)
        except Exception as e:
            elapsed = time.time() - start
            if on_node_error is not None:
                on_node_error(
                    {
                        "pipeline": pipeline.name,
                        "node": str(node.name),
                        "node_index": int(node_idx),
                        "node_count": int(total_nodes),
                        "inputs": list(in_names),
                        "outputs": list(out_names),
                        "seconds": float(elapsed),
                        "error": repr(e),
                        "ts": float(time.time()),
                    }
                )
            raise
        elapsed = time.time() - start

        node_timings.append(
            {
                "pipeline": pipeline.name,
                "node": node.name,
                "seconds": float(elapsed),
                "inputs": list(in_names),
                "outputs": list(out_names),
            }
        )

        if not out_names:
            if verbose:
                print("  done in %.2fs (no outputs)" % elapsed)
            continue

        if len(out_names) == 1:
            catalog[out_names[0]] = result
        else:
            if not isinstance(result, (tuple, list)) or len(result) != len(out_names):
                raise ValueError("Node %s must return %d outputs" % (node.name, len(out_names)))
            for k, v in zip(out_names, result):
                catalog[k] = v

        if verbose:
            print("  done in %.2fs" % elapsed)

        if on_node_end is not None:
            on_node_end(
                {
                    "pipeline": pipeline.name,
                    "node": str(node.name),
                    "node_index": int(node_idx),
                    "node_count": int(total_nodes),
                    "inputs": list(in_names),
                    "outputs": list(out_names),
                    "seconds": float(elapsed),
                    "ts": float(time.time()),
                }
            )

    total = time.time() - pipeline_start
    # Keep timings as a small in-memory list to avoid requiring pandas.
    catalog["run:timings"] = sorted(node_timings, key=lambda r: float(r["seconds"]), reverse=True)
    catalog["run:total_seconds"] = float(total)

    if verbose:
        print("\n=== Timing snapshot (slowest first) ===")
        for r in catalog["run:timings"][:10]:
            print("- %s: %.4fs" % (r["node"], float(r["seconds"])))
        print("Total pipeline time (s):", round(float(catalog["run:total_seconds"]), 4))

    return catalog


def _require_pyspark():
    try:
        import pyspark  # noqa: F401

        return
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "PySpark is required for the Spark pipeline. "
            "Install it (and ensure Java is installed) then re-run.\n" + repr(e)
        )


def build_spark(master: str, shuffle_partitions: int, app_name: str = "kedro_like_propensity_pipeline"):
    _require_pyspark()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .master(str(master))
        .config("spark.sql.shuffle.partitions", str(int(shuffle_partitions)))
        .getOrCreate()
    )
    return spark


# -----------------------
# Spark pipeline nodes
# -----------------------

def create_customers_spark(spark, n_customers: int, seed: int):
    """Generate the synthetic customer dataset as a Spark DataFrame."""

    import pandas as pd

    # For a demo-sized dataset, generating in pandas then parallelizing is simplest.
    # This keeps parity with the original script’s distributions.
    rng = np.random.default_rng(int(seed))
    n = int(n_customers)
    pdf = pd.DataFrame(
        {
            "customer_id": np.arange(100000, 100000 + n),
            "age": rng.integers(21, 75, size=n),
            "annual_income": rng.normal(65000, 22000, size=n).clip(15000, 250000).round(0),
            "credit_score": rng.normal(690, 55, size=n).clip(450, 850).round(0),
            "utilization_rate": rng.beta(2.2, 4.5, size=n).clip(0.01, 0.99),
            "tenure_months": rng.integers(1, 240, size=n),
            "txn_count_3m": rng.poisson(18, size=n).clip(0, 120),
            "avg_txn_amount_3m": rng.normal(55, 30, size=n).clip(1, 600).round(2),
            "late_payments_12m": rng.poisson(0.35, size=n).clip(0, 10),
            "savings_balance": rng.lognormal(mean=8.6, sigma=0.9, size=n).clip(0, 250000).round(0),
            "region": rng.choice(["north", "south", "east", "west"], size=n),
            "is_premium": rng.choice([0, 1], size=n, p=[0.78, 0.22]),
            "has_mortgage": rng.choice([0, 1], size=n, p=[0.55, 0.45]),
            "channel": rng.choice(["branch", "app", "web", "call_center"], size=n, p=[0.18, 0.44, 0.30, 0.08]),
        }
    )
    pdf["email"] = "cust" + pdf["customer_id"].astype(str) + "@example.com"
    area = pd.Series(rng.integers(100, 999, size=n), index=pdf.index).astype(str).str.zfill(3)
    suffix = pd.Series(rng.integers(1000, 9999, size=n), index=pdf.index).astype(str).str.zfill(4)
    pdf["phone"] = "+1-555-" + area + "-" + suffix

    return spark.createDataFrame(pdf)


def add_target_spark(df, seed: int):
    """Add binary target applied_cc based on a noisy scoring function."""

    from pyspark.sql import functions as F

    # Spark-side randomness per row: create a stable pseudo-random uniform via hash.
    # This avoids Python UDFs and keeps it deterministic per seed.
    u = (
        F.abs(F.hash(F.col("customer_id"), F.lit(int(seed)))) % F.lit(10_000_000)
    ) / F.lit(10_000_000.0)

    score = (
        F.lit(0.025) * (F.col("credit_score") - F.lit(650))
        + F.lit(0.000012) * (F.col("annual_income") - F.lit(60000))
        + F.lit(0.030) * (F.col("txn_count_3m") - F.lit(15))
        + F.lit(0.002) * (F.col("tenure_months") - F.lit(24))
        + F.lit(0.55) * F.col("is_premium")
        + F.lit(0.10) * F.col("has_mortgage")
        - F.lit(2.0) * F.col("utilization_rate")
        - F.lit(0.45) * F.col("late_payments_12m")
    )

    # Add a mild noise term (Spark randn) seeded.
    noise = F.randn(int(seed)) * F.lit(1.2)
    p = F.lit(1.0) / (F.lit(1.0) + F.exp(-(score + noise)))

    out = df.withColumn("applied_cc", (u < p).cast("int"))
    return out


def split_xy_spark(df):
    """Split features/label and return the feature columns + cat cols."""

    target_col = "applied_cc"
    id_cols = ["customer_id", "email", "phone"]
    cat_cols = ["region", "channel"]

    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
    return df, feature_cols, target_col, cat_cols


def split_train_valid_spark(df, train_frac: float, seed: int):
    """Return train/valid splits."""

    train, valid = df.randomSplit([float(train_frac), 1.0 - float(train_frac)], seed=int(seed))
    return train, valid


def split_train_test_spark(df, test_frac: float, seed: int):
    train, test = df.randomSplit([1.0 - float(test_frac), float(test_frac)], seed=int(seed))
    return train, test


def _build_lr_pipeline(feature_cols: List[str], cat_cols: List[str], label_col: str):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import (
        StringIndexer,
        VectorAssembler,
        StandardScaler,
    )

    num_cols = [c for c in feature_cols if c not in cat_cols]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}__idx", handleInvalid="keep")
        for c in cat_cols
        if c in feature_cols
    ]
    assembled_inputs = num_cols + [f"{c}__idx" for c in cat_cols if c in feature_cols]

    assembler = VectorAssembler(inputCols=assembled_inputs, outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    lr = LogisticRegression(featuresCol="features", labelCol=label_col, probabilityCol="probability", maxIter=200)

    return Pipeline(stages=[*indexers, assembler, scaler, lr])


def baseline_scaled_model_spark(df_train, df_valid, feature_cols: List[str], label_col: str, cat_cols: List[str]):
    """Train baseline LR and return fitted pipeline + validation AUC."""

    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    pipe = _build_lr_pipeline(feature_cols, cat_cols, label_col)
    model = pipe.fit(df_train)

    pred = model.transform(df_valid)
    evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = float(evaluator.evaluate(pred))

    return model, auc


def predict_proba_spark(model, df, label_col: str):
    """Return a DataFrame with probability for class 1 as `p_apply`."""

    from pyspark.sql import functions as F

    pred = model.transform(df)
    # probability is a vector [p0, p1]
    out = pred.withColumn("p_apply", F.col("probability")[1].cast("double"))
    return out


def metrics_table_from_scores_spark(scored_df, label_col: str, thresholds: Sequence[float], label_prefix: str):
    """Compute a compact metrics table (Spark DF) for a list of thresholds."""

    from pyspark.sql import functions as F

    rows = []
    base = scored_df.select(F.col(label_col).cast("int").alias("y"), F.col("p_apply").cast("double").alias("p"))
    n = base.count()
    pos_rate = float(base.select(F.avg("y")).first()[0]) if n else 0.0

    # AUC / PR AUC via Spark evaluators (threshold-free).
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    # Recreate rawPrediction by mapping p to logit-ish score; evaluator expects rawPrediction vector.
    # Simplest: use the model outputs earlier where rawPrediction exists; if missing, skip.
    auc_roc = None
    auc_pr = None
    if "rawPrediction" in scored_df.columns:
        e_roc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        e_pr = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
        auc_roc = float(e_roc.evaluate(scored_df))
        auc_pr = float(e_pr.evaluate(scored_df))

    for thr in list(thresholds):
        t = float(thr)
        yhat = (F.col("p") >= F.lit(t)).cast("int")
        m = base.withColumn("yhat", yhat)

        tp = int(m.filter((F.col("y") == 1) & (F.col("yhat") == 1)).count())
        fp = int(m.filter((F.col("y") == 0) & (F.col("yhat") == 1)).count())
        tn = int(m.filter((F.col("y") == 0) & (F.col("yhat") == 0)).count())
        fn = int(m.filter((F.col("y") == 1) & (F.col("yhat") == 0)).count())

        accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = (2 * precision * recall) / max(1e-12, (precision + recall))

        rows.append(
            {
                "label": f"{label_prefix}@{t:g}",
                "n": int(n),
                "pos_rate": float(pos_rate),
                "threshold": float(t),
                "auc_roc": float(auc_roc) if auc_roc is not None else None,
                "auc_pr": float(auc_pr) if auc_pr is not None else None,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    return scored_df.sql_ctx.createDataFrame(rows)


def best_f1_metrics_from_scores_spark(scored_df, label_col: str, grid_points: int, label: str):
    from pyspark.sql import functions as F

    base = scored_df.select(F.col(label_col).cast("int").alias("y"), F.col("p_apply").cast("double").alias("p"))
    n = base.count()
    pos_rate = float(base.select(F.avg("y")).first()[0]) if n else 0.0

    grid_points = int(max(2, grid_points))
    thresholds = np.linspace(0.0, 1.0, grid_points)

    best = {"thr": 0.5, "f1": -1.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0, "prec": 0.0, "rec": 0.0, "acc": 0.0}
    for thr in thresholds:
        t = float(thr)
        m = base.withColumn("yhat", (F.col("p") >= F.lit(t)).cast("int"))
        tp = int(m.filter((F.col("y") == 1) & (F.col("yhat") == 1)).count())
        fp = int(m.filter((F.col("y") == 0) & (F.col("yhat") == 1)).count())
        tn = int(m.filter((F.col("y") == 0) & (F.col("yhat") == 0)).count())
        fn = int(m.filter((F.col("y") == 1) & (F.col("yhat") == 0)).count())
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
        if f1 > best["f1"]:
            best = {"thr": t, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "prec": prec, "rec": rec, "acc": acc}

    row = {
        "label": f"{label}_best_f1",
        "n": int(n),
        "pos_rate": float(pos_rate),
        "threshold": float(best["thr"]),
        "auc_roc": None,
        "auc_pr": None,
        "accuracy": float(best["acc"]),
        "precision": float(best["prec"]),
        "recall": float(best["rec"]),
        "f1": float(best["f1"]),
        "tp": int(best["tp"]),
        "fp": int(best["fp"]),
        "tn": int(best["tn"]),
        "fn": int(best["fn"]),
    }
    return scored_df.sql_ctx.createDataFrame([row])


def train_gbt_spark(df_train, df_valid, feature_cols: List[str], label_col: str, cat_cols: List[str], params: Dict[str, Any]):
    """Train a GBT classifier pipeline (string-index cats + vector assemble)."""

    from pyspark.ml import Pipeline
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.feature import StringIndexer, VectorAssembler

    num_cols = [c for c in feature_cols if c not in cat_cols]
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}__idx", handleInvalid="keep")
        for c in cat_cols
        if c in feature_cols
    ]
    assembled_inputs = num_cols + [f"{c}__idx" for c in cat_cols if c in feature_cols]
    assembler = VectorAssembler(inputCols=assembled_inputs, outputCol="features", handleInvalid="keep")

    gbt = GBTClassifier(
        featuresCol="features",
        labelCol=label_col,
        probabilityCol="probability",
        seed=42,
        maxIter=int(params.get("maxIter", 80)),
        maxDepth=int(params.get("maxDepth", 6)),
        stepSize=float(params.get("stepSize", 0.1)),
        subsamplingRate=float(params.get("subsamplingRate", 0.8)),
        maxBins=int(params.get("maxBins", 32)),
    )

    pipe = Pipeline(stages=[*indexers, assembler, gbt])
    model = pipe.fit(df_train)
    return model


def gbt_feature_importance_spark(model, feature_cols: List[str], cat_cols: List[str]):
    """Return a Spark DF of feature importance mapped to original columns.

    Mapping works because we *do not* one-hot encode categoricals; we index them.
    """

    from pyspark.sql import Row

    # model is PipelineModel; last stage is GBTClassificationModel
    gbt_stage = model.stages[-1]
    importances = gbt_stage.featureImportances

    num_cols = [c for c in feature_cols if c not in cat_cols]
    assembled_inputs = num_cols + [c for c in cat_cols if c in feature_cols]

    rows = []
    for i, col in enumerate(assembled_inputs):
        rows.append(Row(feature=str(col), importance=float(importances[i]) if i < len(importances) else 0.0))

    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    return spark.createDataFrame(rows).orderBy("importance", ascending=False)


def select_top_k_spark(fi_df, k: int):
    top = fi_df.limit(int(k)).select("feature").collect()
    return [r[0] for r in top]


def restrict_features_spark(df, selected_features: List[str], id_cols: List[str], label_col: str):
    cols = [c for c in id_cols + [label_col] + list(selected_features) if c in df.columns]
    return df.select(*cols)


def tune_gbt_spark(df_train, df_valid, feature_cols: List[str], label_col: str, cat_cols: List[str], n_trials: int, seed: int, tune_verbose: bool):
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    rng = np.random.default_rng(int(seed))
    best_auc = -1.0
    best_params = None
    best_model = None

    trial_rows: List[Dict[str, Any]] = []

    evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    for i in range(int(n_trials)):
        params = {
            "maxIter": int(rng.integers(40, 160)),
            "maxDepth": int(rng.integers(3, 8)),
            "stepSize": float(rng.uniform(0.03, 0.2)),
            "subsamplingRate": float(rng.uniform(0.7, 1.0)),
            "maxBins": int(rng.integers(16, 64)),
        }

        model = train_gbt_spark(df_train, df_valid, feature_cols, label_col, cat_cols, params)
        pred = model.transform(df_valid)
        auc = float(evaluator.evaluate(pred))

        trial_rows.append({"trial": int(i + 1), "valid_auc": float(auc), **{f"param_{k}": v for k, v in params.items()}})

        if bool(tune_verbose):
            pretty = ", ".join([f"{k}={params[k]}" for k in sorted(params.keys())])
            print("  [tune] trial %d/%d | valid_auc=%.5f | best_so_far=%.5f" % (i + 1, int(n_trials), auc, max(best_auc, auc)))
            print("  [tune] params:", pretty)

        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model

    # Return trials as Spark DF for catalog parity.
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    trials_df = spark.createDataFrame(trial_rows).orderBy("valid_auc", ascending=False)
    return best_model, best_params, float(best_auc), trials_df


def score_test_spark(df_full, df_test, id_cols: List[str], label_col: str, model, selected_features: List[str]):
    """Score test split and return a scored DataFrame with selected feature columns."""

    from pyspark.sql import functions as F
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    # Ensure df_test has all required columns used by model stages.
    pred = model.transform(df_test)
    scored = pred.select(*[c for c in id_cols if c in pred.columns], F.col(label_col).alias(label_col), *[c for c in selected_features if c in pred.columns], F.col("probability"), F.col("rawPrediction"))
    scored = scored.withColumn("p_apply", F.col("probability")[1].cast("double"))

    evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = float(evaluator.evaluate(scored))
    scored = scored.withColumn("auc_on_test", F.lit(float(auc)))
    return scored


def add_deciles_spark(scored_df):
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    w = Window.orderBy(F.col("p_apply").asc())
    return scored_df.withColumn("decile", F.ntile(10).over(w))


def decile_table_spark(scored_with_deciles, label_col: str):
    from pyspark.sql import functions as F

    base_rate = float(scored_with_deciles.select(F.avg(F.col(label_col).cast("double"))).first()[0])

    tbl = (
        scored_with_deciles.groupBy("decile")
        .agg(
            F.count(F.lit(1)).alias("customers"),
            F.sum(F.col(label_col).cast("int")).alias("responders"),
            F.avg(F.col("p_apply")).alias("avg_score"),
        )
        .orderBy(F.col("decile").desc())
    )

    tbl = tbl.withColumn("response_rate", F.col("responders") / F.col("customers"))
    tbl = tbl.withColumn("lift_vs_avg", F.col("response_rate") / F.lit(float(base_rate) if base_rate else 1.0))
    return tbl


def generate_leads_spark(scored_with_deciles, min_decile: int):
    from pyspark.sql import functions as F

    return scored_with_deciles.filter(F.col("decile") >= F.lit(int(min_decile))).orderBy(F.col("p_apply").desc())


def add_rowwise_explanations_spark(leads_df, selected_features: List[str], top_n: int):
    """Add lightweight explanation columns.

    This is NOT SHAP. It simply surfaces the top-N selected feature names and their values.
    """

    from pyspark.sql import functions as F

    top_n = int(max(1, top_n))
    top_feats = [c for c in list(selected_features) if c in leads_df.columns][:top_n]

    out = leads_df
    out = out.withColumn("exp_expected_value", F.lit(None).cast("double"))

    for i in range(top_n):
        feat = top_feats[i] if i < len(top_feats) else None
        out = out.withColumn(f"exp_{i+1}_feature", F.lit(feat))
        if feat is None:
            out = out.withColumn(f"exp_{i+1}_value", F.lit(None))
        else:
            out = out.withColumn(f"exp_{i+1}_value", F.col(feat))
        out = out.withColumn(f"exp_{i+1}_contribution", F.lit(None).cast("double"))

    return out


def save_importance_plot_spark(fi_df, path: str):
    """Save a global feature-importance plot to PNG (matplotlib)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdf = fi_df.toPandas()
    if pdf.empty:
        raise ValueError("feature importance is empty")

    pdf = pdf.sort_values("importance", ascending=True).reset_index(drop=True)
    fig_h = max(4.0, 0.35 * len(pdf))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(pdf["feature"], pdf["importance"], color="C0")
    ax.set_title("Feature importance (GBT)")
    ax.set_xlabel("importance")
    ax.set_ylabel("feature")
    plt.tight_layout()

    out_path = _resolve_output_path(path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def concat_metrics_spark(*dfs):
    dfs2 = [d for d in dfs if d is not None]
    if not dfs2:
        return None
    out = dfs2[0]
    for d in dfs2[1:]:
        out = out.unionByName(d, allowMissingColumns=True)
    return out


def _resolve_output_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((Path(__file__).resolve().parent / p).resolve())


def build_default_catalog(master: str, shuffle_partitions: int) -> DataCatalog:
    spark = build_spark(master=master, shuffle_partitions=shuffle_partitions)

    initial: Dict[str, Any] = {
        "spark": spark,
        "params:n_customers": 6000,
        "params:seed": 42,
        "params:valid_size": 0.2,
        "params:test_size": 0.25,
        "params:top_k": 8,
        "params:tune_trials": 10,
        "params:tune_verbose": True,
        "params:min_decile_for_leads": 9,
        "params:explain_top_n": 5,
        "params:baseline_label": "baseline_valid",
        "params:final_label": "final_test",
        "params:report_thresholds": [0.3, 0.5, 0.7],
        "params:f1_threshold_grid_points": 101,
        # outputs
        "params:leads_csv_path": _resolve_output_path("kedro_style_leads_spark_csv"),
        "params:metrics_csv_path": _resolve_output_path("kedro_style_performance_metrics_spark_csv"),
        "params:model_path": _resolve_output_path("kedro_style_model_spark"),
        "params:importance_png_path": _resolve_output_path("kedro_style_feature_importance_spark.png"),
    }

    catalog = DataCatalog(initial=initial)

    # Register file-backed outputs (Kedro-like “catalog entries”).
    catalog.register("leads", SparkCSVDataSet(spark, initial["params:leads_csv_path"], header=True, mode="overwrite"))
    catalog.register(
        "performance_metrics",
        SparkCSVDataSet(spark, initial["params:metrics_csv_path"], header=True, mode="overwrite"),
    )
    catalog.register("model", SparkMLlibModelDataSet(initial["params:model_path"], mode="overwrite"))

    return catalog


def build_pipelines() -> Dict[str, Pipeline]:
    """Register runnable pipelines.

    Mirrors the pure-Python script’s pipeline names and node ordering.
    """

    common_prefix = [
        Node(create_customers_spark, inputs=["spark", "params:n_customers", "params:seed"], outputs="raw_customers"),
        Node(add_target_spark, inputs=["raw_customers", "params:seed"], outputs="labeled_customers"),
        Node(split_xy_spark, inputs="labeled_customers", outputs=["df_full", "feature_cols", "label_col", "cat_cols"]),
        Node(
            split_train_test_spark,
            inputs=["df_full", "params:test_size", "params:seed"],
            outputs=["df_train_all", "df_test"],
            name="split_train_test",
        ),
        Node(
            split_train_valid_spark,
            inputs=["df_train_all", "params:valid_size", "params:seed"],
            outputs=["df_train", "df_valid"],
            name="split_train_valid",
        ),
    ]

    baseline_nodes = common_prefix + [
        Node(
            baseline_scaled_model_spark,
            inputs=["df_train", "df_valid", "feature_cols", "label_col", "cat_cols"],
            outputs=["baseline_model", "baseline_auc"],
        ),
        Node(
            predict_proba_spark,
            inputs=["baseline_model", "df_valid", "label_col"],
            outputs="baseline_scored_valid",
            name="baseline_score_valid",
        ),
        Node(
            metrics_table_from_scores_spark,
            inputs=["baseline_scored_valid", "label_col", "params:report_thresholds", "params:baseline_label"],
            outputs="baseline_metrics_table",
        ),
        Node(
            best_f1_metrics_from_scores_spark,
            inputs=["baseline_scored_valid", "label_col", "params:f1_threshold_grid_points", "params:baseline_label"],
            outputs="baseline_best_f1_metrics",
        ),
    ]

    train_nodes = baseline_nodes + [
        Node(
            tune_gbt_spark,
            inputs=[
                "df_train",
                "df_valid",
                "feature_cols",
                "label_col",
                "cat_cols",
                "params:tune_trials",
                "params:seed",
                "params:tune_verbose",
            ],
            outputs=["model_all", "best_params", "best_valid_auc", "tuning_trials"],
            name="tune_gbt_all_features",
        ),
        Node(
            gbt_feature_importance_spark,
            inputs=["model_all", "feature_cols", "cat_cols"],
            outputs="feature_importance",
        ),
        Node(select_top_k_spark, inputs=["feature_importance", "params:top_k"], outputs="selected_features"),
        Node(
            restrict_features_spark,
            inputs=["df_full", "selected_features", "params:id_cols", "label_col"],
            outputs="df_full_sel",
            name="restrict_features_full",
        ),
        Node(
            split_train_test_spark,
            inputs=["df_full_sel", "params:test_size", "params:seed"],
            outputs=["df_train_all_sel", "df_test_sel"],
            name="split_train_test_selected",
        ),
        Node(
            split_train_valid_spark,
            inputs=["df_train_all_sel", "params:valid_size", "params:seed"],
            outputs=["df_train_sel", "df_valid_sel"],
            name="split_train_valid_selected",
        ),
        Node(
            tune_gbt_spark,
            inputs=[
                "df_train_sel",
                "df_valid_sel",
                "selected_features",
                "label_col",
                "cat_cols",
                "params:tune_trials",
                "params:seed",
                "params:tune_verbose",
            ],
            outputs=["model_sel", "best_params_sel", "best_valid_auc_sel", "tuning_trials_sel"],
            name="tune_gbt_selected_features",
        ),
    ]

    full_nodes = train_nodes + [
        # Persist final model via the catalog entry.
        Node(lambda m: m, inputs=["model_sel"], outputs="model", name="save_model"),
        Node(
            score_test_spark,
            inputs=["df_full_sel", "df_test_sel", "params:id_cols", "label_col", "model_sel", "selected_features"],
            outputs="scored_test",
        ),
        Node(
            metrics_table_from_scores_spark,
            inputs=["scored_test", "label_col", "params:report_thresholds", "params:final_label"],
            outputs="final_metrics_table",
        ),
        Node(
            best_f1_metrics_from_scores_spark,
            inputs=["scored_test", "label_col", "params:f1_threshold_grid_points", "params:final_label"],
            outputs="final_best_f1_metrics",
        ),
        Node(add_deciles_spark, inputs="scored_test", outputs="scored_test_with_deciles"),
        Node(decile_table_spark, inputs=["scored_test_with_deciles", "label_col"], outputs="decile_table"),
        Node(
            generate_leads_spark,
            inputs=["scored_test_with_deciles", "params:min_decile_for_leads"],
            outputs="leads_raw",
        ),
        Node(
            add_rowwise_explanations_spark,
            inputs=["leads_raw", "selected_features", "params:explain_top_n"],
            outputs="leads",
        ),
        Node(
            save_importance_plot_spark,
            inputs=["feature_importance", "params:importance_png_path"],
            outputs="importance_png_path",
        ),
        Node(
            concat_metrics_spark,
            inputs=["baseline_metrics_table", "baseline_best_f1_metrics", "final_metrics_table", "final_best_f1_metrics"],
            outputs="performance_metrics",
        ),
    ]

    return {
        "baseline": Pipeline(name="cc_propensity_baseline_spark", nodes=baseline_nodes),
        "train": Pipeline(name="cc_propensity_train_spark", nodes=train_nodes),
        "full": Pipeline(name="cc_propensity_full_spark", nodes=full_nodes),
        "__meta__": Pipeline(nodes=[], name="__meta__"),
        "__meta__id_cols": Pipeline(nodes=[], name="__meta__id_cols"),
    }


def _print_outputs(catalog: DataCatalog) -> None:
    c = catalog.as_dict()
    print("\n=== Outputs ===")

    if "baseline_auc" in c:
        print("Baseline (Spark LogisticRegression) AUC on validation:", round(float(c["baseline_auc"]), 4))

    if "best_params_sel" in c:
        print("Best tuning params (selected features):", c["best_params_sel"])
    if "best_valid_auc_sel" in c:
        print("Best validation AUC (selected features):", round(float(c["best_valid_auc_sel"]), 4))

    if "scored_test" in c:
        try:
            auc_on_test = c["scored_test"].select("auc_on_test").first()[0]
            print("Test AUC (GBT on selected features):", round(float(auc_on_test), 4))
        except Exception:
            pass

    if "importance_png_path" in c:
        print("Feature importance plot:", c["importance_png_path"])

    # These are file-backed via the catalog, so print the target paths.
    if "params:leads_csv_path" in c:
        print("Leads CSV folder:", c["params:leads_csv_path"])
    if "params:metrics_csv_path" in c:
        print("Performance metrics CSV folder:", c["params:metrics_csv_path"])
    if "params:model_path" in c:
        print("Model folder:", c["params:model_path"])

    if "decile_table" in c:
        print("\nDecile table:")
        c["decile_table"].show(20, truncate=False)

    if "feature_importance" in c:
        print("\nTop feature importance:")
        c["feature_importance"].show(20, truncate=False)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    pipelines = build_pipelines()

    parser = argparse.ArgumentParser(description="Kedro-style propensity pipeline runner (Spark)")
    parser.add_argument(
        "--pipeline",
        default="full",
        choices=sorted([k for k in pipelines.keys() if not k.startswith("__meta__")]),
        help="Which pipeline to execute",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="Print available pipelines and exit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce node-by-node logging",
    )
    parser.add_argument(
        "--spark-master",
        default=os.environ.get("SPARK_MASTER", "local[*]"),
        help="Spark master URL (default: local[*])",
    )
    parser.add_argument(
        "--spark-shuffle-partitions",
        type=int,
        default=int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8")),
        help="spark.sql.shuffle.partitions (default: 8)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(pipeline_name: str = "full", verbose: bool = True, spark_master: str = "local[*]", shuffle_partitions: int = 8) -> DataCatalog:
    pipelines = build_pipelines()
    if pipeline_name not in pipelines:
        raise SystemExit("Unknown pipeline '%s'. Use --list-pipelines to see options." % pipeline_name)

    catalog = build_default_catalog(master=spark_master, shuffle_partitions=int(shuffle_partitions))
    # Keep id_cols in params for node wiring.
    catalog["params:id_cols"] = ["customer_id", "email", "phone"]

    catalog = run_pipeline(pipelines[pipeline_name], catalog=catalog, verbose=verbose)
    _print_outputs(catalog)

    # Stop Spark at the very end (Kedro-style).
    try:
        spark = catalog["spark"]
        spark.stop()
    except Exception:
        pass

    return catalog


if __name__ == "__main__":
    args = _parse_args()
    if args.list_pipelines:
        print("Available pipelines:")
        for name, pl in build_pipelines().items():
            if name.startswith("__meta__"):
                continue
            print("- %s (%s, %d nodes)" % (name, pl.name, len(pl.nodes)))
        raise SystemExit(0)

    main(
        pipeline_name=str(args.pipeline),
        verbose=not bool(args.quiet),
        spark_master=str(args.spark_master),
        shuffle_partitions=int(args.spark_shuffle_partitions),
    )
