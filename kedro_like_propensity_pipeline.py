"""Kedro-Style ML Flow (Script)

Kedro-inspired structure implemented in pure Python:
- Node: function with declared inputs/outputs
- Pipeline: ordered list of nodes
- Catalog: dict of named datasets/objects
- Runner: executes nodes, materializes outputs back to the catalog

This script mirrors the notebook `kedro_style_propensity_pipeline.ipynb`:
- target creation
- baseline (scaled LogisticRegression)
- CatBoost training
- feature selection via importance
- lightweight hyperparameter tuning
- decile analysis + lead export

Run:
  python kedro_style_propensity_pipeline.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def run_pipeline(
    pipeline: Pipeline,
    catalog: Catalog,
    verbose: bool = True,
    on_node_start: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_node_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_node_error: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Catalog:
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
    if node_timings:
        timings_df = pd.DataFrame(node_timings)
        timings_df["pct_total"] = (timings_df["seconds"] / max(total, 1e-12)) * 100.0
        catalog["run:timings"] = timings_df.sort_values("seconds", ascending=False).reset_index(drop=True)
        catalog["run:total_seconds"] = float(total)

        if verbose:
            print("\n=== Timing snapshot (slowest first) ===")
            view = catalog["run:timings"][["node", "seconds", "pct_total"]].copy()
            view["seconds"] = view["seconds"].map(lambda s: round(float(s), 4))
            view["pct_total"] = view["pct_total"].map(lambda p: round(float(p), 2))
            print(view.to_string(index=False))
            print("Total pipeline time (s):", round(float(catalog["run:total_seconds"]), 4))
    return catalog


def ensure_catboost_installed() -> None:
    try:
        import catboost  # noqa: F401

        return
    except Exception:
        pass

    import subprocess

    print("Installing catboost...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])


def _normalize_cat_cols(X: pd.DataFrame, cat_cols: List[str]) -> List[str]:
    if cat_cols is None:
        return []
    return [c for c in list(cat_cols) if c in X.columns]


def _cast_cat_cols_to_str(X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    if not cat_cols:
        return X
    out = X.copy()
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("NA").astype(str)
    return out


def _make_pool(X: pd.DataFrame, y: Any = None, cat_cols: Optional[List[str]] = None):
    ensure_catboost_installed()
    from catboost import Pool

    cat_cols = _normalize_cat_cols(X, cat_cols or [])
    X2 = _cast_cat_cols_to_str(X, cat_cols)
    cat_indices = [X2.columns.get_loc(c) for c in cat_cols]
    return Pool(X2, y, cat_features=cat_indices), cat_cols


def create_customers(n_customers: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    n = int(n_customers)
    df = pd.DataFrame(
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
    df["email"] = "cust" + df["customer_id"].astype(str) + "@example.com"
    area = pd.Series(rng.integers(100, 999, size=n), index=df.index).astype(str).str.zfill(3)
    suffix = pd.Series(rng.integers(1000, 9999, size=n), index=df.index).astype(str).str.zfill(4)
    df["phone"] = "+1-555-" + area + "-" + suffix
    return df


def add_target(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    score = (
        0.025 * (df["credit_score"] - 650)
        + 0.000012 * (df["annual_income"] - 60000)
        + 0.030 * (df["txn_count_3m"] - 15)
        + 0.002 * (df["tenure_months"] - 24)
        + 0.55 * df["is_premium"]
        + 0.10 * df["has_mortgage"]
        - 2.0 * df["utilization_rate"]
        - 0.45 * df["late_payments_12m"]
    )
    noise = rng.normal(0, 1.2, size=len(df))
    p_apply = sigmoid(score + noise)
    out = df.copy()
    out["applied_cc"] = (rng.random(len(df)) < p_apply).astype(int)
    return out


def split_xy(df: pd.DataFrame):
    target_col = "applied_cc"
    id_cols = ["customer_id", "email", "phone"]
    cat_cols = ["region", "channel"]

    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, cat_cols


def split_train_valid(X: pd.DataFrame, y: pd.Series, valid_size: float, seed: int):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=float(valid_size),
        random_state=int(seed),
        stratify=y,
    )
    return X_tr, X_val, y_tr, y_val


def baseline_scaled_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cat_cols: List[str],
):
    cat_cols = _normalize_cat_cols(X_train, cat_cols)
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    model = SkPipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]
    )
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p))
    return model, auc


def predict_proba_sklearn(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def classification_metrics_from_scores(
    y_true: pd.Series,
    y_score: Union[pd.Series, np.ndarray, List[float]],
    threshold: float,
    label: str,
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    thr = float(threshold)

    y_pred = (y_score_arr >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    row = {
        "label": str(label),
        "n": int(len(y_true_arr)),
        "pos_rate": float(np.mean(y_true_arr)),
        "threshold": thr,
        "auc_roc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "auc_pr": float(average_precision_score(y_true_arr, y_score_arr)),
        "log_loss": float(log_loss(y_true_arr, y_score_arr, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return pd.DataFrame([row])


def metrics_table_from_scores(
    y_true: pd.Series,
    y_score: Union[pd.Series, np.ndarray, List[float]],
    thresholds: Sequence[float],
    label_prefix: str,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for thr in list(thresholds):
        rows.append(classification_metrics_from_scores(y_true, y_score, float(thr), f"{label_prefix}@{float(thr):g}"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def best_f1_metrics_from_scores(
    y_true: pd.Series,
    y_score: Union[pd.Series, np.ndarray, List[float]],
    grid_points: int,
    label: str,
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)

    n_grid = int(max(2, grid_points))
    thresholds = np.linspace(0.0, 1.0, n_grid)
    best_thr = 0.5
    best_f1 = -1.0

    # Simple sweep; n is small in this demo pipeline.
    for thr in thresholds:
        y_pred = (y_score_arr >= float(thr)).astype(int)
        score = float(f1_score(y_true_arr, y_pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)

    return classification_metrics_from_scores(y_true, y_score, best_thr, f"{label}_best_f1")


def metrics_from_scored_df(scored: pd.DataFrame, threshold: float, label: str) -> pd.DataFrame:
    if "applied_cc" not in scored.columns or "p_apply" not in scored.columns:
        raise ValueError("scored must contain applied_cc and p_apply")
    return classification_metrics_from_scores(scored["applied_cc"], scored["p_apply"], threshold, label)


def metrics_table_from_scored_df(scored: pd.DataFrame, thresholds: Sequence[float], label_prefix: str) -> pd.DataFrame:
    if "applied_cc" not in scored.columns or "p_apply" not in scored.columns:
        raise ValueError("scored must contain applied_cc and p_apply")
    return metrics_table_from_scores(scored["applied_cc"], scored["p_apply"], thresholds, label_prefix)


def best_f1_metrics_from_scored_df(scored: pd.DataFrame, grid_points: int, label: str) -> pd.DataFrame:
    if "applied_cc" not in scored.columns or "p_apply" not in scored.columns:
        raise ValueError("scored must contain applied_cc and p_apply")
    return best_f1_metrics_from_scores(scored["applied_cc"], scored["p_apply"], grid_points, label)


def concat_metrics(*dfs: pd.DataFrame) -> pd.DataFrame:
    parts = [d for d in dfs if d is not None and len(d) > 0]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def save_metrics_csv(metrics: pd.DataFrame, path: str) -> str:
    out_path = _resolve_output_path(path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)
    return out_path


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_cols: List[str],
    params: Dict[str, Any],
):
    ensure_catboost_installed()
    from catboost import CatBoostClassifier

    cat_cols = _normalize_cat_cols(X_train, cat_cols)
    train_pool, cat_cols = _make_pool(X_train, y_train, cat_cols)
    valid_pool, _ = _make_pool(X_valid, y_valid, cat_cols)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        **params,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
    )
    return model


def catboost_feature_importance(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_cols: List[str],
) -> pd.DataFrame:
    pool, _ = _make_pool(X_train, y_train, cat_cols)
    imp = model.get_feature_importance(pool)
    fi = pd.DataFrame({"feature": X_train.columns, "importance": imp})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)


def select_top_k(fi: pd.DataFrame, k: int) -> List[str]:
    return fi["feature"].head(int(k)).tolist()


def restrict_features(X_train: pd.DataFrame, X_test: pd.DataFrame, selected: List[str]):
    return X_train[selected], X_test[selected]


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_cols: List[str],
    n_trials: int,
    seed: int,
    tune_verbose: bool,
):
    rng = np.random.default_rng(int(seed))
    best_auc = -1.0
    best_params = None
    best_model = None

    trial_rows: List[Dict[str, Any]] = []

    for i in range(int(n_trials)):
        params = {
            "iterations": int(rng.integers(250, 800)),
            "learning_rate": float(rng.uniform(0.02, 0.15)),
            "depth": int(rng.integers(4, 9)),
            "l2_leaf_reg": float(rng.uniform(1.0, 12.0)),
            "random_strength": float(rng.uniform(0.0, 2.0)),
            "subsample": float(rng.uniform(0.7, 1.0)),
            "early_stopping_rounds": 40,
        }

        model = train_catboost(X_train, y_train, X_valid, y_valid, cat_cols, params)

        valid_pool, _ = _make_pool(X_valid, None, _normalize_cat_cols(X_valid, cat_cols))
        p = model.predict_proba(valid_pool)[:, 1]
        auc = float(roc_auc_score(y_valid, p))

        row: Dict[str, Any] = {
            "trial": int(i + 1),
            "valid_auc": float(auc),
            **{f"param_{k}": v for k, v in params.items()},
        }
        trial_rows.append(row)

        if bool(tune_verbose):
            print(
                "  [tune] trial %d/%d | valid_auc=%.5f | best_so_far=%.5f"
                % (i + 1, int(n_trials), auc, max(best_auc, auc))
            )
            # Keep params readable and stable-order in output.
            pretty = ", ".join([f"{k}={params[k]}" for k in sorted(params.keys())])
            print("  [tune] params:", pretty)
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model

    trials_df = pd.DataFrame(trial_rows).sort_values("valid_auc", ascending=False).reset_index(drop=True)
    return best_model, best_params, best_auc, trials_df


def score_test(
    df_full: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model,
    cat_cols: List[str],
) -> pd.DataFrame:
    scored = df_full.loc[X_test.index, ["customer_id", "email", "phone", "applied_cc"]].copy()
    # Add the exact feature columns used for scoring (e.g., selected top-k features).
    # These will flow into the leads export.
    scored = scored.join(X_test, how="left")
    test_pool, _ = _make_pool(X_test, None, _normalize_cat_cols(X_test, cat_cols))
    scored["p_apply"] = model.predict_proba(test_pool)[:, 1]
    scored["auc_on_test"] = float(roc_auc_score(y_test, scored["p_apply"]))
    return scored


def add_deciles(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["decile"] = pd.qcut(out["p_apply"].rank(method="first"), 10, labels=False) + 1
    return out


def decile_table(scored_with_deciles: pd.DataFrame) -> pd.DataFrame:
    base_rate = float(scored_with_deciles["applied_cc"].mean())
    tbl = (
        scored_with_deciles.groupby("decile", as_index=False)
        .agg(
            customers=("customer_id", "count"),
            responders=("applied_cc", "sum"),
            avg_score=("p_apply", "mean"),
        )
        .sort_values("decile", ascending=False)
    )
    tbl["response_rate"] = tbl["responders"] / tbl["customers"]
    tbl["lift_vs_avg"] = tbl["response_rate"] / base_rate
    return tbl.reset_index(drop=True)


def generate_leads(scored_with_deciles: pd.DataFrame, min_decile: int) -> pd.DataFrame:
    leads = scored_with_deciles[scored_with_deciles["decile"] >= int(min_decile)].copy()
    return leads.sort_values("p_apply", ascending=False).reset_index(drop=True)


def save_leads(leads: pd.DataFrame, selected_features: List[str], path: str) -> str:
    def _safe_shap_col(feature: str) -> str:
        safe = "".join([(ch if (ch.isalnum() or ch == "_") else "_") for ch in str(feature)])
        while "__" in safe:
            safe = safe.replace("__", "_")
        return "shap__" + safe.strip("_")

    ordered_features = [c for c in list(selected_features) if c in leads.columns]
    ordered_shap_cols = [_safe_shap_col(c) for c in ordered_features if _safe_shap_col(c) in leads.columns]

    exp_expected = ["exp_expected_value"] if "exp_expected_value" in leads.columns else []
    exp_ranked: List[str] = []
    for i in range(1, 101):
        for suffix in ("feature", "contribution", "value"):
            c = f"exp_{i}_{suffix}"
            if c in leads.columns:
                exp_ranked.append(c)

    cols = [
        "customer_id",
        "email",
        "phone",
        *ordered_features,
        *ordered_shap_cols,
        "p_apply",
        "decile",
        *exp_expected,
        *exp_ranked,
    ]
    cols = [c for c in cols if c in leads.columns]
    leads[cols].to_csv(path, index=False)
    return path


def save_model_pickle(model: Any, path: str) -> str:
    path = _resolve_output_path(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _catboost_shap_values(model: Any, X: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
    """Return CatBoost SHAP values array with shape (n_rows, n_features + 1).

    The last column is the expected value; feature contributions are [:, :-1].
    """

    pool, _ = _make_pool(X, None, _normalize_cat_cols(X, cat_cols))
    # CatBoost can compute SHAP values natively (no external shap dependency).
    return model.get_feature_importance(pool, type="ShapValues")


def add_rowwise_shap_explanations(
    leads: pd.DataFrame,
    model: Any,
    selected_features: List[str],
    cat_cols: List[str],
    top_n: int,
) -> pd.DataFrame:
    """Attach per-row SHAP explanations to the leads DataFrame.

    Adds columns:
      - exp_expected_value
      - exp_1_feature, exp_1_contribution, exp_1_value
      - ... up to top_n

    Explanations are computed over the `selected_features` used by the final model.
    """

    X = leads[[c for c in selected_features if c in leads.columns]].copy()
    if X.empty:
        return leads

    shap_values = _catboost_shap_values(model, X, cat_cols)
    contrib = shap_values[:, :-1]
    expected = shap_values[:, -1]

    top_n = int(max(1, top_n))
    feature_names = list(X.columns)

    out = leads.copy()
    out["exp_expected_value"] = expected

    def _safe_shap_col(feature: str) -> str:
        safe = "".join([(ch if (ch.isalnum() or ch == "_") else "_") for ch in str(feature)])
        while "__" in safe:
            safe = safe.replace("__", "_")
        return "shap__" + safe.strip("_")

    # Add one column per feature with the SHAP contribution for that row.
    for j, feature_name in enumerate(feature_names):
        col = _safe_shap_col(feature_name)
        if col in out.columns:
            col = col + "_contribution"
        out[col] = contrib[:, j]

    abs_contrib = np.abs(contrib)
    # Indices of top-N absolute contributions per row (descending).
    top_idx = np.argsort(-abs_contrib, axis=1)[:, :top_n]

    for rank in range(top_n):
        idx = top_idx[:, rank]
        out[f"exp_{rank+1}_feature"] = [feature_names[i] for i in idx]
        out[f"exp_{rank+1}_contribution"] = contrib[np.arange(len(X)), idx]
        out[f"exp_{rank+1}_value"] = X.to_numpy()[np.arange(len(X)), idx]

    return out


def save_shap_importance_plot(
    model: Any,
    X_ref: pd.DataFrame,
    selected_features: List[str],
    cat_cols: List[str],
    path: str,
) -> str:
    """Save an overall SHAP importance plot (mean |SHAP| per feature) as a PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = X_ref[[c for c in selected_features if c in X_ref.columns]].copy()
    if X.empty:
        raise ValueError("No selected features found in X_ref")

    shap_values = _catboost_shap_values(model, X, cat_cols)
    contrib = shap_values[:, :-1]
    mean_abs = np.mean(np.abs(contrib), axis=0)

    fi = pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": mean_abs})
    fi = fi.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)

    fig_h = max(4.0, 0.35 * len(fi))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(fi["feature"], fi["mean_abs_shap"], color="C0")
    ax.set_title("SHAP feature importance (mean |SHAP|)")
    ax.set_xlabel("mean(|SHAP contribution|)")
    ax.set_ylabel("feature")
    plt.tight_layout()

    out_path = _resolve_output_path(path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _resolve_output_path(path_str: str) -> str:
    """Resolve relative paths next to this script (more notebook-like when run anywhere)."""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((Path(__file__).resolve().parent / p).resolve())


def build_default_catalog() -> Catalog:
    catalog: Catalog = {
        "params:n_customers": 6000,
        "params:seed": 42,
        "params:top_k": 8,
        "params:valid_size": 0.2,
        "params:tune_trials": 10,
        "params:tune_verbose": True,
        "params:min_decile_for_leads": 9,
        "params:leads_csv_path": _resolve_output_path("kedro_style_leads.csv"),
        "params:model_pickle_path": _resolve_output_path("kedro_style_model.pkl"),
        "params:explain_top_n": 5,
        "params:shap_importance_png_path": _resolve_output_path("kedro_style_shap_importance.png"),
        "params:decision_threshold": 0.5,
        "params:metrics_csv_path": _resolve_output_path("kedro_style_performance_metrics.csv"),
        "params:baseline_label": "baseline_valid",
        "params:final_label": "final_test",
        "params:report_thresholds": [0.3, 0.5, 0.7],
        "params:f1_threshold_grid_points": 101,
    }

    catalog["params:catboost_defaults"] = {
        "iterations": 500,
        "learning_rate": 0.08,
        "depth": 6,
        "early_stopping_rounds": 40,
    }
    return catalog


def build_pipelines() -> Dict[str, Pipeline]:
    """Register runnable pipelines.

    Keep this small and practical for interview practice:
    - `full`: end-to-end (baseline + CatBoost + deciles + leads export)
    - `baseline`: only the baseline model AUC on validation
    - `train`: CatBoost training + feature selection + tuning (no scoring/export)
    """

    common_prefix = [
        Node(create_customers, inputs=["params:n_customers", "params:seed"], outputs="raw_customers"),
        Node(add_target, inputs=["raw_customers", "params:seed"], outputs="labeled_customers"),
        Node(
            split_xy,
            inputs="labeled_customers",
            outputs=["X_train", "X_test", "y_train", "y_test", "cat_cols"],
        ),
        Node(
            split_train_valid,
            inputs=["X_train", "y_train", "params:valid_size", "params:seed"],
            outputs=["X_tr", "X_val", "y_tr", "y_val"],
            name="split_train_valid",
        ),
    ]

    baseline_nodes = common_prefix + [
        Node(
            baseline_scaled_model,
            inputs=["X_tr", "X_val", "y_tr", "y_val", "cat_cols"],
            outputs=["baseline_model", "baseline_auc"],
        )
    ]

    baseline_nodes = baseline_nodes + [
        Node(predict_proba_sklearn, inputs=["baseline_model", "X_val"], outputs="baseline_scores"),
        Node(
            metrics_table_from_scores,
            inputs=["y_val", "baseline_scores", "params:report_thresholds", "params:baseline_label"],
            outputs="baseline_metrics_table",
            name="baseline_metrics_table",
        ),
        Node(
            best_f1_metrics_from_scores,
            inputs=["y_val", "baseline_scores", "params:f1_threshold_grid_points", "params:baseline_label"],
            outputs="baseline_best_f1_metrics",
            name="baseline_best_f1_metrics",
        ),
    ]

    train_nodes = baseline_nodes + [
        Node(
            train_catboost,
            inputs=["X_tr", "y_tr", "X_val", "y_val", "cat_cols", "params:catboost_defaults"],
            outputs="model_all",
            name="train_catboost_all",
        ),
        Node(
            catboost_feature_importance,
            inputs=["model_all", "X_tr", "y_tr", "cat_cols"],
            outputs="feature_importance",
        ),
        Node(select_top_k, inputs=["feature_importance", "params:top_k"], outputs="selected_features"),
        Node(
            restrict_features,
            inputs=["X_train", "X_test", "selected_features"],
            outputs=["X_train_sel", "X_test_sel"],
        ),
        Node(
            split_train_valid,
            inputs=["X_train_sel", "y_train", "params:valid_size", "params:seed"],
            outputs=["X_tr_sel", "X_val_sel", "y_tr_sel", "y_val_sel"],
            name="split_train_valid_selected",
        ),
        Node(
            tune_catboost,
            inputs=[
                "X_tr_sel",
                "y_tr_sel",
                "X_val_sel",
                "y_val_sel",
                "cat_cols",
                "params:tune_trials",
                "params:seed",
                "params:tune_verbose",
            ],
            outputs=["model_sel", "best_params", "best_valid_auc", "tuning_trials"],
        ),
    ]

    full_nodes = train_nodes + [
        Node(save_model_pickle, inputs=["model_sel", "params:model_pickle_path"], outputs="model_pickle_path"),
        Node(
            score_test,
            inputs=["labeled_customers", "X_test_sel", "y_test", "model_sel", "cat_cols"],
            outputs="scored_test",
        ),
        Node(
            metrics_table_from_scored_df,
            inputs=["scored_test", "params:report_thresholds", "params:final_label"],
            outputs="final_metrics_table",
        ),
        Node(
            best_f1_metrics_from_scored_df,
            inputs=["scored_test", "params:f1_threshold_grid_points", "params:final_label"],
            outputs="final_best_f1_metrics",
        ),
        Node(add_deciles, inputs="scored_test", outputs="scored_test_with_deciles"),
        Node(decile_table, inputs="scored_test_with_deciles", outputs="decile_table"),
        Node(
            generate_leads,
            inputs=["scored_test_with_deciles", "params:min_decile_for_leads"],
            outputs="leads",
        ),
        Node(
            add_rowwise_shap_explanations,
            inputs=["leads", "model_sel", "selected_features", "cat_cols", "params:explain_top_n"],
            outputs="leads_explained",
        ),
        Node(
            save_shap_importance_plot,
            inputs=["model_sel", "X_test_sel", "selected_features", "cat_cols", "params:shap_importance_png_path"],
            outputs="shap_importance_png_path",
        ),
        Node(
            save_leads,
            inputs=["leads_explained", "selected_features", "params:leads_csv_path"],
            outputs="leads_csv_path",
        ),
        Node(
            concat_metrics,
            inputs=[
                "baseline_metrics_table",
                "baseline_best_f1_metrics",
                "final_metrics_table",
                "final_best_f1_metrics",
            ],
            outputs="performance_metrics",
        ),
        Node(
            save_metrics_csv,
            inputs=["performance_metrics", "params:metrics_csv_path"],
            outputs="metrics_csv_path",
        ),
    ]

    return {
        "full": Pipeline(name="cc_propensity_full", nodes=full_nodes),
        "baseline": Pipeline(name="cc_propensity_baseline", nodes=baseline_nodes),
        "train": Pipeline(name="cc_propensity_train", nodes=train_nodes),
    }


def _print_outputs(catalog: Catalog) -> None:
    print("\n=== Outputs ===")

    if "baseline_auc" in catalog:
        print("Baseline (scaled LogisticRegression) AUC on validation:", round(float(catalog["baseline_auc"]), 4))

    if "best_params" in catalog:
        print("Best tuning params (selected features):", catalog["best_params"])
    if "best_valid_auc" in catalog:
        print("Best validation AUC (selected features):", round(float(catalog["best_valid_auc"]), 4))

    if "tuning_trials" in catalog:
        print("\nTuning trials (top 5 by validation AUC):")
        cols = [c for c in catalog["tuning_trials"].columns if c in {"trial", "valid_auc"} or c.startswith("param_")]
        view = catalog["tuning_trials"][cols].head(5).copy()
        if "valid_auc" in view.columns:
            view["valid_auc"] = view["valid_auc"].map(lambda x: round(float(x), 5))
        print(view.to_string(index=False))
    if "scored_test" in catalog:
        print("Test AUC (selected features):", round(float(catalog["scored_test"]["auc_on_test"].iloc[0]), 4))
    if "leads_csv_path" in catalog:
        print("Leads CSV:", catalog["leads_csv_path"])
    if "model_pickle_path" in catalog:
        print("Model pickle:", catalog["model_pickle_path"])
    if "shap_importance_png_path" in catalog:
        print("SHAP importance plot:", catalog["shap_importance_png_path"])
    if "metrics_csv_path" in catalog:
        print("Performance metrics CSV:", catalog["metrics_csv_path"])

    if "feature_importance" in catalog:
        print("\nTop feature importance:")
        print(catalog["feature_importance"].head(12).to_string(index=False))
    if "decile_table" in catalog:
        print("\nDecile table:")
        print(catalog["decile_table"].to_string(index=False))
    if "leads" in catalog:
        print("\nSample leads:")
        print(catalog["leads"].head(10).to_string(index=False))

    if "run:timings" in catalog:
        print("\n=== Timing snapshot (slowest first) ===")
        view = catalog["run:timings"][["node", "seconds", "pct_total"]].copy()
        view["seconds"] = view["seconds"].map(lambda s: round(float(s), 4))
        view["pct_total"] = view["pct_total"].map(lambda p: round(float(p), 2))
        print(view.to_string(index=False))
        if "run:total_seconds" in catalog:
            print("Total pipeline time (s):", round(float(catalog["run:total_seconds"]), 4))


def main(pipeline_name: str = "full", verbose: bool = True) -> Catalog:
    pipelines = build_pipelines()
    if pipeline_name not in pipelines:
        raise SystemExit(
            "Unknown pipeline '%s'. Use --list-pipelines to see options." % pipeline_name
        )

    catalog = build_default_catalog()
    catalog = run_pipeline(pipelines[pipeline_name], catalog=catalog, verbose=verbose)
    _print_outputs(catalog)
    return catalog


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    pipelines = build_pipelines()
    parser = argparse.ArgumentParser(description="Kedro-style propensity pipeline runner")
    parser.add_argument(
        "--pipeline",
        default="full",
        choices=sorted(pipelines.keys()),
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
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    args = _parse_args()
    if args.list_pipelines:
        print("Available pipelines:")
        for name, pl in build_pipelines().items():
            print("- %s (%s, %d nodes)" % (name, pl.name, len(pl.nodes)))
        raise SystemExit(0)
    main(pipeline_name=str(args.pipeline), verbose=not bool(args.quiet))
