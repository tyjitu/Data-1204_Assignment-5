from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 42


@dataclass
class ThresholdResult:
    threshold: float
    score: float


def f2_score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return fbeta_score(y_true, y_pred, beta=2, zero_division=0)


def find_best_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    thresholds: np.ndarray | None = None,
    optimize_metric: str = "f2",
) -> ThresholdResult:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        if optimize_metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = f2_score(y_true, y_pred)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    return ThresholdResult(threshold=best_threshold, score=best_score)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray | None = None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "pr_auc": None,
        "roc_auc": None,
        "f2": f2_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": None,
    }

    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["brier"] = brier_score_loss(y_true, y_proba)

    return metrics


def evaluate_experiment(
    exp_id: int,
    model_name: str,
    feature_set: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    main_settings: str,
    notes: str,
    threshold: float = 0.5,
    threshold_metric: str = "f2",
    selected_finalist: str = "No",
) -> dict[str, Any]:
    estimator.fit(X_train, y_train)

    train_proba = estimator.predict_proba(X_train)[:, 1]
    val_proba = estimator.predict_proba(X_val)[:, 1]

    train_pred = (train_proba >= threshold).astype(int)
    val_pred = (val_proba >= threshold).astype(int)

    train_metrics = compute_metrics(y_train, train_pred, train_proba)
    val_metrics = compute_metrics(y_val, val_pred, val_proba)
    threshold_score = val_metrics["f1"] if threshold_metric == "f1" else val_metrics["f2"]

    return {
        "exp_id": exp_id,
        "model": model_name,
        "feature_set": feature_set,
        "main_settings": main_settings,
        "train_pr_auc": train_metrics["pr_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "overfit_gap": train_metrics["pr_auc"] - val_metrics["pr_auc"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_brier": val_metrics["brier"],
        "threshold": threshold,
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1_or_f2": threshold_score,
        "selected_finalist": selected_finalist,
        "notes": notes,
    }


def split_balance(y: pd.Series) -> pd.Series:
    counts = y.value_counts().sort_index()
    percents = y.value_counts(normalize=True).sort_index().mul(100)
    return pd.DataFrame({"count": counts, "percent": percents.round(2)})


def make_scale_pos_weight(y_train: pd.Series) -> float:
    class_counts = y_train.value_counts()
    negative_count = class_counts.get(0, 0)
    positive_count = class_counts.get(1, 1)
    return negative_count / positive_count


def select_top_features(
    model: Any,
    feature_names: list[str],
    top_n: int = 25,
) -> list[str]:
    importance = pd.Series(model.feature_importances_, index=feature_names)
    return importance.sort_values(ascending=False).head(top_n).index.tolist()
