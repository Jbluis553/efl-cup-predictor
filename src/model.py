import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from . import config

logger = logging.getLogger(__name__)

# Closing prices can only be observed after a match; never use them as predictors.
FEATURES_EXCLUDE = {
    "match_id", "season_id", "home_team", "away_team", "kickoff",
    "score_home", "score_away", "result_home", "result_draw", "result_away",
    "total_goals", "over_2_5", "btts",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column for column in df.select_dtypes(include=[np.number]).columns
        if column not in FEATURES_EXCLUDE and not column.endswith("_last_seen")
        and not column.endswith("_ah_line")
    ]


def _classifier() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )


def _positive_probability(model, X: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(1)]


def _oof_probabilities(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(y) < 6:
        raise ValueError("At least 6 completed matches are required for time-series training")
    splitter = TimeSeriesSplit(n_splits=min(5, len(y) - 1))
    predictions = np.full(len(y), np.nan)
    for train_idx, valid_idx in splitter.split(X):
        model = _classifier()
        model.fit(X[train_idx], y[train_idx])
        predictions[valid_idx] = _positive_probability(model, X[valid_idx])
    return predictions


def _binary_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    valid = np.isfinite(probabilities)
    y_valid = y[valid]
    p_valid = probabilities[valid]
    if not len(y_valid):
        raise ValueError("No out-of-fold predictions could be generated")
    hard = (p_valid >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_valid, hard),
        "brier": brier_score_loss(y_valid, p_valid),
        "f1": f1_score(y_valid, hard, zero_division=0),
    }
    metrics["log_loss"] = log_loss(y_valid, p_valid, labels=[0, 1])
    return metrics


def _prepare(df: pd.DataFrame, target: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    if target not in df:
        raise ValueError(f"Missing target column: {target}")
    features = get_feature_columns(df)
    if not features:
        raise ValueError("No numeric pre-match features are available for training")
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
    y = pd.to_numeric(df[target], errors="coerce").to_numpy()
    valid = np.isfinite(y)
    X, y = X[valid], y[valid].astype(int)
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) < 2:
        raise ValueError(f"Target {target} must contain both binary classes")
    return features, X, y


def train_1x2(df: pd.DataFrame) -> dict:
    """Train one-vs-rest 1X2 models with chronological validation."""
    features = get_feature_columns(df)
    if not features:
        raise ValueError("No numeric pre-match features are available for training")
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
    results = {}
    final_models = {}
    for outcome in ("home", "draw", "away"):
        target = f"result_{outcome}"
        y = pd.to_numeric(df[target], errors="coerce").to_numpy()
        valid = np.isfinite(y)
        x_target, y_target = X[valid], y[valid].astype(int)
        if len(np.unique(y_target)) < 2:
            raise ValueError(f"Target {target} must contain both binary classes")
        oof = _oof_probabilities(x_target, y_target)
        results[f"{outcome}_metrics"] = _binary_metrics(y_target, oof)
        final_model = _classifier()
        final_model.fit(x_target, y_target)
        final_models[outcome] = final_model
        logger.info("%s model metrics: %s", outcome, results[f"{outcome}_metrics"])
    results["final_models"] = final_models
    results["features"] = features
    return results


def train_over_under(df: pd.DataFrame) -> dict:
    """Train an Over/Under 2.5 goals model."""
    features, X, y = _prepare(df, "over_2_5")
    oof = _oof_probabilities(X, y)
    model = _classifier()
    model.fit(X, y)
    return {
        "model": model,
        "features": features,
        "oof_accuracy": _binary_metrics(y, oof)["accuracy"],
        "oof_log_loss": _binary_metrics(y, oof)["log_loss"],
    }


def train_btts(df: pd.DataFrame) -> dict:
    """Train a both-teams-to-score model."""
    features, X, y = _prepare(df, "btts")
    oof = _oof_probabilities(X, y)
    model = _classifier()
    model.fit(X, y)
    metrics = _binary_metrics(y, oof)
    return {
        "model": model,
        "features": features,
        "oof_accuracy": metrics["accuracy"],
        "oof_log_loss": metrics["log_loss"],
    }


def save_models(models: dict, output_dir: str = config.MODEL_DIR) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib")
    logger.info("Models saved to %s/", output)


def predict_match(model_dict: dict, features: pd.DataFrame) -> dict:
    """Return normalized 1X2 probabilities for one or more matches."""
    selected = model_dict["features"]
    X = features.reindex(columns=selected, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    probabilities = np.column_stack([
        _positive_probability(model_dict["final_models"][outcome], X)
        for outcome in ("home", "draw", "away")
    ])
    totals = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(probabilities, totals, out=np.full_like(probabilities, 1 / 3), where=totals > 0)
    return [
        {
            "P(home)": round(float(row[0]), 4),
            "P(draw)": round(float(row[1]), 4),
            "P(away)": round(float(row[2]), 4),
            "implied_odds_home": round(1 / row[0], 2) if row[0] > 0 else None,
            "implied_odds_draw": round(1 / row[1], 2) if row[1] > 0 else None,
            "implied_odds_away": round(1 / row[2], 2) if row[2] > 0 else None,
        }
        for row in probabilities
    ]
