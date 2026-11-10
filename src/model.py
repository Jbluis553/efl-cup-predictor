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


def _oof_probabilities(X: np.ndarray, y: np.ndarray, gap: int = 0) -> np.ndarray:
    """Chronological out-of-fold probabilities (honest validation)."""
    if len(y) < 6:
        raise ValueError("At least 6 completed matches are required for time-series training")
    splitter = TimeSeriesSplit(n_splits=min(5, len(y) - 1), gap=gap)
    predictions = np.full(len(y), np.nan)
    for train_idx, valid_idx in splitter.split(X):
        # Añadido: guarda contra folds cuyo train tiene una sola clase.
        if len(np.unique(y[train_idx])) < 2:
            logger.warning("Skipping OOF fold with a single class in the training window")
            continue
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


def train_binary_target(df: pd.DataFrame, target: str,
                        features: list[str] | None = None) -> tuple:
    """Train one binary model and return (model, features, oof_aligned, metrics).

    oof_aligned has one probability per df row (NaN where the target was
    invalid), which lets the backtest merge on match_id without leakage.
    """
    if target not in df:
        raise ValueError(f"Missing target column: {target}")
    features = features if features is not None else get_feature_columns(df)
    if not features:
        raise ValueError("No numeric pre-match features are available for training")
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
    y = pd.to_numeric(df[target], errors="coerce").to_numpy()
    valid = np.isfinite(y)
    x_target, y_target = X[valid], y[valid].astype(int)
    if not set(np.unique(y_target)).issubset({0, 1}) or len(np.unique(y_target)) < 2:
        raise ValueError(f"Target {target} must contain both binary classes")
    oof_valid = _oof_probabilities(x_target, y_target)
    model = _classifier()
    model.fit(x_target, y_target)
    oof = np.full(len(df), np.nan)
    oof[valid] = oof_valid
    return model, features, oof, _binary_metrics(y_target, oof_valid)


def train_1x2(df: pd.DataFrame) -> dict:
    """Train one-vs-rest 1X2 models with chronological validation."""
    features = get_feature_columns(df)
    if not features:
        raise ValueError("No numeric pre-match features are available for training")
    results = {"features": features, "final_models": {}}
    match_ids = df["match_id"] if "match_id" in df else np.arange(len(df))
    kickoffs = df["kickoff"] if "kickoff" in df else [None] * len(df)
    oof = pd.DataFrame({"match_id": match_ids, "kickoff": kickoffs})
    for outcome in ("home", "draw", "away"):
        model, _, oof_col, metrics = train_binary_target(df, f"result_{outcome}", features)
        results["final_models"][outcome] = model
        results[f"{outcome}_metrics"] = metrics
        oof[f"p_{outcome}"] = oof_col
        logger.info("%s model metrics: %s", outcome, metrics)
    results["oof"] = oof
    return results


def train_over_under(df: pd.DataFrame) -> dict:
    """Train an Over/Under 2.5 goals model."""
    model, features, oof, metrics = train_binary_target(df, "over_2_5")
    return {
        "model": model,
        "features": features,
        "oof": oof,
        "oof_accuracy": metrics["accuracy"],
        "oof_log_loss": metrics["log_loss"],
    }


def train_btts(df: pd.DataFrame) -> dict:
    """Train a both-teams-to-score model."""
    model, features, oof, metrics = train_binary_target(df, "btts")
    return {
        "model": model,
        "features": features,
        "oof": oof,
        "oof_accuracy": metrics["accuracy"],
        "oof_log_loss": metrics["log_loss"],
    }


def train_all(df: pd.DataFrame) -> dict:
    """Train every model and assemble a single OOF prediction table."""
    results_1x2 = train_1x2(df)
    results_ou = train_over_under(df)
    results_btts = train_btts(df)
    oof = results_1x2["oof"].copy()
    oof["p_over"] = results_ou["oof"]
    oof["p_btts"] = results_btts["oof"]
    all_models = {
        f"1x2_{outcome}": model
        for outcome, model in results_1x2["final_models"].items()
    }
    all_models.update({"over_under": results_ou["model"], "btts": results_btts["model"]})
    return {
        "models": all_models,
        "features": results_1x2["features"],
        "oof": oof,
        "results_1x2": results_1x2,
        "results_over_under": results_ou,
        "results_btts": results_btts,
    }


def save_models(models: dict, output_dir: str = config.MODEL_DIR) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib")
    logger.info("Models saved to %s/", output)


def save_oof_predictions(oof: pd.DataFrame, output_dir: str = config.MODEL_DIR) -> Path:
    """Persist chronological OOF probabilities used by the honest backtest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / config.OOF_FILE
    oof.to_csv(path, index=False)
    logger.info("OOF predictions saved to %s", path)
    return path


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
