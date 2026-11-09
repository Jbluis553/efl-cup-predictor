import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score,
    classification_report, brier_score_loss
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from . import config

logger = logging.getLogger(__name__)


FEATURES_EXCLUDE = [
    "match_id", "season_id", "home_team", "away_team", "kickoff",
    "score_home", "score_away", "result_home", "result_draw",
    "result_away", "total_goals", "over_2_5", "btts"
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURES_EXCLUDE]


def train_1x2(df: pd.DataFrame, target: str = "result_home") -> dict:
    """
    Entrena modelos para predicción 1X2 (home win, draw, away win).
    Usa TimeSeriesSplit para respetar el orden temporal.
    """
    features = get_feature_columns(df)
    X = df[features].fillna(0).values
    y_home = df["result_home"].values
    y_draw = df["result_draw"].values
    y_away = df["result_away"].values

    tscv = TimeSeriesSplit(n_splits=5)

    results = {}
    for name, model in [
        ("lightgbm_home", LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
        )),
        ("xgboost_home", XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            eval_metric="logloss", use_label_encoder=False
        )),
    ]:
        oof_preds = np.zeros(len(y_home))
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            model.fit(X[train_idx], y_home[train_idx])
            oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_home, (oof_preds > 0.5).astype(int)),
            "log_loss": log_loss(y_home, oof_preds, labels=[0, 1]),
            "brier": brier_score_loss(y_home, oof_preds),
            "f1": f1_score(y_home, (oof_preds > 0.5).astype(int)),
        }
        logger.info(f"  {name}: {results[name]}")

    # Retrain on full data for final model
    final_lgbm = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
    )
    final_lgbm.fit(X, y_home)

    # Draw model
    final_draw = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
    )
    final_draw.fit(X, y_draw)

    # Away model
    final_away = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
    )
    final_away.fit(X, y_away)

    results["final_models"] = {
        "home": final_lgbm,
        "draw": final_draw,
        "away": final_away,
    }
    results["features"] = features

    return results


def train_over_under(df: pd.DataFrame) -> dict:
    """Predicción Over/Under 2.5."""
    features = get_feature_columns(df)
    X = df[features].fillna(0).values
    y = df["over_2_5"].values

    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(y))

    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    model.fit(X, y)

    return {
        "model": model,
        "features": features,
        "oof_accuracy": accuracy_score(y, (oof_preds > 0.5).astype(int)),
        "oof_log_loss": log_loss(y, oof_preds, labels=[0, 1]),
    }


def train_btts(df: pd.DataFrame) -> dict:
    """Predicción BTTS (Both Teams To Score)."""
    features = get_feature_columns(df)
    X = df[features].fillna(0).values
    y = df["btts"].values

    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
    )
    model.fit(X, y)

    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    return {
        "model": model,
        "features": features,
        "oof_accuracy": accuracy_score(y, (oof_preds > 0.5).astype(int)),
        "oof_log_loss": log_loss(y, oof_preds, labels=[0, 1]),
    }


def save_models(models: dict, output_dir: str = config.MODEL_DIR) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        if name == "features":
            continue
        if isinstance(model, dict):
            for sub_name, sub_model in model.items():
                joblib.dump(sub_model, out / f"{name}_{sub_name}.joblib")
        else:
            joblib.dump(model, out / f"{name}.joblib")

    logger.info(f"Models saved to {out}/")


def predict_match(model_dict: dict, features: pd.DataFrame) -> dict:
    """Predicción para un partido futuro."""
    X = features[config.MODEL_DIR and model_dict["features"]].fillna(0).values

    p_home = model_dict["final_models"]["home"].predict_proba(X)[:, 1][0]
    p_draw = model_dict["final_models"]["draw"].predict_proba(X)[:, 1][0]
    p_away = model_dict["final_models"]["away"].predict_proba(X)[:, 1][0]

    # Normalizar
    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home/total, p_draw/total, p_away/total

    return {
        "P(home)": round(p_home, 4),
        "P(draw)": round(p_draw, 4),
        "P(away)": round(p_away, 4),
        "implied_odds_home": round(1/p_home, 2) if p_home > 0 else None,
        "implied_odds_draw": round(1/p_draw, 2) if p_draw > 0 else None,
        "implied_odds_away": round(1/p_away, 2) if p_away > 0 else None,
    }   
