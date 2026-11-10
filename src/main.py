import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .backtest import backtest_1x2, backtest_btts, backtest_over_under
from .data_fetcher import fetch_all_data, get_seasons
from .feature_engineering import build_dataset
from .model import (
    save_models,
    save_oof_predictions,
    train_1x2,
    train_btts,
    train_over_under,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _load_oof_probabilities(df: pd.DataFrame, models_dir: Path):
    """Return (p_1x2, p_over, p_btts) aligned to df rows, or Nones for in-sample."""
    oof_path = models_dir / config.OOF_FILE
    if not oof_path.exists():
        logger.warning(
            "%s not found; backtest will be IN-SAMPLE (optimistic). "
            "Retrain to generate out-of-fold predictions.", config.OOF_FILE,
        )
        return None, None, None
    oof = pd.read_csv(oof_path)
    keep = ["match_id", "p_home", "p_draw", "p_away", "p_over", "p_btts"]
    oof = oof[[c for c in keep if c in oof.columns]].copy()
    oof["match_id"] = oof["match_id"].astype(str)
    merged = df.assign(match_id=df["match_id"].astype(str)).merge(oof, on="match_id", how="left")
    p_1x2 = merged[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    p_over = merged["p_over"].to_numpy(dtype=float)
    p_btts = merged["p_btts"].to_numpy(dtype=float)
    n_missing = int(pd.isna(merged["p_home"]).sum())
    if n_missing:
        logger.warning("%s matches without OOF predictions (excluded from the backtest)", n_missing)
    return p_1x2, p_over, p_btts


def main():
    parser = argparse.ArgumentParser(description="EFL Cup ML Predictor")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-only", action="store_true", help="Solo descargar datos")
    mode.add_argument("--train-only", action="store_true", help="Solo entrenar con datos existentes")
    mode.add_argument("--backtest-only", action="store_true", help="Solo ejecutar el backtest")
    parser.add_argument("--force", action="store_true", help="Volver a descargar cuotas aunque existan")
    args = parser.parse_args()

    if not args.train_only and not args.backtest_only:
        logger.info("=== STEP 1: Fetching data from TheStatsAPI ===")
        seasons = get_seasons()
        seasons = seasons[-config.SEASONS:]
        fetch_all_data(seasons, force=args.force)

    if args.fetch_only:
        return

    logger.info("=== STEP 2: Building feature matrix ===")
    df = build_dataset()
    logger.info("  Shape: %s", df.shape)
    if df["kickoff"].notna().any():
        logger.info("  Date range: %s → %s", df["kickoff"].min(), df["kickoff"].max())

    models_dir = Path(config.MODEL_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.backtest_only:
        with (models_dir / config.FEATURES_FILE).open(encoding="utf-8") as handle:
            features = json.load(handle)
        model_dict = {
            "features": features,
            "final_models": {
                outcome: joblib.load(models_dir / f"1x2_{outcome}.joblib")
                for outcome in ("home", "draw", "away")
            },
        }
        over_under_result = {
            "features": features,
            "model": joblib.load(models_dir / "over_under.joblib"),
        }
        btts_result = {
            "features": features,
            "model": joblib.load(models_dir / "btts.joblib"),
        }
    else:
        logger.info("=== STEP 3: Training models ===")
        results_1x2 = train_1x2(df)
        results_ou = train_over_under(df)
        results_btts = train_btts(df)

        all_models = {
            f"1x2_{outcome}": model
            for outcome, model in results_1x2["final_models"].items()
        }
        all_models.update({"over_under": results_ou["model"], "btts": results_btts["model"]})
        save_models(all_models)

        with (models_dir / config.FEATURES_FILE).open("w", encoding="utf-8") as handle:
            json.dump(results_1x2["features"], handle, indent=2)

        oof = results_1x2["oof"].copy()
        oof["p_over"] = results_ou["oof"]
        oof["p_btts"] = results_btts["oof"]
        save_oof_predictions(oof)

        model_dict = {
            "features": results_1x2["features"],
            "final_models": results_1x2["final_models"],
        }
        over_under_result = results_ou
        btts_result = {"features": results_1x2["features"], "model": results_btts["model"]}

    if not args.train_only:
        logger.info("=== STEP 4: Backtesting (out-of-fold) ===")
        p_1x2, p_over, p_btts = _load_oof_probabilities(df, models_dir)

        bt_1x2 = backtest_1x2(df, model_dict, threshold=config.EDGE_THRESHOLD, probabilities=p_1x2)
        if not bt_1x2.empty:
            logger.info("  1X2 Backtest: %s", bt_1x2.attrs.get("summary", {}))
            bt_1x2.to_csv(models_dir / "backtest_1x2.csv", index=False)

        bt_ou = backtest_over_under(df, over_under_result, threshold=config.EDGE_THRESHOLD, probabilities=p_over)
        if not bt_ou.empty:
            logger.info("  O/U Backtest: %s", bt_ou.attrs.get("summary", {}))
            bt_ou.to_csv(models_dir / "backtest_over_under.csv", index=False)

        bt_btts = backtest_btts(df, btts_result, threshold=config.EDGE_THRESHOLD, probabilities=p_btts)
        if not bt_btts.empty:
            logger.info("  BTTS Backtest: %s", bt_btts.attrs.get("summary", {}))
            bt_btts.to_csv(models_dir / "backtest_btts.csv", index=False)

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
