import argparse
import json
import logging
from pathlib import Path

import joblib

from . import config
from .backtest import backtest_1x2, backtest_over_under
from .data_fetcher import fetch_all_data, get_seasons
from .feature_engineering import build_dataset
from .model import save_models, train_1x2, train_btts, train_over_under

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="EFL Cup ML Predictor")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-only", action="store_true", help="Solo descargar datos")
    mode.add_argument("--train-only", action="store_true", help="Solo entrenar con datos existentes")
    mode.add_argument("--backtest-only", action="store_true", help="Solo ejecutar el backtest")
    args = parser.parse_args()

    if not args.train_only and not args.backtest_only:
        logger.info("=== STEP 1: Fetching data from TheStatsAPI ===")
        seasons = get_seasons()
        seasons = seasons[-config.SEASONS:]
        fetch_all_data(seasons)

    if args.fetch_only:
        return

    logger.info("=== STEP 2: Building feature matrix ===")
    df = build_dataset()
    logger.info("  Shape: %s", df.shape)
    if df["kickoff"].notna().any():
        logger.info("  Date range: %s → %s", df["kickoff"].min(), df["kickoff"].max())

    if args.backtest_only:
        models_dir = Path(config.MODEL_DIR)
        with (models_dir / "features.json").open(encoding="utf-8") as handle:
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

        models_dir = Path(config.MODEL_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        with (models_dir / "features.json").open("w", encoding="utf-8") as handle:
            json.dump(results_1x2["features"], handle, indent=2)
        model_dict = {
            "features": results_1x2["features"],
            "final_models": results_1x2["final_models"],
        }
        over_under_result = results_ou

    if not args.train_only:
        logger.info("=== STEP 4: Backtesting ===")
        bt_1x2 = backtest_1x2(df, model_dict, threshold=0.03)
        if not bt_1x2.empty:
            logger.info("  1X2 Backtest: %s", bt_1x2.attrs.get("summary", {}))
            bt_1x2.to_csv(Path(config.MODEL_DIR) / "backtest_1x2.csv", index=False)

        bt_ou = backtest_over_under(df, over_under_result, threshold=0.03)
        if not bt_ou.empty:
            logger.info("  O/U Backtest: %s", bt_ou.attrs.get("summary", {}))
            bt_ou.to_csv(Path(config.MODEL_DIR) / "backtest_over_under.csv", index=False)

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
