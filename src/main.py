import logging
import argparse
import json
from pathlib import Path

from . import config
from .data_fetcher import get_seasons, fetch_all_data
from .feature_engineering import build_dataset
from .model import train_1x2, train_over_under, train_btts, save_models
from .backtest import backtest_1x2, backtest_over_under

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="EFL Cup ML Predictor")
    parser.add_argument("--fetch-only", action="store_true", help="Solo descargar datos")
    parser.add_argument("--train-only", action="store_true", help="Solo entrenar (usa datos existentes)")
    parser.add_argument("--backtest-only", action="store_true", help="Solo backtest")
    args = parser.parse_args()

    # 1. Fetch data
    if not args.train_only and not args.backtest_only:
        logger.info("=== STEP 1: Fetching data from TheStatsAPI ===")
        seasons = get_seasons()
        # Tomar las últimas N temporadas
        seasons = seasons[-config.SEASONS:]
        fetch_all_data(seasons)

    # 2. Build features
    logger.info("=== STEP 2: Building feature matrix ===")
    df = build_dataset()
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Date range: {df['kickoff'].min()} → {df['kickoff'].max()}")

    if args.fetch_only:
        return

    # 3. Train models
    if not args.backtest_only:
        logger.info("=== STEP 3: Training models ===")

        logger.info("  --- 1X2 Model ---")
        results_1x2 = train_1x2(df)
        for name, metrics in results_1x2.items():
            if name != "final_models" and name != "features":
                logger.info(f"  {name}: {metrics}")

        logger.info("  --- Over/Under 2.5 ---")
        results_ou = train_over_under(df)
        logger.info(f"  OOF Accuracy: {results_ou['oof_accuracy']:.4f}, LogLoss: {results_ou['oof_log_loss']:.4f}")

        logger.info("  --- BTTS ---")
        results_btts = train_btts(df)
        logger.info(f"  OOF Accuracy: {results_btts['oof_accuracy']:.4f}, LogLoss: {results_btts['oof_log_loss']:.4f}")

        # Save
        all_models = {
            "1x2_home": results_1x2["final_models"]["home"],
            "1x2_draw": results_1x2["final_models"]["draw"],
            "1x2_away": results_1x2["final_models"]["away"],
            "over_under": results_ou["model"],
            "btts": results_btts["model"],
        }
        save_models(all_models)

        # Save feature list
        Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        with open(f"{config.MODEL_DIR}/features.json", "w") as f:
            json.dump(results_1x2["features"], f, indent=2)

    # 4. Backtest
    if not args.train_only:
        logger.info("=== STEP 4: Backtesting ===")

        # Reconstruir model dict para backtest
        import joblib
        model_dict = {
            "features": results_1x2["features"] if not args.train_only else json.load(open(f"{config.MODEL_DIR}/features.json")),
            "final_models": {
                "home": joblib.load(f"{config.MODEL_DIR}/1x2_home.joblib"),
                "draw": joblib.load(f"{config.MODEL_DIR}/1x2_draw.joblib"),
                "away": joblib.load(f"{config.MODEL_DIR}/1x2_away.joblib"),
            }
        }

        bt_1x2 = backtest_1x2(df, model_dict, threshold=0.03)
        if not bt_1x2.empty:
            logger.info(f"  1X2 Backtest: {bt_1x2.attrs['summary']}")
            bt_1x2.to_csv(f"{config.MODEL_DIR}/backtest_1x2.csv", index=False)

        bt_ou = backtest_over_under(df, results_ou, threshold=0.03)
        if not bt_ou.empty:
            logger.info(f"  O/U Backtest: {bt_ou.attrs['summary']}")
            bt_ou.to_csv(f"{config.MODEL_DIR}/backtest_over_under.csv", index=False)

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()   
