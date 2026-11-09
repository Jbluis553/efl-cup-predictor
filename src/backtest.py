import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _valid_odds(value) -> bool:
    try:
        return np.isfinite(float(value)) and float(value) > 1
    except (TypeError, ValueError):
        return False


def backtest_1x2(df: pd.DataFrame, model_dict: dict, threshold: float = 0.05) -> pd.DataFrame:
    """Backtest 1X2 predictions against closing Pinnacle prices."""
    features = model_dict["features"]
    X = df.reindex(columns=features, fill_value=0).fillna(0).values
    outcomes = ("home", "draw", "away")
    models = model_dict["final_models"]

    # Predict each outcome for the whole dataset at once. Calling LightGBM
    # once per match creates thousands of tiny predictions and can make CI
    # backtests appear to hang on larger historical datasets.
    probabilities = []
    for outcome in outcomes:
        model = models[outcome]
        classes = list(model.classes_)
        if 1 in classes:
            probabilities.append(model.predict_proba(X)[:, classes.index(1)])
        else:
            probabilities.append(np.zeros(len(df)))
    probabilities = np.column_stack(probabilities)
    totals = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities,
        totals,
        out=np.zeros_like(probabilities),
        where=np.isfinite(totals) & (totals > 0),
    )

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        actual = "home" if row["result_home"] == 1 else "draw" if row["result_draw"] == 1 else "away"
        best_bet = None
        best_edge = 0.0
        for outcome, probability in zip(outcomes, probabilities[i]):
            odds = row.get(f"odds_pinnacle_{outcome}_last_seen", np.nan)
            if not _valid_odds(odds):
                continue
            edge = probability - 1 / float(odds)
            if edge > best_edge:
                best_edge = edge
                best_bet = {
                    "match_id": row.get("match_id"),
                    "outcome": outcome,
                    "model_prob": probability,
                    "closing_odds": float(odds),
                    "edge": edge,
                    "actual_result": actual,
                    "win": outcome == actual,
                }
        if best_bet and best_edge > threshold:
            results.append(best_bet)

    bt = pd.DataFrame(results)
    if bt.empty:
        logger.warning("No 1X2 bets passed the threshold filter")
        return bt

    bt["pnl"] = np.where(bt["win"], bt["closing_odds"] - 1, -1)
    bt["cumulative_pnl"] = bt["pnl"].cumsum()
    bt.attrs["summary"] = {
        "total_bets": len(bt),
        "win_rate": bt["win"].mean(),
        "avg_edge": bt["edge"].mean(),
        "total_pnl": bt["pnl"].sum(),
        "max_drawdown": (bt["cumulative_pnl"].cummax() - bt["cumulative_pnl"]).max(),
        "roi": bt["pnl"].sum() / len(bt),
    }
    logger.info("1X2 backtest: %s", bt.attrs["summary"])
    return bt


def backtest_over_under(df: pd.DataFrame, model_result: dict, threshold: float = 0.05) -> pd.DataFrame:
    """Backtest Over/Under 2.5 using Pinnacle closing odds."""
    features = model_result["features"]
    X = df.reindex(columns=features, fill_value=0).fillna(0).values
    model = model_result["model"]
    classes = list(model.classes_)
    probs = model.predict_proba(X)[:, classes.index(1)] if 1 in classes else np.zeros(len(df))

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        odds = row.get("odds_pinnacle_ou_over_last_seen", np.nan)
        if not _valid_odds(odds):
            continue
        edge = probs[i] - (1 / float(odds))
        if edge > threshold:
            results.append({
                "match_id": row.get("match_id"),
                "model_prob_over": probs[i],
                "closing_odds": float(odds),
                "edge": edge,
                "win": row["over_2_5"] == 1,
            })

    bt = pd.DataFrame(results)
    if bt.empty:
        logger.warning("No Over/Under bets passed the threshold filter")
        return bt
    bt["pnl"] = np.where(bt["win"], bt["closing_odds"] - 1, -1)
    bt["cumulative_pnl"] = bt["pnl"].cumsum()
    bt.attrs["summary"] = {
        "total_bets": len(bt),
        "win_rate": bt["win"].mean(),
        "total_pnl": bt["pnl"].sum(),
        "roi": bt["pnl"].sum() / len(bt),
    }
    return bt
