import numpy as np
import pandas as pd
import logging

from . import config

logger = logging.getLogger(__name__)


def compute_clv(model_probs: np.ndarray, closing_odds: np.ndarray) -> float:
    """
    Closing Line Value: compara la probabilidad del modelo con la cuota de cierre.
    CLV > 0 significa que el modelo encontró valor.
    """
    # Convert closing odds to implied probability
    implied_prob = 1 / closing_odds
    # Remove overround (normalizar)
    # Asumimos que model_probs ya está normalizado
    clv = model_probs - implied_prob
    return np.mean(clv)


def backtest_1x2(df: pd.DataFrame, model_dict: dict,
                 threshold: float = 0.05) -> pd.DataFrame:
    """
    Backtest de 1X2 con valor.
    Solo apuesta cuando el modelo tiene probabilidad > threshold por encima de la implied.
    """
    features = model_dict["features"]
    X = df[features].fillna(0).values

    results = []
    for i in range(len(df)):
        p_home = model_dict["final_models"]["home"].predict_proba(X[i:i+1])[:, 1][0]
        p_draw = model_dict["final_models"]["draw"].predict_proba(X[i:i+1])[:, 1][0]
        p_away = model_dict["final_models"]["away"].predict_proba(X[i:i+1])[:, 1][0]

        total = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home/total, p_draw/total, p_away/total

        # Buscar cuotas de cierre (Pinnacle last_seen)
        odds_row = df.iloc[i]
        best_bet = None
        best_edge = 0

        for outcome, prob in [("home", p_home), ("draw", p_draw), ("away", p_away)]:
            odds_col = f"odds_pinnacle_{outcome}_last_seen"
            if odds_col in df.columns and not np.isnan(odds_row.get(odds_col, np.nan)):
                closing_odds = odds_row[odds_col]
                edge = prob - (1 / closing_odds)
                if edge > best_edge:
                    best_edge = edge
                    best_bet = {
                        "match_id": df.iloc[i]["match_id"],
                        "outcome": outcome,
                        "model_prob": prob,
                        "closing_odds": closing_odds,
                        "edge": edge,
                        "actual_result": (
                            "home" if df.iloc[i]["result_home"] == 1 else
                            "draw" if df.iloc[i]["result_draw"] == 1 else
                            "away"
                        ),
                        "win": (outcome == (
                            "home" if df.iloc[i]["result_home"] == 1 else
                            "draw" if df.iloc[i]["result_draw"] == 1 else
                            "away"
                        )),
                    }

        if best_bet and best_bet["edge"] > threshold:
            results.append(best_bet)

    bt = pd.DataFrame(results)
    if bt.empty:
        logger.warning("No bets passed the threshold filter")
        return bt

    bt["pnl"] = np.where(bt["win"], bt["closing_odds"] - 1, -1)
    bt["cumulative_pnl"] = bt["pnl"].cumsum()

    summary = {
        "total_bets": len(bt),
        "win_rate": bt["win"].mean(),
        "avg_edge": bt["edge"].mean(),
        "total_pnl": bt["pnl"].sum(),
        "max_drawdown": (bt["cumulative_pnl"].cummax() - bt["cumulative_pnl"]).max(),
        "roi": bt["pnl"].sum() / len(bt),
    }
    logger.info(f"Backtest: {summary}")
    bt.attrs["summary"] = summary
    return bt


def backtest_over_under(df: pd.DataFrame, model_result: dict,
                        threshold: float = 0.05) -> pd.DataFrame:
    """Backtest Over/Under 2.5."""
    features = model_result["features"]
    X = df[features].fillna(0).values
    probs = model_result["model"].predict_proba(X)[:, 1]

    results = []
    for i in range(len(df)):
        odds_col = "odds_pinnacle_ou_2_5_over_last_seen"
        if odds_col not in df.columns:
            continue
        odds = df.iloc[i].get(odds_col, np.nan)
        if np.isnan(odds):
            continue

        edge = probs[i] - (1 / odds)
        if edge > threshold:
            actual_over = df.iloc[i]["over_2_5"] == 1
            results.append({
                "match_id": df.iloc[i]["match_id"],
                "model_prob_over": probs[i],
                "closing_odds": odds,
                "edge": edge,
                "win": actual_over,
            })

    bt = pd.DataFrame(results)
    if not bt.empty:
        bt["pnl"] = np.where(bt["win"], bt["closing_odds"] - 1, -1)
        bt["cumulative_pnl"] = bt["pnl"].cumsum()
        bt.attrs["summary"] = {
            "total_bets": len(bt),
            "win_rate": bt["win"].mean(),
            "total_pnl": bt["pnl"].sum(),
            "roi": bt["pnl"].sum() / len(bt),
        }
    return bt   
