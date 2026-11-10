import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def _valid_odds(value) -> bool:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(odds) and config.MIN_ODDS <= odds <= config.MAX_ODDS


def _odds_matrix(df: pd.DataFrame, selections: tuple, price_type: str,
                 bookmaker: str, prefix: str = "") -> np.ndarray:
    """Matrix of odds (NaN when missing/invalid) for a set of selections."""
    matrix = np.full((len(df), len(selections)), np.nan)
    for j, selection in enumerate(selections):
        column = f"odds_{bookmaker}_{prefix}{selection}_{price_type}"
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & (values >= config.MIN_ODDS) & (values <= config.MAX_ODDS)
        matrix[valid, j] = values[valid]
    return matrix


def _fair_probabilities(df: pd.DataFrame, selections: tuple, price_type: str,
                        bookmaker: str, prefix: str = "") -> np.ndarray:
    """Margin-free implied probabilities normalized across the selections."""
    implied = np.full((len(df), len(selections)), np.nan)
    odds = _odds_matrix(df, selections, price_type, bookmaker, prefix)
    valid = np.isfinite(odds) & (odds > 1)
    implied[valid] = 1.0 / odds[valid]
    totals = np.nansum(implied, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fair = implied / np.where(totals > 0, totals[:, None], np.nan)
    fair[~np.isfinite(fair)] = np.nan
    return fair


def _normalize_1x2(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    totals = probabilities.sum(axis=1, keepdims=True)
    return np.divide(
        probabilities,
        totals,
        out=np.full_like(probabilities, 1 / 3),
        where=np.isfinite(totals) & (totals > 0),
    )


def _positive_probability(model, X: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(1)]


def _model_probabilities_1x2(df: pd.DataFrame, model_dict: dict) -> np.ndarray:
    """In-sample 1X2 probabilities (legacy path; prefer passing OOF)."""
    features = model_dict["features"]
    X = df.reindex(columns=features, fill_value=0).fillna(0).values
    probabilities = np.column_stack([
        _positive_probability(model_dict["final_models"][outcome], X)
        for outcome in ("home", "draw", "away")
    ])
    return _normalize_1x2(probabilities)


def backtest_1x2(df: pd.DataFrame, model_dict: dict, threshold: float = 0.05,
                 probabilities: np.ndarray | None = None,
                 bookmaker: str | None = None) -> pd.DataFrame:
    """Backtest 1X2 against Pinnacle prices with margin removed from the close.

    Bets are placed at OPENING odds (the realistically actionable price); the
    edge and CLV are measured against the margin-free CLOSING probability.
    Pass `probabilities` (chronological OOF) for an honest out-of-sample test;
    omitting it falls back to in-sample predictions (optimistic, legacy mode).
    """
    bookmaker = bookmaker or config.BACKTEST_BOOKMAKER
    outcomes = ("home", "draw", "away")
    if probabilities is None:
        probabilities = _model_probabilities_1x2(df, model_dict)
    else:
        probabilities = _normalize_1x2(probabilities)

    n = len(df)
    fair_close = _fair_probabilities(df, outcomes, "last_seen", bookmaker)
    taken = _odds_matrix(df, outcomes, "opening", bookmaker)
    raw_close = _odds_matrix(df, outcomes, "last_seen", bookmaker)

    taken_valid = np.isfinite(taken)
    edges = probabilities - fair_close
    edges[~np.isfinite(edges)] = -np.inf
    edges[~taken_valid] = -np.inf
    best_idx = np.argmax(edges, axis=1)
    best_edge = edges[np.arange(n), best_idx]
    mask = np.isfinite(best_edge) & (best_edge > threshold)
    if not mask.any():
        logger.warning("No 1X2 bets passed the threshold filter")
        return pd.DataFrame()

    result_home = df["result_home"].to_numpy()
    result_draw = df["result_draw"].to_numpy()
    actual_code = np.select(
        [result_home == 1, result_draw == 1],
        [0, 1],
        default=2,
    )
    rows = np.arange(n)[mask]
    sel = best_idx[mask]
    bt = pd.DataFrame({
        "match_id": df["match_id"].to_numpy()[rows],
        "outcome": [outcomes[s] for s in sel],
        "model_prob": probabilities[rows, sel],
        "taken_odds_opening": taken[rows, sel],
        "closing_odds_last_seen": raw_close[rows, sel],
        "closing_fair_prob": fair_close[rows, sel],
        "edge": best_edge[mask],
        "clv": fair_close[rows, sel] * taken[rows, sel] - 1,
        "actual_result": [outcomes[c] for c in actual_code[rows]],
        "win": actual_code[rows] == sel,
    })

    bt["pnl"] = np.where(bt["win"], bt["taken_odds_opening"] - 1, -1)
    bt["cumulative_pnl"] = bt["pnl"].cumsum()
    bt.attrs["summary"] = {
        "total_bets": len(bt),
        "win_rate": bt["win"].mean(),
        "avg_edge": bt["edge"].mean(),
        "avg_clv": bt["clv"].mean(),
        "total_pnl": bt["pnl"].sum(),
        "max_drawdown": (bt["cumulative_pnl"].cummax() - bt["cumulative_pnl"]).max(),
        "roi": bt["pnl"].sum() / len(bt),
    }
    logger.info("1X2 backtest: %s", bt.attrs["summary"])
    return bt


def backtest_over_under(df: pd.DataFrame, model_result: dict, threshold: float = 0.05,
                        probabilities: np.ndarray | None = None,
                        bookmaker: str | None = None) -> pd.DataFrame:
    """Backtest Over/Under 2.5 using Pinnacle prices (margin removed)."""
    bookmaker = bookmaker or config.BACKTEST_BOOKMAKER
    if probabilities is None:
        features = model_result["features"]
        X = df.reindex(columns=features, fill_value=0).fillna(0).values
        probs = _positive_probability(model_result["model"], X)
    else:
        probs = np.asarray(probabilities, dtype=float)

    fair_close = _fair_probabilities(df, ("over", "under"), "last_seen", bookmaker, "ou_")[:, 0]
    taken = _odds_matrix(df, ("over",), "opening", bookmaker, "ou_")[:, 0]
    edge = probs - fair_close
    valid = np.isfinite(edge) & np.isfinite(taken)
    mask = valid & (edge > threshold)
    if not mask.any():
        logger.warning("No Over/Under bets passed the threshold filter")
        return pd.DataFrame()

    bt = pd.DataFrame({
        "match_id": df["match_id"].to_numpy()[mask],
        "model_prob_over": probs[mask],
        "taken_odds_opening": taken[mask],
        "closing_fair_prob": fair_close[mask],
        "edge": edge[mask],
        "clv": fair_close[mask] * taken[mask] - 1,
        "win": df["over_2_5"].to_numpy()[mask] == 1,
    })
    bt["pnl"] = np.where(bt["win"], bt["taken_odds_opening"] - 1, -1)
    bt["cumulative_pnl"] = bt["pnl"].cumsum()
    bt.attrs["summary"] = {
        "total_bets": len(bt),
        "win_rate": bt["win"].mean(),
        "avg_edge": bt["edge"].mean(),
        "avg_clv": bt["clv"].mean(),
        "total_pnl": bt["pnl"].sum(),
        "roi": bt["pnl"].sum() / len(bt),
    }
    logger.info("O/U backtest: %s", bt.attrs["summary"])
    return bt


def backtest_btts(df: pd.DataFrame, model_result: dict, threshold: float = 0.05,
                  probabilities: np.ndarray | None = None,
                  bookmaker: str | None = None) -> pd.DataFrame:
    """Backtest BTTS using Pinnacle prices (margin removed)."""
    bookmaker = bookmaker or config.BACKTEST_BOOKMAKER
    if probabilities is None:
        features = model_result["features"]
        X = df.reindex(columns=features, fill_value=0).fillna(0).values
        probs = _positive_probability(model_result["model"], X)
    else:
        probs = np.asarray(probabilities, dtype=float)

    fair_close = _fair_probabilities(df, ("yes", "no"), "last_seen", bookmaker, "btts_")[:, 0]
    taken = _odds_matrix(df, ("yes",), "opening", bookmaker, "btts_")[:, 0]
    edge = probs - fair_close
    valid = np.isfinite(edge) & np.isfinite(taken)
    mask = valid & (edge > threshold)
    if not mask.any():
        logger.warning("No BTTS bets passed the threshold filter")
        return pd.DataFrame()

    bt = pd.DataFrame({
        "match_id": df["match_id"].to_numpy()[mask],
        "model_prob_yes": probs[mask],
        "taken_odds_opening": taken[mask],
        "closing_fair_prob": fair_close[mask],
        "edge": edge[mask],
        "clv": fair_close[mask] * taken[mask] - 1,
        "win": df["btts"].to_numpy()[mask] == 1,
    })
    bt["pnl"] = np.where(bt["win"], bt["taken_odds_opening"] - 1, -1)
    bt["cumulative_pnl"] = bt["pnl"].cumsum()
    bt.attrs["summary"] = {
        "total_bets": len(bt),
        "win_rate": bt["win"].mean(),
        "avg_edge": bt["edge"].mean(),
        "avg_clv": bt["clv"].mean(),
        "total_pnl": bt["pnl"].sum(),
        "roi": bt["pnl"].sum() / len(bt),
    }
    logger.info("BTTS backtest: %s", bt.attrs["summary"])
    return bt
