"""Predicciones para fixtures reales próximos (sin datos ficticios).

Uso: python -m src.predict [--days 30]
Salida: predictions/predictions_YYYY-MM-DD.csv con columnas fijas.
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_fetcher import get_upcoming_matches
from .feature_engineering import (
    _extract_odds_features,
    build_dataset,
    current_team_state,
    normalize_team_name,
)
from .model import _positive_probability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict")

OUTPUT_COLUMNS = [
    "match_id", "kickoff", "home_team", "away_team",
    "P(home)", "P(draw)", "P(away)", "P(over_2_5)", "P(btts)",
    "implied_odds_home", "implied_odds_draw", "implied_odds_away",
]


def _team_feature_row(team_state, key):
    state = team_state.get(key) if key else None
    base = config.ELO_BASE_RATING
    return {
        "elo": state["elo"] if state else base,
        "form_pts": state["form_pts"] if state else np.nan,
        "form_gf": state["form_gf"] if state else np.nan,
        "form_ga": state["form_ga"] if state else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict upcoming EFL Cup fixtures")
    parser.add_argument("--days", type=int, default=config.FIXTURES_LOOKAHEAD_DAYS)
    args = parser.parse_args()

    models_dir = Path(config.MODEL_DIR)
    with (models_dir / config.FEATURES_FILE).open(encoding="utf-8") as handle:
        features = json.load(handle)
    final_models = {
        outcome: joblib.load(models_dir / f"1x2_{outcome}.joblib")
        for outcome in ("home", "draw", "away")
    }
    over_under_model = joblib.load(models_dir / "over_under.joblib")
    btts_model = joblib.load(models_dir / "btts.joblib")

    fixtures = get_upcoming_matches(days_ahead=args.days)
    if not fixtures:
        logger.warning("No upcoming fixtures found in the next %s days", args.days)
        return

    logger.info("Building team state from historical data...")
    history = build_dataset()
    team_state = current_team_state(history)

    rows = []
    for fixture in fixtures:
        row = {
            "match_id": fixture.get("match_id"),
            "kickoff": fixture.get("kickoff"),
            "home_team": fixture.get("home_team_name"),
            "away_team": fixture.get("away_team_name"),
        }
        row.update(_extract_odds_features(fixture.get("odds", {})))
        home = _team_feature_row(team_state, normalize_team_name(row["home_team"]))
        away = _team_feature_row(team_state, normalize_team_name(row["away_team"]))
        row["elo_home"] = home["elo"]; row["elo_away"] = away["elo"]
        row["elo_diff"] = home["elo"] - away["elo"]
        row["form_pts_home"] = home["form_pts"]; row["form_pts_away"] = away["form_pts"]
        row["form_pts_diff"] = home["form_pts"] - away["form_pts"]
        row["form_gf_home"] = home["form_gf"]; row["form_gf_away"] = away["form_gf"]
        row["form_gf_diff"] = home["form_gf"] - away["form_gf"]
        row["form_ga_home"] = home["form_ga"]; row["form_ga_away"] = away["form_ga"]
        row["form_ga_diff"] = home["form_ga"] - away["form_ga"]
        rows.append(row)

    fx_df = pd.DataFrame(rows)
    X = fx_df.reindex(columns=features, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)

    probs = np.column_stack([_positive_probability(final_models[o], X) for o in ("home", "draw", "away")])
    totals = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, totals, out=np.full_like(probs, 1 / 3), where=totals > 0)
    p_over = _positive_probability(over_under_model, X)
    p_btts = _positive_probability(btts_model, X)

    output = pd.DataFrame({
        "match_id": fx_df["match_id"],
        "kickoff": pd.to_datetime(fx_df["kickoff"], errors="coerce", utc=True),
        "home_team": fx_df["home_team"],
        "away_team": fx_df["away_team"],
        "P(home)": np.round(probs[:, 0], 4),
        "P(draw)": np.round(probs[:, 1], 4),
        "P(away)": np.round(probs[:, 2], 4),
        "P(over_2_5)": np.round(p_over, 4),
        "P(btts)": np.round(p_btts, 4),
        "implied_odds_home": np.where(probs[:, 0] > 0, np.round(1 / probs[:, 0], 2), np.nan),
        "implied_odds_draw": np.where(probs[:, 1] > 0, np.round(1 / probs[:, 1], 2), np.nan),
        "implied_odds_away": np.where(probs[:, 2] > 0, np.round(1 / probs[:, 2], 2), np.nan),
    })[OUTPUT_COLUMNS]

    out_dir = Path("predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"predictions_{datetime.now(timezone.utc):%Y-%m-%d}.csv"
    output.to_csv(path, index=False)
    logger.info("Wrote %s predictions to %s", len(output), path)


if __name__ == "__main__":
    main()
