import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def _safe_get(data: Any, *keys: str, default=np.nan) -> Any:
    """Return a nested value, or the default if the path is missing."""
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
        if data is default:
            return default
    return data


def _number(value: Any) -> float:
    """Convert API values to a finite float; invalid values become NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def _market_price(market: dict, selection: str, price_type: str) -> float:
    value = _safe_get(market, selection, price_type, default=np.nan)
    # Some API responses represent a price directly rather than as an object.
    if isinstance(value, dict):
        value = value.get("odds", value.get("price", np.nan))
    return _number(value)


def _extract_odds_features(odds_data: dict) -> dict:
    """Extract opening and closing odds from the API response."""
    if not isinstance(odds_data, dict):
        return {}
    if isinstance(odds_data.get("data"), dict):
        odds_data = odds_data["data"]

    bookmakers = odds_data.get("bookmakers", [])
    if not isinstance(bookmakers, list):
        return {}

    features = {}
    for bookmaker in bookmakers:
        if not isinstance(bookmaker, dict):
            continue
        name = str(bookmaker.get("bookmaker", "unknown")).strip().lower()
        name = name.replace(" ", "_").replace("-", "_")
        markets = bookmaker.get("markets", {})
        if not isinstance(markets, dict):
            continue

        market_specs = (
            ("match_odds", ("home", "draw", "away"), ""),
            ("total_goals", ("over", "under"), "ou_"),
            ("btts", ("yes", "no"), "btts_"),
            ("corners", ("over", "under"), "corners_"),
            ("asian_handicap", ("home", "away"), "ah_"),
        )
        for market_name, selections, prefix in market_specs:
            market = markets.get(market_name)
            if market is None and market_name == "asian_handicap":
                market = markets.get("ah", {})
            if not isinstance(market, dict):
                continue
            for selection in selections:
                for price_type in ("opening", "last_seen"):
                    price = _market_price(market, selection, price_type)
                    if math.isfinite(price) and price > 1:
                        features[f"odds_{name}_{prefix}{selection}_{price_type}"] = price
            if market_name == "asian_handicap":
                line = _number(market.get("line"))
                if math.isfinite(line):
                    features[f"odds_{name}_ah_line"] = line

    return features


def _is_finished(match: dict) -> bool:
    status = str(match.get("status", "")).strip().lower()
    return status in {"finished", "ft", "full_time", "completed"}


def build_dataset(data_dir: str | Path = config.DATA_DIR) -> pd.DataFrame:
    """Build a labeled dataset from downloaded seasons.

    Only pre-match opening odds are used as model inputs. Closing odds are kept
    for backtesting, while post-match statistics are deliberately excluded to
    avoid target leakage.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    records = []
    for file in sorted(data_path.glob("season_*.json")):
        try:
            with file.open(encoding="utf-8") as handle:
                matches = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable data file %s: %s", file, exc)
            continue
        if not isinstance(matches, list):
            logger.warning("Skipping %s: expected a list of matches", file)
            continue

        for match in matches:
            if not isinstance(match, dict) or not _is_finished(match):
                continue
            home_goals = _number(match.get("score_home"))
            away_goals = _number(match.get("score_away"))
            if not (math.isfinite(home_goals) and math.isfinite(away_goals)):
                score = match.get("score")
                if isinstance(score, dict):
                    home_goals = _number(score.get("home"))
                    away_goals = _number(score.get("away"))
            if not (math.isfinite(home_goals) and math.isfinite(away_goals)):
                continue

            row = {
                "match_id": match.get("match_id", match.get("id")),
                "season_id": match.get("season_id"),
                "home_team": match.get("home_team_name"),
                "away_team": match.get("away_team_name"),
                "kickoff": match.get("kickoff"),
                "score_home": int(home_goals),
                "score_away": int(away_goals),
            }
            row.update(_extract_odds_features(match.get("odds", {})))
            row["result_home"] = int(home_goals > away_goals)
            row["result_draw"] = int(home_goals == away_goals)
            row["result_away"] = int(home_goals < away_goals)
            row["total_goals"] = int(home_goals + away_goals)
            row["over_2_5"] = int(home_goals + away_goals > 2.5)
            row["btts"] = int(home_goals > 0 and away_goals > 0)
            records.append(row)

    if not records:
        raise ValueError(f"No completed matches with valid scores found in {data_path}")

    dataset = pd.DataFrame.from_records(records)
    dataset["kickoff"] = pd.to_datetime(dataset["kickoff"], errors="coerce", utc=True)
    dataset = dataset.sort_values(["kickoff", "match_id"], na_position="last").reset_index(drop=True)
    logger.info("Built dataset with %s completed matches and %s columns", *dataset.shape)
    return dataset
