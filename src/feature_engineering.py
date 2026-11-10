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


def normalize_team_name(name: Any) -> str | None:
    """Canonical key for a team name (lowercase, collapsed whitespace).

    Keeps the repository standard of one canonical name per team so ratings
    are not split across spelling variants.
    """
    if name is None:
        return None
    key = " ".join(str(name).strip().lower().split())
    return key or None


def _chronological_state(df: pd.DataFrame):
    """Walk matches in kickoff order yielding PRE-MATCH team state.

    Elo and form are updated only AFTER each match is processed, so every
    value reflects information available before kickoff (no data leakage).
    Returns (snapshots, final_ratings, final_form).
    """
    ratings: dict[str, float] = {}
    form: dict[str, list[tuple[float, float, float]]] = {}
    snapshots = []

    for row in df.itertuples(index=False):
        home_key = normalize_team_name(getattr(row, "home_team", None))
        away_key = normalize_team_name(getattr(row, "away_team", None))
        home_goals = float(getattr(row, "score_home"))
        away_goals = float(getattr(row, "score_away"))

        def snapshot(key):
            if key is None:
                return (np.nan, np.nan, np.nan, np.nan)
            entries = form.get(key, [])[-config.FORM_WINDOW:]
            elo = ratings.get(key, config.ELO_BASE_RATING)
            if not entries:
                return (elo, np.nan, np.nan, np.nan)
            n = len(entries)
            pts = sum(e[0] for e in entries) / n
            gf = sum(e[1] for e in entries) / n
            ga = sum(e[2] for e in entries) / n
            return (elo, pts, gf, ga)

        snapshots.append((home_key, away_key, snapshot(home_key), snapshot(away_key)))

        # --- update state AFTER recording pre-match values ---
        elo_h = ratings.get(home_key, config.ELO_BASE_RATING) if home_key else config.ELO_BASE_RATING
        elo_a = ratings.get(away_key, config.ELO_BASE_RATING) if away_key else config.ELO_BASE_RATING
        expected_home = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
        if home_goals > away_goals:
            actual_home, home_pts, away_pts = 1.0, 3.0, 0.0
        elif home_goals == away_goals:
            actual_home, home_pts, away_pts = 0.5, 1.0, 1.0
        else:
            actual_home, home_pts, away_pts = 0.0, 0.0, 3.0

        goal_diff = abs(home_goals - away_goals)
        k = config.ELO_K_FACTOR
        if goal_diff == 2:
            k *= 1.25
        elif goal_diff == 3:
            k *= 1.5
        elif goal_diff >= 4:
            k *= 1.75

        if home_key:
            ratings[home_key] = elo_h + k * (actual_home - expected_home)
            form.setdefault(home_key, []).append((home_pts, home_goals, away_goals))
        if away_key:
            ratings[away_key] = elo_a + k * ((1.0 - actual_home) - (1.0 - expected_home))
            form.setdefault(away_key, []).append((away_pts, away_goals, home_goals))

    return snapshots, ratings, form


def add_team_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free Elo and recent-form features (past matches only)."""
    if df.empty:
        return df
    df = df.sort_values(["kickoff", "match_id"], na_position="last").reset_index(drop=True)
    snapshots, _, _ = _chronological_state(df)

    elo_h, elo_a = [], []
    pts_h, pts_a, gf_h, gf_a, ga_h, ga_a = [], [], [], [], [], []
    for _hk, _ak, home_state, away_state in snapshots:
        elo_h.append(home_state[0]); pts_h.append(home_state[1])
        gf_h.append(home_state[2]); ga_h.append(home_state[3])
        elo_a.append(away_state[0]); pts_a.append(away_state[1])
        gf_a.append(away_state[2]); ga_a.append(away_state[3])

    df = df.copy()
    df["elo_home"] = elo_h
    df["elo_away"] = elo_a
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["form_pts_home"] = pts_h
    df["form_pts_away"] = pts_a
    df["form_pts_diff"] = df["form_pts_home"] - df["form_pts_away"]
    df["form_gf_home"] = gf_h
    df["form_gf_away"] = gf_a
    df["form_gf_diff"] = df["form_gf_home"] - df["form_gf_away"]
    df["form_ga_home"] = ga_h
    df["form_ga_away"] = ga_a
    df["form_ga_diff"] = df["form_ga_home"] - df["form_ga_away"]
    return df


def current_team_state(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Latest Elo/form per team after processing all history (for prediction)."""
    if df.empty:
        return {}
    df = df.sort_values(["kickoff", "match_id"], na_position="last")
    _, ratings, form = _chronological_state(df)
    state = {}
    for key, elo in ratings.items():
        entries = form.get(key, [])[-config.FORM_WINDOW:]
        n = len(entries)
        state[key] = {
            "elo": elo,
            "form_pts": sum(e[0] for e in entries) / n if n else np.nan,
            "form_gf": sum(e[1] for e in entries) / n if n else np.nan,
            "form_ga": sum(e[2] for e in entries) / n if n else np.nan,
        }
    return state


def build_dataset(data_dir: str | Path = config.DATA_DIR) -> pd.DataFrame:
    """Build a labeled dataset from downloaded seasons.

    Only pre-match opening odds are used as model inputs. Closing odds are kept
    for backtesting, while post-match statistics are deliberately excluded to
    avoid target leakage. Team-strength features (Elo/form) are computed
    chronologically from past results only.
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

    # Añadido: deduplicar por match_id (un partido no debe aparecer dos veces).
    before = len(dataset)
    dup_mask = dataset["match_id"].notna() & dataset.duplicated(subset=["match_id"], keep="last")
    dataset = dataset[~dup_mask].reset_index(drop=True)
    if before - len(dataset):
        logger.warning("Removed %s duplicated match_ids", before - len(dataset))

    dataset["kickoff"] = pd.to_datetime(dataset["kickoff"], errors="coerce", utc=True)
    dataset = dataset.sort_values(["kickoff", "match_id"], na_position="last").reset_index(drop=True)

    # Añadido: fuerza de equipos calculada solo con información previa al partido.
    dataset = add_team_strength_features(dataset)
    logger.info("Built dataset with %s completed matches and %s columns", *dataset.shape)
    return dataset
