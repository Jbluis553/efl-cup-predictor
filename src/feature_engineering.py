import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from . import config


def _extract_xg(stats: dict) -> dict:
    """Extrae xG de las stats del partido."""
    overview = stats.get("overview", stats)
    xg_data = overview.get("expected_goals", overview.get("xg", {}))
    if isinstance(xg_data, dict):
        all_xg = xg_data.get("all", xg_data)
        return {
            "xg_home": all_xg.get("home", np.nan),
            "xg_away": all_xg.get("away", np.nan),
        }
    return {"xg_home": np.nan, "xg_away": np.nan}


def _extract_team_stats(stats: dict) -> dict:
    """Extrae métricas de equipo de las stats."""
    overview = stats.get("overview", stats)
    teams = overview.get("teams", [overview.get("home", {}), overview.get("away", {})])

    def _safe_get(d, *keys):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, np.nan)
            else:
                return np.nan
        return d

    home = teams[0] if len(teams) > 0 else {}
    away = teams[1] if len(teams) > 1 else {}

    features = {}
    for prefix, team in [("home", home), ("away", away)]:
        features[f"{prefix}_possession"] = _safe_get(team, "possession")
        features[f"{prefix}_shots"] = _safe_get(team, "shots")
        features[f"{prefix}_shots_on_target"] = _safe_get(team, "shots_on_target")
        features[f"{prefix}_passes"] = _safe_get(team, "passes")
        features[f"{prefix}_pass_accuracy"] = _safe_get(team, "pass_accuracy")
        features[f"{prefix}_corners"] = _safe_get(team, "corners")
        features[f"{prefix}_fouls"] = _safe_get(team, "fouls")
        features[f"{prefix}_yellow_cards"] = _safe_get(team, "yellow_cards")
        features[f"{prefix}_red_cards"] = _safe_get(team, "red_cards")
        features[f"{prefix}_tackles"] = _safe_get(team, "tackles")
        features[f"{prefix}_interceptions"] = _safe_get(team, "interceptions")
        features[f"{prefix}_clearances"] = _safe_get(team, "clearances")
    return features


def _extract_odds_features(odds_data: dict) -> dict:
    """Extrae features de cuotas de todos los bookmakers."""
    features = {}
    bookmakers = odds_data.get("bookmakers", odds_data.get("data", []))
    if isinstance(bookmakers, dict):
        bookmakers = [bookmakers]

    for bm in bookmakers:
        name = bm.get("name", "unknown").lower().replace(" ", "_")
        markets = bm.get("markets", {})

        # 1X2
        match_odds = markets.get("match_odds", {})
        for selection in ["home", "draw", "away"]:
            for price_type in ["opening", "last_seen"]:
                val = match_odds.get(selection, {}).get(price_type)
                if val is not None:
                    features[f"odds_{name}_{selection}_{price_type}"] = val

        # Over/Under
        total_goals = markets.get("total_goals", {})
        for line in ["2_5", "1_5", "3_5"]:
            for side in ["over", "under"]:
                for price_type in ["opening", "last_seen"]:
                    val = total_goals.get(line, {}).get(side, {}).get(price_type)
                    if val is not None:
                        features[f"odds_{name}_ou_{line}_{side}_{price_type}"] = val

        # BTTS
        btts = markets.get("btts", {})
        for side in ["yes", "no"]:
            for price_type in ["opening", "last_seen"]:
                val = btts.get(side, {}).get(price_type)
                if val is not None:
                    features[f"odds_{name}_btts_{side}_{price_type}"] = val

        # Corners
        corners = markets.get("corners", {})
        for line in ["9_0", "8_0", "10_0"]:
            for side in ["over", "under"]:
                for price_type in ["opening", "last_seen"]:
                    val = corners.get(line, {}).get(side, {}).get(price_type)
                    if val is not None:
                        features[f"odds_{name}_corners_{line}_{side}_{price_type}"] = val

    # Implied probabilities from Pinnacle (sharp)
    pinnacle = next((bm for bm in bookmakers if "pinnacle" in bm.get("name", "").lower()), None)
    if pinnacle:
        match_odds = pinnacle.get("markets", {}).get("match_odds", {})
        home_odds = match_odds.get("home", {}).get("last_seen")
        draw_odds = match_odds.get("draw", {}).get("last_seen")
        away_odds = match_odds.get("away", {}).get("last_seen")
        if home_odds and draw_odds and away_odds:
            total = 1/home_odds + 1/draw_odds + 1/away_odds
            features["pinnacle_implied_home"] = (1/home_odds) / total
            features["pinnacle_implied_draw"] = (1/draw_odds) / total
            features["pinnacle_implied_away"] = (1/away_odds) / total
            features["overround"] = total - 1

    return features


def _extract_shotmap_features(shotmap: list[dict]) -> dict:
    """Extrae features de calidad de tiros desde el shotmap."""
    if not shotmap:
        return {}

    home_shots = [s for s in shotmap if s.get("team_side") == "home"]
    away_shots = [s for s in shotmap if s.get("team_side") == "away"]

    features = {}
    for prefix, shots in [("home", home_shots), ("away", away_shots)]:
        xg_values = [s.get("expected_goals", 0) for s in shots]
        features[f"{prefix}_shot_count"] = len(shots)
        features[f"{prefix}_avg_shot_xg"] = np.mean(xg_values) if xg_values else 0
        features[f"{prefix}_max_shot_xg"] = np.max(xg_values) if xg_values else 0
        features[f"{prefix}_big_chance_count"] = sum(1 for x in xg_values if x >= 0.25)
        on_target = sum(1 for s in shots if s.get("is_on_target"))
        features[f"{prefix}_on_target_ratio"] = on_target / len(shots) if shots else 0

    return features


def _extract_player_features(player_stats: dict) -> dict:
    """Features agregados de jugadores."""
    features = {}
    players = player_stats.get("data", player_stats.get("players", []))
    if isinstance(players, dict):
        players = players.get("home", []) + players.get("away", [])

    if not players:
        return features

    for prefix, side in [("home", "home"), ("away", "away")]:
        side_players = [p for p in players if p.get("team_side") == side]
        if not side_players:
            continue
        features[f"{prefix}_total_goals"] = sum(p.get("goals", 0) for p in side_players)
        features[f"{prefix}_total_assists"] = sum(p.get("assists", 0) for p in side_players)
        features[f"{prefix}_total_shots"] = sum(p.get("shots", 0) for p in side_players)
        features[f"{prefix}_avg_xg"] = np.mean([p.get("expected_goals", 0) for p in side_players]) if side_players else 0
        features[f"{prefix}_avg_xa"] = np.mean([p.get("expected_assists", 0) for p in side_players]) if side_players else 0
        features[f"{prefix}_total_passes"] = sum(p.get("passes", 0) for p in side_players)
        features[f"{prefix}_total_tackles"] = sum(p.get("tackles", 0) for p in side_players)

    return features


def _extract_form_features(team_season: dict) -> dict:
    """Features de forma/registro de temporada."""
    features = {}
    data = team_season.get("data", team_season)
    if not data:
        return features

    features["wins"] = data.get("wins", np.nan)
    features["draws"] = data.get("draws", np.nan)
    features["losses"] = data.get("losses", np.nan)
    features["goals_for"] = data.get("goals_for", data.get("goals", np.nan))
    features["goals_against"] = data.get("goals_against", np.nan)
    features["points"] = data.get("points", np.nan)
    features["position"] = data.get("position", np.nan)
    features["goal_diff"] = features.get("goals_for", 0) - features.get("goals_against", 0) if not np.isnan(features.get("goals_for", np.nan)) else np.nan

    return features


def build_dataset(data_dir: str = config.DATA_DIR) -> pd.DataFrame:
    """
    Construye el DataFrame completo de features desde los JSON descargados.
    """
    out = Path(data_dir)
    records = []

    for file in sorted(out.glob("season_*.json")):
        with open(file) as f:
            matches = json.load(f)

        for m in matches:
            if m.get("score_home") is None:
                continue  # Skip unfinished

            row = {
                "match_id": m["match_id"],
                "season_id": m["season_id"],
                "home_team": m.get("home_team_name", ""),
                "away_team": m.get("away_team_name", ""),
                "kickoff": m.get("kickoff", ""),
                "score_home": m["score_home"],
                "score_away": m["score_away"],
            }

            # Labels
            row["result_home"] = 1 if m["score_home"] > m["score_away"] else 0
            row["result_draw"] = 1 if m["score_home"] == m["score_away"] else 0
            row["result_away"] = 1 if m["score_home"] < m["score_away"] else 0
            row["total_goals"] = m["score_home"] + m["score_away"]
            row["over_2_5"] = 1 if row["total_goals"] > 2.5 else 0
            row["btts"] = 1 if m["score_home"] > 0 and m["score_away"] > 0 else 0

            # Stats features
            stats = m.get("stats", {})
            row.update(_extract_team_stats(stats))
            row.update(_extract_xg(stats))

            # Odds features
            row.update(_extract_odds_features(m.get("odds", {})))

            # Shotmap features
            row.update(_extract_shotmap_features(m.get("shotmap", [])))

            # Player features
            row.update(_extract_player_features(m.get("player_stats", {})))

            # Team season features (renamed with prefix)
            home_season = _extract_form_features(m.get("home_team_season", {}))
            away_season = _extract_form_features(m.get("away_team_season", {}))
            for k, v in home_season.items():
                row[f"home_{k}"] = v
            for k, v in away_season.items():
                row[f"away_{k}"] = v

            records.append(row)

    df = pd.DataFrame(records)
    df["kickoff"] = pd.to_datetime(df["kickoff"], errors="coerce")
    df = df.sort_values("kickoff").reset_index(drop=True)

    # Drop rows with no useful features at all
    feature_cols = [c for c in df.columns if c not in
                    ["match_id", "season_id", "home_team", "away_team", "kickoff",
                     "score_home", "score_away", "result_home", "result_draw",
                     "result_away", "total_goals", "over_2_5", "btts"]]
    df = df.dropna(subset=feature_cols[:5], how="all").reset_index(drop=True)

    logger.info(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


logger = __import__("logging").getLogger(__name__)   
