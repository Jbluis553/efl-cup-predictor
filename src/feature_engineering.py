import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger(__name__)


def _safe_get(d, *keys, default=np.nan):
    """Acceso seguro a dict anidado."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def _extract_xg(stats: dict) -> dict:
    """Extrae xG de las stats del partido."""
    if not stats:
        return {"xg_home": np.nan, "xg_away": np.nan}

    # La estructura puede variar; intentamos varios caminos
    xg_home = _safe_get(stats, "xg", "home")
    xg_away = _safe_get(stats, "xg", "away")

    if np.isnan(xg_home):
        xg_home = _safe_get(stats, "expected_goals", "home")
    if np.isnan(xg_away):
        xg_away = _safe_get(stats, "expected_goals", "away")

    return {"xg_home": xg_home, "xg_away": xg_away}


def _extract_team_stats(stats: dict) -> dict:
    """Extrae métricas de equipo de las stats del partido."""
    if not stats:
        return {}

    features = {}

    # Estructura esperada: stats tiene campos directos o anidados
    # Posibilidad 1: campos directos con prefijo
    for field in ["possession", "shots", "shots_on_target", "passes",
                  "pass_accuracy", "corners", "fouls", "yellow_cards",
                  "red_cards", "tackles", "interceptions", "clearances"]:
        features[f"home_{field}"] = _safe_get(stats, "home", field)
        features[f"away_{field}"] = _safe_get(stats, "away", field)

    # Posibilidad 2: estructura con "teams" lista
    if "teams" in stats and isinstance(stats["teams"], list):
        teams = stats["teams"]
        if len(teams) >= 2:
            for field in ["possession", "shots", "shots_on_target", "passes",
                          "pass_accuracy", "corners", "fouls", "yellow_cards",
                          "red_cards", "tackles", "interceptions", "clearances"]:
                features[f"home_{field}"] = _safe_get(teams[0], field)
                features[f"away_{field}"] = _safe_get(teams[1], field)

    return features


def _extract_odds_features(odds_data: dict) -> dict:
    """
    Extrae features de cuotas.
    Estructura real de TheStatsAPI:
    {
      "match_id": "mt_...",
      "bookmakers": [
        {
          "bookmaker": "Pinnacle",
          "markets": {
            "match_odds": {
              "home": {"opening": "2.150", "last_seen": "2.080"},
              "draw": {"opening": "3.400", "last_seen": "3.500"},
              "away": {"opening": "3.200", "last_seen": "3.300"}
            },
            "total_goals": {
              "over": {"opening": "1.890", "last_seen": "1.900"},
              "under": {"opening": "1.930", "last_seen": "1.920"}
            },
            "btts": {
              "yes": {"opening": "1.70", "last_seen": "1.68"},
              "no": {"opening": "2.10", "last_seen": "2.12"}
            },
            "corners": {
              "over": {"opening": "1.90", "last_seen": "1.88"},
              "under": {"opening": "1.92", "last_seen": "1.94"}
            }
          }
        }
      ]
    }
    """
    if not odds_data:
        return {}

    features = {}
    bookmakers = odds_data.get("bookmakers", [])

    for bm in bookmakers:
        name = bm.get("bookmaker", "unknown").lower().replace(" ", "_").replace("-", "_")
        markets = bm.get("markets", {})

        # 1X2
        match_odds = markets.get("match_odds", {})
        for selection in ["home", "draw", "away"]:
            for price_type in ["opening", "last_seen"]:
                val = _safe_get(match_odds, selection, price_type)
                if not np.isnan(val):
                    features[f"odds_{name}_{selection}_{price_type}"] = float(val)

        # Over/Under (total_goals)
        total_goals = markets.get("total_goals", {})
        for side in ["over", "under"]:
            for price_type in ["opening", "last_seen"]:
                val = _safe_get(total_goals, side, price_type)
                if not np.isnan(val):
                    features[f"odds_{name}_ou_{side}_{price_type}"] = float(val)

        # BTTS
        btts = markets.get("btts", {})
        for side in ["yes", "no"]:
            for price_type in ["opening", "last_seen"]:
                val = _safe_get(btts, side, price_type)
                if not np.isnan(val):
                    features[f"odds_{name}_btts_{side}_{price_type}"] = float(val)

        # Corners
        corners = markets.get("corners", {})
        for side in ["over", "under"]:
            for price_type in ["opening", "last_seen"]:
                val = _safe_get(corners, side, price_type)
                if not np.isnan(val):
                    features[f"odds_{name}_corners_{side}_{price_type}"] = float(val)

        # Asian Handicap
        ah = markets.get("asian_handicap", markets.get("ah", {}))
        for side in ["home", "away"]:
            for price_type in ["opening", "last_seen"]:
                val = _safe_get(ah, side, price_type)
                if not np.isnan(val):
                    features[f"odds_{name}_ah_{side}_{price_type}"] = float(val)
        ah_line = _safe_get(ah, "line")
        if not np.isnan(ah_line):
            features[f"odds_{name}_ah_line"] = float(ah_line)

    # Implied probabilities from Pinnacle (sharp)
    pinnacle = next(
        (bm for bm in bookmakers if "pinnacle" in bm.get("bookmaker", "").lower()),
        None,
    )
    if pinnacle:
        match_odds = pinnacle.get("markets", {}).get("match_odds", {})
        home_odds = _safe_get(match_odds, "home", "last_seen")
        draw_odds = _safe_get(match_odds, "draw", "last_seen")
        away_odds = _safe_get(match_odds, "away", "last_seen")

        if not any(np.isnan(v) for v in [home_odds, draw_odds, away_odds]):
            total = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)
            features["pinnacle_implied_home"] = (1 / home_odds) / total
            features["pinnacle_implied_draw"] = (1 / draw_odds) / total
            features["pinnacle_implied_away"] = (1 / away_odds) / total
            features["pinnacle_overround"] = total - 1

    return features


def _extract_shotmap_features(shotmap: list) -> dict:
    """Extrae features de calidad de tiros desde el shotmap."""
    if not shotmap:
        return {}

    features = {}
    for prefix, side in [("home", "home"), ("away", "away")]:
        shots = [s for s in shotmap if s.get("team_side", s.get("side")) == side]
        if not shots:
            continue

        xg_values = [s.get("expected_goals", s.get("xg", 0)) for s in shots]
        features[f"{prefix}_shot_count"] = len(shots)
        features[f"{prefix}_avg_shot_xg"] = np.mean(xg_values)
        features[f"{prefix}_max_shot_xg"] = np.max(xg_values)
        features[f"{prefix}_big_chance_count"] = sum(1 for x in xg_values if x >= 0.25)

        on_target = sum(1 for s in shots if s.get("is_on_target", s.get("on_target", False)))
        features[f"{prefix}_on_target_ratio"] = on_target / len(shots)

    return features


def _extract_player_features(player_stats: dict) -> dict:
    """Features agregados de jugadores."""
    if not player_stats:
        return {}

    features = {}
    # Estructura puede ser lista o dict con home/away
    players = player_stats if isinstance(player_stats, list) else []
    if isinstance(player_stats, dict):
        players = (
            player_stats.get("home", []) + player_stats.get("away", [])
            if isinstance(player_stats.get("home"), list)
            else player_stats.get("players", player_stats.get("data", []))
        )

    if not players:
        return features

    for prefix, side in [("home", "home"), ("away", "away")]:
        side_players = [p for p in players if p.get("team_side", p.get("side")) == side]
        if not side_players:
            continue

        features[f"{prefix}_total_goals"] = sum(p.get("goals", 0) for p in side_players)
        features[f"{prefix}_total_assists"] = sum(p.get("assists", 0) for p in side_players)
        features[f"{prefix}_total_shots"] = sum(p.get("shots", 0) for p in side_players)
        features[f"{prefix}_total_passes"] = sum(p.get("passes", 0) for p in side_players)
        features[f"{prefix}_total_tackles"] = sum(p.get("tackles", 0) for p in side_players)

        xg_vals = [p.get("expected_goals", p.get("xg", 0)) for p in side_players]
        features[f"{prefix}_avg_xg"] = np.mean(xg_vals) if xg_vals else 0

    return features


def _extract_form_features(team_season: dict) -> dict:
    """Features de forma/registro de temporada."""
    if not team_season:
        return {}

    features = {
        "wins": team_season.get("wins", np.nan),
        "draws": team_season.get("draws", np.nan),
        "losses": team_season.get("losses", np.nan),
        "goals_for": team_season.get("goals_for", team_season.get("goals", np.nan)),
        "goals_against": team_season.get("goals_against", np.nan),
        "points": team_season.get("points", np.nan),
        "position": team_season.get("position", np.nan),
    }

    gf, ga = features["goals_for"], features["goals_against"]
    if not np.isnan(gf) and not np.isnan(ga):
        features["goal_diff"] = gf - ga
    else:
        features["goal_diff"] = np.nan

    return features


def build_dataset(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """
    Construye el DataFrame completo de features desde los JSON descargados.
    """
    records = []

    for file in sorted(data_dir.glob("season_*.j   
