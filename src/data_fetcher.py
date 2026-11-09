import time
import json
import logging
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from . import config

logger = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({
    "Authorization": f"Bearer {config.API_KEY}",
    "Content-Type": "application/json"
})


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.ConnectionError)),
    reraise=True
)
def _get(endpoint: str, params: Optional[dict] = None) -> dict:
    url = f"{config.API_BASE}/{endpoint}"
    resp = _session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(config.REQUEST_DELAY)
    return resp.json()


def get_seasons() -> list[dict]:
    """Obtiene todas las temporadas de la competición."""
    logger.info(f"Fetching seasons for {config.COMPETITION_ID}")
    data = _get(f"competitions/{config.COMPETITION_ID}/seasons")
    return data.get("data", data)


def get_matches(season_id: str, page: int = 1, per_page: int = 100) -> list[dict]:
    """Obtiene todos los partidos de una temporada (paginado)."""
    all_matches = []
    while True:
        data = _get("matches", {
            "competition_id": config.COMPETITION_ID,
            "season_id": season_id,
            "page": page,
            "per_page": per_page
        })
        matches = data.get("data", data)
        if isinstance(matches, dict):
            matches = matches.get("matches", [])
        if not matches:
            break
        all_matches.extend(matches)
        total_pages = data.get("meta", {}).get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1
    logger.info(f"  Season {season_id}: {len(all_matches)} matches")
    return all_matches


def get_match_stats(match_id: str) -> dict:
    """Estadísticas del partido + xG."""
    try:
        return _get(f"matches/{match_id}/stats")
    except Exception as e:
        logger.warning(f"  Stats unavailable for {match_id}: {e}")
        return {}


def get_match_odds(match_id: str) -> dict:
    """Cuotas pre-match (opening + last-seen) de todos los bookmakers."""
    try:
        return _get(f"matches/{match_id}/odds")
    except Exception as e:
        logger.warning(f"  Odds unavailable for {match_id}: {e}")
        return {}


def get_match_shotmap(match_id: str) -> list[dict]:
    """Shotmap con xG por tiro."""
    try:
        data = _get(f"matches/{match_id}/shotmap")
        return data.get("data", data) if isinstance(data, dict) else data
    except Exception:
        return []


def get_match_player_stats(match_id: str) -> dict:
    """Estadísticas de jugadores del partido."""
    try:
        return _get(f"matches/{match_id}/player-stats")
    except Exception:
        return {}


def get_team_season_stats(team_id: str, season_id: str) -> dict:
    """Estadísticas de temporada del equipo."""
    try:
        return _get(f"teams/{team_id}/stats", {"season_id": season_id})
    except Exception:
        return {}


def get_standings(competition_id: str, season_id: str) -> list[dict]:
    """Tabla de posiciones."""
    try:
        data = _get("standings", {
            "competition_id": competition_id,
            "season_id": season_id
        })
        return data.get("data", data) if isinstance(data, dict) else data
    except Exception:
        return []


def get_head_to_head(team1_id: str, team2_id: str) -> dict:
    """Historial de enfrentamientos."""
    try:
        return _get("head-to-head", {"home": team1_id, "away": team2_id})
    except Exception:
        return {}


def fetch_all_data(seasons: list[dict], output_dir: str = config.DATA_DIR) -> None:
    """
    Pipeline completo de descarga.
    Con el plan Scale (1000 req/min) y 10 temporadas de EFL Cup (~300 partidos/season),
    esto toma aproximadamente 3-5 minutos.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_matches = 0
    for season in seasons:
        season_id = season.get("id", season.get("season_id", ""))
        logger.info(f"=== Season: {season_id} ===")

        matches = get_matches(season_id)
        season_data = []

        for match in matches:
            match_id = match.get("id", match.get("match_id", ""))
            record = {
                "match_id": match_id,
                "season_id": season_id,
                "home_team_id": match.get("home_team", {}).get("id"),
                "away_team_id": match.get("away_team", {}).get("id"),
                "home_team_name": match.get("home_team", {}).get("name"),
                "away_team_name": match.get("away_team", {}).get("name"),
                "kickoff": match.get("kickoff"),
                "status": match.get("status"),
                "score_home": match.get("score", {}).get("home"),
                "score_away": match.get("score", {}).get("away"),
                "xg_available": match.get("xg_available", False),
                "odds_available": match.get("odds_available", False),
                "shotmap_available": match.get("shotmap_available", False),
            }

            # Solo procesamos partidos finalizados
            if match.get("status") not in ("finished", "ft", "completed"):
                season_data.append(record)
                continue

            # Stats + xG
            stats = get_match_stats(match_id)
            record["stats"] = stats

            # Odds
            if record["odds_available"]:
                record["odds"] = get_match_odds(match_id)

            # Shotmap
            if record["shotmap_available"]:
                record["shotmap"] = get_match_shotmap(match_id)

            # Player stats
            record["player_stats"] = get_match_player_stats(match_id)

            # Team season stats
            record["home_team_season"] = get_team_season_stats(record["home_team_id"], season_id)
            record["away_team_season"] = get_team_season_stats(record["away_team_id"], season_id)

            season_data.append(record)
            total_matches += 1

        # Save per season
        season_file = out / f"season_{season_id}.json"
        with open(season_file, "w") as f:
            json.dump(season_data, f, indent=2)
        logger.info(f"  Saved {len(season_data)} matches to {season_file}")

    logger.info(f"Total matches fetched: {total_matches}")   
