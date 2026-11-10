import json
import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

import requests
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from . import config

logger = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({
    "Authorization": f"Bearer {config.API_KEY}",
    "Content-Type": "application/json",
})


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        return status == 408 or status == 429 or (status is not None and status >= 500)
    return False


_fallback_wait = wait_exponential(multiplier=2, min=2, max=60)


def _wait_before_retry(retry_state: RetryCallState) -> float:
    """Honor the API's Retry-After instruction, otherwise back off exponentially."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0), 300)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return min(max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0), 300)
                except (TypeError, ValueError, OverflowError):
                    pass
    return _fallback_wait(retry_state)


@retry(
    stop=stop_after_attempt(5),
    wait=_wait_before_retry,
    retry=retry_if_exception(_is_retryable_error),
    reraise=True,
)
def _get(endpoint: str, params: Optional[dict] = None) -> dict:
    url = f"{config.API_BASE}/{endpoint}"
    response = _session.get(url, params=params, timeout=15)
    response.raise_for_status()
    time.sleep(config.REQUEST_DELAY)
    return response.json()


def get_seasons() -> list[dict]:
    """Fetch all seasons for the configured competition."""
    logger.info("Fetching seasons for %s", config.COMPETITION_ID)
    data = _get(f"competitions/{config.COMPETITION_ID}/seasons")
    return data.get("data", data)


def get_matches(season_id: str, page: int = 1, per_page: int = 100) -> list[dict]:
    """Fetch all matches for one season, using API pagination."""
    all_matches = []
    while True:
        data = _get("matches", {
            "competition_id": config.COMPETITION_ID,
            "season_id": season_id,
            "page": page,
            "per_page": per_page,
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
    logger.info("Season %s: %s matches", season_id, len(all_matches))
    return all_matches


def get_match_odds(match_id: str) -> dict:
    """Fetch pre-match odds for one match."""
    try:
        return _get(f"matches/{match_id}/odds")
    except Exception as exc:
        logger.warning("Odds unavailable for %s: %s", match_id, exc)
        return {}


def fetch_all_data(seasons: list[dict], output_dir: str = config.DATA_DIR) -> None:
    """Fetch match metadata and odds needed to build and evaluate the models.

    Avoid requesting stats, shotmaps, player stats, and team stats for every
    historical fixture: those data are not used by feature_engineering and
    multiply the API calls per match, making scheduled training needlessly slow.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    total_matches = 0

    for season in seasons:
        season_id = season.get("id", season.get("season_id", ""))
        logger.info("=== Season: %s ===", season_id)
        matches = get_matches(season_id)
        season_data = []

        for index, match in enumerate(matches, start=1):
            home_team = match.get("home_team") or {}
            away_team = match.get("away_team") or {}
            record = {
                "match_id": match.get("id", match.get("match_id", "")),
                "season_id": season_id,
                "home_team_id": home_team.get("id"),
                "away_team_id": away_team.get("id"),
                "home_team_name": home_team.get("name"),
                "away_team_name": away_team.get("name"),
                "kickoff": match.get("kickoff"),
                "status": match.get("status"),
                "score_home": (match.get("score") or {}).get("home"),
                "score_away": (match.get("score") or {}).get("away"),
                "odds_available": match.get("odds_available", False),
            }

            if match.get("status") in ("finished", "ft", "completed"):
                if record["odds_available"]:
                    record["odds"] = get_match_odds(record["match_id"])
                total_matches += 1

            season_data.append(record)
            if index % 50 == 0:
                logger.info("Season %s: processed %s/%s matches", season_id, index, len(matches))

        season_file = output / f"season_{season_id}.json"
        with season_file.open("w", encoding="utf-8") as handle:
            json.dump(season_data, handle, indent=2)
        logger.info("Saved %s matches to %s", len(season_data), season_file)

    logger.info("Total completed matches fetched: %s", total_matches)
