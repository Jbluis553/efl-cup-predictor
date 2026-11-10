"""Configuración central del predictor EFL Cup.

Conserva todas las variables originales y añade validación, límites de
cuotas, parámetros de fuerza de equipos y nombres de artefactos.
"""
import os

from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.thestatsapi.com/api/football"
API_KEY = os.getenv("THESTATSAPI_KEY", "")
COMPETITION_ID = os.getenv("COMPETITION_ID", "comp_2504")
SEASONS = int(os.getenv("SEASONS", "10"))

# Use a conservative request pace to avoid bursts against the API rate limit.
RATE_LIMIT_RPM = 60
REQUEST_DELAY = 60 / RATE_LIMIT_RPM

BOOKMAKERS = ["pinnacle", "bet365", "paddy_power", "betfair_sportsbook", "kambi"]

MARKETS = ["match_odds", "total_goals", "btts", "corners", "asian_handicap"]

DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# --- Añadido: fuerza de equipos (Elo + forma reciente, sin leakage) ---
ELO_BASE_RATING = 1500.0
ELO_K_FACTOR = 20.0
FORM_WINDOW = 5

# --- Añadido: backtesting honesto ---
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.03"))
MIN_ODDS = 1.05
MAX_ODDS = 1000.0
BACKTEST_BOOKMAKER = os.getenv("BACKTEST_BOOKMAKER", "pinnacle")

# --- Añadido: artefactos del modelo ---
FEATURES_FILE = "features.json"
OOF_FILE = "oof_predictions.csv"

# --- Añadido: ventana de fixtures futuros para el modo predict ---
FIXTURES_LOOKAHEAD_DAYS = int(os.getenv("FIXTURES_LOOKAHEAD_DAYS", "30"))


def validate_api_key() -> None:
    """Fallar rápido con un mensaje claro si falta la API key."""
    if not API_KEY:
        raise RuntimeError(
            "THESTATSAPI_KEY no está definida. Expórtala en el entorno "
            "o en un archivo .env antes de ejecutar."
        )
