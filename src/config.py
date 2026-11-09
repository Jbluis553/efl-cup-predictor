import os
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.thestatsapi.com/api/football"
API_KEY = os.getenv("THESTATSAPI_KEY", "")
COMPETITION_ID = os.getenv("COMPETITION_ID", "comp_2504")
SEASONS = int(os.getenv("SEASONS", "10"))

# Scale plan: 1,000 req/min
RATE_LIMIT_RPM = 1000
REQUEST_DELAY = 60 / RATE_LIMIT_RPM  # ~0.06s

BOOKMAKERS = ["pinnacle", "bet365", "paddy_power", "betfair_sportsbook", "kambi"]

MARKETS = ["match_odds", "total_goals", "btts", "corners", "asian_handicap"]

DATA_DIR = "data"
MODEL_DIR = "models"   
