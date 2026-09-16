import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Data sources
STATIONS_PATH = DATA_DIR / "metro-stations-in-riyadh-by-metro-line-and-station-type-2024.json"
DISTRICTS_URL = "https://raw.githubusercontent.com/homaily/Saudi-Arabia-Regions-Cities-and-Districts/master/geojson/districts.geojson"
DISTRICTS_PATH = DATA_DIR / "riyadh-districts.geojson"
RIYADH_CITY_ID = 3

# Spatial join
METRIC_CRS = "EPSG:32638"  # UTM zone 38N: distances in metres around Riyadh
NEARBY_RADIUS_M = 150

# Retrieval
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # supports both Arabic and English
TOP_K = 5

# Generation (Amazon Bedrock)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
