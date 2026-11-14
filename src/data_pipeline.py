from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.cleaning import clean_saudi_data
from src.nasa_power import download_saudi_temperature
from src.trend_analysis import compute_trends


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_PATH = DATA_DIR / "processed" / "saudi_t2m_clean.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def run_pipeline(refresh: bool = False, log: Callable[[str], None] | None = None):
    """Download NASA POWER data, mask it to Saudi Arabia, clean and analyze trends."""
    emit = log or print
    raw_path = download_saudi_temperature(DATA_DIR, refresh=refresh)
    emit(f"البيانات الخام: {raw_path}")
    boundary_path = DATA_DIR / "boundaries" / "saudi_arabia_simplified.geojson"
    cleaned, summary = clean_saudi_data(raw_path, boundary_path, CLEAN_PATH)
    emit(json.dumps(summary, ensure_ascii=False, indent=2))
    _, national_trend = compute_trends(cleaned, ARTIFACTS_DIR)
    emit("الاتجاه الوطني: " + json.dumps(national_trend, ensure_ascii=False))
    return cleaned, summary
