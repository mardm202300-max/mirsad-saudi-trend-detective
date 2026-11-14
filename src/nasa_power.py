from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

MONTHLY_URL = "https://power.larc.nasa.gov/api/temporal/monthly/regional"
DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
GEOBOUNDARIES_META_URL = "https://www.geoboundaries.org/api/current/gbOpen/SAU/ADM0/"

# Saudi Arabia bounding box. The downloaded country polygon masks out neighboring
# countries and sea cells after the regional NASA requests.
SAUDI_BBOX = {"lat_min": 16.0, "lat_max": 33.0, "lon_min": 34.0, "lon_max": 57.0}
NASA_PARAMETER = "T2M"  # Monthly mean air temperature at 2 m, degrees Celsius.


def _ranges(low: float, high: float, max_span: float = 9.5) -> list[tuple[float, float]]:
    """Split an axis into valid NASA regional requests (<10 degrees per request)."""
    result: list[tuple[float, float]] = []
    start = low
    while start < high:
        end = min(start + max_span, high)
        result.append((round(start, 6), round(end, 6)))
        start = end
    return result


def _request_json(url: str, params: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=(20, 180),
                headers={"User-Agent": "SaudiClimateTrendApp/1.0 (NASA POWER open-data project)"},
            )
            if response.status_code in (429, 500, 502, 503, 504):
                response.raise_for_status()
            response.raise_for_status()
            payload = response.json()
            if "messages" in payload and payload.get("header"):
                messages = "; ".join(str(m) for m in payload.get("messages", []))
                if messages:
                    raise RuntimeError(f"NASA POWER API returned a message: {messages}")
            return payload
        except Exception as exc:  # retry transient network/API errors only a few times
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"NASA POWER request failed after {retries} attempts: {last_error}")


def get_saudi_boundary(boundary_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download and cache a simplified open GeoJSON outline for Saudi Arabia."""
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = boundary_path.with_suffix(".metadata.json")
    if boundary_path.exists():
        geometry = json.loads(boundary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return geometry, metadata

    metadata = requests.get(GEOBOUNDARIES_META_URL, timeout=45).json()
    geojson_url = metadata.get("simplifiedGeometryGeoJSON") or metadata.get("gjDownloadURL")
    if not geojson_url:
        raise RuntimeError("لم يعثر المشروع على رابط GeoJSON لحدود السعودية في خدمة geoBoundaries.")
    response = requests.get(geojson_url, timeout=90)
    response.raise_for_status()
    geometry = response.json()
    boundary_path.write_text(json.dumps(geometry), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return geometry, metadata


def _feature_rows(feature: dict[str, Any], date_format: str) -> list[dict[str, Any]]:
    coordinates = feature.get("geometry", {}).get("coordinates", [])
    if len(coordinates) < 2:
        return []
    lon, lat = float(coordinates[0]), float(coordinates[1])
    parameter_data = feature.get("properties", {}).get("parameter", {}).get(NASA_PARAMETER, {})
    rows: list[dict[str, Any]] = []
    for date_code, raw_value in parameter_data.items():
        if not date_code.isdigit():
            continue
        try:
            if date_format == "%Y%m":
                if len(date_code) != 6 or int(date_code[4:]) not in range(1, 13):
                    continue  # discard NASA's YYYY13 annual summary
            elif len(date_code) != 8:
                continue
            timestamp = pd.to_datetime(date_code, format=date_format, errors="coerce")
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if pd.isna(timestamp):
            continue
        rows.append({"lat": lat, "lon": lon, "date": timestamp, "T2M": value})
    return rows


def _fetch_grid(
    *,
    url: str,
    start: str | int,
    end: str | int,
    date_format: str,
    time_standard: str | None = None,
) -> pd.DataFrame:
    lat_tiles = _ranges(SAUDI_BBOX["lat_min"], SAUDI_BBOX["lat_max"])
    lon_tiles = _ranges(SAUDI_BBOX["lon_min"], SAUDI_BBOX["lon_max"])
    all_rows: list[dict[str, Any]] = []
    total = len(lat_tiles) * len(lon_tiles)

    for i, (lat_min, lat_max) in enumerate(lat_tiles, start=1):
        for lon_min, lon_max in lon_tiles:
            params: dict[str, Any] = {
                "latitude-min": lat_min,
                "latitude-max": lat_max,
                "longitude-min": lon_min,
                "longitude-max": lon_max,
                "parameters": NASA_PARAMETER,
                "community": "SB",
                "start": start,
                "end": end,
                "format": "JSON",
            }
            if time_standard:
                params["time-standard"] = time_standard
            print(
                f"NASA POWER tile {i}/{len(lat_tiles)} × {lon_tiles.index((lon_min, lon_max)) + 1}/{len(lon_tiles)} "
                f"(lat {lat_min}–{lat_max}, lon {lon_min}–{lon_max}; {total} tiles total)..."
            )
            payload = _request_json(url, params)
            features = payload.get("features", [])
            if not features:
                raise RuntimeError(f"NASA POWER أرجع شبكة فارغة للمنطقة {lat_min}:{lat_max}, {lon_min}:{lon_max}.")
            for feature in features:
                all_rows.extend(_feature_rows(feature, date_format))
            time.sleep(0.25)  # stay polite to the public API

    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise RuntimeError("لم تُرجع NASA POWER أي سجلات قابلة للقراءة.")
    frame = frame.drop_duplicates(subset=["lat", "lon", "date"], keep="last")
    return frame.sort_values(["lat", "lon", "date"]).reset_index(drop=True)


def download_saudi_temperature(data_dir: Path, start_year: int = 1981, refresh: bool = False) -> Path:
    """Download historical monthly T2M plus latest complete months aggregated from daily data.

    NASA POWER's monthly meteorological archive generally ends at the previous full year.
    Daily data are aggregated for the current year through the previous complete month so the
    app's forecast starts after the freshest complete observation available.
    """
    raw_dir = data_dir / "raw"
    boundary_dir = data_dir / "boundaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = raw_dir / "saudi_power_monthly_bbox.csv"
    recent_path = raw_dir / "saudi_power_daily_recent_bbox.csv"
    merged_path = raw_dir / "saudi_power_monthly_all_bbox.csv"
    boundary_path = boundary_dir / "saudi_arabia_simplified.geojson"

    if merged_path.exists() and not refresh:
        return merged_path

    current_year = date.today().year
    monthly_end_year = current_year - 1
    if refresh or not monthly_path.exists():
        monthly = _fetch_grid(
            url=MONTHLY_URL,
            start=start_year,
            end=monthly_end_year,
            date_format="%Y%m",
        )
        monthly.to_csv(monthly_path, index=False)
    else:
        monthly = pd.read_csv(monthly_path, parse_dates=["date"])

    latest_complete_day = date.today().replace(day=1) - timedelta(days=1)
    current_year_start = date(current_year, 1, 1)
    recent_monthly = pd.DataFrame(columns=["lat", "lon", "date", "T2M"])

    if latest_complete_day >= current_year_start:
        if refresh or not recent_path.exists():
            daily = _fetch_grid(
                url=DAILY_URL,
                start=current_year_start.strftime("%Y%m%d"),
                end=latest_complete_day.strftime("%Y%m%d"),
                date_format="%Y%m%d",
                time_standard="UTC",
            )
            daily.to_csv(recent_path, index=False)
        else:
            daily = pd.read_csv(recent_path, parse_dates=["date"])
        daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()
        recent_monthly = (
            daily.groupby(["lat", "lon", "month_start"], as_index=False)["T2M"]
            .mean()
            .rename(columns={"month_start": "date"})
        )

    monthly = pd.concat([monthly, recent_monthly], ignore_index=True)
    monthly = monthly.drop_duplicates(subset=["lat", "lon", "date"], keep="last")
    monthly = monthly.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    monthly.to_csv(merged_path, index=False)

    # Fetch/cache the national border during data preparation, not on every dashboard rerun.
    get_saudi_boundary(boundary_path)
    return merged_path
