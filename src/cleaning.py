from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union


def load_country_geometry(path: Path):
    geojson = json.loads(path.read_text(encoding="utf-8"))
    kind = geojson.get("type")
    if kind == "FeatureCollection":
        geometries = [shape(feature["geometry"]) for feature in geojson.get("features", []) if feature.get("geometry")]
        if not geometries:
            raise ValueError("ملف حدود السعودية لا يحتوي أي مضلع GeoJSON.")
        return unary_union(geometries)
    if kind == "Feature":
        return shape(geojson["geometry"])
    return shape(geojson)


def seasonal_outlier_flags(df: pd.DataFrame, threshold: float = 4.5) -> pd.Series:
    """Flag robust outliers within each grid-cell/calendar-month group.

    Values are flagged, not erased: real heat/cold extremes are scientifically meaningful.
    The model can optionally exclude flagged windows; the raw measurement remains auditable.
    """
    month = df["date"].dt.month
    grouped = df.assign(_calendar_month=month).groupby(["lat", "lon", "_calendar_month"])["T2M"]
    median = grouped.transform("median")
    mad = grouped.transform(lambda x: np.median(np.abs(x - np.median(x))))
    modified_z = 0.6745 * (df["T2M"] - median).abs() / mad.replace(0, np.nan)
    flags = modified_z > threshold

    # If MAD is zero, use a conservative IQR-based rule as a fallback.
    q1 = grouped.transform(lambda x: x.quantile(0.25))
    q3 = grouped.transform(lambda x: x.quantile(0.75))
    robust_scale = (q3 - q1) / 1.349
    fallback = mad.eq(0) & robust_scale.gt(0) & ((df["T2M"] - median).abs() > 6.0 * robust_scale)
    return (flags.fillna(False) | fallback.fillna(False)).astype(bool)


def clean_saudi_data(raw_path: Path, boundary_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean NASA grid-point data, mask to Saudi Arabia, and annotate suspected outliers."""
    df = pd.read_csv(raw_path, parse_dates=["date"])
    required = {"lat", "lon", "date", "T2M"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"الأعمدة المطلوبة غير موجودة: {sorted(missing_columns)}")

    input_rows = len(df)
    df["T2M"] = pd.to_numeric(df["T2M"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["lat", "lon", "date", "T2M"])
    df = df.drop_duplicates(subset=["lat", "lon", "date"], keep="last")

    # NASA POWER fill value and physical sanity limits. These are data errors, not climate extremes.
    invalid = (df["T2M"] <= -90.0) | (df["T2M"] > 65.0)
    invalid_count = int(invalid.sum())
    df = df.loc[~invalid].copy()

    border = load_country_geometry(boundary_path)
    inside = [border.covers(Point(float(lon), float(lat))) for lon, lat in zip(df["lon"], df["lat"])]
    outside_count = int((~pd.Series(inside, index=df.index)).sum())
    df = df.loc[inside].copy()

    df["outlier_flag"] = seasonal_outlier_flags(df)
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df = df.sort_values(["lat", "lon", "date"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("لم تبق أي بيانات بعد تنظيفها وتطبيق حدود السعودية.")

    summary: dict[str, Any] = {
        "input_rows": input_rows,
        "clean_rows": int(len(df)),
        "invalid_physical_rows_removed": invalid_count,
        "outside_saudi_rows_removed": outside_count,
        "statistical_outliers_flagged_but_retained": int(df["outlier_flag"].sum()),
        "grid_cells": int(df[["lat", "lon"]].drop_duplicates().shape[0]),
        "first_month": df["date"].min().strftime("%Y-%m"),
        "last_month": df["date"].max().strftime("%Y-%m"),
        "missing_t2m_after_cleaning": int(df["T2M"].isna().sum()),
        "quality_note": "قيم T2M المتطرفة إحصائياً معلّمة ولا تُحذف تلقائياً حتى لا نمحو موجات حر حقيقية.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    output_path.with_suffix(".quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return df, summary


def national_weighted_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate national monthly mean using cosine-latitude cell-area weights."""
    data = df.copy()
    data["area_weight"] = np.cos(np.deg2rad(data["lat"].astype(float))).clip(lower=0)
    data["weighted_t2m"] = data["T2M"] * data["area_weight"]
    monthly = data.groupby("date", as_index=False).agg(
        weighted_sum=("weighted_t2m", "sum"),
        weight_sum=("area_weight", "sum"),
        grid_cells=("T2M", "count"),
    )
    monthly["national_t2m"] = monthly["weighted_sum"] / monthly["weight_sum"]
    return monthly[["date", "national_t2m", "grid_cells"]].sort_values("date")
