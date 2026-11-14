from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk

from src.cleaning import national_weighted_monthly


def _seasonal_sen_slope(series: pd.Series) -> float:
    slopes: list[float] = []
    for month in range(1, 13):
        subset = series[series.index.month == month].dropna()
        years = subset.index.year.to_numpy(dtype=float)
        values = subset.to_numpy(dtype=float)
        for i in range(len(values) - 1):
            slopes.extend(((values[i + 1 :] - values[i]) / (years[i + 1 :] - years[i])).tolist())
    return float(np.median(slopes)) if slopes else float("nan")


def _test(series: pd.Series) -> dict:
    series = series.dropna().sort_index()
    if len(series) < 24:
        return {"trend": "insufficient data", "p_value": np.nan, "tau": np.nan, "slope_per_year": np.nan}
    result = mk.seasonal_test(series.to_numpy(dtype=float), period=12, alpha=0.05)
    return {
        "trend": result.trend,
        "significant": bool(result.h),
        "p_value": float(result.p),
        "kendall_tau": float(result.Tau),
        "slope_per_year": _seasonal_sen_slope(series),
    }


def compute_trends(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Compute national and cell-level seasonal Mann–Kendall trends."""
    output_dir.mkdir(parents=True, exist_ok=True)
    national = national_weighted_monthly(df).set_index("date")["national_t2m"]
    national_result = _test(national)
    national_result.update(
        {
            "variable": "T2M",
            "units": "°C",
            "start": national.index.min().strftime("%Y-%m"),
            "end": national.index.max().strftime("%Y-%m"),
            "n_months": int(len(national)),
        }
    )

    rows = []
    for (lat, lon), group in df.groupby(["lat", "lon"], sort=False):
        series = group.set_index("date")["T2M"].sort_index()
        result = _test(series)
        rows.append({"lat": float(lat), "lon": float(lon), **result})
    by_grid = pd.DataFrame(rows)
    by_grid.to_csv(output_dir / "trend_by_grid.csv", index=False, encoding="utf-8-sig")
    (output_dir / "national_trend.json").write_text(
        json.dumps(national_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return by_grid, national_result
