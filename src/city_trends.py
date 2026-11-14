from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pymannkendall as mk

CITY_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "city_monthly_trends.csv"
CITY_LABELS = {"Riyadh": "الرياض", "Jeddah": "جدة"}
VARIABLES = {
    "T2M": {"name": "Air temperature", "name_ar": "درجة حرارة الهواء", "unit": "°C", "description": "Monthly mean air temperature at 2 m"},
    "PRECTOTCORR": {"name": "Precipitation", "name_ar": "الهطول", "unit": "mm/month", "description": "NASA POWER corrected precipitation"},
}

def load_city_data() -> pd.DataFrame:
    df = pd.read_csv(CITY_DATA_PATH, parse_dates=["date"])
    return df.sort_values(["city", "date"]).reset_index(drop=True)

def _seasonal_sen(series: pd.Series) -> float:
    series = series.dropna().sort_index()
    slopes = []
    for month in range(1, 13):
        s = series[series.index.month == month]
        years = s.index.year.to_numpy(dtype=float)
        values = s.to_numpy(dtype=float)
        for i in range(len(values) - 1):
            slopes.extend(((values[i + 1:] - values[i]) / (years[i + 1:] - years[i])).tolist())
    return float(np.median(slopes)) if slopes else float("nan")

def _bootstrap_sen_ci(series: pd.Series, n_boot: int = 120) -> tuple[float, float]:
    """Residual bootstrap interval around the seasonal trend; exploratory, not causal."""
    series = series.dropna().sort_index()
    rng = np.random.default_rng(42)
    month_models = []
    for month in range(1, 13):
        subset = series[series.index.month == month]
        years = subset.index.year.to_numpy(dtype=float)
        values = subset.to_numpy(dtype=float)
        if len(values) >= 3:
            coef = np.polyfit(years - years.mean(), values, 1)
            fitted = np.polyval(coef, years - years.mean())
            month_models.append((years, fitted, values - fitted))
    if not month_models:
        return float("nan"), float("nan")
    boot_slopes = []
    for _ in range(n_boot):
        all_slopes = []
        for years, fitted, residuals in month_models:
            sampled = fitted + rng.choice(residuals, size=len(residuals), replace=True)
            for i in range(len(sampled) - 1):
                all_slopes.extend(((sampled[i + 1:] - sampled[i]) / (years[i + 1:] - years[i])).tolist())
        if all_slopes:
            boot_slopes.append(float(np.median(all_slopes)))
    return float(np.percentile(boot_slopes, 2.5)), float(np.percentile(boot_slopes, 97.5))

def trend_result(df: pd.DataFrame, city: str, variable: str) -> dict:
    s = df.loc[df["city"].eq(city)].set_index("date")[variable].dropna().sort_index()
    result = mk.seasonal_test(s.to_numpy(dtype=float), period=12, alpha=0.05)
    ci_low, ci_high = _bootstrap_sen_ci(s)
    return {
        "city": city,
        "variable": variable,
        "trend": result.trend,
        "significant": bool(result.h),
        "p_value": float(result.p),
        "kendall_tau": float(result.Tau),
        "slope_per_year": _seasonal_sen(s),
        "slope_ci_low": ci_low,
        "slope_ci_high": ci_high,
        "start": s.index.min().strftime("%Y-%m"),
        "end": s.index.max().strftime("%Y-%m"),
        "n_months": int(s.size),
    }

def all_results(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    return pd.DataFrame([trend_result(df, city, variable) for city in ["Riyadh", "Jeddah"]])

def proactive_signal(df: pd.DataFrame, city: str, variable: str) -> dict:
    """Compare the latest complete year with each calendar month's historical climatology."""
    city_df = df.loc[df["city"].eq(city)].copy()
    climatology = city_df.groupby(city_df["date"].dt.month)[variable].mean()
    latest_year = int(city_df["date"].dt.year.max())
    latest = city_df.loc[city_df["date"].dt.year.eq(latest_year)].copy()
    latest["anomaly"] = latest.apply(lambda r: float(r[variable] - climatology.get(r["date"].month, np.nan)), axis=1)
    anomaly = float(latest["anomaly"].mean()) if not latest.empty else float("nan")
    return {"city": city, "year": latest_year, "mean_anomaly": anomaly, "months": int(latest.shape[0])}
