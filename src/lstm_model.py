from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LOOKBACK_MONTHS = 36
HORIZON_MONTHS = 12
TRAIN_STRIDE_MONTHS = 3
SEED = 42


class TemperatureLSTM(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, horizon: int = HORIZON_MONTHS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size // 2, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _full_years(df: pd.DataFrame) -> list[int]:
    months_by_year = df.groupby(df["date"].dt.year)["date"].apply(lambda x: set(x.dt.month))
    return sorted(int(year) for year, months in months_by_year.items() if set(range(1, 13)).issubset(months))


def _make_features(
    temperature: np.ndarray,
    dates: pd.DatetimeIndex,
    lat: float,
    lon: float,
    mean: float,
    std: float,
) -> np.ndarray:
    months = dates.month.to_numpy(dtype=np.float32)
    sin_month = np.sin(2.0 * np.pi * (months - 1.0) / 12.0)
    cos_month = np.cos(2.0 * np.pi * (months - 1.0) / 12.0)
    temp_scaled = ((temperature - mean) / std).astype(np.float32)
    lat_scaled = np.full(len(dates), (lat - 24.0) / 10.0, dtype=np.float32)
    lon_scaled = np.full(len(dates), (lon - 45.0) / 12.0, dtype=np.float32)
    return np.column_stack([temp_scaled, sin_month, cos_month, lat_scaled, lon_scaled]).astype(np.float32)


def _prepare_grid_series(df: pd.DataFrame):
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["T2M"] = pd.to_numeric(data["T2M"], errors="coerce")
    if "outlier_flag" not in data:
        data["outlier_flag"] = False
    data["outlier_flag"] = data["outlier_flag"].fillna(False).astype(bool)

    full_dates = pd.date_range(data["date"].min(), data["date"].max(), freq="MS")
    grids = []
    for (lat, lon), group in data.groupby(["lat", "lon"], sort=False):
        indexed = group.set_index("date").sort_index()
        temperatures = indexed["T2M"].reindex(full_dates).to_numpy(dtype=float)
        flagged = indexed["outlier_flag"].reindex(full_dates).fillna(False).to_numpy(dtype=bool)
        grids.append((float(lat), float(lon), full_dates, temperatures, flagged))
    return grids


def _collect_samples(
    grids,
    *,
    mean: float,
    std: float,
    train_end: pd.Timestamp | None = None,
    special_year: int | None = None,
    exclude_flagged: bool = False,
    stride: int = TRAIN_STRIDE_MONTHS,
):
    xs, ys, baselines = [], [], []
    for lat, lon, dates, temps, flagged in grids:
        last_start = len(dates) - HORIZON_MONTHS
        for target_start in range(LOOKBACK_MONTHS, last_start + 1):
            target_dates = dates[target_start : target_start + HORIZON_MONTHS]
            if len(target_dates) != HORIZON_MONTHS:
                continue

            if special_year is not None:
                if target_dates[0].year != special_year or target_dates[0].month != 1:
                    continue
                if target_dates[-1].year != special_year or target_dates[-1].month != 12:
                    continue
            else:
                if target_dates[-1] > train_end:
                    continue
                if (target_start - LOOKBACK_MONTHS) % stride != 0:
                    continue

            input_values = temps[target_start - LOOKBACK_MONTHS : target_start]
            target_values = temps[target_start : target_start + HORIZON_MONTHS]
            input_flags = flagged[target_start - LOOKBACK_MONTHS : target_start]
            target_flags = flagged[target_start : target_start + HORIZON_MONTHS]

            if not (np.isfinite(input_values).all() and np.isfinite(target_values).all()):
                continue
            if exclude_flagged and (input_flags.any() or target_flags.any()):
                continue

            x = _make_features(input_values, dates[target_start - LOOKBACK_MONTHS : target_start], lat, lon, mean, std)
            y = ((target_values - mean) / std).astype(np.float32)
            xs.append(x)
            ys.append(y)
            # Seasonal-naive baseline: repeat the previous 12 months for the next 12.
            baselines.append(input_values[-HORIZON_MONTHS:].astype(np.float32))

    if not xs:
        return (
            np.empty((0, LOOKBACK_MONTHS, 5), dtype=np.float32),
            np.empty((0, HORIZON_MONTHS), dtype=np.float32),
            np.empty((0, HORIZON_MONTHS), dtype=np.float32),
        )
    return np.stack(xs), np.stack(ys), np.stack(baselines)


def _train_one_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_epochs: int,
    patience: int,
    log: Callable[[str], None],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemperatureLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    x_val_tensor = torch.from_numpy(x_val).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)

    best_loss = float("inf")
    best_epoch = 1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_tensor), y_val_tensor).cpu())
        log(f"Epoch {epoch}/{max_epochs} — train Huber={np.mean(train_losses):.4f}, validation Huber={val_loss:.4f}")

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_loss, device


def _fit_fixed_epochs(x: np.ndarray, y: np.ndarray, epochs: int, log: Callable[[str], None]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemperatureLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=512,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        log(f"Final fit epoch {epoch}/{max(1, epochs)} — Huber={np.mean(losses):.4f}")
    return model, device


def train_evaluate_forecast(
    df: pd.DataFrame,
    artifacts_dir: Path,
    *,
    max_epochs: int = 15,
    patience: int = 4,
    exclude_flagged: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Chronological backtest, seasonal-naive comparison, final LSTM fit and 12-month forecast."""
    emit = log or print
    _seed_everything()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    years = _full_years(df)
    if len(years) < 4:
        raise ValueError("نحتاج أربع سنوات كاملة على الأقل لتقسيم التدريب والتحقق والاختبار زمنياً.")
    test_year = years[-1]
    validation_year = test_year - 1
    train_end_year = test_year - 2
    emit(
        f"تقسيم زمني للاختبار: تدريب حتى {train_end_year}، تحقق {validation_year}، اختبار {test_year}. "
        "الفترة الجزئية الحالية تبقى للتدريب النهائي والتوقع فقط."
    )

    grids = _prepare_grid_series(df)
    train_temperatures = df.loc[df["date"].dt.year <= train_end_year, "T2M"].astype(float)
    if exclude_flagged and "outlier_flag" in df:
        train_temperatures = df.loc[
            (df["date"].dt.year <= train_end_year) & (~df["outlier_flag"].fillna(False).astype(bool)), "T2M"
        ].astype(float)
    mean = float(train_temperatures.mean())
    std = float(train_temperatures.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        raise ValueError("تعذر تقدير مقياس درجة الحرارة للتطبيع.")

    x_train, y_train, _ = _collect_samples(
        grids,
        mean=mean,
        std=std,
        train_end=pd.Timestamp(f"{train_end_year}-12-01"),
        exclude_flagged=exclude_flagged,
    )
    x_val, y_val, _ = _collect_samples(
        grids,
        mean=mean,
        std=std,
        special_year=validation_year,
        exclude_flagged=exclude_flagged,
    )
    x_test, y_test, baseline_test = _collect_samples(
        grids,
        mean=mean,
        std=std,
        special_year=test_year,
        exclude_flagged=exclude_flagged,
    )

    if min(len(x_train), len(x_val), len(x_test)) == 0:
        raise ValueError(
            f"عينات غير كافية للتقسيم الزمني: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}. "
            "تحقق من اكتمال الأشهر في ملف البيانات."
        )
    emit(f"العينات — تدريب: {len(x_train):,}، تحقق: {len(x_val):,}، اختبار: {len(x_test):,}.")

    model, best_epoch, best_val_loss, device = _train_one_model(
        x_train,
        y_train,
        x_val,
        y_val,
        max_epochs=max_epochs,
        patience=patience,
        log=emit,
    )
    model.eval()
    with torch.no_grad():
        pred_test_scaled = model(torch.from_numpy(x_test).to(device)).cpu().numpy()
    pred_test = pred_test_scaled * std + mean
    actual_test = y_test * std + mean
    lstm_mae = float(mean_absolute_error(actual_test.ravel(), pred_test.ravel()))
    lstm_rmse = float(np.sqrt(mean_squared_error(actual_test.ravel(), pred_test.ravel())))
    baseline_mae = float(mean_absolute_error(actual_test.ravel(), baseline_test.ravel()))
    baseline_rmse = float(np.sqrt(mean_squared_error(actual_test.ravel(), baseline_test.ravel())))
    lstm_abs_error = np.abs(actual_test - pred_test)
    baseline_abs_error = np.abs(actual_test - baseline_test)
    lstm_within_1c_pct = float(np.mean(lstm_abs_error <= 1.0) * 100.0)
    lstm_within_2c_pct = float(np.mean(lstm_abs_error <= 2.0) * 100.0)
    baseline_within_1c_pct = float(np.mean(baseline_abs_error <= 1.0) * 100.0)
    baseline_within_2c_pct = float(np.mean(baseline_abs_error <= 2.0) * 100.0)

    # Refit scaler and final model on all observed months after preserving the untouched test score.
    final_mean = float(df["T2M"].astype(float).mean())
    final_std = float(df["T2M"].astype(float).std(ddof=0))
    final_cutoff = pd.to_datetime(df["date"].max())
    x_final, y_final, _ = _collect_samples(
        grids,
        mean=final_mean,
        std=final_std,
        train_end=final_cutoff,
        exclude_flagged=exclude_flagged,
    )
    if len(x_final) == 0:
        raise ValueError("لم تتكون عينات كافية لإعادة تدريب النموذج على كل السجل المتاح.")
    final_model, final_device = _fit_fixed_epochs(x_final, y_final, best_epoch, emit)

    last_date = pd.to_datetime(df["date"].max())
    forecast_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=HORIZON_MONTHS, freq="MS")
    forecast_rows = []
    final_model.eval()
    with torch.no_grad():
        for lat, lon, dates, temps, flagged in grids:
            if len(temps) < LOOKBACK_MONTHS or not np.isfinite(temps[-LOOKBACK_MONTHS:]).all():
                continue
            if exclude_flagged and flagged[-LOOKBACK_MONTHS:].any():
                continue
            sequence = _make_features(
                temps[-LOOKBACK_MONTHS:], dates[-LOOKBACK_MONTHS:], lat, lon, final_mean, final_std
            )
            prediction_scaled = final_model(torch.from_numpy(sequence[None, ...]).to(final_device)).cpu().numpy()[0]
            prediction = prediction_scaled * final_std + final_mean
            for forecast_date, value in zip(forecast_dates, prediction):
                forecast_rows.append(
                    {"date": forecast_date, "lat": lat, "lon": lon, "predicted_T2M": float(value)}
                )

    forecast = pd.DataFrame(forecast_rows)
    if forecast.empty:
        raise ValueError("تعذر إنشاء توقعات؛ تحقق من توفر 36 شهراً حديثاً لكل خلية.")
    forecast.to_csv(artifacts_dir / "forecast_saudi_12_months.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "model": "LSTM multi-grid direct 12-month forecast",
        "variable": "T2M",
        "units": "°C",
        "lookback_months": LOOKBACK_MONTHS,
        "horizon_months": HORIZON_MONTHS,
        "train_end_year": train_end_year,
        "validation_year": validation_year,
        "test_year": test_year,
        "n_train_samples": int(len(x_train)),
        "n_validation_samples": int(len(x_val)),
        "n_test_samples": int(len(x_test)),
        "lstm_test_mae_c": lstm_mae,
        "lstm_test_rmse_c": lstm_rmse,
        "seasonal_naive_test_mae_c": baseline_mae,
        "seasonal_naive_test_rmse_c": baseline_rmse,
        "n_test_monthly_predictions": int(lstm_abs_error.size),
        "lstm_within_1c_pct": lstm_within_1c_pct,
        "lstm_within_2c_pct": lstm_within_2c_pct,
        "seasonal_naive_within_1c_pct": baseline_within_1c_pct,
        "seasonal_naive_within_2c_pct": baseline_within_2c_pct,
        "best_epoch": int(best_epoch),
        "best_validation_huber_scaled": float(best_val_loss),
        "device": str(device),
        "final_fit_samples": int(len(x_final)),
        "forecast_start": forecast["date"].min().strftime("%Y-%m"),
        "forecast_end": forecast["date"].max().strftime("%Y-%m"),
        "forecast_grid_cells": int(forecast[["lat", "lon"]].drop_duplicates().shape[0]),
        "exclude_flagged_outliers": bool(exclude_flagged),
        "warning": "اتجاهات النموذج التجريبية ليست توقعاً رسمياً من NASA؛ اختُبر النموذج على سنة كاملة لم تدخل التدريب.",
    }
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    torch.save(
        {
            "state_dict": final_model.cpu().state_dict(),
            "mean": final_mean,
            "std": final_std,
            "lookback_months": LOOKBACK_MONTHS,
            "horizon_months": HORIZON_MONTHS,
            "input_size": 5,
            "hidden_size": 64,
            "training_last_observation": last_date.strftime("%Y-%m"),
            "test_year": test_year,
        },
        artifacts_dir / "saudi_temperature_lstm.pt",
    )

    emit(
        f"الاختبار {test_year}: LSTM MAE={lstm_mae:.3f}°C مقابل خط الأساس MAE={baseline_mae:.3f}°C. "
        f"التنبؤ المحفوظ من {metrics['forecast_start']} إلى {metrics['forecast_end']}."
    )
    return metrics
