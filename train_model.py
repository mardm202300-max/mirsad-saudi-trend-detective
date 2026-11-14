import argparse
from pathlib import Path

import pandas as pd

from src.data_pipeline import CLEAN_PATH, PROJECT_ROOT
from src.lstm_model import train_evaluate_forecast


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Saudi monthly temperature LSTM.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--exclude-flagged", action="store_true", help="Exclude windows containing statistical outlier flags.")
    args = parser.parse_args()

    if not CLEAN_PATH.exists():
        raise SystemExit("ملف البيانات النظيفة غير موجود. شغّل أولاً: python prepare_data.py")
    dataset = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
    metrics = train_evaluate_forecast(
        dataset,
        PROJECT_ROOT / "artifacts",
        max_epochs=args.epochs,
        exclude_flagged=args.exclude_flagged,
    )
    print(metrics)
