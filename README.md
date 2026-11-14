# Mirsad — Saudi Arabia Earth-System Trend Detective

Mirsad is an interactive Streamlit application for investigating Earth-system trends across Saudi Arabia. It follows a simple workflow: **Detect → Explain → Act**.

## Demo video

[Watch or download the English explainer](media/Mirsad_English_Explainer.mp4). This motion-graphics overview uses AI-generated narration and is not a live screen recording of the dashboard.

The national view scans NASA POWER grid data over Saudi Arabia and displays an interactive heatmap for recent temperature values and long-term Sen’s slope. The Trend Detective view compares Riyadh and Jeddah for air temperature and precipitation, then reports seasonal Mann–Kendall statistics, p-value, Kendall’s tau, Sen’s slope, and an exploratory slope interval. Proactive cards compare the latest complete year with the usual monthly climatology. An optional experimental LSTM forecast is included as a separate layer.

> The forecast is experimental and educational. It is not an official NASA forecast and does not replace physical climate or weather models.

> NASA POWER values are gridded/reanalysis-derived estimates. City markers are reference locations; they are not local weather-station measurements.

## Quick start

The project requires Python 3.11 or 3.12 and an internet connection for package installation and external map tiles.

### Windows

```bash
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically. Real map tiles require internet access. If a map looks empty initially, wait 5–10 seconds for OpenStreetMap or Esri tiles to load.

## Main features

- Saudi-wide NASA POWER temperature heatmap.
- Interactive map with street-map and satellite layers.
- National grid-cell Sen’s slope trend view.
- Riyadh and Jeddah temperature and precipitation comparison.
- Seasonal Mann–Kendall test with p-value and Kendall’s tau.
- Seasonal Sen’s slope and an exploratory confidence interval.
- Proactive anomaly signals based on the latest complete year.
- Optional LSTM forecast benchmarked against a seasonal-naive baseline.
- Arabic and English user interface.

## Analysis and model

The national and city trend analysis uses a 12-month seasonal Mann–Kendall test and seasonal Sen’s slope. The LSTM uses the latest 36 months and produces an optional 12-month forecast for the available grid locations. Evaluation follows a chronological holdout design and compares the LSTM with a seasonal-naive baseline.

In the bundled experimental run, the LSTM achieved an MAE of 0.91°C on the 2025 holdout year, compared with 1.35°C for the seasonal-naive baseline. This is one experimental evaluation and must not be presented as an official NASA prediction.

Statistical significance does not prove causation. Extreme values are not automatically deleted because a heatwave may be a real climate signal rather than a data error.

## Project structure

```text
saudi-climate-ai-code/
├── app.py
├── prepare_data.py
├── train_model.py
├── requirements.txt
├── data/
│   ├── raw/
│   ├── boundaries/
│   └── processed/
├── artifacts/
│   ├── national_trend.json
│   ├── trend_by_grid.csv
│   ├── metrics.json
│   ├── forecast_saudi_12_months.csv
│   └── saudi_temperature_lstm.pt
└── src/
    ├── nasa_power.py
    ├── cleaning.py
    ├── trend_analysis.py
    ├── data_pipeline.py
    ├── lstm_model.py
    └── city_trends.py
```

## Updating data and retraining

```bash
python prepare_data.py
python train_model.py
python train_model.py --epochs 20 --exclude-flagged
```

The application also includes controls for updating NASA data and retraining the forecast. Data refresh requires internet access and may take longer than opening the dashboard.

## Data sources

- NASA POWER Monthly API: https://power.larc.nasa.gov/docs/services/api/temporal/monthly/
- NASA POWER Daily API: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- NASA POWER Regional Data Requests: https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/
- geoBoundaries: https://www.geoboundaries.org/
- OpenStreetMap: https://www.openstreetmap.org/
- Esri World Imagery: https://www.arcgis.com/
- pyMannKendall: https://github.com/mmhs013/pyMannKendall

## Submission files

- `PROJECT_PAGE_COPY.md` — English text ready for the NASA Space Apps project page.
- `DEMO_SCRIPT_30S_EN.md` — 30-second English demo script.
- `SUBMISSION_HOW_TO_AR.md` — Arabic upload and publishing instructions.
- `SUBMISSION_CHECKLIST.md` — final submission checklist.
- `README_AR.md` — Arabic project documentation.
