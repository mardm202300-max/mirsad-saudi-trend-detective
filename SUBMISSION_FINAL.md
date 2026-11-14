# Mirsad — NASA Trend Detective

## One-line pitch

Mirsad turns NASA POWER monthly data into an understandable Earth-system trend investigation: users compare Riyadh and Jeddah, measure seasonal Sen's slope, test significance with Mann–Kendall, and inspect early-warning anomalies on a real map.

## Challenge fit

**Selected challenge:** Be An Earth System Trend Detective!

The tool answers four questions: what changed, where did it change, how fast did it change, and is the evidence statistically significant? The central analysis is the seasonal Mann–Kendall test and Sen's slope. The LSTM forecast is an optional exploratory layer, not a replacement for the trend test.

## Why it matters

Climate data are abundant but difficult for non-specialists to interpret. Mirsad reduces the first analysis step to a transparent comparison of two contrasting Saudi locations: an inland city and a coastal city. A user can inspect temperature or precipitation, see the long record, read the slope and p-value, and identify whether the latest complete year is above or below its calendar-month baseline.

## NASA data and method

- **Source:** NASA POWER monthly point product.
- **Variables:** `T2M` (2 m air temperature, °C) and `PRECTOTCORR` (corrected precipitation, mm/month).
- **Locations:** Riyadh and Jeddah, with coordinates recorded in `data/processed/city_monthly_trends.csv`.
- **Trend test:** seasonal Mann–Kendall with a 12-month period.
- **Effect size:** seasonal Sen's slope, reported per year.
- **Uncertainty:** exploratory empirical 95% interval from the seasonal pairwise slopes used by Sen's estimator.
- **Proactive signal:** latest complete year minus the long-term calendar-month climatology.

The trend test indicates evidence for or against a monotonic seasonal trend; it does not prove causation. NASA POWER values are gridded/reanalysis-derived estimates, not local station observations.

## What is novel

The project combines an explainable statistical workflow with an accessible visual interface. Machine learning is deliberately secondary: the user first sees the transparent trend evidence and baseline anomaly, then can inspect the experimental LSTM forecast and its chronological holdout metrics.

## Demo path

1. Open the English interface.
2. Select **Air temperature** and compare both cities.
3. Point to the historical line chart.
4. Show the Sen's slope, p-value, Kendall tau, and 95% interval cards.
5. Switch to **Precipitation** and show that not every apparent change is statistically significant.
6. Point to the proactive signal cards for the latest complete year.
7. Open the real map and explain that map points are geographic context for the NASA data.

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The bundled city comparison CSV allows the Trend Detective view to open immediately. Refreshing the national grid requires internet access to NASA POWER; the real basemap requires internet access to OpenStreetMap or Esri tiles.

## Team and links

- **Team name and members:** Enter the registered team details in the official submission form.
- **Demo video:** `media/Mirsad_English_Explainer.mp4` (included in this repository).
- **Repository:** The public GitHub page containing this README.
- **Challenge page:** https://www.spaceappschallenge.org/2026/challenges/earth-system-trend-detective/
