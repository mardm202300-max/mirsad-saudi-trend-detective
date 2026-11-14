# Mirsad — 2026 NASA Space Apps judging readiness

## What was improved in this version

- **International usability:** the Streamlit dashboard now has an Arabic/English switch. The core controls, headings, map-layer selector, and key dashboard sections are available in both languages.
- **Real geographic context:** the former Plotly geographic view is now complemented by a real interactive map with OpenStreetMap and CartoDB basemap layers, a Saudi Arabia GeoJSON boundary, NASA POWER grid-cell markers, coordinate tooltips, reference-center markers, zoom, and layer control.
- **Scientific honesty:** the interface continues to label the grid values as NASA POWER estimates, distinguishes reference centers from ground stations, and does not claim that an experimental LSTM is an official NASA forecast.

## Main gaps that still affect a winning submission

### Impact
The project demonstrates a climate signal, but the submission should show who uses it and what changes after using it. Add a short decision story: for example, a heat-risk analyst selects a region, sees the historical trend and uncertainty, and exports a briefing for a municipality or public-health team. Quantify the intended benefit without inventing outcomes: reduced analysis time, earlier screening, or better prioritization of field investigation.

### Creativity
A temperature dashboard alone is familiar. The strongest differentiator is the combination of a transparent regional NASA POWER pipeline, a trend-detection layer, and a forecast layer that is explicitly benchmarked against a seasonal-naive baseline. In the demo, make the “why this is different” moment visible within the first minute.

### Validity
The code already includes useful safeguards: spatial clipping, missing-value checks, outlier flags, seasonal Mann–Kendall/Sen trend analysis, chronological holdout evaluation, and a baseline comparison. Before submission, run the refresh path from a clean environment, record the data date range, show the actual test metrics, and explain that MERRA-2/grid estimates are not station observations. Do not present the LSTM as causal or physically based.

### Relevance
The README identifies the Earth-system trend-detection challenge. The project page should explicitly map each feature to the selected challenge prompt and list the NASA data product, variable (`T2M`), spatial resolution/limitations, and how the result helps a real Earth-system question in Saudi Arabia.

### Presentation
Prepare a short screen recording with this order: the problem in one sentence, the real map, one historical trend, the holdout-vs-baseline result, and the intended user decision. Keep the first view uncluttered. The project submission should include a concise README, setup instructions, data sources, limitations, team roles, and a public demo link if available.

## Submission checklist

1. Register the team and submit through the official Space Apps project page before the deadline.
2. Name the exact 2026 challenge selected; do not rely on a general climate description.
3. Add a two-minute demo showing Arabic/English switching and the real basemap.
4. Include the measured test-year MAE/RMSE and seasonal-naive comparison exactly as produced by the code.
5. Include NASA POWER and boundary-data citations, plus a clear note that the map is gridded analysis data rather than local station data.
6. Add one concrete target user and one measurable success criterion.
7. Test the ZIP on a fresh Python environment and confirm that `streamlit run app.py` works with internet access for basemap tiles and data refresh.

## Official references

- NASA Space Apps 2026: https://www.spaceappschallenge.org/2026/
- 2026 Challenges: https://www.spaceappschallenge.org/2026/challenges/
- Judging & Awards Guide: https://www.spaceappschallenge.org/resources/judging-awards-guide/
- Project Submission Guide: https://www.spaceappschallenge.org/resources/project-submission-guide/
