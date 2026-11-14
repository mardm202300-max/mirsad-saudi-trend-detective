# NASA Space Apps 2026 — Project Page Copy

Use the following text in the official project submission page. Replace only the fields marked `REPLACE`.

## Project name

Mirsad — Saudi Arabia Earth-System Trend Detective

## Selected challenge

Be An Earth System Trend Detective!

Challenge URL: https://www.spaceappschallenge.org/2026/challenges/earth-system-trend-detective/

## Short description

Mirsad is an interactive Saudi-wide climate trend detective. It scans NASA POWER grid data across Saudi Arabia, visualizes temperature and long-term Sen’s slope on an interactive heatmap, and explains the evidence with seasonal Mann–Kendall statistics. Users can then compare Riyadh and Jeddah, inspect proactive seasonal anomalies, and explore an optional experimental forecast.

## The problem

Earth-system data are abundant, but a non-specialist still needs a simple way to answer four questions: What changed? Where did it change? How fast did it change? Is the trend statistically significant? Mirsad turns those questions into a visual workflow for Saudi Arabia.

## The solution

Mirsad follows a national-first workflow. The user starts with a full-Saudi heatmap of NASA POWER grid cells, switches between the latest monthly temperature and long-term Sen’s slope, and uses real geographic context to locate signals. The user can then drill down to Riyadh and Jeddah, select air temperature or precipitation, and read the seasonal Mann–Kendall p-value, Kendall’s tau, Sen’s slope, and an exploratory 95% slope interval. Proactive cards compare the latest complete year with the calendar-month climatology.

## Impact

Mirsad makes a first-pass climate investigation faster and more understandable for students, educators, local analysts, and public-sector teams. It does not replace a formal climate assessment. It provides a transparent way to locate a signal, explain its statistical strength, and decide where deeper investigation or local observations may be needed.

## Creativity

The project combines a national heatmap with an explainable statistical workflow instead of treating machine learning as the answer. The product separates three steps: detect a spatial signal, explain it with seasonal Mann–Kendall and Sen’s slope, and monitor whether the latest complete year is unusual. The experimental LSTM is clearly labeled as an optional forecast layer and is benchmarked against a seasonal baseline.

## Validity

NASA POWER monthly point data are used for Riyadh and Jeddah, while NASA POWER grid data are used for the Saudi-wide scan. The analysis uses a 12-month seasonal Mann–Kendall test and seasonal Sen’s slope to avoid confusing normal calendar-month seasonality with a long-term trend. The interface reports the p-value and effect size together. The project clearly states that NASA POWER values are gridded/reanalysis-derived estimates, not local station observations, and that statistical association does not prove causation.

## Relevance to the challenge

The selected challenge asks teams to investigate Earth-system trends and communicate what is changing, where it is changing, and how confidently the change can be detected. Mirsad directly addresses that prompt through a Saudi-wide spatial scan, transparent trend statistics, regional comparison, and an accessible explanation for non-specialists.

## Data sources

- NASA POWER Monthly API: https://power.larc.nasa.gov/docs/services/api/temporal/monthly/
- NASA POWER Regional Data Requests: https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/
- NASA POWER Daily API for recent complete months: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- geoBoundaries Saudi Arabia boundary: https://www.geoboundaries.org/
- OpenStreetMap tiles: https://www.openstreetmap.org/
- Esri World Imagery tiles: https://www.arcgis.com/

## Technology

Python, Streamlit, Pandas, NumPy, Plotly, Folium, Streamlit-Folium, PyMannKendall, Scikit-learn, and PyTorch.

## Measured model result

The optional LSTM forecast was evaluated on a chronological holdout year that was not used for training. In the bundled run, the LSTM achieved an MAE of 0.91°C on 2025, compared with 1.35°C for the seasonal-naive baseline. This result describes one experimental evaluation and is not an official NASA forecast.

## Demo video and code

The repository includes the English explainer at `media/Mirsad_English_Explainer.mp4`. It is a motion-graphics overview with AI-generated narration, not a live screen recording of the dashboard. The public code repository URL is the GitHub page containing this file.

## Team

Enter the official team name, member names, and roles in the competition submission form using the team's registered details. Personal team details are intentionally not duplicated in this repository.
