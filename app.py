from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from src.cleaning import national_weighted_monthly
from src.data_pipeline import ARTIFACTS_DIR, CLEAN_PATH, run_pipeline
from src.lstm_model import train_evaluate_forecast
from src.city_trends import CITY_LABELS, VARIABLES, all_results, load_city_data, proactive_signal


st.set_page_config(page_title="Mirsad | Saudi Climate", layout="wide")

# A lightweight bilingual layer keeps the science and data pipeline unchanged while making
# the demo accessible to an international judging panel.
LANGUAGE = st.sidebar.radio("Language / اللغة", ["العربية", "English"], horizontal=True, index=0)
EN = LANGUAGE == "English"
T = {
    "eyebrow": ("NASA POWER · CLIMATE MONITORING", "NASA POWER · لوحة مراقبة مناخية"),
    "title": ("Mirsad | Saudi Climate Dashboard", "مِرصد | لوحة مناخ السعودية"),
    "subtitle": ("2 m air-temperature record · statistical trends · real basemap · experimental forecast", "السجل الشهري لدرجة حرارة الهواء عند مترين · الاتجاهات الإحصائية · خريطة حقيقية وتوقع تجريبي"),
    "settings": ("Dashboard settings", "إعدادات اللوحة"),
    "layer": ("Map layer", "طبقة الخريطة"),
    "temperature": ("Monthly temperature", "حرارة الشهر"),
    "trend": ("Temperature trend · Sen", "اتجاه الحرارة · Sen"),
    "forecast": ("LSTM forecast", "توقع LSTM"),
    "fetch": ("Fetch NASA data", "جلب بيانات NASA"),
    "refresh": ("Fetch / refresh NASA data", "جلب / تحديث بيانات NASA"),
    "train": ("Train LSTM and update forecast", "تدريب LSTM وتحديث التوقع"),
    "exclude": ("Exclude flagged outliers from LSTM training", "استبعاد الشذوذات من تدريب LSTM"),
    "history": ("Historical temperature trend", "اتجاه درجات الحرارة في السجل التاريخي"),
    "maptitle": ("Interactive real map · NASA grid cells and reference centers", "الخريطة التفاعلية · خلايا المناخ ومراكز المناطق"),
    "season": ("Seasonal temperature pattern", "النمط الموسمي لحرارة الشبكة"),
    "cities": ("Regional-center comparison", "مقارنة مراكز المناطق"),
    "gridtrend": ("Grid-cell trends", "اتجاه خلايا الشبكة"),
    "forecasttitle": ("Experimental forecast for the next 12 months", "توقع تجريبي للأشهر الاثني عشر التالية"),
    "evaluation": ("Chronological holdout evaluation", "تقييم زمني محجوب"),
    "language_note": ("Data labels and chart annotations follow the selected language where available.", "تتبع تسميات البيانات والرسوم اللغة المختارة حيثما أمكن."),
}
def t(key):
    return T[key][0 if EN else 1]

st.markdown(
    """
    <style>
      :root { --ink:#f3f7ff; --muted:#9da8d2; --line:#25256b; --blue:#20e8ff; --blue-dark:#8b7cff; --paper:#05052b; --pink:#ff27b7; --cyan:#27f2ff; }
      .stApp { background:radial-gradient(circle at 0% 0%, #11116d 0, transparent 30%), radial-gradient(circle at 100% 100%, #133e72 0, transparent 34%), var(--paper); color:var(--ink); }
      .block-container { max-width:1580px; padding:1.2rem 1.5rem 2.8rem; }
      h1,h2,h3,h4,p,label,[data-testid="stMetricLabel"] { text-align:right; }
      h1 { color:#ffffff; letter-spacing:-.035em; font-size:2.55rem; margin-bottom:.15rem; text-shadow:0 0 24px rgba(39,242,255,.25); }
      h2 { color:#dbe4ff; letter-spacing:-.02em; }
      h3 { color:#ffffff; }
      [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(16,18,82,.95),rgba(8,8,48,.96)); border:1px solid #34349a; border-radius:17px; padding:15px 16px; box-shadow:0 0 22px rgba(45,40,190,.22), inset 0 0 22px rgba(28,218,255,.04); min-height:108px; }
      [data-testid="stMetricLabel"] { color:#aab5e0; font-size:.84rem; }
      [data-testid="stMetricValue"] { color:#ffffff; font-weight:700; }
      [data-testid="stMetricDelta"] { direction:rtl; }
      div[data-testid="stTabs"] button { font-weight:650; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(14,15,77,.9),rgba(7,7,42,.96)); border:1px solid #282879; border-radius:18px; box-shadow:0 0 28px rgba(20,18,110,.32), inset 0 0 22px rgba(39,242,255,.025); }
      .eyebrow { color:var(--cyan); font-size:.78rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
      .subtitle { color:#aab5e0; margin-top:-.5rem; font-size:1rem; }
      .hero-strip { margin:1rem 0 1.3rem; padding:1.1rem 1.25rem; border-radius:20px; background:linear-gradient(105deg,#101057 0%,#182a8d 50%,#a31383 150%); color:white; border:1px solid #3535b2; box-shadow:0 0 28px rgba(39,242,255,.14), 0 0 22px rgba(255,39,183,.13); }
      .hero-strip strong { font-size:1.05rem; }
      .hero-strip span { opacity:.86; font-size:.88rem; margin-inline-start:1rem; }
      .stButton > button { border-radius:12px; border:1px solid #27f2ff; background:linear-gradient(110deg,#1434a0,#c51a9e); color:#fff; font-weight:800; box-shadow:0 0 16px rgba(39,242,255,.18); }
      .stButton > button:hover { border-color:#ffffff; background:linear-gradient(110deg,#ff27b7,#20e8ff); color:#05052b; }
      div[data-testid="stAlert"] { border-radius:14px; border-left:4px solid #ff27b7; background:#10104d; }
      .section-title { color:#ffffff; font-size:1.02rem; font-weight:700; margin:.15rem 0 .5rem; text-align:right; }
      .tiny-note { color:#9da8d2; font-size:.78rem; line-height:1.55; text-align:right; }
      .pill { display:inline-block; background:linear-gradient(110deg,#1d2989,#93218f); color:#ffffff; border:1px solid #3ddcf3; border-radius:999px; padding:5px 11px; font-size:.76rem; font-weight:700; box-shadow:0 0 10px rgba(39,242,255,.14); }
      section[data-testid="stSidebar"] { background:linear-gradient(180deg,#070737,#0b0b4c); border-right:1px solid #29298a; }
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:#dfe7ff; }
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


REGION_CENTERS = [
    ("الرياض", 24.7136, 46.6753),
    ("مكة المكرمة", 21.3891, 39.8579),
    ("المدينة المنورة", 24.5247, 39.5692),
    ("الشرقية · الدمام", 26.4207, 50.0888),
    ("القصيم · بريدة", 26.3592, 43.9818),
    ("عسير · أبها", 18.2164, 42.5053),
    ("تبوك", 28.3838, 36.5550),
    ("حائل", 27.5114, 41.7208),
    ("الحدود الشمالية · عرعر", 30.9753, 41.0381),
    ("جازان", 16.8894, 42.5511),
    ("نجران", 17.4917, 44.1322),
    ("الجوف · سكاكا", 29.9697, 40.2064),
    ("الباحة", 20.0129, 41.4677),
]


def load_data() -> pd.DataFrame | None:
    if not CLEAN_PATH.exists():
        return None
    data = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
    data["outlier_flag"] = data["outlier_flag"].fillna(False).astype(bool)
    return data


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_p_value(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"= {value:.3g}"


def national_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    data = forecast.copy()
    data["weight"] = np.cos(np.deg2rad(data["lat"].astype(float))).clip(lower=0)
    data["weighted"] = data["predicted_T2M"] * data["weight"]
    result = data.groupby("date", as_index=False).agg(weighted_sum=("weighted", "sum"), weight_sum=("weight", "sum"))
    result["national_forecast"] = result["weighted_sum"] / result["weight_sum"]
    return result[["date", "national_forecast"]]


def reference_centers(month_points: pd.DataFrame, value_col: str = "T2M") -> pd.DataFrame:
    """Attach each regional capital marker to its nearest available NASA grid cell."""
    rows = []
    if month_points.empty:
        return pd.DataFrame(columns=["region", "lat", "lon", "grid_lat", "grid_lon", "T2M", "grid_value"])
    for region, lat, lon in REGION_CENTERS:
        dx = (month_points["lon"].to_numpy(dtype=float) - lon) * np.cos(np.deg2rad(lat))
        dy = month_points["lat"].to_numpy(dtype=float) - lat
        index = int(np.argmin(dx * dx + dy * dy))
        nearest = month_points.iloc[index]
        rows.append({
            "region": region,
            "lat": lat,
            "lon": lon,
            "grid_lat": float(nearest["lat"]),
            "grid_lon": float(nearest["lon"]),
            "T2M": float(nearest[value_col]),
            "grid_value": float(nearest[value_col]),
        })
    return pd.DataFrame(rows)


def map_figure(points: pd.DataFrame, value_col: str, value_label: str, title: str,
               *, colorscale: str = "Turbo", center_zero: bool = False,
               centers: pd.DataFrame | None = None, units: str = "°C") -> go.Figure:
    hover = {"lat": ":.2f", "lon": ":.2f", value_col: ":.3f"}
    fig = px.scatter_geo(
        points,
        lat="lat",
        lon="lon",
        color=value_col,
        hover_data=hover,
        color_continuous_scale=colorscale,
        labels={value_col: value_label, "lat": "خط العرض", "lon": "خط الطول"},
        title=title,
    )
    fig.update_traces(marker={"size": 8, "opacity": .88, "line": {"width": .5, "color": "white"}})
    if center_zero and not points.empty:
        limit = max(abs(float(points[value_col].min())), abs(float(points[value_col].max())))
        fig.update_coloraxes(cmin=-limit, cmax=limit, cmid=0)

    if centers is not None and not centers.empty:
        center_hover = [
            f"{row.region}<br>{value_label}: {row.grid_value:.3f}<br>أقرب خلية شبكية: {row.grid_lat:.2f}, {row.grid_lon:.2f}<br>علامة مرجعية لمركز المنطقة وليست محطة رصد"
            for row in centers.itertuples()
        ]
        visible_labels = {"الرياض", "مكة المكرمة", "المدينة المنورة", "الشرقية · الدمام", "عسير · أبها", "تبوك", "جازان"}
        city_labels = [name if name in visible_labels else "" for name in centers["region"]]
        fig.add_trace(go.Scattergeo(
            lat=centers["lat"], lon=centers["lon"],
            text=city_labels, customdata=center_hover,
            mode="markers+text", textposition="top center",
            textfont={"family": "Arial, Tahoma", "size": 9, "color": "#183354"},
            marker={"size": 7, "color": "#ffffff", "line": {"color": "#203e69", "width": 1.5}, "symbol": "circle"},
            hovertemplate="%{customdata}<extra></extra>", name="مراكز المناطق (مرجعية)",
        ))
    fig.update_geos(
        projection_type="mercator",
        lonaxis_range=[33.5, 57.5],
        lataxis_range=[15.0, 33.5],
        showland=True,
        landcolor="#eef1f5",
        showocean=True,
        oceancolor="#eaf1f8",
        showcountries=True,
        countrycolor="#9aa9b9",
        coastlinecolor="#9aa9b9",
        resolution=50,
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title={"text": title, "x": .98, "xanchor": "right", "y": .97, "font": {"size": 15}},
        height=470,
        margin={"l": 5, "r": 5, "t": 55, "b": 5},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, Tahoma", "color": "#f3f7ff", "size": 11},
        coloraxis_colorbar={"title": units, "thickness": 12, "len": .72},
        legend={"orientation": "h", "y": -.04, "x": .98, "xanchor": "right"},
    )
    return fig


def real_map(points: pd.DataFrame, value_col: str, value_label: str, title: str, centers: pd.DataFrame | None = None, heatmap: bool = False) -> folium.Map:
    """Render actual geographic context using OpenStreetMap tiles and the project boundary."""
    fmap = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(fmap)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     name="Satellite imagery", attr="Esri World Imagery", control=True).add_to(fmap)
    boundary_path = Path(__file__).parent / "data" / "boundaries" / "saudi_arabia_simplified.geojson"
    if boundary_path.exists():
        folium.GeoJson(boundary_path.read_text(encoding="utf-8"), name="Saudi Arabia boundary",
                       style_function=lambda _: {"fillColor": "#9bb7d4", "color": "#284a6d", "weight": 2, "fillOpacity": 0.08}).add_to(fmap)
    if not points.empty:
        vals = pd.to_numeric(points[value_col], errors="coerce")
        lo, hi = float(vals.min()), float(vals.max())
        scale = max(abs(hi - lo), 1e-9)
        if heatmap:
            heat_values = ((vals - lo) / scale).clip(0.05, 1.0)
            HeatMap([[float(row.lat), float(row.lon), float(weight)] for row, weight in zip(points.itertuples(), heat_values)],
                    name="Climate intensity heatmap", radius=18, blur=22, min_opacity=.34,
                    gradient={0.15: "#2c7bb6", 0.4: "#abd9e9", 0.6: "#ffffbf", 0.8: "#fdae61", 1.0: "#d7191c"}).add_to(fmap)
        map_points = points.assign(_map_value=vals).dropna(subset=["_map_value"])
        for _, row in map_points.iterrows():
            value = float(row["_map_value"])
            norm = (value - lo) / scale
            color = "#b2182b" if norm > .66 else "#ef8a62" if norm > .33 else "#2166ac"
            folium.CircleMarker([float(row.lat), float(row.lon)], radius=4 if heatmap else 5, color="white", weight=1, fill=True, fill_color=color, fill_opacity=.62 if heatmap else .86,
                tooltip=f"{value_label}: {value:.2f} · {float(row.lat):.2f}, {float(row.lon):.2f}").add_to(fmap)
    if centers is not None and not centers.empty:
        for row in centers.itertuples():
            folium.Marker([float(row.lat), float(row.lon)], tooltip=str(row.region), icon=folium.Icon(color="darkblue", icon="info-sign")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap

def line_figure(history: pd.DataFrame) -> go.Figure:
    rolling = history.set_index("date")["national_t2m"].rolling(12, min_periods=12).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["date"], y=history["national_t2m"], name="المتوسط الشهري",
        mode="lines", line={"color": "#25e6ff", "width": 1.1}, opacity=.72,
    ))
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling.values, name="متوسط متحرك 12 شهراً",
        mode="lines", line={"color": "#ff2db2", "width": 3},
    ))
    fig.update_layout(
        height=385, margin={"l": 8, "r": 8, "t": 20, "b": 5}, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0b3e",
        xaxis_title="السنة", yaxis_title="°م",
        legend={"orientation": "h", "y": 1.12, "x": .98, "xanchor": "right"},
        font={"family": "Arial, Tahoma", "color": "#c1c9ed", "size": 10},
        xaxis={"gridcolor": "#25256b", "zeroline": False},
        yaxis={"gridcolor": "#25256b", "zeroline": False},
    )
    return fig


def season_figure(data: pd.DataFrame) -> go.Figure:
    monthly = data.groupby("month", as_index=False)["T2M"].mean()
    labels = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
    fig = go.Figure(go.Bar(
        x=[labels[int(m)-1] for m in monthly["month"]], y=monthly["T2M"],
        marker_color="#6d5cff", marker_line_width=0,
        hovertemplate="%{x}<br>متوسط سجل الشبكة: %{y:.1f}°م<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin={"l": 5, "r": 5, "t": 8, "b": 5},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0b3e",
        xaxis={"tickfont": {"size": 9}, "showgrid": False},
        yaxis={"title": "°م", "gridcolor": "#25256b"},
        font={"family": "Arial, Tahoma", "color": "#c1c9ed", "size": 10},
    )
    return fig


def city_figure(cities: pd.DataFrame) -> go.Figure:
    ordered = cities.sort_values("T2M")
    fig = px.bar(
        ordered, x="T2M", y="region", orientation="h", text="T2M",
        labels={"T2M": "°م عند أقرب خلية", "region": "المركز المرجعي"},
        color_discrete_sequence=["#25e6ff"],
    )
    fig.update_traces(texttemplate="%{text:.1f}°", textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}<br>القيمة الشبكية القريبة: %{x:.2f}°م<extra></extra>")
    fig.update_layout(
        height=300, margin={"l": 5, "r": 45, "t": 8, "b": 5},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0b3e",
        xaxis={"gridcolor": "#25256b", "title": "°م"},
        yaxis={"title": "", "tickfont": {"size": 9}},
        font={"family": "Arial, Tahoma", "color": "#c1c9ed", "size": 10},
        showlegend=False,
    )
    return fig


def forecast_figure(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    recent = history.tail(36)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["date"], y=recent["national_t2m"], name="مرصود",
                             mode="lines+markers", line={"color": "#25e6ff", "width": 2}, marker={"size": 5}))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["national_forecast"], name="توقع LSTM",
                             mode="lines+markers", line={"color": "#ff2db2", "width": 3, "dash": "dash"}, marker={"size": 6}))
    fig.update_layout(
        height=300, margin={"l": 8, "r": 8, "t": 10, "b": 5}, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0b3e",
        xaxis_title="التاريخ", yaxis_title="°م",
        legend={"orientation": "h", "y": 1.12, "x": .98, "xanchor": "right"},
        font={"family": "Arial, Tahoma", "color": "#c1c9ed", "size": 10},
        xaxis={"gridcolor": "#25256b"}, yaxis={"gridcolor": "#25256b"},
    )
    return fig


def detective_figure(city_data: pd.DataFrame, variable: str, selected_city: str | None = None) -> go.Figure:
    subset = city_data if selected_city is None else city_data.loc[city_data["city"].eq(selected_city)]
    meta = VARIABLES[variable]
    fig = px.line(
        subset, x="date", y=variable, color="city", labels={variable: f"{meta['name']} ({meta['unit']})", "date": "Date", "city": "City"},
        color_discrete_map={"Riyadh": "#2f6f9f", "Jeddah": "#c77b45"},
    )
    fig.update_layout(
        height=330, margin={"l": 8, "r": 8, "t": 20, "b": 5}, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0b3e",
        font={"family": "Arial, Tahoma", "color": "#c1c9ed", "size": 10},
        legend={"orientation": "h", "y": 1.12, "x": .98, "xanchor": "right"},
        xaxis={"gridcolor": "#25256b"}, yaxis={"gridcolor": "#25256b"},
    )
    return fig


st.markdown(f'<div class="eyebrow">{t("eyebrow")}</div>', unsafe_allow_html=True)
st.title(t("title"))
st.markdown(f'<div class="subtitle">{t("subtitle")}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-strip"><strong>{"Find the signal. Explain the change." if EN else "اكتشف الإشارة · افهم التغيّر"}</strong>'
    f'<span>{"NASA POWER · Mann–Kendall · Sen’s slope · Real maps" if EN else "NASA POWER · Mann–Kendall · ميل Sen · خرائط حقيقية"}</span></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"### {t('settings')}")
    data = load_data()
    if data is None:
        st.warning("No prepared data. Fetch NASA data to clean and analyze it." if EN else "لا توجد بيانات جاهزة. استخدم الزر لجلب بيانات NASA ثم تنظيفها.")
        if st.button(t("fetch"), width="stretch"):
            with st.status("جارٍ تنزيل البيانات وتنظيفها…", expanded=True) as status:
                try:
                    run_pipeline(refresh=True, log=status.write)
                    status.update(label="اكتمل تجهيز البيانات", state="complete")
                    st.rerun()
                except Exception as exc:
                    status.update(label="تعذر إكمال تجهيز البيانات", state="error")
                    st.exception(exc)
        st.stop()

    dates = sorted(pd.to_datetime(data["date"].drop_duplicates()))
    st.caption(f"البيانات الجاهزة: {dates[0]:%Y-%m} — {dates[-1]:%Y-%m}")
    map_mode = st.selectbox(t("layer"), ["temperature", "trend", "forecast"], format_func=lambda x: {"temperature": t("temperature"), "trend": t("trend"), "forecast": t("forecast")} [x])
    map_date = dates[-1]
    if map_mode == "temperature":
        map_date = st.select_slider("الشهر المعروض", options=dates, value=dates[-1], format_func=lambda d: d.strftime("%Y-%m"))
    elif map_mode == "forecast":
        fp = ARTIFACTS_DIR / "forecast_saudi_12_months.csv"
        if fp.exists():
            fdates = sorted(pd.to_datetime(pd.read_csv(fp, usecols=["date"])["date"].unique()))
            forecast_date = st.select_slider("شهر التوقع", options=fdates, value=fdates[0], format_func=lambda d: d.strftime("%Y-%m"))
        else:
            forecast_date = None
            st.info("درّب النموذج أولاً لإظهار طبقة التوقع.")
    else:
        map_date = dates[-1]

    st.divider()
    st.markdown("**إدارة البيانات والنموذج**")
    if st.button(t("refresh"), width="stretch"):
        with st.status("جارٍ تحديث البيانات وتحليلها…", expanded=True) as status:
            try:
                run_pipeline(refresh=True, log=status.write)
                status.update(label="اكتمل التحديث", state="complete")
                st.rerun()
            except Exception as exc:
                status.update(label="فشل تحديث البيانات", state="error")
                st.exception(exc)
    exclude_flagged = st.checkbox(t("exclude"), value=False,
                                  help="القيم المتطرفة قد تمثل موجات حر حقيقية؛ افتراضياً لا تُستبعد.")
    train_button = st.button(t("train"), type="primary", width="stretch")
    if train_button:
        with st.status("تدريب واختبار LSTM زمنياً…", expanded=True) as status:
            try:
                model_metrics = train_evaluate_forecast(data, ARTIFACTS_DIR, max_epochs=15, patience=4,
                                                        exclude_flagged=exclude_flagged, log=status.write)
                status.update(label="اكتمل التدريب وحُفظ التوقع", state="complete")
                st.success(f"اختبار {model_metrics['test_year']}: MAE {model_metrics['lstm_test_mae_c']:.2f}°م")
                st.rerun()
            except Exception as exc:
                status.update(label="تعذر تدريب النموذج", state="error")
                st.exception(exc)
    quality = load_json(CLEAN_PATH.with_suffix(".quality.json"))
    if quality:
        st.caption(f"الخلايا الشبكية: {quality['grid_cells']:,}")
        st.caption(f"قيم معلّمة للمراجعة: {quality['statistical_outliers_flagged_but_retained']:,}")

# Trend Detective workspace: the challenge's core workflow comes before the legacy dashboard.
st.divider()
st.header("Trend Detective" if EN else "محقّق اتجاهات نظام الأرض")
st.markdown(
    "Choose a variable and compare Riyadh with Jeddah. The result answers what changed, where, by how much, and whether the trend is statistically significant."
    if EN else
    "اختر متغيراً وقارن بين الرياض وجدة. النتيجة تجيب: ماذا تغيّر؟ أين؟ ما معدل التغيّر؟ وهل الاتجاه ذو دلالة إحصائية؟"
)
# National-first story: scan all of Saudi Arabia before drilling into two cities.
st.subheader("Saudi-wide signal scan" if EN else "المسح الوطني لإشارات التغيّر")
st.markdown(
    "Start with the full Saudi grid, locate the strongest signal, then open the city-level explanation below."
    if EN else
    "ابدأ بشبكة السعودية كاملة، حدّد أقوى إشارة مكانية، ثم انتقل إلى التفسير التفصيلي للمدن أدناه."
)
national_layer = st.radio(
    "National layer" if EN else "الطبقة الوطنية",
    ["temperature", "trend"], horizontal=True,
    format_func=lambda key: ("Latest monthly temperature" if EN else "حرارة آخر شهر مكتمل") if key == "temperature" else ("Long-term Sen slope" if EN else "ميل Sen طويل الأمد"),
)
national_map_col, national_story_col = st.columns([1.5, .75], gap="medium")
with national_map_col:
    if national_layer == "temperature":
        national_points = data.loc[data["date"].eq(data["date"].max()), ["lat", "lon", "T2M"]].copy()
        national_points["map_value"] = national_points["T2M"]
        national_centers = reference_centers(national_points, "map_value")
        map_label = "Temperature °C" if EN else "درجة الحرارة °م"
        map_title = "Saudi Arabia · latest complete month" if EN else "السعودية · آخر شهر مكتمل"
    else:
        national_trend_path = ARTIFACTS_DIR / "trend_by_grid.csv"
        national_points = pd.read_csv(national_trend_path) if national_trend_path.exists() else pd.DataFrame()
        if not national_points.empty:
            national_points["map_value"] = national_points["slope_per_year"]
        national_centers = reference_centers(national_points, "map_value") if not national_points.empty else None
        map_label = "Sen slope °C/year" if EN else "ميل Sen °م/سنة"
        map_title = "Saudi Arabia · long-term trend" if EN else "السعودية · الاتجاه طويل الأمد"
    if national_points.empty:
        st.warning("No national trend artifact yet." if EN else "لا تتوفر نتائج الاتجاه الوطني بعد.")
    else:
        national_map_state = st_folium(real_map(national_points, "map_value", map_label, map_title, national_centers, heatmap=True), height=560, use_container_width=True, returned_objects=["last_object_clicked"])
        clicked = national_map_state.get("last_object_clicked") if national_map_state else None
        if clicked and clicked.get("lat") is not None:
            st.caption((f"Selected location · {clicked['lat']:.2f}, {clicked['lng']:.2f}" if EN else f"الموقع المختار · {clicked['lat']:.2f}، {clicked['lng']:.2f}"))
with national_story_col:
    with st.container(border=True):
        st.markdown(f'<div class="section-title">{"A decision-ready story" if EN else "قصة جاهزة للقرار"}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pill">{"01 · Detect" if EN else "٠١ · اكتشف"}</div> '
            f'<div class="pill">{"02 · Explain" if EN else "٠٢ · فسّر"}</div> '
            f'<div class="pill">{"03 · Act" if EN else "٠٣ · تحرّك"}</div>', unsafe_allow_html=True)
        st.markdown(
            "The heatmap reveals where the signal is concentrated. The statistical cards explain whether it is credible. The proactive cards show whether the latest year is unusual."
            if EN else
            "الخريطة الحرارية تكشف أين تتركز الإشارة. البطاقات الإحصائية تشرح مدى موثوقيتها. والإشارات الاستباقية توضح هل كانت السنة الأخيرة غير معتادة."
        )
        st.metric("Grid cells" if EN else "خلايا الشبكة", f"{national_points.shape[0]:,}")
        st.caption("NASA POWER grid estimates · not local station observations" if EN else "تقديرات شبكة NASA POWER · وليست قراءات محطات محلية")
try:
    city_data = load_city_data()
    variable = st.radio(
        "Variable" if EN else "المتغير",
        ["T2M", "PRECTOTCORR"], horizontal=True,
        format_func=lambda key: VARIABLES[key]["name"] if EN else VARIABLES[key]["name_ar"],
    )
    city_choice = st.radio(
        "View" if EN else "العرض", ["both", "Riyadh", "Jeddah"], horizontal=True,
        format_func=lambda key: ("Both cities" if EN else "المدينتان") if key == "both" else CITY_LABELS[key],
    )
    detective_results = all_results(city_data, variable)
    detective_subset = None if city_choice == "both" else city_choice
    st.plotly_chart(detective_figure(city_data, variable, detective_subset), width="stretch", config={"displaylogo": False})
    meta = VARIABLES[variable]
    cards = st.columns(2)
    for card, result in zip(cards, detective_results.to_dict("records")):
        trend_word = {"increasing": ("increasing" if EN else "صاعد"), "decreasing": ("decreasing" if EN else "هابط"), "no trend": ("no clear trend" if EN else "غير واضح")}.get(result["trend"], result["trend"])
        significance = ("statistically significant" if EN else "دال إحصائياً") if result["significant"] else ("not statistically significant" if EN else "غير دال إحصائياً")
        with card:
            with st.container(border=True):
                st.subheader(CITY_LABELS[result["city"]] if not EN else result["city"])
                st.metric("Seasonal Sen slope" if EN else "ميل Sen الموسمي", f"{result['slope_per_year']:.4f} {meta['unit']}/year")
                st.metric("Mann–Kendall p-value" if EN else "قيمة p لاختبار Mann–Kendall", format_p_value(result["p_value"]))
                st.caption(f"{trend_word} · {significance} · Kendall τ = {result['kendall_tau']:.3f} · {result['start']}–{result['end']}")
                st.caption((f"95% CI for slope: [{result['slope_ci_low']:.4f}, {result['slope_ci_high']:.4f}] {meta['unit']}/year" if EN else f"فاصل الثقة 95% للميل: [{result['slope_ci_low']:.4f}, {result['slope_ci_high']:.4f}] {meta['unit']}/سنة"))
    st.markdown(f"**{('Proactive signals' if EN else 'الإشارات الاستباقية')}**")
    signal_cards = st.columns(2)
    for card, city in zip(signal_cards, ["Riyadh", "Jeddah"]):
        signal = proactive_signal(city_data, city, variable)
        anomaly = signal["mean_anomaly"]
        direction = ("above its seasonal baseline" if EN else "أعلى من خطه الموسمي") if anomaly > 0 else ("below its seasonal baseline" if EN else "أدنى من خطه الموسمي")
        with card:
            st.metric(f"{CITY_LABELS[city] if not EN else city} · {signal['year']}", f"{anomaly:+.2f} {meta['unit']}", direction)
    st.info(
        f"Interpretation: Sen's slope is the median seasonal change per year. A p-value below 0.05 indicates statistical evidence of a trend; it does not prove causation. Units: {meta['description']} ({meta['unit']})."
        if EN else
        f"التفسير: ميل Sen هو متوسط معدل التغير الموسمي في السنة. قيمة p الأقل من 0.05 تعني وجود دليل إحصائي على اتجاه، لكنها لا تثبت السبب. المتغير: {meta['description']} ({meta['unit']})."
    )
    with st.expander("Method and data notes" if EN else "المنهجية وملاحظات البيانات"):
        st.markdown(
            "The monthly series is evaluated with a seasonal Mann–Kendall test using a 12-month period, and Sen's slope is computed within calendar months to avoid confusing normal summer/winter seasonality with a long-term trend. Values come from NASA POWER's monthly point product for Riyadh and Jeddah."
            if EN else
            "تُحلّل السلسلة الشهرية باختبار Mann–Kendall الموسمي بدورة 12 شهراً، ويُحسب ميل Sen داخل الأشهر التقويمية حتى لا نخلط موسمية الصيف والشتاء مع الاتجاه طويل الأمد. القيم من منتج NASA POWER الشهري لنقطتي الرياض وجدة."
        )
except Exception as exc:
    st.warning(("Trend Detective data are not available yet: " if EN else "بيانات Trend Detective غير متاحة بعد: ") + str(exc))

# Shared data for dashboard panels.
national = national_weighted_monthly(data)
latest_month = national["date"].max()
latest_value = float(national.loc[national["date"].eq(latest_month), "national_t2m"].iloc[0])
trend = load_json(ARTIFACTS_DIR / "national_trend.json")
metrics = load_json(ARTIFACTS_DIR / "metrics.json")
trend_path = ARTIFACTS_DIR / "trend_by_grid.csv"
trend_grid = pd.read_csv(trend_path) if trend_path.exists() else pd.DataFrame()

# KPI strip, similar to an executive snapshot.
cell_count = data[["lat", "lon"]].drop_duplicates().shape[0]
significant_pct = float(trend_grid["significant"].astype(bool).mean() * 100) if not trend_grid.empty else 0.0
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("آخر شهر مكتمل", latest_month.strftime("%Y-%m"))
k2.metric("متوسط المملكة المرجّح", f"{latest_value:.1f} °م")
if trend:
    k3.metric("ميل Sen الموسمي", f"{trend['slope_per_year']:.3f} °م/سنة", "اتجاه صاعد" if trend["trend"] == "increasing" else "اتجاه هابط")
    k4.metric("الخلايا ذات اتجاه دال", f"{significant_pct:.1f}%", "من خلايا الشبكة")
else:
    k3.metric("ميل Sen الموسمي", "غير متاح")
    k4.metric("الخلايا ذات اتجاه دال", "غير متاح")
k5.metric("الخلايا داخل المملكة", f"{cell_count:,}", "NASA POWER · شبكة مكانية")

left, right = st.columns([1.08, 1], gap="medium")
with left:
    with st.container(border=True):
        st.markdown(f'<div class="section-title">{t("history")}</div>', unsafe_allow_html=True)
        st.plotly_chart(line_figure(national), width="stretch", config={"displaylogo": False, "scrollZoom": True})
        if trend:
            label = "اتجاه موسمي دال إحصائياً" if trend.get("significant") else "الاتجاه غير دال إحصائياً"
            st.markdown(
                f'<div class="tiny-note">{label} · ميل Sen: {trend["slope_per_year"]:.4f} °م/سنة · p {format_p_value(trend["p_value"])}. الاختبار يصف الاتجاه ولا يحدد سببه.</div>',
                unsafe_allow_html=True,
            )
with right:
    with st.container(border=True):
        st.markdown(f'<div class="section-title">{t("maptitle")}</div>', unsafe_allow_html=True)
        centers = None
        if map_mode == "temperature":
            selected = data.loc[data["date"].eq(pd.Timestamp(map_date)), ["lat", "lon", "T2M"]].copy()
            selected["map_value"] = selected["T2M"]
            centers = reference_centers(selected, "map_value")
            figure = map_figure(selected, "map_value", "درجة الحرارة °م", f"درجة الحرارة · {pd.Timestamp(map_date):%Y-%m}", centers=centers)
        elif map_mode == "trend":
            if trend_grid.empty:
                st.info("حدّث البيانات لإنتاج خريطة الاتجاهات.")
                selected = pd.DataFrame()
                figure = None
            else:
                selected = trend_grid.copy()
                selected["map_value"] = selected["slope_per_year"]
                centers = reference_centers(selected, "map_value")
                figure = map_figure(selected, "map_value", "Sen · °م/سنة", "معدل اتجاه الحرارة طويل الأمد", colorscale="RdBu_r", center_zero=True, centers=centers, units="°م/سنة")
        else:
            fpath = ARTIFACTS_DIR / "forecast_saudi_12_months.csv"
            if not fpath.exists() or not metrics:
                st.info("لا يوجد توقع محفوظ بعد. استخدم زر تدريب LSTM في الشريط الجانبي.")
                selected = pd.DataFrame()
                figure = None
            else:
                all_forecast = pd.read_csv(fpath, parse_dates=["date"])
                selected = all_forecast.loc[all_forecast["date"].eq(pd.Timestamp(forecast_date))].copy()
                selected["map_value"] = selected["predicted_T2M"]
                centers = reference_centers(selected, "map_value")
                figure = map_figure(selected, "map_value", "توقع °م", f"توقع LSTM · {pd.Timestamp(forecast_date):%Y-%m}", centers=centers)
        if figure is not None:
            st_folium(real_map(selected, "map_value", "Value", "Map", centers), height=520, use_container_width=True, returned_objects=[])
        st.markdown('<div class="tiny-note">النقاط الملوّنة خلايا شبكة NASA POWER. علامات المدن/مراكز المناطق للمرجعية فقط؛ قيمة الحرارة لأقرب خلية شبكية وليست قراءة محطة داخل المدينة. لا تتوفر في هذا المشروع حدود المحافظات الإدارية.</div>', unsafe_allow_html=True)

# Three compact analytical panels.
a, b, c = st.columns([1, 1.15, .9], gap="medium")
with a:
    with st.container(border=True):
        st.markdown(f'<div class="section-title">{t("season")}</div>', unsafe_allow_html=True)
        st.plotly_chart(season_figure(data), width="stretch", config={"displaylogo": False})
        st.markdown('<div class="tiny-note">متوسط كل شهر تقويمي عبر كامل السجل المتاح، وليس توقعاً مستقبلياً.</div>', unsafe_allow_html=True)
with b:
    with st.container(border=True):
        st.markdown(f'<div class="section-title">{t("cities")} · {pd.Timestamp(map_date):%Y-%m}</div>', unsafe_allow_html=True)
        city_points_month = data.loc[data["date"].eq(pd.Timestamp(map_date)), ["lat", "lon", "T2M"]]
        city_values = reference_centers(city_points_month)
        if not city_values.empty:
            st.plotly_chart(city_figure(city_values), width="stretch", config={"displaylogo": False})
        st.markdown('<div class="tiny-note">قيمة كل مدينة مأخوذة من أقرب خلية مكانية متاحة؛ ليست قياساً محلياً مستقلاً.</div>', unsafe_allow_html=True)
with c:
    with st.container(border=True):
        st.markdown(f'<div class="section-title">{t("gridtrend")}</div>', unsafe_allow_html=True)
        if not trend_grid.empty:
            tg = trend_grid.copy()
            tg["الحالة"] = np.where(tg["significant"].astype(bool), tg["trend"].map({"increasing": "اتجاه صاعد دال", "decreasing": "اتجاه هابط دال"}), "غير دال")
            counts = tg["الحالة"].value_counts().rename_axis("الحالة").reset_index(name="عدد الخلايا")
            pie = px.pie(counts, names="الحالة", values="عدد الخلايا", hole=.62,
                         color="الحالة", color_discrete_map={"اتجاه صاعد دال": "#6687b5", "اتجاه هابط دال": "#c58c6e", "غير دال": "#d8dee8"})
            pie.update_traces(textposition="outside", textinfo="label+percent", hovertemplate="%{label}<br>%{value} خلية (%{percent})<extra></extra>")
            pie.update_layout(height=255, margin={"l": 6, "r": 6, "t": 8, "b": 8},
                              showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font={"family": "Arial, Tahoma", "size": 10})
            st.plotly_chart(pie, width="stretch", config={"displaylogo": False})
        else:
            st.info("لا تتوفر نتائج اتجاهات لكل خلية بعد.")
        st.markdown('<div class="tiny-note">الدلالة إحصائية لا تعني تحديد سبب التغير.</div>', unsafe_allow_html=True)

# Forecast panel follows the historical/map summary as a separate lower band.
forecast_path = ARTIFACTS_DIR / "forecast_saudi_12_months.csv"
if forecast_path.exists() and metrics:
    forecast_grid = pd.read_csv(forecast_path, parse_dates=["date"])
    forecast_national = national_forecast(forecast_grid)
    with st.container(border=True):
        head, score = st.columns([1.5, 1])
        with head:
            st.markdown(f'<div class="section-title">{t("forecasttitle")}</div>', unsafe_allow_html=True)
            st.plotly_chart(forecast_figure(national, forecast_national), width="stretch", config={"displaylogo": False, "scrollZoom": True})
        with score:
            st.markdown(f'<div class="section-title">{t("evaluation")}</div>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            s1.metric(f"MAE · {metrics['test_year']}", f"{metrics['lstm_test_mae_c']:.2f} °م")
            s2.metric("خط الأساس", f"{metrics['seasonal_naive_test_mae_c']:.2f} °م")
            s3, s4 = st.columns(2)
            s3.metric("ضمن ±1°م", f"{metrics.get('lstm_within_1c_pct', float('nan')):.1f}%")
            s4.metric("ضمن ±2°م", f"{metrics.get('lstm_within_2c_pct', float('nan')):.1f}%")
            st.markdown(
                f'<div class="tiny-note">اختبار على {metrics["n_test_monthly_predictions"]:,} توقعاً شهرياً للخلايا. النسبة ضمن هامش خطأ معلن، وليست دقة عامة أو ضماناً للتوقع المستقبلي.</div>',
                unsafe_allow_html=True,
            )
            if metrics.get("lstm_test_mae_c", 999) < metrics.get("seasonal_naive_test_mae_c", 0):
                st.success("تفوق على خط الأساس في سنة الاختبار المحجوبة.")
            else:
                st.warning("لم يتفوق على خط الأساس في الاختبار المحجوب.")
            st.markdown(
                f'<div class="tiny-note">الفترة المتوقعة: {metrics["forecast_start"]} — {metrics["forecast_end"]}. إسقاط إحصائي تجريبي، وليس توقعاً رسمياً من NASA أو نموذجاً مناخياً فيزيائياً.</div>',
                unsafe_allow_html=True,
            )

st.divider()
st.markdown(
    '<div class="tiny-note">المصدر: NASA POWER. السجل الشهري التاريخي من MERRA-2، وأحدث الأشهر المكتملة من تجميع البيانات اليومية. الخلايا تمثل تقديراً شبكياً محدود الدقة. عواصم المناطق علامات مرجعية ولا تعني وجود محطات رصد. بيانات الحدود المتاحة هنا حدود المملكة فقط، وليست حدوداً إدارية داخلية.</div>',
    unsafe_allow_html=True,
)
