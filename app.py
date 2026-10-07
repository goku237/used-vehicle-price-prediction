"""Polished Streamlit interface for used vehicle price estimation."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.charts import (
    display_label,
    feature_impact_chart,
    grouped_feature_importance,
    model_comparison_chart,
    price_distribution_chart,
)
from src.data import load_vehicle_data

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "vehicle_price_model.joblib"
IMPORTANCE_PATH = ROOT / "artifacts" / "feature_importance.csv"
METRICS_PATH = ROOT / "artifacts" / "model_metrics.csv"
DATA_PATH = ROOT / "data" / "raw" / "used_cars.csv"

FIELD_GROUPS = [
    ("Age & distance", ["age", "km_driven", "odometer", "mileage"]),
    ("Vehicle identity", ["name", "make", "model", "trim"]),
    ("Specification", ["fuel", "transmission", "body_type", "drivetrain", "engine", "max_power", "horsepower", "cylinders", "seats"]),
    ("Ownership & condition", ["owner", "seller_type", "condition", "accident_history", "clean_title", "exterior_color", "interior_color"]),
]

UNITS = {
    "age": "years", "km_driven": "km", "odometer": "miles", "mileage": "kmpl",
    "engine": "cc", "max_power": "bhp", "horsepower": "hp",
}

STYLES = """
<style>
    .stApp { background: #f6f8fc; color: #172033; }
    .block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 3.2rem; }
    #MainMenu, header, footer { visibility: hidden; }

    .hero { background: linear-gradient(118deg, #0e2a52 0%, #1a4a89 55%, #2b74aa 100%); color: #fff;
            border-radius: 22px; padding: 2.3rem 2.5rem; margin: .1rem 0 1.5rem;
            box-shadow: 0 16px 36px rgba(14, 42, 82, .20); }
    .hero .eyebrow { letter-spacing: .14em; text-transform: uppercase; font-size: .75rem;
                     opacity: .8; font-weight: 700; margin-bottom: .6rem; }
    .hero h1 { font-size: 2.35rem; line-height: 1.1; margin: 0 0 .65rem; color: #fff; font-weight: 800;
               letter-spacing: -.01em; }
    .hero p { font-size: 1.02rem; opacity: .9; margin: 0; max-width: 640px; line-height: 1.55; }
    .hero .chips { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: 1.15rem; }
    .hero .chips span { background: rgba(255, 255, 255, .13); border: 1px solid rgba(255, 255, 255, .22);
                        color: #e8f1ff; border-radius: 999px; padding: .3rem .8rem;
                        font-size: .78rem; font-weight: 600; }

    .section { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin: 2.1rem 0 .7rem; }
    .section .title { font-size: 1.12rem; font-weight: 800; color: #172033; letter-spacing: -.01em; }
    .section .hint { font-size: .85rem; color: #69748a; }
    .card-title { font-size: .95rem; font-weight: 800; color: #172033; margin: 0 0 .55rem; letter-spacing: -.01em; }
    .note { color: #69748a; font-size: .87rem; margin: -.35rem 0 .95rem; }

    div[data-testid="stForm"] { background: #fff; border: 1px solid #e5e9f2; border-radius: 18px;
                                padding: 1.25rem 1.5rem 1.2rem; box-shadow: 0 3px 14px rgba(27, 41, 69, .05); }
    .field-group { font-size: .74rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
                   color: #3465a4; margin: 1.1rem 0 .4rem; }
    .field-group.first { margin-top: .15rem; }

    [data-testid="stWidgetLabel"] p { font-size: .85rem; font-weight: 600; color: #3b4557; }
    [data-testid="stWidgetLabel"] small { font-size: .74rem; color: #7c8598; }
    div[data-baseweb="input"] { background: #fbfcfe; border-radius: 10px; }
    div[data-baseweb="select"] > div { background: #fbfcfe; border-radius: 10px; }

    .stFormSubmitButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: #1d5aa6; color: #fff; border: 0; border-radius: 12px; font-weight: 700;
        height: 3rem; letter-spacing: .01em; transition: background .15s ease; }
    .stFormSubmitButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #124685; color: #fff; border-color: #124685; }

    .result { background: linear-gradient(135deg, #e9f6ef 0%, #f2faf6 100%); border: 1px solid #b7ddc8;
              border-radius: 18px; padding: 1.35rem 1.6rem 1.2rem; color: #123f29; margin: .2rem 0 .4rem; }
    .result .label { font-size: .76rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
                     color: #3d7a5f; }
    .result .value { font-size: 2.5rem; font-weight: 800; letter-spacing: -.02em; margin-top: .15rem;
                     color: #0f3d28; }
    .result .meta { display: flex; gap: 2rem; flex-wrap: wrap; margin-top: .75rem; padding-top: .75rem;
                    border-top: 1px solid #cfe8db; }
    .result .meta div { display: flex; flex-direction: column; gap: .1rem; }
    .result .meta span { font-size: .74rem; text-transform: uppercase; letter-spacing: .07em; color: #4f8a70;
                         font-weight: 700; }
    .result .meta b { font-size: 1rem; color: #14402c; font-weight: 700; }
    .result .caption { font-size: .83rem; color: #4c7a66; margin-top: .8rem; }

    [data-testid="stMetric"] { background: #fff; border: 1px solid #e5e9f2; border-radius: 14px;
                               padding: .7rem .95rem .8rem; box-shadow: 0 2px 8px rgba(27, 41, 69, .04); }
    [data-testid="stMetricLabel"] { color: #69748a; font-weight: 600; font-size: .8rem; }
    [data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 800; color: #172033; }

    div[data-testid="stVerticalBlockBorder"], div[data-testid="stBorderWrapper"] {
        background: #fff; border: 1px solid #e5e9f2 !important; border-radius: 16px;
        box-shadow: 0 3px 12px rgba(27, 41, 69, .05); }
    [data-testid="stDataFrame"] { border: 1px solid #e5e9f2; border-radius: 12px; overflow: hidden; }
    [data-testid="stCaptionContainer"], caption { color: #69748a; }
    h2 { color: #172033; font-size: 1.12rem !important; margin-top: .2rem !important; }
</style>
"""

HERO = """
<section class="hero">
  <div class="eyebrow">Used vehicle intelligence</div>
  <h1>Find a data-backed price for your car</h1>
  <p>Enter the vehicle details below to estimate its resale value from the trained machine-learning model, then review how the candidate models and input fields compare.</p>
  <div class="chips">
    <span>3 candidate models</span>
    <span>Held-out validation</span>
    <span>MAE · RMSE · R²</span>
    <span>Feature impact ranking</span>
  </div>
</section>
"""


@st.cache_resource
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    try:
        return load_vehicle_data(DATA_PATH)
    except Exception:
        return None


def section(title: str, hint: str = "") -> None:
    tail = f'<span class="hint">{hint}</span>' if hint else ""
    st.markdown(f'<div class="section"><span class="title">{title}</span>{tail}</div>', unsafe_allow_html=True)


def card_title(title: str) -> None:
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)


def number_input(column: str, details: dict) -> float:
    lower = float(details["minimum"])
    upper = float(details["maximum"])
    default = float(details["default"])
    unit = UNITS.get(column)
    label = display_label(column) + (f" ({unit})" if unit else "")
    help_text = "Typical training range: " + f"{lower:,.0f} to {upper:,.0f}" + (f" {unit}" if unit else "")
    if column == "age":
        return st.number_input(
            label, min_value=0.0, max_value=max(100.0, upper), value=max(0.0, default),
            step=1.0, format="%.0f", help=help_text,
        )
    kwargs: dict = {}
    if all(float(value).is_integer() for value in (lower, upper, default)):
        kwargs = {"format": "%.0f", "step": 1.0}
    return st.number_input(
        label, min_value=min(0.0, lower), max_value=max(lower + 1.0, upper), value=default,
        help=help_text, **kwargs,
    )


def grouped_fields(profile: dict) -> list[tuple[str, list[str]]]:
    grouped: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for title, columns in FIELD_GROUPS:
        present = [column for column in columns if column in profile]
        if present:
            grouped.append((title, present))
            seen.update(present)
    leftover = [column for column in profile if column not in seen]
    if leftover:
        grouped.append(("Other details", leftover))
    return grouped


def render_form(profile: dict) -> bool:
    values: dict[str, object] = {}
    with st.form("vehicle_form", border=False):
        for index, (title, columns) in enumerate(grouped_fields(profile)):
            suffix = " first" if index == 0 else ""
            st.markdown(f'<div class="field-group{suffix}">{title}</div>', unsafe_allow_html=True)
            for start in range(0, len(columns), 3):
                row = st.columns(3, gap="medium")
                for container, feature in zip(row, columns[start:start + 3]):
                    with container:
                        details = profile[feature]
                        if details["kind"] == "numeric":
                            values[feature] = number_input(feature, details)
                        else:
                            choices = details["values"] or ["Unknown"]
                            values[feature] = st.selectbox(
                                display_label(feature), choices, index=0,
                                help="Values seen in the training data.",
                                placeholder="Search…",
                            )
        submitted = st.form_submit_button(
            "Estimate resale price", type="primary", width="stretch",
            help="Runs the selected model on the values above.",
        )
    if submitted:
        st.session_state["vehicle_values"] = values
    return submitted


def comparison_table(metrics: pd.DataFrame, selected: str):
    table = metrics.rename(columns={"model": "Model", "MAE": "MAE", "RMSE": "RMSE", "R2": "R²"}).copy()
    table["Model"] = table["Model"].astype(str)
    winners = table.index[table["Model"] == selected]
    if len(winners):
        position = winners[0]
        table.loc[position, "Model"] = "★ " + table.loc[position, "Model"]
    styler = table.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R²": "{:.3f}"}).hide(axis="index")
    return styler.set_table_styles([
        {"selector": "th", "props": [
            ("text-align", "left"), ("font-weight", "700"), ("font-size", "0.78rem"),
            ("text-transform", "uppercase"), ("letter-spacing", "0.05em"), ("color", "#5a657a"),
            ("background-color", "#f7f9fc"), ("padding", "0.55rem 0.7rem"),
            ("border-bottom", "1px solid #e5e9f2"),
        ]},
        {"selector": "td", "props": [
            ("text-align", "right"), ("padding", "0.6rem 0.7rem"),
            ("border-bottom", "1px solid #eef1f7"), ("font-size", "0.92rem"), ("color", "#24304a"),
        ]},
        {"selector": "th:first-child, td:first-child", "props": [("text-align", "left")]},
        {"selector": "tbody tr:last-child td", "props": [("border-bottom", "none")]},
        {"selector": "tbody tr:first-child td", "props": [
            ("background-color", "#eaf2fc"), ("font-weight", "700"), ("color", "#123a68"),
        ]},
    ])


def impact_table(impact: pd.DataFrame) -> "pd.io.formats.style.Styler":
    total = float(impact["importance"].sum()) or 1.0
    top = impact.head(10).copy()
    table = pd.DataFrame({
        "Input": [display_label(feature) for feature in top["feature"]],
        "Share of impact": [f"{value / total * 100:.1f}%" for value in top["importance"]],
    })
    styler = table.style.hide(axis="index")
    return styler.set_table_styles([
        {"selector": "th", "props": [
            ("text-align", "left"), ("font-weight", "700"), ("font-size", "0.78rem"),
            ("text-transform", "uppercase"), ("letter-spacing", "0.05em"), ("color", "#5a657a"),
            ("background-color", "#f7f9fc"), ("padding", "0.5rem 0.7rem"),
            ("border-bottom", "1px solid #e5e9f2"),
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"), ("padding", "0.5rem 0.7rem"),
            ("border-bottom", "1px solid #eef1f7"), ("font-size", "0.9rem"), ("color", "#24304a"),
        ]},
        {"selector": "th:last-child, td:last-child", "props": [
            ("text-align", "right"), ("font-weight", "700"), ("color", "#1d5aa6"),
        ]},
        {"selector": "tbody tr:last-child td", "props": [("border-bottom", "none")]},
    ])


st.set_page_config(page_title="AutoValue | Price Estimator", page_icon="🚘", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(STYLES, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.warning("Your dashboard is ready. Train a model to activate estimates.")
    st.code("py -m pip install -r requirements.txt\npy train.py --data data/raw/used_cars.csv", language="powershell")
    st.info("Download the supplied Kaggle dataset, extract its CSV, and place it in `data/raw/` before running the training command.")
    st.stop()

bundle = load_bundle()
profile = bundle["input_profile"]
metrics = bundle.get("metrics")
if metrics is None and METRICS_PATH.exists():
    metrics = pd.read_csv(METRICS_PATH)
if metrics is None or len(metrics) == 0:
    st.warning("Model metrics are missing. Re-run training to regenerate the artifacts folder.")
    st.code("py train.py --data data/raw/used_cars.csv", language="powershell")
    st.stop()
selected_model = str(bundle.get("model_name", ""))
best = metrics.iloc[0]
worst = metrics.iloc[-1]

section("Vehicle details", "Use values that closely match the vehicle — fields come straight from the training dataset.")
submitted = render_form(profile)

values = st.session_state.get("vehicle_values")
if submitted or values:
    prediction = max(0.0, float(bundle["model"].predict(pd.DataFrame([values]))[0]))
    mae = float(best["MAE"])
    st.markdown(f"""
    <div class="result">
      <div class="label">Estimated resale price</div>
      <div class="value">{prediction:,.0f}</div>
      <div class="meta">
        <div><span>Model</span><b>{selected_model}</b></div>
        <div><span>Validation R²</span><b>{float(best['R2']):.3f}</b></div>
        <div><span>Typical error</span><b>±{mae:,.0f}</b></div>
        <div><span>Candidates compared</span><b>{len(metrics)}</b></div>
      </div>
      <div class="caption">Price uses the currency and unit in the training CSV. This is an estimate, not an offer.</div>
    </div>
    """, unsafe_allow_html=True)

section("Model performance", "Every candidate trained on the same split; lowest RMSE wins.")
kpis = st.columns(4)
kpis[0].metric("Selected model", selected_model, delta="lowest validation RMSE", delta_color="off")
kpis[1].metric(
    "Validation R²", f"{float(best['R2']):.3f}",
    delta=f"{float(best['R2']) - float(worst['R2']):+.3f} vs weakest",
)
kpis[2].metric(
    "Validation RMSE", f"{float(best['RMSE']):,.0f}",
    delta=f"{float(best['RMSE']) - float(worst['RMSE']):+,.0f} vs weakest", delta_color="inverse",
)
kpis[3].metric(
    "Validation MAE", f"{float(best['MAE']):,.0f}",
    delta=f"{float(best['MAE']) - float(worst['MAE']):+,.0f} vs weakest", delta_color="inverse",
)

left, right = st.columns([1, 1.25], gap="large")
with left:
    with st.container(border=True):
        card_title("Validation metrics")
        st.dataframe(comparison_table(metrics, selected_model), hide_index=True, width="stretch")
        st.caption("Sorted by validation RMSE — lower is better. ★ marks the deployed model.")
with right:
    with st.container(border=True):
        card_title("Accuracy and error")
        st.pyplot(model_comparison_chart(metrics, selected_model), clear_figure=True, width="stretch")

section("Feature impact", "How much each input field moves the prediction.")
estimator = getattr(bundle["model"], "named_steps", {}).get("model")
linear_model = bool(hasattr(estimator, "coef_") and not hasattr(estimator, "feature_importances_"))
impact = None
if IMPORTANCE_PATH.exists():
    raw_importance = pd.read_csv(IMPORTANCE_PATH)
    try:
        impact = grouped_feature_importance(bundle["model"], raw_importance)
    except Exception:
        impact = None
    if impact is None:
        impact = raw_importance
        impact_note = "Top individual input levels from the deployed model."
    else:
        impact_note = (
            "Category levels are averaged per field for the linear model."
            if linear_model else "Category levels are summed per field."
        )
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        with st.container(border=True):
            card_title("Most influential inputs")
            st.pyplot(feature_impact_chart(impact, top=10), clear_figure=True, width="stretch")
    with right:
        with st.container(border=True):
            card_title("Ranked inputs")
            st.dataframe(impact_table(impact), hide_index=True, width="stretch")
            st.caption(impact_note)
else:
    st.caption("Feature-impact results will appear after training.")

dataset = load_dataset()
if dataset is not None and "price" in dataset:
    section("Dataset snapshot", "Listing prices in the training CSV behind the model above.")
    prices = dataset["price"].astype(float)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        with st.container(border=True):
            card_title("Price distribution")
            st.pyplot(price_distribution_chart(prices), clear_figure=True, width="stretch")
    with right:
        with st.container(border=True):
            card_title("Price summary")
            first, second = st.columns(2)
            first.metric("Listings", f"{len(dataset):,}")
            second.metric("Median price", f"{prices.median():,.0f}")
            third, fourth = st.columns(2)
            third.metric("Mean price", f"{prices.mean():,.0f}")
            fourth.metric("Highest price", f"{prices.max():,.0f}")
            st.caption(f"Range: {prices.min():,.0f} – {prices.max():,.0f} in the dataset's price units.")

st.caption(
    "Estimates come from the trained scikit-learn pipeline on the supplied CSV. "
    "They do not account for inspection condition, location, demand, or negotiation."
)
