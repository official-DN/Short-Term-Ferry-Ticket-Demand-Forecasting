from pathlib import Path
from datetime import datetime, time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Short Ferry Demand Forecasting", page_icon="⛴️", layout="wide")

# ---------------- Theme setup (CSS applied across the app) ----------------
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "Dark"

def get_theme_colors(mode):
    if mode == "Dark":
        return {
            "bg": "#0E1117",
            "panel": "#161B22",
            "text": "#E6EDF3",
            "muted": "#9BA7B4",
            "border": "#30363D",
        }
    return {
        "bg": "#FFFFFF",
        "panel": "#F4F6F9",
        "text": "#17202A",
        "muted": "#596579",
        "border": "#D9E0E8",
    }

theme = get_theme_colors(st.session_state.app_theme)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {theme["bg"]};
        color: {theme["text"]};
    }}
    [data-testid="stSidebar"] {{
        background-color: {theme["panel"]};
    }}
    [data-testid="stHeader"] {{
        background-color: {theme["bg"]};
    }}
    [data-testid="stMarkdownContainer"], [data-testid="stMetric"],
    label, p, h1, h2, h3, h4, h5, h6 {{
        color: {theme["text"]};
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        background-color: {theme["panel"]};
    }}
    div[data-testid="stAlert"] {{
        border-color: {theme["border"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


ROOT = Path(__file__).resolve().parent
DP = ROOT / "dataProcess"
DATA_CANDIDATES = [DP / "Toronto_Island_Ferry_Tickets.csv", ROOT / "Toronto_Island_Ferry_Tickets.csv"]

# Load the user's actual notebook-defined processing/model functions. We execute only
# definition/setup cells and never execute notebook main() blocks or visualizations.
def notebook_namespace(filename, skip_cells=()):
    path = DP / filename
    if not path.exists():
        raise FileNotFoundError(f"Required project notebook not found: {path}")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": f"loaded_{path.stem}"}
    for i, cell in enumerate(notebook.get("cells", [])):
        if i in skip_cells or cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        if not code.strip():
            continue
        # main_03models cell 1 attempts to re-open main_02train_test.ipynb;
        # skip it because we already loaded the preprocessing namespace below.
        if filename == "main_03models.ipynb" and i == 1:
            continue
        exec(compile(code, f"{path.name}:cell_{i}", "exec"), namespace)
    return namespace


@st.cache_resource(show_spinner="Loading your project pipeline...")
def load_project_pipeline():
    prep = notebook_namespace("main_02train_test.ipynb")
    models = notebook_namespace("main_03models.ipynb")
    evaluation = notebook_namespace("main_04evaluation.ipynb")
    models["evaluate_forecasts"] = evaluation["evaluate_forecasts"]
    # Reuse precisely the function definitions from main_02train_test.ipynb.
    models.update({
        "TARGETS": prep["TARGETS"], "HORIZONS": prep["HORIZONS"],
        "load_data": prep["load_data"], "build_features": prep["build_features"],
        "get_feature_columns": prep["get_feature_columns"], "make_xy": prep["make_xy"],
        "time_split": prep["time_split"],
        "rolling_origin_splits": prep["rolling_origin_splits"],
    })
    # Import the user's KPI functions directly.
    kpi_path = ROOT / "key_performence" / "indicator.py"
    score_path = ROOT / "key_performence" / "show_score.py"
    if kpi_path.exists():
        kpi_ns = {}
        exec(compile(kpi_path.read_text(encoding="utf-8"), str(kpi_path), "exec"), kpi_ns)
        models["calculate_kpis"] = kpi_ns["calculate_kpis"]
    if score_path.exists():
        score_ns = {}
        exec(compile(score_path.read_text(encoding="utf-8"), str(score_path), "exec"), score_ns)
        models["regression_scores"] = score_ns["regression_scores"]
        models["classification_scores"] = score_ns["classification_scores"]
    return models


@st.cache_data(show_spinner="Preparing dataset with main_02train_test.ipynb...")
def prepare_data(csv_path):
    pipeline = load_project_pipeline()
    raw = pipeline["load_data"](csv_path)
    features = pipeline["build_features"](raw)
    return raw, features


@st.cache_data(show_spinner="Training models with main_03models.ipynb...")
def run_holdout_evaluation(csv_path, target, horizon):
    p = load_project_pipeline()
    _, features = prepare_data(csv_path)
    X, y = p["make_xy"](features, target, horizon)
    Xtr, Xte, ytr, yte = p["time_split"](X, y, test_fraction=0.20)
    rows, predictions = [], []
    # Baselines and estimators are defined by the original model notebook.
    baselines = {
        "NaiveLastTrainingValue": np.full(len(yte), float(ytr.iloc[-1])),
        "MovingAverage4TrainingValues": np.full(len(yte), float(ytr.tail(4).mean())),
    }
    for name, pred in baselines.items():
        rows.append({"Model": name, **p["metrics"](yte, pred)})
        predictions.append(pd.DataFrame({"Timestamp": Xte.index, "Actual": yte.to_numpy(), "Predicted": pred, "Model": name}))
    for name, estimator in p["build_models"]().items():
        estimator.fit(Xtr, ytr)
        pred = np.maximum(estimator.predict(Xte), 0)
        rows.append({"Model": name, **p["metrics"](yte, pred)})
        predictions.append(pd.DataFrame({"Timestamp": Xte.index, "Actual": yte.to_numpy(), "Predicted": pred, "Model": name}))
    score_df = pd.DataFrame(rows).sort_values("MAE")
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df["Target"] = target
    pred_df["Horizon"] = horizon
    return score_df, pred_df, (Xtr, Xte, ytr, yte)


@st.cache_data(show_spinner="Fitting selected project model for forecast...")
def forecast_at_origin(csv_path, target, horizon, model_name, origin_timestamp):
    p = load_project_pipeline()
    _, features = prepare_data(csv_path)
    X, y = p["make_xy"](features, target, horizon)
    origin = pd.Timestamp(origin_timestamp)
    # Fit only on training samples whose forecast origin is at/before selected origin.
    eligible = X.index <= (origin - pd.Timedelta(minutes=horizon))
    X_train, y_train = X.loc[eligible], y.loc[eligible]
    if len(X_train) < 100:
        raise ValueError("Not enough eligible historical training rows at this forecast origin.")
    candidate = features.loc[features.index <= origin]
    feature_cols = p["get_feature_columns"](features)
    candidate = candidate.dropna(subset=feature_cols)
    if candidate.empty:
        raise ValueError("No complete feature row at the selected date/time.")
    actual_origin = candidate.index[-1]
    Xrow = candidate.loc[[actual_origin], feature_cols]
    if model_name == "NaiveLastTrainingValue":
        prediction = float(y_train.iloc[-1])
        model = None
    elif model_name == "MovingAverage4TrainingValues":
        prediction = float(y_train.tail(4).mean())
        model = None
    else:
        model = p["build_models"]()[model_name]
        model.fit(X_train, y_train)
        prediction = float(np.maximum(model.predict(Xrow)[0], 0))
    return actual_origin, prediction, model, X_train, y_train


path = next((p for p in DATA_CANDIDATES if p.exists()), None)
if path is None:
    st.error("Dataset missing. Put Toronto_Island_Ferry_Tickets.csv inside dataProcess/.")
    st.stop()
try:
    pipeline = load_project_pipeline()
    raw, data = prepare_data(str(path))
except Exception as e:
    st.error(f"Could not load original project notebooks/pipeline: {e}")
    st.stop()

TARGETS = pipeline["TARGETS"]
HORIZONS = pipeline["HORIZONS"]


# The Sidebar Panel

# the sidebar controls are used to select the forecast target, horizon, origin date/time, and alert threshold. The selected values are used to produce the forecast and evaluation tables in the main body of the app.

st.sidebar.title("⛴️ Forecast Controls")
st.sidebar.caption("Using original Data ")
target = st.sidebar.selectbox("Forecast target", TARGETS)
horizon = st.sidebar.selectbox("Forecast horizon", HORIZONS, format_func=lambda x: {15:"15 minutes",30:"30 minutes",60:"1 hour",120:"2 hours"}.get(x, f"{x} minutes"), index=1)
latest = data.index.max()
min_date, max_date = data.index.min().date(), latest.date()
selected_date = st.sidebar.date_input("Forecast origin date", value=max_date, min_value=min_date, max_value=max_date)
selected_time = st.sidebar.time_input("Forecast origin time", value=time(latest.hour, latest.minute))
forecast_origin = pd.Timestamp(datetime.combine(selected_date, selected_time))
peak_threshold = st.sidebar.number_input("Peak alert threshold", min_value=1.0, value=float(max(20, data[target].quantile(.90))), step=5.0)
interval_level = st.sidebar.select_slider("Confidence interval", options=[80, 90, 95], value=90)



# Body of the Streamlit app. The original project notebooks are used to produce the forecast, evaluation tables, and KPI metrics.

st.title("⛴️ Short-Term Ferry Ticket Demand Forecasting")
st.markdown("**Predictive Decision Support System** · Toronto Island Ferry")
st.caption("Preprocessing, feature engineering, model estimators, metrics and KPI functions are loaded from your original project notebooks and Python modules.")

try:
    score_df, holdout_predictions, split_data = run_holdout_evaluation(str(path), target, horizon)
    # Produce the notebook main_04evaluation.ipynb metric tables from holdout outputs.
    eval_input = holdout_predictions.assign(Target=target, Horizon=horizon)
    eval_overall, eval_by_horizon = pipeline["evaluate_forecasts"](eval_input)
    Xtr, Xte, ytr, yte = split_data
    available_models = score_df["Model"].tolist()
    selected_model = st.sidebar.selectbox("Model selection", available_models, index=0)
    origin, prediction, model, X_train, y_train = forecast_at_origin(str(path), target, horizon, selected_model, forecast_origin)
except Exception as e:
    st.error(f"Model evaluation/forecast failed: {e}")
    st.stop()

# The user's uncertainty notebook functions are applied to the holdout predictions.
try:
    uncertainty_ns = notebook_namespace("main_05uncertainty_ana.ipynb")
    uncertainty_input = holdout_predictions.rename(columns={"Timestamp":"Forecast_Origin"}).copy()
    uncertainty_input["Target"] = target
    uncertainty_input["Horizon"] = horizon
    uncertainty_input["Model"] = selected_model
    chosen_holdout = uncertainty_input[uncertainty_input["Model"] == selected_model].copy()
    intervals = uncertainty_ns["add_prediction_intervals"](chosen_holdout, level=interval_level/100)
    radius = float(intervals["Interval_Radius"].median()) if not intervals.empty else 0.0
except Exception:
    # Use the selected model's holdout residuals as a conservative empirical fallback.
    chosen = holdout_predictions[holdout_predictions["Model"] == selected_model]
    radius = float((chosen["Actual"] - chosen["Predicted"]).abs().quantile(interval_level/100)) if len(chosen) else 0.0
lower, upper = max(0.0, prediction-radius), prediction+radius

recent = data.loc[:origin, target].dropna()
current_demand = float(recent.iloc[-1]) if not recent.empty else np.nan
cols = st.columns(4)
cols[0].metric("Current demand", f"{current_demand:,.0f}" if pd.notna(current_demand) else "—")
cols[1].metric(f"Forecast +{horizon}m", f"{prediction:,.1f}")
cols[2].metric(f"{interval_level}% interval", f"{lower:,.1f}–{upper:,.1f}")
cols[3].metric("Peak threshold", f"{peak_threshold:,.0f}")
if prediction >= peak_threshold:
    st.error(f"🚨 Peak-demand alert: forecast {prediction:,.0f} meets/exceeds threshold {peak_threshold:,.0f}.")
else:
    st.success("Forecast is below the configured peak-demand threshold.")

st.subheader("Historical demand and forecast")
hist = data.loc[:origin, target].tail(192).dropna()
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Observed demand"))
future = origin + pd.Timedelta(minutes=horizon)
fig.add_trace(go.Scatter(x=[origin, future], y=[current_demand, prediction], mode="lines+markers", name="Forecast", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=[future, future], y=[lower, upper], mode="lines", name=f"{interval_level}% interval", line=dict(width=8), opacity=.35))
fig.update_layout(
    height=430,
    xaxis_title="Timestamp",
    yaxis_title=target,
    template="plotly_dark" if st.session_state.app_theme == "Dark" else "plotly_white",
    paper_bgcolor=theme["bg"],
    plot_bgcolor=theme["bg"],
    font_color=theme["text"],
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Model comparison · original model notebook")

# from main_03models.ipynb
st.caption("Chronological 80/20 holdout; scores are produced using the metrics() function.")
st.dataframe(score_df, hide_index=True, use_container_width=True)

#from main_04evaluation.ipynb
with st.expander("Evaluation tables "):
    st.markdown("Overall and horizon-wise evaluation for the models evaluated on the selected target/horizon.")
    st.dataframe(eval_overall, hide_index=True, use_container_width=True)
    st.dataframe(eval_by_horizon, hide_index=True, use_container_width=True)
with st.expander("Actual vs predicted holdout values"):
    comp = holdout_predictions[holdout_predictions["Model"] == selected_model]
    cf = go.Figure()
    cf.add_trace(go.Scatter(
        x=comp["Timestamp"],
        y=comp["Actual"],
        mode="lines",
        name="Actual",
        line=dict(color="rgb(18, 216, 80)"),
    ))

    cf.add_trace(go.Scatter(
        x=comp["Timestamp"],
        y=comp["Predicted"],
        mode="lines",
        name="Predicted",
        line=dict(color="rgb(0, 150, 255)"),
    ))
    cf.update_layout(
        height=350,
        xaxis_title="Timestamp",
        yaxis_title=target,
        template="plotly_dark" if st.session_state.app_theme == "Dark" else "plotly_white",
        paper_bgcolor=theme["bg"],
        plot_bgcolor=theme["bg"],
        font_color=theme["text"],
    )
    st.plotly_chart(cf, use_container_width=True)

#· key_performence/indicator.py
st.subheader("Forecast KPI indicators ")
kpi_input = holdout_predictions[holdout_predictions["Model"] == selected_model].copy()
kpi_input["Horizon"] = horizon
try:
    kpi_df = pipeline["calculate_kpis"](kpi_input.rename(columns={"Model":"Model"}))
    st.dataframe(kpi_df, hide_index=True, use_container_width=True)
except Exception as e:
    st.info(f"KPI module could not calculate metrics for this selection: {e}")

#· key_performence/show_score.py
st.subheader("Model performance scores ")
try:
    reg_scores = pipeline["regression_scores"](holdout_predictions.assign(Target=target, Horizon=horizon))
    st.dataframe(reg_scores, hide_index=True, use_container_width=True)
    cls_scores = pipeline["classification_scores"](holdout_predictions.assign(Target=target, Horizon=horizon))
    if cls_scores is not None:
        st.caption("Peak-event classification metrics (derived peak classes)")
        st.dataframe(cls_scores, hide_index=True, use_container_width=True)
except Exception as e:
    st.info(f"Score module could not calculate metrics: {e}")

st.subheader("Operational recommendations")
if prediction >= peak_threshold:
    st.markdown("- **Prepare for elevated demand:** review staffing and ticketing coverage.\n- **Queue readiness:** prepare additional service points or ticket-channel guidance.\n- **Monitor:** refresh forecasts as new ticket counts arrive.")
elif prediction >= peak_threshold * .75:
    st.markdown("- **Watch demand:** keep staffing flexible.\n- **Prepare contingency:** queue measures may be needed if demand rises.")
else:
    st.markdown("- **Normal operating posture:** use expected demand for routine planning.\n- **Continue monitoring:** refresh forecast when newer observations are available.")



#st.subheader("Forecasting assistant")
#st.caption("Project-grounded chat representation using the currently selected forecast and score tables.")
#

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])
question = st.chat_input("Ask about demand, model scores, confidence, alerts, or recommendations...")
if question:
    st.session_state.chat_history.append({"role":"user", "content":question})
    q = question.lower()
    if any(k in q for k in ["score", "accuracy", "mae", "rmse", "best model", "performance"]):
        best = score_df.iloc[0]
        answer = f"For {target} at +{horizon} minutes, the lowest holdout MAE model is {best['Model']} (MAE {best['MAE']:.2f}, RMSE {best['RMSE']:.2f}, sMAPE {best['sMAPE_percent']:.2f}%)."
    elif any(k in q for k in ["peak", "busy", "alert"]):
        answer = f"The forecast is {prediction:.1f}, and the alert threshold is {peak_threshold:.1f}. " + ("This is a peak alert; review staffing and queue management." if prediction >= peak_threshold else "The forecast is below the peak threshold.")
    elif any(k in q for k in ["confidence", "interval", "uncertainty", "band"]):
        answer = f"The empirical {interval_level}% interval is {lower:.1f} to {upper:.1f}. It is estimated from holdout absolute errors; it is not a guaranteed probabilistic bound."
    elif any(k in q for k in ["recommend", "staff", "queue", "operation"]):
        answer = "Use the selected forecast as a planning signal, check recent actual ticket counts, and keep staffing/queue measures flexible around predicted peaks. Refresh forecasts as new data arrive."
    else:
        answer = f"The current selected project model is {selected_model}, forecasting {target} +{horizon} minutes at {origin}. The forecast is {prediction:.1f}. You can ask about model scores, intervals, peak alerts, or operational recommendations."
    st.session_state.chat_history.append({"role":"assistant", "content":answer})
    st.rerun()


#with st.expander("Project objectives and modules"):
#   st.markdown("**Primary objectives**\n- Forecast short-term ferry ticket sales and redemptions.\n- Predict demand for 15m, 30m, 1h, and 2h windows.\n- Compare statistical and ML forecasting approaches.\n\n**Secondary objectives**\n- Quantify uncertainty.\n- Support proactive operations.\n- Demonstrate ML deployment with Streamlit.\n\n**Integrated project files:** main_01.ipynb (data understanding), main_02train_test.ipynb (preprocessing/features/chronological split), main_03models.ipynb (model training/metrics), main_04evaluation.ipynb (evaluation functions), main_05uncertainty_ana.ipynb (interval analysis), indicator.py (KPIs), show_score.py (performance scores).")
#


# Sidebar caption to clarify that the forecasts are based on historical data and not live ticketing data.
st.sidebar.caption("Forecasts use the uploaded historical dataset and are not connected to live ferry ticketing data.")

# Theme switcher is intentionally placed at the bottom of the sidebar.
st.sidebar.divider()
st.sidebar.caption("Appearance")
if st.session_state.app_theme == "Dark":
    if st.sidebar.button("☀️ Switch to Light Mode", key="theme_toggle_bottom", use_container_width=True):
        st.session_state.app_theme = "Light"
        st.rerun()
else:
    if st.sidebar.button("🌙 Switch to Dark Mode", key="theme_toggle_bottom", use_container_width=True):
        st.session_state.app_theme = "Dark"
        st.rerun()
