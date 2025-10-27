import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import time
import os

# ==============================================================================
# 1. CONFIGURATION AND STYLING (Req 1 & 5)
# All parameters are now centralized for quick "re-skinning"
# ==============================================================================
CONFIG = {
    # Req 5: Operational Flexibility - Asset Details
    "ASSET_NAME": "Kroonstad Maize Mill Pump #4",
    "INDUSTRY": "Agricultural Processing",
    "FAILURE_TYPE": "Bearing Seizure (High Temp/Vibe)",
    # Req 1: Branding and Theme
    "PRIMARY_COLOR": "#0052CC",  # Blue - Professional and clean
    "SECONDARY_COLOR": "#FFD700",  # Gold/Yellow - For alerts
    "LOGO_TEXT": "S.I.L.K.E AI",
    "FONT_FAMILY": "Roboto, sans-serif",
    # Req 4: ROI Calculator Defaults
    "DEFAULT_DOWNTIME_COST_PER_HOUR": 50000,  # R50,000
    "DEFAULT_FAILURES_PREVENTED_PER_YEAR": 4,
    "ASSUMED_HOURS_SAVED_PER_FAILURE": 12,
    "POC_COST": 50000,  # R50,000 PoC price
    # Req 3: AI Prediction Logic Tuning (for IsolationForest score mapping)
    "SCORE_THRESHOLD_90_PERCENT": -0.15,  # Score that corresponds to 90% confidence
    "MAX_NORMAL_SCORE": 0.05,  # Max score for 0% confidence
}

st.set_page_config(
    layout="wide",
    page_title=f"{CONFIG['LOGO_TEXT']} Predictive Maintenance Demo",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Branding (Req 1)
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        /* General Font */
        .stApp {{
            font-family: {CONFIG["FONT_FAMILY"]};
            color: #1F2937; /* Dark Gray text */
        }}
        /* Main Header */
        h1 {{
            color: {CONFIG["PRIMARY_COLOR"]};
            font-weight: 700;
            border-bottom: 2px solid #E5E7EB;
            padding-bottom: 10px;
        }}
        /* KPIs and Metrics */
        [data-testid="stMetricValue"] {{
            font-size: 2.2rem;
            color: #111827;
        }}
        /* AI Confidence Gauge/Card */
        .confidence-card {{
            background-color: #F3F4F6;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid {CONFIG["PRIMARY_COLOR"]};
        }}
        .roi-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #10B981; /* Green for savings */
            margin-top: 10px;
        }}
        /* Sidebar branding */
        .sidebar .st-h2 {{
            color: {CONFIG["PRIMARY_COLOR"]};
        }}
    </style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. DATA LOADING AND MODEL INITIALIZATION
# ==============================================================================

# Note: Using a mock file name. The user must ensure this file exists
file_path = "maize_mill_simulated_sensor_data.csv"


@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the asset sensor data."""
    try:
        df = pd.read_csv(file_path, parse_dates=["Timestamp"])
        df = df.set_index("Timestamp").sort_index()

        # Calculate features needed for rule-based detection
        window_size = "12h"
        df["Power_kW_RollingMean"] = df["Power_kW"].rolling(window=window_size).mean()
        df["Power_kW_RollingStd"] = df["Power_kW"].rolling(window=window_size).std()
        df["Amperage_RollingMean"] = df["Amperage"].rolling(window=window_size).mean()
        df["Amperage_RollingStd"] = df["Amperage"].rolling(window=window_size).std()

        # Add columns for later AI prediction/analysis
        df["Is_ML_Anomaly"] = False
        df["ML_Anomaly_Score"] = 0.0
        df["AI_Confidence_Score"] = 0.0

        return df
    except Exception as e:
        st.error(f"Error loading or processing data from CSV: {e}")
        st.stop()


@st.cache_resource
def train_isolation_forest(df_initial):
    """Trains the Isolation Forest model on initial, healthy data."""
    # We train on the first 1000 data points assuming they represent 'normal' operation
    df_train = df_initial.head(1000).copy()

    features_for_ml = ["Power_kW", "Amperage", "Vibration", "Temperature"]
    X_train = df_train[features_for_ml].dropna()

    if X_train.empty or len(X_train) < 100:
        return None, None

    # Use auto contamination to be safe, or a defined value (0.01 is fine)
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train.values)
    return model, features_for_ml


# Global data and model initialization
full_data_df = load_data(file_path)
iso_forest_model, ml_features = train_isolation_forest(full_data_df)


# ==============================================================================
# 3. CORE PREDICTION AND ANOMALY LOGIC (Req 3)
# ==============================================================================


def calculate_ml_confidence(raw_score):
    """
    Scales the raw Isolation Forest score to a 0-100 AI Confidence Score.
    Lower (more negative) raw scores mean higher anomaly confidence.
    """
    if iso_forest_model is None:
        return 0.0

    # Linear scaling: Map MAX_NORMAL_SCORE to 0% and SCORE_THRESHOLD_90_PERCENT to 90%
    # We define a range for normalization
    score_range = CONFIG["MAX_NORMAL_SCORE"] - CONFIG["SCORE_THRESHOLD_90_PERCENT"]

    if raw_score >= CONFIG["MAX_NORMAL_SCORE"]:
        confidence = 0.0
    else:
        # Distance from the 'normal' threshold
        distance_from_normal = CONFIG["MAX_NORMAL_SCORE"] - raw_score

        # Calculate a factor based on how far we are into the anomaly range
        factor = distance_from_normal / score_range

        # Scale factor to 0-100, capping at 100
        confidence = min(100.0, factor * 90.0)  # 90 is the confidence at the threshold

    return round(confidence, 1)


def process_data_slice(df_slice, model, features):
    """Applies ML prediction and calculates confidence for an entire data slice."""

    # Check for anomalies using the ML model
    if model is not None and not df_slice.empty:
        X = df_slice[features].values

        # Get raw scores
        raw_scores = model.decision_function(X)
        df_slice.loc[:, "ML_Anomaly_Score"] = raw_scores

        # Determine if it's an anomaly (-1) or normal (1)
        ml_predictions = model.predict(X)
        df_slice.loc[:, "Is_ML_Anomaly"] = ml_predictions == -1

        # Calculate the confidence score based on the raw score
        df_slice.loc[:, "AI_Confidence_Score"] = df_slice["ML_Anomaly_Score"].apply(
            calculate_ml_confidence
        )

    return df_slice


def get_prediction_metrics(df_slice):
    """Calculates the key AI prediction dates and confidence."""

    latest_confidence = df_slice["AI_Confidence_Score"].iloc[-1]

    # 90%+ confidence is the critical prediction moment (Req 3)
    critical_anomalies = df_slice[df_slice["AI_Confidence_Score"] >= 90.0]

    anomaly_flag_date = None
    predicted_failure_date = None

    # The ultimate failure date is the last point in the simulation data
    ultimate_failure_date = full_data_df.index[-1]

    if not critical_anomalies.empty:
        # Find the first time the AI confidence jumped over 90%
        anomaly_flag_date = critical_anomalies.index[0]

        # Estimate the predicted failure date: 15-30 days after the flag
        # For the demo, we'll make the prediction 25 days after the flag
        # The ultimate failure date will always be the last point of data
        # Let's ensure the prediction is in the future relative to the current slice

        days_to_predict = 25
        # The difference between the ultimate failure and the anomaly flag time is the true lead time
        time_diff_to_failure = ultimate_failure_date - anomaly_flag_date

        # If the critical anomaly happened in the current slice, and we still have time left:
        if anomaly_flag_date <= df_slice.index[-1]:
            # The predicted failure date is the anomaly flag date + 25 days (simulated lead time)
            # This makes the prediction actionable
            predicted_failure_date = anomaly_flag_date + timedelta(days=days_to_predict)

            # Ensure the displayed predicted failure date is always relative to the current time,
            # and only shows if the anomaly flag has been hit.
            if predicted_failure_date < df_slice.index[-1]:
                # If the predicted date has already passed, use the ultimate failure date
                predicted_failure_date = ultimate_failure_date

    if predicted_failure_date is None:
        # Default prediction window if no anomaly is found yet
        predicted_failure_date = df_slice.index[-1] + timedelta(days=30)

    # Days left until the predicted failure (from the current latest timestamp)
    days_left = (predicted_failure_date - df_slice.index[-1]).days

    return latest_confidence, anomaly_flag_date, predicted_failure_date, days_left


# ==============================================================================
# 4. ROI CALCULATOR LOGIC (Req 4)
# ==============================================================================


def calculate_roi(downtime_cost, failures_prevented, hours_saved):
    """Calculates potential annual savings."""
    annual_savings = (downtime_cost * hours_saved) * failures_prevented
    roi_multiplier = annual_savings / CONFIG["POC_COST"]
    return annual_savings, roi_multiplier


# ==============================================================================
# 5. UI RENDERING FUNCTIONS (Role-Based Views - Req 1)
# ==============================================================================


def render_plant_manager_view(
    df_slice, latest_confidence, predicted_failure_date, days_left
):
    """Renders the financial and executive summary focused dashboard."""

    st.subheader(f"Executive Summary for {CONFIG['ASSET_NAME']}")

    # --- Financial and Prediction KPIs (Req 4) ---
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

    with col1:
        st.markdown('<div class="confidence-card">', unsafe_allow_html=True)
        st.markdown(f"**AI Failure Confidence** (0-100%)")
        st.markdown(f"## {latest_confidence:.1f}%")
        st.progress(latest_confidence / 100)
        st.markdown("</div>", unsafe_allow_html=True)

    col2.metric(
        label="Predicted Failure Date (Actionable)",
        value=(
            predicted_failure_date.strftime("%Y-%m-%d")
            if days_left > 0
            else "FAILURE IMMINENT"
        ),
        delta=f"{days_left} Days Lead Time" if days_left > 0 else None,
        delta_color="inverse" if days_left > 15 else "normal",
    )

    col3.metric(
        label="Expected Failures Preventable (Annual)",
        value=f"{CONFIG['DEFAULT_FAILURES_PREVENTED_PER_YEAR']} ",
    )
    col4.metric(
        label="Asset Health Status",
        value=(
            "CRITICAL"
            if latest_confidence >= 90
            else "WARNING" if latest_confidence >= 50 else "NORMAL"
        ),
        delta="Immediate Attention Required" if latest_confidence >= 90 else None,
        delta_color="inverse",
    )

    st.markdown("---")

    # --- ROI Calculator (Req 4) ---
    st.header("Built-in ROI Calculator")
    st.write(
        "Plug in your company's actual numbers to see the immediate financial justification."
    )

    cost_col, failures_col, hours_col, savings_col = st.columns(4)

    with cost_col:
        downtime_cost = st.number_input(
            "Cost of Unplanned Downtime (R/Hour)",
            min_value=1000,
            value=CONFIG["DEFAULT_DOWNTIME_COST_PER_HOUR"],
            step=5000,
            format="%d",
        )

    with failures_col:
        failures_prevented = st.number_input(
            "Estimated Failures Prevented Per Year",
            min_value=1,
            value=CONFIG["DEFAULT_FAILURES_PREVENTED_PER_YEAR"],
            step=1,
            format="%d",
        )

    with hours_col:
        hours_saved = st.number_input(
            "Average Hours Saved per Event",
            min_value=1,
            value=CONFIG["ASSUMED_HOURS_SAVED_PER_FAILURE"],
            step=1,
            format="%d",
        )

    annual_savings, roi_multiplier = calculate_roi(
        downtime_cost, failures_prevented, hours_saved
    )

    with savings_col:
        st.markdown("### Potential Annual Savings:")
        st.markdown(
            f'<div class="roi-value">R {annual_savings:,.0f}</div>',
            unsafe_allow_html=True,
        )

    st.info(
        f"**ROI Justification:** Based on these figures, the potential annual savings of R {annual_savings:,.0f} "
        f"is **{roi_multiplier:.1f}x** the cost of a R{CONFIG['POC_COST']:,.0f} PoC. The solution quickly pays for itself."
    )

    # --- Summary Data Table ---
    with st.expander("Show Latest Sensor Readings"):
        st.dataframe(
            df_slice.iloc[-1:].reset_index().rename(columns={"index": "Timestamp"}),
            use_container_width=True,
        )


def render_technician_view(df_slice, latest_confidence, anomaly_flag_date):
    """Renders the detailed sensor data and technical analysis focused dashboard."""

    st.subheader(f"Detailed Technical Analysis for {CONFIG['ASSET_NAME']}")

    # --- AI Confidence and Current Status ---
    col_status, col_conf = st.columns([2, 1])

    with col_status:
        st.markdown(
            f"**Current Asset State:** {'Anomaly Detected' if latest_confidence >= 50 else 'Normal Operation'} "
            f"| **Last Reading:** {df_slice.index[-1].strftime('%Y-%m-%d %H:%M')}"
        )

    with col_conf:
        st.metric("AI Confidence", f"{latest_confidence:.1f}%")

    st.markdown("---")

    # --- Main Time-Series Charts (Req 2 & 3 - Anomaly Highlight) ---
    st.header("Time-Series Data Stream (Vibration, Temp, Amperage)")

    # We will only plot 3 data streams as requested (Vibration, Temperature, Amperage)
    df_to_plot = df_slice[["Vibration", "Temperature", "Amperage"]].copy()

    fig = px.line(
        df_to_plot,
        y=["Vibration", "Temperature", "Amperage"],
        labels={"value": "Sensor Reading", "Timestamp": "Time"},
        title=f"Sensor Data Trending Towards Failure | Confidence: {latest_confidence:.1f}%",
        height=600,
        color_discrete_map={
            "Vibration": CONFIG["PRIMARY_COLOR"],
            "Temperature": "#EF4444",  # Red
            "Amperage": "#FBBF24",  # Amber
        },
    )

    # Req 3: Anomaly Highlight
    if anomaly_flag_date is not None and latest_confidence >= 90:
        fig.add_vrect(
            x0=anomaly_flag_date,
            x1=anomaly_flag_date
            + timedelta(hours=6),  # Highlight a few hours around the flag
            fillcolor=CONFIG["SECONDARY_COLOR"],
            opacity=0.3,
            line_width=0,
            layer="below",
            name="AI Anomaly Flag",
        )
        fig.add_annotation(
            x=anomaly_flag_date,
            y=1.05,
            xref="x",
            yref="paper",
            text="AI FIRST ANOMALY FLAG (90%+ CONFIDENCE)",
            showarrow=True,
            arrowhead=2,
            arrowcolor=CONFIG["SECONDARY_COLOR"],
            font=dict(
                color=CONFIG["SECONDARY_COLOR"], size=12, family=CONFIG["FONT_FAMILY"]
            ),
            xshift=-10,
        )

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- Detailed Anomaly Report ---
    with st.expander("Detailed Anomaly Events and ML Scores"):
        anomalies_to_show = df_slice[df_slice["AI_Confidence_Score"] >= 50].copy()
        if anomalies_to_show.empty:
            st.info("No high-confidence AI anomalies detected in this timeframe.")
        else:
            st.markdown(
                "The table below shows all timestamps where the AI confidence surpassed 50%."
            )
            st.dataframe(
                anomalies_to_show[
                    [
                        "Vibration",
                        "Temperature",
                        "Amperage",
                        "ML_Anomaly_Score",
                        "AI_Confidence_Score",
                    ]
                ].sort_index(ascending=False),
                use_container_width=True,
                height=300,
            )


# ==============================================================================
# 6. MAIN APP EXECUTION FLOW
# ==============================================================================


def main():
    if full_data_df.empty or iso_forest_model is None:
        st.error(
            "Cannot run demo. Check if your CSV file is correctly formatted and contains enough data."
        )
        return

    # --- Sidebar Controls (Req 1 & 2) ---
    with st.sidebar:
        st.title(CONFIG["LOGO_TEXT"])
        st.markdown(f"**Asset:** {CONFIG['ASSET_NAME']}")
        st.markdown(f"**Failure Mode:** {CONFIG['FAILURE_TYPE']}")

        st.markdown("---")

        # Req 1: Role-Based View Selector
        user_role = st.selectbox(
            "Select User Role:", ("Plant Manager", "Technician"), key="user_role"
        )

        st.markdown("---")

        # Req 2: Dynamic Time-Series Data Simulation Control
        total_data_points = len(full_data_df)
        total_days = (full_data_df.index[-1] - full_data_df.index[0]).days

        days_to_failure = st.slider(
            "Time Until Failure Simulation (Days)",
            min_value=0,
            max_value=total_days,
            value=total_days,  # Start at the end (failure state)
            step=1,
            key="days_until_failure",
            help="Simulate asset degradation by moving this slider backward in time.",
        )

        # Calculate the end index for the data slice
        end_date = full_data_df.index[0] + timedelta(days=days_to_failure)

        # Slice the data up to the simulated end date
        df_slice = full_data_df.loc[full_data_df.index <= end_date].copy()

        # Re-run ML processing on the slice
        df_slice = process_data_slice(df_slice, iso_forest_model, ml_features)

        # Get the prediction metrics based on the current slice
        latest_confidence, anomaly_flag_date, predicted_failure_date, days_left = (
            get_prediction_metrics(df_slice)
        )

    # --- Main Dashboard Rendering (Role Switch) ---

    if user_role == "Plant Manager":
        render_plant_manager_view(
            df_slice, latest_confidence, predicted_failure_date, days_left
        )

    elif user_role == "Technician":
        render_technician_view(df_slice, latest_confidence, anomaly_flag_date)


if __name__ == "__main__":
    main()
