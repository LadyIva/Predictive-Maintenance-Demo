import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, datetime
import time
import os

# ==============================================================================
# 1. CONFIGURATION AND STYLING
# ==============================================================================
CONFIG = {
    # Req 5: Operational Flexibility - Asset Details
    "ASSET_NAME": "Kroonstad Maize Mill Pump #4",
    "INDUSTRY": "Agricultural Processing",
    "FAILURE_TYPE": "Bearing Seizure (High Temp/Vibe)",
    # Req 1: Branding and Theme
    "PRIMARY_COLOR": "#0052CC",  # Blue - Professional and clean
    "SECONDARY_COLOR": "#FFD700",  # Gold/Yellow - For alerts
    "LOGO_TEXT": "S.I.L.K.E. AI",
    "FONT_FAMILY": "Roboto, sans-serif",
    # Req 4: ROI Calculator Defaults
    "DEFAULT_DOWNTIME_COST_PER_HOUR": 50000,  # R50,000
    "DEFAULT_FAILURES_PREVENTED_PER_YEAR": 4,
    "ASSUMED_HOURS_SAVED_PER_FAILURE": 12,
    "POC_COST": 50000,  # R50,000 PoC price
    # Req 3: AI Prediction Logic Tuning (for IsolationForest score mapping)
    "SCORE_THRESHOLD_90_PERCENT": -0.15,  # Score that corresponds to 90% confidence
    "MAX_NORMAL_SCORE": 0.05,  # Max score for 0% confidence
    # Real-time RUL/Lead Time: If AI is confident, predict failure in this many days
    "PREDICTED_LEAD_TIME_DAYS": 25,
    # New Config for Real-Time Loop
    "UPDATE_INTERVAL_SECONDS": 1,  # Reverted to 1 second for the original speed
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

# --- Session State Management for Live Mode ---
if "days_progressed" not in st.session_state:
    # Start the demo at a point just before failure for maximum impact (e.g., 50% of max days)
    st.session_state.days_progressed = 1
if "playing" not in st.session_state:
    st.session_state.playing = False
if "initial_load" in st.session_state:
    del st.session_state.initial_load


# ==============================================================================
# 2. DATA LOADING AND MODEL INITIALIZATION
# ==============================================================================

file_path = "maize_mill_simulated_sensor_data.csv"


@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the asset sensor data."""
    try:
        # Load the CSV content
        if not os.path.exists(file_path):
            st.error(
                f"Error: Data file '{file_path}' not found. Please ensure the file is in the same directory."
            )
            st.stop()

        df = pd.read_csv(file_path, parse_dates=["Timestamp"])
        df = df.set_index("Timestamp").sort_index()

        # Add initial required columns for processing
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
try:
    full_data_df = load_data(file_path)
    if full_data_df.empty:
        st.error("The loaded data frame is empty. Cannot run the simulation.")
        st.stop()
    iso_forest_model, ml_features = train_isolation_forest(full_data_df)
    if iso_forest_model is None:
        st.error("ML Model could not be trained. Not enough training data.")
        st.stop()
except:
    pass


# ==============================================================================
# 3. CORE PREDICTION AND ANOMALY LOGIC
# ==============================================================================


def calculate_ml_confidence(raw_score):
    """
    Scales the raw Isolation Forest score to a 0-100 AI Confidence Score.
    Lower (more negative) raw scores mean higher anomaly confidence.
    """
    if iso_forest_model is None:
        return 0.0

    # Linear scaling: Map MAX_NORMAL_SCORE to 0% and SCORE_THRESHOLD_90_PERCENT to 90%
    score_range = CONFIG["MAX_NORMAL_SCORE"] - CONFIG["SCORE_THRESHOLD_90_PERCENT"]

    if raw_score >= CONFIG["MAX_NORMAL_SCORE"]:
        confidence = 0.0
    else:
        # Distance from the 'normal' threshold
        distance_from_normal = CONFIG["MAX_NORMAL_SCORE"] - raw_score

        # Calculate a factor based on how far we are into the anomaly range
        factor = distance_from_normal / score_range

        # Scale factor to 100, capping at 100
        confidence = min(100.0, factor * 90.0)  # 90 is the confidence at the threshold

    return round(confidence, 1)


def process_data_slice(df_slice, model, features):
    """Applies ML prediction and calculates confidence for an entire data slice."""

    # Check for anomalies using the ML model
    if model is not None and not df_slice.empty:
        # Check if any data point has a null value in the feature columns
        last_row = df_slice.iloc[[-1]]
        if last_row[features].isnull().any().any():
            # If the last row has a missing value, we cannot predict the score, use previous
            return df_slice

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
    """
    Calculates the key AI prediction dates and confidence for a real-time stream.
    The predicted failure date is a forecast: Current Time + Fixed Lead Time
    """
    if df_slice.empty:
        return 0.0, None, datetime.now(), 0

    latest_confidence = df_slice["AI_Confidence_Score"].iloc[-1]

    # The current moment in time for the simulation
    current_time = df_slice.index[-1]

    anomaly_flag_date = None
    predicted_failure_date = None
    days_left = 0

    # 90%+ confidence is the critical prediction moment (realistic action threshold)
    if latest_confidence >= 90.0:

        # 1. Find the first time the AI confidence jumped over 90% in the entire slice
        critical_anomalies = df_slice[df_slice["AI_Confidence_Score"] >= 90.0]
        if not critical_anomalies.empty:
            anomaly_flag_date = critical_anomalies.index[0]
        else:
            anomaly_flag_date = current_time  # Fallback

        # 2. Real-Time Forecasting: Calculate the Predicted Failure Date
        predicted_failure_date = current_time + timedelta(
            days=CONFIG["PREDICTED_LEAD_TIME_DAYS"]
        )

        # 3. Days left until the predicted failure
        days_left = CONFIG["PREDICTED_LEAD_TIME_DAYS"]

    else:
        # If confidence is below 90%, we set a non-actionable, far-out date
        predicted_failure_date = current_time + timedelta(days=90)
        days_left = 90

    return latest_confidence, anomaly_flag_date, predicted_failure_date, days_left


# ==============================================================================
# 4. ROI CALCULATOR LOGIC
# ==============================================================================


def calculate_roi(downtime_cost, failures_prevented, hours_saved):
    """Calculates potential annual savings."""
    annual_savings = (downtime_cost * hours_saved) * failures_prevented
    roi_multiplier = annual_savings / CONFIG["POC_COST"]
    return annual_savings, roi_multiplier


# ==============================================================================
# 5. UI RENDERING FUNCTIONS (Role-Based Views)
# ==============================================================================


def render_plant_manager_view(
    df_slice, latest_confidence, predicted_failure_date, days_left
):
    """Renders the financial and executive summary focused dashboard."""
    st.header(f"Live Asset Monitor: {CONFIG['ASSET_NAME']} (Plant Manager View)")

    # --- Financial and Prediction KPIs (Req 4) ---
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

    is_critical_alert = latest_confidence >= 90

    with col1:
        st.markdown('<div class="confidence-card">', unsafe_allow_html=True)
        st.markdown(f"**AI Failure Confidence** (0-100%)")
        st.markdown(f"## {latest_confidence:.1f}%")
        st.progress(latest_confidence / 100)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Real-Time Predictive KPI (Actionable RUL) ---
    lead_time_display = f"{days_left} Days Lead Time"

    if is_critical_alert:
        date_value_display = predicted_failure_date.strftime("%Y-%m-%d")
        delta_color = "inverse"
    else:
        date_value_display = "---"
        lead_time_display = "System Normal"
        delta_color = "normal"

    col2.metric(
        label="Predicted Failure Date (Actionable RUL)",
        value=date_value_display,
        delta=lead_time_display,
        delta_color=delta_color,
    )

    col3.metric(
        label="Expected Failures Preventable (Annual)",
        value=f"{CONFIG['DEFAULT_FAILURES_PREVENTED_PER_YEAR']} ",
    )

    status_text = (
        "CRITICAL ALERT"
        if is_critical_alert
        else "WARNING (Degradation)" if latest_confidence >= 50 else "NORMAL"
    )
    col4.metric(
        label="Asset Health Status",
        value=status_text,
        delta="Action Required: Initiate Inspection" if is_critical_alert else None,
        delta_color="inverse",
    )

    st.markdown("---")

    # --- ROI Calculator (Req 4) ---
    st.header("Built-in ROI Calculator")

    cost_col, failures_col, hours_col, savings_col = st.columns(4)

    with cost_col:
        downtime_cost = st.number_input(
            "Cost of Unplanned Downtime (R/Hour)",
            min_value=1000,
            value=CONFIG["DEFAULT_DOWNTIME_COST_PER_HOUR"],
            step=5000,
            format="%d",
            key="cost_input_pm",
        )

    with failures_col:
        failures_prevented = st.number_input(
            "Estimated Failures Prevented Per Year",
            min_value=1,
            value=CONFIG["DEFAULT_FAILURES_PREVENTED_PER_YEAR"],
            step=1,
            format="%d",
            key="failures_input_pm",
        )

    with hours_col:
        hours_saved = st.number_input(
            "Average Hours Saved per Event",
            min_value=1,
            value=CONFIG["ASSUMED_HOURS_SAVED_PER_FAILURE"],
            step=1,
            format="%d",
            key="hours_input_pm",
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
        f"**ROI Justification:** The potential annual savings is **{roi_multiplier:.1f}x** the Proof of Concept cost."
    )


def render_technician_view(df_slice, latest_confidence, anomaly_flag_date):
    """Renders the detailed sensor data and technical analysis focused dashboard."""
    st.header(f"Real-Time Diagnostics: {CONFIG['ASSET_NAME']} (Technician View)")

    # --- AI Confidence and Current Status ---
    col_status, col_conf = st.columns([2, 1])

    with col_status:
        st.markdown(
            f"**Current Operation State:** {'Anomaly Detected' if latest_confidence >= 50 else 'Normal Operation'} "
            f"| **Last Data Point Received:** {df_slice.index[-1].strftime('%Y-%m-%d %H:%M')}"
        )

    with col_conf:
        st.metric("AI Confidence", f"{latest_confidence:.1f}%")

    st.markdown("---")

    # --- Main Time-Series Charts (Sensor Data Streams) ---
    st.subheader("Real-time Sensor Data Monitoring")  # Reverted to original subheader

    # Reverted to original data columns (Power_kW, Vibration, Temperature)
    df_to_plot = df_slice[["Power_kW", "Vibration", "Temperature"]].copy()

    fig = px.line(
        df_to_plot,
        # Reverted to original data streams
        y=["Power_kW", "Vibration", "Temperature"],
        labels={"value": "Sensor Reading", "Timestamp": "Time"},
        title=f"Live Sensor Data Stream | Confidence: {latest_confidence:.1f}%",  # Kept dynamic confidence in title
        height=600,
        color_discrete_map={
            "Power_kW": CONFIG["PRIMARY_COLOR"],  # Default Blue
            "Vibration": "#EF4444",  # Red
            "Temperature": "#FBBF24",  # Amber
        },
    )

    # Anomaly Highlight for critical confidence
    if anomaly_flag_date is not None and latest_confidence >= 90:
        fig.add_vrect(
            x0=anomaly_flag_date,
            x1=anomaly_flag_date + timedelta(hours=6),
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
            text="AI CRITICAL ANOMALY FLAG (90%+ CONFIDENCE)",
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
    with st.expander("Show Latest Sensor Readings and ML Scores"):
        st.markdown(
            "The table below shows the latest readings and the confidence calculated by the Machine Learning model."
        )
        # Show only the last 20 rows to keep it manageable in a "live" view
        st.dataframe(
            df_slice.iloc[-20:].sort_index(ascending=False)[
                [
                    "Power_kW",  # Added Power back
                    "Vibration",
                    "Temperature",
                    "ML_Anomaly_Score",
                    "AI_Confidence_Score",
                ]
            ],
            use_container_width=True,
            height=300,
        )


# ==============================================================================
# 6. MAIN APP EXECUTION FLOW (Uses st.rerun for non-blocking stream)
# ==============================================================================


def main():
    if full_data_df.empty or iso_forest_model is None:
        return

    # Calculate total days in data for slider max
    total_days = (full_data_df.index[-1] - full_data_df.index[0]).days

    # Ensure slider is capped at max days
    if st.session_state.days_progressed > total_days:
        st.session_state.days_progressed = total_days

    # --- Sidebar Controls ---
    with st.sidebar:
        st.title(CONFIG["LOGO_TEXT"])
        st.markdown(f"**Asset:** **{CONFIG['ASSET_NAME']}**")
        st.markdown(f"**Failure Mode:** {CONFIG['FAILURE_TYPE']}")

        st.markdown("---")

        # Role Selector (Required feature)
        user_role = st.selectbox(
            "Select User Role:", ("Plant Manager", "Technician"), key="user_role"
        )

        st.markdown("---")
        st.subheader("Data Stream Control")

        # --- REAL-TIME DATA STREAM SIMULATION CONTROL ---

        # Slider linked to session state
        st.session_state.days_progressed = st.slider(
            "Simulated Data Progress (Days)",
            min_value=1,
            max_value=total_days,
            value=st.session_state.days_progressed,
            step=1,
            key="_slider_control_",
            help="Manually advance or rewind the data stream.",
        )

        # Define button handlers to control the loop
        def start_stream():
            st.session_state.playing = True

        def pause_stream():
            st.session_state.playing = False

        # Play/Pause Controls
        col_play, col_pause = st.columns(2)

        with col_play:
            st.button(
                "▶️ Start Live Stream",
                on_click=start_stream,
                use_container_width=True,
                disabled=st.session_state.playing,
            )

        with col_pause:
            st.button(
                "⏸️ Pause Stream",
                on_click=pause_stream,
                use_container_width=True,
                disabled=not st.session_state.playing,
            )

        # Container for displaying the current time
        time_placeholder = st.empty()

    # --- 1. CORE DATA PROCESSING ---
    end_date = full_data_df.index[0] + timedelta(days=st.session_state.days_progressed)
    df_slice = full_data_df.loc[full_data_df.index <= end_date].copy()

    # Process the data slice to get ML metrics (AI Confidence Score Calculation)
    df_slice = process_data_slice(df_slice, iso_forest_model, ml_features)
    latest_confidence, anomaly_flag_date, predicted_failure_date, days_left = (
        get_prediction_metrics(df_slice)
    )

    # --- 2. RENDER DASHBOARD BASED ON ROLE ---
    if user_role == "Plant Manager":
        render_plant_manager_view(
            df_slice, latest_confidence, predicted_failure_date, days_left
        )
    elif user_role == "Technician":
        render_technician_view(df_slice, latest_confidence, anomaly_flag_date)

    # Update the time in the sidebar
    with time_placeholder.container():
        st.info(
            f"**Current System Time:** {df_slice.index[-1].strftime('%Y-%m-%d %H:%M')} "
        )

    # --- 3. AUTO-ADVANCE LOGIC (Correct Live Motion Implementation) ---

    if st.session_state.playing:

        # Check for end condition first
        if st.session_state.days_progressed >= total_days:
            st.session_state.playing = False
            st.sidebar.error("⚠️ **SIMULATION ENDED:** Failure event has occurred.")
            return

        # 1. Increment Data Stream for the NEXT run
        st.session_state.days_progressed += 1

        # 2. Wait for the specified interval (non-blocking)
        time.sleep(CONFIG["UPDATE_INTERVAL_SECONDS"])

        # 3. Tell Streamlit to re-execute the script from the top
        st.rerun()


if __name__ == "__main__":
    main()
