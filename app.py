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
# 0. CONFIGURATION AND STYLING
# ==============================================================================
CONFIG = {
    # Application & Branding
    "ASSET_NAME": "Kroonstad Maize Mill Pump #4",
    "FAILURE_TYPE": "Bearing Seizure (High Temp/Vibe)",
    "PRIMARY_COLOR": "#0052CC",  # Blue
    "SECONDARY_COLOR": "#FFD700",  # Gold/Yellow
    "FONT_FAMILY": "Roboto, sans-serif",
    # AI Prediction Logic Tuning
    "SCORE_THRESHOLD_90_PERCENT": -0.15,  # Raw score that corresponds to 90% confidence
    "MAX_NORMAL_SCORE": 0.05,  # Max score for 0% confidence
    "PREDICTED_LEAD_TIME_DAYS": 25,  # Fixed lead time (RUL) when AI confidence is critical
    # ROI Calculator Defaults
    "DEFAULT_DOWNTIME_COST_PER_HOUR": 50000,  # R50,000
    "DEFAULT_FAILURES_PREVENTED_PER_YEAR": 4,
    "ASSUMED_HOURS_SAVED_PER_FAILURE": 12,
    "POC_COST": 50000,  # R50,000 PoC price for ROI calculation
}

st.set_page_config(
    layout="wide",
    page_title=f"S.I.L.K.E. AI Predictive Maintenance Demo",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Branding
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .stApp {{
            font-family: {CONFIG["FONT_FAMILY"]};
            color: #1F2937;
        }}
        h1, h2, h3 {{
            color: {CONFIG["PRIMARY_COLOR"]};
        }}
        [data-testid="stMetricValue"] {{
            font-size: 2.0rem;
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
            font-size: 2.0rem;
            font-weight: 700;
            color: #10B981;
            margin-top: 10px;
        }}
    </style>
""",
    unsafe_allow_html=True,
)


# --- Configuration ---
DATA_POINT_INTERVAL = 1.0
file_path = "maize_mill_simulated_sensor_data.csv"


# ==============================================================================
# 1. LOAD DATA, INIT MODELS, AND HELPER FUNCTIONS
# ==============================================================================


@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the asset sensor data."""
    try:
        if not os.path.exists(file_path):
            st.error(f"Error: File not found at {file_path}. Please check the path.")
            st.stop()

        df = pd.read_csv(file_path, parse_dates=["Timestamp"])
        required_columns = [
            "Timestamp",
            "Power_kW",
            "Amperage",
            "Vibration",
            "Temperature",
        ]
        if not all(col in df.columns for col in required_columns):
            st.error("Error: The CSV file is missing one of the required columns.")
            st.stop()

        df = df.set_index("Timestamp").sort_index()

        # Add rolling stats for rule-based analysis (as per original code)
        window_size = "12h"
        df["Power_kW_RollingMean"] = df["Power_kW"].rolling(window=window_size).mean()
        df["Power_kW_RollingStd"] = df["Power_kW"].rolling(window=window_size).std()
        df["Amperage_RollingMean"] = df["Amperage"].rolling(window=window_size).mean()
        df["Amperage_RollingStd"] = df["Amperage"].rolling(window=window_size).std()

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


@st.cache_resource
def train_isolation_forest(df_initial):
    """Trains the Isolation Forest model on initial, healthy data."""
    if df_initial.empty:
        return None, None

    features_for_ml = ["Power_kW", "Amperage", "Vibration", "Temperature"]
    # Training on the first 1000 rows, assuming initial normal operation
    X_train = df_initial.head(1000)[features_for_ml].dropna()

    if X_train.empty or len(X_train) < 100:
        st.warning("Not enough data to train the ML model.")
        return None, None

    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train.values)
    return model, features_for_ml


# Global data and model initialization
full_data_df = load_data(file_path)

try:
    iso_forest_model, ml_features = train_isolation_forest(full_data_df)
except Exception as e:
    st.error(f"An error occurred during model training: {e}")
    st.stop()

if iso_forest_model is None:
    st.warning("ML functionality will be disabled due to model training failure.")


def calculate_ml_confidence(raw_score):
    """Scales the raw Isolation Forest score to a 0-100 AI Confidence Score."""
    score_range = CONFIG["MAX_NORMAL_SCORE"] - CONFIG["SCORE_THRESHOLD_90_PERCENT"]

    if raw_score >= CONFIG["MAX_NORMAL_SCORE"]:
        confidence = 0.0
    else:
        distance_from_normal = CONFIG["MAX_NORMAL_SCORE"] - raw_score
        factor = distance_from_normal / score_range
        confidence = min(100.0, factor * 90.0)

    return round(confidence, 1)


def get_prediction_metrics(df_slice):
    """Calculates the key AI prediction dates, confidence, and RUL."""
    if df_slice.empty:
        return 0.0, None, datetime.now(), 90

    latest_score = df_slice["ML_Anomaly_Score"].iloc[-1]
    latest_confidence = calculate_ml_confidence(latest_score)
    current_time = df_slice.index[-1]

    predicted_failure_date = current_time + timedelta(days=90)
    days_left = 90

    # 90%+ confidence is the critical prediction moment (actionable RUL)
    if latest_confidence >= 90.0:
        predicted_failure_date = current_time + timedelta(
            days=CONFIG["PREDICTED_LEAD_TIME_DAYS"]
        )
        days_left = CONFIG["PREDICTED_LEAD_TIME_DAYS"]

    return latest_confidence, predicted_failure_date, days_left


def calculate_roi(downtime_cost, failures_prevented, hours_saved):
    """Calculates potential annual savings and ROI multiplier."""
    annual_savings = (downtime_cost * hours_saved) * failures_prevented
    # Handle division by zero if POC cost is set to 0
    roi_multiplier = annual_savings / CONFIG["POC_COST"] if CONFIG["POC_COST"] else 0
    return annual_savings, roi_multiplier


# ==============================================================================
# 2. ANOMALY DETECTION AND DATA PROCESSING
# (Keeping rule and ML checks consistent with original code)
# ==============================================================================


def check_rule_based_anomalies(row):
    """Applies rule-based anomaly detection to a single data row."""
    # ... (content remains the same as in the original code)
    rule_power_threshold = 600
    rule_temp_threshold = 70.0
    rule_vibration_threshold = 6.5
    rule_temp_gradual_threshold = 75.0

    anomaly_rule_1 = (row["Power_kW"] > rule_power_threshold) and (
        row["Temperature"] > rule_temp_threshold
    )

    try:
        if not pd.isna(row["Power_kW_RollingMean"]):
            rolling_mean = row["Power_kW_RollingMean"]
            rolling_std = row["Power_kW_RollingStd"]
            rolling_mean_amp = row["Amperage_RollingMean"]
            rolling_std_amp = row["Amperage_RollingStd"]

            anomaly_rule_2 = (row["Power_kW"] > (rolling_mean + 3 * rolling_std)) or (
                row["Amperage"] > (rolling_mean_amp + 3 * rolling_std_amp)
            )
        else:
            anomaly_rule_2 = False
    except KeyError:
        anomaly_rule_2 = False

    anomaly_rule_3 = (row["Vibration"] > rule_vibration_threshold) and (
        row["Temperature"] > rule_temp_gradual_threshold
    )

    is_rule_anomaly = anomaly_rule_1 or anomaly_rule_2 or anomaly_rule_3
    reasoning = ""
    if anomaly_rule_1:
        reasoning += "High Power & Temp: Potential inefficiency/overload. "
    if anomaly_rule_2:
        reasoning += "Sudden Power/Amp Spike. "
    if anomaly_rule_3:
        reasoning += "Gradual Vibe/Temp Increase: Possible mechanical wear. "

    return is_rule_anomaly, reasoning.strip()


def check_ml_anomaly(row, model, features):
    """Applies the Isolation Forest model to a single data row."""
    if model is None or features is None or any(pd.isna(row[features])):
        return False, 0.0

    data_point = np.array([row[features].values])
    ml_prediction = model.predict(data_point)[0]
    ml_score = model.decision_function(data_point)[0]
    is_ml_anomaly = ml_prediction == -1

    return is_ml_anomaly, ml_score


# ==============================================================================
# 3. ROLE-BASED VIEW RENDERING
# ==============================================================================


def render_plant_manager_view(latest_confidence, predicted_failure_date, days_left):
    """Renders the financial and executive summary focused dashboard."""
    st.header(f"Live Asset Monitor: {CONFIG['ASSET_NAME']} (Plant Manager View)")

    is_critical_alert = latest_confidence >= 90

    # --- Financial and Prediction KPIs ---
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

    with col1:
        st.markdown('<div class="confidence-card">', unsafe_allow_html=True)
        st.markdown(f"**AI Failure Confidence** (0-100%)")
        st.markdown(f"## {latest_confidence:.1f}%")
        st.progress(latest_confidence / 100)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        date_value_display = (
            predicted_failure_date.strftime("%Y-%m-%d") if is_critical_alert else "---"
        )
        lead_time_display = (
            f"{days_left} Days Lead Time" if is_critical_alert else "System Normal"
        )
        delta_color = "inverse" if is_critical_alert else "normal"

        col2.metric(
            label="Predicted Failure Date (Actionable RUL)",
            value=date_value_display,
            delta=lead_time_display,
            delta_color=delta_color,
        )

    status_text = (
        "CRITICAL ALERT"
        if is_critical_alert
        else "WARNING (Degradation)" if latest_confidence >= 50 else "NORMAL"
    )
    col3.metric(
        label="Asset Health Status",
        value=status_text,
        delta="Action Required: Initiate Inspection" if is_critical_alert else None,
        delta_color="inverse",
    )

    col4.metric(
        label="Expected Failures Preventable (Annual)",
        value=f"{CONFIG['DEFAULT_FAILURES_PREVENTED_PER_YEAR']} ",
    )

    st.markdown("---")

    # --- Built-in ROI Calculator ---
    st.subheader("Built-in ROI Calculator")

    cost_col, failures_col, hours_col, savings_col = st.columns(4)

    with cost_col:
        downtime_cost = st.number_input(
            "Cost of Unplanned Downtime (R/Hour)",
            min_value=1000,
            value=CONFIG["DEFAULT_DOWNTIME_COST_PER_HOUR"],
            step=5000,
            format="%d",
            key="cost_input",
        )

    with failures_col:
        failures_prevented = st.number_input(
            "Estimated Failures Prevented Per Year",
            min_value=1,
            value=CONFIG["DEFAULT_FAILURES_PREVENTED_PER_YEAR"],
            step=1,
            format="%d",
            key="failures_input",
        )

    with hours_col:
        hours_saved = st.number_input(
            "Average Hours Saved per Event",
            min_value=1,
            value=CONFIG["ASSUMED_HOURS_SAVED_PER_FAILURE"],
            step=1,
            format="%d",
            key="hours_input",
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

    st.success(
        f"**ROI Justification:** The potential annual savings is **{roi_multiplier:.1f}x** the Proof of Concept cost (R {CONFIG['POC_COST']:,.0f})."
    )


def render_technician_view(
    kpi_ph, alert_ph, chart_ph, current_df, anomaly_count, latest_confidence
):
    """Renders the detailed sensor data and technical analysis dashboard."""

    # --- KPI Section (Now includes AI Confidence) ---
    with kpi_ph.container():
        st.header(f"Real-Time Diagnostics: {CONFIG['ASSET_NAME']} (Technician View)")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

        col_kpi1.metric("Latest Power (kW)", f"{current_df.iloc[-1]['Power_kW']:.2f}")
        col_kpi2.metric("Latest Vibration", f"{current_df.iloc[-1]['Vibration']:.2f}")
        col_kpi3.metric("Latest Temp (°C)", f"{current_df.iloc[-1]['Temperature']:.2f}")
        col_kpi4.metric(
            "AI Confidence",
            f"{latest_confidence:.1f}%",
            help="Confidence of anomaly detection.",
        )
        col_kpi5.metric("Total Anomalies", anomaly_count)

    st.markdown("---")

    # --- Alerts and Anomaly Reporting (Kept original logic) ---
    with alert_ph.container():
        if anomaly_count > 0:
            last_anomaly = current_df[
                current_df["Is_Rule_Anomaly"] | current_df["Is_ML_Anomaly"]
            ].iloc[-1]
            st.error(
                f"⚠️ Anomaly Detected at {last_anomaly.name.strftime('%Y-%m-%d %H:%M:%S')}!"
            )
            st.markdown(
                f"**Reasoning:** {last_anomaly['Anomaly_Reasoning'] if last_anomaly['Anomaly_Reasoning'] else 'ML Detected: Uncategorized Anomaly.'}"
            )

            with st.expander("Show Detailed Anomaly Report"):
                anomalies_to_show = current_df[
                    current_df["Is_Rule_Anomaly"] | current_df["Is_ML_Anomaly"]
                ]
                st.dataframe(
                    anomalies_to_show[
                        [
                            "Power_kW",
                            "Vibration",
                            "Temperature",
                            "ML_Anomaly_Score",
                            "Anomaly_Reasoning",
                            "Is_ML_Anomaly",
                        ]
                    ].sort_index(ascending=False),
                    use_container_width=True,
                    height=300,
                )

                # Original Anomaly Chart
                if not anomalies_to_show.empty:
                    last_anomaly_time = last_anomaly.name
                    df_anomaly_window = current_df.loc[
                        (current_df.index >= last_anomaly_time - pd.Timedelta(hours=2))
                        & (
                            current_df.index
                            <= last_anomaly_time + pd.Timedelta(hours=2)
                        )
                    ]

                    fig_anomaly = px.line(
                        df_anomaly_window,
                        y=["Power_kW", "Vibration", "Temperature"],
                        title=f"Sensor Readings Around Anomaly at {last_anomaly_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    )
                    st.plotly_chart(fig_anomaly, use_container_width=True)

        else:
            st.success("✅ System Operating Normally.")

    # --- Main Chart (Kept original logic) ---
    with chart_ph.container():
        st.subheader("Real-time Sensor Data Monitoring")
        fig_main = px.line(
            current_df,
            x=current_df.index,
            y=["Power_kW", "Vibration", "Temperature"],
            labels={"value": "Value", "Timestamp": "Time"},
            title="Live Sensor Data Stream",
        )
        fig_main.update_layout(height=500, xaxis_title="Timestamp")
        st.plotly_chart(
            fig_main,
            use_container_width=True,
            key=f"main_chart_{st.session_state.current_row_index}",
        )


# ==============================================================================
# 4. MAIN EXECUTION AND STREAMING LOOP
# ==============================================================================

st.title("S.I.L.K.E Predictive Maintenance Demo")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Asset Details")
    st.markdown(f"**Asset:** **{CONFIG['ASSET_NAME']}**")
    st.markdown(f"**Failure Mode:** {CONFIG['FAILURE_TYPE']}")

    st.markdown("---")

    # NEW: Role Selector Switch
    user_role = st.selectbox(
        "Select User Role:", ("Plant Manager", "Technician"), key="user_role"
    )

    st.markdown("---")
    st.header("The Value Proposition")
    st.markdown(
        """
    Our solution helps you shift from reactive to proactive maintenance by providing:
    - **Predictive Maintenance & Uptime:** AI predicts anomalies, reducing unexpected downtime.
    - **Actionable Intelligence:** Turns complex data into informed financial and operational decisions.
    """
    )


# --- Session State Initialization ---
if "current_df" not in st.session_state:
    initial_row = full_data_df.head(1).copy()
    initial_row["Is_Rule_Anomaly"] = False
    initial_row["Is_ML_Anomaly"] = False
    initial_row["Anomaly_Reasoning"] = ""
    initial_row["ML_Anomaly_Score"] = 0.0
    st.session_state.current_df = initial_row
    st.session_state.current_row_index = 1
    st.session_state.anomaly_count = 0

# Create empty placeholders for all dynamic UI elements
# These placeholders are needed for the while loop to correctly update the UI
kpi_placeholder = st.empty()
alert_placeholder = st.empty()
chart_placeholder = st.empty()


# The continuous simulation loop
while st.session_state.current_row_index < len(full_data_df):
    try:
        # 1. Get next data row
        next_row = full_data_df.iloc[
            st.session_state.current_row_index : st.session_state.current_row_index + 1
        ].copy()

        # 2. Perform anomaly checks
        is_rule_anomaly, rule_reasoning = check_rule_based_anomalies(next_row.iloc[0])
        is_ml_anomaly, ml_score = check_ml_anomaly(
            next_row.iloc[0], iso_forest_model, ml_features
        )

        # 3. Augment data row with results
        next_row.loc[:, "Is_Rule_Anomaly"] = is_rule_anomaly
        next_row.loc[:, "Is_ML_Anomaly"] = is_ml_anomaly
        next_row.loc[:, "Anomaly_Reasoning"] = rule_reasoning
        next_row.loc[:, "ML_Anomaly_Score"] = ml_score

        # 4. Update session state
        st.session_state.current_df = pd.concat([st.session_state.current_df, next_row])
        st.session_state.current_row_index += 1

        if is_rule_anomaly or is_ml_anomaly:
            st.session_state.anomaly_count += 1

        # 5. Calculate global prediction metrics based on the latest data point
        latest_confidence, predicted_failure_date, days_left = get_prediction_metrics(
            st.session_state.current_df
        )

        # 6. Call the appropriate rendering function based on the user's role
        # We clear all placeholders at the start to ensure clean rendering for the selected view
        kpi_placeholder.empty()
        alert_placeholder.empty()
        chart_placeholder.empty()

        if user_role == "Plant Manager":
            render_plant_manager_view(
                latest_confidence, predicted_failure_date, days_left
            )
        elif user_role == "Technician":
            render_technician_view(
                kpi_placeholder,
                alert_placeholder,
                chart_placeholder,
                st.session_state.current_df,
                st.session_state.anomaly_count,
                latest_confidence,
            )

        # 7. Pause for a moment to simulate real-time stream
        time.sleep(DATA_POINT_INTERVAL)

    except Exception as e:
        st.error(
            f"⚠️ Error: The simulation crashed while processing row {st.session_state.current_row_index}."
        )
        st.error(f"**Details:** {e}")
        st.stop()

# Final status message after the loop
if st.session_state.current_row_index >= len(full_data_df):
    st.info("End of simulation. All data has been processed.")
