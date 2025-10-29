import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
import time
import os

# --- NEW: Failure Mode Definitions and Constants ---
FAILURE_MODES_MAPPING = {
    "A": "Inner Race Bearing Failure (NDE)",  # High Vibe & High Temp (Friction/Wear)
    "B": "Misalignment or Imbalance",  # High Vibe & Normal Temp (Mechanical Stress)
    "C": "Electrical Winding Fault",  # High Amperage/Power Spike (Electrical)
    "D": "Oil Contamination/Loss",  # Gradual Temp Rise (Lubrication/Cooling)
    "E": "Normal Operation",
}

# --- NEW: Branding and Currency/Localization Constants ---
LOGO_TEXT = "S.I.L.K.E. AI"
INDUSTRY_TITLE = "Maize Mill Grinding Line Motor"  # Monitoring the main motor/gearbox
CURRENCY_SYMBOL = "R"  # South African Rand
CURRENCY_FORMAT = "R {:,.0f}"  # Format for ZAR without decimal cents

# --- Configuration ---
st.set_page_config(layout="wide", page_title=f"{LOGO_TEXT} PdM: {INDUSTRY_TITLE}")

DATA_POINT_INTERVAL = 0.5
file_path = "maize_mill_simulated_sensor_data.csv"

# --- Placeholder for industry-specific failure/cost data (for ROI calculation) ---
FAILURE_COST_DATA = {
    # Conversion approximated for demo: $15,000/hr -> R250,000/hr
    "Average Downtime Cost per Hour (R)": 250000,
    "Time Saved by PdM (Hours)": 4,
    "Avg. Repair Cost (Reactive R)": 800000,  # Catastrophic failure cost
    "Avg. Repair Cost (Predictive R)": 150000,  # Planned repair cost
    "Equipment Lifespan (Years)": 10,
    "PdM System Annual Cost (R)": 400000,  # Annual investment
}


# --- 1. Load Data and Initialize Models ---
def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"Error: File not found at {file_path}. Please check the path.")
            st.stop()

        df = pd.read_csv(file_path, parse_dates=["Timestamp"])

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "Timestamp",
                    "Power_kW",
                    "Amperage",
                    "Vibration",
                    "Temperature",
                ]
            )

        required_columns = [
            "Timestamp",
            "Power_kW",
            "Amperage",
            "Vibration",
            "Temperature",
        ]
        if not all(col in df.columns for col in required_columns):
            st.error("Error: The CSV file is missing one of the required columns.")
            st.error(f"Required columns: {required_columns}")
            st.error(f"Found columns: {list(df.columns)}")
            st.stop()

        df = df.set_index("Timestamp").sort_index()

        window_size = "12h"
        df["Power_kW_RollingMean"] = df["Power_kW"].rolling(window=window_size).mean()
        df["Power_kW_RollingStd"] = df["Power_kW"].rolling(window=window_size).std()
        df["Amperage_RollingMean"] = df["Amperage"].rolling(window=window_size).mean()
        df["Amperage_RollingStd"] = df["Amperage"].rolling(window=window_size).std()

        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please check the path.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty.")
        st.stop()


def train_isolation_forest(df_initial):
    if df_initial.empty:
        return None, None

    features_for_ml = ["Power_kW", "Amperage", "Vibration", "Temperature"]
    X_train = df_initial[features_for_ml].dropna()
    if X_train.empty or len(X_train) < 2:
        return None, None
    model = IsolationForest(contamination=0.015, random_state=42)
    model.fit(X_train.values)
    return model, features_for_ml


# Global data and model initialization
full_data_df = load_data(file_path)

# --- Define Global Variables for Slider/Index ---
MAX_DATA_INDEX = len(full_data_df)

try:
    # We still only train the model on the first 1000 rows as intended.
    iso_forest_model, ml_features = train_isolation_forest(full_data_df.head(1000))
except Exception as e:
    st.error(f"An error occurred during model training: {e}")
    st.warning(
        "Model training failed. This may be due to insufficient data or inconsistencies."
    )
    st.stop()


# --- 2. Anomaly Detection Logic (as functions) ---
def check_rule_based_anomalies(row):
    """Applies rule-based anomaly detection to a single data row."""
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


def determine_failure_mode(row, rule_anomaly, ml_anomaly):
    """
    Simulates a classifier to determine the most probable failure mode
    based on sensor patterns when an anomaly is detected.
    """
    vibe = row["Vibration"]
    temp = row["Temperature"]
    amp = row["Amperage"]

    # Thresholds (Simulated)
    HIGH_VIBE = 6.0
    HIGH_TEMP = 70.0
    SPIKE_AMP = 65.0

    if not rule_anomaly and not ml_anomaly:
        return FAILURE_MODES_MAPPING["E"]  # Normal Operation

    # 1. High Vibe & High Temp (Likely Bearing/Friction Failure)
    if vibe >= HIGH_VIBE and temp >= HIGH_TEMP:
        return FAILURE_MODES_MAPPING["A"]

    # 2. High Vibe & Normal Temp (Likely Imbalance/Misalignment)
    if vibe >= HIGH_VIBE and temp < HIGH_TEMP:
        return FAILURE_MODES_MAPPING["B"]

    # 3. Electrical Anomaly (High Amp/Power Spike)
    if amp >= SPIKE_AMP:
        return FAILURE_MODES_MAPPING["C"]

    # 4. Catch-all for gradual or uncategorized (e.g., fluid/lubrication issue)
    if temp >= 65.0:
        return FAILURE_MODES_MAPPING["D"]

    return FAILURE_MODES_MAPPING["E"]  # Default fallback if thresholds aren't met yet


def check_ml_anomaly(row, model, features):
    """Applies the Isolation Forest model to a single data row and calculates HI/RUL."""
    if model is None or features is None or any(pd.isna(row[features])):
        return False, 0.0, 0.0, 0, 100, FAILURE_MODES_MAPPING["E"]

    data_point = np.array([row[features].values])
    ml_prediction = model.predict(data_point)[0]
    ml_score = model.decision_function(data_point)[0]
    is_ml_anomaly = ml_prediction == -1

    # Calculate a proxy for AI Confidence and RUL
    if is_ml_anomaly:
        confidence = min(100, max(50, 50 - (ml_score * 500)))
    else:
        confidence = 100 - (abs(ml_score) * 200)
        confidence = max(0, min(100, confidence))

    base_rul_days = 30
    rul_days = max(1, int(base_rul_days + (ml_score * 100)))

    # --- Health Index Calculation ---
    # Health Index is 100% when RUL > 30 days, drops linearly to 1% at RUL = 1 day
    health_index = min(100, max(1, int((rul_days / 30) * 100)))

    # --- Failure Mode Prediction (Placeholder for more complex logic) ---
    is_rule_anomaly, _ = check_rule_based_anomalies(row)
    predicted_failure_mode = determine_failure_mode(row, is_rule_anomaly, is_ml_anomaly)

    return (
        is_ml_anomaly,
        ml_score,
        confidence,
        rul_days,
        health_index,
        predicted_failure_mode,
    )


# --- ROI Calculation Function ---
def calculate_pdm_roi(anomaly_count, cost_data):
    """Calculates ROI metrics based on simulated savings from avoided failures."""

    # One critical failure avoided for every 5 minor anomalies detected
    critical_failures_avoided = anomaly_count // 5

    # Calculate Potential Downtime Cost (Cost-at-Risk)
    potential_downtime_cost = (
        cost_data["Time Saved by PdM (Hours)"]
        * cost_data["Average Downtime Cost per Hour (R)"]
    )

    if critical_failures_avoided == 0:
        return {
            "Total Savings": 0,
            "Net ROI": 0,
            "ROI Status": "Awaiting critical detection...",
            "Justification": "No major failure predicted/avoided yet.",
            "Potential Downtime Cost": potential_downtime_cost,
        }

    # 1. Savings from avoided downtime
    downtime_savings_per_failure = potential_downtime_cost
    total_downtime_savings = downtime_savings_per_failure * critical_failures_avoided

    # 2. Savings from reduced repair costs
    repair_cost_savings_per_failure = (
        cost_data["Avg. Repair Cost (Reactive R)"]
        - cost_data["Avg. Repair Cost (Predictive R)"]
    )
    total_repair_savings = repair_cost_savings_per_failure * critical_failures_avoided

    total_savings = total_downtime_savings + total_repair_savings

    # Annual Investment Cost (PdM System)
    investment_cost = cost_data["PdM System Annual Cost (R)"]

    net_profit = total_savings - investment_cost

    if investment_cost > 0:
        net_roi = (net_profit / investment_cost) * 100
    else:
        net_roi = 0

    justification = (
        f"Avoided **{critical_failures_avoided}** catastrophic failure(s), saving:\n"
        f"- **{CURRENCY_FORMAT.format(total_downtime_savings)}** in potential downtime.\n"
        f"- **{CURRENCY_FORMAT.format(total_repair_savings)}** in reduced repair costs."
    )

    return {
        "Total Savings": total_savings,
        "Net ROI": net_roi,
        "ROI Status": f"ROI: {net_roi:,.1f}%",
        "Justification": justification,
        "Potential Downtime Cost": potential_downtime_cost,
    }


# --- NEW ADDON FUNCTION: Maintenance Scheduling ---
def get_maintenance_schedule(rul_days, current_timestamp):
    """Calculates the proposed maintenance window based on RUL."""

    # If RUL is healthy (>= 30 days)
    if rul_days >= 30:
        return "No immediate maintenance required. Schedule next routine inspection."

    # Calculate the end date of the predicted safe operation period (RUL)
    safe_end_date = current_timestamp + pd.Timedelta(days=rul_days)

    # Suggest a window that is 5 days prior to the RUL end date
    start_window = safe_end_date - pd.Timedelta(days=5)

    # Format dates nicely
    start_str = start_window.strftime("%Y-%m-%d")
    end_str = safe_end_date.strftime("%Y-%m-%d")

    return f"**PROPOSED WINDOW:** {start_str} to {end_str} (Must be completed before RUL expires on {end_str})."


# --- 3. Simulation State Management Helper Function ---
def initialize_simulation_state(start_index):
    """
    Initializes the simulation state (current_df, RUL, HI, Anomaly Count)
    at a specific starting index, simulating a fast-forward jump.
    """
    # Ensure start_index is valid
    start_index = min(max(0, start_index), MAX_DATA_INDEX)

    # Reset core state variables
    st.session_state.current_row_index = start_index
    st.session_state.anomaly_count = 0
    st.session_state.rul_days = 30
    st.session_state.health_index = 100

    # Slice data up to the start_index (exclusive)
    jump_data = full_data_df.iloc[:start_index].copy()

    if jump_data.empty:
        # If jumping to index 0, use the first row and set defaults
        initial_row = full_data_df.head(1).copy()
        initial_row["Is_Rule_Anomaly"] = False
        initial_row["Is_ML_Anomaly"] = False
        initial_row["Anomaly_Reasoning"] = ""
        initial_row["ML_Anomaly_Score"] = 0.0
        initial_row["AI_Confidence"] = 0.0
        initial_row["RUL_Days"] = 30
        initial_row["Health_Index"] = 100
        initial_row["Predicted_Failure_Mode"] = FAILURE_MODES_MAPPING["E"]
        st.session_state.current_df = initial_row
        st.session_state.current_row_index = 1  # Start loop at index 1
        return

    # Process the data slice to calculate the state at the jump point
    processed_data = jump_data.copy()

    # Pre-add the necessary analysis columns
    processed_data.loc[:, "Is_Rule_Anomaly"] = False
    processed_data.loc[:, "Is_ML_Anomaly"] = False
    processed_data.loc[:, "Anomaly_Reasoning"] = ""
    processed_data.loc[:, "ML_Anomaly_Score"] = 0.0
    processed_data.loc[:, "AI_Confidence"] = 0.0
    processed_data.loc[:, "RUL_Days"] = 30
    processed_data.loc[:, "Health_Index"] = 100
    processed_data.loc[:, "Predicted_Failure_Mode"] = FAILURE_MODES_MAPPING["E"]

    # To calculate RUL/HI accurately at the jump point, we re-run the analysis
    # on the last 500 rows leading up to it (or the whole slice if smaller).
    start_idx_for_analysis = max(0, start_index - 500)

    # Note: Using .iloc[i] here for safe access on the sliced DF
    for i in range(start_idx_for_analysis, start_index):
        row = jump_data.iloc[i]

        # Recalculate anomaly and state
        is_rule_anomaly, rule_reasoning = check_rule_based_anomalies(row)
        (
            is_ml_anomaly,
            ml_score,
            ai_confidence,
            rul_days_current,
            health_index_current,
            predicted_failure_mode,
        ) = check_ml_anomaly(row, iso_forest_model, ml_features)

        # Update the processed_data DF
        processed_data.loc[row.name, "Is_Rule_Anomaly"] = is_rule_anomaly
        processed_data.loc[row.name, "Is_ML_Anomaly"] = is_ml_anomaly
        processed_data.loc[row.name, "Anomaly_Reasoning"] = rule_reasoning
        processed_data.loc[row.name, "ML_Anomaly_Score"] = ml_score
        processed_data.loc[row.name, "AI_Confidence"] = ai_confidence
        processed_data.loc[row.name, "RUL_Days"] = rul_days_current
        processed_data.loc[row.name, "Health_Index"] = health_index_current
        processed_data.loc[row.name, "Predicted_Failure_Mode"] = predicted_failure_mode

        # Update Global State (this cumulative effect defines the state at the jump)
        if is_rule_anomaly or is_ml_anomaly:
            st.session_state.anomaly_count += 1
            st.session_state.rul_days = min(st.session_state.rul_days, rul_days_current)
            st.session_state.health_index = min(
                st.session_state.health_index, health_index_current
            )
        elif st.session_state.rul_days < 30 and ml_score > 0:
            st.session_state.rul_days = min(30, st.session_state.rul_days + 1)
            st.session_state.health_index = min(100, st.session_state.health_index + 3)

    # To keep the displayed history manageable, we only keep the last 500 rows of the processed data.
    st.session_state.current_df = processed_data.tail(500)


# --- Initial Setup (Uses the new helper function) ---
if "current_df" not in st.session_state:
    initialize_simulation_state(0)  # Start from the beginning

st.title(INDUSTRY_TITLE)
st.write("Live data stream and anomaly detection for critical equipment.")


# --- Timestamp Formatting Function for Slider (No longer used by slider, but kept for context) ---
def format_slider_index_to_time(index):
    # Ensure index is within bounds
    index = int(index)
    if index < 0:
        return full_data_df.index[0].strftime("START: %Y-%m-%d %H:%M")
    if index >= MAX_DATA_INDEX:
        return full_data_df.index[-1].strftime("END: %Y-%m-%d %H:%M")
    return full_data_df.index[index].strftime("%Y-%m-%d %H:%M")


# --- Sidebar Content ---
with st.sidebar:
    st.header(LOGO_TEXT)
    st.markdown("---")

    # --- NEW: Fast Forward Data Stream Slider (format_func removed) ---
    st.subheader("Simulation Control")

    # The slider value controls the starting index for the stream
    # format_func is removed to fix the TypeError
    new_index_value = st.slider(
        "Fast Forward Data Stream (Select Data Point Index)",
        min_value=0,
        max_value=MAX_DATA_INDEX,
        value=st.session_state.current_row_index,
        key="fast_forward_slider",
        help="Use this to skip forward to a specific data point (row index) in the data. The simulation will resume from that point.",
    )

    # Logic to handle the jump/reset
    if new_index_value != st.session_state.current_row_index:
        initialize_simulation_state(new_index_value)
        # The Streamlit widget change automatically forces a rerun, breaking the loop and restarting
        # the stream from the new index set in initialize_simulation_state.

    st.markdown("---")
    # --- END NEW: Fast Forward Data Stream Slider ---

    # We need to ensure the radio button value is consistent across reruns
    if "selected_role" not in st.session_state:
        st.session_state.selected_role = "Plant Manager"

    role = st.radio(
        "Select User Role:",
        ("Plant Manager", "Technician"),
        key="role_radio",  # Add key for explicit management
        index=0,
    )
    st.session_state.selected_role = role
    st.markdown("---")

    st.header("Asset Context & Location")
    st.info(
        f"""
    **Asset:** Primary {INDUSTRY_TITLE}
    **Location:** Free State, South Africa
    **Objective:** Predict mechanical or electrical failure 
    using real-time **Vibration, Temperature, Power, and Amperage** data.
    """
    )

    st.markdown("---")
    st.header("Value Proposition Summary")

    st.markdown(
        """
    Our solution provides **Actionable Intelligence** by:
    - **Maximizing Uptime:** Predicting RUL (Remaining Useful Life) for proactive scheduling.
    - **Quantifiable ROI:** Tracking and justifying savings by avoiding catastrophic failures.
    - **Optimizing Energy:** Identifying operational inefficiencies via Power/Amperage anomalies.
    """
    )

    st.markdown("---")
    st.caption(f"Powered by {LOGO_TEXT} in the Free State.")


# Conditional title for the main dashboard
if st.session_state.selected_role == "Plant Manager":
    st.header("Executive Financial & Asset Health Overview")
else:
    st.header("Technician's Detailed Sensor & AI Diagnostics")

# Create empty placeholder for all dynamic UI elements
# We wrap the main content in a container with a dynamic key for stability
main_content_placeholder = st.empty()


def get_financial_risk_level(rul_days, anomaly_count, cost_at_risk):
    """Determines the color-coded financial risk level and corresponding message."""

    risk_level = "LOW"
    risk_color = "green"
    risk_summary = (
        "Operation is healthy. The PdM system is maintaining optimal reliability."
    )

    if anomaly_count == 0:
        # If we have jumped to a failure date but anomaly count is 0, still use RUL/HI
        if st.session_state.health_index < 100:
            pass  # allow RUL logic below to take over
        else:
            return risk_level, risk_color, risk_summary

    if rul_days <= 10:
        # High Risk: Imminent failure (1-10 days RUL)
        risk_level = "CRITICAL"
        risk_color = "red"
        risk_summary = (
            f"**IMMINENT FINANCIAL RISK!** Failure predicted in < 10 days. "
            f"Potential Downtime Cost: **{CURRENCY_FORMAT.format(cost_at_risk)}** (Reactive Repair is {CURRENCY_FORMAT.format(FAILURE_COST_DATA['Avg. Repair Cost (Reactive R)'])})."
        )
    elif rul_days <= 30:
        # Medium Risk: Scheduling required (10-30 days RUL)
        risk_level = "ELEVATED"
        risk_color = "orange"
        risk_summary = (
            f"**ELEVATED RISK.** Failure expected in {rul_days} days. "
            f"Proactively scheduling maintenance can save: **{CURRENCY_FORMAT.format(cost_at_risk)}** in downtime."
        )
    # else: LOW Risk (RUL > 30 days)

    return risk_level, risk_color, risk_summary


def update_plant_manager_view(content_ph, current_df, anomaly_count):
    """Updates the dashboard for the Plant Manager (Financial KPIs/ROI)."""

    # 1. Clear the placeholder content to prevent visual artifacts
    content_ph.empty()

    # 2. Draw all content inside the placeholder's context
    # Wrapping in a container with a key specific to the role and the index ensures layout stability
    with content_ph.container(
        key=f"pm_view_container_{st.session_state.current_row_index}"
    ):

        roi_data = calculate_pdm_roi(anomaly_count, FAILURE_COST_DATA)
        cost_at_risk = roi_data["Potential Downtime Cost"]
        rul_days = st.session_state.rul_days

        # Get the latest timestamp for scheduling
        latest_timestamp = current_df.index[-1]
        maintenance_schedule = get_maintenance_schedule(rul_days, latest_timestamp)

        risk_level, risk_color, risk_summary = get_financial_risk_level(
            rul_days, anomaly_count, cost_at_risk
        )

        # --- KPI Section ---
        st.subheader("Financial Performance & Predictive Insights")

        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

        col_kpi1.metric(
            "Total Savings (YTD)", CURRENCY_FORMAT.format(roi_data["Total Savings"])
        )
        col_kpi2.metric(
            "Net ROI", f"{roi_data['Net ROI']:,.1f}%", delta=roi_data["ROI Status"]
        )
        col_kpi3.metric("Total Anomalies", anomaly_count)

        # --- Asset Health Rank ---
        if st.session_state.health_index > 80:
            asset_rank = "1/12 (Best)"
            rank_delta = "High Priority"
        elif st.session_state.health_index > 50:
            asset_rank = "5/12 (High)"
            rank_delta = "Medium Priority"
        else:
            asset_rank = "11/12 (Lowest)"
            rank_delta = "Lowest Priority"

        col_kpi4.metric(
            "Asset Health Rank",
            asset_rank,
            delta=f"HI: {st.session_state.health_index}%",
        )

        if risk_color == "red":
            st_color = "🔥"
        elif risk_color == "orange":
            st_color = "⚠️"
        else:
            st_color = "✅"

        col_kpi5.metric(
            "Financial Risk Level",
            f"{st_color} {risk_level}",
            delta=f"RUL: {rul_days} Days",
        )

        st.markdown("---")

        # --- Alert & Scheduling Section ---
        st.subheader("Risk-Weighted Alert, Schedule & Justification")

        # Display Risk-Weighted Alert
        if risk_color == "red":
            st.error(f"🔴 **CRITICAL ACTION REQUIRED**")
        elif risk_color == "orange":
            st.warning(f"🟠 **ELEVATED WATCH**")
        else:
            st.success(f"🟢 **LOW RISK**")

        st.markdown(risk_summary)

        st.markdown("---")

        # Maintenance Schedule Forecast
        st.markdown(f"🗓️ **Proactive Maintenance Schedule Forecast**")
        if risk_color in ["red", "orange"]:
            st.markdown(
                f"The current Remaining Useful Life (RUL) of **{rul_days} days** dictates the following maintenance window:"
            )
            # Highlighting the schedule for urgent planning
            st.markdown(
                f"<p style='background-color:#ffebeb; padding: 10px; border-radius: 5px; font-weight: bold;'>{maintenance_schedule}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info(maintenance_schedule)

        st.markdown("---")

        # Display ROI Justification (This is the long-term value statement)
        st.info(
            f"**PdM Value Justification:** The system delivered significant savings by preemptively identifying maintenance needs.\n\n"
            f"{roi_data['Justification']}"
        )

        st.markdown("---")

        # --- Chart Section ---
        st.subheader("Key Sensor Data Trends")
        df_display = current_df.tail(200)  # Show a recent window
        fig_main = px.line(
            df_display,
            x=df_display.index,
            y=[
                "Power_kW",
                "Vibration",
            ],  # Power and Vibe are key operational indicators
            labels={"value": "Value", "Timestamp": "Time"},
            title="Recent Power and Vibration Trend",
        )
        fig_main.update_layout(height=400, xaxis_title="Timestamp")
        # Ensure the key is fully dynamic to force a clean redraw
        st.plotly_chart(
            fig_main,
            use_container_width=True,
            key=f"pm_main_chart_{st.session_state.current_row_index}_v{rul_days}",
        )
        # -------------------------------


def get_maintenance_recommendation(predicted_mode):
    """Provides a specific recommendation based on the predicted failure mode."""
    if predicted_mode == FAILURE_MODES_MAPPING["A"]:
        return "Inspect bearings on the Non-Drive End (NDE). Check for lubrication breakdown and excessive heat. Schedule immediate replacement or re-lubrication."
    elif predicted_mode == FAILURE_MODES_MAPPING["B"]:
        return "Perform laser alignment check and dynamic balance test on the motor/gearbox coupling. Correct any detected misalignment or imbalance."
    elif predicted_mode == FAILURE_MODES_MAPPING["C"]:
        return "Perform motor winding insulation test (Megger test) and check for loose connections in the terminal box. Inspect VFD output."
    elif predicted_mode == FAILURE_MODES_MAPPING["D"]:
        return "Check gearbox oil level, quality, and contamination (oil analysis). Top up or replace fluid immediately."
    else:
        return "Review sensor data trends. Monitor for continued degradation of Health Index. A specific failure mode has not been definitively identified yet."


def update_technician_view(content_ph, current_df, anomaly_count):
    """Updates the dashboard for the Technician (Sensor Charts/AI scores)."""

    # 1. Clear the placeholder content to prevent visual artifacts
    content_ph.empty()

    if current_df.empty:
        return

    latest_row = current_df.iloc[-1]

    # Wrapping in a container with a key specific to the role and the index ensures layout stability
    with content_ph.container(
        key=f"tech_view_container_{st.session_state.current_row_index}"
    ):

        # --- KPI Section ---
        st.subheader("Key Sensor Readings & AI Diagnostics")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        col_kpi1.metric("Latest Vibration", f"{latest_row['Vibration']:.2f}g")
        col_kpi2.metric("Latest Temperature", f"{latest_row['Temperature']:.2f}°C")

        # --- KPI: Health Index ---
        # The .diff() calculation will often return NaN or be less accurate in a simulation loop,
        # but we keep it for the concept. We ensure the key is dynamic.
        health_delta_value = st.session_state.current_df["Health_Index"].diff().iloc[-1]
        health_delta = (
            f"{health_delta_value:.1f}"
            if not pd.isna(health_delta_value) and health_delta_value != 0
            else None
        )

        col_kpi3.metric(
            "Health Index", f"{st.session_state.health_index}%", delta=health_delta
        )

        col_kpi4.metric("RUL (Predicted)", f"{st.session_state.rul_days} Days")

        st.markdown("---")

        # --- Alert Section ---
        st.subheader("Immediate Action & Diagnostic Report")

        # Corrected: Only show a detailed alert if RUL is low (critical)
        if st.session_state.rul_days <= 30 and anomaly_count > 0:

            # Find the most recent anomaly that contributed to the low RUL
            anomalies = current_df[
                (current_df["Is_Rule_Anomaly"] | current_df["Is_ML_Anomaly"])
                & (current_df["RUL_Days"] <= st.session_state.rul_days + 1)
            ]

            if not anomalies.empty:
                # We need the last anomaly *before* the current row to be reflective of the warning
                last_anomaly = anomalies.iloc[-1]

                # --- Predicted Failure Mode and Action ---
                predicted_mode = last_anomaly["Predicted_Failure_Mode"]
                recommended_action = get_maintenance_recommendation(predicted_mode)

                st.error(
                    f"IMMEDIATE ACTION REQUIRED: Failure predicted in **{st.session_state.rul_days} days**."
                )

                st.markdown(
                    f"**Predicted Failure Mode:** <span style='color: #FF4B4B; font-weight: bold;'>{predicted_mode}</span>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**RECOMMENDED ACTION:** <span style='color: #FF4B4B; font-weight: bold;'>{recommended_action}</span>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**Sensor Indicators:** {last_anomaly['Anomaly_Reasoning'] if last_anomaly['Anomaly_Reasoning'] else 'ML Detected: Uncategorized Anomaly.'}"
                )
                st.markdown(
                    f"**AI Confidence:** `{last_anomaly['AI_Confidence']:.1f}%`"
                )

                # --- Detailed Anomaly Chart (inside expander) ---
                with st.expander(
                    f"Show Detailed Sensor Readings (2hr window around last critical event)"
                ):
                    df_anomaly_window = current_df.loc[
                        (current_df.index >= last_anomaly.name - pd.Timedelta(hours=2))
                        & (
                            current_df.index
                            <= last_anomaly.name + pd.Timedelta(hours=2)
                        )
                    ]
                    fig_anomaly = px.line(
                        df_anomaly_window,
                        x=df_anomaly_window.index,
                        y=["Power_kW", "Amperage", "Vibration", "Temperature"],
                        title=f"Sensor Readings Around Critical Anomaly at {last_anomaly.name.strftime('%Y-%m-%d %H:%M:%S')}",
                    )

                    fig_anomaly.add_trace(
                        go.Scatter(
                            x=[last_anomaly.name],
                            y=[last_anomaly["Power_kW"]],
                            mode="markers",
                            marker=dict(color="red", size=15, symbol="x"),
                            name="Anomaly Point",
                        )
                    )

                    st.plotly_chart(
                        fig_anomaly,
                        use_container_width=True,
                        # Key uses anomaly timestamp and current index, so it is safe.
                        key=f"tech_anomaly_chart_{last_anomaly.name.isoformat()}_{st.session_state.current_row_index}_v{st.session_state.rul_days}",
                    )
            else:
                st.success(
                    "System Operating Normally - All sensor data within acceptable limits."
                )

        else:
            st.success(
                "System Operating Normally - All sensor data within acceptable limits."
            )

        st.markdown("---")

        # --- Charts Section ---

        # Chart 1: Full Sensor Data Monitoring (Now full width)
        st.subheader("Full History Sensor Data Monitoring")
        df_display_full = current_df.tail(1000)
        fig_main = px.line(
            df_display_full,
            x=df_display_full.index,
            y=["Power_kW", "Amperage", "Vibration", "Temperature"],
            labels={"value": "Value", "Timestamp": "Time"},
            title="Live Sensor Data Stream (Last 1000 Points)",
        )
        fig_main.update_layout(height=400, xaxis_title="Timestamp")
        st.plotly_chart(
            fig_main,
            use_container_width=True,
            key=f"tech_main_chart_{st.session_state.current_row_index}_v{st.session_state.rul_days}",
        )

        # --- NEW CHART: Failure Mode Breakdown ---
        st.markdown("---")
        st.subheader("Historical Predicted Failure Mode Breakdown")

        # Create a dataframe for the failure mode counts
        df_failure_counts = (
            current_df[
                current_df["Predicted_Failure_Mode"] != FAILURE_MODES_MAPPING["E"]
            ]["Predicted_Failure_Mode"]
            .value_counts()
            .reset_index()
        )
        df_failure_counts.columns = ["Failure Mode", "Count"]

        if not df_failure_counts.empty:
            fig_modes = px.bar(
                df_failure_counts,
                x="Failure Mode",
                y="Count",
                color="Failure Mode",
                title="Total Anomaly Count by Predicted Failure Type",
                color_discrete_map={
                    FAILURE_MODES_MAPPING["A"]: "red",
                    FAILURE_MODES_MAPPING["B"]: "orange",
                    FAILURE_MODES_MAPPING["C"]: "blue",
                    FAILURE_MODES_MAPPING["D"]: "purple",
                    FAILURE_MODES_MAPPING["E"]: "gray",
                },
            )
            fig_modes.update_layout(
                height=400, xaxis={"categoryorder": "total descending"}
            )
            st.plotly_chart(
                fig_modes,
                use_container_width=True,
                key=f"tech_failure_modes_chart_{st.session_state.current_row_index}_v{st.session_state.rul_days}",
            )
        else:
            st.info("No predictive failure modes detected yet.")
        # ----------------------------------------


def update_dashboard(content_ph, current_df, anomaly_count, role):
    """Function to call the appropriate view update function."""
    if role == "Plant Manager":
        update_plant_manager_view(content_ph, current_df, anomaly_count)
    else:  # Technician
        update_technician_view(content_ph, current_df, anomaly_count)


# The continuous simulation loop
# We only enter this loop if the index is less than the total data length.
if st.session_state.current_row_index < len(full_data_df):

    # Force the first render right away to show the selected starting point
    update_dashboard(
        main_content_placeholder,
        st.session_state.current_df,
        st.session_state.anomaly_count,
        st.session_state.selected_role,
    )

    # Start the continuous stream from the current index
    while st.session_state.current_row_index < len(full_data_df):
        try:
            next_row = full_data_df.iloc[
                st.session_state.current_row_index : st.session_state.current_row_index
                + 1
            ].copy()

            if next_row.empty:
                st.info("Data stream ended naturally.")
                break

            row_to_process = next_row.iloc[0]  # Safely get the row

            # --- Anomaly Logic ---
            is_rule_anomaly, rule_reasoning = check_rule_based_anomalies(row_to_process)
            (
                is_ml_anomaly,
                ml_score,
                ai_confidence,
                rul_days_current,
                health_index_current,
                predicted_failure_mode,
            ) = check_ml_anomaly(row_to_process, iso_forest_model, ml_features)

            # --- Append Anomaly Data to the new row ---
            next_row.loc[:, "Is_Rule_Anomaly"] = is_rule_anomaly
            next_row.loc[:, "Is_ML_Anomaly"] = is_ml_anomaly
            next_row.loc[:, "Anomaly_Reasoning"] = rule_reasoning
            next_row.loc[:, "ML_Anomaly_Score"] = ml_score
            next_row.loc[:, "AI_Confidence"] = ai_confidence
            next_row.loc[:, "RUL_Days"] = rul_days_current
            next_row.loc[:, "Health_Index"] = health_index_current
            next_row.loc[:, "Predicted_Failure_Mode"] = predicted_failure_mode

            st.session_state.current_df = pd.concat(
                [st.session_state.current_df, next_row]
            )
            st.session_state.current_row_index += 1

            # --- Update Global State (RUL/HI) ---
            if is_rule_anomaly or is_ml_anomaly:
                st.session_state.anomaly_count += 1
                st.session_state.rul_days = min(
                    st.session_state.rul_days, rul_days_current
                )
                st.session_state.health_index = min(
                    st.session_state.health_index, health_index_current
                )
            elif st.session_state.rul_days < 30 and ml_score > 0:
                # If system recovers, RUL and HI should slowly improve
                st.session_state.rul_days = min(30, st.session_state.rul_days + 1)
                st.session_state.health_index = min(
                    100, st.session_state.health_index + 3
                )  # Slowly improve HI
            # ---------------------------

            # Pass the single main placeholder
            update_dashboard(
                main_content_placeholder,
                st.session_state.current_df,
                st.session_state.anomaly_count,
                st.session_state.selected_role,  # Use the role from session state
            )

            time.sleep(DATA_POINT_INTERVAL)

        except Exception as e:
            st.error(
                f"Error: The simulation crashed while processing row {st.session_state.current_row_index}. Details: {e}"
            )
            st.warning(
                "The simulation has stopped. Please check your data at this row for any inconsistencies (e.g., missing values, incorrect data types)."
            )
            # Break the loop on error
            break

# Final status message after the loop finishes
if st.session_state.current_row_index >= len(full_data_df):
    st.info("End of simulation. All data has been processed.")
