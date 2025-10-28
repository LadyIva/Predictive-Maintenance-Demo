import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
import time
import os

# --- Configuration ---
st.set_page_config(layout="wide", page_title="S.I.L.K.E AI Predictive Maintenance Demo")

DATA_POINT_INTERVAL = 0.5  # Increased speed for better demo flow
file_path = "maize_mill_simulated_sensor_data.csv"

# --- Placeholder for industry-specific failure/cost data (for ROI calculation) ---
FAILURE_COST_DATA = {
    "Average Downtime Cost per Hour ($)": 15000,
    "Time Saved by PdM (Hours)": 4,  # Estimated hours saved vs reactive
    "Avg. Repair Cost (Reactive $)": 50000,
    "Avg. Repair Cost (Predictive $)": 10000,  # Cheaper repair before catastrophic failure
    "Equipment Lifespan (Years)": 10,
    "PdM System Annual Cost ($)": 25000,
}


# --- 1. Load Data and Initialize Models (Runs only once) ---
@st.cache_data
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


@st.cache_resource
def train_isolation_forest(df_initial):
    if df_initial.empty:
        return None, None

    features_for_ml = ["Power_kW", "Amperage", "Vibration", "Temperature"]
    X_train = df_initial[features_for_ml].dropna()
    if X_train.empty or len(X_train) < 2:
        return None, None
    # Increased contamination slightly for more frequent demo alerts
    model = IsolationForest(contamination=0.015, random_state=42)
    model.fit(X_train.values)
    return model, features_for_ml


# Global data and model initialization
full_data_df = load_data(file_path)

try:
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


def check_ml_anomaly(row, model, features):
    """Applies the Isolation Forest model to a single data row."""
    if model is None or features is None or any(pd.isna(row[features])):
        # Return False, a neutral score, and 0% confidence
        return False, 0.0, 0.0

    data_point = np.array([row[features].values])
    ml_prediction = model.predict(data_point)[0]
    ml_score = model.decision_function(data_point)[0]
    is_ml_anomaly = ml_prediction == -1

    # NEW: Calculate a proxy for AI Confidence
    # Isolation Forest score is the negative distance to the boundary.
    # A smaller, less negative score means closer to the 'normal' boundary (less confident in anomaly).
    # A larger negative score means deeper into the anomaly region (more confident in anomaly).
    # Confidence in 'Anomaly' is inversely related to the score's magnitude, but we need a 0-100 scale.
    # Use a sigmoid or simple scaling for a proxy RUL and Confidence.

    # Confidence in ANOMALY (0% to 100%)
    # ml_score is typically between -0.3 and 0.3. Let's map it.
    if is_ml_anomaly:
        # For anomalies (negative score), map deeper negative to higher confidence.
        # Example: map -0.3 to 100%, 0.0 to 50% (threshold for detection)
        # We can use a simplified linear map for demo purposes:
        max_negative_score = -0.1  # Assume this is the 'most anomalous' observed
        confidence = min(
            100, max(50, 50 - (ml_score * 500))
        )  # Starts at 50% at threshold (0.0 score)
    else:
        # For normal points (positive score), confidence in 'Normal' is higher for higher score.
        confidence = 100 - (abs(ml_score) * 200)  # Simple inverse mapping
        confidence = max(0, min(100, confidence))

    # NEW: Calculate a simple proxy for Remaining Useful Life (RUL) in Days
    # RUL decreases with increasingly negative ML scores.
    # Assume 30 days RUL at the first anomaly detection (ml_score ~0.0)
    base_rul_days = 30
    rul_days = max(1, int(base_rul_days + (ml_score * 100)))

    return is_ml_anomaly, ml_score, confidence, rul_days


# --- NEW: ROI Calculation Function ---
def calculate_pdm_roi(anomaly_count, cost_data):
    """Calculates ROI metrics based on simulated savings from avoided failures."""

    # Estimate the number of 'critical' failures avoided
    # Assume every 5th detected anomaly (Is_ML_Anomaly or Is_Rule_Anomaly) would have been a catastrophic failure
    critical_failures_avoided = anomaly_count // 5

    if critical_failures_avoided == 0:
        return {
            "Total Savings": 0,
            "Net ROI": 0,
            "ROI Status": "Awaiting critical detection...",
            "Justification": "No major failure predicted/avoided yet.",
        }

    # 1. Savings from avoided downtime
    downtime_savings_per_failure = (
        cost_data["Time Saved by PdM (Hours)"]
        * cost_data["Average Downtime Cost per Hour ($)"]
    )
    total_downtime_savings = downtime_savings_per_failure * critical_failures_avoided

    # 2. Savings from reduced repair costs
    repair_cost_savings_per_failure = (
        cost_data["Avg. Repair Cost (Reactive $)"]
        - cost_data["Avg. Repair Cost (Predictive $)"]
    )
    total_repair_savings = repair_cost_savings_per_failure * critical_failures_avoided

    total_savings = total_downtime_savings + total_repair_savings

    # Annual Investment Cost (PdM System)
    investment_cost = cost_data["PdM System Annual Cost ($)"]

    net_profit = total_savings - investment_cost

    # ROI Formula: (Total Savings - Investment Cost) / Investment Cost * 100
    if investment_cost > 0:
        net_roi = (net_profit / investment_cost) * 100
    else:
        net_roi = 0

    justification = (
        f"Avoided **{critical_failures_avoided}** catastrophic failure(s), saving:\n"
        f"- **${total_downtime_savings:,.0f}** in potential downtime.\n"
        f"- **${total_repair_savings:,.0f}** in reduced repair costs."
    )

    return {
        "Total Savings": total_savings,
        "Net ROI": net_roi,
        "ROI Status": f"ROI: {net_roi:,.1f}%",
        "Justification": justification,
    }


# --- 3. Streamlit UI Rendering and Simulation Logic ---
st.title("S.I.L.K.E Predictive Maintenance Demo")

# --- NEW: Role Selector ---
role = st.sidebar.radio(
    "👤 Select User Role:", ("Plant Manager", "Technician"), index=0
)
st.sidebar.markdown("---")

# Conditional title for the main dashboard
if role == "Plant Manager":
    st.header("Financial & Operational Health Dashboard 💰")
else:
    st.header("Detailed Sensor & AI Anomaly Diagnostics 🛠️")


# Initialize session state for simulation
if "current_df" not in st.session_state:
    initial_row = full_data_df.head(1).copy()
    initial_row["Is_Rule_Anomaly"] = False
    initial_row["Is_ML_Anomaly"] = False
    initial_row["Anomaly_Reasoning"] = ""
    initial_row["ML_Anomaly_Score"] = 0.0
    initial_row["AI_Confidence"] = 0.0  # NEW
    initial_row["RUL_Days"] = 0  # NEW
    st.session_state.current_df = initial_row
    st.session_state.current_row_index = 1
    st.session_state.anomaly_count = 0
    st.session_state.rul_days = 30  # Initial RUL

# Create empty placeholders for all dynamic UI elements
kpi_placeholder = st.empty()
alert_placeholder = st.empty()
chart_placeholder = st.empty()


def update_plant_manager_view(kpi_ph, alert_ph, current_df, anomaly_count):
    """Updates the dashboard for the Plant Manager (Financial KPIs/ROI)."""

    roi_data = calculate_pdm_roi(anomaly_count, FAILURE_COST_DATA)
    latest_row = current_df.iloc[-1]

    with kpi_ph.container():
        st.subheader("Financial Performance & Predictive Insights")

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        col_kpi1.metric("💰 Total Savings (YTD)", f"${roi_data['Total Savings']:,.0f}")
        col_kpi2.metric(
            "📈 Net ROI", f"{roi_data['Net ROI']:,.1f}%", delta=roi_data["ROI Status"]
        )
        col_kpi3.metric("🚨 Total Anomalies", anomaly_count)

        # Display the AI's current RUL (or a neutral status if no anomaly)
        rul_display = (
            f"{st.session_state.rul_days} Days"
            if st.session_state.rul_days < 30
            else "Normal (30+ Days)"
        )
        col_kpi4.metric("⚙️ Predicted RUL", rul_display)

    with alert_ph.container():
        st.subheader("ROI Justification & Operational Status")

        # Display ROI Justification
        st.info(
            f"**Value Proposition:** The PdM system has delivered significant savings by preemptively identifying maintenance needs.\n\n"
            f"{roi_data['Justification']}"
        )

        # Display a simplified alert message
        if anomaly_count > 0:
            last_anomaly = current_df[
                current_df["Is_Rule_Anomaly"] | current_df["Is_ML_Anomaly"]
            ].iloc[-1]
            st.error(
                f"⚠️ **CRITICAL ALERT:** Potential failure predicted! RUL is **{st.session_state.rul_days} days**."
            )
            st.markdown(
                f"**Anomaly Cause:** {last_anomaly['Anomaly_Reasoning'] if last_anomaly['Anomaly_Reasoning'] else 'AI Detected: Uncategorized Anomaly.'}"
            )
        else:
            st.success("✅ System Health: Excellent. No immediate financial risk.")

    # The Plant Manager typically only needs a high-level operational chart
    with chart_placeholder.container():
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
        st.plotly_chart(fig_main, use_container_width=True)


def update_technician_view(kpi_ph, alert_ph, chart_ph, current_df, anomaly_count):
    """Updates the dashboard for the Technician (Sensor Charts/AI scores)."""

    if current_df.empty:
        return

    latest_row = current_df.iloc[-1]

    with kpi_ph.container():
        st.subheader("Key Sensor Readings & AI Diagnostics")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        # Sensor data
        col_kpi1.metric("Latest Vibration", f"{latest_row['Vibration']:.2f}g")
        col_kpi2.metric("Latest Temperature", f"{latest_row['Temperature']:.2f}°C")

        # NEW: AI Metrics
        col_kpi3.metric("🤖 AI Confidence", f"{latest_row['AI_Confidence']:.1f}%")
        col_kpi4.metric("⏳ RUL (Predicted)", f"{st.session_state.rul_days} Days")

    with alert_ph.container():
        if anomaly_count > 0:
            last_anomaly = current_df[
                current_df["Is_Rule_Anomaly"] | current_df["Is_ML_Anomaly"]
            ].iloc[-1]
            st.error(
                f"🚨 Anomaly Detected at {last_anomaly.name.strftime('%Y-%m-%d %H:%M:%S')}!"
            )
            st.markdown(
                f"**Root Cause Analysis:** {last_anomaly['Anomaly_Reasoning'] if last_anomaly['Anomaly_Reasoning'] else 'ML Detected: Uncategorized Anomaly.'}"
            )
            st.markdown(
                f"**ML Score:** `{last_anomaly['ML_Anomaly_Score']:.4f}` **| AI Confidence:** `{last_anomaly['AI_Confidence']:.1f}%`"
            )

            # --- Detailed Anomaly Chart (same as original, but clearer) ---
            with st.expander(
                f"Show Detailed Sensor Readings (2hr window around anomaly)"
            ):
                df_anomaly_window = current_df.loc[
                    (current_df.index >= last_anomaly.name - pd.Timedelta(hours=2))
                    & (current_df.index <= last_anomaly.name + pd.Timedelta(hours=2))
                ]
                fig_anomaly = px.line(
                    df_anomaly_window,
                    x=df_anomaly_window.index,
                    y=["Power_kW", "Amperage", "Vibration", "Temperature"],
                    title=f"Sensor Readings Around Anomaly at {last_anomaly.name.strftime('%Y-%m-%d %H:%M:%S')}",
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
                    key=f"tech_anomaly_chart_{last_anomaly.name.isoformat()}_{st.session_state.current_row_index}",
                )

        else:
            st.success(
                "✅ System Operating Normally - All sensor data within acceptable limits."
            )

    # --- Full Sensor Chart ---
    with chart_ph.container():
        st.subheader("Full History Sensor Data Monitoring")
        df_display_full = current_df.tail(1000)  # Only display the last 1000 points
        fig_main = px.line(
            df_display_full,
            x=df_display_full.index,
            y=["Power_kW", "Amperage", "Vibration", "Temperature"],
            labels={"value": "Value", "Timestamp": "Time"},
            title="Live Sensor Data Stream (Last 1000 Points)",
        )
        fig_main.update_layout(height=500, xaxis_title="Timestamp")
        st.plotly_chart(
            fig_main,
            use_container_width=True,
            key=f"tech_main_chart_{st.session_state.current_row_index}",
        )


def update_dashboard(kpi_ph, alert_ph, chart_ph, current_df, anomaly_count, role):
    """Function to call the appropriate view update function."""
    if role == "Plant Manager":
        update_plant_manager_view(kpi_ph, alert_ph, current_df, anomaly_count)
    else:  # Technician
        update_technician_view(kpi_ph, alert_ph, chart_ph, current_df, anomaly_count)


# The continuous simulation loop
while st.session_state.current_row_index < len(full_data_df):
    try:
        next_row = full_data_df.iloc[
            st.session_state.current_row_index : st.session_state.current_row_index + 1
        ].copy()

        is_rule_anomaly, rule_reasoning = check_rule_based_anomalies(next_row.iloc[0])
        is_ml_anomaly, ml_score, ai_confidence, rul_days = (
            check_ml_anomaly(  # NEW: Unpack new values
                next_row.iloc[0], iso_forest_model, ml_features
            )
        )

        next_row.loc[:, "Is_Rule_Anomaly"] = is_rule_anomaly
        next_row.loc[:, "Is_ML_Anomaly"] = is_ml_anomaly
        next_row.loc[:, "Anomaly_Reasoning"] = rule_reasoning
        next_row.loc[:, "ML_Anomaly_Score"] = ml_score
        next_row.loc[:, "AI_Confidence"] = ai_confidence  # NEW
        next_row.loc[:, "RUL_Days"] = rul_days  # NEW

        st.session_state.current_df = pd.concat([st.session_state.current_df, next_row])
        st.session_state.current_row_index += 1

        # Update RUL globally if an anomaly is detected, and it's the lowest seen
        if is_rule_anomaly or is_ml_anomaly:
            st.session_state.anomaly_count += 1
            st.session_state.rul_days = min(st.session_state.rul_days, rul_days)
        elif st.session_state.rul_days < 30 and ml_score > 0:
            # Simple simulation: RUL can recover slightly if score is positive
            st.session_state.rul_days = min(30, st.session_state.rul_days + 1)

        # Call the new function to update all UI elements
        update_dashboard(
            kpi_placeholder,
            alert_placeholder,
            chart_placeholder,
            st.session_state.current_df,
            st.session_state.anomaly_count,
            role,  # NEW: Pass the role
        )

        time.sleep(DATA_POINT_INTERVAL)

    except Exception as e:
        st.error(
            f"⚠️ Error: The simulation crashed while processing row {st.session_state.current_row_index}."
        )
        st.error(f"**Details:** {e}")
        st.warning(
            "The simulation has stopped. Please check your data at this row for any inconsistencies (e.g., missing values, incorrect data types)."
        )
        st.stop()

# Final status message after the loop
if st.session_state.current_row_index >= len(full_data_df):
    st.info("End of simulation. All data has been processed.")
