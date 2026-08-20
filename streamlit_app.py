import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Nordex Shift Optimization",
    page_icon="⚙️",
    layout="wide"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/",
            timeout=5
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


def predict_shift(payload):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()

        st.error(
            f"Prediction API error: {response.status_code}\n\n"
            f"{response.text}"
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to FastAPI: {e}")

    return None


def optimize_shift(payload):
    try:
        response = requests.post(
            f"{API_URL}/optimize",
            json=payload,
            timeout=600
        )

        if response.status_code == 200:
            return response.json()

        st.error(
            f"Optimization API error: {response.status_code}\n\n"
            f"{response.text}"
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to FastAPI: {e}")

    return None


def retrain_model():
    try:
        response = requests.post(
            f"{API_URL}/retrain",
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        st.error(
            f"Retraining API error: {response.status_code}\n\n"
            f"{response.text}"
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to FastAPI: {e}")

    return None


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("⚙️ Nordex Shift Optimization")
st.markdown(
    "Machine-learning powered shift efficiency prediction "
    "and optimization."
)


# ---------------------------------------------------------
# API status
# ---------------------------------------------------------

if check_api():
    st.success("🟢 FastAPI is online — localhost:8000")
else:
    st.error(
        "🔴 FastAPI is offline. Start your FastAPI server on port 8000."
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select operation",
    [
        "📊 Predict Shift Efficiency",
        "🔧 Optimize Shift",
        "🚀 Retrain Model"
    ]
)


# =========================================================
# PREDICTION PAGE
# =========================================================

if page == "📊 Predict Shift Efficiency":

    st.header("Predict Shift Efficiency")

    st.write(
        "Enter the operational details for a shift and the trained "
        "ML model will predict its efficiency score."
    )

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            units_produced = st.number_input(
                "Units Produced",
                min_value=0,
                value=800,
                step=1
            )

            defect_count = st.number_input(
                "Defect Count",
                min_value=0,
                value=10,
                step=1
            )

            cycle_time_avg = st.number_input(
                "Average Cycle Time",
                min_value=0.0,
                value=37.0,
                step=0.1
            )

            experience_level = st.number_input(
                "Experience Level",
                min_value=0,
                value=3,
                step=1
            )

        with col2:
            runtime_hours = st.number_input(
                "Runtime Hours",
                min_value=0.0,
                value=7.5,
                step=0.1
            )

            downtime_minutes = st.number_input(
                "Downtime Minutes",
                min_value=0.0,
                value=30.0,
                step=1.0
            )

            maintenance_flag = st.selectbox(
                "Maintenance Flag",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            maintenance_downtime = st.number_input(
                "Maintenance Downtime",
                min_value=0.0,
                value=20.0,
                step=1.0
            )

        with col3:
            temperature = st.number_input(
                "Temperature",
                value=24.0,
                step=0.1
            )

            humidity = st.number_input(
                "Humidity",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0
            )

            defect_rate = st.number_input(
                "Defect Rate",
                min_value=0,
                value=2,
                step=1
            )

            day_of_week = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday"
                ][x]
            )

        st.divider()

        col4, col5, col6 = st.columns(3)

        with col4:
            shift_name = st.selectbox(
                "Shift Name",
                [
                    "Morning",
                    "Afternoon",
                    "Night"
                ]
            )

        with col5:
            skill_category = st.selectbox(
                "Skill Category",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )

        with col6:
            machine_status = st.selectbox(
                "Machine Status",
                [
                    "Operational",
                    "Maintenance",
                    "Idle",
                    "Down"
                ]
            )

        submitted = st.form_submit_button(
            "🔮 Predict Efficiency",
            use_container_width=True
        )

    if submitted:

        payload = {
            "units_produced": units_produced,
            "defect_count": defect_count,
            "cycle_time_avg": cycle_time_avg,
            "experience_level": experience_level,
            "runtime_hours": runtime_hours,
            "downtime_minutes": downtime_minutes,
            "maintenance_flag": maintenance_flag,
            "maintenance_downtime": maintenance_downtime,
            "temperature": temperature,
            "humidity": humidity,
            "defect_rate": defect_rate,
            "day_of_week": day_of_week,
            "shift_name": shift_name,
            "skill_category": skill_category,
            "machine_status": machine_status
        }

        with st.spinner("Running prediction..."):

            result = predict_shift(payload)

        if result:

            score = result["predicted_shift_efficiency_score"]

            st.success("Prediction completed!")

            st.metric(
                label="Predicted Shift Efficiency Score",
                value=f"{score:.2f}"
            )

            # Simple interpretation
            if score >= 0.80:
                st.success("🟢 Excellent shift efficiency")
            elif score >= 0.60:
                st.warning("🟡 Moderate shift efficiency")
            else:
                st.error("🔴 Low shift efficiency")


# =========================================================
# OPTIMIZATION PAGE
# =========================================================

elif page == "🔧 Optimize Shift":

    st.header("🔧 optimize Shift")

    st.write(
        "optuna will search through different operational "
        "parameters to find the combination that maximizes "
        "the predicted shift efficiency."
    )

    col1, col2 = st.columns(2)

    with col1:

        shift_name = st.selectbox(
            "Shift Name",
            [
                "Morning",
                "Afternoon",
                "Night"
            ],
            key="optimization_shift"
        )

        skill_category = st.selectbox(
            "Skill Category",
            [
                "Low",
                "Medium",
                "High"
            ],
            key="optimization_skill"
        )

        machine_status = st.selectbox(
            "Machine Status",
            [
                "Operational",
                "Maintenance",
                "Idle",
                "Down"
            ],
            key="optimization_machine"
        )

        n_trials = st.slider(
            "Number of optimization Trials",
            min_value=10,
            max_value=500,
            value=50,
            step=10
        )

    with col2:

        st.subheader("optimization Ranges")

        st.write("Experience Level")

        exp_min, exp_max = st.slider(
            "Experience Range",
            min_value=0,
            max_value=10,
            value=(1, 5)
        )

        st.write("Downtime")

        downtime_min, downtime_max = st.slider(
            "Downtime Range (minutes)",
            min_value=0.0,
            max_value=480.0,
            value=(0.0, 120.0),
            step=5.0
        )

        st.write("Defects")

        defect_min, defect_max = st.slider(
            "Defect Count Range",
            min_value=0,
            max_value=100,
            value=(0, 20)
        )

    st.divider()

    if st.button(
        "🚀 Run optimization",
        type="primary",
        use_container_width=True
    ):

        payload = {
            "shift_name": shift_name,
            "skill_category": skill_category,
            "machine_status": machine_status,
            "exp_range": [
                exp_min,
                exp_max
            ],
            "downtime_range": [
                downtime_min,
                downtime_max
            ],
            "defect_range": [
                defect_min,
                defect_max
            ],
            "n_trials": n_trials
        }

        with st.spinner(
            f"Running {n_trials} optuna trial..."
        ):

            result = optimize_shift(payload)

        if result:

            st.success("optimization completed!")

            # -------------------------------------------------
            # Best result
            # -------------------------------------------------

            st.subheader("🏆 Best Configuration")

            best_score = result["best_shift_efficiency_score"]
            best_params = result["best_parameters"]

            st.metric(
                "Best Shift Efficiency Score",
                f"{best_score:.4f}"
            )

            st.write("### Recommended Parameters")

            best_df = pd.DataFrame(
                [
                    {
                        "Parameter": key,
                        "Value": value
                    }
                    for key, value in best_params.items()
                ]
            )

            st.dataframe(
                best_df,
                use_container_width=True,
                hide_index=True
            )

            # -------------------------------------------------
            # Top trials
            # -------------------------------------------------

            st.subheader("📈 Top 10 Optimization Trials")

            top_trials = result.get(
                "top_trials",
                []
            )

            if top_trials:

                trials_df = pd.DataFrame(top_trials)

                # Put value first if available
                if "value" in trials_df.columns:

                    columns = ["value"] + [
                        col
                        for col in trials_df.columns
                        if col != "value"
                    ]

                    trials_df = trials_df[columns]

                st.dataframe(
                    trials_df,
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------------------------
                # Chart
                # -------------------------------------------------

                if "value" in trials_df.columns:

                    st.subheader(
                        "Efficiency Score — Top Trials"
                    )

                    chart_df = trials_df[
                        ["value"]
                    ].reset_index()

                    chart_df.columns = [
                        "Trial",
                        "Efficiency Score"
                    ]

                    st.bar_chart(
                        chart_df.set_index("Trial")
                    )


# =========================================================
# RETRAIN PAGE
# =========================================================

elif page == "🚀 Retrain Model":

    st.header("🚀 Retrain ML Model")

    st.warning(
        "Retraining starts the training pipeline in a background "
        "thread. The API will reload the model after training."
    )

    st.write(
        "Use this when you want to train a new model using your "
        "latest training data."
    )

    if st.button(
        "🚀 Start Model Retraining",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Starting training process..."):

            result = retrain_model()

        if result:

            st.success(
                result.get(
                    "message",
                    "Retraining started successfully."
                )
            )

            st.info(
                "Training is running in the FastAPI background "
                "thread. The model will be reloaded automatically "
                "when training finishes."
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Nordex Shift Optimization API"
)

st.sidebar.caption(
    "FastAPI: http://localhost:8000"
)
