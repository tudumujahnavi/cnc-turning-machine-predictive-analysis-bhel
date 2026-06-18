import streamlit as st
import matplotlib.pyplot as plt
from model import PredictiveMaintenance

# =================================================
# PAGE CONFIG
# =================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    layout="centered"
)

# =================================================
# CSS (DO NOT shrink fonts globally)
# =================================================

st.markdown("""
<style>

.block-container {
    max-width: 900px;
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.stButton button {
    width: 130px;
    height: 38px;
    font-size: 13px;
}

.element-container {
    margin-bottom: 0.4rem;
}

</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD MODEL
# =================================================

@st.cache_resource
def load_model():
    return PredictiveMaintenance()

system = load_model()

# =================================================
# TITLE
# =================================================

st.markdown(
    "<h2 style='text-align:center;'><color='red'>AI-Based Predictive Maintenance and suggestion System for CNC machine</h2>",
    unsafe_allow_html=True
)

st.markdown("### Machine Parameters")

# =================================================
# INPUTS
# =================================================

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    air_temp = st.number_input("Air Temperature (K)", value=298.0)
    rpm = st.number_input("Rotational Speed (rpm)", value=1500)

with col2:
    process_temp = st.number_input("Process Temperature (K)", value=308.0)
    torque = st.number_input("Torque (Nm)", value=40.0)
    tool_wear = st.number_input("Tool Wear (min)", value=0)

type_map = {"L": 0, "M": 1, "H": 2}

# =================================================
# PREDICT BUTTON
# =================================================

if st.button("Predict"):

    values = [
        type_map[machine_type],
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]

    prediction, probability = system.predict(values)

    # =================================================
    # MODEL TABLE
    # =================================================

    st.markdown("### Model Comparison")

    st.dataframe(
        system.results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        height=380
    )

    st.success(f"Best Model: {system.best_model_name}")

    # =================================================
    # PREDICTION RESULT
    # =================================================

    st.markdown("### Prediction Result")

    if prediction == 1:
        st.error(f"Failure Predicted ({probability:.2%})")
    else:
        st.success(f"No Failure Predicted ({probability:.2%})")

    # =================================================
    # BAR CHART
    # =================================================

    st.markdown("### Model Accuracy Comparison")

    fig1, ax1 = plt.subplots(figsize=(6, 3))

    ax1.bar(
        system.results["Model"],
        system.results["Accuracy"],
        width=0.5
    )

    ax1.set_title("Model Accuracy Comparison", fontsize=10)
    ax1.set_xlabel("Models", fontsize=9)
    ax1.set_ylabel("Accuracy", fontsize=9)

    plt.setp(
        ax1.get_xticklabels(),
        rotation=25,
        ha="right",
        fontsize=9
    )

    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    fig1.tight_layout()
    st.pyplot(fig1)

    # =================================================
    # PIE CHART (ONLY SMALL TEXT HERE)
    # =================================================

    st.markdown("### Machine Performance Analysis")

    failure_percent = probability * 100
    healthy_percent = 100 - failure_percent

    fig2, ax2 = plt.subplots(figsize=(3.5, 3.5), dpi=120)

    sizes = [healthy_percent, failure_percent]
    colors = ["#2ecc71", "#e74c3c"]

    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        radius=0.2,
        colors=colors,
        textprops={
            "fontsize": 4   # ONLY PIE CHART FONT SMALL
        }
    )

    ax2.set_title("Machine Health Status", fontsize=10)
    ax2.axis("equal")

    ax2.legend(
        wedges,
        ["Healthy", "Failure Risk"],
        loc="upper right",
        bbox_to_anchor=(1.25, 1.1),
        fontsize=4,
        frameon=False
    )

    fig2.tight_layout()
    st.pyplot(fig2)

    # =================================================
    # FAILURE EXPLANATION
    # =================================================

    if prediction == 1:

        st.markdown("### Failure Reasons")

        reasons = system.explain_failure(values)

        for r in reasons:
            st.write("•", r)

    # =================================================
    # MAINTENANCE SUGGESTIONS
    # =================================================

    st.markdown("### Maintenance Suggestions")

    suggestions = system.suggestions(values)

    for s in suggestions:
        st.write("✓", s)
