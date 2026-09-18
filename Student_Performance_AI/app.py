import json
from pathlib import Path
import sys
import streamlit as st
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from src.predict import predict_student
from src.train_model import train_model, FEATURES, CLASS_NAMES

MODEL_PATH = BASE / "models" / "student_model.keras"
METRICS_PATH = BASE / "models" / "metrics.json"

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Based Student Performance Prediction")
st.caption("Deep Learning + NumPy + TensorFlow/Keras + Scikit-learn")

with st.sidebar:
    st.header("System")
    st.write("MLP Performance Classifier")
    st.write("Classes: At Risk, Average, Good, Excellent")

    if not MODEL_PATH.exists():
        st.warning("Model is not trained yet.")

        if st.button("Train Model"):
            with st.spinner("Training neural network..."):
                train_model()

            st.success("Training completed. Reload the page.")
            st.rerun()


tab1, tab2, tab3 = st.tabs(
    ["🔮 Prediction", "📊 Model Metrics", "📚 Project Information"]
)


# =========================================================
# TAB 1 - PREDICTION
# =========================================================

with tab1:

    st.subheader("👨‍🎓 Enter Student Information")

    # Student Name
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student name"
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0
        )

        study_hours = st.number_input(
            "Study Hours / Week",
            min_value=0.0,
            max_value=40.0,
            value=12.0
        )

    with c2:
        previous_score = st.number_input(
            "Previous Score (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        assignment_score = st.number_input(
            "Assignment Score (%)",
            min_value=0.0,
            max_value=100.0,
            value=78.0
        )

    with c3:
        internal_score = st.number_input(
            "Internal Assessment (%)",
            min_value=0.0,
            max_value=100.0,
            value=76.0
        )

        practical_score = st.number_input(
            "Practical Score (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )

    with c4:
        participation = st.number_input(
            "Participation (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        gpa = st.number_input(
            "GPA",
            min_value=0.0,
            max_value=10.0,
            value=8.0
        )

    st.divider()

    if st.button(
        "🚀 Predict Performance",
        type="primary",
        use_container_width=True
    ):

        if student_name.strip() == "":
            st.error("Please enter the student name.")

        elif not MODEL_PATH.exists():
            st.error("Train the model first.")

        else:
            data = {
                "attendance": attendance,
                "study_hours": study_hours,
                "previous_score": previous_score,
                "assignment_score": assignment_score,
                "internal_score": internal_score,
                "practical_score": practical_score,
                "participation": participation,
                "gpa": gpa
            }

            result = predict_student(data)

            st.divider()

            st.success(
                f"Prediction completed for **{student_name.strip()}**"
            )

            st.subheader(
                f"🎓 Student: {student_name.strip()}"
            )

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Predicted Performance",
                result["performance"]
            )

            r2.metric(
                "Confidence",
                f'{result["confidence"]:.2f}%'
            )

            r3.metric(
                "Risk Level",
                result["risk"]
            )

            st.subheader("📊 Class Probabilities")

            probs = pd.DataFrame({
                "Performance": list(
                    result["probabilities"].keys()
                ),
                "Probability (%)": list(
                    result["probabilities"].values()
                )
            })

            st.bar_chart(
                probs.set_index("Performance")
            )

            st.subheader("💡 Recommendations")

            for rec in result["recommendations"]:
                st.write("•", rec)


# =========================================================
# TAB 2 - MODEL METRICS
# =========================================================

with tab2:

    st.subheader("📊 Model Evaluation")

    if METRICS_PATH.exists():

        metrics = json.loads(
            METRICS_PATH.read_text(
                encoding="utf-8"
            )
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Accuracy",
            f'{metrics["accuracy"] * 100:.2f}%'
        )

        m2.metric(
            "Precision",
            f'{metrics["precision"] * 100:.2f}%'
        )

        m3.metric(
            "Recall",
            f'{metrics["recall"] * 100:.2f}%'
        )

        m4.metric(
            "F1 Score",
            f'{metrics["f1"] * 100:.2f}%'
        )

    else:
        st.warning(
            "Model metrics are not available. "
            "Train the model first."
        )

    st.subheader("📈 Training Results")

    result_files = [
        "correlation_heatmap.png",
        "loss_curve.png",
        "accuracy_curve.png",
        "confusion_matrix.png",
        "activation_functions.png",
        "manual_nn_loss.png"
    ]

    for name in result_files:

        path = BASE / "results" / name

        if path.exists():

            st.image(
                str(path),
                caption=name.replace(
                    "_", " "
                ).replace(
                    ".png", ""
                ).title(),
                use_container_width=True
            )


# =========================================================
# TAB 3 - PROJECT INFORMATION
# =========================================================

with tab3:

    st.subheader("📚 How the System Works")

    st.markdown("""
    ### AI-Based Student Performance Prediction

    1. Student academic data is collected.

    2. Data is cleaned and normalized using StandardScaler.

    3. NumPy and Pandas are used for numerical processing.

    4. An MLP receives eight input features.

    5. Hidden layers use ReLU activation.

    6. The output layer uses Softmax for four performance classes.

    7. Cross-entropy loss measures classification error.

    8. Adam updates the neural-network weights during training.

    9. Accuracy, Precision, Recall and F1-score evaluate the model.

    10. The dashboard predicts student performance, risk level
        and recommendations.

    ### Performance Classes

    - 🔴 At Risk
    - 🟡 Average
    - 🟢 Good
    - 🔵 Excellent

    ### Input Features

    - Attendance
    - Study Hours
    - Previous Score
    - Assignment Score
    - Internal Assessment
    - Practical Score
    - Participation
    - GPA

    **Important:** This is an educational prediction tool.
    It should support, not replace, teacher judgment and
    student support processes.
    """)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Student Performance Prediction and Early Academic Risk Detection System"
)