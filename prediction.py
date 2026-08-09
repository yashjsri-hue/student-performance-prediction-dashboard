# ==========================================================
# Student Performance Prediction Dashboard
# prediction.py (Part 1A)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------------------------------------
# Project Paths
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "student_performance_model.pkl"

# ----------------------------------------------------------
# Part 2E : Model File Validation
# ----------------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        "❌ Model file not found. Please check model path."
    )
    st.stop()

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------
@st.cache_resource
def load_prediction_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Unable to load model.\n\n{e}")
        st.stop()


# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------
def get_default(feature_name):
    """
    Default value for every feature.
    """
    defaults = {
        "age":18,
        "gender":0,
        "study_hours":4.0,
        "attendance":85,
        "sleep_hours":7,
        "previous_grade":75,
        "assignments_completed":8,
        "practice_tests_taken":5,
        "group_study_hours":2,
        "notes_quality_score":7,
        "time_management_score":7,
        "motivation_level":7,
        "mental_health_score":7,
        "screen_time":3,
        "social_media_hours":2,
        "family_income":50000,
        "parent_education":1,
        "internet_access":1,
        "device_type":0,
        "school_type":0,
        "extracurriculars":1,
        "final_grade":75
    }
    return defaults.get(feature_name,0)
def pretty_name(column):
    return (
        column
        .replace("_"," ")
        .title()
    )

# ----------------------------------------------------------
# Part 2D : PDF Report Generator
# ----------------------------------------------------------

def generate_pdf_report(
        prediction,
        confidence,
        input_data
):
    file_name = "student_prediction_report.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    content = []
    title = Paragraph(
        "Student Performance Prediction Report",
        styles["Title"]
    )
    content.append(title)
    content.append(
        Spacer(1, 20)
    )
    content.append(
        Paragraph(
            f"Generated Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        )
    )
    content.append(
        Spacer(1, 15)
    )
    result = "PASS" if prediction == 0 else "FAIL"
    content.append(
        Paragraph(
            f"Prediction Result: {result}",
            styles["Heading2"]
        )
    )
    if confidence:
        content.append(
            Paragraph(
                f"Confidence Score: {confidence:.2f}%",
                styles["Normal"]
            )
        )
    content.append(
        Spacer(1, 20)
    )
    content.append(
        Paragraph(
            "Student Details",
            styles["Heading2"]
        )
    )
    table_data = [
        ["Feature", "Value"]
    ]
    for key, value in input_data.items():
        table_data.append(
            [
                pretty_name(key),
                str(value)
            ]
        )
    table = Table(table_data)
    table.setStyle(
        TableStyle(
            [
                ("GRID",(0,0),(-1,-1),0.5,None)
            ]
        )
    )
    content.append(table)
    doc.build(content)
    return file_name

# ----------------------------------------------------------
# Prediction Page
# ----------------------------------------------------------
def show_prediction(df):
    #st.success("✅ show_prediction() is running")
    #st.title("🎯 Student Performance Prediction")
    model = load_prediction_model()
    # ----------------------------------------------------------
    # Part 2B : Prediction History Initialization
    # ----------------------------------------------------------
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    input_data = {}
    st.title("🎯 Student Performance Prediction")
    # ----------------------------------------------------------
    # Part 2E : Model Feature Compatibility Check
    # ----------------------------------------------------------
    feature_names = list(model.feature_names_in_)
    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]
    if missing_features:
        st.error(
            f"Missing required features: {missing_features}"
        )
        st.stop()
    input_data = {}
    # ----------------------------------------------------------
    # Professional UI Styling
    # ----------------------------------------------------------
    st.markdown(
        """
        <style>
        .prediction-card {
            padding: 20px;
            border-radius: 12px;
            background-color: #f8f9fa;
            margin-bottom: 15px;
        }
        .footer {
            text-align:center;
            color:gray;
            font-size:14px;
            margin-top:40px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
    """
    Predict a student's performance using the trained
    **Random Forest Classifier**.
    Enter the student's academic and personal information,
    then click **Predict**.
    """
    )
    st.markdown("---")
    st.subheader("📝 Student Information")
    #feature_names = list(model.feature_names_in_)
    input_data = {}
    col1,col2 = st.columns(2)
    # ----------------------------------------------------------
    # Dynamic Input Fields
    # ----------------------------------------------------------
    for i, feature in enumerate(feature_names):
        current_col = col1 if i % 2 == 0 else col2
        with current_col:
            label = pretty_name(feature)
            default = get_default(feature)
            # -------------------------------
            # Numerical Features
            # -------------------------------
            if feature == "age":
                input_data[feature] = st.number_input(
                    label,
                    min_value=10,
                    max_value=35,
                    value=int(default),
                    step=1
                )
            elif feature == "study_hours":
                input_data[feature] = st.slider(
                    label,
                    min_value=0.0,
                    max_value=12.0,
                    value=float(default),
                    step=0.5
                )
            elif feature == "attendance":
                input_data[feature] = st.slider(
                    label,
                    min_value=0,
                    max_value=100,
                    value=int(default)
                )
            elif feature in [
                "sleep_hours",
                "assignments_completed",
                "practice_tests_taken",
                "group_study_hours",
                "screen_time",
                "social_media_hours"
            ]:
                input_data[feature] = st.slider(
                    label,
                    min_value=0,
                    max_value=12,
                    value=int(default)
                )
            elif feature in [
                "previous_grade",
                "notes_quality_score",
                "time_management_score",
                "motivation_level",
                "mental_health_score"
            ]:
                max_val = 100 if feature == "previous_grade" else 10
                input_data[feature] = st.slider(
                    label,
                    min_value=0,
                    max_value=max_val,
                    value=int(default)
                )
            elif feature == "family_income":
                input_data[feature] = st.number_input(
                    label,
                    min_value=0,
                    value=int(default),
                    step=1000
                )
            # -------------------------------
            # Binary Features
            # -------------------------------
            elif feature in [
                "internet_access",
                "extracurriculars"
            ]:
                option = st.selectbox(
                    label,
                    ["No", "Yes"],
                    index=int(default)
                )
                input_data[feature] = 1 if option == "Yes" else 0
            # -------------------------------
            # Gender
            # -------------------------------
            elif feature == "gender":
                gender = st.selectbox(
                    label,
                    ["Male", "Female"],
                    index=int(default)
                )
                input_data[feature] = 0 if gender == "Male" else 1
            # -------------------------------
            # Parent Education
            # -------------------------------
            elif feature == "parent_education":
                education_levels = {
                    "High School": 0,
                    "Graduate": 1,
                    "Post Graduate": 2,
                    "Doctorate": 3
                }
                selected = st.selectbox(
                    label,
                    list(education_levels.keys()),
                    index=min(int(default), len(education_levels) - 1)
                )
                input_data[feature] = education_levels[selected]
            # -------------------------------
            # Device Type
            # -------------------------------
            elif feature == "device_type":
                devices = {
                    "Desktop": 0,
                    "Laptop": 1,
                    "Tablet": 2,
                    "Mobile": 3
                }
                selected = st.selectbox(
                    label,
                    list(devices.keys())
                )
                input_data[feature] = devices[selected]
            # -------------------------------
            # School Type
            # -------------------------------
            elif feature == "school_type":
                schools = {
                    "Government": 0,
                    "Private": 1
                }
                selected = st.selectbox(
                    label,
                    list(schools.keys()),
                    index=int(default)
                )
                input_data[feature] = schools[selected]
            # -------------------------------
            # Any Remaining Numeric Feature
            # -------------------------------
            else:
                input_data[feature] = st.number_input(
                    label,
                    value=float(default)
                )
    st.markdown("---")
    st.subheader("📋 Input Summary")
    summary_df = pd.DataFrame({
        "Feature": [pretty_name(col) for col in input_data.keys()],
        "Value": list(input_data.values())
    })
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )
    st.markdown("---")
    # ==========================================================
    # Prediction Engine
    # ==========================================================
    predict_col, clear_col = st.columns([3, 1])
    with predict_col:
        predict_btn = st.button(
            "🎯 Predict Student Performance",
            use_container_width=True,
            type="primary"
        )
    with clear_col:
        reset_btn = st.button(
            "🔄 Reset",
            use_container_width=True
        )
    # ----------------------------------------------------------
    # Part 2E : Reset Button Functionality
    # ----------------------------------------------------------
    if reset_btn:
        # Clear all session state except prediction history
        for key in list(st.session_state.keys()):
            if key != "prediction_history":
                del st.session_state[key]
        st.rerun()
    if predict_btn:
        with st.spinner("🔄 Analyzing student performance..."):
            #st.write("Button clicked")
            try:
                # -----------------------------------------
                # PART 2A : Input Validation System
                # -----------------------------------------
                validation_errors = []
                if input_data["age"] < 10 or input_data["age"] > 35:
                    validation_errors.append(
                        "Age should be between 10 and 35."
                        )
                if input_data["attendance"] < 0 or input_data["attendance"] > 100:
                    validation_errors.append(
                        "Attendance should be between 0 and 100."
                        )
                if input_data["study_hours"] < 0:
                    validation_errors.append(
                        "Study hours cannot be negative."
                        )
                if input_data["sleep_hours"] < 0:
                    validation_errors.append(
                        "Sleep hours cannot be negative."
                        )
                if validation_errors:
                    st.error("⚠️ Please correct the following errors:")
                    for error in validation_errors:
                        st.write("•", error)
                    st.stop()
                # Create input DataFrame
                input_df = pd.DataFrame([input_data])
                # Match model feature order
                input_df = input_df[feature_names]
                # Make prediction
                prediction = model.predict(input_df)[0]
                #st.write("Prediction:", prediction)
                # ----------------------------------------------------------
                # Confidence score (if available)
                # ----------------------------------------------------------
                confidence = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    confidence = float(np.max(probabilities) * 100)
                # ----------------------------------------------------------
                # Part 2B : Store Prediction History
                # ----------------------------------------------------------
                history_record = {
                    "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "Prediction": "PASS" if prediction == 0 else "FAIL",
                    "Confidence": f"{confidence:.2f}%" if confidence else "N/A"
                }
                st.session_state.prediction_history.append(history_record)
                # Keep only latest 10 predictions
                if len(st.session_state.prediction_history) > 10:
                    st.session_state.prediction_history.pop(0)
                # -----------------------------------------
                # Professional Prediction Display
                # -----------------------------------------
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    if prediction == 0:
                        st.success(
                            """
                            🎉
                        ## PASS
                        Student is predicted to perform successfully.
                        """
                    )
                    else:
                        st.error(
                            """
                            ⚠️
                            ## FAIL
                            Student requires academic improvement.
                            """
                        )
                with result_col2:
                    if confidence is not None:
                        st.metric(
                            "Model Confidence",
                            f"{confidence:.2f}%"
                        )
                    st.metric(
                        "Prediction Class",
                        "PASS" if prediction == 0 else "FAIL"
                    )
                # -----------------------------------------
                # Student Summary
                # -----------------------------------------
                st.markdown("---")
                st.subheader("📋 Student Information Summary")
                summary_df = pd.DataFrame({
                    "Feature": [pretty_name(col) for col in feature_names],
                    "Value": [input_data[col] for col in feature_names]
                })
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )
                # -----------------------------------------
                # Probability Distribution
                # -----------------------------------------
                if hasattr(model, "predict_proba"):
                    st.markdown("---")
                    st.subheader("📈 Class Probability Distribution")
                    probability_df = pd.DataFrame({
                        "Class": model.classes_,
                        "Probability (%)": [
                            round(prob * 100, 2)
                            for prob in probabilities
                            ]
                        })
                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True
                    )
                # -----------------------------------------
                # Prediction Interpretation
                # -----------------------------------------
                st.markdown("---")
                st.subheader("💡 Prediction Interpretation")
                if prediction == 0:
                    st.success("""
                The model predicts that the student is likely to **PASS** based on the
                academic, attendance, and behavioural information provided.
                """)
                else:
                    st.error("""
                The model predicts that the student is at risk of **FAILING**.
                Improving study habits and academic engagement can increase the chances
                of better future performance.
                """)
                # -----------------------------------------
                # Personalized Recommendations
                # -----------------------------------------
                st.subheader("📚 Personalized Recommendations")
                if prediction == 0:
                    st.info("""
                ✅ Maintain regular attendance.
                ✅ Continue your current study routine.
                ✅ Keep completing assignments on time.
                ✅ Practice regularly using mock tests.
                ✅ Maintain a healthy sleep schedule.
                """)
                else:
                    recommendations = []
                    if input_data.get("study_hours", 0) < 4:
                        recommendations.append("• Increase daily study hours.")
                    if input_data.get("attendance", 100) < 75:
                        recommendations.append("• Improve classroom attendance.")
                    if input_data.get("assignments_completed", 10) < 5:
                        recommendations.append("• Complete more assignments.")
                    if input_data.get("practice_tests_taken", 10) < 3:
                        recommendations.append("• Take more practice tests.")
                    if input_data.get("screen_time", 0) > 6:
                        recommendations.append("• Reduce unnecessary screen time.")
                    if input_data.get("sleep_hours", 10) < 6:
                        recommendations.append("• Get sufficient sleep every night.")
                    if not recommendations:
                        recommendations.append("• Continue improving your overall study habits.")
                    st.warning("\n".join(recommendations))
                # -----------------------------------------
                # PART 2C : Explainable AI
                # -----------------------------------------
                st.markdown("---")
                st.subheader("🤖 Why This Prediction?")
                if hasattr(model, "feature_importances_"):
                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance (%)": model.feature_importances_ * 100
                    })
                    importance_df = (
                        importance_df.sort_values(
                            by="Importance (%)",
                            ascending=False
                        )
                        .head(10)
                    )
                    importance_df["Feature"] = (
                        importance_df["Feature"]
                        .apply(pretty_name)
                    )
                    st.markdown(
                        """
                        The prediction is mainly influenced by the following
                        important factors:
                        """
                    )
                    st.dataframe(
                        importance_df.round(2),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.bar_chart(
                        importance_df.set_index("Feature")
                    )
                else:
                    st.info(
                        "Feature importance is not available for this model."
                    )
                # -----------------------------------------
                # Download Report
                # -----------------------------------------
                report_lines = [
                    "STUDENT PERFORMANCE PREDICTION REPORT",
                    "=" * 45,
                    "",
                    f"Generated : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
                    "",
                    f"Prediction : {prediction}"
                ]
                if confidence is not None:
                    report_lines.append(f"Confidence : {confidence:.2f}%")
                report_lines.extend([
                    "",
                    "Student Details",
                    "-" * 25
                ])
                for feature in feature_names:
                    report_lines.append(
                        f"{pretty_name(feature)} : {input_data[feature]}"
                    )
                report_text = "\n".join(report_lines)
                # ----------------------------------------------------------
                # Part 2D : PDF Download
                # ----------------------------------------------------------
                pdf_file = generate_pdf_report(
                    prediction,
                    confidence,
                    input_data
                )
                with open(pdf_file, "rb") as file:
                    st.download_button(
                        "📄 Download PDF Prediction Report",
                        file,
                        file_name="student_prediction_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                # ----------------------------------------------------------
                # Display Prediction History
                # ----------------------------------------------------------
                st.markdown("---")
                st.subheader("📜 Prediction History")
                if len(st.session_state.prediction_history) > 0:
                    history_df = pd.DataFrame(
                        st.session_state.prediction_history
                    )
                    st.dataframe(
                        history_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No prediction history available.")
                # ----------------------------------------------------------
                # Part 2E : Professional Production Footer
                # ----------------------------------------------------------
                st.markdown("---")
                st.markdown(
                    f"""
                    <div class="footer">
                    <h4>🎓 Student Performance Prediction Dashboard</h4>
                    <p>
                    <b>Machine Learning Model:</b> Random Forest Classifier
                    </p>
                    <p>
                    <b>Technologies:</b>
                    Python | Pandas | NumPy | Scikit-learn | Streamlit
                    </p>
                    <p>
                    <b>Report Generated:</b>
                    {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
                    </p>
                    <p style="font-size:13px;">
                    © 2026 MCA Data Science Project
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Prediction failed.\n\n{e}")
