"""
=========================================================
Student Performance Prediction Dashboard
Home Page
=========================================================
"""
import pandas as pd
import streamlit as st

def show_home(df: pd.DataFrame):
    """
    Displays the Home page of the dashboard.
    """

    # --------------------------------------------------
    # Title
    # --------------------------------------------------
    st.title("🎓 Student Performance Prediction Dashboard")
    st.markdown("---")

    # --------------------------------------------------
    # Project Overview
    # --------------------------------------------------
    st.header("📌 Project Overview")
    st.write(
        """
        The **Student Performance Prediction Dashboard** is a Machine Learning
        application that predicts whether a student is likely to perform well
        based on academic, demographic, and lifestyle factors.

        The project assists educators and institutions in identifying students
        who may require additional academic support, enabling timely and
        data-driven interventions.
        """
    )
    st.markdown("---")

    # --------------------------------------------------
    # Dataset Summary
    # --------------------------------------------------
    st.header("📊 Dataset Summary")
    total_students = df.shape[0]
    total_features = df.shape[1]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Students", f"{total_students:,}")
    with col2:
        st.metric("Total Features", total_features)
    st.markdown("---")

    # --------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------
    st.header("👀 Dataset Preview")
    st.dataframe(df, use_container_width=True)
    st.markdown("---")

    # --------------------------------------------------
    # Model Information
    # --------------------------------------------------
    st.header("🤖 Machine Learning Model")
    model_info = pd.DataFrame(
        {
            "Property": [
                "Algorithm",
                "Problem Type",
                "Target Variable",
                "Programming Language",
                "Framework",
            ],
            "Value": [
                "Random Forest Classifier",
                "Classification",
                "Pass / Fail",
                "Python",
                "Scikit-learn",
            ],
        }
    )
    st.table(model_info)
    st.markdown("---")

    # --------------------------------------------------
    # Dashboard Workflow
    # --------------------------------------------------
    st.header("⚙️ Project Workflow")
    workflow = [
        "✅ Data Collection",
        "✅ Data Preprocessing",
        "✅ Exploratory Data Analysis (EDA)",
        "✅ Model Building (Random Forest Classifier)",
        "✅ Interactive Dashboard",
        "✅ Student Performance Prediction",
    ]
    for step in workflow:
        st.write(step)
    st.markdown("---")

    # --------------------------------------------------
    # Key Features
    # --------------------------------------------------
    st.header("✨ Dashboard Features")
    features = [
        "📈 Interactive visualizations",
        "🎯 Student performance prediction",
        "📊 Dataset exploration",
        "🤖 Model performance evaluation",
        "📥 Download prediction results",
        "💻 User-friendly Streamlit interface",
    ]
    for feature in features:
        st.write(feature)
    st.markdown("---")

    # --------------------------------------------------
    # Quick Statistics
    # --------------------------------------------------
    st.header("📋 Quick Statistics")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        st.dataframe(
            numeric_df.describe().T,
            use_container_width=True,
        )
    st.markdown("---")

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------
    st.success("✅ Dashboard Loaded Successfully!")
    st.caption(
        "Student Performance Prediction Dashboard | "
        "Developed using Python, Streamlit & Scikit-learn"
    )
