# ==========================================================
# Student Performance Prediction Dashboard
# about.py (Part 1A)
# ==========================================================
import streamlit as st
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------
# Project Configuration
# ----------------------------------------------------------
PROJECT_NAME = "Student Performance Prediction Dashboard"
VERSION = "1.0.0"
ALGORITHM = "Logistic Regression"
TARGET_VARIABLE = "pass_fail"

# ----------------------------------------------------------
# Helper Function
# ----------------------------------------------------------
def pretty_name(text):
    return (
        text
        .replace("_", " ")
        .title()
    )

# ----------------------------------------------------------
# About Page
# ----------------------------------------------------------
def show_about(df):
    st.title("ℹ️ About")
    st.markdown(
        """
Learn more about the **Student Performance Prediction Dashboard**,
its objectives, machine learning model, technologies used,
and the features available within the application.
"""
    )
    st.markdown("---")
    
    # ==========================================================
    # Part 1B - Dataset Information
    # ==========================================================
    st.markdown("---")
    st.header("📊 Dataset Information")
    # Dataset statistics
    num_records = len(df)
    num_columns = df.shape[1]
    target_column = "pass_fail"
    feature_columns = [
        col for col in df.columns
        if col != target_column
    ]
    feature_count = len(feature_columns)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "📄 Total Records",
            f"{num_records:,}"
        )
        st.metric(
            "📑 Total Features",
            feature_count
        )
    with col2:
        st.metric(
            "🎯 Target Variable",
            target_column
        )
        st.metric(
            "📚 Dataset Columns",
            num_columns
        )
    st.markdown("---")
    st.subheader("📝 Dataset Description")
    st.info(
        """
    This dataset contains academic, demographic, and
    student-related information used to predict whether
    a student is likely to **Pass** or **Fail**.
    The dataset was preprocessed before model training,
    including feature selection and preparation for
    classification using Logistic Regression.
    """
    )
    st.subheader("📋 Dataset Features")
    feature_df = pd.DataFrame({
        "Feature No.": range(1, feature_count + 1),
        "Feature Name": [
            feature.replace("_", " ").title()
            for feature in feature_columns
        ]
    })
    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )
    with st.expander("ℹ️ Dataset Summary"):
        st.write(f"**Total Student Records:** {num_records:,}")
        st.write(f"**Input Features:** {feature_count}")
        st.write(f"**Target Variable:** {target_column}")
        st.write(
            "The dataset has been cleaned and prepared for "
            "binary classification using a Logistic Regression model."
        )

    # ==========================================================
    # Student Performance Prediction Dashboard
    # about.py (Part 2)
    # ==========================================================
    def show_about(df):
        # ------------------------------------------------------
        # Page Header
        # ------------------------------------------------------
        st.title("ℹ️ About Project")
        st.markdown(
            """
            ### Student Performance Prediction System
            This dashboard uses Machine Learning techniques to predict
            student performance based on academic and related factors.
            The project demonstrates the complete Machine Learning workflow:
            - Data Collection
            - Data Preprocessing
            - Exploratory Data Analysis
            - Model Training
            - Model Evaluation
            - Performance Prediction
            """
        )
        st.divider()
        # ------------------------------------------------------
        # Part 2A - Dataset Information
        # ------------------------------------------------------
        st.subheader("📊 Dataset Information")
        if df is not None:
            total_records = df.shape[0]
            total_columns = df.shape[1]
            target_column = "pass_fail"
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Records",
                    f"{total_records:,}"
                )
            with col2:
                st.metric(
                    "Total Features",
                    total_columns - 1
                )
            with col3:
                st.metric(
                    "Target Variable",
                    target_column
                )
            st.write("### Dataset Description")
            st.info(
                """
                The dataset contains student academic information used
                for classification of student performance.
                The target variable **pass_fail** represents the final
                performance category predicted by the Machine Learning model.
                """
            )
            # --------------------------------------------------
            # Feature List
            # --------------------------------------------------
            st.write("### Input Features Used for Prediction")
            feature_columns = [
                col for col in df.columns
                if col != target_column
            ]
            feature_df = pd.DataFrame(
                {
                    "Feature Name": feature_columns,
                    "Data Type": [
                        str(df[col].dtype)
                        for col in feature_columns
                    ]
                }
            )
            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )
            # --------------------------------------------------
            # Dataset Preview
            # --------------------------------------------------
            with st.expander(
                "🔍 View Dataset Preview"
            ):
                st.dataframe(
                    df.head(),
                    use_container_width=True
                )
        else:
            st.warning(
                "Dataset not loaded."
            )
        st.divider()
        # ------------------------------------------------------
        # Project Technology Stack
        # ------------------------------------------------------
        st.subheader("🛠Technologies Used")
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.markdown(
                """
                **Programming & Data Processing**
                - Python
                - Pandas
                - NumPy
                - Scikit-learn
                """
            )
        with tech_col2:
            st.markdown(
                """
                **Dashboard & Visualization**
                - Streamlit
                - Plotly
                - Matplotlib
                - Seaborn
                """
            )

        # ==========================================================
        # Part 3 - Author Information + GitHub + Future Improvements
        # ==========================================================
        st.markdown("---")
        # ----------------------------------------------------------
        # Author Information
        # ----------------------------------------------------------
        st.subheader("👨‍💻 Author Information")
        author_col1, author_col2 = st.columns(2)
        with author_col1:
            st.markdown(
                """
                ### Yash Sri
                **Role:**  
                MCA Data Science Student
                **Project Domain:**  
                Machine Learning | Education Analytics
                **Objective:**  
                Developed this Student Performance Prediction Dashboard
                to analyze student academic patterns and predict pass/fail
                outcomes using Machine Learning.
                """
            )
        with author_col2:
            st.markdown(
                """
                ### 🛠 Technical Skills Used
                - Python Programming
                - Pandas & NumPy
                - Data Preprocessing
                - Machine Learning
                - Scikit-Learn
                - Random Forest Classifier
                - Streamlit Dashboard Development
                - Data Visualization
                """
            )
        # ----------------------------------------------------------
        # GitHub Repository
        # ----------------------------------------------------------
        st.subheader("🔗 GitHub Repository")
        github_link = "https://github.com/yashjsri-hue/student-performance-prediction-dashboard"
        st.markdown(
            f"""
            This project source code is available on GitHub.
            You can explore:
            - Complete Python implementation
            - Machine Learning pipeline
            - Dashboard development
            - Dataset preprocessing steps
            - Project documentation
            """
        )
        st.link_button(
            "🚀 View GitHub Repository",
            github_link
        )
        # ----------------------------------------------------------
        # Future Improvements
        # ----------------------------------------------------------
        st.subheader("🚀 Future Improvements")
        future_col1, future_col2 = st.columns(2)
        with future_col1:
            st.markdown(
                """
                ### 🤖 Machine Learning Improvements
                - Test advanced ML algorithms
                - Hyperparameter tuning
                - Model comparison dashboard
                - Cross-validation implementation
                - Feature importance analysis
                """
            )
        with future_col2:
            st.markdown(
                """
                ### 📊 Dashboard Improvements
                - Add student progress tracking
                - Add interactive reports
                - Add PDF prediction reports
                - Deploy using cloud platforms
                - Add authentication system
                """
            )
        # ----------------------------------------------------------
        # Professional Footer
        # ----------------------------------------------------------
        st.markdown("---")
        st.markdown(
            """
            <div style="
                text-align:center;
                padding:15px;
                background-color:#f0f2f6;
                border-radius:10px;
            ">
            <h4>🎓 Student Performance Prediction Dashboard</h4>
            <p>
            Developed using Python, Machine Learning and Streamlit
            </p>
            <p>
            © 2026 Yash Sri | MCA Data Science Project
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ==========================================================
        # Student Performance Prediction Dashboard
        # about.py (Part 4 - Final Professional UI Upgrade)
        # ==========================================================
        def show_about():
            # ------------------------------------------------------
            # Professional CSS Styling
            # ------------------------------------------------------
            st.markdown(
                """
                <style>
                .about-header {
                    background: linear-gradient(
                        90deg,
                        #667eea,
                        #764ba2
                    );
                    padding: 25px;
                    border-radius: 15px;
                    color: white;
                    text-align: center;
                    margin-bottom: 25px;
                }
                .card {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 15px;
                    border-left: 5px solid #667eea;
                    margin-bottom: 20px;
                }
                .section-title {
                    color: #667eea;
                    font-size: 24px;
                    font-weight: bold;
                }
                .feature-box {
                    background-color: #ffffff;
                    padding: 15px;
                    border-radius: 12px;
                    border: 1px solid #ddd;
                    text-align: center;
                }
                .footer {
                    margin-top: 40px;
                    padding: 15px;
                    text-align: center;
                    color: gray;
                    font-size: 14px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            # ------------------------------------------------------
            # Header
            # ------------------------------------------------------
            st.markdown(
                """
                <div class="about-header">
                <h1>🎓 Student Performance Prediction Dashboard</h1>
                <p>
                Machine Learning Based Academic Performance Analysis System
                </p>
                </div>

                """,
                unsafe_allow_html=True
            )
            # ------------------------------------------------------
            # Project Overview
            # ------------------------------------------------------
            st.markdown(
                '<div class="section-title">📌 Project Overview</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <div class="card">
                This project predicts student academic performance using
                Machine Learning techniques.
                The dashboard provides prediction capabilities,
                dataset analysis, model evaluation and performance insights
                through an interactive Streamlit interface.
                The objective is to identify factors influencing student
                success and support data-driven educational decisions.
                </div>
                """,
                unsafe_allow_html=True
            )
            # ------------------------------------------------------
            # Technology Stack
            # ------------------------------------------------------
            st.markdown(
                '<div class="section-title">🛠 Technology Stack</div>',
                unsafe_allow_html=True
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    """
                    <div class="feature-box">
                    <h3>🐍 Programming</h3>
                    Python<br>
                    Pandas<br>
                    NumPy
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    """
                    <div class="feature-box">
                    <h3>🤖 Machine Learning</h3>
                    Scikit-Learn<br>
                    Random Forest<br>
                    Model Evaluation
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    """
                    <div class="feature-box">
                    <h3>📊 Dashboard</h3>
                    Streamlit<br>
                    Plotly<br>
                    Interactive UI
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # ------------------------------------------------------
            # ML Workflow
            # ------------------------------------------------------
            st.markdown(
                '<div class="section-title">🔄 Machine Learning Workflow</div>',
                unsafe_allow_html=True
            )
            workflow = [
                "1️⃣ Data Collection",
                "2️⃣ Data Cleaning & Preprocessing",
                "3️⃣ Exploratory Data Analysis",
                "4️⃣ Feature Engineering",
                "5️⃣ Model Training",
                "6️⃣ Model Evaluation",
                "7️⃣ Student Performance Prediction"
            ]
            for step in workflow:
                st.info(step)
            # ------------------------------------------------------
            # Model Information
            # ------------------------------------------------------
            st.markdown(
                '<div class="section-title">🤖 Model Information</div>',
                unsafe_allow_html=True
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    """
                    <div class="card">
                    <b>Algorithm:</b><br>
                    Random Forest Classifier
                    <br><br>
                    <b>Learning Type:</b><br>
                    Supervised Machine Learning
                    </div>

                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    """
                    <div class="card">
                    <b>Prediction Target:</b><br>
                    Student Pass / Fail
                    <br><br>
                    <b>Application:</b><br>
                    Academic Performance Analysis
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # ------------------------------------------------------
            # Dashboard Features
            # ------------------------------------------------------
            st.markdown(
                '<div class="section-title">✨ Dashboard Features</div>',
                unsafe_allow_html=True
            )
            features = [
                "📈 Student Performance Prediction",
                "📊 Dataset Overview",
                "🔍 Exploratory Data Analysis",
                "🤖 Model Performance Evaluation",
                "📥 Download Evaluation Reports",
                "🎯 Data-driven Academic Insights"
            ]
            for feature in features:
                st.success(feature)
            # ------------------------------------------------------
            # Developer Information
            # ------------------------------------------------------
            st.markdown(
                '<div class="section-title">👨‍💻 Project Information</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <div class="card">
                <b>Project:</b>
                Student Performance Prediction Dashboard
                <br><br>
                <b>Domain:</b>
                Data Science & Machine Learning
                <br><br>
                <b>Platform:</b>
                Streamlit Web Application
                </div>
                """,
                unsafe_allow_html=True
            )
            # ------------------------------------------------------
            # Footer
            # ------------------------------------------------------
            st.markdown(
                """
                <div class="footer">
                © 2026 Student Performance Prediction Dashboard
                <br>
                Built using Python, Machine Learning & Streamlit
                </div>
                """,
                unsafe_allow_html=True
            )
