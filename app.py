# ==========================================================
# Student Performance Prediction Dashboard
# app.py (Main Application Controller)
# ==========================================================
import streamlit as st
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------------------------------
# Import Modules
# ----------------------------------------------------------
from home import show_home
from dataset_overview import show_dataset_overview
from eda_dashboard import show_eda_dashboard
from prediction import show_prediction
from model_performance import show_model_performance
from about import show_about

# ----------------------------------------------------------
# Base Directory
# ----------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "processed_student_performance_data.csv"

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df
df = load_data()

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        color:#1f77b4;
        text-align:center;
    }
    .footer {
        text-align:center;
        color:grey;
        font-size:14px;
        margin-top:50px;

    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------
# Sidebar Navigation
# ----------------------------------------------------------
st.sidebar.title("🎓 Student Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📋 Dataset Overview",
        "📊 EDA Dashboard",
        "🔮 Prediction",
        "🤖 Model Performance",
        "ℹ️ About"
    ]
)

# ----------------------------------------------------------
# Page Routing
# ----------------------------------------------------------
if page == "🏠 Home":
    show_home(df)
elif page == "📋 Dataset Overview":
    show_dataset_overview(df)
elif page == "📊 EDA Dashboard":
    show_eda_dashboard(df)
elif page == "🔮 Prediction":
    show_prediction(df)
elif page == "🤖 Model Performance":
    show_model_performance(df)
elif page == "ℹ️ About":
    show_about(df)
    
# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------
st.markdown(
    """
    <div class="footer">
    Developed using Python | Streamlit | Machine Learning<br>
    Student Performance Prediction System © 2026
    </div>
    """,
    unsafe_allow_html=True
)
