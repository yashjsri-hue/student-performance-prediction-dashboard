# ==========================================================
# Student Performance Prediction Dashboard
# dataset_overview.py
# ==========================================================

import streamlit as st
import pandas as pd


# ----------------------------------------------------------
# Dataset Overview Page
# ----------------------------------------------------------
def show_dataset_overview(df: pd.DataFrame):

    st.title("📈 Dataset Overview")

    st.markdown(
        """
        Explore the processed student performance dataset.
        This page provides an overview of the dataset structure,
        quality, and statistical information.
        """
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Dataset Metrics
    # ------------------------------------------------------
    st.subheader("📊 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        memory = (
            df.memory_usage(deep=True).sum()
            / (1024 ** 2)
        )

        st.metric(
            "Memory Usage",
            f"{memory:.2f} MB"
        )

    st.markdown("---")

    # ------------------------------------------------------
    # Dataset Preview
    # ------------------------------------------------------
    st.subheader("👀 Dataset Preview")

    #rows = st.slider(
    #    "Number of rows",
    #    min_value=5,
    #    max_value=50,
    #    value=10
    #)

    st.dataframe(
        df,
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Column Information
    # ------------------------------------------------------
    st.subheader("📋 Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Statistical Summary
    # ------------------------------------------------------
    st.subheader("📑 Statistical Summary")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Missing Values
    # ------------------------------------------------------
    st.subheader("⚠️ Missing Values")

    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Percentage (%)":
        (
            df.isnull().sum() / len(df) * 100
        ).round(2)
    })

    st.dataframe(
        missing,
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Download Dataset
    # ------------------------------------------------------
    st.subheader("⬇️ Download Processed Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="processed_student_performance_data.csv",
        mime="text/csv"
    )

    st.success("Dataset overview loaded successfully.")
