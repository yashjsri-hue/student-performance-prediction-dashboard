# ==========================================================
# Student Performance Prediction Dashboard
# eda_dashboard.py
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px


# ----------------------------------------------------------
# EDA Dashboard
# ----------------------------------------------------------
def show_eda_dashboard(df: pd.DataFrame):

    st.title("📊 Exploratory Data Analysis")

    st.markdown(
        """
        Explore the student performance dataset through
        interactive visualizations.
        """
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Select Numerical Feature
    # ------------------------------------------------------
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    selected_feature = st.selectbox(
        "Select Numerical Feature",
        numerical_columns
    )

    # ------------------------------------------------------
    # Histogram
    # ------------------------------------------------------
    st.subheader("📈 Distribution")

    fig = px.histogram(
        df,
        x=selected_feature,
        nbins=30,
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Box Plot
    # ------------------------------------------------------
    st.subheader("📦 Box Plot")

    fig = px.box(
        df,
        y=selected_feature,
        title=f"Box Plot of {selected_feature}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Correlation Heatmap
    # ------------------------------------------------------
    st.subheader("🔥 Correlation Heatmap")

    corr = df[numerical_columns].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Scatter Plot
    # ------------------------------------------------------
    st.subheader("📉 Scatter Plot")

    x_axis = st.selectbox(
        "Select X-axis",
        numerical_columns,
        index=0
    )

    y_axis = st.selectbox(
        "Select Y-axis",
        numerical_columns,
        index=1
    )

    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        title=f"{x_axis} vs {y_axis}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Correlation with Target
    # ------------------------------------------------------
    possible_targets = [
        "pass_fail",
        "grade_category",
        "exam_score",
        "final_score",
        "score"
    ]

    target_column = None

    for col in possible_targets:
        if col in df.columns:
            target_column = col
            break

    if target_column:

        st.subheader("🎯 Correlation with Target")

        correlation = (
            df[numerical_columns]
            .corr()[target_column]
            .sort_values(ascending=False)
            .reset_index()
        )

        correlation.columns = [
            "Feature",
            "Correlation"
        ]

        fig = px.bar(
            correlation,
            x="Feature",
            y="Correlation",
            title=f"Correlation with {target_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------
    st.subheader("📋 Dataset Statistics")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    st.success("EDA Dashboard Loaded Successfully.")

