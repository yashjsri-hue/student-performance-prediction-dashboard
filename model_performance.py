# ==========================================================
# Student Performance Prediction Dashboard
# model_performance.py (Part 1A)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split
import plotly.express as px

# ----------------------------------------------------------
# Project Paths
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_performance_model.pkl"

# ----------------------------------------------------------
# Validate Model File
# ----------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "❌ Model file not found. Please verify the model path."
    )
    st.stop()

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(
            f"Unable to load model.\n\n{e}"
        )
        st.stop()

# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------
def pretty_name(column):
    return (
        column
        .replace("_", " ")
        .title()
    )

# ----------------------------------------------------------
# Model Performance Page
# ----------------------------------------------------------
def show_model_performance(df):
    model = load_model()
    st.title("🤖 Model Performance")
    st.markdown(
        """
        Evaluate the performance of the trained
        **Student Performance Prediction Model**.
        """
    )
    st.markdown("---")

    # ----------------------------------------------------------
    # Part 1D : Model Information
    # ----------------------------------------------------------
    st.header("ℹ️ Model Information")
    # Basic model details
    algorithm_name = type(model).__name__
    model_type = "Classification"
    # Number of input features
    if hasattr(model, "feature_names_in_"):
        num_features = len(model.feature_names_in_)
        feature_list = list(model.feature_names_in_)
    else:
        num_features = df.shape[1] - 1
        feature_list = list(df.drop(columns=["pass_fail"]).columns)
    target_variable = "pass_fail"
    training_records = len(df)
    num_classes = df[target_variable].nunique()
    # Display information
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Algorithm:** {algorithm_name}")
        st.info(f"**Model Type:** {model_type}")
        st.info(f"**Input Features:** {num_features}")
    with col2:
        st.info(f"**Target Variable:** {target_variable}")
        st.info(f"**Training Records:** {training_records:,}")
        st.info(f"**Number of Classes:** {num_classes}")
    # Expandable feature list
    with st.expander("📋 View Model Features"):
        feature_df = pd.DataFrame({
            "Feature No.": range(1, len(feature_list) + 1),
            "Feature Name": [pretty_name(f) for f in feature_list]
        })
        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )
    st.markdown("---")

    # ----------------------------------------------------------
    # Part 1B : Model Evaluation Summary
    # ----------------------------------------------------------
    st.markdown("---")
    st.header("📊 Model Evaluation Summary")
    with st.spinner("🔄 Evaluating model performance..."):
        # Identify target column
        target_column = "pass_fail"
        if target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found.")
            st.stop()
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        # Ensure feature order matches the trained model
        if hasattr(model, "feature_names_in_"):
            missing_features = [
                feature
                for feature in model.feature_names_in_
                if feature not in X.columns
            ]
            if missing_features:
                st.error(f"Missing required model features: {missing_features}")
                st.stop()
            X = X[model.feature_names_in_]
        # Create one train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
        # Predictions
        y_pred = model.predict(X_test)
        # Calculate ROC-AUC (only if supported)
        roc_auc = None
        if (
            hasattr(model, "predict_proba")
            and len(np.unique(y)) == 2
        ):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            except Exception:
                roc_auc = None
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
        # Display KPIs
        metric1, metric2, metric3, metric4, metric5 = st.columns(5)
        with metric1:
            st.metric(
                "🎯 Accuracy",
                f"{accuracy * 100:.2f}%"
            )
        with metric2:
            st.metric(
                "🎯 Precision",
                f"{precision * 100:.2f}%"
            )
        with metric3:
            st.metric(
                "🎯 Recall",
                f"{recall * 100:.2f}%"
            )
        with metric4:
            st.metric(
                "🎯 F1 Score",
                f"{f1 * 100:.2f}%"
            )
        with metric5:
            if roc_auc is not None:
                st.metric(
                    "🎯 ROC-AUC",
                    f"{roc_auc:.3f}"
                )
            else:
                st.metric(
                    "🎯 ROC-AUC",
                    "N/A"
                )
        if roc_auc is not None:
            st.markdown("---")
            st.subheader("📈 ROC-AUC Interpretation")
            if roc_auc >= 0.90:
                st.success(
                    "Outstanding discrimination between pass and fail classes."
                )
            elif roc_auc >= 0.80:
                st.info(
                    "Very good class discrimination."
                )
            elif roc_auc >= 0.70:
                st.warning(
                    "Acceptable discrimination with room for improvement."
                )
            else:
                st.error(
                    "Poor discrimination. Consider improving the model."
                )
        # Interpretation
        st.markdown("---")
        st.subheader("📖 Performance Interpretation")
        if accuracy >= 0.90:
            st.success(
                "Excellent model performance with very high prediction accuracy."
            )
        elif accuracy >= 0.80:
            st.info(
                "Good model performance suitable for most prediction tasks."
            )
        elif accuracy >= 0.70:
            st.warning(
                "Moderate model performance. Further improvement is possible."
            )
        else:
            st.error(
                "Low prediction accuracy. Consider improving the model."
            )
        st.caption(
         f"Evaluation performed on {len(df):,} student records."
        )

        # ==========================================================
        # Part 2 - Classification Performance Analysis
        # ==========================================================
        st.markdown("---")
        st.header("📊 Classification Performance Analysis")
        
        # ----------------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------------
        st.subheader("🔲 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Fail", "Actual Pass"],
            columns=["Predicted Fail", "Predicted Pass"]
        )
        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            title="Confusion Matrix"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------
        # Classification Report
        # ----------------------------------------------------------
        st.subheader("📋 Classification Report")
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(
            report_df,
            use_container_width=True
        )

        # ----------------------------------------------------------
        # Prediction Distribution
        # ----------------------------------------------------------
        st.subheader("📈 Prediction Distribution")
        prediction_df = pd.DataFrame({
            "Prediction": pd.Series(y_pred).map({
                0: "Fail",
                1: "Pass"
            })
        })
        fig = px.histogram(
            prediction_df,
            x="Prediction",
            color="Prediction",
            title="Predicted Class Distribution",
            text_auto=True
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------
        # Feature Importance
        # ----------------------------------------------------------
        st.subheader("⭐ Feature Importance (Logistic Regression)")
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        })
        coef_df["Absolute"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Coefficient")
        fig = px.bar(
            coef_df,
            x="Coefficient",
            y="Feature",
            color="Coefficient",
            orientation="h",
            title="Logistic Regression Feature Coefficients",
            text_auto=".3f"
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------
        # Prediction Samples
        # ----------------------------------------------------------
        st.subheader("📝 Sample Predictions")
        sample_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })
        sample_df["Status"] = sample_df.apply(
            lambda row: "✅ Correct"
            if row["Actual"] == row["Predicted"]
            else "❌ Incorrect",
            axis=1
        )
        st.dataframe(
            sample_df.head(20),
            use_container_width=True
        )
        st.success("Classification performance analysis completed successfully.")

        # ==========================================================
        # Part 3A - ROC Curve Analysis
        # ==========================================================
        st.markdown("---")
        st.header("📈 ROC Curve Analysis")
        if (
            hasattr(model, "predict_proba")
            and roc_auc is not None
        ):
            st.subheader("Receiver Operating Characteristic (ROC) Curve")
            # Probability prediction
            y_probability = model.predict_proba(X_test)[:, 1]
            # Calculate ROC values
            fpr, tpr, thresholds = roc_curve(
                y_test,
                y_probability
            )
            # Create ROC dataframe
            roc_df = pd.DataFrame({
                "False Positive Rate": fpr,
                "True Positive Rate": tpr
            })
            # ROC Curve Plot
            fig = px.line(
                roc_df,
                x="False Positive Rate",
                y="True Positive Rate",
                title="ROC Curve"
            )
            # Add random classifier line
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    dash="dash"
                )
            )
            fig.update_layout(
                height=550,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )
            # Display AUC score
            st.metric(
                "ROC-AUC Score",
                f"{roc_auc:.3f}"
            )
            # Interpretation
            if roc_auc >= 0.90:
                st.success(
                    "Excellent model capability to separate Pass and Fail students."
                )
            elif roc_auc >= 0.80:
                st.info(
                    "Very good classification capability."
                )
            elif roc_auc >= 0.70:
                st.warning(
                    "Acceptable classification capability."
                )
            else:
                st.error(
                    "Model discrimination ability is weak."
                )
        else:
            st.warning(
                "ROC Curve cannot be generated because probability prediction is unavailable."
            )

        # ==========================================================
        # Part 3B - Precision Recall Curve Analysis
        # ==========================================================
        st.markdown("---")
        st.header("📊 Precision-Recall Curve Analysis")
        if (
            hasattr(model, "predict_proba")
            and roc_auc is not None
        ):
            st.subheader(
                "Precision vs Recall Trade-off"
            )
            # Probability prediction
            y_probability = model.predict_proba(X_test)[:, 1]
            # Calculate Precision Recall values
            precision_values, recall_values, thresholds = precision_recall_curve(
                y_test,
                y_probability
            )
            # Average Precision Score
            avg_precision = average_precision_score(
                y_test,
                y_probability
            )
            # Create dataframe
            pr_df = pd.DataFrame({
                "Recall": recall_values,
                "Precision": precision_values
            })
            # Plot Precision Recall Curve
            fig = px.line(
                pr_df,
                x="Recall",
                y="Precision",
                title="Precision-Recall Curve"
            )
            fig.update_layout(
                height=550,
                xaxis_title="Recall",
                yaxis_title="Precision"
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )
            # Display Average Precision
            st.metric(
                "Average Precision Score",
                f"{avg_precision:.3f}"
            )
            # Interpretation
            if avg_precision >= 0.90:
                st.success(
                    "Excellent precision and recall balance."
                )
            elif avg_precision >= 0.80:
                st.info(
                    "Very good balance between identifying pass students and avoiding false predictions."
                )
            elif avg_precision >= 0.70:
                st.warning(
                    "Acceptable performance with possible improvement opportunities."
                )
            else:
                st.error(
                    "Low precision-recall performance."
                )
        else:
            st.warning(
                "Precision-Recall Curve cannot be generated because probability prediction is unavailable."
            )

        # ==========================================================
        # Part 3C - Download Evaluation Report
        # ==========================================================
        st.markdown("---")
        st.header("📥 Download Evaluation Report")
        st.write(
            "Download a summary of the model evaluation metrics "
            "for documentation and reporting."
        )
        # Create evaluation summary dataframe
        evaluation_report = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            "Score": [
                round(accuracy, 4),
                round(precision, 4),
                round(recall, 4),
                round(f1, 4),
                round(roc_auc, 4) if roc_auc is not None else "N/A"
            ]
        })
        # Display evaluation table
        st.subheader("📋 Evaluation Summary")
        st.dataframe(
            evaluation_report,
            use_container_width=True,
            hide_index=True
        )
        # Convert dataframe to CSV
        csv_report = evaluation_report.to_csv(
            index=False
        ).encode("utf-8")
        # Download button
        st.download_button(
            label="📄 Download Evaluation Report (CSV)",
            data=csv_report,
            file_name="student_model_evaluation_report.csv",
            mime="text/csv"
        )
        st.success(
            "Evaluation report is ready for download."
        )

        # ==========================================================
        # Part 3D - Professional UI Enhancements & Footer
        # ==========================================================
        st.markdown("---")
        st.header("📌 Model Evaluation Summary")
        roc_auc_display = (
            f"{roc_auc:.3f}"
            if roc_auc is not None
            else "N/A"
        )
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.info(
                """
                **🎯 Overall Performance**
                The Logistic Regression model demonstrates strong classification performance on the Student Performance Prediction dataset. The evaluation metrics indicate that the model iswell-balanced for predicting student pass/fail outcomes.
                """
            )
        with summary_col2:
            st.success(
                f"""
                **📈 Evaluation Highlights**
                • Accuracy : {accuracy*100:.2f}%
                • Precision : {precision*100:.2f}%
                • Recall : {recall*100:.2f}%
                • F1 Score : {f1*100:.2f}%
                • ROC-AUC : {roc_auc_display}
                """
            )
        st.markdown("---")
        st.subheader("💡 Key Insights")
        insight1, insight2, insight3 = st.columns(3)
        with insight1:
            st.success(
                """
                **Reliable Predictions**
                The model correctly classifies the majority of students based on their academic performance.
                """
            )
        with insight2:
            st.info(
                """
                **Balanced Performance**
                Precision and Recall values indicate a balanced classifier with minimal prediction bias.
                """
            )
        with insight3:
            st.warning(
                """
                **Future Improvements**
                Model performance can be further improved by adding more relevant student-related features.
                """
            )
        st.markdown("---")
        st.subheader("📖 About this Evaluation")
        st.write(
            """
            This dashboard evaluates the trained Logistic Regression model using multiple classification metrics including:
            - Accuracy
            - Precision
            - Recall
            - F1 Score
            - ROC-AUC
            - Confusion Matrix
            - Classification Report
            - ROC Curve
            - Precision-Recall Curve
            These metrics provide a comprehensive understanding of the model's predictive capability and classification performance.
            """
        )
        st.markdown("---")
        st.caption(
            "──────────────────────────────────────────────"
        )
        footer_col1, footer_col2 = st.columns([3, 2])
        with footer_col1:
            st.caption(
                """
            🎓 Student Performance Prediction Dashboard
        Machine Learning Classification Project
        Algorithm: Logistic Regression
        """
            )
        with footer_col2:
            st.caption(
                """
        Version : 1.0 Developed using:
        • Streamlit
        • Scikit-learn
        • Plotly
        • Pandas
        """
            )
        st.caption(
            "© 2026 Student Performance Prediction Dashboard"
        )



    
