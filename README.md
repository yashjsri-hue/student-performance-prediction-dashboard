
### 🎓 Student Performance Prediction Dashboard

## 📌 Project Overview
-> Student Performance Prediction Dashboard is an interactive Machine Learning application developed to predict whether a student is likely to Pass or Fail based on academic and behavioral attributes.
-> The project uses a Logistic Regression Classification Model to analyze student-related factors and provide data-driven predictions. The dashboard enables users to explore student data, understand model performance, and generate predictions through an intuitive Streamlit interface.
-> This project demonstrates the complete Machine Learning lifecycle, including data preprocessing, model training, evaluation, and deployment using an interactive dashboard.

## 🎯 Business Problem
-> Educational institutions need effective methods to identify students who may require additional academic support.
Traditional methods depend mainly on manual analysis, which can be time-consuming and may not identify risk factors early.
-> This project provides a Machine Learning-based solution that helps institutions:
(i) Identify students at risk of failure
(ii) Understand important factors affecting performance
(iii) Provide early academic interventions
(iv) Improve student success rates using data-driven insights

## 🎯 Project Objectives
-> Build a classification model to predict student performance
-> Analyze important academic factors influencing outcomes
-> Develop an interactive dashboard for predictions
-> Evaluate model performance using classification metrics
-> Provide actionable insights for educational decision-making

## 🚀 Key Features
# 🏠 Home Dashboard
-> Project introduction
-> Dataset overview
-> Machine Learning model information
-> Key performance indicators
# 📊 Dataset Analysis
-> Dataset summary
-> Feature information
-> Statistical analysis
-> Data distribution insights
# 🔮 Student Performance Prediction
Users can enter student details and receive:
-> Prediction result
-> Pass/Fail classification
-> Prediction confidence score
-> Input summary
# 🤖 Model Performance Analysis
The dashboard provides:-
-> Accuracy Score
-> Precision Score
-> Recall Score
-> F1-Score
-> Confusion Matrix
-> ROC-AUC Score
-> Classification Report
-> ROC Curve
-> Precision-Recall Curve
# 📈 Data Visualization
-> Interactive visualizations include:-
(i) Student performance distribution
(ii) Feature relationships
(iii) Target variable analysis
(iv) Performance comparisons

## 🧠 Machine Learning Workflow
-> The project follows a complete Machine Learning pipeline:
# Data Collection
(i) Collected student academic and demographic information.
# Data Preprocessing
(i) Performed:
(a) Data cleaning
(b) Missing value handling
(c) Feature selection
(d) Data transformation
(e) Train-test split
# Model Development
(i) Implemented:
(a) Algorithms Used:
=> Logistic Regression Classification Model :- The model learns relationships between student attributes and performance outcomes to classify students into Pass or Fail categories.
# Model Evaluation
(i) The trained model was evaluated using:-
(a) Accuracy
(b) Precision
(c) Recall
(d) F1-Score
(e) ROC-AUC
(f) Confusion Matrix
# Dashboard Development
(i) Built an interactive web application using Streamlit for:-
(a) Data exploration
(b) Student prediction
(c) Model evaluation
(d) Business insights

## Project Architecture
Student_Performance_Prediction/
│
├── dashboard/
│   │
│   ├── app.py
│   ├── home.py
│   ├── prediction.py
│   ├── model_performance.py
│   ├── about.py
│   |-- eda_dashboard.py
|
├── models/
│   └── logistic_regression_model.pkl
│
├── data/
│   └── processed_student_performance_data.csv
│
├── requirements.txt
├── README.md
└── .gitignore

🤖 Machine Learning Model Information
Model Used

Logistic Regression

Problem Type

Binary Classification

Target Variable
pass_fail
Prediction Classes
0 → Fail
1 → Pass
Why Logistic Regression?

Logistic Regression was selected because:

Suitable for binary classification problems
Provides probability-based predictions
Easy interpretation of feature impact
Efficient and computationally lightweight
📊 Model Evaluation Metrics

The model performance is measured using:

Accuracy

Measures the overall percentage of correct predictions.

Precision

Measures how many predicted positive outcomes were actually correct.

Recall

Measures how effectively the model identifies students who successfully pass.

F1-Score

Provides a balance between precision and recall.

ROC-AUC Score

Measures the model's ability to distinguish between Pass and Fail categories.

🛠️ Technologies Used
Programming Language
Python
Machine Learning
Scikit-learn
Logistic Regression
Data Processing
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
Plotly
Dashboard Development
Streamlit
Model Serialization
Joblib
Development Tools
Jupyter Notebook
Visual Studio Code
GitHub
📂 Dataset Features

The dataset contains student-related attributes such as:

Academic performance indicators
Study-related factors
Attendance information
Personal and behavioral attributes
Target variable: pass_fail
⚙️ Installation & Setup
1. Clone Repository
git clone <repository-url>
2. Navigate to Project Directory
cd Student_Performance_Prediction
3. Install Required Libraries
pip install -r requirements.txt
4. Run Streamlit Dashboard
streamlit run dashboard/app.py
📌 Application Workflow
User Input
     |
     ↓
Data Validation
     |
     ↓
Feature Processing
     |
     ↓
Logistic Regression Model
     |
     ↓
Prediction Result
     |
     ↓
Dashboard Display

## Business Recommendations
-> Based on model predictions, educational institutions can:-
(a) Identify students requiring additional support
(b) Provide personalized learning strategies
(c) Improve academic monitoring systems
(d) Develop early intervention programs
(e) Optimize student success initiatives

## Future Enhancements
-> Future improvements may include:
(a) Adding advanced ML algorithms like Random Forest and XGBoost
(b) Real-time student monitoring system
(c) Explainable AI using SHAP values
(d) Student risk scoring system
(e) Cloud deployment
(f) Automated academic recommendation engine

## Deployment
-> The application can be deployed using:- Streamlit Community Cloud

## Author
Yash Sri
MCA Data Science Student
# Skills:
-> Python
-> Machine Learning
-> Data Analytics
-> Streamlit Dashboard Development

## Project Status

-> Data Preprocessing Completed
-> Model Training Completed
-> Logistic Regression Model Implemented
-> Dashboard Development Completed
-> Model Evaluation Completed
-> Deployment Ready

⭐ If you find this project useful, consider giving it a star on GitHub!