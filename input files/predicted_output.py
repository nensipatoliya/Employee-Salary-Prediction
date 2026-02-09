#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained models
linear_regressor = joblib.load("C:/Users/Grancy/salary_price_prediction/linear_regression.pkl")
decision_tree_regressor = joblib.load("C:/Users/Grancy/salary_price_prediction/decision_tree_regression.pkl")
random_forest_regressor = joblib.load("C:/Users/Grancy/salary_price_prediction/random_forest_regression.pkl")

# Streamlit App Title
st.title("Employee Salary Prediction App")
st.sidebar.header("User Input Features")

# User Input for Features
age = st.sidebar.number_input("Age", min_value=18, max_value=65, step=1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
years_experience = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

# Encode categorical inputs
gender_encoded = 1 if gender == "Male" else 0
education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
education_encoded = education_map[education_level]

# Prepare input data
input_features = np.array([[age, gender_encoded, education_encoded, years_experience]])

# Sample Actual Salary Data
y_test = np.linspace(30000, 150000, 50)  # Simulated actual salary values
x_test = np.linspace(1, 50, 50)  # Simulated X values

# Prediction Button
if st.sidebar.button("Predict Salary"):
    predicted_salary = random_forest_regressor.predict(input_features)[0]
    
    # Display Predicted and Actual Salaries
    st.subheader("Salary Comparison")
    salary_df = pd.DataFrame({"Type": ["Actual Salary", "Predicted Salary"],
                              "Salary": [y_test.mean(), predicted_salary]})
    st.table(salary_df)
    
    # Line Plot - Actual vs. Predicted Salaries
    st.subheader("Actual vs. Predicted Salaries (Line Plot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Actual Salary Line
    sns.lineplot(x=x_test, y=y_test, marker='o', linestyle='-', label='Actual Salary', color='black')
    
    # Predicted Salary Line
    ax.axhline(predicted_salary, color='red', linestyle='--', label='Predicted Salary')
    
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Actual vs. Predicted Salaries")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    st.success("Prediction Complete!")

