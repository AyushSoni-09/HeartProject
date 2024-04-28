import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import pickle 

with open('C:\\Users\\91869\\OneDrive\\Desktop\\streamlit\\HeartProject\\model.pkl', 'rb') as f:
    model = pickle.load(f)
st.title("Heart Disease Risk Predictor")

st.sidebar.subheader("Input Parameters")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
sex_mapping = {"Male": 0, "Female": 1}
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex_int = sex_mapping[sex]
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=1, max_value=300, value=120)
chol = st.sidebar.number_input("Cholesterol", min_value=1, max_value=1000, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=1, max_value=300, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0, max_value=10, value=1)
slope = st.sidebar.selectbox("Slopeof the Peak Exercise ST Segment", [0, 1, 2])
ca = st.sidebar.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

input_data = [[age, sex_int, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

# Define function to make predictions
def make_predictions(input_data):
    # Ensure model is loaded and fitted before making predictions
    if model:
        # Make predictions
        prediction =model.predict(input_data)
        return prediction
    else:
        st.error("Model is not loaded. Please check the path to the pickle file.")
# Make prediction on button click
if st.button("Make Prediction"):
     prediction = make_predictions(input_data)
     st.subheader("Prediction Result:")
     if prediction == 1:
        st.error("The patient has a higher probability of having heart disease.")
     else:
        st.success("The patient has a lower probability of having heart disease.")

# Visualize input parameters using a bar chart
input_df = pd.DataFrame({
    'Parameter': ['Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting BS', 'Resting ECG',
                  'Max HR Achieved', 'Exercise Angina', 'ST Depression', 'Slope', 'Major Vessels', 'Thalassemia'],
    'Value': [cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
})

st.write("\n\n")
st.subheader("Input Parameters Visualization:")
fig, ax = plt.subplots()
ax.barh(input_df['Parameter'], input_df['Value'], color='skyblue')
ax.set_xlabel('Value')
ax.set_title('Input Parameters')
st.pyplot(fig)