import streamlit as st
import pandas as pd
import numpy as np
import pickle
# Load trained model (we will create this later)
def load_model():
    with open("fitness_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model
model = load_model()
# Streamlit UI
st.title("🏋️ Personal Fitness Tracker")
st.write("Enter your details to predict calories burned!")

# Sidebar Inputs
age = st.slider("Age:", 10, 100, 25)
bmi = st.slider("BMI:", 15, 40, 22)
duration = st.slider("Duration (min):", 0, 60, 30)
heart_rate = st.slider("Heart Rate:", 60, 130, 80)
body_temp = st.slider("Body Temperature (°C):", 36, 42, 37)
gender = st.radio("Gender:", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0
# Display Inputs
data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Duration": [duration],
    "Heart Rate": [heart_rate],
    "Body Temp": [body_temp],
    "Gender_Male": [gender_val]
})
st.write("### Your Parameters:", data)



# Prediction
prediction = model.predict(data)[0]
st.write("## Prediction:")
st.write(f"🔥 Estimated Calories Burned: **{round(prediction, 2)} kilocalories**")
