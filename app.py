import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model/model.pkl", "rb"))

st.title("Student Performance Predictor")

study_hours = st.slider("Study Hours per Day", 0, 12, 4)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.slider("Sleep Hours", 0, 12, 6)
previous_score = st.slider("Previous Score", 0, 100, 60)

if st.button("Predict"):
    input_data = np.array([[study_hours, attendance, sleep_hours, previous_score]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Score: {prediction[0]:.2f}")
