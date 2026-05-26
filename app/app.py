
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model from Hugging Face
model_path = hf_hub_download(
    repo_id="ShauryaRawatRwt/Predictive_Maintenance_Model",
    filename="best_model.pkl"
)

model = joblib.load(model_path)

# Streamlit UI
st.title("Predictive Maintenance System")

st.write("Enter engine sensor values to predict engine condition.")

# User Inputs
engine_rpm = st.number_input("Engine RPM", value=1500.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure", value=3.0)
fuel_pressure = st.number_input("Fuel Pressure", value=5.0)
coolant_pressure = st.number_input("Coolant Pressure", value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", value=80.0)
coolant_temp = st.number_input("Coolant Temperature", value=90.0)

# Prediction button
if st.button("Predict Engine Condition"):

    input_df = pd.DataFrame({
        'Engine rpm': [engine_rpm],
        'Lub oil pressure': [lub_oil_pressure],
        'Fuel pressure': [fuel_pressure],
        'Coolant pressure': [coolant_pressure],
        'lub oil temp': [lub_oil_temp],
        'Coolant temp': [coolant_temp]
    })

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Maintenance Required")
    else:
        st.success("Engine Healthy")
