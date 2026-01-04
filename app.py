import streamlit as st
import pickle
import numpy as np

with open("crop_loss_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    le = pickle.load(file)

st.title("🌾Crop Loss Prediction")

st.write("Predict crop loss risk based on climate conditions")

rainfall = st.slider("Rainfall (mm)", 500, 2000)
temperature = st.slider("Temperature (°C)", 20, 45)
humidity = st.slider("Humidity (%)", 30, 90)
soil_moisture = st.slider("Soil Moisture", 0.1, 0.9)
drought_index = st.slider("Drought Index", 0.0, 1.0)
crop_yield = st.slider("Crop Yield (tons/hectare)", 0.5, 5.0)

if st.button("Predict Crop Loss Risk"):
    input_data = np.array(
        [rainfall, temperature, humidity, soil_moisture, drought_index, crop_yield]
    ).reshape(1, -1)

    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)[0]

    st.success(f"🌱 Predicted Crop Loss Risk: **{result}**")
