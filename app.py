import streamlit as st
import joblib

# Load the trained model
model = joblib.load("weather_model.pkl")

st.title("🌦 Weather Prediction System")
st.write("Predict the temperature using Machine Learning.")

# User inputs
day_of_year = st.number_input("Day of Year", min_value=1, max_value=366, value=150)
month = st.number_input("Month", min_value=1, max_value=12, value=6)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=70)
wind_speed = st.number_input("Wind Speed (km/h)", value=15.0)
pressure = st.number_input("Pressure (hPa)", value=1015.0)
cloud_cover = st.number_input("Cloud Cover (%)", min_value=0, max_value=100, value=50)
previous_temp = st.number_input("Previous Temperature (°C)", value=28.0)

if st.button("Predict Temperature"):

    input_data = [[
        day_of_year,
        month,
        humidity,
        wind_speed,
        pressure,
        cloud_cover,
        previous_temp
    ]]

    prediction = model.predict(input_data)

    st.success(f"🌡 Predicted Temperature: {prediction[0]:.2f} °C")