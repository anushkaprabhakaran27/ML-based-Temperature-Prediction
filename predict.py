import joblib
import pandas as pd

#loading the trained model
model=joblib.load("weather_model.pkl")

print("\n🌦 Enter Weather Details")
print("-" * 30)

day_of_year=int(input("day of year(1-365):"))
month=int(input("month(1-12):"))
humidity=float(input("humidity:"))
wind_speed=float(input("wind speed:"))
pressure=float(input("pressure:"))
cloud_cover=float(input("cloud cover:"))
previous_temp=float(input("previous temperature:"))

#create a DataFrame with the input values
new_data=pd.DataFrame({
    "day_of_year": [day_of_year],
    "month": [month],
    "humidity": [humidity],
    "wind_speed": [wind_speed],
    "pressure": [pressure],
    "cloud_cover": [cloud_cover],
    "previous_temp": [previous_temp]
})

#predicting the temperature
prediction=model.predict(new_data)

print("\n-------- Prediction Result --------")
print(f"🌡 Predicted Temperature: {prediction[0]:.2f} °C")
print("----------------------------------")