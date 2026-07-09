import pandas  as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt



#loading the dataset
data = pd.read_csv("data/weather_dataset.csv")

#select the input columns
X=data[["day_of_year","month","humidity","wind_speed","pressure","cloud_cover","previous_temp"]]

#select the output column
y=data["temperature"]

#split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#create the ML model
model=RandomForestRegressor(random_state=42)

#train the model
model.fit(X_train,y_train)

#check the accuracy of the model
score=model.score(X_test,y_test)
print("model accuracy:",score)

#prediction on test data
y_pred=model.predict(X_test)

#calculate the metrices
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

print("Mean Absolute Error:",round(mae,2))
print("Mean Squared Error:",round(mse,2))
print("Root Mean Squared Error:",round(rmse,2))

#using matplotlib to plot the actual vs predicted temperature
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Temperature")
plt.plot(y_pred, label="Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.xlabel("Test samples")
plt.ylabel("Temperature(°C)")
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_predicted_temperature.png")
plt.show()

#feature importance
feature_importance=model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(X.columns, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance") 
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

joblib.dump(model,"weather_model.pkl")

print("Model saved succesfully!")