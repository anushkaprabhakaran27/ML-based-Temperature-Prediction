import pandas  as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


#loading the dataset

data = pd.read_csv("data/weather_dataset.csv")

#select the input columns
X=data[["day_of_year","month","humidity","wind_speed","pressure","cloud_cover","previous_temp"]]

#select the output column
y=data["temperature"]

#split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#create the ML model
forest_model=RandomForestRegressor(random_state=42)
linear_model=LinearRegression()
decision_model=DecisionTreeRegressor(random_state=42)

#train the model
forest_model.fit(X_train,y_train)
linear_model.fit(X_train,y_train)
decision_model.fit(X_train,y_train)

#check the accuracy of the model
forest_score=forest_model.score(X_test,y_test)
linear_score=linear_model.score(X_test,y_test)
decision_score=decision_model.score(X_test,y_test)

print("Random Forest Accuracy:",forest_score)
print("Linear Regression Accuracy:",linear_score)
print("Decision Tree Accuracy:",decision_score)

print("\nModel Comparison")
print("------------------------")
print(f"Linear Regression : {linear_score:.4f}")
print(f"Decision Tree     : {decision_score:.4f}")
print(f"Random Forest     : {forest_score:.4f}")

#prediction on test data
forest_pred=forest_model.predict(X_test)
linear_pred=linear_model.predict(X_test)
decision_pred=decision_model.predict(X_test)


#calculate the metrices
mae=mean_absolute_error(y_test,forest_pred)
mse=mean_squared_error(y_test,forest_pred)
rmse=np.sqrt(mse)

print("Mean Absolute Error:",round(mae,2))
print("Mean Squared Error:",round(mse,2))
print("Root Mean Squared Error:",round(rmse,2))

#using matplotlib to plot the actual vs predicted temperature
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Temperature")
plt.plot(forest_pred, label="Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.xlabel("Test samples")
plt.ylabel("Temperature(°C)")
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_predicted_temperature.png")
plt.show()

#feature importance
feature_importance=forest_model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(X.columns, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance") 
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

plt.figure(figsize=(8,5))

models = ["Linear Regression", "Decision Tree", "Random Forest"]
scores = [linear_score, decision_score, forest_score]

plt.bar(models, scores)

plt.title("Model Comparison (R² Score)")
plt.xlabel("Machine Learning Models")
plt.ylabel("R² Score")
plt.ylim(0, 1)

for i, score in enumerate(scores):
    plt.text(i, score + 0.02, f"{score:.2f}", ha="center")

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


joblib.dump(forest_model,"weather_model.pkl")

print("Model saved succesfully!")