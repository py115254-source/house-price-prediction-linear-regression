# House Price Prediction using Linear Regression
# Task 4 - SkillPilots

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

print("=" * 60)
print("HOUSE PRICE PREDICTION - LINEAR REGRESSION")
print("=" * 60)

print("\nLoading dataset...")

housing = fetch_california_housing(as_frame=True)

df = housing.frame

# Rename target column for better readability
df = df.rename(columns={"MedHouseVal": "HousePrice"})

print("Dataset loaded successfully.")


# --------------------------------------------------
# 2. Basic Data Exploration
# --------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# 3. Save Dataset as CSV
# --------------------------------------------------

data_folder = "Data"
os.makedirs(data_folder, exist_ok=True)

csv_path = os.path.join(data_folder, "house_prices.csv")

df.to_csv(csv_path, index=False)

print(f"\nDataset saved to: {csv_path}")


# --------------------------------------------------
# 4. Separate Features and Target
# --------------------------------------------------

X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("HousePrice")


# --------------------------------------------------
# 5. Split Dataset into Training and Testing Data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 6. Create Linear Regression Model
# --------------------------------------------------

model = LinearRegression()


# --------------------------------------------------
# 7. Train the Model
# --------------------------------------------------

print("\nTraining Linear Regression model...")

model.fit(X_train, y_train)

print("Model training completed successfully.")


# --------------------------------------------------
# 8. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 9. Evaluate the Model
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error (MAE)  : {mae:.4f}")
print(f"Mean Squared Error (MSE)   : {mse:.4f}")
print(f"Root Mean Squared Error    : {rmse:.4f}")
print(f"R² Score                   : {r2:.4f}")


# --------------------------------------------------
# 10. Display Sample Predictions
# --------------------------------------------------

results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

print("\nSample Predictions:")
print(results)


# --------------------------------------------------
# 11. Save Trained Model
# --------------------------------------------------

model_folder = "models"
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(
    model_folder,
    "linear_regression_model.pkl"
)

joblib.dump(model, model_path)

print(f"\nTrained model saved to: {model_path}")


# --------------------------------------------------
# 12. Actual vs Predicted Price Graph
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig("house_price_predictions.png", dpi=300)

plt.show()


# --------------------------------------------------
# 13. Project Completion Message
# --------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)