import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Housing.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Select features and target variable
features = ['area', 'bedrooms', 'bathrooms']
X = df[features]
y = df['price']

# Display correlation matrix
print("\nCorrelation Matrix:")
correlation_matrix = df[['price', 'area', 'bedrooms', 'bathrooms']].corr()
print(correlation_matrix)

# Visualize relationships
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(df['area'], df['price'], alpha=0.6)
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('Price vs Area')

plt.subplot(1, 3, 2)
plt.scatter(df['bedrooms'], df['price'], alpha=0.6)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('Price vs Bedrooms')

plt.subplot(1, 3, 3)
plt.scatter(df['bathrooms'], df['price'], alpha=0.6)
plt.xlabel('Number of Bathrooms')
plt.ylabel('Price')
plt.title('Price vs Bathrooms')

plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Display model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:,.2f}")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:,.2f}")

# Create predictions vs actual values plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Function to make predictions for new houses
def predict_house_price(area, bedrooms, bathrooms):
    """
    Predict house price based on area, bedrooms, and bathrooms
    """
    features_array = np.array([[area, bedrooms, bathrooms]])
    predicted_price = model.predict(features_array)
    return predicted_price[0]

# Example predictions
print("\nExample Predictions:")
example_houses = [
    (5000, 3, 2),   # Medium house
    (8000, 4, 3),   # Large house
    (3000, 2, 1)    # Small house
]

for area, bedrooms, bathrooms in example_houses:
    predicted_price = predict_house_price(area, bedrooms, bathrooms)
    print(f"Area: {area} sq ft, {bedrooms} bedrooms, {bathrooms} bathrooms -> "
          f"Predicted Price: ₹{predicted_price:,.2f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': abs(model.coef_)
}).sort_values('Coefficient', ascending=False)

print("\nFeature Importance (by coefficient magnitude):")
print(feature_importance)

# Additional analysis: Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("\nVariance Inflation Factor (VIF):")
print(vif_data)