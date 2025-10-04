Here’s a clean, professional **README.md** for your **House_Prediction.py** project:

---

# 🏠 House Price Prediction using Linear Regression

This project uses **Linear Regression** to predict house prices based on key features such as **area**, **number of bedrooms**, and **number of bathrooms**. It includes data analysis, visualization, model training, evaluation, and prediction functionalities.

---

## 📂 Project Files

* **House_Predition.py** — Main Python script that handles data loading, visualization, model building, evaluation, and predictions.
* **Housing.csv** — Dataset containing house attributes and their corresponding prices.

---

## ⚙️ Requirements

Install the required Python libraries before running the script:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
```

---

## 🚀 How to Run the Project

1. Clone or download the project files.

2. Ensure `Housing.csv` is in the same directory as `House_Predition.py`.

3. Run the script:

   ```bash
   python House_Predition.py
   ```

4. The program will:

   * Display dataset insights and summary.
   * Visualize correlations and feature relationships.
   * Train a **Linear Regression** model.
   * Evaluate the model using **MSE**, **RMSE**, and **R²**.
   * Show actual vs predicted values and residual plots.
   * Provide example predictions for sample house configurations.

---

## 📊 Features

* **Data Analysis**

  * Summary statistics
  * Correlation matrix
  * Visualization of relationships between price and features

* **Model Training**

  * Linear Regression model using Scikit-learn
  * Train-test split (80-20)

* **Evaluation Metrics**

  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)
  * R² Score

* **Visualization**

  * Scatter plots for price vs features
  * Actual vs predicted prices plot
  * Residual plot

* **Prediction Function**

  * `predict_house_price(area, bedrooms, bathrooms)`
    → Returns the predicted house price for given inputs.

* **Feature Importance**

  * Based on regression coefficients
  * Variance Inflation Factor (VIF) to check multicollinearity

---

## 🧠 Example Prediction

```python
predicted_price = predict_house_price(5000, 3, 2)
print(f"Predicted Price: ₹{predicted_price:,.2f}")
```

**Output Example:**

```
Area: 5000 sq ft, 3 bedrooms, 2 bathrooms -> Predicted Price: ₹8,750,000.00
```

---

## 📈 Results Interpretation

* A higher **R² score** indicates better model performance.
* **Residual plots** help check for randomness (indicating good model fit).
* **VIF values** > 5 suggest possible multicollinearity issues.

---

## 🧾 License

This project is open-source and free to use for educational and research purposes.

---
