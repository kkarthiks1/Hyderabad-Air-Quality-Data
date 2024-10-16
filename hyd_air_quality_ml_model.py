import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

#Dropping Rows with at least 1 null value in CSV file
# making new data frame with dropped NA values
df = df.dropna(axis = 0, how ='any')

df.info()

df['AQI']=df['AQI'].astype(float)
df.info()


# Import necessary libraries
from sklearn.ensemble import StackingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LarsCV, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Assuming X and y are your dataset's features and target variable
# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the individual models with their best hyperparameters
xgboost_model = XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=200, objective='reg:squarederror', random_state=42)

larscv_model = LarsCV(eps=2.220446049250313e-16, max_n_alphas=100)

bayesian_ridge_model = BayesianRidge(alpha_1=1e-06, alpha_2=0.0001, lambda_1=0.0001, lambda_2=1e-06)

# Base estimator for AdaBoost (usually a Decision Tree)
base_estimator = DecisionTreeRegressor(max_depth=3)
adaboost_model = AdaBoostRegressor(estimator=base_estimator, learning_rate=0.1, loss='exponential', n_estimators=200, random_state=42)

# Initialize the final meta-model (you can choose another model if desired)
meta_model = BayesianRidge()

# Create a StackingRegressor
stacked_model = StackingRegressor(
    estimators=[
        ('xgboost', xgboost_model),
        ('larscv', larscv_model),
        ('bayesian_ridge', bayesian_ridge_model),
        ('adaboost', adaboost_model)
    ],
    final_estimator=meta_model,
    cv=5  # 5-fold cross-validation
)

# Fit the stacked model on the training data
stacked_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = stacked_model.predict(X_test)

# Define function to print evaluation metrics
def print_evaluate(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square:', r2_square)

# Evaluate the stacked model on the test data
print("Test Performance:")
print_evaluate(y_test, y_pred)

# Evaluate model on training data (optional)
print("\nTraining Performance:")
print_evaluate(y_train, stacked_model.predict(X_train))


# Scatter plot of true vs. predicted values
plt.figure(figsize=(10, 8), dpi=300)
plt.scatter(y_test, y_pred, color='orange', alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_test)], color='blue')
plt.title("Stacked Regression Model - True vs Predicted", fontsize=18)
plt.xlabel("True Values", fontsize=16)
plt.ylabel("Predictions", fontsize=16)
plt.show()

# Residual plot
plt.figure(figsize=(10, 8), dpi=300)
plt.scatter(y_pred, y_test - y_pred, color='orange', alpha=0.5)
plt.plot([min(y_pred), max(y_pred)], [0, 0], color='blue')
plt.title("Stacked Regression Model - Residual Plot", fontsize=18)
plt.xlabel("Predictions", fontsize=16)
plt.ylabel("Residuals", fontsize=16)
plt.show()

# Histogram of residuals
plt.figure(figsize=(10, 8), dpi=300)
plt.hist(y_test - y_pred, color='orange', alpha=0.5, bins=20)
plt.axvline(x=0, color='blue')
plt.title("Stacked Regression Model - Residual Histogram", fontsize=18)
plt.xlabel("Residuals", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.show()