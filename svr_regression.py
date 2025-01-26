import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load Dataset
# Replace 'path_to_dataset.csv' with the actual path to your Kaggle dataset.
data = pd.read_csv("path_to_dataset.csv")

# Assuming the dataset has columns 'YearsExperience' and 'Salary'.
X = data[['YearsExperience']].values
y = data['Salary'].values

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Kernels to Test
kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:
    print(f"Training SVR with {kernel} kernel...")
    # Train SVR Model
    svr = SVR(kernel=kernel, C=1.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train_scaled)

    # Predict
    y_pred_scaled = svr.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{kernel.capitalize()} Kernel: MSE = {mse:.2f}, R2 = {r2:.2f}")

    # Plot Results
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.title(f'SVR with {kernel.capitalize()} Kernel')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()
