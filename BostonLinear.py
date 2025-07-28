import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('HousingData.csv')

# Drop rows with missing values
df = df.dropna()

# Separate features and target
X_full = df.drop(columns=['MEDV'])
y = df['MEDV']

# Correlation threshold
threshold = 0.99

# Compute correlation matrix
cor_matrix = X_full.corr()
cols_to_drop = set()

# Detect and collect highly correlated columns (manual 2-for loop)
for i in range(len(cor_matrix.columns)):
    for j in range(i + 1, len(cor_matrix.columns)):
        feature1 = cor_matrix.columns[i]
        feature2 = cor_matrix.columns[j]
        corr_value = cor_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            if feature2 not in cols_to_drop:
                cols_to_drop.add(feature2)
                print(f"Dropping '{feature2}' due to high correlation with '{feature1}' (|{corr_value:.2f}| > {threshold})")

# Drop the highly correlated columns
X_full = X_full.drop(columns=cols_to_drop)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_full, y)

intercept = model.intercept_
coefficients = model.coef_
features = X_full.columns

# Print regression equation
print("\n===== Linear Regression Equation =====")
terms = [f"{coef:.4f}*{feat}" for coef, feat in zip(coefficients, features)]
equation = f"MEDV = {intercept:.4f} + " + " + ".join(terms)
print(equation)

# Evaluate model
y_pred = model.predict(X_full)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n===== Model Evaluation =====")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Manual predictions using model parameters
input_data_list = [
    {
        'features': {
            'CRIM': 0.08187,
            'ZN': 0,
            'INDUS': 2.89,
            'CHAS': 0,
            'NOX': 0.445,
            'RM': 7.82,
            'AGE': 36.9,
            'DIS': 3.4952,
            'RAD': 2,
            'PTRATIO': 18,
            'LSTAT': 3.57,
            'B': 396.9  # Include only if not dropped
        },
        'actual_MEDV': 43.8
    },
    {
        'features': {
            'CRIM': 0.17876,
            'ZN': 20,
            'INDUS': 4.53,
            'CHAS': 0,
            'NOX': 0.563,
            'RM': 4.95,
            'AGE': 30.9,
            'DIS': 4.7152,
            'RAD': 3,
            'PTRATIO': 15.4,
            'LSTAT': 5.12,
            'B': 392.83  # Include only if not dropped
        },
        'actual_MEDV': 0
    }
]

print("\n===== Manual Predictions Using Intercept and Coefficients =====")
for idx, entry in enumerate(input_data_list, start=1):
    features_input = entry['features']
    actual_y = entry['actual_MEDV']
    
    y_manual = intercept
    for coef, feat in zip(coefficients, features):
        y_manual += coef * features_input.get(feat, 0)  # Safe get in case feature was dropped
    
    print(f"Input Set {idx}:")
    print(f"  Predicted MEDV (y): {y_manual:.2f}")
    print(f"  Actual MEDV:        {actual_y:.2f}")
    print(f"  Error:              {abs(y_manual - actual_y):.2f}\n")
