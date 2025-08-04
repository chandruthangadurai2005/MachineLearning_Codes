
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data from CSV
df = pd.read_csv("student_results_binary.csv")  # Update path if needed

# Features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict for a new student
print("\n--- Predict Result for New Student ---")
maths = float(input("Enter Maths marks: "))
science = float(input("Enter Science marks: "))
english = float(input("Enter English marks: "))
history = float(input("Enter History marks: "))
cs = float(input("Enter CS marks: "))

new_data = pd.DataFrame([{
    'Maths': maths,
    'Science': science,
    'English': english,
    'History': history,
    'CS': cs
}])

prediction = model.predict(new_data)

print("Prediction:", "Pass ✅" if prediction[0] == 1 else "Fail ❌")









import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Load data from CSV
df = pd.read_csv("student_scores.csv")  # Replace path if necessary

# Features and target
X = df.drop('FinalScore', axis=1)
y = df['FinalScore']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize models
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
elastic.fit(X_train, y_train)

# Predictions
y_lasso_pred = lasso.predict(X_test)
y_ridge_pred = ridge.predict(X_test)
y_elastic_pred = elastic.predict(X_test)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} Regression ---")
    print("R² Score:", round(r2_score(y_true, y_pred), 3))
    print("MSE:", round(mean_squared_error(y_true, y_pred), 3))
    print("Predicted Pass/Fail (Threshold = 50):", ["Pass ✅" if p >= 50 else "Fail ❌" for p in y_pred])

# Evaluate each model
evaluate_model("Lasso", y_test, y_lasso_pred)
evaluate_model("Ridge", y_test, y_ridge_pred)
evaluate_model("ElasticNet", y_test, y_elastic_pred)

# Predict for a new student
print("\n--- Predict Final Score for New Student ---")
maths = float(input("Maths: "))
science = float(input("Science: "))
english = float(input("English: "))
history = float(input("History: "))
cs = float(input("CS: "))
new_data = pd.DataFrame([{
    'Maths': maths,
    'Science': science,
    'English': english,
    'History': history,
    'CS': cs
}])


# Make predictions
lasso_score = lasso.predict(new_data)[0]
ridge_score = ridge.predict(new_data)[0]
elastic_score = elastic.predict(new_data)[0]

# Display results
print(f"\nPredicted Final Score using Lasso: {lasso_score:.2f} → {'Pass ✅' if lasso_score >= 50 else 'Fail ❌'}")
print(f"Predicted Final Score using Ridge: {ridge_score:.2f} → {'Pass ✅' if ridge_score >= 50 else 'Fail ❌'}")
print(f"Predicted Final Score using ElasticNet: {elastic_score:.2f} → {'Pass ✅' if elastic_score >= 50 else 'Fail ❌'}")

