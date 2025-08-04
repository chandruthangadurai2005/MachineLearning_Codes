import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset — each row is a student
data = {
    'Maths':    [35, 70, 50, 90, 40, 20, 60, 85, 30, 95],
    'Science':  [40, 80, 55, 95, 38, 25, 70, 88, 35, 98],
    'English':  [30, 75, 48, 92, 45, 28, 68, 89, 33, 96],
    'History':  [25, 65, 42, 88, 36, 22, 60, 84, 30, 90],
    'CS':       [50, 90, 60, 98, 42, 18, 75, 92, 40, 99],
    'Result':   [0, 1, 1, 1, 0, 0, 1, 1, 0, 1]  # 0 = Fail, 1 = Pass
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Custom Prediction
print("\n--- Predict Result for New Student ---")
maths = float(input("Enter Maths marks: "))
science = float(input("Enter Science marks: "))
english = float(input("Enter English marks: "))
history = float(input("Enter History marks: "))
cs = float(input("Enter CS marks: "))

new_data = np.array([[maths, science, english, history, cs]])
prediction = model.predict(new_data)

print("Prediction:", "Pass ✅" if prediction[0] == 1 else "Fail ❌")
