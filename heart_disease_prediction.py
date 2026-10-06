import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load UCI Heart Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

data = pd.read_csv(url, names=columns, na_values="?")
data = data.dropna()

# Select simple features
X = data[["age", "chol", "trestbps", "thalach"]]

# Convert target into 0 = No Disease, 1 = Disease
y = (data["target"] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Machine Learning model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
print("Training started...")
model.fit(X_train, y_train)
print("Training completed!")

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nHeart Disease Risk Prediction")
print("--------------------------------")
print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test with a new patient
new_patient = pd.DataFrame({
    "age": [55],
    "chol": [240],
    "trestbps": [140],
    "thalach": [150]
})

prediction = model.predict(new_patient)[0]

print("\nNew Patient:")
print("Age: 55")
print("Cholesterol: 240")
print("Blood Pressure: 140")
print("Maximum Heart Rate: 150")

if prediction == 1:
    print("Prediction: Heart Disease Risk")
else:
    print("Prediction: No Heart Disease Risk")