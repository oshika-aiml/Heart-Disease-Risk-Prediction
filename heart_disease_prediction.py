import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]

data = pd.read_csv(url, names=cols, na_values="?").dropna()

features = ["age", "chol", "trestbps", "thalach"]
X = data[features]
y = (data["target"] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))])

print("Training started...")
model.fit(X_train, y_train)
print("Training completed!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nHeart Disease Risk Prediction")
print("--------------------------------")
print("Test Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_patient = pd.DataFrame([[55, 240, 140, 150]], columns=features)
prediction = model.predict(new_patient)[0]
prediction_text = "Heart Disease Risk" if prediction == 1 else "No Heart Disease Risk"

print("\nNew Patient:")
print("Age: 55")
print("Cholesterol: 240")
print("Blood Pressure: 140")
print("Maximum Heart Rate: 150")
print("Prediction:", prediction_text)