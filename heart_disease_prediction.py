!pip install ucimlrepo -q

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

heart_disease = fetch_ucirepo(id=45)

X_all = heart_disease.data.features
y = heart_disease.data.targets

y = y.iloc[:, 0].astype(int)
y = (y > 0).astype(int)

features = ['age', 'chol', 'trestbps', 'thalach']

X = X_all[features].copy()
X = X.apply(pd.to_numeric, errors='coerce')

data = pd.concat([X, y], axis=1)
data = data.dropna()

X = data[features]
y = data.iloc[:, -1]

print("Dataset loaded successfully!")
print("Number of records:", len(data))
print("\nFirst 5 rows:")
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)

print("\nModel training completed!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_patient = pd.DataFrame({'age': [55], 'chol': [240], 'trestbps': [140], 'thalach': [150]})

prediction = model.predict(new_patient)

print("\nNew Patient Information:")
print("Age:", new_patient['age'].iloc[0])
print("Cholesterol:", new_patient['chol'].iloc[0])
print("Blood Pressure:", new_patient['trestbps'].iloc[0])
print("Maximum Heart Rate:", new_patient['thalach'].iloc[0])

labels = ["No-disease class", "Disease-present class"]
print("\nPrediction:", labels[prediction[0]])

print("\nNote: This project is for educational purposes only and is NOT a medical diagnosis.")