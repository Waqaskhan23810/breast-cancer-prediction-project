"""Breast Cancer Prediction - simple Data Science project.

Run:  python main.py
Dataset: scikit-learn built-in (koi download nahi chahiye)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

os.makedirs("outputs", exist_ok=True)

# 1. Data load
data = load_breast_cancer(as_frame=True)
df = data.frame
df["target"] = df["target"].map({0: "malignant", 1: "benign"})
print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print(df["target"].value_counts(), "\n")

# 2. Simple EDA charts
df["target"].value_counts().plot(kind="bar", color=["#2a9d8f", "#e76f51"])
plt.title("Class Distribution")
plt.ylabel("Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=120)
plt.close()

top = df.drop(columns="target").corrwith(data.target).abs().sort_values(ascending=False).head(10)
top.sort_values().plot(kind="barh", color="#264653")
plt.title("Top 10 Features Related to Diagnosis")
plt.xlabel("Absolute correlation")
plt.tight_layout()
plt.savefig("outputs/top_features.png", dpi=120)
plt.close()

# 3. Train / test split
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, pred)
    print(f"{name} accuracy: {results[name]:.3f}")

best = max(results, key=results.get)
print(f"\nBest model: {best}\n")
best_pred = models[best].predict(X_test)
print(classification_report(y_test, best_pred, target_names=["malignant", "benign"]))

# 5. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, display_labels=["malignant", "benign"], cmap="Blues"
)
plt.title(f"Confusion Matrix - {best}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=120)
plt.close()

pd.Series(results, name="accuracy").to_csv("outputs/results.csv")
print("Charts aur results 'outputs' folder me save ho gaye.")
