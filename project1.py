# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import joblib


# =========================
# Helper Functions
# =========================

def report_scores(y_true, y_pred, model_name):
    """Print and return accuracy, precision, recall, and F1 score."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n[{model_name}]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1w:.4f}")

    return {"Model": model_name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1_Weighted": f1w}


def plot_confusion_heatmap(y_true, y_pred, title="Confusion Matrix", labels=None):
    """Display a confusion matrix as a color-coded heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
# Step 1: Load and Inspect Data
# =========================
df = pd.read_csv("Project 1 Data.csv")

# Features and target
X = df[["X", "Y", "Z"]]
y = df["Step"].astype(int)

print("First 5 rows of the dataset:")
print(df.head(), "\n")
print("Missing values per column:")
print(df.isnull().sum(), "\n")
print("Class distribution (normalized):")
print(y.value_counts(normalize=True).sort_index(), "\n")


# =========================
# Step 2: Split the Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


# =========================
# Step 3: Visualization
# =========================
# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
points = ax.scatter(df["X"], df["Y"], df["Z"], c=df["Step"], cmap="viridis", s=25)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
fig.colorbar(points, ax=ax, label="Step")
plt.title("3D Scatter Plot of Coordinates by Step")
plt.show()

# Correlation Heatmap
corr = df[["X", "Y", "Z", "Step"]].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="rocket", fmt=".2f")
plt.title("Correlation Heatmap (X, Y, Z, Step)")
plt.tight_layout()
plt.show()


# =========================
# Step 4: Model Setup
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipelines
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, random_state=42))
])

pipe_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(random_state=42))
])

pipe_rf = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", RandomForestClassifier(random_state=42))
])

pipe_dt = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", DecisionTreeClassifier(random_state=42))
])

# Hyperparameter grids
grid_lr = {"clf__C": [0.05, 0.1, 0.5, 1.0, 2.0]}
rand_svc = {
    "clf__C": np.logspace(-2, 2, 21),
    "clf__kernel": ["rbf", "poly", "sigmoid"],
    "clf__gamma": ["scale", "auto"]
}
grid_rf = {
    "clf__n_estimators": [120, 200, 320],
    "clf__max_depth": [None, 8, 12, 16],
    "clf__min_samples_split": [2, 4, 6],
    "clf__min_samples_leaf": [1, 2, 3]
}
grid_dt = {
    "clf__max_depth": [None, 6, 10, 14, 18],
    "clf__min_samples_split": [2, 4, 6, 8],
    "clf__min_samples_leaf": [1, 2, 4]
}

# Model training with cross-validation
print("\nTraining models (please wait)...")
gs_lr = GridSearchCV(pipe_lr, grid_lr, cv=cv, n_jobs=-1, scoring="f1_weighted")
rs_svc = RandomizedSearchCV(pipe_svc, rand_svc, n_iter=25, cv=cv, n_jobs=-1,
                            random_state=42, scoring="f1_weighted")
gs_rf = GridSearchCV(pipe_rf, grid_rf, cv=cv, n_jobs=-1, scoring="f1_weighted")
gs_dt = GridSearchCV(pipe_dt, grid_dt, cv=cv, n_jobs=-1, scoring="f1_weighted")

gs_lr.fit(X_train, y_train)
rs_svc.fit(X_train, y_train)
gs_rf.fit(X_train, y_train)
gs_dt.fit(X_train, y_train)

print("\nBest Parameters Found:")
print("  Logistic Regression:", gs_lr.best_params_)
print("  SVC:", rs_svc.best_params_)
print("  Random Forest:", gs_rf.best_params_)
print("  Decision Tree:", gs_dt.best_params_)


# =========================
# Step 5: Evaluate Models
# =========================
models = {
    "Logistic Regression": gs_lr.best_estimator_,
    "SVC": rs_svc.best_estimator_,
    "Random Forest": gs_rf.best_estimator_,
    "Decision Tree": gs_dt.best_estimator_
}

results = []
for name, model in models.items():
    preds = model.predict(X_test)
    results.append(report_scores(y_test, preds, name))

# Pick best model by F1
best_model_name = max(results, key=lambda x: x["F1_Weighted"])["Model"]
print(f"\nBest model based on weighted F1: {best_model_name}\n")

# Plot confusion matrices for each model
for name, model in models.items():
    preds = model.predict(X_test)
    plot_confusion_heatmap(y_test, preds, title=f"{name} Confusion Matrix", labels=sorted(y.unique()))

# Classification report for the best model
best_preds = models[best_model_name].predict(X_test)
print("\nDetailed classification report for best model:\n")
print(classification_report(y_test, best_preds, zero_division=0))


# =========================
# Step 6: Stacking Ensemble
# =========================
stack_model = StackingClassifier(
    estimators=[
        ("rf", models["Random Forest"]),
        ("svc", models["SVC"])
    ],
    final_estimator=LogisticRegression(max_iter=3000, random_state=42),
    cv=cv,
    n_jobs=-1
)

stack_model.fit(X_train, y_train)
stack_preds = stack_model.predict(X_test)
results.append(report_scores(y_test, stack_preds, "Stacked Model (RF + SVC → LR)"))

plot_confusion_heatmap(y_test, stack_preds, title="Stacked Model Confusion Matrix", labels=sorted(y.unique()))


# =========================
# Step 7: Save Models & Predictions
# =========================
# Save best single model and stacked model
joblib.dump(models[best_model_name], "best_single_model.joblib")
joblib.dump(stack_model, "stacked_model.joblib")
print("\nModels saved as 'best_single_model.joblib' and 'stacked_model.joblib'")

# Save performance table
pd.DataFrame(results).to_csv("model_results.csv", index=False)
print("Model metrics saved as 'model_results.csv'")

# Make predictions on new coordinates
new_points = pd.DataFrame([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.0000, 3.0625, 1.93],
    [9.4000, 3.0000, 1.80],
    [9.4000, 3.0000, 1.30]
], columns=["X", "Y", "Z"])

new_preds = stack_model.predict(new_points)
print("\nPredicted Steps for New Coordinates:")
for coords, pred in zip(new_points.values, new_preds):
    print(f"{coords} → Step {pred}")
