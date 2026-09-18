from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "student_performance.csv"
MODELS = BASE / "models"
RESULTS = BASE / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

FEATURES = [
    "attendance", "study_hours", "previous_score", "assignment_score",
    "internal_score", "practical_score", "participation", "gpa"
]
CLASS_NAMES = ["At Risk", "Average", "Good", "Excellent"]

def performance_category(score):
    if score >= 90:
        return 3
    if score >= 75:
        return 2
    if score >= 50:
        return 1
    return 0

def train_model():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=FEATURES + ["final_score"]).copy()
    df["performance"] = df["final_score"].apply(performance_category)

    X = df[FEATURES].values
    y = df["performance"].values

    # Need enough examples per class for stratified splitting.
    if len(df) < 100:
        raise ValueError("Use at least 100 student records for reliable training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, MODELS / "scaler.pkl")

    y_train_cat = to_categorical(y_train, num_classes=4)
    y_test_cat = to_categorical(y_test, num_classes=4)

    model = Sequential([
        Input(shape=(len(FEATURES),)),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.20,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    print("\nMODEL PERFORMANCE")
    for k, v in metrics.items():
        print(f"{k.title():10}: {v:.4f}")

    print("\nClassification Report")
    print(classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
    ))

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Student Performance Model Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "loss_curve.png", dpi=150)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Student Performance Model Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Student Performance Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS / "confusion_matrix.png", dpi=150)
    plt.close()

    # Correlation heatmap
    corr = df[FEATURES + ["final_score"]].corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Student Performance Feature Correlation")
    plt.tight_layout()
    plt.savefig(RESULTS / "correlation_heatmap.png", dpi=150)
    plt.close()

    model.save(MODELS / "student_model.keras")

    # Save metrics and class names
    import json
    with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved to:", MODELS / "student_model.keras")
    return metrics

if __name__ == "__main__":
    train_model()
