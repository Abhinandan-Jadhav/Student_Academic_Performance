from pathlib import Path
import numpy as np
import joblib
from tensorflow.keras.models import load_model

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "student_model.keras"
SCALER_PATH = BASE / "models" / "scaler.pkl"

FEATURES = [
    "attendance", "study_hours", "previous_score", "assignment_score",
    "internal_score", "practical_score", "participation", "gpa"
]
CLASS_NAMES = ["At Risk", "Average", "Good", "Excellent"]

def calculate_risk(attendance, study_hours, performance):
    if performance == "At Risk" or attendance < 60:
        return "HIGH"
    if performance == "Average" or attendance < 75 or study_hours < 8:
        return "MEDIUM"
    return "LOW"

def recommendations(data, performance):
    rec = []
    if data["attendance"] < 75:
        rec.append("Improve class attendance.")
    if data["study_hours"] < 10:
        rec.append("Increase weekly study hours.")
    if data["assignment_score"] < 70:
        rec.append("Complete assignments regularly.")
    if data["internal_score"] < 70:
        rec.append("Focus on internal assessment.")
    if data["previous_score"] < 60:
        rec.append("Revise previous topics and seek academic support.")
    if performance == "At Risk":
        rec.append("Discuss an academic improvement plan with a teacher.")
    if not rec:
        rec.append("Maintain current academic habits and continue monitoring progress.")
    return rec

def predict_student(data):
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    row = np.array([[float(data[f]) for f in FEATURES]])
    row_scaled = scaler.transform(row)
    probs = model.predict(row_scaled, verbose=0)[0]
    class_id = int(np.argmax(probs))
    performance = CLASS_NAMES[class_id]
    confidence = float(probs[class_id] * 100)
    risk = calculate_risk(data["attendance"], data["study_hours"], performance)

    return {
        "performance": performance,
        "confidence": confidence,
        "risk": risk,
        "probabilities": {
            name: float(prob * 100)
            for name, prob in zip(CLASS_NAMES, probs)
        },
        "recommendations": recommendations(data, performance)
    }
