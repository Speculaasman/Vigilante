import os
import numpy as np
import joblib
import sys
from features import extract_features_from_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'drowsiness_kss_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl') # New

def predict_drowsiness(file_path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model/Scaler not found. Run train_full.py first.")
        return

    # 1. Load Model & Scaler
    print("Loading AI core...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Extract Features
    print(f"Reading brainwaves from {os.path.basename(file_path)}...")
    features = extract_features_from_file(file_path)

    if features is None:
        print("Error: Could not process file.")
        return

    # 3. Normalize Features (CRITICAL STEP)
    # We use the SAME scaler from training so the math matches
    features_scaled = scaler.transform(features)

    # 4. Predict
    epoch_scores = model.predict(features_scaled)
    final_score = np.mean(epoch_scores)
    
    # 5. Visual Output
    bar_len = 20
    filled = int((final_score / 9.0) * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    
    print("\n" + "-"*40)
    print(f"FATIGUE LEVEL: [{bar}] {final_score:.2f} / 9.0")
    
    if final_score < 4: print("STATUS: 🟢 ALERT (Safe to drive)")
    elif final_score < 7: print("STATUS: 🟡 FATIGUE DETECTED (Caution)")
    else: print("STATUS: 🔴 DROWSY (Danger!)")
    print("-"*40)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/drozy/6-2.edf"
    predict_drowsiness(target)