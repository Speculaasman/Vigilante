# src/predict.py
import os
import numpy as np
import joblib
import argparse
from features import extract_features_from_file

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'drowsiness_kss_model.pkl')

def predict_drowsiness(file_path):
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please run train_full.py first.")
        return

    # 1. Load Model
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    # 2. Extract Features
    print(f"Processing {file_path}...")
    features = extract_features_from_file(file_path)

    if features is None:
        print("Could not extract features. Check file path or channels.")
        return

    # 3. Predict
    # This gives a score for every 5-second chunk
    epoch_scores = model.predict(features)
    
    # Average them for the final session score
    final_score = np.mean(epoch_scores)
    
    print("\n" + "="*30)
    print(f"PREDICTION RESULT")
    print("="*30)
    print(f"Average KSS Score: {final_score:.1f} / 9.0")
    
    # Interpretation
    status = ""
    if final_score < 3: status = "🟢 ALERT"
    elif final_score < 5: status = "🟡 MILDLY TIRED"
    elif final_score < 7: status = "🟠 DROWSY"
    else: status = "🔴 DANGER (Sleepy)"
    
    print(f"Status: {status}")
    print("="*30)

if __name__ == "__main__":
    # You can change this filename to test different files
    # Example: python src/predict.py --file data/drozy/6-2.edf
    import sys
    if len(sys.argv) > 1:
        target_file = sys.argv[1] # User provided file in command line
    else:
        # Default for testing
        target_file = os.path.join(BASE_DIR, 'data', 'drozy', '6-2.edf')
        
    predict_drowsiness(target_file)