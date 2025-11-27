import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # <--- NEW IMPORT
from sklearn.metrics import mean_absolute_error, r2_score
from features import extract_features_from_file

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'INSERT_DATA_FOLDER', 'drozy_data') 
KSS_PATH = os.path.join(DATA_DIR, 'KSS.txt')
# We save both the Model AND the Scaler
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'drowsiness_kss_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

def parse_kss_file(kss_path):
    label_map = {}
    if not os.path.exists(kss_path):
        print(f"CRITICAL ERROR: KSS.txt not found at {kss_path}")
        return {}

    with open(kss_path, 'r') as f:
        lines = f.readlines()

    for row_idx, line in enumerate(lines):
        subject_id = row_idx + 1
        scores = line.strip().split()
        for col_idx, score in enumerate(scores):
            test_id = col_idx + 1
            filename = f"{subject_id}-{test_id}.edf"
            if score.replace('.', '', 1).isdigit():
                label_map[filename] = float(score)
    return label_map

def main():
    print(f"--- Starting Training Pipeline (With Normalization) ---")
    
    # 1. Load Labels
    file_label_map = parse_kss_file(KSS_PATH)
    X_all = []
    y_all = []
    files_processed = 0

    # 2. Extract Features
    for filename, kss_score in file_label_map.items():
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            print(f"Processing {filename} (KSS: {kss_score})...")
            features = extract_features_from_file(file_path)
            if features is not None:
                X_all.append(features)
                labels = np.full(features.shape[0], kss_score)
                y_all.append(labels)
                files_processed += 1

    if files_processed == 0:
        print("ERROR: No files found.")
        return

    # 3. Aggregate Data
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    
    # --- NEW STEP: SCALING ---
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # Centers data around 0
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. Train Model
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # 6. Evaluate
    preds = rf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\n--- Model Results ---")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    # 7. Save Model AND Scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH) # Essential for new data
    print(f"Saved model and scaler to 'models/'")

if __name__ == "__main__":
    main()