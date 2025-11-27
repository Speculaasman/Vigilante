# src/train_full.py
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Import our custom modules
from features import extract_features_from_file

# --- PATH SETUP ---
# Get the root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'drozy') # Expects data/drozy/
KSS_PATH = os.path.join(DATA_DIR, 'KSS.txt')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'drowsiness_kss_model.pkl')

def parse_kss_file(kss_path):
    """
    Reads KSS.txt and creates a mapping dictionary.
    Returns: { '1-1.edf': 3, '1-2.edf': 7, ... }
    """
    label_map = {}
    if not os.path.exists(kss_path):
        print(f"CRITICAL ERROR: KSS.txt not found at {kss_path}")
        return {}

    with open(kss_path, 'r') as f:
        lines = f.readlines()

    # KSS.txt structure: Row 1 = Subject 1. Columns = Test 1, 2, 3.
    for row_idx, line in enumerate(lines):
        subject_id = row_idx + 1
        scores = line.strip().split()
        
        for col_idx, score in enumerate(scores):
            test_id = col_idx + 1
            filename = f"{subject_id}-{test_id}.edf"
            
            # Filter out invalid scores (sometimes '?' or empty)
            if score.replace('.', '', 1).isdigit():
                label_map[filename] = float(score)
    
    return label_map

def main():
    print(f"--- Starting Training Pipeline ---")
    print(f"Looking for data in: {DATA_DIR}")

    # 1. Load Labels
    file_label_map = parse_kss_file(KSS_PATH)
    print(f"Found KSS labels for {len(file_label_map)} potential files.")

    X_all = []
    y_all = []
    files_processed = 0

    # 2. Loop through every file in the map
    for filename, kss_score in file_label_map.items():
        file_path = os.path.join(DATA_DIR, filename)
        
        # Check if we actually have the .edf file
        if os.path.exists(file_path):
            print(f"Processing {filename} (KSS: {kss_score})...")
            
            features = extract_features_from_file(file_path)
            
            if features is not None:
                # Add features to list
                X_all.append(features)
                
                # Create a label array same length as features
                # If we have 100 epochs, we need 100 labels of '7'
                labels = np.full(features.shape[0], kss_score)
                y_all.append(labels)
                files_processed += 1
        # Else: silently skip missing files (like 5-1.edf if not downloaded)

    if files_processed == 0:
        print("ERROR: No matching .edf files found. Check your 'data/drozy' folder.")
        return

    # 3. Aggregate Data
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"\nTotal Dataset Size: {X.shape[0]} epochs from {files_processed} files.")

    # 4. Train/Test Split (80/20)
    # We split randomly across all epochs to verify the model learns the patterns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train Model
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # 6. Evaluate
    preds = rf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\n--- Model Results ---")
    print(f"Mean Absolute Error: {mae:.2f} (On average, off by {mae:.2f} KSS points)")
    print(f"R2 Score: {r2:.2f}")

    # 7. Save Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    print(f"\nModel saved successfully to: {MODEL_PATH}")

if __name__ == "__main__":
    main()