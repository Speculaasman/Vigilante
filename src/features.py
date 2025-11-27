# src/features.py
import mne
import numpy as np
import config  # Imports variables from config.py in the same folder

def extract_features_from_file(file_path):
    """
    Reads an EDF file, filters it, and extracts band powers.
    Returns: Numpy array (n_epochs, n_features) or None if failed.
    """
    try:
        # 1. Load Data (verbose=False keeps the console clean)
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # 2. Pick Channels
    # We look for channels that contain 'C3' or 'C4' (case insensitive logic often helps)
    available_chs = raw.ch_names
    picked = [ch for ch in available_chs if any(t in ch for t in config.TARGET_CHANNELS)]
    
    if len(picked) == 0:
        # Fail silently or warn if strictly necessary
        return None
    
    try:
        raw.pick_channels(picked)

        # 3. Filter (Remove drift <1Hz and muscle noise >30Hz)
        raw.filter(config.LOW_CUTOFF, config.HIGH_CUTOFF, fir_design='firwin', verbose=False)

        # 4. Epoching (Cut into fixed windows)
        epochs = mne.make_fixed_length_epochs(raw, duration=config.EPOCH_DURATION, 
                                              overlap=0.0, verbose=False)
        
        # 5. Compute PSD (Power Spectral Density)
        # fmin/fmax matches our filter to avoid edge artifacts
        psd = epochs.compute_psd(fmin=config.LOW_CUTOFF, fmax=config.HIGH_CUTOFF, verbose=False)
        
        # 6. Extract Features
        feature_vectors = []
        
        # Loop through Delta, Theta, Alpha, Beta
        for band, (fmin, fmax) in config.FREQ_BANDS.items():
            # Get mean power in this band across channels (axis 1) and freq bins (axis 2)
            # Result is shape (n_epochs,)
            power = psd.get_data(fmin=fmin, fmax=fmax).mean(axis=(1, 2))
            feature_vectors.append(power)

        # 7. Add Theta/Alpha Ratio (Critical Drowsiness Feature)
        # Theta is index 1, Alpha is index 2 in our config dict
        theta = feature_vectors[1]
        alpha = feature_vectors[2]
        ratio = theta / (alpha + 1e-6) # 1e-6 prevents division by zero
        feature_vectors.append(ratio)

        # Stack into matrix: Rows=Epochs, Cols=Features
        return np.column_stack(feature_vectors)

    except Exception as e:
        print(f"Processing error in {file_path}: {e}")
        return None