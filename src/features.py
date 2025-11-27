# src/features.py
import mne
import numpy as np
import config  # Direct import (no dots)

def extract_features_from_file(file_path):
    """
    Reads an EDF file, filters it, and extracts Log-Band Powers (dB).
    """
    try:
        # 1. Load Data
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # 2. Pick Channels
    available_chs = raw.ch_names
    # Case-insensitive search for channels
    picked = [ch for ch in available_chs if any(t.lower() in ch.lower() for t in config.TARGET_CHANNELS)]
    
    if len(picked) == 0:
        return None
    
    try:
        raw.pick_channels(picked)

        # 3. Filter (1-30Hz)
        raw.filter(config.LOW_CUTOFF, config.HIGH_CUTOFF, fir_design='firwin', verbose=False)

        # 4. Epoching
        epochs = mne.make_fixed_length_epochs(raw, duration=config.EPOCH_DURATION, 
                                              overlap=0.0, verbose=False)
        
        # 5. Compute PSD
        psd = epochs.compute_psd(fmin=config.LOW_CUTOFF, fmax=config.HIGH_CUTOFF, verbose=False)
        
        # 6. Extract Features (NOW WITH LOG TRANSFORM)
        feature_vectors = []
        
        for band, (fmin, fmax) in config.FREQ_BANDS.items():
            # Get mean power (Shape: n_epochs)
            power = psd.get_data(fmin=fmin, fmax=fmax).mean(axis=(1, 2))
            
            # --- THE FIX: Convert to Decibels (Log Scale) ---
            # This handles the "Exponential" nature of brainwaves
            power_db = 10 * np.log10(power + 1e-9) 
            feature_vectors.append(power_db)

        # 7. Ratios (Do ratio on raw power, then log)
        # Re-get raw power for Theta and Alpha
        raw_theta = psd.get_data(fmin=4, fmax=8).mean(axis=(1, 2))
        raw_alpha = psd.get_data(fmin=8, fmax=12).mean(axis=(1, 2))
        
        # Ratio of Theta/Alpha
        ratio = raw_theta / (raw_alpha + 1e-6)
        feature_vectors.append(np.log10(ratio + 1e-9)) # Log the ratio too

        return np.column_stack(feature_vectors)

    except Exception as e:
        print(f"Processing error in {file_path}: {e}")
        return None