TARGET_CHANNELS = ['C3', 'C4'] 

# Length of time (seconds) for each analysis window
EPOCH_DURATION = 5.0 

# Frequency bands for Power Spectral Density (PSD)
FREQ_BANDS = {
    'Delta': (1, 4),   # Deep sleep
    'Theta': (4, 8),   # Drowsiness / Daydreaming
    'Alpha': (8, 12),  # Relaxed / Awake (Eyes Closed)
    'Beta':  (12, 30)  # Active processing / Stress
}

# Filter settings (Hz)
LOW_CUTOFF = 1.0
HIGH_CUTOFF = 30.0