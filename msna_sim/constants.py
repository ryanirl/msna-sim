# Physiological limits
RR_INTERVAL_MIN = 0.4  # seconds
RR_INTERVAL_MAX = 1.5  # seconds
BURST_DURATION_MIN = 0.3  # seconds
BURST_DURATION_MAX = 0.8  # seconds

# Noise frequency bands (Hz)
NOISE_BANDS = {
    "low_freq": (0.01, 0.5),
    "mid_freq": (0.5, 10), 
    "high_freq": (50, 200),
    "powerline": [60, 120]
}


