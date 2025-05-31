import numpy as np


def generate_colored_noise(
    n_samples: int, fs: int, beta: float = 1.0, amplitude: float = 1.0, integration_smoothing: float = 0.95
) -> np.ndarray:
    """Generate colored noise with 1/f^beta characteristics."""
    white_noise = np.random.normal(0, 1, n_samples)
    
    if beta <= 0:
        return white_noise / np.std(white_noise) * amplitude
    
    # Apply exponential smoothing
    filtered_noise = np.zeros_like(white_noise)
    filtered_noise[0] = white_noise[0]
    
    for i in range(1, n_samples):
        filtered_noise[i] = (
            integration_smoothing * filtered_noise[i-1] + (1 - integration_smoothing) * white_noise[i])
    
    # Normalize and scale
    if np.std(filtered_noise) > 0:
        filtered_noise = filtered_noise / np.std(filtered_noise) * amplitude
    
    return filtered_noise


def generate_band_limited_noise(
    n_samples: int, fs: int, low_freq: float, high_freq: float, amplitude: float
) -> np.ndarray:
    """Generate band-limited noise."""
    white_noise = np.random.normal(0, 1, n_samples)
    
    # Validate frequency bounds
    nyquist = fs / 2
    if low_freq <= 0 or high_freq >= nyquist or low_freq >= high_freq:
        return white_noise / np.std(white_noise) * amplitude
    
    # FFT-based filtering
    fft_noise = np.fft.fft(white_noise)
    freqs = np.fft.fftfreq(n_samples, 1 / fs)
    
    # Create frequency mask
    freq_mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
    fft_filtered = fft_noise.copy()
    fft_filtered[~freq_mask] = 0
    
    # Convert back to time domain
    filtered_noise = np.fft.ifft(fft_filtered).real
    
    # Scale amplitude
    if np.std(filtered_noise) > 0:
        filtered_noise = filtered_noise / np.std(filtered_noise) * amplitude
    
    return filtered_noise


def generate_powerline_noise(t: np.ndarray, amplitude_60: float, amplitude_120: float) -> np.ndarray:
    """Generate powerline interference at 60Hz and 120Hz."""
    powerline_60 = amplitude_60 * np.sin(2 * np.pi * 60 * t)
    powerline_120 = amplitude_120 * np.sin(2 * np.pi * 120 * t)
    return powerline_60 + powerline_120


def generate_respiratory_artifacts(t: np.ndarray, resp_rate: float, amplitude: float) -> np.ndarray:
    """Generate respiratory movement artifacts."""
    resp_freq = resp_rate / 60.0
    return amplitude * np.sin(2 * np.pi * resp_freq * t)


def generate_spike_artifacts(n_samples: int, amplitude: float, poisson_factor: float) -> np.ndarray:
    """Generate random spike artifacts."""
    spike_signal = np.zeros(n_samples)
    
    # Generate random spikes
    n_spikes = np.random.poisson(poisson_factor)
    if n_spikes > 0:
        spike_indices = np.random.randint(0, n_samples, n_spikes)
        spike_amplitudes = np.random.exponential(amplitude, n_spikes)
        spike_signs = np.random.choice([-1, 1], n_spikes)
        
        for idx, amp, sign in zip(spike_indices, spike_amplitudes, spike_signs):
            spike_signal[idx] += amp * sign
    
    return spike_signal


