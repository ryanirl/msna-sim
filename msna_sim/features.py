import numpy as np

from typing import Dict
from typing import List
from typing import Any

from scipy import stats

from msna_sim.simulator import SimulationResults


def add_burst_features(results: SimulationResults, fs: int) -> List[Dict[str, Any]]:
    """Compute features that reflect what an expert annotator would see in the noisy signal."""
    clean_msna = results.clean_msna
    noisy_msna = results.noisy_msna
    burst_occurrences = results.burst_occurrences
    pure_noise = noisy_msna - clean_msna
    
    for burst in burst_occurrences:
        start_idx = burst["start_idx"]
        end_idx = burst["end_idx"]
        peak_idx = burst["peak_idx"]
        
        # Extract burst windows
        burst_window = slice(start_idx, end_idx)
        
        # Extended window for baseline (2 seconds around burst)
        window_samples = int(4.0 * fs)
        extended_start = max(0, peak_idx - window_samples // 2)
        extended_end = min(len(noisy_msna), peak_idx + window_samples // 2)
        extended_window = slice(extended_start, extended_end)
        
        # Extract signals
        clean_burst = clean_msna[burst_window]
        noisy_burst = noisy_msna[burst_window] 
        noise_burst = pure_noise[burst_window]
        clean_extended = clean_msna[extended_window]
        noisy_extended = noisy_msna[extended_window]
        noise_extended = pure_noise[extended_window]
        
        # Estimate baselines (exclude burst region)
        baseline_mask = np.ones(len(clean_extended), dtype=bool)
        burst_start_rel = max(0, start_idx - extended_start)
        burst_end_rel = min(len(clean_extended), end_idx - extended_start)
        baseline_mask[burst_start_rel:burst_end_rel] = False
        
        if np.sum(baseline_mask) > 10:
            clean_baseline = float(np.median(clean_extended[baseline_mask]))
            noisy_baseline = float(np.median(noisy_extended[baseline_mask]))
            baseline_noise_mad = float(stats.median_abs_deviation(noise_extended[baseline_mask]))
        else:
            clean_baseline = 0.0
            noisy_baseline = 0.0
            baseline_noise_mad = float(stats.median_abs_deviation(noise_extended))
        
        # Peak detection in noisy signal
        noisy_peak_idx = int(np.argmax(noisy_burst))
        noisy_peak_amplitude = float(noisy_burst[noisy_peak_idx])

        # Baseline-corrected amplitudes
        clean_corrected = burst["peak_amplitude"] - clean_baseline

        # Compute features of interest 
        width_samples = get_width_samples(noisy_burst, noisy_baseline, noisy_peak_amplitude)
        energy_snr_db = get_energy_snr_db(clean_burst, noise_burst, clean_baseline)
        mad_prominence = get_mad_prominence_score(clean_corrected, baseline_noise_mad)
        noise_contamination_ratio = get_noise_contamination_ratio(clean_corrected, noise_burst)
        
        # Update burst with computed features
        burst.update({
            "noisy_peak_idx": noisy_peak_idx,
            "noisy_burst_width": width_samples / fs,
            "energy_snr_db": energy_snr_db,
            "mad_prominence_score": mad_prominence,
            "noise_contamination_ratio": noise_contamination_ratio
        })

    return burst_occurrences


def get_mad_prominence_score(clean_corrected: float, baseline_noise_mad: float) -> float:
    """Get the MAD prominence score."""
    return clean_corrected / (1.4826 * baseline_noise_mad + 1e-10)


def get_noise_contamination_ratio(clean_corrected: float, noise_burst: np.ndarray) -> float:
    """Get the noise contamination ratio."""
    return 20 * float(np.var(noise_burst) / (clean_corrected ** 2 + 1e-12))


def get_width_samples(noisy_burst: np.ndarray, noisy_baseline: float, noisy_peak_amplitude: float) -> float:
    """Get the width of a burst in samples."""
    corrected_amplitude = noisy_peak_amplitude - noisy_baseline
    threshold_25 = noisy_baseline + 0.25 * corrected_amplitude
    width_samples = float(np.sum(noisy_burst >= threshold_25) * 2) if corrected_amplitude > 0 else 1.0
    return width_samples


def get_energy_snr_db(clean_burst: np.ndarray, noise_burst: np.ndarray, clean_baseline: float) -> float:
    """Get the energy SNR in dB."""
    clean_burst_corrected = clean_burst - clean_baseline
    signal_rms = np.sqrt(np.mean(clean_burst_corrected ** 2))
    noise_rms = np.sqrt(np.mean(noise_burst ** 2))
    energy_snr_db = 20 * np.log10((signal_rms + 1e-8) / (noise_rms + 1e-8))
    return energy_snr_db


