from dataclasses import dataclass
import numpy as np

from typing import Optional
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any

from msna_sim.config import PatientConfig
from msna_sim.config import SignalConfig
from msna_sim.constants import *
from msna_sim.noise import *


@dataclass
class SimulationResults:
    """Container for MSNA simulation results."""
    
    time: np.ndarray
    clean_msna: np.ndarray
    noisy_msna: np.ndarray
    respiratory_signal: np.ndarray
    respiratory_modulation: np.ndarray
    r_times: np.ndarray
    rr_intervals: np.ndarray
    burst_occurrences: List[Dict[str, Any]]
    fs: int
    
    @property
    def duration(self) -> float:
        """Total recording duration in seconds."""
        return len(self.time) / self.fs
    
    @property
    def n_bursts(self) -> int:
        """Total number of bursts."""
        return len(self.burst_occurrences)
    
    @property
    def burst_rate(self) -> float:
        """Burst rate in bursts per minute."""
        return (self.n_bursts / self.duration) * 60 if self.duration > 0 else 0
    
    @property
    def actual_burst_incidence(self) -> float:
        """Actual burst incidence as percentage."""
        if len(self.rr_intervals) == 0:
            return 0.0
        return (self.n_bursts / len(self.rr_intervals)) * 100
    
    @property
    def mean_heart_rate(self) -> float:
        """Mean heart rate in bpm."""
        return 60.0 / float(np.mean(self.rr_intervals)) if len(self.rr_intervals) > 0 else 0.0
    
    def get_burst_times(self) -> np.ndarray:
        """Get array of burst occurrence times."""
        return np.array([burst["time"] for burst in self.burst_occurrences])


class Simulation:
    def __init__(self, patient_config: PatientConfig, signal_config: Optional[SignalConfig] = None) -> None:
        """
        MSNA simulator with realistic signal generation.

        Args:
            patient_config: Patient physiological configuration
            signal_config: Signal generation configuration (uses defaults if None)
        """
        self.patient_config = patient_config
        self.signal_config = signal_config if signal_config is not None else SignalConfig()
    
    def _generate_cardiac_timing(self, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic cardiac timing with HRV."""
        mean_rr = 60.0 / self.patient_config.heart_rate
        num_beats = int(duration / mean_rr) + 5
        
        # Generate RR intervals with HRV
        rr_intervals = np.random.normal(mean_rr, self.patient_config.hrv_std, num_beats)
        rr_intervals = np.clip(rr_intervals, RR_INTERVAL_MIN, RR_INTERVAL_MAX)
        
        # Create R-wave times
        r_times = np.cumsum(np.concatenate([[0], rr_intervals]))
        r_times = r_times[r_times < duration]
        
        return r_times, rr_intervals[:len(r_times)-1]
    
    def _generate_respiratory_modulation(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate respiratory modulation of sympathetic activity."""
        resp_freq = self.patient_config.resp_rate / 60.0
        resp_phase = 2 * np.pi * resp_freq * t
        
        # Add breathing irregularity
        phase_noise = (
            self.signal_config.phase_noise_amplitude * np.sin(3.7 * resp_phase) + 
            self.signal_config.breathing_irregularity * np.random.randn(len(t))
        )
        resp_signal = np.cos(resp_phase + phase_noise)
        
        # Convert to modulation factor
        base_level = 1.0 - self.patient_config.resp_modulation_strength/2
        modulation = base_level + self.patient_config.resp_modulation_strength * (1 + resp_signal) / 2
        
        return modulation, resp_signal
    
    def _determine_burst_occurrences(
        self, r_times: np.ndarray, resp_modulation: np.ndarray, fs: int, duration: float
    ) -> List[Dict[str, Any]]:
        """Determine burst occurrences with phase-locking constraint."""
        burst_occurrences = []
        for i in range(len(r_times) - 1):
            r_time = r_times[i]
            next_r_time = r_times[i + 1]
            rr_interval = next_r_time - r_time
            
            # Get respiratory modulation
            r_idx = min(int(r_time * fs), len(resp_modulation) - 1)
            resp_mod = resp_modulation[r_idx]
            
            # Calculate burst probability
            base_prob = self.patient_config.burst_incidence / 100.0
            modulated_prob = base_prob * resp_mod
            
            # Decide if burst occurs
            if np.random.random() < modulated_prob:
                # Calculate burst timing
                burst_delay = np.random.normal(
                    self.patient_config.burst_delay_mean, 
                    self.patient_config.burst_delay_std
                )
                burst_delay = np.clip(burst_delay, 0.8, rr_interval - 0.2)
                burst_time = r_time + burst_delay
                
                if burst_time < duration:
                    burst_occurrences.append({
                        "time": burst_time,
                        "rr_interval": rr_interval,
                        "resp_modulation": resp_mod
                    })
        
        return burst_occurrences

    def _generate_burst_shape(self, duration_samples: int, amplitude: float) -> np.ndarray:
        """Generate realistic burst shape."""
        if duration_samples <= 0:
            return np.array([])
        
        t_burst = np.linspace(-2, 2, duration_samples)
        
        # Primary Gaussian component
        burst_shape = np.exp(-t_burst ** 2 / self.signal_config.burst_gaussian_sigma)
        
        # Add slight asymmetry
        asymmetry = 0.15 * np.exp(
            -(t_burst - self.signal_config.burst_asymmetry_offset) ** 2 / 
            self.signal_config.burst_asymmetry_sigma
        )
        burst_shape += asymmetry
        
        # Normalize and scale
        if np.max(burst_shape) > 0:
            burst_shape = burst_shape / np.max(burst_shape) * amplitude
        
        return burst_shape

    def _generate_clean_msna(
        self, burst_occurrences: List[Dict[str, Any]], n_samples: int, fs: int
    ) -> np.ndarray:
        """Generate clean MSNA signal."""
        msna_signal = np.zeros(n_samples)
        
        for burst in burst_occurrences:
            # Variable burst amplitude with gamma distribution
            burst_amplitude = (
                self.patient_config.signal_amplitude * 
                np.random.gamma(
                    shape = self.signal_config.burst_amplitude_shape, 
                    scale = self.signal_config.burst_amplitude_scale
                )
            )
            burst_amplitude *= burst["resp_modulation"]
            
            # Variable burst duration
            burst_duration = np.random.normal(
                self.patient_config.burst_duration_mean, 
                self.patient_config.burst_duration_std
            )
            burst_duration = np.clip(burst_duration, BURST_DURATION_MIN, BURST_DURATION_MAX)
            
            # Generate burst
            duration_samples = int(burst_duration * fs)
            burst_shape = self._generate_burst_shape(duration_samples, burst_amplitude)
            
            # Place burst in signal
            start_idx = int(burst["time"] * fs)
            end_idx = min(start_idx + duration_samples, n_samples)
            actual_samples = end_idx - start_idx
            
            if actual_samples > 0:
                msna_signal[start_idx:end_idx] += burst_shape[:actual_samples]
        
        return msna_signal
    
    def _add_realistic_noise(self, signal: np.ndarray, t: np.ndarray, n_samples: int, fs: int) -> np.ndarray:
        """Add realistic noise to clean signal."""
        noise_floor = self.patient_config.noise_floor
        
        # Add spike artifacts first (additive)
        spike_noise = generate_spike_artifacts(
            n_samples,
            noise_floor * self.signal_config.spike_artifact_amplitude,
            self.signal_config.spike_poisson_factor
        )
        
        # Pink noise (dominant biological component)
        pink_noise = generate_colored_noise(
            n_samples, fs, beta = 1.0, 
            amplitude = noise_floor * self.signal_config.pink_noise_amplitude,
            integration_smoothing = self.signal_config.integration_smoothing
        )
        
        # Frequency band limited noise
        low_freq, high_freq = NOISE_BANDS["low_freq"]
        lf_noise = generate_band_limited_noise(
            n_samples, fs, low_freq, high_freq, 
            noise_floor * self.signal_config.lf_noise_amplitude
        )
        low_freq, high_freq = NOISE_BANDS["mid_freq"]
        mf_noise = generate_band_limited_noise(
            n_samples, fs, low_freq, high_freq, 
            noise_floor * self.signal_config.mf_noise_amplitude
        )
        low_freq, high_freq = NOISE_BANDS["high_freq"]
        max_hf = min(high_freq, fs / 3)
        hf_noise = generate_band_limited_noise(
            n_samples, fs, low_freq, max_hf, 
            noise_floor * self.signal_config.hf_noise_amplitude
        )
        
        # Powerline interference
        powerline_noise = generate_powerline_noise(
            t,
            noise_floor * self.signal_config.powerline_60_amplitude,
            noise_floor * self.signal_config.powerline_120_amplitude
        )
        
        # Respiratory artifacts
        breathing_artifact = generate_respiratory_artifacts(
            t,
            self.patient_config.resp_rate,
            noise_floor * self.signal_config.breathing_artifact_amplitude
        )
        
        # Combine the noise sources
        total_noise = pink_noise + lf_noise + mf_noise + hf_noise + powerline_noise + breathing_artifact + spike_noise
        
        return signal + total_noise
    
    def simulate(self, duration: float, sampling_rate: int = 250, seed: Optional[int] = None) -> SimulationResults:
        """
        Run complete MSNA simulation.

        Args:
            duration: Recording duration in seconds
            sampling_rate: Sampling frequency in Hz
            seed: Random seed for reproducibility

        Returns:
            SimulationResults: Dataclass containing the simulation results
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")

        if sampling_rate < 100:
            raise ValueError("Sampling frequency must be at least 100 Hz")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create time array, an get the number of samples
        t = np.arange(0, duration, 1 / sampling_rate)
        n_samples = len(t)
        
        # Generate physiological timing
        r_times, rr_intervals = self._generate_cardiac_timing(duration)
        resp_modulation, resp_signal = self._generate_respiratory_modulation(t)
        
        # Generate bursts
        burst_occurrences = self._determine_burst_occurrences(r_times, resp_modulation, sampling_rate, duration)
        
        # Generate signals
        clean_msna = self._generate_clean_msna(burst_occurrences, n_samples, sampling_rate)
        noisy_msna = self._add_realistic_noise(clean_msna, t, n_samples, sampling_rate)
        
        return SimulationResults(
            time = t,
            clean_msna = clean_msna,
            noisy_msna = noisy_msna,
            respiratory_signal = resp_signal,
            respiratory_modulation = resp_modulation,
            r_times = r_times,
            rr_intervals = rr_intervals,
            burst_occurrences = burst_occurrences,
            fs = sampling_rate
        )


