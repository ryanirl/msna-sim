# MSNA-Sim: Realistic MSNA Signal Simulation
# 
# A library for generating synthetic Muscle Sympathetic Nerve Activity (MSNA)
# data with realistic and physiologically accurate characteristics.

__version__ = "0.1.2"
__author__ = "Ryan 'RyanIRL' Peters"
__email__ = "ryanirl@icloud.com"

# Core functionality
from .simulator import Simulation
from .simulator import SimulationResults
from .config import PatientConfig
from .config import SignalConfig
from .config import create_preset_config

from typing import Optional

# Constants for advanced users
from .constants import (
    RR_INTERVAL_MIN,
    RR_INTERVAL_MAX, 
    BURST_DURATION_MIN,
    BURST_DURATION_MAX,
    NOISE_BANDS
)


def quick_simulation(
    duration: float = 60.0, 
    preset: str = "normal_adult", 
    sampling_rate: int = 1000, 
    seed: Optional[int] = None, 
    **kwargs
) -> SimulationResults:
    """
    Convenience function for quick simulations with sensible defaults.
    
    Args:
        duration: Recording duration in seconds
        preset: Configuration preset name  
        sampling_rate: Sampling frequency in Hz
        seed: Random seed for reproducibility
        **kwargs: Additional parameters passed to PatientConfig
    
    Returns:
        SimulationResults object
    """
    patient_config = create_preset_config(preset)
    
    # Allow parameter overrides
    if kwargs:
        config_dict = patient_config.__dict__.copy()
        config_dict.update(kwargs)
        patient_config = PatientConfig(**config_dict)
    
    simulation = Simulation(patient_config = patient_config)
    return simulation.simulate(duration = duration, sampling_rate = sampling_rate, seed = seed)


__all__ = [
    "Simulation",
    "SimulationResults", 
    "PatientConfig",
    "SignalConfig",
    "create_preset_config",
    "quick_simulation",
    "RR_INTERVAL_MIN",
    "RR_INTERVAL_MAX",
    "BURST_DURATION_MIN", 
    "BURST_DURATION_MAX",
    "NOISE_BANDS"
]


