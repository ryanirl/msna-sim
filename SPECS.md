# MSNA Signal Simulation - Technical Specifications

## Overview

This document describes the technical implementation for simulating realistic
integrated Muscle Sympathetic Nerve Activity (MSNA) obtained through
microneurography recordings. The simulator generates physiologically accurate
integrated MSNA signals with realistic noise profiles suitable for algorithm
development, testing, and educational purposes.


## Physiological Background

### MSNA Characteristics

Muscle Sympathetic Nerve Activity (MSNA) is a direct measurement of the
sympathetic nervous system activity to skeletal muscle vasculature. Key
physiological characteristics include:

- **Cardiac Phase Locking**: MSNA bursts are phase-locked to the cardiac cycle, occurring during diastole (~1.3 seconds after the R-wave)
- **Burst Incidence**: Percentage of cardiac cycles containing sympathetic bursts (typically 30-80% in healthy adults)
- **Respiratory Modulation**: Sympathetic activity is modulated by respiration, with reduced activity during inspiration
- **Burst Morphology**: Individual bursts have symmetric, roughly Gaussian-shaped profiles after rectifying and integration. 

### Recording Methodology

MSNA is recorded using microneurography, where tungsten microelectrodes are
inserted into peripheral nerves (commonly the peroneal nerve). The raw signal
undergoes amplification, bandpass filtering, and integration to get the integrated
MSNA signal which is typically used for analysis.


## Signal Generation Methodology

### Cardiac Timing Generation

The simulator generates realistic ECG R-wave timing with physiological heart
rate variability (HRV). The approach models natural cardiac rhythm variations
observed in healthy individuals through normally distributed RR intervals with
appropriate variability.

$$\text{RR}_{\text{intervals}} \sim \mathcal{N}(\text{mean}_{\text{RR}}, \sigma_{\text{HRV}})$$

Where $\text{mean}_{\text{RR}} = 60 / \text{heart rate}$ (seconds) and
$\sigma_{\text{HRV}} \approx 0.04$ seconds (40ms standard deviation).
Physiological constraints ensure all intervals fall within realistic bounds
(0.4s < RR < 1.5s). The implementation uses a cumulative summation of these
normally distributed RR intervals with clipping to reasonable physiological
limits, creating a realistic sequence of R-wave times that serves as the
temporal foundation for the sympathetic burst occurrences.


### Respiratory Modulation

Respiratory influence on sympathetic activity represents one of the most
important physiological coupling mechanisms in autonomic control. The simulator
models this through sinusoidal modulation with irregular breathing patterns,
reflecting the central respiratory-cardiovascular coupling observed in humans.

$$\text{modulation}(t) = 0.6 + 0.4 \times \frac{1 + \cos(2\pi f_{\text{resp}} \times t + \phi(t))}{2}$$


Where $f_{\text{resp}} = \text{respiratory rate} / 60$ (Hz) and $\phi(t)$
represents breathing irregularity through a harmonic distortion.  Inspiration
corresponds to the negative phase of the cosine function, resulting in reduced
sympathetic outflow during inspiratory phases. This coupling ensures that burst
probability varies systematically with the respiratory cycle, mimicking the
inhibitory effects of lung inflation on sympathetic nerve activity.


### Burst Occurrence Determination

MSNA bursts cannot occur more frequently than once per cardiac cycle due to the
refractory period of the sympathetic nervous system and its tight coupling to
cardiac timing.

$$P_{\text{burst}} = \frac{\text{burst incidence}}{100} \times \text{respiratory modulation}(t_{\text{cycle}})$$


For each cardiac cycle, the simulator evaluates burst probability based on the
target burst incidence modulated by current respiratory phase. This ensures that
respiratory coupling affects not just burst amplitude but also burst occurrence
probability. Within each cardiac cycle that receives a burst, timing follows a
normal distribution centered at 1.3 seconds post-R-wave with 0.15 seconds
standard deviation, constrained to fall within [0.8s, RR_interval - 0.2s].


### Burst Morphology Generation

Through studying real MSNA signals, we find (to out surprise) that MSNA bursts
exhibits a more symmetric and Gaussian-like profiles rather than the sharp rise
and exponential decay. The simulator generates these realistic burst shapes
through a combination of primary Gaussian components with subtle asymmetric
features that reflect the complex neural population dynamics underlying each
burst.

$$\text{burst}(t) = \exp\left(-\frac{t^2}{\sigma^2}\right) + \text{asymmetry component}(t)$$


The primary component uses a Gaussian with $\sigma \approx 0.8$, while the asymmetry
component adds a secondary Gaussian offset in time to create the slight
irregularities observed in real recordings. Burst duration follows
$\mathcal{N}(0.5, 0.1)$ seconds, clipped to [0.3, 0.8]s. Amplitude varies
exponentially with additional respiratory modulation, ensuring that bursts
occurring during expiration tend to be larger than those during inspiration.


## Noise Modeling

### Multi-Frequency Noise Architecture

Real MSNA recordings contain many distinct distributions of noise, with complex
nuances, and of multiple frequency ranges. The simulator tries to be as 
comprehensive as possible, and models noise that captures this complexity
through frequency-specific components rather than simple white noise addition.


### Pink Noise Component

Pink noise (1/f noise) represents the dominant component in most biological
signals, arising from the intrinsic variability of biological systems across
multiple time scales. This component provides the foundational noise floor that
characterizes all neural recordings. The implementation uses exponential
smoothing of white noise to create the characteristic 1/f power spectral
density:

$$
\text{filtered noise}[i] = \alpha \times \text{white noise}[i] + (1-\alpha) \times \text{filtered noise}[i-1]
$$

Where $\alpha = 0.05$ provides appropriate correlation structure. This component
provides the pervasive background noise that gives MSNA recordings their
characteristic "fuzzy" appearance. The pink noise ensures that low frequencies
have higher power than high frequencies, matching the spectral characteristics
observed in real neural recordings.


### Band-Limited Noise Components

Different frequency bands in MSNA recordings correspond to distinct
physiological and technical noise sources, each requiring specialized modeling
approaches to capture their unique characteristics. We generate four frequency
bands of noise, that are generated through bandpass filtering white noise. 
* Low frequency noise (0.01-0.5 Hz)
* Mid frequency noise (0.5-10 Hz)
* High frequency noise (50-200 Hz)

### Structured Artifacts

Beyond random noise components, MSNA recordings contain structured artifacts
that arise from specific environmental and physiological sources. These
artifacts have characteristic patterns that distinguish them from random noise.

Powerline interference represents one of the most common structured artifacts in
biomedical recordings. The simulator includes 60 Hz fundamental frequency
interference at 3% of noise floor, along with 120 Hz harmonic at 1.5% of noise
floor. These sinusoidal components create a characteristic periodic
interference pattern.

Breathing artifacts manifest as very low frequency modulation synchronized with
respiratory cycles. Beyond the physiological respiratory modulation of
sympathetic activity itself, mechanical effects of breathing create additional
signal variations through electrode movement, tissue displacement, and changes
in electrical impedance. This component operates at the respiratory frequency
with 10% noise floor amplitude, creating slow oscillations.

Burst-like artifacts occur sporadically throughout recordings and represent
transient events such as muscle twitches, electrode movement, or electrical
transients. These artifacts have exponential decay profiles with sinusoidal
modulation, making them superficially similar to actual MSNA bursts but lacking
the precise cardiac timing relationship. Their occurrence follows Poisson
statistics based on the noise floor parameter, with higher noise floors
corresponding to more frequent artifacts.


### Noise Floor Parameter Control

The noise floor parameter provides general control over signal quality, scaling
all noise components proportionally. In general, values of 0.1 represent clean
recordings with good electrode contact and minimal interference, while values
around 0.3 represent typical clinical recordings with moderate noise. Value of 
0.5 or higher represent poor quality recordings with significant interference,
movement artifacts, or suboptimal electrode placement.



