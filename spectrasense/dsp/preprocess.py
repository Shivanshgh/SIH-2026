"""
SpectraSense DSP - Pre-processing
Implements Stage 2: DC offset removal, energy normalization, and optional bandpass filtering.
"""

try:
    import numpy as np
    from scipy import signal as sp_signal
except ImportError:
    np = None
    sp_signal = None


def remove_dc_offset(iq_data):
    """Removes DC component by subtracting the complex mean."""
    if np is not None:
        mean_val = np.mean(iq_data)
        return iq_data - mean_val, mean_val
    else:
        n = len(iq_data)
        mean_r = sum(s.real for s in iq_data) / max(n, 1)
        mean_i = sum(s.imag for s in iq_data) / max(n, 1)
        mean_c = complex(mean_r, mean_i)
        return [s - mean_c for s in iq_data], mean_c


def normalize_signal(iq_data):
    """Normalizes signal to unit average power."""
    if np is not None:
        power = np.mean(np.abs(iq_data) ** 2)
        if power <= 0:
            return iq_data, 1.0
        scale = np.sqrt(power)
        return iq_data / scale, float(power)
    else:
        p_sum = sum(abs(s) ** 2 for s in iq_data)
        power = p_sum / max(len(iq_data), 1)
        scale = (power ** 0.5) if power > 0 else 1.0
        return [s / scale for s in iq_data], power


def bandpass_filter(iq_data, sample_rate, center_freq, bandwidth, num_taps=65):
    """
    Optional digital FIR bandpass filter around center frequency.
    """
    if np is None or sp_signal is None:
        return iq_data
    
    # Frequency bounds normalized to Nyquist
    nyquist = sample_rate / 2.0
    f_low = max(0.001, (center_freq - bandwidth / 2.0) / nyquist)
    f_high = min(0.999, (center_freq + bandwidth / 2.0) / nyquist)
    
    if f_low >= f_high or f_low <= 0 or f_high >= 1:
        return iq_data
        
    taps = sp_signal.firwin(num_taps, [f_low, f_high], pass_zero=False)
    
    # Filter I and Q independently
    filtered_i = sp_signal.lfilter(taps, 1.0, np.real(iq_data))
    filtered_q = sp_signal.lfilter(taps, 1.0, np.imag(iq_data))
    return filtered_i + 1j * filtered_q


def preprocess_pipeline(iq_data, sample_rate=200000, apply_dc=True, apply_norm=True, filter_params=None):
    """Executes the pre-processing chain."""
    metadata = {}
    sig = iq_data
    
    if apply_dc:
        sig, dc_val = remove_dc_offset(sig)
        metadata["dc_offset_removed"] = (float(np.real(dc_val)), float(np.imag(dc_val))) if np is not None else (dc_val.real, dc_val.imag)
        
    if apply_norm:
        sig, orig_power = normalize_signal(sig)
        metadata["original_power"] = orig_power
        
    if filter_params and filter_params.get("enabled"):
        sig = bandpass_filter(
            sig,
            sample_rate,
            filter_params.get("fc", 0),
            filter_params.get("bw", sample_rate * 0.4)
        )
        metadata["filtered"] = True
        
    return sig, metadata
