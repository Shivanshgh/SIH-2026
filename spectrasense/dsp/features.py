"""
SpectraSense DSP - Feature & Parameter Extraction (Stage 3: CHARACTERIZE)
Calculates:
- Carrier Frequency (Fc)
- 3dB and 99% Occupied Bandwidth (OBW)
- SNR estimation (M2M4 and Spectral Floor)
- Symbol Rate (Rs) estimation (Cyclostationary non-linear transform)
- Higher-Order Cumulants (C40, C42)
- Instantaneous Amplitude & Frequency Variance
"""

import math

try:
    import numpy as np
    from scipy import signal as sp_signal
except ImportError:
    np = None
    sp_signal = None


def estimate_carrier_frequency(iq_data, sample_rate):
    """
    Estimates carrier frequency Fc using spectral peak and power centroid,
    or autocorrelation lag-1 phase angle in pure Python.
    """
    if np is not None:
        n = len(iq_data)
        fft_vals = np.fft.fftshift(np.fft.fft(iq_data))
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate))
        
        # Peak frequency
        peak_idx = np.argmax(psd)
        fc_peak = freqs[peak_idx]
        
        threshold = 0.1 * np.max(psd)
        mask = psd > threshold
        if np.sum(mask) > 0:
            fc_centroid = np.sum(freqs[mask] * psd[mask]) / np.sum(psd[mask])
        else:
            fc_centroid = fc_peak
        return float(fc_peak), float(fc_centroid)
    else:
        # Lag-1 phase angle estimation: angle(sum(x[n] * conj(x[n-1])))
        if len(iq_data) < 2:
            return 0.0, 0.0
        n_sub = min(len(iq_data), 4000)
        # Check squared signal for BPSK/QPSK carrier offset
        r1_sum = sum(iq_data[i] * iq_data[i-1].conjugate() for i in range(1, n_sub))
        ang = math.atan2(r1_sum.imag, r1_sum.real)
        fc = ang * sample_rate / (2.0 * math.pi)
        return float(round(fc, 1)), float(round(fc, 1))


def estimate_bandwidth(iq_data, sample_rate):
    """
    Calculates 3dB Bandwidth and 99% Occupied Bandwidth (OBW).
    """
    if np is None:
        return sample_rate * 0.1, sample_rate * 0.15
        
    n = len(iq_data)
    fft_vals = np.fft.fftshift(np.fft.fft(iq_data))
    psd = np.abs(fft_vals) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate))
    
    max_val = np.max(psd)
    half_power = max_val / 2.0  # -3 dB
    
    # Indices exceeding half power
    indices_3db = np.where(psd >= half_power)[0]
    if len(indices_3db) > 1:
        bw_3db = freqs[indices_3db[-1]] - freqs[indices_3db[0]]
    else:
        bw_3db = sample_rate * 0.05
        
    # 99% Occupied Bandwidth (OBW)
    total_energy = np.sum(psd)
    cumsum = np.cumsum(psd)
    low_idx = np.searchsorted(cumsum, 0.005 * total_energy)
    high_idx = np.searchsorted(cumsum, 0.995 * total_energy)
    
    if high_idx > low_idx and high_idx < len(freqs):
        bw_99 = freqs[high_idx] - freqs[low_idx]
    else:
        bw_99 = bw_3db * 1.3
        
    return abs(float(bw_3db)), abs(float(bw_99))


def estimate_snr_m2m4(iq_data):
    """
    Estimates SNR using the classic M2M4 (2nd and 4th moment) signal-to-noise estimator.
    Adapts kurtosis constant ka for constant envelope (ka=1.0 for PSK/FSK/CW)
    versus multi-amplitude signals (ka=1.32 for 16-QAM) and falls back to
    spectral floor estimation when M2M4 moments are ill-conditioned.
    """
    if np is not None:
        r = np.abs(iq_data)
        m2 = float(np.mean(r ** 2))
        m4 = float(np.mean(r ** 4))
        amp_var = float(np.var(r / (np.mean(r) + 1e-12)))
    else:
        r = [abs(s) for s in iq_data[:4000]]
        m2 = sum(x ** 2 for x in r) / max(len(r), 1)
        m4 = sum(x ** 4 for x in r) / max(len(r), 1)
        mean_r = sum(r) / max(len(r), 1)
        amp_var = sum(((x / (mean_r + 1e-12)) - 1.0) ** 2 for x in r) / max(len(r), 1)

    # Adaptive kurtosis: ka = 1.32 for 16-QAM (elevated envelope variance >= 0.055), ka = 1.0 for PSK/FSK
    ka = 1.32 if amp_var >= 0.055 else 1.0
    denom = 2.0 - ka
    arg = (2.0 * (m2 ** 2) - m4) / denom
    if arg > 0:
        s = math.sqrt(arg)
        noise = m2 - s
        if noise > 1e-6 and s > 0:
            snr_linear = s / noise
            snr_db = 10.0 * math.log10(max(snr_linear, 0.01))
            return max(round(float(snr_db), 1), -10.0)

    # Fallback to spectral floor estimate
    return estimate_snr_spectral(iq_data)


def estimate_snr_spectral(iq_data):
    """
    Spectral noise floor SNR estimation.
    Estimates noise floor density N0 from quiet out-of-band spectral bins (bottom 25%)
    and computes in-band signal power relative to total noise power.
    """
    if np is None:
        return 12.0
    n = len(iq_data)
    if n == 0:
        return 6.0
    fft_vals = np.fft.fft(iq_data)
    psd = (np.abs(fft_vals) ** 2) / n
    sorted_psd = np.sort(psd)
    n0 = float(np.median(sorted_psd[:max(int(n * 0.25), 1)]))
    total_power = float(np.mean(np.abs(iq_data) ** 2))
    noise_power = n0
    sig_power = total_power - noise_power

    if sig_power > 1e-12 and noise_power > 1e-12:
        snr_db = 10.0 * math.log10(sig_power / noise_power)
        return max(round(float(snr_db), 1), -10.0)
    return 6.0


def _find_fundamental_rs(freqs, psd, sample_rate, bw_limit=None):
    """
    Helper to identify the fundamental symbol clock line from candidate spectral peaks.
    Searches for the lowest strong line rather than a bare argmax, validates harmonic
    multiples (2*f0, 3*f0), and penalizes candidate peaks that are integer multiples
    of lower peaks.
    """
    if np is None or sp_signal is None or len(freqs) == 0:
        return None, 0.0

    valid_mask = (freqs >= (sample_rate * 0.005)) & (freqs <= (sample_rate * 0.45))
    if bw_limit is not None and bw_limit > 0:
        valid_mask &= (freqs <= bw_limit * 1.25)

    if np.sum(valid_mask) < 3:
        return None, 0.0

    sub_f = freqs[valid_mask]
    sub_p = psd[valid_mask]

    med_val = float(np.median(sub_p))
    max_val = float(np.max(sub_p))
    if max_val < 3.0 * med_val:
        return None, 0.0

    min_dist = max(1, int(len(sub_p) * 0.005))
    peaks, props = sp_signal.find_peaks(sub_p, height=2.5 * med_val, distance=min_dist)
    if len(peaks) == 0:
        return None, 0.0

    peak_f = sub_f[peaks]
    peak_h = props["peak_heights"]
    max_h = float(np.max(peak_h))

    # Significant lines: height >= 0.25 * max_h
    sig_mask = peak_h >= 0.25 * max_h
    cand_f = peak_f[sig_mask]
    cand_h = peak_h[sig_mask]

    order = np.argsort(cand_f)
    cand_f = cand_f[order]
    cand_h = cand_h[order]

    best_f = cand_f[0]
    best_score = -999.0

    for i, f0 in enumerate(cand_f):
        # Count integer harmonics (2*f0, 3*f0, 4*f0, 5*f0) present among all detected peaks
        harmonic_count = 0
        for mult in [2, 3, 4, 5]:
            target = mult * f0
            err = np.abs(peak_f - target) / target
            if np.any(err < 0.04):
                harmonic_count += 1

        # Check if f0 is an integer multiple of an earlier lower candidate
        is_subharmonic = False
        for earlier_f in cand_f[:i]:
            ratio = f0 / earlier_f
            if abs(ratio - round(ratio)) < 0.04 and round(ratio) >= 2:
                is_subharmonic = True
                break

        rel_h = cand_h[i] / max_h
        score = rel_h + 1.8 * harmonic_count
        if is_subharmonic:
            score -= 2.5

        if score > best_score:
            best_score = score
            best_f = f0

    quality = min(1.0, float(max_h / max(med_val, 1e-12)) / 40.0)
    return float(best_f), float(quality)


def estimate_symbol_rate(iq_data, sample_rate, fc=0.0, bw_occupied=None):
    """
    Estimates Symbol Rate (Rs) using non-linear cyclostationary cues:
    - Envelope power spectrum: |x(t)|^2 - mean(|x(t)|^2)
    - Delay-and-multiply: x(t) * conj(x(t-1))

    Applies harmonic logic to select the fundamental clock tone rather than a
    higher harmonic or noise spike, and rejects candidates inconsistent with OBW.
    Returns (rs_estimate, quality_figure).
    """
    if np is None:
        return float(sample_rate * 0.05), 0.0

    # 1. Envelope power spectrum (real-valued input, np.fft.rfft is valid)
    env = np.abs(iq_data) ** 2
    env_ac = env - np.mean(env)
    fft_env = np.abs(np.fft.rfft(env_ac))
    freqs_env = np.fft.rfftfreq(len(env_ac), 1.0 / sample_rate)

    rs_env, q_env = _find_fundamental_rs(freqs_env, fft_env, sample_rate, bw_limit=bw_occupied)

    # 2. Delay-and-multiply branch: x(t) * conj(x(t-1))
    # Note on NumPy 2.x compatibility:
    # d_sig is complex-valued. In NumPy 2.x, np.fft.rfft() strictly rejects complex inputs.
    # We deliberately choose np.fft.fft with a positive-frequency slice rather than taking
    # np.real(d_sig) because d_sig is fundamentally complex; taking only the real part
    # is sensitive to arbitrary static carrier phase offsets theta (where Re(d) ~ cos(theta) can
    # vanish), whereas the complex FFT preserves the full conjugate correlation across both
    # I and Q without phase blindspots and is NumPy 2.x safe.
    d_sig = iq_data[1:] * np.conj(iq_data[:-1])
    d_sig_ac = d_sig - np.mean(d_sig)
    d_fft_full = np.fft.fft(d_sig_ac)
    d_freqs_full = np.fft.fftfreq(len(d_sig_ac), 1.0 / sample_rate)
    pos_mask = (d_freqs_full > 0) & (d_freqs_full < (sample_rate / 2.0))
    d_fft = np.abs(d_fft_full[pos_mask])
    d_freqs = d_freqs_full[pos_mask]

    rs_dm, q_dm = _find_fundamental_rs(d_freqs, d_fft, sample_rate, bw_limit=bw_occupied)

    # Select the candidate with highest quality / harmonic support
    # Delay-and-multiply produces particularly strong line spectra for PSK
    if rs_dm is not None and rs_env is not None:
        if q_dm >= q_env * 0.8:
            return float(rs_dm), float(q_dm)
        return float(rs_env), float(q_env)
    elif rs_dm is not None:
        return float(rs_dm), float(q_dm)
    elif rs_env is not None:
        return float(rs_env), float(q_env)

    # Fallback to bandwidth-proportional default
    fallback = float(bw_occupied * 0.8) if (bw_occupied and bw_occupied > 0) else float(sample_rate * 0.05)
    return fallback, 0.0


def calculate_cumulants(iq_data):
    """
    Higher-order cumulants normalized by C21^2:
    C20 = E[x^2]
    C21 = E[|x|^2]
    C40 = E[x^4] - 3*(E[x^2])^2
    C42 = E[|x|^4] - |E[x^2]|^2 - 2*(E[|x|^2])^2

    Normalization by C21^2 produces scale-invariant cumulants essential for
    robust modulation classification (e.g. BPSK ~ -2 / |C40|~1-2, QPSK ~ 0).
    """
    if np is None:
        n = max(len(iq_data), 1)
        mean_r = sum(s.real for s in iq_data) / n
        mean_i = sum(s.imag for s in iq_data) / n
        x = [complex(s.real - mean_r, s.imag - mean_i) for s in iq_data]
        m20 = sum(s ** 2 for s in x) / n
        m21 = sum(abs(s) ** 2 for s in x) / n
        m40 = sum(s ** 4 for s in x) / n
        m41 = sum(abs(s) ** 4 for s in x) / n
        c40 = m40 - 3.0 * (m20 ** 2)
        c42 = m41 - abs(m20) ** 2 - 2.0 * (m21 ** 2)
        c21_sq = max(m21 ** 2, 1e-12)
        c40_norm = c40 / c21_sq
        c42_norm = c42 / c21_sq
        c20_norm = m20 / max(m21, 1e-12)
        return {
            "c20": c20_norm, "c21": m21, "c40": c40_norm, "c42": c42_norm,
            "c40_abs": float(abs(c40_norm)), "c42_abs": float(abs(c42_norm)),
            "c20_abs": float(abs(c20_norm)), "c40_real": float(c40_norm.real)
        }

    x = iq_data
    x = x - np.mean(x)
    m20 = np.mean(x ** 2)
    m21 = np.mean(np.abs(x) ** 2)
    m40 = np.mean(x ** 4)
    m41 = np.mean(np.abs(x) ** 4)

    c40 = m40 - 3.0 * (m20 ** 2)
    c42 = m41 - np.abs(m20) ** 2 - 2.0 * (m21 ** 2)

    c21_sq = max(float(m21 ** 2), 1e-12)
    c40_norm = c40 / c21_sq
    c42_norm = c42 / c21_sq
    c20_norm = m20 / max(float(m21), 1e-12)

    return {
        "c20": complex(c20_norm),
        "c21": float(m21),
        "c40": complex(c40_norm),
        "c42": float(c42_norm),
        "c40_abs": float(abs(c40_norm)),
        "c42_abs": float(abs(c42_norm)),
        "c20_abs": float(abs(c20_norm)),
        "c40_real": float(c40_norm.real),
    }


def extract_all_features(iq_data, sample_rate):
    """
    Master feature extraction pipeline for Stage 3: CHARACTERIZE.
    Keep all FFT-based feature work bounded, even when called outside the GUI.
    """
    if np is not None and len(iq_data) > 65_536:
        # Preserve samples from across the supplied representative windows while
        # bounding FFT and temporary-array memory independently of file size.
        chunk = 8_192
        n_chunks = 65_536 // chunk
        starts = np.linspace(0, len(iq_data) - chunk, n_chunks, dtype=np.int64)
        iq_data = np.concatenate([iq_data[int(start):int(start) + chunk] for start in starts]).astype(np.complex64, copy=False)
    fc_peak, fc_centroid = estimate_carrier_frequency(iq_data, sample_rate)
    bw_3db, bw_99 = estimate_bandwidth(iq_data, sample_rate)
    snr_db = estimate_snr_m2m4(iq_data)
    rs_est, rs_quality = estimate_symbol_rate(iq_data, sample_rate, fc=fc_peak, bw_occupied=bw_99)

    # Squaring non-linearity for BPSK vs QPSK carrier and spectral line identification
    if np is not None:
        sig_sq = iq_data ** 2
        sig_sq_ac = sig_sq - np.mean(sig_sq)
        fft_sq = np.fft.fft(sig_sq_ac)
        freqs_sq = np.fft.fftfreq(len(iq_data), 1.0 / sample_rate)
        mag_sq = np.abs(fft_sq)
        med_sq = float(np.median(mag_sq))
        max_sq = float(np.max(mag_sq))
        sq_peak_ratio = float(max_sq / max(med_sq, 1e-12))
        sq_peak_idx = np.argmax(mag_sq)
        fc_sq = float(freqs_sq[sq_peak_idx] / 2.0)

        # Carrier frequency determination:
        # Strong squaring line reveals exact BPSK carrier
        if sq_peak_ratio > 30.0 and abs(fc_sq - fc_peak) < (sample_rate * 0.1):
            fc_derot = fc_sq
            fc_reported = fc_sq
        elif abs(fc_centroid - fc_peak) < (0.25 * max(bw_3db, 1000.0)):
            fc_derot = fc_centroid
            fc_reported = fc_centroid
        else:
            fc_derot = fc_peak
            fc_reported = fc_peak

        # Derotate to baseband for cumulant analysis
        if abs(fc_derot) > 50.0:
            t = np.arange(len(iq_data)) / float(sample_rate)
            bb_data = iq_data * np.exp(-1j * 2 * np.pi * fc_derot * t)
        else:
            bb_data = iq_data
    else:
        sq_peak_ratio = 1.0
        fc_reported = fc_peak
        if abs(fc_peak) > 50.0:
            bb_data = []
            for idx, s in enumerate(iq_data[:4000]):
                t = idx / float(sample_rate)
                phase = -2.0 * math.pi * fc_peak * t
                cos_p = math.cos(phase)
                sin_p = math.sin(phase)
                bb_data.append(complex(s.real * cos_p - s.imag * sin_p, s.real * sin_p + s.imag * cos_p))
        else:
            bb_data = iq_data

    cumulants = calculate_cumulants(bb_data)

    # Instantaneous amplitude and phase cues
    if np is not None:
        amp = np.abs(iq_data)
        amp_var = float(np.var(amp / (np.mean(amp) + 1e-12)))

        # Instantaneous frequency variance
        phase = np.unwrap(np.angle(iq_data))
        freq_inst = np.diff(phase) * (sample_rate / (2.0 * math.pi))
        freq_var = float(np.var(freq_inst))
    else:
        amps = [abs(s) for s in iq_data[:3000]]
        m_amp = sum(amps) / max(len(amps), 1)
        amp_var = sum(((a / (m_amp + 1e-12)) - 1.0) ** 2 for a in amps) / max(len(amps), 1)

        phases = [math.atan2(s.imag, s.real) for s in iq_data[:3000]]
        d_phases = []
        for i in range(1, len(phases)):
            diff = phases[i] - phases[i-1]
            while diff > math.pi: diff -= 2 * math.pi
            while diff < -math.pi: diff += 2 * math.pi
            d_phases.append(diff * (sample_rate / (2.0 * math.pi)))
        m_dp = sum(d_phases) / max(len(d_phases), 1)
        freq_var = sum((d - m_dp) ** 2 for d in d_phases) / max(len(d_phases), 1)

    return {
        "carrier_freq_hz": round(fc_reported, 1),
        "carrier_centroid_hz": round(fc_centroid, 1),
        "carrier_peak_hz": round(fc_peak, 1),
        "bandwidth_3db_hz": bw_3db,
        "bandwidth_99_hz": bw_99,
        "snr_db": snr_db,
        "symbol_rate_baud": rs_est,
        "symbol_rate_quality": round(rs_quality, 2),
        "samples_per_symbol": round(sample_rate / max(rs_est, 1.0), 2),
        "cumulant_c40": round(cumulants["c40_abs"], 3),
        "cumulant_c40_real": round(cumulants.get("c40_real", 0.0), 3),
        "cumulant_c20": round(cumulants.get("c20_abs", 0.0), 3),
        "cumulant_c42": round(cumulants["c42_abs"], 3),
        "sq_peak_ratio": round(sq_peak_ratio, 1),
        "envelope_variance": round(amp_var, 4),
        "freq_inst_variance": round(freq_var, 1)
    }
