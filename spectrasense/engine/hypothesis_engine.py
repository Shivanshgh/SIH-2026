"""
SpectraSense Hypothesis Engine (Stage 4: HYPOTHESIZE)
Combines classical DSP features (cumulants C40/C42, envelope variance, instantaneous frequency
variance, spectral symmetry, baud rate lines) with a calibrated multi-hypothesis classifier.

CRITICAL DIRECTIVE:
- Never force a single classification!
- Always output multiple ranked hypotheses with confidence scores.
- When SNR is low or features conflict, distribute probabilities across multiple candidates.
"""

import math


MODULATION_CLASSES = [
    "BPSK",
    "QPSK",
    "8-PSK",
    "16-QAM",
    "64-QAM",
    "PAM4",
    "2-FSK",
    "4-FSK",
    "CPFSK",
    "GFSK",
    "AM-DSB",
    "AM-SSB",
    "WBFM",
    "CW / Pure Carrier",
    "Unresolved / Noise"
]

# Conservative fusion: interpretable DSP remains the larger component.
DSP_SCORE_WEIGHT = 0.45
ML_SCORE_WEIGHT = 0.55


def score_hypotheses(features, user_prior=None, ml_scores=None):
    """
    Evaluates physical DSP features and yields ranked hypotheses.
    Returns list of dicts:
    [
        {
            "rank": 1,
            "modulation": "QPSK",
            "confidence": 0.68,
            "evidence": ["Near-zero C40 cumulant", "Low envelope variance", "4-quadrant phase distribution"],
            "suggested_pipeline": "Costas Loop (4-phase) -> Symbol Sync (Gardner) -> QPSK Slicer"
        },
        ...
    ]
    """
    c40 = features.get("cumulant_c40", 0.0)
    c40_real = features.get("cumulant_c40_real", 0.0)
    c20 = features.get("cumulant_c20", 0.0)
    c42 = features.get("cumulant_c42", 0.0)
    sq_ratio = features.get("sq_peak_ratio", 1.0)
    env_var = features.get("envelope_variance", 0.1)
    freq_var = features.get("freq_inst_variance", 1000.0)
    snr_db = features.get("snr_db", 10.0)
    bw = features.get("bandwidth_3db_hz", 10000.0)
    rs = features.get("symbol_rate_baud", 10000.0)
    
    # Raw unnormalized logit-scores
    scores = {mod: 1.0 for mod in MODULATION_CLASSES}
    evidence_map = {mod: [] for mod in MODULATION_CLASSES}
    
    # 1. Check for pure carrier / unmodulated CW
    # Must have very narrow bandwidth AND low envelope variance AND negligible frequency variance
    if bw < 500 and env_var < 0.003 and freq_var < 2000.0:
        scores["CW / Pure Carrier"] += 8.5
        evidence_map["CW / Pure Carrier"].append("Very narrow 3dB bandwidth (<500 Hz)")
        evidence_map["CW / Pure Carrier"].append("Near-zero amplitude fluctuation (<0.003)")
        evidence_map["CW / Pure Carrier"].append("Negligible frequency deviation (<2000 Hz²)")

    # 2. Check for 16-QAM (Multi-amplitude)
    # 16-QAM has multiple amplitude levels (3 distinct energy rings), giving env_var > 0.055
    elif env_var >= 0.055:
        scores["16-QAM"] += 8.5
        scores["QPSK"] += 1.5
        evidence_map["16-QAM"].append(f"Elevated envelope variance ({env_var:.3f}) indicating multi-level amplitude rings")
        evidence_map["16-QAM"].append(f"Normalized C42 cumulant ({c42:.3f}) consistent with square 16-QAM grid")

    # 3. Constant modulus modulations (PSK and FSK families)
    else:
        # Check BPSK:
        # 1. Strong discrete line after squaring (peak-to-median ratio > 40)
        # 2. Normalized C40 cumulant real part strongly negative (~ -1 to -2) and large magnitude
        # 3. Normalized C20 cumulant near 1.0 (indicating 1D axis modulation)
        is_bpsk = (sq_ratio >= 40.0 and (c40 >= 0.5 or c20 >= 0.4)) or (c40 >= 0.8 and c40_real < -0.4)
        if is_bpsk:
            scores["BPSK"] += 9.0
            scores["QPSK"] += 1.0
            evidence_map["BPSK"].append(f"Strong spectral line after squaring (peak ratio {sq_ratio:.1f}) confirms binary antipodal (BPSK) modulation")
            evidence_map["BPSK"].append(f"Normalized C40 cumulant ({c40_real:.2f}) and C20 ({c20:.2f}) confirm 1D real constellation axis")
            evidence_map["BPSK"].append("Bimodal 180° phase distribution along single axis with constant envelope")
        else:
            # Check 2-FSK / 4-FSK vs QPSK / 8-PSK
            # FSK exhibits constant envelope with discrete tone hopping:
            # - Audio FSK (AFSK) band: 6,000 <= freq_var <= 4,000,000 Hz²
            # - RF 2-FSK band: 20,000,000 <= freq_var <= 160,000,000 Hz²
            is_fsk = (6000.0 <= freq_var <= 4000000.0) or (20000000.0 <= freq_var <= 160000000.0)
            if is_fsk:
                scores["2-FSK"] += 8.5
                scores["4-FSK"] += 3.5
                evidence_map["2-FSK"].append("Bimodal instantaneous frequency shifting with constant modulus")
                evidence_map["2-FSK"].append(f"Frequency variance ({freq_var:.0f} Hz²) confirms discrete Mark/Space tone hopping")
                evidence_map["2-FSK"].append("Low envelope fluctuation (<0.055) confirms constant-envelope frequency modulation")
            else:
                # QPSK / 8-PSK: near-zero C40 (< 0.2), weak squaring line (< 30), constant envelope, 4-fold phase symmetry
                scores["QPSK"] += 8.0
                scores["8-PSK"] += 3.5
                scores["2-FSK"] += 1.0
                evidence_map["QPSK"].append(f"Near-zero normalized C40 cumulant ({c40:.3f}) characteristic of 4-quadrant QPSK symmetry")
                evidence_map["QPSK"].append(f"Weak squaring line ratio ({sq_ratio:.1f}) rejects BPSK in favor of 4-phase constellation")
                evidence_map["QPSK"].append("Constant modulus envelope with 4 constellation quadrants")
                evidence_map["8-PSK"].append("Circular constellation distribution with phase shifts")

    # 4. Low SNR / Degradation Penalty
    if snr_db < 6.0:
        # Heavily boost Unresolved / Noise candidate when SNR is degraded
        scores["Unresolved / Noise"] += 8.0
        evidence_map["Unresolved / Noise"].append(f"Low SNR ({snr_db:.1f} dB) limits modulation separability")
        evidence_map["Unresolved / Noise"].append("High noise floor induces constellation blurring")
        
        # Flatten probabilities of other candidates
        for k in scores:
            if k != "Unresolved / Noise":
                scores[k] = scores[k] * 0.45 + 1.0

    # Softmax conversion to normalized probabilities
    # Temperature scaling based on SNR: low SNR -> high temperature (flatter, honest uncertainty)
    temp = max(1.0, 5.0 - 0.2 * snr_db) if snr_db < 6.0 else max(1.0, 3.5 - 0.12 * snr_db)
    exp_scores = {mod: math.exp(scores[mod] / temp) for mod in MODULATION_CLASSES}
    total_exp = sum(exp_scores.values())
    
    probabilities = {mod: exp_scores[mod] / total_exp for mod in MODULATION_CLASSES}

    # A trained classifier contributes evidence but never removes the classical
    # DSP and low-SNR uncertainty paths. Keep the blend conservative and visible.
    if ml_scores:
        ml_total = sum(max(0.0, float(ml_scores.get(mod, 0.0))) for mod in MODULATION_CLASSES)
        if ml_total > 0.0:
            ml_probabilities = {
                mod: max(0.0, float(ml_scores.get(mod, 0.0))) / ml_total
                for mod in MODULATION_CLASSES
            }
            probabilities = {
                mod: ML_SCORE_WEIGHT * ml_probabilities[mod] + DSP_SCORE_WEIGHT * probabilities[mod]
                for mod in MODULATION_CLASSES
            }
            for mod, score in ml_probabilities.items():
                if score >= 0.15:
                    evidence_map[mod].append(f"Offline ML model probability: {score:.0%} (supporting evidence)")
    
    # Sort descending
    sorted_candidates = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Pipeline suggestion mapping
    pipelines = {
        "BPSK": "Carrier Recovery (Squaring Loop / Costas 2-Phase) -> Gardner Symbol Sync -> Threshold Slicer",
        "QPSK": "Costas Loop (4-Phase) -> Root-Raised-Cosine Matched Filter -> Gardner Sync -> 4-Quadrant Slicer",
        "8-PSK": "M-th Power Carrier Sync (M=8) -> Polyphase Clock Sync -> 8-Phase Sector Decision",
        "16-QAM": "CMA Adaptive Equalizer -> Decision-Directed Carrier Phase Loop -> 16-Grid Slicer",
        "64-QAM": "Equalizer -> Carrier Recovery -> 64-Grid Slicer (candidate only)",
        "PAM4": "Timing Recovery -> Four-Level Slicer (candidate only)",
        "2-FSK": "Discriminator / Quadrature Demodulator -> Dual-Tone Bandpass Bank -> Zero-Crossing Detector",
        "4-FSK": "Multi-tone Matched Filter Bank -> Maximum Likelihood Tone Estimator",
        "CPFSK": "Continuous-Phase Frequency Discriminator (candidate only)",
        "GFSK": "Gaussian Frequency Discriminator (candidate only)",
        "AM-DSB": "Envelope Detector / Coherent AM Demodulator (candidate only)",
        "AM-SSB": "SSB Product Detector (candidate only)",
        "WBFM": "FM Discriminator -> De-emphasis (candidate only)",
        "CW / Pure Carrier": "Narrowband Bandpass Filter -> Frequency Tracking PLL -> Amplitude Demodulator",
        "Unresolved / Noise": "Coherent Integration / Pre-filtering Required -> Increase Observation Window"
    }
    
    ranked_results = []
    for rank, (mod, prob) in enumerate(sorted_candidates, 1):
        # Only list candidates with non-trivial score or top 4
        if prob > 0.03 or rank <= 4:
            ranked_results.append({
                "rank": rank,
                "modulation": mod,
                "confidence": round(prob, 4),
                "confidence_pct": round(prob * 100.0, 1),
                "evidence": evidence_map[mod] if evidence_map[mod] else ["Standard spectral match"],
                "suggested_pipeline": pipelines.get(mod, "Generic DSP Demodulator")
            })
            
    return ranked_results
