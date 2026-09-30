"""
SpectraSense - Synthetic Signal Generator for SIH 2026 (PS 26147)
Generates real binary .iq (complex float32) and .wav signals with known ground truth
to demonstrate the 4-stage pipeline:
  CHARACTERIZE -> HYPOTHESIZE -> PROCESS -> VALIDATE
Includes clear cases and one purposefully ambiguous low-SNR case.
"""

import os
import math
import random
import struct
import wave

SAMPLE_RATE = 200_000  # 200 kSps standard test rate
NUM_SAMPLES = 20_000   # 100 ms of observation


def gaussian_noise(std_dev=1.0):
    # Box-Muller transform
    u1 = max(random.random(), 1e-12)
    u2 = random.random()
    r = math.sqrt(-2.0 * math.log(u1)) * std_dev
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta)


def generate_bpsk(filepath, fc=25000, rs=10000, snr_db=18):
    """BPSK with root raised cosine shape approximation + carrier offset."""
    sps = int(SAMPLE_RATE / rs)
    num_symbols = NUM_SAMPLES // sps + 2
    bits = [1 if random.random() > 0.5 else -1 for _ in range(num_symbols)]
    
    noise_power = 10.0 ** (-snr_db / 10.0)
    noise_std = math.sqrt(noise_power / 2.0)
    
    with open(filepath, 'wb') as f:
        for i in range(NUM_SAMPLES):
            sym_idx = i // sps
            base_val = bits[sym_idx]
            
            # Carrier modulation
            t = i / SAMPLE_RATE
            phase = 2.0 * math.pi * fc * t
            
            i_sig = base_val * math.cos(phase)
            q_sig = base_val * math.sin(phase)
            
            # Add AWGN
            n_i, n_q = gaussian_noise(noise_std)
            f.write(struct.pack('<ff', i_sig + n_i, q_sig + n_q))


def generate_qpsk(filepath, fc=40000, rs=20000, snr_db=16):
    """QPSK with 4 distinct constellation states and carrier offset."""
    sps = int(SAMPLE_RATE / rs)
    num_symbols = NUM_SAMPLES // sps + 2
    constellation = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
    syms = [random.choice(constellation) for _ in range(num_symbols)]
    
    noise_power = 10.0 ** (-snr_db / 10.0)
    noise_std = math.sqrt(noise_power / 2.0)
    
    with open(filepath, 'wb') as f:
        for i in range(NUM_SAMPLES):
            sym_idx = i // sps
            i_sym, q_sym = syms[sym_idx]
            
            # Carrier
            t = i / SAMPLE_RATE
            phase = 2.0 * math.pi * fc * t
            cos_p = math.cos(phase)
            sin_p = math.sin(phase)
            
            # Complex mixing: (I + jQ) * exp(j*phase)
            i_mix = i_sym * cos_p - q_sym * sin_p
            q_mix = i_sym * sin_p + q_sym * cos_p
            
            n_i, n_q = gaussian_noise(noise_std)
            f.write(struct.pack('<ff', i_mix * 0.707 + n_i, q_mix * 0.707 + n_q))


def generate_2fsk(filepath, f_mark=30000, f_space=45000, rs=5000, snr_db=15):
    """Continuous-phase 2FSK with distinct tone separation."""
    sps = int(SAMPLE_RATE / rs)
    num_symbols = NUM_SAMPLES // sps + 2
    bits = [1 if random.random() > 0.5 else 0 for _ in range(num_symbols)]
    
    noise_power = 10.0 ** (-snr_db / 10.0)
    noise_std = math.sqrt(noise_power / 2.0)
    
    phase = 0.0
    with open(filepath, 'wb') as f:
        for i in range(NUM_SAMPLES):
            sym_idx = i // sps
            freq = f_mark if bits[sym_idx] == 1 else f_space
            phase += 2.0 * math.pi * freq * (1.0 / SAMPLE_RATE)
            
            i_sig = math.cos(phase)
            q_sig = math.sin(phase)
            
            n_i, n_q = gaussian_noise(noise_std)
            f.write(struct.pack('<ff', i_sig + n_i, q_sig + n_q))


def generate_16qam(filepath, fc=0, rs=15000, snr_db=22):
    """16-QAM baseband complex signal with 16 constellation points."""
    sps = int(SAMPLE_RATE / rs)
    levels = [-3.0, -1.0, 1.0, 3.0]
    num_symbols = NUM_SAMPLES // sps + 2
    syms = [(random.choice(levels), random.choice(levels)) for _ in range(num_symbols)]
    
    # Normalize power: avg power of 16-QAM with levels [-3,-1,1,3] is (9+1)/2 = 5 -> scale by 1/sqrt(10)
    scale = 1.0 / math.sqrt(10.0)
    noise_power = 10.0 ** (-snr_db / 10.0)
    noise_std = math.sqrt(noise_power / 2.0)
    
    with open(filepath, 'wb') as f:
        for i in range(NUM_SAMPLES):
            sym_idx = i // sps
            i_sym, q_sym = syms[sym_idx]
            
            i_val = i_sym * scale
            q_val = q_sym * scale
            
            n_i, n_q = gaussian_noise(noise_std)
            f.write(struct.pack('<ff', i_val + n_i, q_val + n_q))


def generate_ambiguous_signal(filepath):
    """
    CRITICAL SIH TEST CASE:
    Ambiguous / Degraded signal with:
    - Low SNR (~3 dB)
    - Strong multipath / phase jitter
    - Overlapping interference tone
    This forces the hypothesis engine to output close confidence scores (e.g. QPSK 42% vs 8PSK 38%),
    causing validation EVM to fail thresholds and legitimately transition to
    'Unresolved / Inconclusive' state with explicit uncertainty notes.
    """
    sps = int(SAMPLE_RATE / 12000)
    num_symbols = NUM_SAMPLES // sps + 2
    # Jittery phase constellation
    constellation = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
    syms = [random.choice(constellation) for _ in range(num_symbols)]
    
    noise_std = 0.65  # Very high noise floor (~3-4 dB SNR)
    carrier_freq = 28000
    interferer_freq = 32000
    
    with open(filepath, 'wb') as f:
        phase_jitter = 0.0
        for i in range(NUM_SAMPLES):
            sym_idx = i // sps
            i_sym, q_sym = syms[sym_idx]
            
            t = i / SAMPLE_RATE
            phase_jitter += (random.random() - 0.5) * 0.1
            phase = 2.0 * math.pi * carrier_freq * t + phase_jitter
            
            i_mix = i_sym * math.cos(phase) - q_sym * math.sin(phase)
            q_mix = i_sym * math.sin(phase) + q_sym * math.cos(phase)
            
            # Interference CW tone
            i_interf = 0.45 * math.cos(2.0 * math.pi * interferer_freq * t)
            q_interf = 0.45 * math.sin(2.0 * math.pi * interferer_freq * t)
            
            n_i, n_q = gaussian_noise(noise_std)
            
            f.write(struct.pack('<ff', 0.5 * i_mix + i_interf + n_i, 0.5 * q_mix + q_interf + n_q))


def generate_wav(filepath, f_mark=1200, f_space=2200, baud=300):
    """Standard 16-bit PCM WAV audio file (Bell 202 style FSK) for .wav input demonstration."""
    sample_rate = 44100
    duration_sec = 1.0
    total_samples = int(sample_rate * duration_sec)
    sps = int(sample_rate / baud)
    
    bits = [1 if random.random() > 0.5 else 0 for _ in range(total_samples // sps + 2)]
    
    phase = 0.0
    audio_data = bytearray()
    
    for i in range(total_samples):
        sym_idx = i // sps
        freq = f_mark if bits[sym_idx] == 1 else f_space
        phase += 2.0 * math.pi * freq / sample_rate
        sample_val = int(math.sin(phase) * 20000.0)
        audio_data.extend(struct.pack('<h', sample_val))
        
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating synthetic test signals into {out_dir}...")
    generate_bpsk(os.path.join(out_dir, "bpsk_25k_18db.iq"))
    generate_qpsk(os.path.join(out_dir, "qpsk_40k_16db.iq"))
    generate_2fsk(os.path.join(out_dir, "2fsk_30k_15db.iq"))
    generate_16qam(os.path.join(out_dir, "16qam_bb_22db.iq"))
    generate_ambiguous_signal(os.path.join(out_dir, "ambiguous_degraded_case.iq"))
    generate_wav(os.path.join(out_dir, "audio_fsk_sample.wav"))
    
    print("Done! Generated 5 .iq files and 1 .wav file.")


if __name__ == "__main__":
    main()
