"""Small, explicit rate-1/2 K=3 convolutional-code demonstration."""

import random
from spectrasense.engine.fec import (analyze_fec, block_deinterleave,
                                     block_interleave, convolutional_encode,
                                     viterbi_decode)


_block_interleave = block_interleave
_block_deinterleave = block_deinterleave


def block_interleave(bits, rows=8):
    """Compatibility wrapper retaining the demo's original return shape."""
    return _block_interleave(bits, rows), len(bits)


def block_deinterleave(bits, original_length, rows=8):
    return _block_deinterleave(bits, rows)[:original_length]


def run_coding_demo(seed=2026, payload_bits=96, flip_probability=0.035):
    rng = random.Random(seed)
    payload = [rng.randrange(2) for _ in range(payload_bits)]
    uncoded_noisy = [bit ^ int(rng.random() < flip_probability) for bit in payload]
    terminated = payload + [0, 0]
    encoded = convolutional_encode(terminated)
    interleaved = block_interleave(encoded, rows=8)
    noisy = [bit ^ int(rng.random() < flip_probability) for bit in interleaved]
    restored = block_deinterleave(noisy, rows=8)[:len(encoded)]
    decoded = viterbi_decode(restored)[:payload_bits]
    coded_errors = sum(a != b for a, b in zip(decoded, payload))
    return {
        "scheme": "rate-1/2 convolutional code, K=3, generators (7,5) octal; 8-row block interleaver",
        "payload_bits": payload_bits,
        "channel_bit_flip_probability": flip_probability,
        "interleaved_channel_bit_errors": sum(a != b for a, b in zip(noisy, interleaved)),
        "uncoded_payload_bit_errors": sum(a != b for a, b in zip(uncoded_noisy, payload)),
        "decoded_payload_bit_errors": coded_errors,
        "payload_recovered_exactly": coded_errors == 0,
        "transmitted_bits_preview": "".join(map(str, payload[:64])),
        "decoded_bits_preview": "".join(map(str, decoded[:64])),
        "blind_candidate_search": analyze_fec("".join(map(str, noisy))),
        "warning": "Synthetic known-parameter demo only; this does not infer or decode coding on an input capture.",
    }
