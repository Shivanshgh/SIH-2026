"""Bounded hard-bit search over a small set of convolutional/FEC profiles.

This is a candidate detector, not a general blind FEC decoder. It deliberately
returns unresolved when the observed bits do not strongly fit the supported code.
"""

import math

from spectrasense.engine.reed_solomon import search_rs_profiles
from spectrasense.engine.ldpc import find_blind_80211n_648_rate_half


CONVOLUTIONAL_PROFILES = (
    ("rate-1/2 convolutional, K=3, generators (7,5) octal", (0o7, 0o5)),
    ("rate-1/2 convolutional, K=7, generators (171,133) octal", (0o171, 0o133)),
)


def convolutional_encode(bits, generators=(0o7, 0o5)):
    memory = max(generators).bit_length() - 1
    state_mask = (1 << memory) - 1
    state = 0
    output = []
    for bit in bits:
        register = (state << 1) | int(bit)
        output.extend((register & polynomial).bit_count() & 1 for polynomial in generators)
        state = register & state_mask
    return output


def viterbi_decode(encoded_bits, generators=(0o7, 0o5)):
    if len(encoded_bits) % 2:
        raise ValueError("Rate-1/2 input must contain bit pairs")
    memory = max(generators).bit_length() - 1
    state_count, state_mask = 1 << memory, (1 << memory) - 1
    metrics = [0.0] + [float("inf")] * (state_count - 1)
    history = []
    for i in range(0, len(encoded_bits), 2):
        received = encoded_bits[i:i + 2]
        next_metrics = [float("inf")] * state_count
        back = [None] * state_count
        for state, cost in enumerate(metrics):
            if not math.isfinite(cost):
                continue
            for bit in (0, 1):
                register = (state << 1) | bit
                target = register & state_mask
                expected = tuple((register & polynomial).bit_count() & 1 for polynomial in generators)
                metric = cost + sum(a != b for a, b in zip(received, expected))
                if metric < next_metrics[target]:
                    next_metrics[target] = metric
                    back[target] = (state, bit)
        metrics = next_metrics
        history.append(back)
    state = min(range(state_count), key=lambda item: metrics[item])
    decoded = []
    for back in reversed(history):
        previous, bit = back[state]
        decoded.append(bit)
        state = previous
    return decoded[::-1]


def _deinterleave(bits, rows, layout="rectangular"):
    if rows == 1 and layout == "rectangular":
        return bits
    columns = len(bits) // rows
    usable = rows * columns
    restored = [0] * usable
    index = 0
    for column in range(columns):
        for row in range(rows):
            target_column = (column + row) % columns if layout == "diagonal" else column
            restored[row * columns + target_column] = bits[index]
            index += 1
    return restored


def block_interleave(bits, rows=8, layout="rectangular"):
    """Interleave a bounded block using rectangular or diagonal row offsets."""
    if rows < 1 or layout not in ("rectangular", "diagonal"):
        raise ValueError("Use positive rows and rectangular/diagonal layout")
    padded = list(map(int, bits))
    padded.extend([0] * (-len(padded) % rows))
    columns = len(padded) // rows
    return [padded[row * columns + ((column + row) % columns if layout == "diagonal" else column)]
            for column in range(columns) for row in range(rows)]


def block_deinterleave(bits, rows=8, layout="rectangular"):
    if len(bits) % rows:
        raise ValueError("Interleaved block length must be divisible by row count")
    return _deinterleave(list(map(int, bits)), rows, layout)


def analyze_fec(bit_string, min_bits=128, max_rows=16, fit_threshold=0.12):
    """Search two convolutional profiles and rectangular/diagonal block layouts.

    Fit is the re-encoded Viterbi path's Hamming error rate. Even a match is only
    evidence for this candidate family; framing, termination, and payload validity
    are not established.
    """
    bits = [int(char) for char in bit_string if char in "01"]
    if len(bits) < min_bits:
        return {"status": "INSUFFICIENT_BITS", "tested_bit_count": len(bits),
                "candidates": [], "decoded_preview": "",
                "reed_solomon": search_rs_profiles(bit_string),
                "ldpc": find_blind_80211n_648_rate_half(bit_string),
                "note": f"Need at least {min_bits} recovered bits for a useful FEC search."}
    candidates = []
    for code_name, generators in CONVOLUTIONAL_PROFILES:
        for offset in (0, 1):
            shifted = bits[offset:1024]
            for layout in ("rectangular", "diagonal"):
                for rows in range(1, max_rows + 1):
                    if layout == "diagonal" and rows == 1:
                        continue
                    usable = (len(shifted) // rows) * rows
                    usable -= usable % 2
                    if usable < min_bits:
                        continue
                    encoded = _deinterleave(shifted[:usable], rows, layout)
                    decoded = viterbi_decode(encoded, generators)
                    rebuilt = convolutional_encode(decoded, generators)
                    errors = sum(a != b for a, b in zip(encoded, rebuilt))
                    fit = errors / max(len(encoded), 1)
                    candidates.append({"code": code_name, "interleaver_rows": rows,
                                       "interleaver_layout": layout, "bit_offset": offset,
                                       "fit_error_rate": round(fit, 4),
                                       "fit_percent": round(100 * (1 - fit), 1),
                                       "tested_bit_count": usable,
                                       "_decoded": "".join(map(str, decoded))})
    candidates.sort(key=lambda item: (item["fit_error_rate"], item["code"], item["interleaver_rows"], item["bit_offset"]))
    best = candidates[0] if candidates else None
    conv_matched = bool(best and best["fit_error_rate"] <= fit_threshold)
    rs_result = search_rs_profiles(bit_string)
    ldpc_result = find_blind_80211n_648_rate_half(bit_string)
    rs_matched = rs_result.get("status") == "CANDIDATE_FOUND"
    ldpc_matched = ldpc_result.get("status") == "CANDIDATE_FOUND"
    matched = conv_matched or rs_matched or ldpc_matched
    public_candidates = [{key: value for key, value in item.items() if key != "_decoded"}
                         for item in candidates[:5]]
    result = {"status": "CANDIDATE_FOUND" if matched else "NO_SUPPORTED_MATCH",
              "tested_bit_count": len(bits),
              "code_family": (best["code"] if conv_matched else rs_result.get("code") if rs_matched else ldpc_result.get("code")),
              "interleaver_rows": best["interleaver_rows"] if conv_matched else None,
              "interleaver_layout": best["interleaver_layout"] if conv_matched else None,
              "fit_error_rate": best["fit_error_rate"] if conv_matched else None,
              "decoded_preview": (best["_decoded"][:128] if conv_matched else
                                  rs_result.get("decoded_preview", "") if rs_matched else
                                  ldpc_result.get("decoded_preview", "")),
              "candidates": public_candidates,
              "reed_solomon": rs_result,
              "ldpc": ldpc_result,
              "note": "Supported candidates: K=3/K=7 rate-1/2 convolutional, rectangular/diagonal block interleaving, RS(255,223)/(255,239), and one 802.11n LDPC profile. Matches are tentative."}
    return result
