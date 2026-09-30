"""LDPC decoders and a bounded IEEE 802.11n candidate search."""


def decode_min_sum_sparse(llrs, checks, max_iterations=30, normalization=0.8):
    """Decode LLRs (positive favors bit 0) using check-node variable indices."""
    if not checks or not llrs:
        raise ValueError("LLRs and parity-check matrix are required")
    width = len(llrs)
    variable_checks = [[] for _ in range(width)]
    for check_index, row in enumerate(checks):
        indices = list(row)
        if any(i < 0 or i >= width for i in indices):
            raise ValueError("Parity-check matrix width must equal LLR count")
        if not indices:
            raise ValueError("Parity-check matrix contains an empty row")
        for var in indices:
            variable_checks[var].append(check_index)

    variable_messages = {(check, var): float(llrs[var])
                         for check, variables in enumerate(checks) for var in variables}
    beliefs = [float(value) for value in llrs]
    for iteration in range(1, max_iterations + 1):
        check_messages = {}
        for check, variables in enumerate(checks):
            incoming = [variable_messages[(check, var)] for var in variables]
            signs = [-1.0 if value < 0 else 1.0 for value in incoming]
            magnitudes = [abs(value) for value in incoming]
            for pos, var in enumerate(variables):
                other = magnitudes[:pos] + magnitudes[pos + 1:]
                sign = 1.0
                for i, value in enumerate(signs):
                    if i != pos:
                        sign *= value
                check_messages[(check, var)] = normalization * sign * min(other, default=0.0)
        beliefs = [float(llrs[var]) + sum(check_messages[(check, var)]
                    for check in variable_checks[var]) for var in range(width)]
        hard = [int(value < 0) for value in beliefs]
        unsatisfied = sum(sum(hard[var] for var in variables) % 2 for variables in checks)
        if unsatisfied == 0:
            return {"bits": hard, "converged": True, "iterations": iteration,
                    "unsatisfied_checks": 0}
        variable_messages = {(check, var): beliefs[var] - check_messages[(check, var)]
                             for check, variables in enumerate(checks) for var in variables}
    return {"bits": hard, "converged": False, "iterations": max_iterations,
            "unsatisfied_checks": unsatisfied}


def decode_min_sum(llrs, parity_check, max_iterations=30, normalization=0.8):
    """Decode with a dense binary H matrix; positive LLR favors bit zero."""
    checks = [[i for i, value in enumerate(row) if value] for row in parity_check]
    return decode_min_sum_sparse(llrs, checks, max_iterations, normalization)


_IEEE_80211N_648_R12_LEFT = (
    "0 - - - 0 0 - - 0 - - 0",
    "22 0 - - 17 - 0 0 12 - - -",
    "6 - 0 - 10 - - - 24 - 0 -",
    "2 - - 0 20 - - - 25 0 - -",
    "23 - - - 3 - - - 0 - 9 11",
    "24 - 23 1 17 - 3 - 10 - - -",
    "25 - - - 8 - - - 7 18 - -",
    "13 24 - - 0 - 8 - 6 - - -",
    "7 20 - 16 22 10 - - 23 - - -",
    "11 - - - 19 - - - 13 - 3 17",
    "25 - 8 - 23 18 - 14 9 - - -",
    "3 - - - 16 - - 2 25 5 - -",
)


def make_80211n_648_rate_half_checks():
    """Build the 324x648 QC-LDPC H from IEEE 802.11n Annex R prototype."""
    z = 27
    checks = [[] for _ in range(12 * z)]
    for block_row, encoded_row in enumerate(_IEEE_80211N_648_R12_LEFT):
        left = [None if token == "-" else int(token) for token in encoded_row.split()]
        if len(left) != 12:
            raise ValueError("Invalid 802.11n prototype row")
        parity = [None] * 12
        if block_row == 0:
            parity[0], parity[1] = 1, 0
        elif block_row == 6:
            parity[0], parity[5], parity[6] = 0, 0, 0
        elif block_row == 11:
            parity[0], parity[11] = 1, 0
        else:
            parity[block_row], parity[block_row + 1] = 0, 0
        shifts = left + parity
        for block_col, shift in enumerate(shifts):
            if shift is None:
                continue
            for local_row in range(z):
                checks[block_row * z + local_row].append(
                    block_col * z + ((local_row + shift) % z)
                )
    return checks


def find_blind_80211n_648_rate_half(bit_string, max_flip_fraction=0.06):
    """Scan unpermuted hard bits for a decodable 648-bit WLAN LDPC codeword.

    Candidate matching covers codeword start offsets in a recovered bitstream;
    WLAN PHY bit interleaving, puncturing, and other LDPC profiles are not searched.
    """
    bits = [int(bit) for bit in bit_string if bit in "01"]
    n = 648
    if len(bits) < n:
        return {"status": "INSUFFICIENT_BITS", "minimum_bits": n,
                "code": "IEEE 802.11n LDPC, N=648, rate 1/2"}
    checks = make_80211n_648_rate_half_checks()
    columns = [0] * n
    for check, variables in enumerate(checks):
        for variable in variables:
            columns[variable] |= 1 << check
    best = None
    for start in range(len(bits) - n + 1):
        # H is sparse but not cyclic across bit positions, so each shifted
        # codeword window needs its own syndrome calculation.
        syndrome = 0
        for index, bit in enumerate(bits[start:start + n]):
            if bit:
                syndrome ^= columns[index]
        weight = syndrome.bit_count()
        if best is None or weight < best["syndrome_weight"]:
            best = {"start_bit": start, "syndrome_weight": weight}
        if weight <= max(2, int(0.12 * len(checks))):
            window = bits[start:start + n]
            llrs = [-3.0 if bit else 3.0 for bit in window]
            decoded = decode_min_sum_sparse(llrs, checks, max_iterations=20)
            changes = sum(a != b for a, b in zip(window, decoded["bits"]))
            if decoded["converged"] and changes <= int(max_flip_fraction * n):
                return {"status": "CANDIDATE_FOUND",
                        "code": "IEEE 802.11n QC-LDPC, N=648, rate 1/2",
                        "start_bit": start, "syndrome_weight_before": weight,
                        "syndrome_weight_after": decoded["unsatisfied_checks"],
                        "corrected_bits": changes, "iterations": decoded["iterations"],
                        "decoded_preview": "".join(map(str, decoded["bits"][:128])),
                        "note": "Matrix checks pass after decoding; confirm WLAN framing/interleaving independently."}
    return {"status": "NO_SUPPORTED_MATCH", "code": "IEEE 802.11n LDPC, N=648, rate 1/2",
            "best_start_bit": best["start_bit"], "best_syndrome_weight": best["syndrome_weight"],
            "checks": len(checks),
            "note": "No candidate met the parity and correction thresholds; other LDPC profiles are not covered."}
