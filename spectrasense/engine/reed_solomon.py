"""GF(256) Reed-Solomon(255,k) decoders (primitive polynomial 0x11D)."""

_EXP = [0] * 510
_LOG = [0] * 256
_value = 1
for _i in range(255):
    _EXP[_i] = _value
    _LOG[_value] = _i
    _value <<= 1
    if _value & 0x100:
        _value ^= 0x11D
for _i in range(255, 510):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a, b):
    return 0 if not a or not b else _EXP[_LOG[a] + _LOG[b]]


def _div(a, b):
    if not b:
        raise ZeroDivisionError("GF(256) division by zero")
    return 0 if not a else _EXP[(_LOG[a] - _LOG[b]) % 255]


def _pow(a, power):
    if power == 0:
        return 1
    if a == 0:
        return 0
    return _EXP[(_LOG[a] * power) % 255]


def _poly_eval(poly, x):
    value = 0
    for coefficient in poly:
        value = _mul(value, x) ^ coefficient
    return value


def _syndromes(codeword, parity=32):
    return [_poly_eval(codeword, _EXP[i]) for i in range(parity)]


def _locator(syndromes):
    # Berlekamp-Massey; coefficients are stored from lowest degree upward.
    count = len(syndromes)
    current = [1] + [0] * count
    previous = [1] + [0] * count
    degree, shift, last = 0, 1, 1
    for n in range(count):
        discrepancy = syndromes[n]
        for i in range(1, degree + 1):
            discrepancy ^= _mul(current[i], syndromes[n - i])
        if discrepancy == 0:
            shift += 1
            continue
        saved = current[:]
        scale = _div(discrepancy, last)
        for i in range(count + 1 - shift):
            if previous[i]:
                current[i + shift] ^= _mul(scale, previous[i])
        if 2 * degree <= n:
            degree = n + 1 - degree
            previous, last, shift = saved, discrepancy, 1
        else:
            shift += 1
    return current[:degree + 1]


def _solve_gf(matrix, vector):
    size = len(vector)
    augmented = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = next((row for row in range(col, size) if augmented[row][col]), None)
        if pivot is None:
            raise ValueError("Singular error-locator system")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [_div(value, scale) for value in augmented[col]]
        for row in range(size):
            if row == col or not augmented[row][col]:
                continue
            factor = augmented[row][col]
            augmented[row] = [a ^ _mul(factor, b)
                              for a, b in zip(augmented[row], augmented[col])]
    return [augmented[row][-1] for row in range(size)]


def decode_rs_255(codeword, parity=32):
    """Correct RS(255,255-parity), returning corrected bytes and error count."""
    received = list(codeword)
    if len(received) != 255 or any(not 0 <= int(byte) <= 255 for byte in received):
        raise ValueError("RS(255,k) requires exactly 255 bytes")
    if parity not in (16, 32):
        raise ValueError("Supported RS parity lengths are 16 and 32 bytes")
    syndromes = _syndromes(received, parity)
    if not any(syndromes):
        return bytes(received), 0
    locator = _locator(syndromes)
    errors = len(locator) - 1
    if errors < 1 or errors > parity // 2:
        raise ValueError(f"Too many errors for RS(255,{255 - parity})")

    positions = []
    for pos in range(255):
        exponent = 254 - pos
        if _poly_eval(locator[::-1], _EXP[(-exponent) % 255]) == 0:
            positions.append(pos)
    if len(positions) != errors:
        raise ValueError("Could not locate all byte errors")

    xs = [_EXP[254 - pos] for pos in positions]
    matrix = [[_pow(x, row) for x in xs] for row in range(errors)]
    magnitudes = _solve_gf(matrix, syndromes[:errors])
    for pos, magnitude in zip(positions, magnitudes):
        received[pos] ^= magnitude
    if any(_syndromes(received, parity)):
        raise ValueError("RS correction failed syndrome verification")
    return bytes(received), errors


def decode_rs_255_223(codeword):
    """Compatibility wrapper for RS(255,223), correcting up to 16 bytes."""
    return decode_rs_255(codeword, 32)


def decode_rs_255_239(codeword):
    """RS(255,239), correcting up to 8 bytes."""
    return decode_rs_255(codeword, 16)


def search_rs_profiles(bit_string, max_checks=512):
    """Scan alignments for verified RS(255,223) and RS(255,239) codewords."""
    bits = [int(bit) for bit in bit_string if bit in "01"]
    if len(bits) < 255 * 8:
        return {"status": "INSUFFICIENT_BITS", "minimum_bits": 2040,
                "profiles": ["RS(255,223)", "RS(255,239)"],
                "note": "Supported RS profiles need at least one 255-byte codeword."}
    checks = 0
    # First inspect frame-start alignment for every bit phase, then scan byte
    # offsets in the most common phase while keeping runtime bounded.
    for parity in (32, 16):
        k = 255 - parity
        for bit_offset in range(8):
            aligned = bits[bit_offset:]
            byte_count = len(aligned) // 8
            raw = bytes(sum(aligned[i + b] << (7 - b) for b in range(8))
                        for i in range(0, byte_count * 8, 8))
            all_starts = max(1, len(raw) - 254)
            starts = range(all_starts) if bit_offset == 0 else range(1)
            for start in starts:
                if checks >= max_checks:
                    return {"status": "SEARCH_LIMIT", "checks": checks,
                            "profiles": ["RS(255,223)", "RS(255,239)"],
                            "note": "No verified codeword in bounded alignment search."}
                checks += 1
                try:
                    decoded, corrected = decode_rs_255(raw[start:start + 255], parity)
                except ValueError:
                    continue
                payload_bits = "".join(f"{byte:08b}" for byte in decoded[:k])
                return {"status": "CANDIDATE_FOUND", "code": f"RS(255,{k}), GF(256), primitive polynomial 0x11D",
                        "bit_offset": bit_offset, "byte_offset": start,
                        "corrected_bytes": corrected, "verified_codeword": True,
                        "decoded_preview": payload_bits[:128], "checks": checks,
                        "note": "Codeword syndromes verified; framing and higher-layer integrity remain unverified."}
    return {"status": "NO_SUPPORTED_MATCH", "checks": checks,
            "profiles": ["RS(255,223)", "RS(255,239)"],
            "note": "No verified RS(255,223) or RS(255,239) codeword found in the bounded alignment search."}


def search_rs_255_223(bit_string, max_checks=512):
    """Compatibility wrapper; scans supported RS profiles for verified codewords."""
    return search_rs_profiles(bit_string, max_checks=max_checks)
