"""Bounded IQ/WAV input helpers with explicit format selection."""
import os
import wave

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is a required app dependency
    np = None


class SignalFormat:
    FLOAT32_IQ = "Complex Float32 (fc32 / cfile)"
    INT16_IQ = "Complex Int16 (sc16)"
    INT8_IQ = "Complex Int8 (sc8)"
    WAV_AUDIO = "WAV Audio (.wav)"


_FORMAT_ALIASES = {
    SignalFormat.FLOAT32_IQ: "float32", "float32": "float32", "fc32": "float32",
    "cfile": "float32", "complex float32": "float32",
    SignalFormat.INT16_IQ: "int16", "int16": "int16", "sc16": "int16",
    "complex int16": "int16",
    SignalFormat.INT8_IQ: "int8", "int8": "int8", "sc8": "int8",
    "complex int8": "int8",
}
_EXT_FORMAT = {".fc32": "float32", ".cfile": "float32", ".iq": "float32",
               ".sc16": "int16", ".sc8": "int8"}
_CANONICAL_LABEL = {"float32": SignalFormat.FLOAT32_IQ, "int16": SignalFormat.INT16_IQ,
                    "int8": SignalFormat.INT8_IQ}


def infer_iq_format(filepath):
    """Return the GUI format label suggested by the extension, or None."""
    kind = _EXT_FORMAT.get(os.path.splitext(filepath)[1].lower())
    return _CANONICAL_LABEL.get(kind)


def iq_pair_bytes(format_type):
    """Return encoded bytes per complex pair for a selected format."""
    kind = _FORMAT_ALIASES.get(format_type, _FORMAT_ALIASES.get(str(format_type).lower()))
    return {"float32": 8, "int16": 4, "int8": 2}.get(kind)


def _canonical_format(filepath, format_type):
    if format_type is None or str(format_type).lower() == "auto":
        label = infer_iq_format(filepath)
        if label is None:
            raise ValueError(f"Unsupported IQ format for '{os.path.basename(filepath)}'; select Float32, Int16, or Int8.")
        return {value: key for key, value in _CANONICAL_LABEL.items()}[label]
    try:
        return _FORMAT_ALIASES[format_type] if format_type in _FORMAT_ALIASES else _FORMAT_ALIASES[str(format_type).lower()]
    except KeyError as exc:
        raise ValueError(f"Invalid IQ sample format: {format_type}") from exc


def read_iq_file(filepath, format_type=None, max_samples=1_000_000,
                 representative=False, chunks=8):
    """Read interleaved I/Q pairs. Explicit format always overrides the extension.

    For large files, representative=True reads bounded, evenly spaced windows.
    The returned array remains at most max_samples complex values.
    """
    if np is None:
        raise RuntimeError("NumPy is required to load IQ files.")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    sample_format = _canonical_format(filepath, format_type)
    dtype = {"float32": np.dtype("<f4"), "int16": np.dtype("<i2"), "int8": np.dtype("i1")}[sample_format]
    size = os.path.getsize(filepath)
    bytes_per_complex = 2 * dtype.itemsize
    if size == 0:
        raise ValueError("Invalid IQ file: file is empty")
    if size % bytes_per_complex:
        raise ValueError(f"Incomplete IQ samples: file size {size} is not a multiple of {bytes_per_complex} bytes per I/Q pair.")
    total = size // bytes_per_complex
    if total <= max_samples or not representative:
        starts = [0]
        lengths = [min(total, max_samples)]
    else:
        n_chunks = min(max(1, chunks), max_samples)
        per_window = max(1, max_samples // n_chunks)
        starts = np.linspace(0, max(0, total - per_window), n_chunks, dtype=np.int64).tolist()
        lengths = [per_window] * len(starts)
    parts = []
    with open(filepath, "rb") as stream:
        for start, count in zip(starts, lengths):
            stream.seek(start * bytes_per_complex)
            raw = np.fromfile(stream, dtype=dtype, count=count * 2)
            if raw.size != count * 2:
                raise ValueError("Invalid IQ file: incomplete I/Q sample pair encountered while reading.")
            vals = raw.astype(np.float32, copy=False)
            if sample_format == "int16":
                vals = vals / 32768.0
            elif sample_format == "int8":
                vals = vals / 128.0
            i = vals[0::2]
            q = vals[1::2]
            parts.append((i + 1j * q).astype(np.complex64, copy=False))
    result = parts[0] if len(parts) == 1 else np.concatenate(parts)
    if sample_format == "float32" and not np.all(np.isfinite(result)):
        raise ValueError("Invalid IQ file: Float32 samples contain NaN or infinity.")
    return result


def read_wav_file(filepath, max_samples=1_000_000, representative=False, chunks=8):
    """Read PCM mono or stereo WAV; stereo channels are interpreted as I/Q."""
    if np is None:
        raise RuntimeError("NumPy is required to load WAV files.")
    try:
        with wave.open(filepath, "rb") as wf:
            channels, width, rate, frames = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
            if channels not in (1, 2) or width not in (1, 2, 3, 4) or rate <= 0:
                raise ValueError("Invalid WAV sample format or channel count.")
            if frames > max_samples and representative:
                n_chunks = min(max(1, chunks), max_samples)
                per_chunk = max(1, max_samples // n_chunks)
                starts = np.linspace(0, frames - per_chunk, n_chunks, dtype=np.int64)
                blocks = []
                for start in starts:
                    wf.setpos(int(start))
                    blocks.append(wf.readframes(per_chunk))
                data = b"".join(blocks)
            else:
                data = wf.readframes(min(frames, max_samples))
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError(f"Corrupted or unsupported WAV file: {exc}") from exc
    if width == 1:
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    else:  # PCM 24-bit little-endian with sign extension
        b = np.frombuffer(data, dtype=np.uint8)
        if b.size % 3:
            raise ValueError("Corrupted WAV: incomplete 24-bit PCM sample.")
        b = b.reshape(-1, 3).astype(np.int32)
        vals = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        vals = (vals ^ 0x800000) - 0x800000
        samples = vals.astype(np.float32) / 8388608.0
    if samples.size % channels:
        raise ValueError("Corrupted WAV: incomplete channel frame.")
    if channels == 2:
        stereo = samples.reshape(-1, 2)
        signal = stereo[:, 0] + 1j * stereo[:, 1]
    else:
        from scipy.signal import hilbert
        signal = hilbert(samples).astype(np.complex64, copy=False)
    return np.asarray(signal, dtype=np.complex64), rate


def get_file_metadata(filepath):
    size = os.path.getsize(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".wav":
        try:
            with wave.open(filepath, "rb") as stream:
                count = stream.getnframes()
        except (wave.Error, OSError):
            count = None
    else:
        fmt = _EXT_FORMAT.get(ext, "float32")
        count = size // (2 * {"float32": 4, "int16": 2, "int8": 1}[fmt])
    return {"filepath": filepath, "filename": os.path.basename(filepath), "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2), "extension": ext,
            "format_guess": "WAV Audio" if ext == ".wav" else "Raw Complex IQ",
            "available_sample_count": count}
