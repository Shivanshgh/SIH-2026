"""Train an offline feature-based classifier using reproducible synthetic IQ.

Run from the project root with ``python -m spectrasense.ml.train_model``.
The reported holdout score is a synthetic benchmark, not a claim of field accuracy.
"""

import argparse
import bz2
import json
import os
import pickle
import tarfile

import numpy as np

from spectrasense.dsp.features import extract_all_features
from spectrasense.engine.ml_classifier import FEATURE_NAMES, FEATURE_WINDOW_SAMPLES
from spectrasense.ml.forest import RandomForest
from spectrasense.engine.ml_classifier import feature_vector
from spectrasense.dsp.preprocess import preprocess_pipeline

try:
    _numpy_multiarray = np._core.multiarray
except AttributeError:
    _numpy_multiarray = np.core.multiarray


SAMPLE_RATE = 200_000
MODULATIONS = ("BPSK", "QPSK", "8-PSK", "2-FSK", "16-QAM")


def synthesize(modulation, snr_db, rng, samples=FEATURE_WINDOW_SAMPLES):
    """Generate varied, labelled captures with phase/CFO and mild multipath."""
    rates = {"BPSK": (8_000, 25_000), "QPSK": (8_000, 25_000),
             "8-PSK": (8_000, 25_000), "2-FSK": (5_000, 15_000),
             "16-QAM": (8_000, 25_000)}
    rate = int(rng.integers(*rates[modulation]))
    sps = SAMPLE_RATE // rate
    count = (samples + sps - 1) // sps
    if modulation == "BPSK":
        symbols = rng.choice(np.array([-1, 1]), count).astype(np.complex128)
    elif modulation == "QPSK":
        symbols = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, count)))
    elif modulation == "8-PSK":
        symbols = np.exp(1j * (rng.uniform(-np.pi, np.pi) + 2 * np.pi / 8 * rng.integers(0, 8, count)))
    elif modulation == "16-QAM":
        levels = np.array([-3, -1, 1, 3])
        symbols = (rng.choice(levels, count) + 1j * rng.choice(levels, count)) / np.sqrt(10)
    else:
        symbols = np.ones(count, dtype=np.complex128)

    # Random fractional timing phase, then truncate to the fixed inference window.
    timing = int(rng.integers(0, sps))
    base = np.repeat(symbols, sps)[timing:timing + samples]
    if len(base) < samples:
        base = np.pad(base, (0, samples - len(base)), mode="wrap")
    if modulation == "2-FSK":
        bits = rng.integers(0, 2, count)
        tones = np.repeat(np.where(bits > 0, 1.0, -1.0), sps)[timing:timing + samples]
        if len(tones) < samples:
            tones = np.pad(tones, (0, samples - len(tones)), mode="wrap")
        deviation = float(rng.choice([rate / 2, rate]))
        inst_freq = tones * deviation
        phase = np.cumsum(2 * np.pi * inst_freq / SAMPLE_RATE)
        base = np.exp(1j * phase)

    fc = float(rng.uniform(-45_000, 45_000))
    phase0 = float(rng.uniform(-np.pi, np.pi))
    t = np.arange(samples) / SAMPLE_RATE
    drift = float(rng.uniform(-150, 150))
    iq = base * rng.uniform(0.3, 2.5) * np.exp(1j * (2 * np.pi * fc * t + np.pi * drift * t * t + phase0))
    # Mild two-path channel and gain variation prevent exact waveform memorization.
    if rng.random() < 0.5:
        iq = iq + rng.uniform(0.05, 0.2) * np.roll(iq, int(rng.integers(1, max(2, sps // 2))))
    power = float(np.mean(np.abs(iq) ** 2))
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(samples) + 1j * rng.standard_normal(samples))
    dc = complex(rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15))
    return (iq + noise + dc).astype(np.complex64)


def build_dataset(per_class_snr, seed):
    rng = np.random.default_rng(seed)
    rows, labels, snrs = [], [], []
    for modulation in MODULATIONS:
        for snr in (-2, 2, 6, 10, 14, 18):
            for _ in range(per_class_snr):
                iq = synthesize(modulation, snr, rng)
                iq, _ = preprocess_pipeline(iq, sample_rate=SAMPLE_RATE)
                features = extract_all_features(iq, SAMPLE_RATE)
                rows.append(feature_vector(features, SAMPLE_RATE))
                labels.append(modulation)
                snrs.append(snr)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels), np.asarray(snrs)


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Read NumPy arrays from this public corpus while refusing arbitrary globals."""
    _allowed = {
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("numpy.core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): _numpy_multiarray.scalar,
        ("numpy._core.multiarray", "scalar"): _numpy_multiarray.scalar,
        ("_codecs", "encode"): __import__("_codecs").encode,
    }

    def find_class(self, module, name):
        try:
            return self._allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"Blocked dataset pickle global: {module}.{name}") from exc


def read_radioml(path):
    """Load RadioML 2016.10a from its pickle file or the published tar.bz2 archive."""
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:bz2") as archive:
            member = next((item for item in archive.getmembers() if item.name.endswith(".pkl")), None)
            if member is None:
                raise ValueError("RadioML archive contains no .pkl dataset")
            stream = archive.extractfile(member)
            return _NumpyOnlyUnpickler(stream, encoding="latin1").load()
    opener = bz2.open if path.endswith((".bz2", ".bz")) else open
    with opener(path, "rb") as stream:
        return _NumpyOnlyUnpickler(stream, encoding="latin1").load()


def load_radioml_features(path, examples_per_snr, seed):
    dataset = read_radioml(path)
    rng = np.random.default_rng(seed + 1)
    labels = {"BPSK": "BPSK", "QPSK": "QPSK", "QAM16": "16-QAM"}
    train_rows, train_labels, train_snr = [], [], []
    test_rows, test_labels, test_snr = [], [], []
    for key, samples in dataset.items():
        raw_label, snr = key
        raw_label = raw_label.decode("utf-8") if isinstance(raw_label, bytes) else str(raw_label)
        if raw_label not in labels:
            continue
        samples = np.asarray(samples)
        indices = rng.permutation(len(samples))[:min(examples_per_snr, len(samples))]
        # Four independent 128-sample records form one 512-sample model window.
        groups = [indices[i:i + FEATURE_WINDOW_SAMPLES // 128]
                  for i in range(0, len(indices) - (FEATURE_WINDOW_SAMPLES // 128) + 1,
                                 FEATURE_WINDOW_SAMPLES // 128)]
        if len(groups) < 2:
            continue
        held_count = max(1, int(round(len(groups) * 0.2)))
        for local_index, group in enumerate(groups):
            iq = np.concatenate([samples[index, 0, :] + 1j * samples[index, 1, :]
                                 for index in group])
            iq, _ = preprocess_pipeline(iq, sample_rate=SAMPLE_RATE)
            vector = feature_vector(extract_all_features(iq, SAMPLE_RATE), SAMPLE_RATE)
            target_rows, target_labels, target_snr = (
                (test_rows, test_labels, test_snr) if local_index < held_count
                else (train_rows, train_labels, train_snr)
            )
            target_rows.append(vector)
            target_labels.append(labels[raw_label])
            target_snr.append(int(snr))
    if not train_rows or not test_rows:
        raise ValueError("No supported RadioML classes found (expected BPSK, QPSK, QAM16)")
    return (np.asarray(train_rows), np.asarray(train_labels), np.asarray(train_snr),
            np.asarray(test_rows), np.asarray(test_labels), np.asarray(test_snr))


def accuracy(expected, predicted):
    return sum(a == b for a, b in zip(expected, predicted)) / max(len(expected), 1)


def confusion(expected, predicted, classes):
    return [[sum(actual == cls and guess == target for actual, guess in zip(expected, predicted))
             for target in classes] for cls in classes]


def class_report(expected, predicted, classes):
    report = {}
    for cls in classes:
        tp = sum(a == cls and b == cls for a, b in zip(expected, predicted))
        support = sum(a == cls for a in expected)
        predicted_count = sum(b == cls for b in predicted)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[cls] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class-snr", type=int, default=24,
                        help="captures per modulation per SNR (default: 24; 720 total)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=os.path.join("models", "modulation_model.json"))
    parser.add_argument("--radioml", help="Path to RadioML2016.10a tar.bz2 archive or pickle")
    parser.add_argument("--radioml-per-snr", type=int, default=30,
                        help="examples per supported class and SNR (default: 30)")
    args = parser.parse_args()
    if args.per_class_snr < 2:
        parser.error("--per-class-snr must be at least 2")
    if args.radioml_per_snr < 8:
        parser.error("--radioml-per-snr must be at least 8 (two independent 512-sample windows)")

    x, y, snrs = build_dataset(args.per_class_snr, args.seed)
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = [], []
    for cls in MODULATIONS:
        indices = np.flatnonzero(y == cls)
        rng.shuffle(indices)
        held = max(1, int(round(len(indices) * 0.25)))
        test_idx.extend(indices[:held].tolist())
        train_idx.extend(indices[held:].tolist())
    synthetic_model = RandomForest.fit(x[train_idx].tolist(), y[train_idx].tolist(), seed=args.seed)
    synthetic_test_pred = [synthetic_model.classes[int(np.argmax(synthetic_model.predict_proba_one(row.tolist())))]
                          for row in x[test_idx]]
    training_rows = x[train_idx].tolist()
    training_labels = y[train_idx].tolist()
    radio_test = None
    radio_baseline_accuracy = None
    if args.radioml:
        (radio_x, radio_y, _radio_snr,
         radio_test_x, radio_test_y, radio_test_snr) = load_radioml_features(
            args.radioml, args.radioml_per_snr, args.seed
        )
        training_rows.extend(radio_x.tolist())
        training_labels.extend(radio_y.tolist())
        radio_test = (radio_test_x, radio_test_y, radio_test_snr)
        baseline_radio_pred = [synthetic_model.classes[int(np.argmax(synthetic_model.predict_proba_one(row.tolist())))]
                               for row in radio_test_x]
        radio_baseline_accuracy = accuracy(radio_test_y.tolist(), baseline_radio_pred)
    model = RandomForest.fit(training_rows, training_labels, seed=args.seed)
    y_test, snr_test = y[test_idx], snrs[test_idx]
    predicted = [model.classes[int(np.argmax(model.predict_proba_one(row.tolist())))] for row in x[test_idx]]
    holdout_accuracy = accuracy(y_test.tolist(), predicted)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump({"forest": model.to_dict(), "feature_names": FEATURE_NAMES,
                   "feature_version": 2, "feature_window_samples": FEATURE_WINDOW_SAMPLES,
                   "classes": model.classes, "sample_rate": SAMPLE_RATE,
                   "synthetic_only": not bool(args.radioml), "seed": args.seed,
                   "training_sources": ["local synthetic generator"] + (["RadioML2016.10a"] if args.radioml else []),
                   "randomized_conditions": ["SNR", "frequency offset", "phase", "amplitude", "symbol rate",
                                             "timing offset", "AWGN", "DC offset", "mild multipath", "frequency drift"],
                   "dataset_license": "CC BY-NC-SA 4.0" if args.radioml else None,
                   "dataset_attribution": "O'Shea, West, and Rothe, RadioML 2016.10a (2016), DOI 10.5281/zenodo.18397070" if args.radioml else None}, stream)
    metrics = {
        "evaluation": ("RadioML2016.10a and synthetic holdouts; both are synthetic benchmarks, not field performance"
                       if args.radioml else "Synthetic holdout only; not a field-performance estimate"),
        "seed": args.seed, "training_sample_count": int(len(training_labels)),
        "split_strategy": "stratified by modulation; each row is one independently generated capture",
        "synthetic_feature_window_samples": FEATURE_WINDOW_SAMPLES,
        "synthetic_holdout_sample_count": int(len(y_test)),
        "combined_model_synthetic_holdout_accuracy": holdout_accuracy,
        "synthetic_only_baseline_accuracy": accuracy(y_test.tolist(), synthetic_test_pred),
        "classes": list(MODULATIONS),
        "macro_average": {
            metric: float(np.mean([row[metric] for row in class_report(y_test.tolist(), predicted, MODULATIONS).values()]))
            for metric in ("precision", "recall", "f1-score")
        },
        "confusion_matrix": confusion(y_test.tolist(), predicted, MODULATIONS),
        "classification_report": class_report(y_test.tolist(), predicted, MODULATIONS),
        "accuracy_by_snr_db": {
            str(int(snr)): float(accuracy([label for label, level in zip(y_test.tolist(), snr_test.tolist()) if level == snr],
                                          [label for label, level in zip(predicted, snr_test.tolist()) if level == snr]))
            for snr in sorted(set(snr_test.tolist())) if np.any(snr_test == snr)
        },
        "model_file": os.path.relpath(output_path, os.getcwd()),
        "training_sources": ["local synthetic generator"] + (["RadioML2016.10a (BPSK, QPSK, QAM16)"] if args.radioml else []),
    }
    if radio_test is not None:
        radio_x, radio_y, radio_snr = radio_test
        radio_pred = [model.classes[int(np.argmax(model.predict_proba_one(row.tolist())))] for row in radio_x]
        metrics["radioml2016_10a_holdout_accuracy"] = accuracy(radio_y.tolist(), radio_pred)
        metrics["synthetic_only_model_on_radioml_holdout_accuracy"] = radio_baseline_accuracy
        metrics["radioml2016_10a_samples"] = int(len(radio_y))
        radio_classes = sorted(set(radio_y.tolist()))
        metrics["radioml2016_10a_classes"] = radio_classes
        metrics["radioml2016_10a_confusion_matrix"] = confusion(radio_y.tolist(), radio_pred, radio_classes)
        metrics["radioml2016_10a_classification_report"] = class_report(radio_y.tolist(), radio_pred, radio_classes)
        metrics["radioml2016_10a_accuracy_by_snr_db"] = {
            str(int(snr)): accuracy([a for a, b in zip(radio_y.tolist(), radio_snr.tolist()) if b == snr],
                                    [a for a, b in zip(radio_pred, radio_snr.tolist()) if b == snr])
            for snr in sorted(set(radio_snr.tolist()))
        }
    metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(f"Saved offline model: {output_path}")
    print(f"Synthetic-only baseline holdout: {accuracy(y_test.tolist(), synthetic_test_pred):.1%}")
    print(f"Combined model synthetic holdout: {holdout_accuracy:.1%}")
    if radio_test is not None:
        print(f"RadioML holdout accuracy: {metrics['radioml2016_10a_holdout_accuracy']:.1%} (synthetic benchmark)")
    print("Neither score estimates field performance.")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
