"""Optional offline modulation classifier trained from synthetic IQ captures."""

import json
import os

import numpy as np


FEATURE_NAMES = (
    "cumulant_c40", "cumulant_c40_real", "cumulant_c20", "cumulant_c42",
    "sq_peak_ratio", "envelope_variance", "freq_inst_variance", "snr_db",
    "bandwidth_3db_hz", "bandwidth_99_hz", "symbol_rate_baud",
    "symbol_rate_quality",
)
# Longer captures stabilize spectral and cumulant features, especially for
# higher-order PSK/QAM and low-SNR IQ. Inference remains capped at 32 windows.
FEATURE_WINDOW_SAMPLES = 2048
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "modulation_model.json")


def feature_vector(features, sample_rate=200_000.0):
    """Use sample-rate-normalized spectral features for variable-rate captures."""
    sr = max(float(sample_rate), 1.0)
    normalized = dict(features)
    for name in ("bandwidth_3db_hz", "bandwidth_99_hz", "symbol_rate_baud"):
        normalized[name] = float(features.get(name, 0.0)) / sr
    normalized["freq_inst_variance"] = float(features.get("freq_inst_variance", 0.0)) / (sr * sr)
    return np.asarray([
        np.nan_to_num(float(normalized.get(name, 0.0)), nan=0.0, posinf=0.0, neginf=0.0)
        for name in FEATURE_NAMES
    ], dtype=np.float64)


class ModulationClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = os.path.abspath(model_path)
        self.model = None
        self.error = None
        self.training_sources = []
        self.synthetic_only = True
        self.dataset_license = None
        self.dataset_attribution = None
        try:
            if os.path.isfile(self.model_path):
                with open(self.model_path, "r", encoding="utf-8") as stream:
                    bundle = json.load(stream)
                if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
                    self.error = "Model feature version does not match this application. Retrain it."
                else:
                    from spectrasense.ml.forest import RandomForest
                    self.model = RandomForest.from_dict(bundle["forest"])
                    self.training_sources = bundle.get("training_sources", [])
                    self.synthetic_only = bundle.get("synthetic_only", True)
                    self.dataset_license = bundle.get("dataset_license")
                    self.dataset_attribution = bundle.get("dataset_attribution")
            else:
                self.error = "No trained model found. Run: python -m spectrasense.ml.train_model"
        except Exception as exc:
            self.error = f"Could not load model: {exc}"

    @property
    def available(self):
        return self.model is not None

    def predict(self, features):
        if not self.available:
            return {}
        probs = self.model.predict_proba_one(feature_vector(features).tolist())
        return {str(label): float(prob) for label, prob in zip(self.model.classes, probs)}

    def predict_signal(self, iq_data, sample_rate, max_windows=32):
        """Average predictions from the same 512-sample windows used by synthetic training."""
        if not self.available or len(iq_data) < FEATURE_WINDOW_SAMPLES:
            return {}
        from spectrasense.dsp.preprocess import preprocess_pipeline
        from spectrasense.dsp.features import extract_all_features

        window_size = FEATURE_WINDOW_SAMPLES
        window_count = min(max_windows, max(1, len(iq_data) // window_size))
        starts = np.linspace(0, len(iq_data) - window_size, window_count, dtype=int)
        totals = {label: 0.0 for label in self.model.classes}
        used = 0
        for start in sorted(set(starts.tolist())):
            window = iq_data[start:start + window_size]
            window, _ = preprocess_pipeline(window, sample_rate=sample_rate)
            features = extract_all_features(window, sample_rate)
            scores = self.model.predict_proba_one(feature_vector(features, sample_rate).tolist())
            for label, score in zip(self.model.classes, scores):
                totals[label] += score
            used += 1
        return {label: score / max(used, 1) for label, score in totals.items()}
