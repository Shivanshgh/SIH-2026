"""
SpectraSense Master Pipeline Coordinator
Orchestrates:
  STAGE 1: INPUT & FORMAT HANDLING
  STAGE 2: PRE-PROCESSING (DC removal, normalization, filtering)
  STAGE 3: CHARACTERIZE (Parameter extraction: Fc, BW, SNR, Rs, Cumulants)
  STAGE 4: HYPOTHESIZE (Multi-candidate ranked scoring, honest distribution)
  STAGE 5: PROCESS & VALIDATE (Downstream demod trial, EVM check, refinement loop, unresolved flag)
  STAGE 6: STRUCTURED SIGNAL PROFILE (JSON & HTML report generation)
"""

from spectrasense.dsp.io import (read_iq_file, read_wav_file, get_file_metadata,
                                 SignalFormat, infer_iq_format, iq_pair_bytes)
from spectrasense.dsp.preprocess import preprocess_pipeline
from spectrasense.dsp.features import extract_all_features
from spectrasense.engine.hypothesis_engine import score_hypotheses
from spectrasense.engine.validation import run_process_and_validate
from spectrasense.engine.profile import build_signal_profile, generate_profile_html
from spectrasense.engine.ml_classifier import ModulationClassifier


class SpectraSensePipeline:
    def __init__(self, sample_rate=200000):
        self.sample_rate = sample_rate
        self.raw_iq = None
        self.preprocessed_iq = None
        self.file_metadata = {}
        self.preprocess_metadata = {}
        self.features = {}
        self.hypotheses = []
        self.validation_result = {}
        self.profile = None
        self.ml_classifier = ModulationClassifier()
        self.ml_scores = {}
        self.classification_assessment = {}

    def load_file(self, filepath, format_type=None, sample_rate=None,
                  max_samples=500_000, representative=True):
        """Stage 1: Load file and extract metadata."""
        if sample_rate is not None:
            self.sample_rate = sample_rate
            
        self.file_metadata = get_file_metadata(filepath)
        
        if filepath.lower().endswith(".wav"):
            sig, sr = read_wav_file(filepath, max_samples=max_samples, representative=representative)
            self.raw_iq = sig
            self.sample_rate = sr
            self.file_metadata["sample_rate"] = sr
        else:
            self.raw_iq = read_iq_file(filepath, format_type=format_type,
                                       max_samples=max_samples, representative=representative)
            self.file_metadata["sample_rate"] = self.sample_rate
            chosen = infer_iq_format(filepath) if format_type is None or str(format_type).lower() == "auto" else format_type
            bytes_per_pair = iq_pair_bytes(chosen)
            if bytes_per_pair:
                self.file_metadata["available_sample_count"] = self.file_metadata["size_bytes"] // bytes_per_pair
                self.file_metadata["selected_format"] = chosen
            
        self.file_metadata["sample_count"] = len(self.raw_iq)
        available = self.file_metadata.get("available_sample_count")
        self.file_metadata["analysis_truncated"] = bool(available and len(self.raw_iq) < available)
        self.file_metadata["analysis_method"] = "representative chunks" if self.file_metadata["analysis_truncated"] and representative else "leading segment"
        return self.file_metadata

    def run_preprocess(self, apply_dc=True, apply_norm=True, filter_params=None):
        """Stage 2: Pre-processing."""
        if self.raw_iq is None:
            raise ValueError("No signal loaded. Run load_file() first.")
        self.preprocessed_iq, self.preprocess_metadata = preprocess_pipeline(
            self.raw_iq,
            sample_rate=self.sample_rate,
            apply_dc=apply_dc,
            apply_norm=apply_norm,
            filter_params=filter_params
        )
        return self.preprocess_metadata

    def run_characterize(self):
        """Stage 3: Characterize and extract physical DSP features."""
        sig = self.preprocessed_iq if self.preprocessed_iq is not None else self.raw_iq
        self.features = extract_all_features(sig, self.sample_rate)
        return self.features

    def run_hypothesize(self):
        """Stage 4: Generate multi-candidate ranked hypotheses."""
        if not self.features:
            self.run_characterize()
        sig = self.preprocessed_iq if self.preprocessed_iq is not None else self.raw_iq
        self.ml_scores = self.ml_classifier.predict_signal(sig, self.sample_rate)
        self.hypotheses = score_hypotheses(self.features, ml_scores=self.ml_scores)
        top = self.hypotheses[0] if self.hypotheses else {}
        runner_up = self.hypotheses[1] if len(self.hypotheses) > 1 else {}
        top_score = float(top.get("confidence", 0.0))
        margin = top_score - float(runner_up.get("confidence", 0.0))
        reasons = []
        if float(self.features.get("snr_db", 0.0)) < 6.0:
            reasons.append("low SNR")
        if top_score < 0.50:
            reasons.append("low top score")
        if margin < 0.12:
            reasons.append("close alternatives")
        ml_confidence = max(self.ml_scores.values(), default=0.0)
        if self.ml_classifier.available and ml_confidence < 0.45:
            reasons.append("low ML confidence")
        self.classification_assessment = {
            "status": "REVIEW RECOMMENDED" if reasons else "LEADING CANDIDATE",
            "candidate": top.get("modulation"), "top_score": round(top_score, 4),
            "margin_over_runner_up": round(margin, 4), "reasons": reasons,
            "ml_top_probability": round(ml_confidence, 4),
            "note": "Scores are relative model/DSP evidence, not calibrated probabilities.",
        }
        return self.hypotheses

    def run_validation(self, refinement_attempt=0):
        """Stage 5: Downstream process & validation check."""
        if not self.hypotheses:
            self.run_hypothesize()
        top_h = self.hypotheses[0]
        sig = self.preprocessed_iq if self.preprocessed_iq is not None else self.raw_iq
        self.validation_result = run_process_and_validate(
            sig,
            top_h,
            self.features,
            refinement_attempt=refinement_attempt,
            sample_rate=self.sample_rate
        )
        return self.validation_result

    def generate_profile(self):
        """Stage 6: Assemble final Structured Signal Profile."""
        self.profile = build_signal_profile(
            self.file_metadata,
            self.preprocess_metadata,
            self.features,
            self.hypotheses,
            self.validation_result
        )
        self.profile["ml_assistance"] = {
            "enabled": self.ml_classifier.available,
            "model_file": "models/modulation_model.json",
            "training_sources": self.ml_classifier.training_sources,
            "synthetic_only": self.ml_classifier.synthetic_only,
            "dataset_license": self.ml_classifier.dataset_license,
            "dataset_attribution": self.ml_classifier.dataset_attribution,
            "ml_scores": self.ml_scores,
            "notice": "Model scores are supporting evidence, not field-accuracy claims."
            if self.ml_classifier.available else self.ml_classifier.error,
        }
        self.profile["classification_assessment"] = self.classification_assessment
        self.profile["processing_and_validation"]["recovered_bits_preview"] = self.validation_result.get("recovered_bits_preview", "")
        self.profile["processing_and_validation"]["recovered_bit_count"] = self.validation_result.get("recovered_bit_count", 0)
        self.profile["processing_and_validation"]["bit_recovery_method"] = self.validation_result.get("bit_recovery_method", "")
        self.profile["processing_and_validation"]["sync_word_candidates"] = self.validation_result.get("sync_word_candidates", [])
        self.profile["processing_and_validation"]["fec_analysis"] = self.validation_result.get("fec_analysis", {})
        return self.profile

    def execute_all(self, filepath, sample_rate=200000):
        """Executes full pipeline end-to-end."""
        self.load_file(filepath, sample_rate=sample_rate)
        self.run_preprocess()
        self.run_characterize()
        self.run_hypothesize()
        self.run_validation(refinement_attempt=0)
        
        # Bounded refinement loop: if marginal, attempt 1 bounded refinement cycle
        if self.validation_result.get("status") == "REFINEMENT_RECOMMENDED":
            self.run_validation(refinement_attempt=1)
            
        return self.generate_profile()
