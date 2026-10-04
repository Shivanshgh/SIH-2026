"""
SpectraSense - Main Application Window (PyQt6)
Automated Analysis of .IQ and .wav Files along with Signal Parameter Extraction

Features:
- Windows desktop architecture with native menus, status bar, and dockable sidebars
- Embedded Matplotlib canvases for Time-Domain, PSD, Spectrogram, and Constellation
- Stage Navigator: Characterize -> Hypothesize -> Process -> Validate
- Ranked Multi-Hypothesis view with confidence scores
- Bounded Refinement & Honest Uncertainty loop
- Structured Signal Profile export (JSON & HTML)
"""

import os
import sys

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QCheckBox, QTextEdit, QFrame,
    QMessageBox, QProgressBar, QHeaderView, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QFont, QColor

from spectrasense.dsp.io import SignalFormat, infer_iq_format
from spectrasense.engine.pipeline import SpectraSensePipeline
from spectrasense.engine.profile import SignalJSONEncoder, export_profile_json, generate_profile_html
from spectrasense.gui.theme import DARK_STYLESHEET

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    np = None


class SpectraSenseMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraSense — Signal Workflow Orchestration & Evidence Layer")
        self.resize(1340, 860)
        self.setStyleSheet(DARK_STYLESHEET)

        self.pipeline = SpectraSensePipeline(sample_rate=200000)
        self.current_filepath = None
        self.refinement_count = 0

        self.init_ui()
        self.create_menus()
        self.init_sample_signals()

    def create_menus(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open IQ / WAV File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.on_open_file)
        file_menu.addAction(open_action)

        export_json_action = QAction("Export &Structured Profile (JSON)...", self)
        export_json_action.setShortcut("Ctrl+S")
        export_json_action.triggered.connect(self.on_export_json)
        file_menu.addAction(export_json_action)

        export_html_action = QAction("Export &Report (HTML / PDF Print)...", self)
        export_html_action.triggered.connect(self.on_export_html)
        file_menu.addAction(export_html_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Pipeline Menu
        pipeline_menu = menubar.addMenu("&Pipeline")
        run_all_action = QAction("&Execute Complete Pipeline", self)
        run_all_action.setShortcut("F5")
        run_all_action.triggered.connect(self.on_run_full_pipeline)
        pipeline_menu.addAction(run_all_action)

        refine_action = QAction("&Trigger Bounded Refinement Loop", self)
        refine_action.setShortcut("F6")
        refine_action.triggered.connect(self.on_trigger_refine)
        pipeline_menu.addAction(refine_action)

        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        sample_gen_action = QAction("&Generate Synthetic Test Signals...", self)
        sample_gen_action.triggered.connect(self.on_generate_samples_dialog)
        tools_menu.addAction(sample_gen_action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About SpectraSense", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter)

        # Scrollable, user-resizable controls column.
        sidebar = self.build_sidebar()
        sidebar.setMinimumWidth(250)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setMinimumWidth(250)
        splitter.addWidget(sidebar_scroll)
        splitter.setStretchFactor(0, 0)

        # Right Main Workspace (Tabs: Visualizations, Parameters, Hypotheses, Validation, Profile)
        workspace = self.build_workspace()
        workspace.setMinimumWidth(480)
        splitter.addWidget(workspace)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1020])
        self.main_splitter = splitter

        # Bottom Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("SpectraSense Ready. Load an .iq or .wav signal to begin workflow orchestration.")

    def build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebarPanel")
        sidebar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Header Title
        title_label = QLabel("SPECTRASENSE")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")
        sub_label = QLabel("Signal Workflow Orchestration & Evidence Layer")
        sub_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(title_label)
        layout.addWidget(sub_label)
        self.ml_status_label = QLabel()
        self.ml_status_label.setWordWrap(True)
        layout.addWidget(self.ml_status_label)
        self.update_ml_status()

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #232d3d;")
        layout.addWidget(divider)

        # Stage 1: Ingest Card
        ingest_box = QFrame()
        ingest_box.setObjectName("cardPanel")
        ib_layout = QVBoxLayout(ingest_box)
        ib_layout.setSpacing(8)

        lbl_ingest = QLabel("STAGE 1: INPUT INGEST")
        lbl_ingest.setObjectName("sectionHeader")
        ib_layout.addWidget(lbl_ingest)

        self.btn_browse = QPushButton("Load Signal File (.iq / .wav)")
        self.btn_browse.setObjectName("primaryButton")
        self.btn_browse.clicked.connect(self.on_open_file)
        ib_layout.addWidget(self.btn_browse)

        format_hint = QLabel("Raw IQ format (manual choice overrides extension):")
        format_hint.setWordWrap(True)
        ib_layout.addWidget(format_hint)
        self.format_combo = QComboBox()
        self.format_combo.addItem("Auto (from extension)", None)
        self.format_combo.addItem("Complex Float32 (fc32 / cfile)", SignalFormat.FLOAT32_IQ)
        self.format_combo.addItem("Complex Int16 (sc16)", SignalFormat.INT16_IQ)
        self.format_combo.addItem("Complex Int8 (sc8)", SignalFormat.INT8_IQ)
        ib_layout.addWidget(self.format_combo)

        ib_layout.addWidget(QLabel("Analysis size:"))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItem("Quick · 100k samples", 100_000)
        self.analysis_combo.addItem("Standard · 500k samples", 500_000)
        self.analysis_combo.addItem("Large · 1M samples", 1_000_000)
        self.analysis_combo.setCurrentIndex(1)
        ib_layout.addWidget(self.analysis_combo)

        lbl_samples = QLabel("Quick Load Synthetic Case:")
        lbl_samples.setStyleSheet("font-size: 11px; color: #94a3b8;")
        ib_layout.addWidget(lbl_samples)

        self.sample_combo = QComboBox()
        self.sample_combo.currentIndexChanged.connect(self.on_sample_selected)
        ib_layout.addWidget(self.sample_combo)

        lbl_sr = QLabel("Sample Rate (Sps):")
        lbl_sr.setStyleSheet("font-size: 11px; color: #94a3b8;")
        ib_layout.addWidget(lbl_sr)

        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(1000, 100000000)
        self.sr_spin.setValue(200000)
        self.sr_spin.setSingleStep(50000)
        ib_layout.addWidget(self.sr_spin)

        layout.addWidget(ingest_box)

        # Stage 2: Preprocess Settings
        pre_box = QFrame()
        pre_box.setObjectName("cardPanel")
        pb_layout = QVBoxLayout(pre_box)
        pb_layout.setSpacing(6)

        lbl_pre = QLabel("STAGE 2: PRE-PROCESSING")
        lbl_pre.setObjectName("sectionHeader")
        pb_layout.addWidget(lbl_pre)

        self.chk_dc = QCheckBox("DC Offset Removal")
        self.chk_dc.setChecked(True)
        pb_layout.addWidget(self.chk_dc)

        self.chk_norm = QCheckBox("Energy Normalization")
        self.chk_norm.setChecked(True)
        pb_layout.addWidget(self.chk_norm)

        self.chk_filter = QCheckBox("Digital Bandpass Filter")
        self.chk_filter.setChecked(False)
        pb_layout.addWidget(self.chk_filter)

        layout.addWidget(pre_box)

        # Pipeline Execution Buttons
        exec_box = QFrame()
        exec_box.setObjectName("cardPanel")
        eb_layout = QVBoxLayout(exec_box)
        eb_layout.setSpacing(8)

        lbl_orch = QLabel("ORCHESTRATION CONTROLS")
        lbl_orch.setObjectName("sectionHeader")
        eb_layout.addWidget(lbl_orch)

        self.btn_run_all = QPushButton("Run Pipeline (All Stages)")
        self.btn_run_all.setObjectName("accentButton")
        self.btn_run_all.clicked.connect(self.on_run_full_pipeline)
        eb_layout.addWidget(self.btn_run_all)

        self.btn_refine = QPushButton("Bounded Refinement Loop")
        self.btn_refine.setObjectName("refineButton")
        self.btn_refine.clicked.connect(self.on_trigger_refine)
        eb_layout.addWidget(self.btn_refine)

        layout.addWidget(exec_box)

        # Export Buttons
        export_box = QFrame()
        export_box.setObjectName("cardPanel")
        exp_layout = QVBoxLayout(export_box)
        exp_layout.setSpacing(8)

        lbl_exp = QLabel("STAGE 6: STRUCTURED PROFILE")
        lbl_exp.setObjectName("sectionHeader")
        exp_layout.addWidget(lbl_exp)

        self.btn_export_json = QPushButton("Export JSON Profile")
        self.btn_export_json.clicked.connect(self.on_export_json)
        exp_layout.addWidget(self.btn_export_json)

        self.btn_export_html = QPushButton("Export HTML / PDF Report")
        self.btn_export_html.clicked.connect(self.on_export_html)
        exp_layout.addWidget(self.btn_export_html)

        layout.addWidget(export_box)

        layout.addStretch()
        return sidebar

    def build_workspace(self):
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Visualizations (PSD, Time Domain, Waterfall, Constellation)
        self.vis_tab = self.build_visualizations_tab()
        self.tabs.addTab(self.vis_tab, "Signal Visualizations")

        # Tab 2: Parameter Extraction (Stage 3: Characterize)
        self.char_tab = self.build_characterize_tab()
        self.tabs.addTab(self.char_tab, "Stage 3: Characterize")

        # Tab 3: Hypothesize (Stage 4: Ranked Candidates)
        self.hypo_tab = self.build_hypothesize_tab()
        self.tabs.addTab(self.hypo_tab, "Stage 4: Hypothesize")

        # Tab 4: Process & Validate (Stage 5: Verification & Refinement)
        self.val_tab = self.build_validate_tab()
        self.tabs.addTab(self.val_tab, "Stage 5: Process & Validate")

        # Tab 5: Structured Profile (Stage 6: Output & JSON)
        self.profile_tab = self.build_profile_tab()
        self.tabs.addTab(self.profile_tab, "Stage 6: Structured Signal Profile")

        self.coding_tab = self.build_coding_demo_tab()
        self.tabs.addTab(self.coding_tab, "Signal Coding Analysis")

        return workspace

    def build_visualizations_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(10, 6), facecolor='#121822')
            self.canvas = FigureCanvas(self.fig)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.canvas.setMinimumSize(0, 0)
            
            # 2x2 Subplots:
            # [0, 0] = Time Domain (I & Q)
            # [0, 1] = Power Spectral Density (PSD)
            # [1, 0] = Spectrogram / Waterfall
            # [1, 1] = Constellation (I vs Q)
            self.ax_time = self.fig.add_subplot(221)
            self.ax_psd = self.fig.add_subplot(222)
            self.ax_spec = self.fig.add_subplot(223)
            self.ax_const = self.fig.add_subplot(224)

            self.format_axes()
            layout.addWidget(self.canvas)
        else:
            lbl = QLabel("Matplotlib not available. Install matplotlib to view graphics.")
            layout.addWidget(lbl)

        return tab

    def format_axes(self):
        for ax in [self.ax_time, self.ax_psd, self.ax_spec, self.ax_const]:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#94a3b8', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#232d3d')
            ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)

        self.ax_time.set_title("Time Domain: I (Cyan) & Q (Amber)", color='#cbd5e1', fontsize=10, pad=6)
        self.ax_psd.set_title("Power Spectral Density (PSD)", color='#cbd5e1', fontsize=10, pad=6)
        self.ax_spec.set_title("Spectrogram / Waterfall", color='#cbd5e1', fontsize=10, pad=6)
        self.ax_const.set_title("IQ Constellation Diagram", color='#cbd5e1', fontsize=10, pad=6)
        self.fig.tight_layout()

    def build_characterize_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QLabel("AUTOMATED PHYSICAL PARAMETER EXTRACTION (DSP EVIDENCE)")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #38bdf8;")
        layout.addWidget(header)

        self.param_table = QTableWidget(10, 3)
        self.param_table.setHorizontalHeaderLabels(["Parameter Metric", "Estimated Value", "Extraction Method"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.param_table)

        return tab

    def build_hypothesize_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QLabel("MULTI-CANDIDATE RANKED HYPOTHESES (STAGE 4)")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #38bdf8;")
        layout.addWidget(header)

        info = QLabel("Philosophy: Never force a single classification. Output multiple ranked candidates with confidence scores and evidence drivers.")
        info.setStyleSheet("color: #94a3b8; font-style: italic;")
        layout.addWidget(info)

        self.hypo_table = QTableWidget(6, 5)
        self.hypo_table.setHorizontalHeaderLabels(["Rank", "Modulation Candidate", "Relative score (%)", "Evidence Drivers", "Suggested Downstream Pipeline"])
        self.hypo_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.hypo_table)

        return tab

    def build_validate_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QLabel("DOWNSTREAM PROCESS & VALIDATION VERDICT (STAGE 5)")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #38bdf8;")
        layout.addWidget(header)

        # Status Verdict Banner
        self.verdict_banner = QLabel("STATUS: PENDING VALIDATION")
        self.verdict_banner.setStyleSheet("padding: 12px; font-size: 14px; font-weight: bold; background-color: #1e293b; color: #94a3b8; border-radius: 6px;")
        layout.addWidget(self.verdict_banner)

        # Details Table
        self.val_table = QTableWidget(8, 2)
        self.val_table.setHorizontalHeaderLabels(["Validation Metric", "Result"])
        self.val_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.val_table)

        # Uncertainty Notes Box
        lbl_unc = QLabel("AUDIT TRAIL & HONEST UNCERTAINTY NOTES:")
        lbl_unc.setStyleSheet("font-weight: bold; color: #f59e0b;")
        layout.addWidget(lbl_unc)

        self.txt_uncertainty = QTextEdit()
        self.txt_uncertainty.setReadOnly(True)
        self.txt_uncertainty.setStyleSheet("background-color: #141b24; color: #cbd5e1; border: 1px solid #334155; font-family: monospace;")
        layout.addWidget(self.txt_uncertainty)

        return tab

    def build_profile_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QLabel("STRUCTURED SIGNAL PROFILE (STAGE 6 - JSON AUDIT)")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #38bdf8;")
        layout.addWidget(header)

        self.txt_profile = QTextEdit()
        self.txt_profile.setReadOnly(True)
        self.txt_profile.setStyleSheet("background-color: #0d1117; color: #38bdf8; border: 1px solid #232d3d; font-family: 'Consolas', monospace; font-size: 12px;")
        layout.addWidget(self.txt_profile)

        return tab

    def build_coding_demo_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        header = QLabel("CURRENT SIGNAL CODING CHECK + SYNTHETIC DEMO")
        header.setStyleSheet("color: #38bdf8; font-weight: bold;")
        layout.addWidget(header)
        explanation = QLabel(
            "Analyze sync/FEC checks from the currently loaded signal, or run a separate known-parameter "
            "synthetic demonstration. Real-signal coding matches are tentative."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #94a3b8;")
        layout.addWidget(explanation)
        signal_button = QPushButton("Analyze Loaded Signal")
        signal_button.clicked.connect(self.on_analyze_loaded_signal_coding)
        layout.addWidget(signal_button)
        run_button = QPushButton("Run synthetic coding demo")
        run_button.clicked.connect(self.on_run_coding_demo)
        layout.addWidget(run_button)
        self.txt_coding_demo = QTextEdit()
        self.txt_coding_demo.setReadOnly(True)
        self.txt_coding_demo.setStyleSheet("background-color: #0d1117; color: #cbd5e1; font-family: Consolas;")
        layout.addWidget(self.txt_coding_demo)
        return tab

    def on_analyze_loaded_signal_coding(self):
        import json
        if not self.current_filepath or self.pipeline.raw_iq is None:
            self.txt_coding_demo.setPlainText("Load a signal and run the pipeline first.")
            return
        validation = self.pipeline.validation_result
        if not validation:
            self.txt_coding_demo.setPlainText("Run the pipeline on the loaded signal first.")
            return
        features = self.pipeline.features
        top = self.pipeline.hypotheses[0] if self.pipeline.hypotheses else {}
        result = {
            "analysis": "Coding and synchronization checks for the loaded signal",
            "source_file": os.path.basename(self.current_filepath),
            "sample_rate_hz": self.pipeline.sample_rate,
            "analyzed_samples": len(self.pipeline.raw_iq),
            "leading_modulation_hypothesis": top.get("modulation", "Unknown"),
            "estimated_snr_db": features.get("snr_db"),
            "validation_status": validation.get("status"),
            "recovered_bit_count": validation.get("recovered_bit_count", 0),
            "bit_recovery_method": validation.get("bit_recovery_method", ""),
            "sync_word_candidates": validation.get("sync_word_candidates", []),
            "fec_candidate_search": validation.get("fec_analysis", {}),
            "notice": "Supported candidate checks only; a match does not confirm framing or payload."
        }
        self.txt_coding_demo.setPlainText(json.dumps(result, indent=2, cls=SignalJSONEncoder))
        self.status_bar.showMessage(f"Coding checks shown for {os.path.basename(self.current_filepath)}.")

    def on_run_coding_demo(self):
        import json
        try:
            from spectrasense.engine.coding_demo import run_coding_demo
            result = run_coding_demo()
            self.txt_coding_demo.setPlainText(json.dumps(result, indent=2, cls=SignalJSONEncoder))
            self.status_bar.showMessage("Synthetic coding demo completed.")
        except Exception as exc:
            message = f"Coding demo failed: {exc}"
            self.txt_coding_demo.setPlainText(message)
            self.status_bar.showMessage(message)
            QMessageBox.critical(self, "Coding Demo Error", message)

    def init_sample_signals(self):
        resource_root = getattr(sys, "_MEIPASS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        self.sample_combo.addItem("-- Select Test Case --", "")
        samples = os.path.join(resource_root, "samples")
        self.sample_combo.addItem("1. BPSK (Fc=25kHz, Rs=10k, SNR=18dB)", os.path.join(samples, "bpsk_25k_18db.iq"))
        self.sample_combo.addItem("2. QPSK (Fc=40kHz, Rs=20k, SNR=16dB)", os.path.join(samples, "qpsk_40k_16db.iq"))
        self.sample_combo.addItem("3. 2-FSK (Tone Sep=15kHz, Rs=5k)", os.path.join(samples, "2fsk_30k_15db.iq"))
        self.sample_combo.addItem("4. 16-QAM (Baseband, 16 States)", os.path.join(samples, "16qam_bb_22db.iq"))
        self.sample_combo.addItem("5. Ambiguous Degraded (Low SNR 3dB, Co-channel CW)", os.path.join(samples, "ambiguous_degraded_case.iq"))
        self.sample_combo.addItem("6. Audio FSK (.wav format)", os.path.join(samples, "audio_fsk_sample.wav"))

    def on_sample_selected(self, index):
        path = self.sample_combo.currentData()
        if path and os.path.exists(path):
            self.load_and_process_signal(path)

    def on_open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Signal File", "", "Signal Files (*.iq *.fc32 *.cfile *.sc16 *.sc8 *.wav);;All Files (*.*)"
        )
        if filepath:
            self.load_and_process_signal(filepath)

    def load_and_process_signal(self, filepath):
        self.current_filepath = filepath
        self.refinement_count = 0
        sr = self.sr_spin.value()

        try:
            fmt = self.format_combo.currentData()
            ext = os.path.splitext(filepath)[1].lower()
            expected = infer_iq_format(filepath)
            if fmt and expected and fmt != expected:
                QMessageBox.warning(self, "Format differs from extension",
                                    f"Using your selected format for {os.path.basename(filepath)}.\n"
                                    "The extension suggests a different sample format.")
            if ext == ".wav":
                fmt = SignalFormat.FLOAT32_IQ  # WAV parser handles its own encoding.
            self.pipeline.load_file(filepath, sample_rate=sr, format_type=fmt,
                                    max_samples=int(self.analysis_combo.currentData()), representative=True)
            self.on_run_full_pipeline()
            loaded = len(self.pipeline.raw_iq)
            note = (" • large file: analyzing representative chunks" if self.pipeline.file_metadata.get("analysis_truncated") else "")
            self.status_bar.showMessage(f"Loaded: {os.path.basename(filepath)} ({loaded:,} samples){note}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")

    def on_run_full_pipeline(self):
        if not self.current_filepath or self.pipeline.raw_iq is None:
            QMessageBox.warning(self, "No Signal", "Please select a signal file first.")
            return

        sr = self.sr_spin.value()
        apply_dc = self.chk_dc.isChecked()
        apply_norm = self.chk_norm.isChecked()

        # Execute Pipeline
        self.pipeline.run_preprocess(apply_dc=apply_dc, apply_norm=apply_norm)
        features = self.pipeline.run_characterize()
        hypotheses = self.pipeline.run_hypothesize()
        assessment = self.pipeline.classification_assessment
        reasons = ", ".join(assessment.get("reasons", []))
        message = assessment.get("status", "Analysis complete")
        if reasons:
            message += f" — {reasons}"
        self.status_bar.showMessage(message)
        validation = self.pipeline.run_validation(refinement_attempt=self.refinement_count)
        profile = self.pipeline.generate_profile()
        self.update_ml_status()

        # Update Visualizations
        self.update_plots()

        # Update UI Tables & Views
        self.populate_characterize_tab(features)
        self.populate_hypothesize_tab(hypotheses)
        self.populate_validation_tab(validation)
        self.populate_profile_tab(profile)

    def on_trigger_refine(self):
        """Bounded refinement loop execution."""
        if self.pipeline.raw_iq is None:
            return
        self.refinement_count += 1
        validation = self.pipeline.run_validation(refinement_attempt=self.refinement_count)
        profile = self.pipeline.generate_profile()
        self.populate_validation_tab(validation)
        self.populate_profile_tab(profile)
        self.status_bar.showMessage(f"Refinement cycle {self.refinement_count} completed. Verdict updated.")

    def update_plots(self):
        if not MATPLOTLIB_AVAILABLE or self.pipeline.raw_iq is None:
            return

        sig = self.pipeline.preprocessed_iq if self.pipeline.preprocessed_iq is not None else self.pipeline.raw_iq
        sr = self.pipeline.sample_rate

        self.ax_time.clear()
        self.ax_psd.clear()
        self.ax_spec.clear()
        self.ax_const.clear()

        # 1. Time domain (first 400 samples)
        n_disp = min(400, len(sig))
        t = np.arange(n_disp) / (sr / 1000.0)  # ms
        self.ax_time.plot(t, np.real(sig[:n_disp]), color='#00d2ff', lw=1.0, label='I (In-Phase)')
        self.ax_time.plot(t, np.imag(sig[:n_disp]), color='#f59e0b', lw=1.0, label='Q (Quadrature)', alpha=0.8)
        self.ax_time.set_xlabel("Time (ms)", color='#94a3b8', fontsize=8)
        self.ax_time.set_ylabel("Amplitude", color='#94a3b8', fontsize=8)
        self.ax_time.legend(loc='upper right', facecolor='#161e2a', edgecolor='#232d3d', labelcolor='#cbd5e1', fontsize=7)

        # 2. Power Spectral Density (PSD)
        fft_vals = np.fft.fftshift(np.fft.fft(sig[:2048]))
        psd = 10.0 * np.log10(np.abs(fft_vals) ** 2 + 1e-12)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(fft_vals), 1.0 / (sr / 1000.0)))  # kHz
        self.ax_psd.plot(freqs, psd, color='#10b981', lw=1.0)
        self.ax_psd.set_xlabel("Frequency (kHz)", color='#94a3b8', fontsize=8)
        self.ax_psd.set_ylabel("Power (dB)", color='#94a3b8', fontsize=8)

        # Mark carrier frequency
        fc_khz = self.pipeline.features.get("carrier_freq_hz", 0) / 1000.0
        self.ax_psd.axvline(fc_khz, color='#ef4444', linestyle=':', label=f"Fc: {fc_khz:.1f} kHz")
        self.ax_psd.legend(loc='upper right', facecolor='#161e2a', edgecolor='#232d3d', labelcolor='#cbd5e1', fontsize=7)

        # 3. Spectrogram
        spec_n = min(4096, len(sig))
        self.ax_spec.specgram(sig[:spec_n], NFFT=128, Fs=sr / 1000.0, noverlap=64, cmap='viridis')
        self.ax_spec.set_xlabel("Time (ms)", color='#94a3b8', fontsize=8)
        self.ax_spec.set_ylabel("Freq (kHz)", color='#94a3b8', fontsize=8)

        # 4. Constellation Diagram
        sps = max(int(self.pipeline.features.get("samples_per_symbol", 8)), 2)
        const_syms = sig[::sps][:600]
        self.ax_const.scatter(np.real(const_syms), np.imag(const_syms), color='#00d2ff', s=10, alpha=0.7, edgecolors='none')
        self.ax_const.axhline(0, color='#334155', lw=0.8)
        self.ax_const.axvline(0, color='#334155', lw=0.8)
        self.ax_const.set_xlabel("In-Phase (I)", color='#94a3b8', fontsize=8)
        self.ax_const.set_ylabel("Quadrature (Q)", color='#94a3b8', fontsize=8)

        self.format_axes()
        self.canvas.draw()

    def populate_characterize_tab(self, f):
        params = [
            ("Carrier Frequency (Fc)", f"{f.get('carrier_freq_hz', 0):,.1f} Hz", "Welch Spectral Peak"),
            ("Carrier Centroid", f"{f.get('carrier_centroid_hz', 0):,.1f} Hz", "Power Energy Centroid"),
            ("3 dB Bandwidth", f"{f.get('bandwidth_3db_hz', 0):,.1f} Hz", "Half-Power Threshold"),
            ("99% Occupied Bandwidth", f"{f.get('bandwidth_99_hz', 0):,.1f} Hz", "Cumulative Spectral Energy"),
            ("Estimated SNR", f"{f.get('snr_db', 0):.1f} dB", "M2M4 Moment & Spectral Noise Floor"),
            ("Estimated Symbol Rate (Rs)", f"{f.get('symbol_rate_baud', 0):,.1f} Baud", "Cyclostationary Non-linear Squaring"),
            ("Samples per Symbol (SpS)", f"{f.get('samples_per_symbol', 0)}", "Ratio Fs / Rs"),
            ("4th-Order Cumulant C40", f"{f.get('cumulant_c40', 0):.4f}", "Kurtosis-based Phase Symmetry"),
            ("4th-Order Cumulant C42", f"{f.get('cumulant_c42', 0):.4f}", "Conjugate 4th Moment Metric"),
            ("Envelope Variance", f"{f.get('envelope_variance', 0):.4f}", "Instantaneous Modulus Fluctuations")
        ]
        self.param_table.setRowCount(len(params))
        for row, (name, val, method) in enumerate(params):
            self.param_table.setItem(row, 0, QTableWidgetItem(name))
            self.param_table.setItem(row, 1, QTableWidgetItem(val))
            self.param_table.setItem(row, 2, QTableWidgetItem(method))

    def populate_hypothesize_tab(self, hypotheses):
        self.hypo_table.setRowCount(len(hypotheses))
        for row, h in enumerate(hypotheses):
            self.hypo_table.setItem(row, 0, QTableWidgetItem(f"#{h['rank']}"))
            self.hypo_table.setItem(row, 1, QTableWidgetItem(h['modulation']))
            self.hypo_table.setItem(row, 2, QTableWidgetItem(f"{h['confidence_pct']}%"))
            self.hypo_table.setItem(row, 3, QTableWidgetItem("; ".join(h['evidence'])))
            self.hypo_table.setItem(row, 4, QTableWidgetItem(h.get('suggested_pipeline', '-')))

    def populate_validation_tab(self, val):
        import json
        status = val.get("status", "PENDING")
        if status.startswith("VALIDATED"):
            self.verdict_banner.setStyleSheet("padding: 12px; font-size: 14px; font-weight: bold; background-color: #064e3b; color: #34d399; border-radius: 6px;")
        elif "REFINEMENT" in status:
            self.verdict_banner.setStyleSheet("padding: 12px; font-size: 14px; font-weight: bold; background-color: #451a03; color: #fbbf24; border-radius: 6px;")
        else:
            self.verdict_banner.setStyleSheet("padding: 12px; font-size: 14px; font-weight: bold; background-color: #450a0a; color: #f87171; border-radius: 6px;")

        self.verdict_banner.setText(f"VERDICT: {val.get('verdict', 'N/A')}")

        items = [
            ("Verification Status", status),
            ("EVM Residual", f"{val.get('evm_percent')}%" if val.get("evm_percent") is not None else "N/A"),
            ("Refinement Loops Executed", f"{val.get('refinement_count', 0)} cycles"),
            ("Processing Demodulator Used", val.get("processing_path_used", "N/A")),
            ("Conclusive Profile Approved", "YES" if val.get("passed") else "NO (Honest Uncertainty)"),
            ("Recovered Bit Preview", val.get("recovered_bits_preview") or "Unavailable / unresolved"),
            ("Recovered Bit Count", str(val.get("recovered_bit_count", 0))),
            ("Sync Word Candidates", json.dumps(val.get("sync_word_candidates", []), cls=SignalJSONEncoder) or "[]"),
            ("FEC / Interleaver Search", self._fec_summary(val.get("fec_analysis", {}))),
            ("Reed–Solomon Search", self._rs_summary(val.get("fec_analysis", {}).get("reed_solomon", {}))),
            ("LDPC Matrix Search", self._ldpc_summary(val.get("fec_analysis", {}).get("ldpc", {}))),
            ("Tentative FEC Decoded Preview", val.get("fec_analysis", {}).get("decoded_preview") or "Unavailable / no supported match")
        ]
        self.val_table.setRowCount(len(items))
        for row, (k, v) in enumerate(items):
            self.val_table.setItem(row, 0, QTableWidgetItem(k))
            self.val_table.setItem(row, 1, QTableWidgetItem(str(v)))

        self.txt_uncertainty.setPlainText("\n".join(f"• {n}" for n in val.get("uncertainty_notes", [])))

    @staticmethod
    def _fec_summary(result):
        if not result:
            return "Not analyzed"
        if result.get("status") == "CANDIDATE_FOUND":
            if result.get("code_family", "").startswith("RS("):
                return f"Tentative {result['code_family']} candidate"
            return (f"Tentative {result.get('code_family')} • {result.get('interleaver_layout')} interleaver, "
                    f"rows={result.get('interleaver_rows')} • "
                    f"fit error={100 * result.get('fit_error_rate', 0):.1f}%")
        return f"{result.get('status', 'Unknown')} • {result.get('note', '')}"

    @staticmethod
    def _rs_summary(result):
        status = result.get("status", "Not analyzed")
        if status == "CANDIDATE_FOUND":
            return f"Verified {result.get('code', 'RS candidate')} • corrected {result.get('corrected_bytes', 0)} bytes"
        return f"{status} • {result.get('note', '')}"

    @staticmethod
    def _ldpc_summary(result):
        status = result.get("status", "Not analyzed")
        if status == "CANDIDATE_FOUND":
            return (f"{result.get('code')} • corrected {result.get('corrected_bits')} bits • "
                    f"syndrome {result.get('syndrome_weight_before')}→{result.get('syndrome_weight_after')}")
        return f"{status} • {result.get('note', '')}"

    def update_ml_status(self):
        classifier = self.pipeline.ml_classifier
        if classifier.available:
            source_label = "+".join(classifier.training_sources) or "unknown source"
            self.ml_status_label.setText(f"OFFLINE ML loaded • blended with DSP evidence • {source_label}")
            self.ml_status_label.setStyleSheet("color: #34d399; font-size: 10px;")
        else:
            self.ml_status_label.setText(f"OFFLINE ML unavailable • {classifier.error}")
            self.ml_status_label.setStyleSheet("color: #fbbf24; font-size: 10px;")

    def populate_profile_tab(self, profile):
        import json
        self.txt_profile.setPlainText(json.dumps(profile, indent=2, cls=SignalJSONEncoder))

    def on_export_json(self):
        if not self.pipeline.profile:
            QMessageBox.warning(self, "No Profile", "Run the pipeline first to generate a profile.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Export JSON Profile", "signal_profile.json", "JSON Files (*.json)")
        if filepath:
            export_profile_json(self.pipeline.profile, filepath)
            QMessageBox.information(self, "Export Complete", f"Structured Signal Profile exported to:\n{filepath}")

    def on_export_html(self):
        if not self.pipeline.profile:
            QMessageBox.warning(self, "No Profile", "Run the pipeline first to generate a profile.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Export HTML Report", "signal_profile_report.html", "HTML Files (*.html)")
        if filepath:
            html = generate_profile_html(self.pipeline.profile)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            QMessageBox.information(self, "Export Complete", f"Report saved to:\n{filepath}")

    def on_generate_samples_dialog(self):
        import generate_samples
        generate_samples.main()
        self.init_sample_signals()
        QMessageBox.information(self, "Samples Generated", "Synthetic signals generated successfully into samples/ directory.")

    def on_about(self):
        QMessageBox.about(
            self,
            "About SpectraSense",
            "<h3>SpectraSense v1.0.0</h3>"
            "<p><b>Automated Analysis of .IQ and .wav Files</b><br>Signal Parameter Extraction &amp; Evidence Layer</p>"
            "<p>Workflow Orchestration + Evidence Layer for signal identification.<br>"
            "Pipeline: <b>CHARACTERIZE &rarr; HYPOTHESIZE &rarr; PROCESS &rarr; VALIDATE</b></p>"
            "<p>Architected for native Windows execution with PyQt6.</p>"
        )
