#!/usr/bin/env python3
"""
SpectraSense - Windows Desktop Application Entry Point
Smart India Hackathon 2026 (Problem Statement 26147 – NTRO / Space Technology)

To run on Windows:
    python -m venv venv
    venv\\Scripts\\activate
    pip install -r requirements.txt
    python main.py
"""

import sys
import os

# Ensure package root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def main():
    bundled_root = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    if getattr(sys, "frozen", False):
        sys.path.insert(0, bundled_root)
        # Explicitly prioritize the bundled Qt DLLs over any system Qt installation.
        qt_bin = os.path.join(bundled_root, "PyQt6", "Qt6", "bin")
        if os.path.isdir(qt_bin):
            os.environ["PATH"] = qt_bin + os.pathsep + os.environ.get("PATH", "")
            _qt_dll_directory = os.add_dll_directory(qt_bin)

    if "--self-test" in sys.argv[1:]:
        from PyQt6 import QtCore
        from PyQt6.QtWidgets import QApplication
        from spectrasense.engine.pipeline import SpectraSensePipeline
        sample = os.path.join(bundled_root, "samples", "qpsk_40k_16db.iq")
        profile = SpectraSensePipeline().execute_all(sample, sample_rate=200000)
        status = (f"Qt interface loaded (Qt {QtCore.qVersion()})\n"
                  f"Standalone check: {profile['processing_and_validation']['validation_status']}")
        if sys.stdout is not None:
            print(status)
        else:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, status, "SpectraSense self-test", 0x40)
        return

    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from spectrasense.gui.main_window import SpectraSenseMainWindow
    except ImportError as e:
        if getattr(sys, "frozen", False):
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None,
                f"SpectraSense could not load its desktop interface:\n\n{e}",
                "SpectraSense startup error",
                0x10,
            )
            return

        print("\n" + "=" * 60)
        print(" [!] PyQt6 is not yet installed in this environment.")
        print(f" Error: {e}")
        print(" Run: pip install -r requirements.txt")
        print("=" * 60 + "\n")
        
        # Test pipeline directly via CLI
        print("Executing SpectraSense CLI Signal Pipeline Test...")
        from spectrasense.engine.pipeline import SpectraSensePipeline
        pipe = SpectraSensePipeline()
        sample_path = os.path.join(bundled_root, "samples", "qpsk_40k_16db.iq")
        if not os.path.exists(sample_path):
            import generate_samples
            generate_samples.main()
            
        profile = pipe.execute_all(sample_path, sample_rate=200000)
        print(f"Pipeline executed successfully for: {sample_path}")
        print(f"Top Hypothesis: {profile['processing_and_validation']['top_candidate']}")
        print(f"Confidence: {profile['processing_and_validation']['top_candidate_confidence_pct']}%")
        print(f"Verdict: {profile['processing_and_validation']['validation_verdict']}")
        return

    # Enable High DPI scaling for modern Windows 10/11 4K displays
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("SpectraSense")
    app.setOrganizationName("NTRO-SIH2026")

    window = SpectraSenseMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
