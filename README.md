# SpectraSense

Windows desktop app for IQ/WAV signal analysis.

## Run
- **Windows app:** extract the repository and open `SpectraSense.exe`.
- **Run from source:** install Python 3.10–3.12, then open `run.bat` (downloads dependencies listed in `requirements.txt`).

The offline model and example signals are included. `generate_samples.py` recreates examples if removed. To retrain the model: `python -m spectrasense.ml.train_model`.

The model proposes candidates from 13 common families (BPSK, QPSK, 8-PSK, 16/64-QAM, PAM4, 2/4-FSK, CPFSK, GFSK, AM-DSB, AM-SSB, WBFM). Its 64.1% score is from a synthetic holdout only, not a real-world accuracy claim. Other signals can be inspected, but recognition is not guaranteed. Real over-the-air recordings are needed to measure field performance.


Latest model build: run SpectraSense-latest.exe. The original SpectraSense.exe was already running during this update and remains in use; close it before replacing the original filename.
