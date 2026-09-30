# SpectraSense

Windows desktop app for IQ/WAV signal analysis.

## Run
- **Windows app:** extract the repository and open `SpectraSense.exe`.
- **Run from source:** install Python 3.10–3.12, then open `run.bat` (downloads dependencies listed in `requirements.txt`).

The offline model and example signals are included. `generate_samples.py` recreates examples if removed. To retrain the model: `python -m spectrasense.ml.train_model`.

