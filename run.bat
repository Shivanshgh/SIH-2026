@echo off
echo ====================================================================
echo  SpectraSense - Signal Workflow Orchestration & Evidence Layer
echo  Smart India Hackathon 2026 (Problem Statement 26147 - NTRO)
echo ====================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Setup virtual environment if not exists
if not exist "venv\Scripts\activate.bat" (
    echo [*] Creating Python virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
) else (
    call venv\Scripts\activate.bat
)

echo [*] Checking required packages...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)

REM Generate synthetic test signals if missing
if not exist "samples\bpsk_25k_18db.iq" (
    echo [*] Generating synthetic sample IQ/WAV files...
    python generate_samples.py
)

echo [*] Launching SpectraSense Windows Desktop Application (PyQt6)...
python main.py

pause
