@echo off
echo ============================================
echo   aeras Environment Setup
echo ============================================
echo.

echo [1/4] Creating virtual environment with Python 3.11...
py -3.11 -m venv .venv

echo [2/4] Activating environment...
call .venv\Scripts\activate.bat

echo [3/4] Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [4/4] Installing project dependencies...
pip install -r requirements.txt

echo.
echo   Activate with: .venv\Scripts\activate
pause
