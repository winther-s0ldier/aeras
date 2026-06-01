@echo off
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
echo Activate with: .venv\Scripts\activate