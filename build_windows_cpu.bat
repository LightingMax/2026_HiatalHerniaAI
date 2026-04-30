@echo off
setlocal

REM Run this on a Windows machine (or Windows CI runner)
python --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found. Please install Python 3.10+ first.
  exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements_windows_cpu.txt
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

pyinstaller --clean --noconfirm liekongshanapp.spec

if errorlevel 1 (
  echo [ERROR] Build failed.
  exit /b 1
)

echo [OK] Build finished. EXE output in dist\HiatalHerniaAI_CPU.exe
endlocal
