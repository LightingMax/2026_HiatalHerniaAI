$ErrorActionPreference = 'Stop'

Write-Host '[1/3] Installing dependencies (Windows CPU)...'
python -m pip install --upgrade pip
python -m pip install -r requirements_windows_cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

Write-Host '[2/3] Cleaning previous build artifacts...'
if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path dist) { Remove-Item dist -Recurse -Force }

Write-Host '[3/3] Building EXE...'
pyinstaller --clean --noconfirm liekongshanapp.spec

Write-Host 'Build complete. Output: dist/HiatalHerniaAI_CPU.exe'
