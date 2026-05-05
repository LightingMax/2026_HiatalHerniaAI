# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

project_dir = Path.cwd()

# Pack both weights if present
candidate_weights = [
    project_dir / "best_model_liekongshan.pth",
    project_dir / "best_model_liekongshan_2classes.pth",
]

datas = [(str(p), ".") for p in candidate_weights if p.exists()]
datas += collect_data_files("timm")
binaries = collect_dynamic_libs("torch")
hiddenimports = collect_submodules("timm")

icon_file = project_dir / "app_icon.ico"
icon_arg = str(icon_file) if icon_file.exists() else None

block_cipher = None


a = Analysis(
    ['liekongshanapp.py'],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HiatalHerniaAI_CPU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_arg,
)
