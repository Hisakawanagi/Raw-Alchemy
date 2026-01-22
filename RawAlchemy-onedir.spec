# -*- mode: python ; coding: utf-8 -*-
import sys


# --- Platform-specific settings ---
# Enable strip on Linux and macOS for a smaller executable.
# On Windows, stripping can sometimes cause issues with antivirus software
# or runtime behavior, so it's safer to leave it disabled.
strip_executable = True if sys.platform.startswith('linux') else False

# --- Platform-specific binaries ---
import os
import glob

binaries_list = []

# Add math_ops_ext compiled module (.pyd on Windows, .so on Linux/macOS)
# First check in src directory (development)
math_ops_ext_files = glob.glob(os.path.join('src', 'raw_alchemy', 'math_ops_ext*.pyd'))
if not math_ops_ext_files:
    math_ops_ext_files = glob.glob(os.path.join('src', 'raw_alchemy', 'math_ops_ext*.so'))
# If not found, check in installed package
if not math_ops_ext_files:
    try:
        import raw_alchemy
        pkg_dir = os.path.dirname(raw_alchemy.__file__)
        math_ops_ext_files = glob.glob(os.path.join(pkg_dir, 'math_ops_ext*.pyd'))
        if not math_ops_ext_files:
            math_ops_ext_files = glob.glob(os.path.join(pkg_dir, 'math_ops_ext*.so'))
    except ImportError:
        pass

for pyd_file in math_ops_ext_files:
    binaries_list.append((pyd_file, 'raw_alchemy'))

if sys.platform == 'darwin' or sys.platform.startswith('linux'):
    import rawpy

    # Find the path to libraw_r library within the rawpy package
    rawpy_path = os.path.dirname(rawpy.__file__)
    lib_file = None
    for f in os.listdir(rawpy_path):
        if f.startswith('libraw_r'):
            lib_file = os.path.join(rawpy_path, f)
            break
    if lib_file:
        binaries_list.append((lib_file, '.'))


a = Analysis(
    ['src/raw_alchemy/gui.py'],
    pathex=[],
    binaries=binaries_list,
    datas=[('src/raw_alchemy/vendor', 'vendor'),('src/raw_alchemy/locales', 'locales'), ('icon.ico', '.'), ('icon.png', '.')],
    hiddenimports=['tkinter', 'loguru', 'pyexiv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pandas',
        'IPython',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'qtpy',
        'test',
        'doctest',
        'distutils',
        'setuptools',
        'wheel',
        'pkg_resources',
        'Cython',
        'PyInstaller',
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RawAlchemy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=strip_executable,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=strip_executable,
    upx=False,
    upx_exclude=[],
    name='RawAlchemy',
)