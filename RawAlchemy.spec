# -*- mode: python ; coding: utf-8 -*-
import sys

# --- Platform-specific settings ---
# Enable strip on Linux and macOS for a smaller executable.
# On Windows, stripping can sometimes cause issues with antivirus software
# or runtime behavior, so it's safer to leave it disabled.
# On macOS, stripping the Python shared library can lead to runtime errors,
# such as "Failed to load Python shared library". Disabling strip for macOS
# is a safer approach to ensure all necessary symbols are preserved.
strip_executable = True if sys.platform.startswith('linux') else False


# --- Platform-specific binaries ---
binaries_list = []
if sys.platform == 'darwin' or sys.platform.startswith('linux'):
    import os
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
    hiddenimports=['tkinter', 'raw_alchemy.math_ops_ext'],
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

# On macOS, BUNDLE is used, which has its own icon parameter.
# The .ico format is for Windows, so we remove it from datas on macOS.
if sys.platform == 'darwin':
    a.datas = [item for item in a.datas if item[0] != 'icon.ico']

pyz = PYZ(a.pure)

# Platform-specific EXE and BUNDLE for macOS .app creation
# Create a one-file executable.
# Binaries and data are included directly in the EXE for a one-file build.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RawAlchemy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=strip_executable,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Set the icon based on the platform.
    icon='icon.icns' if sys.platform == 'darwin' else 'icon.ico',
)

# If on macOS, bundle the one-file executable into a .app directory.
# This is required for a proper GUI application on macOS.
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='RawAlchemy.app',
        icon='icon.icns',
        bundle_identifier=None,
    )
