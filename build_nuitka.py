#!/usr/bin/env python3
"""
Nuitka build script for Raw Alchemy Studio

Usage:
    python build_nuitka.py

Requirements:
    pip install nuitka ordered-set zstandard
"""

import sys
import os
import platform
import subprocess

def get_nuitka_command():
    """Generate Nuitka build command based on platform"""
    
    system = platform.system()
    
    # Base command
    cmd = [
        sys.executable,
        "-m", "nuitka",
        "--standalone",  # 独立模式,包含所有依赖
        "--onefile",     # 单文件模式(可选,也可以用 --standalone 生成文件夹)
        
        # 入口点
        "src/raw_alchemy/gui.py",
        
        # 输出设置
        "--output-dir=dist",
        "--output-filename=RawAlchemy",
        
        # 优化选项
        "--lto=yes",  # Link Time Optimization
        "--jobs=32",   # 并行编译
        
        # Python 标志
        "--python-flag=no_site",
        "--python-flag=no_warnings",
        
        # 包含必要的包
        "--include-package=numpy",
        "--include-package=numba",
        "--include-package=rawpy",
        "--include-package=colour",
        "--include-package=scipy",
        "--include-package=PIL",
        "--include-package=pillow_heif",
        "--include-package=PyQt6",
        "--include-package=qfluentwidgets",
        "--include-package=send2trash",
        
        # 包含数据文件
        "--include-data-dir=src/raw_alchemy/vendor=vendor",
        "--include-data-dir=src/raw_alchemy/locales=locales",
        "--include-data-files=icon.ico=icon.ico",
        "--include-data-files=icon.png=icon.png",
        
        # PyQt6 插件支持
        "--enable-plugin=pyqt6",
        
        # 排除不需要的模块以减小体积
        "--nofollow-import-to=pandas",
        "--nofollow-import-to=IPython",
        "--nofollow-import-to=PyQt5",
        "--nofollow-import-to=PySide2",
        "--nofollow-import-to=test",
        "--nofollow-import-to=distutils",
        "--nofollow-import-to=setuptools",
        
        # 警告控制
        "--assume-yes-for-downloads",
        "--warn-implicit-exceptions",
        "--warn-unusual-code",
    ]
    
    # 平台特定设置
    if system == "Windows":
        cmd.extend([
            "--windows-icon-from-ico=icon.ico",
            "--windows-console-mode=disable",  # 新版参数：GUI 应用不显示控制台
            "--windows-company-name=Raw Alchemy",
            "--windows-product-name=Raw Alchemy Studio",
            "--windows-file-version=0.3.2",
            "--windows-product-version=0.3.2",
        ])
    elif system == "Darwin":  # macOS
        cmd.extend([
            "--macos-create-app-bundle",
            "--macos-app-icon=icon.icns",
            "--macos-app-name=RawAlchemy",
        ])
        # macOS 上不需要 strip
    elif system == "Linux":
        cmd.extend([
            "--linux-icon=icon.png",
        ])
    
    return cmd

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import nuitka
        # Nuitka 没有 __version__ 属性，尝试通过命令行获取版本
        try:
            result = subprocess.run([sys.executable, "-m", "nuitka", "--version"],
                                  capture_output=True, text=True, timeout=5)
            version = result.stdout.strip().split('\n')[0] if result.stdout else "installed"
            print(f"✓ Nuitka: {version}")
        except:
            print("✓ Nuitka installed")
    except ImportError:
        print("✗ Nuitka not found. Install with: pip install nuitka")
        return False
    
    # 检查其他依赖
    required = ['numpy', 'numba', 'rawpy', 'colour', 'PyQt6', 'qfluentwidgets']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg} found")
        except ImportError:
            missing.append(pkg)
            print(f"✗ {pkg} not found")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    print("=" * 60)
    print("Raw Alchemy Studio - Nuitka Build Script")
    print("=" * 60)
    print()
    
    # 检查依赖
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\nStarting Nuitka compilation...")
    print("This may take 10-30 minutes depending on your system.")
    print()
    
    # 生成命令
    cmd = get_nuitka_command()
    
    # 打印命令(用于调试)
    print("Build command:")
    print(" ".join(cmd))
    print()
    
    # 执行编译
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("✓ Build completed successfully!")
        print("=" * 60)
        print(f"\nExecutable location: dist/RawAlchemy{'.exe' if platform.system() == 'Windows' else ''}")
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("✗ Build failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
