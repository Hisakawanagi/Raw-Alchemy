#!/usr/bin/env python3
"""
Nuitka FAST build script for Raw Alchemy Studio
使用增量编译和优化选项来加速构建过程

Usage:
    python build_nuitka_fast.py

特点:
    - 使用 --onedir 模式（比 --onefile 快很多）
    - 禁用 LTO 优化（节省编译时间）
    - 启用增量编译缓存
    - 减少并行任务数以降低内存占用
"""

import sys
import os
import platform
import subprocess

def get_nuitka_command():
    """Generate fast Nuitka build command"""
    
    system = platform.system()
    
    # Base command - 优化编译速度
    cmd = [
        sys.executable,
        "-m", "nuitka",
        "--mingw64",
        "--standalone",  # 独立模式
        # 注意：使用 onedir 而不是 onefile，速度快很多
        
        # 入口点
        "src/raw_alchemy/gui.py",
        
        # 输出设置
        "--output-dir=dist",
        "--output-filename=RawAlchemy-Fast",
        
        # 优化选项 - 为速度优化
        "--lto=no",   # 禁用 LTO 以加快编译
        "--jobs=32",   # 减少并行任务数
        
        # Python 标志
        "--python-flag=no_site",
        
        # 包含必要的包
        "--include-package=numpy",
        "--include-package=numba",
        "--include-module=raw_alchemy.math_ops_ext",
        "--include-package=rawpy",
        "--include-package=colour",
        "--include-package=scipy",
        "--include-package=PIL",
        "--include-package=pillow_heif",
        "--include-package=PySide6",
        "--include-package=qfluentwidgets",
        "--include-package=send2trash",
        
        # 排除不需要的模块
        "--nofollow-import-to=nuitka",
        "--nofollow-import-to=pandas",
        "--nofollow-import-to=IPython",
        "--nofollow-import-to=PyQt6",
        "--nofollow-import-to=PyQt5",
        "--nofollow-import-to=PySide2",
        "--nofollow-import-to=test",
        "--nofollow-import-to=distutils",
        "--nofollow-import-to=setuptools",
        
        # 包含数据文件
        "--include-data-dir=src/raw_alchemy/vendor=raw_alchemy/vendor",
        "--include-data-dir=src/raw_alchemy/locales=raw_alchemy/locales",
        "--include-data-files=icon.ico=icon.ico",
        "--include-data-files=icon.png=icon.png",
        
        # PySide6 插件支持
        "--enable-plugin=pyside6",
        
        # 警告控制
        "--assume-yes-for-downloads",
        
        # 关键：启用增量编译
        "--show-progress",
        "--show-memory",
    ]
    
    # 强制包含 vendor 中的 DLL (Nuitka 默认可能会在 data-dir 中过滤掉 DLL)
    vendor_dir = os.path.join("src", "raw_alchemy", "vendor")
    if os.path.exists(vendor_dir):
        print(f"Scanning for DLLs in {vendor_dir}...")
        for root, _, files in os.walk(vendor_dir):
            for file in files:
                if file.lower().endswith(('.dll', '.so', '.dylib')):
                    src_path = os.path.join(root, file)
                    # 计算相对路径: src/raw_alchemy/vendor/... -> raw_alchemy/vendor/...
                    # 我们希望它在 dist 中的位置与源码结构一致
                    rel_path = os.path.relpath(src_path, "src")
                    print(f"  + Force including DLL: {src_path} -> {rel_path}")
                    cmd.append(f"--include-data-files={src_path}={rel_path}")

    # 平台特定设置
    if system == "Windows":
        cmd.extend([
            "--windows-icon-from-ico=icon.ico",
            "--windows-console-mode=force",  # 显示控制台便于调试
            "--windows-company-name=Raw Alchemy",
            "--windows-product-name=Raw Alchemy Studio Fast",
            "--windows-file-version=0.3.2",
            "--windows-product-version=0.3.2",
        ])
    elif system == "Darwin":  # macOS
        cmd.extend([
            "--macos-create-app-bundle",
            "--macos-app-icon=icon.icns",
            "--macos-app-name=RawAlchemy-Debug",
        ])
    elif system == "Linux":
        cmd.extend([
            "--linux-icon=icon.png",
        ])
    
    return cmd

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import nuitka
        try:
            result = subprocess.run([sys.executable, "-m", "nuitka", "--version"],
                                  capture_output=True, text=True, timeout=5)
            version = result.stdout.strip().split('\n')[0] if result.stdout else "installed"
            print(f"[OK] Nuitka: {version}")
        except:
            print("[OK] Nuitka installed")
    except ImportError:
        print("[ERROR] Nuitka not found. Install with: pip install nuitka")
        return False
    
    # 检查其他依赖
    required = ['numpy', 'numba', 'rawpy', 'colour', 'PySide6', 'qfluentwidgets']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"[OK] {pkg} found")
        except ImportError:
            missing.append(pkg)
            print(f"[MISSING] {pkg} not found")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    print("=" * 60)
    print("Raw Alchemy Studio - Nuitka FAST Build Script")
    print("=" * 60)
    print()
    print("[FAST] 快速构建模式:")
    print("  - 使用 onedir 模式（输出为文件夹）")
    print("  - 禁用 LTO 优化")
    print("  - 启用控制台窗口")
    print("  - 编译时间约为正式版本的 50-70%")
    print()
    
    # 检查依赖
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\nStarting FAST Nuitka compilation...")
    print("This should take 5-15 minutes (faster than full build).")
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
        print("[OK] Build completed successfully!")
        print("=" * 60)
        
        if platform.system() == "Windows":
            exe_path = "dist/RawAlchemy-Fast.dist/RawAlchemy-Debug.exe"
        else:
            exe_path = "dist/RawAlchemy-Fast.dist/RawAlchemy-Debug"
        
        print(f"\n可执行文件位置: {exe_path}")
        print("\n⚠️  这是快速构建版本，用于测试。")
        print("    正式发布请使用: python build_nuitka.py")
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("[ERROR] Build failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
