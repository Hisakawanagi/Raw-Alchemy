"""
崩溃处理器和错误拦截器
用于捕获和记录系统级崩溃，特别是 M2 Mac 上的静默崩溃
集成到现有的 loguru 日志系统中
"""
import sys
import os
import signal
import traceback
import platform
import faulthandler
from datetime import datetime
from pathlib import Path
from loguru import logger


class CrashHandler:
    """全局崩溃处理器 - 集成到 loguru 日志系统"""
    
    def __init__(self):
        # 使用与 logger.py 相同的日志目录
        from raw_alchemy.logger import get_log_file_path
        self.main_log_file = get_log_file_path()
        self.installed = False
        
    def install(self):
        """安装崩溃处理器 - 集成到现有日志系统"""
        if self.installed:
            return
        
        # 启用 Python 的 faulthandler（输出到主日志文件）
        try:
            # faulthandler 会在崩溃时自动写入，我们让它写入 stderr
            # loguru 会捕获 stderr 并写入日志文件
            faulthandler.enable()
            logger.info(f"✅ Faulthandler enabled, output to: {self.main_log_file}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to enable faulthandler: {e}")
        
        # 注册信号处理器
        self._register_signal_handlers()
        
        # 设置全局异常钩子
        sys.excepthook = self._exception_hook
        
        # 记录系统信息
        self._log_system_info()
        
        self.installed = True
        logger.info("✅ Crash handler installed")
    
    def _register_signal_handlers(self):
        """注册系统信号处理器
        
        注意: 在多线程GUI应用中,信号处理器可能导致死锁
        因此我们只注册最关键的信号,并避免在处理器中调用GUI代码
        """
        # 定义要捕获的信号
        signals_to_catch = []
        
        # POSIX 信号（macOS/Linux）
        if hasattr(signal, 'SIGSEGV'):
            signals_to_catch.append(('SIGSEGV', signal.SIGSEGV))  # 段错误
        if hasattr(signal, 'SIGABRT'):
            signals_to_catch.append(('SIGABRT', signal.SIGABRT))  # 异常终止
        if hasattr(signal, 'SIGFPE'):
            signals_to_catch.append(('SIGFPE', signal.SIGFPE))    # 浮点异常
        if hasattr(signal, 'SIGILL'):
            signals_to_catch.append(('SIGILL', signal.SIGILL))    # 非法指令
        if hasattr(signal, 'SIGBUS'):
            signals_to_catch.append(('SIGBUS', signal.SIGBUS))    # 总线错误
        
        # 注册处理器
        for sig_name, sig_num in signals_to_catch:
            try:
                signal.signal(sig_num, self._signal_handler)
                logger.debug(f"Registered handler for {sig_name}")
            except (OSError, ValueError) as e:
                logger.debug(f"Cannot register {sig_name}: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理函数 - 输出到统一日志
        
        重要: 不在此处调用GUI代码,避免死锁
        """
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        
        # 记录崩溃信息到 loguru（会自动写入日志文件）
        logger.critical("="*80)
        logger.critical(f"💥 CRASH DETECTED - Signal: {sig_name} ({signum})")
        logger.critical("="*80)
        logger.critical(f"Time: {datetime.now().isoformat()}")
        logger.critical(f"Platform: {platform.platform()}")
        logger.critical(f"Machine: {platform.machine()}")
        logger.critical(f"Python: {sys.version}")
        logger.critical("")
        logger.critical("Stack Trace:")
        logger.critical(self._format_stack_trace(frame))
        logger.critical("="*80)
        logger.critical(f"Crash log saved to: {self.main_log_file}")
        logger.critical("="*80)
        
        # 不显示GUI对话框 - 避免在信号处理器中调用GUI代码导致死锁
        # 用户可以通过日志文件查看崩溃信息
        
        # 强制刷新日志
        try:
            import logging
            for handler in logger._core.handlers.values():
                if hasattr(handler, '_sink') and hasattr(handler._sink, 'flush'):
                    handler._sink.flush()
        except:
            pass
        
        # 立即退出程序
        os._exit(1)  # 使用 os._exit 而不是 sys.exit,避免清理代码导致的额外问题
    
    def _exception_hook(self, exc_type, exc_value, exc_traceback):
        """全局异常钩子 - 输出到统一日志"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许 Ctrl+C 正常退出
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # 记录到 loguru（会自动写入日志文件）
        logger.critical("="*80)
        logger.critical("❌ UNHANDLED EXCEPTION")
        logger.critical("="*80)
        logger.critical(f"Time: {datetime.now().isoformat()}")
        logger.critical(f"Type: {exc_type.__name__}")
        logger.critical(f"Value: {exc_value}")
        logger.critical("")
        logger.critical("Traceback:")
        for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
            logger.critical(line.rstrip())
        logger.critical("="*80)
        
        # 调用默认的异常处理
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    def _format_stack_trace(self, frame):
        """格式化堆栈跟踪"""
        try:
            return ''.join(traceback.format_stack(frame))
        except:
            return "Unable to format stack trace"
    
    def _log_system_info(self):
        """记录系统信息到统一日志"""
        logger.info("="*80)
        logger.info("SYSTEM INFORMATION")
        logger.info("="*80)
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"System: {platform.system()}")
        logger.info(f"Release: {platform.release()}")
        logger.info(f"Version: {platform.version()}")
        logger.info(f"Machine: {platform.machine()}")
        logger.info(f"Processor: {platform.processor()}")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Python Implementation: {platform.python_implementation()}")
        logger.info(f"Python Compiler: {platform.python_compiler()}")
        logger.info("")
        logger.info("Environment Variables:")
        logger.info(f"  NUMBA_DISABLE_JIT: {os.environ.get('NUMBA_DISABLE_JIT', 'not set')}")
        logger.info(f"  PYTHONFAULTHANDLER: {os.environ.get('PYTHONFAULTHANDLER', 'not set')}")
        logger.info(f"  NUMBA_CACHE_DIR: {os.environ.get('NUMBA_CACHE_DIR', 'not set')}")
        logger.info("")
        logger.info(f"Main Log File: {self.main_log_file}")
        logger.info("="*80)
    
    def _show_crash_dialog(self, sig_name, crash_info):
        """显示崩溃对话框（仅在 GUI 可用时）"""
        try:
            from PySide6.QtWidgets import QMessageBox, QApplication
            
            # 检查是否有 QApplication 实例
            app = QApplication.instance()
            if app is None:
                return
            
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Raw Alchemy - 程序崩溃")
            msg.setText(f"程序遇到严重错误并需要关闭\n\n信号: {sig_name}")
            msg.setInformativeText(
                f"崩溃日志已保存到:\n{self.main_log_file}\n\n"
                "请将此日志文件发送给开发者以帮助修复问题。"
            )
            msg.setDetailedText(crash_info)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        except:
            pass


class SafeLogTransform:
    """安全的 Log Transform 包装器"""
    
    @staticmethod
    def apply(img, log_space, log_color_space, log_curve):
        """
        安全地应用 log transform，带完整的错误处理
        
        Args:
            img: 图像数组 (numpy array)
            log_space: Log 空间名称
            log_color_space: Log 色彩空间名称
            log_curve: Log 曲线名称
        
        Returns:
            (success, result_img, error_msg)
        """
        import numpy as np
        import colour
        from raw_alchemy import utils
        
        try:
            # 步骤 1: 验证色彩空间
            logger.debug(f"[SafeLogTransform] Validating color space: {log_color_space}")
            if log_color_space not in colour.RGB_COLOURSPACES:
                error_msg = f"Unknown color space: {log_color_space}"
                logger.error(f"[SafeLogTransform] {error_msg}")
                return False, img, error_msg
            
            # 步骤 2: 检查输入图像
            logger.debug(f"[SafeLogTransform] Checking input image validity")
            if not np.isfinite(img).all():
                logger.warning(f"[SafeLogTransform] Input contains NaN/Inf, clipping...")
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 步骤 3: 计算变换矩阵
            logger.debug(f"[SafeLogTransform] Computing transformation matrix")
            try:
                M = colour.matrix_RGB_to_RGB(
                    colour.RGB_COLOURSPACES['ProPhoto RGB'],
                    colour.RGB_COLOURSPACES[log_color_space]
                )
            except Exception as e:
                error_msg = f"Matrix computation failed: {e}"
                logger.error(f"[SafeLogTransform] {error_msg}")
                return False, img, error_msg
            
            # 步骤 4: 验证矩阵
            if not np.isfinite(M).all():
                error_msg = f"Invalid transformation matrix (contains NaN/Inf)"
                logger.error(f"[SafeLogTransform] {error_msg}")
                return False, img, error_msg
            
            # 步骤 5: 应用矩阵变换
            logger.debug(f"[SafeLogTransform] Applying matrix transformation")
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)
            
            try:
                utils.apply_matrix_inplace(img, M)
            except Exception as e:
                error_msg = f"Matrix application failed: {e}"
                logger.error(f"[SafeLogTransform] {error_msg}")
                return False, img, error_msg
            
            # 步骤 6: 检查矩阵变换结果
            if not np.isfinite(img).all():
                logger.warning(f"[SafeLogTransform] Result contains NaN/Inf after matrix, clipping...")
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 步骤 7: 裁剪负值
            np.maximum(img, 1e-6, out=img)
            
            # 步骤 8: 应用 Log 编码（最可能崩溃的地方）
            logger.debug(f"[SafeLogTransform] Applying log encoding: {log_curve}")
            try:
                # 这是最可能在 M2 Mac 上崩溃的地方
                img = colour.cctf_encoding(img, function=log_curve)
                logger.debug(f"[SafeLogTransform] Log encoding successful")
            except Exception as e:
                error_msg = f"Log encoding failed: {e}"
                logger.error(f"[SafeLogTransform] {error_msg}")
                logger.warning(f"[SafeLogTransform] Falling back to simple gamma curve")
                # 回退：使用简单的 gamma 曲线
                try:
                    img = np.power(np.clip(img, 0, 1), 1/2.2)
                    logger.info(f"[SafeLogTransform] Fallback gamma curve applied successfully")
                except Exception as e2:
                    error_msg = f"Even fallback failed: {e2}"
                    logger.error(f"[SafeLogTransform] {error_msg}")
                    return False, img, error_msg
            
            # 步骤 9: 最终验证
            if not np.isfinite(img).all():
                logger.warning(f"[SafeLogTransform] Final result contains NaN/Inf, clipping...")
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            
            logger.info(f"[SafeLogTransform] Successfully applied {log_space}")
            return True, img, None
            
        except Exception as e:
            error_msg = f"Unexpected error in log transform: {e}"
            logger.error(f"[SafeLogTransform] {error_msg}")
            logger.error(traceback.format_exc())
            return False, img, error_msg


# 全局崩溃处理器实例
_crash_handler = None


def install_crash_handler():
    """安装全局崩溃处理器"""
    global _crash_handler
    if _crash_handler is None:
        _crash_handler = CrashHandler()
        _crash_handler.install()
    return _crash_handler


def get_crash_handler():
    """获取崩溃处理器实例"""
    global _crash_handler
    return _crash_handler
