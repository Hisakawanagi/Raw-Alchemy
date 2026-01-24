from pathlib import Path
import sys
import os
import platform

# 1. 定义一个用于存放 Matplotlib 缓存的目录
# Windows通常在 AppData, Mac通常在 ~/Library/Caches
config_dir = os.path.expanduser('~/.raw_alchemy/matplotlib_cache')
# 2. 确保目录存在
os.makedirs(config_dir, exist_ok=True)

# 3. 设置环境变量 (必须在 import matplotlib 之前！)
os.environ['MPLCONFIGDIR'] = config_dir

import time
import threading
import multiprocessing
import numpy as np
import rawpy
import colour
import gc
import urllib.request
import json
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QFrame,
    QSplitter, QSizePolicy, QGraphicsDropShadowEffect, QGridLayout,
    QInputDialog
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QObject, QTimer, QEvent
from PySide6.QtGui import QIcon, QPixmap, QImage, QPainter, QColor, QResizeEvent, QTransform, QPen

from qfluentwidgets import (
    FluentWindow, SubtitleLabel, PrimaryPushButton, PushButton,
    ComboBox, Slider, CaptionLabel, SwitchButton, StrongBodyLabel,
    BodyLabel, LineEdit, ToolButton, FluentIcon as FIF,
    CardWidget, SimpleCardWidget, ScrollArea, IndeterminateProgressRing,
    InfoBar, InfoBarPosition, Theme, setTheme, CheckBox, ProgressRing
)

from raw_alchemy import config, utils, orchestrator, metering, lensfun_wrapper, i18n
from raw_alchemy.i18n import tr
from raw_alchemy.orchestrator import SUPPORTED_RAW_EXTENSIONS
from loguru import logger

# ==============================================================================
#                               Version & License Info
# ==============================================================================

def get_version_info():
    """Read version and license from __init__.py"""
    version = '0.0.0'
    license_info = 'AGPL-3.0'
    
    try:
        from raw_alchemy import __version__
        version = __version__
    except (ImportError, AttributeError):
        logger.warning("Warning: Could not read version from __init__.py")
    
    return version, license_info

# ==============================================================================
#                               Data Structures
# ==============================================================================

class ImageState:
    """
    Unified state for a single image. 
    Replaces the mess of: original_pixmap_raw, original_pixmap_scaled, 
    last_processed_pixmap, _last_processed_pixmap_full, etc.
    
    Three states total:
    - original: RAW decoded image
    - current: processed with current params
    - baseline: saved baseline (optional)
    """
    def __init__(self):
        self.full: Optional[QPixmap] = None
        self.display: Optional[QPixmap] = None
        self.float_data: Optional[np.ndarray] = None  # For histogram
    
    def update_full(self, pixmap: QPixmap, float_data: Optional[np.ndarray] = None):
        """Update the full-size image and clear cached display version"""
        self.full = pixmap
        self.float_data = float_data
        self.display = None  # Invalidate cached display
    
    def get_display(self, size: QSize) -> Optional[QPixmap]:
        """Get display-sized version, caching the result"""
        if not self.full:
            return None
        
        if self.display is None:
            self.display = self.full.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        return self.display
    
    def clear(self):
        """Clear all cached data"""
        self.full = None
        self.display = None
        self.float_data = None


class ProcessRequest:
    """Immutable processing request. Eliminates race conditions."""
    def __init__(self, path: str, params: dict, request_id: int):
        self.path = path
        self.params = params.copy()  # Defensive copy
        self.request_id = request_id


# ==============================================================================
#                               Worker Threads
# ==============================================================================

class ThumbnailWorker(QThread):
    """Scan folder and generate thumbnails - 优化版本使用线程池"""
    thumbnail_ready = Signal(str, QImage)
    progress_update = Signal(int, int)  # current, total
    finished_scanning = Signal()

    def __init__(self, folder_path, max_workers=4):
        super().__init__()
        self.folder_path = folder_path
        self.stopped = False
        self.max_workers = max_workers

    @staticmethod
    def extract_thumbnail(full_path):
        """静态方法用于线程池并行处理"""
        try:
            with rawpy.imread(full_path) as raw:
                image = None
                from_raw_data = False  # 标记是否从 RAW 数据生成
                
                # 尝试提取嵌入的缩略图
                try:
                    thumb = raw.extract_thumb()
                    if thumb and thumb.format == rawpy.ThumbFormat.JPEG:
                        image = QImage()
                        image.loadFromData(thumb.data)
                        if image.isNull():
                            image = None
                except rawpy.LibRawNoThumbnailError:
                    # 没有嵌入缩略图，稍后从 RAW 数据生成
                    image = None
                
                # 如果没有嵌入缩略图，从 RAW 数据生成
                if image is None:
                    from_raw_data = True
                    # 使用快速解码生成缩略图
                    # postprocess() 默认会自动应用 EXIF 旋转，所以不需要手动旋转
                    rgb = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=True,
                        use_auto_wb=False,
                        output_bps=8,
                        half_size=True,  # 使用半尺寸加速
                        demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR  # 使用最快的算法
                    )
                    
                    # 转换为 QImage
                    h, w, c = rgb.shape
                    bytes_per_line = w * 3
                    image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                
                if not image.isNull():
                    # 只对嵌入缩略图应用旋转，从 RAW 数据生成的已经自动旋转了
                    if not from_raw_data:
                        orientation = raw.sizes.flip
                        if orientation == 3:
                            image = image.transformed(QTransform().rotate(180))
                        elif orientation == 5:
                            image = image.transformed(QTransform().rotate(-90))
                        elif orientation == 6:
                            image = image.transformed(QTransform().rotate(90))
                    
                    # 使用FastTransformation提速 - 缩略图不需要高质量
                    scaled = image.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.FastTransformation)
                    return full_path, scaled
        except Exception as e:
            logger.error(f"Error generating thumb for {os.path.basename(full_path)}: {e}")
            import traceback
            traceback.print_exc()
        
        return None, None

    def run(self):
        if not os.path.exists(self.folder_path):
            return

        files = sorted([f for f in os.listdir(self.folder_path)
                        if os.path.splitext(f)[1].lower() in SUPPORTED_RAW_EXTENSIONS])
        
        full_paths = [os.path.join(self.folder_path, f) for f in files]
        total = len(full_paths)
        
        # 使用滑动窗口策略：维持有限的并发任务，确保顺序发送
        from concurrent.futures import Future
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 使用字典维护 (index, future) 对，确保按顺序处理
            pending_futures = {}
            next_to_emit = 0  # 下一个应该发送的索引
            
            # 初始提交一批任务
            for i in range(min(self.max_workers * 2, total)):
                if self.stopped:
                    break
                future = executor.submit(self.extract_thumbnail, full_paths[i])
                pending_futures[i] = future
            
            submitted = min(self.max_workers * 2, total)
            
            # 主循环：等待并按顺序发送结果
            while next_to_emit < total:
                if self.stopped:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                # 等待下一个应该发送的任务完成
                if next_to_emit in pending_futures:
                    future = pending_futures[next_to_emit]
                    try:
                        full_path, scaled_image = future.result(timeout=30)
                        if full_path and scaled_image:
                            self.thumbnail_ready.emit(full_path, scaled_image)
                            # 强制主线程立即处理这个信号
                            time.sleep(0.02)  # 20ms 让UI有足够时间渲染
                        
                        self.progress_update.emit(next_to_emit + 1, total)
                    except Exception as e:
                        logger.error(f"Error processing thumbnail {full_paths[next_to_emit]}: {e}")
                        self.progress_update.emit(next_to_emit + 1, total)
                    
                    # 移除已完成的任务
                    del pending_futures[next_to_emit]
                    next_to_emit += 1
                    
                    # 提交新任务填充窗口
                    if submitted < total and not self.stopped:
                        future = executor.submit(self.extract_thumbnail, full_paths[submitted])
                        pending_futures[submitted] = future
                        submitted += 1
                else:
                    # 如果索引不在队列中（理论上不应该发生），跳过
                    next_to_emit += 1
        
        self.finished_scanning.emit()

    def stop(self):
        self.stopped = True


class VersionCheckWorker(QThread):
    """Check for new version from GitHub releases"""
    version_checked = Signal(bool, str, str)  # success, latest_version, error_msg
    
    def __init__(self, current_version):
        super().__init__()
        self.current_version = current_version
    
    def run(self):
        try:
            # GitHub API endpoint for latest release
            url = "https://api.github.com/repos/shenmintao/raw-alchemy/releases/latest"
            
            # Set timeout and user agent
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Raw-Alchemy-Studio')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                tag_name = data.get('tag_name', '')
                
                # Extract version from tag (e.g., "studio-v0.3.0" -> "0.3.0")
                # Remove common prefixes like "v", "studio-v", etc.
                latest_version = tag_name
                for prefix in ['studio-v', 'v']:
                    if latest_version.startswith(prefix):
                        latest_version = latest_version[len(prefix):]
                        break
                
                if latest_version:
                    self.version_checked.emit(True, latest_version, "")
                else:
                    self.version_checked.emit(False, "", "Invalid response from server")
                    
        except Exception as e:
            self.version_checked.emit(False, "", str(e))


class ImageProcessor(QThread):
    """
    Image processing worker. Uses request queue pattern to eliminate race conditions.
    No more params_dirty, no more mode/running_mode confusion.
    """
    result_ready = Signal(np.ndarray, np.ndarray, str, int, float)  # img_uint8, img_float, path, request_id, applied_ev
    load_complete = Signal(str, int)  # path, request_id - signals RAW loading is done
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        
        # Request management
        self.pending_request: Optional[ProcessRequest] = None
        self.current_request_id = 0
        
        # Caches (shared across loads)
        self.cached_linear = None
        self.cached_corrected = None
        self.cached_lens_key = None
        self.exif_data = None
        self.current_path = None

    def load_image(self, path: str):
        """Load RAW image - creates a special load request"""
        with self.lock:
            self.current_request_id += 1
            self.pending_request = ProcessRequest(path, {'_load': True}, self.current_request_id)
        
        if not self.isRunning():
            self.start()

    def update_preview(self, path: str, params: dict):
        """Process image with parameters"""
        with self.lock:
            self.current_request_id += 1
            self.pending_request = ProcessRequest(path, params, self.current_request_id)
        
        if not self.isRunning():
            self.start()

    def run(self):
        """Keep thread alive to process requests without restart overhead"""
        idle_count = 0
        max_idle = 10  # Exit after 10 empty checks (1 second)
        
        while True:
            # Atomically get pending request
            with self.lock:
                request = self.pending_request
                if request:
                    self.pending_request = None
                    idle_count = 0
                else:
                    idle_count += 1
            
            if not request:
                if idle_count >= max_idle:
                    break  # Exit after idle timeout
                time.sleep(0.1)  # Sleep 100ms, check again
                continue
            
            try:
                if '_load' in request.params:
                    self._do_load(request)
                else:
                    self._do_process(request)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.error_occurred.emit(str(e))

    def _do_load(self, request: ProcessRequest):
        """Load and decode RAW file"""
        path = request.path
        
        # Invalidate caches if switching images
        if path != self.current_path:
            self.cached_linear = None
            self.cached_corrected = None
            self.cached_lens_key = None
            self.exif_data = None
            self.current_path = path
        
        with rawpy.imread(path) as raw:
            self.exif_data, _ = utils.extract_lens_exif(path, raw)
            raw_post = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=True,
                use_auto_wb=False,
                output_bps=16,
                output_color=rawpy.ColorSpace.ProPhoto,
                bright=1.0,
                highlight_mode=2,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
                half_size=True
            )
            
            img = (raw_post / 65535.0).astype(np.float32)
            
            # Downsample if needed
            h, w = img.shape[:2]
            max_dim = 2048
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                step = int(1.0 / scale)
                if step > 1:
                    img = img[::step, ::step, :]
            
            self.cached_linear = img
            
            # Signal that loading is complete
            # Don't generate a processed preview here - let _do_process handle all processing
            # This eliminates duplicate lens correction and ensures consistent processing
            self.load_complete.emit(path, request.request_id)

    def _do_process(self, request: ProcessRequest):
        """Process image with parameters"""
        logger.debug(f"[Worker] _do_process called for: {os.path.basename(request.path)}, request_id={request.request_id}")
        
        # Ensure image is loaded and matches request path
        if self.cached_linear is None or self.current_path != request.path:
            logger.debug(f"[Worker] Need to load first (cached_linear={'None' if self.cached_linear is None else 'exists'}, current_path={os.path.basename(self.current_path) if self.current_path else 'None'})")
            # Need to load first
            self._do_load(ProcessRequest(request.path, {'_load': True}, request.request_id))
            if self.cached_linear is None:
                return
        
        params = request.params
        
        # Lens correction check
        current_lens_key = (params.get('lens_correct'), params.get('custom_db_path'))
        
        if current_lens_key != self.cached_lens_key or self.cached_corrected is None:
            temp = self.cached_linear.copy()
            if params.get('lens_correct') and self.exif_data:
                temp = utils.apply_lens_correction(
                    temp, self.exif_data, custom_db_path=params.get('custom_db_path')
                )
            self.cached_corrected = temp
            self.cached_lens_key = current_lens_key
        
        img = self.cached_corrected.copy()
        
        # Exposure
        if params.get('exposure_mode') == 'Manual':
            logger.debug(f"[Worker] Applying manual exposure: {params.get('exposure', 0.0)} EV")
            gain = 2.0 ** params.get('exposure', 0.0)
            utils.apply_gain_inplace(img, gain)
            applied_ev = params.get('exposure', 0.0)
        else:
            source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
            mode = params.get('metering_mode', 'matrix')
            
            logger.debug(f"[Worker] *** CALLING AUTO EXPOSURE *** mode={mode}, request_id={request.request_id}")
            # metering 模块会使用 loguru
            _, gain = metering.apply_auto_exposure(img, source_cs, mode)
            applied_ev = np.log2(gain)  # Convert gain to EV
            logger.debug(f"[Worker] Auto exposure complete: applied_ev={applied_ev:.2f}")
        
        # White Balance
        temp_val = params.get('wb_temp', 0.0)
        tint = params.get('wb_tint', 0.0)
        utils.apply_white_balance(img, temp_val, tint)
        
        # Highlight / Shadow
        hl = params.get('highlight', 0.0)
        sh = params.get('shadow', 0.0)
        utils.apply_highlight_shadow(img, hl, sh)
        
        # Saturation / Contrast
        sat = params.get('saturation', 1.0)
        con = params.get('contrast', 1.0)
        utils.apply_saturation_and_contrast(img, saturation=sat, contrast=con)
        
        # Log Transform
        log_space = params.get('log_space')
        if log_space and log_space != 'None':
            log_color_space = config.LOG_TO_WORKING_SPACE.get(log_space)
            log_curve = config.LOG_ENCODING_MAP.get(log_space, log_space)
            
            if log_color_space:
                M = colour.matrix_RGB_to_RGB(
                    colour.RGB_COLOURSPACES['ProPhoto RGB'],
                    colour.RGB_COLOURSPACES[log_color_space]
                )
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                utils.apply_matrix_inplace(img, M)
                np.maximum(img, 1e-6, out=img)
                img = colour.cctf_encoding(img, function=log_curve)
        
        # LUT
        lut_path = params.get('lut_path')
        if lut_path and os.path.exists(lut_path):
            try:
                lut = colour.read_LUT(lut_path)
                if isinstance(lut, colour.LUT3D):
                    if not img.flags['C_CONTIGUOUS']:
                        img = np.ascontiguousarray(img)
                    if img.dtype != np.float32:
                        img = img.astype(np.float32)
                    
                    # Ensure LUT table is float32 and C-contiguous
                    lut_table = lut.table
                    if lut_table.dtype != np.float32:
                        lut_table = lut_table.astype(np.float32)
                    if not lut_table.flags['C_CONTIGUOUS']:
                        lut_table = np.ascontiguousarray(lut_table)
                        
                    # Ensure domains are float64 and C-contiguous
                    domain_min = lut.domain[0].astype(np.float64)
                    domain_max = lut.domain[1].astype(np.float64)
                    if not domain_min.flags['C_CONTIGUOUS']:
                        domain_min = np.ascontiguousarray(domain_min)
                    if not domain_max.flags['C_CONTIGUOUS']:
                        domain_max = np.ascontiguousarray(domain_max)

                    utils.apply_lut_inplace(img, lut_table, domain_min, domain_max)
                else:
                    img = lut.apply(img)
            except Exception as e:
                logger.error(f"LUT application error: {e}")
        
        # Display transform - sRGB Standard
        if not log_space or log_space == 'None':
            utils.linear_to_srgb_inplace(img)
        
        img = np.clip(img, 0, 1)
        img_float = img.copy()
        img_uint8 = (img * 255).astype(np.uint8)
        
        self.result_ready.emit(img_uint8, img_float, request.path, request.request_id, applied_ev)


# ==============================================================================
#                               UI Components
# ==============================================================================

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.hist_data = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # 优化: 添加更新定时器防抖
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(50)  # 50ms防抖
        self.update_timer.timeout.connect(self._do_update)
        self.pending_data = None

    def update_data(self, img_array):
        """异步更新直方图数据 - 使用防抖避免频繁计算"""
        if img_array is None:
            return
        
        # 存储待处理数据，避免跨线程数据竞争
        try:
            self.pending_data = img_array.copy() if img_array is not None else None
        except Exception:
            return
        self.update_timer.start()
    
    def _do_update(self):
        """实际执行直方图计算"""
        if self.pending_data is None:
            return
        
        data = self.pending_data
        self.pending_data = None
            
        try:
            if data is None or data.size == 0:
                return
            
            # 使用utils中的快速计算函数
            self.hist_data = utils.compute_histogram_fast(data, bins=100, sample_rate=4)
            self.update()
        except (RuntimeError, ValueError, TypeError):
            pass
        except Exception as e:
            logger.error(f"Histogram update error: {type(e).__name__}: {e}")

    def paintEvent(self, event):
        if not self.hist_data:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # 填充深色背景（加色混合模式在深色背景下效果最佳）
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        
        # 检查是否有有效数据
        try:
            # 策略1：忽略两端极值 + 对数缩放
            # 这样可以避免过曝/欠曝区域的极高峰值压缩中间调细节
            
            # 对每个通道分别计算显示用的最大值（忽略首尾bin）
            display_max_vals = []
            for hist in self.hist_data:
                if len(hist) > 2:
                    # 忽略第0项（纯黑）和最后一项（纯白），在中间找最大值
                    inner_max = np.max(hist[1:-1])
                    display_max_vals.append(inner_max if inner_max > 0 else 1)
                else:
                    display_max_vals.append(np.max(hist) if len(hist) > 0 else 1)
            
            # 使用所有通道中的最大值作为统一缩放基准
            display_max = max(display_max_vals) if display_max_vals else 1
            if display_max == 0 or display_max < 1e-10:
                display_max = 1
            
            # 启用对数缩放来进一步压缩动态范围
            log_scale = True
            
            # 预计算对数缩放的分母（避免循环中重复计算）
            if log_scale:
                log_max_height = np.log1p(display_max)
            else:
                log_max_height = display_max
            
        except Exception as e:
            logger.error(f"Error computing display_max: {e}")
            return
        
        # RGB颜色定义（降低Alpha以获得更好的混合效果）
        colors = [
            QColor(255, 0, 0, 160),    # 红色
            QColor(0, 255, 0, 160),    # 绿色
            QColor(0, 0, 255, 160)     # 蓝色
        ]
        
        # 使用加色混合模式（Additive Blending）
        # 红+绿=黄，红+蓝=洋红，绿+蓝=青，红+绿+蓝=白
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        
        for i, hist in enumerate(self.hist_data):
            if len(hist) == 0:
                continue
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colors[i])
            
            bin_w = w / len(hist)
            
            # Draw polygon
            from PySide6.QtGui import QPolygonF
            from PySide6.QtCore import QPointF
            
            points = [QPointF(0, h)]  # 起始点：左下角
            
            for j, val in enumerate(hist):
                x = j * bin_w
                
                # 计算比例
                if log_scale:
                    current_h = np.log1p(float(val))
                    ratio = current_h / log_max_height
                else:
                    ratio = float(val) / display_max
                
                # [关键] 截断到 [0, 1] 范围
                # index 0 和 255 的值可能远大于 display_max，会被截断为1.0（顶格）
                # 这正是我们想要的效果：显示溢出，同时保留中间调细节
                ratio = min(1.0, max(0.0, ratio))
                
                # 计算 Y 坐标（Qt坐标系：0在顶部，h在底部）
                y = h - (ratio * h)
                
                points.append(QPointF(x, y))
            
            points.append(QPointF(w, h))  # 结束点：右下角
            painter.drawPolygon(QPolygonF(points))


class WaveformWidget(QWidget):
    """示波器组件 - 显示图像的亮度分布"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.waveform_data = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # 优化: 添加更新定时器防抖
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(50)  # 50ms防抖
        self.update_timer.timeout.connect(self._do_update)
        self.pending_data = None

    def update_data(self, img_array):
        """异步更新示波器数据 - 使用防抖避免频繁计算"""
        if img_array is None:
            return
        
        # 存储待处理数据，避免跨线程数据竞争
        try:
            self.pending_data = img_array.copy() if img_array is not None else None
        except Exception:
            return
        self.update_timer.start()
    
    def _do_update(self):
        """实际执行示波器计算"""
        if self.pending_data is None:
            return
        
        data = self.pending_data
        self.pending_data = None
            
        try:
            if data is None or data.size == 0:
                return
            
            # 使用utils中的快速计算函数
            self.waveform_data = utils.compute_waveform_fast(data, bins=100, sample_rate=4)
            self.update()
        except (RuntimeError, ValueError, TypeError):
            pass
        except Exception as e:
            logger.error(f"Waveform update error: {type(e).__name__}: {e}")

    def paintEvent(self, event):
        if self.waveform_data is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # 填充深色背景
        painter.fillRect(self.rect(), QColor(10, 10, 10))
        
        # 绘制专业网格线（达芬奇风格）
        try:
            line_color = QColor(0, 255, 0, 180)  # 绿色网格线
            painter.setPen(line_color)
            
            # IRE范围：-4% 到 109% (总共113%范围)
            # 计算关键IRE线的Y坐标
            def ire_to_y(ire_value):
                """将IRE值转换为屏幕Y坐标"""
                # -4% IRE 在底部 (y=h)
                # 109% IRE 在顶部 (y=0)
                normalized = (ire_value - (-4.0)) / 113.0  # 0到1
                return h - (normalized * h)
            
            # A: 109% IRE - 虚线
            y_109 = ire_to_y(109)
            painter.setPen(QPen(line_color, 0.5, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(y_109), w, int(y_109))
            
            # B: 100% IRE - 实线加粗
            y_100 = ire_to_y(100)
            painter.setPen(QPen(line_color, 1.0, Qt.PenStyle.SolidLine))
            painter.drawLine(0, int(y_100), w, int(y_100))
            
            # C: 50% IRE - 实线加粗
            y_50 = ire_to_y(50)
            painter.setPen(QPen(line_color, 1.0, Qt.PenStyle.SolidLine))
            painter.drawLine(0, int(y_50), w, int(y_50))
            
            # D: 0% IRE - 实线加粗
            y_0 = ire_to_y(0)
            painter.setPen(QPen(line_color, 1.0, Qt.PenStyle.SolidLine))
            painter.drawLine(0, int(y_0), w, int(y_0))
            
            # E: -4% IRE - 虚线
            y_minus4 = ire_to_y(-4)
            painter.setPen(QPen(line_color, 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(y_minus4), w, int(y_minus4))
            
            # 0-100%之间每10%画虚线
            painter.setPen(QPen(line_color, 0.5, Qt.PenStyle.DashLine))
            for ire in [10, 20, 30, 40, 60, 70, 80, 90]:
                y_ire = ire_to_y(ire)
                painter.drawLine(0, int(y_ire), w, int(y_ire))
            
        except Exception as e:
            logger.error(f"Error drawing grid: {e}")
        
        # 绘制波形数据
        try:
            waveform = self.waveform_data
            num_cols, num_bins = waveform.shape
            
            if num_cols == 0:
                return
            
            col_width = w / num_cols
            
            # 使用灰白色点状绘制（达芬奇风格）
            for col_idx in range(num_cols):
                x = col_idx * col_width
                
                # 获取该列的数据
                column_data = waveform[col_idx, :]
                
                # 绘制每个有值的点
                for bin_idx in range(num_bins):
                    if column_data[bin_idx] > 0:
                        # Y坐标：bin_idx越大（IRE越高），Y越小（屏幕上越高）
                        # bin 0 对应 -4% IRE (底部)
                        # bin (num_bins-1) 对应 109% IRE (顶部)
                        y = h - (bin_idx / float(num_bins - 1) * h)
                        
                        # 根据密度设置透明度和亮度（增强显示效果）
                        density = column_data[bin_idx]
                        # 提高基础透明度和密度系数
                        alpha = int(density * 255) + 100
                        alpha = min(255, alpha)
                        
                        # 使用灰白色（亮度波形用灰度显示更专业）
                        # 密度越高，颜色越亮 - 提高基础亮度和密度系数
                        brightness = int(density * 200) + 120
                        brightness = min(255, brightness)
                        color = QColor(brightness, brightness, brightness, alpha)
                        painter.setPen(color)
                        
                        # 绘制点
                        painter.drawPoint(int(x), int(y))
            
        except Exception as e:
            logger.error(f"Error painting waveform: {e}")


class GalleryItem(QWidget):
    """Custom widget for gallery item (Image + Text)"""
    def __init__(self, path, pixmap, parent=None):
        super().__init__(parent)
        self.path = path
        self.base_name = os.path.basename(path)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.img_label = QLabel()
        self.img_label.setPixmap(pixmap)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setFixedSize(140, 100)
        self.img_label.setScaledContents(True)
        
        # Text label with green dot indicator
        self.text_label = CaptionLabel(self.base_name)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.img_label)
        layout.addWidget(self.text_label)
    
    def set_marked(self, marked):
        """Show or hide the green dot indicator in the filename"""
        if marked:
            self.text_label.setText(f"🟢 {self.base_name}")
        else:
            self.text_label.setText(self.base_name)

class InspectorPanel(ScrollArea):
    """Right side control panel"""
    param_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QWidget()
        self.view.setObjectName("view")
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        self.view.setStyleSheet("#view { background-color: transparent; }")
        
        self.v_layout = QVBoxLayout(self.view)
        self.v_layout.setSpacing(20)
        self.v_layout.setContentsMargins(20, 20, 20, 20)
        
        # 保存的基准参数
        self.saved_baseline_params = None
        
        # 保存各模式的EV值
        self.manual_ev_value = 0.0  # 手动模式的EV
        self.auto_ev_value = 0.0    # 自动模式计算的EV（只读）
        
        # --- Histogram / Waveform with Switch ---
        self.hist_widget = HistogramWidget()
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.hide()  # Initially hidden
        
        # Create a container for the display mode switch
        display_mode_card = SimpleCardWidget()
        display_mode_layout = QVBoxLayout(display_mode_card)
        display_mode_layout.setSpacing(5)
        
        # Switch button for histogram/waveform
        self.display_mode_switch = SwitchButton()
        self.display_mode_switch.setChecked(False)  # False = Histogram, True = Waveform
        self.display_mode_switch.checkedChanged.connect(self._on_display_mode_changed)
        self._update_display_mode_switch_text()
        
        display_mode_layout.addWidget(self.display_mode_switch)
        
        self.add_section(tr('histogram_waveform'), display_mode_card)
        self.v_layout.addWidget(self.hist_widget)
        self.v_layout.addWidget(self.waveform_widget)

        # --- Exposure ---
        self.exp_card = SimpleCardWidget()
        exp_layout = QVBoxLayout(self.exp_card)
        
        self.auto_exp_radio = SwitchButton()
        self.auto_exp_radio.setChecked(True)  # Default to Auto Exposure
        self.auto_exp_radio.checkedChanged.connect(self._on_exposure_mode_changed)
        self._update_exposure_switch_text()
        
        self.metering_lbl = BodyLabel(tr('metering_mode'))
        self.metering_combo = ComboBox()
        # Store metering mode mapping: display text -> internal key
        self.metering_mode_map = {
            tr('matrix'): 'matrix',
            tr('average'): 'average',
            tr('center_weighted'): 'center-weighted',
            tr('highlight_safe'): 'highlight-safe',
            tr('hybrid'): 'hybrid'
        }
        # Reverse mapping: internal key -> display text
        self.metering_mode_reverse_map = {v: k for k, v in self.metering_mode_map.items()}
        
        self.metering_combo.addItems([tr('matrix'), tr('average'), tr('center_weighted'), tr('highlight_safe'), tr('hybrid')])
        self.metering_combo.setCurrentText(tr('matrix'))
        self.metering_combo.currentTextChanged.connect(self._on_param_change)
        
        self.exp_slider = Slider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(-100, 100) # -10.0 to 10.0
        self.exp_slider.setValue(0)
        self.exp_slider.update()
        
        # Add exposure value label
        self.exp_value_label = BodyLabel(tr('exposure_ev') + ": 0.0")
        
        def update_exp_label(val):
            """Update label and trigger debounced parameter change"""
            real_val = val / 10.0
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {real_val:+.1f}")
            # Trigger parameter change - will be debounced by 100ms timer in on_param_changed
            self._on_param_change()
        
        # 保存回调函数引用，以便后续临时断开连接
        self.exp_slider_callback = update_exp_label
        self.exp_slider.valueChanged.connect(update_exp_label)
        
        exp_layout.addWidget(self.auto_exp_radio)
        exp_layout.addWidget(self.metering_lbl)
        exp_layout.addWidget(self.metering_combo)
        exp_layout.addWidget(self.exp_value_label)
        exp_layout.addWidget(self.exp_slider)
        
        self._update_exposure_ui_state()
        
        self.add_section(tr('exposure'), self.exp_card)
        
        # --- Color / Log ---
        self.color_card = SimpleCardWidget()
        color_layout = QVBoxLayout(self.color_card)
        
        # Log Space
        color_layout.addWidget(BodyLabel(tr('log_space')))
        self.log_combo = ComboBox()
        # Store log space mapping: display text -> internal key
        self.log_space_map = {tr('none'): 'None'}
        # Map actual log space names to themselves
        for log_name in config.LOG_TO_WORKING_SPACE.keys():
            self.log_space_map[log_name] = log_name
        # Reverse mapping: internal key -> display text
        self.log_space_reverse_map = {v: k for k, v in self.log_space_map.items()}
        
        log_items = [tr('none')] + list(config.LOG_TO_WORKING_SPACE.keys())
        self.log_combo.addItems(log_items)
        self.log_combo.setCurrentText(tr('none'))
        self.log_combo.currentTextChanged.connect(self._on_param_change)
        color_layout.addWidget(self.log_combo)
        
        # LUT
        color_layout.addWidget(BodyLabel(tr('lut')))
        lut_layout = QHBoxLayout()
        self.lut_combo = ComboBox()
        self.lut_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.lut_combo.addItem(tr('none'))
        self.lut_combo.currentTextChanged.connect(self._on_param_change)
        
        self.lut_btn = ToolButton(FIF.FOLDER)
        self.lut_btn.clicked.connect(self._browse_lut_folder)
        
        lut_layout.addWidget(self.lut_combo, 1)
        lut_layout.addWidget(self.lut_btn)
        color_layout.addLayout(lut_layout)
        
        self.add_section(tr('color_management'), self.color_card)
        
        # --- Lens Correction ---
        self.lens_card = SimpleCardWidget()
        lens_layout = QVBoxLayout(self.lens_card)
        
        self.lens_correct_switch = SwitchButton(text=tr('enable_lens_correction'))
        self.lens_correct_switch.setChecked(True)  # Default enabled
        self.lens_correct_switch.checkedChanged.connect(self._on_param_change)
        lens_layout.addWidget(self.lens_correct_switch)
        
        # Custom Lensfun DB
        lens_layout.addWidget(BodyLabel(tr('custom_lensfun_db')))
        db_layout = QHBoxLayout()
        self.db_path_edit = LineEdit()
        self.db_path_edit.setPlaceholderText(tr('optional_db_path'))
        self.db_path_edit.setReadOnly(True)
        self.db_path_edit.textChanged.connect(self._on_param_change)
        
        self.db_browse_btn = ToolButton(FIF.FOLDER)
        self.db_browse_btn.clicked.connect(self._browse_lensfun_db)
        
        self.db_clear_btn = ToolButton(FIF.CLOSE)
        self.db_clear_btn.clicked.connect(self._clear_lensfun_db)
        
        db_layout.addWidget(self.db_path_edit, 1)
        db_layout.addWidget(self.db_browse_btn)
        db_layout.addWidget(self.db_clear_btn)
        lens_layout.addLayout(db_layout)
        
        self.add_section(tr('lens_correction'), self.lens_card)
        
        # --- Adjustments ---
        self.adj_card = SimpleCardWidget()
        adj_layout = QVBoxLayout(self.adj_card)
        
        self.sliders = {}
        self.slider_labels = {}  # 存储标签引用以便更新
        
        def add_slider(key, name, min_v, max_v, default_v, scale=1.0):
            layout = QVBoxLayout()
            lbl = BodyLabel(f"{name}: {default_v}")
            slider = Slider(Qt.Orientation.Horizontal)
            # slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.setRange(int(min_v*scale), int(max_v*scale))
            slider.setValue(int(default_v*scale))
            
            def update_lbl(val):
                """Update label and trigger debounced parameter change"""
                real_val = val / scale
                lbl.setText(f"{name}: {real_val:.2f}")
                # Trigger parameter change - will be debounced by 100ms timer in on_param_changed
                self._on_param_change()
            
            slider.valueChanged.connect(update_lbl)
                
            layout.addWidget(lbl)
            layout.addWidget(slider)
            adj_layout.addLayout(layout)
            self.sliders[key] = (slider, scale, default_v, name)  # 添加 name 到元组
            self.slider_labels[key] = lbl  # 存储标签引用

        add_slider('wb_temp', tr('temp'), -100, 100, 0, 1)
        add_slider('wb_tint', tr('tint'), -100, 100, 0, 1)
        add_slider('saturation', tr('saturation'), 0, 3, 1.25, 100)
        add_slider('contrast', tr('contrast'), 0, 3, 1.1, 100)
        add_slider('highlight', tr('highlights'), -100, 100, 0, 1)
        add_slider('shadow', tr('shadows'), -100, 100, 0, 1)
        
        # 按钮布局：保存参数和Reset ALL并排
        btn_layout = QHBoxLayout()
        self.save_baseline_btn = PushButton(tr('save_baseline'))
        self.save_baseline_btn.clicked.connect(self.save_baseline_params)
        self.reset_btn = PushButton(tr('reset_all'))
        self.reset_btn.clicked.connect(self.reset_adjustments)
        btn_layout.addWidget(self.save_baseline_btn)
        btn_layout.addWidget(self.reset_btn)
        adj_layout.addLayout(btn_layout)
        
        self.add_section(tr('adjustments'), self.adj_card)
        
        # Filler
        self.v_layout.addStretch()
        
        self.lut_folder = None

    def set_params(self, params):
        """Update UI controls from params dict"""
        if not params: return
        
        self.blockSignals(True) # Pause signals to avoid triggering processing loops
        
        # Exposure
        if 'exposure_mode' in params:
            self.auto_exp_radio.setChecked(params['exposure_mode'] == 'Auto')
        
        if 'metering_mode' in params:
            # Use reverse map to convert internal key to display text
            display_text = self.metering_mode_reverse_map.get(params['metering_mode'], tr('matrix'))
            self.metering_combo.setCurrentText(display_text)

        self._update_exposure_ui_state()

        if 'exposure' in params:
            exp_val = params['exposure']
            self.exp_slider.setValue(int(exp_val * 10))
            self.exp_slider.update()
            # Update the exposure value label
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {exp_val:+.1f}")
            
        # Color
        if 'log_space' in params:
            # Use reverse map to convert internal key to display text
            display_text = self.log_space_reverse_map.get(params['log_space'], tr('none'))
            self.log_combo.setCurrentText(display_text)
        
        # LUT (Path reconstruction logic needed if we only store path)
        # Assuming lut_path is full path
        if 'lut_path' in params and params['lut_path']:
            lut_name = os.path.basename(params['lut_path'])
            idx = self.lut_combo.findText(lut_name)
            if idx >= 0:
                self.lut_combo.setCurrentIndex(idx)
            else:
                 # Maybe LUT folder changed? For now set to None or handle gracefully
                 pass
        else:
            self.lut_combo.setCurrentIndex(0)
        
        # Lens Correction
        if 'lens_correct' in params:
            self.lens_correct_switch.setChecked(params['lens_correct'])
        
        if 'custom_db_path' in params and params['custom_db_path']:
            self.db_path_edit.setText(params['custom_db_path'])
        else:
            self.db_path_edit.clear()
            
        # Sliders
        for key, (slider, scale, _, name) in self.sliders.items():
            if key in params:
                slider.setValue(int(params[key] * scale))
                # 更新标签文本
                if key in self.slider_labels:
                    real_val = params[key]
                    self.slider_labels[key].setText(f"{name}: {real_val:.2f}")
                
        self.blockSignals(False)
        # Emit one signal to update view if needed?
        # Usually calling code handles the logic update, here we just update UI.

    def add_section(self, title, widget):
        self.v_layout.addWidget(StrongBodyLabel(title))
        self.v_layout.addWidget(widget)

    def _browse_lut_folder(self):
        # Get the main window to access last_lut_folder_path
        main_window = self.window()
        
        # Use last LUT folder path, or fall back to last gallery folder, or home
        start_dir = ""
        if hasattr(main_window, 'last_lut_folder_path') and main_window.last_lut_folder_path and os.path.exists(main_window.last_lut_folder_path):
            start_dir = main_window.last_lut_folder_path
        elif hasattr(main_window, 'last_folder_path') and main_window.last_folder_path and os.path.exists(main_window.last_folder_path):
            start_dir = main_window.last_folder_path
        
        folder = QFileDialog.getExistingDirectory(self, tr('select_lut_folder'), start_dir)
        if folder:
            self.lut_folder = folder
            # Remember this path in main window
            if hasattr(main_window, 'last_lut_folder_path'):
                main_window.last_lut_folder_path = folder
            self.refresh_lut_list()

    def refresh_lut_list(self):
        if not self.lut_folder: return
        self.lut_combo.clear()
        self.lut_combo.addItem(tr('none'))
        files = sorted([f for f in os.listdir(self.lut_folder) if f.lower().endswith('.cube')])
        self.lut_combo.addItems(files)
    
    def _browse_lensfun_db(self):
        # Get the main window to access last_lensfun_db_path
        main_window = self.window()
        
        # Use last Lensfun DB path's directory, or fall back to last gallery folder, or home
        start_dir = ""
        if hasattr(main_window, 'last_lensfun_db_path') and main_window.last_lensfun_db_path and os.path.exists(main_window.last_lensfun_db_path):
            start_dir = os.path.dirname(main_window.last_lensfun_db_path)
        elif hasattr(main_window, 'last_folder_path') and main_window.last_folder_path and os.path.exists(main_window.last_folder_path):
            start_dir = main_window.last_folder_path
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr('select_lensfun_db'),
            start_dir,
            "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.db_path_edit.setText(file_path)
            # Remember this path in main window
            if hasattr(main_window, 'last_lensfun_db_path'):
                main_window.last_lensfun_db_path = file_path
            # 重新加载lensfun数据库
            try:
                lensfun_wrapper.reload_lensfun_database(custom_db_path=file_path)
                InfoBar.success(tr('db_loaded'), tr('using_custom_db', name=os.path.basename(file_path)), parent=self)
            except Exception as e:
                InfoBar.error(tr('db_load_failed'), tr('failed_to_load_db', error=str(e)), parent=self)
                self.db_path_edit.clear()
    
    def _clear_lensfun_db(self):
        self.db_path_edit.clear()
        # 重新加载默认lensfun数据库
        try:
            lensfun_wrapper.reload_lensfun_database(custom_db_path=None)
            InfoBar.info(tr('db_cleared'), tr('using_default_db'), parent=self)
        except Exception as e:
            InfoBar.warning(tr('db_cleared'), f"Warning: {str(e)}", parent=self)

    def _update_display_mode_switch_text(self):
        """Update the display mode switch button text based on its state"""
        if self.display_mode_switch.isChecked():
            self.display_mode_switch.setText(tr('waveform'))
        else:
            self.display_mode_switch.setText(tr('histogram'))
    
    def _on_display_mode_changed(self):
        """Handle display mode switch between histogram and waveform"""
        is_waveform = self.display_mode_switch.isChecked()
        
        if is_waveform:
            # Switch to waveform
            self.hist_widget.hide()
            self.waveform_widget.show()
        else:
            # Switch to histogram
            self.waveform_widget.hide()
            self.hist_widget.show()
        
        self._update_display_mode_switch_text()

    def _update_exposure_switch_text(self):
        """Update the switch button text based on its state"""
        if self.auto_exp_radio.isChecked():
            self.auto_exp_radio.setText(tr('auto_exposure'))
        else:
            self.auto_exp_radio.setText(tr('manual_exposure'))
    
    def _update_exposure_ui_state(self):
        is_auto = self.auto_exp_radio.isChecked()
        self.metering_combo.setEnabled(is_auto)
        self.metering_lbl.setEnabled(is_auto)
        self.exp_slider.setEnabled(not is_auto)
        self._update_exposure_switch_text()

    def _on_exposure_mode_changed(self):
        is_auto = self.auto_exp_radio.isChecked()
        
        # Update UI state FIRST (enable/disable controls)
        self._update_exposure_ui_state()
        
        if is_auto:
            # Switching to auto mode: save current manual value
            self.manual_ev_value = self.exp_slider.value() / 10.0
            # Display auto mode EV
            # self.exp_slider.blockSignals(True)
            self.exp_slider.setValue(int(self.auto_ev_value * 10))
            self.exp_slider.update()  # Force immediate visual refresh
            # self.exp_slider.blockSignals(False)
            # Update label manually
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {self.auto_ev_value:+.1f}")
        else:
            # Switching to manual mode: save current auto value, restore manual value
            self.auto_ev_value = self.exp_slider.value() / 10.0
            # self.exp_slider.blockSignals(True)
            self.exp_slider.setValue(int(self.manual_ev_value * 10))
            self.exp_slider.update()  # Force immediate visual refresh
            # self.exp_slider.blockSignals(False)
            # Update label manually
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {self.manual_ev_value:+.1f}")
        
        self._on_param_change()

    def _on_param_change(self):
        self.param_changed.emit(self.get_params())

    def save_baseline_params(self):
        """保存当前参数作为基准点"""
        self.saved_baseline_params = self.get_params().copy()
        InfoBar.success(tr('baseline_saved'), tr('baseline_saved_message'), parent=self)

    def reset_adjustments(self):
        """重置到保存的基准点，如果没有保存则重置到默认值"""
        if self.saved_baseline_params:
            # 重置到保存的基准点
            self.set_params(self.saved_baseline_params)
            self._on_param_change()
        else:
            # 重置到默认值
            for key, (slider, scale, default, name) in self.sliders.items():
                slider.setValue(int(default * scale))
            self._on_param_change()

    def reset_params(self):
        self.auto_exp_radio.setChecked(True)  # Default to Auto Exposure
        self.metering_combo.setCurrentText(tr('matrix'))
        self._update_exposure_ui_state()
        self.exp_slider.setValue(0)
        self.exp_slider.update()
        self.exp_value_label.setText(tr('exposure_ev') + ": 0.0")
        self.log_combo.setCurrentText(tr('none'))
        self.lut_combo.setCurrentIndex(0)
        self.lens_correct_switch.setChecked(True)  # Default enabled
        self.db_path_edit.clear()
        
        # 重置所有滑块,不阻塞信号以确保UI正确更新
        for key, (slider, scale, default, name) in self.sliders.items():
            slider.setValue(int(default * scale))
            # 标签会通过valueChanged信号自动更新,无需手动设置
        
        self._on_param_change()

    def get_params(self):
        # Get internal metering mode key from display text
        metering_display_text = self.metering_combo.currentText()
        metering_internal_key = self.metering_mode_map.get(metering_display_text, 'matrix')
        
        # Get internal log space key from display text
        log_display_text = self.log_combo.currentText()
        log_internal_key = self.log_space_map.get(log_display_text, 'None')
        
        p = {
            'exposure_mode': 'Auto' if self.auto_exp_radio.isChecked() else 'Manual',
            'metering_mode': metering_internal_key,
            'exposure': self.exp_slider.value() / 10.0, # Scale factor for EV
            'log_space': log_internal_key,
            'lut_path': os.path.join(self.lut_folder, self.lut_combo.currentText()) if self.lut_folder and self.lut_combo.currentIndex() > 0 else None,
            'lens_correct': self.lens_correct_switch.isChecked(),
            'custom_db_path': self.db_path_edit.text() if self.db_path_edit.text() else None
        }
        
        for key, (slider, scale, _, _) in self.sliders.items():
            p[key] = slider.value() / scale
            
        return p


class MainWindow(FluentWindow):
    def __init__(self):
        # Initialize image states BEFORE super().__init__() to avoid resizeEvent issues
        # FluentWindow.__init__ may trigger resizeEvent during initialization
        self.original = ImageState()  # RAW decoded
        self.current = ImageState()   # Processed with current params
        self.baseline = ImageState()  # Saved baseline (optional)
        
        super().__init__()
        self.base_title = "Raw Alchemy Studio"
        self.setWindowTitle(self.base_title)
        self.setWindowIcon(QIcon(self._get_icon_path()))
        self.resize(1900, 1200)
        
        # State
        self.current_folder = None
        self.current_raw_path = None
        self.marked_files = set()
        self.file_params_cache = {}  # path -> params dict
        self.thumbnail_cache = {}  # path -> original QPixmap
        self.file_baseline_params_cache = {}  # path -> baseline params dict
        
        # Last used paths for file dialogs
        self.last_folder_path = None  # Last opened gallery folder
        self.last_lut_folder_path = None  # Last LUT folder
        self.last_lensfun_db_path = None  # Last Lensfun DB path
        self.last_export_path = None  # Last export folder
        
        # Request tracking
        self.current_request_id = 0
        
        # 预加载lensfun数据库（在后台线程中）
        self._preload_lensfun_database()
        
        self.create_ui()
        self.create_settings_interface()
        self.create_help_interface()
        self.create_about_interface()
        
        # Workers
        self.thumb_worker = None
        self.processor = ImageProcessor()
        self.processor.result_ready.connect(self.on_process_result)
        self.processor.load_complete.connect(self.on_load_complete)
        self.processor.error_occurred.connect(self.on_error)
        
        # Baseline processor
        self.baseline_processor = ImageProcessor()
        self.baseline_processor.result_ready.connect(self.on_baseline_result)
        
        # Processing Debounce
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(100) # 100ms debounce
        self.update_timer.timeout.connect(self.trigger_processing)
        
        # Load saved settings
        self.load_settings()
        
        # Restore UI state from saved settings
        self.restore_ui()
    
    def update_window_title(self):
        """更新窗口标题以显示当前文件名"""
        if self.current_raw_path:
            filename = os.path.basename(self.current_raw_path)
            self.setWindowTitle(f"{self.base_title} - {filename}")
        else:
            self.setWindowTitle(self.base_title)
    
    def _get_icon_path(self):
        """Get the path to the application icon (supports PyInstaller and Nuitka)."""
        # Check if running as frozen executable (PyInstaller or Nuitka)
        if getattr(sys, 'frozen', False):
            # For onefile mode, resources are in _MEIPASS
            if hasattr(sys, '_MEIPASS'):
                base_path = sys._MEIPASS
            else:
                # For standalone mode, resources are next to executable
                base_path = os.path.dirname(sys.executable)
        else:
            # Running as script - use current directory
            base_path = os.path.abspath(".")
        
        # Windows 优先使用 .ico 格式，其他平台使用 .png
        if sys.platform == 'win32':
            icon_path = os.path.join(base_path, "icon.ico")
            if not os.path.exists(icon_path):
                # Fallback to PNG if ICO not found
                icon_path = os.path.join(base_path, "icon.png")
        else:
            icon_path = os.path.join(base_path, "icon.png")
            if not os.path.exists(icon_path):
                # Fallback to ICO if PNG not found
                icon_path = os.path.join(base_path, "icon.ico")
            
        return icon_path

    def _preload_lensfun_database(self):
        """在后台线程中预加载lensfun数据库，避免阻塞GUI启动"""
        def preload():
            try:
                # 预加载默认数据库
                lensfun_wrapper._get_or_create_database(custom_db_path=None)
            except Exception as e:
                logger.error(f"  ⚠️ [Lensfun] Failed to preload database: {e}")
        
        # 在后台线程中执行，不阻塞GUI
        import threading
        preload_thread = threading.Thread(target=preload, daemon=True)
        preload_thread.start()
    
    def load_settings(self):
        """Load saved application settings"""
        settings = i18n.load_app_settings()
        
        # Restore window geometry and state
        if 'window_geometry' in settings:
            geom = settings['window_geometry']
            if all(k in geom for k in ['x', 'y', 'width', 'height']):
                self.setGeometry(geom['x'], geom['y'], geom['width'], geom['height'])
        
        # Restore maximized state
        if settings.get('window_maximized', False):
            self.showMaximized()
        
        # Restore last folder path (but don't auto-open)
        if 'last_folder_path' in settings:
            self.last_folder_path = settings['last_folder_path']
        
        # Restore LUT folder
        if 'last_lut_folder_path' in settings:
            self.last_lut_folder_path = settings['last_lut_folder_path']
        
        # Restore Lensfun DB path
        if 'last_lensfun_db_path' in settings:
            self.last_lensfun_db_path = settings['last_lensfun_db_path']
        
        # Restore export path
        if 'last_export_path' in settings:
            self.last_export_path = settings['last_export_path']
    
    def restore_ui(self):
        """Restore UI state from saved settings"""
        # Restore LUT folder and refresh list if saved
        if self.last_lut_folder_path and os.path.exists(self.last_lut_folder_path):
            self.right_panel.lut_folder = self.last_lut_folder_path
            self.right_panel.refresh_lut_list()
    
    def save_settings(self):
        """Save current application settings"""
        # Save window geometry and maximized state
        is_maximized = self.isMaximized()
        
        # Get geometry (if not maximized, save current geometry; if maximized, save normal geometry)
        if is_maximized:
            # Get the normal geometry (before maximization)
            geom = self.normalGeometry()
        else:
            geom = self.geometry()
        
        settings = {
            'window_geometry': {
                'x': geom.x(),
                'y': geom.y(),
                'width': geom.width(),
                'height': geom.height()
            },
            'window_maximized': is_maximized,
            'last_folder_path': self.current_folder,
            'last_lut_folder_path': self.last_lut_folder_path,
            'last_lensfun_db_path': self.last_lensfun_db_path,
            'last_export_path': self.last_export_path
        }
        i18n.save_app_settings(settings)

    def create_ui(self):
        # Central Layout
        self.main_widget = QWidget()
        self.main_widget.setObjectName("mainWidget")
        self.h_layout = QHBoxLayout(self.main_widget)
        self.h_layout.setContentsMargins(0, 0, 0, 0)
        self.h_layout.setSpacing(0)
        
        # 1. Left Panel (Gallery)
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(400)
        self.left_panel.setStyleSheet("background-color: transparent;")
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(5, 10, 5, 10)
        
        self.gallery_list = QListWidget()
        self.gallery_list.setIconSize(QSize(130, 100))
        self.gallery_list.setGridSize(QSize(160, 140))
        self.gallery_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery_list.setSpacing(10)

        # 禁用左侧缩略图拖拽（避免误拖动导致内部移动/拖放）
        self.gallery_list.setDragEnabled(False)
        self.gallery_list.setAcceptDrops(False)
        self.gallery_list.setDropIndicatorShown(False)
        self.gallery_list.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self.gallery_list.setDefaultDropAction(Qt.DropAction.IgnoreAction)

        self.gallery_list.itemClicked.connect(self.on_gallery_item_clicked)
        self.gallery_list.currentItemChanged.connect(lambda current, prev: self.on_gallery_item_clicked(current))
        self.gallery_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                color: white;
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
            }
            QListWidget::item:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
        """)
        
        self.open_btn = PrimaryPushButton(FIF.FOLDER, tr('open_folder'))
        self.open_btn.clicked.connect(self.browse_folder)
        
        # 添加加载提示标签
        self.loading_label = CaptionLabel("")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.hide()
        
        self.left_layout.addWidget(SubtitleLabel(tr('library')))
        self.left_layout.addWidget(self.gallery_list)
        self.left_layout.addWidget(self.loading_label)
        self.left_layout.addWidget(self.open_btn)
        
        # 2. Center Panel (Preview)
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(10, 10, 10, 10)
        
        # Preview Area
        self.preview_lbl = QLabel(tr('no_image_selected'))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setStyleSheet("background-color: #202020; border-radius: 8px; color: white;")
        self.preview_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Handle compare (Mouse Press/Release)
        self.preview_lbl.mousePressEvent = self.show_original
        self.preview_lbl.mouseReleaseEvent = self.show_processed
        
        # Toolbar
        self.toolbar = QFrame()
        self.toolbar.setFixedHeight(60)
        self.toolbar_layout = QHBoxLayout(self.toolbar)
        
        self.btn_prev = ToolButton(FIF.LEFT_ARROW)
        self.btn_next = ToolButton(FIF.RIGHT_ARROW)
        self.btn_mark = ToolButton(FIF.TAG)
        self.btn_mark.setCheckable(True)  # Make it a toggle button
        self.btn_delete = ToolButton(FIF.DELETE)
        self.btn_compare = PushButton(tr('hold_to_compare')) # Visual cue
        self.btn_compare.setToolTip(tr('hold_to_compare'))
        
        self.btn_export_curr = PushButton(tr('export_current'))
        self.btn_export_all = PrimaryPushButton(tr('export_all_marked'))
        
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_mark.clicked.connect(self.toggle_mark)
        self.btn_delete.clicked.connect(self.delete_image)
        self.btn_export_curr.clicked.connect(self.export_current)
        self.btn_export_all.clicked.connect(self.export_all)
        
        # Progress Ring for Batch Export
        self.export_progress = ProgressRing()
        self.export_progress.setFixedSize(40, 40)
        self.export_progress.setTextVisible(True)
        self.export_progress.hide()

        # Make compare button toggle between original and processed
        self.btn_compare.pressed.connect(lambda: self.show_original(None))
        self.btn_compare.released.connect(lambda: self.show_processed(None))
        
        self.toolbar_layout.addWidget(self.btn_prev)
        self.toolbar_layout.addWidget(self.btn_next)
        self.toolbar_layout.addStretch()
        self.toolbar_layout.addWidget(self.btn_mark)
        self.toolbar_layout.addWidget(self.btn_delete)
        self.toolbar_layout.addStretch()
        self.toolbar_layout.addWidget(self.btn_compare)
        self.toolbar_layout.addWidget(self.export_progress) # Add progress ring
        self.toolbar_layout.addWidget(self.btn_export_curr)
        self.toolbar_layout.addWidget(self.btn_export_all)
        
        self.center_layout.addWidget(self.preview_lbl)
        self.center_layout.addWidget(self.toolbar)

        # 3. Right Panel (Inspector)
        self.right_panel = InspectorPanel()
        self.right_panel.setFixedWidth(400)
        self.right_panel.param_changed.connect(self.on_param_changed)
        
        # 连接保存基准点按钮到保存基准点图像的方法
        self.right_panel.save_baseline_btn.clicked.connect(self.save_baseline_image)

        # Assemble
        self.h_layout.addWidget(self.left_panel)
        self.h_layout.addWidget(self.center_panel, 1) # Expand
        self.h_layout.addWidget(self.right_panel)
        
        self.addSubInterface(self.main_widget, FIF.PHOTO, tr('editor'))
        
        # Apply Dark Theme
        setTheme(Theme.DARK)

        # Install event filter to capture keys globally
        QApplication.instance().installEventFilter(self)
    
    def create_settings_interface(self):
        """Create settings interface with language selection"""
        self.settings_widget = QWidget()
        self.settings_widget.setObjectName("settingsWidget")
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_layout.setContentsMargins(40, 40, 40, 40)
        settings_layout.setSpacing(20)
        
        # Title
        title = SubtitleLabel(tr('settings'))
        settings_layout.addWidget(title)
        
        # Language Card
        lang_card = SimpleCardWidget()
        lang_layout = QVBoxLayout(lang_card)
        lang_layout.setSpacing(10)
        
        lang_label = StrongBodyLabel(tr('language'))
        lang_layout.addWidget(lang_label)
        
        # Language ComboBox
        self.lang_combo = ComboBox()
        self.lang_combo.addItems([tr('english'), tr('chinese')])
        
        # Set current language
        current_lang = i18n.get_current_language()
        if current_lang == 'zh':
            self.lang_combo.setCurrentIndex(1)
        else:
            self.lang_combo.setCurrentIndex(0)
        
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        lang_layout.addWidget(self.lang_combo)
        
        settings_layout.addWidget(lang_card)
        settings_layout.addStretch()
        
        # Add settings interface to navigation
        self.addSubInterface(self.settings_widget, FIF.SETTING, tr('settings'))
    
    def create_help_interface(self):
        """Create help interface with user guide"""
        self.help_widget = QWidget()
        self.help_widget.setObjectName("helpWidget")
        
        # Create scroll area for help content
        help_scroll = ScrollArea(self.help_widget)
        help_scroll.setWidgetResizable(True)
        help_scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        
        help_content = QWidget()
        help_content.setObjectName("helpContent")
        help_content.setStyleSheet("#helpContent { background-color: transparent; }")
        help_layout = QVBoxLayout(help_content)
        help_layout.setContentsMargins(40, 40, 40, 40)
        help_layout.setSpacing(20)
        
        # Title
        title = SubtitleLabel(tr('help_title'))
        help_layout.addWidget(title)
        
        # Overview Section
        overview_card = SimpleCardWidget()
        overview_layout = QVBoxLayout(overview_card)
        overview_layout.setSpacing(10)
        overview_title = StrongBodyLabel(tr('help_overview'))
        overview_text = BodyLabel(tr('help_overview_text'))
        overview_text.setWordWrap(True)
        overview_layout.addWidget(overview_title)
        overview_layout.addWidget(overview_text)
        help_layout.addWidget(overview_card)
        
        # Workflow Section
        workflow_card = SimpleCardWidget()
        workflow_layout = QVBoxLayout(workflow_card)
        workflow_layout.setSpacing(10)
        workflow_title = StrongBodyLabel(tr('help_workflow'))
        workflow_text = BodyLabel(tr('help_workflow_text'))
        workflow_text.setWordWrap(True)
        workflow_layout.addWidget(workflow_title)
        workflow_layout.addWidget(workflow_text)
        help_layout.addWidget(workflow_card)
        
        # Shortcuts Section
        shortcuts_card = SimpleCardWidget()
        shortcuts_layout = QVBoxLayout(shortcuts_card)
        shortcuts_layout.setSpacing(10)
        shortcuts_title = StrongBodyLabel(tr('help_shortcuts'))
        shortcuts_text = BodyLabel(tr('help_shortcuts_text'))
        shortcuts_text.setWordWrap(True)
        shortcuts_layout.addWidget(shortcuts_title)
        shortcuts_layout.addWidget(shortcuts_text)
        help_layout.addWidget(shortcuts_card)
        
        # Features Guide Section
        features_title = StrongBodyLabel(tr('help_features'))
        help_layout.addWidget(features_title)
        
        # Exposure Control
        exposure_card = SimpleCardWidget()
        exposure_layout = QVBoxLayout(exposure_card)
        exposure_layout.setSpacing(10)
        exposure_title = StrongBodyLabel(tr('help_exposure'))
        exposure_text = BodyLabel(tr('help_exposure_text'))
        exposure_text.setWordWrap(True)
        exposure_layout.addWidget(exposure_title)
        exposure_layout.addWidget(exposure_text)
        help_layout.addWidget(exposure_card)
        
        # Color Management
        color_card = SimpleCardWidget()
        color_layout = QVBoxLayout(color_card)
        color_layout.setSpacing(10)
        color_title = StrongBodyLabel(tr('help_color'))
        color_text = BodyLabel(tr('help_color_text'))
        color_text.setWordWrap(True)
        color_layout.addWidget(color_title)
        color_layout.addWidget(color_text)
        help_layout.addWidget(color_card)
        
        # Lens Correction
        lens_card = SimpleCardWidget()
        lens_layout = QVBoxLayout(lens_card)
        lens_layout.setSpacing(10)
        lens_title = StrongBodyLabel(tr('help_lens'))
        lens_text = BodyLabel(tr('help_lens_text'))
        lens_text.setWordWrap(True)
        lens_layout.addWidget(lens_title)
        lens_layout.addWidget(lens_text)
        help_layout.addWidget(lens_card)
        
        # Adjustments
        adjustments_card = SimpleCardWidget()
        adjustments_layout = QVBoxLayout(adjustments_card)
        adjustments_layout.setSpacing(10)
        adjustments_title = StrongBodyLabel(tr('help_adjustments'))
        adjustments_text = BodyLabel(tr('help_adjustments_text'))
        adjustments_text.setWordWrap(True)
        adjustments_layout.addWidget(adjustments_title)
        adjustments_layout.addWidget(adjustments_text)
        help_layout.addWidget(adjustments_card)
        
        # Export Options
        export_card = SimpleCardWidget()
        export_layout = QVBoxLayout(export_card)
        export_layout.setSpacing(10)
        export_title = StrongBodyLabel(tr('help_export'))
        export_text = BodyLabel(tr('help_export_text'))
        export_text.setWordWrap(True)
        export_layout.addWidget(export_title)
        export_layout.addWidget(export_text)
        help_layout.addWidget(export_card)
        
        help_layout.addStretch()
        
        help_scroll.setWidget(help_content)
        
        # Set layout for help widget
        help_widget_layout = QVBoxLayout(self.help_widget)
        help_widget_layout.setContentsMargins(0, 0, 0, 0)
        help_widget_layout.addWidget(help_scroll)
        
        # Add help interface to navigation
        self.addSubInterface(self.help_widget, FIF.QUESTION, tr('help'))
    
    def create_about_interface(self):
        """Create about interface"""
        self.about_widget = QWidget()
        self.about_widget.setObjectName("aboutWidget")
        about_layout = QVBoxLayout(self.about_widget)
        about_layout.setContentsMargins(40, 40, 40, 40)
        about_layout.setSpacing(20)
        
        # Get version and license info
        self.current_version, license_info = get_version_info()
        
        # Title
        title = SubtitleLabel(tr('about_title'))
        about_layout.addWidget(title)
        
        # Description Card
        desc_card = SimpleCardWidget()
        desc_layout = QVBoxLayout(desc_card)
        desc_layout.setSpacing(10)
        desc_text = BodyLabel(tr('about_description'))
        desc_text.setWordWrap(True)
        desc_layout.addWidget(desc_text)
        about_layout.addWidget(desc_card)
        
        # Version Card with Check Update Button
        version_card = SimpleCardWidget()
        version_layout = QVBoxLayout(version_card)
        version_layout.setSpacing(10)
        version_title = StrongBodyLabel(tr('about_version'))
        version_text = BodyLabel(self.current_version)
        
        # Check Update Button
        self.check_update_btn = PushButton(tr('check_update'))
        self.check_update_btn.clicked.connect(self.check_for_updates)
        
        # Export Logs Button
        self.export_logs_btn = PushButton(tr('export_logs'))
        self.export_logs_btn.clicked.connect(self.export_logs)
        
        version_layout.addWidget(version_title)
        version_layout.addWidget(version_text)
        version_layout.addWidget(self.check_update_btn)
        version_layout.addWidget(self.export_logs_btn)
        about_layout.addWidget(version_card)
        
        # License Card
        license_card = SimpleCardWidget()
        license_layout = QVBoxLayout(license_card)
        license_layout.setSpacing(10)
        license_title = StrongBodyLabel(tr('about_license'))
        license_text = BodyLabel(license_info)
        license_layout.addWidget(license_title)
        license_layout.addWidget(license_text)
        about_layout.addWidget(license_card)
        
        # Features Card
        features_card = SimpleCardWidget()
        features_layout = QVBoxLayout(features_card)
        features_layout.setSpacing(10)
        features_title = StrongBodyLabel(tr('about_features'))
        features_text = BodyLabel(tr('about_features_list'))
        features_text.setWordWrap(True)
        features_layout.addWidget(features_title)
        features_layout.addWidget(features_text)
        about_layout.addWidget(features_card)
        
        # GitHub Card
        github_card = SimpleCardWidget()
        github_layout = QVBoxLayout(github_card)
        github_layout.setSpacing(10)
        github_title = StrongBodyLabel(tr('about_github'))
        github_link = BodyLabel('<a href="https://github.com/shenmintao/Raw-alchemy">https://github.com/shenmintao/Raw-alchemy</a>')
        github_link.setOpenExternalLinks(True)
        github_layout.addWidget(github_title)
        github_layout.addWidget(github_link)
        about_layout.addWidget(github_card)
        
        about_layout.addStretch()
        
        # Add about interface to navigation
        self.addSubInterface(self.about_widget, FIF.INFO, tr('about'))
    
    def check_for_updates(self):
        """Check for new version from GitHub"""
        # Disable button and show checking message
        self.check_update_btn.setEnabled(False)
        self.check_update_btn.setText(tr('checking_update'))
        
        # Create and start version check worker
        self.version_worker = VersionCheckWorker(self.current_version)
        self.version_worker.version_checked.connect(self.on_version_checked)
        self.version_worker.start()
    
    def on_version_checked(self, success, latest_version, error_msg):
        """Handle version check result"""
        # Re-enable button
        self.check_update_btn.setEnabled(True)
        self.check_update_btn.setText(tr('check_update'))
        
        if not success:
            # Show error message
            InfoBar.error(
                tr('update_check_failed'),
                tr('update_check_error', error=error_msg),
                parent=self
            )
            return
        
        # Compare versions
        from packaging import version as pkg_version
        try:
            current = pkg_version.parse(self.current_version)
            latest = pkg_version.parse(latest_version)
            
            if latest > current:
                # New version available - show dialog
                from PySide6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self,
                    tr('update_available'),
                    tr('update_available_message',
                       version=latest_version,
                       current=self.current_version,
                       latest=latest_version),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Open releases page in browser
                    import webbrowser
                    webbrowser.open('https://github.com/shenmintao/raw-alchemy/releases/latest')
            else:
                # Already up to date
                InfoBar.success(
                    tr('no_update'),
                    tr('no_update_message', version=self.current_version),
                    parent=self
                )
        except Exception as e:
            # Fallback: simple string comparison
            if latest_version != self.current_version:
                from PySide6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self,
                    tr('update_available'),
                    tr('update_available_message',
                       version=latest_version,
                       current=self.current_version,
                       latest=latest_version),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open('https://github.com/shenmintao/raw-alchemy/releases/latest')
            else:
                InfoBar.success(
                    tr('no_update'),
                    tr('no_update_message', version=self.current_version),
                    parent=self
                 )
    
    def export_logs(self):
        """Export application logs to a user-selected location"""
        from raw_alchemy.logger import get_log_file_path
        import shutil
        
        # Get the log file path
        log_file = get_log_file_path()
        
        # Check if log file exists
        if not os.path.exists(log_file):
            InfoBar.warning(
                tr('no_logs_found'),
                tr('no_logs_found'),
                parent=self
            )
            return
        
        # Ask user where to save the log file
        default_name = f"raw_alchemy_logs_{time.strftime('%Y%m%d_%H%M%S')}.log"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            tr('export_logs'),
            default_name,
            "Log Files (*.log);;All Files (*)"
        )
        
        if save_path:
            try:
                # Copy log file to selected location
                shutil.copy2(log_file, save_path)
                InfoBar.success(
                    tr('export_logs_success'),
                    tr('logs_saved_to', path=save_path),
                    parent=self
                )
            except Exception as e:
                InfoBar.error(
                    tr('export_logs_failed'),
                    str(e),
                    parent=self
                )
    
    def on_language_changed(self, index):
        """Handle language change"""
        # Map index to language code
        lang_code = 'en' if index == 0 else 'zh'
        
        # Set language
        i18n.set_language(lang_code)
        
        # Show restart message
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            tr('restart_required'),
            tr('restart_message')
        )

    def eventFilter(self, obj, event):
        if isinstance(obj, QWidget) and obj.window() == self:
            if event.type() == QEvent.Type.KeyPress:
                key = event.key()
                if key == Qt.Key.Key_Left:
                    self.prev_image()
                    return True
                elif key == Qt.Key.Key_Right:
                    self.next_image()
                    return True
                elif key == Qt.Key.Key_Space:
                    if not event.isAutoRepeat():
                        self.show_original(None)
                    return True
                elif key == Qt.Key.Key_Delete:
                    self.delete_image()
                    return True
                elif key == Qt.Key.Key_T:
                    self.toggle_mark()
                    return True
            elif event.type() == QEvent.Type.KeyRelease:
                if event.key() == Qt.Key.Key_Space:
                    if not event.isAutoRepeat():
                        self.show_processed(None)
                    return True
        
        return super().eventFilter(obj, event)

    # --- Actions ---

    def browse_folder(self):
        # Use last folder path as starting directory, or home directory as fallback
        start_dir = self.last_folder_path if self.last_folder_path and os.path.exists(self.last_folder_path) else ""
        
        folder = QFileDialog.getExistingDirectory(self, tr('select_folder'), start_dir)
        if folder:
            self.current_folder = folder
            self.last_folder_path = folder  # Remember this path
            self.gallery_list.clear()
            self.start_thumbnail_scan(folder)

    def start_thumbnail_scan(self, folder):
        if self.thumb_worker:
            self.thumb_worker.stop()
            self.thumb_worker.wait()
        
        # 显示加载提示
        self.loading_label.setText(tr('loading_thumbnails'))
        self.loading_label.show()
        
        self.thumb_worker = ThumbnailWorker(folder)
        self.thumb_worker.thumbnail_ready.connect(self.add_gallery_item)
        self.thumb_worker.progress_update.connect(self.on_thumbnail_progress)
        self.thumb_worker.finished_scanning.connect(self.on_thumbnail_finished)
        self.thumb_worker.start()
    
    def on_thumbnail_progress(self, current, total):
        """更新缩略图加载进度"""
        self.loading_label.setText(f"{tr('loading_thumbnails')}: {current}/{total}")
    
    def on_thumbnail_finished(self):
        """缩略图加载完成"""
        self.loading_label.hide()

    def add_gallery_item(self, path, image):
        name = os.path.basename(path)
        
        # Convert QImage to QPixmap in main thread
        pixmap = QPixmap.fromImage(image)
        
        # Store original pixmap in cache
        self.thumbnail_cache[path] = pixmap

        # Custom Item with icon and text
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setIcon(QIcon(pixmap))
        
        # Set the text with or without green dot based on marked status
        is_marked = path in self.marked_files
        if is_marked:
            item.setText(f"🟢 {name}")
        else:
            item.setText(name)
        
        self.gallery_list.addItem(item)
        
        # 每次添加后立即刷新UI，确保缩略图逐个显示
        QApplication.processEvents()

    def on_gallery_item_clicked(self, item):
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path == self.current_raw_path:
            return

        logger.debug(f"[Gallery] Clicked item: {os.path.basename(path)}")

        # Save current params before switching
        if self.current_raw_path:
            self.file_params_cache[self.current_raw_path] = self.right_panel.get_params()

        # Switch path
        self.current_raw_path = path
        logger.debug(f"[Gallery] Switched to: {os.path.basename(path)}")
        
        # Update window title with current filename
        self.update_window_title()
        
        # Clear images for new selection
        self.original.clear()
        self.current.clear()
        self.baseline.clear()
        
        # Reset auto EV display if in auto mode (will be updated after processing)
        if self.right_panel.auto_exp_radio.isChecked():
            logger.debug("[Gallery] Resetting auto EV display")
            self.right_panel.auto_ev_value = 0.0
            # Block signals to prevent triggering param_changed
            self.right_panel.exp_slider.blockSignals(True)
            self.right_panel.exp_slider.setValue(0)
            self.right_panel.exp_slider.update()  # Force visual refresh
            self.right_panel.exp_slider.blockSignals(False)
            self.right_panel.exp_value_label.setText(f"{tr('exposure_ev')}: 0.0")
        
        # 4. Restore params or Reset
        if path in self.file_params_cache:
            self.right_panel.set_params(self.file_params_cache[path])
        else:
            # Keep previous params (inherit from previous image)
            # We do NOT reset params here, so the new image inherits current UI settings
            # Processing will be triggered automatically by on_original_ready()
            # This ensures ALL settings (Exposure, WB, LUT, etc.) are carried over
            pass
        
        # 5. Update mark button state
        self.update_mark_button_state()
        
        # 6. Load Image
        logger.debug(f"[Gallery] Loading image: {os.path.basename(path)}")
        self.load_image(path)
        
        # 7. 如果这张图像有保存的基准点参数，加载后重新生成基准点图像
        if path in self.file_baseline_params_cache:
            # 延迟生成，等待图像加载完成
            QTimer.singleShot(500, self.regenerate_baseline_for_current_image)

    def load_image(self, path):
        self.preview_lbl.setText(tr('loading'))
        self.current_request_id = self.processor.current_request_id + 1
        self.processor.load_image(path)
        
    def on_param_changed(self, params):
        logger.debug(f"[Params] Parameter changed, starting debounce timer")
        # Debounce - trigger processing after brief delay
        self.update_timer.start()
    
    def trigger_processing(self):
        if not self.current_raw_path:
            return
        logger.debug(f"[Processing] Trigger processing for: {os.path.basename(self.current_raw_path)}")
        params = self.right_panel.get_params()
        self.current_request_id += 1
        self.processor.update_preview(self.current_raw_path, params)
    
    def save_baseline_image(self):
        """Save current params as baseline"""
        if not self.current_raw_path:
            return
        
        current_params = self.right_panel.get_params()
        self.file_baseline_params_cache[self.current_raw_path] = current_params.copy()
        
        # Generate baseline image immediately
        if self.processor.cached_linear is not None:
            # Copy cache state to baseline processor
            self.baseline_processor.cached_linear = self.processor.cached_linear
            self.baseline_processor.cached_corrected = self.processor.cached_corrected
            self.baseline_processor.cached_lens_key = self.processor.cached_lens_key
            self.baseline_processor.exif_data = self.processor.exif_data
            self.baseline_processor.current_path = self.current_raw_path
            self.baseline_processor.update_preview(self.current_raw_path, current_params)
        
        InfoBar.success(tr('baseline_saved'), tr('baseline_saved_message'), parent=self)
    
    def on_baseline_result(self, img_uint8, img_float, image_path, request_id):
        """Handle baseline image generation result"""
        if image_path != self.current_raw_path:
            return
        
        h, w, c = img_uint8.shape
        bytes_per_line = w * 3
        qimg = QImage(img_uint8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        
        # Update baseline state
        self.baseline.update_full(pixmap, img_float)
    
    def regenerate_baseline_for_current_image(self):
        """Regenerate baseline image for current image when switching"""
        if not self.current_raw_path:
            return
        
        if self.current_raw_path not in self.file_baseline_params_cache:
            return
        
        if self.processor.cached_linear is None:
            return
        
        baseline_params = self.file_baseline_params_cache[self.current_raw_path]
        self.baseline_processor.cached_linear = self.processor.cached_linear
        self.baseline_processor.cached_corrected = self.processor.cached_corrected
        self.baseline_processor.cached_lens_key = self.processor.cached_lens_key
        self.baseline_processor.exif_data = self.processor.exif_data
        self.baseline_processor.current_path = self.current_raw_path
        self.baseline_processor.update_preview(self.current_raw_path, baseline_params)

    def on_process_result(self, img_uint8, img_float, image_path, request_id, applied_ev):
        """Handle processed image result"""
        h, w, c = img_uint8.shape
        bytes_per_line = w * 3
        qimg = QImage(img_uint8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        
        # Always update thumbnail for the processed image, regardless of current selection
        for i in range(self.gallery_list.count()):
            item = self.gallery_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                thumb_pixmap = pixmap.scaled(
                    200, 200,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
                item.setIcon(QIcon(thumb_pixmap))
                break
        
        # Only update display and histogram if this is the current image
        if request_id != self.current_request_id or image_path != self.current_raw_path:
            return
        
        # Update current state
        self.current.update_full(pixmap, img_float.copy())  # Copy to avoid race conditions
        
        # If original is not set yet, set it to the first processed result
        # This allows compare function to work (show original vs current after adjustments)
        if self.original.full is None:
            self.original.update_full(pixmap.copy(), img_float.copy())
        
        # Update auto EV value if in auto mode
        if self.right_panel.auto_exp_radio.isChecked():
            self.right_panel.auto_ev_value = applied_ev
            # 临时断开valueChanged信号连接，避免触发参数变化
            try:
                self.right_panel.exp_slider.valueChanged.disconnect(self.right_panel.exp_slider_callback)
            except:
                pass  # 如果已经断开则忽略
            
            # 更新滑条值（此时不会触发信号）
            self.right_panel.exp_slider.setValue(int(applied_ev * 10))
            # 手动更新标签
            self.right_panel.exp_value_label.setText(f"{tr('exposure_ev')}: {applied_ev:+.1f}")
            
            # 重新连接信号
            self.right_panel.exp_slider.valueChanged.connect(self.right_panel.exp_slider_callback)
            logger.debug(f"[Result] Updated auto EV display to {applied_ev:+.1f} (signal temporarily disconnected)")
        
        # Display
        display_pixmap = self.current.get_display(self.preview_lbl.size())
        if display_pixmap:
            self.preview_lbl.setPixmap(display_pixmap)
            self.preview_lbl.update()
        
        # Update histogram and waveform
        self.right_panel.hist_widget.update_data(img_float)
        self.right_panel.waveform_widget.update_data(img_float)

    def on_load_complete(self, image_path, request_id):
        """Handle RAW loading completion"""
        logger.debug(f"[Load] Load complete: {os.path.basename(image_path)}, request_id={request_id}")
        
        # Ignore stale results
        if request_id != self.current_request_id or image_path != self.current_raw_path:
            logger.debug(f"[Load] Ignoring stale result (current_request_id={self.current_request_id}, current_path={os.path.basename(self.current_raw_path) if self.current_raw_path else 'None'})")
            return
        
        # RAW loading is complete, show processing message
        self.preview_lbl.setText(tr('processing'))
        
        # Cancel any pending debounce timer to avoid duplicate processing
        if self.update_timer.isActive():
            logger.debug("[Load] Stopping active debounce timer to avoid duplicate processing")
            self.update_timer.stop()
        
        # Trigger processing - but only if still the current image
        # Capture path to avoid race condition with fast image switching
        if image_path == self.current_raw_path:
            logger.debug(f"[Load] Scheduling processing for: {os.path.basename(image_path)}")
            QTimer.singleShot(0, lambda: self._trigger_processing_for_path(image_path))
    
    def _trigger_processing_for_path(self, path):
        """Trigger processing for specific path - prevents race conditions"""
        logger.debug(f"[Processing] _trigger_processing_for_path called for: {os.path.basename(path)}")
        
        if path != self.current_raw_path:
            logger.debug(f"[Processing] Path mismatch, aborting (current: {os.path.basename(self.current_raw_path) if self.current_raw_path else 'None'})")
            return  # User switched away, don't process
        
        params = self.right_panel.get_params()
        self.current_request_id += 1
        logger.debug(f"[Processing] Calling processor.update_preview, request_id={self.current_request_id}, exposure_mode={params.get('exposure_mode')}")
        self.processor.update_preview(path, params)

    def on_error(self, msg):
        self.preview_lbl.setText(f"{tr('error')}: {msg}")
        InfoBar.error(tr('error'), msg, parent=self)


    # --- Toolbar Actions ---
    
    def prev_image(self):
        count = self.gallery_list.count()
        if count == 0: return
        
        row = self.gallery_list.currentRow()
        # Loop to the end if at beginning
        new_row = (row - 1) % count
        self.gallery_list.setCurrentRow(new_row)

    def next_image(self):
        count = self.gallery_list.count()
        if count == 0: return

        row = self.gallery_list.currentRow()
        # Loop to start if at end
        new_row = (row + 1) % count
        self.gallery_list.setCurrentRow(new_row)

    def toggle_mark(self):
        if not self.current_raw_path: return
        
        if self.current_raw_path in self.marked_files:
            self.marked_files.remove(self.current_raw_path)
            InfoBar.info(tr('unmarked'), os.path.basename(self.current_raw_path), parent=self)
        else:
            self.marked_files.add(self.current_raw_path)
            InfoBar.success(tr('marked'), os.path.basename(self.current_raw_path), parent=self)
        
        # Update button state and gallery item indicator
        self.update_mark_button_state()
        self.update_gallery_item_mark_indicator(self.current_raw_path)
    
    def update_mark_button_state(self):
        """Update the mark button's checked state based on whether current image is marked"""
        if not self.current_raw_path:
            self.btn_mark.setChecked(False)
            return
        
        # Block signals to prevent triggering toggle_mark when programmatically setting state
        self.btn_mark.blockSignals(True)
        self.btn_mark.setChecked(self.current_raw_path in self.marked_files)
        self.btn_mark.blockSignals(False)

    def update_gallery_item_mark_indicator(self, path):
        """Update the green dot indicator on a gallery item"""
        for i in range(self.gallery_list.count()):
            item = self.gallery_list.item(i)
            item_path = item.data(Qt.ItemDataRole.UserRole)
            if item_path == path:
                # Update the item's text to show/hide the green dot
                is_marked = path in self.marked_files
                name = os.path.basename(path)
                
                if is_marked:
                    item.setText(f"🟢 {name}")
                else:
                    item.setText(name)
                break

    def delete_image(self):
        if not self.current_raw_path: return
        
        from PySide6.QtWidgets import QMessageBox
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            tr('delete_image'),
            tr('confirm_delete', filename=os.path.basename(self.current_raw_path)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Move to recycle bin using send2trash
                import send2trash
                # Normalize path to fix mixed slashes issue on Windows
                normalized_path = os.path.normpath(self.current_raw_path)
                send2trash.send2trash(normalized_path)
                
                # 优化: 清理所有相关缓存
                if self.current_raw_path in self.marked_files:
                    self.marked_files.remove(self.current_raw_path)
                
                if self.current_raw_path in self.file_params_cache:
                    del self.file_params_cache[self.current_raw_path]
                
                if self.current_raw_path in self.file_baseline_params_cache:
                    del self.file_baseline_params_cache[self.current_raw_path]
                
                if self.current_raw_path in self.thumbnail_cache:
                    del self.thumbnail_cache[self.current_raw_path]
                
                # Find and remove from gallery
                current_row = self.gallery_list.currentRow()
                for i in range(self.gallery_list.count()):
                    item = self.gallery_list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == self.current_raw_path:
                        self.gallery_list.takeItem(i)
                        break
                
                # Move to next image or previous if at end
                if self.gallery_list.count() > 0:
                    if current_row >= self.gallery_list.count():
                        current_row = self.gallery_list.count() - 1
                    self.gallery_list.setCurrentRow(current_row)
                else:
                    # No more images
                    self.current_raw_path = None
                    self.preview_lbl.setText(tr('no_image_selected'))
                    # Update window title when no image is selected
                    self.update_window_title()
                
                InfoBar.success(tr('delete_image'), tr('delete_image'), parent=self)
                
            except ImportError:
                # Fallback if send2trash is not installed
                InfoBar.error(tr('error'), tr('send2trash_error'), parent=self)
            except Exception as e:
                InfoBar.error(tr('delete_failed'), str(e), parent=self)

    def show_original(self, event):
        """Show baseline or original when comparing"""
        # Priority: baseline > original
        img_to_show = self.baseline if self.baseline.full else self.original
        
        if not img_to_show.full:
            return  # Image not loaded yet
        
        display_pixmap = img_to_show.get_display(self.preview_lbl.size())
        if display_pixmap:
            self.preview_lbl.setPixmap(display_pixmap)

        InfoBar.info(
            tr('compare_showing_baseline'),
            "",
            parent=self
        )

    def show_processed(self, event):
        """Show current processed image"""
        if not self.current.full:
            # Fallback to original if current not ready
            if self.original.full:
                display_pixmap = self.original.get_display(self.preview_lbl.size())
                if display_pixmap:
                    self.preview_lbl.setPixmap(display_pixmap)
            return
        
        display_pixmap = self.current.get_display(self.preview_lbl.size())
        if display_pixmap:
            self.preview_lbl.setPixmap(display_pixmap)

    def export_current(self):
        if not self.current_raw_path: return
        
        # Save Dialog with HEIF support
        # Remove the RAW extension from the default filename to avoid "123.RW2.jpg"
        base_name_without_ext = os.path.splitext(os.path.basename(self.current_raw_path))[0]
        
        # Use last export path, or fall back to gallery folder
        start_dir = ""
        if self.last_export_path and os.path.exists(self.last_export_path):
            start_dir = self.last_export_path
        elif self.last_folder_path and os.path.exists(self.last_folder_path):
            start_dir = self.last_folder_path
        
        if start_dir:
            default_path = os.path.join(start_dir, base_name_without_ext)
        else:
            default_path = base_name_without_ext
        
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            tr('export_image'),
            default_path,
            "JPEG (*.jpg);;HEIF (*.heif);;TIFF (*.tif);;DNG (*.dng)"
        )
        
        # Check if user clicked OK and provided a valid path
        if path and len(path.strip()) > 0:
            # Remember the export directory
            self.last_export_path = os.path.dirname(path)
            # 显示保存中通知
            self.saving_infobar = InfoBar.info(
                tr('saving'),
                tr('saving_image'),
                duration=-1,  # 持续显示直到手动关闭
                parent=self
            )
            
            # 禁用保存当前图片按钮
            self.btn_export_curr.setEnabled(False)
            
            self.run_export(
                input_path=self.current_raw_path,
                output_path=path,
                is_single_export=True
            )
        else:
            logger.debug("[Export] User cancelled or no path selected")

    def export_all(self):
        if not self.marked_files:
             InfoBar.warning(tr('no_files_marked'), tr('please_mark_files'), parent=self)
             return
        
        # Ask for format
        formats = ["JPEG", "HEIF", "TIFF", "DNG"]
        format_str, ok = QInputDialog.getItem(self, tr('select_export_format'), "Format:", formats, 0, False)
        
        if not ok:
            return
        
        # Use last export path, or fall back to gallery folder
        start_dir = ""
        if self.last_export_path and os.path.exists(self.last_export_path):
            start_dir = self.last_export_path
        elif self.last_folder_path and os.path.exists(self.last_folder_path):
            start_dir = self.last_folder_path
        
        folder = QFileDialog.getExistingDirectory(self, tr('select_export_folder'), start_dir)
        if folder:
             # Remember the export directory
             self.last_export_path = folder
             # Batch export marked files
             self.batch_export_list = list(self.marked_files)
             self.batch_export_folder = folder
             
             # Map format string to extension
             fmt_map = {"JPEG": "jpg", "HEIF": "heif", "TIFF": "tif", "DNG": "dng"}
             self.batch_export_ext = fmt_map.get(format_str, "jpg")
             
             # 显示批量导出中通知
             self.batch_saving_infobar = InfoBar.info(
                 tr('saving'),
                 tr('batch_exporting'),
                 duration=-1,  # 持续显示直到手动关闭
                 parent=self
             )
             
             # Initialize Progress UI
             self.export_progress.setRange(0, len(self.batch_export_list))
             self.export_progress.setValue(0)
             self.export_progress.show()
             self.btn_export_all.setEnabled(False) # Disable button during export
             
             self.batch_export_idx = 0
             self.batch_export_next()

    def batch_export_next(self):
         if self.batch_export_idx >= len(self.batch_export_list):
             # 关闭批量导出通知
             if hasattr(self, 'batch_saving_infobar') and self.batch_saving_infobar:
                 self.batch_saving_infobar.close()
                 self.batch_saving_infobar = None
             
             InfoBar.success(tr('batch_export'), tr('all_exported'), parent=self)
             self.export_progress.hide()
             self.btn_export_all.setEnabled(True)
             return
         
         # Update Progress
         self.export_progress.setValue(self.batch_export_idx)

         input_path = self.batch_export_list[self.batch_export_idx]
         filename = os.path.basename(input_path)
         
         # Use selected extension
         output_path = os.path.join(self.batch_export_folder, os.path.splitext(filename)[0] + "." + self.batch_export_ext)
         
         # Determine params for this file
         if input_path == self.current_raw_path:
             # Use current UI params
             params = self.right_panel.get_params()
         else:
             # Use cached params
             params = self.file_params_cache.get(input_path)
             if not params:
                 # Fallback to current if not found (should not happen for marked files)
                 params = self.right_panel.get_params()

         self.batch_export_idx += 1
         
         # Trigger single export but chain the next one
         # We'll use a modified run_export that accepts a callback
         self.run_export(input_path, output_path, params=params, callback=self.batch_export_next)


    def run_export(self, input_path, output_path, params=None, callback=None, is_single_export=False):
        # Gather params
        p = params if params else self.right_panel.get_params()
        
        # Determine format from extension
        ext = os.path.splitext(output_path)[1].lower().replace('.', '')
        if ext not in ['jpg', 'heif', 'tif', 'tiff', 'dng']: ext = 'jpg'
        
        # 确定曝光值：无论是手动还是自动模式，都使用界面上的值
        # 这样可以确保导出的图像与预览一致
        if p['exposure_mode'] == 'Manual':
            # 手动模式：使用滑条的值
            export_exposure = p['exposure']
        else:
            # 自动模式：使用已经计算好的auto_ev_value
            export_exposure = self.right_panel.auto_ev_value
        
        # Create a thread to run orchestrator (it blocks otherwise)
        # Using a simple QThread wrapper
        
        class ExportThread(QThread):
            finished_sig = Signal(bool, str)
            
            def run(self):
                try:
                    orchestrator.process_path(
                        input_path=input_path,
                        output_path=output_path,
                        log_space=p['log_space'],
                        lut_path=p['lut_path'],
                        exposure=export_exposure,  # 始终使用界面上的曝光值
                        lens_correct=p['lens_correct'],
                        custom_db_path=p['custom_db_path'],
                        metering_mode=p.get('metering_mode', 'matrix'),
                        jobs=1,
                        logger_func=lambda msg: None, # Mute log for now
                        output_format=ext,
                        wb_temp=p['wb_temp'],
                        wb_tint=p['wb_tint'],
                        saturation=p['saturation'],
                        contrast=p['contrast'],
                        highlight=p['highlight'],
                        shadow=p['shadow']
                    )
                    self.finished_sig.emit(True, "")
                except Exception as e:
                    self.finished_sig.emit(False, str(e))
        
        self.export_thread = ExportThread()
        
        def on_finish(success, msg):
            # 如果是单个导出，关闭保存中通知并重新启用按钮
            if is_single_export:
                if hasattr(self, 'saving_infobar') and self.saving_infobar:
                    self.saving_infobar.close()
                    self.saving_infobar = None
                self.btn_export_curr.setEnabled(True)
            
            if success:
                if not callback:
                    InfoBar.success(tr('export_success'), tr('saved_to', path=os.path.basename(output_path)), parent=self)
                if callback: callback()
            else:
                InfoBar.error(tr('export_failed'), msg, parent=self)
        
        self.export_thread.finished_sig.connect(on_finish)
        self.export_thread.start()
    
    
    def resizeEvent(self, event):
        """Clear display caches on resize, let ImageState handle re-scaling"""
        super().resizeEvent(event)
        
        # Clear display caches - full pixmaps remain
        self.original.display = None
        self.current.display = None
        self.baseline.display = None
        
        # Debounce re-display
        if not hasattr(self, 'resize_timer'):
            self.resize_timer = QTimer()
            self.resize_timer.setSingleShot(True)
            self.resize_timer.setInterval(100)
            self.resize_timer.timeout.connect(self._on_resize_complete)
        
        self.resize_timer.start()
    
    def _on_resize_complete(self):
        """Re-display current image after resize"""
        if self.current.full:
            display_pixmap = self.current.get_display(self.preview_lbl.size())
            if display_pixmap:
                self.preview_lbl.setPixmap(display_pixmap)
    
    def closeEvent(self, event):
        """Save settings when closing the application"""
        self.save_settings()
        super().closeEvent(event)


def launch_gui():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    launch_gui()
