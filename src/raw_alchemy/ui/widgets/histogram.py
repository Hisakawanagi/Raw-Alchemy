from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPolygonF
import numpy as np
from loguru import logger
from raw_alchemy import utils

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.hist_data = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # 优化: 添加更新定时器防抖
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(25)  # 50ms防抖
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
        except BaseException as e:
            # 捕获所有BaseException子类，避免异常处理错误
            try:
                logger.error(f"Histogram update error: {type(e).__name__}: {e}")
            except Exception:
                # 如果日志记录也失败，完全静默（不捕获系统级异常）
                pass

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
