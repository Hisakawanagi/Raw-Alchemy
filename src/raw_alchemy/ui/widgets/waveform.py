from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen
import numpy as np
from loguru import logger
from raw_alchemy import utils

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
        self.update_timer.setInterval(25)  # 50ms防抖
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
            
            # 使用utils中的快速计算函数 - 增加bins数量以提高垂直分辨率
            waveform_result = utils.compute_waveform_fast(data, bins=150, sample_rate=8)
            
            # 检查结果是否有效
            if waveform_result is not None:
                self.waveform_data = waveform_result
                self.update()
        except (RuntimeError, ValueError, TypeError, OSError, SystemError) as e:
            # 静默处理这些异常，避免干扰UI
            logger.warning(f"Waveform update error: {type(e).__name__}: {e}")
        except BaseException as e:
            # 捕获所有BaseException子类，避免异常处理错误
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
                        
                        # 根据密度设置透明度和亮度（大幅增强显示效果）
                        density = column_data[bin_idx]
                        
                        # 使用更激进的映射策略，让低密度区域也更明显
                        # 对密度进行非线性映射，提升低密度值的可见度
                        enhanced_density = np.power(density, 0.6)  # 0.6次方让低值更明显
                        
                        # 大幅提高基础透明度和密度系数
                        alpha = int(enhanced_density * 200) + 150
                        alpha = min(255, alpha)
                        
                        # 使用灰白色（亮度波形用灰度显示更专业）
                        # 大幅提高基础亮度，让波形更明显
                        brightness = int(enhanced_density * 150) + 180
                        brightness = min(255, brightness)
                        color = QColor(brightness, brightness, brightness, alpha)
                        
                        # 使用更粗的笔触绘制，提高可见度
                        pen = QPen(color)
                        pen.setWidth(2)  # 使用2像素宽度
                        painter.setPen(pen)
                        
                        # 绘制点
                        painter.drawPoint(int(x), int(y))
            
        except Exception as e:
            logger.error(f"Error painting waveform: {e}")
