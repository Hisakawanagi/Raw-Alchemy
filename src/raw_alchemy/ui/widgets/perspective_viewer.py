"""
Perspective Correction Viewer Widget
Allows user to drag 4 corner control points to correct perspective distortion.
"""
from typing import Tuple, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGraphicsDropShadowEffect, QFrame
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap, QPolygonF

from qfluentwidgets import (
    BodyLabel, PushButton, PrimaryPushButton, FluentIcon as FIF
)
from raw_alchemy.i18n import tr


class PerspectiveViewer(QWidget):
    """
    Interactive Perspective Correction Viewer.
    - Displays the image with 4 draggable corner control points.
    - Previews the perspective transformation in real-time.
    - Emits applied signal with corner coordinates when user confirms.
    """
    
    # Signal emitted when user confirms changes (returns 4 corner tuples)
    applied = Signal(tuple)  # ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.original_pixmap: Optional[QPixmap] = None
        
        # Corner positions in normalized coordinates (0-1)
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        self.corners = [
            (0.0, 0.0),  # TL
            (1.0, 0.0),  # TR
            (1.0, 1.0),  # BR
            (0.0, 1.0),  # BL
        ]
        
        # Interaction state
        self.dragging = False
        self.dragging_corner = -1  # Which corner is being dragged (0-3)
        self.drag_start_pos = QPointF()
        
        # UI
        self._init_ui()
        self.setMouseTracking(True)
        
    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # 1. Viewport (Canvas)
        self.viewport = QWidget()
        from PySide6.QtWidgets import QSizePolicy
        self.viewport.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 2. Toolbar (Bottom)
        self.toolbar = QFrame()
        self.toolbar.setObjectName("PerspectiveToolbar")
        self.toolbar.setFixedHeight(80)
        self.toolbar.setStyleSheet("""
            #PerspectiveToolbar {
                background-color: #2b2b2b;
                border-top: 1px solid #3a3a3a;
            }
        """)
        
        tb_layout = QHBoxLayout(self.toolbar)
        
        self.hint_label = BodyLabel(tr('perspective_hint'))
        self.hint_label.setStyleSheet("color: #888888;")
        
        self.btn_reset = PushButton(tr('reset'))
        self.btn_reset.clicked.connect(self.reset_corners)
        
        self.btn_cancel = PushButton(tr('cancel'))
        self.btn_cancel.clicked.connect(self.cancelled.emit)
        
        self.btn_done = PrimaryPushButton(tr('done'))
        self.btn_done.setIcon(FIF.ACCEPT)
        self.btn_done.clicked.connect(self._on_done)
        
        tb_layout.addWidget(self.hint_label)
        tb_layout.addStretch()
        tb_layout.addWidget(self.btn_reset)
        tb_layout.addWidget(self.btn_cancel)
        tb_layout.addWidget(self.btn_done)
        
        self.layout.addWidget(self.viewport, 1)
        self.layout.addWidget(self.toolbar)
        
        # Event filter for viewport painting
        self.viewport.installEventFilter(self)

    def set_image(self, pixmap: QPixmap, current_params: dict):
        """
        Load image and initial corner state.
        pixmap: The image to display.
        current_params: Dict containing 'perspective_corners' if previously set.
        """
        self.original_pixmap = pixmap
        
        # Parse existing corners from params
        corners = current_params.get('perspective_corners')
        if corners and len(corners) == 4:
            self.corners = list(corners)
        else:
            # Default: corners at image edges
            self.corners = [
                (0.0, 0.0),  # TL
                (1.0, 0.0),  # TR
                (1.0, 1.0),  # BR
                (0.0, 1.0),  # BL
            ]
        
        self.viewport.update()

    def reset_corners(self):
        """Reset corners to default (image edges)"""
        self.corners = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        self.viewport.update()

    def eventFilter(self, obj, event):
        if obj == self.viewport and event.type() == event.Type.Paint:
            self._paint_viewport()
            return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if not self.original_pixmap:
            return
        
        pos = event.position()
        rect_map = self._get_image_screen_rect()
        if not rect_map:
            return
        
        dx, dy, dw, dh = rect_map.x(), rect_map.y(), rect_map.width(), rect_map.height()
        
        # Check if clicking near a corner
        hit_radius = 20.0
        for i, (cx, cy) in enumerate(self.corners):
            sx = dx + cx * dw
            sy = dy + cy * dh
            if (pos - QPointF(sx, sy)).manhattanLength() < hit_radius:
                self.dragging = True
                self.dragging_corner = i
                self.drag_start_pos = pos
                return
    
    def mouseMoveEvent(self, event):
        if not self.dragging or not self.original_pixmap:
            return
        
        pos = event.position()
        rect_map = self._get_image_screen_rect()
        if not rect_map:
            return
        
        dx, dy, dw, dh = rect_map.x(), rect_map.y(), rect_map.width(), rect_map.height()
        
        # Convert screen position to normalized coordinates
        nx = (pos.x() - dx) / dw
        ny = (pos.y() - dy) / dh
        
        # Clamp to image bounds (strict)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        
        # Update corner position
        self.corners[self.dragging_corner] = (nx, ny)
        self.viewport.update()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.dragging_corner = -1

    def _paint_viewport(self):
        if not self.original_pixmap:
            return
        
        painter = QPainter(self.viewport)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # 1. Draw Image (Scaled to fit)
        view_w = self.viewport.width()
        view_h = self.viewport.height()
        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()
        
        # Scale to fit
        scale = min(view_w / img_w, view_h / img_h) * 0.85  # Margin
        
        draw_w = img_w * scale
        draw_h = img_h * scale
        
        dx = (view_w - draw_w) / 2
        dy = (view_h - draw_h) / 2
        
        self.display_rect = QRectF(dx, dy, draw_w, draw_h)
        
        painter.drawPixmap(self.display_rect.toRect(), self.original_pixmap)
        
        # 2. Draw semi-transparent overlay outside the quad
        # (Optional: darken areas outside the perspective quad)
        
        # 3. Draw corner points and connecting lines
        # Convert normalized corners to screen coordinates
        screen_corners = []
        for (cx, cy) in self.corners:
            sx = dx + cx * draw_w
            sy = dy + cy * draw_h
            screen_corners.append(QPointF(sx, sy))
        
        # Draw quad outline
        pen_outline = QPen(QColor(255, 255, 255))
        pen_outline.setWidth(2)
        painter.setPen(pen_outline)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        quad = QPolygonF(screen_corners)
        painter.drawPolygon(quad)
        
        # Draw diagonal guide lines
        pen_guide = QPen(QColor(255, 255, 255, 80))
        pen_guide.setWidth(1)
        pen_guide.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen_guide)
        
        # Diagonals
        painter.drawLine(screen_corners[0], screen_corners[2])  # TL to BR
        painter.drawLine(screen_corners[1], screen_corners[3])  # TR to BL
        
        # Draw corner handles
        handle_radius = 8
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(QBrush(QColor(100, 150, 255)))
        
        for i, pt in enumerate(screen_corners):
            # Highlight dragging corner
            if self.dragging and self.dragging_corner == i:
                painter.setBrush(QBrush(QColor(255, 200, 100)))
            else:
                painter.setBrush(QBrush(QColor(100, 150, 255)))
            
            painter.drawEllipse(pt, handle_radius, handle_radius)
        
        # Draw corner labels
        labels = ['TL', 'TR', 'BR', 'BL']
        painter.setPen(QPen(QColor(255, 255, 255)))
        for i, pt in enumerate(screen_corners):
            offset = QPointF(12, -12) if i < 2 else QPointF(12, 20)
            painter.drawText(pt + offset, labels[i])

    def _get_image_screen_rect(self):
        if not hasattr(self, 'display_rect'):
            return None
        return self.display_rect

    def _on_done(self):
        # Convert corners list to tuple of tuples
        corners_tuple = tuple(tuple(c) for c in self.corners)
        self.applied.emit(corners_tuple)
