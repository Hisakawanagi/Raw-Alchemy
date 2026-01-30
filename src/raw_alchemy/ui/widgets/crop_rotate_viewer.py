import math
from typing import Tuple, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGraphicsDropShadowEffect, QFrame
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap, QAction

from qfluentwidgets import (
    SubtitleLabel, StrongBodyLabel, BodyLabel, 
    PushButton, PrimaryPushButton, ToolButton, Slider, 
    FluentIcon as FIF, ComboBox, SwitchButton
)
from raw_alchemy.i18n import tr
from raw_alchemy import utils

class CropRotateViewer(QWidget):
    """
    Unified Crop & Rotate Viewer.
    - Displays the rotated image.
    - Overlays crop rectangle on top of the rotated image.
    - Handles view scaling so the rotated image fits.
    """
    
    # Signal emitted when user confirms changes (returns Rotation, FlipH, FlipV, CropRect)
    applied = Signal(int, bool, bool, tuple)
    cancelled = Signal()
    
    # Internal signal for preview updates (fast-path)
    # preview_changed = Signal(int, bool, bool) 

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.original_pixmap: Optional[QPixmap] = None
        self.display_pixmap: Optional[QPixmap] = None # Rotated/Flipped internal cache
        
        # State
        self.rotation = 0
        self.flip_h = False
        self.flip_v = False
        self.crop_rect = (0.0, 0.0, 1.0, 1.0) # Normalized (x, y, w, h)
        
        # Interaction
        self.dragging = False
        self.drag_mode = None # 'move', 'tl', 'tr', 'bl', 'br', 'l', 't', 'r', 'b'
        self.drag_start_pos = QPointF()
        self.rect_start_drag = None # Saved rect state
        
        # Config
        self.min_crop_size = 0.05 # Minimum 5% width/height
        self.aspect_ratio = 0.0 # 0 = Free
        
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
        # We handle painting in paintEvent
        
        # 2. Toolbar (Bottom)
        self.toolbar = QFrame()
        self.toolbar.setObjectName("CropToolbar")
        self.toolbar.setFixedHeight(140) # Taller for controls
        self.toolbar.setStyleSheet("""
            #CropToolbar {
                background-color: #2b2b2b;
                border-top: 1px solid #3a3a3a;
            }
        """)
        
        tb_layout = QVBoxLayout(self.toolbar)
        
        # Row 1: Rotation Slider & Flip
        row1 = QHBoxLayout()
        
        self.rotate_slider = Slider(Qt.Orientation.Horizontal)
        self.rotate_slider.setRange(-45, 45)
        self.rotate_slider.setValue(0)
        self.rotate_slider.valueChanged.connect(self._on_rotate_slider)
        
        self.rotate_val_lbl = BodyLabel("0°")
        self.rotate_val_lbl.setFixedWidth(40)
        self.rotate_val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_rot_left = ToolButton(FIF.ROTATE)
        self.btn_rot_left.clicked.connect(self._rotate_left_90)
        
        self.btn_rot_right = ToolButton(FIF.ROTATE)
        self.btn_rot_right.clicked.connect(self._rotate_right_90)
        
        self.btn_flip_h = ToolButton(FIF.SYNC) # Placeholder icon for Flip H
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_h.setToolTip(tr('flip_horizontal'))
        self.btn_flip_h.clicked.connect(self._on_flip_changed)
        
        self.btn_flip_v = ToolButton(FIF.SYNC) # Placeholder for V
        self.btn_flip_v.setCheckable(True)
        self.btn_flip_v.setToolTip(tr('flip_vertical'))
        self.btn_flip_v.clicked.connect(self._on_flip_changed)
        
        row1.addWidget(BodyLabel(tr('rotation')))
        row1.addWidget(self.btn_rot_left)
        row1.addWidget(self.rotate_slider)
        row1.addWidget(self.rotate_val_lbl)
        row1.addWidget(self.btn_rot_right)
        row1.addSpacing(20)
        row1.addWidget(self.btn_flip_h)
        row1.addWidget(self.btn_flip_v)
        
        # Row 2: Aspect Ratio & Actions
        row2 = QHBoxLayout()
        
        self.combo_ratio = ComboBox()
        self.combo_ratio.addItems([
            tr('free'), tr('original'), "1:1", "3:2", "4:3", "16:9", 
            "2:3", "3:4", "9:16"
        ])
        self.combo_ratio.currentIndexChanged.connect(self._on_ratio_changed)
        
        self.btn_reset = PushButton(tr('reset'))
        self.btn_reset.clicked.connect(self.reset_state)
        
        self.btn_cancel = PushButton(tr('cancel'))
        self.btn_cancel.clicked.connect(self.cancelled.emit)
        
        self.btn_done = PrimaryPushButton(tr('done'))
        self.btn_done.setIcon(FIF.ACCEPT)
        self.btn_done.clicked.connect(self._on_done)
        
        row2.addWidget(BodyLabel(tr('aspect_ratio')))
        row2.addWidget(self.combo_ratio)
        row2.addStretch()
        row2.addWidget(self.btn_reset)
        row2.addWidget(self.btn_cancel)
        row2.addWidget(self.btn_done)
        
        tb_layout.addLayout(row1)
        tb_layout.addLayout(row2)
        
        self.layout.addWidget(self.viewport, 1)
        self.layout.addWidget(self.toolbar)
        
        # Event filter for viewport painting
        self.viewport.installEventFilter(self)

    def set_image(self, pixmap: QPixmap, current_params: dict):
        """
        Load image and initial state. 
        pixmap: The linear/corrected image (before geometry/crop).
        """
        self.original_pixmap = pixmap
        
        # Parse params
        self.rotation = current_params.get('rotation', 0)
        self.flip_h = current_params.get('flip_horizontal', False)
        self.flip_v = current_params.get('flip_vertical', False)
        self.crop_rect = current_params.get('crop', (0.0, 0.0, 1.0, 1.0))
        if self.crop_rect is None: self.crop_rect = (0.0, 0.0, 1.0, 1.0)
        
        # Set UI state
        self._update_ui_controls()
        
        # Generate initial display image
        self._update_display_pixmap()
        self.viewport.update()

    def _update_ui_controls(self):
        # Rotation logic: Split into mod 90 and fine
        # But here we treat total rotation as slider + base if we want
        # Or just slider for fine (-45..45) and buttons for coarse
        # Let's decompose
        base = round(self.rotation / 90.0) * 90
        fine = self.rotation - base
        
        # Prevent slider jumps: if fine is > 45, shift base?
        # Actually simplest is: Slider is exclusively fine tune.
        # But we need to display total properly?
        # For simplicity, let internal self.rotation be the master.
        # Slider shows `rotation % 90` if we treated it wrapped? No.
        # Let's do: Slider wraps -45 to 45. 
        # If rotation is 95, base=90, slider=5.
        
        # Normalize fine to -45..45
        while fine > 45:
            fine -= 90
            base += 90
        while fine < -45:
            fine += 90
            base -= 90
            
        self.rotate_slider.blockSignals(True)
        self.rotate_slider.setValue(int(fine))
        self.rotate_slider.blockSignals(False)
        self.rotate_val_lbl.setText(f"{int(fine)}°")
        
        self.btn_flip_h.setChecked(self.flip_h)
        self.btn_flip_v.setChecked(self.flip_v)

    def _update_display_pixmap(self):
        """Generates the rotated/flipped background image"""
        if not self.original_pixmap: return
        
        # Check cache logic?
        # For UI fluidity, we might want to do this cheaply.
        # QPixmap transform
        
        transform = list()
        
        # 1. Flip (Pre-rotation) or Post? Standard is usually before or after?
        # Our utils.apply_geometry: Rotate THEN Flip.
        # Wait, `apply_geometry` code: 
        #   if rotation... rot90 or rotate
        #   if flip_h... fliplr
        # So Rotation happens first.
        
        # QTransform
        t =  Qt.TransformationMode.FastTransformation
        img = self.original_pixmap
        
        # Rotate
        if self.rotation != 0:
            import math
            # QPixmap.transformed wraps around center usually.
            from PySide6.QtGui import QTransform
            trans = QTransform()
            trans.rotate(self.rotation)
            img = img.transformed(trans, Qt.TransformationMode.SmoothTransformation)
            
        # Flip
        if self.flip_h or self.flip_v:
             img = img.transformed(
                 from_scale( -1 if self.flip_h else 1, -1 if self.flip_v else 1 ),
                 Qt.TransformationMode.FastTransformation
             )
             
        self.display_pixmap = img
        
        # Update validation of crop rect (if image shape changed drastically?)
        # Since we use normalized coordinates, they technically remain valid relative to the *new* geometry.
        # BUT: rotating a rectangle might make the old crop content invalid conceptually?
        # Lightroom strategy: Crop rotates WITH the image.
        # Which means: Crop Rect (Norm) is relative to the Rotated Canvas.
        # So if I rotate 90, the canvas becomes HxW. Crop (0,0,1,1) is still full image.
        # This matches our pipeline logic: Stage 2.5 (Geometry) -> Stage 2.6 (Crop).
        pass

    def _on_rotate_slider(self, val):
        # We need to know previous base
        base = round(self.rotation / 90.0) * 90
        # Re-calc base based on current slider vs previous?
        # Simpler: Just take current nearest 90-step of self.rotation, add new slider val.
        # But this snaps if I drag slider across boundary?
        # Implementation: Slider is purely "Add to base".
        # If user wants to rotate 90, they click button.
        self.rotation = base + val
        self.rotate_val_lbl.setText(f"{val}°")
        
        # Re-generate pixmap
        self._update_display_pixmap()
        
        # Auto-Constrain Crop if needed
        # self._constrain_crop_to_valid() # TODO
        
        self.viewport.update()

    def _rotate_left_90(self):
        self.rotation = (self.rotation - 90)
        self._update_display_pixmap()
        self._update_ui_controls()
        self.viewport.update()

    def _rotate_right_90(self):
        self.rotation = (self.rotation + 90)
        self._update_display_pixmap()
        self._update_ui_controls()
        self.viewport.update()

    def _on_flip_changed(self):
        self.flip_h = self.btn_flip_h.isChecked()
        self.flip_v = self.btn_flip_v.isChecked()
        self._update_display_pixmap()
        self.viewport.update()

    def _on_ratio_changed(self):
        txt = self.combo_ratio.currentText()
        if txt == tr('original'):
             # Need original aspect ratio
             if self.original_pixmap:
                 self.aspect_ratio = self.original_pixmap.width() / self.original_pixmap.height()
        elif txt == tr('free'):
            self.aspect_ratio = 0.0
        elif ":" in txt:
            w, h = map(float, txt.split(":"))
            self.aspect_ratio = w / h
            
        # Apply ratio immediately to current crop?
        if self.aspect_ratio > 0:
            self._apply_aspect_ratio_to_current()
            self.viewport.update()

    def _apply_aspect_ratio_to_current(self):
        if self.aspect_ratio <= 0: return
        # Adjust current crop rect to match aspect ratio
        # Try to keep Center fixed
        x, y, w, h = self.crop_rect
        
        # Convert to screen/image aspect space? 
        # Normalized coords imply W and H depend on image dimensions.
        if not self.display_pixmap: return
        img_w = self.display_pixmap.width()
        img_h = self.display_pixmap.height()
        
        # Current pixel dims
        px_w = w * img_w
        px_h = h * img_h
        
        target_h = px_w / self.aspect_ratio
        
        if target_h <= img_h:
            # Fit by height adjustment
            px_h = target_h
        else:
            # Fit by width adjustment
            px_w = px_h * self.aspect_ratio
            
        # Norm back
        new_w = px_w / img_w
        new_h = px_h / img_h
        
        # Center
        cx = x + w/2
        cy = y + h/2
        
        new_x = cx - new_w/2
        new_y = cy - new_h/2
        
        # Constrain
        if new_x < 0: new_x = 0
        if new_y < 0: new_y = 0
        if new_x + new_w > 1: new_x = 1 - new_w
        if new_y + new_h > 1: new_y = 1 - new_h
        
        self.crop_rect = (new_x, new_y, new_w, new_h)

    def eventFilter(self, obj, event):
        if obj == self.viewport and event.type() == event.Type.Paint:
            self._paint_viewport()
            return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if not self.display_pixmap: return
        pos = event.position()
        
        # Get screen rect of the image
        rect_map = self._get_image_screen_rect()
        if not rect_map: return
        
        dx, dy, dw, dh = rect_map.x(), rect_map.y(), rect_map.width(), rect_map.height()
        
        # Current crop in screen coords
        cx, cy, cw, ch = self.crop_rect
        sx = dx + cx * dw
        sy = dy + cy * dh
        sw = cw * dw
        sh = ch * dh
        
        handles = {
            'tl': QPointF(sx, sy),
            'tr': QPointF(sx + sw, sy),
            'bl': QPointF(sx, sy + sh),
            'br': QPointF(sx + sw, sy + sh),
            'l': QPointF(sx, sy + sh/2),
            'r': QPointF(sx + sw, sy + sh/2),
            't': QPointF(sx + sw/2, sy),
            'b': QPointF(sx + sw/2, sy + sh)
        }
        
        hit_radius = 15.0
        
        # Check corners first
        for key in ['tl', 'tr', 'bl', 'br']:
            if (pos - handles[key]).manhattanLength() < hit_radius:
                self.drag_mode = key
                self.dragging = True
                self.drag_start_pos = pos
                self.rect_start_drag = self.crop_rect
                return
        
        # Check edges (optional, for now just corners for simplicity or all?)
        # Let's support edges
        for key in ['l', 'r', 't', 'b']:
            if (pos - handles[key]).manhattanLength() < hit_radius:
                 self.drag_mode = key
                 self.dragging = True
                 self.drag_start_pos = pos
                 self.rect_start_drag = self.crop_rect
                 return
                 
        # Check inside (Move)
        if sx < pos.x() < sx + sw and sy < pos.y() < sy + sh:
            self.drag_mode = 'move'
            self.dragging = True
            self.drag_start_pos = pos
            self.rect_start_drag = self.crop_rect
            return
            
    def mouseMoveEvent(self, event):
        if not self.dragging or not self.display_pixmap: return
        
        pos = event.position()
        rect_map = self._get_image_screen_rect()
        dw, dh = rect_map.width(), rect_map.height() # Image draw dimensions
        
        # Delta in normalized coords
        delta_x = (pos.x() - self.drag_start_pos.x()) / dw
        delta_y = (pos.y() - self.drag_start_pos.y()) / dh
        
        start_x, start_y, start_w, start_h = self.rect_start_drag
        
        # Initialize defaults to prevent UnboundLocalError
        new_x, new_y, new_w, new_h = start_x, start_y, start_w, start_h

        # Calculate proposed new edges based on drag
        left, top = start_x, start_y
        right = start_x + start_w
        bottom = start_y + start_h
        
        mode = self.drag_mode
        
        if mode == 'move':
            new_x = start_x + delta_x
            new_y = start_y + delta_y
            
            # Constrain to bounds
            if new_x < 0: new_x = 0
            if new_y < 0: new_y = 0
            if new_x + start_w > 1: new_x = 1 - start_w
            if new_y + start_h > 1: new_y = 1 - start_h
            
            self.crop_rect = (new_x, new_y, start_w, start_h)
            
        else:
            # Resizing logic with constraints
            
            # 1. Apply delta to get raw proposed edges
            if 'l' in mode: left += delta_x
            if 'r' in mode: right += delta_x
            if 't' in mode: top += delta_y
            if 'b' in mode: bottom += delta_y
            
            # 2. Enforce minimum size (flip if needed, or clamp)
            # For simplicity, prevent flipping by clamping min size
            if right - left < self.min_crop_size:
                if 'l' in mode: left = right - self.min_crop_size
                else: right = left + self.min_crop_size
            if bottom - top < self.min_crop_size:
                if 't' in mode: top = bottom - self.min_crop_size
                else: bottom = top + self.min_crop_size

            # 3. Apply Aspect Ratio & Bounds Constraint
            if self.aspect_ratio > 0:
                # Target Aspect Ratio constant K = (W_norm / H_norm)
                # K = AR * (image_h_px / image_w_px)
                K = self.aspect_ratio * (dh / dw)
                
                # Determine Anchor Point (the side/corner that stays fixed)
                # and the growing direction.
                
                # Corner dragging
                if mode == 'tl':
                    anchor_x, anchor_y = self.rect_start_drag[0] + self.rect_start_drag[2], self.rect_start_drag[1] + self.rect_start_drag[3] # Bottom-Right
                    # Allowed region for Top-Left is (0,0) to (anchor_x, anchor_y)
                    # Vector from Anchor to Mouse: (-w, -h)
                    # We want to maximize size inside bounds.
                    
                    # Current proposed width/height from anchor
                    curr_w = anchor_x - left
                    curr_h = anchor_y - top
                    
                    # Constrain to max available space
                    max_w = anchor_x # Distance to left edge 0
                    max_h = anchor_y # Distance to top edge 0
                    
                    # Solve: w / h = K  => w = h * K
                    # Bound 1: w <= max_w
                    # Bound 2: h <= max_h  =>  w/K <= max_h => w <= max_h * K
                    
                    # Also respect drag direction (user intention):
                    # Usually we take the largest possible rect that fits in the user's dragged box?
                    # Or we project the user's mouse pos onto the diagonal?
                    # Projecting is better for UX.
                    
                    # Clamped proposed dims
                    curr_w = max(self.min_crop_size, min(curr_w, max_w))
                    curr_h = max(self.min_crop_size, min(curr_h, max_h))
                    
                    # Enforce AR
                    if curr_w / curr_h > K:
                        # Width is too wide for height -> limited by height OR user drag
                        # If we are limited by bounds, we must shrink.
                        # If limited by mouse, we adjust strictly.
                        curr_w = curr_h * K
                        if curr_w > max_w: # If still too wide (bound hit), shrink h
                             curr_w = max_w
                             curr_h = curr_w / K
                    else:
                        curr_h = curr_w / K
                        if curr_h > max_h:
                            curr_h = max_h
                            curr_w = curr_h * K
                            
                    left = anchor_x - curr_w
                    top = anchor_y - curr_h
                    right = anchor_x
                    bottom = anchor_y

                elif mode == 'tr':
                    anchor_x, anchor_y = self.rect_start_drag[0], self.rect_start_drag[1] + self.rect_start_drag[3] # Bottom-Left
                    # Growing Right, Up
                    # Bounds: Right <= 1, Top >= 0
                    
                    curr_w = right - anchor_x
                    curr_h = anchor_y - top
                    
                    max_w = 1.0 - anchor_x
                    max_h = anchor_y
                    
                    # Clamp inputs
                    curr_w = max(self.min_crop_size, min(curr_w, max_w))
                    curr_h = max(self.min_crop_size, min(curr_h, max_h))
                    
                    if curr_w / curr_h > K:
                        curr_w = curr_h * K
                        if curr_w > max_w:
                            curr_w = max_w
                            curr_h = curr_w / K
                    else:
                        curr_h = curr_w / K
                        if curr_h > max_h:
                            curr_h = max_h
                            curr_w = curr_h * K
                            
                    left = anchor_x
                    top = anchor_y - curr_h
                    right = anchor_x + curr_w
                    bottom = anchor_y

                elif mode == 'bl':
                    anchor_x, anchor_y = self.rect_start_drag[0] + self.rect_start_drag[2], self.rect_start_drag[1] # Top-Right
                    # Growing Left, Down
                    # Bounds: Left >= 0, Bottom <= 1
                    
                    curr_w = anchor_x - left
                    curr_h = bottom - anchor_y
                    
                    max_w = anchor_x
                    max_h = 1.0 - anchor_y
                    
                    curr_w = max(self.min_crop_size, min(curr_w, max_w))
                    curr_h = max(self.min_crop_size, min(curr_h, max_h))
                    
                    if curr_w / curr_h > K:
                        curr_w = curr_h * K
                        if curr_w > max_w:
                            curr_w = max_w
                            curr_h = curr_w / K
                    else:
                        curr_h = curr_w / K
                        if curr_h > max_h:
                            curr_h = max_h
                            curr_w = curr_h * K
                            
                    left = anchor_x - curr_w
                    top = anchor_y
                    right = anchor_x
                    bottom = anchor_y + curr_h

                elif mode == 'br':
                    anchor_x, anchor_y = self.rect_start_drag[0], self.rect_start_drag[1] # Top-Left
                    # Growing Right, Down
                    # Bounds: Right <= 1, Bottom <= 1
                    
                    curr_w = right - anchor_x
                    curr_h = bottom - anchor_y
                    
                    max_w = 1.0 - anchor_x
                    max_h = 1.0 - anchor_y
                    
                    curr_w = max(self.min_crop_size, min(curr_w, max_w))
                    curr_h = max(self.min_crop_size, min(curr_h, max_h))
                    
                    if curr_w / curr_h > K:
                        curr_w = curr_h * K
                        if curr_w > max_w:
                            curr_w = max_w
                            curr_h = curr_w / K
                    else:
                        curr_h = curr_w / K
                        if curr_h > max_h:
                            curr_h = max_h
                            curr_w = curr_h * K
                            
                    left = anchor_x
                    top = anchor_y
                    right = anchor_x + curr_w
                    bottom = anchor_y + curr_h

                # Edge Dragging (Center Symmetrical Expansion usually, or fixed opposite edge)
                # Lightroom style: Side drag expands that side only, but forces other dim to match AR. 
                # Since other dim has 2 directions (e.g. dragging Right affects Height (Top/Bottom)),
                # we usually expand symmetrically around center for the non-dragged axis.
                
                elif 'l' in mode or 'r' in mode:
                    # Dragging Width. Height must adjust. 
                    # Center Y is fixed.
                    cy = self.rect_start_drag[1] + self.rect_start_drag[3]/2
                    
                    if 'l' in mode:
                        # Anchor Right
                        anchor_x = self.rect_start_drag[0] + self.rect_start_drag[2]
                        # Growing Left
                        width = anchor_x - left
                        max_w = anchor_x # To 0
                        width = max(self.min_crop_size, min(width, max_w))
                        
                        # Height is derived
                        height = width / K
                        
                        # Helper to check vertical bounds centered at cy
                        # Top = cy - h/2 >= 0  => h/2 <= cy => h <= 2*cy
                        # Bottom = cy + h/2 <= 1 => h/2 <= 1-cy => h <= 2*(1-cy)
                        max_h = 2 * min(cy, 1.0 - cy)
                        
                        if height > max_h:
                            height = max_h
                            width = height * K # Shrink width to fit height bounds
                        
                        left = anchor_x - width
                        right = anchor_x
                        top = cy - height/2
                        bottom = cy + height/2
                        
                    else: # 'r'
                        # Anchor Left
                        anchor_x = self.rect_start_drag[0]
                        # Growing Right
                        width = right - anchor_x
                        max_w = 1.0 - anchor_x
                        width = max(self.min_crop_size, min(width, max_w))
                        
                        height = width / K
                        max_h = 2 * min(cy, 1.0 - cy)
                        
                        if height > max_h:
                            height = max_h
                            width = height * K
                            
                        left = anchor_x
                        right = anchor_x + width
                        top = cy - height/2
                        bottom = cy + height/2

                elif 't' in mode or 'b' in mode:
                    # Dragging Height. Width must adjust symmetrically around Center X.
                    cx = self.rect_start_drag[0] + self.rect_start_drag[2]/2
                    
                    if 't' in mode:
                        # Anchor Bottom
                        anchor_y = self.rect_start_drag[1] + self.rect_start_drag[3]
                        # Growing Up
                        height = anchor_y - top
                        max_h = anchor_y
                        height = max(self.min_crop_size, min(height, max_h))
                        
                        # Width derived
                        width = height * K
                        
                        # Width bounds centered at cx:
                        # Left >= 0, Right <= 1
                        max_w = 2 * min(cx, 1.0 - cx)
                        
                        if width > max_w:
                            width = max_w
                            height = width / K
                            
                        top = anchor_y - height
                        bottom = anchor_y
                        left = cx - width/2
                        right = cx + width/2
                        
                    else: # 'b'
                        # Anchor Top
                        anchor_y = self.rect_start_drag[1]
                        # Growing Down
                        height = bottom - anchor_y
                        max_h = 1.0 - anchor_y
                        height = max(self.min_crop_size, min(height, max_h))
                        
                        width = height * K
                        max_w = 2 * min(cx, 1.0 - cx)
                        
                        if width > max_w:
                            width = max_w
                            height = width / K
                            
                        top = anchor_y
                        bottom = anchor_y + height
                        left = cx - width/2
                        right = cx + width/2
            
            else:
                # FREE MODE (Just clamp to 0..1)
                left = max(0, min(1, left))
                top = max(0, min(1, top))
                right = max(0, min(1, right))
                bottom = max(0, min(1, bottom))
            
            new_x, new_y = left, top
            new_w, new_h = right - left, bottom - top
            
        self.crop_rect = (new_x, new_y, new_w, new_h)
        self.viewport.update()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.drag_mode = None

    def _paint_viewport(self):
        if not self.display_pixmap: return
        
        painter = QPainter(self.viewport)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # 1. Draw Image (Scaled to fit)
        view_w = self.viewport.width()
        view_h = self.viewport.height()
        img_w = self.display_pixmap.width()
        img_h = self.display_pixmap.height()
        
        # Scale to fit
        scale = min(view_w / img_w, view_h / img_h) * 0.9 # Margin
        
        draw_w = img_w * scale
        draw_h = img_h * scale
        
        dx = (view_w - draw_w) / 2
        dy = (view_h - draw_h) / 2
        
        self.display_rect = QRectF(dx, dy, draw_w, draw_h)
        
        painter.drawPixmap(self.display_rect.toRect(), self.display_pixmap)
        
        # 2. Draw Dim/Mask (Overlay)
        # Areas outside crop
        cx, cy, cw, ch = self.crop_rect
        
        # Screen coords of crop
        sx = dx + cx * draw_w
        sy = dy + cy * draw_h
        sw = cw * draw_w
        sh = ch * draw_h
        screen_crop = QRectF(sx, sy, sw, sh)
        
        # Darken outside
        mask = QColor(0, 0, 0, 150)
        painter.fillRect(0, 0, view_w, sy, mask) # Top
        painter.fillRect(0, sy + sh, view_w, view_h - (sy+sh), mask) # Bottom
        painter.fillRect(0, sy, sx, sh, mask) # Left
        painter.fillRect(sx + sw, sy, view_w - (sx+sw), sh, mask) # Right
        
        # 3. Draw Grid (Rule of Thirds)
        pen_grid = QPen(QColor(255, 255, 255, 100))
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)
        
        # Thirds
        painter.drawLine(sx + sw/3, sy, sx + sw/3, sy + sh)
        painter.drawLine(sx + 2*sw/3, sy, sx + 2*sw/3, sy + sh)
        painter.drawLine(sx, sy + sh/3, sx + sw, sy + sh/3)
        painter.drawLine(sx, sy + 2*sh/3, sx + sw, sy + 2*sh/3)
        
        # 4. Draw Border & Handles
        pen_border = QPen(QColor(255, 255, 255))
        pen_border.setWidth(2)
        painter.setPen(pen_border)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(screen_crop)
        
        # Handles (Corners)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        r = 4
        painter.drawEllipse(QPointF(sx, sy), r, r)
        painter.drawEllipse(QPointF(sx+sw, sy), r, r)
        painter.drawEllipse(QPointF(sx, sy+sh), r, r)
        painter.drawEllipse(QPointF(sx+sw, sy+sh), r, r)

    def _get_image_screen_rect(self):
        if not hasattr(self, 'display_rect'): return None
        return self.display_rect

    def reset_state(self):
        self.rotation = 0
        self.flip_h = False
        self.flip_v = False
        self.crop_rect = (0.0, 0.0, 1.0, 1.0)
        self.rotate_slider.setValue(0)
        self._update_display_pixmap()
        self.viewport.update()
        
    def _on_done(self):
        self.applied.emit(self.rotation, self.flip_h, self.flip_v, self.crop_rect)

def from_scale(sx, sy):
    """Helper for QTransform scale since API varies"""
    from PySide6.QtGui import QTransform
    t = QTransform()
    t.scale(sx, sy)
    return t
