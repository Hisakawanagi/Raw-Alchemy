from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from qfluentwidgets import CaptionLabel
import os

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
