from PySide6.QtWidgets import QWidget, QVBoxLayout, QMessageBox
from PySide6.QtCore import Signal

from qfluentwidgets import (
    SubtitleLabel, StrongBodyLabel, BodyLabel,
    SimpleCardWidget, ScrollArea, ComboBox
)

from raw_alchemy import i18n
from raw_alchemy.i18n import tr


class SettingsPanel(QWidget):
    """Settings panel widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsWidget")
        self.setStyleSheet("#settingsWidget { background-color: transparent; }")
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI"""
        # Create scroll area
        settings_scroll = ScrollArea(self)
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        
        # Content widget
        settings_content = QWidget()
        settings_content.setObjectName("settingsContent")
        settings_content.setStyleSheet("#settingsContent { background-color: transparent; }")
        settings_layout = QVBoxLayout(settings_content)
        settings_layout.setContentsMargins(40, 40, 40, 40)
        settings_layout.setSpacing(20)
        
        # Title
        title = SubtitleLabel(tr('settings'))
        settings_layout.addWidget(title)
        
        # Language Settings Card
        lang_card = SimpleCardWidget()
        lang_layout = QVBoxLayout(lang_card)
        lang_layout.setSpacing(10)
        lang_title = StrongBodyLabel(tr('language'))
        self.lang_combo = ComboBox()
        self.lang_combo.addItems([tr('english'), tr('chinese')])
        current_lang = i18n.get_current_language()
        self.lang_combo.setCurrentIndex(1 if current_lang == 'zh' else 0)
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        lang_layout.addWidget(lang_title)
        lang_layout.addWidget(self.lang_combo)
        settings_layout.addWidget(lang_card)
        
        settings_layout.addStretch()
        
        # Set scroll area widget
        settings_scroll.setWidget(settings_content)
        
        # Set main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(settings_scroll)
    
    def on_language_changed(self, index):
        """Handle language change"""
        lang_code = 'en' if index == 0 else 'zh'
        current = i18n.get_current_language()
        if lang_code == current:
            return
        i18n.set_language(lang_code)
        QMessageBox.information(self, tr('restart_required'), tr('restart_message'))
