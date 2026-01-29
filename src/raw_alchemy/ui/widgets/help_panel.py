from PySide6.QtWidgets import QWidget, QVBoxLayout

from qfluentwidgets import (
    SubtitleLabel, StrongBodyLabel, BodyLabel,
    SimpleCardWidget, ScrollArea
)

from raw_alchemy.i18n import tr


class HelpPanel(QWidget):
    """Help panel widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("helpWidget")
        self.setStyleSheet("#helpWidget { background-color: transparent; }")
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI"""
        # Create scroll area
        help_scroll = ScrollArea(self)
        help_scroll.setWidgetResizable(True)
        help_scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        
        # Content widget
        help_content = QWidget()
        help_content.setObjectName("helpContent")
        help_content.setStyleSheet("#helpContent { background-color: transparent; }")
        help_layout = QVBoxLayout(help_content)
        help_layout.setContentsMargins(40, 40, 40, 40)
        help_layout.setSpacing(20)
        
        # Title
        help_layout.addWidget(SubtitleLabel(tr('help_title')))
        
        # Help sections
        help_sections = [
            (tr('help_overview'), tr('help_overview_text')),
            (tr('help_workflow'), tr('help_workflow_text')),
            (tr('help_shortcuts'), tr('help_shortcuts_text')),
        ]
        
        for title, text in help_sections:
            card = SimpleCardWidget()
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(10)
            card_layout.addWidget(StrongBodyLabel(title))
            
            text_label = BodyLabel(text)
            text_label.setWordWrap(True)
            card_layout.addWidget(text_label)
            
            help_layout.addWidget(card)
        
        help_layout.addStretch()
        
        # Set scroll area widget
        help_scroll.setWidget(help_content)
        
        # Set main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(help_scroll)
