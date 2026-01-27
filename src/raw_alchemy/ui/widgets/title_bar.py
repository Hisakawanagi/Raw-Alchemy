from PySide6.QtWidgets import QWidget, QHBoxLayout
from PySide6.QtCore import Qt
from qfluentwidgets.window.fluent_window import FluentTitleBar

class CenteredFluentTitleBar(FluentTitleBar):
    """让 titleLabel/iconLabel 以“整个窗口宽度”绝对居中。

    说明：你选择了“按整窗宽度绝对居中，即使被右侧最小化/最大化/关闭遮挡也无所谓”。
    在这个目标下，不应该尝试在 hBoxLayout 里用 stretch 平衡（因为左右控件不对称必然偏）。

    最可靠的做法是：
    - 把 titleLabel/iconLabel 取出来放进一个 overlay 容器
    - overlay 容器覆盖整个标题栏区域
    - overlay 内用居中布局（或 setGeometry）保证绝对居中

    这里用一个透明的 overlay widget，并在 resizeEvent 时更新其几何。
    """

    def __init__(self, parent):
        super().__init__(parent)

        if getattr(self, "_center_overlay_installed", False):
            return
        self._center_overlay_installed = True

        # 创建 overlay 容器覆盖整个标题栏
        self._title_overlay = QWidget(self)
        self._title_overlay.setObjectName("titleOverlay")
        self._title_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        overlay_layout = QHBoxLayout(self._title_overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(6)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 把 icon/title 从原布局移除，放到 overlay 中
        self.hBoxLayout.removeWidget(self.titleLabel)
        self.hBoxLayout.removeWidget(self.iconLabel)
        overlay_layout.addWidget(self.iconLabel)
        overlay_layout.addWidget(self.titleLabel)

        # 原位置留空即可；如果你希望左侧仍有占位避免布局抖动，可选择性加一个 stretch
        # self.hBoxLayout.insertStretch(0, 1)

        # 初次同步 overlay 几何
        self._sync_title_overlay_geometry()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_title_overlay_geometry()

    def _sync_title_overlay_geometry(self):
        # 覆盖整个 titlebar 区域，实现“整窗绝对居中”
        self._title_overlay.setGeometry(self.rect())
