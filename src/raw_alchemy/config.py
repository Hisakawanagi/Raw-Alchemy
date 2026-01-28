"""[config.py](src/raw_alchemy/config.py)
Raw Alchemy 配置文件
包含 Log 空间映射、编码映射、测光模式定义和 GUI 配置
"""

# ==========================================
#           文件类型支持
# ==========================================

# 缩略图/扫描等功能支持的 RAW 扩展名（小写，包含点号）
SUPPORTED_RAW_EXTENSIONS = {
'.dng', '.cr2', '.cr3', '.nef', '.arw', '.rw2', '.raf', '.orf', '.pef', '.srw', '.x3f', 'fff', '3fr'
}

# ==========================================
#           核心处理配置
# ==========================================

# 映射：Log 空间名称 -> 对应的线性色域 (Linear Gamut)
LOG_TO_WORKING_SPACE = {
    'F-Log': 'F-Gamut',
    'F-Log2': 'F-Gamut',
    'F-Log2C': 'F-Gamut C',
    'V-Log': 'V-Gamut',
    'N-Log': 'N-Gamut',
    'L-Log': 'ITU-R BT.2020',
    'Canon Log 2': 'Cinema Gamut',
    'Canon Log 3': 'Cinema Gamut',
    'S-Log3': 'S-Gamut3',
    'S-Log3.Cine': 'S-Gamut3.Cine',
    'Arri LogC3': 'ARRI Wide Gamut 3',
    'Arri LogC4': 'ARRI Wide Gamut 4',
    'Log3G10': 'REDWideGamutRGB',
    'D-Log': 'DJI D-Gamut',
}

# 映射：复合名称 -> colour 库识别的 Log 编码函数名称
LOG_ENCODING_MAP = {
    'S-Log3.Cine': 'S-Log3',
    'F-Log2C': 'F-Log2',
}

# 测光模式选项
METERING_MODES = [
    'average',        # 几何平均 (默认)
    'center-weighted',# 中央重点
    'highlight-safe', # 高光保护 (ETTR)
    'hybrid',         # 混合 (平均 + 高光限制)
    'matrix',         # 矩阵/评价测光
]

# ==========================================
#           GUI 配置
# ==========================================

# GUI 窗口配置
GUI_WINDOW_WIDTH = 1000
GUI_WINDOW_HEIGHT = 950
GUI_WINDOW_TITLE = "Raw Alchemy"

# GUI 更新间隔（毫秒）
GUI_QUEUE_UPDATE_INTERVAL = 50
GUI_INITIAL_UPDATE_INTERVAL = 100

# 默认值
DEFAULT_CPU_THREADS = 4
DEFAULT_OUTPUT_FORMAT = 'tif'
DEFAULT_METERING_MODE = 'matrix'
DEFAULT_EXPOSURE_STOPS = 0.0
DEFAULT_LENS_CORRECTION = True

# 曝光调整范围
EXPOSURE_MIN = -5.0
EXPOSURE_MAX = 5.0

# 日志字体
LOG_FONT_FAMILY = "Consolas"
LOG_FONT_SIZE = 9

# 进度条配置
PROGRESS_BAR_LENGTH = 400
PROGRESS_LABEL_WIDTH = 16
