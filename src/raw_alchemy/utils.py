import os
import sys
from typing import Optional, Tuple
import rawpy
import numpy as np
from loguru import logger
from raw_alchemy import lensfun_wrapper as lf
import pyexiv2
try:
    from raw_alchemy.math_ops_ext import (
        apply_matrix_inplace,
        apply_lut_inplace,
        apply_saturation_contrast_inplace,
        apply_white_balance_inplace,
        apply_highlight_shadow_inplace,
        apply_gain_inplace,
        linear_to_srgb_inplace,
        bt709_to_srgb_inplace,
        compute_histogram_channel
    )
except ImportError:
    logger.error("Warning: AOT module 'math_ops_ext' not found. Please run 'python src/raw_alchemy/math_ops.py' to compile it.")
    raise


def resource_path(relative_path):
    """
    获取资源的绝对路径，兼容 Dev, PyInstaller, 和 Nuitka (Onefile & Standalone).
    """
    # 1. 处理 PyInstaller (它把资源解压到 _MEIPASS)
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    
    # 2. 处理 Nuitka 和 普通 Python 脚本
    # Nuitka 会巧妙地处理 __file__，使其指向解压后的临时目录(Onefile)或发布目录(Standalone)
    else:
        # 获取当前脚本所在的目录
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

# =========================================================
# Numba 加速核函数 (In-Place / 无内存分配)
# =========================================================


def compute_histogram_fast(img_array, bins=100, sample_rate=4):
    """
    快速计算 RGB 三通道直方图（使用 numba 加速）
    
    Args:
        img_array: HxWx3 numpy array with float values in range [0, 1]
        bins: number of histogram bins
        sample_rate: subsample rate (e.g., 4 means take every 4th pixel)
    
    Returns:
        list of 3 histogram arrays (R, G, B) as float arrays
    """
    # 子采样以提高速度
    sample = img_array[::sample_rate, ::sample_rate, :]
    
    hist_data = []
    for channel in range(3):
        # 展平通道数据
        channel_data = sample[:, :, channel].ravel()
        # 使用 numba 加速的直方图计算
        hist = compute_histogram_channel(channel_data, bins, 0.0, 1.0)
        # 转换为浮点数以便绘制
        hist_data.append(hist.astype(np.float64))
    
    return hist_data

# =========================================================
# 辅助计算函数 (用于测光)
# =========================================================

def get_luminance_coeffs(colourspace):
    """从 colour 空间对象中提取 RGB -> Y (Luminance) 的系数"""
    # RGB_to_XYZ 矩阵的第二行就是 Y 通道的系数 [Lr, Lg, Lb]
    return colourspace.matrix_RGB_to_XYZ[1, :]

def get_subsampled_view(img, target_size=1024):
    """
    获取图像的下采样视图。
    对于测光来说，分析 1000px 宽的缩略图和分析 8000px 的原图，结果差异可忽略不计。
    """
    h, w, _ = img.shape
    # 计算步长，使得长边大约为 target_size
    step = max(1, max(h, w) // target_size)
    # Numpy切片是视图(View)，不占用新内存
    return img[::step, ::step, :]

# =========================================================
# 业务逻辑函数 (优化版)
# =========================================================

def apply_saturation_and_contrast(img_linear, saturation=1.25, contrast=1.10, colourspace=None):
    """
    In-Place 应用饱和度和对比度。
    """
    import colour
    
    if colourspace is None:
        colourspace = colour.RGB_COLOURSPACES['ProPhoto RGB']
    
    luma_coeffs = get_luminance_coeffs(colourspace).astype(np.float32)
    
    if not img_linear.flags['C_CONTIGUOUS']:
        img_linear = np.ascontiguousarray(img_linear)
        
    apply_saturation_contrast_inplace(
        img_linear,
        float(saturation),
        float(contrast),
        0.18,
        luma_coeffs
    )
    return img_linear

def apply_white_balance(img_linear, temp=0.0, tint=0.0):
    """
    Apply White Balance.
    temp: -100 to 100 (Blue <-> Amber)
    tint: -100 to 100 (Green <-> Magenta)
    """
    # Simple gain calculation
    # Temp > 0: Warm (R+, B-)
    # Temp < 0: Cool (R-, B+)
    # Tint > 0: Magenta (G-)  -- Wait, usually tint + is magenta?
    # Let's define: Tint > 0 (Magenta/Purple), Tint < 0 (Green)
    # Standard: Tint slider usually goes Green (-) to Magenta (+)
    
    r_gain = 1.0
    g_gain = 1.0
    b_gain = 1.0
    
    # Temperature (strength factor 0.01 per unit)
    t_val = temp * 0.005 # Sensitivity
    r_gain += t_val
    b_gain -= t_val
    
    # Tint
    g_val = tint * 0.005
    g_gain -= g_val # Tint + (Magenta) means Green decreases
    
    if not img_linear.flags['C_CONTIGUOUS']:
        img_linear = np.ascontiguousarray(img_linear)
        
    apply_white_balance_inplace(img_linear, float(r_gain), float(g_gain), float(b_gain))
    return img_linear

def apply_highlight_shadow(img_linear, highlight=0.0, shadow=0.0, colourspace=None):
    """
    highlight: -100 to 100
    shadow: -100 to 100
    """
    import colour
    if colourspace is None:
        colourspace = colour.RGB_COLOURSPACES['ProPhoto RGB']
    luma_coeffs = get_luminance_coeffs(colourspace).astype(np.float32)

    # Normalize inputs to -1.0 to 1.0 roughly
    h_val = highlight / 100.0
    s_val = shadow / 100.0
    
    if not img_linear.flags['C_CONTIGUOUS']:
        img_linear = np.ascontiguousarray(img_linear)

    apply_highlight_shadow_inplace(img_linear, float(h_val), float(s_val), luma_coeffs)
    return img_linear

# ----------------- 镜头校正 (保持逻辑，优化注释) -----------------

def apply_lens_correction(image: np.ndarray, exif_data: dict, custom_db_path: Optional[str] = None, **kwargs) -> np.ndarray:
    """
    镜头校正通常需要几何变换，很难完全 In-Place。
    这是整个流程中少数几个必然会产生内存拷贝的地方。
    """
    # exif_data is now passed directly
    
    # 简单的字典合并
    params = {**exif_data, **kwargs}
    
    # 必要的 key 检查
    if not params.get('camera_model') or not params.get('lens_model'):
        logger.warning("  ⚠️  [Lens] Missing camera model info, skipping.")
        return image
    
    if not params.get('focal_length') or not params.get('aperture'):
        logger.warning("  ⚠️  [Lens] Missing optical info, skipping.")
        return image
    
    logger.info(f"  🧬 [Lens] {params.get('camera_maker')} {params.get('camera_model')} + {params.get('lens_model')}")
    
    try:
        # lensfun_wrapper 内部通常会调用 cv2.remap 或 scipy.map_coordinates
        # 这必然返回新图像
        corrected = lf.apply_lens_correction(
            image=image,
            camera_maker=params.get('camera_maker'),
            camera_model=params.get('camera_model'),
            lens_maker=params.get('lens_maker'),
            lens_model=params.get('lens_model'),
            focal_length=params.get('focal_length'),
            aperture=params.get('aperture'),
            crop_factor=params.get('crop_factor'),
            correct_distortion=params.get('correct_distortion', True),
            correct_tca=params.get('correct_tca', True),
            correct_vignetting=params.get('correct_vignetting', True),
            distance=params.get('distance', 1000.0),
            custom_db_path=custom_db_path,
        )
        
        # 显式帮助 GC (虽然 Python 会自动处理，但在大内存压力下 explicit is better)
        # 这里原来的 image 引用计数会减少，如果外面没有引用，旧内存会被释放
        return corrected
        
    except Exception as e:
        logger.error(f"  ❌ [Lens Error] {e}")
        return image # 失败则返回原图

def extract_lens_exif(raw_path: str) -> Tuple[dict, pyexiv2.Image]:
    """
    使用 pyexiv2 从 RAW 文件中提取 EXIF 和镜头信息。
    
    Args:
        raw_path: RAW 文件路径
        
    Returns:
        Tuple[dict, pyexiv2.Image]: (镜头校正所需的参数字典, pyexiv2 图像对象用于后续写入)
    """
    result = {}
    exif_img = None
    
    try:
        # 使用 pyexiv2 读取 EXIF 数据
        exif_img = pyexiv2.Image(raw_path)
        exif_data = exif_img.read_exif()
        
        # 提取镜头校正所需的信息
        # 相机制造商和型号
        result['camera_maker'] = exif_data.get('Exif.Image.Make', '').strip()
        result['camera_model'] = exif_data.get('Exif.Image.Model', '').strip()
        
        # 镜头信息 (不同厂商的标签可能不同)
        lens_model = (
            exif_data.get('Exif.Photo.LensModel') or
            exif_data.get('Exif.Canon.LensModel') or
            exif_data.get('Exif.Nikon3.Lens') or
            exif_data.get('Exif.Panasonic.LensType') or
            exif_data.get('Exif.OlympusEq.LensModel') or
            ''
        )
        result['lens_model'] = lens_model.strip() if lens_model else ''
        
        # 镜头制造商
        lens_maker = exif_data.get('Exif.Photo.LensMake', '').strip()
        result['lens_maker'] = lens_maker if lens_maker else ''
        
        # 焦距
        focal_length_str = exif_data.get('Exif.Photo.FocalLength', '')
        if focal_length_str:
            try:
                # 焦距通常是 "50/1" 这样的分数格式
                if '/' in str(focal_length_str):
                    num, denom = map(float, str(focal_length_str).split('/'))
                    result['focal_length'] = num / denom if denom != 0 else 0
                else:
                    result['focal_length'] = float(focal_length_str)
            except (ValueError, ZeroDivisionError):
                pass
        
        # 光圈
        aperture_str = exif_data.get('Exif.Photo.FNumber', '')
        if aperture_str:
            try:
                # 光圈通常是 "28/10" 这样的分数格式
                if '/' in str(aperture_str):
                    num, denom = map(float, str(aperture_str).split('/'))
                    result['aperture'] = num / denom if denom != 0 else 0
                else:
                    result['aperture'] = float(aperture_str)
            except (ValueError, ZeroDivisionError):
                pass
                
    except Exception as e:
        logger.error(f"  ❌ [EXIF Error] {e}")
        if exif_img:
            exif_img.close()
            exif_img = None
    
    # 过滤掉空值，防止下游出错
    result = {k: v for k, v in result.items() if v}
    
    return result, exif_img
