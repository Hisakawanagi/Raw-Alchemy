import os
import sys
from typing import Optional, Tuple, Dict
import rawpy
import numpy as np
from loguru import logger
from raw_alchemy import lensfun_wrapper as lf
import pyexiv2
from raw_alchemy.math_ops import (
    apply_matrix_inplace,
    apply_lut_inplace,
    apply_saturation_contrast_inplace,
    apply_white_balance_inplace,
    apply_highlight_shadow_inplace,
    apply_gain_inplace,
    linear_to_srgb_inplace,
    bt709_to_srgb_inplace,
    compute_histogram_channel,
    compute_waveform_channel
)
from scipy import ndimage

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
    try:
        # 数据验证
        if img_array is None or img_array.size == 0:
            return None
        
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return None
        
        # 子采样 - 使用copy()创建独立副本，避免数据竞争
        sample = img_array[::sample_rate, ::sample_rate, :].copy()
        
        # 确保数据类型正确
        if sample.dtype != np.float32:
            sample = sample.astype(np.float32)
        
        # 确保数据在有效范围内
        sample = np.clip(sample, 0.0, 1.0)
        
        hist_data = []
        for channel in range(3):
            # 展平通道数据 - 使用copy()确保连续内存
            channel_data = sample[:, :, channel].ravel().copy()
            
            # 确保是C连续数组
            if not channel_data.flags['C_CONTIGUOUS']:
                channel_data = np.ascontiguousarray(channel_data)
            
            # 使用 numba 加速的直方图计算
            hist = compute_histogram_channel(channel_data, bins, 0.0, 1.0)
            # 转换为浮点数以便绘制
            hist_data.append(hist.astype(np.float64))
        
        return hist_data
    except Exception as e:
        # 记录错误但不抛出异常
        logger.warning(f"Histogram computation failed: {type(e).__name__}: {e}")
        return None


def compute_waveform_fast(img_array, bins=100, sample_rate=4):
    """
    快速计算亮度波形图数据（使用 numba 加速）
    类似达芬奇的波形图，显示图像的亮度分布
    
    Args:
        img_array: HxWx3 numpy array with float values in range [0, 1]
        bins: number of vertical bins (亮度级别)
        sample_rate: horizontal subsample rate (水平采样率)
    
    Returns:
        numpy array of shape [sampled_width, bins] - 亮度波形数据
    """
    try:
        # 数据验证
        if img_array is None or img_array.size == 0:
            return None
        
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return None
        
        h, w, c = img_array.shape
        
        # 水平方向采样以提高性能
        sampled_width = w // sample_rate
        if sampled_width == 0:
            sampled_width = 1
        
        # 创建数据副本以避免竞争条件
        img_copy = img_array.copy()
        
        # 确保数据类型正确
        if img_copy.dtype != np.float32:
            img_copy = img_copy.astype(np.float32)
        
        # 确保数据在有效范围内
        img_copy = np.clip(img_copy, 0.0, 1.0)
        
        # 计算亮度（使用 Rec.709 系数）
        # Y = 0.2126*R + 0.7152*G + 0.0722*B
        luma = (img_copy[:, :, 0] * 0.2126 +
                img_copy[:, :, 1] * 0.7152 +
                img_copy[:, :, 2] * 0.0722).astype(np.float32)
        
        # 确保是C连续数组
        if not luma.flags['C_CONTIGUOUS']:
            luma = np.ascontiguousarray(luma)
        
        # 创建波形输出数组
        waveform = np.zeros((sampled_width, bins), dtype=np.float32)
        
        # 确保输出数组也是C连续的
        if not waveform.flags['C_CONTIGUOUS']:
            waveform = np.ascontiguousarray(waveform)
        
        # 使用numba加速的计算函数
        compute_waveform_channel(luma, waveform, bins, sample_rate)
        
        # 归一化
        max_val = np.max(waveform)
        if max_val > 0:
            waveform = waveform / max_val
        
        return waveform
    except Exception as e:
        # 记录错误但不抛出异常
        logger.warning(f"Waveform computation failed: {type(e).__name__}: {e}")
        return None

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

def extract_lens_exif(raw_path: str, raw) -> Tuple[dict, Optional[Dict[str, dict]]]:
    """
    使用 pyexiv2 从 RAW 文件中提取 EXIF 和镜头信息。
    Returns:
        Tuple[dict, Optional[Dict[str, dict]]]: (镜头校正参数, 完整的元数据字典{'exif', 'iptc', 'xmp'})
    """
    result = {}
    metadata = None
    pyexiv2_failed = False
    
    try:
        # 使用 pyexiv2 读取 EXIF 数据
        # 使用 verify_supported=False 防止某些 raw 格式检查报错
        # 使用 ignore_xmp_decoding_errors 防止 XMP 解析错误
        with pyexiv2.Image(raw_path) as exif_img:
            exif_data = exif_img.read_exif() or {}
            iptc_data = exif_img.read_iptc() or {}
            xmp_data = exif_img.read_xmp() or {}
            
            metadata = {
                'exif': exif_data,
                'iptc': iptc_data,
                'xmp': xmp_data
            }
        
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
                     # 尝试直接解析为浮点数
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
                    # 尝试直接解析为浮点数
                    result['aperture'] = float(aperture_str)
            except (ValueError, ZeroDivisionError):
                pass
        
        # ISO 感光度
        result['iso'] = (
            exif_data.get('Exif.Photo.ISOSpeedRatings') or 
            exif_data.get('Exif.Photo.ISOSpeed') or 
            ''
        )
        
        # 快门速度 / 曝光时间 (保留原始字符串，如 "1/100")
        result['exposure_time'] = exif_data.get('Exif.Photo.ExposureTime', '')
        
        # 拍摄时间
        result['datetime'] = (
            exif_data.get('Exif.Photo.DateTimeOriginal') or 
            exif_data.get('Exif.Image.DateTime') or 
            ''
        )
                
    except Exception as e:
        error_msg = str(e)
        pyexiv2_failed = True
        
        # Sony2 目录错误是已知的 exiv2 库限制，不影响其他 EXIF 数据读取
        logger.error(f"  ❌ [EXIF Error] {error_msg}")
        logger.info("  ℹ️  Trying to extract basic info from rawpy...")
    
    # 如果 pyexiv2 失败或数据不完整，尝试从 rawpy 获取基本信息
    if pyexiv2_failed:
        try:
            # 使用新的 rawpy 参数对象 (rawpy >= 0.20.0)
            result['camera_maker'] = raw.camera_params.make
            result['camera_model'] = raw.camera_params.model
            result['lens_maker'] = raw.lens_params.make
            result['lens_model'] = raw.lens_params.model
            result['focal_length'] = raw.other_params.focal_len
            result['aperture'] = raw.other_params.aperture
            result['iso'] = raw.other_params.iso_speed
            result['exposure_time'] = raw.other_params.shutter # float seconds
            
            import time
            if raw.other_params.timestamp > 0:
                result['datetime'] = time.strftime('%Y:%m:%d %H:%M:%S', time.localtime(raw.other_params.timestamp))
            
            # 如果 pyexiv2 失败，metadata 为 None，这里可以考虑不构造或者提供简单的 None
        except Exception as e:
            # logger(f"  ❌ [EXIF Error] {e}") # logger is not available globally in utils mostly? 
            # actually logger is imported in utils.py
            print(f"  ❌ [EXIF Error (Fallback)] {e}")

    # 过滤掉空值
    result = {k: v for k, v in result.items() if v}
    
    return result, metadata

def get_version_info():
    """Get version and license information"""
    try:
        from raw_alchemy import __version__
        version = __version__
    except ImportError:
        version = "0.0.0"
    
    current_year = "2025"
    license_info = f"Copyright © {current_year} MinQ.\nAGPL-V3 License."
    return version, license_info

def apply_geometry(img: np.ndarray, rotation: int = 0, flip_h: bool = False, flip_v: bool = False) -> np.ndarray:
    """
    应用几何变换（旋转和翻转）
    rotation: degrees clockwise. If divisible by 90, uses fast numpy.rot90, 
              otherwise uses higher quality interpolation.
    """
    if rotation == 0 and not flip_h and not flip_v:
        return img
    
    out = img
    
    # Rotation
    # Normalize rotation to [0, 360)
    rotation = rotation % 360
    
    # 1. Fast 90-degree steps rotation
    if rotation % 90 == 0:
        k = 0
        if rotation == 90:
            k = -1 # numpy rot90 is CCW
        elif rotation == 180:
            k = 2
        elif rotation == 270:
            k = 1
            
        if k != 0:
            out = np.rot90(out, k=k)
            
    # 2. Arbitrary angle rotation
    else:
        # ndimage.rotate uses CCW angle, so we use -rotation
        # reshape=True ensures the whole image is kept
        # order=3 (cubic) or order=1 (bilinear). Keep order=1 for speed in preview? 
        # Actually for quality we might want 3, but let's stick to default or 1 for responsiveness first.
        # User requested "specific angle", so let's allow arbitrary.
        # Note: This is computationally expensive!
        out = ndimage.rotate(out, -rotation, reshape=True, order=1, prefilter=False)
        
    if flip_h:
        out = np.fliplr(out)
        
    if flip_v:
        out = np.flipud(out)
        
    # Ensure contiguous
    if not out.flags['C_CONTIGUOUS']:
        out = np.ascontiguousarray(out)
        
    return out

