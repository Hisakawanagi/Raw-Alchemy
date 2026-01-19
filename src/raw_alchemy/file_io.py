"""
文件输入输出模块
处理各种格式的图像保存
"""
import os
import numpy as np
import tifffile
from PIL import Image
import pillow_heif
from typing import Optional, Dict, Any
from raw_alchemy.logger import Logger

def save_image(
    img: np.ndarray,
    output_path: str,
    logger: Optional[Logger] = None,
    exif_data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    保存图像到指定路径，根据扩展名自动选择格式
    
    Args:
        img: 图像数据 (float32, 0.0-1.0)
        output_path: 输出路径
        logger: 日志处理器
        exif_data:可选的EXIF数据字典
    
    Returns:
        bool: 是否保存成功
    """
    if logger is None:
        from .logger import create_logger
        logger = create_logger()
    
    # 确保数据在有效范围内
    np.clip(img, 0.0, 1.0, out=img)
    
    file_ext = os.path.splitext(output_path)[1].lower()
    
    try:
        if file_ext in ['.tif', '.tiff']:
            _save_tiff(img, output_path, logger, exif_data)
        elif file_ext in ['.heic', '.heif']:
            _save_heif(img, output_path, logger, exif_data)
        else:
            _save_jpeg_or_other(img, output_path, file_ext, logger, exif_data)
        
        logger.info(f"  ✅ Saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Failed to save file: {e}")
        import traceback
        traceback.print_exc()
        return False


def _save_tiff(img: np.ndarray, output_path: str, logger: Logger, exif_data: Optional[Dict[str, Any]] = None):
    """保存为 16-bit TIFF 格式并通过 pyexiv2写入EXIF信息"""
    logger.info("    Format: TIFF (16-bit, ZLIB Optimized)")
    output_image_uint16 = (img * 65535).astype(np.uint16)
    
    tifffile.imwrite(
        output_path,
        output_image_uint16,
        photometric='rgb',
        compression='zlib',
        predictor=2,    # 水平差分，提升压缩率
        compressionargs={'level': 8}    # 平衡速度和体积
    )
    
    # Add EXIF if data is provided
    if exif_data and len(exif_data) > 0:
        logger.info(f"    📍 Embedding EXIF data: {', '.join(exif_data.keys())}")
        _add_exif_with_pyexiv2(output_path, exif_data, logger)


def _add_exif_with_pyexiv2(tiff_path: str, exif_data: Dict[str, Any], logger: Logger):
    """Add EXIF data to TIFF file using pyexiv2"""
    try:
        import pyexiv2
        
        # Open image and read current EXIF
        image = pyexiv2.Image(tiff_path)
        
        # Build EXIF dictionary
        exif_dict = {}
        
        # 0th IFD / Image tags
        if 'camera_maker' in exif_data:
            exif_dict['Exif.Image.Make'] = exif_data['camera_maker']
        
        if 'camera_model' in exif_data:
            exif_dict['Exif.Image.Model'] = exif_data['camera_model']
        
        if 'timestamp' in exif_data:
            exif_dict['Exif.Image.DateTime'] = exif_data['timestamp']
            # Also add to Photo SubIFD as DateTimeOriginal and DateTimeDigitized
            exif_dict['Exif.Photo.DateTimeOriginal'] = exif_data['timestamp']
            exif_dict['Exif.Photo.DateTimeDigitized'] = exif_data['timestamp']
        
        # Photo tags (Exif SubIFD)
        if 'iso' in exif_data:
            exif_dict['Exif.Photo.ISOSpeedRatings'] = str(int(exif_data['iso']))
        
        if 'focal_length' in exif_data:
            try:
                focal = float(exif_data['focal_length'])
                exif_dict['Exif.Photo.FocalLength'] = f"{int(focal * 10)}/10"
            except (ValueError, TypeError):
                pass
        
        if 'shutter_speed' in exif_data:
            try:
                shutter = float(exif_data['shutter_speed'])
                # shutter_speed is stored as the fraction value (e.g., 0.025 = 1/40)
                # Convert to proper fraction notation "1/denominator"
                if shutter > 0:
                    exif_dict['Exif.Photo.ExposureTime'] = f"1/{int(1/shutter)}"
                else:
                    exif_dict['Exif.Photo.ExposureTime'] = f"{int(shutter)}/1"
            except (ValueError, TypeError, ZeroDivisionError):
                pass
        
        if 'aperture' in exif_data:
            try:
                aperture = float(exif_data['aperture'])
                # FNumber should be stored as a fraction
                exif_dict['Exif.Photo.FNumber'] = f"{int(aperture * 10)}/10"
            except (ValueError, TypeError):
                pass
        
        if 'lens_model' in exif_data:
            exif_dict['Exif.Photo.LensModel'] = exif_data['lens_model']
        
        if 'exposure_bias' in exif_data:
            try:
                bias = float(exif_data['exposure_bias'])
                exif_dict['Exif.Photo.ExposureBiasValue'] = f"{int(bias * 10)}/10"
            except (ValueError, TypeError):
                pass
        
        if 'metering_mode' in exif_data:
            try:
                metering = int(exif_data['metering_mode'])
                exif_dict['Exif.Photo.MeteringMode'] = str(metering)
            except (ValueError, TypeError):
                pass
        
        if 'flash' in exif_data:
            try:
                flash = int(exif_data['flash'])
                exif_dict['Exif.Photo.Flash'] = str(flash)
            except (ValueError, TypeError):
                pass
        
        if 'exposure_program' in exif_data:
            try:
                exposure_program = int(exif_data['exposure_program'])
                exif_dict['Exif.Photo.ExposureProgram'] = str(exposure_program)
            except (ValueError, TypeError):
                pass
        
        # Add software tag
        exif_dict['Exif.Image.Software'] = 'Raw Alchemy'
        
        # Write EXIF data
        image.modify_exif(exif_dict)
        image.close()
        
        logger.info(f"    ✅ EXIF data embedded successfully")
        
    except ImportError:
        logger.warning(f"    ⚠️  pyexiv2 not available, EXIF not embedded")
    except Exception as e:
        logger.warning(f"    ⚠️  Failed to add EXIF with pyexiv2: {e}")


def _create_exif_text_summary(exif_data: Dict[str, Any]) -> str:
    """
    创建 EXIF 数据的文本摘要，可以存储在 TIFF 的 ImageDescription 标签中。
    
    Args:
        exif_data: EXIF 数据字典
    
    Returns:
        EXIF 数据的文本表示
    """
    parts = []
    
    if 'camera_maker' in exif_data:
        parts.append(f"Make: {exif_data['camera_maker']}")
    
    if 'camera_model' in exif_data:
        parts.append(f"Model: {exif_data['camera_model']}")
    
    if 'timestamp' in exif_data:
        parts.append(f"DateTime: {exif_data['timestamp']}")
    
    if 'iso' in exif_data:
        parts.append(f"ISO: {exif_data['iso']}")
    
    if 'focal_length' in exif_data:
        parts.append(f"FocalLength: {exif_data['focal_length']}mm")
    
    if 'shutter_speed' in exif_data:
        shutter = exif_data['shutter_speed']
        parts.append(f"ExposureTime: 1/{int(shutter)}s")
    
    if 'aperture' in exif_data:
        parts.append(f"FNumber: f/{exif_data['aperture']}")
    
    if 'lens_model' in exif_data:
        parts.append(f"Lens: {exif_data['lens_model']}")
    
    return " | ".join(parts)


def _save_heif(img: np.ndarray, output_path: str, logger: Logger, exif_data: Optional[Dict[str, Any]] = None):
    """保存为 10-bit HEIF 格式"""
    logger.info("    Format: HEIF (10-bit, High Quality)")
    output_image_uint16 = (img * 65535).astype(np.uint16)
    
    heif_file = pillow_heif.from_bytes(
        mode='RGB;16',
        size=(output_image_uint16.shape[1], output_image_uint16.shape[0]),
        data=output_image_uint16.tobytes()
    )
    heif_file.save(output_path, quality=-1, bit_depth=10)


def _save_jpeg_or_other(img: np.ndarray, output_path: str, file_ext: str, logger: Logger, exif_data: Optional[Dict[str, Any]] = None):
    """保存为 8-bit JPEG 或其他格式"""
    logger.info(f"    Format: {file_ext.upper()} (8-bit High Quality)")
    
    # 转换为 8-bit（img 已经在 save_image 中被 clip 过了）
    output_image_uint8 = (img * 255).astype(np.uint8)
    
    # JPEG 特殊优化参数
    save_params = {}
    if file_ext in ['.jpg', '.jpeg']:
        save_params = {
            'quality': 95,
            'subsampling': 0,
            'optimize': True
        }
    
    Image.fromarray(output_image_uint8).save(output_path, **save_params)
