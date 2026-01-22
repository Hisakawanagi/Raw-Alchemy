"""
文件输入输出模块
处理各种格式的图像保存
"""
import os
import numpy as np
import tifffile
from PIL import Image
import pillow_heif
from typing import Optional
from raw_alchemy.logger import Logger
import pyexiv2

def save_image(
    img: np.ndarray,
    output_path: str,
    logger: Optional[Logger] = None,
    exif_img: Optional[pyexiv2.Image] = None,
    exif_dict: Optional[dict] = None
) -> bool:
    """
    保存图像到指定路径，根据扩展名自动选择格式
    
    Args:
        img: 图像数据 (float32, 0.0-1.0)
        output_path: 输出路径
        logger: 日志处理器
        exif_img: pyexiv2 图像对象，用于复制完整 EXIF 数据（优先使用）
        exif_dict: 从 rawpy 提取的 EXIF 数据字典，当 exif_img 不可用时使用
    
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
            _save_tiff(img, output_path, logger)
        elif file_ext in ['.heic', '.heif']:
            _save_heif(img, output_path, logger)
        else:
            _save_jpeg_or_other(img, output_path, file_ext, logger)
        
        # 写入 EXIF 数据
        if exif_img:
            # 优先使用完整的 EXIF 数据（从 pyexiv2.Image）
            _write_exif(output_path, exif_img, logger)
        elif exif_dict:
            # 降级方案：从 rawpy 数据构造基本 EXIF
            _write_exif_from_dict(output_path, exif_dict, logger)
        
        logger.info(f"  ✅ Saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Failed to save file: {e}")
        import traceback
        traceback.print_exc()
        return False


def _save_tiff(img: np.ndarray, output_path: str, logger: Logger):
    """保存为 16-bit TIFF 格式"""
    logger.info("    Format: TIFF (16-bit, ZLIB Optimized)")
    output_image_uint16 = (img * 65535).astype(np.uint16)
    
    tifffile.imwrite(
        output_path,
        output_image_uint16,
        photometric='rgb',
        compression='zlib',
        predictor=2,  # 水平差分，提升压缩率
        compressionargs={'level': 8}  # 平衡速度和体积
    )


def _save_heif(img: np.ndarray, output_path: str, logger: Logger):
    """保存为 10-bit HEIF 格式"""
    logger.info("    Format: HEIF (10-bit, High Quality)")
    output_image_uint16 = (img * 65535).astype(np.uint16)
    
    heif_file = pillow_heif.from_bytes(
        mode='RGB;16',
        size=(output_image_uint16.shape[1], output_image_uint16.shape[0]),
        data=output_image_uint16.tobytes()
    )
    heif_file.save(output_path, quality=85, bit_depth=10)


def _save_jpeg_or_other(img: np.ndarray, output_path: str, file_ext: str, logger: Logger):
    """保存为 8-bit JPEG 或其他格式"""
    logger.info(f"    Format: {file_ext.upper()} (8-bit High Quality)")
    
    # 转换为 8-bit（img 已经在 save_image 中被 clip 过了）
    output_image_uint8 = (img * 255).astype(np.uint8)
    
    # JPEG 特殊优化参数
    save_params = {}
    if file_ext in ['.jpg', '.jpeg']:
        save_params = {
            'quality': 90,
            'subsampling': 2,
            'optimize': True
        }
    
    Image.fromarray(output_image_uint8).save(output_path, **save_params)


def _write_exif(output_path: str, exif_img: pyexiv2.Image, logger: Logger):
    """将 EXIF 数据写入输出文件（排除旋转信息）"""
    try:
        # 读取源文件的 EXIF 数据
        exif_data = exif_img.read_exif()
        iptc_data = exif_img.read_iptc()
        xmp_data = exif_img.read_xmp()
        
        # 打开输出文件并写入 EXIF 数据
        output_img = pyexiv2.Image(output_path)
        
        # 写入 EXIF 数据，但排除旋转相关标签
        if exif_data:
            # 移除旋转相关的 EXIF 标签
            rotation_tags = ['Exif.Image.Orientation']
            for tag in rotation_tags:
                if tag in exif_data:
                    del exif_data[tag]
            
            output_img.modify_exif(exif_data)
        
        # 写入 IPTC 数据
        if iptc_data:
            output_img.modify_iptc(iptc_data)
        
        # 写入 XMP 数据
        if xmp_data:
            output_img.modify_xmp(xmp_data)
        
        output_img.close()
        logger.info("    ✅ EXIF data written successfully (rotation info excluded)")
        
    except Exception as e:
        logger.warning(f"    ⚠️  Failed to write EXIF data: {e}")


def _write_exif_from_dict(output_path: str, exif_dict: dict, logger: Logger):
    """
    从 rawpy 提取的数据字典构造并写入基本 EXIF 标签
    
    Args:
        output_path: 输出文件路径
        exif_dict: 包含相机和镜头信息的字典 (从 utils.extract_lens_exif 返回)
        logger: 日志处理器
    """
    try:
        # 打开输出文件
        output_img = pyexiv2.Image(output_path)
        
        # 构造基本的 EXIF 数据
        basic_exif = {}
        
        # 相机信息
        if exif_dict.get('camera_maker'):
            basic_exif['Exif.Image.Make'] = exif_dict['camera_maker']
        
        if exif_dict.get('camera_model'):
            basic_exif['Exif.Image.Model'] = exif_dict['camera_model']
        
        # 镜头信息
        if exif_dict.get('lens_model'):
            basic_exif['Exif.Photo.LensModel'] = exif_dict['lens_model']
        
        if exif_dict.get('lens_maker'):
            basic_exif['Exif.Photo.LensMake'] = exif_dict['lens_maker']
        
        # 拍摄参数
        if exif_dict.get('focal_length'):
            # 将浮点数转换为分数格式（例如 50.0 -> "50/1"）
            focal_mm = exif_dict['focal_length']
            basic_exif['Exif.Photo.FocalLength'] = f"{int(focal_mm * 10)}/10"
        
        if exif_dict.get('aperture'):
            # 光圈值（例如 2.8 -> "28/10"）
            aperture = exif_dict['aperture']
            basic_exif['Exif.Photo.FNumber'] = f"{int(aperture * 10)}/10"
        
        # 写入 EXIF 数据
        if basic_exif:
            output_img.modify_exif(basic_exif)
            output_img.close()
            logger.info(f"    ✅ Basic EXIF written from rawpy data ({len(basic_exif)} tags)")
        else:
            output_img.close()
            logger.warning("    ⚠️  No valid EXIF data to write from rawpy")
        
    except Exception as e:
        logger.warning(f"    ⚠️  Failed to write EXIF from dict: {e}")
