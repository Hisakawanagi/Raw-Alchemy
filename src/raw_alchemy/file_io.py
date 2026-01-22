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
    exif_img: Optional[pyexiv2.Image] = None
) -> bool:
    """
    保存图像到指定路径，根据扩展名自动选择格式
    
    Args:
        img: 图像数据 (float32, 0.0-1.0)
        output_path: 输出路径
        logger: 日志处理器
        exif_img: pyexiv2 图像对象，用于复制 EXIF 数据
    
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
            _write_exif(output_path, exif_img, logger)
        
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
    """将 EXIF 数据写入输出文件"""
    try:
        # 读取源文件的 EXIF 数据
        exif_data = exif_img.read_exif()
        iptc_data = exif_img.read_iptc()
        xmp_data = exif_img.read_xmp()
        
        # 打开输出文件并写入 EXIF 数据
        output_img = pyexiv2.Image(output_path)
        
        # 写入 EXIF 数据
        if exif_data:
            output_img.modify_exif(exif_data)
        
        # 写入 IPTC 数据
        if iptc_data:
            output_img.modify_iptc(iptc_data)
        
        # 写入 XMP 数据
        if xmp_data:
            output_img.modify_xmp(xmp_data)
        
        output_img.close()
        logger.info("    ✅ EXIF data written successfully")
        
    except Exception as e:
        logger.warning(f"    ⚠️  Failed to write EXIF data: {e}")
