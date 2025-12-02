import rawpy
import numpy as np
import colour
import tifffile
from typing import Optional

def auto_expose_linear(img_linear: np.ndarray, source_colorspace: colour.RGB_Colourspace, target_gray: float = 0.18) -> np.ndarray:
    """
    自动计算曝光增益，将画面的“几何平均亮度”拉升到 target_gray (默认0.18)。
    这模拟了相机的自动测光。
    """
    # 1. 转换为亮度 (Luminance) 以便分析
    # 从源色彩空间转换到 CIE XYZ，然后取 Y 通道作为精确的亮度


    # # 注意：这里只是为了测光，不用太精确的色彩空间转换
    # luminance = (0.2126 * img_linear[:, :, 0] + 
    #              0.7152 * img_linear[:, :, 1] + 
    #              0.0722 * img_linear[:, :, 2])
    
    xyz_image = colour.RGB_to_XYZ(img_linear, source_colorspace)
    luminance = xyz_image[:, :, 1]
    
    # 2. 计算几何平均值 (Geometric Mean)
    # 使用几何平均值可以避免画面中极亮的高光点（如太阳）把整体曝光压得太低
    # 加一个极小值 1e-6 防止 log(0)
    avg_log_lum = np.mean(np.log(luminance + 1e-6))
    avg_lum = np.exp(avg_log_lum)
    
    # 3. 计算增益
    # 如果是一张该死的全黑图片，避免除以0
    if avg_lum < 0.0001: 
        gain = 1.0 
    else:
        gain = target_gray / avg_lum

    # 4. 限制增益范围（可选）
    # 防止对噪点图进行疯狂提亮，通常限制在 1.0 到 10.0 之间
    # 如果你的RAW普遍非常暗，可以把上限调高，比如 64.0 (相当于+6档快门)
    gain = np.clip(gain, 1.0, 50.0)
    
    print(f"  > Auto Exposure Gain: {gain:.4f} (Base Avg: {avg_lum:.5f})")
    
    return img_linear * gain

# 1. 映射：Log 空间名称 -> 对应的线性色域 (Linear Gamut)
LOG_TO_WORKING_SPACE = {
    'F-Log': 'F-Gamut',
    'F-Log2': 'F-Gamut',
    'F-Log2C': 'F-Gamut C',
    'V-Log': 'V-Gamut',
    'N-Log': 'N-Gamut',
    'Canon Log 2': 'Cinema Gamut',
    'Canon Log 3': 'Cinema Gamut',
    'S-Log3': 'S-Gamut3',
    'S-Log3.Cine': 'S-Gamut3.Cine',
    'LogC3': 'ARRI Wide Gamut 3',
    'LogC4': 'ARRI Wide Gamut 4',
    'Log3G10': 'RED Wide Gamut RGB',
}

# 2. 映射：复合名称 -> colour 库识别的 Log 编码函数名称
# 例如：S-Log3.Cine 使用的是 S-Gamut3.Cine 色域，但曲线依然是 S-Log3
LOG_ENCODING_MAP = {
    'S-Log3.Cine': 'S-Log3',
    'F-Log2C': 'F-Log2',
    # 其他名称如果跟 colour 库一致，可以在代码逻辑中直接 fallback
}

# 3. 映射：用户友好的 LUT 空间名 -> colour 库标准名称
LUT_SPACE_MAP = {
    "Rec.709": "ITU-R BT.709",
    "Rec.2020": "ITU-R BT.2020",
}

def process_image(
    raw_path: str,
    output_path: str,
    log_space: str,
    lut_path: Optional[str],
    lut_space: Optional[str],
    matrix_method: str = "adobe", # 推荐用 adobe
    exposure: Optional[float] = None,
):
    print(f"Processing: {raw_path}")

    with rawpy.imread(raw_path) as raw:
        
        print(f"Step 2: Decoding RAW (Adobe RGB Linear)...")
        # 1. 解码 RAW -> Linear Adobe RGB
        # bright=1.0 保持原始数据
        prophoto_linear = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.ProPhoto, 
            bright=1.0, 
            highlight_mode=2,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
        )
        prophoto_linear = prophoto_linear.astype(np.float32) / 65535.0
        source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
        img_exposed = auto_expose_linear(prophoto_linear, source_cs, target_gray=0.18)

    log_color_space_name = LOG_TO_WORKING_SPACE.get(log_space)
    log_curve_name = LOG_ENCODING_MAP.get(log_space, log_space)

    log_linear_image = colour.RGB_to_RGB(
        img_exposed,
        colour.RGB_COLOURSPACES['ProPhoto RGB'],
        colour.RGB_COLOURSPACES[log_color_space_name],
    )
    log_image = colour.cctf_encoding(log_linear_image, function=log_curve_name)

    image_to_save = log_image

    if lut_path and lut_space:
        print(f"Step 4: Applying LUT {lut_path}...")
        lut = colour.read_LUT(lut_path)
        image_after_lut = lut.apply(log_image)

        image_to_save = image_after_lut

    image_16bit = (image_to_save * 65535).astype(np.uint16)
    tifffile.imwrite(output_path, image_16bit)
    