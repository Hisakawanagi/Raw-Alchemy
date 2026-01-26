from numba.pycc import CC
from numba import prange
import numpy as np

cc = CC('math_ops_ext')
cc.verbose = True

# =========================================================
# Numba AOT Compilation Module
# =========================================================

# Note: parallel=True is not fully supported in AOT via pycc in all versions.
# We use prange, but it might run serially depending on the numba version and build config.

@cc.export('apply_matrix_inplace', 'void(float32[:,:,::1], float64[:,::1])')
def apply_matrix_inplace(img, matrix):
    # 获取图像总像素数
    rows, cols, channels = img.shape
    n_pixels = rows * cols
    
    # 创建 (N, 3) 的视图
    flat_img = img.reshape(n_pixels, channels)

    # 预加载矩阵参数
    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    for i in range(n_pixels):
        r = flat_img[i, 0]
        g = flat_img[i, 1]
        b = flat_img[i, 2]
        
        flat_img[i, 0] = r * m00 + g * m01 + b * m02
        flat_img[i, 1] = r * m10 + g * m11 + b * m12
        flat_img[i, 2] = r * m20 + g * m21 + b * m22

@cc.export('apply_lut_inplace', 'void(float32[:,:,::1], float32[:,:,:,::1], float64[:], float64[:])')
def apply_lut_inplace(img, lut_table, domain_min, domain_max):
    if img.ndim == 2:
        n_pixels = img.shape[0]
        flat_img = img.reshape(n_pixels, 1)
    else:
        rows, cols, _ = img.shape
        n_pixels = rows * cols
        flat_img = img.reshape(n_pixels, 3)

    size = lut_table.shape[0]
    size_minus_1 = size - 1
    size_float = float(size_minus_1)
    
    scale_r = size_minus_1 / (domain_max[0] - domain_min[0])
    scale_g = size_minus_1 / (domain_max[1] - domain_min[1])
    scale_b = size_minus_1 / (domain_max[2] - domain_min[2])
    
    min_r, min_g, min_b = domain_min[0], domain_min[1], domain_min[2]

    for i in range(n_pixels):
        in_r = flat_img[i, 0]
        in_g = flat_img[i, 1]
        in_b = flat_img[i, 2]

        raw_idx_r = (in_r - min_r) * scale_r
        raw_idx_g = (in_g - min_g) * scale_g
        raw_idx_b = (in_b - min_b) * scale_b

        idx_r = min(max(raw_idx_r, 0.0), size_float)
        idx_g = min(max(raw_idx_g, 0.0), size_float)
        idx_b = min(max(raw_idx_b, 0.0), size_float)

        x0 = int(idx_r)
        y0 = int(idx_g)
        z0 = int(idx_b)

        x1 = x0 + 1
        if x0 == size_minus_1: x1 = x0
        
        y1 = y0 + 1
        if y0 == size_minus_1: y1 = y0
        
        z1 = z0 + 1
        if z0 == size_minus_1: z1 = z0

        dx = idx_r - x0
        dy = idx_g - y0
        dz = idx_b - z0

        r_val = 0.0
        g_val = 0.0
        b_val = 0.0

        if dx >= dy:
            if dy >= dz:
                w0 = 1.0 - dx
                c_r = lut_table[x0, y0, z0, 0] * w0
                c_g = lut_table[x0, y0, z0, 1] * w0
                c_b = lut_table[x0, y0, z0, 2] * w0
                
                w1 = dx - dy
                c_r += lut_table[x1, y0, z0, 0] * w1
                c_g += lut_table[x1, y0, z0, 1] * w1
                c_b += lut_table[x1, y0, z0, 2] * w1
                
                w2 = dy - dz
                c_r += lut_table[x1, y1, z0, 0] * w2
                c_g += lut_table[x1, y1, z0, 1] * w2
                c_b += lut_table[x1, y1, z0, 2] * w2
                
                c_r += lut_table[x1, y1, z1, 0] * dz
                c_g += lut_table[x1, y1, z1, 1] * dz
                c_b += lut_table[x1, y1, z1, 2] * dz

                r_val, g_val, b_val = c_r, c_g, c_b

            elif dx >= dz:
                w0 = 1.0 - dx
                c_r = lut_table[x0, y0, z0, 0] * w0
                c_g = lut_table[x0, y0, z0, 1] * w0
                c_b = lut_table[x0, y0, z0, 2] * w0
                
                w1 = dx - dz
                c_r += lut_table[x1, y0, z0, 0] * w1
                c_g += lut_table[x1, y0, z0, 1] * w1
                c_b += lut_table[x1, y0, z0, 2] * w1
                
                w2 = dz - dy
                c_r += lut_table[x1, y0, z1, 0] * w2
                c_g += lut_table[x1, y0, z1, 1] * w2
                c_b += lut_table[x1, y0, z1, 2] * w2
                
                c_r += lut_table[x1, y1, z1, 0] * dy
                c_g += lut_table[x1, y1, z1, 1] * dy
                c_b += lut_table[x1, y1, z1, 2] * dy
                
                r_val, g_val, b_val = c_r, c_g, c_b
                
            else:
                w0 = 1.0 - dz
                c_r = lut_table[x0, y0, z0, 0] * w0
                c_g = lut_table[x0, y0, z0, 1] * w0
                c_b = lut_table[x0, y0, z0, 2] * w0
                
                w1 = dz - dx
                c_r += lut_table[x0, y0, z1, 0] * w1
                c_g += lut_table[x0, y0, z1, 1] * w1
                c_b += lut_table[x0, y0, z1, 2] * w1
                
                w2 = dx - dy
                c_r += lut_table[x1, y0, z1, 0] * w2
                c_g += lut_table[x1, y0, z1, 1] * w2
                c_b += lut_table[x1, y0, z1, 2] * w2
                
                c_r += lut_table[x1, y1, z1, 0] * dy
                c_g += lut_table[x1, y1, z1, 1] * dy
                c_b += lut_table[x1, y1, z1, 2] * dy

                r_val, g_val, b_val = c_r, c_g, c_b

        else: 
            if dz >= dy:
                w0 = 1.0 - dz
                c_r = lut_table[x0, y0, z0, 0] * w0
                c_g = lut_table[x0, y0, z0, 1] * w0
                c_b = lut_table[x0, y0, z0, 2] * w0
                
                w1 = dz - dy
                c_r += lut_table[x0, y0, z1, 0] * w1
                c_g += lut_table[x0, y0, z1, 1] * w1
                c_b += lut_table[x0, y0, z1, 2] * w1
                
                w2 = dy - dx
                c_r += lut_table[x0, y1, z1, 0] * w2
                c_g += lut_table[x0, y1, z1, 1] * w2
                c_b += lut_table[x0, y1, z1, 2] * w2
                
                c_r += lut_table[x1, y1, z1, 0] * dx
                c_g += lut_table[x1, y1, z1, 1] * dx
                c_b += lut_table[x1, y1, z1, 2] * dx
                
                r_val, g_val, b_val = c_r, c_g, c_b

            elif dz >= dx:
                w0 = 1.0 - dy
                c_r = lut_table[x0, y0, z0, 0] * w0
                c_g = lut_table[x0, y0, z0, 1] * w0
                c_b = lut_table[x0, y0, z0, 2] * w0
                
                w1 = dy - dz
                c_r += lut_table[x0, y1, z0, 0] * w1
                c_g += lut_table[x0, y1, z0, 1] * w1
                c_b += lut_table[x0, y1, z0, 2] * w1
                
                w2 = dz - dx
                c_r += lut_table[x0, y1, z1, 0] * w2
                c_g += lut_table[x0, y1, z1, 1] * w2
                c_b += lut_table[x0, y1, z1, 2] * w2
                
                c_r += lut_table[x1, y1, z1, 0] * dx
                c_g += lut_table[x1, y1, z1, 1] * dx
                c_b += lut_table[x1, y1, z1, 2] * dx
                
                r_val, g_val, b_val = c_r, c_g, c_b

            else:
                w0 = 1.0 - dy
                c_r = lut_table[x0, y0, z0, 0] * w0
                c_g = lut_table[x0, y0, z0, 1] * w0
                c_b = lut_table[x0, y0, z0, 2] * w0
                
                w1 = dy - dx
                c_r += lut_table[x0, y1, z0, 0] * w1
                c_g += lut_table[x0, y1, z0, 1] * w1
                c_b += lut_table[x0, y1, z0, 2] * w1
                
                w2 = dx - dz
                c_r += lut_table[x1, y1, z0, 0] * w2
                c_g += lut_table[x1, y1, z0, 1] * w2
                c_b += lut_table[x1, y1, z0, 2] * w2
                
                c_r += lut_table[x1, y1, z1, 0] * dz
                c_g += lut_table[x1, y1, z1, 1] * dz
                c_b += lut_table[x1, y1, z1, 2] * dz

                r_val, g_val, b_val = c_r, c_g, c_b

        flat_img[i, 0] = r_val
        flat_img[i, 1] = g_val
        flat_img[i, 2] = b_val

@cc.export('apply_saturation_contrast_inplace', 'void(float32[:,:,::1], float64, float64, float64, float32[::1])')
def apply_saturation_contrast_inplace(img, saturation, contrast, pivot, luma_coeffs):
    rows, cols, _ = img.shape
    cr, cg, cb = luma_coeffs[0], luma_coeffs[1], luma_coeffs[2]

    for r in range(rows):
        for c in range(cols):
            r_val = img[r, c, 0]
            g_val = img[r, c, 1]
            b_val = img[r, c, 2]

            lum = r_val * cr + g_val * cg + b_val * cb

            r_sat = lum + (r_val - lum) * saturation
            g_sat = lum + (g_val - lum) * saturation
            b_sat = lum + (b_val - lum) * saturation

            r_fin = (r_sat - pivot) * contrast + pivot
            g_fin = (g_sat - pivot) * contrast + pivot
            b_fin = (b_sat - pivot) * contrast + pivot

            if r_fin < 0.0: r_fin = 0.0
            if g_fin < 0.0: g_fin = 0.0
            if b_fin < 0.0: b_fin = 0.0

            img[r, c, 0] = r_fin
            img[r, c, 1] = g_fin
            img[r, c, 2] = b_fin

@cc.export('apply_white_balance_inplace', 'void(float32[:,:,::1], float64, float64, float64)')
def apply_white_balance_inplace(img, r_gain, g_gain, b_gain):
    rows, cols, _ = img.shape
    for r in range(rows):
        for c in range(cols):
            img[r, c, 0] *= r_gain
            img[r, c, 1] *= g_gain
            img[r, c, 2] *= b_gain

@cc.export('apply_highlight_shadow_inplace', 'void(float32[:,:,::1], float64, float64, float32[::1])')
def apply_highlight_shadow_inplace(img, highlight, shadow, luma_coeffs):
    rows, cols, _ = img.shape
    cr, cg, cb = luma_coeffs[0], luma_coeffs[1], luma_coeffs[2]

    for r in range(rows):
        for c in range(cols):
            r_v = img[r, c, 0]
            g_v = img[r, c, 1]
            b_v = img[r, c, 2]
            
            lum = r_v * cr + g_v * cg + b_v * cb
            
            if shadow != 0.0:
                mask = (1.0 - lum)
                if mask > 0:
                    factor = 1.0 + shadow * (mask * mask * mask)
                    r_v *= factor
                    g_v *= factor
                    b_v *= factor
            
            if highlight != 0.0:
                # 高光压缩应当在接近 1.0 的区域逐渐生效：
                # - highlight < 0: 压高光（将亮部拉回）
                # - highlight > 0: 推高光（可能导致溢出/后续剪裁）
                # 旧实现用 lum^3 作为 mask，会把暗部也一起整体乘因子，
                # 当 highlight 为负时会把暗部进一步压暗，形成“黑洞”。
                mask = lum
                if mask < 0.0:
                    mask = 0.0
                elif mask > 1.0:
                    mask = 1.0
                t = 1.0 - mask  # 离 1 越近，t 越小
                # 让高光调整集中在亮部：使用 (1-lum)^3 作为 roll-off
                factor = 1.0 + highlight * (1.0 - t * t * t)
                if factor < 0.0:
                    factor = 0.0
                r_v *= factor
                g_v *= factor
                b_v *= factor
            
            if r_v < 0.0: r_v = 0.0
            if g_v < 0.0: g_v = 0.0
            if b_v < 0.0: b_v = 0.0

            img[r, c, 0] = r_v
            img[r, c, 1] = g_v
            img[r, c, 2] = b_v

@cc.export('apply_gain_inplace', 'void(float32[:,:,::1], float64)')
def apply_gain_inplace(img, gain):
    rows, cols, _ = img.shape
    for r in range(rows):
        for c in range(cols):
            img[r, c, 0] *= gain
            img[r, c, 1] *= gain
            img[r, c, 2] *= gain

@cc.export('linear_to_srgb_inplace', 'void(float32[:,:,::1])')
def linear_to_srgb_inplace(img):
    rows, cols, _ = img.shape
    
    for r in range(rows):
        for c in range(cols):
            for ch in range(3):
                linear = img[r, c, ch]
                
                if linear <= 0.0031308:
                    result = linear * 12.92
                else:
                    result = 1.055 * (linear ** (1.0 / 2.4)) - 0.055
                
                img[r, c, ch] = result

@cc.export('bt709_to_srgb_inplace', 'void(float32[:,:,::1])')
def bt709_to_srgb_inplace(img):
    rows, cols, _ = img.shape
    
    for r in range(rows):
        for c in range(cols):
            for ch in range(3):
                val = img[r, c, ch]
                
                if val < 0.081:
                    linear = val / 4.5
                else:
                    linear = ((val + 0.099) / 1.099) ** (1.0 / 0.45)
                
                if linear <= 0.0031308:
                    result = linear * 12.92
                else:
                    result = 1.055 * (linear ** (1.0 / 2.4)) - 0.055
                
                img[r, c, ch] = result

@cc.export('compute_histogram_channel', 'int64[:](float32[::1], int64, float64, float64)')
def compute_histogram_channel(data, bins, min_val, max_val):
    hist = np.zeros(bins, dtype=np.int64)
    bin_width = (max_val - min_val) / bins
    
    # Remove prange to avoid threading conflicts with main processing loop
    for i in range(data.shape[0]):
        val = data[i]
        if min_val <= val <= max_val:
            bin_idx = int((val - min_val) / bin_width)
            if bin_idx >= bins:
                bin_idx = bins - 1
            hist[bin_idx] += 1
    
    return hist

@cc.export('compute_waveform_channel', 'void(float32[:,:], float32[:,:], int64, int64)')
def compute_waveform_channel(channel_data, waveform_out, bins, sample_rate):
    """
    计算单个通道的示波器数据 - 专业达芬奇风格
    支持 -4% 到 109% IRE 范围的显示
    
    Args:
        channel_data: HxW 的通道数据 (float32, 0-1范围)
        waveform_out: sampled_width x bins 的输出数组 (float32)
        bins: 垂直bins数量
        sample_rate: 水平采样率
    """
    h, w = channel_data.shape
    sampled_width = w // sample_rate
    if sampled_width == 0:
        sampled_width = 1
    
    # 专业IRE范围：-4% 到 109% (总共113%的范围)
    # 0-1的输入值映射到0-100% IRE
    # bins需要覆盖-4到109的范围
    ire_min = -4.0
    ire_max = 109.0
    ire_range = ire_max - ire_min  # 113
    
    bins_minus_1 = bins - 1
    
    for i in range(sampled_width):
        col_idx = i * sample_rate
        if col_idx >= w:
            break
        
        # 处理该列的所有像素
        for row in range(h):
            val = channel_data[row, col_idx]
            
            # 将0-1的值转换为0-100% IRE
            ire_value = val * 100.0
            
            # 映射到-4到109的范围内的bin索引
            # bin 0 对应 -4% IRE
            # bin (bins-1) 对应 109% IRE
            normalized_pos = (ire_value - ire_min) / ire_range
            bin_idx = int(normalized_pos * bins_minus_1)
            
            # 限制在有效范围内
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= bins:
                bin_idx = bins - 1
            
            # 累加到对应的bin
            waveform_out[i, bin_idx] += 1.0

if __name__ == "__main__":
    cc.compile()
