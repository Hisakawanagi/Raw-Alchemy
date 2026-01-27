from numba import njit, prange, float32, float64, int64, void
import numpy as np
from loguru import logger

# =========================================================
# JIT Compiled Kernels (Parallel, FastMath, Cached)
# =========================================================

# Helper for parallel configuration
JIT_CONFIG = {
    'nopython': True,
    'parallel': True,
    'fastmath': True,
    'cache': True,
    'nogil': False
}

@njit(**JIT_CONFIG)
def apply_matrix_inplace(img, matrix):
    # Ensure matrix is contiguous for better access pattern
    m = np.ascontiguousarray(matrix)
    rows = img.shape[0]
    cols = img.shape[1]
    
    m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
    m10, m11, m12 = m[1, 0], m[1, 1], m[1, 2]
    m20, m21, m22 = m[2, 0], m[2, 1], m[2, 2]

    # Parallel loop over rows
    for r in prange(rows):
        for c in range(cols):
            r_val = img[r, c, 0]
            g_val = img[r, c, 1]
            b_val = img[r, c, 2]
            
            img[r, c, 0] = r_val * m00 + g_val * m01 + b_val * m02
            img[r, c, 1] = r_val * m10 + g_val * m11 + b_val * m12
            img[r, c, 2] = r_val * m20 + g_val * m21 + b_val * m22

@njit(**JIT_CONFIG)
def apply_lut_inplace(img, lut_table, domain_min, domain_max):
    rows = img.shape[0]
    cols = img.shape[1]
    
    size = lut_table.shape[0]
    size_minus_1 = size - 1
    size_float = float(size_minus_1)
    
    scale_r = size_minus_1 / (domain_max[0] - domain_min[0])
    scale_g = size_minus_1 / (domain_max[1] - domain_min[1])
    scale_b = size_minus_1 / (domain_max[2] - domain_min[2])
    
    min_r, min_g, min_b = domain_min[0], domain_min[1], domain_min[2]

    for r in prange(rows):
        for c in range(cols):
            in_r = img[r, c, 0]
            in_g = img[r, c, 1]
            in_b = img[r, c, 2]

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

            # Tetrahedral interpolation
            if dx >= dy:
                if dy >= dz:
                    w0 = 1.0 - dx
                    c_0 = lut_table[x0, y0, z0, 0] * w0
                    c_1 = lut_table[x0, y0, z0, 1] * w0
                    c_2 = lut_table[x0, y0, z0, 2] * w0
                    
                    w1 = dx - dy
                    c_0 += lut_table[x1, y0, z0, 0] * w1
                    c_1 += lut_table[x1, y0, z0, 1] * w1
                    c_2 += lut_table[x1, y0, z0, 2] * w1
                    
                    w2 = dy - dz
                    c_0 += lut_table[x1, y1, z0, 0] * w2
                    c_1 += lut_table[x1, y1, z0, 1] * w2
                    c_2 += lut_table[x1, y1, z0, 2] * w2
                    
                    c_0 += lut_table[x1, y1, z1, 0] * dz
                    c_1 += lut_table[x1, y1, z1, 1] * dz
                    c_2 += lut_table[x1, y1, z1, 2] * dz

                elif dx >= dz:
                    w0 = 1.0 - dx
                    c_0 = lut_table[x0, y0, z0, 0] * w0
                    c_1 = lut_table[x0, y0, z0, 1] * w0
                    c_2 = lut_table[x0, y0, z0, 2] * w0
                    
                    w1 = dx - dz
                    c_0 += lut_table[x1, y0, z0, 0] * w1
                    c_1 += lut_table[x1, y0, z0, 1] * w1
                    c_2 += lut_table[x1, y0, z0, 2] * w1
                    
                    w2 = dz - dy
                    c_0 += lut_table[x1, y0, z1, 0] * w2
                    c_1 += lut_table[x1, y0, z1, 1] * w2
                    c_2 += lut_table[x1, y0, z1, 2] * w2
                    
                    c_0 += lut_table[x1, y1, z1, 0] * dy
                    c_1 += lut_table[x1, y1, z1, 1] * dy
                    c_2 += lut_table[x1, y1, z1, 2] * dy
                    
                else: # dz > dx >= dy
                    w0 = 1.0 - dz
                    c_0 = lut_table[x0, y0, z0, 0] * w0
                    c_1 = lut_table[x0, y0, z0, 1] * w0
                    c_2 = lut_table[x0, y0, z0, 2] * w0
                    
                    w1 = dz - dx
                    c_0 += lut_table[x0, y0, z1, 0] * w1
                    c_1 += lut_table[x0, y0, z1, 1] * w1
                    c_2 += lut_table[x0, y0, z1, 2] * w1
                    
                    w2 = dx - dy
                    c_0 += lut_table[x1, y0, z1, 0] * w2
                    c_1 += lut_table[x1, y0, z1, 1] * w2
                    c_2 += lut_table[x1, y0, z1, 2] * w2
                    
                    c_0 += lut_table[x1, y1, z1, 0] * dy
                    c_1 += lut_table[x1, y1, z1, 1] * dy
                    c_2 += lut_table[x1, y1, z1, 2] * dy

            else: # dy > dx
                if dz >= dy:
                    w0 = 1.0 - dz
                    c_0 = lut_table[x0, y0, z0, 0] * w0
                    c_1 = lut_table[x0, y0, z0, 1] * w0
                    c_2 = lut_table[x0, y0, z0, 2] * w0
                    
                    w1 = dz - dy
                    c_0 += lut_table[x0, y0, z1, 0] * w1
                    c_1 += lut_table[x0, y0, z1, 1] * w1
                    c_2 += lut_table[x0, y0, z1, 2] * w1
                    
                    w2 = dy - dx
                    c_0 += lut_table[x0, y1, z1, 0] * w2
                    c_1 += lut_table[x0, y1, z1, 1] * w2
                    c_2 += lut_table[x0, y1, z1, 2] * w2
                    
                    c_0 += lut_table[x1, y1, z1, 0] * dx
                    c_1 += lut_table[x1, y1, z1, 1] * dx
                    c_2 += lut_table[x1, y1, z1, 2] * dx

                elif dz >= dx:
                    w0 = 1.0 - dy
                    c_0 = lut_table[x0, y0, z0, 0] * w0
                    c_1 = lut_table[x0, y0, z0, 1] * w0
                    c_2 = lut_table[x0, y0, z0, 2] * w0
                    
                    w1 = dy - dz
                    c_0 += lut_table[x0, y1, z0, 0] * w1
                    c_1 += lut_table[x0, y1, z0, 1] * w1
                    c_2 += lut_table[x0, y1, z0, 2] * w1
                    
                    w2 = dz - dx
                    c_0 += lut_table[x0, y1, z1, 0] * w2
                    c_1 += lut_table[x0, y1, z1, 1] * w2
                    c_2 += lut_table[x0, y1, z1, 2] * w2
                    
                    c_0 += lut_table[x1, y1, z1, 0] * dx
                    c_1 += lut_table[x1, y1, z1, 1] * dx
                    c_2 += lut_table[x1, y1, z1, 2] * dx
                    
                else: # dy > dx > dz
                    w0 = 1.0 - dy
                    c_0 = lut_table[x0, y0, z0, 0] * w0
                    c_1 = lut_table[x0, y0, z0, 1] * w0
                    c_2 = lut_table[x0, y0, z0, 2] * w0
                    
                    w1 = dy - dx
                    c_0 += lut_table[x0, y1, z0, 0] * w1
                    c_1 += lut_table[x0, y1, z0, 1] * w1
                    c_2 += lut_table[x0, y1, z0, 2] * w1
                    
                    w2 = dx - dz
                    c_0 += lut_table[x1, y1, z0, 0] * w2
                    c_1 += lut_table[x1, y1, z0, 1] * w2
                    c_2 += lut_table[x1, y1, z0, 2] * w2
                    
                    c_0 += lut_table[x1, y1, z1, 0] * dz
                    c_1 += lut_table[x1, y1, z1, 1] * dz
                    c_2 += lut_table[x1, y1, z1, 2] * dz

            img[r, c, 0] = c_0
            img[r, c, 1] = c_1
            img[r, c, 2] = c_2

@njit(**JIT_CONFIG)
def apply_saturation_contrast_inplace(img, saturation, contrast, pivot, luma_coeffs):
    rows = img.shape[0]
    cols = img.shape[1]
    cr, cg, cb = luma_coeffs[0], luma_coeffs[1], luma_coeffs[2]

    for r in prange(rows):
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

            # Clamp negative values
            if r_fin < 0.0: r_fin = 0.0
            if g_fin < 0.0: g_fin = 0.0
            if b_fin < 0.0: b_fin = 0.0

            img[r, c, 0] = r_fin
            img[r, c, 1] = g_fin
            img[r, c, 2] = b_fin

@njit(**JIT_CONFIG)
def apply_white_balance_inplace(img, r_gain, g_gain, b_gain):
    rows = img.shape[0]
    cols = img.shape[1]
    # Simple multiplication is very vectorizable
    for r in prange(rows):
        for c in range(cols):
            img[r, c, 0] *= r_gain
            img[r, c, 1] *= g_gain
            img[r, c, 2] *= b_gain

@njit(**JIT_CONFIG)
def apply_highlight_shadow_inplace(img, highlight, shadow, luma_coeffs):
    rows = img.shape[0]
    cols = img.shape[1]
    cr, cg, cb = luma_coeffs[0], luma_coeffs[1], luma_coeffs[2]

    for r in prange(rows):
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
                mask = lum
                if mask < 0.0:
                    mask = 0.0
                elif mask > 1.0:
                    mask = 1.0
                t = 1.0 - mask
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

@njit(**JIT_CONFIG)
def apply_gain_inplace(img, gain):
    rows = img.shape[0]
    cols = img.shape[1]
    for r in prange(rows):
        for c in range(cols):
            img[r, c, 0] *= gain
            img[r, c, 1] *= gain
            img[r, c, 2] *= gain

@njit(**JIT_CONFIG)
def linear_to_srgb_inplace(img):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for r in prange(rows):
        for c in range(cols):
            for ch in range(3):
                linear = img[r, c, ch]
                
                if linear <= 0.0031308:
                    result = linear * 12.92
                else:
                    # Pow can be expensive, fastmath helps
                    result = 1.055 * (linear ** (1.0 / 2.4)) - 0.055
                
                img[r, c, ch] = result

@njit(**JIT_CONFIG)
def bt709_to_srgb_inplace(img):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for r in prange(rows):
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

@njit(**JIT_CONFIG)
def compute_histogram_channel(data, bins, min_val, max_val):
    hist = np.zeros(bins, dtype=np.int64)
    # Avoid div by zero
    if max_val <= min_val:
        return hist
        
    bin_width = (max_val - min_val) / bins
    n = data.shape[0]
    
    # Simple histogram logic. Parallelism for histogram reduction is complex in Numba
    # without race conditions. Given specific channel subsampling, this is fast enough.
    # We can stick to serial for correct reduction or use explicit atomic add if needed.
    # But for a subsampled array, serial is usually fine. Let's keep it simple JIT.
    for i in range(n):
        val = data[i]
        if min_val <= val <= max_val:
            bin_idx = int((val - min_val) / bin_width)
            if bin_idx >= bins:
                bin_idx = bins - 1
            hist[bin_idx] += 1
    
    return hist

@njit(**JIT_CONFIG)
def compute_waveform_channel(channel_data, waveform_out, bins, sample_rate):
    h = channel_data.shape[0]
    w = channel_data.shape[1]
    sampled_width = w // sample_rate
    if sampled_width == 0:
        sampled_width = 1
    
    ire_min = -4.0
    ire_max = 109.0
    ire_range = ire_max - ire_min
    bins_minus_1 = bins - 1
    
    # Parallel over columns (independent output memory locations)
    for i in prange(sampled_width):
        col_idx = i * sample_rate
        # if col_idx >= w: break  <-- Removed this check as it breaks parallelism and is redundant (loop bounds cover it)

        
        for row in range(h):
            val = channel_data[row, col_idx]
            
            ire_value = val * 100.0
            normalized_pos = (ire_value - ire_min) / ire_range
            bin_idx = int(normalized_pos * bins_minus_1)
            
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= bins:
                bin_idx = bins - 1
            
            waveform_out[i, bin_idx] += 1.0

def warmup():
    """Compiles all JIT functions with dummy data"""
    logger.info("  ♨️ Warming up JIT kernels...")
    
    # Small dummy image: 64x64
    dummy_img = np.zeros((64, 64, 3), dtype=np.float32)
    dummy_coeffs = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    
    # 1. Gain
    apply_gain_inplace(dummy_img, 2.0)
    
    # 2. White Balance
    apply_white_balance_inplace(dummy_img, 1.1, 1.0, 1.2)
    
    # 3. Saturation / Contrast
    apply_saturation_contrast_inplace(dummy_img, 1.2, 1.1, 0.18, dummy_coeffs)
    
    # 4. Highlight / Shadow
    apply_highlight_shadow_inplace(dummy_img, -20.0, 20.0, dummy_coeffs)
    
    # 5. LUT
    # 3D LUT: 4x4x4
    lut_size = 4
    lut_table = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float32)
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    # Prevent div by zero in kernel
    apply_lut_inplace(dummy_img, lut_table, domain_min, domain_max)
    
    # 6. Matrix
    matrix = np.eye(3, dtype=np.float64)
    apply_matrix_inplace(dummy_img, matrix)
    
    # 7. sRGB
    linear_to_srgb_inplace(dummy_img)
    bt709_to_srgb_inplace(dummy_img)

    # 8. Histogram
    flat_data = np.zeros(1000, dtype=np.float32)
    compute_histogram_channel(flat_data, 100, 0.0, 1.0)
    
    # 9. Waveform
    # Input HxW, Output W/sample x Bins
    wf_in = np.zeros((100, 100), dtype=np.float32)
    wf_out = np.zeros((25, 100), dtype=np.float32) # width=100, sample=4 -> 25
    compute_waveform_channel(wf_in, wf_out, 100, 4)
    
    logger.info("✅ JIT Warmup complete.")

if __name__ == '__main__':
    warmup()
