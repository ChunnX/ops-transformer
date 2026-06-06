"""
HiFloat8 向量化解码 (numpy), 与原始 generate_hifloat8_data.py 逐元素版本数值等价。
编码 (f32->hif8) 复用原始函数, 解码 (hif8->f32) 使用 numpy 向量化加速 ~5x.
"""
import numpy as np


def cvt_hif8_to_f32_vec(x_np, over_mode=True):
    """向量化 hif8 -> f32 解码, 与 cvt_hifuint8_to_float 逐元素等价"""
    x = x_np.ravel().astype(np.uint8)
    N = len(x)
    out = np.zeros(N, dtype=np.float32)

    # ─ 特殊值 ─
    if over_mode:
        m_zero, m_infp, m_infn, m_nan = (x == 0), (x == 111), (x == 239), (x == 128)
    else:
        m_zero, m_infp, m_infn, m_nan = (x == 0), (x == 110), (x == 238), (x == 128)
    out[m_zero] = np.float32(0.0)
    out[m_infp] = np.float32(np.inf)
    out[m_infn] = np.float32(-np.inf)
    out[m_nan]  = np.float32(np.nan) if over_mode else np.float32(0.0)
    m_special = m_zero | m_infp | m_infn | m_nan

    m_norm = ~m_special
    if not m_norm.any():
        return out.reshape(x_np.shape)

    xn = x[m_norm]
    sgn = np.where(xn >= 128, -1.0, 1.0).astype(np.float32)
    xn7 = xn & 0x7F
    dot = xn7 >> 3

    # 分桶: dot_prefix 区分子格式
    m_dml = (dot == 0)
    m_d0  = (dot == 1)
    m_d1  = (dot == 2) | (dot == 3)
    m_d2  = (dot >= 4) & (dot <= 7)
    m_d3  = (dot >= 8) & (dot <= 11)
    m_d4  = (dot >= 12) & (dot <= 15)

    exp_raw = np.zeros_like(xn, dtype=np.int32)
    frac_raw = np.zeros_like(xn, dtype=np.int32)
    exp_bw   = np.zeros_like(xn, dtype=np.int32)
    frac_bw  = np.zeros_like(xn, dtype=np.int32)
    exp_bias = np.zeros_like(xn, dtype=np.int32)

    # DML: exp[2:0] = bits[2:0], frac=none
    exp_raw[m_dml] = (xn7[m_dml] & 7).astype(np.int32)
    exp_bw[m_dml] = 3

    # D0: exp=0, frac[2:0]
    frac_raw[m_d0] = (xn7[m_d0] & 7).astype(np.int32)
    frac_bw[m_d0] = 3

    # D1: exp_sign=bit3, frac[2:0]
    exp_raw[m_d1]  = ((xn7[m_d1] >> 3) & 1).astype(np.int32)
    frac_raw[m_d1] = (xn7[m_d1] & 7).astype(np.int32)
    exp_bw[m_d1]   = 1
    exp_bias[m_d1] = 1
    frac_bw[m_d1]  = 3

    # D2: exp[1:0]=bits[4:3], frac[2:0]
    exp_raw[m_d2]  = ((xn7[m_d2] >> 3) & 3).astype(np.int32)
    frac_raw[m_d2] = (xn7[m_d2] & 7).astype(np.int32)
    exp_bw[m_d2]   = 2
    exp_bias[m_d2] = 2
    frac_bw[m_d2]  = 3

    # D3: exp[2:0]=bits[4:2], frac[1:0]=bits[1:0]
    exp_raw[m_d3]  = ((xn7[m_d3] >> 2) & 7).astype(np.int32)
    frac_raw[m_d3] = (xn7[m_d3] & 3).astype(np.int32)
    exp_bw[m_d3]   = 3
    exp_bias[m_d3] = 4
    frac_bw[m_d3]  = 2

    # D4: exp[3:0]=bits[4:1], frac=bit0
    exp_raw[m_d4]  = ((xn7[m_d4] >> 1) & 15).astype(np.int32)
    frac_raw[m_d4] = (xn7[m_d4] & 1).astype(np.int32)
    exp_bw[m_d4]   = 4
    exp_bias[m_d4] = 8
    frac_bw[m_d4]  = 1

    # 解码指数
    exp_val = np.where(m_dml,
                       exp_raw.astype(np.float32) - 23.0,
                       np.where(exp_raw < exp_bias,
                                (exp_raw + exp_bias).astype(np.float32),
                                (-exp_raw).astype(np.float32)))
    exp_val = np.where(exp_bw > 0, exp_val, np.float32(0.0))

    # 解码尾数
    frac_den = np.power(np.float32(2.0), frac_bw.astype(np.float32))
    mant_val = np.float32(1.0) + frac_raw.astype(np.float32) / np.where(frac_bw > 0, frac_den, np.float32(1.0))
    mant_val = np.where(frac_bw > 0, mant_val, np.float32(1.0))
    mant_val[m_dml] = np.float32(1.0)

    pr = np.power(np.float32(2.0), exp_val)
    out[m_norm] = (sgn * pr * mant_val).astype(np.float32)
    return out.reshape(x_np.shape)


def trans_np_hifuint8_tensor_to_float32_vec(in_tensor, over_mode=True):
    """向量化 hif8 -> f32 批量转换, 替代 trans_np_hifuint8_tensor_to_float32"""
    shape = in_tensor.shape
    return cvt_hif8_to_f32_vec(in_tensor, over_mode).reshape(shape).astype(np.float32)
