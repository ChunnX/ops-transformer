#!/usr/bin/python3
import sys, os, struct, math
import numpy as np

def factor_dims(seq_len):
    """Match FactorVideoDims from tiling.cpp"""
    if seq_len <= 1:
        return 1, 1, seq_len
    max_dim = min(128, int(math.isqrt(seq_len)))
    for w in range(max_dim, 0, -1):
        if seq_len % w == 0:
            rest = seq_len // w
            for h in range(min(128, int(math.isqrt(rest))), 0, -1):
                if rest % h == 0:
                    return rest // h, h, w
    return 1, 1, seq_len

def compute_band_dims(head_dim):
    """Match ComputeBandDims3D: 2:1:1 ratio, all even"""
    unit = head_dim // 4
    if unit % 2 != 0:
        unit -= 1
    tb = unit * 2
    hb = unit
    wb = head_dim - tb - hb
    if wb % 2 != 0:
        wb -= 1; tb += 1
        if tb % 2 != 0:
            tb -= 1; hb += 1
    if hb % 2 != 0:
        hb -= 1; tb += 1
    if tb % 2 != 0:
        tb -= 1; hb += 1
    return tb, hb, wb

def gen_data(b, s, d, dtype_str):
    shape = (int(b), int(s), int(d))
    half_d = d // 2
    freq_base = 10000.0

    # Match tiling's FactorVideoDims
    T, H, W = factor_dims(s)
    hw, wv = H * W, W
    tb, hb, wb = compute_band_dims(d)
    half_tb, half_hb, half_wb = tb // 2, hb // 2, wb // 2

    rt = freq_base ** (-2.0 / tb)
    rh = freq_base ** (-2.0 / hb)
    rw = freq_base ** (-2.0 / wb)

    # Theta tables
    theta_t = np.array([rt ** i for i in range(half_tb)], dtype=np.float64)
    theta_h = np.array([rh ** i for i in range(half_hb)], dtype=np.float64)
    theta_w = np.array([rw ** i for i in range(half_wb)], dtype=np.float64)

    # x1: random input
    rng = np.random.default_rng(42)
    data_x = rng.uniform(-3.0, 3.0, shape).astype(dtype_str)

    # x2: true 3D cos/sin with position decoding
    data_cs = np.zeros(shape, dtype=dtype_str)
    for batch in range(b):
        for pos in range(s):
            t_coord = pos // hw
            rem = pos % hw
            i_coord = rem // wv
            j_coord = rem % wv

            # T-band
            for k in range(half_tb):
                a = float(t_coord) * theta_t[k]
                data_cs[batch, pos, k] = math.cos(a)
                data_cs[batch, pos, half_d + k] = math.sin(a)
            # H-band
            for k in range(half_hb):
                a = float(i_coord) * theta_h[k]
                data_cs[batch, pos, half_tb + k] = math.cos(a)
                data_cs[batch, pos, half_d + half_tb + k] = math.sin(a)
            # W-band
            for k in range(half_wb):
                a = float(j_coord) * theta_w[k]
                data_cs[batch, pos, half_tb + half_hb + k] = math.cos(a)
                data_cs[batch, pos, half_d + half_tb + half_hb + k] = math.sin(a)

    # Golden: reference for validation
    data_golden = np.zeros(shape, dtype=np.float64)
    for batch in range(b):
        for pos in range(s):
            t_coord = pos // hw
            rem = pos % hw
            i_coord = rem // wv
            j_coord = rem % wv
            base_x = batch * s * d + pos * d
            # T-band
            for k in range(half_tb):
                a = float(t_coord) * theta_t[k]
                c, sn = math.cos(a), math.sin(a)
                xl = float(data_x.flat[base_x + k])
                xr = float(data_x.flat[base_x + half_d + k])
                data_golden.flat[base_x + k] = xl * c - xr * sn
                data_golden.flat[base_x + half_d + k] = xl * sn + xr * c
            # H-band
            for k in range(half_hb):
                a = float(i_coord) * theta_h[k]
                c, sn = math.cos(a), math.sin(a)
                off = half_tb + k
                xl = float(data_x.flat[base_x + off])
                xr = float(data_x.flat[base_x + half_d + off])
                data_golden.flat[base_x + off] = xl * c - xr * sn
                data_golden.flat[base_x + half_d + off] = xl * sn + xr * c
            # W-band
            for k in range(half_wb):
                a = float(j_coord) * theta_w[k]
                c, sn = math.cos(a), math.sin(a)
                off = half_tb + half_hb + k
                xl = float(data_x.flat[base_x + off])
                xr = float(data_x.flat[base_x + half_d + off])
                data_golden.flat[base_x + off] = xl * c - xr * sn
                data_golden.flat[base_x + half_d + off] = xl * sn + xr * c

    data_x.tofile("./x.bin")
    data_cs.tofile("./cos_sin.bin")
    data_golden.astype(dtype_str).tofile("./golden.bin")

if __name__ == "__main__":
    os.system("rm -rf ./*.bin")
    gen_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
