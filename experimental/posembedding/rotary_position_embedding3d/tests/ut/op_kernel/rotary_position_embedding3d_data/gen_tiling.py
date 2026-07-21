#!/usr/bin/python3
import sys, math
import numpy as np

TILE_NUM = 1
BLOCK_DIM = 8
FREQ_BASE = 10000.0

def factor_dims(seq_len):
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

case_configs = {
    "case_2_4_8":    (2, 4, 8),
    "case_4_16_64":  (4, 16, 64),
    "case_1_32_128": (1, 32, 128),
}

def gen_tiling(b, s, d):
    total = b * s * d
    block_len = total // BLOCK_DIM
    tb, hb, wb = compute_band_dims(d)
    T, H, W = factor_dims(s)
    rt = FREQ_BASE ** (-2.0 / tb)
    rh = FREQ_BASE ** (-2.0 / hb)
    rw = FREQ_BASE ** (-2.0 / wb)

    # Struct layout: 11 int64 + 4 float = 88 + 16 = 104 bytes
    params = np.array([
        total,      # totalLength
        d,          # headDim
        s,          # seqLen
        T,          # T (frames)
        H,          # H (height tokens)
        W,          # W (width tokens)
        tb,         # tBand
        hb,         # hBand
        wb,         # wBand
        TILE_NUM,   # tileNum
        block_len,  # blockLength
    ], dtype=np.int64)

    floats = np.array([
        FREQ_BASE,  # freqBase
        rt,         # rT
        rh,         # rH
        rw,         # rW
    ], dtype=np.float32)

    with open("tiling.bin", "wb") as f:
        params.tofile(f)
        floats.tofile(f)
    print(f"Tiling: B={b} S={s} D={d} total={total}")
    print(f"  Video: T={T} H={H} W={W}")
    print(f"  Bands: tBand={tb} hBand={hb} wBand={wb}")
    print(f"  Freq: rt={rt:.6f} rh={rh:.6f} rw={rw:.6f}")

def main():
    case_name = sys.argv[1]
    if case_name not in case_configs:
        print(f"Unknown case: {case_name}, available: {list(case_configs.keys())}")
        sys.exit(1)
    b, s, d = case_configs[case_name]
    gen_tiling(b, s, d)

if __name__ == "__main__":
    main()
