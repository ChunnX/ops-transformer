#!/usr/bin/python3
# coding=utf-8

import math
import os
import sys

import numpy as np
import torch


def write_artifacts(base_dir, rank_id, a_fp8, b_fp8, a_scale, b_scale, out):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_fp8.tofile(os.path.join(input_dir, "input_a.bin"))
    # B must be in DNExtLayoutPtn (col-major): element(k,n) at k + n*K.
    # np default is C-order row-major; transpose+copy produces correct layout.
    np.ascontiguousarray(b_fp8.T).tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    # ScaleB must be in ScaleBDNLayoutPtn (col-major with N outermost).
    # offset(g,s,j) = g*2 + s + j*ceilK64*2.  numpy C-order [N,ceilK64,2] matches.
    np.ascontiguousarray(b_scale.transpose(2, 0, 1)).tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def float8_e4m3fn_to_float(data_uint8):
    """Convert FP8E4M3FN (uint8 representation) to FP32.
       NPU hardware uses bias=7 (not standard IEEE bias=8)."""
    data = data_uint8.astype(np.uint8)
    sign = (data >> 7) & 1
    exp = (data >> 3) & 15
    mant = data & 7

    fp32 = np.zeros_like(data, dtype=np.float32)

    # Zero case
    zero_mask = (exp == 0) & (mant == 0)
    fp32[zero_mask] = 0.0

    # Normal case: exp bias = 7 for AscendC NPU fp8_e4m3fn_t
    normal_mask = ~zero_mask
    fp32[normal_mask] = (1.0 - 2.0 * sign[normal_mask]) * \
                        np.power(2.0, exp[normal_mask] - 7.0) * \
                        (1.0 + mant[normal_mask] / 8.0)

    return fp32


def float8_e8m0_to_float(scale_uint8):
    """Convert FP8E8M0 (uint8 representation) to FP32 power of 2."""
    exp = scale_uint8.astype(np.float32)
    return np.power(2.0, exp - 127.0)


def float_to_fp8_e4m3fn_vec(fp32_arr):
    """Vectorized FP32→FP8 E4M3FN conversion.
       NPU hardware uses exp bias=7 (not standard IEEE bias=8)."""
    fp32 = fp32_arr.astype(np.float32)
    sign = (fp32 < 0).astype(np.uint8)
    abs_val = np.abs(fp32)

    MAX_NORMAL = 1.875 * 128.0   # 240.0: exp=14, bias=7 → 2^(14-7)*1.875
    MIN_NORMAL = 2.0 ** (-6)      # 0.015625: exp=1, bias=7 → 2^(1-7)

    log2_val = np.log2(np.clip(abs_val, 1e-30, None))
    e = np.floor(log2_val).astype(np.int32) + 7    # bias = 7 for AscendC NPU
    e = np.clip(e, 1, 14)

    normalized = abs_val / np.power(2.0, (e - 7).astype(np.float32))
    mant = np.clip(np.round((normalized - 1.0) * 8.0).astype(np.int32), 0, 7)

    overflow = abs_val >= MAX_NORMAL
    underflow = abs_val < MIN_NORMAL
    e[overflow] = 14
    mant[overflow] = 7
    e[underflow] = 0
    mant[underflow] = 0
    sign[underflow] = 0

    result = (sign << 7) | (e.astype(np.uint8) << 3) | mant.astype(np.uint8)
    result[fp32 == 0.0] = 0
    return result


def generate_fp8_e4m3fn_data(shape, min_val=1.0, max_val=8.0):
    """Generate random FP8E4M3FN data with proper quantization."""
    fp32_data = np.random.uniform(min_val, max_val, shape).astype(np.float32)
    fp8_data = float_to_fp8_e4m3fn_vec(fp32_data)
    return fp8_data, fp32_data


def generate_fp8_e8m0_scale(shape):
    """Generate FP8E8M0 scale values (exponent only, no mantissa)."""
    # Generate exponent values that produce scales in reasonable range
    # FP8E8M0: value = 2^(exp - 127)
    # For scale values around 1-4, use exp around 127-129
    exp_values = np.random.randint(127, 129, shape).astype(np.uint8)
    return exp_values


def dequantize_mxfp8(data_fp8, scale_fp8, divisor=64, c0=2):
    """
    MXFP8 dequantization: FP8 E4M3FN data × FP8 E8M0 scale → FP32.

    Scale layout per MXFP8 convention:
      A: scale shape [M, ceil(K/divisor), c0]  → last dim = c0
      B: scale shape [ceil(K/divisor), c0, N]  → mid dim = c0

    Each group of `divisor` elements shares `c0` scale values.
    Sub-group size = divisor / c0 elements per scale.
    """
    sub_group = divisor // c0
    fp32_data = float8_e4m3fn_to_float(data_fp8.astype(np.uint8))
    fp32_scale = float8_e8m0_to_float(scale_fp8.astype(np.uint8))

    # DEBUG: print raw uint8 and decoded float for first element
    raw0 = data_fp8.flat[0].astype(np.uint8)
    print(f"  [DIAG-deq] data.dtype={data_fp8.dtype}, raw[0]=0x{raw0:02x}={raw0}, fp32={fp32_data.flat[0]:.3f}, scale[0]={fp32_scale.flat[0]:.3f}")

    is_a_scale = (scale_fp8.ndim == 3 and scale_fp8.shape[-1] == c0 and scale_fp8.shape[-2] > c0)
    is_b_scale = (scale_fp8.ndim == 3 and scale_fp8.shape[1] == c0)

    k_shape = data_fp8.shape[1] if is_a_scale else data_fp8.shape[0]
    n_groups = (k_shape + divisor - 1) // divisor

    k_indices = np.arange(k_shape)
    group_idx = np.minimum(k_indices // divisor, n_groups - 1)
    sub_idx = np.clip((k_indices % divisor) // sub_group, 0, c0 - 1)

    scales = np.zeros(data_fp8.shape, dtype=np.float32)

    if is_a_scale:  # A: [M, n_groups, c0]
        M = data_fp8.shape[0]
        for i in range(M):
            for k in range(k_shape):
                scales[i, k] = fp32_scale[i, group_idx[k], sub_idx[k]]
    elif is_b_scale:  # B: [n_groups, c0, N]
        N = data_fp8.shape[1]
        for k in range(k_shape):
            scales[k, :] = fp32_scale[group_idx[k], sub_idx[k], :]
    else:
        raise ValueError(f"Unexpected scale shape: {scale_fp8.shape}")

    return fp32_data * scales


def gen_golden_data_matmul_allto_all(m, k, n, rank_num):
    """
    Generate golden data for MatmulAllToAll fusion operator.

    Each rank computes full C[M,N] = A[M,K] × B[K,N].
    M-direction tiling (4 tiles of tileM=512), N=full.

    CPU golden:
      1. C = A * B  → [M, N]
      2. permute: C.view(M, rankSize, N/rankSize).permute(1,0,2) → [rankSize, M, Np]
      3. AlltoAll exchange across ranks (each rank gets all other ranks' blocks)
    """
    M = m
    K = k
    N = n
    np_ = N // rank_num

    print(f"  M={M}, K={K}, N={N}, Np={np_} (per rank)")

    # Generate A: [M, K] — same for all ranks (or different per rank)
    a_fp8_list = []
    a_fp32_list = []
    a_scale_list = []

    for rank_id in range(rank_num):
        a_fp8, a_fp32 = generate_fp8_e4m3fn_data((M, K), 1.0, 8.0)
        a_scale = generate_fp8_e8m0_scale((M, math.ceil(K / 64), 2))
        if rank_id == 0:
            print(f"  [DIAG] A: fp8.shape={a_fp8.shape}, scale.shape={a_scale.shape}")
        a_fp8_list.append(a_fp8)
        a_fp32_list.append(a_fp32)
        a_scale_list.append(a_scale)

    # Generate B: [K, N] — full B per rank (complete weight matrix)
    b_fp8_list = []
    b_fp32_list = []
    b_scale_list = []

    for rank_id in range(rank_num):
        b_fp8, b_fp32 = generate_fp8_e4m3fn_data((K, N), 1.0, 8.0)
        b_scale = generate_fp8_e8m0_scale((math.ceil(K / 64), 2, N))
        if rank_id == 0:
            print(f"  [DIAG] B: fp8.shape={b_fp8.shape}, scale.shape={b_scale.shape}")
        b_fp8_list.append(b_fp8)
        b_fp32_list.append(b_fp32)
        b_scale_list.append(b_scale)

    # CPU Golden: MXFP8 dequantize → FP32 Matmul → permute → AlltoAll
    cpu_outputs_full = []
    for rank_id in range(rank_num):
        # Dequantize A: FP8E4M3FN × FP8E8M0 scale → FP32
        a_deq = dequantize_mxfp8(a_fp8_list[rank_id], a_scale_list[rank_id])
        b_deq = dequantize_mxfp8(b_fp8_list[rank_id], b_scale_list[rank_id])

        # DEBUG: verify dequantized shapes and sample values.
        # If b_deq.shape[0] != K (3584), the B matrix was generated with wrong dimensions.
        # NPU outputs 8064 (random scale) or 3584 (all-1 scale);
        # CPU outputs 2016 which is K/4 → suggests K is 896 not 3584.
        # Expected: a_deq=(M,K), b_deq=(K,N), K=3584, mean≈4.5 (data 1-8 × scale 1-2).
        if rank_id == 0:
            print(f"  [DIAG] a_deq.shape={a_deq.shape} (expect ({M},{K}))")
            print(f"  [DIAG] b_deq.shape={b_deq.shape} (expect ({K},{N}))")
            print(f"  [DIAG] a_deq[0,:5]={a_deq[0,:5]} (sample row 0)")
            print(f"  [DIAG] b_deq[:5,0]={b_deq[:5,0]} (sample col 0)")
            print(f"  [DIAG] a_deq.mean={a_deq.mean():.3f}, b_deq.mean={b_deq.mean():.3f}")
        a_cpu = torch.from_numpy(a_deq)
        b_cpu = torch.from_numpy(b_deq)
        c_full = torch.matmul(a_cpu, b_cpu)  # [M, N]
        cpu_outputs_full.append(c_full)
        print(f"  Rank {rank_id}: C shape {c_full.shape}, range [{c_full.min():.2f}, {c_full.max():.2f}]")

    # Step 2: permute — C.view(M, rankSize, Np).permute(1,0,2) → [rankSize, M, Np]
    allto_all_inputs = [c.view(M, rank_num, np_).permute(1, 0, 2) for c in cpu_outputs_full]

    # Step 3: AlltoAll exchange — each rank i sends block[s] to rank s
    allto_all_outputs = []
    for dst_rank in range(rank_num):
        out_blocks = []
        for src_rank in range(rank_num):
            # Rank src sends block[dst_rank] to rank dst
            # = permuted[src][dst_rank, :, :] shape [M, Np]
            out_blocks.append(allto_all_inputs[src_rank][dst_rank])
        allto_all_out = torch.cat(out_blocks, dim=0)  # [rankSize*M, Np]
        allto_all_outputs.append(allto_all_out)
        print(f"  Rank {dst_rank}: AllToAll output shape {allto_all_out.shape}")

    # Convert to BF16
    allto_all_outputs_bf16 = [out.to(torch.bfloat16) for out in allto_all_outputs]

    # Write artifacts
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    dirs_to_write = [current_dir, script_dir] if current_dir != script_dir else [current_dir]

    for base_dir in dirs_to_write:
        for rank_id in range(rank_num):
            write_artifacts(
                base_dir,
                rank_id,
                a_fp8_list[rank_id],
                b_fp8_list[rank_id],
                a_scale_list[rank_id],
                b_scale_list[rank_id],
                allto_all_outputs_bf16[rank_id]
            )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 gen_data.py m k n rank_num")
        print("  m: matrix A row dimension M")
        print("  k: matrix A column dimension K (full K, not split)")
        print("  n: matrix B column dimension N (split to N/rank_num per rank)")
        print("  rank_num: number of ranks")
        print("\nExample: python3 gen_data.py 2048 3584 4096 4")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])

    if k % 32 != 0:
        print(f"Error: K={k} not divisible by 32")
        sys.exit(1)

    if math.ceil(k / 64) % 2 != 0:
        print(f"Error: ceil(K/64)={math.ceil(k/64)} not even")
        sys.exit(1)

    if n % rank_num != 0:
        print(f"Error: n={n} not divisible by rank_num={rank_num}")
        sys.exit(1)

    print(f"Generating MatmulAllToAll test data:")
    print(f"  M={m}, K_full={k}, N={n}, rankSize={rank_num}")
    print(f"  Np={n//rank_num} (per rank, K not split)")

    gen_golden_data_matmul_allto_all(m, k, n, rank_num)

    print("Test data generation completed!")