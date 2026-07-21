#!/usr/bin/env python3
"""Unified compressor_grad case set.

Each case is a dict with the same field schema, consumed by
``tests/test_compressor_grad.py`` to drive the golden reference
``compressor_grad_golden.compressor_grad``.

Field schema:
    seed, grad_scale, weight_scale  — RNG / value scaling
    coff                            — overlap factor (1 or 2)
    layout                          — 0 = BSH, 1 = TH  (cube convention)
    dataType                        — 0 = BF16, 1 = FP16
    b_size                          — batch count
    s_size                          — per-batch seq len (BSH); informational for TH
    d_size                          — head dim (128 or 512)
    h_size                          — hidden size (default 4096)
    cmp_ratio                       — compression ratio (4 or 128)
    sps                             — start_pos list; None => all zeros
    sqs                             — seqused list; None => full s_size
    desc                            — human-readable description

Sources:
    Cube cases (case_01 .. case_36): from tests/test_cube.py — cmp_ratio=4,
        sps=None (all zeros), sqs=None (full s_size), h_size=4096,
        grad_scale=0.1, weight_scale=10.0.
    Vec STANDARD_CASES (case_37 .. case_47): from tests/test_vec_cases.py —
        explicit cmp_ratio / sps / sqs, exercise Phase-1 scatter path
        (prefill + decode).
    Vec QUICK_CASES (case_48 .. case_50): from tests/test_vec_cases.py —
        smoke-test subset.

Note on layout convention: vec cases originally used string ``layout`` with
"TH" -> 0 and "BSH" -> 1 (opposite of cube).  Here all cases are normalized
to the cube convention (0 = BSH, 1 = TH) so a single code path handles both.
"""

# ════════════════════════════════════════════════════════════════════════════
# Unified case set
# ════════════════════════════════════════════════════════════════════════════
CASES = {
    # ── Cube stage: 36 cases (from test_cube.py) ──
    "case_01_bsh_bf16_coff2": dict(
        seed=42, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=0,
        b_size=1, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="BSH/BF16/coff=2 base case (single-tile, no k_acc).",
    ),
    "case_02_bsh_bf16_coff1": dict(
        seed=123, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=0, dataType=0,
        b_size=1, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="BSH/BF16/coff=1 base case (single M/K/N-tile).",
    ),
    "case_03_th_bf16_coff2": dict(
        seed=42, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=0,
        b_size=1, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="TH/BF16/coff=2 base case.",
    ),
    "case_04_th_bf16_coff1": dict(
        seed=123, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=0,
        b_size=1, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="TH/BF16/coff=1 base case.",
    ),
    "case_05_bsh_bf16_coff2_b2": dict(
        seed=42, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=0,
        b_size=2, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="B=2 -> T=256 (multi M-tile d_x, still no k_acc).",
    ),
    "case_06_bsh_bf16_coff2_b4": dict(
        seed=0, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=0,
        b_size=4, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="B=4 -> T=512.",
    ),
    "case_07_th_bf16_coff2_b8": dict(
        seed=7, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=0,
        b_size=8, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="B=8 -> T=1024.",
    ),
    "case_08_bsh_bf16_coff1_b16": dict(
        seed=123, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=0, dataType=0,
        b_size=16, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="B=16 -> T=2048 (coff=1, max B without k_acc).",
    ),
    "case_09_th_bf16_coff2_b32": dict(
        seed=314, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=0,
        b_size=32, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="B=32 -> T=4096 (k_acc=True, coff=2).",
    ),
    "case_10_bsh_bf16_coff2_b64": dict(
        seed=1618, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=0,
        b_size=64, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="B=64 -> T=8192 (k_acc=True, largest batch).",
    ),
    "case_11_th_fp16_coff2_b4": dict(
        seed=2718, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=4, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="FP16 dtype, B=4.",
    ),
    "case_12_bsh_fp16_coff1_b8": dict(
        seed=5772, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=0, dataType=1,
        b_size=8, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="FP16 dtype, coff=1, B=8.",
    ),
    "case_13_bsh_bf16_coff2_s4096": dict(
        seed=42, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=0,
        b_size=1, s_size=4096, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="S=4096 -> T=4096 (k_acc=True).",
    ),
    "case_14_th_bf16_coff2_s4096": dict(
        seed=0, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=0,
        b_size=1, s_size=4096, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="S=4096, TH layout.",
    ),
    "case_15_bsh_bf16_coff1_s8192": dict(
        seed=7, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=0, dataType=0,
        b_size=1, s_size=8192, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="S=8192, coff=1 (k_acc=True).",
    ),
    "case_16_th_bf16_coff2_s8192": dict(
        seed=123, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=0,
        b_size=1, s_size=8192, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="S=8192.",
    ),
    "case_17_bsh_bf16_coff2_s16384": dict(
        seed=314, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=0,
        b_size=1, s_size=16384, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="S=16384 (largest sequence).",
    ),
    "case_18_th_fp16_coff2_s16384": dict(
        seed=2718, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=1, s_size=16384, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="S=16384, FP16.",
    ),
    "case_19_bsh_bf16_coff2_b2_s4096": dict(
        seed=5772, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=0,
        b_size=2, s_size=4096, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="B=2 + S=4096 -> T=8192.",
    ),
    "case_20_th_bf16_coff1_b4_s4096": dict(
        seed=1618, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=0,
        b_size=4, s_size=4096, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="B=4 + S=4096 -> T=16384, coff=1.",
    ),
    "case_21_bsh_bf16_coff2_d512": dict(
        seed=42, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=0,
        b_size=1, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="D=512 -> C=1024 (multi M/K-tile, no k_acc).",
    ),
    "case_22_th_bf16_coff2_d512": dict(
        seed=0, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=0,
        b_size=1, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="D=512, TH.",
    ),
    "case_23_bsh_bf16_coff1_d512": dict(
        seed=7, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=0, dataType=0,
        b_size=1, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="D=512, coff=1 -> C=512.",
    ),
    "case_24_th_bf16_coff1_d512": dict(
        seed=123, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=0,
        b_size=1, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="D=512, coff=1, TH.",
    ),
    "case_25_bsh_fp16_coff2_d512": dict(
        seed=314, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=1,
        b_size=1, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="D=512, FP16.",
    ),
    "case_26_th_fp16_coff2_d512": dict(
        seed=2718, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=1, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="D=512, FP16, TH.",
    ),
    "case_27_bsh_bf16_coff2_d512_b4": dict(
        seed=5772, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=0, dataType=0,
        b_size=4, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="D=512 + B=4 -> T=512.",
    ),
    "case_28_th_bf16_coff2_d512_s4096": dict(
        seed=1618, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=0,
        b_size=1, s_size=4096, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="D=512 + S=4096 (k_acc + multi M/K-tile).",
    ),
    "case_29_bsh_fp16_coff1": dict(
        seed=42, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=0, dataType=1,
        b_size=1, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="FP16 coff=1 single-tile path (B=1, S=128, D=128).",
    ),
    "case_30_th_fp16_coff1_b32": dict(
        seed=123, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=32, s_size=128, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="FP16 coff=1 k_acc path (B=32 -> T=4096).",
    ),
    "case_31_th_fp16_coff1_d512": dict(
        seed=2718, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=1, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="FP16 coff=1 D=512 single M-tile (B=1).",
    ),
    "case_32_th_fp16_coff1_d512_b4": dict(
        seed=5772, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=4, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="FP16 coff=1 D=512 B>1 no k_acc.",
    ),
    "case_33_th_fp16_coff1_d512_b32": dict(
        seed=314, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=32, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="FP16 coff=1 D=512 k_acc (B=32).",
    ),
    "case_34_th_bf16_coff1_d512_b4": dict(
        seed=1618, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=0,
        b_size=4, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="BF16 coff=1 D=512 B>1 no k_acc.",
    ),
    "case_35_th_bf16_coff1_d512_b32": dict(
        seed=42, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=0,
        b_size=32, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="BF16 coff=1 D=512 k_acc (B=32).",
    ),
    "case_36_th_fp16_coff2_d512_b16": dict(
        seed=0, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=16, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=4, sps=None, sqs=None,
        desc="FP16 coff=2 D=512 k_acc (B=16 -> T=2048).",
    ),

    # ── Vec stage: 11 STANDARD_CASES (from test_vec_cases.py) ──
    # Prefill C4A: large head, coff=2, cmpRatio=4
    "case_37_prefill_c4a": dict(
        seed=42, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=1, s_size=8192, d_size=512, h_size=4096,
        cmp_ratio=4, sps=[0], sqs=[8192],
        desc="Prefill C4A: large head, coff=2, cmpRatio=4.",
    ),
    # Prefill C4Li: small head, coff=2, cmpRatio=4
    "case_38_prefill_c4li": dict(
        seed=43, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=1, s_size=8192, d_size=128, h_size=4096,
        cmp_ratio=4, sps=[0], sqs=[8192],
        desc="Prefill C4Li: small head, coff=2, cmpRatio=4.",
    ),
    # Prefill C128A: large head, coff=1, cmpRatio=128
    "case_39_prefill_c128a": dict(
        seed=44, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=1, s_size=8192, d_size=512, h_size=4096,
        cmp_ratio=128, sps=[0], sqs=[8192],
        desc="Prefill C128A: large head, coff=1, cmpRatio=128.",
    ),
    # Decode — 有压缩块
    "case_40_decode_c4a_b1_s1": dict(
        seed=45, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=1, s_size=1, d_size=512, h_size=4096,
        cmp_ratio=4, sps=[3], sqs=[1],
        desc="Decode C4A B=1 S=1 (有压缩块).",
    ),
    "case_41_decode_c4li_b1_s1": dict(
        seed=46, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=1, s_size=1, d_size=128, h_size=4096,
        cmp_ratio=4, sps=[3], sqs=[1],
        desc="Decode C4Li B=1 S=1 (有压缩块).",
    ),
    # Decode — 无压缩块（边界验证）
    "case_42_decode_c128a_b1_s1": dict(
        seed=47, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=1, s_size=1, d_size=512, h_size=4096,
        cmp_ratio=128, sps=[8192], sqs=[1],
        desc="Decode C128A B=1 S=1 (无压缩块, 边界验证).",
    ),
    # Decode — 有压缩块（多 batch）
    "case_43_decode_c4a_b8_s3": dict(
        seed=48, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=8, s_size=4, d_size=512, h_size=4096,
        cmp_ratio=4, sps=[0]*8, sqs=[4]*8,
        desc="Decode C4A B=8 (有压缩块, 多 batch).",
    ),
    # Decode — 无压缩块（多 batch 边界验证）
    "case_44_decode_c4li_b8_s3": dict(
        seed=49, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=8, s_size=3, d_size=128, h_size=4096,
        cmp_ratio=4, sps=[8193]*8, sqs=[3]*8,
        desc="Decode C4Li B=8 (无压缩块, 多 batch 边界验证).",
    ),
    # Decode — 有压缩块（cr=128 多 batch）
    "case_45_decode_c128a_b8_s3": dict(
        seed=50, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=8, s_size=128, d_size=512, h_size=4096,
        cmp_ratio=128, sps=[0]*8, sqs=[128]*8,
        desc="Decode C128A B=8 (有压缩块, cr=128 多 batch).",
    ),
    # Decode — start_pos=None (default 0), seqused provided
    "case_46_decode_c4a_sp_none": dict(
        seed=51, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=2, s_size=4, d_size=128, h_size=4096,
        cmp_ratio=4, sps=None, sqs=[4, 4],
        desc="Decode C4a start_pos=None (default 0), seqused provided.",
    ),
    # Decode — cu_seqlens None path (TH layout, kernel-only; golden still
    # receives cu_seqlens so it can identify the layout)
    "case_47_decode_c4a_cu_none": dict(
        seed=52, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=2, s_size=4, d_size=128, h_size=4096,
        cmp_ratio=4, sps=[0, 0], sqs=[4, 4],
        desc="Decode C4a cu_seqlens=None path (TH, seqused provided).",
    ),

    # ── Vec stage: 3 QUICK_CASES (from test_vec_cases.py) ──
    "case_48_quick_coff1_cr4": dict(
        seed=42, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=1, s_size=8, d_size=128, h_size=4096,
        cmp_ratio=4, sps=[0], sqs=[8],
        desc="Quick coff=1 cr=4.",
    ),
    "case_49_quick_coff2_cr4": dict(
        seed=43, grad_scale=0.1, weight_scale=10.0,
        coff=2, layout=1, dataType=1,
        b_size=1, s_size=8, d_size=128, h_size=4096,
        cmp_ratio=4, sps=[0], sqs=[8],
        desc="Quick coff=2 cr=4.",
    ),
    "case_50_quick_coff1_cr4_small": dict(
        seed=44, grad_scale=0.1, weight_scale=10.0,
        coff=1, layout=1, dataType=1,
        b_size=1, s_size=8, d_size=128, h_size=4096,
        cmp_ratio=4, sps=[0], sqs=[8],
        desc="Quick coff=1 cr=4 small.",
    ),
}
