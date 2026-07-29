#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# ======================================================================================================
"""
npu_block_sparse_attention mxfp4 (FLOAT4_E2M1) 正向单测 + CPU golden。

golden = bsa block_sparse_mask 网格逻辑(来自 test_bsa_golden.py)
       ⊕ qfa mxfp4 量化逻辑(来自 quant_flash_attn_golden.py / flash_attention_cpu_golden.py):
         - Q/K/V 做 mxfp4 qdq (Q/K 沿 head_dim block=32、V 沿 seq_k block=32), qdq 后用 fp32 算
         - S = Q@K^T fp32 matmul -> 降到 s_dtype(默认 fp16) -> 在 s_dtype 上乘 scale
         - P(softmax probs) 量化(blockwise_snap_local)与 P 本身均在 s_dtype(fp16),
           FP4 量化网格在 s_dtype 命中(snap_local 内含一次 bf16 round-trip);
         - sum(P) / PV matmul 在 fp32(P_eff.float() 后), 与 qfa flash golden 一致.
NPU 调用参考 test_pv_systematic.py: cann_ops_transformer.ops.block_sparse_attention_v3(quant_mode=2).
判据: qfa check_result (np.isclose rtol=1/128 atol=1e-4, 满足率>=99.5% 且 maxRE<10).

注: 本文件盲写(无 NPU 环境)。已据 aclnnBlockSparseAttentionV3.md (L678-684) 核实:
  - quantMode=2 仅 Ascend 950PR/950DT 支持; blockShapeX 仅 64/128; D 仅 64/128;
    不支持 attenMask/PagedAttention/softmaxLse (故 softmax_lse_flag=False, 只比 out 不比 lse)。
  - e8m0 形状与 doc L46-64 一致: q/k [B,N,S,D//64,2]、v [B,N2,S2//64,D,2], dtype float8_e8m0fnu。
  - NPU 调用签名经 test_pv_systematic.py + test_verify_pv_correctness.py 双重核对一致。
  - qfa dtype 流已逐行核对(flash_attention_cpu_golden.py L685-774): S=fp16, P_eff=fp16(量化网格在
    fp16 命中), sum(P)/PV=fp32(P_fp32=P_eff_s.float())。本 golden 默认 s_dtype='fp16' 对齐之。
  - golden 逻辑已由 validate_bsa_vs_qfa.py 交叉验证: 同 QKV + dense + MHA 下与 qfa flash golden
    在 fp16 P-量化噪声内逐位吻合(pass@1/128=100%, max_rel≈0.39%)。
仍需上机核验的盲写假设 (check_result 会暴露):
  (A) e8m0 (D//32)->(D//64,2) 与 (S2//32)->(S2//64,D,2) 的配对方向 (现取连续配对, 可能需 flip/交换);
  (B) nibble 顺序: pack 低 nibble=elem0、高 nibble=elem1 (OCP 惯例, 可能需交换);
  (C) P 量化方案【承重假设】: memory 记 BSA mxfp4 kernel 用 qfa blockwise_snap_local
      (E8M0=K_block-gmax+125), 但 V3 doc 公式为 Softmax(scale·Q·K^T)·value 无显式 P 量化、
      pDequantScale 标“预留不支持”。二者冲突。本 golden 默认 p_quant='snap_local' (按 memory+qfa 逻辑);
      若首跑 4 用例精度同挂, 改 run_bsa_mxfp4_eager(..., p_quant='none') (纯 softmax) 再试。
  (D) s_dtype 默认 fp16 对齐 qfa; 但 BSA 用 inner_precise=4(非 qfa 的 1), 4 对应的中间精度未知。
      若首跑同挂且 p_quant 切换无效, 试 s_dtype='bf16' 或 'fp32'(经 run_bsa_mxfp4_eager(..., s_dtype=) 切换)。
mxfp4 工具与 blockwise_snap_local 移植自 qfa 树 (ops-transformer-6060.../quant_flash_attn/tests/pytest),
0703 树无该工具, 故内嵌。
"""

# 基础库: math(sqrt/log/log2)、numpy(check_result 用)、torch(主计算)
import math
import numpy as np
import torch
# torch_npu 仅 NPU 环境有; 用 try/except guard, 使本模块在纯 CPU 也能被 import
# (golden 本身纯 CPU 可跑, 交叉验证就靠这个)。缺失时给 dummy 的 TestCase/装饰器。
try:
    import torch_npu  # noqa: F401  (NPU 环境)
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion
except Exception:  # CPU-only: 提供 dummy, 使模块可导入(golden 本身纯 CPU 可跑)
    torch_npu = None

    class TestCase:  # 最小替身, 仅供模块在无 torch_npu 时能定义测试类
        pass

    def run_tests():
        pass

    class _NoopDecorator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, fn):
            return fn

    SkipIfNotGteCANNVersion = _NoopDecorator

# BSA mxfp4 的 NPU 算子(python 包装)。缺失则置 None, 运行时跳过 NPU 调用。
try:
    from cann_ops_transformer.ops import block_sparse_attention_v3
    _HAS_BSA_V3 = True
    _BSA_V3_IMPORT_ERROR = None
except Exception as _e:  # 非 NPU 环境或包未装
    block_sparse_attention_v3 = None
    _HAS_BSA_V3 = False
    _BSA_V3_IMPORT_ERROR = _e

# 固定随机种子: 让 randn 造的 Q/K/V 可复现, 便于调试与交叉验证对照。
RAND_SEED = 21
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)


# =====================================================================
# 第一部分: MXFP4 (E2M1) 量化/反量化/打包 工具 (移植自 mx_quant_fp4_tool.py)
# =====================================================================
# ── FP4 E2M1 格式说明 ──────────────────────────────────────────────
# 4 bit = S(1) + E(2) + M(1), bias=1。可表示 16 个值, 无 inf/NaN(e=3 是正规数):
#   正规(e>0): (-1)^S * 2^(E-1) * (1 + M/2)   -> ±1,±1.5,±2,±3,±4,±6
#   次正规(e=0): (-1)^S * 2^0 * (M/2)         -> ±0,±0.5
# FP4_E2M1_VALUES 即这 16 个值的 fp32 模拟(量化时取最近)。
FP4_E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)
# E8M0: 8 bit 纯指数缩放因子, value = 2^(byte - bias), bias=127, 范围 2^-127..2^127。
E8M0_BIAS = 127
# FP4 元素格式的最大正规指数: 6.0 = 1.5*2^2, 故 emax_elem=2。
# e8m0 scale 取 exp = log2(max_abs) - emax, 使 block max 映射到 FP4 高端(4~6)以充分利用精度。
FP4_E2M1_EMAX = 2  # max 正规化指数 (6.0 = 1.5*2^2)


def generate_fp4_e2m1_encoding_table():
    """枚举 16 个 FP4 E2M1 编码, 返回 {fp32值: 4bit编码} 映射(供 convert 反查编码)。"""
    encodings, values = [], []
    for s in [0, 1]:          # 符号位 S
        for e in range(4):    # 2 位指数 E(0..3)
            for m in range(2):  # 1 位尾数 M(0/1)
                enc = (s << 3) | (e << 1) | m  # S-E-E-M 位序
                # 正规数(e>0): (-1)^S * 2^(E-1) * (1 + M/2); 次正规(e=0): (-1)^S * M/2
                val = (-1) ** s * (2.0 ** (e - 1)) * (1.0 + m / 2.0) if e != 0 \
                    else (-1) ** s * (2.0 ** 0) * (m / 2.0)
                encodings.append(enc)
                values.append(val)
    fp32_to_encoding = {}
    for enc, val in zip(encodings, values):
        # +0 与 -0 在 python 同 hash; 让正零(0b0000)优先, 保证正零往返为正零。
        if val not in fp32_to_encoding:  # 正零优先
            fp32_to_encoding[val] = enc
    return fp32_to_encoding


FP32_TO_FP4 = generate_fp4_e2m1_encoding_table()  # {fp32值: 4bit编码}, convert 中使用


def quantize_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """把每个元素量化到最近的 FP4 E2M1 值, 返回 FP32 模拟(值是 16 个 FP4 值之一)。"""
    with torch.no_grad():
        # 广播: 最后一维对齐 16 个候选 FP4 值, 取 abs 差最小的索引 = 最近邻量化。
        expanded = tensor.unsqueeze(-1)
        idx = torch.argmin(torch.abs(expanded - FP4_E2M1_VALUES.to(tensor.device)), dim=-1)
        return FP4_E2M1_VALUES.to(tensor.device)[idx]


def compute_e8m0_scale_block(block: torch.Tensor, mode: str = "baseline") -> torch.Tensor:
    """为一个 32 元素 block 算 E8M0 scale(2 的整数次幂)。
    baseline: exp = floor(log2(max_abs)) - emax  -> 最大化 FP4 范围, 允许少量饱和。
    oas:      exp = ceil(log2(max_abs)) - emax   -> 防饱和, 牺牲一点精度。"""
    max_abs = torch.max(torch.abs(block))                 # block 内绝对值最大
    if max_abs.item() < 1e-30:                            # 全 0 block: scale=1(避免 log2(0))
        return torch.tensor(1.0, device=block.device)
    log_max = torch.log2(max_abs)                         # log2(max_abs)
    exp = torch.floor(log_max) - FP4_E2M1_EMAX if mode == "baseline" \
        else torch.ceil(log_max) - FP4_E2M1_EMAX         # 整数指数
    exp = torch.clamp(exp, -127.0, 127.0)                 # E8M0 指数范围
    return 2.0 ** exp                                     # scale = 2^exp


def mx_quantize(tensor: torch.Tensor, block_size: int = 32, mode: str = "baseline"):
    """MXFP4 量化(FP32 模拟版): 沿最后一维每 block_size 个元素共享一个 E8M0 scale,
    再各自量化到 FP4。返回 (fp4_sim_fp32[同输入shape], scales_real[每个block一个])。"""
    original_shape = tensor.shape
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    last = tensor.shape[-1]
    # 最后一维不足 block_size 倍数时 pad 0(尾部补零, 不影响 max/scale)。
    pad = 0
    if last % block_size != 0:
        pad = block_size - (last % block_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad), value=0.0)
    reshaped = tensor.reshape(-1, block_size)   # [num_blocks, block_size]
    n_blk = reshaped.shape[0]
    scales = torch.empty(n_blk, device=tensor.device)   # 每 block 一个 scale
    fp4 = torch.empty_like(reshaped)
    for i in range(n_blk):
        sc = compute_e8m0_scale_block(reshaped[i], mode=mode)  # 该 block 的 E8M0 scale
        scales[i] = sc
        fp4[i] = quantize_fp4(reshaped[i] / sc)               # 先除 scale 归一化, 再量化到 FP4
    fp4 = fp4.view(tensor.shape)                              # 还原(含 pad)shape
    if pad > 0:
        fp4 = fp4[..., :-pad]                                 # 裁掉 pad
    return fp4, scales


def mx_dequantize(fp4_vals: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """MXFP4 反量化(FP32 模拟值 -> FP32): 每 block 乘回各自的 scale。沿最后一维分块。"""
    original_shape = fp4_vals.shape
    last = original_shape[-1] if fp4_vals.dim() > 0 else 1
    pad = 0
    if fp4_vals.dim() > 0 and last % block_size != 0:        # 同样 pad 到 block 倍数
        pad = block_size - (last % block_size)
        fp4_vals = torch.nn.functional.pad(fp4_vals, (0, pad), value=0.0)
    reshaped = fp4_vals.reshape(-1, block_size)
    # 每 block 乘其 scale: scales[blk] 广播到 block_size 个元素。
    dq = reshaped * scales.unsqueeze(1).to(reshaped.device)
    dq = dq.view(fp4_vals.shape)
    if pad > 0:
        dq = dq[..., :last]                                   # 裁 pad
    return dq


def mxfp4_qdq(t: torch.Tensor, axis: int, block_size: int = 32, mode: str = "baseline") -> torch.Tensor:
    """沿任意 axis 做 MXFP4 量化-反量化 round-trip, 返回带量化噪声的 FP32。
    golden 用它给 Q/K/V 引入与 NPU(packed fp4+e8m0)一致的量化误差, 使 CPU 参考与 NPU 公平对比。"""
    t_last = t.movedim(axis, -1).contiguous()   # 把量化轴搬到最后一维(mx_quantize 只沿最后维)
    fp4_sim, scales = mx_quantize(t_last, block_size=block_size, mode=mode)
    dq = mx_dequantize(fp4_sim, scales, block_size=block_size)   # 反量化回 fp32(带噪声)
    return dq.movedim(-1, axis).contiguous()    # 搬回原轴


def convert_fp32_simulated_to_fp4_encoding(fp32_sim: torch.Tensor) -> torch.Tensor:
    """把量化后的 FP32 模拟值(必为 16 个 FP4 值之一)转成真实 4-bit 编码(0-15) uint8。
    通过 FP32_TO_FP4 字典反查: 值匹配则填编码。"""
    flat = fp32_sim.flatten().cpu()
    enc = torch.zeros_like(flat, dtype=torch.uint8)
    for fp32_val, encoding in FP32_TO_FP4.items():
        enc[flat == fp32_val] = encoding        # 该值的位置填对应 4-bit 编码
    return enc.to(fp32_sim.device).reshape(fp32_sim.shape)


def pack_fp4_to_uint8(fp4_enc: torch.Tensor) -> torch.Tensor:
    """2 个 4-bit 编码 -> 1 个 uint8(OCP MX: 低 4 位=elem[2k], 高 4 位=elem[2k+1])。
    沿最后一维打包, 最后一维须偶数。"""
    shape = fp4_enc.shape
    if shape[-1] % 2 != 0:
        raise ValueError(f"最后一维须偶数, 当前 {shape[-1]}")
    reshaped = fp4_enc.reshape(-1, 2)                       # 两两一组
    packed = (reshaped[:, 1] << 4) | (reshaped[:, 0] & 0x0F)  # 高 nibble=后, 低 nibble=前
    return packed.reshape(*shape[:-1], shape[-1] // 2)      # 最后一维减半


def mxfp4_quantize_pack_last(tensor: torch.Tensor, quant_axis: int,
                             block_size: int = 32, mode: str = "baseline"):
    """NPU 侧打包入口: 量化轴任意, 打包轴固定最后一维。
    返回 (packed_fp4 uint8[最后一维=D//2], e8m0_scales uint8 biased[量化轴维=ceil(qd/32)])。
    Q/K 用 quant_axis=-1(沿 head_dim); V 用 quant_axis=-2(沿 seq_k)。"""
    nd = tensor.dim()
    qax = quant_axis % nd                                   # 负索引 -> 正索引
    assert tensor.shape[-1] % 2 == 0, f"最后一维须偶数, 当前 {tensor.shape[-1]}"
    t_q_last = tensor.movedim(qax, -1).contiguous()        # 量化轴搬到最后
    fp4_sim, scales_flat = mx_quantize(t_q_last, block_size=block_size, mode=mode)  # 量化(fp32 模拟)
    codes = convert_fp32_simulated_to_fp4_encoding(fp4_sim).movedim(-1, qax).contiguous()  # ->4bit编码
    packed = pack_fp4_to_uint8(codes)                      # 打包成 uint8
    qdim = tensor.shape[qax]                               # 量化轴长度
    n_blk = (qdim + block_size - 1) // block_size          # 该轴 block 数
    rest = list(t_q_last.shape[:-1])
    scales_real = scales_flat.reshape(*rest, n_blk).movedim(-1, qax).contiguous()  # 还原 scale 形状
    exp_int = torch.log2(scales_real).round().long()      # scale=2^exp -> 整数 exp
    e8m0 = (exp_int + E8M0_BIAS).clamp(0, 254).to(torch.uint8)  # E8M0 biased uint8(待 NPU 核验配对方向)
    return packed, e8m0


# =====================================================================
# 第二部分: blockwise_snap_local P 量化 (移植自 flash_attention_cpu_golden.py)
# 方案 B-snap-local: m_block 立即 floor 到 log2 网格, corr=2^integer 精确 E8M0,
# online-softmax 不变式保持. (memory: BSA mxfp4 kernel 用此方案)
# 注: helper 用 src_dtype=S.dtype, 传入 s_dtype 的 scores 则全程在 s_dtype 算
# (含 m_block/K_block/P_local/bucketize/P_q/corr/P_eff), 与 qfa 一致; 内部 P_local 有一次
# bf16 round-trip. 返回 P_eff(同 s_dtype) + m_new(fp32).
# =====================================================================
# P 是 softmax 概率(非负), 只用 FP4 正值 grid: {0,0.5,1,1.5,2,3,4,6}。
_FP4_POS_LEVELS_FP32 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# 量化边界(round-half-down 的中点), bucketize right=True 用。
_FP4_POS_MIDPOINTS_FP32 = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def _blockwise_snap_local_quantize_p_eff(S: torch.Tensor, m_running, seqk_dim: int,
                                          mx_block_size: int = 32, mx_mode: str = "baseline"):
    """对 scores(S, seqk_dim 轴)做 blockwise_snap_local P 量化, 返回 (P_eff, m_new)。
    思路: 每 32 元素 block 内 m_block 立即 snap 到 log2 网格 -> 局部归一化得 P_local(∈[4,8))
    -> FP4 round 得 raw P_q -> corr=2^(K_block-K_new-2) 折回 per-block E8M0 -> P_eff=P_q*corr。
    P_eff 是反量化后的有效 P(未归一化, 需外部 /sum); m_new=K_new(fp32 整数, 供 online softmax)。"""
    NEG_INF = float('-inf')
    device = S.device
    src_dtype = S.dtype                       # 全程跟随 S 的 dtype(对齐 qfa: fp16)
    ndim = S.dim()
    skp = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim  # seqk 轴正索引
    seqk_len = S.shape[skp]
    LN2 = math.log(2.0)

    # 阶段1: pad seqk 到 32 倍数(-inf padding 不影响 max; qfa 同款)。
    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pshape = list(S.shape)
        pshape[skp] = pad
        S_padded = torch.cat([S, torch.full(pshape, NEG_INF, dtype=src_dtype, device=device)], dim=skp)
    else:
        S_padded = S
    S_moved = S_padded.movedim(skp, -1)       # seqk 搬到最后, 便于分块
    other = S_moved.shape[:-1]                # 其余维度
    n_blk = S_moved.shape[-1] // mx_block_size
    S_r = S_moved.reshape(*other, n_blk, mx_block_size)  # [..., n_blk, 32]

    # 阶段2: 每 block 的 m_block_raw, 处理 -inf, snap 到 log2 网格。
    m_block_raw = S_r.max(dim=-1).values                  # 每 block max [..., n_blk]
    m_block_raw_safe = torch.where(torch.isinf(m_block_raw) & (m_block_raw < 0),
                                   torch.zeros_like(m_block_raw), m_block_raw)  # 全 -inf block 视作 0
    K_block = torch.clamp(torch.floor(m_block_raw_safe / LN2), -127.0, 127.0)  # snap: floor(m/ln2)
    # m_block_snap = (K-2)*ln2: 使 block max 经 exp 后落在 [4,8) -> FP4 高端(4/6)精度最高。
    m_block_snap = (K_block - 2) * LN2
    # 阶段3: 局部归一化 P_local = exp(S - m_block_snap), block max ∈ [4,8)。
    P_local = torch.exp(S_r - m_block_snap.unsqueeze(-1))
    # bf16 round-trip(对齐 qfa: 模拟 NPU 中间 bf16 精度)。
    P_local_bf = P_local.to(torch.bfloat16).to(P_local.dtype)
    # 阶段4: FP4 round。levels/midpoints 跟随 src_dtype。
    levels = _FP4_POS_LEVELS_FP32.to(device=device, dtype=src_dtype)
    midpoints = _FP4_POS_MIDPOINTS_FP32.to(device=device, dtype=src_dtype)
    idx = torch.bucketize(P_local_bf.contiguous(), midpoints, right=True)  # 落桶 -> level 索引
    P_q = levels[idx]                         # FP4 raw data(未乘 0.25, ∈{0,0.5,...,6})

    # 阶段5: K_new = 跨 block 取 max(在整数 K 域), 供 corr 与 online softmax。
    K_ij = K_block.max(dim=-1).values         # 跨 block 的 max K, s_dtype 整数
    K_ij_fp32 = K_ij.float()                  # 提到 fp32(供外部 alpha=exp2(K_diff))
    m_new = K_ij_fp32 if m_running is None else torch.maximum(m_running, K_ij_fp32)  # 单 tile 时 m_running=None
    K_new_s = m_new.to(src_dtype)             # cast 回 s_dtype 与 K_block 同 dtype 算
    # 阶段6: corr = 2^(K_block-K_new-2) = 0.25 * 2^(K_block-K_new), per-block E8M0。
    # (把 FP4 dequant 因子 2^-2 与跨 block rescale 2^(K_diff) 合并为单一整数指数)
    K_diff = K_block - K_new_s.unsqueeze(-1)  # ≤0, s_dtype 整数
    corr = torch.exp2(K_diff - 2.0)           # 2^(K_diff-2), 位精确
    # 阶段7: corr 广播回 seqk_len, 与 P_q 相乘得 P_eff。
    corr_broad = corr.repeat_interleave(mx_block_size, dim=-1)[..., :seqk_len].movedim(-1, skp)
    P_q_moved = P_q.reshape(*other, n_blk * mx_block_size)[..., :seqk_len].movedim(-1, skp)  # 去 pad
    P_eff = P_q_moved * corr_broad            # 反量化后的有效 P(未归一化)
    return P_eff, m_new


# =====================================================================
# 第三部分: BSA 数值辅助 (移植自 test_bsa_golden.py) + s_dtype 解析
# =====================================================================
# s_dtype: S/P 中间精度。fp16 对齐 qfa(inner_precise=1); bf16/fp32 供 BSA inner_precise=4 探索。
_S_DTYPE_MAP = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def _resolve_s_dtype(s_dtype):
    """字符串 s_dtype -> torch dtype。"""
    if s_dtype not in _S_DTYPE_MAP:
        raise ValueError(f"s_dtype must be one of {list(_S_DTYPE_MAP)}, got '{s_dtype}'")
    return _S_DTYPE_MAP[s_dtype]


def _logsumexp_np(scores):
    """对一行 scores 算 log(sum(exp(scores))), 数值稳定: 先减 max。全 mask(max<=-1e9)直接返回。"""
    s = scores.astype(np.float32)
    m = np.max(s)
    if m <= -1e9:                             # 全 -inf/-1e10 行: 避免 log(0)
        return m
    return float(m + np.log(np.sum(np.exp(s - m))))


def randn_x(shape, x=1.0, dtype=torch.float32, device=None):
    """x * 标准正态, 用于造 Q/K/V(qfa 风格)。"""
    return x * torch.randn(shape, dtype=dtype, device=device)


# =====================================================================
# 第四部分: CPU golden — BNSD block-sparse MXFP4 attention (qfa dtype 对齐)
# 这是本文件的【核心 golden】: 给定 Q/K/V + block_sparse_mask, 输出 attention_out(+lse)。
# 数据流: Q/K/V mxfp4 qdq(fp32) -> 按 q-block gather selected kv -> S=Q@K^T(fp32)->s_dtype->scale
#         -> P blockwise_snap_local 量化(s_dtype) -> P.float() -> sum/PV(fp32) -> out(->bf16)。
# =====================================================================
def cpu_block_sparse_attention_bnsd_mxfp4_with_lse(query, key, value, block_sparse_mask,
                                                    block_shape, scale_value, num_kv_heads,
                                                    p_quant='snap_local', s_dtype='fp16'):
    """CPU 标杆 (BNSD, MXFP4), 对齐 qfa flash golden 的 dtype 流:
       Q/K/V mxfp4 qdq(fp32); S=Q@K^T fp32 matmul -> .to(s_dtype) -> s_dtype 上乘 scale;
       P 量化(snap_local)与 P 本身在 s_dtype(FP4 网格在 s_dtype 命中, 含 bf16 round-trip);
       sum(P)/PV 在 fp32(P_eff.float() 后). lse=logsumexp(fp32 scores) 仅参考(mxfp4 NPU 不支持 lse).
       p_quant: 'snap_local'(blockwise_snap_local, memory: BSA mxfp4 kernel 同方案) | 'none'(纯 softmax).
       s_dtype: 'fp16'(默认, qfa 对齐) | 'bf16' | 'fp32'(对应 BSA inner_precise, 待上机确认)."""
    # ── 准备: 统一升 fp32, 解析形状 ───────────────────────────────
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    B, N, S, D = q.shape                      # B=batch, N=query头数, S=q序列长, D=head_dim
    N2 = k.shape[1]                           # kv 头数(GQA 时 N2<N)
    S2 = k.shape[2]                           # KV 序列长 (不与 q 的 S 混用, 支持 s1≠s2)
    block_x, block_y = int(block_shape[0]), int(block_shape[1])  # q块大小 / kv块大小
    ceil_q = (S + block_x - 1) // block_x     # q 方向块数(向上取整)
    ceil_kv = (S2 + block_y - 1) // block_y   # kv 方向块数
    s_torch_dtype = _resolve_s_dtype(s_dtype)  # S/P 中间精度(默认 fp16)

    # ── Q/K/V mxfp4 qdq: 引入与 NPU packed fp4+e8m0 一致的量化噪声 ──
    q_q = mxfp4_qdq(q, axis=-1)            # Q: 沿 head_dim(rowwise, block32) 量化
    k_q = mxfp4_qdq(k, axis=-1)            # K: 同 Q
    v_q = mxfp4_qdq(v, axis=2)             # V: 沿 seq_k(columnwise, block32) 量化

    # ── 输出 buffer ──────────────────────────────────────────────
    out = torch.zeros((B, N, S, D), dtype=torch.float32)             # attention 输出(fp32 累加)
    lse = torch.full((B, N, S, 1), float('-inf'), dtype=torch.float32)  # logsumexp(空行=-inf)

    # ── 主循环: 逐 batch / head / q-token ────────────────────────
    for b in range(B):
        for n in range(N):
            n2 = n % N2                    # GQA 取模映射: query 头 n -> kv 头 n%N2(与 test_bsa_golden 一致)
            for s1 in range(S):
                i_block = s1 // block_x    # 该 q-token 所属的 q 块号
                # 收集该 q 块允许 attend 的 kv 块(ascending j_block) —— 与 NPU sparse gather 一致。
                sel_j = [j for j in range(ceil_kv) if block_sparse_mask[b, n, i_block, j] != 0]
                if not sel_j:              # 该 q 块无任何 kv 块: 跳过(out 保持 0, lse 保持 -inf)
                    continue
                # 把选中 kv 块展开成具体 token 位置(每块 block_y 个, 尾块截断到 S2)。
                positions = []
                for j in sel_j:
                    st = j * block_y
                    ed = min(st + block_y, S2)
                    positions.extend(range(st, ed))
                if not positions:
                    continue
                pos = torch.as_tensor(positions, dtype=torch.long)
                qvec = q_q[b, n, s1]                      # [D] fp32, 该 q-token 的 query 向量
                kvec = k_q[b, n2][pos]                    # [npos, D] fp32, gathered key
                vvec = v_q[b, n2][pos]                    # [npos, D] fp32, gathered value
                # ── S: fp32 matmul -> 降到 s_dtype -> s_dtype 上乘 scale (对齐 qfa L685-696) ──
                # 注意 scale 须 round 到 s_dtype(qfa 用 np.float16(scale)), 否则 snap_local 的
                # log2 网格 snap 会在 m_block 边界翻转, 放大个别 P 元素误差。
                scores_fp32 = (kvec @ qvec).to(torch.float32)   # [npos] = scale·Q·K^T, fp32
                scale_s = torch.as_tensor(scale_value, dtype=s_torch_dtype)  # scale round 到 s_dtype
                scores_s = scores_fp32.to(s_torch_dtype) * scale_s          # fp16(默认) S
                if p_quant == 'snap_local':
                    # P 量化在 s_dtype (FP4 网格在 s_dtype 命中) -> P_eff 为 s_dtype
                    P_eff_s, _ = _blockwise_snap_local_quantize_p_eff(scores_s, None, seqk_dim=-1)
                    P_eff_s = torch.nan_to_num(P_eff_s, nan=0.0)             # 防 NaN(全 mask 行)
                    P_fp32 = P_eff_s.float()             # 升 fp32 供 sum/PV (对齐 qfa L723)
                else:  # 'none': 纯 softmax, P 在 s_dtype, sum/PV 在 fp32
                    m_s = scores_s.max()
                    m_safe_s = torch.where(torch.isinf(m_s) & (m_s < 0), torch.zeros_like(m_s), m_s)
                    P_ij = torch.exp(scores_s - m_safe_s)   # s_dtype softmax(未量化)
                    P_fp32 = torch.nan_to_num(P_ij, nan=0.0).float()
                # ── sum(P) / PV 均在 fp32 (对齐 qfa L767-769) ─────────────
                l = P_fp32.sum()                          # 该行概率和(归一化分母)
                l_safe = l if float(l) > 0 else torch.tensor(1.0)  # 全 0 行用 1 防 0/0
                out[b, n, s1] = (P_fp32 @ vvec) / l_safe   # PV: P@V 再除以 sum(P), fp32
                # lse 用未降精度的 fp32 scores 算(仅参考; mxfp4 模式 NPU 不返回 lse)。
                lse[b, n, s1, 0] = _logsumexp_np(scores_fp32.detach().cpu().numpy())
    return out.to(torch.bfloat16), lse.to(torch.float32)   # out->bf16(对齐 NPU 输出 dtype), lse->fp32


# =====================================================================
# 第五部分: check_result / compare (移植自 quant_flash_attn_golden.py)
# 判据: 逐元素 isclose(rtol,atol), 算满足率; PASS 需 满足率>=99.5% 且 maxRE<10。
# =====================================================================
def check_result(expect, result, data_type, pct_thd=0.005):
    """expect=golden, result=NPU 输出。返回 (fulfill_percent, "Pass"/"Failed")。"""
    real_data = result.cpu().numpy().flatten()        # NPU 输出, 展平
    data_compe = expect.cpu().numpy().flatten()       # golden, 展平
    if real_data.size == 0 and real_data.size == data_compe.size:  # 空且同空 -> Pass
        return 100.0, "Pass"
    if real_data.size != data_compe.size:             # 尺寸不符 -> 直接 Failed
        return 0.0, "Failed"
    # 按 dtype 选容差: bf16 用 1/128(0.78125%); 其余(fp32 等)用 0.5%。
    if data_type == 'bfloat16':
        rtol, atol, max_diff_hd = 0.0078125, 0.0001, 10.0      # 1/128
    else:
        rtol, atol, max_diff_hd = 0.005, 0.000025, 10.0
    # 逐元素 isclose(都先转 fp32 比), equal_nan 把两边 NaN 视作相等。
    diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32),
                             rtol=rtol, atol=atol, equal_nan=True)
    err_idx = np.where(diff_result != np.array((True,)))[0]   # 不达标的元素下标
    split_count = real_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0  # 满足率%
    # 另算相对误差(只在 err 元素上), 用于 maxRE 判据。
    diff_abs = np.abs(data_compe - real_data)
    b = np.add(np.maximum(np.abs(real_data), np.abs(data_compe)), 10e-10)
    err_diff = (diff_abs / b)[err_idx]
    pct_thd = (1 - pct_thd) * 100.0                   # pct_thd=0.005 -> 99.5%
    result_str = "Pass" if fulfill_percent >= pct_thd else "Failed"   # 满足率>=99.5% 才 Pass
    if len(err_diff) > 0 and np.max(err_diff) >= max_diff_hd:        # maxRE>=10 强制 Failed
        result_str = "Failed"
    return fulfill_percent, result_str


def compare(golden, real, src_dtype):
    """封装: 两边转 fp32 后调 check_result。"""
    if golden is not None and real is not None:
        return check_result(golden.to(torch.float32), real.cpu().to(torch.float32), src_dtype)
    return 0.0, "Failed"


# =====================================================================
# 第六部分: e8m0 -> V3 形状 重排 (假设 A: 连续配对, 待 NPU 核验)
# mxfp4_quantize_pack_last 产出的 e8m0 是 per-32(uint8 biased), 需重排成 V3 doc 形状
# (q/k: 末维 (D//32)->(D//64,2); v: (S2//32)->(S2//64,2) 且与 D 维换序), 再 view 成 float8_e8m0fnu。
# =====================================================================
def _e8m0_qk_to_v3(e8m0_uint8):
    """q/k e8m0 [B,N,S,D//32] -> [B,N,S,D//64,2] (连续配对: 两两 32-block 折成 64-group 的 2 子槽)。"""
    *lead, nd = e8m0_uint8.shape                 # nd = D//32
    return e8m0_uint8.reshape(*lead, nd // 2, 2).contiguous().view(torch.float8_e8m0fnu)


def _e8m0_v_to_v3(e8m0_uint8):
    """v e8m0 [B,N2,S2//32,D] -> [B,N2,S2//64,D,2] (连续配对 + 把 '2' 摞到 D 之后)。"""
    B, N2, nb, D = e8m0_uint8.shape             # nb = S2//32
    # 先 (nb)->(nb//2,2), 再 permute 把 2 维挪到 D 后: [B,N2,nb//2,2,D] -> [B,N2,nb//2,D,2]
    return e8m0_uint8.reshape(B, N2, nb // 2, 2, D).permute(0, 1, 2, 4, 3).contiguous().view(torch.float8_e8m0fnu)


# =====================================================================
# 第七部分: qfa 风格 driver — 造数据/quantize/golden/NPU/compare
# 串起整条链路: fp32 造数据 -> 官方 mask -> CPU golden -> mxfp4 打包喂 NPU -> check_result。
# =====================================================================
def _build_block_sparse_mask(b, n1, ceil_q, ceil_kv, kind='dense'):
    """构造官方 block_sparse_mask 网格 (B,n1,ceil_q,ceil_kv) int8: !=0 表示该 q 块 attend 该 kv 块。"""
    if kind == 'dense':                          # 全 1 = 稠密(所有 kv 块都 attend)
        return torch.ones(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
    if kind == 'sparse':  # 每个 q_block i 仅 attend kv_block (i % ceil_kv)
        m = torch.zeros(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
        for i in range(ceil_q):
            m[:, :, i, i % ceil_kv] = 1           # 每 q 块只选一个 kv 块(保证每行至少一个有效块)
        return m
    raise ValueError(f"unsupported mask kind: {kind}")


def run_bsa_mxfp4_eager(b, n1, n2, g, s1, s2, qk_d, v_d=None,
                        input_layout='BNSD', out_dtype='bfloat16',
                        block_shape=(128, 128), mask_kind='dense',
                        inner_precise=4, quant_mode=2,
                        p_quant='snap_local', s_dtype='fp16', device_id=0):
    """qfa 风格: fp32 造数据 -> mxfp4 打包 -> CPU golden -> NPU -> check_result.
       p_quant: 'snap_local'|'none'; s_dtype: 'fp16'(默认,qfa对齐)|'bf16'|'fp32' (对应 inner_precise).
       返回 (fulfill_percent, result_str). NPU 不可用时返回 (-1.0, 'SKIP')."""
    # ── 参数合法性 (MXFP4 spec 约束) ─────────────────────────────
    assert input_layout == 'BNSD', "首版仅支持 BNSD"
    assert n1 == n2 * g, f"GQA 约束 n1==n2*g: {n1} vs {n2}*{g}"
    assert block_shape[0] in (64, 128), f"MXFP4 (quantMode=2) blockShapeX 仅支持 64/128, 当前 {block_shape[0]}"
    assert qk_d in (64, 128), f"MXFP4 D 仅支持 64/128, 当前 {qk_d}"
    if v_d is None:
        v_d = qk_d                             # V 的 head_dim 默认同 Q/K
    scale = 1.0 / math.sqrt(qk_d)             # 标准 attention 缩放 1/√D
    src_dtype = out_dtype                     # 比对 dtype(传给 check_result 选容差)

    # 1. 造 fp32 数据 (qfa 风格 randn, 固定种子)
    query = randn_x((b, n1, s1, qk_d), 1.0).to(torch.float32)
    key = randn_x((b, n2, s2, qk_d), 1.0).to(torch.float32)
    value = randn_x((b, n2, s2, v_d), 1.0).to(torch.float32)

    # 2. 官方 block_sparse_mask 网格
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (s1 + block_x - 1) // block_x
    ceil_kv = (s2 + block_y - 1) // block_y
    mask = _build_block_sparse_mask(b, n1, ceil_q, ceil_kv, mask_kind)

    # 3. CPU golden (mxfp4 qdq + P 量化, p_quant/s_dtype 可切换)
    cpu_out, _ = cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
        query, key, value, mask, block_shape, scale, n2,
        p_quant=p_quant, s_dtype=s_dtype)

    # 4. mxfp4 打包喂 NPU: Q/K axis=-1, V axis=-2
    q_packed, q_e8m0 = mxfp4_quantize_pack_last(query, quant_axis=-1)   # [b,n1,s1,d//2], e8m0 [b,n1,s1,d//32]
    k_packed, k_e8m0 = mxfp4_quantize_pack_last(key, quant_axis=-1)
    v_packed, v_e8m0 = mxfp4_quantize_pack_last(value, quant_axis=-2)   # [b,n2,s2,d//2], e8m0 [b,n2,s2//32,d]
    # 防御性 dtype 断言(个别 torch 版 uint8<<4 会静默升 int64)
    assert q_packed.dtype == k_packed.dtype == v_packed.dtype == torch.uint8, "packed fp4 须为 uint8"
    assert q_e8m0.dtype == k_e8m0.dtype == v_e8m0.dtype == torch.uint8, "e8m0 scale 须为 uint8"
    # e8m0 重排成 V3 doc 形状 + view 成 float8_e8m0fnu
    q_scale = _e8m0_qk_to_v3(q_e8m0)   # [b,n1,s1,d//64,2] float8_e8m0fnu
    k_scale = _e8m0_qk_to_v3(k_e8m0)   # [b,n2,s2,d//64,2]
    v_scale = _e8m0_v_to_v3(v_e8m0)     # [b,n2,s2//64,d,2]

    # 5. NPU 调用 (照搬 test_pv_systematic.py: packed uint8 fp4 + e8m0 + block_sparse_mask)
    if not _HAS_BSA_V3:
        return -1.0, "SKIP"                    # 纯 CPU/包未装: 跳过 NPU(golden 仍可独立跑)
    torch_npu.npu.set_device(device_id)
    npu_out, _ = block_sparse_attention_v3(
        q_packed.contiguous().npu(),          # query: packed fp4 uint8
        k_packed.contiguous().npu(),          # key
        v_packed.contiguous().npu(),          # value
        mask.contiguous().npu(),              # block_sparse_mask int8
        list(block_shape),                     # [blockShapeX, blockShapeY]
        [s1] * b,                 # actual_seq_lengths (每 batch 全长)
        [s2] * b,                 # actual_seq_lengths_kv
        num_heads=n1,                          # query 头数
        num_key_value_heads=n2,                # kv 头数(GQA)
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        mask_type=0,                           # 无 causal/prefix
        scale_value=scale,                     # 1/√D
        inner_precise=inner_precise,           # 4(MXFP4 模式, 待上机确认对应 s_dtype)
        block_size=0,                           # 非 PagedAttention
        pre_tokens=2147483647,                  # 无 window(全 attend)
        next_tokens=2147483647,
        softmax_lse_flag=False,                # MXFP4 不支持 lse -> 不取/不比 lse
        quant_mode=quant_mode,                 # 2 = MXFP4
        dst_type_max=0,
        q_dequant_scale=q_scale.npu(),         # e8m0 descale
        k_dequant_scale=k_scale.npu(),
        v_dequant_scale=v_scale.npu(),
        attention_out_dtype=torch.bfloat16,    # 输出 bf16
    )
    torch_npu.npu.synchronize()

    # 6. 比对 (qfa check_result)
    fulfill_percent, result = compare(cpu_out, npu_out, src_dtype)
    return fulfill_percent, result


# =====================================================================
# 第八部分: 测试类 — 4 条 BNSD 用例
# =====================================================================
class TestBSAMxfp4Golden(TestCase):
    """BSA mxfp4 正向 vs CPU golden, qfa check_result 判据。
    注: quantMode=2 仅 Ascend 950PR/950DT 支持 (V3 doc L680), 故不锁 SupportedDevices,
    靠 @SkipIfNotGteCANNVersion + 运行期能力检查。"""

    def _run(self, **cfg):
        """共用: 跳过无 NPU 环境, 跑 run_bsa_mxfp4_eager, 断言 result=='Pass'。"""
        if not _HAS_BSA_V3:
            self.skipTest(f"cann_ops_transformer.ops.block_sparse_attention_v3 不可用: "
                          f"{_BSA_V3_IMPORT_ERROR}")
        pct, res = run_bsa_mxfp4_eager(**cfg)
        # 失败时打印 + 给排查指引(假设 A/B/C/D)。
        assert res == "Pass", (f"BSA mxfp4 case 失败: {cfg}\n  fulfill={pct}% result={res}\n"
                               f"  (若 fulfill<99.5% 或 maxRE>=10, 先核验 spec 假设 A/B/C/D; "
                               f"4 用例同挂优先怀疑 C(p_quant) 或 D(s_dtype), 改 p_quant='none' / s_dtype='bf16'|'fp32' 再试)")
        print(f"[PASS] {cfg.get('mask_kind','dense')} b={cfg['b']} n1={cfg['n1']} n2={cfg['n2']} "
              f"s1={cfg['s1']} s2={cfg['s2']} block={cfg['block_shape']} s_dtype={cfg.get('s_dtype','fp16')} "
              f"fulfill={pct:.4f}%")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_dense_single_block(self, device="npu"):
        """① BNSD dense 单块: 基本精度 (block_shape=[128,128], S=128 -> 1x1)."""
        self._run(b=2, n1=8, n2=8, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_multi_block(self, device="npu"):
        """② BNSD 多块 dense: block_shape=[64,128], S=256 -> ceil_q=4, ceil_kv=2 (4x2)."""
        self._run(b=2, n1=8, n2=8, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[64, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_sparse_mask(self, device="npu"):
        """③ BNSD 稀疏 mask: 每 q_block i attend kv_block i%ceil_kv (S=256, [64,128] -> 4x2)."""
        self._run(b=2, n1=8, n2=8, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[64, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa(self, device="npu"):
        """④ BNSD GQA: g=2, N1=8/N2=4 (block_shape=[128,128] -> 1x1)."""
        self._run(b=2, n1=8, n2=4, g=2, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense')


if __name__ == "__main__":
    run_tests()
