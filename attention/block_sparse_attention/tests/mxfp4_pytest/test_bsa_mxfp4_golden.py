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
【kernel 模拟 + dump 能力】(对齐 qfa flash_attention_cpu_golden.py):
         - golden 内层为分块 flash: per q-block + 2-kv-block chunk(对齐 kernel s2LoopCount*2)
           + online softmax(m_run 跨 chunk rescale), 与 qfa _flash_attn_single_batch_kernel 同构。
         - dump=True 时把中间结果落盘 dump_data/*.bin, 镜像 qfa 的 bin 集合:
           mm1Res(S) / mm2ReduceSum(sum P) / mm2Res_1(PV) / p_packed_x4(P FP4) /
           p_scale_e8m0(P E8M0) / local_group_max / global_max, per-chunk append(kernel tile 序)。
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
  (E) dump 的 bin 字节序/转置方向(尤其 p_packed_x4/p_scale_e8m0 的 .T)为镜像 qfa 的盲写,
      首跑需对照 NPU kernel GM dump(desc=5/6/21/8/16/14/15/29/36/18/19/50) 校正方向。
mxfp4 工具与 blockwise_snap_local 移植自 qfa 树, 0703 树无该工具, 故内嵌。
"""

import math
import os
import numpy as np
import torch
# torch_npu 仅 NPU 环境有; guard 使本模块纯 CPU 也能 import(golden 纯 CPU 可跑, 交叉验证靠这个)。
try:
    import torch_npu  # noqa: F401  (NPU 环境)
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion
except Exception:  # CPU-only: 提供 dummy, 使模块可导入
    torch_npu = None

    class TestCase:  # 最小替身
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

# 固定随机种子: 让 randn 造的 Q/K/V 可复现。
RAND_SEED = 21
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)


# =====================================================================
# 第零部分: dump 基础设施 (镜像 qfa flash_attention_cpu_golden.py 的 dump 机制)
# golden dump=True 时, 把每个 chunk(kernel tile)的中间结果以 raw bytes 追加到 dump_data/*.bin,
# 便于与 NPU kernel 的 GM tile dump 逐 tile 对照。dump=False(默认)时全部 no-op, 零开销。
# =====================================================================
_DUMP_DIR = "dump_data"
_DUMP_ENABLED = True  # 全局开关, 由 golden 的 dump= 参数设置


def _enable_dump(on):
    """开/关 dump。开时建 dump_data/ 并清空各 dump 文件(一次 golden 调用 = 一份完整 dump)。"""
    global _DUMP_ENABLED
    _DUMP_ENABLED = bool(on)
    if _DUMP_ENABLED:
        os.makedirs(_DUMP_DIR, exist_ok=True)
        for name in ("mm1Res.bin", "mm2ReduceSum.bin", "mm2Res_1.bin",
                     "p_packed_x4.bin", "p_scale_e8m0.bin",
                     "local_group_max.bin", "global_max.bin"):
            with open(os.path.join(_DUMP_DIR, name), "wb"):
                pass  # truncate


def _dump_tensor(t, name):
    """把张量以 raw bytes 追加到 dump_data/<name>(per-chunk append, kernel tile 序)。
    任意 dtype 都按其原生字节写(qfa 用 .numpy().tofile 覆盖; 这里用 append 以支持多 chunk)。"""
    if not _DUMP_ENABLED:
        return
    path = os.path.join(_DUMP_DIR, name)
    arr = t.contiguous().cpu().numpy()
    with open(path, "ab") as f:
        arr.tofile(f)   # 与 qfa 统一: numpy().tofile (append 句柄保留全部 tile; qfa 用字符串=覆盖仅留最后 tile)

# 把 2 个 4-bit FP4 编码 打包成 1 个 uint8 （OCP MX 标准）
def _pack_fp4_e2m1_along_last(idx_tensor):
    """FP4 element 索引(0..7) -> nibble-packed uint8 (low=e[2k], high=e[2k+1], OCP MX)。"""
    *batch, n = idx_tensor.shape
    if n % 2 == 1:
        pad = torch.zeros(*batch, 1, dtype=idx_tensor.dtype, device=idx_tensor.device)
        idx_tensor = torch.cat([idx_tensor, pad], dim=-1)
        n += 1
    pairs = idx_tensor.reshape(*batch, n // 2, 2)
    low = pairs[..., 0].to(torch.uint8) & 0xF
    high = pairs[..., 1].to(torch.uint8) & 0xF
    return (high << 4) | low

# 把 E8M0 整数指数 用uint8表示
def _to_e8m0_uint8(exp_int):
    """整数指数 -> E8M0 (uint8, bias 127): byte = clamp(exp + 127, 0, 254)。"""
    biased = exp_int.to(torch.int64) + 127
    biased = torch.clamp(biased, min=0, max=254)
    return biased.to(torch.uint8)


# =====================================================================
# 第一部分: MXFP4 (E2M1) 量化/反量化/打包 工具 (移植自 mx_quant_fp4_tool.py)
# =====================================================================
# FP4 E2M1: 4bit = S(1)+E(2)+M(1), bias=1。16 个值, 无 inf/NaN(e=3 是正规数 ±4/±6):
#   正规(e>0): (-1)^S * 2^(E-1) * (1 + M/2) -> ±1,±1.5,±2,±3,±4,±6
#   次正规(e=0): (-1)^S * M/2            -> ±0,±0.5
FP4_E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)
E8M0_BIAS = 127                       # E8M0: value = 2^(byte - 127), 范围 2^-127..2^127
FP4_E2M1_EMAX = 2                     # max 正规指数 (6.0 = 1.5*2^2); e8m0 exp = log2(max) - emax

# 枚举 FP4 E2M1 的 16 个编码值 ，生成 {fp32值 → 4-bit编码} 的查表字典
def generate_fp4_e2m1_encoding_table():
    """枚举 16 个 FP4 编码, 返回 {fp32值: 4bit编码}(供 convert 反查编码)。"""
    encodings, values = [], []
    for s in [0, 1]:                    # 符号位
        for e in range(4):              # 2 位指数
            for m in range(2):          # 1 位尾数
                enc = (s << 3) | (e << 1) | m  # S-E-E-M 位序
                val = (-1) ** s * (2.0 ** (e - 1)) * (1.0 + m / 2.0) if e != 0 \
                    else (-1) ** s * (2.0 ** 0) * (m / 2.0)  # 正规 / 次正规
                encodings.append(enc)
                values.append(val)
    fp32_to_encoding = {}
    for enc, val in zip(encodings, values):
        if val not in fp32_to_encoding:  # +0/-0 同 hash, 正零优先
            fp32_to_encoding[val] = enc
    return fp32_to_encoding


FP32_TO_FP4 = generate_fp4_e2m1_encoding_table()  # {fp32值: 4bit编码}

# 最近邻量化 ：把 FP32 张量的每个元素映射到 16 个 FP4 值中最近的那个，返回 FP32 模拟值。
def quantize_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """量化到最近 FP4 值, 返回 FP32 模拟(16 值之一)。"""
    with torch.no_grad():
        expanded = tensor.unsqueeze(-1)  # 末维对齐 16 候选
        idx = torch.argmin(torch.abs(expanded - FP4_E2M1_VALUES.to(tensor.device)), dim=-1)
        return FP4_E2M1_VALUES.to(tensor.device)[idx]

# 为一个 32 元素 block 计算 E8M0 scale（ 2^exp ），使 block 内 max 经过除 scale 后落在 FP4 高精度区 [4, 6]
def compute_e8m0_scale_block(block: torch.Tensor, mode: str = "baseline") -> torch.Tensor:
    """为一个 32 元素 block 算 E8M0 scale(2^exp)。
    baseline: floor(log2(max))-emax(最大化范围, 允许少量饱和); oas: ceil-2(防饱和)。"""
    """
    为一个 block 计算 E8M0 缩放因子 (2 的整数次幂)。

    按 OCP MX 规范:
        scale_exp = floor(log2(max_abs(block))) - emax_elem   (baseline)
        scale_exp = ceil(log2(max_abs(block)))  - emax_elem   (oas, 防饱和)

    参数:
        block:     shape [..., block_size] 的张量
        mode:      "baseline" - floor, 最大化 FP4 范围利用, 允许少量饱和
                   "oas"      - ceil,  保证不饱和, 牺牲一点精度
        emax_elem: 元素格式最大正规化指数, FP4 E2M1 为 2

    返回:
        scale: 标量, 2 的整数次幂, 指数范围 [-127, 127]
    """
    max_abs = torch.max(torch.abs(block))
    if max_abs.item() < 1e-30:          # 全 0 block: scale=1
        return torch.tensor(1.0, device=block.device)
    log_max = torch.log2(max_abs)
    exp = torch.floor(log_max) - FP4_E2M1_EMAX if mode == "baseline" \
        else torch.ceil(log_max) - FP4_E2M1_EMAX
    exp = torch.clamp(exp, -127.0, 127.0)
    return 2.0 ** exp

# 完整的 MXFP4 量化流程 ：沿最后一维，每 32 元素一个 block，每个 block 共享一个 E8M0 scale，先除 scale 归一化再量化到 FP4
def mx_quantize(tensor: torch.Tensor, block_size: int = 32, mode: str = "baseline"):
    """MXFP4 量化(FP32 模拟): 沿最后一维每 32 元素共享一个 E8M0 scale, 再各自量化到 FP4。
    返回 (fp4_sim_fp32[同 shape], scales_real[每 block 一个])。"""
    original_shape = tensor.shape
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    last = tensor.shape[-1]
    pad = 0
    if last % block_size != 0:          # 末维不足 32 倍数: pad 0(不影响 max)
        pad = block_size - (last % block_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad), value=0.0)
    reshaped = tensor.reshape(-1, block_size)   # [num_blocks, 32]
    n_blk = reshaped.shape[0]
    scales = torch.empty(n_blk, device=tensor.device)
    fp4 = torch.empty_like(reshaped)
    for i in range(n_blk):
        sc = compute_e8m0_scale_block(reshaped[i], mode=mode)
        scales[i] = sc
        fp4[i] = quantize_fp4(reshaped[i] / sc)   # 归一化后量化到 FP4
    fp4 = fp4.view(tensor.shape)
    if pad > 0:
        fp4 = fp4[..., :-pad]
    return fp4, scales


# MXFP4 反量化 ：把量化后的 FP32 模拟值乘回 E8M0 scale，恢复为带量化噪声的原始范围。
def mx_dequantize(fp4_vals: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """MXFP4 反量化(FP32 模拟 -> FP32): 每 block 乘回 scale。沿最后一维分块。"""
    original_shape = fp4_vals.shape
    last = original_shape[-1] if fp4_vals.dim() > 0 else 1
    pad = 0
    if fp4_vals.dim() > 0 and last % block_size != 0:
        pad = block_size - (last % block_size)
        fp4_vals = torch.nn.functional.pad(fp4_vals, (0, pad), value=0.0)
    reshaped = fp4_vals.reshape(-1, block_size)
    dq = reshaped * scales.unsqueeze(1).to(reshaped.device)   # scale 广播到 32 元素
    dq = dq.view(fp4_vals.shape)
    if pad > 0:
        dq = dq[..., :last]
    return dq

# 沿任意轴做 mx_quantize → 立即mx_dequantize ，返回带量化噪声的 FP32。用于给 CPU golden 的 Q/K/V 引入与 NPU 一致的量化误差 ，使对比公平
def mxfp4_qdq(t: torch.Tensor, axis: int, block_size: int = 32, mode: str = "baseline") -> torch.Tensor:
    """沿任意 axis 做 MXFP4 qdq round-trip, 返回带量化噪声的 FP32。
    golden 用它给 Q/K/V 引入与 NPU(packed fp4+e8m0)一致的量化误差。"""
    t_last = t.movedim(axis, -1).contiguous()     # 量化轴搬到最后
    fp4_sim, scales = mx_quantize(t_last, block_size=block_size, mode=mode)
    dq = mx_dequantize(fp4_sim, scales, block_size=block_size)
    return dq.movedim(-1, axis).contiguous()      # 搬回原轴

# 把 FP32 模拟值（16 个离散值之一） 反查回 4-bit 编码 （0-15），通过 FP32_TO_FP4 字典
def convert_fp32_simulated_to_fp4_encoding(fp32_sim: torch.Tensor) -> torch.Tensor:
    """FP32 模拟值(必为 16 值之一) -> 4-bit 编码(0-15) uint8, 经 FP32_TO_FP4 字典反查。"""
    flat = fp32_sim.flatten().cpu()
    enc = torch.zeros_like(flat, dtype=torch.uint8)
    for fp32_val, encoding in FP32_TO_FP4.items():
        enc[flat == fp32_val] = encoding
    return enc.to(fp32_sim.device).reshape(fp32_sim.shape)

# 把 4-bit 编码两两打包成 uint8（OCP MX 标准：低 nibble=前元素，高 nibble=后元素）：
def pack_fp4_to_uint8(fp4_enc: torch.Tensor) -> torch.Tensor:
    """2 个 4-bit 编码 -> 1 uint8(低 nibble=elem[2k], 高 nibble=elem[2k+1])。沿最后一维打包。"""
    shape = fp4_enc.shape
    if shape[-1] % 2 != 0:
        raise ValueError(f"最后一维须偶数, 当前 {shape[-1]}")
    reshaped = fp4_enc.reshape(-1, 2)
    packed = (reshaped[:, 1] << 4) | (reshaped[:, 0] & 0x0F)
    return packed.reshape(*shape[:-1], shape[-1] // 2)

# NPU 打包入口 ：完整的 MXFP4 量化 → 编码 → 打包 + E8M0 指数转 biased uint8。全流程整合：
def mxfp4_quantize_pack_last(tensor: torch.Tensor, quant_axis: int,
                             block_size: int = 32, mode: str = "baseline"):
    """NPU 侧打包入口: 量化轴任意, 打包轴=最后一维。
    返回 (packed_fp4 uint8[last=D//2], e8m0_scales uint8 biased[量化轴维=ceil(qd/32)])。
    Q/K 用 quant_axis=-1(沿 head_dim); V 用 -2(沿 seq_k)。"""
    nd = tensor.dim()
    qax = quant_axis % nd
    assert tensor.shape[-1] % 2 == 0, f"最后一维须偶数, 当前 {tensor.shape[-1]}"
    t_q_last = tensor.movedim(qax, -1).contiguous()
    fp4_sim, scales_flat = mx_quantize(t_q_last, block_size=block_size, mode=mode)
    codes = convert_fp32_simulated_to_fp4_encoding(fp4_sim).movedim(-1, qax).contiguous()
    packed = pack_fp4_to_uint8(codes)
    qdim = tensor.shape[qax]
    n_blk = (qdim + block_size - 1) // block_size
    rest = list(t_q_last.shape[:-1])
    scales_real = scales_flat.reshape(*rest, n_blk).movedim(-1, qax).contiguous()
    exp_int = torch.log2(scales_real).round().long()
    e8m0 = (exp_int + E8M0_BIAS).clamp(0, 254).to(torch.uint8)
    return packed, e8m0


# =====================================================================
# 第二部分: blockwise_snap_local P 量化 (移植自 flash_attention_cpu_golden.py)
# 方案 B-snap-local: m_block 立即 floor 到 log2 网格, corr=2^integer 精确 E8M0,
# online-softmax 不变式保持. (memory: BSA mxfp4 kernel 用此方案)
# 注: helper 用 src_dtype=S.dtype, 传入 s_dtype 的 scores 则全程在 s_dtype 算
# (含 m_block/K_block/P_local/bucketize/P_q/corr/P_eff), 与 qfa 一致; 内部 P_local 有一次
# bf16 round-trip. 返回 P_eff(同 s_dtype) + m_new(fp32).
# dump 开启时, 额外落盘 p_packed_x4 / p_scale_e8m0 / local_group_max / global_max(镜像 qfa)。
# =====================================================================
_FP4_POS_LEVELS_FP32 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_FP4_POS_MIDPOINTS_FP32 = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)

# 这是 P 矩阵的 blockwise snap-local 量化 核心算法。逐阶段说明：
# ### 整体目的
# 把 softmax 之前的 scores 矩阵 S （形状 [..., seq_k, nq] ）做 FP4 量化和 per-block E8M0 scale 反量化，得到 P_eff （有效概率，未归一化），
# 供外部 sum(P) 和 PV matmul 在 fp32 上计算
def _blockwise_snap_local_quantize_p_eff(S: torch.Tensor, m_running, seqk_dim: int,
                                          mx_block_size: int = 32, mx_mode: str = "baseline"):
    """对 scores(S, seqk_dim 轴)做 blockwise_snap_local P 量化, 返回 (P_eff, m_new)。
    每 32 元素 block: m_block snap 到 log2 -> P_local(∈[4,8)) -> FP4 round 得 P_q(raw) ->
    corr=2^(K_block-K_new-2) 折 per-block E8M0 -> P_eff=P_q*corr。
    P_eff 未归一化(需外部 /sum); m_new=K_new(fp32 整数, 供 online softmax alpha=exp2(K_diff))。"""
    NEG_INF = float('-inf')
    device = S.device
    src_dtype = S.dtype                       # 全程跟随 S dtype(对齐 qfa: fp16)
    ndim = S.dim()
    skp = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim
    seqk_len = S.shape[skp]
    LN2 = math.log(2.0)

    # 阶段1: pad seqk 到 32 倍数(-inf padding 不影响 max)。
    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pshape = list(S.shape)
        pshape[skp] = pad
        S_padded = torch.cat([S, torch.full(pshape, NEG_INF, dtype=src_dtype, device=device)], dim=skp)
    else:
        S_padded = S
    S_moved = S_padded.movedim(skp, -1)
    other = S_moved.shape[:-1]
    n_blk = S_moved.shape[-1] // mx_block_size
    S_r = S_moved.reshape(*other, n_blk, mx_block_size)

    # 阶段2: m_block_raw -> snap 到 log2 网格。
    m_block_raw = S_r.max(dim=-1).values
    m_block_raw_safe = torch.where(torch.isinf(m_block_raw) & (m_block_raw < 0),
                                   torch.zeros_like(m_block_raw), m_block_raw)  # 全 -inf block -> 0
    K_block = torch.clamp(torch.floor(m_block_raw_safe / LN2), -127.0, 127.0)  # snap: floor(m/ln2)
    if _DUMP_ENABLED: 
        _dump_tensor(K_block.T, "local_group_max.bin")         # dump: 每 block 的 K(镜像 qfa local_group_max, 对齐转置)
    m_block_snap = (K_block - 2) * LN2                     # 使 P_local block max ∈ [4,8)
    # 阶段3: 局部归一化 P_local = exp(S - m_block_snap)。
    P_local = torch.exp(S_r - m_block_snap.unsqueeze(-1))
    P_local_bf = P_local.to(torch.bfloat16).to(P_local.dtype)   # bf16 round-trip(模拟 NPU 中间)
    # 阶段4: FP4 round(bucketize 到 levels)。
    levels = _FP4_POS_LEVELS_FP32.to(device=device, dtype=src_dtype)
    midpoints = _FP4_POS_MIDPOINTS_FP32.to(device=device, dtype=src_dtype)
    idx = torch.bucketize(P_local_bf.contiguous(), midpoints, right=True)  # 落桶 -> level 索引(0-7)
    P_q = levels[idx]                                      # FP4 raw data(未乘 0.25)
    if _DUMP_ENABLED:                                       # dump: P FP4 nibble-packed(镜像 qfa p_packed_x4)
        p_packed_x4 = _pack_fp4_e2m1_along_last(
            idx.reshape(*other, n_blk * mx_block_size).T)    # 先 reshape 转置再 pack, 对齐 qfa
        _dump_tensor(p_packed_x4, "p_packed_x4.bin")

    # 阶段5: K_new = 跨 block 取 max(整数 K 域), 供 corr 与 online softmax。
    K_ij = K_block.max(dim=-1).values
    K_ij_fp32 = K_ij.float()
    m_new = K_ij_fp32 if m_running is None else torch.maximum(m_running, K_ij_fp32)
    K_new_s = m_new.to(src_dtype)                          # cast 回 s_dtype 与 K_block 同算
    if _DUMP_ENABLED: 
        _dump_tensor(K_new_s, "global_max.bin")                # dump: 全局 K_new
    # 阶段6: corr = 2^(K_block-K_new-2) = 0.25*2^(K_diff), per-block E8M0(折 2^-2 dequant 因子)。
    K_diff = K_block - K_new_s.unsqueeze(-1)               # ≤0, s_dtype 整数
    corr = torch.exp2(K_diff - 2.0)
    if _DUMP_ENABLED:                                       # dump: P scale E8M0(镜像 qfa p_scale_e8m0)
        p_scale_e8m0 = _to_e8m0_uint8(K_diff.to(torch.int32) - 2).T.contiguous()
        _dump_tensor(p_scale_e8m0, "p_scale_e8m0.bin")
    # 阶段7: corr 广播回 seqk_len, P_eff = P_q * corr。
    corr_broad = corr.repeat_interleave(mx_block_size, dim=-1)[..., :seqk_len].movedim(-1, skp)
    P_q_moved = P_q.reshape(*other, n_blk * mx_block_size)[..., :seqk_len].movedim(-1, skp)
    P_eff = P_q_moved * corr_broad
    return P_eff, m_new


# =====================================================================
# 第三部分: BSA 数值辅助 + s_dtype 解析
# =====================================================================
_S_DTYPE_MAP = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}

# 把 'fp16'/'bf16'/'fp32' 字符串转为 torch dtype
def _resolve_s_dtype(s_dtype):
    """字符串 s_dtype -> torch dtype。用于将 S/P 中间精度字符串('fp16'/'bf16'/'fp32')转为 torch dtype 类型。"""
    if s_dtype not in _S_DTYPE_MAP:
        raise ValueError(f"s_dtype must be one of {list(_S_DTYPE_MAP)}, got '{s_dtype}'")
    return _S_DTYPE_MAP[s_dtype]

# 数值稳定的 log(sum(exp))，供 lse 参考输出
def _logsumexp_np(scores):
    """log(sum(exp(scores))), 数值稳定(先减 max)。全 mask(max<=-1e9)直接返回 m。
    用于计算 lse(logsumexp), 供参考输出(非 mxfp4 路径, 纯 CPU 验证用)。"""
    s = scores.astype(np.float32)
    m = np.max(s)
    if m <= -1e9:
        return m
    return float(m + np.log(np.sum(np.exp(s - m))))

# qfa 风格造 Q/K/V 数据
def randn_x(shape, x=1.0, dtype=torch.float32, device=None):
    """x * 标准正态, qfa 风格造 Q/K/V 数据(固定种子保证可复现)。"""
    return x * torch.ones(shape, dtype=dtype, device=device)


# =====================================================================
# 第四部分: CPU golden — BNSD block-sparse MXFP4 分块 flash attention (kernel 模拟 + dump)
# 这是【核心 golden】: 分块 flash 结构对齐 qfa _flash_attn_single_batch_kernel + kernel 的
# s2LoopCount*2(2 kv 块/chunk); online softmax 跨 chunk rescale; dump 中间结果到 dump_data/。
# 数据流(per q-block): gather selected kv -> 切 2-block chunk -> [S=Q@K^T fp32->s_dtype->scale
#   -> snap_local P 量化(s_dtype, 含 dump p_packed_x4/p_scale_e8m0/local_group_max/global_max)
#   -> P.float() -> alpha rescale -> sum/PV fp32(含 dump mm1Res/mm2ReduceSum/mm2Res_1)] -> out/=l。
# =====================================================================
def cpu_block_sparse_attention_bnsd_mxfp4_with_lse(query, key, value, block_sparse_mask,
                                                    block_shape, scale_value, num_kv_heads,
                                                    p_quant='snap_local', s_dtype='fp16',
                                                    dump=False):
    """CPU 标杆(BNSD, MXFP4, 分块 flash), 对齐 qfa flash golden 的 dtype 流 + kernel tile 结构:
       Q/K/V mxfp4 qdq(fp32); per q-block + 2-kv-block chunk + online softmax;
       S fp32 matmul -> s_dtype -> scale; P 量化(snap_local)与 P 本身在 s_dtype; sum/PV 在 fp32。
       dump=True 时落盘中间结果(dump_data/*.bin, per-chunk append)。
       p_quant: 'snap_local'|'none'; s_dtype: 'fp16'(默认,qfa 对齐)|'bf16'|'fp32'。"""
    if dump:
        _enable_dump(True)
    try:
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        v = value.to(torch.float32)
        B, N, S, D = q.shape
        N2 = k.shape[1]
        S2 = k.shape[2]                       # KV 序列长(不与 q 的 S 混)
        block_x, block_y = int(block_shape[0]), int(block_shape[1])
        ceil_q = (S + block_x - 1) // block_x
        ceil_kv = (S2 + block_y - 1) // block_y
        s_torch_dtype = _resolve_s_dtype(s_dtype)

        # Q/K/V mxfp4 qdq -> fp32(引入与 NPU 一致的量化噪声)
        q_q = mxfp4_qdq(q, axis=-1)            # Q/K: head_dim(rowwise)
        k_q = mxfp4_qdq(k, axis=-1)
        v_q = mxfp4_qdq(v, axis=2)            # V: seq_k(columnwise)

        out = torch.zeros((B, N, S, D), dtype=torch.float32)
        lse = torch.full((B, N, S, 1), float('-inf'), dtype=torch.float32)
        scale_s = torch.as_tensor(scale_value, dtype=s_torch_dtype)  # scale round 到 s_dtype

        for batch_idx in range(B):
            for qn_idx in range(N):
                kvn_idx = qn_idx // N2                    # GQA 映射(query 头 -> kv 头)
                for q_block_idx in range(ceil_q):      # ── q-block(kernel q-tile)──
                    q_start = q_block_idx * block_x
                    q_end = min(q_start + block_x, S)
                    if q_start >= S:
                        break
                    # 该 q-block允许 attend 的 kv 块(ascending), 同 q-block内所有 q-token 共享。
                    selected_kv_blocks = [j for j in range(ceil_kv) if block_sparse_mask[batch_idx, qn_idx, q_block_idx, j] != 0]
                    if not selected_kv_blocks:             # 无 kv 块: out 保持 0, lse 保持 -inf
                        continue
                    # 切成 2-kv-block chunk(对齐 kernel s2LoopCount*2)。
                    chunks = [selected_kv_blocks[i:i + 2] for i in range(0, len(selected_kv_blocks), 2)]
                    Q_block = q_q[batch_idx, qn_idx, q_start:q_end]              # [num_q_tokens, D] fp32
                    num_q_tokens = q_end - q_start
                    running_max_k = None                            # online running max(None=首块)
                    running_sum_p = torch.zeros(num_q_tokens, dtype=torch.float32)
                    output_acc = torch.zeros((num_q_tokens, D), dtype=torch.float32)
                    lse_running_max = torch.full((num_q_tokens,), float('-inf'), dtype=torch.float32)  # online lse
                    lse_running_sum = torch.zeros(num_q_tokens, dtype=torch.float32)
                    for chunk in chunks:                   # ── 2-kv-block chunk(kernel kv-tile)──
                        positions = []
                        for j in chunk:
                            kv_start = j * block_y
                            kv_end = min(kv_start + block_y, S2)
                            positions.extend(range(kv_start, kv_end))
                        if not positions:
                            continue
                        positions_tensor = torch.as_tensor(positions, dtype=torch.long)
                        K_block = k_q[batch_idx, kvn_idx][positions_tensor]            # [num_kv_tokens, D] fp32
                        V_block = v_q[batch_idx, kvn_idx][positions_tensor]            # [num_kv_tokens, D] fp32
                        # S: fp32 matmul -> 降到 s_dtype -> s_dtype 上乘 scale
                        scores_fp32 = (K_block @ Q_block.t()).to(torch.float32)   # [num_kv_tokens, num_q_tokens] (DN: S=K@Q^T, 对齐 qfa s_layout="DN" + kernel)
                        if _DUMP_ENABLED:   
                            _dump_tensor(scores_fp32.to(s_torch_dtype), "mm1Res.bin")   # dump: QK result(对应 kernel desc=5/6)
                        scores_sdtype = scores_fp32.to(s_torch_dtype) * scale_s
                        if p_quant == 'snap_local':
                            # P 量化在 s_dtype; snap_local 内部已 dump p_packed_x4/p_scale_e8m0/
                            # local_group_max/global_max。running_max_k 跨 chunk online。
                            P_eff_sdtype, new_max_k = _blockwise_snap_local_quantize_p_eff(
                                scores_sdtype, m_running=running_max_k, seqk_dim=0)  # DN: seqk=dim0
                            P_eff_sdtype = torch.nan_to_num(P_eff_sdtype, nan=0.0)
                            P_fp32 = P_eff_sdtype.float()      # 升 fp32 供 sum/PV
                            # rescale_alpha = exp2(K_old - K_new)(snap_local 的 m 存 K 整数, 位精确)
                            rescale_alpha = (torch.zeros(num_q_tokens, dtype=torch.float32) if running_max_k is None
                                     else torch.exp2(running_max_k.float() - new_max_k.float()))
                        else:  # 'none': 纯 softmax, P 在 s_dtype, online 用实数 max
                            scores_max = scores_sdtype.max(dim=0).values   # DN: 沿 seqk=dim0
                            scores_max_safe = torch.where(torch.isinf(scores_max) & (scores_max < 0), torch.zeros_like(scores_max), scores_max)
                            P_softmax = torch.exp(scores_sdtype - scores_max_safe.unsqueeze(0))  # [num_kv_tokens, num_q_tokens] (DN)
                            P_fp32 = torch.nan_to_num(P_softmax, nan=0.0).float()
                            new_max_k = scores_max_safe.float()
                            rescale_alpha = (torch.zeros(num_q_tokens, dtype=torch.float32) if running_max_k is None
                                     else torch.exp(running_max_k - new_max_k))
                        # sum(P) / PV 均在 fp32
                        sum_P = P_fp32.sum(dim=0)                # [num_q_tokens] (DN: 沿 seqk=dim0)
                        if _DUMP_ENABLED:   
                            _dump_tensor(sum_P, "mm2ReduceSum.bin")  # dump: rowsum(对应 desc=21/8/...)
                        PV_result = V_block.t() @ P_fp32                      # [D, num_q_tokens] (DN: V.t()@P, 对应 kernel PV_i dump)
                        if _DUMP_ENABLED:   
                            _dump_tensor(PV_result, "mm2Res_1.bin")              # dump: PV(对应 desc=36/18/19)
                        running_sum_p = rescale_alpha * running_sum_p + sum_P
                        output_acc = rescale_alpha.unsqueeze(-1) * output_acc + (P_fp32.t() @ V_block)  # [num_q_tokens, D] (DN: P.t()@V)
                        running_max_k = new_max_k
                        # online lse(用 fp32 scaled S, 仅参考; mxfp4 NPU 不返回 lse)
                        scores_scaled = scores_fp32 * scale_value             # fp32 scaled scores
                        max_chunk = scores_scaled.max(dim=0).values   # DN: 沿 seqk=dim0
                        max_chunk = torch.where(torch.isinf(max_chunk) & (max_chunk < 0), torch.full_like(max_chunk, -1e10), max_chunk)
                        lse_new_max = torch.maximum(lse_running_max, max_chunk)
                        lse_alpha = (torch.zeros(num_q_tokens, dtype=torch.float32) if bool((lse_running_max == float('-inf')).all())
                                 else torch.exp(lse_running_max - lse_new_max))
                        lse_running_sum = lse_alpha * lse_running_sum + torch.exp(scores_scaled - lse_new_max.unsqueeze(0)).sum(dim=0)  # [num_q_tokens] (DN)
                        lse_running_max = lse_new_max
                    sum_p_safe = torch.where(running_sum_p > 0, running_sum_p, torch.ones_like(running_sum_p))
                    output_acc = output_acc / sum_p_safe.unsqueeze(-1)     # 归一化
                    out[batch_idx, qn_idx, q_start:q_end] = output_acc
                    lse_out = torch.where(lse_running_sum > 0, lse_running_max + torch.log(lse_running_sum),
                                        torch.full_like(lse_running_max, float('-inf')))
                    lse[batch_idx, qn_idx, q_start:q_end, 0] = lse_out
        return out.to(torch.bfloat16), lse.to(torch.float32)
    finally:
        if dump:
            _enable_dump(False)


# =====================================================================
# 第五部分: check_result / compare (移植自 quant_flash_attn_golden.py)
# 判据: isclose(rtol,atol) 满足率; PASS 需 满足率>=99.5% 且 maxRE<10。
# =====================================================================
def check_result(expect, result, data_type, pct_thd=0.005):
    """expect=golden, result=NPU。返回 (fulfill_percent, "Pass"/"Failed")。"""
    real_data = result.cpu().numpy().flatten()
    data_compe = expect.cpu().numpy().flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        return 100.0, "Pass"
    if real_data.size != data_compe.size:
        return 0.0, "Failed"
    if data_type == 'bfloat16':
        rtol, atol, max_diff_hd = 0.0078125, 0.0001, 10.0      # 1/128
    else:
        rtol, atol, max_diff_hd = 0.005, 0.000025, 10.0
    diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32),
                             rtol=rtol, atol=atol, equal_nan=True)
    err_idx = np.where(diff_result != np.array((True,)))[0]
    split_count = real_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    diff_abs = np.abs(data_compe - real_data)
    b = np.add(np.maximum(np.abs(real_data), np.abs(data_compe)), 10e-10)
    err_diff = (diff_abs / b)[err_idx]
    pct_thd = (1 - pct_thd) * 100.0
    result_str = "Pass" if fulfill_percent >= pct_thd else "Failed"
    if len(err_diff) > 0 and np.max(err_diff) >= max_diff_hd:
        result_str = "Failed"
    return fulfill_percent, result_str


def compare(golden, real, src_dtype):
    """两边转 fp32 后调 check_result。"""
    if golden is not None and real is not None:
        return check_result(golden.to(torch.float32), real.cpu().to(torch.float32), src_dtype)
    return 0.0, "Failed"


# =====================================================================
# 第六部分: e8m0 -> V3 形状 重排 (假设 A: 连续配对, 待 NPU 核验)
# =====================================================================
# Q/K 的 E8M0 scale 从 [B,N,S,D//32] 重排为 V3 算子 doc 规定的 [B,N,S,D//64,2] 形状（每 2 个连续的 32-block 配对成 1 个 64-group）
def _e8m0_qk_to_v3(e8m0_uint8):
    """q/k e8m0 [B,N,S,D//32] -> [B,N,S,D//64,2] (连续配对)。"""
    *lead, nd = e8m0_uint8.shape
    return e8m0_uint8.reshape(*lead, nd // 2, 2).contiguous().view(torch.float8_e8m0fnu)


def _e8m0_v_to_v3(e8m0_uint8):
    """v e8m0 [B,N2,S2//32,D] -> [B,N2,S2//64,D,2] (连续配对 + 2 维挪到 D 后)。"""
    B, N2, nb, D = e8m0_uint8.shape
    return e8m0_uint8.reshape(B, N2, nb // 2, 2, D).permute(0, 1, 2, 4, 3).contiguous().view(torch.float8_e8m0fnu)


# =====================================================================
# 第七部分: qfa 风格 driver — 造数据/quantize/golden/NPU/compare
# =====================================================================
def _build_block_sparse_mask(b, n1, ceil_q, ceil_kv, kind='dense'):
    """官方 block_sparse_mask 网格 (B,n1,ceil_q,ceil_kv) int8: !=0 表示该 q 块 attend 该 kv 块。"""
    if kind == 'dense':
        return torch.ones(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
    if kind == 'sparse':  # 每 q_block 随机 attend 1~ceil_kv 个 kv 块
        m = torch.zeros(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
        for i in range(ceil_q):
            n_attend = torch.randint(1, ceil_kv + 1, (1,)).item()  # 随机选 1~ceil_kv 个
            idx = torch.randperm(ceil_kv)[:n_attend]                # 随机打乱取前 n_attend 个
            m[:, :, i, idx] = 1
        return m
    raise ValueError(f"unsupported mask kind: {kind}")


def run_bsa_mxfp4_eager(b, n1, n2, g, s1, s2, qk_d, v_d=None,
                        input_layout='BNSD', out_dtype='bfloat16',
                        block_shape=(128, 128), mask_kind='dense',
                        inner_precise=4, quant_mode=2,
                        p_quant='snap_local', s_dtype='fp16',
                        dump=False, device_id=0):
    """qfa 风格: fp32 造数据 -> mxfp4 打包 -> CPU golden -> NPU -> check_result.
       p_quant: 'snap_local'|'none'; s_dtype: 'fp16'(默认)|'bf16'|'fp32'; dump: 落盘中间结果。
       返回 (fulfill_percent, result_str). NPU 不可用时返回 (-1.0, 'SKIP')."""
    assert input_layout == 'BNSD', "首版仅支持 BNSD"
    assert n1 == n2 * g, f"GQA 约束 n1==n2*g: {n1} vs {n2}*{g}"
    assert block_shape[0] in (64, 128), f"MXFP4 blockShapeX 仅支持 64/128, 当前 {block_shape[0]}"
    assert qk_d in (64, 128), f"MXFP4 D 仅支持 64/128, 当前 {qk_d}"
    if v_d is None:
        v_d = qk_d
    scale = 1.0 / math.sqrt(qk_d)
    src_dtype = out_dtype

    # 1. 造 fp32 数据(qfa 风格 randn, 固定种子)
    query = randn_x((b, n1, s1, qk_d), 1.0).to(torch.float32)
    key = randn_x((b, n2, s2, qk_d), 1.0).to(torch.float32)
    value = randn_x((b, n2, s2, v_d), 1.0).to(torch.float32)

    # 2. 官方 block_sparse_mask 网格
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (s1 + block_x - 1) // block_x
    ceil_kv = (s2 + block_y - 1) // block_y
    mask = _build_block_sparse_mask(b, n1, ceil_q, ceil_kv, mask_kind)

    # 3. CPU golden(分块 flash + 可选 dump)
    cpu_out, _ = cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
        query, key, value, mask, block_shape, scale, n2,
        p_quant=p_quant, s_dtype=s_dtype, dump=dump)

    # 4. mxfp4 打包喂 NPU: Q/K axis=-1, V axis=-2
    q_packed, q_e8m0 = mxfp4_quantize_pack_last(query, quant_axis=-1)
    k_packed, k_e8m0 = mxfp4_quantize_pack_last(key, quant_axis=-1)
    v_packed, v_e8m0 = mxfp4_quantize_pack_last(value, quant_axis=-2)
    assert q_packed.dtype == k_packed.dtype == v_packed.dtype == torch.uint8, "packed fp4 须为 uint8"
    assert q_e8m0.dtype == k_e8m0.dtype == v_e8m0.dtype == torch.uint8, "e8m0 scale 须为 uint8"
    q_scale = _e8m0_qk_to_v3(q_e8m0)
    k_scale = _e8m0_qk_to_v3(k_e8m0)
    v_scale = _e8m0_v_to_v3(v_e8m0)

    # 5. NPU 调用(照搬 test_pv_systematic.py)
    if not _HAS_BSA_V3:
        return -1.0, "SKIP"
    torch_npu.npu.set_device(device_id)
    npu_out, _ = block_sparse_attention_v3(
        q_packed.contiguous().npu(), k_packed.contiguous().npu(),
        v_packed.contiguous().npu(), mask.contiguous().npu(), list(block_shape),
        [s1] * b, [s2] * b,
        num_heads=n1, num_key_value_heads=n2,
        q_input_layout="BNSD", kv_input_layout="BNSD",
        mask_type=0, scale_value=scale, inner_precise=inner_precise, block_size=0,
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=False, quant_mode=quant_mode, dst_type_max=0,
        q_dequant_scale=q_scale.npu(), k_dequant_scale=k_scale.npu(), v_dequant_scale=v_scale.npu(),
        attention_out_dtype=torch.bfloat16,
    )
    torch_npu.npu.synchronize()

    # 6. 比对(qfa check_result)
    fulfill_percent, result = compare(cpu_out, npu_out, src_dtype)
    return fulfill_percent, result


# =====================================================================
# 第八部分: 测试类 — 4 条 BNSD 用例
# =====================================================================
class TestBSAMxfp4Golden(TestCase):
    """BSA mxfp4 正向 vs CPU golden, qfa check_result 判据。
    注: quantMode=2 仅 Ascend 950PR/950DT 支持(V3 doc L680), 故不锁 SupportedDevices,
    靠 @SkipIfNotGteCANNVersion + 运行期能力检查。"""

    _cfg_overrides = {}  # conftest.py 注入 CLI 覆盖参数

    def _run(self, **cfg):
        """共用: 跳过无 NPU, 跑 run_bsa_mxfp4_eager, 断言 result=='Pass'。
        CLI 参数(conftest.py --b/--n1/--n2/--s1/--s2/--qk_d)覆盖 cfg。"""
        if not _HAS_BSA_V3:
            self.skipTest(f"cann_ops_transformer.ops.block_sparse_attention_v3 不可用: "
                          f"{_BSA_V3_IMPORT_ERROR}")
        # conftest 覆盖的参数字典(只覆盖 cfg 中已有的 key)
        ov = getattr(self, '_cfg_overrides', {})
        cfg_merged = {**cfg, **{k: v for k, v in ov.items() if k in cfg}}
        pct, res = run_bsa_mxfp4_eager(**cfg_merged)
        assert res == "Pass", (f"BSA mxfp4 case 失败: {cfg}\n  fulfill={pct}% result={res}\n"
                               f"  (若 fulfill<99.5% 或 maxRE>=10, 先核验 spec 假设 A/B/C/D; "
                               f"4 用例同挂优先怀疑 C(p_quant) 或 D(s_dtype), 改 p_quant='none' / s_dtype='bf16'|'fp32' 再试)")
        print(f"[PASS] {cfg_merged.get('mask_kind','dense')} b={cfg_merged['b']} n1={cfg_merged['n1']} n2={cfg_merged['n2']} "
              f"s1={cfg_merged['s1']} s2={cfg_merged['s2']} block={cfg_merged['block_shape']} s_dtype={cfg_merged.get('s_dtype','fp16')} "
              f"fulfill={pct:.4f}%")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_dense(self, device="npu"):
        """① BNSD dense 单块: 基本精度(block_shape=[128,128], S=128 -> 1x1)。"""
        self._run(b=2, n1=8, n2=8, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense')

    # @SkipIfNotGteCANNVersion("9.0.0")
    # def test_bsa_mxfp4_bnsd_multi_block(self, device="npu"):
    #     """② BNSD 多块 dense: block_shape=[64,128], S=256 -> ceil_q=4, ceil_kv=2 (4x2)。"""
    #     self._run(b=2, n1=8, n2=8, g=1, s1=256, s2=256, qk_d=128,
    #               block_shape=[64, 128], mask_kind='dense')

    # @SkipIfNotGteCANNVersion("9.0.0")
    # def test_bsa_mxfp4_bnsd_sparse_mask(self, device="npu"):
    #     """③ BNSD 稀疏 mask: 每 q_block i attend kv_block i%ceil_kv(S=256, [64,128] -> 4x2)。"""
    #     self._run(b=2, n1=8, n2=8, g=1, s1=256, s2=256, qk_d=128,
    #               block_shape=[64, 128], mask_kind='sparse')

    # @SkipIfNotGteCANNVersion("9.0.0")
    # def test_bsa_mxfp4_bnsd_gqa(self, device="npu"):
    #     """④ BNSD GQA: g=2, N1=8/N2=4(block_shape=[128,128] -> 1x1)。"""
    #     self._run(b=2, n1=8, n2=4, g=2, s1=128, s2=128, qk_d=128,
    #               block_shape=[128, 128], mask_kind='dense')


if __name__ == "__main__":
    run_tests()
