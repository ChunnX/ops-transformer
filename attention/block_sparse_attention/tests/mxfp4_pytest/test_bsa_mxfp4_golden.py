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
import unittest
import numpy as np
import torch

# =====================================================================
# torch_npu 导入与 CPU-only 模式说明
# =====================================================================
# 为什么要这层 guard? 不是 `import torch_npu` 本身会崩, 而是继承的基类会自启动 NPU:
#
#   torch_npu 框架(pip 包)自带一个测试基类 torch_npu.testing.testcase.TestCase, 它的
#   setUpClass 写死了 set_npu_device():
#       @classmethod
#       def setUpClass(cls):           # testcase.py:70-72 (框架自带, 非本文件)
#           cls.npu_device = set_npu_device()   # -> torch.npu.set_device -> aclInit
#
#   框架设计意图(见 testcase.py L1-3 注释): import 阶段不初始化 NPU(惰性), 但把初始化推迟到
#   setUpClass —— 即 pytest/unittest 跑测试前的那一刻。所以一旦 `class TestX(TestCase)` 继承
#   它, 就等于声明"我在 NPU 环境跑", 框架会在每个测试类启动时帮你选卡 + aclInit。
#   在 NPU 机器上这是便利; 在纯 CPU 开发机(无 /dev/davinci 或 aclInit 507008)上就崩。
#
#   触发链: pytest 收集测试类 -> 自动调 setUpClass -> 调基类 setUpClass
#           -> set_npu_device -> torch.npu.set_device(0) -> _lazy_init -> aclInit -> 失败
#
#   解法: CPU-only 时不继承框架那个 TestCase, 改继承纯 unittest.TestCase(空 setUpClass)。
#   用环境变量 BSA_CPU_ONLY=1 切换继承源, 从根上切断 setUpClass -> set_npu_device 链。
#
# 另一个坑: 较新 torch 在 `import torch` 时会自动加载已装后端扩展(含 torch_npu), 若
#   torch_npu 缺 libhccl.so(没 source CANN)会在 import torch 阶段崩。需同时设
#   TORCH_DEVICE_BACKEND_AUTOLOAD=0 禁用自动加载(见 run_cpu.sh)。
# =====================================================================
# CPU-only 模式: BSA_CPU_ONLY=1 强制走 dummy 路径(不继承框架 TestCase)。
_FORCE_CPU = os.environ.get("BSA_CPU_ONLY", "0") == "1" #0 npu 1 cpu

# torch_npu 仅 NPU 环境有; guard 使本模块纯 CPU 也能 import(golden 纯 CPU 可跑, 交叉验证靠这个)。
try:
    if _FORCE_CPU:
        raise ImportError("BSA_CPU_ONLY=1: 强制 CPU-only, 跳过真实 torch_npu")
    import torch_npu  # noqa: F401  (NPU 环境)
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion
except Exception:  # CPU-only: 提供 dummy, 使模块可导入
    torch_npu = None

    class TestCase(unittest.TestCase):  # 继承 unittest: 提供 skipTest; 无 setUpClass 不触发 NPU init
        pass

    def run_tests():
        unittest.main()

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

# 详细打印对比结果的工具(从 compare_utils.py 导入)。
# PRINTF_CHECK_RES=False(默认): 走本文件原 check_result(静默, 只返回满足率);
# PRINTF_CHECK_RES=True:        走 compare_utils.check_result_printf_res(打印逐元素明细)。
from compare_utils import check_result_printf_res

# 固定随机种子: 让 randn 造的 Q/K/V 可复现。
RAND_SEED = 21
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)

# conftest.py 的 pytest_configure 注入 CLI 覆盖参数(供 _run 读取)。
# 注: pytest_generate_tests 对非参数化测试不触发, 故用 pytest_configure + 模块级全局。
_CLI_OVERRIDES = {}


# =====================================================================
# 第零部分: dump 基础设施 (镜像 qfa flash_attention_cpu_golden.py 的 dump 机制)
# golden dump=True 时, 把每个 chunk(kernel tile)的中间结果以 raw bytes 追加到 dump_data/*.bin,
# 便于与 NPU kernel 的 GM tile dump 逐 tile 对照。dump=False(默认)时全部 no-op, 零开销。
# =====================================================================
_DUMP_DIR = "/srv/workspace/mnt1/work_code/lwg/0707/ops-transformer_0703/attention/block_sparse_attention/tests/mxfp4_pytest/dump_data"
_DUMP_ENABLED = False  # 全局开关, 默认关; golden 的 dump=True 时由 _enable_dump(True) 开启(并建 dump_data/ 目录)
PRINTF_CHECK_RES = True


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
    # 全1case
    # return x * torch.ones(shape, dtype=dtype, device=device)
    # 任意case
    data_range=[-x, x]
    return rand_range(shape, data_range, dtype=dtype, device=device)

def rand_range(shape, data_range=[-10, 10], dtype=torch.bfloat16, device=None):
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(shape, dtype=dtype, device=device)

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

        for b in range(B):
            for n in range(N):
                n2 = n % N2                    # GQA 取模映射(query 头 n -> kv 头 n%N2)
                for qb in range(ceil_q):      # ── q-block(kernel q-tile)──
                    q_lo = qb * block_x
                    q_hi = min(q_lo + block_x, S)
                    if q_lo >= S:
                        break
                    # 该 q-block允许 attend 的 kv 块(ascending), 同 q-block内所有 q-token 共享。
                    sel_j = [j for j in range(ceil_kv) if block_sparse_mask[b, n, qb, j] != 0]
                    if not sel_j:             # 无 kv 块: out 保持 0, lse 保持 -inf
                        continue
                    # 切成 2-kv-block chunk(对齐 kernel s2LoopCount*2)。
                    chunks = [sel_j[i:i + 2] for i in range(0, len(sel_j), 2)]
                    Q_blk = q_q[b, n, q_lo:q_hi]              # [nq, D] fp32
                    nq = q_hi - q_lo
                    m_run = None                            # online running max(None=首块)
                    l_run = torch.zeros(nq, dtype=torch.float32)
                    acc = torch.zeros((nq, D), dtype=torch.float32)
                    m_lse = torch.full((nq,), float('-inf'), dtype=torch.float32)  # online lse
                    l_lse = torch.zeros(nq, dtype=torch.float32)
                    for chunk in chunks:                   # ── 2-kv-block chunk(kernel kv-tile)──
                        positions = []
                        for j in chunk:
                            st = j * block_y
                            ed = min(st + block_y, S2)
                            positions.extend(range(st, ed))
                        if not positions:
                            continue
                        pos = torch.as_tensor(positions, dtype=torch.long)
                        K_blk = k_q[b, n2][pos]            # [npos, D] fp32
                        V_blk = v_q[b, n2][pos]            # [npos, D] fp32
                        # S: fp32 matmul -> 降到 s_dtype -> s_dtype 上乘 scale
                        S_fp32 = (K_blk @ Q_blk.t()).to(torch.float32)   # [npos, nq] (DN: S=K@Q^T, 对齐 qfa s_layout="DN" + kernel)
                        if _DUMP_ENABLED: 
                            _dump_tensor(S_fp32.to(s_torch_dtype), "mm1Res.bin")   # dump: QK result(对应 kernel desc=5/6)
                        S_s = S_fp32.to(s_torch_dtype) * scale_s
                        if p_quant == 'snap_local':
                            # P 量化在 s_dtype; snap_local 内部已 dump p_packed_x4/p_scale_e8m0/
                            # local_group_max/global_max。m_run 跨 chunk online。
                            P_eff_s, m_new = _blockwise_snap_local_quantize_p_eff(
                                S_s, m_running=m_run, seqk_dim=0)  # DN: seqk=dim0
                            P_eff_s = torch.nan_to_num(P_eff_s, nan=0.0)
                            P_fp32 = P_eff_s.float()      # 升 fp32 供 sum/PV
                            # alpha = exp2(K_old - K_new)(snap_local 的 m 存 K 整数, 位精确)
                            alpha = (torch.zeros(nq, dtype=torch.float32) if m_run is None
                                     else torch.exp2(m_run.float() - m_new.float()))
                        else:  # 'none': 纯 softmax, P 在 s_dtype, online 用实数 max
                            m_s = S_s.max(dim=0).values   # DN: 沿 seqk=dim0
                            m_safe_s = torch.where(torch.isinf(m_s) & (m_s < 0), torch.zeros_like(m_s), m_s)
                            P_ij = torch.exp(S_s - m_safe_s.unsqueeze(0))  # [npos, nq] (DN)
                            P_fp32 = torch.nan_to_num(P_ij, nan=0.0).float()
                            m_new = m_safe_s.float()
                            alpha = (torch.zeros(nq, dtype=torch.float32) if m_run is None
                                     else torch.exp(m_run - m_new))
                        # sum(P) / PV 均在 fp32
                        reduce_sum = P_fp32.sum(dim=0)                # [nq] (DN: 沿 seqk=dim0)
                        if _DUMP_ENABLED: 
                            _dump_tensor(reduce_sum, "mm2ReduceSum.bin")  # dump: rowsum(对应 desc=21/8/...)
                        pv = V_blk.t() @ P_fp32                      # [D, nq] (DN: V.t()@P, 对应 kernel PV_i dump)
                        if _DUMP_ENABLED: 
                            _dump_tensor(pv, "mm2Res_1.bin")              # dump: PV(对应 desc=36/18/19)
                        l_run = alpha * l_run + reduce_sum
                        acc = alpha.unsqueeze(-1) * acc + (P_fp32.t() @ V_blk)  # [nq, D] (DN: P.t()@V)
                        m_run = m_new
                        # online lse(用 fp32 scaled S, 仅参考; mxfp4 NPU 不返回 lse)
                        s_scaled = S_fp32 * scale_value             # fp32 scaled scores
                        m_ch = s_scaled.max(dim=0).values   # DN: 沿 seqk=dim0
                        m_ch = torch.where(torch.isinf(m_ch) & (m_ch < 0), torch.full_like(m_ch, -1e10), m_ch)
                        m_new_lse = torch.maximum(m_lse, m_ch)
                        a_lse = (torch.zeros(nq, dtype=torch.float32) if bool((m_lse == float('-inf')).all())
                                 else torch.exp(m_lse - m_new_lse))
                        l_lse = a_lse * l_lse + torch.exp(s_scaled - m_new_lse.unsqueeze(0)).sum(dim=0)  # [nq] (DN)
                        m_lse = m_new_lse
                    l_safe = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
                    acc = acc / l_safe.unsqueeze(-1)     # 归一化
                    out[b, n, q_lo:q_hi] = acc
                    lse_q = torch.where(l_lse > 0, m_lse + torch.log(l_lse),
                                        torch.full_like(m_lse, float('-inf')))
                    lse[b, n, q_lo:q_hi, 0] = lse_q
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
    """两边转 fp32 后调 check_result。
    PRINTF_CHECK_RES=False(默认): 走本文件原 check_result(静默, 只返回满足率/结果)。
    PRINTF_CHECK_RES=True:        走 compare_utils.check_result_printf_res(逐元素打印对比明细)。"""
    if golden is None or real is None:
        return 0.0, "Failed"
    golden_fp32 = golden.to(torch.float32)
    real_fp32 = real.cpu().to(torch.float32)
    if PRINTF_CHECK_RES:
        return check_result_printf_res(golden_fp32, real_fp32, src_dtype)
    return check_result(golden_fp32, real_fp32, src_dtype)


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
    if False :
        print(f"[run_bsa_mxfp4_eager] shapes: b={b} n1={n1} n2={n2} g={g} "
          f"s1={s1} s2={s2} qk_d={qk_d} v_d={v_d} block_shape={block_shape} "
          f"layout={input_layout} out_dtype={out_dtype} mask={mask_kind} "
          f"inner_precise={inner_precise} quant_mode={quant_mode} "
          f"p_quant={p_quant} s_dtype={s_dtype} dump={dump} device_id={device_id}")
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

    
    # CPU-only 模式(_FORCE_CPU 或无 NPU 算子): 无 NPU 可比对, 直接 SKIP, 不做任何 golden 自检。
    if _FORCE_CPU or not _HAS_BSA_V3:
        print(f"cpu_out: {cpu_out}")
        return -1.0, "SKIP"

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
        """共用: 跑 run_bsa_mxfp4_eager, 断言 result=='Pass'。
        CLI 参数(conftest.py --b/--n1/--n2/--s1/--s2/--qk_d/--dump)覆盖 cfg 或新增 key。
        CPU-only 模式(BSA_CPU_ONLY=1 或无 NPU 算子): 无 NPU 可比对, 直接 SKIP。"""
        # CLI 覆盖参数(conftest.py pytest_configure 注入到模块级 _CLI_OVERRIDES)。
        # 覆盖 cfg 已有 key + 注入新 key 如 dump。
        cfg_merged = {**cfg, **_CLI_OVERRIDES}
        pct, res = run_bsa_mxfp4_eager(**cfg_merged)
        cpu_mode = _FORCE_CPU or not _HAS_BSA_V3
        if res == "SKIP":
            if cpu_mode:
                self.skipTest("CPU-only 模式: 无 NPU 算子可比对, SKIP")
            else:
                self.skipTest(f"cann_ops_transformer.ops.block_sparse_attention_v3 不可用: "
                              f"{_BSA_V3_IMPORT_ERROR}")
        assert res == "Pass", (f"BSA mxfp4 case 失败: {cfg}\n  fulfill={pct}% result={res}\n"
                               f"  (NPU: 若 fulfill<99.5% 或 maxRE>=10, 先核验 spec 假设 A/B/C/D; "
                               f"4 用例同挂优先怀疑 C(p_quant) 或 D(s_dtype), 改 p_quant='none' / s_dtype='bf16'|'fp32' 再试)")
        mode_tag = "NPU vs golden"
        print(f"[PASS][{mode_tag}] {cfg_merged.get('mask_kind','dense')} b={cfg_merged['b']} n1={cfg_merged['n1']} n2={cfg_merged['n2']} "
              f"s1={cfg_merged['s1']} s2={cfg_merged['s2']} block={cfg_merged['block_shape']} s_dtype={cfg_merged.get('s_dtype','fp16')} "
              f"fulfill={pct:.4f}%")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs128_d128(self, device="npu"):
        """① BNSD dense 单块: 基本精度(block_shape=[128,128], S=128 -> 1x1)。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense')
    
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs256_d128(self, device="npu"):
        """① BNSD dense 单块: 基本精度(block_shape=[128,128], S=128 -> 1x1)。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense')
    
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs512_d128(self, device="npu"):
        """① BNSD dense 单块: 基本精度(block_shape=[128,128], S=128 -> 1x1)。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense')



if __name__ == "__main__":
    run_tests()
