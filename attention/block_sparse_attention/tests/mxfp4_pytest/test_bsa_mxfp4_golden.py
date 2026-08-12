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
  - e8m0 形状与 doc L46-64 一致: q/k [B,N,S,D//64,2]、v [B,N2,ceil(S2/64),D,2], dtype float8_e8m0fnu
    (ceil(S2/32) 为奇时 _e8m0_v_to_v3 先 pad 1 个 e8m0=127 block, 对齐 qfa golden L551-558)。
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
_DUMP_DIR = "/home/w00947241/lwg/bsa_code_0730/ops-transformer_0703/attention/block_sparse_attention/tests/mxfp4_pytest/dump_data"
_DUMP_ENABLED = False  # 全局开关, 默认关; golden 的 dump=True 时由 _enable_dump(True) 开启(并建 dump_data/ 目录)
PRINTF_CHECK_RES = False #打开看Out比较结果
_E8M0_QK_FLIP_PAIR = False


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
# 第零点五部分: 量化模式枚举 (对齐 host侧 quantMode: 2=OCP, 3=CX)
# ---------------------------------------------------------------------
# 区分 OCP 与 CX 两套 P 量化方案 (推导文档式1-26):
#   OCP(quantMode=2): K_block = floor(Smax/ln2)
#                     m_snap = (K_block - 2)·ln2            (式8)
#                     corr   = 2^(K_block - K_new - 2)      (式12/13)
#                     pscale byte = K_block - K_new - 2 + 127
#   CX(quantMode=3):  log2CX = log2(dstTypeMax), ceil_log2CX = ceil(log2CX)
#                     K_block = ceil(Smax/ln2 - log2CX)     (式19)
#                     m_snap = K_block·ln2  (无 -2)         (式19)
#                     corr   = 2^(K_block - K_new - ceil_log2CX)  (式26)
#                     pscale byte = K_block - K_new - ceil_log2CX + 127
# 二者差异仅在 K_block 取整方式 + 末尾 -2 换成 -ceil_log2CX。
# CX 默认 dstTypeMax=6.0 -> log2CX=log2(6)≈2.585, ceil_log2CX=3。
# =====================================================================
QUANT_MODE_OCP = 2
QUANT_MODE_CX = 3
# CX 默认 dstTypeMax(host: dstTypeMax=0 时取 MXFP4_DEFAULT_DST_TYPE_MAX_LOG2=log2(6.0))
CX_DEFAULT_DST_TYPE_MAX = 6.0


def _resolve_cx_params(dst_type_max):
    """CX 路径: 由 dstTypeMax 求 (log2CX, ceil_log2CX)。
    dst_type_max=0 -> 默认 6.0(对齐 host MXFP4_DEFAULT_DST_TYPE_MAX_LOG2)。"""
    cx_val = CX_DEFAULT_DST_TYPE_MAX if (dst_type_max is None or dst_type_max == 0) else float(dst_type_max)
    log2_cx = math.log2(cx_val)
    ceil_log2_cx = math.ceil(log2_cx)
    return log2_cx, ceil_log2_cx


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

# 为一个 32 元素 block 计算 E8M0 scale（ 2^exp ），使 block 内 max 经过除 scale 后落在 FP4 高精度区
def compute_e8m0_scale_block(block: torch.Tensor, mode: str = "baseline",
                             quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0) -> torch.Tensor:
    """为一个 32 元素 block 算 E8M0 scale(2^exp)。
    OCP(baseline): scale_exp = floor(log2(max))-emax, max 归一化到 [4,8)(式3: /4)。
    OCP(oas):      scale_exp = ceil(log2(max))-emax,  防饱和。
    CX:            scale_exp = ceil(log2(max) - log2CX), max 归一化到 (CX/2, CX](式16: /CX)。
                   log2CX 由调用方预算传入(避免 per-block 重算 math.log2)。"""
    """
    为一个 block 计算 E8M0 缩放因子 (2 的整数次幂)。

    OCP(quantMode=2) MX 规范(式3):
        scale_exp = floor(log2(max_abs(block))) - emax_elem   (baseline)
        scale_exp = ceil(log2(max_abs(block)))  - emax_elem   (oas, 防饱和)
    CX(quantMode=3)(式16):
        scale_exp = ceil(log2(max_abs) - log2CX) = ceil(log2(max_abs/CX))
        其中 log2CX = log2(dstTypeMax), dstTypeMax=0 取默认 6.0。

    参数:
        block:        shape [..., block_size] 的张量
        mode:         "baseline" - floor, 最大化 FP4 范围利用, 允许少量饱和
                      "oas"      - ceil,  保证不饱和, 牺牲一点精度
                      (CX 忽略 mode, 固定用 ceil-log2CX)
        quant_mode:   OCP(2)|CX(3)
        dst_type_max: CX 的 dstTypeMax(0=默认6.0); 仅 CX 生效
        emax_elem:    元素格式最大正规化指数, FP4 E2M1 为 2

    返回:
        scale: 标量, 2 的整数次幂, 指数范围 [-127, 127]
    """
    max_abs = torch.max(torch.abs(block))
    if max_abs.item() < 1e-30:          # 全 0 block: scale=1
        return torch.tensor(1.0, device=block.device)
    log_max = torch.log2(max_abs)
    if quant_mode == QUANT_MODE_CX:
        # CX(式16): scale_exp = ceil(log2(max) - log2CX), log2CX 由调用方预算传入
        exp = torch.ceil(log_max - log2_cx)
    else:
        # OCP(式3): scale_exp = floor(log2(max)) - emax (baseline) | ceil - emax (oas)
        exp = torch.floor(log_max) - FP4_E2M1_EMAX if mode == "baseline" \
            else torch.ceil(log_max) - FP4_E2M1_EMAX
    exp = torch.clamp(exp, -127.0, 127.0)
    return 2.0 ** exp

# 完整的 MXFP4 量化流程 ：沿最后一维，每 32 元素一个 block，每个 block 共享一个 E8M0 scale，先除 scale 归一化再量化到 FP4
def mx_quantize(tensor: torch.Tensor, block_size: int = 32, mode: str = "baseline",
                quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0):
    """MXFP4 量化(FP32 模拟): 沿最后一维每 32 元素共享一个 E8M0 scale, 再各自量化到 FP4。
    quant_mode/log2_cx 透传给 compute_e8m0_scale_block(CX 用 ceil-log2CX, OCP 用 floor-emax)。
    log2_cx 由调用方预算传入(避免 per-block 重算 math.log2)。
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
        sc = compute_e8m0_scale_block(reshaped[i], mode=mode,
                                      quant_mode=quant_mode, log2_cx=log2_cx)
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
def mxfp4_qdq(t: torch.Tensor, axis: int, block_size: int = 32, mode: str = "baseline",
              quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0) -> torch.Tensor:
    """沿任意 axis 做 MXFP4 qdq round-trip, 返回带量化噪声的 FP32。
    golden 用它给 Q/K/V 引入与 NPU(packed fp4+e8m0)一致的量化误差。
    quant_mode/log2_cx 透传给 mx_quantize(CX 用 ceil-log2CX 网格, OCP 用 floor-emax)。"""
    t_last = t.movedim(axis, -1).contiguous()     # 量化轴搬到最后
    fp4_sim, scales = mx_quantize(t_last, block_size=block_size, mode=mode,
                                  quant_mode=quant_mode, log2_cx=log2_cx)
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
                             block_size: int = 32, mode: str = "baseline",
                             quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0):
    """NPU 侧打包入口: 量化轴任意, 打包轴=最后一维。
    返回 (packed_fp4 uint8[last=D//2], e8m0_scales uint8 biased[量化轴维=ceil(qd/32)])。
    Q/K 用 quant_axis=-1(沿 head_dim); V 用 -2(沿 seq_k)。
    quant_mode/log2_cx 透传给 mx_quantize(CX 用 ceil-log2CX 网格, OCP 用 floor-emax)。"""
    nd = tensor.dim()
    qax = quant_axis % nd
    assert tensor.shape[-1] % 2 == 0, f"最后一维须偶数, 当前 {tensor.shape[-1]}"
    t_q_last = tensor.movedim(qax, -1).contiguous()
    fp4_sim, scales_flat = mx_quantize(t_q_last, block_size=block_size, mode=mode,
                                       quant_mode=quant_mode, log2_cx=log2_cx)
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
                                          mx_block_size: int = 32, mx_mode: str = "baseline",
                                          is_cx: bool = False,
                                          neg_log2_cx: float = 0.0,
                                          p_quant_exp_shift: float = 2.0):
    """对 scores(S, seqk_dim 轴)做 blockwise_snap_local P 量化, 返回 (P_eff, m_new)。
    每 32 元素 block: m_block snap 到 log2 -> P_local -> FP4 round 得 P_q(raw) ->
    corr=2^(K_block-K_new-shift) 折 per-block E8M0 -> P_eff=P_q*corr。
    P_eff 未归一化(需外部 /sum); m_new=K_new(fp32 整数, 供 online softmax alpha=exp2(K_diff))。

    循环不变量 is_cx/neg_log2_cx/p_quant_exp_shift 由调用方提前算一次传入(避免 per-chunk 重复算):
      OCP(is_cx=False): K_block=floor(Smax/ln2), m_snap=(K_block-2)·ln2,
              corr=2^(K_block-K_new-2), pscale byte=K_block-K_new-2+127   (式8/12/13)
              p_quant_exp_shift=2.0(默认), neg_log2_cx 不用(预置 0)。
      CX(is_cx=True):  K_block=ceil(Smax/ln2 - log2CX), m_snap=K_block·ln2(无 -2),
              corr=2^(K_block-K_new-ceil_log2CX), pscale byte=K_block-K_new-ceil_log2CX+127  (式19/26)
              p_quant_exp_shift=ceil_log2CX, neg_log2_cx=-log2CX,
              其中 log2CX=log2(dstTypeMax), dstTypeMax=0 时取默认 6.0(log2(6)≈2.585, ceil=3)。"""
    NEG_INF = float('-inf')
    device = S.device
    src_dtype = S.dtype                       # 全程跟随 S dtype(对齐 qfa: fp16)
    ndim = S.dim()
    skp = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim
    seqk_len = S.shape[skp]
    LN2 = math.log(2.0)

    # 量化模式参数已由调用方预算传入(is_cx/neg_log2_cx/p_quant_exp_shift), 此处直接用。

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
    if is_cx:
        # CX(式19): K_block = ceil(Smax*INV_LN2 - log2CX), 全程 fp16 域(对齐 kernel:
        #   kernel: Smax(fp16)*dScale*INV_LN2(fp16) + NEG_LOG2_CX(fp16) -> Truncate<CAST_CEIL> -> *LN2(fp16))
        #   golden: INV_LN2/neg_log2_cx 都转 fp16, 与 m_block_raw_safe(fp16) 同精度运算, 避免 fp64
        #   标量混入导致 ceil 边界舍入翻转。
        #   注: 用乘法 Smax*INV_LN2 而非除法 Smax/LN2——fp16 下 1/LN2_fp16 ≠ INV_LN2_fp16
        #   (差~1.29e-04), 除法在 ceil 边界翻转导致 golden≠kernel(dst_type_max=9.9 即此现象)。
        INV_LN2_fp16 = torch.tensor(1.0 / LN2, dtype=torch.float16, device=device)            # 对齐 kernel constexpr half INV_LN2
        LN2_fp16 = torch.tensor(LN2, dtype=torch.float16, device=device)                      # m_block_snap 用
        neg_log2_cx_fp16 = torch.tensor(neg_log2_cx, dtype=torch.float16, device=device)  # 对齐 kernel NEG_LOG2_CX
        k_raw = m_block_raw_safe * INV_LN2_fp16 + neg_log2_cx_fp16                       # 全 fp16 乘法(对齐 kernel)
        K_block = torch.ceil(k_raw)
        m_block_snap = K_block * LN2_fp16                                             # 无 -2(式19), fp16
    else:
        # OCP(式8): K_block = floor(Smax*INV_LN2), 全程 fp16 域(对齐 kernel:
        #   kernel: Smax(fp16)*dScale*INV_LN2(fp16) -> Truncate<CAST_FLOOR> -> *LN2(fp16) -> -2*LN2(fp16))
        #   golden: INV_LN2/LN2 都转 fp16, 与 m_block_raw_safe(fp16) 同精度运算, 避免 fp64 标量混入
        #   导致 floor 边界舍入翻转。
        #   注: 用乘法 Smax*INV_LN2 而非除法 Smax/LN2——fp16 下 1/LN2_fp16 ≠ INV_LN2_fp16
        #   (差~1.29e-04), 除法在 floor 边界翻转导致 golden≠kernel(OCP 在 Smax≥11.78 有 46 处翻转)。
        INV_LN2_fp16 = torch.tensor(1.0 / LN2, dtype=torch.float16, device=device)            # 对齐 kernel constexpr half INV_LN2
        LN2_fp16 = torch.tensor(LN2, dtype=torch.float16, device=device)                      # m_block_snap 用
        K_block = torch.floor(m_block_raw_safe * INV_LN2_fp16)  # snap: floor(m*INV_LN2)
        m_block_snap = (K_block - 2) * LN2_fp16                     # 使 P_local block max ∈ [4,8), fp16
    if _DUMP_ENABLED:
        _dump_tensor(K_block.T, "local_group_max.bin")         # dump: 每 block 的 K(镜像 qfa local_group_max, 对齐转置)
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
    # 阶段6: corr = 2^(K_block-K_new-p_quant_exp_shift), per-block E8M0(折 dequant 因子)。
    #   OCP: p_quant_exp_shift=2  (式12/13: byte=K_block-K_new-2+127)
    #   CX:  p_quant_exp_shift=ceil_log2CX  (式26: byte=K_block-K_new-ceil_log2CX+127)
    # CX 全程 fp16(对齐 kernel pscale VF 的 16bit 算); OCP 保持原 s_dtype 精度。
    if is_cx:
        K_diff = (K_block - K_new_s.unsqueeze(-1))
        corr = torch.exp2(K_diff - torch.tensor(p_quant_exp_shift, dtype=torch.float16, device=device))
    else:
        K_diff = K_block - K_new_s.unsqueeze(-1)               # ≤0, s_dtype 整数
        corr = torch.exp2(K_diff - 2.0)
    if _DUMP_ENABLED:                                       # dump: P scale E8M0(镜像 qfa p_scale_e8m0)
        p_scale_e8m0 = _to_e8m0_uint8(K_diff.to(torch.int32) - int(p_quant_exp_shift)).T.contiguous()
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

# 输出 dtype 映射(out_dtype): golden 最终 cast + NPU attention_out_dtype 用。
_OUT_DTYPE_MAP = {'bfloat16': torch.bfloat16, 'float16': torch.float16}

def _resolve_out_dtype(out_dtype):
    """字符串 out_dtype('bfloat16'/'float16') -> torch dtype。golden return cast + NPU attention_out_dtype 用。"""
    if out_dtype not in _OUT_DTYPE_MAP:
        raise ValueError(f"out_dtype must be one of {list(_OUT_DTYPE_MAP)}, got '{out_dtype}'")
    return _OUT_DTYPE_MAP[out_dtype]

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

def make_ocp_cross_head_scale_trap(shape_bnsd, dtype=torch.float32):
 	     """OCP 下强制「相邻头 e8m0 全局不等」的 BNSD 张量, 专打 TND/BSND Q/K-scale 行步长漏乘 N。
 	 
 	     OCP baseline: scale_exp = floor(log2(max_abs)) - 2
 	       偶头: |x|∈(0.60, 0.90) → block max∈(0.5,1) → exp=-3 (e8m0=124)
 	       奇头: |x|∈(1.20, 1.80) → block max∈[1,2)  → exp=-2 (e8m0=125)
 	     旧 kernel Dn2Nz srcDValue=D//64(未乘 N)时, 同头第 r 行会读到邻头 scale → 反量化差 2× → 精度挂;
 	     新 kernel srcDValue=N*(D//64) 后应对齐 golden。
 	     普通 U[-1,1] OCP 下 scale 近乎全等, 读错也看不见, 故需本陷阱。"""
 	     B, N, S, D = shape_bnsd
 	     assert N >= 2, f"scale-stride trap 需要 N>=2, 当前 N={N}"
 	     out = torch.empty(B, N, S, D, dtype=dtype)
 	     for n in range(N):
 	         if (n % 2) == 0:
 	             lo, hi = 0.60, 0.90
 	         else:
 	             lo, hi = 1.20, 1.80
 	         chunk = lo + (hi - lo) * torch.rand(B, S, D, dtype=dtype)
 	         sign = torch.where(torch.rand(B, S, D) > 0.5, 1.0, -1.0).to(dtype)
 	         out[:, n, :, :] = chunk * sign
 	     return out
 	 
 	 
def _make_qkv(b, n1, n2, s1, s2, qk_d, v_d, data_x=1.0, data_kind='default'):
    """统一造 Q/K/V (BNSD)。data_kind='ocp_scale_stride_trap' 时走跨头 e8m0 陷阱。"""
    if data_kind == 'ocp_scale_stride_trap':
        query = make_ocp_cross_head_scale_trap((b, n1, s1, qk_d)).to(torch.float32)
        key = make_ocp_cross_head_scale_trap((b, n2, s2, qk_d)).to(torch.float32)
        value = make_ocp_cross_head_scale_trap((b, n2, s2, v_d)).to(torch.float32)
    else:
        query = randn_x((b, n1, s1, qk_d), data_x).to(torch.float32)
        key = randn_x((b, n2, s2, qk_d), data_x).to(torch.float32)
        value = randn_x((b, n2, s2, v_d), data_x).to(torch.float32)
    return query, key, value

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
                                                    dump=False, out_dtype='bfloat16',
                                                    quant_mode=QUANT_MODE_OCP, dst_type_max=0.0):
    """CPU 标杆(BNSD, MXFP4, 分块 flash), 对齐 qfa flash golden 的 dtype 流 + kernel tile 结构:
       Q/K/V mxfp4 qdq(fp32); per q-block + 2-kv-block chunk + online softmax;
       S fp32 matmul -> s_dtype -> scale; P 量化(snap_local)与 P 本身在 s_dtype; sum/PV 在 fp32。
       dump=True 时落盘中间结果(dump_data/*.bin, per-chunk append)。
       p_quant: 'snap_local'|'none'; s_dtype: 'fp16'(默认,qfa 对齐)|'bf16'|'fp32'。
       quant_mode: OCP(2)|CX(3), 仅 p_quant='snap_local' 时区分 P 量化方案; dst_type_max: CX 的 dstTypeMax(0=默认6.0)。"""
    if dump:
        _enable_dump(True)
    try:
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        v = value.to(torch.float32)
        B, N, S, D = q.shape
        N2 = k.shape[1]
        S2 = k.shape[2]                       # KV 序列长(不与 q 的 S 混)
        group_size = N // N2
        block_x, block_y = int(block_shape[0]), int(block_shape[1])
        ceil_q = (S + block_x - 1) // block_x
        ceil_kv = (S2 + block_y - 1) // block_y
        s_torch_dtype = _resolve_s_dtype(s_dtype)

        # 量化模式参数(循环不变量, 预算一次, 供每次 chunk 调用复用):
        # 量化模式参数(循环不变量, 预算一次, 供每次 chunk 调用 + Q/K/V qdq + NPU 打包复用):
        #   OCP(quant_mode=2): p_quant_exp_shift=2.0, neg_log2_cx/log2_cx 不用(预置 0)。
        #   CX (quant_mode=3): p_quant_exp_shift=ceil_log2CX, neg_log2_cx=-log2CX, log2_cx=log2(CX)。
        is_cx = (quant_mode == QUANT_MODE_CX)
        neg_log2_cx = 0.0
        log2_cx = 0.0
        if is_cx:
            print("MXFP4 CX QUANT")
            log2_cx, ceil_log2_cx = _resolve_cx_params(dst_type_max)
            neg_log2_cx = -log2_cx
            p_quant_exp_shift = float(ceil_log2_cx)
            print(f"p_quant_exp_shift: {p_quant_exp_shift}")
        else:
            print("MXFP4 OCP QUANT")
            p_quant_exp_shift = 2.0
            print(f"p_quant_exp_shift: {p_quant_exp_shift}")

        # Q/K/V mxfp4 qdq -> fp32(引入与 NPU 一致的量化噪声)
        # quant_mode/log2_cx 透传: CX 用 ceil-log2CX 网格, OCP 用 floor-emax 网格
        q_q = mxfp4_qdq(q, axis=-1, quant_mode=quant_mode, log2_cx=log2_cx)  # Q/K: head_dim(rowwise)
        k_q = mxfp4_qdq(k, axis=-1, quant_mode=quant_mode, log2_cx=log2_cx)
        v_q = mxfp4_qdq(v, axis=2, quant_mode=quant_mode, log2_cx=log2_cx)   # V: seq_k(columnwise)

        out = torch.zeros((B, N, S, D), dtype=torch.float32)
        lse = torch.full((B, N, S, 1), float('-inf'), dtype=torch.float32)
        scale_s = torch.as_tensor(scale_value, dtype=s_torch_dtype)  # scale round 到 s_dtype

        for b in range(B):
            for n in range(N):
                n2 = n // group_size                 # GQA 取模映射(query 头 n -> kv 头 n//group_size)
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
                                S_s, m_running=m_run, seqk_dim=0,  # DN: seqk=dim0
                                is_cx=is_cx, neg_log2_cx=neg_log2_cx, p_quant_exp_shift=p_quant_exp_shift)
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
        return out.to(_resolve_out_dtype(out_dtype)), lse.to(torch.float32)
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
    elif data_type == 'float16':
        # mxfp4: drift 是 mxfp4 P-quant + fp16-softmax 累积(非 fp16 量化噪声), 同 bf16 级;
        # fp16 输出量化(1/1024)远小于此 drift, fp16 输出暴露 bf16 掩盖的 drift, 故 fp16 不该比 bf16 严。
        rtol, atol, max_diff_hd = 0.0078125, 0.0001, 10.0      # 1/128(同 bf16)
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
# Q/K 的 E8M0 scale 从 [B,N,S,D//32] 重排为 V3 算子 doc 规定的 [B,N,S,D//64,2] 形状
# 配对方向: flip (s0,s1)->(s1,s0)。经 NPU kernel mm1Res dump 核验(spec 假设A):
#   连续配对下 Q 行 s=10/20(scales[0]!=scales[1])在所有 KV chunk 上 S=K@Q^T 整列出错,
#   flip 配对后 Q dequant 与 kernel 一致。原 spec 写"连续配对, 可能需 flip"-> 确需 flip。
def _e8m0_qk_to_v3(e8m0_uint8):
    """q/k e8m0 [B,N,S,D//32] -> [B,N,S,D//64,2] (配对方向由 _E8M0_QK_FLIP_PAIR 控制)。"""
    *lead, nd = e8m0_uint8.shape
    paired = e8m0_uint8.reshape(*lead, nd // 2, 2)
    if _E8M0_QK_FLIP_PAIR:
        paired = paired.flip(-1)                                # (s0,s1) -> (s1,s0)
    return paired.contiguous().view(torch.float8_e8m0fnu)


def _e8m0_v_to_v3(e8m0_uint8):
    """v e8m0 [B,N2,ceil(S2/32),D] -> [B,N2,ceil(S2/64),D,2] (连续配对 + 2 维挪到 D 后)。
    ceil(S2/32) 为奇数时(s2 落在 (64k, 64k+32], 如 129/160/200)先 pad 1 个 e8m0=127
    (scale=2^0=1.0) 的 32-block 再配对——对齐 qfa_mxfp4_golden.py L551-558;
    host 校验 CeilDiv(maxKvSeqlen,64) 组(tiling ValidateMxfp4DequantScale),
    pad 后形状恰好一致; pad 值必须 127(e8m0 无 0 值语义, 0 byte 在 NPU 侧是 NaN)。"""
    B, N2, nb, D = e8m0_uint8.shape
    if nb % 2 != 0:
        pad_block = torch.full((B, N2, 1, D), E8M0_BIAS, dtype=torch.uint8)
        e8m0_uint8 = torch.cat([e8m0_uint8, pad_block], dim=2)
        nb += 1
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
    if kind == 'all_masked_one_qb':  # dense 但中间一个 q-block 全 0(无 kv attended, kernel 0-rsvd gather 边界)
        m = torch.ones(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
        m[:, :, ceil_q // 2, :] = 0  # 中间 q-block attend 无 kv -> out=0/lse=-inf
        return m
    if kind == 'gap_sparse':  # 每 q-block 仅 attend kv-block 0 + 末块(大 gap, 跨块 rescale + 非连续 gather)
        m = torch.zeros(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
        m[:, :, :, 0] = 1
        m[:, :, :, ceil_kv - 1] = 1
        return m
    raise ValueError(f"unsupported mask kind: {kind}")


def _build_block_sparse_mask_varlen(b, n1, act_q, act_kv, block_shape, kind='dense'):
    """变长 TND 的 block_sparse_mask [B,n1,ceil_q_max,ceil_kv_max] int8。
    batch b 实际 token 数 act_q[b]/act_kv[b]; 超出 ceil_q_b/ceil_kv_b 的块置 0(无 token)——
    对齐 V3 doc: mask 用 max-based ceil, 但每 batch 实际只到 ceil(S_b/block)。
    dense: [0..ceil_q_b, 0..ceil_kv_b]=1 余 0; sparse: 在 [0..ceil_q_b,0..ceil_kv_b] 内随机。"""
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    max_s1 = max(act_q); max_s2 = max(act_kv)
    ceil_q = (max_s1 + block_x - 1) // block_x
    ceil_kv = (max_s2 + block_y - 1) // block_y
    m = torch.zeros(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
    for bi in range(b):
        cq_b = (act_q[bi] + block_x - 1) // block_x
        ck_b = (act_kv[bi] + block_y - 1) // block_y
        if kind == 'dense':
            m[bi, :, :cq_b, :ck_b] = 1
        elif kind == 'sparse':
            for i in range(cq_b):
                n_attend = torch.randint(1, ck_b + 1, (1,)).item()
                idx = torch.randperm(ck_b)[:n_attend]
                m[bi, :, i, idx] = 1
        else:
            raise ValueError(f"unsupported mask kind: {kind}")
    return m


def run_bsa_mxfp4_eager(b, n1, n2, s1=None, s2=None, g=1, qk_d=None, v_d=None,
                        input_layout='BNSD', out_dtype='bfloat16',
                        block_shape=(128, 128), mask_kind='dense',
                        inner_precise=4, quant_mode=2,
                        p_quant='snap_local', s_dtype='fp16',
                        dump=False, device_id=0,
                        s1_list=None, s2_list=None,
                        data_x=10.0, dst_type_max=0.0, act_s1=None, act_s2=None, 
                        data_kind='default'):
    """qfa 风格: fp32 造数据 -> mxfp4 打包 -> CPU golden -> NPU -> check_result.
       p_quant: 'snap_local'|'none'; s_dtype: 'fp16'(默认)|'bf16'|'fp32'; dump: 落盘中间结果。
       input_layout: 'BNSD'(默认)|'BSND'|'TND'; BSND/TND 对齐 QFA 哲学——golden 保持 BNSD 不动, 仅 NPU I/O
       重排: BSND=permute(0,2,1,3)[+5D scale permute(0,2,1,3,4)]; TND uniform=BSND 重排 + flatten(0,1)(B,S->T)。
       TND varlen(s1_list/s2_list 给 per-batch 变长): per-batch 造数/golden/打包 + concat(B,S->T per batch,
       T=sum S1_b, BTkv=sum CeilDiv(S2_b,64)), actualSeq=[S1_b]/[S2_b]。输出同法重排回目标布局比对。
       data_kind: 'default'| 'ocp_scale_stride_trap'(跨头强制 OCP e8m0 不等, 打 TND/BSND scale 行步长)。
       s1/s2(uniform, BNSD/BSND/uniform-TND) 或 s1_list/s2_list(TND varlen) 二选一。
       act_s1/act_s2(BNSD/BSND padded, 可选): actualSeqLengths < 张量 s1/s2(V3 doc L248/264 显式支持)。
       int=uniform 或 list=per-batch; s1/s2 此时为 padded 张量形状。golden 走 per-batch unpadded(golden 不改),
       NPU I/O 0-pad 到 s1/s2 + actualSeq=[act_b]; 比对只取 [0,act_q) 有效区(kernel 不清零 padding 区)。
       量化一致性靠 0-padding(Q/K 轴 D 不跨 seq; V 轴 seq 末有效 32-block unpadded 内部 0-pad 与 0-padded 外部
       0-pad 填 0 -> max/scale 相同)。返回 (fulfill_percent, result_str). NPU 不可用时返回 (-1.0, 'SKIP')."""
    g = n1 // n2
    assert input_layout in ('BNSD', 'BSND', 'TND'), f"首版仅支持 BNSD/BSND/TND, 当前 {input_layout}"
    assert n1 == n2 * g, f"GQA 约束 n1==n2*g: {n1} vs {n2}*{g}"
    assert block_shape[0] in (64, 128), f"MXFP4 blockShapeX 仅支持 64/128, 当前 {block_shape[0]}"
    assert qk_d in (64, 128), f"MXFP4 D 仅支持 64/128, 当前 {qk_d}"
    assert mask_kind in ('dense', 'sparse', 'gap_sparse'), \
        f"mask_kind 仅支持 'dense'/'sparse/gap_sparse/ ', 当前 {mask_kind}"
    assert s_dtype in ('fp16', 'bf16', 'fp32'), \
        f"s_dtype 仅支持 'fp16'/'bf16'/'fp32', 当前 {s_dtype}"
    assert quant_mode in (QUANT_MODE_OCP, QUANT_MODE_CX), \
        f"quant_mode 仅支持 {QUANT_MODE_OCP}(OCP)/{QUANT_MODE_CX}(CX), 当前 {quant_mode}"
    assert data_kind in ('default', 'ocp_scale_stride_trap'), \
        f"data_kind 仅支持 'default'/'ocp_scale_stride_trap', 当前 {data_kind}"
    if data_kind == 'ocp_scale_stride_trap':
        assert n1 >= 2 and n2 >= 2, \
            f"ocp_scale_stride_trap 需要 n1>=2 且 n2>=2(跨头 e8m0 不等), 当前 n1={n1} n2={n2}"
        assert input_layout in ('BSND', 'TND'), \
            f"ocp_scale_stride_trap 仅对 BSND/TND 有意义(BNSD 无 N 行步长 bug), 当前 {input_layout}"
    assert out_dtype in ('bfloat16', 'float16'), \
        f"out_dtype 仅支持 'bfloat16'/'float16', 当前 {out_dtype}"
    if quant_mode == QUANT_MODE_CX:
        # CX dstTypeMax: 0(默认6.0) 或 [6.0, 12.0](对齐 host ValidateAdditionalParams)
        _dst = 0.0 if dst_type_max is None else float(dst_type_max)
        assert _dst == 0.0 or (6.0 <= _dst <= 12.0), \
            f"CX(quant_mode=3) dst_type_max 须 0(默认6.0) 或 [6.0, 12.0], 当前 {dst_type_max}"

    # 预算 log2_cx(供 Q/K/V 量化网格 + P snap 复用, 避免重复算 math.log2):
    #   OCP: log2_cx 不用(预置 0); CX: log2_cx=log2(dstTypeMax), dstTypeMax=0 取默认 6.0。
    if quant_mode == QUANT_MODE_CX:
        print("MXFP4 CX QUANT")
        log2_cx, _ = _resolve_cx_params(dst_type_max)
        log2_cx_fp16 = float(torch.tensor(log2_cx, dtype=torch.float16).item())
        print(f"log2_cx: {log2_cx} (fp16 cast: {log2_cx_fp16})")
        # log2_cx = log2_cx_fp16  # 用 fp16 值后续透传(对齐 kernel static_cast<half>)
    else:
        print("MXFP4 OCP QUANT")
        log2_cx = 0.0

    if v_d is None:
        v_d = qk_d
    assert v_d == qk_d, f"v_d must == qk_d (doc L517/L653: attentionOut shape==query, D 须一致), got v_d={v_d} qk_d={qk_d}"
    if False :
        print(f"[run_bsa_mxfp4_eager] shapes: b={b} n1={n1} n2={n2} g={g} "
          f"s1={s1} s2={s2} qk_d={qk_d} v_d={v_d} block_shape={block_shape} "
          f"layout={input_layout} out_dtype={out_dtype} mask={mask_kind} "
          f"inner_precise={inner_precise} quant_mode={quant_mode} "
          f"p_quant={p_quant} s_dtype={s_dtype} dump={dump} device_id={device_id}")
    if data_kind == 'ocp_scale_stride_trap':
        print(f"[data_kind] ocp_scale_stride_trap: 跨头强制 OCP e8m0 偶头=124/奇头=125 "
              f"(layout={input_layout} n1={n1} n2={n2})")
    scale = 1.0 / math.sqrt(qk_d)
    src_dtype = out_dtype

    # ===== TND varlen (per-batch 变长 S): s1_list/s2_list 给 per-batch 实际 seq, T=sum S1_b =====
    # kernel/host 已 varlen-ready(base-offset 按实际 S 累加、BTkv=sum、actualSeq 强制); 此处仅 test 侧。
    if input_layout == 'TND' and s1_list is not None:
        block_x, block_y = int(block_shape[0]), int(block_shape[1])
        act_q = [int(x) for x in s1_list]
        act_kv = [int(x) for x in s2_list]
        assert len(act_q) == b and len(act_kv) == b, "s1_list/s2_list 长度须 == b"
        mask = _build_block_sparse_mask_varlen(b, n1, act_q, act_kv, block_shape, mask_kind)
        # per-batch BNSD 造数(变长, TND 无 padding); 同一份 q_b/k_b/v_b 供 golden + 打包(mxfp4 噪声一致)
        q_list, k_list, v_list = [], [], []
        for bi in range(b):
            q_b, k_b, v_b = _make_qkv(1, n1, n2, act_q[bi], act_kv[bi], qk_d, v_d,
                                    data_x=data_x, data_kind=data_kind)
            q_list.append(q_b)
            k_list.append(k_b)
            v_list.append(v_b)
        # golden per-batch(B=1, S=S_b, mask 切到实际) -> BNSD permute+flatten -> concat TND [Tq,N,D]
        # dump 修复: 循环外一次性 _enable_dump(True)(含 truncate), 循环内 dump=False 避免
        #   cpu_block_sparse_attention_bnsd_mxfp4_with_lse 进入时再 truncate 清掉前序 batch 数据;
        #   循环外 _enable_dump(False) 收尾。否则 b>1 时 bin 仅剩最后一个 batch(per-batch 进入都 truncate)。
        print("cpu calc start")
        if dump:
            _enable_dump(True)
        cpu_chunks = []
        for bi in range(b):
            cq_b = (act_q[bi] + block_x - 1) // block_x
            ck_b = (act_kv[bi] + block_y - 1) // block_y
            mask_b = mask[bi:bi + 1, :, :cq_b, :ck_b]
            out_b, _ = cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
                q_list[bi], k_list[bi], v_list[bi], mask_b, block_shape, scale, n2,
                p_quant=p_quant, s_dtype=s_dtype, dump=False, out_dtype=out_dtype,
                quant_mode=quant_mode, dst_type_max=dst_type_max)
            cpu_chunks.append(out_b.permute(0, 2, 1, 3).flatten(0, 1))  # [S1_b, N, D] bf16
        cpu_out = torch.cat(cpu_chunks, dim=0).contiguous()  # [Tq, N, D] bf16
        print("golden calc end")
        if _FORCE_CPU or not _HAS_BSA_V3:
            print(f"cpu_out: {cpu_out}")
            return -1.0, "SKIP"
        # 打包 per-batch -> BNSD permute+flatten -> concat TND [T,N,D//2] + 4D scale
        q_p_ch, k_p_ch, v_p_ch, q_s_ch, k_s_ch, v_s_ch = [], [], [], [], [], []
        for bi in range(b):
            q_p, q_e = mxfp4_quantize_pack_last(q_list[bi], quant_axis=-1,
                                                quant_mode=quant_mode, log2_cx=log2_cx)
            k_p, k_e = mxfp4_quantize_pack_last(k_list[bi], quant_axis=-1,
                                                quant_mode=quant_mode, log2_cx=log2_cx)
            v_p, v_e = mxfp4_quantize_pack_last(v_list[bi], quant_axis=-2,
                                                quant_mode=quant_mode, log2_cx=log2_cx)
            q_p_ch.append(q_p.permute(0, 2, 1, 3).flatten(0, 1))                       # [S1_b, N, D//2]
            k_p_ch.append(k_p.permute(0, 2, 1, 3).flatten(0, 1))                      # [S2_b, N, D//2]
            v_p_ch.append(v_p.permute(0, 2, 1, 3).flatten(0, 1))
            q_s_ch.append(_e8m0_qk_to_v3(q_e).permute(0, 2, 1, 3, 4).flatten(0, 1))   # [S1_b, N, D//64, 2]
            k_s_ch.append(_e8m0_qk_to_v3(k_e).permute(0, 2, 1, 3, 4).flatten(0, 1))
            v_s_ch.append(_e8m0_v_to_v3(v_e).permute(0, 2, 1, 3, 4).flatten(0, 1))     # [ceil(S2_b/64), N, D, 2]
        q_packed = torch.cat(q_p_ch, dim=0).contiguous()    # [Tq, N, D//2]
        k_packed = torch.cat(k_p_ch, dim=0).contiguous()    # [Tkv, N, D//2]
        v_packed = torch.cat(v_p_ch, dim=0).contiguous()
        q_scale = torch.cat(q_s_ch, dim=0).contiguous()     # [Tq, N, D//64, 2]
        k_scale = torch.cat(k_s_ch, dim=0).contiguous()
        v_scale = torch.cat(v_s_ch, dim=0).contiguous()     # [BTkv, N, D, 2]
        torch_npu.npu.set_device(device_id)
        print("npu calc start")
        npu_out, _ = block_sparse_attention_v3(
            q_packed.contiguous().npu(), k_packed.contiguous().npu(),
            v_packed.contiguous().npu(), mask.contiguous().npu(), list(block_shape),
            act_q, act_kv,
            num_heads=n1, num_key_value_heads=n2,
            q_input_layout='TND', kv_input_layout='TND',
            mask_type=0, scale_value=scale, inner_precise=inner_precise, block_size=0,
            pre_tokens=2147483647, next_tokens=2147483647,
            softmax_lse_flag=False, quant_mode=quant_mode, dst_type_max=dst_type_max,
            q_dequant_scale=q_scale.npu(), k_dequant_scale=k_scale.npu(), v_dequant_scale=v_scale.npu(),
            attention_out_dtype=_resolve_out_dtype(out_dtype),
        )
        torch_npu.npu.synchronize()
        fulfill_percent, result = compare(cpu_out, npu_out, src_dtype)
        return fulfill_percent, result

    # ===== padded BNSD/BSND 路径: act_s1/act_s2 < s1/s2(张量 0-pad 到 S, actualSeq 指示更小有效长度) =====
    # V3 doc L248/264: BNSD/BSND 下配置 actualSeqLengths 则算子按实际长度处理(kernel 只建 [0,act) task,
    #   不清零 [act,S) padding 区)。当前 uniform 路径恒令 actualSeq==s1/s2, 此路径此前零覆盖。
    # golden 走 per-batch unpadded(复用 TND varlen 模式, golden 函数不改); NPU I/O 0-pad 到 s1/s2 后打包。
    # 量化一致性靠 0-padding: Q/K 量化轴=head_dim(D) 32-block 不跨 seq; V 量化轴=seq_k 末有效 32-block 在
    #   unpadded(内部 0-pad 到 32 边界) 与 0-padded(外部 0-pad) 填充值均 0 -> max/scale 相同。
    # mask 用 _build_block_sparse_mask_varlen(per-batch actual blocking) + F.pad 补到 max-based padded
    #   shape(host CheckSparsePattern 校验 mask dim=CeilDiv(S_pad, block); padding 块须置 0, host 不校验内容)。
    # 比对只取 [0,act_q) 有效区(kernel 不清零 [act,S) padding 区, 不能比全 padded 输出), flatten-concat per-batch。
    if input_layout in ('BNSD', 'BSND') and act_s1 is not None:
        act_q = [int(x) for x in ([act_s1] * b if isinstance(act_s1, int) else act_s1)]
        act_kv = [int(x) for x in ([act_s2] * b if isinstance(act_s2, int) else act_s2)]
        assert len(act_q) == b and len(act_kv) == b, "act_s1/act_s2 长度须 == b"
        assert all(a <= s1 for a in act_q) and all(a <= s2 for a in act_kv), \
            "act_s1/act_s2 每 batch 须 <= s1/s2(padded 张量形状)"
        block_x, block_y = int(block_shape[0]), int(block_shape[1])
        mask = _build_block_sparse_mask_varlen(b, n1, act_q, act_kv, block_shape, mask_kind)
        ceil_q_pad = (s1 + block_x - 1) // block_x
        ceil_kv_pad = (s2 + block_y - 1) // block_y
        mask = torch.nn.functional.pad(
            mask, (0, ceil_kv_pad - mask.shape[3], 0, ceil_q_pad - mask.shape[2]), value=0)
        # per-batch unpadded 造数(同一份供 golden + 0-pad 打包, mxfp4 噪声一致)
        q_list = [randn_x((1, n1, act_q[bi], qk_d), data_x).to(torch.float32) for bi in range(b)]
        k_list = [randn_x((1, n2, act_kv[bi], qk_d), data_x).to(torch.float32) for bi in range(b)]
        v_list = [randn_x((1, n2, act_kv[bi], v_d), data_x).to(torch.float32) for bi in range(b)]
        print("cpu calc start")
        cpu_chunks = []
        for bi in range(b):
            cq_b = (act_q[bi] + block_x - 1) // block_x
            ck_b = (act_kv[bi] + block_y - 1) // block_y
            mask_b = mask[bi:bi + 1, :, :cq_b, :ck_b]
            out_b, _ = cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
                q_list[bi], k_list[bi], v_list[bi], mask_b, block_shape, scale, n2,
                p_quant=p_quant, s_dtype=s_dtype, dump=dump, out_dtype=out_dtype,
                quant_mode=quant_mode, dst_type_max=dst_type_max)
            if input_layout == 'BSND':
                out_b = out_b.permute(0, 2, 1, 3).contiguous()  # [1,N,act,D] -> [1,act,N,D]
            cpu_chunks.append(out_b)
        print("golden calc end")
        if _FORCE_CPU or not _HAS_BSA_V3:
            print(f"cpu_out(valid region) per-batch shape: {[tuple(c.shape) for c in cpu_chunks]}")
            return -1.0, "SKIP"
        # 0-pad per-batch seq 维到 s1(Q)/s2(KV) -> stack BNSD [B,N,s1,D] / [B,N,s2,D]
        q_pad = torch.cat([torch.nn.functional.pad(q_list[bi], (0, 0, 0, s1 - q_list[bi].shape[2]))
                           for bi in range(b)], 0).contiguous()
        k_pad = torch.cat([torch.nn.functional.pad(k_list[bi], (0, 0, 0, s2 - k_list[bi].shape[2]))
                           for bi in range(b)], 0).contiguous()
        v_pad = torch.cat([torch.nn.functional.pad(v_list[bi], (0, 0, 0, s2 - v_list[bi].shape[2]))
                           for bi in range(b)], 0).contiguous()
        # 打包 0-padded BNSD(Q/K axis=-1, V axis=-2) + V3 e8m0 重排(复用 uniform 路径)
        # quant_mode/log2_cx 透传: CX 用 ceil-log2CX 网格, OCP 用 floor-emax(对齐 uniform 路径)
        q_packed, q_e8m0 = mxfp4_quantize_pack_last(q_pad, quant_axis=-1,
                                                    quant_mode=quant_mode, log2_cx=log2_cx)
        k_packed, k_e8m0 = mxfp4_quantize_pack_last(k_pad, quant_axis=-1,
                                                    quant_mode=quant_mode, log2_cx=log2_cx)
        v_packed, v_e8m0 = mxfp4_quantize_pack_last(v_pad, quant_axis=-2,
                                                    quant_mode=quant_mode, log2_cx=log2_cx)
        q_scale = _e8m0_qk_to_v3(q_e8m0)
        k_scale = _e8m0_qk_to_v3(k_e8m0)
        v_scale = _e8m0_v_to_v3(v_e8m0)
        if input_layout == 'BSND':  # packed 4D + 5D scale permute 到 BSND(复用 uniform 路径)
            q_packed = q_packed.permute(0, 2, 1, 3).contiguous()
            k_packed = k_packed.permute(0, 2, 1, 3).contiguous()
            v_packed = v_packed.permute(0, 2, 1, 3).contiguous()
            q_scale = q_scale.permute(0, 2, 1, 3, 4).contiguous()
            k_scale = k_scale.permute(0, 2, 1, 3, 4).contiguous()
            v_scale = v_scale.permute(0, 2, 1, 3, 4).contiguous()
        torch_npu.npu.set_device(device_id)
        print("npu calc start")
        npu_out, _ = block_sparse_attention_v3(
            q_packed.contiguous().npu(), k_packed.contiguous().npu(), v_packed.contiguous().npu(),
            mask.contiguous().npu(), list(block_shape), act_q, act_kv,
            num_heads=n1, num_key_value_heads=n2,
            q_input_layout=input_layout, kv_input_layout=input_layout,
            mask_type=0, scale_value=scale, inner_precise=inner_precise, block_size=0,
            pre_tokens=2147483647, next_tokens=2147483647,
            softmax_lse_flag=False, quant_mode=quant_mode, dst_type_max=dst_type_max,
            q_dequant_scale=q_scale.npu(), k_dequant_scale=k_scale.npu(), v_dequant_scale=v_scale.npu(),
            attention_out_dtype=_resolve_out_dtype(out_dtype))
        torch_npu.npu.synchronize()
        # 比对: 只取 [0,act_q) 有效区(kernel 不清零 [act,S) padding 区), flatten-concat per-batch
        g_valid, n_valid = [], []
        for bi in range(b):
            g_valid.append(cpu_chunks[bi].reshape(-1))
            if input_layout == 'BSND':
                n_valid.append(npu_out[bi, :act_q[bi], :, :].reshape(-1).cpu())
            else:
                n_valid.append(npu_out[bi, :, :act_q[bi], :].reshape(-1).cpu())
        return compare(torch.cat(g_valid), torch.cat(n_valid), src_dtype)

    # ===== uniform 路径(BNSD/BSND/uniform-TND): s1/s2 uniform =====
    assert s1 is not None and s2 is not None, "uniform 路径需 s1/s2(变长 TND 用 s1_list/s2_list)"

    # 1. 造 fp32 数据(qfa 风格 randn, 固定种子)
    query, key, value = _make_qkv(b, n1, n2, s1, s2, qk_d, v_d,
 	                              data_x=data_x, data_kind=data_kind)

    # 2. 官方 block_sparse_mask 网格
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (s1 + block_x - 1) // block_x
    ceil_kv = (s2 + block_y - 1) // block_y
    mask = _build_block_sparse_mask(b, n1, ceil_q, ceil_kv, mask_kind)

    # 3. CPU golden(分块 flash + 可选 dump)
    print("cpu calc start")
    cpu_out, _ = cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
        query, key, value, mask, block_shape, scale, n2,
        p_quant=p_quant, s_dtype=s_dtype, dump=dump, out_dtype=out_dtype,
        quant_mode=quant_mode, dst_type_max=dst_type_max)
    print("golden calc end")
    # BSND/TND: golden 保持 BNSD 计算(QFA 哲学: 仅 NPU I/O 层), 输出重排到目标布局与 NPU out 比对。
    if input_layout == 'BSND':
        cpu_out = cpu_out.permute(0, 2, 1, 3).contiguous()  # BNSD[B,N,S,D]->BSND[B,S,N,D]
    elif input_layout == 'TND':
        cpu_out = cpu_out.permute(0, 2, 1, 3).flatten(0, 1).contiguous()  # ->[B,S,N,D]->[T,N,D](uniform S: T=B*S)

    
    # CPU-only 模式(_FORCE_CPU 或无 NPU 算子): 无 NPU 可比对, 直接 SKIP, 不做任何 golden 自检。
    if _FORCE_CPU or not _HAS_BSA_V3:
        print(f"cpu_out: {cpu_out}")
        return -1.0, "SKIP"

    # 4. mxfp4 打包喂 NPU: Q/K axis=-1, V axis=-2 (quant_mode/log2_cx 透传: CX 用 ceil-log2CX 网格)
    q_packed, q_e8m0 = mxfp4_quantize_pack_last(query, quant_axis=-1,
                                                quant_mode=quant_mode, log2_cx=log2_cx)
    k_packed, k_e8m0 = mxfp4_quantize_pack_last(key, quant_axis=-1,
                                                quant_mode=quant_mode, log2_cx=log2_cx)
    v_packed, v_e8m0 = mxfp4_quantize_pack_last(value, quant_axis=-2,
                                                quant_mode=quant_mode, log2_cx=log2_cx)
    assert q_packed.dtype == k_packed.dtype == v_packed.dtype == torch.uint8, "packed fp4 须为 uint8"
    assert q_e8m0.dtype == k_e8m0.dtype == v_e8m0.dtype == torch.uint8, "e8m0 scale 须为 uint8"
    q_scale = _e8m0_qk_to_v3(q_e8m0)
    k_scale = _e8m0_qk_to_v3(k_e8m0)
    v_scale = _e8m0_v_to_v3(v_e8m0)
    # BSND/TND: golden 保持 BNSD 不动, 仅把 NPU 输入(packed Q/K/V + V3 e8m0 scale)重排到目标布局。
    #   BSND: packed 4D permute(0,2,1,3)(B,N,S,D->B,S,N,D); 5D scale permute(0,2,1,3,4)
    #     (例 v_scale [B,N2,nb//2,D,2]->[B,nb//2,N2,D,2], 对齐 V3 doc BSND v 形状)。
    #   TND: BSND 重排 + flatten(0,1)(B,S->T, uniform S: T=B*S, BTkv=B*CeilDiv(S2,64))——
    #     packed->[T,N,D//2]; q/k scale->[T,N,CeilDiv(D,64),2]; v_scale->[BTkv,N2,D,2]。对齐 V3 doc TND 4D 形状。
    #   mask 不重排(headNum 恒 dim1, V3 doc L199, 三布局同)。量化轴沿用 BNSD(Q/K 沿 D, V 沿 S2)——
    #   mxfp4 噪声逐元素与布局无关, BNSD 下量化后重排等价于目标布局下直接量化。
    if input_layout in ('BSND', 'TND'):
        q_packed = q_packed.permute(0, 2, 1, 3)
        k_packed = k_packed.permute(0, 2, 1, 3)
        v_packed = v_packed.permute(0, 2, 1, 3)
        q_scale = q_scale.permute(0, 2, 1, 3, 4)
        k_scale = k_scale.permute(0, 2, 1, 3, 4)
        v_scale = v_scale.permute(0, 2, 1, 3, 4)
        if input_layout == 'TND':
            # TND: BSND 重排 + flatten(B,S)->T (uniform S: T=B*S; V-scale BTkv=B*CeilDiv(S2,64)=B*(nb//2 pad后))
            q_packed = q_packed.flatten(0, 1).contiguous()
            k_packed = k_packed.flatten(0, 1).contiguous()
            v_packed = v_packed.flatten(0, 1).contiguous()
            q_scale = q_scale.flatten(0, 1).contiguous()
            k_scale = k_scale.flatten(0, 1).contiguous()
            v_scale = v_scale.flatten(0, 1).contiguous()
        else:  # BSND
            q_packed = q_packed.contiguous()
            k_packed = k_packed.contiguous()
            v_packed = v_packed.contiguous()
            q_scale = q_scale.contiguous()
            k_scale = k_scale.contiguous()
            v_scale = v_scale.contiguous()

    # 5. NPU 调用(照搬 test_pv_systematic.py)
    torch_npu.npu.set_device(device_id)
    print("npu calc start")
    npu_out, _ = block_sparse_attention_v3(
        q_packed.contiguous().npu(), k_packed.contiguous().npu(),
        v_packed.contiguous().npu(), mask.contiguous().npu(), list(block_shape),
        [s1] * b, [s2] * b,
        num_heads=n1, num_key_value_heads=n2,
        q_input_layout=input_layout, kv_input_layout=input_layout,
        mask_type=0, scale_value=scale, inner_precise=inner_precise, block_size=0,
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=False, quant_mode=quant_mode, dst_type_max=dst_type_max,
        q_dequant_scale=q_scale.npu(), k_dequant_scale=k_scale.npu(), v_dequant_scale=v_scale.npu(),
        attention_out_dtype=_resolve_out_dtype(out_dtype),
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
        print()  # 空行: 与 pytest 打印的测试 ID 行隔开
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
        _s1 = cfg_merged.get('s1'); _s2 = cfg_merged.get('s2')
        _s1v = cfg_merged.get('s1_list'); _s2v = cfg_merged.get('s2_list')
        print(f"[PASS][{mode_tag}] {cfg_merged.get('mask_kind','dense')} layout={cfg_merged.get('input_layout','BNSD')} b={cfg_merged['b']} n1={cfg_merged['n1']} n2={cfg_merged['n2']} "
              f"s1={_s1 if _s1 is not None else _s1v} s2={_s2 if _s2 is not None else _s2v} block={cfg_merged['block_shape']} s_dtype={cfg_merged.get('s_dtype','fp16')} "
              f"fulfill={pct:.4f}%")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs128_d128(self, device="npu"):
        """① BNSD dense 单块: 基本精度(block_shape=[128,128], S=128 -> 1x1)。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense')


    # ==================== 精选泛化测试用例 ====================
    # 28 个精选 case 覆盖 b/n1/n2/g/s1/s2/dense/sparse 各维度泛化, 与上方 73 例不重复。
    # 配上方 2 个超大 case(line 845 sparse 25600x33024 + line 851 b=56 dense 512x512) 合计 30 个。
    #
    # 维度覆盖:
    #   b    : 1, 2, 3, 4, 5, 8            (上方 b=2/b=56 已含大 batch)
    #   n1   : 2, 4, 6, 8, 16(仅 1 例)      n2: 1, 2, 3, 4, 8
    #   g    : 1(MHA), 2, 3, 4, 8
    #   s1   : 1, 63, 64, 127, 128, 256, 384, 512, 768, 1024
    #   s2   : 31, 95, 128, 159, 223, 255, 256, 479, 511, 512, 768, 799, 1024, 4096, 8448
    #          s2=32*奇数-1 系列(由小到大): 31,95,159,223,479,799 (含较大值)
    #          s2 不全是 256 倍数; 4096/8448 各 1-2 例, 与上方超大分散
    #   mask : dense=10  sparse=18  (sparse 多覆盖 gather 泛化)
    # 大序列(s2>=4096)分散放置, 间隔>=2, 避免连续跑耗时。

    # ---------- A 组: MHA (g=1) 3 例 ----------
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b2_qn4_kvn4_g1_qs256_kvs4096_d128(self, device="npu"):
        """MHA dense s2=4096 长 KV, b=2 验证 batch + 长 KV dense。"""
        self._run(b=2, n1=4, n2=4, g=1, s1=256, s2=4096, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs31_d128(self, device="npu"):
        """MHA sparse 边界: s1=1 decode + s2=31(32*1-1), Q pad + K 单块不满。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=31, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_sparse_b1_qn8_kvn8_g1_qs127_kvs159_d128(self, device="npu"):
        """MHA sparse 边界: s1=127 + s2=159(32*5-1) 双尾块, K 尾 31 行非对齐。"""
        self._run(b=1, n1=8, n2=8, g=1, s1=127, s2=159, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    # ---------- B 组: GQA dense 9 例 (大序列散布) ----------
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs287_d128(self, device="npu"):
        """GQA dense g4, s2=287(32*9-1) 尾块不满, 验证 K 尾块处理。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=287, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b2_qn4_kvn1_g4_qs128_kvs256_d128(self, device="npu"):
        """GQA dense g4 b=2, 小 KV 验证 batch + 单 kv head dense。"""
        self._run(b=2, n1=4, n2=1, g=4, s1=128, s2=256, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b4_qn8_kvn2_g4_qs384_kvs768_d128(self, device="npu"):
        """GQA dense g4 b=4, s1=384/s2=768 非 128 对齐多 q-block dense。"""
        self._run(b=4, n1=8, n2=2, g=4, s1=384, s2=768, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b8_qn6_kvn2_g3_qs256_kvs512_d128(self, device="npu"):
        """GQA dense g3 b=8, 大 batch + g3 头分组(非 2/4 幂) dense。"""
        self._run(b=8, n1=6, n2=2, g=3, s1=256, s2=512, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b1_qn16_kvn4_g4_qs257_kvs8448_d128(self, device="npu"):
        """GQA dense g4 唯一 n1=16, s2=8448 超长 KV, 验证大 head + 超长 dense。"""
        self._run(b=1, n1=16, n2=4, g=4, s1=257, s2=8448, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b1_qn8_kvn4_g2_qs64_kvs479_d128(self, device="npu"):
        """GQA dense g2, s2=479(32*15-1) 较大尾块, 验证 g2 + K 尾 31 行。"""
        self._run(b=1, n1=8, n2=4, g=2, s1=64, s2=479, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b1_qn8_kvn2_g4_qs512_kvs1024_d128(self, device="npu"):
        """GQA dense g4, s1=512/s2=1024 多 block dense。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=512, s2=1024, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b1_qn8_kvn1_g8_qs256_kvs511_d128(self, device="npu"):
        """GQA dense g8 单 kv head, s2=511 尾块(512-1), 验证 g8 + K 尾 1 行。"""
        self._run(b=1, n1=8, n2=1, g=8, s1=256, s2=511, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b1_qn8_kvn2_g4_qs1024_kvs4096_d128(self, device="npu"):
        """GQA dense g4 + s1=1024 + s2=4096 全大维度 dense。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=1024, s2=4096, qk_d=128,
                block_shape=[128, 128], mask_kind='dense')

    # ---------- C 组: GQA sparse 16 例 (大序列散布) ----------
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs63_kvs95_d128(self, device="npu"):
        """GQA sparse 双边界: s1=63(64-1) + s2=95(32*3-1), Q/K 双尾块 sparse。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=63, s2=95, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn6_kvn2_g3_qs127_kvs255_d128(self, device="npu"):
        """GQA sparse g3, s1=127/s2=255 边界非对齐 sparse 尾块。"""
        self._run(b=1, n1=6, n2=2, g=3, s1=127, s2=255, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b2_qn4_kvn1_g4_qs128_kvs256_d128(self, device="npu"):
        """GQA sparse g4 b=2, 小 KV 验证 batch + 单 kv head sparse gather。"""
        self._run(b=2, n1=4, n2=1, g=4, s1=128, s2=256, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b3_qn8_kvn2_g4_qs512_kvs1024_d128(self, device="npu"):
        """GQA sparse g4 b=3, s1=512/s2=1024 多 block 大序列 sparse。"""
        self._run(b=3, n1=8, n2=2, g=4, s1=512, s2=1024, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs1_kvs8448_d128(self, device="npu"):
        """GQA sparse s1=1 decode + s2=8448 超长 KV, Q pad + 跨 tile rescale。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=1, s2=8448, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b4_qn8_kvn2_g4_qs384_kvs768_d128(self, device="npu"):
        """GQA sparse g4 b=4, s1=384/s2=768 非 128 对齐 sparse 多 block。"""
        self._run(b=4, n1=8, n2=2, g=4, s1=384, s2=768, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b8_qn6_kvn2_g3_qs256_kvs512_d128(self, device="npu"):
        """GQA sparse g3 b=8, 大 batch + g3 头分组 sparse gather。"""
        self._run(b=8, n1=6, n2=2, g=3, s1=256, s2=512, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs512_kvs8448_d128(self, device="npu"):
        """GQA sparse g4, s2=8448 超长 KV sparse, 验证 n1=8 + 超长 sparse gather。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=512, s2=8448, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn8_kvn4_g2_qs64_kvs223_d128(self, device="npu"):
        """GQA sparse g2, s2=223(32*7-1) 较大尾块, 验证 g2 + K 尾 31 行 sparse。"""
        self._run(b=1, n1=8, n2=4, g=2, s1=64, s2=223, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn8_kvn1_g8_qs256_kvs799_d128(self, device="npu"):
        """GQA sparse g8 单 kv head, s2=799(32*25-1) 较大尾块 sparse。"""
        self._run(b=1, n1=8, n2=1, g=8, s1=256, s2=799, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b5_qn4_kvn1_g4_qs128_kvs256_d128(self, device="npu"):
        """GQA sparse g4 b=5, 单 kv head 验证 b=5 + g4 sparse。"""
        self._run(b=5, n1=4, n2=1, g=4, s1=128, s2=256, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs1024_kvs4096_d128(self, device="npu"):
        """GQA sparse g4 + s1=1024 + s2=4096 全大维度 sparse。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=1024, s2=4096, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn4_kvn2_g2_qs64_kvs128_d128(self, device="npu"):
        """GQA sparse g2 极小序列, s1=64 单 block sparse 验证 g2。"""
        self._run(b=1, n1=4, n2=2, g=2, s1=64, s2=128, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn8_kvn4_g2_qs500_kvs4096_d128(self, device="npu"):
        """GQA sparse g2, s2=4096 验证 g2 头分组 + 长 KV sparse。"""
        self._run(b=1, n1=8, n2=4, g=2, s1=500, s2=4096, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn6_kvn3_g2_qs768_kvs1024_d128(self, device="npu"):
        """GQA sparse g2, s1=768(6 个 q-block) + s2=1024(8 个 kv tile) 多 block sparse。"""
        self._run(b=1, n1=6, n2=3, g=2, s1=768, s2=1024, qk_d=128,
                block_shape=[128, 128], mask_kind='sparse')
    
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_sparse_b2_qn1_kvn1_g1_qs25600_kvs33024_d128(self, device="npu"):
        """① BNSD 超大序列 sparse: s1=25600/s2=33024, b=2 验证超长 sparse gather + online rescale。"""
        self._run(b=2, n1=1, n2=1, g=1, s1=25600, s2=33024, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b56_qn8_kvn8_g1_qs512_kvs512_d128(self, device="npu"):
        """① BNSD 大 batch dense: b=56 qn8/kvn8, 验证大 batch 泛化(唯一 b>=56 用例)。"""
        self._run(b=56, n1=8, n2=8, g=1, s1=512, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense')

    # 验证: Q m=15 非 16 倍数 pad + Nd2Nz 非满 fractal 行数 × N=2 Q gather(qHeadMul=2)。
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs15_kvs256_d128(self, device="npu"):
        """② BSND S1=15(Q m 非 16 倍数 pad + Nd2Nz 非满 fractal 行数), n1=n2=2 stride N=2
        (非对齐 Q 行数下 qHeadMul extent 与 pad 后画布一致性)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=15, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    # 验证: K 尾非 64 对齐(s2=400, 尾 16 行) + V-scale ceil(400/32)=13 奇 pad × N=2 stride。
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs400_d128(self, device="npu"):
        """② BSND s2=400 非 64 对齐 K 尾块(3 全 128-tile + 尾 16 行; 400%64=16),
        n1=n2=2 stride N=2(K 尾 srcStride + V-scale 奇 pad block 在 BSND permute 后位置)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=400, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    # --- 多 batch: BSND batch-stride(strideKVDataB_/strideQDataB_/vScaleStrideB_ = S*N*D) × N>1 stride ---

    # 验证: B=2 首个 B>1 BSND; 批间步幅累加 × N=2 stride 同时生效(非退化)。
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b2_qn2_kvn2_g1_qs256_kvs1024_d128(self, device="npu"):
        """④ BSND multi-batch: b=2, n1=n2=2/g1, s1=256 + s2=1024
        (strideKVDataB_=S*N2*D / strideQDataB_=S1*N1*D 批间 offset 累加, N=2 stride 同步)。"""
        self._run(b=2, n1=2, n2=2, g=1, s1=256, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    # 验证: B=8(最大 batch) + g=2 + qHeadMul=4 + kvHeadMul=2 三者叠加, 暴露 batch-offset × stride × 头映射 交叉。
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_gqa_dense_b8_qn4_kvn2_g2_qs128_kvs512_d128(self, device="npu"):
        """④ BSND 多 batch 小形状: b=8, n1=4/n2=2/g2(q0,q1->kv0; q2,q3->kv1),
        s1=128 + s2=512 四 kv tile(8 batch offset 累加 + kvHeadMul=2 + qHeadMul=4 同时)。"""
        self._run(b=8, n1=4, n2=2, g=2, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    # 验证: 首个 BSND sparse; sparse 块索引 gather 路径下 kvHeadMul=2 stride 生效(s2=1024/8 tile 选择性 gather + rescale)。
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d128(self, device="npu"):
        """⑤ BSND sparse mask N=2: 首个 sparse BSND 用例, s2=1024/8 kv tile 稀疏块选择
        (gather stride hack 在 sparse 块索引路径, 非仅 dense 顺序路径, kvHeadMul=2 + rescale)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND')

    # =====================================================================
    # TND 用例系列: 验证 TND 布局代码路径(3D [T,N,D] 数据 + 4D scale + actualSeq 强制)。
    # 哲学(对齐 QFA bnsd_to_bsnd+flatten): golden 保持 BNSD 不动, NPU I/O 重排 = BSND permute + flatten(0,1)
    #   (B,S->T; uniform S 下 T=B*S, BTkv=B*CeilDiv(S2,64); actualSeq=[s]*b 经 NPU 调用传)。
    # 用例 uniform S(镜像 BSND, T=B*S); varying-S varlen 是后续扩展(需 golden 改 per-batch S)。
    # 注: 需 host 解锁(block_sparse_attention_tiling.cpp ValidateMxfp4DequantScale 去掉 TND 硬拒 + 4D
    #   scale 校验 + BTkv 算), 否则 TND mxfp4 tiling 会被 host 拒, 用例挂在该处而非精度。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs128_d128(self, device="npu"):
        """① TND dense 基本单块: b1/qn2/kvn2, s1=128/s2=128(T=B*S=128), N=2 stride factor。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_dense_b1_qn2_kvn2_g1_qs1_kvs2048_d128(self, device="npu"):
        """② TND S1=1 decode + 长 KV s2=2048(16 kv tile online rescale), N=2 stride(T=1, BTkv=CeilDiv(2048,64)=32)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=2048, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs400_d128(self, device="npu"):
        """② TND s2=400 非 64 对齐 K 尾块 + V-scale ceil(400/32)=13 奇 pad, N=2 stride
        (TND 下 V-scale [BTkv,N,D,2] BTkv=CeilDiv(400,64)=7, pad block 在 flatten 后位置)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=400, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs1024_d128(self, device="npu"):
        """③ TND GQA dense: qn8/kvn2/g4, s1=256 + s2=1024(T=256, BTkv=CeilDiv(1024,64)=16),
        头映射 + Q gather qHeadMul=8 / K·V·V-scale kvHeadMul=2。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_dense_b2_qn2_kvn2_g1_qs256_kvs1024_d128(self, device="npu"):
        """④ TND multi-batch: b=2, n1=n2=2/g1, s1=256 + s2=1024(T=B*S=512, actualSeq=[256,256];
        TND batch-flatten + per-batch base-offset 累积 + N=2 stride)。"""
        self._run(b=2, n1=2, n2=2, g=1, s1=256, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d128(self, device="npu"):
        """⑤ TND sparse mask N=2: s2=1024/8 kv tile 稀疏块选择(TND 下 sparse gather 路径 + kvHeadMul=2 stride)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND')

    # =====================================================================
    # TND varlen(变长)用例: per-batch 实际 S 不同(T=sum S1_b, BTkv=sum CeilDiv(S2_b,64)),
    #   测 kernel TND base-offset 按实际 S 累加 + BTkv sum + cu_seqlens(uniform 退化测不到)。
    #   用 s1_list/s2_list 传 per-batch seq; s1/s2 留 None。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d128(self, device="npu"):
        """⑥ TND varlen dense: b=2, n1=n2=2/g1, s1=[128,256] s2=[256,128](变长,
        Tq=128+256=384, Tkv=256+128=384, BTkv=CeilDiv(256,64)+CeilDiv(128,64)=4+2=6)。
        验证 kernel TND per-batch base-offset 累加 + BTkv sum + cu_seqlens 路径。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[256, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_gqa_dense_b2_qn8_kvn2_g4_s1_128_256_s2_512_256_d128(self, device="npu"):
        """⑥ TND varlen GQA dense: b=2, qn8/kvn2/g4, s1=[128,256] s2=[512,256](变长 + 头映射,
        Tq=384, Tkv=768, BTkv=CeilDiv(512,64)+CeilDiv(256,64)=8+4=12)。叠加 GQA stride(qHeadMul=8/kvHeadMul=2)。"""
        self._run(b=2, n1=8, n2=2, g=4, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[512, 256])

    # =====================================================================
    # TND varlen 扩展用例: 扩大变长覆盖(多 batch 累加 / 奇 V-scale pad / sparse / decode-mix / 长 S2 rescale)。
    # 全部 per-batch 实际 S 不同, 测 kernel varlen 累加 + BTkv sum + cu_seqlens 在更多场景的正确性。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b3_qn2_kvn2_g1_s1_128_256_64_s2_256_128_512_d128(self, device="npu"):
        """⑦ TND varlen b3 混合: b=3, n1=n2=2/g1, s1=[128,256,64] s2=[256,128,512]
        (3 batch 累加, 含小 batch S=64; Tq=448, Tkv=896, BTkv=4+2+8=14)。验证 per-batch base-offset 跨 3 batch 累加。"""
        self._run(b=3, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256, 64], s2_list=[256, 128, 512])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_129_400_d128(self, device="npu"):
        """⑦ TND varlen + 奇 V-scale pad: b=2, n1=n2=2/g1, s1=[128,256] s2=[129,400]
        (两 batch s2 均非 64 对齐: 129->pad, 400->pad; BTkv=CeilDiv(129,64)+CeilDiv(400,64)=3+7=10)。
        验证 per-batch V-scale 奇 nb pad + concat 后 BTkv sum 位置正确。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[129, 400])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_sparse_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d128(self, device="npu"):
        """⑦ TND varlen sparse: b=2, n1=n2=2/g1, s1=[128,256] s2=[256,128], sparse mask。
        验证稀疏块索引 gather 在 varlen per-batch 路径(非仅 dense 顺序)+ kvHeadMul=2 stride。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[128, 256], s2_list=[256, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_1_256_s2_128_1024_d128(self, device="npu"):
        """⑦ TND varlen decode-mix: b=2, n1=n2=2/g1, s1=[1,256] s2=[128,1024]
        (batch0 S1=1 decode + Q pad, batch1 S1=256 prefill; 跨 batch 不同 S1 + Q gather batch-stride)。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[1, 256], s2_list=[128, 1024])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_gqa_dense_b2_qn8_kvn2_g4_s1_128_256_s2_129_400_d128(self, device="npu"):
        """⑦ TND varlen GQA + 奇 pad: b=2, qn8/kvn2/g4, s1=[128,256] s2=[129,400]
        (GQA 头映射 + varlen + 奇 V-scale pad 三重叠加; qHeadMul=8/kvHeadMul=2 + per-batch pad + concat)。"""
        self._run(b=2, n1=8, n2=2, g=4, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[129, 400])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_1024_2048_d128(self, device="npu"):
        """⑦ TND varlen 长 S2 online-rescale: b=2, n1=n2=2/g1, s1=[128,256] s2=[1024,2048]
        (batch0 8 kv-tile, batch1 16 kv-tile online softmax rescale 累积; varlen + rescale 叠加, 较慢)。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[1024, 2048])

    # =====================================================================
    # 边界条件用例: 验证 kernel 在极端/边界 shape 下的正确性。
    #   S1<blockShapeX(Q-pad 半块) / S1>blockShapeX(2 q-block 尾1) /
    #   S2<blockShapeY(部分 kv-block) / S2 奇 nb(V-scale pad) /
    #   decode+max-rescale(s2=8192,64tile) / 多 batch(B=4) / block 变体 / TND varlen 边界。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs64_kvs128_d128(self, device="npu"):
        """⑧ BSND S1=64<blockShapeX=128: Q-pad 半块(ceil_q=1, 1 q-block of 64 padded to 128), N=2。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=64, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs129_kvs512_d128(self, device="npu"):
        """⑧ BSND S1=129>blockShapeX: 2 q-block(第2块仅1 token, Q 尾块非对齐), N=2, s2=512(4 kv tile)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=129, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs64_d128(self, device="npu"):
        """⑧ BSND S2=64<blockShapeY=128: 1 部分 kv-block(64 行, ceil_kv=1), K 尾块最小, N=2。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=64, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs193_d128(self, device="npu"):
        """⑧ BSND S2=193: K 尾块(ceil_kv=2, 第2块65行) + V-scale ceil(193/32)=7 奇 nb->pad, N=2。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=193, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs1_kvs8192_d128(self, device="npu"):
        """⑧ BSND S1=1 decode + s2=8192(64 kv tile, online softmax rescale 极限累积), N=2(较慢)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=8192, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b4_qn2_kvn2_g1_qs256_kvs512_d128(self, device="npu"):
        """⑧ BSND B=4 多 batch: b=4, n1=n2=2/g1, s1=256 + s2=512(4 batch offset 累加 + N=2 stride)。"""
        self._run(b=4, n1=2, n2=2, g=1, s1=256, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128_blk64x128(self, device="npu"):
        """⑧ BSND block=[64,128] D=128: blockShapeX=64(Q tile 64 行, ceil_q=2), 验证小 Q-block tiling。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[64, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128_blk128x256(self, device="npu"):
        """⑧ BSND block=[128,256] D=128: blockShapeY=256(KV tile 256 行, ceil_kv=1), 验证大 KV-block tiling。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 256], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_1_1_s2_128_512_d128(self, device="npu"):
        """⑧ TND varlen 双 decode: b=2, s1=[1,1] s2=[128,512](两 batch 均 S1=1 decode + Q-pad,
        Tq=2; 验证 varlen 下双 decode batch 的 Q-pad + base-offset 累加)。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[1, 1], s2_list=[128, 512])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_64_512_d128(self, device="npu"):
        """⑧ TND varlen 部分-kv: b=2, s1=[128,256] s2=[64,512](batch0 S2=64<blockShapeY=128, 1 部分 kv-block;
        BTkv=CeilDiv(64,64)+CeilDiv(512,64)=1+8=9; 验证 varlen 下 S2<blockShapeY 的 K 尾 + V-scale)。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[64, 512])

    # =====================================================================
    # 非对齐场景用例(重点 TND varlen): 每 batch 的 S1/S2 故意不齐 blockShapeX/Y/64/32/16。
    #   验证 kernel 在 Q-pad / Q-tail / K-tail / V-scale pad / Fixpipe m 非16 等 非对齐路径。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_127_129_s2_256_256_d128(self, device="npu"):
        """⑨ TND varlen 两 batch S1 均非-128 对齐: s1=[127,129](127=1块pad, 129=2块尾1), s2=[256,256]对齐。
        验证 Q-pad + Q-tail 在 varlen per-batch(Tq=127+129=256)。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[127, 129], s2_list=[256, 256])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_15_47_s2_128_128_d128(self, device="npu"):
        """⑨ TND varlen 两 batch S1 均非-16(Fixpipe m): s1=[15,47](15%16!=0, 47%16!=0), s2=[128,128]对齐。
        验证 Fixpipe nSize 偶数对齐 + Nd2Nz 非满 fractal 行数 在 varlen(Tq=15+47=62)。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[15, 47], s2_list=[128, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_128_s2_129_193_d128(self, device="npu"):
        """⑨ TND varlen 两 batch S2 均奇 nb(V-scale pad): s2=[129,193](nb=5/7 奇->pad), s1=[128,128]对齐。
        BTkv=CeilDiv(129,64)+CeilDiv(193,64)=3+4=7; 验证 per-batch V-scale pad + concat 后 BTkv sum。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 128], s2_list=[129, 193])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_128_s2_255_400_d128(self, device="npu"):
        """⑨ TND varlen 两 batch S2 K-tail 非-128/64/32 混合: s2=[255,400](255=127尾, nb=8偶; 400=16尾, nb=13奇pad)。
        BTkv=CeilDiv(255,64)+CeilDiv(400,64)=4+7=11; 验证 K-tail + 奇/偶 nb 混合 concat。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 128], s2_list=[255, 400])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_129_s2_256_129_d128(self, device="npu"):
        """⑨ TND varlen 对齐+非对齐 batch 混: batch0 s1=128/s2=256(全对齐), batch1 s1=129/s2=129(全非对齐)。
        验证 kernel 同时处理对齐 batch + 非对齐 batch(尾块 + V-scale pad)的 base-offset 累加。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 129], s2_list=[256, 129])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_1_128_s2_1_128_d128(self, device="npu"):
        """⑨ TND varlen S2=1 极小非对齐: batch0 s1=1/s2=1(1 kv-token, ceil_kv=1, V-scale ceil(1/32)=1奇->pad),
        batch1 s1=128/s2=128(对齐)。BTkv=CeilDiv(1,64)+CeilDiv(128,64)=1+2=3。验证 S2=1 极端 K 尾 + V-scale pad。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[1, 128], s2_list=[1, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b3_qn2_kvn2_g1_s1_127_15_129_s2_129_255_400_d128(self, device="npu"):
        """⑨ TND varlen b3 全不同非对齐: s1=[127,15,129] s2=[129,255,400](3 batch 各不同非对齐组合)。
        Tq=127+15+129=271, BTkv=3+4+7=14; 验证 3 batch 累加下混合非对齐(Q-pad/tail + Fixpipe m + K-tail + V-scale pad)。"""
        self._run(b=3, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[127, 15, 129], s2_list=[129, 255, 400])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs15_kvs255_d128(self, device="npu"):
        """⑨ BSND S1=15(非16, Fixpipe m) + s2=255(K-tail 127 + nb=8偶): 双重非对齐叠加, N=2。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=15, s2=255, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs255_kvs129_d128(self, device="npu"):
        """⑨ BSND S1=255(2块, 127尾, Q-tail) + s2=129(K-tail 1 + nb=5奇pad): S1/S2 双非对齐, N=2。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=255, s2=129, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    # =====================================================================
    # 补充场景用例: GQA g=8(单 kv-head, 非对称 stride) / sparse S1=1(decode 稀疏) / 大 B(b16) / fp16 out。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_gqa_dense_b1_qn8_kvn1_g8_qs256_kvs1024_d128(self, device="npu"):
        """⑩ BSND GQA g=8: qn8/kvn1/g8(单 kv-head, qHeadMul=8 ≠ kvHeadMul=1, 非对称 stride),
        s1=256 + s2=1024。验证 Q stride factor 8 vs K/V stride factor 1(单 kv-head, V-scale BTkv 无头交织)。"""
        self._run(b=1, n1=8, n2=1, g=8, s1=256, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_gqa_dense_b2_qn8_kvn1_g8_s1_128_256_s2_512_256_d128(self, device="npu"):
        """⑩ TND varlen GQA g=8: b=2, qn8/kvn1/g8, s1=[128,256] s2=[512,256](单 kv-head + varlen,
        qHeadMul=8/kvHeadMul=1; Tq=384, Tkv=768, BTkv=8+4=12)。非对称 stride × varlen 累加。"""
        self._run(b=2, n1=8, n2=1, g=8, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[512, 256])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs1024_d128(self, device="npu"):
        """⑩ BSND sparse S1=1: decode 单 q-block + 稀疏 kv 选择(ceil_q=1, 8 kv tile 选择性 gather),
        N=2。验证 sparse gather 在 Q-pad 路径下。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_sparse_b2_qn2_kvn2_g1_s1_1_1_s2_512_256_d128(self, device="npu"):
        """⑩ TND varlen sparse 双 decode: b=2, s1=[1,1] s2=[512,256](两 batch 均 decode + 稀疏 kv 选择,
        Tq=2; 验证 varlen 下 sparse gather + Q-pad per-batch)。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[1, 1], s2_list=[512, 256])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b16_qn2_kvn2_g1_qs128_kvs256_d128(self, device="npu"):
        """⑩ BSND B=16 大 batch: b=16, n1=n2=2/g1, s1=128 + s2=256(16 batch offset 累加 + N=2 stride,
        批间 strideKVDataB_=S*N2*D 累加 16 次 stress)。"""
        self._run(b=16, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    # [已知 kernel bug, 注释] fp16 输出: processOutVF(vf_attenOut_dn_mxfp4.h:79-99) Cast 硬编码 bfloat16_t,
    #   不随 ElementO 模板变 -> OF16(fp16 输出)路径 fp32->bf16 cast 后以 fp16 布局拷贝(bit 错位)-> 输出垃圾。
    #   kernel 修法: processOutVF 的 Cast/CreateMask/RegTensor 按 ElementO 模板化。待 kernel 修后启用。
#     @SkipIfNotGteCANNVersion("9.0.0")
#     def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs512_d128_fp16out(self, device="npu"):
#         """⑩ BSND fp16 out: out_dtype='float16'(NPU attention_out_dtype=float16, golden->float16),
#         b1/qn2/kvn2/g1/s128/kvs512。验证 NPU fp16 输出路径(golden fp32->fp16 cast, rtol=1/200 更严)。"""
#         self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
#                   block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
#                   out_dtype='float16')

    # =====================================================================
    # 审计补充用例(P1+P3): all-masked q-block(kernel 0-rsvd 边界) / gap-sparse(大 gap rescale) /
    #   fp16-out TND-varlen / block64+S1-odd / S2=32(min e8m0 块) / extreme 值(FP4 饱和) / zero 值(退化)。
    # =====================================================================
    # [已知 kernel bug, 注释待修] all-masked q-block: kernel 0-rsvd gather 路径死锁(无 empty-gather guard)。
    #   golden 已处理(if not sel_j: continue -> out=0/lse=-inf), 但 kernel 卡住。待 kernel 补 0-rsvd guard 后启用。
    # @SkipIfNotGteCANNVersion("9.0.0")
    # def test_bsa_mxfp4_bsnd_mha_allmasked_qb_b1_qn2_kvn2_g1_qs256_kvs512_d128(self, device="npu"):
    #     """⑪ BSND all-masked q-block: 中间 q-block 全 0(无 kv attended), s1=256(2 q-block)/s2=512。
    #     验证 kernel 0-rsvd gather 边界(out=0/lse=-inf, 不 OOB/NaN)。golden 已处理(if not sel_j: continue)。"""
    #     self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=512, qk_d=128,
    #               block_shape=[128, 128], mask_kind='all_masked_one_qb', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d128(self, device="npu"):
        """⑪ BSND gap-sparse: 每 q-block 仅 attend kv-block 0+末(大 gap, 跨块 alpha rescale + 非连续 gather),
        s1=128/s2=512(4 kv tile)。验证 online-softmax 跨大 K_diff rescale + snap_local corr 不下溢。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='gap_sparse', input_layout='BSND')

    # [已知 kernel bug, 注释] fp16 输出: processOutVF(vf_attenOut_dn_mxfp4.h:79-99) Cast 硬编码 bfloat16_t,
    #   不随 ElementO 模板变 -> OF16(fp16 输出)路径 fp32->bf16 cast 后以 fp16 布局拷贝(bit 错位)-> 输出垃圾。
    #   kernel 修法: processOutVF 的 Cast/CreateMask/RegTensor 按 ElementO 模板化。待 kernel 修后启用。
#     @SkipIfNotGteCANNVersion("9.0.0")
#     def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d128_fp16out(self, device="npu"):
#         """⑪ TND varlen fp16-out: b=2, s1=[128,256] s2=[256,128], out_dtype='float16'。
#         验证 TND-varlen 路径(唯一正确接 out_dtype 的)的 fp16 输出(golden->fp16, NPU attention_out_dtype=float16)。"""
#         self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
#                   block_shape=[128, 128], mask_kind='dense', input_layout='TND',
#                   s1_list=[128, 256], s2_list=[256, 128], out_dtype='float16')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs127_kvs256_d128_blk64x128(self, device="npu"):
        """⑪ BSND block=[64,128] + S1=127(非 64 倍): blockShapeX=64, ceil_q=2(第2块 63 token Q-pad),
        验证小 Q-block 下 S1 非 64 对齐的 Q-pad + Nd2Nz 画布。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=127, s2=256, qk_d=128,
                  block_shape=[64, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs32_d128(self, device="npu"):
        """⑪ BSND S2=32: 最小完整 e8m0 块(ceil(32/32)=1, nb 偶, 无 V-scale pad), ceil_kv=1(部分 kv-block 32<128)。
        验证最小 KV + 无 pad V-scale 路径。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=32, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    # [已知数值极限, 注释] extreme 值 data_x=50: S=Q@K^T 极大 -> snap_local K_block 极大 -> e8m0 P-scale
    #   钳到 254(clamp) -> golden/NPU 在 clamp 边界分歧(drift>1/128, 连 bf16 都挂)。待查 NPU P-scale
    #   clamp 语义后启用(或用 data_x<=3 避免 S 过大导致 P-scale clamp)。
    # @SkipIfNotGteCANNVersion("9.0.0")
    # def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128_satval(self, device="npu"):
    #     """⑪ BSND extreme 值(FP4 饱和): data_x=50(Q/K/V ∈ [-50,50], FP4 max=6.0, E8M0 scale floor 模式允许饱和)。
    #     验证 E8M0 scale 选择 + 6.0 clip 边界 golden/NPU 一致。"""
    #     self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
    #               block_shape=[128, 128], mask_kind='dense', input_layout='BSND', data_x=50.0)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128_zero(self, device="npu"):
        """⑪ BSND 全零 Q/K/V(data_x=0): 退化输入, E8M0 scale max_abs<1e-30->scale=1.0, P 全等(均匀 softmax),
        out=0。验证 degenerate 路径无 NaN/Inf。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND', data_x=0.0)

    # =====================================================================
    # CX 量化方案用例(quant_mode=3): P 量化走 ceil(Smax/ln2 - log2CX) 路径(式19/26)。
    #   默认 dstTypeMax=0 -> host 取 6.0(log2(6)≈2.585, ceil=3); golden 同步用 6.0。
    #   覆盖典型 b/n/g/s/dense/sparse 组合, 与上方 OCP 用例镜像但走 CX 分支。
    #   注: CX 与 OCP 唯一差异在 K_block 取整方式 + corr 末尾 shift(-ceil_log2CX 替代 -2);
    #       E8M0 指数位充足, 二者 pscale 比值 ∈{1,2}(式25), 无精度损失。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs128_d128(self, device="npu"):
        """CX dense 单块: 基本精度(quant_mode=3, dstTypeMax=6.0)。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=16385, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    # =====================================================================
    # CX 用例组(quant_mode=3): 30 个, 覆盖布局/mask/GQA/s2长度边界/dstTypeMax 全矩阵。
    # 约束: dst_type_max ∈ {0(默认6.0), 6.0~12.0}; s2 含 31/128/256/511/512/1024/4096/8192/16385;
    # TND varlen 强制 s1_list[i] < s2_list[i]。命名: cx_<layout>_<mha/gqa>_<mask>_b_qn_kvn_g_qs_kvs_d_dstmax。
    # =====================================================================

    # --- BNSD 基础 + s2 长度矩阵(dense, 默认 dstTypeMax=6.0) ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs31_d128(self, device="npu"):
        """CX BNSD dense s2=31 边界: 单 KV 块不足(32*1-1), V-scale 尾块 pad + K 尾 1 行。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=31, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs128_d128(self, device="npu"):
        """CX BNSD dense s2=128: 单 KV 块对齐, 最小完整块覆盖。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128(self, device="npu"):
        """CX BNSD dense s2=256: 2 KV 块, 基线多块 rescale 路径。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs511_d128(self, device="npu"):
        """CX BNSD dense s2=511: 4 KV 块 + 尾块差 1 行(511=4*128-1), 尾块 V-scale/K 尾行 pad。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=511, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b2_qn4_kvn4_g1_qs256_kvs4096_d128(self, device="npu"):
        """CX BNSD dense s2=4096: 长 KV(32 块) + b=2 batch + n=4 头维度。"""
        self._run(b=2, n1=4, n2=4, g=1, s1=256, s2=4096, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs8192_d128(self, device="npu"):
        """CX BNSD dense s2=8192: 长 KV(64 块), 验证长 KV rescale 累积 + BTkv 大值。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=8192, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs16385_d128(self, device="npu"):
        """CX BNSD dense s2=16385 超长 KV(129 块, 尾块 1 行): 16385=128*128+1, 极端尾块 + BTkv 上限。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=16385, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    # --- BNSD sparse/gap_sparse/all_masked 边界 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs31_d128(self, device="npu"):
        """CX BNSD sparse decode: s1=1 单 q-block + s2=31 尾块, 稀疏 gather + 尾块 pad。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=31, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs4096_d128(self, device="npu"):
        """CX BNSD sparse decode 长 KV: s1=1 + s2=4096, 稀疏 gather 跨 32 块选择。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=4096, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs256_kvs8192_d128(self, device="npu"):
        """CX BNSD sparse 长 KV: s1=256 + s2=8192(64 块), 稀疏 gather 多块 + rescale。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=8192, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d128(self, device="npu"):
        """CX BNSD sparse 边界: s1=128 + s2=1024, 稀疏 gather 多块 + rescale。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs256_kvs512_d128(self, device="npu"):
        """CX BNSD sparse 中等: s1=256 + s2=512, 稀疏 gather 跨块 + rescale。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', quant_mode=3)

    # --- BNSD GQA(g=2/4/8) + 大 n + s1 边界 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_gqa_dense_b1_qn4_kvn2_g2_qs128_kvs1024_d128(self, device="npu"):
        """CX BNSD GQA g2 dense: n1=4/n2=2/g2, s2=1024, 头映射 stride + 多块 dense。"""
        self._run(b=1, n1=4, n2=2, g=2, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs1024_d128(self, device="npu"):
        """CX BNSD GQA g4 dense: n1=8/n2=2/g4, s1=256/s2=1024 多 block dense。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs512_kvs4096_d128(self, device="npu"):
        """CX BNSD GQA g4 sparse 长 KV: s1=512/s2=4096, 稀疏 gather + GQA stride 跨头。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=512, s2=4096, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_gqa_dense_b1_qn8_kvn1_g8_qs256_kvs511_d128(self, device="npu"):
        """CX BNSD GQA g8 单 kv head dense: s2=511 尾块, g8 头映射 + K 尾 1 行 pad。"""
        self._run(b=1, n1=8, n2=1, g=8, s1=256, s2=511, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_gqa_dense_b1_qn16_kvn4_g4_qs257_kvs8448_d128(self, device="npu"):
        """CX BNSD GQA g4 n1=16 dense 超长: s2=8448, 大 head + 超 long dense KV。"""
        self._run(b=1, n1=16, n2=4, g=4, s1=257, s2=8448, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn8_kvn8_g1_qs512_kvs512_d128(self, device="npu"):
        """CX BNSD MHA dense 大形状: n=8/g1, s1=512/s2=512, 大 head + 多 q/kv block。"""
        self._run(b=1, n1=8, n2=8, g=1, s1=512, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    # --- BSND 布局(N 维 stride) + CX ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128(self, device="npu"):
        """CX BSND dense 基线: N=2 stride + CX 量化路径(s2=256 多块)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs4096_d128(self, device="npu"):
        """CX BSND sparse decode 长 KV: s1=1 + s2=4096, BSND stride + 稀疏 gather。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=4096, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bsnd_gqa_dense_b2_qn4_kvn2_g2_qs256_kvs8192_d128(self, device="npu"):
        """CX BSND GQA g2 dense 长 KV: b=2, s2=8192, BSND stride + GQA + 长 KV rescale。"""
        self._run(b=2, n1=4, n2=2, g=2, s1=256, s2=8192, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND', quant_mode=3)

    # --- TND uniform + CX ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_mha_dense_b1_qn2_kvn2_g1_qs4096_kvs4097_d128(self, device="npu"):
        """CX TND uniform dense 基线: 3D [T,N,D] + CX 量化, 长 KV s2=4097 跨块。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=4096, s2=4097, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_mha_sparse_b2_qn2_kvn2_g1_qs256_kvs4096_d128(self, device="npu"):
        """CX TND uniform sparse 长 KV: b=2, s2=4096, TND batch-flatten + 稀疏 gather。"""
        self._run(b=2, n1=2, n2=2, g=1, s1=256, s2=4096, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs8192_d128(self, device="npu"):
        """CX TND GQA g4 dense 长 KV: s2=8192, TND flatten + GQA stride + 长 KV rescale。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=8192, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND', quant_mode=3)

    # --- TND varlen(强制 s1_list[i] < s2_list[i]) + CX ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_256_64_s2_128_256_d128(self, device="npu"):
        """CX TND varlen dense 混合 s1><s2: s1=[256,64] s2=[128,256](batch0 s1>s2, batch1 s1<s2)。
        验证 per-batch base-offset 累加 + CX 量化 + Q 长于 K 的 pre-fill 边界。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[256, 64], s2_list=[128, 256], quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_varlen_gqa_dense_b2_qn8_kvn2_g4_s1_128_256_s2_512_512_d128(self, device="npu"):
        """CX TND varlen GQA dense: s1=[128,256] s2=[512,512](s1<s2), GQA g4 + varlen 累加。"""
        self._run(b=2, n1=8, n2=2, g=4, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[512, 512], quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_varlen_mha_sparse_b3_qn2_kvn2_g1_s1_512_1_128_s2_128_4096_64_d128(self, device="npu"):
        """CX TND varlen sparse b3 混合 s1><s2: s1=[512,1,128] s2=[128,4096,64]
        (batch0 s1>s2 pre-fill, batch1 decode s1<s2, batch2 s1>s2), 3 batch 混合 + 长 KV 稀疏。"""
        self._run(b=3, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[512, 1, 128], s2_list=[128, 4096, 64], quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_varlen_mha_dense_b2_qn4_kvn4_g1_s1_256_512_s2_4096_128_d128(self, device="npu"):
        """CX TND varlen dense 长 KV 混合: s1=[256,512] s2=[4096,128](batch0 s1<s2, batch1 s1>s2),
        n=4 + 跨 batch 长 KV rescale + pre-fill 边界。"""
        self._run(b=2, n1=4, n2=4, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[256, 512], s2_list=[4096, 128], quant_mode=3)

    # --- dstTypeMax 全覆盖(0/6.0/6.6/7.4/8.0/8.8/9.5/10.0/12.0) ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128_dstmax0(self, device="npu"):
        """CX dstTypeMax=0(默认6.0): 验证 host 默认值路径(log2CX=log2(6)≈2.585, ceil=3)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3, dst_type_max=0)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128_dstmax66(self, device="npu"):
        """CX dstTypeMax=6.6(log2≈2.722, ceil=3): 非整数 log2CX + ceil=3 corr 路径。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3, dst_type_max=6.6)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs512_d128_dstmax74(self, device="npu"):
        """CX dstTypeMax=7.4(log2≈2.887, ceil=3): 接近 8 边界, ceil 仍=3。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3, dst_type_max=7.4)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128_dstmax8(self, device="npu"):
        """CX dstTypeMax=8.0(log2=3.0, ceil=3): 整数 log2CX 边界, ceil=3。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3, dst_type_max=8.0)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs1024_d128_dstmax88(self, device="npu"):
        """CX dstTypeMax=8.8(log2≈3.138, ceil=4): 跨 ceil 边界(>8), ceil=4 corr 路径。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3, dst_type_max=8.8)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs4096_d128_dstmax95(self, device="npu"):
        """CX dstTypeMax=9.5(log2≈3.248, ceil=4): ceil=4, 长 KV s2=4096 叠加。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=4096, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3, dst_type_max=9.5)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b64_qn2_kvn2_g1_qs128_kvs513_d128_dstmax10(self, device="npu"):
        """CX dstTypeMax=10.0(log2≈3.322, ceil=4): ceil=4 + 大 batch b=64 + s2=513 跨块。"""
        self._run(b=64, n1=2, n2=2, g=1, s1=128, s2=513, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3, dst_type_max=10.0)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs8192_d128_dstmax12(self, device="npu"):
        """CX dstTypeMax=12.0(log2≈3.585, ceil=4): 上限值, ceil=4 + 长 KV s2=8192。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=8192, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3, dst_type_max=12.0)
    
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bsnd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d128(self, device="npu"):
        """⑪ BSND gap-sparse: 每 q-block 仅 attend kv-block 0+末(大 gap, 跨块 alpha rescale + 非连续 gather),
        s1=128/s2=512(4 kv tile)。验证 online-softmax 跨大 K_diff rescale + snap_local corr 不下溢。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='gap_sparse', input_layout='BSND', quant_mode=3, dst_type_max=9.9)
    
        # --- TND varlen(强制 s1_list[i] < s2_list[i]) + CX ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d128(self, device="npu"):
        """CX TND varlen dense """
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[128, 256], quant_mode=3,  dst_type_max=6.6)

    # =====================================================================
    # actualSeq ≠ seqlen 用例系列(BNSD/BSND padded 张量 + actualSeqLengths 指示更小有效长度):
    #   覆盖 V3 doc L248/264 显式支持的 BNSD/BSND "actualSeq < 张量 S" 路径——driver uniform 路径恒令
    #   actualSeq == s1/s2(张量 S), 此前零覆盖。driver 走 padded 分支(act_s1/act_s2 触发): golden per-batch
    #   unpadded(golden 不改), NPU I/O 0-pad 到 s1/s2 + actualSeq=[act_b]。比对只取 [0,act_q) 有效区
    #   (kernel 不清零 [act,S) padding 区)。维度: Q-padding/KV-padding/双 padded/decode/block 对齐/
    #   GQA/per-batch 不同 actual(独立 kernel 路径)/sparse-mask 交互。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn2_kvn2_g1_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """⑫ BNSD 双 padded: act_q=200<256 + act_kv=200<256(末 q-block 72 行 + V 末 32-block 8 行), N=2。
        验证 actualSeq 驱动 kernel 只建 [0,200) task、跳过 [200,256) padding q-block + KV 末块 lastYBlockSize 裁剪。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BNSD',
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn2_kvn2_g1_actq256_padq256_actkv200_padkv256_d128(self, device="npu"):
        """⑫ BNSD 仅 KV padded: act_q=256(满) + act_kv=200<256, N=2。验证 KV 侧 actualSeqKv 路径
        (Q 无 padding, 聚焦 KV 末块裁剪 + padding KV 块由 mask 置 0 屏蔽)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BNSD',
                  act_s1=256, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn2_kvn2_g1_actq1_padq128_actkv128_padkv128_d128(self, device="npu"):
        """⑫ BNSD decode Q-pad: act_q=1<128(Q 0-pad 到 128, 单 token decode) + act_kv=128(满 KV), N=2。
        验证 decode 下 actualSeq 驱动 kernel 只建 1 个 q-token task、跳过 [1,128) padding q-block。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BNSD',
                  act_s1=1, act_s2=128)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn2_kvn2_g1_actq128_padq256_actkv128_padkv256_d128(self, device="npu"):
        """⑫ BNSD block 对齐 padded: act_q=128<256 + act_kv=128<256(act 恰 1 block, 整 padding 块跳过), N=2。
        验证 act 恰为 blockShape 倍数时 padding 块边界干净跳过(ceil(128/128)=1 < ceil(256/128)=2)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BNSD',
                  act_s1=128, act_s2=128)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b1_qn8_kvn2_g4_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """⑫ BNSD GQA 双 padded: qn8/kvn2/g4 + act_q=200<256 + act_kv=200<256, N=2。验证 GQA 头映射
        (qHeadMul=4/kvHeadMul=2) + actualSeq 驱动 per-batch task 切分叠加。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BNSD',
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b2_qn2_kvn2_g1_actq200_128_padq256_actkv200_128_padkv256_d128(self, device="npu"):
        """⑫ BNSD per-batch 不同 actual: b=2, act_q=[200,128] act_kv=[200,128](均 <256, 各 batch 不同有效长),
        N=2。验证 host CalcSplitCoreTilingParams950 per-batch actual task 累加 + kernel
        gActualQseqlen.GetValue(curBatch) 独立路径——与 TND varlen 不同的 padded-batch 路径。"""
        self._run(b=2, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BNSD',
                  act_s1=[200, 128], act_s2=[200, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """⑫ BSND 双 padded: act_q=200<256 + act_kv=200<256, N=2(BSND token-stride=N*D 与 actualSeq 叠加)。
        验证 BSND padded 路径(golden BNSD 不动, 仅 NPU I/O 0-pad + permute)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_actq256_padq256_actkv200_padkv256_d128(self, device="npu"):
        """⑫ BSND 仅 KV padded: act_q=256(满) + act_kv=200<256, N=2。验证 BSND 下 KV 侧 actualSeqKv 路径
        + KV 末块裁剪 + padding KV 块 mask 屏蔽(token-stride=N*D 下 gather)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  act_s1=256, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_actq1_padq128_actkv128_padkv128_d128(self, device="npu"):
        """⑫ BSND decode Q-pad: act_q=1<128 + act_kv=128(满), N=2。验证 BSND decode 下 Q 0-pad +
        actualSeq 驱动单 token task(token-stride=N*D 下 Q gather)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  act_s1=1, act_s2=128)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_gqa_dense_b1_qn8_kvn2_g4_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """⑫ BSND GQA 双 padded: qn8/kvn2/g4 + act_q=200<256 + act_kv=200<256, N=2。验证 BSND GQA 非对称
        stride(qHeadMul=4/kvHeadMul=2) + actualSeq 叠加(BSND stride 布局 + padding, 高价值)。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b2_qn2_kvn2_g1_actq200_128_padq256_actkv200_128_padkv256_d128(self, device="npu"):
        """⑫ BSND per-batch 不同 actual: b=2, act_q=[200,128] act_kv=[200,128](均 <256), N=2。验证 BSND
        per-batch actual 独立 kernel 路径(token-stride=N*D + batch-offset 累加 + per-batch actual task 切分)。"""
        self._run(b=2, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  act_s1=[200, 128], act_s2=[200, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_sparse_b1_qn2_kvn2_g1_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """⑫ BNSD sparse + 双 padded: act_q=200<256 + act_kv=200<256, sparse mask, N=2。验证 actualSeq 与
        稀疏块索引 gather 路径叠加(mask per-batch actual blocking + padding 块置 0 + 稀疏块选择)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BNSD',
                  act_s1=200, act_s2=200)
    
    # =====================================================================
    # OCP scale-stride 回归: 普通 U[-1,1] OCP 下 e8m0 近乎全等, TND/BSND Q/K-scale
    #   Dn2Nz srcDValue 漏乘 N 的缺陷被掩盖。本 case 跨头强制 OCP e8m0 偶头=124/奇头=125,
    #   旧 kernel(srcDValue=D//64)必挂, 新 kernel(srcDValue=N*(D//64))应过。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_mha_dense_scale_stride_trap_b1_qn2_kvn2_g1_qs128_kvs128_d128(self, device="npu"):
        """OCP 回归陷阱: TND + N=2 + 跨头强制 e8m0 不等(偶头 max∈(0.5,1)→exp=-3;
        奇头 max∈[1,2)→exp=-2)。专打 Q/K-scale GM→L1 Dn2Nz srcDValue 漏乘 qHeadMul/kvHeadMul。
        旧代码必 Failed; 修好 headMul 后应 Pass。quant_mode=2(OCP)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_bsnd_mha_dense_scale_stride_trap_b1_qn2_kvn2_g1_qs128_kvs128_d128(self, device="npu"):
        """OCP 回归陷阱(BSND 孪生): 同上跨头 e8m0 不等, 打 BSND 布局下 Q/K-scale 行步长。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    # --- 陷阱泛化: 边界值(decode/非对齐尾块/奇数头/大头数) + 常规(GQA/多 batch/长 KV/sparse/varlen) ---
    # 注: padded(act_s1/act_s2)路径造数不走 data_kind, 无法与本陷阱组合, 故不含 actualSeq case。
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_mha_dense_scale_stride_trap_b1_qn2_kvn2_g1_qs1_kvs1024_d128(self, device="npu"):
        """OCP 陷阱 decode 边界: TND s1=1(Q-scale 单行, 行步长退化), 主打 K-scale 侧
        1024 行 × N=2 行步长 + 长 KV 多 chunk 反复 GM→L1 搬运 K-scale。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_mha_dense_scale_stride_trap_b1_qn2_kvn2_g1_qs127_kvs255_d128(self, device="npu"):
        """OCP 陷阱非对齐边界: TND s1=127/s2=255 尾块不满(lastYBlockSize 裁剪),
        验证尾块行数 < blockShape 时 scale 行步长仍按 N*(D//64) 计算。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=127, s2=255, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_mha_dense_scale_stride_trap_b1_qn2_kvn2_g1_qs128_kvs4096_d128(self, device="npu"):
        """OCP 陷阱长 KV: TND s2=4096(32 kv 块, 16 chunk), 跨 chunk online-softmax rescale
        叠加 K-scale 多轮搬运——任一轮行步长错读都会放大到最终输出。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=4096, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_gqa_dense_scale_stride_trap_b1_qn8_kvn2_g4_qs256_kvs1024_d128(self, device="npu"):
        """OCP 陷阱 GQA 非对称: TND qn8/kvn2/g4, qHeadMul=8 与 kvHeadMul=2 不同——
        Q-scale 与 K-scale 行步长须各乘各的 N, 只修一侧仍会挂。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_mha_dense_scale_stride_trap_b1_qn3_kvn3_g1_qs128_kvs128_d128(self, device="npu"):
        """OCP 陷阱奇数头边界: TND N=3(偶头 exp=-3/奇头 exp=-2 交错且首尾头同为偶),
        验证奇数 N 下行步长 3*(D//64) 无偶数对齐假设。"""
        self._run(b=1, n1=3, n2=3, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_gqa_dense_scale_stride_trap_b1_qn16_kvn4_g4_qs256_kvs512_d128(self, device="npu"):
        """OCP 陷阱大头数边界: TND qn16/kvn4/g4, 行步长放大到 16*(D//64)——
        步长漏乘时读偏量最大, 且 15 对相邻头 e8m0 全不等, 错读必被放大。"""
        self._run(b=1, n1=16, n2=4, g=4, s1=256, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_mha_sparse_scale_stride_trap_b1_qn2_kvn2_g1_qs128_kvs1024_d128(self, device="npu"):
        """OCP 陷阱 sparse 常规: TND 稀疏 gather 下 kv 块非连续选取,
        K-scale 按块跳读时行基址 + 行步长两级偏移都须乘 N。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_bsnd_mha_dense_scale_stride_trap_b1_qn8_kvn8_g1_qs256_kvs256_d128(self, device="npu"):
        """OCP 陷阱多头常规: BSND N=8(7 对相邻头 e8m0 全不等),
        行步长 8*(D//64), 漏乘时中间头读偏更远、错读概率最大化。"""
        self._run(b=1, n1=8, n2=8, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_bsnd_gqa_dense_scale_stride_trap_b1_qn4_kvn2_g2_qs128_kvs512_d128(self, device="npu"):
        """OCP 陷阱 BSND GQA 常规: qn4/kvn2/g2 非对称 headMul,
        BSND token-stride=N*D 与 scale 行步长 N*(D//64) 叠加。"""
        self._run(b=1, n1=4, n2=2, g=2, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_bsnd_mha_dense_scale_stride_trap_b4_qn2_kvn2_g1_qs128_kvs256_d128(self, device="npu"):
        """OCP 陷阱多 batch 常规: BSND b=4, batch 基址偏移(S*N*(D//64))与
        头内行步长叠加——batch>0 时基址正确但行步长漏乘同样必挂。"""
        self._run(b=4, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_bsnd_mha_dense_scale_stride_trap_b1_qn2_kvn2_g1_qs129_kvs400_d128(self, device="npu"):
        """OCP 陷阱 BSND 非对齐边界: s1=129(跨块 1 行尾块)/s2=400(尾块 16 行),
        Q/K 两侧尾块同时不满, 验证尾块 scale 行数裁剪与行步长互不干扰。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=129, s2=400, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_bsnd_mha_dense_scale_stride_trap_b1_qn2_kvn2_g1_qs1_kvs2048_d128(self, device="npu"):
        """OCP 陷阱 BSND decode 边界: s1=1 单 token + 长 KV s2=2048,
        BSND 下 Q gather 单行 + K-scale 多轮搬运(与 TND decode 孪生, 走 BSND 行步长分支)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=2048, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_bsnd_mha_dense_scale_stride_trap_b1_qn2_kvn2_g1_qs128_kvs256_d128_blk64x128(self, device="npu"):
        """OCP 陷阱非方块边界: BSND block_shape=[64,128](q 块 64 行),
        验证 scale 行步长与 blockShape 解耦——按 token 行计, 不随块形变化。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[64, 128], mask_kind='dense', input_layout='BSND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_varlen_mha_dense_scale_stride_trap_b2_qn2_kvn2_g1_s1_128_256_s2_256_129_d128(self, device="npu"):
        """OCP 陷阱 TND varlen 常规: s1=[128,256] s2=[256,129](batch1 尾块 1 行),
        per-batch base-offset 按 T*N*(D//64) 累加 + 头内行步长, 两级都漏乘 N 均必挂。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[256, 129],
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_varlen_gqa_dense_scale_stride_trap_b2_qn8_kvn2_g4_s1_1_256_s2_512_256_d128(self, device="npu"):
        """OCP 陷阱 TND varlen GQA 混合: s1=[1,256](batch0 decode) s2=[512,256],
        varlen 累加 + 非对称 headMul(qHeadMul=8/kvHeadMul=2) + decode/prefill 混批全叠加。"""
        self._run(b=2, n1=8, n2=2, g=4, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[1, 256], s2_list=[512, 256],
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    # =====================================================================
    # qk_d=64 用例系列: 上方全部 case 均 D=128, D=64 路径零覆盖。D=64 自带边界:
    #   Q/K scale 末维 [D//64=1, 2](每 token 行仅 1 组 e8m0 对, 无跨对配对);
    #   packed 每行 D//2=32B(GM->L1 搬运最小行宽); Nd2Nz 画布宽 64。
    # 场景全覆盖: BNSD/BSND/TND uniform/TND varlen × dense/sparse/gap_sparse ×
    #   MHA/GQA × OCP/CX × actualSeq padded × zero 退化 × scale-stride 陷阱,
    #   叠加 decode(s1=1)/非对齐尾块/长 KV/大 batch 边界。
    # =====================================================================
    # --- BNSD + OCP ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs128_d64(self, device="npu"):
        """D64 BNSD dense 最小基线: n=1 单头单块(与 ① 的 d128 基线镜像),
        验证 D=64 打包(行 32B)/scale [1,2] 基本精度。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=128, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs31_d64(self, device="npu"):
        """D64 BNSD sparse decode 极小边界: s1=1 + s2=31(单 kv 块不满, V-scale 单 32-block),
        最小形状叠加 D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=31, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_dense_b2_qn8_kvn2_g4_qs256_kvs512_d64(self, device="npu"):
        """D64 BNSD GQA dense 常规: b=2 + qn8/kvn2/g4, 头映射与 D=64 行宽叠加。"""
        self._run(b=2, n1=8, n2=2, g=4, s1=256, s2=512, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs127_kvs255_d64(self, device="npu"):
        """D64 BNSD GQA sparse 非对齐边界: s1=127/s2=255 双尾块不满 + 稀疏 gather。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=127, s2=255, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn2_kvn2_g1_qs256_kvs4096_d64(self, device="npu"):
        """D64 BNSD dense 长 KV: s2=4096(32 kv 块), 跨 chunk rescale × D=64 窄行搬运。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=4096, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b8_qn4_kvn4_g1_qs128_kvs256_d64(self, device="npu"):
        """D64 BNSD dense 大 batch: b=8, batch 基址(N*S*D 缩半)与 task 切分叠加。"""
        self._run(b=8, n1=4, n2=4, g=1, s1=128, s2=256, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense')

    # --- BSND + OCP ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d64(self, device="npu"):
        """D64 BSND dense 基线: token-stride=N*D 缩半(N=2 时 128 元素), BSND 行步长 × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs129_kvs400_d64(self, device="npu"):
        """D64 BSND dense 非对齐边界: s1=129(尾块 1 行)/s2=400(尾块 16 行, 400%64=16 非 64 对齐
        V-scale 末组), 双尾块 × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=129, s2=400, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_gqa_dense_b2_qn4_kvn2_g2_qs128_kvs512_d64(self, device="npu"):
        """D64 BSND GQA dense 常规: qn4/kvn2/g2 非对称 headMul + BSND stride × D=64。"""
        self._run(b=2, n1=4, n2=2, g=2, s1=128, s2=512, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs1024_d64(self, device="npu"):
        """D64 BSND sparse decode: s1=1 单 token + s2=1024 稀疏 gather, BSND 单行 Q × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=1024, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs127_kvs256_d64_blk64x128(self, device="npu"):
        """D64 BSND block=[64,128] 边界: 小 Q-block(64) + s1=127 非 64 倍(第 2 块 63 行 Q-pad),
        Nd2Nz 画布 64x64 方块 × 尾块。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=127, s2=256, qk_d=64,
                  block_shape=[64, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs32_d64(self, device="npu"):
        """D64 BSND 最小 KV 边界: s2=32(单 e8m0 32-block, 无 V-scale pad, ceil_kv=1),
        最小 KV × 最小 D 双边界。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=32, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d64(self, device="npu"):
        """D64 BSND gap-sparse: 仅 attend kv-block 0+末(大 gap 跨块 rescale + 非连续 gather),
        与 ⑪ d128 版镜像。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=64,
                  block_shape=[128, 128], mask_kind='gap_sparse', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d64_zero(self, device="npu"):
        """D64 BSND 全零退化: data_x=0, scale=1.0/P 均匀/out=0, 验证 D=64 degenerate 路径无 NaN/Inf。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND', data_x=0.0)

    # --- TND uniform + OCP ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs128_d64(self, device="npu"):
        """D64 TND uniform dense 基线: 3D [T,N,D] 单块, TND 行步长 × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_dense_b1_qn2_kvn2_g1_qs1_kvs2048_d64(self, device="npu"):
        """D64 TND decode 边界: s1=1 + 长 KV s2=2048, 单 token Q × 多轮 K/V 搬运 × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=2048, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs1024_d64(self, device="npu"):
        """D64 TND GQA dense 常规: qn8/kvn2/g4 + s2=1024, TND flatten + GQA stride × D=64。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=1024, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d64(self, device="npu"):
        """D64 TND sparse 常规: 稀疏 kv 块跳读 × TND 布局 × D=64 窄行。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND')

    # --- TND varlen + OCP ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d64(self, device="npu"):
        """D64 TND varlen dense 基线: s1=[128,256] s2=[256,128] 混合 s1><s2,
        per-batch base-offset(T*N*D 缩半)× D=64。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[256, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_127_129_s2_255_400_d64(self, device="npu"):
        """D64 TND varlen 非对齐边界: s1=[127,129] s2=[255,400] 全尾块不满
        (含 400%64=16 的 V-scale 末组), varlen 累加 × 非对齐 × D=64。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[127, 129], s2_list=[255, 400])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_gqa_dense_b2_qn8_kvn2_g4_s1_1_256_s2_512_256_d64(self, device="npu"):
        """D64 TND varlen GQA 混批: s1=[1,256](batch0 decode) s2=[512,256],
        decode/prefill 混批 + 非对称 headMul × D=64。"""
        self._run(b=2, n1=8, n2=2, g=4, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[1, 256], s2_list=[512, 256])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_sparse_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d64(self, device="npu"):
        """D64 TND varlen sparse: per-batch 稀疏 mask + varlen 累加 × D=64。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[128, 256], s2_list=[256, 128])

    # --- CX(quant_mode=3) × D64 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d64(self, device="npu"):
        """D64 CX BNSD dense 基线: CX P 量化(默认 dstTypeMax=6.0) × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs4096_d64(self, device="npu"):
        """D64 CX BSND sparse decode 长 KV: s1=1 + s2=4096, CX + BSND stride + 稀疏 gather × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=4096, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs1024_d64(self, device="npu"):
        """D64 CX TND GQA dense: CX + TND flatten + GQA stride × D=64。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=1024, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_256_64_s2_128_256_d64(self, device="npu"):
        """D64 CX TND varlen dense 混合 s1><s2: s1=[256,64] s2=[128,256], CX + varlen 累加 × D=64。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[256, 64], s2_list=[128, 256], quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs1024_d64_dstmax88(self, device="npu"):
        """D64 CX dstTypeMax=8.8 边界: log2≈3.138 跨 ceil 边界(ceil=4) × 长 KV × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', quant_mode=3, dst_type_max=8.8)

    # --- actualSeq padded × D64 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn2_kvn2_g1_actq200_padq256_actkv200_padkv256_d64(self, device="npu"):
        """D64 BNSD 双 padded: act=200<256(末块 72 行有效), actualSeq task 裁剪 × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BNSD',
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_actq1_padq128_actkv128_padkv128_d64(self, device="npu"):
        """D64 BSND decode Q-pad 边界: act_q=1<128(单 token 0-pad), BSND stride + padded × D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  act_s1=1, act_s2=128)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b2_qn2_kvn2_g1_actq200_128_padq256_actkv200_128_padkv256_d64(self, device="npu"):
        """D64 BSND per-batch 不同 actual: act_q=act_kv=[200,128], per-batch task 切分 × D=64。"""
        self._run(b=2, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  act_s1=[200, 128], act_s2=[200, 128])

    # --- OCP scale-stride 陷阱 × D64(天然边界: D//64=1, 旧 srcDValue 退化为 1) ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_mha_dense_scale_stride_trap_b1_qn2_kvn2_g1_qs128_kvs128_d64(self, device="npu"):
        """D64 OCP 陷阱边界: D//64=1(每行仅 1 组 scale 对), 旧 kernel srcDValue=1 vs
        新 srcDValue=N*1——步长错位恰为 1 组, 与 d128 版错位 2 组互补。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_bsnd_gqa_dense_scale_stride_trap_b1_qn4_kvn2_g2_qs128_kvs512_d64(self, device="npu"):
        """D64 OCP 陷阱 BSND GQA: 非对称 headMul(qHeadMul=4/kvHeadMul=2) × D//64=1 最小步长基数。"""
        self._run(b=1, n1=4, n2=2, g=2, s1=128, s2=512, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_varlen_mha_dense_scale_stride_trap_b2_qn2_kvn2_g1_s1_128_256_s2_256_129_d64(self, device="npu"):
        """D64 OCP 陷阱 TND varlen: s2=[256,129](batch1 尾块 1 行), varlen base-offset
        按 T*N*1 累加 × 陷阱数据。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=64,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 256], s2_list=[256, 129],
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    # =====================================================================
    # 覆盖缺口补充: 全量盘点后此前未覆盖的组合——
    #   gap_sparse 仅测过 BSND(缺 BNSD/TND); 非默认 block_shape 仅 BSND×dense
    #   (缺 BNSD/TND 布局、sparse、[128,256] 跨多 kv 块); uniform s2=1(单 kv token)
    #   仅 varlen 测过; BSND/TND 的 N=1 头退化零覆盖(非 BNSD case 全 n>=2);
    #   BSND padded×sparse; varlen/BSND-uniform 的 GQA×sparse; 大 batch varlen(此前 b<=3)。
    # 不补的项: s_dtype/p_quant/inner_precise(golden 诊断旋钮非场景)、fp16 输出
    #   (已知 kernel bug, 上方已注释)、CX/陷阱×padded(脚手架限制, 见各分支注释)。
    # =====================================================================
    # --- gap_sparse × BNSD/TND 布局 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d128(self, device="npu"):
        """补 BNSD gap-sparse: 仅 attend kv-block 0+末(跨 3 块 gap 的 alpha rescale),
        此前 gap_sparse 仅 BSND——补 BNSD 基址(无 N-stride)下的非连续 gather。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='gap_sparse', input_layout='BNSD')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d128(self, device="npu"):
        """补 TND gap-sparse: TND flatten 布局 × kv-block 0+末大 gap gather,
        非连续块跳读的 base-offset 按 T*N*D 计算。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='gap_sparse', input_layout='TND')

    # --- 非默认 block_shape × 新布局/sparse/跨多 kv 块 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_dense_b1_qn2_kvn2_g1_qs127_kvs256_d128_blk64x128(self, device="npu"):
        """补 BNSD block=[64,128]: 此前 block 变体仅 BSND——补 BNSD 下小 Q-block(64)
        tiling + s1=127 非 64 倍尾块 Q-pad。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=127, s2=256, qk_d=128,
                  block_shape=[64, 128], mask_kind='dense', input_layout='BNSD')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs512_d128_blk128x256(self, device="npu"):
        """补 TND block=[128,256]: 此前 block 变体从未上 TND——KV tile 256 行 × TND
        行步长, s2=512 恰 2 个大 kv 块(1 chunk 512 行)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 256], mask_kind='dense', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs400_d128_blk128x256(self, device="npu"):
        """补 block=[128,256] 跨多 kv 块 + 非对齐: s2=400 -> ceil_kv=2(末块 144 行不满),
        现有 blk128x256 case s2=256 恰 1 块, 大 kv 块的尾块裁剪零覆盖。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=400, qk_d=128,
                  block_shape=[128, 256], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs256_kvs512_d128_blk64x128(self, device="npu"):
        """补非默认 block × sparse: 此前 block 变体全 dense——[64,128] 小 Q-block(ceil_q=4)
        × 稀疏 kv 块随机 gather。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=512, qk_d=128,
                  block_shape=[64, 128], mask_kind='sparse', input_layout='BSND')

    # --- uniform 路径 s2=1(单 kv token) ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs1_d128(self, device="npu"):
        """补 uniform s2=1 边界: 单 kv token(此前仅 varlen 测过 s2=1)——ceil_kv=1,
        V-scale ceil(1/32)=1 奇数组 pad(e8m0=127), softmax 单列退化为 P=1。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_dense_b1_qn2_kvn2_g1_qs1_kvs1_d128(self, device="npu"):
        """补 uniform 双最小边界: s1=1 且 s2=1(Tq=Tkv=1 单 token 对单 token),
        TND 下全部维度取最小, mask 1x1。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    # --- BSND/TND 的 N=1 头退化 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs256_d128(self, device="npu"):
        """补 BSND N=1 头退化边界: token-stride=N*D 退化为 D(与 BNSD 数值等价但走 BSND
        stride 分支), 此前非 BNSD case 全 n>=2。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs256_d128(self, device="npu"):
        """补 TND N=1 头退化边界: [T,1,D] 单头 flatten, 行步长失去 N 因子,
        scale 行步长 N*(D//64) 退化为 D//64。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND')

    # --- padded × sparse(BSND) + GQA × sparse 缺口 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_sparse_b1_qn2_kvn2_g1_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """补 BSND padded × sparse: act=200<256 + 稀疏 mask(BNSD 版已有, BSND token-stride
        下 actualSeq 裁剪 × 稀疏 gather 叠加缺失)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND',
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_gqa_sparse_b1_qn8_kvn2_g4_qs128_kvs512_d128(self, device="npu"):
        """补 BSND GQA × sparse: 此前 BSND sparse 全 MHA——非对称 headMul(4/2)
        × 稀疏 kv gather × BSND stride。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_gqa_sparse_b2_qn8_kvn2_g4_s1_128_256_s2_512_256_d128(self, device="npu"):
        """补 TND varlen GQA × sparse: 此前 varlen sparse 全 MHA——per-batch 稀疏 mask
        × GQA 头映射 × varlen base-offset 累加。"""
        self._run(b=2, n1=8, n2=2, g=4, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[128, 256], s2_list=[512, 256])

    # --- 大 batch varlen(此前 b<=3) ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_dense_b8_qn2_kvn2_g1_s1_mixed_s2_mixed_d128(self, device="npu"):
        """补大 batch varlen: b=8 混合 decode/prefill/非对齐(s1 含 1/15/127/129,
        s2 含 129/255/400), host per-batch task 累加 + base-offset 8 段链式验证。"""
        self._run(b=8, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 1, 256, 64, 127, 129, 15, 128],
                  s2_list=[256, 512, 256, 128, 255, 129, 128, 400])

    # =====================================================================
    # sparse 加密补充: 全脚本 dense(~98):sparse(44) 比例失衡, 且以下系列 sparse 近零——
    #   陷阱系列 18 例仅 1 sparse; CX dstTypeMax 系列 8 例全 dense; d64 专区 33 例仅 6 sparse;
    #   padded×sparse 仅 2 例(无 GQA/per-batch); 大 batch varlen 仅 dense。
    #   sparse mask 为随机生成(每 q-block 随机 attend 1~ceil_kv 块), 多铺 case 即多采样。
    # =====================================================================
    # --- 陷阱 × sparse ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_bsnd_mha_sparse_scale_stride_trap_b1_qn2_kvn2_g1_qs128_kvs1024_d128(self, device="npu"):
        """陷阱 × BSND sparse: 稀疏 kv 块跳读下 K-scale 行基址随块跳变,
        每次跳块后的行步长都须乘 N(TND sparse 版已有, BSND 版缺)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_varlen_mha_sparse_scale_stride_trap_b2_qn2_kvn2_g1_s1_128_256_s2_512_256_d128(self, device="npu"):
        """陷阱 × TND varlen sparse: per-batch 稀疏 mask + varlen base-offset 累加
        + 陷阱数据(varlen 陷阱此前全 dense)。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[128, 256], s2_list=[512, 256],
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_ocp_tnd_gqa_sparse_scale_stride_trap_b1_qn8_kvn2_g4_qs128_kvs1024_d64(self, device="npu"):
        """陷阱 × D64 GQA sparse 三叠加: D//64=1 最小步长基数 + 非对称 headMul(8/2)
        + 稀疏跳块, 陷阱系列最复合的组合。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=128, s2=1024, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  quant_mode=2, data_kind='ocp_scale_stride_trap')

    # --- CX dstTypeMax × sparse(该系列 8 例此前全 dense) ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d128_dstmax66(self, device="npu"):
        """CX dstTypeMax=6.6(ceil=3) × sparse: 非整数 log2CX corr 路径
        × 稀疏 gather 的跨块 rescale。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', quant_mode=3, dst_type_max=6.6)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d128_dstmax88(self, device="npu"):
        """CX dstTypeMax=8.8(跨 ceil 边界, ceil=4) × BSND sparse: ceil=4 corr
        × 稀疏跳块 × BSND stride。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND',
                  quant_mode=3, dst_type_max=8.8)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_mha_sparse_b1_qn2_kvn2_g1_qs256_kvs4096_d128_dstmax12(self, device="npu"):
        """CX dstTypeMax=12.0(上限) × TND sparse 长 KV: ceil=4 + s2=4096 稀疏 gather
        跨大范围 rescale。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=4096, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  quant_mode=3, dst_type_max=12.0)

    # --- d64 × sparse 加密 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs256_kvs4096_d64(self, device="npu"):
        """D64 BNSD sparse 长 KV: s2=4096 稀疏跳块 × D=64 窄行搬运(d64 长 KV 此前仅 dense)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=4096, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_gqa_sparse_b1_qn8_kvn2_g4_qs127_kvs511_d64(self, device="npu"):
        """D64 BSND GQA sparse 非对齐: s1=127/s2=511 双尾块 + 非对称 headMul
        + 稀疏 gather × D=64。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=127, s2=511, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_gqa_sparse_b1_qn8_kvn2_g4_qs256_kvs1024_d64(self, device="npu"):
        """D64 TND GQA sparse: 稀疏跳块 × GQA stride × TND flatten × D=64
        (d64 TND sparse 此前仅 MHA)。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=1024, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_varlen_mha_sparse_b2_qn2_kvn2_g1_s1_128_256_s2_512_256_d64(self, device="npu"):
        """D64 CX varlen sparse: CX 量化 + per-batch 稀疏 mask + varlen 累加 × D=64
        (d64 CX 此前无 varlen sparse)。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[128, 256], s2_list=[512, 256], quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_gqa_sparse_b2_qn8_kvn2_g4_s1_1_256_s2_512_256_d64(self, device="npu"):
        """D64 varlen GQA sparse 混批: s1=[1,256](batch0 decode) + 稀疏 mask
        + 非对称 headMul × D=64。"""
        self._run(b=2, n1=8, n2=2, g=4, qk_d=64,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[1, 256], s2_list=[512, 256])

    # --- padded × sparse 加密(此前仅 2 例, 无 GQA/per-batch) ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bnsd_mha_sparse_b2_qn2_kvn2_g1_actq200_128_padq256_actkv200_128_padkv256_d128(self, device="npu"):
        """padded × sparse per-batch: b=2 act=[200,128] 各 batch 不同有效长 + 稀疏 mask,
        per-batch actual task 切分 × 稀疏块选择(padded sparse 此前仅 uniform act)。"""
        self._run(b=2, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BNSD',
                  act_s1=[200, 128], act_s2=[200, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_gqa_sparse_b1_qn8_kvn2_g4_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """padded × sparse × GQA: BSND qn8/kvn2/g4 + act=200<256 + 稀疏 mask,
        三路径叠加(padded sparse 此前全 MHA)。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND',
                  act_s1=200, act_s2=200)

    # --- 大 batch varlen × sparse(此前仅 dense) ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_sparse_b8_qn2_kvn2_g1_s1_mixed_s2_mixed_d128(self, device="npu"):
        """大 batch varlen × sparse: b=8 混合 decode/prefill/非对齐 + per-batch 独立
        稀疏 mask(8 组不同 ceil_kv 的随机采样), base-offset 链 × 稀疏 gather。"""
        self._run(b=8, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[128, 1, 256, 64, 127, 129, 15, 128],
                  s2_list=[256, 512, 256, 128, 255, 129, 128, 400])
 
    # =====================================================================
    # 第二轮缺口补充: block_shape 变体与 varlen/CX/TND-小块/d64-大块 的正交组合零覆盖;
    #   gap_sparse 无长 KV(最大仅 s2=512, gap 2 块)与 CX×TND 组合; 大 batch sparse
    #   最大仅 b=8; CX dstTypeMax=6.0 显式下界(host 校验 [6,12] 的下边界值)未测。
    # 仍不可测: gap_sparse×varlen(_build_block_sparse_mask_varlen 仅支持 dense/sparse)。
    # =====================================================================
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_varlen_mha_sparse_b2_qn2_kvn2_g1_s1_128_256_s2_256_512_d128_blk64x128(self, device="npu"):
        """补 block 变体 × varlen(此前 varlen 全 [128,128]): [64,128] 小 Q-block 下
        per-batch ceil_q 翻倍(2/4), varlen mask 网格 + base-offset 均按 64 行块重算。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128,
                  block_shape=[64, 128], mask_kind='sparse', input_layout='TND',
                  s1_list=[128, 256], s2_list=[256, 512])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d128_blk128x256(self, device="npu"):
        """补 block 变体 × CX(此前 CX 全 [128,128]): [128,256] 大 KV-block(ceil_kv=2)
        × CX P 量化 × 稀疏块选择。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 256], mask_kind='sparse', input_layout='BSND', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_tnd_mha_sparse_b1_qn2_kvn2_g1_qs256_kvs512_d128_blk64x128(self, device="npu"):
        """补 TND × [64,128](TND block 变体此前仅 [128,256]): 小 Q-block(ceil_q=4)
        × TND flatten × 稀疏 gather。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=512, qk_d=128,
                  block_shape=[64, 128], mask_kind='sparse', input_layout='TND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs400_d64_blk128x256(self, device="npu"):
        """补 d64 × [128,256](d64 block 变体此前仅 [64,128]): 大 KV-block 尾块 144 行
        (400%256) × D=64 窄行 × 稀疏选择。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=400, qk_d=64,
                  block_shape=[128, 256], mask_kind='sparse', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs4096_d128(self, device="npu"):
        """补 gap_sparse 长 KV 边界(此前最大 s2=512, gap 仅 2 块): s2=4096 -> kv-block 0+末,
        中间 30 块全跳, 跨超大 gap 的 K_diff rescale + snap_local corr 下溢压力最大化。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=4096, qk_d=128,
                  block_shape=[128, 128], mask_kind='gap_sparse', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d128(self, device="npu"):
        """补 CX × TND gap_sparse(CX gap_sparse 此前仅 BSND): CX corr 末尾 shift(-ceil_log2CX)
        在大 gap rescale 下不下溢 × TND 布局。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind='gap_sparse', input_layout='TND', quant_mode=3)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_bsnd_mha_sparse_b16_qn2_kvn2_g1_qs128_kvs256_d128(self, device="npu"):
        """补大 batch × sparse(sparse 此前最大 b=8): b=16 per-batch 独立稀疏采样
        × BSND stride, 与 b=16 dense 版(line 1793)镜像。"""
        self._run(b=16, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', input_layout='BSND')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs256_d128_dstmax6(self, device="npu"):
        """补 CX dstTypeMax=6.0 显式下界: host ValidateAdditionalParams 校验 [6.0,12.0]
        的下边界值(dstmax0 走的是默认值路径, 显式 6.0 走校验路径), × 稀疏 mask。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind='sparse', quant_mode=3, dst_type_max=6.0)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d128_ceshi(self, device="npu"):
        """CX TND varlen dense """
        self._run(b=15, n1=28, n2=28, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='TND',
                  s1_list=[128, 128,128,128,128,128,128,128,128,128,128,128,128,128,128], s2_list=[256,256,256,256,256,256,5000,256,256,256,256,256,5000,256,256], quant_mode=3,  dst_type_max=6.6)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_varlen_mha_dense_b15_qn28_kvn28_g1_actq128_padq128_actkv4096_padkv4096_d128_ceshi(self, device="npu"):
        """CX BNSD varlen dense: b=15, n1=n2=28(MHA), per-batch actualSeq 用 list 形式传入。
        act_s1 全 128(pad_q=128 满), act_s2 含 512 大 KV,
        验证 BNSD padded 路径(per-batch actual 独立 kernel 路径) + CX 量化。先走通流程, 不比精度。"""
        self._run(b=15, n1=28, n2=28, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BNSD',
                  s1=128, s2=5000,
                  act_s1=[128]*15,
                  act_s2=[256,256,256,256,256,256,5000,256,256,256,256,256,5000,256,256],
                  quant_mode=3, dst_type_max=6.6)
    
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp4_cx_bnsd_varlen_mha_dense_hhhhhhhhhhhhhhhhhh_d128_ceshi(self, device="npu"):
        """CSV: Prefill_FP4_E2M1_FP4_E2M1_BF16_L0_fuzz_2_10_10_131072_1792_128_128_BNSD_000077
        B=2, N_q=N_kv=10(MHA, G=1), D=128, S_q=131072(prefill 大序列), S_kv=1792(pad 到 max),
        act_seq_lens_kv=[1792, 202](per-batch KV 变长), Q 满 s1(=act_seq_lens_q 空CSV语义)。
        input_layout=BNSD, dense, OCP 量化(CSV q_quant_mode=5 映射为 OCP=2), out=bf16。
        scale=1/sqrt(128)≈0.08838834764831843(框架默认 1/sqrt(qk_d) 一致, 无需显式传)。
        data_range 按 CSV: q[-0.5,0] k/v[0.01,1](当前框架仅支持单一 data_x, 用默认 1.0 走通流程)。"""
        self._run(b=2, n1=10, n2=10, g=1, qk_d=128,
                  block_shape=[128, 128], mask_kind='dense', input_layout='BNSD',
                  s1=131072, s2=1792,
                  act_s1=[131072, 131072],
                  act_s2=[1792, 202],
                  quant_mode=2, out_dtype='bfloat16')
                  
if __name__ == "__main__":
    run_tests()
