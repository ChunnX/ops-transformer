#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ======================================================================================================================
"""
GMM A4W4 独立调用 + 真值对比脚本。

场景轴 (共 24 个, itertools.product 遍历; 输出 dtype 每次可选):
  weightFormat   : ND / NZ                                            (2)
  量化模式       : perchannel scale[E,N] / pergroup scale[E,G,N]      (2, 由 scale 维度自动判定)
  groupListType  : 0 cumsum / 1 count / 2 [E,2]=(groupIdx,groupSize)  (3)
  weightTranspose: False / True                                       (2, 仅 NZ 生效; ND 不支持转置)
  输出 dtype     : FLOAT16 / BFLOAT16

A4W4 约束 (详见 ops-transformer_A4W4/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md):

  [公共约束]
  - x/weight 若需转置, 转置对应 tensor 必须非连续。
  - x/weight 每组 tensor 最后一维 < 65536 (x 不转置指 K 轴 / 转置指 M 轴; weight 不转置指 N 轴 / 转置指 K 轴)。
  - weight 为 FRACTAL_NZ 时, shape 需满足 NZ 格式要求。
  - perTokenScaleOptional: 一般只支持 1 维且长度 = x 的 M; 仅 x/weight/out 均单 tensor 场景。
  - groupListOptional: 输出 TensorList 长度为 1 时, 约束输出有效部分, 未指定部分不参与更新。
  - groupListType: 0 = 非负单调非递减 cumsum; 1 = 非负每组大小 count;
                   2 = 非负, shape [E,2] = [[groupIdx, groupSize]...], groupSize 为每组大小, 非零组前置。
  - groupType: -1 不分组 / 0 M 轴分组 / 2 K 轴分组。
  - actType: 0 NONE / 1 RELU / 2 GELU_TANH / 3 GELU_ERR_FUNC(不支持) / 4 FAST_GELU / 5 SILU。

  [A4W4 场景约束]
  - GroupType=0 (M 轴分组), actType=0, groupListType=0/1/2。
  - x/weight/out 均为单 tensor (TensorList 长度 1)。
  - x: 2D (M,K), 不支持转置; weight: 3D (E,K,N); ND 仅非转置, NZ 支持转置。
  - weight ND 时 N%8==0。
  - scale: perchannel [E,N] / pergroup [E,G,N]; pergroup 时 G 整除 K 且 K/G(=groupSize) 为偶数。
  - 开启 NZ 转置后: K%64==0, N%16==0, 且 pergroup 时 groupSize%64==0。
  - dtype: x/weight=INT4, scale=UINT64(内部 int64 打包), perTokenScale=FLOAT, groupList=INT64, out=FLOAT16/BFLOAT16。

用法:
    python gmm_A4W4_golden.py                     # 跑全部 24 场景
    python gmm_A4W4_golden.py --scenario 3        # 只跑第 3 个场景
    python gmm_A4W4_golden.py --m 512 --k 2048 --n 512 --e 8
"""

import argparse
import itertools

import numpy as np
import torch
import torch_npu

# npu_quantize 在 NZ 块序张量上打 int4 包时, 需要开启内部格式 (allow_internal_format)
torch_npu.npu.config.allow_internal_format = True


# ======================================================================================================================
# 真值 (golden) 实现 —— 对齐 ATK 反量化链路 (MM) + fp64 高精度理论参考 (MM_fp64, 诊断用)
# ======================================================================================================================
def MM(x, weight, per_channel_scale, per_token_scale, m, out_dtype,
       k_num_per_group, dequant_mode):
    """单组量化 matmul. 对齐 ATK 反量化链路 (PASS/FAIL 判定用此版).

    参数:
        x: (m, k) int4 值域整数;
        weight: (k, n) int4 值域整数;
        per_channel_scale: pergroup 时 (k//k_num_per_group, n), perchannel 时 (n,);
        per_token_scale: (m,) fp32;
        m: token 数 (x 行数).
    返回: (m, n), 类型 out_dtype.

    精度链 (反量化/累加固定 fp16, 不跟随 out_dtype):
      - pergroup: fp32 bmm -> 乘 scale 截 fp16 -> fp16 组间累加 -> perTokenScale 最后乘;
      - perchannel: fp32 matmul -> 乘 scale 截 fp16 -> 乘 perTokenScale.
    pergroup 各组 scale 正负有抵消, bf16 (千分之八) 累加会灾难性丢精度, 必须用 fp16 (千分之一).
    """
    N = weight.shape[1]
    if dequant_mode == 0:  # pergroup: per_channel_scale.shape = (K//k_num_per_group, N)
        groups = per_channel_scale.shape[0]
        ele_per_group = k_num_per_group
        # 1) reshape: x -> [groups, m, ele_per_group], weight -> [groups, ele_per_group, N]
        x_reshaped = x.reshape(m, groups, ele_per_group).transpose(0, 1).type(torch.float32)
        weight_reshaped = weight.reshape(groups, ele_per_group, N).type(torch.float32)
        # 2) batch matmul: [g, m, ele] @ [g, ele, N] -> [g, m, N]
        batch_matmul = torch.bmm(x_reshaped, weight_reshaped)
        # 3) 乘 perChannelScale (广播到 [g, m, N]) 后截 fp16
        scaled_results = (batch_matmul * per_channel_scale.reshape(groups, 1, N)).type(torch.float16)
        # 4) groups 维度 fp16 累加 -> [m, N]
        c = torch.zeros(m, N).type(torch.float16)
        for kg in range(scaled_results.shape[0]):
            c += scaled_results[kg].type(torch.float16)
        # 5) 最后乘 perTokenScale (升 fp32 乘, 对齐硬件两段式反量化)
        c = per_token_scale.reshape(m, 1) * c.to(torch.float32)
        return c.type(out_dtype)
    else:                  # perchannel: per_channel_scale.shape = (N,)
        # fp32 matmul -> 乘 scale 截 fp16 -> 升回 fp32 乘 perTokenScale
        c = torch.matmul(x.type(torch.float32), weight.type(torch.float32))
        c = torch.mul(c, per_channel_scale).type(torch.float16).to(torch.float32)
        c = torch.mul(c, per_token_scale.reshape(m, 1))
        return c.type(out_dtype)


def MM_fp64(x, weight, per_channel_scale, per_token_scale, m, out_dtype,
            k_num_per_group, dequant_mode):
    """高精度理论参考 (诊断用): fp64 全程不截断, 不模拟硬件反量化精度链.

    参数同 MM. 用于度量 NPU 输出偏离"无精度损失理论值"的程度, 不参与 PASS/FAIL 判定.

    与 ATK 原版 MM_fp64 的区别: ATK 版仍保留 ×scale 截断 (pergroup 截 fp32 / perchannel 截 fp16),
    仅升高累加精度; 本版全程 fp64 无截断, 作为纯数学真值. 与 MM 对比可反推硬件真实反量化精度档位.
    """
    N = weight.shape[1]
    if dequant_mode == 0:  # pergroup
        groups = per_channel_scale.shape[0]
        # 同 MM 的 reshape/bmm 结构, 但全程 fp64 不截断
        x_r = x.reshape(m, groups, k_num_per_group).transpose(0, 1).type(torch.float64)
        w_r = weight.reshape(groups, k_num_per_group, N).type(torch.float64)
        c = torch.bmm(x_r, w_r)                                          # fp64 batch matmul, 不截断
        c = (c * per_channel_scale.reshape(groups, 1, N).type(torch.float64)).sum(0)  # fp64 组间累加
        c = per_token_scale.reshape(m, 1).type(torch.float64) * c
        return c.type(out_dtype)
    else:                  # perchannel
        c = torch.matmul(x.type(torch.float64), weight.type(torch.float64))
        c = c * per_channel_scale.type(torch.float64)
        c = c * per_token_scale.reshape(m, 1).type(torch.float64)
        return c.type(out_dtype)


def gmm_a4w4_golden(x, weight, per_channel_scale, per_token_scale, group_list,
                    out_dtype, k_num_per_group, group_list_type, dequant_mode,
                    mm_func=MM):
    """分组遍历标杆. 返回 (M, N). mm_func 可传 MM (对齐版, 默认) 或 MM_fp64 (高精度参考)."""
    M, N = x.shape[0], weight.shape[2]
    out = torch.zeros(M, N).type(out_dtype)
    start = 0
    prev = 0
    for i, v in enumerate(group_list.tolist()):
        # type=2: v = [groupIdx, groupSize], weight 按 groupIdx(非行号 i) 取
        if group_list_type == 2:
            group_idx, temp = int(v[0]), int(v[1])
        elif group_list_type == 0:
            temp = v - prev
            prev = v
            group_idx = i
        else:  # type=1
            temp = v
            group_idx = i
        if temp > 0:
            out[start:start + temp] = mm_func(
                x[start:start + temp], weight[group_idx], per_channel_scale[group_idx],
                per_token_scale[start:start + temp], temp, out_dtype,
                k_num_per_group, dequant_mode)
        start += temp
    return out


def make_group_list(e, m, group_list_type, seed):
    """生成 groupList.
    type=0: 1D cumsum (非负单调非递减, 末值 <= M);
    type=1: 1D count  (每组大小, 和 <= M);
    type=2: 2D [E,2] = [[groupIdx, groupSize], ...], 非零组前置, groupSize 和 <= M。
    type 0/1 对齐 ATK cumsum/count 逻辑, type 2 对齐 aclnnGroupedMatmulV5 文档。
    """
    torch.manual_seed(seed)
    if group_list_type == 2:
        # 每组随机大小, 随机激活部分专家(其余 groupSize=0), 非零组前置, 总和 <= M
        sizes = torch.randint(0, max(2, m // e), (e,))
        active = torch.rand(e) > 0.3                       # ~70% 激活
        if not active.any():                               # 保证至少 1 个激活
            active[torch.randint(0, e, (1,))] = True
        sizes = torch.where(active, sizes.clamp_min(1), torch.zeros_like(sizes))
        if sizes.sum() > m:                                # 超过 M 则整体缩放
            sizes = (sizes.to(torch.float32) * (m / sizes.sum().item())).to(torch.int64)
        else:
            sizes = sizes.to(torch.int64)
        order = torch.argsort(~(sizes > 0), stable=True)   # 激活(sizes>0)前置, 组内保原序
        return torch.stack([order.to(torch.int64), sizes[order]], dim=1)
    # type=0 / 1
    inc = torch.randint(0, 128, (e,))
    cumsum = torch.cumsum(inc, dim=0)
    if cumsum[-1] >= m:
        scale = m / cumsum[-1].item()
        cumsum = (cumsum * scale).to(torch.int64)
        inc = (inc * scale).to(torch.int64)
    return cumsum.to(torch.int64) if group_list_type == 0 else inc.to(torch.int64)


def pack_scale_to_int64(scale_fp32):
    """(E, KG, N) fp32 → (E, KG, N) int64. fp32 比特位放低 32 位, 高 32 位 = 0."""
    e, kg, n = scale_fp32.shape
    arr = scale_fp32.cpu().numpy().astype(np.float32)
    arr.dtype = np.uint32                               # 比特重解释, 不改数值
    packed = np.zeros((e, kg, n * 2), dtype=np.uint32)
    packed[..., ::2] = arr
    packed.dtype = np.int64
    return torch.from_numpy(packed)

def pack_int4_weight_to_nz(weight_int8):
    """[E,K,N] signed INT4 values -> standard INT4 KN-NZ in an INT32 carrier.

    torch_npu 2.9 cannot format-cast a 3D INT32 INT4 carrier to NZ on DAV_3510.
    Pack adjacent N values into INT8 first, use the supported INT8 ND2NZ path,
    then reinterpret every four bytes as INT32 for the ACLNN INT4 ABI.
    """
    low = weight_int8[..., 0::2] & 0x0F
    high = (weight_int8[..., 1::2] & 0x0F) << 4
    packed_int8 = (low | high).to(torch.int8).contiguous().npu()
    return torch_npu.npu_format_cast(packed_int8, 29).view(torch.int32)

# ======================================================================================================================
# 构造输入 (整数 int4 数据流, 与 golden 完全一致)
# ======================================================================================================================
def build_inputs(m, k, n, e, weight_format, dequant_mode, group_list_type,
                 k_num_per_group, weight_transpose=False, seed=42):
    """返回 (npu_inputs, golden_inputs).

    x, weight 为 [-8, 7] 整数(int4 值域); scale 走 bf16 round-trip 保证 NPU 与 golden 一致。
    weight_transpose 仅影响 NPU 侧 weight 的内存排布(NZ 转置), golden 始终基于逻辑 [E,K,N] 计算。
    """
    torch.manual_seed(seed)
    npu = "npu"

    # 1) x (M, K) int4 值域整数
    x_int = torch.randint(-8, 7, (m, k), dtype=torch.int8)

    # 2) weight: 先判断是否转置, 再判断是否为 NZ, 以决定"转NZ"与"转置"的先后。
    #    非转置: 逻辑 weight 即 (E,K,N), 直接使用。
    #    转置:   逻辑 weight 仍为 (E,K,N)(golden 基于此计算), 源矩阵为 (E,N,K)。
    #    关键顺序: 必须**先转NZ、再转置** —— 即先对源 (E,N,K) 打包成 NZ 物理数据,
    #    再在 NZ 张量上做转置得到 view shape [E,K,N]。若反过来先 permute 成 (E,K,N)
    #    再打包, 得到的是非转置 NZ, 算子无法识别转置状态, 结果错位。
    if weight_transpose:
        w_source = torch.randint(-8, 7, (e, n, k), dtype=torch.int8)   # 源 (E, N, K)
        w_int = w_source.permute(0, 2, 1).contiguous()                 # 逻辑 (E, K, N)
    else:
        w_source = None
        w_int = torch.randint(-8, 7, (e, k, n), dtype=torch.int8)      # 逻辑 (E, K, N)

    # 3) scale: pergroup [E, K//kg, N] / perchannel [E, 1, N]
    kg = (k // k_num_per_group) if dequant_mode == 0 else 1
    scale_fp32 = (0.1 * torch.randn(e, kg, n)).to(torch.float32)
    # NPU 侧 scale 以 bf16 精度参与计算(to(bf16).to(fp32)); golden 用同一份保证对齐
    scale_ref = scale_fp32.to(torch.bfloat16).to(torch.float32)

    # 4) per-token scale (M,) fp32
    pt_scale = (0.1 * torch.randn(m)).abs().to(torch.float32)

    # 5) groupList (E,) int64
    group_list = make_group_list(e, m, group_list_type, seed)

    # ---- NPU 侧: 量化打包 ----
    x_quant = torch_npu.npu_quantize(
        x_int.to(torch.float32).to(npu),
        torch.tensor([1.], device=npu), None, torch.quint4x2, -1, False)

    if weight_format == "nz":
        # 转NZ 逻辑: 先打包 INT4, 再经 INT8 ND2NZ 生成标准 INT4 KN-NZ。
        # 普通 INT4 KN-NZ 物理块 [N/64,K/16,16,64]。
        if weight_transpose:
            # ---- 先转NZ、再转置 ----
            # 第一步 转NZ: 对源 (E,N,K) 打包成 NZ 物理数据(对应文档"输入为 [E,N,K]")。
            weight_quant = pack_int4_weight_to_nz(w_source)
            # 第二步 转置: 在 NZ 张量上做转置, 得到 view shape [E,K,N]
            # (对应文档"view shape 为 [E,K,N] 以保证算子识别转置状态")。
            weight_quant = weight_quant.transpose(1, 2)
        else:
            weight_quant = pack_int4_weight_to_nz(w_int)
    else:
        weight_quant = torch_npu.npu_quantize(
            w_int.to(torch.float32).to(npu),
            torch.tensor([1.], device=npu), None, torch.quint4x2, -1, False)

    scale_i64 = pack_scale_to_int64(scale_ref).to(npu)

    npu_inputs = {
        "x": x_quant,
        "weight": weight_quant,
        "scale": scale_i64,
        "per_token_scale": pt_scale.to(npu),
        "group_list": group_list.to(npu),
        "group_list_type": group_list_type,
    }
    golden_inputs = {
        "x": x_int,
        "weight": w_int,
        # golden 的 per_channel_scale 形状: pergroup [E,KG,N]; perchannel [E,N]
        "scale": scale_ref if dequant_mode == 0 else scale_ref.reshape(e, n),
        "per_token_scale": pt_scale,
        "group_list": group_list,
        "group_list_type": group_list_type,
        "k_num_per_group": k_num_per_group,
        "dequant_mode": dequant_mode,
    }
    return npu_inputs, golden_inputs


def call_gmm_a4w4(inputs, out_dtype):
    """调用 torch_npu.npu_grouped_matmul 完成一次 A4W4 GMM."""
    y_list = torch_npu.npu_grouped_matmul(
        [inputs["x"]],
        [inputs["weight"]],
        bias=None,
        scale=[inputs["scale"]],
        offset=None,
        antiquant_scale=None,
        antiquant_offset=None,
        per_token_scale=[inputs["per_token_scale"]],
        group_list=inputs["group_list"],
        split_item=3,
        group_type=0,
        group_list_type=inputs["group_list_type"],
        act_type=0,
        output_dtype=out_dtype,
    )
    return y_list[0]


# ======================================================================================================================
# 真值对比
# ======================================================================================================================

def verify_result(output, golden, dtype=torch.float32, tol = 2e-8):
    relative_tol = tol
    absolute_tol = tol
    error_tol = tol
    output = output.reshape(-1).to(dtype).cpu()
    golden = golden.reshape(-1).to(dtype).cpu()
    # 默认情况下，np.isclose(NaN, NaN) 返回 False，但设置 equal_nan=True 后，会将两个 NaN 判定为 “相等”（返回 True）
    # absolute(a - b) <= (atol + rtol * absolute(b))
    different_element_results = np.isclose(output, golden,
                                           rtol=relative_tol, atol=absolute_tol, equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print("data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
              (real_index, golden_data, output_data,
               abs(output_data - golden_data) / (abs(golden_data) + 1e-9)))
        if index == 10:
            break
    error_ratio = float(different_element_indexes.size) / (golden.size()[0])
    print("错误率: %.6f, 精度: %.6f" % (error_ratio, error_tol))
    print("错误个数: ", different_element_indexes.size, '/', golden.reshape(-1).size()[0])
    return error_ratio <= error_tol


def valid_rows(group_list, group_list_type):
    """有效 token 数: type=0 取 cumsum 末元素; type=1 取累加和; type=2 取第 2 列 groupSize 之和."""
    if group_list_type == 0:
        return int(group_list[-1].item())
    if group_list_type == 2:
        return int(group_list[:, 1].sum().item())
    return int(torch.cumsum(group_list, dim=0)[-1].item())


def compare(npu_out, golden_out, golden_fp64_out, e_valid, tag, out_dtype):
    """量化算子精度: 相对误差均值 + 误差占比. 阈值跟随 out_dtype.

    PASS/FAIL 按 NPU vs 对齐版 golden (golden_out) 判定;
    另打印 NPU vs 高精度理论值 (golden_fp64_out) 的偏差, 仅作诊断 (反推硬件反量化精度).
    bf16 机器精度 ~2^-7≈千分之 8, 1 ULP round 误差固有, 阈值用千分之十;
    fp16 机器精度 ~2^-10≈千分之一, 仍用 CANN 量化标准的双千分之二.
    """
    y = npu_out[:e_valid].cpu().to(torch.float32)
    g = golden_out[:e_valid].cpu().to(torch.float32)
    g_fp64 = golden_fp64_out[:e_valid].cpu().to(torch.float32)

    verify_result(y, g, tol=1e-3)

    # vs 对齐版 golden (判定)
    diff = (y - g).abs()
    rel = diff / g.abs().clamp_min(1e-9)
    err_thr = 1.0 / 1000.0 # if out_dtype == torch.bfloat16 else 2.0 / 1000.0
    err_ratio = (rel > err_thr).float().mean().item()
    mean_rel = rel.mean().item()
    max_abs = diff.max().item()

    # vs 高精度理论值 (诊断, 不参与判定): 偏离越大说明硬件反量化截断越多
    mean_rel_fp64 = ((y - g_fp64).abs() / g_fp64.abs().clamp_min(1e-9)).mean().item()

    passed = (err_ratio <= err_thr) and (mean_rel <= err_thr)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {tag}  mean_rel={mean_rel:.3e}  err_ratio={err_ratio:.3e}  "
          f"max_abs={max_abs:.3e}  (valid_rows={e_valid})")
    print(f"          vs_fp64(诊断): mean_rel={mean_rel_fp64:.3e}")
    return passed


def run_scenario(idx, cfg, args):
    weight_format, dequant_mode, group_list_type, weight_transpose = cfg
    mode_name = "pergroup" if dequant_mode == 0 else "perchannel"
    out_dtype = torch.float16 if args.out_dtype == "fp16" else torch.bfloat16

    m, k, n, e = args.m, args.k, args.n, args.e
    kg_size = args.k_num_per_group

    tag = (f"#{idx} fmt={weight_format} {mode_name} "
           f"groupListType={group_list_type} wt={int(weight_transpose)} out={args.out_dtype}")
    print(f"[run] {tag}  shape M={m} K={k} N={n} E={e}")

    npu_inputs, golden_inputs = build_inputs(
        m, k, n, e, weight_format, dequant_mode, group_list_type, kg_size,
        weight_transpose, args.seed)

    # NPU 执行
    y = call_gmm_a4w4(npu_inputs, out_dtype)
    torch_npu.npu.synchronize()

    # 真值计算 (CPU) —— 与 weight_transpose 无关: golden 始终基于逻辑 [E,K,N] 整数 weight
    golden_args = (golden_inputs["x"], golden_inputs["weight"], golden_inputs["scale"],
                   golden_inputs["per_token_scale"], golden_inputs["group_list"],
                   out_dtype, golden_inputs["k_num_per_group"],
                   golden_inputs["group_list_type"], golden_inputs["dequant_mode"])
    g = gmm_a4w4_golden(*golden_args, mm_func=MM)              # 对齐版 (判定用)
    g_fp64 = gmm_a4w4_golden(*golden_args, mm_func=MM_fp64)    # 高精度理论值 (诊断用)

    e_valid = valid_rows(golden_inputs["group_list"], group_list_type)

    return compare(y, g, g_fp64, e_valid, tag, out_dtype)


# 24 个功能场景: weightFormat × 量化模式 × groupListType × weightTranspose
# product(A,B,C,D) 遍历顺序：外层 A，中层 B，内层 C, 最内 D
# 注: 约束校验已注释(由算子内部判断); ND + weightTranspose 会以非转置送算子。
SCENARIOS = list(itertools.product(
    ["nd", "nz"],   # weight_format
    [1, 0],         # dequant_mode: 1 perchannel, 0 pergroup
    [0, 1, 2],      # group_list_type: 0 cumsum / 1 count / 2 [E,2]
    [False, True]   # weight_transpose (仅 nz 生效)
))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4984)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=176)
    parser.add_argument("--e", type=int, default=8)
    parser.add_argument("--k-num-per-group", type=int, default=64, help="pergroup 模式下 K 方向每组长度")
    parser.add_argument("--out-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=int, default=-1, help="只跑指定场景 (1..24), 默认全跑")
    args = parser.parse_args()

    torch_npu.npu.set_device(7)

    print(f"[info] 共 {len(SCENARIOS)} 个功能场景 "
          f"(weightFormat×量化模式×groupListType×weightTranspose); 输出 dtype={args.out_dtype}\n")

    ####################################################################### 可选：启动打印全部场景清单方便核对
    # print("场景编号对照表：")
    # for idx, cfg in enumerate(SCENARIOS, start=1):
    #     print(f"{idx:2d} | weight_format={cfg[0]}, dequant_mode={cfg[1]}, "
    #           f"group_list_type={cfg[2]}, weight_transpose={cfg[3]}")
    # print()
    #######################################################################

    results = []
    for idx, cfg in enumerate(SCENARIOS, start=1):
        if args.scenario != -1 and idx != args.scenario:
            continue
        if idx in [2,4,6,8,10,12]:  # 去除nd转置场景
            continue
        res = run_scenario(idx, cfg, args)
        results.append((idx, res))
        print()

    ok = [i for i, r in results if r is True]
    fail = [i for i, r in results if r is False]
    skip = [i for i, r in results if r is None]
    print("=" * 60)
    print(f"[summary] PASS={len(ok)}{ok}  FAIL={len(fail)}{fail}  SKIP={len(skip)}{skip}")


if __name__ == "__main__":
    main()


# 场景编号对照表：
#  1 | weight_format=nd, dequant_mode=1, group_list_type=0, weight_transpose=False
#  2 | weight_format=nd, dequant_mode=1, group_list_type=0, weight_transpose=True
#  3 | weight_format=nd, dequant_mode=1, group_list_type=1, weight_transpose=False
#  4 | weight_format=nd, dequant_mode=1, group_list_type=1, weight_transpose=True
#  5 | weight_format=nd, dequant_mode=1, group_list_type=2, weight_transpose=False
#  6 | weight_format=nd, dequant_mode=1, group_list_type=2, weight_transpose=True
#  7 | weight_format=nd, dequant_mode=0, group_list_type=0, weight_transpose=False
#  8 | weight_format=nd, dequant_mode=0, group_list_type=0, weight_transpose=True
#  9 | weight_format=nd, dequant_mode=0, group_list_type=1, weight_transpose=False
# 10 | weight_format=nd, dequant_mode=0, group_list_type=1, weight_transpose=True
# 11 | weight_format=nd, dequant_mode=0, group_list_type=2, weight_transpose=False
# 12 | weight_format=nd, dequant_mode=0, group_list_type=2, weight_transpose=True
# 13 | weight_format=nz, dequant_mode=1, group_list_type=0, weight_transpose=False
# 14 | weight_format=nz, dequant_mode=1, group_list_type=0, weight_transpose=True
# 15 | weight_format=nz, dequant_mode=1, group_list_type=1, weight_transpose=False
# 16 | weight_format=nz, dequant_mode=1, group_list_type=1, weight_transpose=True
# 17 | weight_format=nz, dequant_mode=1, group_list_type=2, weight_transpose=False
# 18 | weight_format=nz, dequant_mode=1, group_list_type=2, weight_transpose=True
# 19 | weight_format=nz, dequant_mode=0, group_list_type=0, weight_transpose=False
# 20 | weight_format=nz, dequant_mode=0, group_list_type=0, weight_transpose=True
# 21 | weight_format=nz, dequant_mode=0, group_list_type=1, weight_transpose=False
# 22 | weight_format=nz, dequant_mode=0, group_list_type=1, weight_transpose=True
# 23 | weight_format=nz, dequant_mode=0, group_list_type=2, weight_transpose=False
# 24 | weight_format=nz, dequant_mode=0, group_list_type=2, weight_transpose=True
