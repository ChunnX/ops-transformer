# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
MatmulAlltoAllV2 golden 精度测试脚本 (torch extension 方式, 调用
``cann_ops_transformer.quant_matmul_all_to_all``).

算子语义 (先计算后通信, 与 ST 工程
``mc2/common/op_kernel/apace/tests/st/quant_matmul_all_to_all`` 的 gen_data.py 一致):
  1. 每 rank 持有量化后的左矩阵 A [M, K]、完整右矩阵 B (逻辑 [K, N], 以 [N, K]
     连续存储 + 转置视图传入)、e8m0 scale 及可选 fp32 bias;
  2. Matmul: C_i = dequant(A_i) x dequant(B_i) (+ bias) -> [M, N];
  3. AllToAll: rank j 从每个 rank i 收取 C_i 的第 j 个 N 切片 [M, Np]
     (Np = N/world_size), 沿 M 拼接 -> y [world_size*M, Np].

Golden 计算与输入构造沿用 ST 的 gen_data.py / verify_result.py 逻辑, 全程内存
进行 (无 bin 文件), 量化采用 ml_dtypes 就近舍入 (输入码值与 golden 反量化使用
同一份码值, 保证对比基准一致), 精度标准沿用 ST: rtol=0.02, atol=0.2.

运行前置条件:
  - 已安装 cann_ops_transformer 包, 并部署 MatmulAlltoAllV2 算子
    (libopapi_transformer.so / libcust_opapi.so);
  - Ascend 910_95 (ascend950) 设备, 通过 ASCEND_RT_VISIBLE_DEVICES 指定可用
    卡, 例如: export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3;
  - HCCL 通信 buffer 默认按 m*n*2 字节估算 (不小于 200MB), 也可通过
    HCCL_BUFFSIZE 环境变量 (单位 MB) 显式指定.

用法示例:
  python3 test_golden_matmul_alltoall.py --m 2048 --k 3584 --n 4096 --rank-per-dev 4
  python3 test_golden_matmul_alltoall.py --quant-type 2 --is-fp16 1 --is-bias 1
    --quant-type: 0=E2M1E2M1(FP4), 1=E4M3E4M3, 2=E5M2E5M2, 3=E4M3E5M2, 4=E5M2E4M3
"""

import os
import argparse
import math
import traceback
import multiprocessing as mp
from torch.multiprocessing import Process, Manager

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ml_dtypes import float4_e2m1fn as np_fp4_e2m1fn
from ml_dtypes import float8_e4m3fn as np_fp8_e4m3fn
from ml_dtypes import float8_e5m2 as np_fp8_e5m2
from ml_dtypes import float8_e8m0fnu as np_fp8_e8m0fnu

import cann_ops_transformer

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50101

# at::ScalarType / acl dtype 枚举 (acl 侧需 +256, 见 cann_ops_transformer DType)
Y_DTYPE_BF16 = 15
Y_DTYPE_FP16 = 5
SCALE_DTYPE_FP8_E8M0 = 293
QUANT_DTYPE_FP4 = 296

# MX 量化常量
MX_QUANT_MODE = 6
MX_GROUP_SIZES = [1, 1, 32]
MX_GROUP_SIZE = 32  # 每个 e8m0 scale 覆盖连续 32 个 K 元素 (64 元素组共享 2 个 scale)

# MXFP 量化输入 + BF16/FP16 输出精度标准 (与 ST verify_result.py 一致)
ERROR_TOL_RTOL = 0.02  # 2% 相对误差
ERROR_TOL_ATOL = 0.2  # 绝对误差

# 固定种子, 保证可复现 (所有 rank 生成完全相同的输入数据)
FIXED_SEED = 42

torch_npu.npu.config.allow_internal_format = True

# quant_type -> (x1 量化类型, x2 量化类型)
QUANT_TYPES = {
    0: ("e2m1", "e2m1"),
    1: ("e4m3", "e4m3"),
    2: ("e5m2", "e5m2"),
    3: ("e4m3", "e5m2"),
    4: ("e5m2", "e4m3"),
}

NP_QUANT_DTYPE = {"e2m1": np_fp4_e2m1fn, "e4m3": np_fp8_e4m3fn, "e5m2": np_fp8_e5m2}
# fp8 直接按位 view 成对应 torch fp8 dtype; fp4 无 torch 侧 dtype, 打包成 uint8
# (每字节 2 个元素) 并通过 x1_dtype/x2_dtype=296 传入算子 (与 all_to_all_quant_matmul
# 的 MXFP4 用法一致, 不依赖 torch 的 fp4 dtype 支持)
TORCH_QUANT_DTYPE = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
DATA_RANGE = {"e2m1": (1.0, 6.0), "e4m3": (1.0, 8.0), "e5m2": (1.0, 8.0)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="matmul_allto_all_v2 golden 精度测试 (MX 量化: fp8/fp4 x fp8/fp4, "
        "e8m0 scale, bf16/fp16 输出)"
    )
    parser.add_argument(
        "--m", type=int, default=2048, help="左矩阵 A 的行数 M (每 rank, 不切分)"
    )
    parser.add_argument(
        "--k", type=int, default=3584, help="左矩阵 A 的列数 K (完整 K, 需被 64 整除)"
    )
    parser.add_argument(
        "--n", type=int, default=4096, help="右矩阵 B 的列数 N (切分为 N/world_size)"
    )
    parser.add_argument(
        "--quant-type",
        type=int,
        default=1,
        help="0=E2M1E2M1(FP4), 1=E4M3E4M3, 2=E5M2E5M2, 3=E4M3E5M2, 4=E5M2E4M3",
    )
    parser.add_argument(
        "--is-fp16", type=int, default=0, help="输出 float16 (1) 或 bfloat16 (0)"
    )
    parser.add_argument(
        "--is-bias", type=int, default=0, help="是否使能 bias 输入 [N] (1/0)"
    )
    parser.add_argument("--warmup", type=int, default=3, help="warmup 迭代次数")
    parser.add_argument(
        "--rank-per-dev",
        type=int,
        default=4,
        help="每服务器的 rank 数 (每 rank 一张卡)",
    )
    parser.add_argument("--server-num", type=int, default=1, help="服务器数量")
    return parser.parse_args()


def set_device(rank, rank_per_dev):
    real_device = rank % rank_per_dev
    torch_npu.npu.set_device(real_device)
    print(f"[INFO] rank: {rank}, device: {real_device}")


def init_hccl_comm(rank, world_size):
    print(f"[INFO] device_{rank} 创建HCCL通信链路")
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{MASTER_IP}:{MASTER_PORT}",
    )
    print(f"[INFO] device_{rank} HCCL通信链路创建成功")
    global_group = dist.new_group(ranks=list(range(world_size)))
    print(f"[INFO] device_{rank} 全局通信组创建成功")
    return global_group


# ---------------------------------------------------------------------------
# 输入构造与 golden 计算 (沿用 ST gen_data.py 逻辑, 全程内存进行, 无 bin 文件)
# ---------------------------------------------------------------------------


def mxfp_dequant(codes, scale):
    """对已量化的码值施加 e8m0 scale, 返回 fp32 数据.

    codes: [rows, cols] 量化码值 (fp8/fp4, 每 ML_Dtypes 元素 1 字节);
    scale: [rows, ceil(cols/64), 2] e8m0, 每 64 个元素一组、组内 2 个 scale,
    每个 scale 覆盖连续 32 个元素 (沿最后一轴).

    A 侧: rows=M, cols=K; B 侧 (存储布局): rows=N, cols=K, 二者用法相同.
    """
    cols = codes.shape[-1]
    bcast = np.repeat(scale.reshape(scale.shape[0], -1), MX_GROUP_SIZE, axis=-1)[
        ..., :cols
    ]
    return codes.astype(np.float32) * bcast.astype(np.float32)


def generate_inputs(m, k, n, world_size, quant_type, is_bias):
    """用固定种子生成所有 rank 的输入数据 (每个 rank 生成完全相同的数据).

    布局 (每 rank i):
      x1:      (M, K) 量化, ML_Dtypes 存储;
      x1_scale: (M, ceil(K/64), 2) e8m0;
      x2:      (N, K) 量化 (DN 布局存储, 调用时取转置视图);
      x2_scale: (N, ceil(K/64), 2) e8m0;
      bias:    (N,) fp32, 幅度自适应到典型 C 量级的 ~1/8 (与 ST gen_data.py 一致).

    Returns: 包含所有 rank 数据 (含 golden 所需 fp32 反量化结果) 的 dict.
    """
    a_kind, b_kind = QUANT_TYPES[quant_type]
    ck = math.ceil(k / 64)

    np.random.seed(FIXED_SEED)

    a_local_list, a_scale_list = [], []
    b_ori_list, b_scale_list = [], []
    for _ in range(world_size):
        # x1 per rank: (M, K)
        a_local = np.random.uniform(*DATA_RANGE[a_kind], size=(m, k)).astype(
            NP_QUANT_DTYPE[a_kind]
        )
        a_scale = np.random.uniform(1, 8, size=(m, ck, 2)).astype(np_fp8_e8m0fnu)
        # x2 per rank: (N, K) 存储布局 (逻辑 [K, N] 的转置), 每 rank 持有完整 B
        b_ori = np.random.uniform(*DATA_RANGE[b_kind], size=(n, k)).astype(
            NP_QUANT_DTYPE[b_kind]
        )
        b_scale = np.random.uniform(1, 8, size=(n, ck, 2)).astype(np_fp8_e8m0fnu)

        a_local_list.append(a_local)
        a_scale_list.append(a_scale)
        b_ori_list.append(b_ori)
        b_scale_list.append(b_scale)

    # golden 所需的 fp32 反量化结果 (A: [M, K]; B: 逻辑 [K, N])
    a_dequant_list = [mxfp_dequant(a, s) for a, s in zip(a_local_list, a_scale_list)]
    b_dequant_list = [mxfp_dequant(b, s).T for b, s in zip(b_ori_list, b_scale_list)]

    # bias 幅度按典型 C 量级自适应 (~1/8), 与 ST gen_data.py 一致
    bias_list = [None] * world_size
    if is_bias:
        a_mean = float(np.abs(a_dequant_list[0]).mean())
        b_mean = float(np.abs(b_dequant_list[0]).mean())
        bias_amp = 0.125 * a_mean * b_mean * k
        for rank_id in range(world_size):
            bias_list[rank_id] = np.random.uniform(
                -bias_amp, bias_amp, size=(n,)
            ).astype(np.float32)
        print(
            f"[INFO] bias range [-{bias_amp:.1f}, {bias_amp:.1f}] (auto-scaled to typical C magnitude)"
        )

    return {
        "m": m,
        "k": k,
        "n": n,
        "world_size": world_size,
        "a_kind": a_kind,
        "b_kind": b_kind,
        "a_local_list": a_local_list,
        "a_scale_list": a_scale_list,
        "b_ori_list": b_ori_list,
        "b_scale_list": b_scale_list,
        "a_dequant_list": a_dequant_list,
        "b_dequant_list": b_dequant_list,
        "bias_list": bias_list,
    }


def compute_golden(inputs, dst_rank, is_fp16):
    """计算指定目的 rank 的 golden 输出 (先计算后通信, 与 ST gen_data.py 一致).

    1. C_i = dequant(A_i) x dequant(B_i) (+ bias_i) -> [M, N];
    2. permute: C_i.view(M, world_size, Np).permute(1, 0, 2) -> [world_size, M, Np];
    3. AllToAll: rank j 沿 M 拼接每个 rank i 的第 j 个 N 切片 -> [world_size*M, Np].

    Returns: (world_size*M, Np) 的 bf16/fp16 torch 张量.
    """
    m = inputs["m"]
    n = inputs["n"]
    world_size = inputs["world_size"]
    np_ = n // world_size
    out_dtype = torch.float16 if is_fp16 else torch.bfloat16

    # Step 1: 每 rank 的 MXFP 反量化 -> fp32 matmul (+ bias) -> C_i [M, N]
    cpu_outputs_full = []
    for src in range(world_size):
        c_full = torch.matmul(
            torch.from_numpy(inputs["a_dequant_list"][src]),
            torch.from_numpy(inputs["b_dequant_list"][src]),
        )
        if inputs["bias_list"][src] is not None:
            c_full = c_full + torch.from_numpy(inputs["bias_list"][src])
        cpu_outputs_full.append(c_full)

    # Step 2 + 3: permute 后取第 dst_rank 个 N 切片, 沿 M 拼接
    allto_all_inputs = [
        c.view(m, world_size, np_).permute(1, 0, 2) for c in cpu_outputs_full
    ]
    golden = torch.cat(
        [allto_all_inputs[src][dst_rank] for src in range(world_size)], dim=0
    )

    return golden.to(out_dtype)


# ---------------------------------------------------------------------------
# numpy 量化数据 -> torch 量化张量转换
# ---------------------------------------------------------------------------


def pack_fp4x2(codes_f4):
    """ML_Dtypes float4_e2m1fn (每元素 1 字节) -> 打包 uint8 (每字节 2 个元素, 低 nibble 在前).

    与 ST gen_data.py 的 pack_fp4x2 及 torch float4_e2m1fn_x2 的存储约定一致.
    """
    nib = codes_f4.view(np.uint8)
    even = nib[..., 0::2] & 0xF
    odd = nib[..., 1::2] & 0xF
    return (even | (odd << 4)).astype(np.uint8)


def np_quant_to_torch(np_arr, kind):
    """numpy 量化数组 -> torch 张量.

    fp8: 按位 view 成对应 torch fp8 dtype;
    fp4: 打包成 uint8 (每字节 2 个元素, 低 nibble 在前), 不做 dtype view,
         通过 x1_dtype/x2_dtype=296 告知算子按 ACL_FLOAT4_E2M1 解释.
    """
    if kind == "e2m1":
        return torch.from_numpy(pack_fp4x2(np_arr))
    return torch.from_numpy(np_arr.view(np.uint8)).view(TORCH_QUANT_DTYPE[kind])


def np_e8m0_to_torch(np_arr):
    """numpy e8m0 数组 -> torch float8_e8m0fnu 张量."""
    return torch.from_numpy(np_arr.view(np.uint8)).view(torch.float8_e8m0fnu)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def golden_test_worker(queue, rank, args, world_size):
    try:
        set_device(rank, args.rank_per_dev)
        group = init_hccl_comm(rank, world_size)

        m, k, n = args.m, args.k, args.n
        np_ = n // world_size

        print(
            f"[INFO] rank {rank} world_size={world_size}, m={m}, k={k}, n={n}, Np={np_}"
        )
        print(
            f"[INFO] rank {rank} scenario: x1={QUANT_TYPES[args.quant_type][0]}, "
            f"x2={QUANT_TYPES[args.quant_type][1]}, scale=fp8_e8m0, "
            f"y={'fp16' if args.is_fp16 else 'bf16'} (MX quant), bias={args.is_bias}"
        )

        # 1. 生成输入 (固定种子, 所有 rank 数据一致)
        inputs = generate_inputs(m, k, n, world_size, args.quant_type, args.is_bias)

        # 本 rank 的输入数据
        a_local = inputs["a_local_list"][rank]
        a_scale = inputs["a_scale_list"][rank]
        b_ori = inputs["b_ori_list"][rank]
        b_scale = inputs["b_scale_list"][rank]
        bias = inputs["bias_list"][rank]

        # numpy 量化数据 -> torch 张量并搬至 NPU
        x1 = np_quant_to_torch(a_local, inputs["a_kind"]).npu()  # [M, K]
        x2 = (
            np_quant_to_torch(b_ori, inputs["b_kind"]).npu().t()
        )  # [K, N] 转置视图 (存储 [N, K])
        x1_scale = np_e8m0_to_torch(a_scale).npu()  # [M, ck, 2]
        x2_scale = np_e8m0_to_torch(b_scale).npu()  # [N, ck, 2]
        bias_t = torch.from_numpy(bias).npu() if bias is not None else None

        print(
            f"[INFO] rank {rank} x1.shape={tuple(x1.shape)} dtype={x1.dtype}, "
            f"x2.shape={tuple(x2.shape)} dtype={x2.dtype}"
        )
        print(
            f"[INFO] rank {rank} x1_scale.shape={tuple(x1_scale.shape)} dtype={x1_scale.dtype}, "
            f"x2_scale.shape={tuple(x2_scale.shape)} dtype={x2_scale.dtype}"
        )

        # 2. Warmup
        for i in range(args.warmup):
            print(
                f"[INFO] rank {rank} warmup iter {i}/{args.warmup}: "
                f"calling quant_matmul_all_to_all ...",
                flush=True,
            )
            y = cann_ops_transformer.quant_matmul_all_to_all(
                x1,
                x2,
                group,
                bias=bias_t,
                x1_scale=x1_scale,
                x2_scale=x2_scale,
                y_dtype=Y_DTYPE_FP16 if args.is_fp16 else Y_DTYPE_BF16,
                x1_quant_mode=MX_QUANT_MODE,
                x2_quant_mode=MX_QUANT_MODE,
                group_sizes=MX_GROUP_SIZES,
                x1_dtype=QUANT_DTYPE_FP4 if inputs["a_kind"] == "e2m1" else None,
                x2_dtype=QUANT_DTYPE_FP4 if inputs["b_kind"] == "e2m1" else None,
                x1_scale_dtype=SCALE_DTYPE_FP8_E8M0,
                x2_scale_dtype=SCALE_DTYPE_FP8_E8M0,
                comm_mode="urma",
            )
            torch.npu.synchronize()
        if args.warmup > 0:
            print(
                f"[INFO] rank {rank} Warmup done. y.shape={tuple(y.shape)} dtype={y.dtype}"
            )

        # 3. Golden 对比运行
        print(f"[INFO] rank {rank} golden comparison run ...", flush=True)
        y = cann_ops_transformer.quant_matmul_all_to_all(
            x1,
            x2,
            group,
            bias=bias_t,
            x1_scale=x1_scale,
            x2_scale=x2_scale,
            y_dtype=Y_DTYPE_FP16 if args.is_fp16 else Y_DTYPE_BF16,
            x1_quant_mode=MX_QUANT_MODE,
            x2_quant_mode=MX_QUANT_MODE,
            group_sizes=MX_GROUP_SIZES,
            x1_dtype=QUANT_DTYPE_FP4 if inputs["a_kind"] == "e2m1" else None,
            x2_dtype=QUANT_DTYPE_FP4 if inputs["b_kind"] == "e2m1" else None,
            x1_scale_dtype=SCALE_DTYPE_FP8_E8M0,
            x2_scale_dtype=SCALE_DTYPE_FP8_E8M0,
            comm_mode="urma",
        )
        torch.npu.synchronize()

        expected_shape = (m * world_size, np_)
        if tuple(y.shape) != expected_shape:
            raise ValueError(
                f"rank {rank} output shape {tuple(y.shape)} != expected {expected_shape}"
            )

        # 4. CPU 侧计算 golden
        golden = compute_golden(inputs, rank, args.is_fp16)

        # 5. 精度对比 (标准与 ST verify_result.py 一致: rtol=0.02, atol=0.2)
        y_npu_cpu = y.cpu()

        abs_err = (y_npu_cpu.float() - golden.float()).abs()
        denom = golden.float().abs().clamp_min(1e-8)
        rel_err = abs_err / denom

        over_tol = (abs_err > ERROR_TOL_ATOL).sum().item()
        numel = m * world_size * np_
        passed = torch.allclose(
            golden, y_npu_cpu, rtol=ERROR_TOL_RTOL, atol=ERROR_TOL_ATOL, equal_nan=True
        )

        print(f"\n{'=' * 60}")
        print(f"[Rank {rank}] shape=({m * world_size}, {np_}), elements={numel}")
        print(
            f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}, "
            f"rmse={(abs_err.pow(2).mean().sqrt()).item():.6e}"
        )
        print(f"  rel_err: max={rel_err.max().item():.6e}")
        print(f"  count(|abs_err| > {ERROR_TOL_ATOL:g}): {over_tol} / {numel}")

        # 打印左上 4x4 角块便于快速人工检查
        cr = min(4, m * world_size)
        cc = min(4, np_)
        if cr > 0 and cc > 0:
            print(f"  golden (top-left {cr}x{cc}):\n{golden[:cr, :cc]}")
            print(f"  npu   (top-left {cr}x{cc}):\n{y_npu_cpu[:cr, :cc]}")

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] Rank {rank}: {'match' if passed else 'mismatch'} "
            f"(rtol={ERROR_TOL_RTOL}, atol={ERROR_TOL_ATOL})"
        )

        dist.barrier()
        torch.npu.synchronize()

        queue.put([rank, "PASS" if passed else "FAIL", None])

    except Exception as e:
        print(f"[ERROR] rank {rank} 执行失败: {str(e)}", flush=True)
        traceback.print_exc()
        try:
            dist.barrier(timeout=5)
        except Exception:
            pass
        queue.put([rank, f"error: {str(e)}", None])

    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        torch.npu.empty_cache()
        print(f"[INFO] rank {rank} 资源清理完成", flush=True)


if __name__ == "__main__":
    args = parse_args()
    world_size = args.rank_per_dev * args.server_num

    # 形状约束 (与 ST gen_data.py 及算子 tiling 校验一致)
    if args.m <= 0 or args.k <= 0 or args.n <= 0:
        raise SystemExit(
            f"Error: m/k/n must be positive, got {args.m}/{args.k}/{args.n}"
        )
    if args.k % 64 != 0:
        raise SystemExit(f"Error: k={args.k} not divisible by 64 (MX quantization)")
    if math.ceil(args.k / 64) % 2 != 0:
        raise SystemExit(
            f"Error: ceil(k/64)={math.ceil(args.k / 64)} not even (scale layout)"
        )
    if args.n % world_size != 0:
        raise SystemExit(f"Error: n={args.n} not divisible by world_size={world_size}")
    if world_size not in (2, 4, 8, 16):
        raise SystemExit(f"Error: world_size={world_size} must be one of 2/4/8/16")
    if args.quant_type not in QUANT_TYPES:
        raise SystemExit(f"Error: quant_type={args.quant_type} must be 0-4")

    # HCCL 通信 buffer: 每 rank 的窗口需容纳所有 rank 写来的 y 分片
    # (totalTileCnt * world_size * maxTileM * Np * 2B + 2MB <= m*n*2B + 2MB)
    est_mb = math.ceil((args.m * args.n * 2 + 2 * 1024 * 1024) / (1024 * 1024))
    os.environ.setdefault("HCCL_BUFFSIZE", str(max(est_mb, 200)))

    print(
        f"[INFO] 开始 matmul_allto_all_v2 golden 精度对比测试, world_size={world_size}"
    )
    print(
        f"[INFO] m={args.m}, k={args.k}, n={args.n}, Np={args.n // world_size} (per rank, K 不切分)"
    )
    print(
        f"[INFO] quant_type={args.quant_type} (x1={QUANT_TYPES[args.quant_type][0]}, "
        f"x2={QUANT_TYPES[args.quant_type][1]}), y_dtype={'fp16' if args.is_fp16 else 'bf16'}, "
        f"bias={args.is_bias}, warmup={args.warmup}"
    )
    print(
        f"[INFO] HCCL_BUFFSIZE={os.environ.get('HCCL_BUFFSIZE')} MB, "
        f"精度标准: rtol={ERROR_TOL_RTOL}, atol={ERROR_TOL_ATOL}"
    )

    proc_list = []
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("spawn", force=True)

    for rank in range(world_size):
        proc = Process(
            target=golden_test_worker, args=(result_queue, rank, args, world_size)
        )
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    # 默认值兜底: worker 异常退出未上报结果时也能正确判定 FAIL
    rank_outputs = [("error: no result reported", None)] * world_size
    for _ in range(world_size):
        rank_id, result, detail = result_queue.get()
        rank_outputs[rank_id] = (result, detail)
        print(f"[INFO] rank {rank_id} result: {result}")

    all_pass = all(r == "PASS" for r, _ in rank_outputs)
    if all_pass:
        print(f"\n{'=' * 60}")
        print(f"[PASS] 所有 {world_size} 个 rank 精度对比通过!")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print("[FAIL] 存在失败的 rank:")
        for i, (r, _) in enumerate(rank_outputs):
            if r != "PASS":
                print(f"  rank {i}: {r}")
        print(f"{'=' * 60}")
        raise SystemExit(1)
