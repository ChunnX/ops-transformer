import os
import argparse
import math
import multiprocessing as mp
from torch.multiprocessing import Process, Manager

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ml_dtypes import float4_e2m1fn as np_fp4_e2m1
from ml_dtypes import float8_e8m0fnu as np_fp8_e8m0

import cann_ops_transformer

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50010

os.environ["HCCL_BUFFSIZE"] = "6000"

# at::ScalarType enums
Y_DTYPE_BF16 = 15
Y_DTYPE_FP16 = 5
# torch_npu dtype 枚举（= ACL_FLOAT4_E2M1 + 256），客户侧传入的 fp4 枚举值
FP4_X_DTYPE = 296
SCALE_DTYPE_FP8_E8M0 = 293

MX_QUANT_MODE = 6
MX_GROUP_SIZES = [1, 1, 32]
MX_GROUP_SIZE = 32
FP4_ELE_EMAX = 2  # fp4_e2m1 元素指数上限

# 双千分之一精度标准：误差超过千分之一(1e-3)的元素占比不超过千分之一(1e-3)。
ERROR_TOL = 1e-3
FAIL_RATE_LIMIT = 1e-3
FIXED_SEED = 42

torch_npu.npu.config.allow_internal_format = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="allto_all_matmul fp4 (MXFP4) golden precision test"
    )
    parser.add_argument(
        "--bs", type=int, default=1024, help="batch size (BS, total rows of x1)"
    )
    parser.add_argument(
        "--h", type=int, default=4096, help="hidden size (H, per-rank K)"
    )
    parser.add_argument("--n", type=int, default=8192, help="output dim (N)")
    parser.add_argument("--warmup", type=int, default=3, help="warmup iterations")
    parser.add_argument(
        "--rank-per-dev", type=int, default=2, help="ranks per NPU device"
    )
    parser.add_argument("--server-num", type=int, default=1, help="server count")
    parser.add_argument(
        "--ydtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Output data type",
    )
    parser.add_argument("--bias", action="store_true", help="Enable bias addition test")
    parser.add_argument(
        "--precision-mode",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="precision mode",
    )
    return parser.parse_args()


def set_device(rank, rank_per_dev):
    real_device = rank % rank_per_dev
    torch_npu.npu.set_device(real_device)
    print(f"[INFO] rank: {rank}, device: {real_device}")


def init_hccl_comm(rank, world_size):
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{MASTER_IP}:{MASTER_PORT}",
    )
    global_group = dist.new_group(ranks=list(range(world_size)))
    return global_group


# ---------------------------------------------------------------------------
# MXFP4 quantization (mirrors the old test framework's mx_quantize for fp4_e2m1)
# ---------------------------------------------------------------------------


def mx_quantize_fp4(fp_array, block_size=32):
    """Quantize (M, K) fp32 to fp4_e2m1 with per-block e8m0 scale.

    Returns:
      packed   : (M, K/2) uint8, two fp4 per byte (low nibble = even idx)
      scale    : (M, K/32, 2) fp8_e8m0 (group_size=32, stored as 2 per 64)
      dequant  : (M, K) float32 (fp4 value * scale, for golden)
    """
    m, k = fp_array.shape
    fp = fp_array.reshape(m, k // block_size, block_size)

    fp_abs_max = np.max(np.abs(fp), axis=2, keepdims=True)
    fp32_min_normal = 2.0 ** (-127 + 1)
    share_exp = (
        np.floor(
            np.log2(fp_abs_max.astype(np.float32) + fp32_min_normal * (fp_abs_max == 0))
        )
        - FP4_ELE_EMAX
    )
    share_exp[fp_abs_max == 0] = -np.inf
    scale_block = (2.0**share_exp).astype(np.float32)  # (M, K/32, 1)

    # quantize elements to fp4_e2m1
    q = (fp / scale_block).astype(np_fp4_e2m1)  # (M, K/32, 32)
    deq = q.astype(np.float32) * scale_block
    deq = deq.reshape(m, k)

    # pack two fp4 into one uint8 (low nibble = even idx)
    q_flat = q.reshape(m, k)
    codes = q_flat.view(np.uint8)
    packed = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)  # (M, K/2)

    # scale: (M, K/32) -> (M, K/64, 2) e8m0
    scale_flat = scale_block.reshape(m, k // block_size)
    scale = scale_flat.reshape(m, k // (block_size * 2), 2).astype(np_fp8_e8m0)

    return packed, scale, deq


def generate_inputs(bs, h, n, world_size, use_bias=False):
    k = h * world_size
    ka = h
    if k % world_size != 0:
        raise ValueError(f"K={k} must be divisible by world_size={world_size}")

    np.random.seed(FIXED_SEED)

    # x2 (shared): (N, K) fp32 -> fp4 + scale, then transpose logic matches kernel (TransB)
    b_fp32 = np.random.uniform(1, 8, (n, k)).astype(np.float32)
    b_packed, b_scale, b_dequant = mx_quantize_fp4(b_fp32)
    b_dequant_transpose = np.swapaxes(b_dequant, -1, -2)  # (K, N)

    a_packed_list = []
    a_scale_list = []
    a_dequant_list = []
    for _ in range(world_size):
        a_fp32 = np.random.uniform(1, 8, (bs, ka)).astype(np.float32)
        a_packed, a_scale, a_dequant = mx_quantize_fp4(a_fp32)
        a_packed_list.append(a_packed)
        a_scale_list.append(a_scale)
        a_dequant_list.append(a_dequant)

    return {
        "bs": bs,
        "h": h,
        "n": n,
        "world_size": world_size,
        "k": k,
        "ka": ka,
        "b_packed": b_packed,
        "b_scale": b_scale,
        "b_dequant_transpose": b_dequant_transpose,
        "a_packed_list": a_packed_list,
        "a_scale_list": a_scale_list,
        "a_dequant_list": a_dequant_list,
        "bias": np.random.uniform(-100000, 100000, n).astype(np.float32)
        if use_bias
        else None,
    }


def compute_golden(inputs, dst_rank, precision_mode=0):
    bs = inputs["bs"]
    n = inputs["n"]
    world_size = inputs["world_size"]
    k = inputs["k"]
    ka = inputs["ka"]
    local_bs = bs // world_size
    a_dequant_list = inputs["a_dequant_list"]
    b_dequant = inputs["b_dequant_transpose"]  # (K, N)
    out_dtype = (
        torch.bfloat16 if inputs.get("ydtype", "bf16") == "bf16" else torch.float16
    )

    a_after_alltoall = np.zeros((bs, ka), dtype=np.float32)
    for src_rank in range(world_size):
        chunk = a_dequant_list[src_rank][
            dst_rank * local_bs : (dst_rank + 1) * local_bs, :
        ]
        a_after_alltoall[src_rank * local_bs : (src_rank + 1) * local_bs, :] = chunk

    a_permuted = np.zeros((local_bs, k), dtype=np.float32)
    for src_rank in range(world_size):
        a_permuted[:, src_rank * ka : (src_rank + 1) * ka] = a_after_alltoall[
            src_rank * local_bs : (src_rank + 1) * local_bs, :
        ]

    a_cpu = torch.from_numpy(a_permuted)
    b_cpu = torch.from_numpy(b_dequant)
    bias_t = (
        torch.from_numpy(inputs["bias"]) if inputs.get("bias") is not None else None
    )

    if precision_mode == 2:
        # GM accumulate 两次中间 bf16 舍入（模拟 kernel 行为）
        local_fp32 = torch.matmul(
            a_cpu[:, dst_rank * ka : (dst_rank + 1) * ka],
            b_cpu[dst_rank * ka : (dst_rank + 1) * ka, :],
        )
        if bias_t is not None:
            local_fp32 = local_fp32 + bias_t
        local_out = local_fp32.to(out_dtype)

        remote_fp32 = torch.zeros_like(local_fp32)
        for r in range(world_size):
            if r == dst_rank:
                continue
            remote_fp32 = remote_fp32 + torch.matmul(
                a_cpu[:, r * ka : (r + 1) * ka], b_cpu[r * ka : (r + 1) * ka, :]
            )
        remote_out = remote_fp32.to(out_dtype)
        out = (local_out.float() + remote_out.float()).to(out_dtype)
    else:
        out = torch.matmul(a_cpu, b_cpu)
        if bias_t is not None:
            out = out + bias_t
        out = out.to(out_dtype)

    return out


def np_fp8_to_torch(np_arr, torch_dtype):
    uint8_arr = np_arr.view(np.uint8)
    t = torch.from_numpy(uint8_arr)
    return t.view(torch_dtype)


def golden_test_worker(queue, rank, args, world_size):
    try:
        set_device(rank, args.rank_per_dev)
        group = init_hccl_comm(rank, world_size)

        bs, h, n = args.bs, args.h, args.n
        local_bs = bs // world_size

        inputs = generate_inputs(bs, h, n, world_size, args.bias)
        inputs["ydtype"] = args.ydtype
        y_dtype_val = Y_DTYPE_BF16 if args.ydtype == "bf16" else Y_DTYPE_FP16

        bias_npu = (
            torch.from_numpy(inputs["bias"]).npu()
            if inputs["bias"] is not None
            else None
        )

        a_packed = inputs["a_packed_list"][rank]  # (bs, ka/2) uint8
        a_scale = inputs["a_scale_list"][rank]  # (bs, ka/64, 2) e8m0
        b_packed = inputs["b_packed"]  # (n, k/2) uint8
        b_scale = inputs["b_scale"]  # (n, k/64, 2) e8m0

        # fp4 tensor as uint8 storage, dtype wrapper tells the kernel it is fp4
        x1 = torch.from_numpy(a_packed).npu()
        x2 = torch.from_numpy(b_packed).npu().t()  # (k/2, n), transpose like fp8 path
        x1_scale = np_fp8_to_torch(a_scale, torch.float8_e8m0fnu).npu()
        x2_scale = np_fp8_to_torch(b_scale, torch.float8_e8m0fnu).npu()

        print(
            f"[INFO] rank {rank} x1.shape={tuple(x1.shape)} x2.shape={tuple(x2.shape)} "
            f"x1_scale.shape={tuple(x1_scale.shape)} x2_scale.shape={tuple(x2_scale.shape)}"
        )

        for i in range(args.warmup):
            cann_ops_transformer.all_to_all_quant_matmul(
                x1,
                x2,
                group,
                bias=bias_npu,
                x1_scale=x1_scale,
                x2_scale=x2_scale,
                y_dtype=y_dtype_val,
                x1_quant_mode=MX_QUANT_MODE,
                x2_quant_mode=MX_QUANT_MODE,
                group_sizes=MX_GROUP_SIZES,
                x1_scale_dtype=SCALE_DTYPE_FP8_E8M0,
                x2_scale_dtype=SCALE_DTYPE_FP8_E8M0,
                comm_mode="urma",
                x1_dtype=FP4_X_DTYPE,
                x2_dtype=FP4_X_DTYPE,
                precision_mode=args.precision_mode,
            )
            torch.npu.synchronize()

        y, all2all_out = cann_ops_transformer.all_to_all_quant_matmul(
            x1,
            x2,
            group,
            bias=bias_npu,
            x1_scale=x1_scale,
            x2_scale=x2_scale,
            y_dtype=y_dtype_val,
            x1_quant_mode=MX_QUANT_MODE,
            x2_quant_mode=MX_QUANT_MODE,
            group_sizes=MX_GROUP_SIZES,
            x1_scale_dtype=SCALE_DTYPE_FP8_E8M0,
            x2_scale_dtype=SCALE_DTYPE_FP8_E8M0,
            comm_mode="urma",
            x1_dtype=FP4_X_DTYPE,
            x2_dtype=FP4_X_DTYPE,
            precision_mode=args.precision_mode,
        )
        torch.npu.synchronize()

        golden = compute_golden(inputs, rank, args.precision_mode)
        y_npu_cpu = y.cpu().to(golden.dtype)

        abs_err = (y_npu_cpu.float() - golden.float()).abs()
        rel_err = abs_err / golden.float().abs().clamp_min(1e-8)
        # 双千分之一标准：误差超过 atol+rtol*|golden| 的元素占比不超过 FAIL_RATE_LIMIT
        threshold = ERROR_TOL + ERROR_TOL * golden.float().abs()
        exceed = abs_err > threshold
        fail_rate = exceed.float().mean().item()
        passed = fail_rate <= FAIL_RATE_LIMIT

        print(
            f"[Rank {rank}] abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}; "
            f"rel_err: max={rel_err.max().item():.6e}; "
            f"exceed={exceed.sum().item()} (rate={fail_rate:.6e}, limit={FAIL_RATE_LIMIT:g}); "
            f"[{'PASS' if passed else 'FAIL'}]"
        )

        dist.barrier()
        torch.npu.synchronize()
        queue.put([rank, "PASS" if passed else "FAIL", None])

    except Exception as e:
        print(f"[ERROR] rank {rank} 执行失败: {str(e)}", flush=True)
        import traceback

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


if __name__ == "__main__":
    args = parse_args()
    world_size = args.rank_per_dev * args.server_num
    print(
        f"[INFO] 开始 allto_all_matmul fp4(MXFP4) golden 精度测试, world_size={world_size}"
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

    rank_outputs = [None] * world_size
    for _ in range(world_size):
        rank_id, result, detail = result_queue.get()
        rank_outputs[rank_id] = result

    all_pass = all(r == "PASS" for r in rank_outputs)
    print(f"\n{'=' * 60}")
    print(
        f"[{'PASS' if all_pass else 'FAIL'}] 所有 rank fp4 精度{'通过' if all_pass else '未通过'}: {rank_outputs}"
    )
    print(f"{'=' * 60}")
