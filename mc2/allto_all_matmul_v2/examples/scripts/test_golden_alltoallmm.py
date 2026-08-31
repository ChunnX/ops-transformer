import os
import argparse
import math
import multiprocessing as mp
from torch.multiprocessing import Process, Manager

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ml_dtypes import float8_e4m3fn as np_fp8_e4m3fn
from ml_dtypes import float8_e5m2 as np_fp8_e5m2
from ml_dtypes import float8_e8m0fnu as np_fp8_e8m0fnu

FP8_NUMPY = {"e4m3fn": np_fp8_e4m3fn, "e5m2": np_fp8_e5m2}
FP8_TORCH = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
FP8_ELE_EMAX = {"e4m3fn": 8, "e5m2": 15}

import cann_ops_transformer

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50010

os.environ["HCCL_BUFFSIZE"] = "6000"

# at::ScalarType dtype enums
Y_DTYPE_BF16 = 15
Y_DTYPE_FP16 = 5
SCALE_DTYPE_FP8_E8M0 = 293

# MX quant constants
MX_QUANT_MODE = 6
MX_GROUP_SIZES = [1, 1, 32]
MXFP_DIVISOR = 64
MX_GROUP_SIZE = 32

# 双千分之一精度标准：误差超过千分之一(1e-3)的元素占比不超过千分之一(1e-3)。
# golden 对 precision_mode=2(GM 累加) 模拟 kernel 的两次中间 bf16 舍入，不再与理想结果比对。
ERROR_TOL = 1e-3
FAIL_RATE_LIMIT = 1e-3

# fixed seed for reproducibility (same as test_allto_all_matmul_mx.py)
FIXED_SEED = 42

torch_npu.npu.config.allow_internal_format = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="allto_all_matmul golden precision test (MX quant: fp8e4m3/e5m2, e8m0 scale, bf16/fp16 output)"
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
        "--dtype",
        type=str,
        default="e4m3fn",
        choices=["e4m3fn", "e5m2"],
        help="FP8 data type for x1 and x2",
    )
    parser.add_argument(
        "--x1dtype",
        type=str,
        default=None,
        choices=["e4m3fn", "e5m2"],
        help="FP8 data type for x1 (overrides --dtype)",
    )
    parser.add_argument(
        "--x2dtype",
        type=str,
        default=None,
        choices=["e4m3fn", "e5m2"],
        help="FP8 data type for x2 (overrides --dtype)",
    )
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
        help="precision mode (0=remote fusion, 1=L0c accumulate/DEFERRED_SYNC, 2=GM accumulate)",
    )
    args = parser.parse_args()
    if args.x1dtype is None:
        args.x1dtype = args.dtype
    if args.x2dtype is None:
        args.x2dtype = args.dtype
    return args


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
# Golden computation (mirrors gen_data.py logic, in-memory, no bin files)
# ---------------------------------------------------------------------------


def mx_quantize_fp8(fp_array, np_dtype, ele_emax, block_size=32):
    """Quantize (M, K) fp32 to fp8 with per-block e8m0 scale (real MX quantization).

    share_exp = floor(log2(block_abs_max)) - ele_emax, scale = 2^share_exp.

    Returns:
      q       : (M, K) fp8
      scale   : (M, K/64, 2) fp8_e8m0 (group_size=32, stored as 2 per 64)
      dequant : (M, K) float32 (fp8 value * scale, for golden)
    """
    m, k = fp_array.shape
    fp = fp_array.reshape(m, k // block_size, block_size)

    fp_abs_max = np.max(np.abs(fp), axis=2, keepdims=True)
    fp32_min_normal = 2.0 ** (-127 + 1)
    share_exp = (
        np.floor(
            np.log2(fp_abs_max.astype(np.float32) + fp32_min_normal * (fp_abs_max == 0))
        )
        - ele_emax
    )
    share_exp[fp_abs_max == 0] = -np.inf
    scale_block = (2.0**share_exp).astype(np.float32)  # (M, K/32, 1)

    # fp8 溢出时 ml_dtypes 会产生 NaN/Inf，需按 fp8 最大值截断（对齐 kernel 的 saturate 行为）
    max_norm = 1.75 * (2.0**ele_emax)
    q = (fp / scale_block).clip(-max_norm, max_norm).astype(np_dtype)  # (M, K/32, 32)
    deq = q.astype(np.float32) * scale_block
    deq = deq.reshape(m, k)

    q_flat = q.reshape(m, k)
    scale_flat = scale_block.reshape(m, k // block_size)
    scale = scale_flat.reshape(m, k // (block_size * 2), 2).astype(np_fp8_e8m0fnu)

    return q_flat, scale, deq


def generate_inputs(
    bs, h, n, world_size, x1dtype="e4m3fn", x2dtype="e4m3fn", use_bias=False
):
    np_dtype_x1 = FP8_NUMPY[x1dtype]
    np_dtype_x2 = FP8_NUMPY[x2dtype]
    ele_emax_x1 = FP8_ELE_EMAX[x1dtype]
    ele_emax_x2 = FP8_ELE_EMAX[x2dtype]
    """Generate all ranks' input data with a fixed seed.

    Layout:
      x1 (per rank):     (BS, H),          fp8_e4m3fn/e5m2
      x1_scale (per rank): (BS, ceil(H/64), 2), fp8_e8m0fnu
      x2 (shared):       (N, H*world_size), fp8_e4m3fn/e5m2
      x2_scale (shared): (N, ceil(H*world_size/64), 2), fp8_e8m0fnu

    Inputs are real MX quantized from fp32 (block=32 shared e8m0 scale).
    Returns a dict containing numpy arrays for all ranks.
    """
    k = h * world_size
    ka = h

    if k % world_size != 0:
        raise ValueError(f"K={k} must be divisible by world_size={world_size}")

    np.random.seed(FIXED_SEED)

    # x2 (shared): (N, K) fp32 -> MX quantize
    b_fp32 = np.random.uniform(1, 8, (n, k)).astype(np.float32)
    b_ori, b_scale_ori, b_dequant = mx_quantize_fp8(b_fp32, np_dtype_x2, ele_emax_x2)
    b_dequant_transpose = np.swapaxes(b_dequant, -1, -2)  # (K, N)

    # x1 per rank: (BS, H) fp32 -> MX quantize
    a_local_list = []
    a_scale_list = []
    a_dequant_list = []

    for _ in range(world_size):
        a_fp32 = np.random.uniform(1, 8, (bs, ka)).astype(np.float32)
        a_local, a_scale, a_dequant = mx_quantize_fp8(a_fp32, np_dtype_x1, ele_emax_x1)

        a_local_list.append(a_local)
        a_scale_list.append(a_scale)
        a_dequant_list.append(a_dequant)

    return {
        "bs": bs,
        "h": h,
        "n": n,
        "world_size": world_size,
        "k": k,
        "ka": ka,
        "b_ori": b_ori,
        "b_scale_ori": b_scale_ori,
        "a_local_list": a_local_list,
        "a_scale_list": a_scale_list,
        "a_dequant_list": a_dequant_list,
        "b_dequant": b_dequant_transpose,
        # bias magnitude ~1e5 is ~3% of the matmul output (~3.3e6), so an omitted/wrong bias
        # exceeds the tolerance and is actually detected.
        "bias": np.random.uniform(-100000, 100000, n).astype(np.float32)
        if use_bias
        else None,
    }


def compute_golden(inputs, dst_rank, precision_mode=0):
    """Compute golden output for a specific destination rank.

    Mirrors the alltoall + permute + matmul logic from gen_data.py.
    For precision_mode=2 (GM accumulate), it also simulates the kernel's
    two intermediate bf16 roundings (LOCAL -> bf16 -> GM, REMOTE -> bf16 ->
    atomic add), so the golden tracks the kernel's exact behavior.
    Returns a (local_bs, N) bfloat16 torch tensor.
    """
    bs = inputs["bs"]
    n = inputs["n"]
    world_size = inputs["world_size"]
    k = inputs["k"]
    ka = inputs["ka"]
    a_dequant_list = inputs["a_dequant_list"]
    b_dequant = inputs["b_dequant"]

    local_bs = bs // world_size
    out_dtype = (
        torch.bfloat16 if inputs.get("ydtype", "bf16") == "bf16" else torch.float16
    )

    # Step 1: AlltoAll along M dimension
    # For dst_rank, gather rows [dst_rank*local_bs : (dst_rank+1)*local_bs] from each src_rank,
    # place them at positions [src_rank*local_bs : (src_rank+1)*local_bs]
    a_after_alltoall = np.zeros((bs, ka), dtype=np.float32)
    for src_rank in range(world_size):
        chunk_row_start = dst_rank * local_bs
        chunk_row_end = chunk_row_start + local_bs
        chunk = a_dequant_list[src_rank][chunk_row_start:chunk_row_end, :]

        dst_row_start = src_rank * local_bs
        dst_row_end = dst_row_start + local_bs
        a_after_alltoall[dst_row_start:dst_row_end, :] = chunk

    # Step 2: Permute — reshape (BS, ka) into (local_bs, K)
    # Take blocks of local_bs rows and concatenate along K axis
    a_permuted = np.zeros((local_bs, k), dtype=np.float32)
    for src_rank in range(world_size):
        src_row_start = src_rank * local_bs
        src_row_end = src_row_start + local_bs
        src_chunk = a_after_alltoall[src_row_start:src_row_end, :]

        dst_col_start = src_rank * ka
        dst_col_end = dst_col_start + ka
        a_permuted[:, dst_col_start:dst_col_end] = src_chunk

    # Step 3: Matmul
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


# ---------------------------------------------------------------------------
# numpy fp8 -> torch fp8 conversion helpers
# ---------------------------------------------------------------------------


def np_fp8_to_torch(np_arr, torch_dtype):
    """Convert a numpy fp8 array to a torch tensor of the given fp8 dtype."""
    uint8_arr = np_arr.view(np.uint8)
    t = torch.from_numpy(uint8_arr)
    return t.view(torch_dtype)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def golden_test_worker(queue, rank, args, world_size):
    try:
        set_device(rank, args.rank_per_dev)
        group = init_hccl_comm(rank, world_size)

        bs = args.bs
        h = args.h
        n = args.n
        local_bs = bs // world_size

        print(
            f"[INFO] rank {rank} world_size={world_size}, bs={bs}, h={h}, n={n}, "
            f"local_bs={local_bs}"
        )
        print(
            f"[INFO] rank {rank} scenario: x1/x2=fp8e4m3fn, scale=fp8e8m0, y=bfloat16 (MX quant)"
        )

        # 1. Generate inputs (identical on every rank due to fixed seed)
        inputs = generate_inputs(
            bs, h, n, world_size, args.x1dtype, args.x2dtype, args.bias
        )
        inputs["ydtype"] = args.ydtype
        y_dtype_val = Y_DTYPE_BF16 if args.ydtype == "bf16" else Y_DTYPE_FP16

        bias_npu = None
        if args.bias and inputs["bias"] is not None:
            bias_npu = torch.from_numpy(inputs["bias"]).npu()

        # This rank's x1 and x1_scale
        a_local = inputs["a_local_list"][rank]
        a_scale = inputs["a_scale_list"][rank]
        b_ori = inputs["b_ori"]
        b_scale_ori = inputs["b_scale_ori"]

        torch_fp8_x1 = FP8_TORCH[args.x1dtype]
        torch_fp8_x2 = FP8_TORCH[args.x2dtype]

        # Convert numpy fp8 -> torch fp8 and move to NPU
        x1 = np_fp8_to_torch(a_local, torch_fp8_x1).npu()
        x2 = np_fp8_to_torch(b_ori, torch_fp8_x2).npu()
        x2 = x2.t()
        x1_scale = np_fp8_to_torch(a_scale, torch.float8_e8m0fnu).npu()
        x2_scale = np_fp8_to_torch(b_scale_ori, torch.float8_e8m0fnu).npu()

        print(
            f"[INFO] rank {rank} x1.shape={x1.shape} dtype={x1.dtype}, "
            f"x2.shape={x2.shape} dtype={x2.dtype}"
        )
        print(
            f"[INFO] rank {rank} x1_scale.shape={x1_scale.shape} dtype={x1_scale.dtype}, "
            f"x2_scale.shape={x2_scale.shape} dtype={x2_scale.dtype}"
        )

        # 2. Warmup
        for i in range(args.warmup):
            print(
                f"[INFO] rank {rank} warmup iter {i}/{args.warmup}: calling allto_all_matmul ...",
                flush=True,
            )
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
                precision_mode=args.precision_mode,
            )
            torch.npu.synchronize()
        print(
            f"[INFO] rank {rank} Warmup done. y.shape={y.shape} dtype={y.dtype}, "
            f"all2all_out.shape={all2all_out.shape} dtype={all2all_out.dtype}"
        )

        # 3. Golden comparison run
        print(f"[INFO] rank {rank} golden comparison run ...", flush=True)
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
            precision_mode=args.precision_mode,
        )
        torch.npu.synchronize()

        # 4. Compute golden on CPU
        golden = compute_golden(inputs, rank, args.precision_mode)

        # 5. Compare
        golden_dtype = torch.bfloat16 if args.ydtype == "bf16" else torch.float16
        y_npu_cpu = y.cpu().to(golden_dtype)

        abs_err = (y_npu_cpu.float() - golden.float()).abs()
        denom = golden.float().abs().clamp_min(1e-8)
        rel_err = abs_err / denom

        # 双千分之一标准：误差超过 atol + rtol*|golden| 的元素占比不超过 FAIL_RATE_LIMIT
        threshold = ERROR_TOL + ERROR_TOL * golden.float().abs()
        exceed = abs_err > threshold
        over_tol = exceed.sum().item()
        numel = local_bs * n
        fail_rate = exceed.float().mean().item()
        passed = fail_rate <= FAIL_RATE_LIMIT

        print(f"\n{'=' * 60}")
        print(f"[Rank {rank}] shape=({local_bs}, {n}), elements={numel}")
        print(
            f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}"
        )
        print(f"  rel_err: max={rel_err.max().item():.6e}")
        print(
            f"  count(exceed 1e-3): {over_tol} / {numel} (rate={fail_rate:.6e}, limit={FAIL_RATE_LIMIT:g})"
        )

        # Print top-left 8x8 corner for quick visual inspection
        cr = min(8, local_bs)
        cc = min(8, n)
        if cr > 0 and cc > 0:
            print(f"  golden (top-left {cr}x{cc}):\n{golden[:cr, :cc]}")
            print(f"  npu   (top-left {cr}x{cc}):\n{y_npu_cpu[:cr, :cc]}")

        # Per-row error summary (first 8 rows)
        pr = min(8, local_bs)
        if pr > 0:
            row_abs = abs_err[:pr].float()
            row_rel = row_abs / golden[:pr].float().abs().clamp_min(1e-8)
            print(
                f"  per-row (first {pr} rows): row_idx | max_abs_err | mean_abs_err | max_rel_err"
            )
            for r in range(pr):
                print(
                    f"    row {r:2d}: max_abs={row_abs[r].max().item():.6e}  "
                    f"mean_abs={row_abs[r].mean().item():.6e}  max_rel={row_rel[r].max().item():.6e}"
                )

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] Rank {rank}: {'match' if passed else 'mismatch'} "
            f"(exceed-rate limit={FAIL_RATE_LIMIT:g}, atol=rtol={ERROR_TOL:g})"
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
        except:
            pass
        queue.put([rank, f"error: {str(e)}", None])

    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except:
                pass
        torch.npu.empty_cache()
        print(f"[INFO] rank {rank} 资源清理完成", flush=True)


if __name__ == "__main__":
    args = parse_args()
    world_size = args.rank_per_dev * args.server_num

    print(f"[INFO] 开始 allto_all_matmul golden 精度对比测试, world_size={world_size}")

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
        rank_outputs[rank_id] = (result, detail)
        print(f"[INFO] rank {rank_id} result: {result}")

    all_pass = all(r == "PASS" for r, _ in rank_outputs)
    if all_pass:
        print(f"\n{'=' * 60}")
        print("[PASS] 所有 rank 精度对比通过!")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print("[FAIL] 存在失败的 rank:")
        for i, (r, d) in enumerate(rank_outputs):
            if r != "PASS":
                print(f"  rank {i}: {r}")
        print(f"{'=' * 60}")
