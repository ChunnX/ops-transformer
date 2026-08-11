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

import cann_ops_transformer

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50010

os.environ['HCCL_BUFFSIZE'] = '6000'

# at::ScalarType dtype enums
Y_DTYPE_BF16 = 15
Y_DTYPE_FP16 = 5
SCALE_DTYPE_FP8_E8M0 = 293

# MX quant constants
MX_QUANT_MODE = 6
MX_GROUP_SIZES = [1, 1, 32]
MXFP_DIVISOR = 64
MX_GROUP_SIZE = 32

# FP8(MXFP8) quantization input + BF16 output, theoretical relative error ~0.5-1%.
ERROR_TOL = 1e-2

# fixed seed for reproducibility (same as test_allto_all_matmul_mx.py)
FIXED_SEED = 42

torch_npu.npu.config.allow_internal_format = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="allto_all_matmul golden precision test (MX quant: fp8e4m3/e5m2, e8m0 scale, bf16/fp16 output)")
    parser.add_argument("--bs", type=int, default=1024, help="batch size (BS, total rows of x1)")
    parser.add_argument("--h", type=int, default=4096, help="hidden size (H, per-rank K)")
    parser.add_argument("--n", type=int, default=8192, help="output dim (N)")
    parser.add_argument("--warmup", type=int, default=3, help="warmup iterations")
    parser.add_argument("--rank-per-dev", type=int, default=2, help="ranks per NPU device")
    parser.add_argument("--server-num", type=int, default=1, help="server count")
    parser.add_argument("--dtype", type=str, default="e4m3fn", choices=["e4m3fn", "e5m2"],
                        help="FP8 data type for x1 and x2")
    parser.add_argument("--x1dtype", type=str, default=None, choices=["e4m3fn", "e5m2"],
                        help="FP8 data type for x1 (overrides --dtype)")
    parser.add_argument("--x2dtype", type=str, default=None, choices=["e4m3fn", "e5m2"],
                        help="FP8 data type for x2 (overrides --dtype)")
    parser.add_argument("--ydtype", type=str, default="bf16", choices=["bf16", "fp16"],
                        help="Output data type")
    parser.add_argument("--bias", action="store_true", help="Enable bias addition test")
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
        backend='hccl',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}'
    )
    print(f"[INFO] device_{rank} HCCL通信链路创建成功")
    global_group = dist.new_group(ranks=list(range(world_size)))
    print(f"[INFO] device_{rank} 全局通信组创建成功")
    return global_group


# ---------------------------------------------------------------------------
# Golden computation (mirrors gen_data.py logic, in-memory, no bin files)
# ---------------------------------------------------------------------------

def generate_inputs(bs, h, n, world_size, x1dtype="e4m3fn", x2dtype="e4m3fn", use_bias=False):
    np_dtype_x1 = FP8_NUMPY[x1dtype]
    np_dtype_x2 = FP8_NUMPY[x2dtype]
    """Generate all ranks' input data with a fixed seed.

    Layout:
      x1 (per rank):     (BS, H),          fp8_e4m3fn/e5m2
      x1_scale (per rank): (BS, ceil(H/64), 2), fp8_e8m0fnu
      x2 (shared):       (N, H*world_size), fp8_e4m3fn/e5m2
      x2_scale (shared): (N, ceil(H*world_size/64), 2), fp8_e8m0fnu

    Returns a dict containing numpy arrays for all ranks.
    """
    k = h * world_size
    ka = h

    if k % world_size != 0:
        raise ValueError(f"K={k} must be divisible by world_size={world_size}")

    np.random.seed(FIXED_SEED)

    # x2 (shared): shape (N, K), fp8
    b_ori = np.random.uniform(1, 8, (n, k)).astype(np_dtype_x2)
    b_ori_transpose = np.swapaxes(b_ori, -1, -2)  # (K, N)

    # x2_scale: shape (N, ceil(K/64), 2), fp8_e8m0fnu
    b_scale_ori = np.random.uniform(1, 8, size=(n, math.ceil(k / 64), 2)).astype(np_fp8_e8m0fnu)

    # Dequantize x2: broadcast scale to (K, N)
    b_scale_reshape = b_scale_ori.reshape(n, -1)  # (N, ceil(K/32))
    b_scale_broadcast = np.repeat(b_scale_reshape, MX_GROUP_SIZE, axis=-1)[..., :k]  # (N, K)
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)  # (K, N)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(np.float32)

    # x1 per rank: shape (BS, H), fp8_e4m3fn
    # x1_scale per rank: shape (BS, ceil(H/64), 2), fp8_e8m0fnu
    a_local_list = []
    a_scale_list = []
    a_dequant_list = []

    for _ in range(world_size):
        a_local = np.random.uniform(1, 8, (bs, ka)).astype(np_dtype_x1)
        a_scale = np.random.uniform(1, 8, size=(bs, math.ceil(ka / 64), 2)).astype(np_fp8_e8m0fnu)

        # Dequantize x1: broadcast scale to (BS, H)
        a_scale_reshape = a_scale.reshape(bs, -1)  # (BS, ceil(H/32))
        a_scale_broadcast = np.repeat(a_scale_reshape, MX_GROUP_SIZE, axis=-1)[..., :ka]  # (BS, H)
        a_dequant = a_local.astype(np.float32) * a_scale_broadcast.astype(np.float32)

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
        "b_dequant": b_dequant,
        "bias": np.random.uniform(-1, 1, n).astype(np.float32) if use_bias else None,
    }


def compute_golden(inputs, dst_rank):
    """Compute golden output for a specific destination rank.

    Mirrors the alltoall + permute + matmul logic from gen_data.py.
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
    out = torch.matmul(a_cpu, b_cpu)
    if "bias" in inputs and inputs["bias"] is not None:
        bias_t = torch.from_numpy(inputs["bias"])
        out = out + bias_t
    out_dtype = torch.bfloat16 if inputs.get("ydtype", "bf16") == "bf16" else torch.float16
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

        print(f"[INFO] rank {rank} world_size={world_size}, bs={bs}, h={h}, n={n}, "
              f"local_bs={local_bs}")
        print(f"[INFO] rank {rank} scenario: x1/x2=fp8e4m3fn, scale=fp8e8m0, y=bfloat16 (MX quant)")

        # 1. Generate inputs (identical on every rank due to fixed seed)
        inputs = generate_inputs(bs, h, n, world_size, args.x1dtype, args.x2dtype, args.bias)
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

        print(f"[INFO] rank {rank} x1.shape={x1.shape} dtype={x1.dtype}, "
              f"x2.shape={x2.shape} dtype={x2.dtype}")
        print(f"[INFO] rank {rank} x1_scale.shape={x1_scale.shape} dtype={x1_scale.dtype}, "
              f"x2_scale.shape={x2_scale.shape} dtype={x2_scale.dtype}")

        # 2. Warmup
        for i in range(args.warmup):
            print(f"[INFO] rank {rank} warmup iter {i}/{args.warmup}: calling allto_all_matmul ...", flush=True)
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
                alltoall_out_flag=True,
            )
            torch.npu.synchronize()
        print(f"[INFO] rank {rank} Warmup done. y.shape={y.shape} dtype={y.dtype}, "
              f"all2all_out.shape={all2all_out.shape} dtype={all2all_out.dtype}")

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
            alltoall_out_flag=True,
        )
        torch.npu.synchronize()

        # 4. Compute golden on CPU
        golden = compute_golden(inputs, rank)

        # 5. Compare
        golden_dtype = torch.bfloat16 if args.ydtype == "bf16" else torch.float16
        y_npu_cpu = y.cpu().to(golden_dtype)

        abs_err = (y_npu_cpu.float() - golden.float()).abs()
        denom = golden.float().abs().clamp_min(1e-8)
        rel_err = abs_err / denom

        over_tol = (abs_err > ERROR_TOL).sum().item()
        numel = local_bs * n
        passed = torch.allclose(
            golden, y_npu_cpu, rtol=ERROR_TOL, atol=ERROR_TOL, equal_nan=True
        )

        print(f"\n{'=' * 60}")
        print(f"[Rank {rank}] shape=({local_bs}, {n}), elements={numel}")
        print(f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}")
        print(f"  rel_err: max={rel_err.max().item():.6e}")
        print(f"  count(|abs_err| > {ERROR_TOL:g}): {over_tol} / {numel}")

        # Print top-left 4x4 corner for quick visual inspection
        cr = min(4, local_bs)
        cc = min(4, n)
        if cr > 0 and cc > 0:
            print(f"  golden (top-left {cr}x{cc}):\n{golden[:cr, :cc]}")
            print(f"  npu   (top-left {cr}x{cc}):\n{y_npu_cpu[:cr, :cc]}")

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Rank {rank}: {'match' if passed else 'mismatch'} (rtol=atol={ERROR_TOL})")

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
    mp.set_start_method('spawn', force=True)

    for rank in range(world_size):
        proc = Process(
            target=golden_test_worker,
            args=(result_queue, rank, args, world_size)
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
        print(f"\n{'='*60}")
        print("[PASS] 所有 rank 精度对比通过!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("[FAIL] 存在失败的 rank:")
        for i, (r, d) in enumerate(rank_outputs):
            if r != "PASS":
                print(f"  rank {i}: {r}")
        print(f"{'='*60}")