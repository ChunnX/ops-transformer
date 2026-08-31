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

import cann_ops_transformer

FP8_NUMPY = {"e4m3fn": np_fp8_e4m3fn, "e5m2": np_fp8_e5m2}
FP8_TORCH = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}

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
MX_GROUP_SIZE = 32

# fixed seed for reproducibility (same as test_golden_alltoallmm.py)
FIXED_SEED = 42

torch_npu.npu.config.allow_internal_format = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="allto_all_matmul bit-matching golden test (bit-exact compare vs NPU)"
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
    parser.add_argument(
        "--matmul-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp64"],
        help="golden matmul accumulation dtype (fp32 mirrors kernel L0C, fp64 is true-math)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="for precision_mode=2, probe multiple GM-accumulate rounding models",
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
    global_group = dist.new_group(ranks=list(range(world_size)))
    print(f"[INFO] device_{rank} 全局通信组创建成功")
    return global_group


# ---------------------------------------------------------------------------
# Golden computation (exact MX dequant + fp32/fp64 matmul, mirror kernel math)
# ---------------------------------------------------------------------------


def generate_inputs(
    bs, h, n, world_size, x1dtype="e4m3fn", x2dtype="e4m3fn", use_bias=False
):
    np_dtype_x1 = FP8_NUMPY[x1dtype]
    np_dtype_x2 = FP8_NUMPY[x2dtype]
    """Generate all ranks' input data with a fixed seed.

    Layout:
      x1 (per rank):     (BS, H),          fp8_e4m3fn/e5m2
      x1_scale (per rank): (BS, ceil(H/64), 2), fp8_e8m0fnu
      x2 (shared):       (N, H*world_size), fp8_e4m3fn/e5m2
      x2_scale (shared): (N, ceil(H*world_size/64), 2), fp8_e8m0fnu
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
    b_scale_ori = np.random.uniform(1, 8, size=(n, math.ceil(k / 64), 2)).astype(
        np_fp8_e8m0fnu
    )

    # Dequantize x2: broadcast scale to (K, N)
    # NOTE: fp8->fp32 and fp32 * 2^E are both exact, so b_dequant is bit-exact.
    b_scale_reshape = b_scale_ori.reshape(n, -1)  # (N, ceil(K/32))
    b_scale_broadcast = np.repeat(b_scale_reshape, MX_GROUP_SIZE, axis=-1)[
        ..., :k
    ]  # (N, K)
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)  # (K, N)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(
        np.float32
    )

    a_local_list = []
    a_scale_list = []
    a_dequant_list = []

    for _ in range(world_size):
        a_local = np.random.uniform(1, 8, (bs, ka)).astype(np_dtype_x1)
        a_scale = np.random.uniform(1, 8, size=(bs, math.ceil(ka / 64), 2)).astype(
            np_fp8_e8m0fnu
        )

        # Dequantize x1: broadcast scale to (BS, H)
        a_scale_reshape = a_scale.reshape(bs, -1)  # (BS, ceil(H/32))
        a_scale_broadcast = np.repeat(a_scale_reshape, MX_GROUP_SIZE, axis=-1)[
            ..., :ka
        ]  # (BS, H)
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
        "bias": np.random.uniform(-100000, 100000, n).astype(np.float32)
        if use_bias
        else None,
    }


def _matmul(a, b, matmul_dtype):
    if matmul_dtype == "fp64":
        return torch.matmul(a.double(), b.double()).float()
    return torch.matmul(a, b)


def _truncate_to_out_dtype(x, out_dtype):
    """Round-toward-zero (truncate) an fp32 tensor to bf16/fp16 (positive values)."""
    if out_dtype == torch.bfloat16:
        xi = x.view(torch.int32)
        return (xi & torch.tensor(-65536, dtype=torch.int32)).view(torch.float32)
    # fp16: keep top 10 mantissa bits -> mask low 13 bits of fp32 mantissa
    xi = x.view(torch.int32)
    return (xi & torch.tensor(~8191, dtype=torch.int32)).view(torch.float32)


def compute_golden(inputs, dst_rank, matmul_dtype="fp32", precision_mode=0):
    """Compute golden output for a specific destination rank (final output dtype).

    Mirrors the alltoall + permute + matmul logic from the kernel. The MX dequant is
    bit-exact (fp8 -> fp32 exact, fp32 * 2^E exact). The matmul accumulation order
    differs slightly from the cube (a few 1-ULP output mismatches are expected).

    precision_mode=2 (GM accumulate) performs two-stage output rounding in the kernel:
      LOCAL  : round(local_matmul + bias)  -> output dtype (written to GM)
      REMOTE : round(local_gm + remote_matmul) -> output dtype (atomic add)
    This must be reproduced to bit-match mode 2.
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
    a_after_alltoall = np.zeros((bs, ka), dtype=np.float32)
    for src_rank in range(world_size):
        chunk_row_start = dst_rank * local_bs
        chunk_row_end = chunk_row_start + local_bs
        chunk = a_dequant_list[src_rank][chunk_row_start:chunk_row_end, :]
        dst_row_start = src_rank * local_bs
        dst_row_end = dst_row_start + local_bs
        a_after_alltoall[dst_row_start:dst_row_end, :] = chunk

    # Step 2: Permute — reshape (BS, ka) into (local_bs, K)
    a_permuted = np.zeros((local_bs, k), dtype=np.float32)
    for src_rank in range(world_size):
        src_row_start = src_rank * local_bs
        src_row_end = src_row_start + local_bs
        src_chunk = a_after_alltoall[src_row_start:src_row_end, :]
        dst_col_start = src_rank * ka
        dst_col_end = dst_col_start + ka
        a_permuted[:, dst_col_start:dst_col_end] = src_chunk

    a_cpu = torch.from_numpy(a_permuted)
    b_cpu = torch.from_numpy(b_dequant)
    bias_t = (
        torch.from_numpy(inputs["bias"]) if inputs.get("bias") is not None else None
    )

    if precision_mode == 2:
        # GM accumulate performs TWO output-dtype casts before the final sum:
        #   LOCAL  phase: round(local_matmul + bias) -> output dtype (written to GM)
        #   REMOTE phase: round(remote_matmul) -> output dtype (atomic-add operand)
        #   final        : round(local_gm + remote_gm) -> output dtype
        # Empirically the atomic add quantizes BOTH operands to the output dtype
        # (bit-match ~99.9986% with this model vs ~75% with a single-cast model).
        local_fp32 = _matmul(
            a_cpu[:, dst_rank * ka : (dst_rank + 1) * ka],
            b_cpu[dst_rank * ka : (dst_rank + 1) * ka, :],
            matmul_dtype,
        )
        if bias_t is not None:
            local_fp32 = local_fp32 + bias_t
        local_out = local_fp32.to(out_dtype)

        remote_fp32 = torch.zeros_like(local_fp32)
        for r in range(world_size):
            if r == dst_rank:
                continue
            remote_fp32 = remote_fp32 + _matmul(
                a_cpu[:, r * ka : (r + 1) * ka],
                b_cpu[r * ka : (r + 1) * ka, :],
                matmul_dtype,
            )
        remote_out = remote_fp32.to(out_dtype)
        out = (local_out.float() + remote_out.float()).to(out_dtype)
    else:
        out_fp32 = _matmul(a_cpu, b_cpu, matmul_dtype)
        if bias_t is not None:
            out_fp32 = out_fp32 + bias_t
        out = out_fp32.to(out_dtype)

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
# precision_mode=2 (GM accumulate) rounding probe
# ---------------------------------------------------------------------------


def _build_permuted_a(inputs, dst_rank):
    bs = inputs["bs"]
    world_size = inputs["world_size"]
    k = inputs["k"]
    ka = inputs["ka"]
    a_dequant_list = inputs["a_dequant_list"]
    local_bs = bs // world_size

    a_after_alltoall = np.zeros((bs, ka), dtype=np.float32)
    for src_rank in range(world_size):
        chunk_row_start = dst_rank * local_bs
        chunk_row_end = chunk_row_start + local_bs
        chunk = a_dequant_list[src_rank][chunk_row_start:chunk_row_end, :]
        dst_row_start = src_rank * local_bs
        dst_row_end = dst_row_start + local_bs
        a_after_alltoall[dst_row_start:dst_row_end, :] = chunk

    a_permuted = np.zeros((local_bs, k), dtype=np.float32)
    for src_rank in range(world_size):
        src_row_start = src_rank * local_bs
        src_row_end = src_row_start + local_bs
        src_chunk = a_after_alltoall[src_row_start:src_row_end, :]
        dst_col_start = src_rank * ka
        dst_col_end = dst_col_start + ka
        a_permuted[:, dst_col_start:dst_col_end] = src_chunk
    return torch.from_numpy(a_permuted)


def _probe_pm2_rounding(inputs, dst_rank, npu_cpu, ydtype, matmul_dtype):
    """Probe different GM-accumulate rounding models against the NPU output."""
    world_size = inputs["world_size"]
    ka = inputs["ka"]
    out_dtype = torch.bfloat16 if ydtype == "bf16" else torch.float16
    a = _build_permuted_a(inputs, dst_rank)
    b = torch.from_numpy(inputs["b_dequant"])
    bias_t = (
        torch.from_numpy(inputs["bias"]) if inputs.get("bias") is not None else None
    )

    def mm(x, y):
        return _matmul(x, y, matmul_dtype)

    local = mm(
        a[:, dst_rank * ka : (dst_rank + 1) * ka],
        b[dst_rank * ka : (dst_rank + 1) * ka, :],
    )
    if bias_t is not None:
        local = local + bias_t
    remote = torch.zeros_like(local)
    for r in range(world_size):
        if r == dst_rank:
            continue
        remote = remote + mm(a[:, r * ka : (r + 1) * ka], b[r * ka : (r + 1) * ka, :])

    variants = {}
    variants["single_rne"] = (local + remote).to(out_dtype)
    variants["local_cast"] = (local.to(out_dtype).float() + remote).to(out_dtype)
    variants["remote_cast"] = (local + remote.to(out_dtype).float()).to(out_dtype)
    variants["both_cast"] = (
        local.to(out_dtype).float() + remote.to(out_dtype).float()
    ).to(out_dtype)

    print(f"  --- pm=2 rounding model probe (rank {dst_rank}) ---")
    for name, g in variants.items():
        r = bit_compare(g, npu_cpu, ydtype)
        print(
            f"    {name:12s}: exact={r['match_rate'] * 100:.4f}%  max_ulp={r['max_ulp']:.2f}"
        )


# ---------------------------------------------------------------------------
# Bit-level comparison helpers
# ---------------------------------------------------------------------------


def _output_mantissa_bits(ydtype):
    # bf16: 7 explicit mantissa bits; fp16: 10 explicit mantissa bits
    return 7 if ydtype == "bf16" else 10


def compute_ulp_diff(a, b, ydtype):
    """ULP difference between two output-dtype values (given as fp32 tensors).

    ULP at magnitude m (normal value) = 2^(floor(log2(m)) - mantissa_bits).
    Returns a float32 tensor of (|a-b| / ulp), ~0 for bit-identical elements.
    """
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    mbits = _output_mantissa_bits(ydtype)
    max_abs = torch.maximum(af.abs(), bf.abs())
    bits = max_abs.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127  # floor(log2(m)) for normal fp32
    ulp = torch.pow(2.0, (exp - mbits).float())
    ulp = torch.clamp(ulp, min=torch.tensor(2.0**-126, dtype=torch.float32))
    return diff / ulp


def bit_compare(golden, npu_out, ydtype):
    """Bit-exact comparison of two output-dtype tensors.

    Returns a dict with:
      total       : element count
      exact       : number of bit-identical elements
      match_rate  : exact / total
      ulp_hist    : {1: n_1ulp, 2: n_2ulp, 3: n_3ulp, 'gt3': n_gt3} for mismatched elements
      max_ulp     : max ULP difference among mismatches
    """
    bits_bits = 16
    golden_bits = golden.view(torch.uint16)
    npu_bits = npu_out.view(torch.uint16)
    exact_mask = golden_bits == npu_bits

    total = int(golden_bits.numel())
    exact = int(exact_mask.sum().item())
    match_rate = exact / total if total else 0.0

    ulp_diff = compute_ulp_diff(golden, npu_out, ydtype)
    mismatch_ulp = ulp_diff[~exact_mask]

    ulp_hist = {1: 0, 2: 0, 3: 0, "gt3": 0}
    if mismatch_ulp.numel() > 0:
        rounded = mismatch_ulp.round().int()
        ulp_hist[1] = int((rounded == 1).sum().item())
        ulp_hist[2] = int((rounded == 2).sum().item())
        ulp_hist[3] = int((rounded == 3).sum().item())
        ulp_hist["gt3"] = int((rounded > 3).sum().item())
        max_ulp = float(mismatch_ulp.max().item())
    else:
        max_ulp = 0.0

    return {
        "total": total,
        "exact": exact,
        "mismatch": total - exact,
        "match_rate": match_rate,
        "ulp_hist": ulp_hist,
        "max_ulp": max_ulp,
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def bitmatch_worker(queue, rank, args, world_size):
    try:
        set_device(rank, args.rank_per_dev)
        group = init_hccl_comm(rank, world_size)

        bs = args.bs
        h = args.h
        n = args.n
        local_bs = bs // world_size

        # 1. Generate inputs (identical on every rank due to fixed seed)
        inputs = generate_inputs(
            bs, h, n, world_size, args.x1dtype, args.x2dtype, args.bias
        )
        inputs["ydtype"] = args.ydtype
        y_dtype_val = Y_DTYPE_BF16 if args.ydtype == "bf16" else Y_DTYPE_FP16

        bias_npu = None
        if args.bias and inputs["bias"] is not None:
            bias_npu = torch.from_numpy(inputs["bias"]).npu()

        a_local = inputs["a_local_list"][rank]
        a_scale = inputs["a_scale_list"][rank]
        b_ori = inputs["b_ori"]
        b_scale_ori = inputs["b_scale_ori"]

        torch_fp8_x1 = FP8_TORCH[args.x1dtype]
        torch_fp8_x2 = FP8_TORCH[args.x2dtype]

        x1 = np_fp8_to_torch(a_local, torch_fp8_x1).npu()
        x2 = np_fp8_to_torch(b_ori, torch_fp8_x2).npu()
        x2 = x2.t()
        x1_scale = np_fp8_to_torch(a_scale, torch.float8_e8m0fnu).npu()
        x2_scale = np_fp8_to_torch(b_scale_ori, torch.float8_e8m0fnu).npu()

        # 2. Warmup
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
                precision_mode=args.precision_mode,
            )
            torch.npu.synchronize()

        # 3. Golden comparison run
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

        # 4. Compute golden on CPU and bit-compare
        golden = compute_golden(inputs, rank, args.matmul_dtype, args.precision_mode)
        npu_cpu = y.cpu()

        result = bit_compare(golden, npu_cpu, args.ydtype)

        print(f"\n{'=' * 60}")
        print(
            f"[Rank {rank}] bit-match (matmul_dtype={args.matmul_dtype}, pm={args.precision_mode}, y={args.ydtype})"
        )
        print(f"  shape=({local_bs}, {n}), elements={result['total']}")
        print(
            f"  bit-exact: {result['exact']} / {result['total']} ({result['match_rate'] * 100:.6f}%)"
        )
        print(f"  mismatch : {result['mismatch']}")
        print(
            f"  ULP hist : 1ulp={result['ulp_hist'][1]}, 2ulp={result['ulp_hist'][2]}, "
            f"3ulp={result['ulp_hist'][3]}, >3ulp={result['ulp_hist']['gt3']}"
        )
        print(f"  max ULP  : {result['max_ulp']:.4f}")

        if args.precision_mode == 2 and args.probe:
            _probe_pm2_rounding(inputs, rank, npu_cpu, args.ydtype, args.matmul_dtype)

        dist.barrier()
        torch.npu.synchronize()

        queue.put([rank, result, None])

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

    print(
        f"[INFO] 开始 allto_all_matmul bit-matching golden 测试, world_size={world_size}"
    )

    proc_list = []
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("spawn", force=True)

    for rank in range(world_size):
        proc = Process(
            target=bitmatch_worker, args=(result_queue, rank, args, world_size)
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

    # Aggregate summary
    print(f"\n{'=' * 60}")
    all_results = [r for r, _ in rank_outputs if isinstance(r, dict)]
    if all_results:
        total_exact = sum(r["exact"] for r in all_results)
        total_elems = sum(r["total"] for r in all_results)
        total_mismatch = sum(r["mismatch"] for r in all_results)
        agg_rate = total_exact / total_elems if total_elems else 0.0
        print(
            f"[SUMMARY] bit-exact: {total_exact}/{total_elems} ({agg_rate * 100:.6f}%), "
            f"mismatch={total_mismatch}"
        )
        for i, (r, _) in enumerate(rank_outputs):
            if isinstance(r, dict):
                print(
                    f"  rank {i}: rate={r['match_rate'] * 100:.6f}% "
                    f"max_ulp={r['max_ulp']:.4f}"
                )
    print(f"{'=' * 60}")
