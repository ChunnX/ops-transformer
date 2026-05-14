#!/usr/bin/env python3
"""
QAR/QRS MTE bandwidth measurement — batch mode.

One HCCL comm domain, loop all shapes per process.  Shared-counter sync
prevents inter-shape overlap (same pattern as mc2_test --scene one_hccl).

Usage:
    python3 perf_bandwidth.py                          # both ops, 4 cards
    python3 perf_bandwidth.py --op qar                 # QAR only
    python3 perf_bandwidth.py --op qrs --world_size 8  # QRS only, 8 cards
    python3 perf_bandwidth.py --dry_run                # print shapes only
"""

import sys
import os
import csv
import time
import argparse
import traceback
import numpy as np
from collections import OrderedDict
from torch.multiprocessing import Process, Lock, Value
import torch.multiprocessing as mp

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script import func
from op_class.aclnnQuantAllReduce import QuantAllReduce
from op_class.aclnnQuantReduceScatter import QuantReduceScatter

PEAK_MTE_BW_GBs = 56.0

# ── payload-oriented shape definition ───────────────────────
# (V_target_MB, preferred_h)
# Derived shapes respect: bs >= world_size, bs % world_size == 0
PAYLOAD_SPECS = [
    (0.016,  4096),
    (0.048,  4096),
    (0.096,  4096),
    (0.192,  4096),
    (0.384,  4096),
    (0.672,  7168),
    (0.768,  7168),
    (1.500, 8192),
    (3.000, 8192),
    (4.500, 8192),
    (6.000, 8192),
    (9.000, 8192),
    (12.00, 8192),
]


def derive_shapes(world_size, op='both', dtype='int8'):
    """Return [(op_name, [bs, h])], bs aligned to world_size.
    Note: V calculation uses sizeof=1B for both int8 and MXFP (fp8).
    """
    tasks = []
    seen = set()

    bytes_per_elem = 1  # int8 and fp8 both 1B

    for v_mb, h in PAYLOAD_SPECS:
        v_elems = int(v_mb * 1024 * 1024 / bytes_per_elem)

        if op in ('qar', 'both'):
            bs = max(1, round(v_elems / h))
            bs = max(world_size, ((bs + world_size - 1) // world_size) * world_size)
            key = ('aclnnQuantAllReduce', bs, h)
            if key not in seen:
                seen.add(key)
                tasks.append(('aclnnQuantAllReduce', [bs, h]))

        if op in ('qrs', 'both'):
            input_elems = v_elems * world_size
            bs = max(1, round(input_elems / h))
            bs = max(world_size, ((bs + world_size - 1) // world_size) * world_size)
            key = ('aclnnQuantReduceScatter', bs, h)
            if key not in seen:
                seen.add(key)
                tasks.append(('aclnnQuantReduceScatter', [bs, h]))

    return tasks


# ── helpers ──────────────────────────────────────────────────

def build_case(op_name, shape, world_size, dtype='int8'):
    """Build a case dict matching what MC2_OP / op classes expect."""
    bs = int(np.prod(shape[:-1]))
    h = shape[-1]
    tag = 'mxfp' if dtype == 'mxfp' else 'int8'
    case_name = f"bw_{op_name}_{bs}x{h}_{tag}"

    is_mxfp = (dtype == 'mxfp')
    if is_mxfp:
        x_dtype = 'fp8_e4m3fn'
        scales_dtype = 'fp8_e8m0'
        scales_shape = [bs, h // 32, 2]
    else:
        x_dtype = 'int8'
        scales_dtype = 'fp32'
        scales_shape = [bs, h // 128]

    return {
        'case_name': case_name,
        'bin_name': case_name,
        'op_name': op_name,
        'input_tensor_shape': shape,
        'input_tensor_range': (-1.0, 1.0),
        'input_tensor_dtype': x_dtype,
        'scales_shape': scales_shape,
        'scales_dtype': scales_dtype,
        'scales_range': (-1.0, 1.0),
        'outdtype': 'fp16',
        'world_size': world_size,
        'reduce_op': 'sum',
        'seed': 42,
        'graph_type': 0,
        'scene': 'one_hccl',
        'soc': '910D',
        'save_rank': 0,
        'sheet_name': 'bw_perf',
        'master_ip': '127.0.0.1',
        'master_port': '50001',
        'server_num': 1,
        'server_index': 0,
        'dev_num': 0,
        'weight_same': 0,
        'ep_size': None,
        'tp_size': None,
    }


def compute_metrics(op_name, shape, world_size, t_kernel_us):
    """Return (V_MB, BW_GBs, Util_pct)."""
    total_elements = int(np.prod(shape))
    bytes_per_elem = 1  # int8

    if 'ReduceScatter' in op_name:
        v_bytes = (total_elements // world_size) * bytes_per_elem
    else:
        v_bytes = total_elements * bytes_per_elem

    v_mb = v_bytes / (1024 * 1024)
    bw = (v_bytes / (1024 ** 3)) / (t_kernel_us * 1e-6)
    util = bw / PEAK_MTE_BW_GBs * 100.0
    return round(v_mb, 4), round(bw, 2), round(util, 1)


def read_perf_from_csv(csv_path, case_name):
    """Read performance (us) from the CSV that get_performace wrote."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, 'r') as f:
        for row in csv.reader(f):
            if row and row[0] == case_name:
                return float(row[3])  # column 3 = performance(us)
    return None


# ── batch worker ─────────────────────────────────────────────

def batch_worker(rank, world_size, tasks, shared_counters, lock, results_out, dtype='int8'):
    """One rank: init hccl → loop all shapes → profile → sync."""
    try:
        torch_npu.npu.set_device(rank)
        print(f"[rank{rank}] device={torch_npu.npu.current_device()}", flush=True)

        dist_ctx, hcomm_info = func.get_hcomm_info(
            world_size, rank,
            ep_size=0, tp_size=0,
            master_ip='127.0.0.1',
            master_port='50001',
        )

        for idx, (op_name, shape) in enumerate(tasks):
            case = build_case(op_name, shape, world_size, dtype)
            case_name = case['case_name']
            profiling_dir = f"Profiling_Result_mc2_{case_name}"
            csv_path = f"mc2_{op_name}_pro_result.csv"

            # clean before each case so csv has exactly one row
            if rank == 0:
                os.system(f"rm -rf {profiling_dir}")
                os.system(f"rm -f {csv_path}")

            if op_name == 'aclnnQuantAllReduce':
                mc2_op = QuantAllReduce(case, rank, dist_ctx, hcomm_info)
            else:
                mc2_op = QuantReduceScatter(case, rank, dist_ctx, hcomm_info)

            mc2_op.get_input_weight(case['weight_same'])
            mc2_op.get_tmp_tensor()
            mc2_op.get_performace('npu', case['soc'])

            # sync: wait until all ranks finish this shape
            with lock:
                shared_counters[idx].value += 1

            start = time.time()
            while True:
                with lock:
                    if shared_counters[idx].value == world_size:
                        break
                if time.time() - start > 300:
                    print(f"[rank{rank}] TIMEOUT at shape {idx}: {shape}", flush=True)
                    break

            # rank 0 reads result
            if rank == 0:
                t_kernel = read_perf_from_csv(csv_path, case_name)
                if t_kernel is None or t_kernel <= 0:
                    t_kernel = 0.0
                tag = 'QAR' if 'AllReduce' in op_name else 'QRS'
                v_mb, bw, util = compute_metrics(op_name, shape, world_size, t_kernel)
                results_out.append((tag, str(shape), v_mb, t_kernel, bw, util))
                print(f"[{tag}] shape={shape}  V={v_mb:.4f}MB  T={t_kernel:.2f}us  "
                      f"BW={bw:.1f}GB/s  Util={util:.1f}%", flush=True)

                # cleanup
                os.system(f"rm -rf {profiling_dir}")
                os.system(f"rm -f {csv_path}")
                os.system(f"rm -rf bin_path/{case_name}")

    except Exception as e:
        print(f"[rank{rank}] FATAL: {e}\n{traceback.format_exc()}", flush=True)


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QAR/QRS bandwidth measurement (batch)")
    parser.add_argument('--op', choices=['qar', 'qrs', 'both'], default='both')
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--dtype', choices=['int8', 'mxfp'], default='int8',
                        help='int8=fp32 per-128-group, mxfp=fp8_e4m3fn+fp8_e8m0 per-32-group')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    world_size = args.world_size
    dtype = args.dtype
    tasks = derive_shapes(world_size, args.op, dtype)

    dtype_tag = 'fp8_e4m3fn+fp8_e8m0' if dtype == 'mxfp' else 'int8'
    print(f"world_size={world_size}  dtype={dtype_tag}  PEAK_MTE_BW={PEAK_MTE_BW_GBs} GB/s")
    print(f"Shapes: {len(tasks)}")
    print(f"QAR V = bs*h*1B    QRS V = bs*h/{world_size}*1B\n")

    print("Derived shapes:")
    for op_name, shape in tasks:
        tag = 'QAR' if 'AllReduce' in op_name else 'QRS'
        v_mb, _, _ = compute_metrics(op_name, shape, world_size, 1.0)
        print(f"  [{tag}] V≈{v_mb:.4f} MB  →  input {shape}")
    print()

    if args.dry_run:
        print("[dry_run] Done.")
        return

    # clean stale profiling / bin output
    os.system("rm -rf Profiling_Result_mc2_*")
    os.system("rm -f mc2_aclnnQuant*_pro_result.csv")
    os.system("rm -f bw_perf_results_w*.csv")

    # batch mode: one process group, loop shapes
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    ctx = mp.get_context('forkserver')
    shared_counters = [ctx.Value('i', 0) for _ in range(len(tasks))]
    lock = ctx.Lock()
    manager = ctx.Manager()
    results_out = manager.list()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=batch_worker,
                        args=(rank, world_size, tasks, shared_counters, lock, results_out, dtype))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)

    for p in procs:
        if p.is_alive():
            print(f"WARNING: process {p.pid} still alive, terminating", flush=True)
            p.terminate()
            p.join()

    # write CSV
    results = list(results_out)
    csv_path = f"bw_perf_results_w{world_size}_{dtype}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['operator', 'input_shape', 'V_MB', 'T_kernel_us', 'BW_GBs', 'Util_pct'])
        for r in results:
            writer.writerow(r)

    print(f"\n{'=' * 72}")
    print(f"  Results  ({len(results)}/{len(tasks)} succeeded)")
    print(f"{'=' * 72}")
    print(f"{'Op':<5} {'Shape':<16} {'V(MB)':<10} {'T(us)':<10} {'BW(GB/s)':<10} {'Util%':<8}")
    print("-" * 62)
    for r in results:
        print(f"{r[0]:<5} {r[1]:<16} {r[2]:<10} {r[3]:<10} {r[4]:<10} {r[5]:<8}")

    print(f"\nSaved: {csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()
