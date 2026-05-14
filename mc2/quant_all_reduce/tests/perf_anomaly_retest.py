#!/usr/bin/env python3
"""Retest anomalous shapes."""
import sys, os, time, csv
sys.path.insert(0, '.')
import torch, torch_npu
from script import func
from op_class.aclnnQuantAllReduce import QuantAllReduce
from op_class.aclnnQuantReduceScatter import QuantReduceScatter
from torch.multiprocessing import Process, Lock, Value
import torch.multiprocessing as mp

WS = 4
SHAPES = [('aclnnQuantAllReduce', [100, 4096])]
DTYPE = 'mxfp'  # fp8_e4m3fn + fp8_e8m0, per-32-group
PEAK = 56.0

def build_case(op_name, shape, ws, dtype):
    bs, h = shape[0], shape[1]
    tag = 'mxfp' if dtype == 'mxfp' else 'int8'
    cn = f'anomaly_{op_name}_{bs}x{h}_{tag}'
    if dtype == 'mxfp':
        x_dtype = 'fp8_e4m3fn'
        s_dtype = 'fp8_e8m0'
        s_shape = [bs, h // 32, 2]
    else:
        x_dtype = 'int8'
        s_dtype = 'fp32'
        s_shape = [bs, h // 128]
    return {
        'case_name': cn, 'bin_name': cn, 'op_name': op_name,
        'input_tensor_shape': shape, 'input_tensor_range': (-1.0, 1.0),
        'input_tensor_dtype': x_dtype, 'scales_shape': s_shape,
        'scales_dtype': s_dtype, 'scales_range': (-1.0, 1.0),
        'outdtype': 'fp16', 'world_size': ws, 'reduce_op': 'sum',
        'seed': 42, 'graph_type': 0, 'scene': 'one_hccl', 'soc': '910D',
        'save_rank': 0, 'sheet_name': 'a',
        'master_ip': '127.0.0.1', 'master_port': '50001',
        'server_num': 1, 'server_index': 0, 'dev_num': 0, 'weight_same': 0,
        'ep_size': None, 'tp_size': None,
    }

def compute_metrics(op_name, shape, ws, t):
    total = shape[0] * shape[-1]
    v_bytes = total  # QAR
    v_mb = v_bytes / (1024 * 1024)
    bw = (v_bytes / (1024 ** 3)) / (t * 1e-6)
    util = bw / PEAK * 100
    return round(v_mb, 4), round(bw, 2), round(util, 1)

def read_csv(fp, case_name):
    if not os.path.exists(fp): return None
    with open(fp) as f:
        for row in csv.reader(f):
            if row and row[0] == case_name: return float(row[3])
    return None

def worker(rank, ws, tasks, counters, lock, results, dtype):
    torch_npu.npu.set_device(rank)
    print(f'[rank{rank}] dev={torch_npu.npu.current_device()}', flush=True)
    d, hc = func.get_hcomm_info(ws, rank, ep_size=0, tp_size=0,
                                master_ip='127.0.0.1', master_port='50001')
    for idx, (op_name, shape) in enumerate(tasks):
        case = build_case(op_name, shape, ws, dtype)
        cn = case['case_name']
        pdir = f'Profiling_Result_mc2_{cn}'
        csv_path = f'mc2_{op_name}_pro_result.csv'
        if rank == 0: os.system(f'rm -rf {pdir} {csv_path}')
        op = QuantAllReduce(case, rank, d, hc)
        op.get_input_weight(0)
        op.get_tmp_tensor()
        op.get_performace('npu', '910D')
        with lock: counters[idx].value += 1
        t0 = time.time()
        while True:
            with lock:
                if counters[idx].value == ws: break
            if time.time() - t0 > 300: print(f'[rank{rank}] TIMEOUT idx={idx}'); break
        if rank == 0:
            tk = read_csv(csv_path, cn)
            vm, bw, ut = compute_metrics(op_name, shape, ws, tk)
            tag = 'QAR'
            results.append((tag, str(shape), vm, tk, bw, ut))
            print(f'[{tag}] shape={shape} dtype={dtype} V={vm:.4f}MB T={tk:.2f}us '
                  f'BW={bw:.1f}GB/s Util={ut:.1f}%', flush=True)

if __name__ == '__main__':
    try: mp.set_start_method('forkserver', force=True)
    except: pass
    ctx = mp.get_context('forkserver')
    cnt = [ctx.Value('i', 0) for _ in SHAPES]
    lk = ctx.Lock(); mgr = ctx.Manager(); out = mgr.list()
    ps = []
    for r in range(WS):
        p = ctx.Process(target=worker, args=(r, WS, SHAPES, cnt, lk, out, DTYPE))
        p.start(); ps.append(p)
    for p in ps: p.join(timeout=300)
    for p in ps:
        if p.is_alive(): p.terminate(); p.join()
    print('\n=== RESULTS ===')
    for r in out: print(f'{r[0]} {r[1]} T={r[3]}us Util={r[5]}%')
