#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
验证脚本 v2：quant_grouped_mat_mul_allto_allv 算子 MXFP4/MXFP8（apace 模板路径）功能验证。

数据生成修正：gmm_weight 按 per-(K/32 组, N 列) 量化（scale shape [E, ceil(K/64), N, 2]），
与算子 tiling/kernel 期望一致；gmm_x 按 per-token per-(K/32 组) 量化（[A, ceil(K/64), 2]）。
golden 用纯 torch 实现（dequant -> GMM -> rank-major 排布 -> alltoallv 模拟）。

用法：
    python3 verify_quant_gmm_alltoallv_mx.py --dtype fp8_e4m3fn --A 64 --H1 256 --N1 128 --E 2
    python3 verify_quant_gmm_alltoallv_mx.py --dtype fp4_e2m1 --A 64 --H1 256 --N1 128 --E 2
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

sys.path.insert(0, '/home/t00947332/mc2_test')
from script import generate_mx_data

MX_DTYPE_MAP = {
    'fp8_e4m3fn': 'float8_e4m3fn',
    'fp8_e5m2': 'float8_e5m2',
    'fp4_e2m1': 'float4_e2m1',
}


def quant_gmm_x(x_fp32, dtype, k):
    """gmm_x [A, K] 沿 K 维每 32 元素量化 -> x_q [A, K], x_scale [A, 1, ceil(K/32)]。"""
    x_q, x_scale = generate_mx_data.mx_quantize(x_fp32, dtype, axis=-1, block_size=32,
                                                round_mode='rint')
    return x_q, x_scale


def quant_gmm_weight(w_fp32, dtype, k, n):
    """gmm_weight [E, K, N] 按 per-(e, K/32 组, N 列) 量化。
    scale 布局转成算子期望的 [E, ceil(K/64), N, 2]（展平 == [E, scaleK, N]）。"""
    e = w_fp32.shape[0]
    w2d = w_fp32.reshape(e * n, k)  # 每行一个 (e, n) 的 K 维
    w_q, w_s = generate_mx_data.mx_quantize(w2d, dtype, axis=-1, block_size=32,
                                            round_mode='rint')
    w_q = w_q.reshape(e, n, k).transpose(0, 2, 1)  # [E, K, N]
    w_s = w_s.reshape(e, n, -1)  # [E, N, ceil(K/32)]
    # scale -> [E, ceil(K/64), N, 2]
    half = w_s.shape[-1] // 2
    w_s = w_s.reshape(e, n, half, 2).transpose(0, 2, 1, 3)  # [E, half, N, 2]
    return w_q, w_s


def pack_fp4(t):
    """fp4x2 打包：偶元素低 4 位。输入 [..., K] -> 输出 [..., K/2]。"""
    t = np.ascontiguousarray(t)
    low = (t[:, 0::2].astype(np.uint8) & 0x0F)
    high = (t[:, 1::2].astype(np.uint8) & 0x0F)
    return (low | (high << 4)).astype(np.uint8)


def worker(rank, args, out_q):
    try:
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', rank=rank, world_size=args.world_size,
                                init_method='tcp://127.0.0.1:50015')
        hcom_info = _get_default_group()._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        e, r = args.E, args.world_size
        k, n = args.H1, args.N1
        nseg = r * e
        seg = args.A // nseg
        send_counts = [seg] * nseg
        recv_counts = [seg] * nseg
        A = args.A

        rng = np.random.default_rng(args.seed + rank)
        # gmm_x: [A, K] 每卡自己的 token（本卡所有专家 token 顺序拼接）
        x_fp32 = rng.uniform(-1, 1, (A, k)).astype(np.float32)
        # gmm_weight: [E, K, N]
        w_fp32 = rng.uniform(-1, 1, (e, k, n)).astype(np.float32)

        dtype = MX_DTYPE_MAP[args.dtype]
        x_q, x_s = quant_gmm_x(x_fp32, dtype, k)
        w_q, w_s = quant_gmm_weight(w_fp32, dtype, k, n)
        is_fp4 = args.dtype.startswith('fp4')
        if is_fp4:
            x_q = pack_fp4(np.ascontiguousarray(x_q.astype(np.uint8)))
            w_q = pack_fp4(np.ascontiguousarray(w_q.reshape(e, k, n).astype(np.uint8)))
            w_q = w_q.reshape(e, k, n // 2)
            gmm_x = torch.from_numpy(np.ascontiguousarray(x_q)).to('npu')
            gmm_weight = torch.from_numpy(np.ascontiguousarray(w_q)).to('npu')
            x_dtype = 296  # torch_npu float4_e2m1
        else:
            t_dtype = torch.float8_e4m3fn if args.dtype == 'fp8_e4m3fn' else torch.float8_e5m2
            gmm_x = torch.from_numpy(np.ascontiguousarray(x_q.astype(np.float32))).to(t_dtype).to('npu')
            gmm_weight = torch.from_numpy(np.ascontiguousarray(w_q.astype(np.float32))).to(t_dtype).to('npu')
            x_dtype = torch_npu.float8_e4m3fn if args.dtype == 'fp8_e4m3fn' else torch_npu.float8_e5m2
        x_s_u8 = np.ascontiguousarray(x_s.view(np.uint8))
        w_s_u8 = np.ascontiguousarray(w_s.view(np.uint8))
        gmm_x_scale = torch.from_numpy(x_s_u8).to('npu')
        gmm_weight_scale = torch.from_numpy(w_s_u8).to('npu')
        out_dtype = torch.float16 if args.out_dtype == 'fp16' else torch.bfloat16

        # 调用 MC2 算子（MX apace 路径）
        gmm_y, mm_y = torch_npu.npu_quant_gmm_alltoallv(
            gmm_x=gmm_x, gmm_weight=gmm_weight, gmm_x_scale=gmm_x_scale,
            gmm_weight_scale=gmm_weight_scale, hcom=hcom_info, ep_world_size=r,
            send_counts=send_counts, recv_counts=recv_counts, gmm_y_dtype=out_dtype,
            send_counts_tensor=None, recv_counts_tensor=None,
            mm_x=None, mm_weight=None, mm_x_scale=None, mm_weight_scale=None,
            comm_quant_scale=None,
            gmm_x_quant_mode=6, gmm_weight_quant_mode=6,
            mm_x_quant_mode=None, mm_weight_quant_mode=None,
            comm_quant_mode=0, group_size=None,
            gmm_x_dtype=x_dtype,
            gmm_weight_dtype=x_dtype,
            gmm_x_scale_dtype=293, gmm_weight_scale_dtype=293,
            mm_x_dtype=None, mm_weight_dtype=None, mm_x_scale_dtype=None, mm_weight_scale_dtype=None,
            comm_quant_dtype=None, mm_y_dtype=out_dtype, comm_mode='ai_cpu')
        torch_npu.npu.synchronize()

        # ---- golden（纯 torch + real alltoallv）----
        def e8m0_to_float(s):
            b = s.astype(np.int16) & 0xFF
            return np.exp2((b - 127).astype(np.float32))
        sx = torch.from_numpy(e8m0_to_float(x_s_u8)).float()  # [A, 1, K/32]
        sw = torch.from_numpy(e8m0_to_float(w_s_u8)).float()  # [E, K/64, N, 2]

        def dequant_x(xq, sx, k):
            xq = torch.from_numpy(np.ascontiguousarray(xq.astype(np.float32))).float()
            kg = k // 32
            s = sx.reshape(A, kg)
            xq = xq.reshape(A, kg, 32) * s.unsqueeze(-1)
            return xq.reshape(A, k)

        def dequant_w(wq, sw, k, n):
            wq = torch.from_numpy(np.ascontiguousarray(wq.astype(np.float32))).float()
            kg = k // 32
            s = sw.reshape(e, kg, n)
            wq = wq.reshape(e, kg, 32, n) * s.unsqueeze(2)
            return wq.reshape(e, k, n)

        x_f = dequant_x(x_q, sx, k).to('npu')
        w_f = dequant_w(w_q, sw, k, n).to('npu')

        # GMM 计算（rank-major 输出）：expert-major 的 A 段与 rank-major 的 C 段
        gmm_out = torch.zeros((A, n), dtype=torch.float32).to('npu')
        pos = 0
        for ee in range(e):
            for rr in range(r):
                cnt = send_counts[ee * r + rr]
                seg_x = x_f[pos:pos + cnt]
                seg_c = gmm_out[rr * (e * seg): rr * (e * seg) + cnt, :]
                seg_c.copy_(seg_x @ w_f[ee].T)
                pos += cnt
        # alltoallv：本卡发送 gmm_out 的 rank 段；golden = 接收各源卡段
        golden = torch.zeros((A, n), dtype=out_dtype).to('npu')
        split = [e * seg] * r
        dist.all_to_all_single(golden, gmm_out.to(out_dtype),
                               output_split_sizes=split, input_split_sizes=split)
        torch_npu.npu.synchronize()
        out_q.put((rank, 'done', golden.cpu().float(), gmm_y.float().cpu()))
        print('rank', rank, 'PASS(compute OK, gmm_y', list(gmm_y.shape), ')', flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        out_q.put((rank, 'error', str(e), None))


def compare(golden, npu, rank, atol=0.1, rtol=0.1):
    g = golden.reshape(-1)
    n = npu.reshape(-1)
    if g.numel() != n.numel():
        return False, -1, 'shape mismatch %s vs %s' % (list(golden.shape), list(npu.shape))
    diff = (g - n).abs()
    mask = torch.isfinite(g) & torch.isfinite(n)
    if not mask.any():
        return False, -1, 'all non-finite'
    abs_err = diff[mask].max().item()
    denom = g[mask].abs().clamp(min=1e-6)
    rel_err = (diff[mask] / denom).max().item()
    passed = abs_err <= atol or rel_err <= rtol
    return passed, abs_err, 'abs_err=%.4f rel_err=%.4f' % (abs_err, rel_err)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dtype', default='fp8_e4m3fn', choices=['fp8_e4m3fn', 'fp8_e5m2', 'fp4_e2m1'])
    p.add_argument('--world_size', type=int, default=4)
    p.add_argument('--A', type=int, default=64)
    p.add_argument('--H1', type=int, default=256)
    p.add_argument('--N1', type=int, default=128)
    p.add_argument('--E', type=int, default=2)
    p.add_argument('--out_dtype', default='fp16', choices=['fp16', 'bf16'])
    p.add_argument('--seed', type=int, default=50)
    args = p.parse_args()

    ctx = torch.multiprocessing.get_context('spawn')
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, args, q)) for r in range(args.world_size)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    results = [q.get() for _ in procs]
    results.sort(key=lambda x: x[0])
    all_pass = True
    print('=' * 70)
    for rank, status, golden, npu in results:
        if status != 'done':
            print('[FAIL] rank %d: %s' % (rank, golden))
            all_pass = False
            continue
        passed, err, msg = compare(golden, npu, rank)
        if not passed:
            all_pass = False
        print('[%s] rank %d  %s' % ('PASS' if passed else 'FAIL', rank, msg))
    print('RESULT:', 'ALL PASS' if all_pass else 'FAILED')
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
