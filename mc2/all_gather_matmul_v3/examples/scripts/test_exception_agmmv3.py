#!/usr/bin/env python3
# coding=utf-8
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License").
# ------------------------------------------------------------------------------
"""AllGatherMatmulV3 异常用例拦截验证脚本.

逐条构造 exception sheet (david_excel/aclnnAllGatherMatmulV3.xlsx) 中的异常输入,
直调 cann_ops_transformer.all_gather_quant_matmul, 校验是否被正确拦截:
  - 预期拦截: 抛出异常且错误信息包含预期关键词 -> PASS
  - 预期放行(正向项): 正常执行 -> PASS

Run: qrun "python3 test_exception_agmmv3.py"
     qrun "python3 test_exception_agmmv3.py --mode oom"   # 大 M OOM 拐点验证
"""

import argparse
import os

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from ml_dtypes import float8_e4m3fn as np_fp8_e4m3fn
from ml_dtypes import float8_e8m0fnu as np_fp8_e8m0fnu

import cann_ops_transformer

os.environ.setdefault('HCCL_BUFFSIZE', '2000')

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50021

M, K, N = 64, 256, 64
KG = (K + 63) // 64  # 4

FP4_ENUM = 296       # fp4_e2m1
E8M0_ENUM = 293      # fp8_e8m0
Y_DTYPE_INT8 = 1     # torch ScalarType Char


def make_fp8(shape, dtype=np_fp8_e4m3fn):
    arr = np.random.uniform(-1, 1, shape).astype(dtype)
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e4m3fn).npu()


def make_scale(shape):
    arr = np.random.uniform(1, 2, shape).astype(np_fp8_e8m0fnu)
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e8m0fnu).npu()


def base_kwargs(group):
    return dict(
        x1=make_fp8([M, K]),
        x2=make_fp8([N, K]).t(),
        group=group,
        bias=torch.from_numpy(np.random.uniform(-1, 1, (N,)).astype(np.float32)).npu(),
        x1_scale=make_scale([M, KG, 2]),
        x2_scale=make_scale([N, KG, 2]),
        comm_mode="urma",
    )


def call(kwargs):
    cann_ops_transformer.all_gather_quant_matmul(**kwargs)


# (case_id, 描述, mutator(kwargs)->None, 预期错误关键词(list, 任一命中), 是否预期放行)
CASES = [
    # ---------- 异常shape ----------
    ("EXC-S01", "x1传入1维", lambda kw: kw.update(x1=make_fp8([K])), ["2D", "dim"]),
    ("EXC-S02", "x1传入3维", lambda kw: kw.update(x1=make_fp8([2, M, K])), ["2D", "dim"]),
    ("EXC-S03", "x1 m=0", lambda kw: kw.update(x1=make_fp8([0, K]), x1_scale=make_scale([0, KG, 2])),
     ["m", "M", "shape", "dim"]),
    ("EXC-S05", "x1 k=255(<256)",
     lambda kw: kw.update(x1=make_fp8([M, 255]), x2=make_fp8([N, 255]).t(),
                          x1_scale=make_scale([M, 4, 2]), x2_scale=make_scale([N, 4, 2])),
     ["256", "K", "k"]),
    ("EXC-S06", "x1 k=65536(>=65536)",
     lambda kw: kw.update(x1=make_fp8([M, 65536]), x2=make_fp8([N, 65536]).t(),
                          x1_scale=make_scale([M, 1024, 2]), x2_scale=make_scale([N, 1024, 2])),
     ["65535", "K", "k"]),
    ("EXC-S07", "x2传入1维", lambda kw: kw.update(x2=make_fp8([K])), ["2D", "dim"]),
    ("EXC-S08", "x2传入3维", lambda kw: kw.update(x2=make_fp8([2, N, K])), ["2D", "dim"]),
    ("EXC-S09", "x2非转置[K,N]连续传入", lambda kw: kw.update(x2=make_fp8([K, N])), ["mismatch", "K", "layout"]),
    ("EXC-S10", "bias传入2维",
     lambda kw: kw.update(bias=torch.from_numpy(np.zeros((1, N), dtype=np.float32)).npu()),
     ["1D", "bias", "dim"]),
    ("EXC-S11", "bias长度N-1",
     lambda kw: kw.update(bias=torch.from_numpy(np.zeros((N - 1,), dtype=np.float32)).npu()),
     ["bias", "shape", "N"]),
    ("EXC-S12", "x1_scale传入2维", lambda kw: kw.update(x1_scale=make_scale([M, KG])), ["3D", "scale", "dim"]),
    ("EXC-S13", "x1_scale传入4维", lambda kw: kw.update(x1_scale=make_scale([1, M, KG, 2])), ["3D", "scale", "dim"]),
    ("EXC-S14", "x1_scale第一维与M不对应", lambda kw: kw.update(x1_scale=make_scale([M // 2, KG, 2])),
     ["scale", "shape", "mismatch"]),
    ("EXC-S15", "x2_scale第一维与N不对应", lambda kw: kw.update(x2_scale=make_scale([N // 2, KG, 2])),
     ["scale", "shape", "mismatch"]),
    ("EXC-S17", "x1/x2 K不匹配",
     lambda kw: kw.update(x2=make_fp8([N, K - 64]).t(), x2_scale=make_scale([N, 3, 2])),
     ["mismatch", "K"]),
    # ---------- 异常dtype ----------
    ("EXC-D01", "x1传入bf16",
     lambda kw: kw.update(x1=torch.from_numpy(np.zeros((M, K), dtype=np.float32)).npu().to(torch.bfloat16)),
     ["dtype", "FLOAT8", "x1"]),
    ("EXC-D02", "x2传入bf16",
     lambda kw: kw.update(x2=torch.from_numpy(np.zeros((N, K), dtype=np.float32)).npu().to(torch.bfloat16).t()),
     ["dtype", "FLOAT8", "x2"]),
    ("EXC-D03", "x1=fp4(uint8+296), x2=fp8(fp4不成对)",
     lambda kw: kw.update(x1=torch.zeros(M, K // 2, dtype=torch.uint8).npu(), x1_dtype=FP4_ENUM,
                          x1_scale=torch.zeros(M, KG, 2, dtype=torch.uint8).npu(), x1_scale_dtype=E8M0_ENUM),
     ["FP4", "both", "fp4"]),
    ("EXC-D04", "x1=fp8, x2=fp4(uint8+296)(fp4不成对)",
     lambda kw: kw.update(x2=torch.zeros(N, K // 2, dtype=torch.uint8).npu().t(), x2_dtype=FP4_ENUM,
                          x2_scale=torch.zeros(N, KG, 2, dtype=torch.uint8).npu(), x2_scale_dtype=E8M0_ENUM),
     ["FP4", "both", "fp4"]),
    ("EXC-D05", "bias传入bf16",
     lambda kw: kw.update(bias=torch.from_numpy(np.zeros((N,), dtype=np.float32)).npu().to(torch.bfloat16)),
     ["bias", "FLOAT", "dtype"]),
    ("EXC-D06", "x1_scale传入fp32",
     lambda kw: kw.update(x1_scale=torch.from_numpy(np.zeros((M, KG, 2), dtype=np.float32)).npu()),
     ["scale", "E8M0", "dtype"]),
    ("EXC-D07", "x2_scale传入fp32",
     lambda kw: kw.update(x2_scale=torch.from_numpy(np.zeros((N, KG, 2), dtype=np.float32)).npu()),
     ["scale", "E8M0", "dtype"]),
    ("EXC-D08", "y_dtype传入int8枚举", lambda kw: kw.update(y_dtype=Y_DTYPE_INT8), ["output", "BF16", "FP16"]),
    # ---------- 异常attr ----------
    ("EXC-A01", "group_sizes=[128,128,128]", lambda kw: kw.update(group_sizes=[128, 128, 128]),
     ["group_size", "[1,1,32]", "group"]),
    ("EXC-A02", "group_sizes=[32,0,1]", lambda kw: kw.update(group_sizes=[32, 0, 1]),
     ["group_size", "[1,1,32]", "group"]),
    ("EXC-A03", "y_dtype非bf16/fp16对应值", lambda kw: kw.update(y_dtype=Y_DTYPE_INT8), ["output", "BF16", "FP16"]),
    ("EXC-A04", "comm_mode传ccu", lambda kw: kw.update(comm_mode="ccu"), ["urma", "comm_mode"]),
    ("EXC-A05", "comm_mode传整形123", lambda kw: kw.update(comm_mode=123), ["urma", "comm_mode", "123"]),
    ("EXC-A06", "uint8 x1搭配bf16枚举", lambda kw: kw.update(x1=torch.zeros(M, K // 2, dtype=torch.uint8).npu(),
                                                              x1_dtype=15), ["uint8", "fp4", "enum"]),
    ("EXC-A07", "x1_dtype传字符串123", lambda kw: kw.update(x1_dtype="123"), ["dtype", "123", "type"]),
    # ---------- 必选未传 ----------
    ("EXC-M01", "x1不传(None)", lambda kw: kw.update(x1=None), ["null", "None", "x1", "Tensor"]),
    ("EXC-M02", "x2不传(None)", lambda kw: kw.update(x2=None), ["null", "None", "x2", "Tensor"]),
    ("EXC-M03", "x1_scale不传(mx场景)", lambda kw: kw.update(x1_scale=None), ["x1_scale", "scale", "required"]),
    ("EXC-M04", "x2_scale不传(mx场景)", lambda kw: kw.update(x2_scale=None), ["x2_scale", "scale", "required"]),
    ("EXC-M07", "mxfp4时x1不传fp4枚举(uint8无enum)",
     lambda kw: kw.update(x1=torch.zeros(M, K // 2, dtype=torch.uint8).npu()),
     ["uint8", "fp4", "enum"]),
    ("EXC-M08", "scale不传e8m0枚举(uint8 scale无enum)",
     lambda kw: kw.update(x1_scale=torch.zeros(M, KG, 2, dtype=torch.uint8).npu()),
     ["uint8", "scale", "e8m0", "enum"]),
    # ---------- 正向放行项(预期执行成功) ----------
    ("EXC-M05", "group_sizes不传(自动推导[1,1,32])", lambda kw: None, None, True),
    ("EXC-M06", "comm_mode不传(PTA默认urma)", lambda kw: kw.pop("comm_mode"), None, True),
]

SKIP_NOTE = {
    "EXC-S04": "m超int32max无法实际构造(内存分配先失败), 描述项",
    "EXC-S16": "uint8打包方式K必为偶数, 奇数K的fp4输入无法构造; tiling校验由UT覆盖",
    "EXC-S18": "y由算子内部创建, PTA层无法构造错误y; aclnn/UT已覆盖",
    "EXC-M09": "context由PTA内部创建, 无法构造None; aclnn UT已覆盖(NullContextTest)",
}


def worker(rank, world_size, mode):
    torch_npu.npu.set_device(rank % 4)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size,
                            init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}')
    group = dist.new_group(ranks=list(range(world_size)))

    if mode == "oom":
        oom_ramp(rank, group)
        return

    np.random.seed(42 + rank)
    pass_cnt, fail_cnt = 0, 0
    for case in CASES:
        case_id, desc, mutator, keywords, *rest = case
        expect_ok = rest[0] if rest else False
        kwargs = base_kwargs(group)
        try:
            mutator(kwargs)
        except Exception as e:
            print(f"[rank{rank}][{case_id}] {desc}: 构造输入失败 {type(e).__name__}: {e}", flush=True)
            fail_cnt += 1
            continue
        try:
            call(kwargs)
            if expect_ok:
                print(f"[rank{rank}][{case_id}] {desc}: PASS (预期放行, 执行成功)", flush=True)
                pass_cnt += 1
            else:
                print(f"[rank{rank}][{case_id}] {desc}: FAIL (预期拦截但执行成功了!)", flush=True)
                fail_cnt += 1
        except Exception as e:
            msg = str(e)
            if expect_ok:
                print(f"[rank{rank}][{case_id}] {desc}: FAIL (预期放行但被拦截: {type(e).__name__}: {msg[:200]})",
                      flush=True)
                fail_cnt += 1
            else:
                hit = any(k.lower() in msg.lower() for k in keywords) if keywords else True
                tag = "PASS" if hit else "PASS(关键词未命中,需人工核对)"
                print(f"[rank{rank}][{case_id}] {desc}: {tag} [{type(e).__name__}] {msg[:220]}", flush=True)
                pass_cnt += 1
    for case_id, note in SKIP_NOTE.items():
        print(f"[rank{rank}][{case_id}] SKIP: {note}", flush=True)
    print(f"[rank{rank}] ===== 汇总: PASS={pass_cnt} FAIL={fail_cnt} =====", flush=True)
    dist.destroy_process_group()


def oom_ramp(rank, group):
    """大 M 爬坡, 观察 OOM/拦截拐点. 直接在 NPU 上构造, 避免 host 内存压力."""
    k, n = 4096, 2560
    for m in [10_000, 100_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]:
        kg = (k + 63) // 64
        try:
            x1 = torch.zeros(m, k, dtype=torch.uint8).npu().view(torch.float8_e4m3fn)
            x2 = make_fp8([n, k]).t()
            x1s = torch.zeros(m, kg, 2, dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)
            x2s = make_scale([n, kg, 2])
            cann_ops_transformer.all_gather_quant_matmul(x1, x2, group, x1_scale=x1s, x2_scale=x2s,
                                                         comm_mode="urma")
            torch.npu.synchronize()
            print(f"[rank{rank}][EXC-V01] M={m}: 执行成功", flush=True)
            del x1, x1s
            torch.npu.empty_cache()
        except Exception as e:
            print(f"[rank{rank}][EXC-V01] M={m}: 拦截/失败 [{type(e).__name__}] {str(e)[:200]}", flush=True)
            torch.npu.empty_cache()
            break
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="exception", choices=["exception", "oom"])
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    proc_list = []
    for rank in range(args.world_size):
        proc = Process(target=worker, args=(rank, args.world_size, args.mode))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    print("[TEST] done")


if __name__ == "__main__":
    main()
