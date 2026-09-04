#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
单算子脚本: FlashMlaWithKvcache, TND + PA_NZ, B1 N1=96 S2=128000 D=576 DV=512 mask3。

用例如下(与 test_cases/flash_mla_with_kvcache/mla_longkv.py 的 PA_NZ 条一致):
  layout_q=TND, layout_kv=PA_NZ, layout_out=TND
  B=1, N1=96, N2=1, S1=8, S2=128000, D=576, DV=512
  block_size=128, mask_mode=3, cu_seqlens_q=[0,8], cache_seqlens=[128000]

================================================================================
【1. 环境准备】
  必须 source custom 算子包环境(flash_mla_with_kvcache 在 custom 包中):
    source /workspace/Custom/flash_mla/vendors/custom_transformer/bin/set_env.bash
  （不要 source CANN 9.2.0 的 set_env.sh, 否则 cann_ops_transformer 被 9.2.0 覆盖
    且该包不导出 flash_mla_with_kvcache; 深描 msprof op 也会因 profiler 库与
    driver 25.7.rc1.6 不兼容而采集不到数据）

  环境变量(可选):
    PEAK_HBM_BW=1.6e12 / PEAK_CUBE_FLOPS=378.47e12 覆盖 MFU/MBU 峰值参数

================================================================================
【2. 普通性能采集(msprof task-based, 出 op_summary / perf)】
  方式 A: 走框架(生成 sweep.py 并 msprof 采集):
    python run.py --backend flash_mla_with_kvcache --case-file mla_longkv \
        --case-filter "PA_NZ_B1_N1_96" --runs 3 --skip-precision

  方式 B: 直接 msprof 采集本脚本:
    msprof --output=/tmp/fmla_panz/msprof --aic-mode=task-based \
        python3 scripts/flash_mla_panz_single.py
    输出在 /tmp/fmla_panz/msprof/PROF_*/mindstudio_profiler_output/op_summary_*.csv

================================================================================
【3. 深度信息采集(msprof op / msopprof, 出 Memory/L2/Pipe/Roofline 指标)】
  kernel-name 用算子在 msprof 中注册的名字 FlashMlaWithKvcache(前缀匹配,
  实际 kernel 名为 FlashMlaWithKvcache_<hash>_<id>_mix_aic):

    rm -rf /tmp/fmla_panz/deep && mkdir -p /tmp/fmla_panz/deep
    msprof op --output=/tmp/fmla_panz/deep \
        --aic-metrics=Default,Occupancy,Roofline \
        --launch-count=1 --kernel-name=FlashMlaWithKvcache \
        python3 scripts/flash_mla_panz_single.py

  产物解析:
    OPPROF_*/OpBasicInfo.csv        算子耗时 / block dim / freq
    OPPROF_*/Memory.csv             HBM read/write、GM->L1/UB 数据量与带宽使用率
    OPPROF_*/L2Cache.csv            L2 read/write 命中率
    OPPROF_*/PipeUtilization.csv    Cube/MTE2/MTE1/Vector 各流水线活跃比
    OPPROF_*/ArithmeticUtilization.csv  各核 cube/vec ratio(真正的计算占比)
    OPPROF_*/ResourceConflictRatio.csv  cube_wait / mte2_wait 等等待比

  或走框架一键深描(等价):
    python run.py --backend flash_mla_with_kvcache --case-file mla_longkv \
        --case-filter "PA_NZ_B1_N1_96" --deep-profile --skip-precision --runs 1

================================================================================
【4. 流水时间线(PipeTimeline, trace.json / visualize_data.bin)】
  采集(独立目录, 数据大, ~8MB/核):
    rm -rf /tmp/fmla_panz/pipe && mkdir -p /tmp/fmla_panz/pipe
    msprof op --output=/tmp/fmla_panz/pipe \
        --aic-metrics=PipeTimeline \
        --launch-count=1 --kernel-name=FlashMlaWithKvcache \
        python3 scripts/flash_mla_panz_single.py

  产物:
    OPPROF_*/trace.json            chrome://tracing 格式, 拖到
                                   chrome://tracing 或 Perfetto UI
                                   (https://ui.perfetto.dev) 打开看流水图:
                                   横轴时间, 纵轴各核(core0..5 各含
                                   cubecore/veccore0/veccore1), 每条
                                   流水线(MTE1/MTE2/MTE3/CUBE/VECTOR/
                                   SCALAR/FIXP)的活跃与空隙一目了然
    OPPROF_*/visualize_data.bin    同数据二进制形式, 供可视化工具读
    OPPROF_*/dump/timeline.bin.0   原始时间线二进制

  流水图怎么看(以本 case 为例):
    - CUBE 事件时长集中在 0.617/0.685us(即 c1=QK^T 与 c2=P*V 的 matmul)
    - CUBE 事件间空隙 = cube 空转; 本 case 每核约 209us 空转(27%),
      中位空隙仅 0.092us 但有少量 0.5~2.8us 大空隙 → 依赖链等待
    - VECTOR 上跑 softmax(每核 ~872 次, 总 249us), 与 CUBE 重叠不充分
    - MTE2 每次搬运 ~1.17us(428 次/核), 带宽使用率仅 45.6%
    - 结论: 瓶颈是 QK^T→softmax→P*V 串行依赖链, 不是 HBM 带宽
      (HBM read 仅 74 GB/s = 1.6TB/s 峰值的 4.6%; 4T 带宽卡无提升佐证)

================================================================================
【5. 参考: 本 case 已实测指标】
  task_duration : ~795 us (runs=3 cold)
  MFU           : 0.70   (N1=96 头数下已优于 PA_BBND/PA_BNBD 的 0.65)
  MBU           : 0.12   (带宽非瓶颈, 低 MBU 属正常)
  L2 read hit   : 17.4%  (冷启动)
  bound_type    : Latency Bound
================================================================================
"""

_AIC_CORE_NUM = 36
_AIV_CORE_NUM = 72
_FA_META_SIZE = 16
_FD_META_SIZE = 16
_HEADER_SIZE = 16

_FA_FIELDS = [
    "BN2_START",
    "M_START",
    "S2_START",
    "BN2_END",
    "M_END",
    "S2_END",
    "FIRST_FD_WS_IDX",
]
_FD_FIELDS = ["BN2_IDX", "M_IDX", "WS_IDX", "WS_NUM", "M_START", "M_NUM"]

import torch
import torch_npu
import numpy as np

_RAND_CHUNK_ELEMS = 250 * 1024 * 1024
_DATA_RANGE = None


def _randn(*shape, dtype, device):
    """大张量分块填充, 避免一次 randn 的显存尖峰。"""
    if _DATA_RANGE is not None:
        _lo, _hi = _DATA_RANGE
        _x = torch.empty(*shape, dtype=dtype, device=device)
        _flat = _x.reshape(-1)
        if _flat.numel() <= _RAND_CHUNK_ELEMS:
            _flat.uniform_(_lo, _hi)
        else:
            for _start in range(0, _flat.numel(), _RAND_CHUNK_ELEMS):
                _flat[_start : _start + _RAND_CHUNK_ELEMS].uniform_(_lo, _hi)
        return _x
    if device.type != "npu":
        return torch.randn(*shape, dtype=dtype, device=device)
    _x = torch.empty(*shape, dtype=dtype, device=device)
    _flat = _x.reshape(-1)
    if _flat.numel() <= _RAND_CHUNK_ELEMS:
        _flat.normal_()
    else:
        for _start in range(0, _flat.numel(), _RAND_CHUNK_ELEMS):
            _flat[_start : _start + _RAND_CHUNK_ELEMS].normal_()
    return _x


try:
    from cann_ops_transformer.ops import (
        flash_mla_with_kvcache,
        flash_mla_with_kvcache_metadata,
    )
except ImportError as e:
    print(f"[ERROR] import failed: {e}")
    exit(1)


def print_metadata(metadata):
    arr = metadata.cpu().contiguous().to(torch.int32).numpy().astype(np.uint32)

    section_num = int(arr[0])
    is_fd = int(arr[1])
    m_base_size = int(arr[2])
    s2_base_size = int(arr[3])

    SEP = "=" * 78
    DASH = "-" * 78

    print(f"\n{SEP}")
    print("  [Metadata Header]")
    print(f"  sectionNum  = {section_num}")
    print(f"  isFD        = {is_fd}")
    print(f"  mBaseSize   = {m_base_size}")
    print(f"  s2BaseSize  = {s2_base_size}")

    fa_base = _HEADER_SIZE
    fd_base = _HEADER_SIZE + section_num * _AIC_CORE_NUM * _FA_META_SIZE

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(
            f"  [Section {sec}] FA Metadata — AIC cores  "
            f"(36 cores × 16 slots, 7 fields used)"
        )
        print(DASH)
        cw = 15
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in _FA_FIELDS)
        print(hdr)
        print(
            f"  {'':<7}"
            + "".join(f"{'[' + str(i) + ']':>{cw}}" for i in range(len(_FA_FIELDS)))
        )
        print(f"  {'─' * 7}" + "─" * (cw * len(_FA_FIELDS)))
        for core in range(_AIC_CORE_NUM):
            off = fa_base + sec * _AIC_CORE_NUM * _FA_META_SIZE + core * _FA_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(_FA_FIELDS))]
            note = "  (inactive)" if all(v == 0 for v in vals) else ""
            print(f"  AIC{core:02d}  " + "".join(f"{v:>{cw}}" for v in vals) + note)

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(
            f"  [Section {sec}] FD Metadata — active AIV cores only  "
            f"(M_NUM > 0 shown, 72 cores × 16 slots, 6 fields used)"
        )
        print(DASH)
        cw = 12
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in _FD_FIELDS)
        print(hdr)
        print(
            f"  {'':<7}"
            + "".join(f"{'[' + str(i) + ']':>{cw}}" for i in range(len(_FD_FIELDS)))
        )
        print(f"  {'─' * 7}" + "─" * (cw * len(_FD_FIELDS)))
        active = 0
        for core in range(_AIV_CORE_NUM):
            off = fd_base + sec * _AIV_CORE_NUM * _FD_META_SIZE + core * _FD_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(_FD_FIELDS))]
            if vals[5] == 0:
                continue
            print(f"  AIV{core:02d} " + "".join(f"{v:>{cw}}" for v in vals))
            active += 1
        if active == 0:
            print("  (no active FD cores)")

    print(f"{SEP}\n")


def main():
    dev = torch.device("npu:0")
    torch.npu.synchronize()
    torch.manual_seed(42)
    _dtype = torch.bfloat16

    # ---- 用例参数 ----
    n1 = 96
    n2 = 1
    d = 576  # head_dim_qk = nope512 + rope64
    dv = 512  # head_dim_v = nope512
    block_size = 128
    s2 = 128000
    layout_q, layout_kv, layout_out = "TND", "PA_NZ", "TND"
    mask_mode = 3
    cu_seqlens_q = [0, 8]
    1 * 96 * 8 * 1 * (576 + 512) / (16 * 16 * 16 * 32 * 1.65 * 1024)
    # ---- 数据生成 ----
    # Q: TND -> (总token, N1, D)
    _q = _randn(8, n1, d, dtype=_dtype, device=dev)

    # K cache: PA_NZ -> (total_blocks, N2, D/blk_elem, block_size, blk_elem)
    #   blk_elem=16 (bf16), DSUB = 576/16 = 36
    blk_elem = 16
    dsub = d // blk_elem
    total_blocks = (s2 + block_size - 1) // block_size  # 1000
    _k = _randn(total_blocks, n2, dsub, block_size, blk_elem, dtype=_dtype, device=dev)

    # block_table: (B, max_blocks_per_seq), 连续分配
    _block_table = (
        torch.arange(total_blocks, dtype=torch.int32, device="cpu").reshape(1, -1).npu()
    )
    # 本 case B=1 且长度刚好整除, 无需 -1 填充; 通用写法见框架 gen_pa_kv_tensor

    _v = None  # V 复用 K(nope 部分), 无独立 V 张量

    # mask: mask_mode=3 右对齐因果模板, 0=可见, 1=屏蔽
    _mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=dev), diagonal=1)

    _cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=dev)
    _seqused_q = None
    _cache_seqlens = torch.tensor([s2], dtype=torch.int32, device=dev)

    _metadata = flash_mla_with_kvcache_metadata(
        cache_seqlens=_cache_seqlens,
        cu_seqlens_q=_cu_q,
        seqused_q=_seqused_q,
        max_seqlen_q=8,
        max_seqlen_kv=s2,
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim_qk=d,
        head_dim_v=dv,
        mask_mode=mask_mode,
        layout_q=layout_q,
    )
    print_metadata(_metadata)
    # ---- 冷启动: 每 run 刷 L2 + 重建张量 ----
    torch.npu.synchronize()
    for _run_i in range(1):
        _flush_buf = torch.empty(
            256 * 1024 * 1024 // 4, dtype=torch.float32, device="npu"
        )
        _ = _flush_buf.sum()
        torch.npu.synchronize()
        del _flush_buf
        torch.npu.empty_cache()

        _op_kwargs = dict(
            k_cache=_k,
            block_table=_block_table,
            cache_seqlens=_cache_seqlens,
            attn_mask=_mask,
            metadata=_metadata,
            cu_seqlens_q=_cu_q,
            head_dim_v=dv,
            softmax_scale=1.0 / (d**0.5),
            mask_mode=mask_mode,
            max_seqlen_q=8,
            max_seqlen_kv=s2,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
            return_softmax_lse=False,
        )
        if _seqused_q is not None:
            _op_kwargs["seqused_q"] = _seqused_q  # 未声明就不传(造参零容忍)
        _op_ret = flash_mla_with_kvcache(_q, **_op_kwargs)
        torch.npu.synchronize()
        torch.npu.empty_cache()

    print("  -> DONE", flush=True)


if __name__ == "__main__":
    main()
