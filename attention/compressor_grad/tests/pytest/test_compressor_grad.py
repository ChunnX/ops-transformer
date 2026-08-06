#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import random

import os
import sys
import torch
import pytest

# PyPTO kernel 直跑（通路 2）：不经 aclnn/custom 包，运行时 JIT 编译
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "op_kernel"))
from compressor_grad import compressor_grad, CompressorGradTiling  # noqa: E402

from test_compressor_grad_cases import REDLINE_CASES
from compressor_grad_cpu_golden import compressor_grad_golden
from compressor_grad_npu import NPUBackend
import compressor_forward
from compressor_grad_check import (
    check_one_output,
    check_result,
    print_log,
    get_pct_thd,
    format_case_line,
)

DEVICE_ID = 0


def _ceil_div(a, b):
    return (a + b - 1) // b

# ── 输出精度对比开关 ──
ENABLE_DWKV_CHECK  = True
ENABLE_DWGATE_CHECK  = True
ENABLE_APE_CHECK  = True
ENABLE_DX_CHECK   = True

# ================================================================
#  Workspace reduction
# ================================================================

def reduce_ape_workspace(workspace_raw, headDim, coff, cmpRatio, roundCnt, totalBlocks):
    """
    Reduce d_ape workspace partials to final d_ape: shape (cmpRatio, coff*headDim).

    Each Vec core writes a tile of shape [coff*cmpRatio, dBaseSize] per round,
    covering its D-slice columns. All slots summed produce the final d_ape.
    """
    coreNum = 64
    dBaseSize = 128 // coff
    groupSize = headDim // 128
    nCorePerGroup = (coreNum // 2) // groupSize
    mPerRoundPerGroup = 128 * 2 // coff // cmpRatio
    mBaseSize = 128 // cmpRatio
    mPerRoundAll = nCorePerGroup * mPerRoundPerGroup

    dbRows = mPerRoundAll * cmpRatio
    ws = workspace_raw[:2 * dbRows * headDim].view(2 * dbRows, headDim).float()
    d_ape = torch.zeros(cmpRatio, coff * headDim)

    for coreIdx in range(coreNum):
        cubeCoreIdx = coreIdx // 2
        groupIdx = cubeCoreIdx // groupSize
        intraGroupIdx = cubeCoreIdx % groupSize
        nStart = intraGroupIdx * 128 + (coreIdx % coff) * dBaseSize

        for roundIdx in range(roundCnt):
            roundBlocks = min(mPerRoundAll,
                              totalBlocks - roundIdx * mPerRoundAll)
            preDealTcSize = groupIdx * mPerRoundPerGroup
            if coff == 1 and coreIdx % 2 == 1:
                preDealTcSize += mBaseSize
            if preDealTcSize >= roundBlocks:
                continue
            dealTcSize = min(mBaseSize,
                             max(0, roundBlocks - preDealTcSize))
            if dealTcSize <= 0:
                continue

            dbBase = (roundIdx % 2) * dbRows
            localSlot = groupIdx * mPerRoundPerGroup * cmpRatio
            if coff == 1 and coreIdx % 2 == 1:
                localSlot += mBaseSize * cmpRatio

            # [coff*cmpRatio, dBaseSize] at (dbBase + localSlot, nStart)
            tile = ws[dbBase + localSlot:dbBase + localSlot + coff * cmpRatio,
                      nStart:nStart + dBaseSize]

            for oid in range(coff):
                row_start = oid * cmpRatio
                col_start = oid * headDim + nStart
                d_ape[:, col_start:col_start + dBaseSize] += \
                    tile[row_start:row_start + cmpRatio, :]

    # Note: with 2x db, rounds N and N+2 share the same slot.
    # Phase 3 will consume per-round; standalone test sees only last 2 rounds.

    return d_ape

# ================================================================
#  Test runner
# ================================================================

def _check_block_size(case, batchSize, seqSize, layoutId, cmpRatio, coff, cacheMode,
                     cuNpu, sqs):
    """校验 case 数据的 block_size 字段合法性；不符则终止+告警（不静默替换）。

    cache_mode=1: block_size 为自由参数，需在 [1, 1024]；
    cache_mode=2: block_size 必须等于 min(coff*cmpRatio + S_for_bs - 1, 1024)
                  （S_for_bs: TH 布局 = max(seqused) 或 max(cu_seqlens 差)；BSH 布局 = S1）
    """
    bs = case["block_size"]
    name = case["testcase_name"]
    if cacheMode == 1:
        if not (1 <= bs <= 1024):
            raise SystemExit(
                f"[block_size 校验失败] {name}: cache_mode=1 block_size={bs} "
                f"不在 [1,1024] 范围内，请修正 case 数据后重跑")
        return
    # cache_mode=2
    if layoutId == 1:  # TH
        if sqs:
            S_for_bs = max(sqs)
        elif cuNpu is not None:
            S_for_bs = max(int(cuNpu[i + 1]) - int(cuNpu[i])
                           for i in range(batchSize)) if batchSize > 0 else 0
        else:
            S_for_bs = seqSize
    else:  # BSH
        S_for_bs = seqSize
    need = coff * cmpRatio + S_for_bs - 1
    expect = min(need, 1024)
    if bs != expect:
        raise SystemExit(
            f"[block_size 校验失败] {name}: cache_mode=2 block_size={bs} "
            f"与要求 {expect} 不一致 (coff*cmpRatio+S-1={need}, S_for_bs={S_for_bs})，"
            f"请修正 case 数据后重跑")

def check_forward_outputs(npu_result, xCpu, wkvCpu, wgateCpu, stateCacheCpu, apeCpu,
                           blockTableCpu, spGolden, sqGolden, cuGolden,
                           cmpRatio, coff, cacheMode, headDim, dataType):
    """通路 1 增强：正向输出（cmp_kv/softmax_score/kv）与小算子拼接参考比对。

    参考实现：compressor_forward.compressor_forward（纯 PyTorch 小算子拼接）。
    仅比对有效压缩块行（padding 行参考为 0、NPU 不保证写）。
    """
    coffD = coff * headDim
    kv_state = stateCacheCpu[..., :coffD]
    score_state = stateCacheCpu[..., coffD:]
    ref_cmp, ref_mask, _, _, ref_sm, ref_kv = compressor_forward.compressor_forward(
        xCpu, wkvCpu, wgateCpu, kv_state, score_state, apeCpu, blockTableCpu,
        [int(v) for v in spGolden] if spGolden is not None else None,
        cmp_ratio=cmpRatio, coff=coff, cache_mode=cacheMode,
        cu_seqlens=cuGolden,
        seqused=[int(v) for v in sqGolden] if sqGolden is not None else None,
        return_intermediates=True)

    valid = ref_mask.any(dim=-1)  # (B, blocks) 或 (rows,)

    def interleave(t):
        # 参考为顺序布局 (prev|cur)；NPU 正向输出为交错布局
        # (偶行=prev、奇行=cur)——比对前重排参考
        n = t.shape[-2]
        half = n // 2
        return torch.stack([t[..., :half, :], t[..., half:, :]], dim=-2).reshape(t.shape)

    def gather(t):
        t = t.detach().float()  # npu 侧输出/参考输入可能带 requires_grad（autograd 图）
        return t[valid]

    results = []
    for name, ref_t, npu_t, dt in (
        ("cmp_kv", ref_cmp, npu_result["cmp_kv"], dataType),
        ("softmax_score", interleave(ref_sm), npu_result["softmax_score"], "float32"),
        ("kv", interleave(ref_kv), npu_result["kv"], "float32"),
    ):
        pct, res = check_result(gather(ref_t), gather(npu_t), dt, get_pct_thd(dt))
        results.append(f"{name}={pct:.2f}%[{res.upper()}]".replace("PASS", "PASS"))
    ok = all("PASS" in r for r in results)
    return {"status": "PASS" if ok else "FAIL", "detail": "  ".join(results)}

def _parse_case(case, device):
    """解析 case 公共参数（通路 1/2 共用）。"""
    B, S1, H, D = case["B"], case["S1"], case["hidden_size"], case["D"]
    cr, coff = case["cmp_ratio"], case["coff"]
    layout = case["input_layout"]
    layoutId = 1 if layout == "TH" else 0
    dataType = case["dtype"]
    ioDtype = torch.float16 if dataType == "float16" else torch.bfloat16
    if layoutId == 1:
        tokenSize = int(case["seqlens_list_q"][-1])
        outputRows = min(tokenSize, tokenSize // cr + B)
        seqSize = 0                       # 与 host tiling 一致（TH 不设置 seqSize）
    else:
        tokenSize = B * S1
        outputRows = B * _ceil_div(S1, cr)
        seqSize = S1
    return dict(B=B, S1=S1, H=H, D=D, cr=cr, coff=coff, layout=layout,
                layoutId=layoutId, dataType=dataType, ioDtype=ioDtype,
                tokenSize=tokenSize, seqSize=seqSize, outputRows=outputRows)

def run_case(case, device):
    """Run one test case and return result dict."""
    name = case["testcase_name"]
    p = _parse_case(case, device)
    batchSize, seqSize, headDim, hiddenSize = p["B"], p["S1"], p["D"], p["H"]
    cmpRatio, coff = p["cr"], p["coff"]
    layoutId, ioDtype, dataType = p["layoutId"], p["ioDtype"], p["dataType"]
    layout = p["layout"]
    tokenSize, outputRows = p["tokenSize"], p["outputRows"]
    sps = case["start_pos"]
    if sps is None:
        sps = [0] * case["B"]
    sqs = case["seqused_q"]
    seqlens_list_q = case["seqlens_list_q"]
    seed = case["manual_seed"]

    dataTypeId = 0 if dataType == "bfloat16" else 1

    sqNpu = torch.tensor(sqs, device=device, dtype=torch.int32) if sqs else None
    spNpu = torch.tensor(sps, device=device, dtype=torch.int32) if sps else None

    # ── 构造 cu_seqlens（TH 布局；BSH 为 None）──
    if layoutId == 1:
        # case 数据统一为"0 开始累加"的累积形式（B+1 项、首项 0），直用不再调整
        if seqlens_list_q is not None:
            cuNpu = torch.tensor(seqlens_list_q, dtype=torch.int32)
        elif sqs is not None:
            cuNpu = torch.tensor([0] + torch.tensor(sqs).cumsum(0).tolist(),
                                 dtype=torch.int32)
        else:
            cuNpu = torch.zeros(batchSize + 1, dtype=torch.int32)
    else:
        cuNpu = None

    # ── Total valid compression blocks ──
    totalValid = 0
    for i in range(batchSize):
        sp = sps[i] if sps else 0
        if sqs:
            sq = sqs[i]
        elif cuNpu is not None:
            sq = int(cuNpu[i + 1].item() - cuNpu[i].item())
        else:
            sq = tokenSize if layoutId == 1 else tokenSize // batchSize
        cmpLimit = (sp + sq) // cmpRatio * cmpRatio
        if cmpLimit > sp:
            # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（与 kernel 一致）
            totalValid += (cmpLimit - sp + cmpRatio - 1) // cmpRatio

    scKvRowCount = coff * cmpRatio

    # ── Workspace sizing ──
    coreNum = 64
    groupSize = headDim // 128
    nCorePerGroup = (coreNum // 2) // groupSize
    mPerRoundPerGroup = 128 * 2 // coff // cmpRatio
    mPerRoundAll = nCorePerGroup * mPerRoundPerGroup
    roundCnt = (totalValid + mPerRoundAll - 1) // mPerRoundAll if totalValid > 0 else 0
    dbRows = mPerRoundAll * cmpRatio
    apeWsSize = 2 * dbRows * headDim
    dxPerGroup = 256 * hiddenSize
    dxPerRoundAll = 32 * dxPerGroup       # pl.get_block_num() = 32 Cube cores
    dxWsSize   = 2 * dxPerRoundAll * 2    # double: dkv@wkv + dsb@wgate
    # GM storage for dkv/dsb: 32 cube cores * [128*coff, 128//coff] FP16 each
    cvGmPerCore = 128 * coff * (128 // coff)  # = 128*128 = 16384 elements
    cvGmSize   = 32 * cvGmPerCore * 2          # dkv + dsb, FP16 (2 bytes each, but count as FP32 elements)
    wsSize     = apeWsSize + dxWsSize + cvGmSize

    # ── Generate random inputs ──
    torch.manual_seed(seed)
    if layoutId == 0:  # BSH: dc 与 cmp_kv 同为 (B, blocks, D)，与 golden 约定一致；kernel 按扁平读兼容
        dcNpu  = torch.randn(batchSize, (seqSize + cmpRatio - 1) // cmpRatio, headDim,
                             device=device, dtype=ioDtype)
    else:
        dcNpu  = torch.randn(outputRows, headDim, device=device, dtype=ioDtype)
    # 注：autograd 路径下 kernel 与 golden 实际使用前向计算的真实 kv/softmax_score
    #（见 backend.compute 的 saved tensors），此处不生成占位（避免与真实值不一致的误导）

    # ── Output tensors ──
    # d_x/d_wkv/d_wgate/d_ape: Phase 2/3 写入，占位用随机值（模拟真实未初始化内存，
    # 不依赖初始 0——kernel 未写输出时比对必须 FAIL 而非被零掩盖）
    if layoutId == 1:  # TH
        totalTokens = tokenSize
    else:  # BSH
        totalTokens = batchSize * seqSize
    d_xNpu     = torch.rand(totalTokens, hiddenSize, device=device,
                             dtype=ioDtype)
    d_wkvNpu   = torch.rand(coff * headDim, hiddenSize, device=device,
                              dtype=torch.float32)
    d_wgateNpu = torch.rand(coff * headDim, hiddenSize, device=device,
                              dtype=torch.float32)
    dapeWsNpu  = torch.zeros(wsSize, device=device, dtype=torch.float32)
    # d_ape final output (dummy — Phase 3 写入)
    dapeOutNpu = torch.rand(coff * cmpRatio * headDim, device=device, dtype=torch.float32)

    # TODO: 如需要 dkv/dsb GM 输出，需在 kernel 添加 pl.store 并在下方分配:
    #   dkvNpu = torch.zeros(outputRows, scKvRowCount, headDim,
    #                        device=device, dtype=ioDtype)
    #   dsbNpu = torch.zeros(outputRows, scKvRowCount, headDim,
    #                        device=device, dtype=ioDtype)
    #   然后将 dkvNpu/dsbNpu 加入 kernel 参数列表

    

    # cu_seqlens argument (None if BSH or noCu)
    if layoutId == 0 or case.get('noCu'):
        cuArg = None
    else:
        cuArg = cuNpu

    # x / wkv / wgate input tensors
    xNpu    = torch.randn(totalTokens, hiddenSize, device=device, dtype=ioDtype)
    wkvInp  = torch.randn(coff * headDim, hiddenSize, device=device, dtype=ioDtype)  # 单位阵
    wgateInp= torch.randn(coff * headDim, hiddenSize, device=device, dtype=ioDtype)

    # ── NPU: 通过 autograd 自动调用 compressor_grad ──
    backend = NPUBackend(device_id=DEVICE_ID)

    # 构建前向所需的 state_cache 和 ape 输入
    # state_cache: dtype 固定 fp32, shape = (block_num, block_size, 2*coff*D)
    # block_num 计算参考 attention/compressor 正向算子：
    #   compressor_golden.py:565-567  -> block_num = B * ceil(S_max / block_size)
    #   S_max 因 BSH/TH 布局而异，参考 compressor_operator_single.py:62-85
    #     BSH: S_max = max(start_pos) + S  (有 start_pos) 否则 S_max = S
    #     TH : S_max = max(start_pos[i] + cu_seqlens[i+1] - cu_seqlens[i]) (有 start_pos)
    #                       否则 max(cu_seqlens[i+1] - cu_seqlens[i])
    #   cache_mode=2: block_num = B (每 batch 一个 block)
    #                 参考 compressor_golden.py:608-616 (kv_state shape = (B, block_size, coff*D))
    cache_mode = case.get("cache_mode", 1)
    if cache_mode == 1:
        block_size = case.get("block_size", 128)
        # S_max 计算因 BSH/TH 布局而异
        if layoutId == 1:  # TH
            # 统一用修复后的 cuNpu（与 kernel 侧一致）
            cu_lens = cuNpu.tolist() if cuNpu is not None else [0] * (batchSize + 1)
            if sps:
                S_max = max(sps[i] + (cu_lens[i + 1] - cu_lens[i])
                            for i in range(batchSize)) if batchSize > 0 else 0
            else:
                S_max = max((cu_lens[i + 1] - cu_lens[i])
                            for i in range(batchSize)) if batchSize > 0 else 0
        else:  # BSH
            if sps:
                S_max = max(sps) + seqSize
            else:
                S_max = seqSize
        max_block_num_per_batch = (S_max + block_size - 1) // block_size
        block_num = batchSize * max_block_num_per_batch
        state_shape = (block_num, block_size, 2 * coff * headDim)
    else:
        # cache_mode=2: block_num = B (每 batch 一个 block)
        # block_size (第二维) 直接使用 case 数据（不再推导替换）；
        # 数据合法性由 _check_block_size 在下方统一校验（不符则终止+告警）
        token_cache_size_ex = case["block_size"]
        state_shape = (batchSize, token_cache_size_ex, 2 * coff * headDim)

    _check_block_size(case, batchSize, seqSize, layoutId, cmpRatio, coff, cache_mode,
                      cuNpu, sqs)

    state_cache_npu = torch.zeros(state_shape, device=device, dtype=torch.float32)
    ape_npu = torch.randn(torch.Size([cmpRatio, coff * headDim]), device=device, dtype=torch.float32)

    # block_table 构造，与正向 golden 一致：
    #   cache_mode=1: 参考 compressor_golden.py:565-600
    #     block_table shape = (B, max_block_num_per_batch), dtype int32
    #     需要按 start_pos/seqused 真实写入有效 block_id (1..block_num-1),
    #     block_id=0 表示无效 block (read_state_page_cache 会报错、write 跳过)
    #   cache_mode=2: 参考 compressor_golden.py:609
    #     block_table shape = (B,), dtype int32, 值为 [0, B) 的排列
    #     (每 batch 一个独立 block_id, 全 0 会导致多 batch 写同一行)
    block_table_npu = None
    if cache_mode == 1:
        block_table = torch.zeros(batchSize, max_block_num_per_batch,
                                  dtype=torch.int32)
        if batchSize > 0 and totalValid > 0:
            next_block_id = 1
            for i in range(batchSize):
                sp = sps[i] if sps else 0
                if sqs:
                    sq = sqs[i]
                elif cuNpu is not None:
                    sq = int(cuNpu[i + 1].item() - cuNpu[i].item())
                else:
                    sq = tokenSize if layoutId == 1 else tokenSize // batchSize
                end_pos = sq
                # 读取范围 (与 golden compressor_golden.py:572-583 一致)
                cur_start = sp // cmpRatio * cmpRatio - cmpRatio
                cur_end = sp // cmpRatio * cmpRatio + cmpRatio
                if sp % cmpRatio == 0:
                    cur_end = sp
                cur_end = min(cur_end, sp + sq)
                cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
                cur_end_block_id = (cur_end - 1) // block_size
                for j in range(cur_start_block_id, cur_end_block_id + 1):
                    if next_block_id < block_num:
                        block_table[i][j] = next_block_id
                        next_block_id = next_block_id + 1
                # 写入范围 (与 golden compressor_golden.py:589-600 一致)
                next_start = (sp + end_pos) // cmpRatio * cmpRatio - cmpRatio
                next_end = (sp + end_pos) // cmpRatio * cmpRatio + cmpRatio
                if (sp + end_pos) % cmpRatio == 0:
                    next_end = sp + end_pos
                next_end = min(next_end, sp + end_pos)
                next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
                next_end_block_id = (next_end - 1) // block_size
                for j in range(next_start_block_id, next_end_block_id + 1):
                    if next_block_id < block_num and block_table[i][j] == 0:
                        block_table[i][j] = next_block_id
                        next_block_id = next_block_id + 1
        block_table_npu = block_table.to(device)
    else:
        # cache_mode=2: block_table = [0, B) 的排列，与正向 golden 一致
        block_table_npu = torch.tensor(random.sample(list(range(batchSize)), batchSize),
                                       device=device, dtype=torch.int32) if batchSize > 0 else \
                          torch.zeros(0, device=device, dtype=torch.int32)

    if layoutId == 0:  # BSH
        x_npu_input = xNpu.view(batchSize, seqSize, hiddenSize)
    else:  # TH
        x_npu_input = xNpu

    npu_result = backend.compute(
        inputs={
            "x": x_npu_input,
            "wkv": wkvInp,
            "wgate": wgateInp,
            "state_cache": state_cache_npu,
            "ape": ape_npu,
            "d_cpm_kv": dcNpu,
            "cu_seqlens": cuArg,
            "seqused": sqNpu,
            "start_pos": spNpu,
            "block_table": block_table_npu,
            "cmp_ratio": cmpRatio,
            "coff": coff,
            "cache_mode": cache_mode,
            "input_layout": layout,
        },
    )

    d_xNpu.copy_(npu_result["d_x"].view_as(d_xNpu))
    d_wkvNpu.copy_(npu_result["d_wkv"])
    d_wgateNpu.copy_(npu_result["d_wgate"])
    dapeOutNpu.copy_(npu_result["d_ape"].flatten())
    torch.npu.synchronize()

    # ── CPU Golden: 使用 compressor_grad.py 作为完整反向参考 ──
    dcCpu  = dcNpu.cpu().float()
    spGolden = sps if sps else None
    if spGolden is not None and not isinstance(spGolden, torch.Tensor):
        spGolden = torch.tensor(spGolden, dtype=torch.int32)
    sqGolden = sqNpu.cpu() if sqNpu is not None else None
    # golden 与 kernel 共用同一份 cuNpu（统一来源）
    cuGolden = cuNpu.cpu() if cuNpu is not None else None

    # x 输入：TH 布局为 (T, H)，BSH 布局为 (B, S, H)
    xCpu = xNpu.cpu()
    if layoutId == 0:  # BSH
        xCpu = xCpu.view(batchSize, seqSize, hiddenSize)
    wkvCpu = wkvInp.cpu()
    wgateCpu = wgateInp.cpu()

    dXGold, dWkvGold, dWgateGold, apeGold = compressor_grad_golden(
        x=xCpu, wkv=wkvCpu, wgate=wgateCpu,
        # golden 约定 dc 为 (B, blocks, D)；测试构造为扁平 (blocks, D)，BSH 时 view 对齐
        d_cpm_kv=(dcCpu.view(batchSize, -1, headDim) if layoutId == 0 else dcCpu),
        # golden 输入需与 kernel（autograd）一致：用前向真实输出，而非占位 randn
        softmax_score=npu_result["softmax_score"].cpu(),
        kv=npu_result["kv"].cpu(),
        cu_seqlens=cuGolden, seqused=sqGolden, start_pos=spGolden,
        cmp_ratio=cmpRatio, coff=coff,
    )

    # ── Phase 3 writes d_ape directly ──
    apeNp = dapeOutNpu.cpu().view(cmpRatio, coff * headDim)

    result = dict(name=name, totalValid=totalValid,
                  coff=coff, cmpRatio=cmpRatio, headDim=headDim,
                  batchSize=batchSize)

    p_pct_thd = get_pct_thd(dataType)

    dWkvNp = d_wkvNpu.cpu().float()
    dWgateNp = d_wgateNpu.cpu().float()

    # ── 输出精度对比 ──
    # 通过 ENABLE_XXX_CHECK 控制是否对比；未开启 → SKIP，便于按需启用任意输出
    print_log("=" * 80)
    print_log(f"Start precision check for case '{name}' (validBlocks={totalValid})")

    # d_wkv check
    r_dwkv = check_one_output("d_wkv", dWkvGold.float(), dWkvNp, dataType,
                             ENABLE_DWKV_CHECK, totalValid, p_pct_thd)
    result["dwkvDiff"] = r_dwkv["diff"]
    result["dwkvPct"] = r_dwkv["pct"]
    result["dwkvStatus"] = r_dwkv["status"]

    # d_wgate check
    r_dwgate = check_one_output("d_wgate", dWgateGold.float(), dWgateNp, dataType,
                             ENABLE_DWGATE_CHECK, totalValid, p_pct_thd)
    result["dwgateDiff"] = r_dwgate["diff"]
    result["dwgatePct"] = r_dwgate["pct"]
    result["dwgateStatus"] = r_dwgate["status"]

    # d_ape check
    r_ape = check_one_output("d_ape", apeGold, apeNp, "float32",
                             ENABLE_APE_CHECK, totalValid, get_pct_thd("float32"))
    result["apeDiff"] = r_ape["diff"]
    result["apePct"] = r_ape["pct"]
    result["apeStatus"] = r_ape["status"]

    # d_x check
    dXNp = d_xNpu.cpu().float()
    dXGoldF = dXGold.float()
    r_dx = check_one_output("d_x", dXGoldF, dXNp, dataType,
                            ENABLE_DX_CHECK, totalValid, p_pct_thd)
    result["dxDiff"] = r_dx["diff"]
    result["dxPct"] = r_dx["pct"]
    result["dxStatus"] = r_dx["status"]

    # ── 通路 1 增强：正向输出校验（cmp_kv/softmax_score/kv vs 小算子拼接参考）──
    fwd = check_forward_outputs(npu_result, xCpu, wkvCpu, wgateCpu,
                                 state_cache_npu.cpu(), ape_npu.cpu(),
                                 block_table_npu.cpu(), spGolden, sqGolden, cuGolden,
                                 cmpRatio, coff, cache_mode, headDim, dataType)
    result["fwdStatus"] = fwd["status"]
    result["fwdDetail"] = fwd["detail"]
    print_log(f"Forward check: {fwd['detail']}")

    # ── Overall status ──
    # 任一 FAIL → FAIL；任一 ERROR → ERROR
    # 注：totalValid=0 不再 trivial PASS——四输出必须真实写 0（golden 全 0），
    #      kernel 未写输出（垃圾）时校验必须 FAIL（见 _zero_outputs）
    statuses = [result["dwkvStatus"], result["dwgateStatus"],
                result["apeStatus"], result["dxStatus"], result["fwdStatus"]]
    if any(s == "ERROR" for s in statuses):
        result["status"] = "ERROR"
    elif any(s == "FAIL" for s in statuses):
        result["status"] = "FAIL"
    elif all(s == "SKIP" for s in statuses):
        result["status"] = "SKIP"
    else:
        result["status"] = "PASS"

    return result

def run_backward_case(case, device):
    """通路 2：单反向——随机构造中间量 → PyPTO 直跑 kernel → golden 比对。

    不经 aclnn/custom 包（kernel 运行时 JIT 编译）；TilingData 29 字段
    派生逻辑与 op_host/compressor_grad_tiling.cpp 一致。
    """
    p = _parse_case(case, device)
    B, H, D = p["B"], p["H"], p["D"]
    cr, coff = p["cr"], p["coff"]
    layoutId, ioDtype = p["layoutId"], p["ioDtype"]
    tokenSize, seqSize, outputRows = p["tokenSize"], p["seqSize"], p["outputRows"]
    sqs, sps = case.get("seqused_q"), case.get("start_pos")

    if layoutId == 1:
        cuNpu = torch.tensor(case["seqlens_list_q"], device=device, dtype=torch.int32)
    else:
        cuNpu = None
    sqNpu = torch.tensor(sqs, device=device, dtype=torch.int32) if sqs else None
    spNpu = torch.tensor(sps, device=device, dtype=torch.int32) if sps else None

    # ── TilingData（派生与 op_host/compressor_grad_tiling.cpp 一致）──
    cubeCoreNum, coreNum = 32, 64
    coffCoef = 2 // coff
    totalHeadDim = coff * D
    cmpSize = coff * cr * D
    cmpKvBatchStride = _ceil_div(seqSize, cr)
    xRows = tokenSize if layoutId == 1 else B * seqSize
    groupSize = D // 128
    groupNum = cubeCoreNum // groupSize
    dealScNum = 128 // cr
    groupDealScNum = dealScNum * coffCoef
    totalScNumPerRound = groupNum * groupDealScNum
    groupRowStride = groupDealScNum * cr + (coff - 1) * cr
    dbRowCnt = groupNum * groupRowStride
    tiling = CompressorGradTiling(
        batchSize=B, tokenSize=tokenSize, seqSize=seqSize,
        cmpRatio=cr, hiddenSize=H, headDim=D,
        cubeCoreNum=cubeCoreNum, coreNum=coreNum,
        totalHeadDim=totalHeadDim, cmpRowCnt=coff * cr, cmpSize=cmpSize,
        cmpKvBatchStride=cmpKvBatchStride, cmpKvRows=outputRows, xRows=xRows,
        groupSize=groupSize, groupNum=groupNum,
        groupDealScNum=groupDealScNum, dealScNum=dealScNum,
        totalScNumPerRound=totalScNumPerRound, dbRowCnt=dbRowCnt,
        groupRowStride=groupRowStride,
        coffCoef=coffCoef, cubeMBaseSize=128 * coffCoef,
        dDealSize=128 // coff, mDealSize=128 * coff,
        dapeWsSize=groupNum * cmpSize * coffCoef,
        dXWsSize=2 * cubeCoreNum * 256 * H,
        dWWeightWsSize=groupNum * totalHeadDim * H,
        xWsSize=2 * groupNum * groupRowStride * H,
        dXCacheWsSize=2 * cr * H,
    )

    # ── 随机构造输入/中间量（不依赖正向输出）──
    torch.manual_seed(case["manual_seed"])
    xNpu = torch.randn(xRows, H, device=device, dtype=ioDtype)
    wkvInp = torch.randn(coff * D, H, device=device, dtype=ioDtype)
    wgateInp = torch.randn(coff * D, H, device=device, dtype=ioDtype)
    dcNpu = torch.randn(outputRows, D, device=device, dtype=ioDtype)
    kvNpu = torch.randn(outputRows, coff * cr, D, device=device, dtype=torch.float32)
    smNpu = torch.softmax(torch.randn(outputRows, coff * cr, D, device=device), dim=1)

    # ── 输出与 workspace（精确大小，FP32 元素数）──
    dX = torch.zeros(xRows, H, device=device, dtype=ioDtype)
    dWkv = torch.zeros(coff * D, H, device=device, dtype=ioDtype)
    dWgate = torch.zeros(coff * D, H, device=device, dtype=ioDtype)
    dApe = torch.zeros(cr * coff * D, device=device, dtype=torch.float32)
    wsSize = (tiling.dapeWsSize + tiling.dXWsSize + 2 * tiling.dWWeightWsSize +
              tiling.xWsSize + tiling.dXCacheWsSize)
    workspace = torch.zeros(wsSize, device=device, dtype=torch.float32)

    # ── PyPTO 直跑（不经 aclnn/custom 包）──
    compressor_grad[None, cubeCoreNum,
                    {"Coff": coff, "Layout": layoutId,
                     "DataType": 0 if ioDtype == torch.bfloat16 else 1}](
        xNpu, wkvInp, wgateInp, dcNpu, smNpu, kvNpu,
        cuNpu, sqNpu, spNpu,
        dX, dWkv, dWgate, dApe, workspace, tiling)
    torch.npu.synchronize()

    # ── golden（BSH 输入需 3 维；dc/sm/kv 需 (B, blocks, ...) 维）──
    xCpu = xNpu.cpu().view(B, seqSize, H) if layoutId == 0 else xNpu.cpu()
    smCpu = smNpu.cpu().view(B, -1, coff * cr, D) if layoutId == 0 else smNpu.cpu()
    kvCpu = kvNpu.cpu().view(B, -1, coff * cr, D) if layoutId == 0 else kvNpu.cpu()
    g_dx, g_dwkv, g_dwgate, g_dape = compressor_grad_golden(
        x=xCpu, wkv=wkvInp.cpu().float(), wgate=wgateInp.cpu().float(),
        d_cpm_kv=(dcNpu.cpu().view(B, -1, D) if layoutId == 0 else dcNpu.cpu()),
        softmax_score=smCpu, kv=kvCpu,
        cu_seqlens=cuNpu.cpu() if cuNpu is not None else None,
        seqused=sqNpu.cpu() if sqNpu is not None else None,
        start_pos=spNpu.cpu() if spNpu is not None else None,
        cmp_ratio=cr, coff=coff)

    # ── 四输出比对 ──
    p_pct_thd = get_pct_thd(p["dataType"])
    checks = [
        ("d_wkv", g_dwkv.float(), dWkv.cpu().float(), p["dataType"], p_pct_thd),
        ("d_wgate", g_dwgate.float(), dWgate.cpu().float(), p["dataType"], p_pct_thd),
        ("d_ape", g_dape, dApe.cpu().float(), "float32", get_pct_thd("float32")),
        ("d_x", g_dx.float(), dX.cpu().float(), p["dataType"], p_pct_thd),
    ]
    statuses = []
    for name, exp, act, dt, thd in checks:
        r = check_one_output(name, exp, act, dt, True, 1, thd)
        statuses.append(f"{name}={r['status']}")
    status = "ERROR" if any("ERROR" in s for s in statuses) else \
             "FAIL" if any("FAIL" in s for s in statuses) else "PASS"
    return {"status": status, "detail": "  ".join(statuses)}


def _pathway(request):
    """当前验证通路（--pathway 入参）。"""
    return request.config.getoption("--pathway")


@pytest.mark.parametrize(
    "case",
    [c for c in REDLINE_CASES if c.get("enable", True)],
    ids=[c["testcase_name"] for c in REDLINE_CASES if c.get("enable", True)],
)
def test_compressor_grad(case, request):
    """通路 1：autograd 正反向全链路 + 反向四输出校验
    + 正向输出校验（cmp_kv/softmax_score/kv vs compressor_forward 参考）。

    运行：pytest test_compressor_grad.py -k <case名子串> -q
    """
    if _pathway(request) != 1:
        pytest.skip(f"pathway={_pathway(request)}")
    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)
    result = run_case(case, device)
    assert result["status"] == "PASS", format_case_line(result)


@pytest.mark.parametrize(
    "case",
    [c for c in REDLINE_CASES if c.get("enable", True)],
    ids=[c["testcase_name"] for c in REDLINE_CASES if c.get("enable", True)],
)
def test_compressor_grad_backward(case, request):
    """通路 2：单反向（PyPTO 直跑，随机构造中间量，不经 custom 包）。

    运行：pytest test_compressor_grad.py --pathway 2 -k <case名子串> -q
    """
    if _pathway(request) != 2:
        pytest.skip(f"pathway={_pathway(request)}")
    result = run_backward_case(case, f"npu:{DEVICE_ID}")
    assert result["status"] == "PASS", result["detail"]
