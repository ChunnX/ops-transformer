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

import sys
import argparse
from dataclasses import dataclass

import torch

from test_cases import REDLINE_CASES
from cpu_golden import compressor_grad_golden
from npu_core import NPUBackend


DEVICE_ID = 0


# Precision thresholds (from ops-transformer compressor test standards)
PRECISION = {
    "bf16": {"rtol": 0.0078125, "atol": 0.0001, "pctThd": 99.5},
    "fp16": {"rtol": 0.005,     "atol": 0.000025, "pctThd": 99.5},
}


# ── Output toggle flags ──────────────────────────────────────────
# dkv/dsb: unified kernel 不输出 dkv/dsb 到 GM（作为 CV 中间数据传给 Cube）
#   如需测试 dkv/dsb，需在 kernel 中添加 pl.store 写 GM 指令，
#   然后在此处分配输出 tensor 并开启对应 flag
ENABLE_DKV_CHECK  = False
ENABLE_DSB_CHECK  = False
ENABLE_APE_CHECK  = False
ENABLE_DX_CHECK   = True    # Phase 3 已实现


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

def npu_test(case, device, seed=42):
    """Run one test case and return result dict."""
    name = case["testcase_name"]
    batchSize = case["B"]
    # tokenSize = case["tokenSize"]  ## 需计算
    seqSize = case["S1"]
    headDim = case["D"]
    hiddenSize = case["hidden_size"]
    cmpRatio = case["cmp_ratio"]
    coff = case["coff"]
    layout = case["input_layout"]
    sps = case["start_pos"]
    sqs = case["seqused_q"]
    seqlens_list_q = case["seqlens_list_q"]
    dataType = case["dtype"]

    tokenSize = batchSize * seqSize
    if layoutId == 0:
        tokenSize = seqlens_list_q[-1]

    dtypeMap   = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    ioDtype    = dtypeMap[dataType]
    dataTypeId = 0 if dataType == "bfloat16" else 1
    layoutId   = 0 if layout == "TH" else 1

    scKvRowCount = coff * cmpRatio

    # ── Build cu_seqlens / seqused / start_pos ──
    if layoutId == 0:
        seqlens_list = case.get('seqlens_list_q')
        if seqlens_list is not None:
            cuNpu = torch.tensor(seqlens_list)
        else:
            cuNpu = torch.tensor([0] + torch.tensor(sqs).cumsum(0).tolist(), 
                                device=device, dtype=torch.int32)
    else:
        cuNpu = None

    if layoutId == 0:
        outputRows = min(tokenSize, (tokenSize // cmpRatio) + batchSize)
    else:
        outputRows = batchSize * ((seqSize + cmpRatio - 1) // cmpRatio)

    sqNpu = torch.tensor(sqs, device=device, dtype=torch.int32) if sqs else None
    spNpu = torch.tensor(sps, device=device, dtype=torch.int32) if sps else None

    # ── Total valid compression blocks ──
    totalValid = 0
    for i in range(batchSize):
        sp = sps[i] if sps else 0
        if sqs:
            sq = sqs[i]
        elif cuNpu is not None:
            sq = int(cuNpu[i + 1].item() - cuNpu[i].item())
        else:
            sq = tokenSize // batchSize if layoutId == 1 else tokenSize
        cmpLimit = (sp + sq) // cmpRatio * cmpRatio
        if cmpLimit > sp:
            totalValid += (cmpLimit - sp) // cmpRatio

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
    dcNpu  = torch.randn(outputRows, headDim, device="npu:0", dtype=ioDtype)
    kvNpu  = torch.randn(outputRows, scKvRowCount, headDim, device="npu:0",
                         dtype=torch.float32)
    smNpu  = torch.softmax(
        torch.randn(outputRows, scKvRowCount, headDim, device="npu:0"), dim=1)

    # ── Output tensors ──
    # d_x/d_wkv/d_wgate: Phase 2/3 写入，当前仅占位
    if layoutId == 0:
        totalTokens = tokenSize
    else:
        totalTokens = batchSize * seqSize
    d_xNpu     = torch.zeros(totalTokens, hiddenSize, device="npu:0",
                             dtype=ioDtype)
    d_wkvNpu   = torch.zeros(coff * headDim, hiddenSize, device="npu:0",
                              dtype=torch.float32)
    d_wgateNpu = torch.zeros(coff * headDim, hiddenSize, device="npu:0",
                              dtype=torch.float32)
    dapeWsNpu  = torch.zeros(wsSize, device="npu:0", dtype=torch.float32)
    # d_ape final output (dummy — Phase 3 写入)
    dapeOutNpu = torch.zeros(coff * cmpRatio * headDim, device="npu:0", dtype=torch.float32)

    # TODO: 如需要 dkv/dsb GM 输出，需在 kernel 添加 pl.store 并在下方分配:
    #   dkvNpu = torch.zeros(outputRows, scKvRowCount, headDim,
    #                        device="npu:0", dtype=ioDtype)
    #   dsbNpu = torch.zeros(outputRows, scKvRowCount, headDim,
    #                        device="npu:0", dtype=ioDtype)
    #   然后将 dkvNpu/dsbNpu 加入 kernel 参数列表

    

    # cu_seqlens argument (None if BSH or noCu)
    if layoutId == 1 or case.get('noCu'):
        cuArg = None
    else:
        cuArg = cuNpu

    # x / wkv / wgate input tensors
    xNpu    = torch.randn(totalTokens, hiddenSize, device="npu:0", dtype=ioDtype)
    wkvInp  = torch.randn(coff * headDim, hiddenSize, device="npu:0", dtype=ioDtype)  # 单位阵
    wgateInp= torch.randn(coff * headDim, hiddenSize, device="npu:0", dtype=ioDtype)

    # ── NPU: 通过 autograd 自动调用 compressor_grad ──
    backend = NPUBackend(device_id=DEVICE_ID)

    # 构建前向所需的 state_cache 和 ape 输入
    # cache_mode=1： state_cache shape = (block_num, block_size, 2*coff*D)
    # cache_mode=2： state_cache shape = (B, token_size, 2*coff*D)
    cache_mode = case.get("cache_mode", 1)
    if cache_mode == 1:
        block_size = case.get("block_size", 128)
        max_block_num_per_batch = ((seqSize - 1) // block_size) + 1
        block_num = batchSize * max_block_num_per_batch
        state_shape = (block_num, block_size, 2 * coff * headDim)
    else:
        token_cache_size_ex = 2 * cmpRatio + seqSize - 1 if coff == 2 else cmpRatio + seqSize - 1
        state_shape = (batchSize, token_cache_size_ex, 2 * coff * headDim)

    state_cache_npu = torch.zeros(state_shape, device="npu:{}".format(DEVICE_ID), dtype=torch.float32)
    ape_npu = torch.randn(torch.Size([cmpRatio, coff * headDim]), device="npu:{}".format(DEVICE_ID), dtype=torch.float32)

    block_table_npu = None
    if cache_mode == 1:
        block_table_npu = torch.zeros(batchSize, max_block_num_per_batch,
                                      device="npu:{}".format(DEVICE_ID), dtype=torch.int32)

    if layoutId == 1:  # BSH
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
    kvCpu  = kvNpu.cpu()
    smCpu  = smNpu.cpu()
    spGolden = sps if sps else None
    if spGolden is not None and not isinstance(spGolden, torch.Tensor):
        spGolden = torch.tensor(spGolden, dtype=torch.int32)
    sqGolden = sqNpu.cpu() if sqNpu is not None else None
    cuGolden = cuNpu.cpu() if cuNpu is not None else None
    # golden 的 compressor_grad 用 cu_seqlens 判断 TH/BSH 布局，
    # 而 noCu 测试用例不传 cu_seqlens 给 kernel，需为 golden 单独构造
    if layoutId == 0 and cuGolden is None and sqs is not None:
        cuGolden = torch.tensor([0] + torch.tensor(sqs).cumsum(0).tolist(), dtype=torch.int32)

    # x 输入：TH 布局为 (T, H)，BSH 布局为 (B, S, H)
    xCpu = xNpu.cpu()
    if layoutId == 1:  # BSH
        xCpu = xCpu.view(batchSize, seqSize, hiddenSize)
    wkvCpu = wkvInp.cpu()
    wgateCpu = wgateInp.cpu()

    dXGold, _, _, apeGold = compressor_grad_golden(
        x=xCpu, wkv=wkvCpu, wgate=wgateCpu,
        d_cpm_kv=dcCpu, softmax_score=smCpu, kv=kvCpu,
        cu_seqlens=cuGolden, seqused=sqGolden, start_pos=spGolden,
        cmp_ratio=cmpRatio, coff=coff,
    )
    # print("d_cpm_kv:")
    # print(d_cpm_kv.shape)
    # print(d_cpm_kv[0])
    # print(dXGold.shape)
    # print(dXGold)
    # print(wkvCpu.shape)
    # print(wkvCpu[0])
    # print(wgateCpu.shape)
    # print(wgateCpu[0])
    # ── Phase 3 writes d_ape directly ──
    apeNp = dapeOutNpu.cpu().view(cmpRatio, coff * headDim)

    # ── DEBUG: inspect workspace partials for Quick_coff1_cr4 ──
    if name == "Quick_coff1_cr4":
        ws_cpu = dapeWsNpu.cpu().float()
        dbRows_dbg = mPerRoundAll * cmpRatio
        apeWsSize_dbg = 2 * dbRows_dbg * headDim
        dxWsElems_dbg = 2 * dxPerRoundAll
        dx_ws = ws_cpu[apeWsSize_dbg:apeWsSize_dbg + dxWsElems_dbg].view(-1, hiddenSize)
        ds_ws = ws_cpu[apeWsSize_dbg + dxWsElems_dbg:apeWsSize_dbg + 2*dxWsElems_dbg].view(-1, hiddenSize)
        print(f"\n[DBG] {name}: d_x_ws shape={dx_ws.shape}, d_s_ws shape={ds_ws.shape}")
        print(f"[DBG] d_x_ws[0, 0:8] = {dx_ws[0, 0:8].tolist()}")
        print(f"[DBG] d_x_ws[128, 0:8] = {dx_ws[128, 0:8].tolist()}")
        print(f"[DBG] d_s_ws[0, 0:8] = {ds_ws[0, 0:8].tolist()}")
        print(f"[DBG] d_s_ws[128, 0:8] = {ds_ws[128, 0:8].tolist()}")
        d_x_out_dbg = d_xNpu.cpu().float()
        print(f"[DBG] d_x_out[0, 0:8] = {d_x_out_dbg[0, 0:8].tolist()}")
        print(f"[DBG] d_x_out[1, 0:8] = {d_x_out_dbg[1, 0:8].tolist()}")
        try:
            dc_block0 = dcCpu[0]
            kv_block0 = kvCpu[0]
            sm_block0 = smCpu[0]
            dkv_block0 = dc_block0.unsqueeze(0) * sm_block0
            print(f"[DBG] golden dkv_block0[0, 0:8] = {dkv_block0[0, 0:8].tolist()}")
            wkvCpu_f32 = wkvCpu.float()
            dkv_wkv_0 = dkv_block0 @ wkvCpu_f32
            print(f"[DBG] golden dkv@wkv[0, 0:8] = {dkv_wkv_0[0, 0:8].tolist()}")
            dsw_block0 = dc_block0.unsqueeze(0) * kv_block0
            col_sum0 = (sm_block0 * dsw_block0).sum(dim=0, keepdim=True)
            dsb_block0 = sm_block0 * (dsw_block0 - col_sum0)
            wgateCpu_f32 = wgateCpu.float()
            dsb_wgate_0 = dsb_block0 @ wgateCpu_f32
            print(f"[DBG] golden dsb@wgate[0, 0:8] = {dsb_wgate_0[0, 0:8].tolist()}")
            print(f"[DBG] golden d_x[0, 0:8] = {(dkv_wkv_0[0] + dsb_wgate_0[0])[0:8].tolist()}")
            print(f"[DBG] golden dXGold[0, 0:8] = {dXGold[0, 0:8].tolist()}")
        except Exception as e:
            print(f"[DBG] golden computation error: {e}")
            import traceback; traceback.print_exc()

    result = dict(name=name, totalValid=totalValid,
                  coff=coff, cmpRatio=cmpRatio, headDim=headDim,
                  batchSize=batchSize)

    p = PRECISION[dataType]
    pctThd = p["pctThd"]

    # ── dkv check (disabled — kernel 未输出 dkv 到 GM) ──
    if ENABLE_DKV_CHECK and totalValid > 0:
        # TODO: kernel 添加 dkv GM store 后:
        #   dkvNp = dkvNpu.cpu().float()[:totalValid]
        #   ... compare with dkvGold ...
        result["dkvDiff"] = float("nan")
        result["dkvPct"] = 100.0
        result["dkvStatus"] = "SKIP"
    else:
        result["dkvDiff"] = float("nan")
        result["dkvPct"] = 100.0
        result["dkvStatus"] = "SKIP"

    # ── dsb check (disabled — kernel 未输出 dsb 到 GM) ──
    if ENABLE_DSB_CHECK and totalValid > 0:
        # TODO: kernel 添加 dsb GM store 后:
        #   dsbNp = dsbNpu.cpu().float()[:totalValid]
        #   ... compare with dsbGold ...
        result["dsbDiff"] = float("nan")
        result["dsbPct"] = 100.0
        result["dsbStatus"] = "SKIP"
    else:
        result["dsbDiff"] = float("nan")
        result["dsbPct"] = 100.0
        result["dsbStatus"] = "SKIP"

    # ── d_ape check (enabled) ──
    if ENABLE_APE_CHECK and totalValid > 0:
        apeDiff = (apeNp - apeGold).abs().max().item()
        apeClose = torch.isclose(apeNp, apeGold,
                                 rtol=1e-4, atol=1e-4, equal_nan=True)
        apePct = apeClose.float().mean().item() * 100.0
        result["apeDiff"] = apeDiff
        result["apePct"] = apePct
        result["apeStatus"] = "PASS" if apePct >= pctThd else "FAIL"
    else:
        result["apeDiff"] = float("nan")
        result["apePct"] = 100.0
        result["apeStatus"] = "SKIP"

    # ── d_x check (enabled) ──
    if ENABLE_DX_CHECK and totalValid > 0:
        dXNp = d_xNpu.cpu().float()
        dXGoldF = dXGold.float()
        dxDiff = (dXNp - dXGoldF).abs().max().item()
        dxClose = torch.isclose(dXNp, dXGoldF,
                                rtol=1e-4, atol=1e-4, equal_nan=True)
        dxPct = dxClose.float().mean().item() * 100.0
        result["dxDiff"] = dxDiff
        result["dxPct"] = dxPct
        result["dxStatus"] = "PASS" if dxPct >= pctThd else "FAIL"
    else:
        result["dxDiff"] = float("nan")
        result["dxPct"] = 100.0
        result["dxStatus"] = "SKIP"

    # ── Overall status ──
    if totalValid == 0:
        result["status"] = "PASS"  # No valid blocks → trivial pass
    else:
        statuses = [result.get("dkvStatus", "SKIP"),
                    result.get("dsbStatus", "SKIP"),
                    result.get("apeStatus", "SKIP"),
                    result.get("dxStatus", "SKIP")]
        if any(s == "FAIL" for s in statuses):
            result["status"] = "FAIL"
        elif all(s == "SKIP" for s in statuses):
            result["status"] = "SKIP"
        else:
            result["status"] = "PASS"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",  type=str, default=None)
    parser.add_argument("--dataType", type=str, default="fp16",
                        choices=["fp16", "bf16"])
    args = parser.parse_args()

    if args.case:
        allCases = REDLINE_CASES
        cases = [c for c in allCases if c["testcase_name"] == args.case]
        if not cases:
            print(f"Case '{args.case}' not found. "
                  f"Available: {[c['testcase_name'] for c in allCases]}")
            return 1
    else:
        cases = REDLINE_CASES

    # Override dataType
    for c in cases:
        c["dataType"] = args.dataType

    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)

    print("=" * 70)
    print(f"Compressor Grad (unified kernel) — {len(cases)} case(s), "
          f"dtype={args.dataType}")
    print(f"  Checks: dkv={ENABLE_DKV_CHECK}  dsb={ENABLE_DSB_CHECK}  "
          f"ape={ENABLE_APE_CHECK}  dx={ENABLE_DX_CHECK}")
    print("=" * 70)

    results = []
    for i, c in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {c['testcase_name']}  "
              f"B={c['B']} hd={c['D']} cr={c['cmp_ratio']} "
              f"coff={c['coff']} {c['input_layout']}")
        try:
            r = npu_test(c, device, seed=42 + i)
            results.append(r)

            parts = []
            if ENABLE_DKV_CHECK:
                parts.append(f"dkv={r['dkvPct']:.2f}%")
            if ENABLE_DSB_CHECK:
                parts.append(f"dsb={r['dsbPct']:.2f}%")
            if ENABLE_APE_CHECK:
                parts.append(f"ape={r['apePct']:.2f}%")
            if ENABLE_DX_CHECK:
                parts.append(f"dx={r['dxPct']:.2f}%")
            detail = "  ".join(parts) if parts else "(no checks enabled)"
            print(f"  {r['status']:5s}  {detail}  "
                  f"(validBlocks={r['totalValid']})")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(dict(name=c["testcase_name"], status="ERROR"))

    # ── Summary ──
    print("\n" + "=" * 70)
    passed  = sum(1 for r in results if r.get("status") == "PASS")
    failed  = sum(1 for r in results if r.get("status") == "FAIL")
    skipped = sum(1 for r in results if r.get("status") == "SKIP")
    errors  = sum(1 for r in results if r.get("status") == "ERROR")
    print(f"SUMMARY: {passed} PASS, {failed} FAIL, {skipped} SKIP, "
          f"{errors} ERROR out of {len(results)}")
    print("=" * 70)

    return 0 if (failed + errors) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
