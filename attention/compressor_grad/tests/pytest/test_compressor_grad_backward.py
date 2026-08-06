# ======================================================================================================================
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""通路 2：单反向通路——PyPTO 直跑反向 kernel（不经 aclnn / custom 包）。

中间结果（d_cmp_kv / softmax_score / kv）由测试框架随机构造（不依赖正向算子输出）；
golden 用 compressor_grad_cpu_golden；比对四输出（d_x/d_wkv/d_wgate/d_ape）。

用法:
    pytest test_compressor_grad_backward.py -k TEST_100 -q     # 单 case
    pytest test_compressor_grad_backward.py -q                 # 全部
"""

import os
import sys

import torch
import torch_npu
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "op_kernel"))
from compressor_grad import compressor_grad, CompressorGradTiling  # noqa: E402

from test_compressor_grad_cases import REDLINE_CASES
from compressor_grad_cpu_golden import compressor_grad_golden
from compressor_grad_check import check_one_output, get_pct_thd, format_case_line  # noqa: E402

DEVICE_ID = 0
ENABLE_DWKV_CHECK = True
ENABLE_DWGATE_CHECK = True
ENABLE_APE_CHECK = True
ENABLE_DX_CHECK = True


def _ceil_div(a, b):
    return (a + b - 1) // b


def run_backward_case(case, device):
    """单反向：随机构造中间量 → PyPTO 直跑 kernel → golden 比对。"""
    B, S1, H, D = case["B"], case["S1"], case["hidden_size"], case["D"]
    cr, coff = case["cmp_ratio"], case["coff"]
    layout = case["input_layout"]
    sqs, sps = case.get("seqused_q"), case.get("start_pos")
    layoutId = 1 if layout == "TH" else 0
    ioDtype = torch.bfloat16 if case["dtype"] == "bfloat16" else torch.float16

    if layoutId == 1:
        cu = case["seqlens_list_q"]
        tokenSize = int(cu[-1])
        seqSize = 0                       # 与 host tiling 一致（TH 不设置 seqSize）
        cuNpu = torch.tensor(cu, device=device, dtype=torch.int32)
        outputRows = min(tokenSize, tokenSize // cr + B)
    else:
        tokenSize = B * S1
        seqSize = S1
        cuNpu = None
        outputRows = B * _ceil_div(seqSize, cr)
    sqNpu = torch.tensor(sqs, device=device, dtype=torch.int32) if sqs else None
    spNpu = torch.tensor(sps, device=device, dtype=torch.int32) if sps else None

    # ── TilingData（29 字段，派生逻辑与 op_host/compressor_grad_tiling.cpp 一致）──
    cubeCoreNum, coreNum = 32, 64
    coffCoef = 2 // coff
    totalHeadDim = coff * D
    cmpRowCnt = coff * cr
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
        totalHeadDim=totalHeadDim, cmpRowCnt=cmpRowCnt, cmpSize=cmpSize,
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
                    {"Coff": coff, "Layout": layoutId, "DataType": 0 if ioDtype == torch.bfloat16 else 1}](
        xNpu, wkvInp, wgateInp, dcNpu, smNpu, kvNpu,
        cuNpu, sqNpu, spNpu,
        dX, dWkv, dWgate, dApe, workspace, tiling)
    torch.npu.synchronize()

    # ── golden（BSH 输入需 3 维；dc/sm/kv 需 (B, blocks, ...) 4 维/3 维）──
    xCpu = xNpu.cpu().view(B, S1, H) if layoutId == 0 else xNpu.cpu()
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

    p_pct_thd = get_pct_thd(case["dtype"])
    checks = [
        ("d_wkv", g_dwkv.float(), dWkv.cpu().float(), ENABLE_DWKV_CHECK),
        ("d_wgate", g_dwgate.float(), dWgate.cpu().float(), ENABLE_DWGATE_CHECK),
        ("d_ape", g_dape, dApe.cpu().float(), ENABLE_APE_CHECK),
        ("d_x", g_dx.float(), dX.cpu().float(), ENABLE_DX_CHECK),
    ]
    result = {}
    for name, exp, act, en in checks:
        r = check_one_output(name, exp, act, "float32" if name == "d_ape" else case["dtype"],
                             en, 1, get_pct_thd("float32") if name == "d_ape" else p_pct_thd)
        result[name] = r["status"]
    status = "ERROR" if any(s == "ERROR" for s in result.values()) else \
             "FAIL" if any(s == "FAIL" for s in result.values()) else "PASS"
    return {"status": status, "detail": "  ".join(f"{k}={v}" for k, v in result.items())}


@pytest.mark.parametrize(
    "case",
    [c for c in REDLINE_CASES if c.get("enable", True)],
    ids=[c["testcase_name"] for c in REDLINE_CASES if c.get("enable", True)],
)
def test_compressor_grad_backward(case):
    """通路 2：单反向（PyPTO 直跑，随机构造中间量）。"""
    result = run_backward_case(case, f"npu:{DEVICE_ID}")
    assert result["status"] == "PASS", result["detail"]
