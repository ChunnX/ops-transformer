/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file k2q_csr.cpp
 * \brief ACLNN wrapper：按命名阶段串行调用
 *   empty(scratch) → Meta → empty(row_ptr)+zero → Hist → RowPrefix → TilePrefix
 *   → empty(q_ind/slot)+fill_(-1) → Scatter
 * total_rows / max_kv：CPU 传入则直接用；未传（<0）时再 D2H。
 * use_simt：Hist/Scatter SIMT（非 950 由各算子 tiling 强制 0）。
 * q_global_offset：1 时 q_ind 为全局 Q 下标；0 为 batch-local（默认）。
 */

#include <algorithm>
#include <cstdint>
#include <tuple>

#include <torch/extension.h>
#include "aclnn_common.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_THREE = 3;
const int64_t K_MIN_Q_PER_CTA = 256;

static int64_t GetAivNumFromDevice()
{
    int32_t deviceId = 0;
    aclError derr = c10_npu::GetDevice(&deviceId);
    if (derr != ACL_SUCCESS) {
        deviceId = static_cast<int32_t>(c10_npu::current_device());
    }

    int64_t aiv = 0;
    aclError ret = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv);
    if (ret != ACL_SUCCESS || aiv <= 0) {
        ret = aclGetDeviceCapability(static_cast<uint32_t>(deviceId), ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv);
    }
    TORCH_CHECK(ret == ACL_SUCCESS && aiv > 0, "k2q_csr: query VECTOR_CORE_NUM failed, device=", deviceId,
                " ret=", static_cast<int>(ret), " aiv=", aiv);
    return aiv;
}

/** 对齐 op_host FillCudaLikeGroups（kMinQPerCta=256） */
static int64_t GroupCount(int64_t tokenNum, int64_t aiv)
{
    if (tokenNum <= 0) {
        return 1;
    }
    if (aiv < 1) {
        aiv = 1;
    }
    int64_t targetG = aiv;
    int64_t maxGForQ = (tokenNum + K_MIN_Q_PER_CTA - 1) / K_MIN_Q_PER_CTA;
    if (maxGForQ < 1) {
        maxGForQ = 1;
    }
    int64_t groupNum = targetG;
    if (groupNum > maxGForQ) {
        groupNum = maxGForQ;
    }
    if (groupNum > tokenNum) {
        groupNum = tokenNum;
    }
    if (groupNum < 1) {
        groupNum = 1;
    }
    int64_t qpc = (tokenNum + groupNum - 1) / groupNum;
    if (qpc < 1) {
        qpc = 1;
    }
    groupNum = (tokenNum + qpc - 1) / qpc;
    if (groupNum < 1) {
        groupNum = 1;
    }
    if (groupNum > aiv) {
        groupNum = aiv;
    }
    return groupNum;
}

static void CalcCuBlockStatsFromDevice(const at::Tensor &cuBlockLens, int64_t &totalRows, int64_t &maxKv)
{
    TORCH_CHECK(cuBlockLens.dim() == DIM_ONE, "cu_block_lens must be 1-D");
    at::Tensor host = cuBlockLens.contiguous().to(at::kCPU);
    const int64_t n = host.numel();
    if (n <= 0) {
        totalRows = 0;
        maxKv = 0;
        return;
    }

    const int32_t *ptr = host.data_ptr<int32_t>();
    totalRows = static_cast<int64_t>(ptr[n - 1]);
    maxKv = 0;
    for (int64_t i = 0; i + 1 < n; ++i) {
        int64_t diff = static_cast<int64_t>(ptr[i + 1]) - static_cast<int64_t>(ptr[i]);
        if (diff > maxKv) {
            maxKv = diff;
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> k2q_csr(const at::Tensor &q2k, const at::Tensor &cuSeqlens,
                                                       const at::Tensor &cuBlockLens, int64_t orderMethod,
                                                       int64_t totalRows, int64_t maxKv, int64_t useSimt,
                                                       int64_t qGlobalOffset)
{
    TORCH_CHECK(
        torch_npu::utils::is_npu(q2k) && torch_npu::utils::is_npu(cuSeqlens) && torch_npu::utils::is_npu(cuBlockLens),
        "k2q_csr: q2k / cu_seqlens / cu_block_lens must be on NPU");
    TORCH_CHECK(q2k.scalar_type() == at::kInt, "q2k must be int32");
    TORCH_CHECK(cuSeqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(cuBlockLens.scalar_type() == at::kInt, "cu_block_lens must be int32");
    TORCH_CHECK(q2k.dim() == DIM_THREE, "q2k must be 3-D [H, T, topk], got dim=", q2k.dim());
    TORCH_CHECK(orderMethod == 0 || orderMethod == 1, "order_method must be 0 or 1, got ", orderMethod);

    if (totalRows < 0 || maxKv < 0) {
        CalcCuBlockStatsFromDevice(cuBlockLens, totalRows, maxKv);
    }
    TORCH_CHECK(totalRows >= 0, "total_rows must be >= 0, got ", totalRows);
    TORCH_CHECK(maxKv >= 0, "max_kv must be >= 0, got ", maxKv);

    const int64_t useSimtI = (useSimt != 0) ? 1 : 0;
    const int64_t qGlobalOffsetI = (qGlobalOffset != 0) ? 1 : 0;

    const int64_t numHeads = q2k.size(0);
    const int64_t numTokens = q2k.size(1);
    const int64_t topk = q2k.size(2);
    const int64_t batch = cuBlockLens.numel() > 0 ? cuBlockLens.numel() - 1 : 0;
    auto opts = q2k.options().dtype(at::kInt);

    at::Tensor q2kContig = q2k.contiguous();
    at::Tensor cuQ = cuSeqlens.contiguous();
    at::Tensor cuB = cuBlockLens.contiguous();

    const int64_t aiv = GetAivNumFromDevice();
    int64_t groupNum = 1;
    if (useSimtI != 0) {
        groupNum = GroupCount(numTokens, aiv);
    } else {
        groupNum = aiv;
        if (numTokens > 0 && numTokens < groupNum) {
            groupNum = numTokens;
        }
        if (groupNum < 1) {
            groupNum = 1;
        }
    }

    int64_t meta = std::max<int64_t>(batch, 0) * std::max<int64_t>(maxKv, 0) + std::max<int64_t>(numTokens, 0);
    int64_t ghr = groupNum * std::max<int64_t>(numHeads, 1) * std::max<int64_t>(totalRows, 1);
    int64_t hist = 2 * ghr + std::max<int64_t>(numHeads, 1) * std::max<int64_t>(totalRows, 1);
    int64_t scratchElems = std::max<int64_t>(meta + hist, 1);

    at::Tensor scratch{nullptr};
    at::Tensor rowPtr{nullptr};
    at::Tensor qInd{nullptr};
    at::Tensor slot{nullptr};
    {
        const c10::OptionalDeviceGuard deviceGuard(c10::Device(q2k.device()));
        scratch = at::empty({scratchElems}, opts);

        ACLNN_CMD(aclnnK2qCsrMeta, cuQ, cuB, scratch, orderMethod, totalRows, maxKv, numHeads, numTokens, topk);

        rowPtr = at::empty({numHeads, totalRows + 1}, opts);
        rowPtr.zero_();

        ACLNN_CMD(aclnnK2qCsrHist, q2kContig, scratch, totalRows, maxKv, useSimtI, batch);
        ACLNN_CMD(aclnnK2qCsrRowPrefix, scratch, totalRows, maxKv, useSimtI, numHeads, numTokens, topk, batch, rowPtr);
        ACLNN_CMD(aclnnK2qCsrTilePrefix, scratch, rowPtr, totalRows, maxKv, useSimtI, numHeads, numTokens, topk, batch);

        qInd = at::empty({numHeads, numTokens * topk}, opts);
        slot = at::empty({numHeads, numTokens * topk}, opts);
        qInd.fill_(-1);
        slot.fill_(-1);

        ACLNN_CMD(aclnnK2qCsrScatter, q2kContig, cuQ, scratch, totalRows, maxKv, useSimtI, qGlobalOffsetI, qInd, slot);
    }

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(rowPtr, qInd, slot);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("k2q_csr", &k2q_csr, "k2q_csr");
}

} // namespace op_api
