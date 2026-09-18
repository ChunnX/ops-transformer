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
 * \file moe_tutel_combine_x_tiling_arch35.cpp
 * \brief MoeTutelCombineX tiling implementation for arch35 (Ascend 950).
 */
#include "moe_tutel_combine_x_tiling_arch35.h"

#include <algorithm>
#include <cstring>

namespace optiling {

// Design-time UB budget cap (bytes). perBufBytes/hChunk are derived from this
// budget capped by the platform's actual UB (ubSize_), so TilingData stays
// deterministic on current silicon while never exceeding the real on-chip UB.
constexpr int64_t UB_BUDGET_CAP = 245760; // bytes
constexpr int64_t P = 3;

// APT 框架系统保留 workspace（launch 参数契约，非算子算法 workspace）：
// ascend950（arch35）的 RESERVED_WORKSPACE = 16 MiB，自动生成的 kernel 入口
// wrapper 无条件前切该保留区 —— TilingFunc 必须按 tiling 输出契约声明同等尺寸，
// 否则裸 launch 的实参表比 kernel 形参少一个指针，workspace 形参错位。
constexpr uint64_t kSysReservedWorkspaceSize = 16ULL * 1024 * 1024;

MoeTutelCombineXTilingArch35::MoeTutelCombineXTilingArch35(gert::TilingContext *context)
    : Ops::Transformer::OpTiling::TilingBaseClass(context)
{}

bool MoeTutelCombineXTilingArch35::IsCapable()
{
    return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
}

ge::graphStatus MoeTutelCombineXTilingArch35::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTilingArch35::GetShapeAttrsInfo()
{
    // Dims are either >= 0 (0 allowed for empty tensors) or -1 (unknown) —
    // the same lower bound the graph-mode InferShape enforces; anything below
    // -1 must be rejected here instead of flowing into TilingData.
    auto ValidateDims = [](const gert::Shape &s) -> bool {
        for (size_t d = 0; d < s.GetDimNum(); ++d) {
            if (s.GetDim(d) < -1) {
                return false;
            }
        }
        return true;
    };
    // Known-dim equality (unknown -1 dims tolerated), mirroring the InferShape
    // metadata shape check.
    auto DimsMatch = [](const gert::Shape &a, const gert::Shape &b) -> bool {
        if (a.GetDimNum() != b.GetDimNum()) {
            return false;
        }
        for (size_t d = 0; d < a.GetDimNum(); ++d) {
            const int64_t av = a.GetDim(d);
            const int64_t bv = b.GetDim(d);
            if (av >= 0 && bv >= 0 && av != bv) {
                return false;
            }
        }
        return true;
    };

    auto ygShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ygShape);
    {
        gert::Shape s = ygShape->GetStorageShape();
        rankYGrad_ = static_cast<int64_t>(s.GetDimNum());
        if (rankYGrad_ == 2) {
            eC_ = s.GetDim(0);
            hidden_ = s.GetDim(1);
            batchSize_ = 1;
        } else if (rankYGrad_ == 3) {
            batchSize_ = s.GetDim(0);
            eC_ = s.GetDim(1);
            hidden_ = s.GetDim(2);
        } else {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "y_grad",
                                                     "rank must be 2 or 3, got " + std::to_string(rankYGrad_));
            return ge::GRAPH_FAILED;
        }
        if (!ValidateDims(s)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "y_grad", "storage dims",
                                                  "must be >= -1 (-1 marks an unknown dim)");
            return ge::GRAPH_FAILED;
        }
    }

    auto gatesShape = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gatesShape);
    int64_t gatesBatchDim = 0;
    {
        gert::Shape s = gatesShape->GetStorageShape();
        int64_t rankGates = static_cast<int64_t>(s.GetDimNum());
        if (rankGates == 1) {
            samples_ = s.GetDim(0);
        } else if (rankGates == 2) {
            gatesBatchDim = s.GetDim(0);
            samples_ = s.GetDim(1);
        } else {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "gates",
                                                     "rank must be 1 or 2, got " + std::to_string(rankGates));
            return ge::GRAPH_FAILED;
        }
        if (!ValidateDims(s)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "gates", "storage dims",
                                                  "must be >= -1 (-1 marks an unknown dim)");
            return ge::GRAPH_FAILED;
        }
    }

    // Batch form: the micro-batch dim must agree (the same contract the
    // graph-mode InferShape enforces).
    if (rankYGrad_ == 3 && gatesBatchDim >= 0 && batchSize_ >= 0 && gatesBatchDim != batchSize_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "y_grad, gates",
                                              std::to_string(batchSize_) + ", " + std::to_string(gatesBatchDim),
                                              "batch dims must agree when rank(y_grad) is 3");
        return ge::GRAPH_FAILED;
    }

    auto indicesShape = context_->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShape);
    auto locationsShape = context_->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsShape);
    {
        gert::Shape gi = indicesShape->GetStorageShape();
        gert::Shape gl = locationsShape->GetStorageShape();
        gert::Shape gg = gatesShape->GetStorageShape();
        if (!ValidateDims(gi) || !ValidateDims(gl)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "indices, locations", "storage dims",
                                                  "must be >= -1 (-1 marks an unknown dim)");
            return ge::GRAPH_FAILED;
        }
        if (!DimsMatch(gg, gi) || !DimsMatch(gg, gl)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "gates/indices/locations",
                                                     "shapes must match");
            return ge::GRAPH_FAILED;
        }
    }

    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        dtypeSize_ = 2;
    } else if (dtype == ge::DT_FLOAT) {
        dtypeSize_ = 4;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y_grad",
                                              std::to_string(static_cast<int>(dtype)), "must be FP16/FP32/BF16");
        return ge::GRAPH_FAILED;
    }
    // Independent dtype gate for the metadata inputs (the same contract the
    // graph-mode InferDataType enforces): gates shares the y_grad dtype and
    // indices/locations are INT32.
    {
        auto gatesDesc = context_->GetInputDesc(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, gatesDesc);
        if (gatesDesc->GetDataType() != dtype) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "gates",
                                                  std::to_string(static_cast<int>(gatesDesc->GetDataType())),
                                                  "must match y_grad dtype");
            return ge::GRAPH_FAILED;
        }
        auto indicesDesc = context_->GetInputDesc(2);
        OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
        auto locationsDesc = context_->GetInputDesc(3);
        OP_CHECK_NULL_WITH_CONTEXT(context_, locationsDesc);
        if (indicesDesc->GetDataType() != ge::DT_INT32 || locationsDesc->GetDataType() != ge::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "indices, locations",
                                                   std::to_string(static_cast<int>(indicesDesc->GetDataType())) + ", " +
                                                       std::to_string(static_cast<int>(locationsDesc->GetDataType())),
                                                   "must be INT32");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTilingArch35::DeriveCapacity()
{
    // The kernel also derives capacity from GM indices, so we can safely set
    // capacity = eC (E=1) here when indices data is not host-accessible.
    const gert::Tensor *indicesTensor = context_->GetInputTensor(2);
    if (indicesTensor == nullptr) {
        // No tensor object available (e.g. data-less host tiling / unit test):
        // fall back to E=1 and let the kernel correct the capacity from GM.
        OP_LOGI(context_, "indices tensor not available, setting capacity=eC (kernel will correct)");
        E_ = 1;
        capacity_ = (E_ > 0) ? (eC_ / E_) : 0;
        return ge::GRAPH_SUCCESS;
    }
    gert::TensorPlacement placement = indicesTensor->GetTensorData().GetPlacement();
    bool hostData = (placement == gert::kOnHost) || (placement == gert::kFollowing);
    const int32_t *indicesData = hostData ? indicesTensor->GetData<int32_t>() : nullptr;
    size_t indicesSize = hostData ? indicesTensor->GetSize() : 0;
    if (indicesData == nullptr || indicesSize == 0) {
        OP_LOGI(context_, "indices data not available, setting capacity=eC (kernel will correct)");
        E_ = 1;
    } else {
        int64_t totalIndices = 1;
        auto indicesShape = context_->GetInputShape(2);
        gert::Shape s = indicesShape->GetStorageShape();
        for (size_t d = 0; d < s.GetDimNum(); ++d) {
            totalIndices *= s.GetDim(d);
        }
        int64_t maxIdx = -1;
        for (int64_t i = 0; i < totalIndices; i++) {
            if (indicesData[i] > maxIdx) {
                maxIdx = indicesData[i];
            }
        }
        E_ = (maxIdx >= 0) ? (maxIdx + 1) : 1;
    }

    capacity_ = (E_ > 0) ? (eC_ / E_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTilingArch35::HandleEmptyTensor()
{
    std::memset(&td_, 0, sizeof(MoetutelcombinexTilingData));
    // Empty-branch kernel zeroes the output x_grad so the op produces a
    // well-defined all-zero output even when the caller does not pre-zero the
    // buffer. Fill the shape fields the kernel needs to compute
    // outputElems = batchSize * samples * hidden.
    td_.batchSize = batchSize_;
    td_.samples = samples_;
    td_.hidden = hidden_;
    blockDim_ = 1;
    tilingKey_ = TILING_KEY_EMPTY;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTilingArch35::DoOpTiling()
{
    bool isEmptyTensor = (batchSize_ == 0) || (samples_ == 0) || (eC_ == 0) || (hidden_ == 0);
    if (isEmptyTensor) {
        return HandleEmptyTensor();
    }

    if (DeriveCapacity() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool hasBatch = (rankYGrad_ == 3);

    tokenTotal_ = batchSize_ * samples_;
    int64_t coreNumMax = static_cast<int64_t>(coreNum_);
    usedCoreNum_ = static_cast<int32_t>(std::min(coreNumMax, tokenTotal_));
    if (usedCoreNum_ <= 0) {
        // Defensive guard: usedCoreNum_==0 would make the token-split division
        // below a divide-by-zero UB. A non-empty input reaching here would
        // silently produce an all-zero x_grad and swallow the whole backward
        // gradient — reject instead.
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "usedCoreNum", std::to_string(usedCoreNum_),
                                              "must be > 0 for a non-empty input (coreNum=" + std::to_string(coreNum_) +
                                                  " tokenTotal=" + std::to_string(tokenTotal_) + ")");
        return ge::GRAPH_FAILED;
    }
    coreTokenBase_ = tokenTotal_ / usedCoreNum_;
    bigCoreCnt_ = static_cast<int32_t>(tokenTotal_ % usedCoreNum_);

    // UB budget: consume the platform's actual UB size (ubSize_) capped to the
    // design-time budget, then reserve the two 32-byte gate-scalar staging
    // buffers (gateTmp_/gateS_, allocated unconditionally by the kernel's
    // InitBuffer) so the kernel's peak UB footprint
    // 3*alignUp(hChunk,64)*sizeof(T) + 64 never exceeds the budget cap.
    int64_t ubAvailable = static_cast<int64_t>(ubSize_);
    if (ubAvailable <= 0) {
        ubAvailable = UB_BUDGET_CAP; // platform UB unavailable -> design budget
    }
    constexpr int64_t kGateScalarBufBytes = 64; // gateTmp_(32B) + gateS_(32B)
    int64_t ubForLines = std::min(ubAvailable, UB_BUDGET_CAP) - kGateScalarBufBytes;
    if (ubForLines < P * 64 * dtypeSize_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "ubSize", std::to_string(ubSize_),
            "too small: cannot fit three 64-element line buffers plus the gate staging buffers (min viable budget "
            "is " +
                std::to_string(P * 64 * dtypeSize_ + kGateScalarBufBytes) + " B)");
        return ge::GRAPH_FAILED;
    }
    // Per-buffer element budget aligned down to the kernel's 64-element
    // allocation granularity so alignUp(hChunk, 64) == hChunk and the byte
    // account stays exact: 3*hChunk*sizeof(T) + 64 <= min(ubSize, UB_BUDGET_CAP).
    int64_t perBufElems = (ubForLines / P) / dtypeSize_ / 64 * 64;
    // Branch-0 (no-batch) uniform chunk fixes hChunk = min(H, 20480), which is
    // the FP32 per-buffer element budget (at the design cap: (245760-64)/3/4
    // aligned down to 64 elements). Derived dynamically so it scales with the
    // budget.
    int64_t perBufElemsFp32 = (ubForLines / P) / 4 / 64 * 64;
    hChunk_ = hasBatch ? std::min(hidden_, std::max<int64_t>(1, perBufElems)) :
                         std::min(hidden_, std::max<int64_t>(1, perBufElemsFp32));
    hChunkCnt_ = Ops::Base::CeilDiv(hidden_, hChunk_);
    hChunkTail_ = hidden_ - (hChunkCnt_ - 1) * hChunk_;

    return FillAndSet(hasBatch);
}

ge::graphStatus MoeTutelCombineXTilingArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeTutelCombineXTilingArch35::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MoeTutelCombineXTilingArch35::GetWorkspaceSize()
{
    // APT 框架系统保留 workspace 契约声明（16 MiB）：kernel 入口 wrapper 无条件
    // 前切该保留区，tiling 必须声明同等尺寸以免实参表错位。算法自身 workspace 为 0。
    workspaceSize_ = kSysReservedWorkspaceSize;
    auto *wsPtr = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wsPtr);
    wsPtr[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTilingArch35::PostTiling()
{
    context_->SetBlockDim(blockDim_);

    auto tilingDataSize = sizeof(MoetutelcombinexTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&td_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTilingArch35::FillAndSet(bool hasBatch)
{
    std::memset(&td_, 0, sizeof(MoetutelcombinexTilingData));

    td_.batchSize = batchSize_;
    td_.samples = samples_;
    td_.hidden = hidden_;
    td_.capacity = capacity_;
    td_.eC = eC_;

    td_.tokenTotal = tokenTotal_;
    td_.coreTokenBase = coreTokenBase_;
    td_.bigCoreCnt = bigCoreCnt_;

    td_.hChunk = hChunk_;
    td_.hChunkCnt = hChunkCnt_;
    td_.hChunkTail = hChunkTail_;

    blockDim_ = static_cast<uint32_t>(usedCoreNum_);
    tilingKey_ = hasBatch ? TILING_KEY_NORMAL_BATCH : TILING_KEY_NORMAL_NO_BATCH;

    OP_LOGI(context_,
            "Tiling: B=%ld S=%ld H=%ld C=%ld eC=%ld cores=%u tokens=%ld base=%ld big=%d "
            "hChunk=%ld cnt=%ld tail=%ld key=%lu",
            batchSize_, samples_, hidden_, capacity_, eC_, blockDim_, tokenTotal_, coreTokenBase_, bigCoreCnt_, hChunk_,
            hChunkCnt_, hChunkTail_, tilingKey_);

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MoeTutelCombineX, MoeTutelCombineXTilingArch35,
                             1000); // If 950, use this tiling class.
} // namespace optiling
