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
 * \file mhc_pre_backward_tiling.cpp
 * \brief
 */
#include "mhc_pre_backward_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;
using MmConfig = MhcPreBackward::MhcPreBackwardMmConfig0;
static_assert(MHC_PRE_BACKWARD_MM_CONFIG_0 == MmConfig::KEY, "host and kernel MM config keys must match");
const constexpr int64_t BSD_DIM_NUM = 3;
const constexpr int64_t BSNN_DIM_NUM = 4;
const constexpr int64_t TD_DIM_NUM = 2;
const constexpr int64_t TN_DIM_NUM = 2;
const constexpr int64_t TNN_DIM_NUM = 3;
const constexpr uint32_t GRAD_H_IN_INDEX = 3;
const constexpr uint32_t GRAD_H_POST_INDEX = 4;
const constexpr uint32_t GRAD_H_RES_INDEX = 5;

const constexpr int64_t INDEX_B = 0;
const constexpr int64_t INDEX_S = 1;
const constexpr int64_t INDEX_D = 2;

const constexpr int64_t INDEX_T = 0;
const constexpr int64_t INDEX_N = 1;
const constexpr int64_t INDEX_D_TND = 1;

const constexpr uint32_t MIN_D_LENGTH = 1;
const constexpr uint32_t MAX_D_LENGTH = 16384;
const constexpr uint32_t D_ALIGN = 64;

const constexpr uint32_t DEFAULT_TILING_PRIORITY = 1000;
const constexpr uint32_t LEGAL_N_VALUES[] = {4, 6, 8};
const constexpr uint32_t LEGAL_N_COUNT = 3;
const constexpr float DEFAULT_HC_EPS = 1e-6f;
const constexpr int64_t IMPL_MODE_FP32 = 0;
const constexpr int64_t IMPL_MODE_HF32 = 1;
const constexpr uint32_t IMPL_MODE_ATTR_INDEX = 1;

const constexpr uint32_t ALPHA_GRAD_CORE_FACTOR = 24;
const constexpr uint32_t BUFFER_NUM = 2;
const constexpr uint32_t EXTRA_BUFFER_SIZE = 2 * 1024 * 1024;
const constexpr uint32_t WORKSPACE_ALIGN_SIZE = 32;
const constexpr uint64_t SYSTEM_WORKSPACE_SIZE = 40 * 1024 * 1024;
const constexpr int32_t SCHEDULE_MODE = 1;

const constexpr uint32_t C0_SET_SHAPE_M = MmConfig::C0_M_BLOCK;
const constexpr uint32_t C0_SET_SHAPE_N = MmConfig::C0_N_BLOCK;

REGISTER_OPS_TILING_TEMPLATE(MhcPreBackward, MhcPreBackwardTiling, DEFAULT_TILING_PRIORITY);

ge::graphStatus MhcPreBackwardTiling::GetPlatformInfo()
{
    const auto *compileInfo = context_->GetCompileInfo<MhcPreBackwardCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "get compile info failed"),
                return ge::GRAPH_FAILED);

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "get platform info failed"),
                return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);

    uint64_t aicNum = compileInfo->aicNum != 0U ? compileInfo->aicNum : ascendcPlatform.GetCoreNumAic();
    uint64_t aivNum = compileInfo->aivNum != 0U ? compileInfo->aivNum : ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0U || aivNum == 0U || aicNum > UINT32_MAX || aivNum > UINT32_MAX,
                OP_LOGE(context_->GetNodeName(), "invalid core count, aic=%lu, aiv=%lu", aicNum, aivNum),
                return ge::GRAPH_FAILED);
    blockDim_ = static_cast<uint32_t>(aicNum);
    vecCoreNum_ = static_cast<uint32_t>(aivNum);

    auto getMemorySize = [&ascendcPlatform](uint64_t parsedSize, platform_ascendc::CoreMemType type) {
        if (parsedSize != 0U) {
            return parsedSize;
        }
        uint64_t platformSize = 0U;
        ascendcPlatform.GetCoreMemSize(type, platformSize);
        return platformSize;
    };
    ubSize_ = getMemorySize(compileInfo->ubSize, platform_ascendc::CoreMemType::UB);
    l1Size_ = getMemorySize(compileInfo->l1Size, platform_ascendc::CoreMemType::L1);
    l2Size_ = getMemorySize(compileInfo->l2Size, platform_ascendc::CoreMemType::L2);
    l0ASize_ = getMemorySize(compileInfo->l0ASize, platform_ascendc::CoreMemType::L0_A);
    l0BSize_ = getMemorySize(compileInfo->l0BSize, platform_ascendc::CoreMemType::L0_B);
    l0CSize_ = getMemorySize(compileInfo->l0CSize, platform_ascendc::CoreMemType::L0_C);
    OP_CHECK_IF(ubSize_ == 0U || l1Size_ == 0U || l0ASize_ == 0U || l0BSize_ == 0U || l0CSize_ == 0U,
                OP_LOGE(context_->GetNodeName(),
                        "invalid platform memory, ub=%lu, l1=%lu, l0a=%lu, l0b=%lu, l0c=%lu", ubSize_, l1Size_,
                        l0ASize_, l0BSize_, l0CSize_),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(),
            "platform info: aic=%u, aiv=%u, ub=%lu, l1=%lu, l2=%lu, l0a=%lu, l0b=%lu, l0c=%lu", blockDim_,
            vecCoreNum_, ubSize_, l1Size_, l2Size_, l0ASize_, l0BSize_, l0CSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::GetInputTensors(const gert::Tensor *&gradHInTensor,
                                                      const gert::Tensor *&gradHPostTensor,
                                                      const gert::Tensor *&gradHResTensor)
{
    gradHInTensor = context_->GetDynamicInputTensor(GRAD_H_IN_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradHInTensor);
    gradHPostTensor = context_->GetDynamicInputTensor(GRAD_H_POST_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradHPostTensor);
    gradHResTensor = context_->GetDynamicInputTensor(GRAD_H_RES_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradHResTensor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ValidateInputDims(int64_t gradHInDims, int64_t gradHPostDims,
                                                        int64_t gradHResDims)
{
    if ((gradHInDims != BSD_DIM_NUM && gradHInDims != TD_DIM_NUM) ||
        (gradHPostDims != BSD_DIM_NUM && gradHPostDims != TN_DIM_NUM) ||
        (gradHResDims != BSNN_DIM_NUM && gradHResDims != TNN_DIM_NUM)) {
        OP_LOGE(context_->GetNodeName(),
                "input dims invalid for MhcPreBackward, gradHInDims=%ld (expected %ld or %ld), gradHPostDims=%ld "
                "(expected %ld or %ld), gradHResDims=%ld (expected %ld or %ld)",
                gradHInDims, BSD_DIM_NUM, TD_DIM_NUM, gradHPostDims, BSD_DIM_NUM, TN_DIM_NUM, gradHResDims,
                BSNN_DIM_NUM, TNN_DIM_NUM);
        return ge::GRAPH_FAILED;
    }
    if (gradHInDims != gradHPostDims) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_in and grad_h_post must have the same dim num, gradHInDims=%ld, gradHPostDims=%ld", gradHInDims,
                gradHPostDims);
        return ge::GRAPH_FAILED;
    }
    if ((gradHInDims == BSD_DIM_NUM && gradHResDims != BSNN_DIM_NUM) ||
        (gradHInDims == TD_DIM_NUM && gradHResDims != TNN_DIM_NUM)) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_in and grad_h_res dims mismatch, gradHInDims=%ld requires gradHResDims=%ld, but got %ld",
                gradHInDims, gradHInDims == BSD_DIM_NUM ? BSNN_DIM_NUM : TNN_DIM_NUM, gradHResDims);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ParseBSDFormat(const gert::Tensor *gradHInTensor,
                                                     const gert::Tensor *gradHPostTensor,
                                                     const gert::Tensor *gradHResTensor)
{
    uint64_t batch = gradHInTensor->GetStorageShape().GetDim(INDEX_B);
    uint64_t sequence = gradHInTensor->GetStorageShape().GetDim(INDEX_S);
    D_ = gradHInTensor->GetStorageShape().GetDim(INDEX_D);
    N_ = gradHPostTensor->GetStorageShape().GetDim(INDEX_D);

    if (gradHPostTensor->GetStorageShape().GetDim(INDEX_B) != batch ||
        gradHPostTensor->GetStorageShape().GetDim(INDEX_S) != sequence) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_post shape must align with grad_h_in on B and S dims, grad_h_post[B,S]=[%ld,%ld], expected "
                "[%lu,%lu]",
                gradHPostTensor->GetStorageShape().GetDim(INDEX_B), gradHPostTensor->GetStorageShape().GetDim(INDEX_S),
                batch, sequence);
        return ge::GRAPH_FAILED;
    }
    if (gradHResTensor->GetStorageShape().GetDim(0) != batch ||
        gradHResTensor->GetStorageShape().GetDim(1) != sequence || gradHResTensor->GetStorageShape().GetDim(2) != N_ ||
        gradHResTensor->GetStorageShape().GetDim(3) != N_) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_res shape must be [B, S, N, N], actual=[%ld,%ld,%ld,%ld], expected=[%lu,%lu,%lu,%lu]",
                gradHResTensor->GetStorageShape().GetDim(0), gradHResTensor->GetStorageShape().GetDim(1),
                gradHResTensor->GetStorageShape().GetDim(2), gradHResTensor->GetStorageShape().GetDim(3), batch,
                sequence, N_, N_);
        return ge::GRAPH_FAILED;
    }
    totalLength_ = batch * sequence;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ParseTNDFormat(const gert::Tensor *gradHInTensor,
                                                     const gert::Tensor *gradHPostTensor,
                                                     const gert::Tensor *gradHResTensor)
{
    uint64_t t = gradHInTensor->GetStorageShape().GetDim(INDEX_T);
    D_ = gradHInTensor->GetStorageShape().GetDim(INDEX_D_TND);
    N_ = gradHPostTensor->GetStorageShape().GetDim(INDEX_N);

    if (gradHPostTensor->GetStorageShape().GetDim(INDEX_T) != t) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_post shape must align with grad_h_in on T dim, grad_h_post[T]=%ld, expected %lu",
                gradHPostTensor->GetStorageShape().GetDim(INDEX_T), t);
        return ge::GRAPH_FAILED;
    }
    if (gradHResTensor->GetStorageShape().GetDim(0) != t || gradHResTensor->GetStorageShape().GetDim(1) != N_ ||
        gradHResTensor->GetStorageShape().GetDim(2) != N_) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_res shape must be [T, N, N], actual=[%ld,%ld,%ld], expected=[%lu,%lu,%lu]",
                gradHResTensor->GetStorageShape().GetDim(0), gradHResTensor->GetStorageShape().GetDim(1),
                gradHResTensor->GetStorageShape().GetDim(2), t, N_, N_);
        return ge::GRAPH_FAILED;
    }
    totalLength_ = t;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ValidateShapeParams()
{
    static const std::vector<uint64_t> legalN(LEGAL_N_VALUES, LEGAL_N_VALUES + LEGAL_N_COUNT);
    if (std::find(legalN.begin(), legalN.end(), N_) == legalN.end()) {
        OP_LOGE(context_->GetNodeName(), "Invalid input shape N=%lu. Expected one of {4,6,8}", N_);
        return ge::GRAPH_FAILED;
    }
    if (D_ < MIN_D_LENGTH || D_ > MAX_D_LENGTH || D_ % D_ALIGN != 0) {
        OP_LOGE(context_->GetNodeName(),
                "Invalid input shape D=%lu. Expected to be in [%u, %u] and %u-element alignment", D_, MIN_D_LENGTH,
                MAX_D_LENGTH, D_ALIGN);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ValidateMmConfig()
{
    constexpr uint64_t cubeAlign = 16U;
    constexpr uint64_t elementSize = sizeof(float);
    auto alignUp = [](uint64_t value, uint64_t align) { return (value + align - 1U) / align * align; };
    auto maxValue = [](uint64_t lhs, uint64_t rhs) { return lhs > rhs ? lhs : rhs; };

    uint64_t l1Bytes =
        (MmConfig::L1_A_BUFFER_DEPTH + MmConfig::L1_B_BUFFER_DEPTH) * MmConfig::L1_SLOT_BYTES;
    uint64_t l0ABytes = MmConfig::L0A_BUFFER_DEPTH * MmConfig::L0A_SLOT_BYTES;
    uint64_t l0BBytes = MmConfig::L0B_BUFFER_DEPTH * MmConfig::L0B_SLOT_BYTES;
    uint64_t l0CBytes = MmConfig::L0C_BUFFER_DEPTH * MmConfig::L0C_SLOT_BYTES;
    OP_CHECK_IF(l1Bytes > l1Size_ || l0ABytes > l0ASize_ || l0BBytes > l0BSize_ || l0CBytes > l0CSize_,
                OP_LOGE(context_->GetNodeName(),
                        "MM config %u exceeds platform memory: L1=%lu/%lu, L0A=%lu/%lu, L0B=%lu/%lu, "
                        "L0C=%lu/%lu",
                        MmConfig::KEY, l1Bytes, l1Size_, l0ABytes, l0ASize_, l0BBytes, l0BSize_, l0CBytes,
                        l0CSize_),
                return ge::GRAPH_FAILED);

    bool cubeShapeAligned = MmConfig::C0_BASE_M % cubeAlign == 0U && MmConfig::C0_BASE_N % cubeAlign == 0U &&
                            MmConfig::C0_BASE_K % cubeAlign == 0U && MmConfig::C1_BASE_M % cubeAlign == 0U &&
                            MmConfig::C1_BASE_N % cubeAlign == 0U && MmConfig::C1_BASE_K % cubeAlign == 0U;
    bool kSplitValid = MmConfig::C0_L1_K >= MmConfig::C0_BASE_K &&
                       MmConfig::C0_L1_K % MmConfig::C0_BASE_K == 0U &&
                       MmConfig::C1_L1_K >= MmConfig::C1_BASE_K &&
                       MmConfig::C1_L1_K % MmConfig::C1_BASE_K == 0U;
    bool depthValid = MmConfig::L1_A_BUFFER_DEPTH == MmConfig::L1_B_BUFFER_DEPTH &&
                      MmConfig::L1_A_BUFFER_DEPTH == MmConfig::L0A_BUFFER_DEPTH &&
                      MmConfig::L0A_BUFFER_DEPTH == MmConfig::L0B_BUFFER_DEPTH &&
                      MmConfig::L0A_BUFFER_DEPTH == MmConfig::L0C_BUFFER_DEPTH;
    OP_CHECK_IF(!cubeShapeAligned || !kSplitValid || !depthValid,
                OP_LOGE(context_->GetNodeName(), "MM config %u has invalid alignment, K split or buffer depth",
                        MmConfig::KEY),
                return ge::GRAPH_FAILED);

    uint64_t maxBaseM = maxValue(MmConfig::C0_BASE_M, MmConfig::C1_BASE_M);
    uint64_t maxBaseN = maxValue(MmConfig::C0_BASE_N, MmConfig::C1_BASE_N);
    uint64_t maxBaseK = maxValue(MmConfig::C0_BASE_K, MmConfig::C1_BASE_K);
    uint64_t maxL1K = maxValue(MmConfig::C0_L1_K, MmConfig::C1_L1_K);
    uint64_t a1SlotBytes = alignUp(maxBaseM, cubeAlign) * alignUp(maxL1K, cubeAlign) * elementSize;
    uint64_t b1SlotBytes = alignUp(maxBaseN, cubeAlign) * alignUp(maxL1K, cubeAlign) * elementSize;
    uint64_t a2SlotBytes = alignUp(maxBaseM, cubeAlign) * alignUp(maxBaseK, cubeAlign) * elementSize;
    uint64_t b2SlotBytes = alignUp(maxBaseN, cubeAlign) * alignUp(maxBaseK, cubeAlign) * elementSize;
    uint64_t c0SlotBytes = alignUp(maxBaseM, cubeAlign) * alignUp(maxBaseN, cubeAlign) * elementSize;
    OP_CHECK_IF(a1SlotBytes > MmConfig::L1_SLOT_BYTES || b1SlotBytes > MmConfig::L1_SLOT_BYTES ||
                    a2SlotBytes > MmConfig::L0A_SLOT_BYTES || b2SlotBytes > MmConfig::L0B_SLOT_BYTES ||
                    c0SlotBytes > MmConfig::L0C_SLOT_BYTES,
                OP_LOGE(context_->GetNodeName(), "MM config %u slot capacity is insufficient", MmConfig::KEY),
                return ge::GRAPH_FAILED);

    uint64_t c0B1ChunkCount = (fusionSize_ + MmConfig::C0_L1_K - 1U) / MmConfig::C0_L1_K;
    uint64_t c0B1Bytes = 0U;
    for (uint64_t offsetK = 0U; offsetK < fusionSize_; offsetK += MmConfig::C0_L1_K) {
        uint64_t currentK = fusionSize_ - offsetK;
        currentK = currentK < MmConfig::C0_L1_K ? currentK : MmConfig::C0_L1_K;
        c0B1Bytes += alignUp(currentK, cubeAlign) * alignUp(MmConfig::C0_N_BLOCK, cubeAlign) * elementSize;
    }
    bool c0B1Valid = MmConfig::C0_B1_FULL_LOAD == 1U &&
                     c0B1ChunkCount <= MmConfig::L1_B_BUFFER_DEPTH &&
                     c0B1Bytes <= MmConfig::L1_B_BUFFER_DEPTH * MmConfig::L1_SLOT_BYTES;
    bool pipelineValid = MmConfig::C0_TO_V2_ROWS == MmConfig::C0_BASE_M &&
                         MmConfig::V2_TO_C1_ROWS == MmConfig::C1_K_BLOCK &&
                         MmConfig::C1_K_BLOCK == MmConfig::C0_M_BLOCK &&
                         MmConfig::C0_M_BLOCK % MmConfig::C0_TO_V2_ROWS == 0U &&
                         MmConfig::C0_M_BLOCK % MmConfig::V2_CHUNK_ROWS == 0U;
    OP_CHECK_IF(!c0B1Valid || !pipelineValid,
                OP_LOGE(context_->GetNodeName(), "MM config %u has invalid C0 B1 residency or pipeline granularity",
                        MmConfig::KEY),
                return ge::GRAPH_FAILED);
    mmConfigKey_ = MmConfig::KEY;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::GetInputShape()
{
    const gert::Tensor *gradHInTensor = nullptr;
    const gert::Tensor *gradHPostTensor = nullptr;
    const gert::Tensor *gradHResTensor = nullptr;

    auto ret = GetInputTensors(gradHInTensor, gradHPostTensor, gradHResTensor);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto gradHInDims = gradHInTensor->GetStorageShape().GetDimNum();
    auto gradHPostDims = gradHPostTensor->GetStorageShape().GetDimNum();
    auto gradHResDims = gradHResTensor->GetStorageShape().GetDimNum();

    ret = ValidateInputDims(gradHInDims, gradHPostDims, gradHResDims);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (gradHInDims == BSD_DIM_NUM) {
        ret = ParseBSDFormat(gradHInTensor, gradHPostTensor, gradHResTensor);
    } else if (gradHInDims == TD_DIM_NUM) {
        ret = ParseTNDFormat(gradHInTensor, gradHPostTensor, gradHResTensor);
    }
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ValidateShapeParams();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    fusionSize_ = (2 * N_) + (N_ * N_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ParseInputAndAttr()
{
    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "get input shape failed");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get attrs failed");
        return ge::GRAPH_FAILED;
    }
    auto hcEpsPtr = attrs->GetAttrPointer<float>(0);
    hcEps_ = (hcEpsPtr != nullptr) ? *hcEpsPtr : DEFAULT_HC_EPS;

    auto implModePtr = attrs->GetAttrPointer<int64_t>(IMPL_MODE_ATTR_INDEX);
    int64_t implMode = (implModePtr != nullptr) ? *implModePtr : IMPL_MODE_FP32;
    if (implMode != IMPL_MODE_FP32 && implMode != IMPL_MODE_HF32) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "op_impl_mode", std::to_string(implMode).c_str(),
                                              "op_impl_mode must be 0 (FP32) or 1 (HF32)");
        return ge::GRAPH_FAILED;
    }
    implMode_ = static_cast<uint32_t>(implMode);

    return ge::GRAPH_SUCCESS;
}

void MhcPreBackwardTiling::SetCommonTilingParams()
{
    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_vecCoreNum(vecCoreNum_);
    tilingData_.set_totalLength(totalLength_);
    tilingData_.set_nD(N_ * D_);
    tilingData_.set_fusionSize(fusionSize_);
    tilingData_.set_N(N_);
    tilingData_.set_D(D_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_implMode(implMode_);
    tilingData_.set_mmConfigKey(mmConfigKey_);
    tilingData_.set_c0BaseM(MmConfig::C0_BASE_M);
    tilingData_.set_c0BaseN(MmConfig::C0_BASE_N);
    tilingData_.set_c0BaseK(MmConfig::C0_BASE_K);
    tilingData_.set_c1BaseM(MmConfig::C1_BASE_M);
    tilingData_.set_c1BaseN(MmConfig::C1_BASE_N);
    tilingData_.set_c1BaseK(MmConfig::C1_BASE_K);
    tilingData_.set_c0L1K(MmConfig::C0_L1_K);
    tilingData_.set_c1L1K(MmConfig::C1_L1_K);
    tilingData_.set_l1ADepth(MmConfig::L1_A_BUFFER_DEPTH);
    tilingData_.set_l1BDepth(MmConfig::L1_B_BUFFER_DEPTH);
    tilingData_.set_l0ADepth(MmConfig::L0A_BUFFER_DEPTH);
    tilingData_.set_l0BDepth(MmConfig::L0B_BUFFER_DEPTH);
    tilingData_.set_l0CDepth(MmConfig::L0C_BUFFER_DEPTH);
    tilingData_.set_c0B1FullLoad(MmConfig::C0_B1_FULL_LOAD);
    tilingData_.set_c0MBlock(MmConfig::C0_M_BLOCK);
    tilingData_.set_c0NBlock(MmConfig::C0_N_BLOCK);
    tilingData_.set_c1KBlock(MmConfig::C1_K_BLOCK);
    tilingData_.set_c0ToV2Rows(MmConfig::C0_TO_V2_ROWS);
    tilingData_.set_v2ToC1Rows(MmConfig::V2_TO_C1_ROWS);
}

void MhcPreBackwardTiling::FillTilingData()
{
    SetCommonTilingParams();
}

uint64_t MhcPreBackwardTiling::CalculateWorkspaceSize(uint64_t totalLength, uint64_t fusionSize, uint64_t cubeCoreNum,
                                                      uint64_t vecCoreNum, uint64_t elementSize)
{
    uint64_t v1Elements = totalLength * fusionSize +            // h_mix_grad
                          ALPHA_GRAD_CORE_FACTOR * vecCoreNum + // alpha_grad
                          vecCoreNum * fusionSize +             // bias_grad
                          totalLength;                          // inv_rms_grad

    uint64_t v2Elements = C0_SET_SHAPE_M * C0_SET_SHAPE_N * cubeCoreNum * BUFFER_NUM + // x_rs_grad_mm
                          C0_SET_SHAPE_M * C0_SET_SHAPE_N * cubeCoreNum * BUFFER_NUM + // x_rs
                          EXTRA_BUFFER_SIZE;
    uint64_t totalElements =
        ((v1Elements + WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE * WORKSPACE_ALIGN_SIZE) + v2Elements;
    return totalElements * elementSize;
}

ge::graphStatus MhcPreBackwardTiling::TilingProcess()
{
    size_t userWorkspaceSize = CalculateWorkspaceSize(totalLength_, fusionSize_, blockDim_, vecCoreNum_, sizeof(float));
    size_t systemWorkspaceSize = SYSTEM_WORKSPACE_SIZE;
    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::DoOpTiling()
{
    auto inputXDesc = context_->GetInputDesc(0);
    if (inputXDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "invalid input pointer: x");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ValidateMmConfig() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

void MhcPreBackwardTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "blockDim: [%u]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "totalLength: [%lu]", tilingData_.get_totalLength());
    OP_LOGD(context_->GetNodeName(), "nD: [%lu]", tilingData_.get_nD());
    OP_LOGD(context_->GetNodeName(), "fusionSize: [%lu]", tilingData_.get_fusionSize());
    OP_LOGD(context_->GetNodeName(), "hcEps: [%f]", tilingData_.get_hcEps());
    OP_LOGD(context_->GetNodeName(), "implMode: [%u]", tilingData_.get_implMode());
    OP_LOGD(context_->GetNodeName(),
            "mmConfig: key=%u, C0=[%u,%u,%u,L1K=%u], C1=[%u,%u,%u,L1K=%u], depth=[%u,%u,%u,%u,%u], "
            "block=[%u,%u,%u], sync=[%u,%u]",
            tilingData_.get_mmConfigKey(), tilingData_.get_c0BaseM(), tilingData_.get_c0BaseN(),
            tilingData_.get_c0BaseK(), tilingData_.get_c0L1K(), tilingData_.get_c1BaseM(),
            tilingData_.get_c1BaseN(), tilingData_.get_c1BaseK(), tilingData_.get_c1L1K(),
            tilingData_.get_l1ADepth(), tilingData_.get_l1BDepth(), tilingData_.get_l0ADepth(),
            tilingData_.get_l0BDepth(), tilingData_.get_l0CDepth(), tilingData_.get_c0MBlock(),
            tilingData_.get_c0NBlock(), tilingData_.get_c1KBlock(), tilingData_.get_c0ToV2Rows(),
            tilingData_.get_v2ToC1Rows());
}

uint64_t MhcPreBackwardTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(mmConfigKey_));
}

ge::graphStatus MhcPreBackwardTiling::PostTiling()
{
    OP_CHECK_IF(
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(SCHEDULE_MODE);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);

    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
