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
 * \file quant_block_sparse_attn_tiling.cpp
 * \brief QuantBlockSparseAttn tiling: three-stage pipeline (Parse -> Check -> Tiling).
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include <tiling/platform/platform_ascendc.h>
#include "../op_kernel/quant_block_sparse_attn_template_tiling_key.h"
#include "quant_block_sparse_attn_tiling.h"
#include "quant_block_sparse_attn_check.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr uint32_t BSA_LAYOUT_Q_BSND_VALUE = 0U;
constexpr uint32_t BSA_LAYOUT_Q_TND_VALUE = 2U;
constexpr uint32_t BSA_LAYOUT_Q_NTD_VALUE = 5U;
constexpr uint32_t BSA_LAYOUT_KV_PA_BNSD_VALUE = 1U;
constexpr uint32_t BSA_LAYOUT_SPARSE_B_N_QB_KB_VALUE = 0U;
constexpr uint8_t BSA_LAYOUT_TYPE_BSH = 1U;
constexpr uint8_t BSA_LAYOUT_TYPE_TND = 2U;
constexpr uint8_t BSA_LAYOUT_TYPE_NTD = 5U;
constexpr uint8_t BSA_PA_LAYOUT_TYPE_BNBD = 0U;
constexpr uint8_t BSA_COMPAT_MASK_NONE = 0U;
constexpr uint8_t BSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL = 2U;
constexpr uint32_t BSA_COMBINE_ALIGN_BYTES = 32U;
constexpr uint32_t BSA_K_SCALE_BYTES = sizeof(float);
constexpr uint32_t BSA_ATTEN_MASK_DEFAULT_BATCH = 1U;
constexpr uint32_t BSA_ATTEN_MASK_DEFAULT_S2_SIZE = 2048U;
constexpr uint32_t BSA_MXFP8_P_SCALE_SHAPE_SIZE = 1U;
constexpr uint32_t BSA_MXFP8_VALUE_SCALE_LAST_DIM = 2U;
static_assert(BSA_MX_CORE_SPLIT_NUM == BSA_CORE_SPLIT_NUM, "MX and FP8 core split array sizes must match");

uint32_t GetAicCoreNum(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return BSA_MAX_CORE_NUM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    return aicNum == 0U ? BSA_MAX_CORE_NUM : std::min<uint32_t>(aicNum, BSA_MAX_CORE_NUM);
}

uint32_t CalcMxScaleDSize(uint32_t dSize)
{
    // MX scale shape 保留 [D/64, 2] 两个独立维度。该 helper 只返回 D/64；
    // 调用方需要展平元素个数时，再乘以 scaleLastDim。
    return BSACeilDiv(dSize, BSA_MXFP8_SCALE_GROUP_SIZE);
}

QuantBlockSparseAttnMxStrideParams MakePaBnbdStride(uint32_t n2Size, uint32_t blockSize, uint32_t lastDim)
{
    // PA BNBD layout：[physicalBlock, N, blockSize, lastDim]。
    // n2Stride 表示同一个物理 block 内两个 head 之间的偏移；
    // bnStride 表示两个物理 block 之间的偏移。
    QuantBlockSparseAttnMxStrideParams stride;
    stride.n2Stride = static_cast<uint64_t>(blockSize) * lastDim;
    stride.bnStride = static_cast<uint64_t>(n2Size) * stride.n2Stride;
    return stride;
}
} // namespace

QuantBlockSparseAttnTiling::QuantBlockSparseAttnTiling(gert::TilingContext *context) : context_(context)
{
}

void QuantBlockSparseAttnTiling::FillAttrParams()
{
    const auto &info = *tilingInfo_;
    auto &attrParams = tilingData_.attrParams;
    attrParams.set_layoutQ(info.layoutQValue);
    attrParams.set_layoutKv(BSA_LAYOUT_KV_PA_BNSD_VALUE);
    attrParams.set_layoutSparseIndices(BSA_LAYOUT_SPARSE_B_N_QB_KB_VALUE);
    attrParams.set_quantMode(info.quantModeVal);
    attrParams.set_maskMode(info.maskModeVal);
    attrParams.set_returnSoftmaxLse(info.returnSoftmaxLseVal ? 1U : 0U);
}

void QuantBlockSparseAttnTiling::FillPaParams()
{
    const auto &info = *tilingInfo_;
    auto &paParams = tilingData_.paParams;
    paParams.set_blockSize(info.kvBlockSizeVal);
    paParams.set_blockTableDim2(info.maxBlockNumPerBatch);
    paParams.set_paBlockNumSum(info.paBlockNumSum);
    paParams.set_paLayoutType(BSA_PA_LAYOUT_TYPE_BNBD);
}

void QuantBlockSparseAttnTiling::FillSparseParams()
{
    const auto &info = *tilingInfo_;
    auto &sparseParams = tilingData_.sparseParams;
    sparseParams.set_gS1OuterSize(info.gS1OuterSize);
    sparseParams.set_sparseSeqLenStride(info.qbMax);
    sparseParams.set_sparseIndicesStride(info.sparseCount);
}

void QuantBlockSparseAttnTiling::FillInputParams()
{
    const auto &info = *tilingInfo_;
    const uint8_t compatMaskMode =
        (info.maskModeVal == BSA_MASK_MODE_CAUSAL) ? BSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL : BSA_COMPAT_MASK_NONE;
    auto &inputParams = tilingData_.inputParamsRegbase;
    inputParams.set_bSize(info.bSize);
    inputParams.set_t1Size(info.qTokenNum);
    inputParams.set_t2Size(info.s2Size);
    inputParams.set_n2Size(info.n2Size);
    inputParams.set_gSize(info.gSize);
    inputParams.set_s1Size(info.s1Size);
    inputParams.set_s2Size(info.s2Size);
    inputParams.set_alignedS2(info.kbMax * info.kvBlockSizeVal);
    inputParams.set_dSize(BSA_D_SIZE);
    inputParams.set_dSizeV(BSA_D_SIZE);
    inputParams.set_dSizeRope(0);
    inputParams.set_keepProb(1.0F);
    inputParams.set_scaleValue(info.softmaxScaleVal);
    inputParams.set_preTokens(std::numeric_limits<int32_t>::max());
    inputParams.set_nextTokens(0);
    inputParams.set_pseS1Size(0);
    inputParams.set_pseS2Size(0);
    inputParams.set_pseBSize(0);
    inputParams.set_bandIndex(0);
    uint8_t layoutTypeVal = BSA_LAYOUT_TYPE_BSH;
    if (info.layoutQValue == BSA_LAYOUT_Q_TND_VALUE) {
        layoutTypeVal = BSA_LAYOUT_TYPE_TND;
    } else if (info.layoutQValue == BSA_LAYOUT_Q_NTD_VALUE) {
        layoutTypeVal = BSA_LAYOUT_TYPE_NTD;
    }
    inputParams.set_layoutType(layoutTypeVal);
    inputParams.set_pseShapeType(0);
    inputParams.set_attenMaskShapeType(2);
    inputParams.set_attenMaskDataType(1);
    inputParams.set_attenMaskCompressMode(compatMaskMode);
    inputParams.set_implMode(0);
    inputParams.set_sparseType(0);
    inputParams.set_needDropMaskOp(0);
    inputParams.set_dropMaskOuter(0);
    inputParams.set_pseEncodeType(0);
    inputParams.set_tndSoftmaxOut(0);
    inputParams.set_remain(0);
    inputParams.set_attenMaskS2Size(2048);
    inputParams.set_pseType(0);
    inputParams.set_rsv1(0);
    inputParams.set_qStartIdx(0);
    inputParams.set_kvStartIdx(0);
    inputParams.set_s1SparseValidSize(info.s1Size);
    inputParams.set_s2SparseValidSize(info.s2Size);
    inputParams.set_seed(0);
    inputParams.set_offset(0);
    inputParams.set_keepProbUint8(255);
    inputParams.set_pseAlibiBaseS1(0);
    inputParams.set_pseAlibiBaseS2(0);
    inputParams.set_deqScaleFlag(1);
    inputParams.set_deqScale2Flag(1);
    inputParams.set_isActualSeqLengthsNull(info.opParamInfo.seqUsedQ.tensor == nullptr ? 1 : 0);
    inputParams.set_isActualSeqLengthsKVNull(info.opParamInfo.seqUsedKV.tensor == nullptr ? 1 : 0);
    inputParams.set_actualSeqLengthsSize(info.bSize);
    inputParams.set_actualSeqLengthsKVSize(info.bSize);
    inputParams.set_isKvContinuous(1);
    inputParams.set_fromFused(0);
    inputParams.set_isBSNDOut(0);
    inputParams.set_transposeLayout(0);
    inputParams.set_isGqa(info.isGqa ? 1 : 0);
    inputParams.set_isSoftMaxLseEnable(info.returnSoftmaxLseVal ? 1 : 0);
    inputParams.set_isActualSharedPrefixLenNull(1);
    inputParams.set_isQHasLeftPadding(0);
    inputParams.set_isKVHasLeftPadding(0);
    inputParams.set_ropeHeadSize(0);
    inputParams.set_prefixSeqInnerSize(0);
    inputParams.set_headNumRatio(1U);
    inputParams.set_blockSize(info.kvBlockSizeVal);
    inputParams.set_blockTableDim2(info.maxBlockNumPerBatch);
    inputParams.set_paBlockNumSum(info.paBlockNumSum);
    inputParams.set_attenMaskS1Size(info.s1Size);
    inputParams.set_paLayoutType(BSA_PA_LAYOUT_TYPE_BNBD);
    inputParams.set_isRowInvalid(1);
    inputParams.set_paBlockStride(info.paBlockStrideVal);
    inputParams.set_qBlockSize(info.qBlockSizeVal);
    inputParams.set_kvBlockSize(info.kvBlockSizeVal);
}

void QuantBlockSparseAttnTiling::FillMultiCoreParams()
{
    const auto &info = *tilingInfo_;
    auto &multiCoreParams = tilingData_.multiCoreParamsRegbase;
    multiCoreParams.set_coreNum(static_cast<int32_t>(usedCoreNum_));
    multiCoreParams.set_totalSize(static_cast<int64_t>(totalTaskNum_));
    multiCoreParams.set_s1OuterSize(info.qbMax);
    multiCoreParams.set_splitFactorSize(0);
    multiCoreParams.set_splitFactorTailSize(0);

    uint32_t bnStartIdx[BSA_CORE_SPLIT_NUM] = {};
    int64_t sparseStartIdx[BSA_CORE_SPLIT_NUM] = {};
    for (uint32_t boundaryIdx = 0U; boundaryIdx <= usedCoreNum_; ++boundaryIdx) {
        const uint64_t taskOffset = totalTaskNum_ * boundaryIdx / usedCoreNum_;
        bnStartIdx[boundaryIdx] = static_cast<uint32_t>(taskOffset / info.gS1OuterSize);
        sparseStartIdx[boundaryIdx] = static_cast<int64_t>(taskOffset % info.gS1OuterSize);
    }
    multiCoreParams.set_bnStartIdx(bnStartIdx);
    multiCoreParams.set_sparseStartIdx(sparseStartIdx);
    multiCoreParams.set_firstFullLoadS1OuterIdx(0);
    multiCoreParams.set_splitCoreMode(0);
    uint8_t reserve[3] = {};
    multiCoreParams.set_reserve(reserve);
}

void QuantBlockSparseAttnTiling::FillDropmaskParams()
{
    auto &dropmaskParams = tilingData_.dropmaskParamsRegbase;
    dropmaskParams.set_multiCoreFactorSize(0);
    dropmaskParams.set_baseUbCalSize(0);
    dropmaskParams.set_multiCoreTotalSize(0);
    dropmaskParams.set_shapeTotalSize(0);
    dropmaskParams.set_dropMaskAddrOffset(0);
}

void QuantBlockSparseAttnTiling::FillInitOutputParams()
{
    const auto &info = *tilingInfo_;
    auto &initOutputParams = tilingData_.initOutputParams;
    const int64_t totalOutputSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size * BSA_D_SIZE;
    const int64_t totalSoftmaxLseSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size;
    initOutputParams.set_singleCoreSize((totalOutputSize + static_cast<int64_t>(usedCoreNum_) - 1) /
                                        static_cast<int64_t>(usedCoreNum_));
    initOutputParams.set_needInit(1);
    initOutputParams.set_isOneN(info.n1Size == 1U ? 1 : 0);
    uint8_t rsvd[2] = {};
    initOutputParams.set_rsvd(rsvd);
    initOutputParams.set_totalOutputSize(totalOutputSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(totalSoftmaxLseSize);
}

void QuantBlockSparseAttnTiling::FillMxTilingData()
{
    // MXFP8 全量化使用独立 tiling payload。能复用 FP8 host 字段语义的地方尽量保持一致；
    // 但 MX 专属的 scale 分组方式显式写入 tilingdata，避免 kernel 再通过 quant_mode
    // 反推 shape 语义。
    const auto &info = *tilingInfo_;
    const uint8_t compatMaskMode =
        (info.maskModeVal == BSA_MASK_MODE_CAUSAL) ? BSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL : BSA_COMPAT_MASK_NONE;
    const uint32_t dSize = info.dSize == 0U ? BSA_D_SIZE : info.dSize;
    const uint32_t dSizeV = info.dSizeV == 0U ? BSA_D_SIZE : info.dSizeV;
    const uint32_t queryScaleDSize = CalcMxScaleDSize(dSize);
    const uint32_t keyScaleDSize = CalcMxScaleDSize(dSize);
    // valueAntiquantScale 在 token 维度上是 per-channel-group：
    // 每个物理 PA block 对应 [ceil(blockSize/64), N, DV, 2]。
    const uint32_t valueScaleBlockSize = BSACeilDiv(info.kvBlockSizeVal, BSA_MXFP8_SCALE_GROUP_SIZE);
    const uint32_t valueScaleDSize = dSizeV * BSA_MXFP8_VALUE_SCALE_LAST_DIM;
    const bool actualSeqQNull = info.opParamInfo.seqUsedQ.tensor == nullptr;
    const bool actualSeqKVNull = info.opParamInfo.seqUsedKV.tensor == nullptr;

    auto &attrParams = mxTilingData_.attrParams;
    attrParams.layoutQ = info.layoutQValue;
    attrParams.layoutKv = BSA_LAYOUT_KV_PA_BNSD_VALUE;
    attrParams.layoutSparseIndices = BSA_LAYOUT_SPARSE_B_N_QB_KB_VALUE;
    attrParams.quantMode = info.quantModeVal;
    attrParams.maskMode = info.maskModeVal;
    attrParams.returnSoftmaxLse = info.returnSoftmaxLseVal ? 1U : 0U;

    auto &baseParams = mxTilingData_.baseParams;
    baseParams.bSize = info.bSize;
    baseParams.t1Size = info.qTokenNum;
    baseParams.t2Size = info.s2Size;
    baseParams.n2Size = info.n2Size;
    baseParams.gSize = info.gSize;
    baseParams.s1Size = info.s1Size;
    baseParams.s2Size = info.s2Size;
    baseParams.dSize = dSize;
    baseParams.dSizeV = dSizeV;
    baseParams.dSizeRope = 0U;
    baseParams.actualSeqLengthsQSize = actualSeqQNull ? 0U : info.bSize;
    baseParams.actualSeqLengthsKVSize = actualSeqKVNull ? 0U : info.bSize;
    baseParams.scaleValue = info.softmaxScaleVal;
    baseParams.isKvContinuous = 0U;
    baseParams.isSoftMaxLseEnable = info.returnSoftmaxLseVal ? 1U : 0U;
    baseParams.isActualSeqLengthsNull = actualSeqQNull ? 1U : 0U;
    baseParams.isActualSeqLengthsKVNull = actualSeqKVNull ? 1U : 0U;
    baseParams.coreNum = usedCoreNum_;
    baseParams.outputLayout = info.layoutQValue;
    // K/V 数据均为 PA BNBD。K scale 为 token-major，lastDim = (D/64)*2；
    // V scale 为 tokenGroup-major，lastDim = DV*2。
    baseParams.keyStrides = MakePaBnbdStride(info.n2Size, info.kvBlockSizeVal, dSize);
    baseParams.valueStrides = MakePaBnbdStride(info.n2Size, info.kvBlockSizeVal, dSizeV);
    baseParams.kScaleStrides =
        MakePaBnbdStride(info.n2Size, info.kvBlockSizeVal, keyScaleDSize * BSA_MXFP8_SCALE_LAST_DIM);
    baseParams.vScaleStrides = MakePaBnbdStride(info.n2Size, valueScaleBlockSize, valueScaleDSize);

    auto &attenMaskParams = mxTilingData_.attenMaskParams;
    attenMaskParams.sparseMode = static_cast<uint8_t>(info.maskModeVal);
    attenMaskParams.attenMaskDataType = 1U;
    attenMaskParams.attenMaskCompressMode = compatMaskMode;
    attenMaskParams.isRowInvalidOpen = 1U;
    attenMaskParams.preTokens = std::numeric_limits<int32_t>::max();
    attenMaskParams.nextTokens = 0;
    attenMaskParams.attenMaskBatch = BSA_ATTEN_MASK_DEFAULT_BATCH;
    attenMaskParams.attenMaskS1Size = info.s1Size;
    attenMaskParams.attenMaskS2Size = BSA_ATTEN_MASK_DEFAULT_S2_SIZE;
    attenMaskParams.isExistRowInvalid = 1U;

    auto &pageAttentionParams = mxTilingData_.pageAttentionParams;
    pageAttentionParams.paLayoutType = BSA_PA_LAYOUT_TYPE_BNBD;
    pageAttentionParams.blockSize = info.kvBlockSizeVal;
    pageAttentionParams.maxBlockNumPerBatch = info.maxBlockNumPerBatch;
    pageAttentionParams.paBlockNumSum = info.paBlockNumSum;
    pageAttentionParams.paBlockStride = info.paBlockStrideVal;
    pageAttentionParams.qBlockSize = info.qBlockSizeVal;
    pageAttentionParams.kvBlockSize = info.kvBlockSizeVal;

    auto &sparseParams = mxTilingData_.sparseParams;
    sparseParams.gS1OuterSize = info.gS1OuterSize;
    sparseParams.sparseSeqLenStride = info.qbMax;
    sparseParams.sparseIndicesStride = info.sparseCount;
    sparseParams.maxQb = info.qbMax;
    sparseParams.maxKb = info.kbMax;
    sparseParams.sparseCount = info.sparseCount;

    auto &multiCoreParams = mxTilingData_.multiCoreParams;
    // MX kernel 当前仍复用 BSA metadata 作为实际切核边界。
    // 这里仍填充这些数组，使 MX tilingdata 保持自描述，
    // 后续切换到 common-MX 风格切核逻辑时可以继续兼容。
    multiCoreParams.coreNum = static_cast<int32_t>(usedCoreNum_);
    multiCoreParams.totalSize = static_cast<int64_t>(totalTaskNum_);
    multiCoreParams.s1OuterSize = info.qbMax;
    multiCoreParams.splitFactorSize = 0;
    multiCoreParams.splitFactorTailSize = 0;
    for (uint32_t boundaryIdx = 0U; boundaryIdx <= usedCoreNum_ && boundaryIdx < BSA_MX_CORE_SPLIT_NUM;
         ++boundaryIdx) {
        const uint64_t taskOffset = totalTaskNum_ * boundaryIdx / usedCoreNum_;
        multiCoreParams.bnStartIdx[boundaryIdx] = static_cast<uint32_t>(taskOffset / info.gS1OuterSize);
        multiCoreParams.sparseStartIdx[boundaryIdx] = static_cast<int64_t>(taskOffset % info.gS1OuterSize);
    }
    multiCoreParams.firstFullLoadS1OuterIdx = 0;
    multiCoreParams.splitCoreMode = 0;

    auto &workspaceParams = mxTilingData_.workspaceParams;
    workspaceParams.accumOutSize = 0U;
    workspaceParams.logSumExpSize = 0U;

    auto &scaleParams = mxTilingData_.scaleParams;
    // 同时记录 shape 分组信息和 antiquant mode 值。当前 kernel 主要使用 size；
    // check/诊断仍可通过这些字段确认 MX 模式约束：
    // query/key 为 per-token-group(6)，value 为 per-channel-group(8)。
    scaleParams.scaleGroupSize = BSA_MXFP8_SCALE_GROUP_SIZE;
    scaleParams.scaleLastDim = BSA_MXFP8_SCALE_LAST_DIM;
    scaleParams.queryScaleDSize = queryScaleDSize;
    scaleParams.keyScaleDSize = keyScaleDSize;
    scaleParams.valueScaleBlockSize = valueScaleBlockSize;
    scaleParams.valueScaleDSize = valueScaleDSize;
    scaleParams.pScaleShapeSize = BSA_MXFP8_P_SCALE_SHAPE_SIZE;
    scaleParams.queryQuantMode = BSA_MXFP8_PER_TOKEN_GROUP_MODE;
    scaleParams.keyAntiquantMode = BSA_MXFP8_PER_TOKEN_GROUP_MODE;
    scaleParams.valueAntiquantMode = BSA_MXFP8_PER_CHANNEL_GROUP_MODE;

    auto &emptyTensorParams = mxTilingData_.emptyTensorParams;
    const uint64_t totalOutputSize = static_cast<uint64_t>(info.qTokenNum) * info.n1Size * dSizeV;
    const uint64_t totalSoftmaxLseSize = static_cast<uint64_t>(info.qTokenNum) * info.n1Size;
    emptyTensorParams.singleCoreSize =
        static_cast<uint32_t>((totalOutputSize + static_cast<uint64_t>(usedCoreNum_) - 1U) / usedCoreNum_);
    emptyTensorParams.needInit = 1U;
    emptyTensorParams.isOneN = info.n1Size == 1U ? 1U : 0U;
    emptyTensorParams.totalOutputSize = totalOutputSize;
    emptyTensorParams.totalSoftMaxLseOutputSize = totalSoftmaxLseSize;
}

void QuantBlockSparseAttnTiling::CalcTilingKey()
{
    const auto &info = *tilingInfo_;
    // 当前 FP8 与 MXFP8 kernel 都走 DN 路径。这里仍把 use_dn 写入 tiling key，
    // 让后续需要接入非 DN 路径时，可以只扩展 tiling 选择和 kernel 分支，而不改变 key 结构。
    constexpr uint32_t useDn = BSA_USE_DN_TRUE;
    if (info.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        // MX 使用独立 tiling key 和 S2=512 config，使 kernel dispatch 能选择
        // QuantBlockSparseAttnMxKernel，并读取 QuantBlockSparseAttnMxTilingData。
        tilingKey_ = GET_TPL_TILING_KEY(
            BSA_DTYPE_FP8_E4M3FN, info.layoutQValue, BSA_KV_LAYOUT_PA_BNSD, info.maskModeVal,
            info.returnSoftmaxLseVal ? 1U : 0U, Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128,
            useDn, MXFullQuantMode);
    } else {
        tilingKey_ = GET_TPL_TILING_KEY(
            BSA_DTYPE_FP8_E4M3FN, info.layoutQValue, BSA_KV_LAYOUT_PA_BNSD, info.maskModeVal,
            info.returnSoftmaxLseVal ? 1U : 0U, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128,
            useDn, FP8QuantMode);
    }
}

void QuantBlockSparseAttnTiling::CalcWorkspaceSize()
{
    workspaceSize_ = 0;
}

void QuantBlockSparseAttnTiling::PrintAllTilingData()
{
    auto &attrParams = tilingData_.attrParams;
    OP_LOGD(kOpName, "===== AttrParams =====");
    OP_LOGD(kOpName, "layoutQ:%u", attrParams.get_layoutQ());
    OP_LOGD(kOpName, "layoutKv:%u", attrParams.get_layoutKv());
    OP_LOGD(kOpName, "layoutSparseIndices:%u", attrParams.get_layoutSparseIndices());
    OP_LOGD(kOpName, "quantMode:%u", attrParams.get_quantMode());
    OP_LOGD(kOpName, "maskMode:%u", attrParams.get_maskMode());
    OP_LOGD(kOpName, "returnSoftmaxLse:%u", attrParams.get_returnSoftmaxLse());

    auto &paParams = tilingData_.paParams;
    OP_LOGD(kOpName, "===== PaParams =====");
    OP_LOGD(kOpName, "blockSize:%u", paParams.get_blockSize());
    OP_LOGD(kOpName, "blockTableDim2:%u", paParams.get_blockTableDim2());
    OP_LOGD(kOpName, "paBlockNumSum:%u", paParams.get_paBlockNumSum());
    OP_LOGD(kOpName, "paLayoutType:%u", paParams.get_paLayoutType());

    auto &sparseParams = tilingData_.sparseParams;
    OP_LOGD(kOpName, "===== SparseParams =====");
    OP_LOGD(kOpName, "gS1OuterSize:%u", sparseParams.get_gS1OuterSize());
    OP_LOGD(kOpName, "sparseSeqLenStride:%u", sparseParams.get_sparseSeqLenStride());
    OP_LOGD(kOpName, "sparseIndicesStride:%u", sparseParams.get_sparseIndicesStride());

    auto &inputParams = tilingData_.inputParamsRegbase;
    OP_LOGD(kOpName, "===== InputParamsRegbase =====");
    OP_LOGD(kOpName, "bSize:%ld", inputParams.get_bSize());
    OP_LOGD(kOpName, "t1Size:%ld", inputParams.get_t1Size());
    OP_LOGD(kOpName, "t2Size:%ld", inputParams.get_t2Size());
    OP_LOGD(kOpName, "n2Size:%ld", inputParams.get_n2Size());
    OP_LOGD(kOpName, "gSize:%ld", inputParams.get_gSize());
    OP_LOGD(kOpName, "s1Size:%ld", inputParams.get_s1Size());
    OP_LOGD(kOpName, "s2Size:%ld", inputParams.get_s2Size());
    OP_LOGD(kOpName, "alignedS2:%ld", inputParams.get_alignedS2());
    OP_LOGD(kOpName, "dSize:%ld", inputParams.get_dSize());
    OP_LOGD(kOpName, "dSizeV:%ld", inputParams.get_dSizeV());
    OP_LOGD(kOpName, "dSizeRope:%ld", inputParams.get_dSizeRope());
    OP_LOGD(kOpName, "keepProb:%f", inputParams.get_keepProb());
    OP_LOGD(kOpName, "scaleValue:%f", inputParams.get_scaleValue());
    OP_LOGD(kOpName, "preTokens:%ld", inputParams.get_preTokens());
    OP_LOGD(kOpName, "nextTokens:%ld", inputParams.get_nextTokens());
    OP_LOGD(kOpName, "layoutType:%u", inputParams.get_layoutType());
    OP_LOGD(kOpName, "attenMaskCompressMode:%u", inputParams.get_attenMaskCompressMode());
    OP_LOGD(kOpName, "attenMaskS2Size:%u", inputParams.get_attenMaskS2Size());
    OP_LOGD(kOpName, "isActualSeqLengthsNull:%u", inputParams.get_isActualSeqLengthsNull());
    OP_LOGD(kOpName, "isActualSeqLengthsKVNull:%u", inputParams.get_isActualSeqLengthsKVNull());
    OP_LOGD(kOpName, "isKvContinuous:%u", inputParams.get_isKvContinuous());
    OP_LOGD(kOpName, "isGqa:%u", inputParams.get_isGqa());
    OP_LOGD(kOpName, "isSoftMaxLseEnable:%u", inputParams.get_isSoftMaxLseEnable());
    OP_LOGD(kOpName, "headNumRatio:%u", inputParams.get_headNumRatio());
    OP_LOGD(kOpName, "blockSize:%d", inputParams.get_blockSize());
    OP_LOGD(kOpName, "blockTableDim2:%d", inputParams.get_blockTableDim2());
    OP_LOGD(kOpName, "paBlockNumSum:%d", inputParams.get_paBlockNumSum());
    OP_LOGD(kOpName, "paLayoutType:%u", inputParams.get_paLayoutType());
    OP_LOGD(kOpName, "qBlockSize:%u", inputParams.get_qBlockSize());
    OP_LOGD(kOpName, "kvBlockSize:%u", inputParams.get_kvBlockSize());

    auto &multiCoreParams = tilingData_.multiCoreParamsRegbase;
    OP_LOGD(kOpName, "===== MultiCoreParams =====");
    OP_LOGD(kOpName, "coreNum:%d", multiCoreParams.get_coreNum());
    OP_LOGD(kOpName, "totalSize:%ld", multiCoreParams.get_totalSize());
    OP_LOGD(kOpName, "s1OuterSize:%ld", multiCoreParams.get_s1OuterSize());
    OP_LOGD(kOpName, "splitFactorSize:%ld", multiCoreParams.get_splitFactorSize());
    OP_LOGD(kOpName, "splitFactorTailSize:%ld", multiCoreParams.get_splitFactorTailSize());
    for (uint32_t i = 0; i <= usedCoreNum_ && i < BSA_CORE_SPLIT_NUM; ++i) {
        OP_LOGD(kOpName, "bnStartIdx[%u]:%u, sparseStartIdx[%u]:%ld", i, multiCoreParams.get_bnStartIdx()[i], i,
                multiCoreParams.get_sparseStartIdx()[i]);
    }
    OP_LOGD(kOpName, "firstFullLoadS1OuterIdx:%ld", multiCoreParams.get_firstFullLoadS1OuterIdx());
    OP_LOGD(kOpName, "splitCoreMode:%u", multiCoreParams.get_splitCoreMode());

    auto &dropmaskParams = tilingData_.dropmaskParamsRegbase;
    OP_LOGD(kOpName, "===== DropmaskParams =====");
    OP_LOGD(kOpName, "multiCoreFactorSize:%d", dropmaskParams.get_multiCoreFactorSize());
    OP_LOGD(kOpName, "baseUbCalSize:%d", dropmaskParams.get_baseUbCalSize());
    OP_LOGD(kOpName, "multiCoreTotalSize:%ld", dropmaskParams.get_multiCoreTotalSize());
    OP_LOGD(kOpName, "shapeTotalSize:%ld", dropmaskParams.get_shapeTotalSize());
    OP_LOGD(kOpName, "dropMaskAddrOffset:%ld", dropmaskParams.get_dropMaskAddrOffset());

    auto &initOutputParams = tilingData_.initOutputParams;
    OP_LOGD(kOpName, "===== InitOutputParams =====");
    OP_LOGD(kOpName, "singleCoreSize:%u", initOutputParams.get_singleCoreSize());
    OP_LOGD(kOpName, "needInit:%u", initOutputParams.get_needInit());
    OP_LOGD(kOpName, "isOneN:%u", initOutputParams.get_isOneN());
    OP_LOGD(kOpName, "totalOutputSize:%ld", initOutputParams.get_totalOutputSize());
    OP_LOGD(kOpName, "totalSoftMaxLseOutputSize:%ld", initOutputParams.get_totalSoftMaxLseOutputSize());

    OP_LOGD(kOpName, "===== Summary =====");
    OP_LOGD(kOpName, "usedCoreNum:%u", usedCoreNum_);
    OP_LOGD(kOpName, "totalTaskNum:%lu", totalTaskNum_);
    OP_LOGD(kOpName, "tilingKey:%lu", tilingKey_);
    OP_LOGD(kOpName, "workspaceSize:%lu", workspaceSize_);
    OP_LOGD(kOpName, "tilingDataSize:%lu", tilingData_.GetDataSize());
    auto rawTilingData = context_->GetRawTilingData();
    OP_LOGD(kOpName, "tilingDataCapacity:%lu", rawTilingData == nullptr ? 0UL : rawTilingData->GetCapacity());
}

void QuantBlockSparseAttnTiling::PrintMxTilingData()
{
    const auto &attrParams = mxTilingData_.attrParams;
    OP_LOGD(kOpName, "===== MX AttrParams =====");
    OP_LOGD(kOpName, "layoutQ:%u", attrParams.layoutQ);
    OP_LOGD(kOpName, "layoutKv:%u", attrParams.layoutKv);
    OP_LOGD(kOpName, "layoutSparseIndices:%u", attrParams.layoutSparseIndices);
    OP_LOGD(kOpName, "quantMode:%u", attrParams.quantMode);
    OP_LOGD(kOpName, "maskMode:%u", attrParams.maskMode);
    OP_LOGD(kOpName, "returnSoftmaxLse:%u", attrParams.returnSoftmaxLse);

    const auto &baseParams = mxTilingData_.baseParams;
    OP_LOGD(kOpName, "===== MX BaseParams =====");
    OP_LOGD(kOpName, "bSize:%u", baseParams.bSize);
    OP_LOGD(kOpName, "t1Size:%u", baseParams.t1Size);
    OP_LOGD(kOpName, "t2Size:%u", baseParams.t2Size);
    OP_LOGD(kOpName, "n2Size:%u", baseParams.n2Size);
    OP_LOGD(kOpName, "gSize:%u", baseParams.gSize);
    OP_LOGD(kOpName, "s1Size:%u", baseParams.s1Size);
    OP_LOGD(kOpName, "s2Size:%u", baseParams.s2Size);
    OP_LOGD(kOpName, "dSize:%u", baseParams.dSize);
    OP_LOGD(kOpName, "dSizeV:%u", baseParams.dSizeV);
    OP_LOGD(kOpName, "actualSeqLengthsQSize:%u", baseParams.actualSeqLengthsQSize);
    OP_LOGD(kOpName, "actualSeqLengthsKVSize:%u", baseParams.actualSeqLengthsKVSize);
    OP_LOGD(kOpName, "scaleValue:%f", baseParams.scaleValue);
    OP_LOGD(kOpName, "isKvContinuous:%u", baseParams.isKvContinuous);
    OP_LOGD(kOpName, "isSoftMaxLseEnable:%u", baseParams.isSoftMaxLseEnable);
    OP_LOGD(kOpName, "coreNum:%u", baseParams.coreNum);
    OP_LOGD(kOpName, "keyStrides:%lu/%lu", baseParams.keyStrides.bnStride, baseParams.keyStrides.n2Stride);
    OP_LOGD(kOpName, "valueStrides:%lu/%lu", baseParams.valueStrides.bnStride, baseParams.valueStrides.n2Stride);
    OP_LOGD(kOpName, "kScaleStrides:%lu/%lu", baseParams.kScaleStrides.bnStride,
            baseParams.kScaleStrides.n2Stride);
    OP_LOGD(kOpName, "vScaleStrides:%lu/%lu", baseParams.vScaleStrides.bnStride,
            baseParams.vScaleStrides.n2Stride);

    const auto &attenMaskParams = mxTilingData_.attenMaskParams;
    OP_LOGD(kOpName, "===== MX AttenMaskParams =====");
    OP_LOGD(kOpName, "sparseMode:%u", attenMaskParams.sparseMode);
    OP_LOGD(kOpName, "attenMaskCompressMode:%u", attenMaskParams.attenMaskCompressMode);
    OP_LOGD(kOpName, "preTokens:%d", attenMaskParams.preTokens);
    OP_LOGD(kOpName, "nextTokens:%d", attenMaskParams.nextTokens);
    OP_LOGD(kOpName, "attenMaskS1Size:%u", attenMaskParams.attenMaskS1Size);
    OP_LOGD(kOpName, "attenMaskS2Size:%u", attenMaskParams.attenMaskS2Size);

    const auto &pageAttentionParams = mxTilingData_.pageAttentionParams;
    OP_LOGD(kOpName, "===== MX PageAttentionParams =====");
    OP_LOGD(kOpName, "paLayoutType:%u", pageAttentionParams.paLayoutType);
    OP_LOGD(kOpName, "blockSize:%u", pageAttentionParams.blockSize);
    OP_LOGD(kOpName, "maxBlockNumPerBatch:%u", pageAttentionParams.maxBlockNumPerBatch);
    OP_LOGD(kOpName, "paBlockNumSum:%u", pageAttentionParams.paBlockNumSum);
    OP_LOGD(kOpName, "paBlockStride:%u", pageAttentionParams.paBlockStride);
    OP_LOGD(kOpName, "qBlockSize:%u", pageAttentionParams.qBlockSize);
    OP_LOGD(kOpName, "kvBlockSize:%u", pageAttentionParams.kvBlockSize);

    const auto &sparseParams = mxTilingData_.sparseParams;
    OP_LOGD(kOpName, "===== MX SparseParams =====");
    OP_LOGD(kOpName, "gS1OuterSize:%u", sparseParams.gS1OuterSize);
    OP_LOGD(kOpName, "sparseSeqLenStride:%u", sparseParams.sparseSeqLenStride);
    OP_LOGD(kOpName, "sparseIndicesStride:%u", sparseParams.sparseIndicesStride);
    OP_LOGD(kOpName, "maxQb:%u", sparseParams.maxQb);
    OP_LOGD(kOpName, "maxKb:%u", sparseParams.maxKb);
    OP_LOGD(kOpName, "sparseCount:%u", sparseParams.sparseCount);

    const auto &scaleParams = mxTilingData_.scaleParams;
    OP_LOGD(kOpName, "===== MX ScaleParams =====");
    OP_LOGD(kOpName, "scaleGroupSize:%u", scaleParams.scaleGroupSize);
    OP_LOGD(kOpName, "scaleLastDim:%u", scaleParams.scaleLastDim);
    OP_LOGD(kOpName, "queryScaleDSize:%u", scaleParams.queryScaleDSize);
    OP_LOGD(kOpName, "keyScaleDSize:%u", scaleParams.keyScaleDSize);
    OP_LOGD(kOpName, "valueScaleBlockSize:%u", scaleParams.valueScaleBlockSize);
    OP_LOGD(kOpName, "valueScaleDSize:%u", scaleParams.valueScaleDSize);
    OP_LOGD(kOpName, "pScaleShapeSize:%u", scaleParams.pScaleShapeSize);
    OP_LOGD(kOpName, "queryQuantMode/keyAntiquantMode/valueAntiquantMode:%u/%u/%u", scaleParams.queryQuantMode,
            scaleParams.keyAntiquantMode, scaleParams.valueAntiquantMode);

    const auto &multiCoreParams = mxTilingData_.multiCoreParams;
    OP_LOGD(kOpName, "===== MX MultiCoreParams =====");
    OP_LOGD(kOpName, "coreNum:%d", multiCoreParams.coreNum);
    OP_LOGD(kOpName, "totalSize:%ld", multiCoreParams.totalSize);
    OP_LOGD(kOpName, "s1OuterSize:%ld", multiCoreParams.s1OuterSize);
    for (uint32_t i = 0; i <= usedCoreNum_ && i < BSA_MX_CORE_SPLIT_NUM; ++i) {
        OP_LOGD(kOpName, "bnStartIdx[%u]:%u, sparseStartIdx[%u]:%ld", i, multiCoreParams.bnStartIdx[i], i,
                multiCoreParams.sparseStartIdx[i]);
    }

    const auto &emptyTensorParams = mxTilingData_.emptyTensorParams;
    OP_LOGD(kOpName, "===== MX EmptyTensorParams =====");
    OP_LOGD(kOpName, "singleCoreSize:%u", emptyTensorParams.singleCoreSize);
    OP_LOGD(kOpName, "needInit:%u", emptyTensorParams.needInit);
    OP_LOGD(kOpName, "isOneN:%u", emptyTensorParams.isOneN);
    OP_LOGD(kOpName, "totalOutputSize:%lu", emptyTensorParams.totalOutputSize);
    OP_LOGD(kOpName, "totalSoftMaxLseOutputSize:%lu", emptyTensorParams.totalSoftMaxLseOutputSize);

    auto rawTilingData = context_->GetRawTilingData();
    OP_LOGD(kOpName, "===== MX Summary =====");
    OP_LOGD(kOpName, "usedCoreNum:%u", usedCoreNum_);
    OP_LOGD(kOpName, "totalTaskNum:%lu", totalTaskNum_);
    OP_LOGD(kOpName, "tilingKey:%lu", tilingKey_);
    OP_LOGD(kOpName, "workspaceSize:%lu", workspaceSize_);
    OP_LOGD(kOpName, "mxTilingDataSize:%lu", sizeof(QuantBlockSparseAttnMxTilingData));
    OP_LOGD(kOpName, "tilingDataCapacity:%lu", rawTilingData == nullptr ? 0UL : rawTilingData->GetCapacity());
}

ge::graphStatus QuantBlockSparseAttnTiling::SaveTilingData()
{
    auto rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr) {
        OP_LOGE(kOpName, "DoOpTiling: rawTilingData is nullptr");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo_->quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        const auto dataSize = sizeof(QuantBlockSparseAttnMxTilingData);
        if (rawTilingData->GetCapacity() < dataSize) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "tilingDataCapacity",
                                                  std::to_string(rawTilingData->GetCapacity()),
                                                  "must be at least " + std::to_string(dataSize) +
                                                      " for MXFP8 full-quant tiling data");
            return ge::GRAPH_FAILED;
        }
        std::memcpy(rawTilingData->GetData(), &mxTilingData_, dataSize);
        rawTilingData->SetDataSize(dataSize);
        return ge::GRAPH_SUCCESS;
    }

    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnTiling::DoOpTiling(QuantBlockSparseAttnTilingInfo *tilingInfo)
{
    if (tilingInfo == nullptr || context_ == nullptr) {
        OP_LOGE(kOpName, "DoOpTiling: tilingInfo=%p or context=%p is nullptr", tilingInfo, context_);
        return ge::GRAPH_FAILED;
    }
    tilingInfo_ = tilingInfo;

    const uint64_t bnCount = tilingInfo_->bSize * tilingInfo_->n1Size;
    totalTaskNum_ = bnCount * tilingInfo_->gS1OuterSize;
    if (totalTaskNum_ == 0U || totalTaskNum_ > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "bSize * n1Size * gS1OuterSize", std::to_string(totalTaskNum_),
                                              "total task num must be in range [1, " +
                                                  std::to_string(std::numeric_limits<uint32_t>::max()) + "]");
        return ge::GRAPH_FAILED;
    }

    const uint32_t maxAicCoreNum = GetAicCoreNum(context_);
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(maxAicCoreNum), totalTaskNum_));

    if (tilingInfo_->quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        FillMxTilingData();
    } else {
        FillAttrParams();
        FillPaParams();
        FillSparseParams();
        FillInputParams();
        FillMultiCoreParams();
        FillDropmaskParams();
        FillInitOutputParams();
    }
    CalcTilingKey();
    CalcWorkspaceSize();
    if (tilingInfo_->quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        PrintMxTilingData();
    } else {
        PrintAllTilingData();
    }

    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    context_->SetScheduleMode(1);
    return SaveTilingData();
}

ge::graphStatus TilingQuantBlockSparseAttn(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "Tiling: tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttnTilingInfo tilingInfo;
    QuantBlockSparseAttnInfoParser parser(context);
    if (parser.Parse(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttnCheck checker(tilingInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttnTiling tiling(context);
    return tiling.DoOpTiling(&tilingInfo);
}

} // namespace optiling

namespace ge {
graphStatus TilingPrepareQuantBlockSparseAttn(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        OP_LOGE("QuantBlockSparseAttn", "TilingPrepare: context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class QuantBlockSparseAttnTiling {
public:
    static ge::graphStatus Tiling(gert::TilingContext *context)
    {
        return optiling::TilingQuantBlockSparseAttn(context);
    }
};
} // namespace ops

namespace optiling {
IMPL_OP_OPTILING(QuantBlockSparseAttn).Tiling(TilingQuantBlockSparseAttn);
} // namespace optiling
