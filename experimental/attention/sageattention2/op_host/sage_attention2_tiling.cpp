/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tiling/platform/platform_ascendc.h>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "sage_attention2_tiling.h"

#ifndef SAGE_ATTENTION2_PROBE_BLOCK_DIM
#define SAGE_ATTENTION2_PROBE_BLOCK_DIM 0U
#endif
#ifndef SAGE_ATTENTION2_PROBE_SCHEDULE_MODE
#define SAGE_ATTENTION2_PROBE_SCHEDULE_MODE 1U
#endif
#ifndef SAGE_ATTENTION2_PROBE_ACTIVE_CORE_NUM
#define SAGE_ATTENTION2_PROBE_ACTIVE_CORE_NUM 20U
#endif

namespace optiling {
namespace {
constexpr const char *kOpName = "SageAttention2";
constexpr uint64_t kWorkspaceAlign = 512ULL;
constexpr uint32_t kQTile = 128U;
// Must match SageAttention2Kernel::KV_TILE. A mismatch silently under-schedules attention blocks.
constexpr uint32_t kKvTile = 256U;
constexpr uint32_t kKQuantBlock = 64U;
constexpr float kQuantEpsilon = 1.0e-7F;

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &out)
{
    if (lhs != 0U && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t &out)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    out = lhs + rhs;
    return true;
}

bool AppendWorkspace(uint64_t bytes, uint64_t &cursor, uint64_t &offset)
{
    if (cursor > std::numeric_limits<uint64_t>::max() - (kWorkspaceAlign - 1ULL)) {
        return false;
    }
    cursor = (cursor + kWorkspaceAlign - 1ULL) / kWorkspaceAlign * kWorkspaceAlign;
    offset = cursor;
    return CheckedAdd(cursor, bytes, cursor);
}

bool GetDim(const gert::Shape &shape, size_t axis, uint32_t &value)
{
    if (axis >= shape.GetDimNum()) {
        return false;
    }
    const int64_t dim = shape.GetDim(axis);
    if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(dim);
    return true;
}

std::string GetStringAttr(const gert::RuntimeAttrs *attrs, uint32_t index, const char *defaultValue)
{
    const char *value = attrs->GetAttrPointer<char>(index);
    return value == nullptr ? std::string(defaultValue) : std::string(value);
}

bool GetBoolAttr(const gert::RuntimeAttrs *attrs, uint32_t index, bool defaultValue)
{
    const bool *value = attrs->GetAttrPointer<bool>(index);
    return value == nullptr ? defaultValue : *value;
}

float GetFloatAttr(const gert::RuntimeAttrs *attrs, uint32_t index, float defaultValue)
{
    const float *value = attrs->GetAttrPointer<float>(index);
    return value == nullptr ? defaultValue : *value;
}

ge::graphStatus Fail(const std::string &message)
{
    OP_LOGE(kOpName, "%s", message.c_str());
    return ge::GRAPH_FAILED;
}
} // namespace

ge::graphStatus TilingSageAttention2(gert::TilingContext *context)
{
    if (context == nullptr) {
        return Fail("tiling context is nullptr");
    }
    const gert::StorageShape *qStorage = context->GetInputShape(SAGE_Q_INDEX);
    const gert::StorageShape *kStorage = context->GetInputShape(SAGE_K_INDEX);
    const gert::StorageShape *vStorage = context->GetInputShape(SAGE_V_INDEX);
    const gert::CompileTimeTensorDesc *qDesc = context->GetInputDesc(SAGE_Q_INDEX);
    const gert::CompileTimeTensorDesc *kDesc = context->GetInputDesc(SAGE_K_INDEX);
    const gert::CompileTimeTensorDesc *vDesc = context->GetInputDesc(SAGE_V_INDEX);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (qStorage == nullptr || kStorage == nullptr || vStorage == nullptr || qDesc == nullptr || kDesc == nullptr ||
        vDesc == nullptr || attrs == nullptr) {
        return Fail("required shape, descriptor, or attrs is nullptr");
    }
    const gert::Shape &qShape = qStorage->GetStorageShape();
    const gert::Shape &kShape = kStorage->GetStorageShape();
    const gert::Shape &vShape = vStorage->GetStorageShape();
    if (qShape.GetDimNum() != 4U || kShape.GetDimNum() != 4U || vShape.GetDimNum() != 4U) {
        return Fail("query, key, and value must all be rank 4");
    }
    if (qDesc->GetDataType() != kDesc->GetDataType() || qDesc->GetDataType() != vDesc->GetDataType() ||
        (qDesc->GetDataType() != ge::DT_FLOAT16 && qDesc->GetDataType() != ge::DT_BF16)) {
        return Fail("query, key, and value must have the same FP16 or BF16 dtype");
    }

    const std::string layoutString = GetStringAttr(attrs, SAGE_LAYOUT_ATTR, "HND");
    SageLayout layout;
    if (layoutString == "HND") {
        layout = SageLayout::HND;
    } else if (layoutString == "NHD") {
        layout = SageLayout::NHD;
    } else {
        return Fail("tensor_layout must be HND or NHD");
    }
    const size_t headAxis = layout == SageLayout::HND ? 1U : 2U;
    const size_t seqAxis = layout == SageLayout::HND ? 2U : 1U;
    uint32_t batch = 0U;
    uint32_t qHeads = 0U;
    uint32_t kvHeads = 0U;
    uint32_t qSeq = 0U;
    uint32_t kvSeq = 0U;
    uint32_t headDim = 0U;
    uint32_t kBatch = 0U;
    uint32_t vBatch = 0U;
    uint32_t vHeads = 0U;
    uint32_t vSeq = 0U;
    uint32_t kDim = 0U;
    uint32_t vDim = 0U;
    if (!GetDim(qShape, 0U, batch) || !GetDim(qShape, headAxis, qHeads) || !GetDim(qShape, seqAxis, qSeq) ||
        !GetDim(qShape, 3U, headDim) || !GetDim(kShape, 0U, kBatch) || !GetDim(kShape, headAxis, kvHeads) ||
        !GetDim(kShape, seqAxis, kvSeq) || !GetDim(kShape, 3U, kDim) || !GetDim(vShape, 0U, vBatch) ||
        !GetDim(vShape, headAxis, vHeads) || !GetDim(vShape, seqAxis, vSeq) || !GetDim(vShape, 3U, vDim)) {
        return Fail("all dimensions must be positive and fit uint32");
    }
    if (batch != kBatch || batch != vBatch || kvHeads != vHeads || kvSeq != vSeq || headDim != kDim ||
        headDim != vDim) {
        return Fail("query/key/value batch and head_dim must match; key/value heads and sequence must match");
    }
    if (headDim > 128U || qHeads % kvHeads != 0U) {
        return Fail("head_dim must be in [1, 128] and q_heads must be divisible by kv_heads");
    }

    const bool causal = GetBoolAttr(attrs, SAGE_CAUSAL_ATTR, false);
    if (causal && qSeq != kvSeq) {
        return Fail("causal mode requires q_seq == kv_seq");
    }
    const std::string granString = GetStringAttr(attrs, SAGE_GRAN_ATTR, "per_thread");
    SageQuantGran gran;
    if (granString == "per_warp") {
        gran = SageQuantGran::PER_WARP;
    } else if (granString == "per_thread") {
        gran = SageQuantGran::PER_THREAD;
    } else {
        return Fail("qk_quant_gran must be per_warp or per_thread");
    }
    const std::string accumString = GetStringAttr(attrs, SAGE_ACCUM_ATTR, "fp32+fp16");
    SageAccumMode accum;
    if (accumString == "fp32") {
        accum = SageAccumMode::FP32;
    } else if (accumString == "fp32+fp32") {
        accum = SageAccumMode::FP32_FP32;
    } else if (accumString == "fp32+fp16") {
        accum = SageAccumMode::FP32_FP16;
    } else {
        return Fail("pv_accum_dtype must be fp32, fp32+fp32, or fp32+fp16");
    }

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return Fail("platform info is nullptr");
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    if (platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND950) {
        return Fail("SageAttention2 only supports Ascend 950");
    }
    const uint32_t reportedCoreNum = platform.GetCoreNumAic();
    const uint32_t reportedVecCoreNum = platform.GetCoreNumAiv();
    const uint32_t coreNum = SAGE_ATTENTION2_PROBE_ACTIVE_CORE_NUM == 0U ? reportedCoreNum :
        std::min(reportedCoreNum, SAGE_ATTENTION2_PROBE_ACTIVE_CORE_NUM);
    const uint32_t vecCoreNum = SAGE_ATTENTION2_PROBE_ACTIVE_CORE_NUM == 0U ? reportedVecCoreNum :
        std::min(reportedVecCoreNum, coreNum * 2U);
    if (coreNum == 0U || vecCoreNum < coreNum * 2U) {
        return Fail("SageAttention2 requires Ascend 950 mixed AIC:AIV=1:2 cores");
    }

    const uint32_t headDimPad = headDim <= 64U ? 64U : 128U;
    const uint32_t qBlocks = CeilDiv(qSeq, kQTile);
    const uint32_t kvBlocks = CeilDiv(kvSeq, kKvTile);
    const uint32_t kvSeqPad = CeilDiv(kvSeq, kKQuantBlock) * kKQuantBlock;
    const uint32_t qGroups = gran == SageQuantGran::PER_WARP ? 4U : 32U;
    const uint32_t kGroups = gran == SageQuantGran::PER_WARP ? 1U : 4U;
    const bool smoothK = GetBoolAttr(attrs, SAGE_SMOOTH_K_ATTR, true);
    // Upstream ignores smooth_v in hierarchical accumulation modes.
    const bool smoothV = GetBoolAttr(attrs, SAGE_SMOOTH_V_ATTR, false) && accum == SageAccumMode::FP32;
    const bool returnLse = GetBoolAttr(attrs, SAGE_RETURN_LSE_ATTR, false);
    float smScale = GetFloatAttr(attrs, SAGE_SCALE_ATTR, 0.0F);
    if (!(smScale > 0.0F) || !std::isfinite(smScale)) {
        smScale = 1.0F / std::sqrt(static_cast<float>(headDim));
    }

    uint64_t qElements = 0U;
    uint64_t kElements = 0U;
    uint64_t vElements = 0U;
    uint64_t tmp = 0U;
    if (!CheckedMul(batch, qSeq, tmp) || !CheckedMul(tmp, qHeads, tmp) || !CheckedMul(tmp, headDimPad, qElements) ||
        !CheckedMul(batch, kvSeq, tmp) || !CheckedMul(tmp, kvHeads, tmp) || !CheckedMul(tmp, headDimPad, kElements) ||
        !CheckedMul(batch, kvSeqPad, tmp) || !CheckedMul(tmp, kvHeads, tmp) || !CheckedMul(tmp, headDimPad, vElements)) {
        return Fail("quantized tensor element count overflow");
    }
    uint64_t qScaleElements = 0U;
    uint64_t kScaleElements = 0U;
    uint64_t channelElements = 0U;
    if (!CheckedMul(batch, qHeads, tmp) || !CheckedMul(tmp, qBlocks, tmp) || !CheckedMul(tmp, qGroups, qScaleElements) ||
        !CheckedMul(batch, kvHeads, tmp) || !CheckedMul(tmp, CeilDiv(kvSeq, kKQuantBlock), tmp) ||
        !CheckedMul(tmp, kGroups, kScaleElements) || !CheckedMul(batch, kvHeads, tmp) ||
        !CheckedMul(tmp, headDimPad, channelElements)) {
        return Fail("scale element count overflow");
    }

    uint64_t cursor = 0U;
    uint64_t q8Offset = 0U;
    uint64_t k8Offset = 0U;
    uint64_t v8Offset = 0U;
    uint64_t qScaleOffset = 0U;
    uint64_t kScaleOffset = 0U;
    uint64_t vScaleOffset = 0U;
    uint64_t kMeanOffset = 0U;
    uint64_t vMeanOffset = 0U;
    uint64_t bytes = 0U;
    if (!AppendWorkspace(qElements, cursor, q8Offset) || !AppendWorkspace(kElements, cursor, k8Offset) ||
        !AppendWorkspace(vElements, cursor, v8Offset) || !CheckedMul(qScaleElements, sizeof(float), bytes) ||
        !AppendWorkspace(bytes, cursor, qScaleOffset) || !CheckedMul(kScaleElements, sizeof(float), bytes) ||
        !AppendWorkspace(bytes, cursor, kScaleOffset) || !CheckedMul(channelElements, sizeof(float), bytes) ||
        !AppendWorkspace(bytes, cursor, vScaleOffset) ||
        !CheckedMul(channelElements, smoothK ? sizeof(uint16_t) : 0U, bytes) ||
        !AppendWorkspace(bytes, cursor, kMeanOffset) ||
        !CheckedMul(channelElements, smoothV ? sizeof(float) : 0U, bytes) ||
        !AppendWorkspace(bytes, cursor, vMeanOffset)) {
        return Fail("workspace size overflow");
    }
    cursor = (cursor + kWorkspaceAlign - 1ULL) / kWorkspaceAlign * kWorkspaceAlign;
    if (cursor > std::numeric_limits<size_t>::max()) {
        return Fail("workspace does not fit size_t");
    }

    SageAttention2TilingData tiling;
    tiling.shape.set_batch(batch);
    tiling.shape.set_qHeads(qHeads);
    tiling.shape.set_kvHeads(kvHeads);
    tiling.shape.set_qSeq(qSeq);
    tiling.shape.set_kvSeq(kvSeq);
    tiling.shape.set_headDim(headDim);
    tiling.shape.set_headDimPad(headDimPad);
    tiling.shape.set_gqaRatio(qHeads / kvHeads);
    tiling.shape.set_qBlocks(qBlocks);
    tiling.shape.set_kvBlocks(kvBlocks);
    tiling.shape.set_kvSeqPad(kvSeqPad);

    const uint32_t dtypeCode = qDesc->GetDataType() == ge::DT_BF16 ? 1U : 0U;
    tiling.attrs.set_layout(static_cast<uint32_t>(layout));
    tiling.attrs.set_causal(causal ? 1U : 0U);
    tiling.attrs.set_quantGran(static_cast<uint32_t>(gran));
    tiling.attrs.set_accumMode(static_cast<uint32_t>(accum));
    tiling.attrs.set_smoothK(smoothK ? 1U : 0U);
    tiling.attrs.set_smoothV(smoothV ? 1U : 0U);
    tiling.attrs.set_returnLse(returnLse ? 1U : 0U);
    tiling.attrs.set_inputDtype(dtypeCode);
    tiling.attrs.set_smScale(smScale);
    tiling.attrs.set_vScaleMax(accum == SageAccumMode::FP32_FP16 ? 2.25F : 448.0F);
    tiling.attrs.set_quantEpsilon(kQuantEpsilon);
    tiling.attrs.set_accumFlushInterval(accum == SageAccumMode::FP32 ? kvBlocks : 4U);

    uint64_t attentionTasks = 0U;
    if (!CheckedMul(batch, qHeads, tmp) || !CheckedMul(tmp, qBlocks, attentionTasks) ||
        attentionTasks > std::numeric_limits<uint32_t>::max()) {
        return Fail("attention task count overflow");
    }
    tiling.cores.set_coreNum(coreNum);
    tiling.cores.set_vecCoreNum(vecCoreNum);
    tiling.cores.set_totalAttentionTasks(static_cast<uint32_t>(attentionTasks));
    tiling.cores.set_qTile(kQTile);
    tiling.cores.set_kvTile(kKvTile);
    tiling.cores.set_qScaleGroupsPerBlock(qGroups);
    tiling.cores.set_kScaleGroupsPerBlock(kGroups);

    tiling.workspace.set_q8Offset(q8Offset);
    tiling.workspace.set_k8Offset(k8Offset);
    tiling.workspace.set_v8Offset(v8Offset);
    tiling.workspace.set_qScaleOffset(qScaleOffset);
    tiling.workspace.set_kScaleOffset(kScaleOffset);
    tiling.workspace.set_vScaleOffset(vScaleOffset);
    tiling.workspace.set_kMeanOffset(kMeanOffset);
    tiling.workspace.set_vMeanOffset(vMeanOffset);
    tiling.workspace.set_workspaceBytes(cursor);

    const uint64_t tilingKey = static_cast<uint64_t>(dtypeCode) + 2ULL * static_cast<uint32_t>(layout) +
        4ULL * (headDimPad == 128U ? 1ULL : 0ULL) + 8ULL * (causal ? 1ULL : 0ULL) +
        16ULL * static_cast<uint32_t>(gran) + 32ULL * static_cast<uint32_t>(accum) +
        128ULL * (smoothK ? 1ULL : 0ULL) + 256ULL * (smoothV ? 1ULL : 0ULL) +
        512ULL * (returnLse ? 1ULL : 0ULL);

    size_t *workspaceSizes = context->GetWorkspaceSizes(1U);
    auto *raw = context->GetRawTilingData();
    if (workspaceSizes == nullptr || raw == nullptr) {
        return Fail("workspace sizes or raw tiling buffer is nullptr");
    }
    const uint64_t systemWorkspaceBytes = static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize());
    uint64_t totalWorkspaceBytes = 0U;
    if (!CheckedAdd(systemWorkspaceBytes, cursor, totalWorkspaceBytes) ||
        totalWorkspaceBytes > std::numeric_limits<size_t>::max()) {
        return Fail("system and user workspace size overflow");
    }
    workspaceSizes[0] = static_cast<size_t>(totalWorkspaceBytes);
    // Driver 25.7.rc1.6 on the current 950PR instance is stable up to 20 MIX blocks; 21 and above stall.
    // Keep an explicit override for isolated launch probes while deriving the candidate launch from active cores.
    const uint32_t blockDim = SAGE_ATTENTION2_PROBE_BLOCK_DIM == 0U ?
        platform.CalcTschBlockDim(vecCoreNum, coreNum, vecCoreNum) : SAGE_ATTENTION2_PROBE_BLOCK_DIM;
    context->SetBlockDim(blockDim); // KERNEL_TYPE_MIX_AIC_1_2 launch geometry.
    context->SetScheduleMode(SAGE_ATTENTION2_PROBE_SCHEDULE_MODE);
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());
    OP_LOGD(kOpName,
        "B=%u Hq=%u Hkv=%u Sq=%u Sk=%u D=%u Dpad=%u key=%lu cores=%u blockDim=%u userWorkspace=%lu "
        "totalWorkspace=%lu",
        batch, qHeads, kvHeads, qSeq, kvSeq, headDim, headDimPad, tilingKey, coreNum, blockDim, cursor,
        totalWorkspaceBytes);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SageAttention2).Tiling(TilingSageAttention2);
} // namespace optiling
