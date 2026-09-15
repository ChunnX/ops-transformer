/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_mla_tiling.h"
#include "flash_mla_platform.h"
#include "flash_mla_shape.h"
#include "flash_mla_value_tiling.h"
#include "flash_mla_sequence_policy.h"
#include "arch22/flash_mla_static_tiling.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include <algorithm>

namespace optiling {
namespace {
using namespace FlashMla;
ge::graphStatus StaticTilingError(const char *stage)
{
    OP_LOGE("FlashMla", "Static tiling validation failed at %s.", stage);
    return ge::GRAPH_FAILED;
}
// Platform quotas are queried for each ACLNN invocation, so compile parsing
// carries no resource snapshot or input values.
struct MlaCompileInfo {};
ge::graphStatus ParseMlaCompileInfo(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}
const gert::Shape *ShapeAt(gert::TilingContext *context, size_t index)
{
    auto shape = context->GetOptionalInputShape(index);
    return shape == nullptr ? nullptr : &shape->GetStorageShape();
}
bool ValidShape(const gert::Shape *shape)
{
    if (shape == nullptr) {
        return false;
    }
    uint64_t elements = 1;
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        const auto dim = shape->GetDim(i);
        if (dim < 0 || dim > UINT32_MAX || !MlaCheckedMul(elements, dim, elements)) {
            return false;
        }
    }
    return true;
}
bool CheckSeq(gert::TilingContext *context, size_t index, uint32_t batch, bool accum, uint32_t &present,
              uint64_t &count, bool checkBatchCount = true)
{
    auto shape = ShapeAt(context, index);
    present = shape != nullptr;
    count = 0;
    if (shape == nullptr) {
        return !accum || batch == 0;
    }
    auto desc = context->GetOptionalInputDesc(index);
    if (desc == nullptr || desc->GetDataType() != ge::DT_INT64 || shape->GetDimNum() != 1 || shape->GetDim(0) < 0 ||
        (checkBatchCount && !ValidShape(shape))) {
        return false;
    }
    count = shape->GetDim(0);
    present = count != 0;
    return accum ? count == batch : (!checkBatchCount || count == 0 || count == 1 || count >= batch);
}
TilingKeyLayout KernelLayout(const std::string &layout)
{
    if (layout == "BNSD") {
        return TilingKeyLayout::BNSD;
    }
    if (layout == "TND") {
        return TilingKeyLayout::TND;
    }
    if (layout == "NZ") {
        return TilingKeyLayout::NZ;
    }
    if (layout == "NBSD") {
        return TilingKeyLayout::NBSD;
    }
    if (layout == "NTD") {
        return TilingKeyLayout::NTD;
    }
    return TilingKeyLayout::BSH_BSND;
}
FiaLayout OriginalLayout(const std::string &layout, bool pa = false)
{
    if (layout == "BSH") {
        return pa ? FiaLayout::BnBsH : FiaLayout::BSH;
    }
    if (layout == "BSND") {
        return FiaLayout::BSND;
    }
    if (layout == "BNSD") {
        return pa ? FiaLayout::BnNBsD : FiaLayout::BNSD;
    }
    if (layout == "TND") {
        return FiaLayout::TND;
    }
    if (layout == "NZ") {
        return FiaLayout::NZ;
    }
    if (layout == "NBSD") {
        return FiaLayout::NBSD;
    }
    return FiaLayout::NTD;
}
bool CacheShape(const gert::Shape &shape, const std::string &layout, bool pa, uint32_t block, uint32_t dim, bool quant,
                MlaAxes &axes)
{
    if (layout != "NZ") {
        return GetMlaAxes(shape, layout, 1, axes) && axes.dim == dim && (!pa || axes.seq == block);
    }
    if (shape.GetDimNum() != 5) {
        return false;
    }
    const uint32_t d0 = quant ? 32 : 16;
    if (!MlaShapeEquals(shape, {shape.GetDim(0), 1, dim / d0, block, d0})) {
        return false;
    }
    axes = {shape.GetDim(0), block, 1, dim};
    return true;
}
const gert::Stride *UsableCacheStride(const gert::Stride *stride)
{
    return stride != nullptr && stride->GetDimNum() != 0 ? stride : nullptr;
}
bool CacheStride(const char *name, const gert::Shape &shape, const gert::Stride *stride, bool allowDim0,
                 uint64_t &bnStride, int32_t &nonContiguousDim)
{
    uint64_t expected = 1;
    bnStride = 0;
    nonContiguousDim = -1;
    if (stride != nullptr && stride->GetDimNum() != shape.GetDimNum()) {
        OP_LOGE("FlashMla", "%s stride rank=%zu differs from shape rank=%zu.", name, stride->GetDimNum(),
                shape.GetDimNum());
        return false;
    }
    for (size_t d = shape.GetDimNum(); d > 0; --d) {
        size_t i = d - 1;
        const int64_t actual = stride == nullptr ? static_cast<int64_t>(expected) : stride->GetStride(i);
        if (shape.GetDim(i) > 1 &&
            (actual < 0 || (actual != static_cast<int64_t>(expected) && !(allowDim0 && i == 0)))) {
            OP_LOGE("FlashMla", "%s stride at axis=%zu is %ld, expected=%lu.", name, i, actual, expected);
            return false;
        }
        if (i == 0) {
            if (allowDim0 && shape.GetDim(i) > 1 && actual == 0) {
                return false;
            }
            bnStride = actual;
            if (actual != static_cast<int64_t>(expected) && shape.GetDim(0) > 1) {
                nonContiguousDim = 0;
            }
        }
        if (!MlaCheckedMul(expected, std::max<int64_t>(shape.GetDim(i), 1), expected)) {
            return false;
        }
    }
    return true;
}
bool FillMask(gert::TilingContext *context, MlaScheduleConfig &config, MlaStaticLaunchInfo &launch, FiaTilingInfo &info)
{
    const auto mask = ShapeAt(context, 3);
    config.hasAttenMask = mask != nullptr && mask->GetShapeSize() != 0;
    info.attenMaskFlag = config.hasAttenMask;
    if (!config.hasAttenMask) {
        return config.sparseMode == 0;
    }
    auto desc = context->GetOptionalInputDesc(3);
    if (!ValidShape(mask) || desc == nullptr ||
        (desc->GetDataType() != ge::DT_BOOL && desc->GetDataType() != ge::DT_INT8 &&
         desc->GetDataType() != ge::DT_UINT8)) {
        return false;
    }
    auto rank = mask->GetDimNum();
    // Original nonquant MLA always has split RoPE. Its ordinary-mask checker
    // rejects rank 2; compressed sparse3/4 masks retain their 2D support.
    if (config.queryDataType != kMlaInt8 && rank == 2 && (config.sparseMode == 0 || config.sparseMode == 1)) {
        return false;
    }
    if (config.queryDataType != kMlaInt8 && config.sparseMode == 9) {
        // The active FIA tree checker accepts ranks 1 and 3 without checking
        // storage extents. Match its parser's strides, including cross-layout ranks.
        if (rank != 1 && rank != 3) {
            return false;
        }
        launch.maskRank = rank;
        launch.maskDim0 = mask->GetDim(0);
        if (rank == 3) {
            launch.maskDim1 = mask->GetDim(1);
            launch.maskDim2 = mask->GetDim(2);
        }
        uint64_t stride = 0;
        if (config.layoutQ == MlaLayout::TND && rank == 1) {
            stride = 1;
            info.attenMaskStride = 1;
        } else if (config.layoutQ != MlaLayout::TND && rank == 3) {
            if (!MlaCheckedMul(launch.maskDim1, launch.maskDim2, stride) || stride > UINT32_MAX) {
                return false;
            }
            info.attenMaskStride = launch.maskDim2;
        }
        launch.maskBatchStride = launch.maskBatchStrideAlt = stride;
        info.attenMaskBatchStride = stride;
        return true;
    }
    if (config.queryDataType == kMlaInt8 && config.sparseMode == 3 && rank != 2) {
        return false;
    }
    if (config.sparseMode == 9 && config.layoutQ == MlaLayout::TND) {
        if (rank != 1) {
            return false;
        }
        launch.maskRank = 1;
        launch.maskDim0 = mask->GetDim(0);
        launch.maskBatchStride = launch.maskBatchStrideAlt = 1;
        info.attenMaskBatchStride = info.attenMaskStride = 1;
        return true;
    }
    if (config.sparseMode == 9 && (rank != 3 || mask->GetDim(0) != config.batchSize)) {
        return false;
    }
    if (rank < 2 || rank > 4 || (rank == 4 && mask->GetDim(1) != 1)) {
        return false;
    }
    launch.maskRank = rank;
    launch.maskDim0 = mask->GetDim(0);
    launch.maskDim1 = mask->GetDim(rank == 4 ? 2 : 1);
    launch.maskDim2 = rank >= 3 ? mask->GetDim(rank - 1) : 0;
    if (rank >= 3 && mask->GetDim(0) != 1 && mask->GetDim(0) != config.batchSize) {
        return false;
    }
    if (config.sparseMode != 0 && config.sparseMode != 1 && config.sparseMode != 9 &&
        ((rank >= 3 && mask->GetDim(0) != 1) || mask->GetDim(rank - 2) != 2048 || mask->GetDim(rank - 1) != 2048)) {
        return false;
    }
    // Only KV extent and TND maximum Q depend on device sequence values.
    if ((config.sparseMode == 0 || config.sparseMode == 1) && config.layoutQ != MlaLayout::TND &&
        static_cast<uint64_t>(mask->GetDim(rank - 2)) < config.qSeqSize) {
        return false;
    }
    uint64_t stride = 0;
    if (rank == 2) {
        launch.maskBatchStride = mask->GetDim(1);
        launch.maskBatchStrideAlt = 0;
    } else {
        if (mask->GetDim(0) == config.batchSize &&
            !MlaCheckedMul(mask->GetDim(rank - 2), mask->GetDim(rank - 1), stride)) {
            return false;
        }
        launch.maskBatchStride = stride;
        launch.maskBatchStrideAlt = stride;
    }
    if (launch.maskBatchStride > UINT32_MAX) {
        return false;
    }
    info.attenMaskBatchStride = launch.maskBatchStride;
    info.attenMaskStride = config.sparseMode == 0 || config.sparseMode == 1 ? mask->GetDim(rank - 1) : 2048;
    if (config.sparseMode == 9) {
        info.attenMaskStride = mask->GetDim(rank - 1);
    }
    return true;
}
bool CheckQuantScales(gert::TilingContext *context, const gert::Shape &qShape, const std::string &layout, int64_t heads)
{
    auto qs = ShapeAt(context, 11);
    auto ks = ShapeAt(context, 7);
    auto vs = ShapeAt(context, 8);
    if (!ValidShape(qs) || !ValidShape(ks) || !ValidShape(vs) || !MlaShapeEquals(*ks, {1}) ||
        !MlaShapeEquals(*vs, {1})) {
        return false;
    }
    // RoPE inputs separate the KV scales from the query scale in FIA order.
    for (size_t i : {7U, 8U, 11U}) {
        auto desc = context->GetOptionalInputDesc(i);
        if (desc == nullptr || desc->GetDataType() != ge::DT_FLOAT) {
            return false;
        }
    }
    std::vector<int64_t> expected;
    if (layout == "BSH") {
        expected = {qShape.GetDim(0), qShape.GetDim(1), heads};
    } else {
        for (size_t i = 0; i + 1 < qShape.GetDimNum(); ++i) {
            expected.push_back(qShape.GetDim(i));
        }
    }
    return MlaShapeEquals(*qs, expected);
}
} // namespace

ge::graphStatus PrepareMlaStaticTiling(gert::TilingContext *context, const FlashMla::MlaPlatformInfo &platform,
                                       FlashMla::MlaStaticTiling &tiling)
{
    using namespace FlashMla;
    if (context == nullptr || context->GetAttrs() == nullptr || platform.aicNum == 0 ||
        platform.aicNum > kMlaMaxAicNum ||
        (platform.aivNum != platform.aicNum && platform.aivNum != 2 * platform.aicNum)) {
        return StaticTilingError("context/platform quota");
    }
    tiling = {};
    MlaScheduleConfig config;
    FiaTilingInfo info;
    auto &launch = tiling.launch;
    auto attrs = context->GetAttrs();
    auto heads = attrs->GetAttrPointer<int64_t>(0);
    auto scale = attrs->GetAttrPointer<float>(1);
    auto pre = attrs->GetAttrPointer<int64_t>(2);
    auto next = attrs->GetAttrPointer<int64_t>(3);
    auto layout = attrs->GetAttrPointer<char>(4);
    auto kvHeads = attrs->GetAttrPointer<int64_t>(5);
    auto sparse = attrs->GetAttrPointer<int64_t>(6);
    auto precise = attrs->GetAttrPointer<int64_t>(7);
    auto block = attrs->GetAttrPointer<int64_t>(8);
    auto lse = attrs->GetAttrPointer<bool>(9);
    auto km = attrs->GetAttrPointer<int64_t>(10);
    auto vm = attrs->GetAttrPointer<int64_t>(11);
    auto qm = attrs->GetAttrPointer<int64_t>(12);
    if (heads == nullptr || scale == nullptr || pre == nullptr || next == nullptr || kvHeads == nullptr ||
        sparse == nullptr || precise == nullptr || block == nullptr || lse == nullptr || km == nullptr ||
        vm == nullptr || qm == nullptr || *heads < 1 || *heads > 128 ||
        (*kvHeads != 1 && !(*kvHeads == 0 && *heads == 1)) || *block < 0 || *block > INT32_MAX || *sparse < 0 ||
        *sparse > 9 || *precise < 0 || *precise > 3) {
        return StaticTilingError("attributes");
    }
    const auto qShape = ShapeAt(context, 0);
    const auto kStorage = context->GetDynamicInputShape(1, 0);
    const auto vStorage = context->GetDynamicInputShape(2, 0);
    const auto qrShape = ShapeAt(context, 9);
    const auto krShape = ShapeAt(context, 10);
    const auto meta = ShapeAt(context, 12);
    const auto qDesc = context->GetInputDesc(0);
    const auto kDesc = context->GetDynamicInputDesc(1, 0);
    const auto vDesc = context->GetDynamicInputDesc(2, 0);
    // Optional inputs before RoPE may be absent, so resolve descriptors by IR index.
    const auto qrDesc = context->GetRequiredInputDesc(9);
    const auto krDesc = context->GetRequiredInputDesc(10);
    if (!ValidShape(qShape) || !ValidShape(qrShape) || !ValidShape(krShape) || kStorage == nullptr ||
        vStorage == nullptr || context->GetDynamicInputShape(1, 1) != nullptr ||
        context->GetDynamicInputShape(2, 1) != nullptr || qDesc == nullptr || kDesc == nullptr || vDesc == nullptr ||
        qrDesc == nullptr || krDesc == nullptr || !ValidShape(meta) || !MlaShapeEquals(*meta, {kMlaMetadataElements}) ||
        (context->GetRequiredInputDesc(12) == nullptr ||
         context->GetRequiredInputDesc(12)->GetDataType() != ge::DT_INT32)) {
        OP_LOGE("FlashMla",
                "Required descriptors: Q=%d QR=%d KR=%d K=%d V=%d extraK=%d extraV=%d metadata=%d metadataType=%d.",
                ValidShape(qShape), ValidShape(qrShape), ValidShape(krShape), kStorage != nullptr, vStorage != nullptr,
                context->GetDynamicInputShape(1, 1) != nullptr, context->GetDynamicInputShape(2, 1) != nullptr,
                ValidShape(meta),
                context->GetRequiredInputDesc(12) == nullptr ?
                    -1 :
                    static_cast<int>(context->GetRequiredInputDesc(12)->GetDataType()));
        return StaticTilingError("required input descriptors");
    }
    const auto &kShape = kStorage->GetStorageShape();
    const auto &vShape = vStorage->GetStorageShape();
    const bool quant = qDesc->GetDataType() == ge::DT_INT8;
    const auto outputType = quant ? ge::DT_BF16 : qDesc->GetDataType();
    if ((!quant && qDesc->GetDataType() != ge::DT_BF16 && qDesc->GetDataType() != ge::DT_FLOAT16) ||
        kDesc->GetDataType() != qDesc->GetDataType() || vDesc->GetDataType() != qDesc->GetDataType() ||
        qrDesc->GetDataType() != outputType || krDesc->GetDataType() != outputType ||
        context->GetOutputDesc(0) == nullptr || context->GetOutputDesc(0)->GetDataType() != outputType ||
        context->GetOutputDesc(1) == nullptr || context->GetOutputDesc(1)->GetDataType() != ge::DT_FLOAT || *km != 0 ||
        *vm != 0 || *qm != (quant ? 3 : 0)) {
        return StaticTilingError("input/output dtype and quant modes");
    }
    std::string qLayout, outLayout;
    MlaAxes q, qr, k, v, kr;
    if (!SplitMlaLayout(layout, qLayout, outLayout) || !GetMlaAxes(*qShape, qLayout, *heads, q) ||
        !GetMlaAxes(*qrShape, qLayout, *heads, qr) || q.dim != 512 || qr.dim != 64 || q.batch != qr.batch ||
        q.seq != qr.seq) {
        return StaticTilingError("query/RoPE layout and axes");
    }
    if (qLayout == "TND") {
        const auto sq = ShapeAt(context, 4);
        if (!ValidShape(sq) || sq->GetDimNum() != 1) {
            return StaticTilingError("TND sequence descriptor");
        }
        q.batch = sq->GetDim(0);
    }
    if (q.batch > 65536) {
        return StaticTilingError("logical batch limit");
    }
    config.numHeads = *heads;
    config.batchSize = q.batch;
    config.qSeqSize = q.seq;
    ParseMlaLayout(qLayout.c_str(), config.layoutQ);
    config.preTokens = *pre;
    config.nextTokens = *next;
    config.sparseMode = *sparse;
    config.queryDataType = quant ? kMlaInt8 : (qDesc->GetDataType() == ge::DT_BF16 ? kMlaBf16 : kMlaFp16);
    auto blocks = ShapeAt(context, 6);
    const bool pa = blocks != nullptr;
    std::string kvLayout = qLayout;
    if (pa) {
        config.blockSize = *block;
        auto blockDesc = context->GetOptionalInputDesc(6);
        if (!ValidShape(blocks) || blocks->GetDimNum() != 2 || blocks->GetDim(0) != q.batch || blockDesc == nullptr ||
            blockDesc->GetDataType() != ge::DT_INT32 || *block == 0 || *block > 1024 || *block % 16 != 0 ||
            !MlaCheckedMul(blocks->GetDim(1), *block, config.kvSeqSize) || config.kvSeqSize > UINT32_MAX) {
            return StaticTilingError("page block table");
        }
        kvLayout = kShape.GetDimNum() == 3 ? "BSH" : (kShape.GetDimNum() == 4 ? "BNSD" : "NZ");
    }
    if (!ValidShape(&kShape) || !ValidShape(&vShape) ||
        !CacheShape(kShape, kvLayout, pa, config.blockSize, 512, quant, k) ||
        !CacheShape(vShape, kvLayout, pa, config.blockSize, 512, quant, v) ||
        !CacheShape(*krShape, kvLayout, pa, config.blockSize, 64, false, kr) || k.batch != v.batch ||
        k.batch != kr.batch || k.seq != v.seq || k.seq != kr.seq || (!pa && qLayout != "TND" && k.batch != q.batch)) {
        return StaticTilingError("KV/RoPE cache shapes");
    }
    if (!pa) {
        config.kvSeqSize = k.seq;
    }
    ParseMlaLayout(kvLayout.c_str(), config.layoutKv);
    if (quant && (*precise > 1 || !pa || kvLayout != "NZ" || config.blockSize != 128 || (*heads & (*heads - 1)) != 0 ||
                  q.seq > (qLayout == "TND" ? 1024 * 1024 : 32) || (kvLayout == "NZ" && qLayout == "BNSD") ||
                  !CheckQuantScales(context, *qShape, qLayout, *heads))) {
        return StaticTilingError("quant layout/scales");
    }
    if (!quant &&
        (ShapeAt(context, 11) != nullptr || ShapeAt(context, 7) != nullptr || ShapeAt(context, 8) != nullptr)) {
        return StaticTilingError("unexpected nonquant scales");
    }
    if (!CheckSeq(context, 4, config.batchSize, qLayout == "TND", config.seqQPresent, config.seqQCount,
                  !IsMlaQuerySeqIgnored(config.queryDataType, qLayout == "TND")) ||
        !CheckSeq(context, 5, config.batchSize, qLayout == "TND", config.seqKvPresent, config.seqKvCount) ||
        ((pa || qLayout == "TND") && (!config.seqKvPresent || config.seqKvCount == 0) && config.batchSize != 0)) {
        return StaticTilingError("sequence descriptors");
    }
    const bool emptyInput = qShape->GetShapeSize() == 0 || kShape.GetShapeSize() == 0 || vShape.GetShapeSize() == 0;
    // The nonquant FIA RopeChecker requires the explicit KV head attribute.
    // Legacy IFA normalizes zero to numHeads; both bypass this check for empty inputs.
    if (!emptyInput && !quant && *kvHeads != 1) {
        return StaticTilingError("KV head attribute");
    }
    // FIA skips its PA checker for empty inputs; otherwise both table axes must be positive.
    if (pa && !emptyInput && blocks->GetShapeSize() == 0) {
        return StaticTilingError("page block table");
    }
    if (!FillMask(context, config, launch, info)) {
        return StaticTilingError("attention mask");
    }
    const auto normalizeStatus = NormalizeMlaScheduleConfig(config);
    if (normalizeStatus != MlaStatus::SUCCESS) {
        OP_LOGE("FlashMla", "Schedule normalization status=%u.", static_cast<uint32_t>(normalizeStatus));
        return StaticTilingError("schedule normalization");
    }
    if (ValidateMlaSparseConfig(config) != MlaStatus::SUCCESS) {
        return StaticTilingError("sparse mode/mask/query length");
    }
    auto outShape = context->GetOutputShape(0);
    auto lseShape = context->GetOutputShape(1);
    if (outShape == nullptr || lseShape == nullptr ||
        !MlaShapeEquals(outShape->GetStorageShape(), MlaOutputShape(q, outLayout)) ||
        (*lse && !MlaShapeEquals(lseShape->GetStorageShape(), qLayout == "TND" ?
                                                                  std::vector<int64_t>{q.seq, q.heads, 1} :
                                                                  std::vector<int64_t>{q.batch, q.heads, q.seq, 1}))) {
        return StaticTilingError("output/LSE shape");
    }
    launch.family = SelectMlaFamily(config);
    launch.cvRatio = platform.aivNum / platform.aicNum;
    launch.aicNum = platform.aicNum;
    launch.aivNum = platform.aivNum;
    launch.queryDataType = config.queryDataType;
    launch.layoutQ = config.layoutQ;
    launch.layoutKv = config.layoutKv;
    launch.pageAttention = pa;
    launch.treeMask = config.sparseMode == 9;
    launch.softmaxLseFlag = *lse;
    launch.qStorageSize = config.qSeqSize;
    launch.kvStorageSize = config.kvSeqSize;
    launch.outputElements = outShape->GetStorageShape().GetShapeSize();
    launch.lseElements = *lse ? lseShape->GetStorageShape().GetShapeSize() : 0;
    launch.sysWorkspaceSize = platform.sysWorkspaceSize;
    info.bSize = config.batchSize;
    info.n1Size = config.numHeads;
    info.n2Size = 1;
    info.qkHeadDim = 512;
    info.vHeadDim = 512;
    info.ropeHeadDim = 64;
    info.scaleValue = *scale;
    info.innerPrecise = *precise;
    info.actualLenQDims = config.seqQCount;
    info.actualLenKvDims = config.seqKvCount;
    info.isAccumQSeq = config.layoutQ == MlaLayout::TND;
    info.isAccumKVSeq = config.layoutKv == MlaLayout::TND;
    info.outputLayout = KernelLayout(outLayout);
    info.inputLayout = KernelLayout(qLayout);
    info.inputKvLayout = KernelLayout(kvLayout);
    info.qLayout = OriginalLayout(qLayout);
    info.kvLayout = OriginalLayout(kvLayout, pa);
    info.outLayout = OriginalLayout(outLayout);
    info.kvStorageMode = pa ? KvStorageMode::PAGE_ATTENTION : KvStorageMode::BATCH_CONTINUOUS;
    info.blockSize = config.blockSize;
    info.softmaxLseFlag = *lse;
    info.sparseMode = config.sparseMode;
    info.preToken = config.preTokens;
    info.nextToken = config.nextTokens;
    info.slidingFlag = config.sparseMode == 4;
    info.keyStrides = UsableCacheStride(context->GetDynamicInputStride(1, 0));
    info.valueStrides = UsableCacheStride(context->GetDynamicInputStride(2, 0));
    info.kRopeStrides = UsableCacheStride(context->GetRequiredInputStride(10));
    // As in the original parser, a TensorV1/Contiguous result can expose an
    // empty stride object. With no usable cache strides the API has already
    // materialized dense tensors. Once any view stride is supplied, missing
    // metadata on another view is an error and must not silently become dense.
    const bool hasCacheStrides =
        info.keyStrides != nullptr || info.valueStrides != nullptr || info.kRopeStrides != nullptr;
    if (hasCacheStrides && ((context->DynamicInputIsView(1, 0) && info.keyStrides == nullptr) ||
                            (context->DynamicInputIsView(2, 0) && info.valueStrides == nullptr) ||
                            (context->RequiredInputIsView(10) && info.kRopeStrides == nullptr))) {
        return StaticTilingError("incomplete cache view strides");
    }
    if (!CacheStride("key", kShape, info.keyStrides, pa && !quant, info.keyBnStride, info.keyNonContigDim) ||
        !CacheStride("value", vShape, info.valueStrides, pa && !quant, info.valueBnStride, info.valueNonContigDim) ||
        !CacheStride("keyRope", *krShape, info.kRopeStrides, pa && !quant, info.kRopeBnStride,
                     info.keyRopeNonContigDim)) {
        return StaticTilingError("cache strides");
    }
    info.pageAttentionFlag = pa;
    // Static empty input follows the original vector initialization path and never consumes metadata.
    if (emptyInput) {
        launch.family = MlaFamily::NO_SCHEDULE;
        launch.physicalBlockDim = platform.aivNum;
        launch.workspaceCap = platform.sysWorkspaceSize;
    } else {
        platform_ascendc::PlatformAscendC ascend(context->GetPlatformInfo());
        launch.physicalBlockDim = ascend.CalcTschBlockDim(platform.aivNum, platform.aicNum, platform.aivNum);
        const auto capacityStatus = ComputeMlaWorkspaceCapacity(config, platform, launch.workspaceCap);
        if (capacityStatus != MlaStatus::SUCCESS) {
            OP_LOGE("FlashMla", "Workspace capacity status=%u.", static_cast<uint32_t>(capacityStatus));
            return StaticTilingError("workspace capacity");
        }
        if (launch.family == MlaFamily::ARCH22_MLA) {
            FusedInferAttentionScoreTilingData original;
            FlashMla::FillArch22MlaStaticTiling(info, original);
            // This is the block-table row stride, independent of actual KV
            // lengths. A shorter sequence must still index the next row using
            // the complete descriptor width, as in the original MLA parser.
            original.pageAttenParams.set_maxBlockNumPerBatch(pa ? blocks->GetDim(1) : 0);
            original.SaveToBuffer(tiling.payload, kMlaNonQuantPayloadBytes);
        } else {
            auto &base = reinterpret_cast<MlaLegacyTiling *>(tiling.payload)->baseParams;
            base.batchSize = config.batchSize;
            base.blockSize = config.blockSize;
            base.maxBlockNumPerBatch = config.blockSize == 0 ? 0 : config.kvSeqSize / config.blockSize;
            base.scaleValue = *scale;
            base.nNumOfQInOneGroup = config.numHeads;
            base.actualLenQDims =
                config.layoutQ == MlaLayout::TND ? std::min<uint64_t>(config.seqQCount, config.batchSize) : 0;
            base.actualLenDims = std::min<uint64_t>(config.seqKvCount, config.batchSize);
            base.antiquantMode = 0;
            base.attenMaskFlag = config.hasAttenMask;
            // Original IFA returns before the generic mask-size assignment for
            // every supported INT8 MLA combination (no mask, sparse3 or tree).
            base.attenMaskSize = 0;
            base.sparseMode = config.sparseMode;
            base.outputLayout = static_cast<uint32_t>(info.outputLayout);
            base.softmaxLseFlag = *lse;
            base.tSeqSize = config.layoutQ == MlaLayout::TND ? config.qSeqSize : 1;
        }
    }
    launch.payloadAbiId = MlaPayloadAbiId(launch.family);
    launch.platformSignature = MlaPlatformSignature(platform);
    auto signature = MlaScheduleSignature(config, platform);
    launch.scheduleSignatureLo = signature.lo;
    launch.scheduleSignatureHi = signature.hi;
    launch.staticLaunchKey = MakeMlaStaticLaunchKey(launch.family, config.queryDataType, config.layoutQ,
                                                    config.layoutKv, pa, launch.treeMask, launch.cvRatio);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareMlaStaticTiling(gert::TilingContext *context, FlashMla::MlaStaticTiling &tiling)
{
    FlashMla::MlaPlatformInfo platform;
    if (!GetMlaTilingPlatform(context, platform)) {
        return StaticTilingError("live platform resources");
    }
    return PrepareMlaStaticTiling(context, platform, tiling);
}

ge::graphStatus TilingFlashMla(gert::TilingContext *context)
{
    FlashMla::MlaStaticTiling tiling;
    if (PrepareMlaStaticTiling(context, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto data = context->GetTilingData<FlashMla::MlaStaticTiling>();
    auto workspace = context->GetWorkspaceSizes(1);
    if (data == nullptr || workspace == nullptr) {
        return StaticTilingError("framework tiling/workspace buffers");
    }
    *data = tiling;
    // BATCH_MODE: fixed physical launch for every legal sequence value.
    if (context->SetTilingKey(tiling.launch.staticLaunchKey) != ge::GRAPH_SUCCESS ||
        context->SetBlockDim(tiling.launch.physicalBlockDim) != ge::GRAPH_SUCCESS ||
        context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
        return StaticTilingError("framework launch attributes");
    }
    workspace[0] = tiling.launch.workspaceCap;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(FlashMla)
    .Tiling(TilingFlashMla, sizeof(FlashMla::MlaStaticTiling))
    .TilingParse<MlaCompileInfo>(ParseMlaCompileInfo);
} // namespace optiling
