#pragma once
#include "../sparse_flash_attention_grad_tiling.h"
#include "../sparse_flash_attention_grad_tiling_common.h"
#include "platform/platform_infos_def.h"
#include "platform/soc_spec.h"
#include <cstring>
#include <string>
namespace optiling { namespace sfag {
inline void TryKeyOwnedFullReduce(gert::TilingContext *context)
{
    auto *platform = context->GetPlatformInfo();
    if (platform == nullptr) return;
    std::string soc;
    bool found = platform->GetPlatformRes("version", "SoC_version", soc);
    OP_LOGI(context, "SFAG_KEY_OWNED_FULL_REDUCE platform found=%d SoC_version=%s", found, soc.c_str());
    const bool supportedSoc = soc == "Ascend910B1" || soc == "Ascend910B2" ||
                              soc == "Ascend910B2C" || soc == "Ascend910B3";
    if (!found || !supportedSoc) return;
    platform_ascendc::PlatformAscendC hw(platform);
    const uint32_t aicNum = hw.GetCoreNumAic();
    const uint32_t aivNum = hw.GetCoreNumAiv();
    const bool supportedCoreCount = (aicNum == 20 && aivNum == 40) || (aicNum == 24 && aivNum == 48);
    if (hw.GetCurNpuArch() != NpuArch::DAV_2201 || !supportedCoreCount) return;
    if (context->GetTilingKey() != 111111 || context->GetDeterministic() != 1) return;
    auto *qdesc = context->GetInputDesc(0);
    if (qdesc == nullptr || qdesc->GetDataType() != ge::DT_FLOAT16) return;
    auto *queryShape = context->GetInputShape(0);
    auto *keyShape = context->GetInputShape(1);
    auto *indexShape = context->GetInputShape(2);
    // The tiling context exposes supplied optional tensor descriptors through
    // GetInputShape; GetOptionalInputShape is not populated consistently for
    // these two rope inputs on the arch22 path.
    auto *queryRopeShape = context->GetInputShape(11);
    auto *keyRopeShape = context->GetInputShape(12);
    if (queryShape == nullptr || keyShape == nullptr || indexShape == nullptr || queryRopeShape == nullptr ||
        keyRopeShape == nullptr) return;
    const auto &q = queryShape->GetStorageShape();
    const auto &k = keyShape->GetStorageShape();
    const auto &indices = indexShape->GetStorageShape();
    const auto &qr = queryRopeShape->GetStorageShape();
    const auto &kr = keyRopeShape->GetStorageShape();
    if (q.GetDimNum() != 4 || k.GetDimNum() != 4 || indices.GetDimNum() != 4 || qr.GetDimNum() != 4 ||
        kr.GetDimNum() != 4) return;
    const int64_t b = q.GetDim(0);
    const int64_t s1 = q.GetDim(1);
    const int64_t g = q.GetDim(2);
    const int64_t d = q.GetDim(3);
    const int64_t s2 = k.GetDim(1);
    const int64_t d2 = k.GetDim(3);
    const int64_t selected = indices.GetDim(3);
    const int64_t ropeD = qr.GetDim(3);
    if (b <= 0 || s1 <= 0 || s2 <= 0 || g <= 0 || d <= 0 || d2 <= 0 || selected <= 0 || ropeD <= 0 ||
        k.GetDim(0) != b || k.GetDim(2) != 1 || k.GetDim(3) != d || indices.GetDim(0) != b ||
        indices.GetDim(1) != s1 || indices.GetDim(2) != 1 || qr.GetDim(0) != b || qr.GetDim(1) != s1 ||
        qr.GetDim(2) != g || kr.GetDim(0) != b || kr.GetDim(1) != s2 || kr.GetDim(2) != 1 || kr.GetDim(3) != ropeD)
        return;
    const uint64_t queryCount = static_cast<uint64_t>(b) * s1;
    const uint64_t dTotal = static_cast<uint64_t>(d) + ropeD;
    const uint64_t bucketCount = (static_cast<uint64_t>(s2) + 63) / 64;
    if (queryCount > 65536 || selected > 2048 || bucketCount > 2048 || g > 64 || dTotal > 576 || d2 > 512 ||
        (dTotal & 15U) != 0 || (d2 & 15U) != 0) return;
    for (size_t i = 7; i <= 10; ++i) {
        if (context->GetOptionalInputTensor(i) != nullptr) return;
    }
    auto *attrs = context->GetAttrs();
    if (attrs == nullptr) return;
    auto *block = attrs->GetAttrPointer<int>(1);
    auto *layout = attrs->GetAttrPointer<char>(2);
    auto *mode = attrs->GetAttrPointer<int>(3);
    if (!attrs->GetAttrPointer<float>(0) || !block || *block != 1 || !layout || std::strcmp(layout,"BSND") != 0 ||
        !mode || *mode != 3) return;
    constexpr uint32_t markerValue = 21;
    // The generated class maps the existing raw buffer; only rsv is written.
    SparseFlashAttentionGradBasicTilingData td(context->GetRawTilingData()->GetData());
    bool marker[6] = {true, false, false, false, false, false};
    for (uint32_t i = 0; i < 5; ++i) marker[i+1] = (markerValue >> i) & 1U;
    td.opInfo.set_rsv(marker);
    context->SetScheduleMode(1);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    const uint64_t dAlign = (static_cast<uint64_t>(d) + ropeD + 15) / 16 * 16;
    const uint64_t selectedCount = static_cast<uint64_t>(selected);
    const uint64_t edgeBytes = queryCount * selectedCount * dAlign * sizeof(float);
    const uint64_t headerBytes = bucketCount * queryCount * 4 * sizeof(uint32_t);
    const uint64_t sortedBytes = queryCount * selectedCount * sizeof(int32_t);
    workspaces[0] += edgeBytes + headerBytes + sortedBytes;
    OP_LOGI(context, "SFAG_KEY_OWNED_FULL_REDUCE active host marker=%u", markerValue);
}
}}
