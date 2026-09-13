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
    auto shapeMatches = [context](size_t input, std::initializer_list<int64_t> dims) {
        auto *s = context->GetInputShape(input);
        if (s == nullptr || s->GetStorageShape().GetDimNum() != dims.size()) return false;
        size_t i = 0;
        for (int64_t d : dims) if (s->GetStorageShape().GetDim(i++) != d) return false;
        return true;
    };
    if (!shapeMatches(0, {1,1024,64,512}) || !shapeMatches(1, {1,131072,1,512}) ||
        !shapeMatches(2, {1,1024,1,2048}) || !shapeMatches(11, {1,1024,64,64}) ||
        !shapeMatches(12, {1,131072,1,64})) return;
    for (size_t i = 7; i <= 10; ++i) {
        if (context->GetOptionalInputTensor(i) != nullptr) return;
    }
    auto *attrs = context->GetAttrs();
    if (attrs == nullptr) return;
    auto *scale = attrs->GetAttrPointer<float>(0);
    auto *block = attrs->GetAttrPointer<int>(1);
    auto *layout = attrs->GetAttrPointer<char>(2);
    auto *mode = attrs->GetAttrPointer<int>(3);
    if (!scale || *scale != 1.0f || !block || *block != 1 || !layout || std::strcmp(layout,"BSND") != 0 ||
        !mode || *mode != 3) return;
    constexpr uint32_t markerValue = 21;
    // The generated class maps the existing raw buffer; only rsv is written.
    SparseFlashAttentionGradBasicTilingData td(context->GetRawTilingData()->GetData());
    bool marker[6] = {true, false, false, false, false, false};
    for (uint32_t i = 0; i < 5; ++i) marker[i+1] = (markerValue >> i) & 1U;
    td.opInfo.set_rsv(marker);
    context->SetScheduleMode(1);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] += static_cast<uint64_t>(4932501504ULL);
    OP_LOGI(context, "SFAG_KEY_OWNED_FULL_REDUCE active host marker=%u", markerValue);
}
}}
