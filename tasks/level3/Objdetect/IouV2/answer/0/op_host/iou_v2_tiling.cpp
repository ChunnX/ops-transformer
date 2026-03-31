/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file iou_v2_tiling.cpp
 * \brief
 */
#include <algorithm>
#include <cstdio>

#include "iou_v2_tiling.h"
// #include "log/log.h"
#define OP_LOGD(fmt, ...) \
    do { \
        std::printf(fmt, ##__VA_ARGS__); \
        std::printf("\n"); \
    } while (0)
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{

    static constexpr uint32_t FLOAT_DTYPE_KEY = 0;
    static constexpr uint32_t FLOAT16_DTYPE_KEY = 1;
    static constexpr uint32_t BFLOAT16_DTYPE_KEY = 2;
    static constexpr uint32_t ALIGNED_KEY = 4;
    static constexpr uint32_t UNALIGNED_KEY = 7;
    static constexpr uint32_t MODE_IOF_KEY = 10;
    static constexpr uint32_t MIN_SIZE_PER_CORE = 4096;
    static constexpr uint32_t FLOAT_SIZE = 4;
    static constexpr uint32_t HALF_SIZE = 2;
    static constexpr uint32_t BLOCK_SIZE = 32;

    static constexpr uint32_t ATTR_STR = 0;
    static constexpr uint32_t ATTR_FLOAT = 1;
    static constexpr uint32_t ATTR_BOOL = 2;

    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b)
    {
        T1 bTemp(b);
        return bTemp == 0 ? a : (a + bTemp - 1) / bTemp;
    };

    template <typename T1, typename T2>
    inline T1 CeilAlign(T1 a, T2 b)
    {
        T1 bTemp(b);
        return bTemp == 0 ? a : CeilDiv(a, bTemp) * bTemp;
    }

    template <typename T>
    inline T ClampSub(T a, T b)
    {
        return a <= b ? 0 : (a - b);
    }

    inline bool IsOutOfBound(uint64_t maxLen, uint64_t bBoxLen, uint64_t ubSize, bool isFloat)
    {
        if (bBoxLen > maxLen)
        {
            bBoxLen = maxLen;
        }
        if (isFloat)
        {
            return static_cast<uint64_t>(36) * maxLen * bBoxLen + BLOCK_SIZE * (maxLen + bBoxLen) > ubSize; // 36: 9个f32_size, 32: 8个f32_size
        }
        else
        {
            return static_cast<uint64_t>(38) * maxLen * bBoxLen + BLOCK_SIZE * (maxLen + bBoxLen) > ubSize; // 38: 9个f32_size + 1个f16_size, 32: 8个f32_size
        }
    }

    inline uint32_t GetTilingKey(const gert::TilingContext *context, uint64_t &dataSize)
    {
        // 根据数据类型（3种）、计算模式（2种）和align（2种），计算tilingKey（3*2*2=12种）
        uint32_t tilingKey = 0;
        // 1. 数据类型
        auto dyDataType = context->GetInputDesc(0)->GetDataType();
        if (ge::DT_FLOAT == dyDataType)
        {
            tilingKey += FLOAT_DTYPE_KEY;
            dataSize = FLOAT_SIZE;
        }
        else if (ge::DT_FLOAT16 == dyDataType)
        {
            tilingKey += FLOAT16_DTYPE_KEY;
            dataSize = HALF_SIZE;
        }
        else if (ge::DT_BF16 == dyDataType)
        {
            tilingKey += BFLOAT16_DTYPE_KEY;
            dataSize = HALF_SIZE;
        }
        // 2.计算模式
        const char *mode = context->GetAttrs()->GetStr(0);
        if (strcmp(mode, "iof") == 0)
        {
            tilingKey += MODE_IOF_KEY;
        }
        // 3.是否aligned
        bool aligned = *context->GetAttrs()->GetBool(2);
        if (aligned)
        {
            tilingKey += ALIGNED_KEY;
        }
        else
        {
            tilingKey += UNALIGNED_KEY;
        }
        return tilingKey;
    }

    static ge::graphStatus Tiling4IouV2(gert::TilingContext *context)
    {
        if (context->GetAttrs()->GetStr(ATTR_STR) == nullptr || context->GetAttrs()->GetFloat(ATTR_FLOAT) == nullptr ||
            context->GetAttrs()->GetBool(ATTR_BOOL) == nullptr)
        { // 2表示第三个参数
            OP_LOGD("Tiling4IouV2: attrs have nullptr.");
            return ge::GRAPH_FAILED;
        }
        uint64_t dataSize = 0;
        context->SetTilingKey(GetTilingKey(context, dataSize));

        // 确定tiling参数
        IouV2TilingData tiling;
        float eps = *context->GetAttrs()->GetFloat(1);
        tiling.set_eps(eps);

        uint64_t bBoxLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize() / 4;
        uint64_t gtBoxLength = context->GetInputShape(1)->GetStorageShape().GetShapeSize() / 4;
        tiling.set_bBoxLength(bBoxLength);
        tiling.set_gtBoxLength(gtBoxLength);

        // bBoxLength和gtBoxLength都处理为32B对齐
        uint64_t alignBase = BLOCK_SIZE / dataSize;
        bBoxLength = CeilAlign(bBoxLength, alignBase);
        gtBoxLength = CeilAlign(gtBoxLength, alignBase);

        // ubSize：单核的UB大小
        uint64_t ubSize = 0;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        uint64_t maxCoreNum = ascendcPlatform.GetCoreNumAiv();
        // 根据UB大小计算单次循环能处理的最大数据长度n
        uint64_t totalDataSize;
        uint64_t tempLoop;
        uint64_t maxLength; // 单核单次最大循环长度
        bool aligned = *context->GetAttrs()->GetBool(2);
        if (!aligned)
        {
            uint64_t validSubLen = bBoxLength >= gtBoxLength ? gtBoxLength : bBoxLength;                    // bbox超出gtbox长度，就按照gtbox长度切分
            totalDataSize = (gtBoxLength * validSubLen * 9 + (gtBoxLength + validSubLen) * 8) * FLOAT_SIZE; // 9: 输入输出的buffer个数，8：中间变量的buffer个数
            totalDataSize += (dataSize == FLOAT_SIZE ? 0 : gtBoxLength * validSubLen * 2);                  // f16/bf16需要额外分配2块buffer用于cast
            tempLoop = CeilDiv(totalDataSize, MIN_SIZE_PER_CORE);
            maxLength = 64; // 64是由于Max和Min的mask不能超过64，否则引入精度问题；正常非对齐的最大长度应为72，牺牲部分长度换精度
            while (IsOutOfBound(maxLength, bBoxLength, ubSize, dataSize == FLOAT_SIZE) && maxLength > alignBase)
            {
                maxLength -= alignBase;
            }
        }
        else
        {
            // 对齐模式：bboxex: (4, n), gtboxes: (4, n), overlap: (n, 1), buffer: (n, 8), cast: (n, 4)
            uint64_t rowNum = dataSize == FLOAT_SIZE ? 17 : 21; // 对齐模式：float需要17个n，f16/bf16需要额外4个n用于cast
            totalDataSize = gtBoxLength * rowNum * FLOAT_SIZE;  // gtBoxLength已经是32B对齐，totalDataSize也会是32B对齐
            tempLoop = std::min(CeilDiv(totalDataSize, MIN_SIZE_PER_CORE), CeilDiv(gtBoxLength, alignBase));
            maxLength = ubSize / (rowNum * FLOAT_SIZE) / alignBase * alignBase;
        }

        // 分核策略：保证单核至少用4K数据
        uint64_t coreNum = 1;
        uint64_t loopNum = 1;
        uint64_t tileLength = 0;
        uint64_t frontCoreNum = 0;
        if (totalDataSize <= MIN_SIZE_PER_CORE)
        {
            tileLength = CeilAlign(gtBoxLength, alignBase);
        }
        else if (tempLoop <= maxCoreNum)
        {
            tileLength = CeilAlign(CeilDiv(gtBoxLength, tempLoop), alignBase);
            coreNum = CeilDiv(gtBoxLength, tileLength);
        }
        else if (tiling.get_gtBoxLength() <= maxLength * maxCoreNum)
        {
            tileLength = CeilAlign(CeilDiv(gtBoxLength, maxCoreNum), alignBase);
            coreNum = CeilDiv(gtBoxLength, tileLength);
        }
        else
        {
            coreNum = maxCoreNum;
            uint64_t totalLoop = CeilDiv(tiling.get_gtBoxLength(), maxLength);
            loopNum = maxCoreNum == 0 ? 0 : (totalLoop / maxCoreNum);
            frontCoreNum = maxCoreNum == 0 ? 0 : (totalLoop % maxCoreNum);
            tileLength = CeilAlign(CeilDiv(gtBoxLength, totalLoop), alignBase);
        }
        context->SetBlockDim(coreNum);
        tiling.set_loopNum(loopNum);
        tiling.set_tileLength(tileLength);
        tiling.set_frontCoreNum(frontCoreNum);
        tiling.set_subTileLen(std::min(tileLength, bBoxLength));

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus TilingParse4IouV2([[maybe_unused]] gert::TilingParseContext *context)
    {
        return ge::GRAPH_SUCCESS;
    }

    IMPL_OP_OPTILING(IouV2)
        .Tiling(Tiling4IouV2)
        .TilingParse<IouV2CompileInfo>(TilingParse4IouV2);
} // namespace optiling
