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
 * \file stack_group_points_tiling.cpp
 * \brief
 */

#include <cstdio>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "stack_group_points_tiling.h"

#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE(_ctx, fmt, ...) \
    do { \
        std::printf(fmt, ##__VA_ARGS__); \
        std::printf("\n"); \
    } while (0)
#define OP_LOGD(...) ((void)0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        std::printf("nullptr error: compileInfo\n"); \
        return ge::GRAPH_FAILED; \
    }

using namespace ge;
using namespace AscendC;

namespace optiling
{

    constexpr int64_t TILING_KEY_HALF = 0;
    constexpr int64_t TILING_KEY_FLOAT = 1;
    constexpr int64_t HALF_TYPE = 2;
    constexpr int64_t FLOAT_TYPE = 4;
    constexpr int64_t INT_TYPE = 4;
    constexpr int64_t DIM_ZERO = 0;
    constexpr int64_t DIM_ONE = 1;
    constexpr int64_t TEMP = 31;
    constexpr int64_t NUM = 16;
    constexpr int64_t BYTE_SIZE = 32;
    constexpr int64_t WORKSPACE = 16 * 1024 * 1024;

    class StackGroupPointsTiling
    {
    public:
        explicit StackGroupPointsTiling(gert::TilingContext *context) : tilingContext(context) {};
        ge::graphStatus Init();
        ge::graphStatus RunKernelTiling();

    private:
        StackGroupPointsTilingData tilingData;
        gert::TilingContext *tilingContext = nullptr;
        int64_t b;
        int64_t m;
        int64_t c;
        int64_t nsample;
        int64_t res;
        int64_t featuresSize;
        int64_t indicesSize;
        int64_t fbcSize;
        int64_t ibcSize;
        int64_t reminder;
        int64_t outLength;
        int64_t n;
        int64_t standard;
        int64_t actCore;
    };
    ge::graphStatus StackGroupPointsTiling::Init()
    {
        // GetWorkspaceSize 路径下 GetCompileInfo() 可能为空，核数从 PlatformAscendC 取。
        auto *platformInfo = tilingContext->GetPlatformInfo();
        OP_CHECK_IF(
            platformInfo == nullptr,
            OP_LOGE(tilingContext, "StackGroupPoints Tiling Error, platformInfo is null."),
            return ge::GRAPH_FAILED);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        int64_t actCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        if (actCoreNum <= 0)
        {
            actCoreNum = 1;
        }
        this->res = 0;
        this->reminder = 0;
        this->outLength = 0;
        int64_t coreReminder = 0;
        int64_t coreRes = 0;
        // Get Input Shape
        const gert::StorageShape *featuresShape = tilingContext->GetInputShape(0);
        OP_CHECK_IF(
            featuresShape == nullptr, OP_LOGE(tilingContext, "StackGroupPoints Tiling Error, featuresShape is null."),
            return ge::GRAPH_FAILED);
        const gert::Shape &fsShape = featuresShape->GetStorageShape();
        const gert::StorageShape *indicesShape = tilingContext->GetInputShape(2);
        OP_CHECK_IF(
            indicesShape == nullptr, OP_LOGE(tilingContext, "StackGroupPoints Tiling Error, indicesShape is null."),
            return ge::GRAPH_FAILED);
        const gert::Shape &isShape = indicesShape->GetStorageShape();
        const gert::StorageShape *indicesBatchCntShape = tilingContext->GetInputShape(3);
        OP_CHECK_IF(
            indicesBatchCntShape == nullptr,
            OP_LOGE(tilingContext, "StackGroupPoints Tiling Error, indicesBatchCntShape is null."),
            return ge::GRAPH_FAILED);
        const gert::Shape &ibcShape = indicesBatchCntShape->GetStorageShape();

        this->n = fsShape.GetDim(DIM_ZERO);
        this->m = isShape.GetDim(DIM_ZERO);
        this->b = ibcShape.GetDim(DIM_ZERO);
        this->c = fsShape.GetDim(DIM_ONE);
        this->nsample = isShape.GetDim(DIM_ONE);
        this->actCore = actCoreNum;
        this->standard = this->m * this->c * this->nsample;

        coreRes = this->m * this->c * this->nsample / NUM;
        coreReminder = this->m * this->c * this->nsample % NUM;
        if (coreReminder > 0)
        {
            coreRes += 1;
        }
        if (coreRes == 0)
        {
            coreRes = 1;
        }
        // Set Block Num
        this->res = coreRes / actCoreNum;
        this->reminder = coreRes % actCoreNum;
        if (actCoreNum <= coreRes)
        {
            tilingContext->SetBlockDim(actCoreNum);
        }
        else if (actCoreNum > coreRes)
        {
            tilingContext->SetBlockDim(coreRes);
        }
        // Allocate Input and Output Size
        auto features = tilingContext->GetInputTensor(0);
        OP_CHECK_IF(
            features == nullptr, OP_LOGE(tilingContext, "StackGroupPoints Tiling Error, features is null."),
            return ge::GRAPH_FAILED);
        this->featuresSize = features->GetShapeSize();
        auto inputTensor = tilingContext->GetInputTensor(2);
        OP_CHECK_IF(
            inputTensor == nullptr, OP_LOGE(tilingContext, "StackGroupPoints Tiling Error, inputTensor is null."),
            return ge::GRAPH_FAILED);
        int64_t indicesS = inputTensor->GetShapeSize();
        this->indicesSize = (indicesS * INT_TYPE + TEMP) / BYTE_SIZE * BYTE_SIZE;
        this->fbcSize = ((this->b + 1) * INT_TYPE + TEMP) / BYTE_SIZE * BYTE_SIZE;
        this->ibcSize = (this->b * INT_TYPE + TEMP) / BYTE_SIZE * BYTE_SIZE;

        auto zeroInputDesc = tilingContext->GetInputDesc(DIM_ZERO);
        OP_CHECK_IF(
            zeroInputDesc == nullptr, OP_LOGE(tilingContext, "StackGroupPoints Tiling Error, zeroInputDesc is null."),
            return ge::GRAPH_FAILED);
        auto srcDtype = zeroInputDesc->GetDataType();
        if (srcDtype == ge::DT_FLOAT16)
        {
            tilingContext->SetTilingKey(TILING_KEY_HALF);
            this->featuresSize = (this->featuresSize * HALF_TYPE + TEMP) / BYTE_SIZE * BYTE_SIZE;
            this->outLength = (this->m * this->c * this->nsample * HALF_TYPE + TEMP) / BYTE_SIZE * BYTE_SIZE;
        }
        else
        {
            tilingContext->SetTilingKey(TILING_KEY_FLOAT);
            this->featuresSize = (this->featuresSize * FLOAT_TYPE + TEMP) / BYTE_SIZE * BYTE_SIZE;
            this->outLength = (this->m * this->c * this->nsample * FLOAT_TYPE + TEMP) / BYTE_SIZE * BYTE_SIZE;
        }
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus StackGroupPointsTiling::RunKernelTiling()
    {
        // Tiling Set
        tilingData.set_m(this->m);
        tilingData.set_c(this->c);
        tilingData.set_n(this->n);
        tilingData.set_b(this->b);
        tilingData.set_nsample(this->nsample);
        tilingData.set_res(this->res);
        tilingData.set_reminder(this->reminder);
        tilingData.set_featuresSize(this->featuresSize);
        tilingData.set_indicesSize(this->indicesSize);
        tilingData.set_fbcSize(this->fbcSize);
        tilingData.set_ibcSize(this->ibcSize);
        tilingData.set_outLength(this->outLength);
        tilingData.set_actCore(this->actCore);
        tilingData.set_standard(this->standard);

        tilingData.SaveToBuffer(
            tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
        size_t *currentWorkspace = tilingContext->GetWorkspaceSizes(1);
        OP_CHECK_IF(
            currentWorkspace == nullptr, OP_LOGE(tilingContext, "StackGroupPoints Tiling Error, currentWorkspace is null."),
            return ge::GRAPH_FAILED);
        currentWorkspace[0] = WORKSPACE;
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingStackGroupPoints(gert::TilingContext *context)
    {
        OP_CHECK_IF(context == nullptr, OP_LOGE("StackGroupPoints", "Tiling context is nullptr"), return ge::GRAPH_FAILED);
        OP_LOGD(context, "Entering TilingStackGroupPoints");
        StackGroupPointsTiling tilingObject(context);
        ge::graphStatus initSt = tilingObject.Init();
        if (initSt != ge::GRAPH_SUCCESS)
        {
            return initSt;
        }
        return tilingObject.RunKernelTiling();
    }

    static ge::graphStatus TilingPrepare4SGP(gert::TilingParseContext *context)
    {
        OP_LOGD("StackGroupPoints", "TilingPrepareForStackGroupPoints start.");
        auto compileInfo = context->GetCompiledInfo<StackGroupPointsCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        auto platformInfo = context->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
        OP_CHECK_IF(
            (compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
            OP_LOGE(
                context, "StartGroupPoints GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%ld.", compileInfo->totalCoreNum,
                compileInfo->ubSize),
            return ge::GRAPH_FAILED);
        OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
        return ge::GRAPH_SUCCESS;
    }

    IMPL_OP_OPTILING(StackGroupPoints)
        .Tiling(TilingStackGroupPoints)
        .TilingParse<StackGroupPointsCompileInfo>(TilingPrepare4SGP);
} // namespace optiling