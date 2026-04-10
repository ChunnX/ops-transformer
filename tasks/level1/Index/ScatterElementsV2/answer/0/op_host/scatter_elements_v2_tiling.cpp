/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2_tiling.cpp
 * \brief
 */
#include "scatter_elements_v2_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdio>
#include <cstring>

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        std::printf("nullptr error!"); \
        return ge::GRAPH_FAILED; \
    }

using namespace std;

namespace {
const int OPERATOR_TYPE = 1;

const int SIZE_OF_FP16 = 2;
const int SIZE_OF_FP32 = 4;
const int SIZE_OF_INT32 = 4;
const int SIZE_OF_INT64 = 8;
const int SIZE_OF_UINT8 = 1;
const int SIZE_OF_INT8 = 1;
const int SIZE_OF_BF16 = 2;

const int NONE = 1;
const int ADD = 2;
const int MUL = 3;
const int BUFFER_NUM = 1;
const int HALF_UB = 2;

const int INPUT_0 = 0;
const int INPUT_1 = 1;
const int INPUT_2 = 2;
const int WORK_SPACE_SIZE = 1024 * 1024 * 16;

const int SMALL_MODE = 1;
} // namespace

namespace optiling {

class ScatterElementsV2Tiling
{
public:
    explicit ScatterElementsV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;
    bool CacheOpSupport();
    ge::graphStatus RunCacheOpTiling();
    ge::graphStatus SetCacheOpTiling();

private:
    void SetTilingData(ScatterElementsV2TilingData& tiling);
    void LogTilingData();
    size_t CalculateWorkspaceSize();
    void ParseAttrs();
    void ProcessUpdatesShape(gert::Shape& updatesShape, const gert::Shape& indicesShape);
    void SetDimsForFirstAxis(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                             const gert::Shape& updatesShape, size_t inputDimNum);
    void SetDimsForMiddleAxis(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                              const gert::Shape& updatesShape, size_t inputDimNum);
    void SetDimsForLastAxis(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                            const gert::Shape& updatesShape, size_t inputDimNum);
    void SetDimsByAxisType(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                           const gert::Shape& updatesShape, size_t inputDimNum);
    ScatterElementsV2TilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    int32_t mode = 0;
    int32_t updatesIsScalar = 0;
    uint64_t xDim0 = 1;
    uint64_t xDim1 = 1;
    uint64_t indicesDim0 = 1;
    uint64_t indicesDim1 = 1;
    uint64_t updatesDim0 = 1;
    uint64_t updatesDim1 = 1;
    uint64_t batchSize = 1;
    uint64_t realDim = 0;
    uint32_t tilingKey = 0;
    uint64_t usedCoreNum = 0;
    uint64_t eachNum = 1;
    uint64_t extraTaskCore = 0;
    uint64_t inputCount = 1;
    uint64_t indicesCount = 1;
    uint64_t updatesCount = 1;
    uint64_t inputOneTime = 0;
    uint64_t indicesOneTime = 0;
    uint64_t updatesOneTime = 0;
    uint64_t inputLoop = 0;
    uint64_t indicesLoop = 0;
    uint64_t inputEach = 0;
    uint64_t indicesEach = 0;
    uint64_t inputLast = 0;
    uint64_t indicesLast = 0;
    uint64_t eachPiece = 1;
    uint64_t inputAlign = 8;
    uint64_t indicesAlign = 8;
    uint64_t updatesAlign = 8;
    uint64_t inputOnePiece = 0;
    uint64_t modeFlag = 0;
    uint64_t lastIndicesLoop = 1;
    uint64_t lastIndicesEach = 1;
    uint64_t lastIndicesLast = 1;
    uint64_t oneTime = 1;
    uint64_t lastOneTime = 1;
    uint64_t workspaceSize = 1024 * 1024 * 16;
    uint64_t max_ub = 20480;
};

ge::graphStatus ScatterElementsV2Tiling::Init()
{
    if (tilingContext == nullptr) {
        OP_LOGE("ScatterElementsV2", "tilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    static uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (coreNum == 0) {
        OP_LOGE(tilingContext, "coreNum must greater than 0.");
        return ge::GRAPH_FAILED;
    }
    workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
    OP_LOGD(tilingContext, "ubSizePlatForm: %lu.", ubSizePlatForm);

    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr || tilingContext->GetInputShape(INPUT_0) == nullptr ||
        tilingContext->GetInputShape(INPUT_1) == nullptr || tilingContext->GetInputShape(INPUT_2) == nullptr ||
        tilingContext->GetInputDesc(INPUT_0) == nullptr || tilingContext->GetRawTilingData() == nullptr) {
        OP_LOGE(tilingContext, "tilingContext inputshape or outputshape is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    uint32_t inputSize = 0;
    if (ge::DT_FLOAT == inputDtype) {
        inputSize = SIZE_OF_FP32;
    } else if (ge::DT_FLOAT16 == inputDtype) {
        inputSize = SIZE_OF_FP16;
    } else if (ge::DT_INT32 == inputDtype) {
        inputSize = SIZE_OF_INT32;
    } else if (ge::DT_UINT8 == inputDtype) {
        inputSize = SIZE_OF_UINT8;
    } else if (ge::DT_INT8 == inputDtype) {
        inputSize = SIZE_OF_INT8;
    } else if (ge::DT_BF16 == inputDtype) {
        inputSize = SIZE_OF_BF16;
    } else {
        OP_LOGE(tilingContext, "var only support float, float16, int32, uint8, int8, bf16.");
        return ge::GRAPH_FAILED;
    }
    uint32_t indicesSize = 0;
    auto indicesDtype = tilingContext->GetInputDesc(1)->GetDataType();
    if (ge::DT_INT32 == indicesDtype) {
        indicesSize = SIZE_OF_INT32;
    } else if (ge::DT_INT64 == indicesDtype) {
        indicesSize = SIZE_OF_INT64;
    } else {
        OP_LOGE(tilingContext, "indices only support int64, int32.");
        return ge::GRAPH_FAILED;
    }

    uint32_t dataAlign = 32;
    uint32_t inputDataAlign = dataAlign / inputSize;
    uint32_t indexDataAlign = dataAlign / indicesSize;

    const int* dim = (attrs->GetAttrPointer<int>(0));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, dim);
    const char* reduce = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, reduce);
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto inputDimNum = inputShape.GetDimNum();
    if (strcmp(reduce, "none") == 0) {
        tilingKey += OPERATOR_TYPE * NONE;
    } else if (strcmp(reduce, "add") == 0) {
        tilingKey += OPERATOR_TYPE * ADD;
    } else if (strcmp(reduce, "mul") == 0) {
        tilingKey += OPERATOR_TYPE * MUL;
        OP_LOGE(tilingContext, "scatter_elements_v2 not support mul.");
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGE(tilingContext, "scatter_elements_v2 only support none, add.");
        return ge::GRAPH_FAILED;
    }

    if (inputDimNum != indicesShape.GetDimNum() ||
        inputDimNum != tilingContext->GetInputShape(INPUT_2)->GetStorageShape().GetDimNum()) {
        OP_LOGE(tilingContext, "the dimNum of input must equal the dimNum of indices.");
        return ge::GRAPH_FAILED;
    }

    if (*dim < 0) {
        realDim = *dim + inputDimNum;
    } else {
        realDim = *dim;
    }

    if (realDim != inputDimNum - 1) {
        OP_LOGE(tilingContext, "scatter_elements_v2 not support dim != -1.");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < inputDimNum; ++i) {
        auto dimInput = inputShape.GetDim(i);
        auto dimIndices = indicesShape.GetDim(i);
        auto dimUpdates = updatesShape.GetDim(i);
        if (dimUpdates < dimIndices) {
            OP_LOGE(tilingContext, "the dim of updates must greater than or equal the dim of indices.");
            return ge::GRAPH_FAILED;
        }
        if (realDim == i) {
            inputOneTime = dimInput;
            indicesOneTime = dimIndices;
            updatesOneTime = dimUpdates;
        }
        inputCount *= dimInput;
        indicesCount *= dimIndices;
        updatesCount *= dimUpdates;
    }

    if (inputOneTime == 0 || indicesOneTime == 0 || updatesOneTime == 0 || inputCount == 0 || indicesCount == 0 ||
        updatesCount == 0) {
        OP_LOGE(tilingContext, "shape cannot equal 0.");
        return ge::GRAPH_FAILED;
    }

    if (ge::DT_INT64 == indicesDtype) {
        indicesSize += SIZE_OF_INT32;
    }
    if ((strcmp(reduce, "add") == 0) && (ge::DT_FLOAT16 == inputDtype || ge::DT_BF16 == inputDtype)) {
        inputSize += sizeof(float) / BUFFER_NUM;
        indicesSize += sizeof(float) / BUFFER_NUM;
    }

    uint32_t times = indicesCount / indicesOneTime;
    uint32_t totalSize = inputSize * inputOneTime + indicesSize * (updatesOneTime + indicesOneTime);
    // 小包场景，一次性可以搬入一轮尾轴，按ub上限尽可能的搬多轮，特殊处理走small分支
    if (totalSize <= max_ub) {
        modeFlag = SMALL_MODE;
        max_ub = max_ub / totalSize;
        if (times < coreNum) {
            usedCoreNum = times;
            indicesEach = 1;
            indicesLoop = 1;
            indicesLast = 1;
        } else {
            oneTime = (times + coreNum - 1) / coreNum;
            usedCoreNum = (times + oneTime - 1) / oneTime;
            indicesLoop = (oneTime + max_ub - 1) / max_ub;
            indicesEach = indicesLoop == 1 ? oneTime : max_ub;
            indicesLast = oneTime - (indicesLoop - 1) * indicesEach;

            lastOneTime = times - oneTime * (usedCoreNum - 1);
            lastIndicesLoop = (lastOneTime + max_ub - 1) / max_ub;
            lastIndicesEach = lastIndicesLoop == 1 ? lastOneTime : max_ub;
            lastIndicesLast = lastOneTime - (lastIndicesLoop - 1) * lastIndicesEach;
        }
        indicesAlign = ((indicesEach - 1) / indexDataAlign + 1) * indexDataAlign;

        OP_LOGD(tilingContext, "Tiling inited.");
        return ge::GRAPH_SUCCESS;
    }

    inputOnePiece = inputOneTime;
    if (times < coreNum) {                                          // 一个任务可以分给多个核
        uint32_t need = (inputOneTime + dataAlign - 1) / dataAlign; // 每个核至少处理32个数，每个任务需要多少个核
        eachPiece = coreNum / times;                                // 每个任务可用的核数
        eachPiece = eachPiece > need ? need : eachPiece;
        eachNum = eachPiece == 1 ? 1 : 0;
        extraTaskCore = 0;
        inputOnePiece = (inputOneTime + eachPiece - 1) / eachPiece; // 每个核需要处理的数量
        usedCoreNum = (inputOneTime + inputOnePiece - 1) / inputOnePiece *
                      times; // 再次根据每个核需要处理的数量重新计算需要的核数
    } else {                 // 一个任务单独由一个核处理
        usedCoreNum = coreNum;
        eachNum = times / coreNum;
        extraTaskCore = times - eachNum * coreNum;
    }

    uint32_t indicesSum = indicesOneTime * (inputSize + indicesSize);
    uint32_t inputSum = inputOnePiece * inputSize;
    if (inputSum + indicesSum > static_cast<uint32_t>(max_ub)) {
        if (indicesSum < static_cast<uint32_t>(max_ub / HALF_UB)) {
            inputEach = (max_ub - indicesSum) / inputSize;
            inputEach = inputEach > inputOnePiece ? inputOnePiece : inputEach;
            inputLoop = (inputOnePiece - 1) / inputEach + 1;
            inputLast = inputOnePiece - inputEach * (inputLoop - 1);

            indicesLoop = 1;
            indicesEach = indicesLast = indicesOneTime;
        } else if (inputSum < static_cast<uint32_t>(max_ub / HALF_UB)) {
            indicesEach = (max_ub - inputSum) / (inputSize + indicesSize);
            indicesEach = indicesEach > indicesOneTime ? indicesOneTime : indicesEach;
            indicesLoop = (indicesOneTime - 1) / indicesEach + 1;
            indicesLast = indicesOneTime - indicesEach * (indicesLoop - 1);

            inputLoop = 1;
            inputEach = inputLast = inputOnePiece;
        } else {
            inputEach = max_ub / (HALF_UB * inputSize);
            inputEach = inputEach > inputOnePiece ? inputOnePiece : inputEach;
            inputLoop = (inputOnePiece - 1) / inputEach + 1;
            inputLast = inputOnePiece - inputEach * (inputLoop - 1);

            indicesEach = max_ub / (HALF_UB * (inputSize + indicesSize));
            indicesEach = indicesEach > indicesOneTime ? indicesOneTime : indicesEach;
            indicesLoop = (indicesOneTime - 1) / indicesEach + 1;
            indicesLast = indicesOneTime - indicesEach * (indicesLoop - 1);
        }
    } else {
        inputLoop = indicesLoop = 1;
        inputEach = inputLast = inputOnePiece;
        indicesEach = indicesLast = indicesOneTime;
    }

    inputAlign = ((inputEach - 1) / inputDataAlign + 1) * inputDataAlign;
    indicesAlign = ((indicesEach - 1) / indexDataAlign + 1) * indexDataAlign;
    updatesAlign = ((indicesEach - 1) / inputDataAlign + 1) * inputDataAlign;

    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2Tiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");

    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_eachNum(eachNum);
    tilingData.set_extraTaskCore(extraTaskCore);
    tilingData.set_eachPiece(eachPiece);
    tilingData.set_inputAlign(inputAlign);
    tilingData.set_indicesAlign(indicesAlign);
    tilingData.set_updatesAlign(updatesAlign);
    tilingData.set_inputCount(inputCount);
    tilingData.set_indicesCount(indicesCount);
    tilingData.set_updatesCount(updatesCount);
    tilingData.set_inputOneTime(inputOneTime);
    tilingData.set_indicesOneTime(indicesOneTime);
    tilingData.set_updatesOneTime(updatesOneTime);
    tilingData.set_inputEach(inputEach);
    tilingData.set_indicesEach(indicesEach);
    tilingData.set_inputLast(inputLast);
    tilingData.set_indicesLast(indicesLast);
    tilingData.set_inputLoop(inputLoop);
    tilingData.set_indicesLoop(indicesLoop);
    tilingData.set_inputOnePiece(inputOnePiece);
    tilingData.set_modeFlag(modeFlag);
    tilingData.set_lastIndicesLoop(lastIndicesLoop);
    tilingData.set_lastIndicesEach(lastIndicesEach);
    tilingData.set_lastIndicesLast(lastIndicesLast);
    tilingData.set_oneTime(oneTime);
    tilingData.set_lastOneTime(lastOneTime);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(usedCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;
    TilingDataPrint();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void ScatterElementsV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "usedCoreNum: %lu.", usedCoreNum);
    OP_LOGD(tilingContext, "eachNum: %lu.", eachNum);
    OP_LOGD(tilingContext, "extraTaskCore: %lu.", extraTaskCore);
    OP_LOGD(tilingContext, "eachPiece: %lu.", eachPiece);
    OP_LOGD(tilingContext, "inputAlign: %lu.", inputAlign);
    OP_LOGD(tilingContext, "indicesAlign: %lu.", indicesAlign);
    OP_LOGD(tilingContext, "updatesAlign: %lu.", updatesAlign);
    OP_LOGD(tilingContext, "inputCount: %lu.", inputCount);
    OP_LOGD(tilingContext, "indicesCount: %lu.", indicesCount);
    OP_LOGD(tilingContext, "updatesCount: %lu.", updatesCount);
    OP_LOGD(tilingContext, "inputOneTime: %lu.", inputOneTime);
    OP_LOGD(tilingContext, "indicesOneTime: %lu.", indicesOneTime);
    OP_LOGD(tilingContext, "updatesOneTime: %lu.", updatesOneTime);
    OP_LOGD(tilingContext, "inputEach: %lu.", inputEach);
    OP_LOGD(tilingContext, "indicesEach: %lu.", indicesEach);
    OP_LOGD(tilingContext, "inputLast: %lu.", inputLast);
    OP_LOGD(tilingContext, "indicesLast: %lu.", indicesLast);
    OP_LOGD(tilingContext, "inputLoop: %lu.", inputLoop);
    OP_LOGD(tilingContext, "indicesLoop: %lu.", indicesLoop);
    OP_LOGD(tilingContext, "inputOnePiece: %lu.", inputOnePiece);
    OP_LOGD(tilingContext, "modeFlag: %lu.", modeFlag);
    OP_LOGD(tilingContext, "lastIndicesLoop: %lu.", lastIndicesLoop);
    OP_LOGD(tilingContext, "lastIndicesEach: %lu.", lastIndicesEach);
    OP_LOGD(tilingContext, "lastIndicesLast: %lu.", lastIndicesLast);
    OP_LOGD(tilingContext, "oneTime: %lu.", oneTime);
    OP_LOGD(tilingContext, "lastOneTime: %lu.", lastOneTime);
    OP_LOGD(tilingContext, "tilingKey: %u.", tilingKey);
    OP_LOGD(tilingContext, "max_ub: %lu.", max_ub);
}

bool ScatterElementsV2Tiling::CacheOpSupport() {
    if (tilingContext == nullptr) {
        OP_LOGD("ScatterElementsV2", "tilingContext is nullptr.");
        return false;
    }

    auto attrs = tilingContext->GetAttrs();
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();

    const int* dim = (attrs->GetAttrPointer<int>(0));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, dim);
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto inputDimNum = inputShape.GetDimNum();
    realDim = (*dim < 0 ? *dim + inputDimNum : *dim);
    // 尾轴场景且非bool类型输入，不支持
    if (realDim == inputDimNum - 1 && inputDtype != ge::DT_BOOL) {
        OP_LOGD("ScatterElementsV2", "Low memory when realDim = -1, only support bool dtype.");
        return false;
    }
    return true;
}

void ScatterElementsV2Tiling::SetTilingData(ScatterElementsV2TilingData& tiling) {
    tiling.set_batchSize(batchSize);
    tiling.set_realDim(realDim);
    tiling.set_mode(mode);
    tiling.set_coreNums(usedCoreNum);
    tiling.set_updatesIsScalar(updatesIsScalar);
    tiling.set_xDim0(xDim0);
    tiling.set_xDim1(xDim1);
    tiling.set_indicesDim0(indicesDim0);
    tiling.set_indicesDim1(indicesDim1);
    tiling.set_updatesDim0(updatesDim0);
    tiling.set_updatesDim1(updatesDim1);
}

void ScatterElementsV2Tiling::LogTilingData() {
    OP_LOGD(tilingContext, "batchSize: %lu.", batchSize);
    OP_LOGD(tilingContext, "realDim: %lu.", realDim);
    OP_LOGD(tilingContext, "mode: %d.", mode);
    OP_LOGD(tilingContext, "coreNums: %d.", usedCoreNum);
    OP_LOGD(tilingContext, "updatesIsScalar: %d.", updatesIsScalar);
    OP_LOGD(tilingContext, "xDim0: %lu.", xDim0);
    OP_LOGD(tilingContext, "xDim1: %lu.", xDim1);
    OP_LOGD(tilingContext, "indicesDim0: %lu.", indicesDim0);
    OP_LOGD(tilingContext, "indicesDim1: %lu.", indicesDim1);
    OP_LOGD(tilingContext, "updatesDim0: %lu.", updatesDim0);
    OP_LOGD(tilingContext, "updatesDim1: %lu.", updatesDim1);
}

size_t ScatterElementsV2Tiling::CalculateWorkspaceSize() {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    size_t libApiworkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto sizeOfVar = 4;
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    if (ge::DT_INT8 == inputDtype || ge::DT_UINT8 == inputDtype || ge::DT_BOOL == inputDtype) {
        sizeOfVar = 2;
    }

    if (batchSize == 1 && realDim == 1) { // 首轴需要转置
        auto targetParts = indicesDim1 / usedCoreNum;
        targetParts = targetParts > 0 ? targetParts : 1;
        targetParts = targetParts > 5 ? 5 : targetParts;
        auto partSize = indicesDim1 / targetParts;
        libApiworkspaceSize += partSize * xDim0 * sizeOfVar + partSize * indicesDim0 * sizeof(int32_t) +
                               partSize * updatesDim0 * sizeOfVar;
    }
    if (batchSize > 1 && realDim == 1) { // 中轴需要转置
        libApiworkspaceSize += xDim0 * xDim1 * sizeOfVar + indicesDim0 * indicesDim1 * sizeof(int32_t) +
                               updatesDim0 * updatesDim1 * sizeOfVar;
    }
    return libApiworkspaceSize;
}

ge::graphStatus ScatterElementsV2Tiling::SetCacheOpTiling() {
    ScatterElementsV2TilingData tiling;
    SetTilingData(tiling);
    LogTilingData();

    size_t calculateResult = CalculateWorkspaceSize();
    size_t *currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = calculateResult;

    tilingContext->SetBlockDim(usedCoreNum);
    tilingContext->SetTilingKey(0);
    tiling.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void ScatterElementsV2Tiling::ParseAttrs() {
    auto attrs = tilingContext->GetAttrs();
    const char* reduce = attrs->GetAttrPointer<char>(1);
    if (strcmp(reduce, "none") == 0) {
        mode = 0;
    } else {
        mode = 1;
    }
}

void ScatterElementsV2Tiling::ProcessUpdatesShape(gert::Shape& updatesShape, const gert::Shape& indicesShape) {
    if ((updatesShape.GetDimNum() == 1 && updatesShape.GetDim(0) == 1) || updatesShape.GetDimNum() == 0) {
        updatesIsScalar = 1;
        updatesShape = indicesShape;
    } else {
        updatesIsScalar = 0;
    }
}

void ScatterElementsV2Tiling::SetDimsForFirstAxis(const gert::Shape& inputShape,
                                                    const gert::Shape& indicesShape,
                                                    const gert::Shape& updatesShape,
                                                    size_t inputDimNum) {
    batchSize = 1;
    realDim = 1;
    xDim0 = inputShape.GetDim(0);
    indicesDim0 = indicesShape.GetDim(0);
    updatesDim0 = updatesShape.GetDim(0);
    for (size_t i = 1; i < inputDimNum; i++) {
        xDim1 *= inputShape.GetDim(i);
        indicesDim1 *= indicesShape.GetDim(i);
        updatesDim1 *= updatesShape.GetDim(i);
    }
}

void ScatterElementsV2Tiling::SetDimsForMiddleAxis(const gert::Shape& inputShape,
                                                     const gert::Shape& indicesShape,
                                                     const gert::Shape& updatesShape,
                                                     size_t inputDimNum) {
    for (size_t i = 0; i < realDim; i++) {
        batchSize *= indicesShape.GetDim(i);
    }
    xDim0 = inputShape.GetDim(realDim);
    indicesDim0 = indicesShape.GetDim(realDim);
    updatesDim0 = updatesShape.GetDim(realDim);
    for (size_t i = realDim + 1; i < inputDimNum; i++) {
        xDim1 *= inputShape.GetDim(i);
        indicesDim1 *= indicesShape.GetDim(i);
        updatesDim1 *= updatesShape.GetDim(i);
    }
    realDim = 1;
}

void ScatterElementsV2Tiling::SetDimsForLastAxis(const gert::Shape& inputShape,
                                                   const gert::Shape& indicesShape,
                                                   const gert::Shape& updatesShape,
                                                   size_t inputDimNum) {
    batchSize = 1;
    realDim = 2;
    xDim1 = inputShape.GetDim(inputDimNum - 1);
    indicesDim1 = indicesShape.GetDim(inputDimNum - 1);
    updatesDim1 = updatesShape.GetDim(inputDimNum - 1);
    for (size_t i = 0; i < inputDimNum - 1; i++) {
        xDim0 *= inputShape.GetDim(i);
        indicesDim0 *= indicesShape.GetDim(i);
        updatesDim0 *= updatesShape.GetDim(i);
    }
}

void ScatterElementsV2Tiling::SetDimsByAxisType(const gert::Shape& inputShape,
                                                  const gert::Shape& indicesShape,
                                                  const gert::Shape& updatesShape,
                                                  size_t inputDimNum) {
    if (realDim == 0) {
        SetDimsForFirstAxis(inputShape, indicesShape, updatesShape, inputDimNum);
    } else if (realDim > 0 && realDim < inputDimNum - 1) {
        SetDimsForMiddleAxis(inputShape, indicesShape, updatesShape, inputDimNum);
    } else {
        SetDimsForLastAxis(inputShape, indicesShape, updatesShape, inputDimNum);
    }
}

ge::graphStatus ScatterElementsV2Tiling::RunCacheOpTiling() {
    ParseAttrs();

    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();

    ProcessUpdatesShape(updatesShape, indicesShape);

    auto inputDimNum = inputShape.GetDimNum();
    auto attrs = tilingContext->GetAttrs();
    const int* dim = (attrs->GetAttrPointer<int>(0));
    realDim = (*dim < 0 ? *dim + inputDimNum : *dim);

    SetDimsByAxisType(inputShape, indicesShape, updatesShape, inputDimNum);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    usedCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(tilingContext, "ascendcPlatform CoreNum is: %d.", usedCoreNum);
    return SetCacheOpTiling();
}

ge::graphStatus TilingScatterElementsV2(gert::TilingContext* context)
{
    ScatterElementsV2Tiling tilingObject(context);
    if (tilingObject.CacheOpSupport()) {
        return tilingObject.RunCacheOpTiling();
    } else {
        if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return tilingObject.RunKernelTiling();
    }
}

IMPL_OP_OPTILING(ScatterElementsV2).Tiling(TilingScatterElementsV2);
} // namespace optiling
