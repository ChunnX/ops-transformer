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
 * \file angle_v2_tiling.cpp
 * \brief
 */

#include "angle_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

namespace optiling {
constexpr uint64_t COMPLEX64_MODE = 1UL;
constexpr uint64_t FP32_MODE = 2UL;
constexpr uint64_t FP16_MODE = 3UL;
constexpr uint64_t BOOL_MODE = 4UL;
constexpr uint64_t UINT8_MODE = 5UL;
constexpr uint64_t INT8_MODE = 6UL;
constexpr uint64_t INT16_MODE = 7UL;
constexpr uint64_t INT32_MODE = 8UL;
constexpr uint64_t INT64_MODE = 9UL;
constexpr int64_t SIZE_OF_B8 = 1;
constexpr int64_t SIZE_OF_B16 = 2;
constexpr int64_t SIZE_OF_B32 = 4;
constexpr int64_t BYTE_BLOCK = 32;
constexpr int64_t BYTE_REPEAT = 256;                 // The amount of data that can be processed by a repeat.
constexpr int64_t SELECT_MODE_GE_ZERO_TMP_UB = 8000; // select mode 2 need 8000B
class AngleV2Tiling
{
public:
    explicit AngleV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    void SetTilingKeyMode(ge::DataType dType);
    int64_t GetNeedCoreNum(const int64_t coreNumPlatform, ge::DataType dType);
    void GetUsedBytesPerDataInKernel(ge::DataType dType);
    void CalTilingAligned(ge::DataType dType);
    void GetAlignNum(ge::DataType dType);
    AngleV2TilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    int64_t coreNum = 0;
    int64_t dataPerRepeat = 0;
    int64_t tileLength = 0;         // align to 256B
    int64_t totalLength = 1;        // the length of input
    int64_t formerNum = 0;          // deal more data core num
    int64_t tailNum = 0;            // deal less data core num
    int64_t formerLength = 0;       // deal more data length
    int64_t tailLength = 0;         // deal less data length
    int64_t alignNum = 0;           // data count per block
    int64_t totalLengthAligned = 0; // length to align 32B
    uint64_t ubSizePlatForm = 0UL;
    int64_t bytesPerData = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

void AngleV2Tiling::GetUsedBytesPerDataInKernel(ge::DataType dType)
{
    // Calculate the bytes of buffer for one element used
    int64_t bytesI8 = 1;
    int64_t bytesI16 = 2;
    int64_t bytesI32 = 4;
    int64_t bytesI64 = 8;
    int64_t coefficentTwo = 2;
    int64_t coefficentThree = 3;
    int64_t coefficentTen = 10;
    switch (dType) {
        case ge::DT_COMPLEX64:
            // double buffer for input(complex64) and output(float32)
            bytesPerData = bytesI64 * coefficentTwo + bytesI32 * coefficentTwo;
            // two masks(uint8) and ten localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 * coefficentTwo + bytesI32 * coefficentTen;
            break;
        case ge::DT_FLOAT:
            // double buffer for input(float32) and output(float32)
            bytesPerData = bytesI32 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) and three localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentThree;
            break;
        case ge::DT_INT8:
            // double buffer for input(int8) and output(float32)
            bytesPerData = bytesI8 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) ,two localTensor(float32) and one castTensor(float16) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentTwo + bytesI16;
            break;
        case ge::DT_INT16:
            // double buffer for input(int16) and output(float32)
            bytesPerData = bytesI16 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) and two localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentTwo;
            break;
        case ge::DT_INT32:
            // double buffer for input(int32) and output(float32)
            bytesPerData = bytesI32 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) and two localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentTwo;
            break;
        case ge::DT_INT64:
            // double buffer for input(int64) and output(float32)
            bytesPerData = bytesI64 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) and two localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentTwo;
            break;
        default:
            // double buffer for input(float16) and output(float16)
            bytesPerData = bytesI16 * coefficentTwo + bytesI16 * coefficentTwo;
            // one masks(uint8) and three localTensor(float16) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI16 * coefficentThree;
            break;
    }
}

void AngleV2Tiling::SetTilingKeyMode(ge::DataType dType)
{
    switch (dType) {
        case ge::DT_COMPLEX64:
            tilingContext->SetTilingKey(COMPLEX64_MODE);
            break;
        case ge::DT_FLOAT16:
            tilingContext->SetTilingKey(FP16_MODE);
            break;
        case ge::DT_BOOL:
            tilingContext->SetTilingKey(BOOL_MODE);
            break;
        case ge::DT_UINT8:
            tilingContext->SetTilingKey(UINT8_MODE);
            break;
        case ge::DT_INT8:
            tilingContext->SetTilingKey(INT8_MODE);
            break;
        case ge::DT_INT16:
            tilingContext->SetTilingKey(INT16_MODE);
            break;
        case ge::DT_INT32:
            tilingContext->SetTilingKey(INT32_MODE);
            break;
        case ge::DT_INT64:
            tilingContext->SetTilingKey(INT64_MODE);
            break;
        default:
            tilingContext->SetTilingKey(FP32_MODE);
            break;
    }
}

int64_t AngleV2Tiling::GetNeedCoreNum(const int64_t coreNumPlatform, ge::DataType dType)
{
    if (dType == ge::DT_FLOAT16) {
        dataPerRepeat = BYTE_REPEAT / SIZE_OF_B16;
    } else {
        dataPerRepeat = BYTE_REPEAT / SIZE_OF_B32;
    }
    int64_t tempCoreNum = (totalLength - 1) / dataPerRepeat + 1;
    if (tempCoreNum == 0) {
        tempCoreNum = 1;
    }
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void AngleV2Tiling::GetAlignNum(ge::DataType dType)
{
    switch (dType) {
        case ge::DT_FLOAT16:
            alignNum = BYTE_BLOCK / SIZE_OF_B16;
            break;
        case ge::DT_INT8:
            alignNum = BYTE_BLOCK / SIZE_OF_B8;
            break;
        case ge::DT_INT16:
            alignNum = BYTE_BLOCK / SIZE_OF_B16;
            break;
        default:
            alignNum = BYTE_BLOCK / SIZE_OF_B32;
            break;
    }
}
void AngleV2Tiling::CalTilingAligned(ge::DataType dType)
{
    GetAlignNum(dType);
    totalLengthAligned = ((totalLength + alignNum - 1) / alignNum) * alignNum;
    // Divide blocks evenly into each core
    auto blockNum = totalLengthAligned / alignNum;
    formerNum = blockNum % coreNum;
    tailNum = coreNum - formerNum;
    formerLength = ((blockNum + coreNum - 1) / coreNum) * alignNum;
    tailLength = (blockNum / coreNum) * alignNum;

    if (socVersion == platform_ascendc::SocVersion::ASCEND910) {
        tileLength = static_cast<int64_t>(ubSizePlatForm) / bytesPerData / dataPerRepeat * dataPerRepeat;
    } else {
        tileLength = (static_cast<int64_t>(ubSizePlatForm) - SELECT_MODE_GE_ZERO_TMP_UB) /
                     bytesPerData / dataPerRepeat * dataPerRepeat;
    }
}

ge::graphStatus AngleV2Tiling::Init()
{
    OP_LOGD(tilingContext, "Tiling initing.");
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    auto inputShape0 = tilingContext->GetInputShape(0);
    auto xShape = inputShape0->GetStorageShape();
    totalLength = xShape.GetShapeSize();
    OP_LOGD(tilingContext, "totalLength %ld.", totalLength);

    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion = ascendcPlatform.GetSocVersion();
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(tilingContext, "coreNum %ld.", coreNum);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_LOGD(tilingContext, "ub_size_platform is %lu.", ubSizePlatForm);
    auto inputDesc0 = tilingContext->GetInputDesc(0);
    auto dType = inputDesc0->GetDataType();
    coreNum = GetNeedCoreNum(coreNum, dType);

    SetTilingKeyMode(dType);
    GetUsedBytesPerDataInKernel(dType);
    CalTilingAligned(dType);
    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AngleV2Tiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "SetTiling start.");
    tilingContext->SetBlockDim(coreNum);
    tilingData.set_totalLength(totalLength);
    tilingData.set_formerNum(formerNum);
    tilingData.set_tailNum(tailNum);
    tilingData.set_formerLength(formerLength);
    tilingData.set_tailLength(tailLength);
    tilingData.set_alignNum(alignNum);
    tilingData.set_totalLengthAligned(totalLengthAligned);
    tilingData.set_tileLength(tileLength);
    tilingData.set_dataPerRepeat(dataPerRepeat);
    auto rawTilingData = tilingContext->GetRawTilingData();
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(tilingContext, "SetTiling end.");
    return ge::GRAPH_SUCCESS;
}

void AngleV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "usedCoreNum: %ld.", coreNum);
    OP_LOGD(tilingContext, "totalLength: %ld.", totalLength);
    OP_LOGD(tilingContext, "formerNum: %ld.", formerNum);
    OP_LOGD(tilingContext, "tailNum: %ld.", tailNum);
    OP_LOGD(tilingContext, "formerLength: %ld.", formerLength);
    OP_LOGD(tilingContext, "tailLength: %ld.", tailLength);
    OP_LOGD(tilingContext, "alignNum: %ld.", alignNum);
    OP_LOGD(tilingContext, "totalLengthAligned: %ld.", totalLengthAligned);
    OP_LOGD(tilingContext, "tileLength: %ld.", tileLength);
    OP_LOGD(tilingContext, "dataPerRepeat: %ld.", dataPerRepeat);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AngleV2Tiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AngleV2 : public OpDef {
public:
    explicit AngleV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_COMPLEX64, ge::DT_BOOL, ge::DT_UINT8, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(AngleV2);
} // namespace optiling