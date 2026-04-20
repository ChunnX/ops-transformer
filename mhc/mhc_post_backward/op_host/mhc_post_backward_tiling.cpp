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
 * \file mhc_post_backward_tiling.cpp
 * \brief
 */

// #include "ge/utils.h"
#include "mhc_post_backward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace {
constexpr uint8_t GRAD_Y_IDX=0;
constexpr uint8_t X_IDX=1;
constexpr uint8_t H_RES_IDX=2;
constexpr uint8_t H_OUT_IDX=3;
constexpr uint8_t H_POST_IDX=4;

constexpr uint8_t X_MIX_GRAD_IDX=0;
constexpr uint8_t H_MIX_GRAD_IDX=1;

constexpr uint8_t SIZE_BFLOAT16=2;
constexpr uint8_t SIZE_FLOAT=4;

static int32_t GetCeilInt(int32_t value1, int32_t value2){
    if (value2==0) {
        return value1;
    }
    return static_cast<int32_t>((value1 + value2-1) / value2);
}

}

namespace optiling{
namespace mhc_post_backward{
const uint32_t BLOCK_C = 1024;

ge::graphStatus TilingCompute(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    MhcPostBackwardTilingData tiling;
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    const auto coreNumber = ascendplatformInfo.GetCoreNumAiv();

    // CHECK_NULLPTR(context->GetInputTensor(GRAD_Y_IDX));
    const auto dFPostResShape = context->GetInputTensor(GRAD_Y_IDX)->GetStorageShape();

    const uint32_t totalTasks = dFPostResShape.GetDim(0);

    uint64_t frontCore = totalTasks % coreNumber != 0 ? static_cast<uint64_t>(totalTasks % coreNumber) : coreNumber;
    uint64_t tailCore = totalTasks <= coreNumber ? 0 : coreNumber - frontCore;

    int32_t singleCoreBS = GetCeilInt(totalTasks, coreNumber);
    int32_t tailBS = totalTasks / coreNumber;

    // const int32_t coreUsed = GetCeilInt(totalTasks, singleCoreBS);
    const int32_t coreUsed = frontCore + tailCore;
    uint64_t ubSizePlatForm;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    uint32_t dFPostResSize = context->GetInputTensor(0)->GetShapeSize();
    uint32_t xSize = context->GetInputTensor(1)->GetShapeSize();
    uint32_t hResSize = context->GetInputTensor(2)->GetShapeSize();
    uint32_t hOutSize = context->GetInputTensor(3)->GetShapeSize();
    uint32_t hPostSize = context->GetInputTensor(4)->GetShapeSize();

    const uint32_t n=dFPostResShape.GetDim(1);
    const uint32_t alignN = GetCeilInt(n * SIZE_FLOAT, 32) * 32 /SIZE_FLOAT;
    const uint32_t channel = dFPostResShape.GetDim(2);
    const uint32_t blockChannel = BLOCK_C > channel ? channel : BLOCK_C;
    const uint32_t loopC = channel / blockChannel;
    const uint32_t tailC = channel % blockChannel;

    auto dtype = context->GetInputTensor(GRAD_Y_IDX)->GetDataType();
    if (dtype == ge::DT_BF16) {
        context->SetTilingKey(0);
    } else if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(1);
    }

    context->SetBlockDim(coreUsed);

    tiling.set_singleCoreBS(singleCoreBS);
    tiling.set_tailBS(tailBS);
    tiling.set_coreUsed(coreUsed);
    tiling.set_frontCore(frontCore);
    tiling.set_tailCore(tailCore);

    tiling.set_dFPostResSize(dFPostResSize);
    tiling.set_xSize(xSize);
    tiling.set_hResSize(hResSize);
    tiling.set_hOutSize(hOutSize);
    tiling.set_hPostSize(hPostSize);

    tiling.set_channel(channel);
    tiling.set_blockChannel(blockChannel);
    tiling.set_n(n);
    tiling.set_alignN(alignN);
    tiling.set_tailC(tailC);
    tiling.set_loopC(loopC);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4MhcPostBackward(gert::TilingContext *context)
{
    // auto attrPtr = context->GetAttrs();
    // OP_CHECK_NULL_WITH_INFOPTR(context, attrPtr);
    return TilingCompute(context);
}
ge::graphStatus TilingPrepareForMhcPostBackward(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcPostBackward)
    .Tiling(Tiling4MhcPostBackward)
    .TilingParse<MhcPostBackwardCompileInfo>(TilingPrepareForMhcPostBackward);
}
}