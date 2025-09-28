/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_to_all_all_gather_batch_matmul_tiling.cpp
 * \brief
 */

#include "allto_all_all_gather_batch_mat_mul_tiling.h"

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "tiling/mc2_tiling_common_var.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "allto_all_all_gather_formulaic_tiling.h"
#include "op_mc2.h"
#include "mc2_moe_utils.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Moe;

namespace {

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t OUTPUT_Y1_INDEX = 0;
constexpr uint32_t OUTPUT_Y2_INDEX = 1;
constexpr uint32_t OUTPUT_Y3_INDEX = 2;

constexpr uint32_t ATTR_EP_GROUP_INDEX = 0;
constexpr uint32_t ATTR_TP_GROUP_INDEX = 1;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 2;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 3;
constexpr uint32_t ATTR_X_SHARD_TYPE_INDEX = 4;
constexpr uint32_t ATTR_ACT_TYPE_INDEX = 5;
constexpr uint32_t ATTR_IS_WEIGHT_TRANS_INDEX = 6;
constexpr uint32_t ATTR_OUTPUT_Y2_FLAG_INDEX = 7;
constexpr uint32_t ATTR_OUTPUT_Y3_FLAG_INDEX = 8;

constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8; // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_ALL_GATHER = 6; // numeric representation of AllGather

constexpr uint64_t INIT_TILINGKEY = 1000000000000000000;
constexpr uint32_t FASTGELU_MINSIZE = 3 * 256;
constexpr uint32_t GELU = 1;
constexpr uint32_t SILU = 2;
constexpr uint32_t RELU = 3;
constexpr uint32_t FASTGELU = 4;

constexpr uint64_t TILINGKEY_X_SHARD = 1;           // When X_shard = 1
constexpr uint64_t TILINGKEY_WEIGHT_TRANSPOSE = 10; // When weight trans is true
constexpr uint64_t TILINGKEY_IS_BIAS = 100;         // When bias is given as input
constexpr uint64_t TILINGKEY_Y2_ONLY = 1000;        // When only need to output Y2
constexpr uint64_t TILINGKEY_Y3_ONLY = 2000;        // When only need to output Y3
constexpr uint64_t TILINGKEY_Y2_Y3 = 3000;          // When have to output both Y2 and Y3

} // namespace

namespace optiling {

struct TransposeConfig {
    uint64_t ubSize;
    uint64_t tileCWhole;
    uint64_t tileEWhole;
    uint64_t inputDatasize;
    bool isLocal;
    bool isTail;
};

struct ActivationParams {
    uint64_t ubSize;
    bool xCastFlag;
    uint64_t activateType;
};

struct TileShardParams {
    uint64_t tileCWhole;
    uint64_t tileEWhole;
    uint64_t m;
    bool isLocal;
    bool isTail;
};

static void PrintCommonTilingVariables(AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - epGroupSize is %u.",
            tilingData.commonTiling.get_epGroupSize());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - tpGroupSize is %u.",
            tilingData.commonTiling.get_tpGroupSize());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - expert is %lu.",
            tilingData.commonTiling.get_expert());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - EOverEp is %lu.",
            tilingData.commonTiling.get_EOverEp());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - C is %lu.", tilingData.commonTiling.get_C());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - COverTp is %lu.",
            tilingData.commonTiling.get_COverTp());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - H is %lu.", tilingData.commonTiling.get_H());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - HOverTp is %lu.",
            tilingData.commonTiling.get_HOverTp());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - MOverTp is %lu.",
            tilingData.commonTiling.get_MOverTp());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - aivCoreNum is %u.",
            tilingData.commonTiling.get_aivCoreNum());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - inputDatasize is %u.",
            tilingData.commonTiling.get_inputDatasize());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - biasDatasize is %u.",
            tilingData.commonTiling.get_biasDatasize());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - ubCapacityForTrans is %lu.",
            tilingData.commonTiling.get_ubCapacityForTrans());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - ubCapacityForAddActivate is %lu.",
            tilingData.commonTiling.get_ubCapacityForAddActivate());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - isBias is %d.",
            tilingData.commonTiling.get_isBias());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - y2Flag is %d.",
            tilingData.commonTiling.get_y2Flag());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - y3Flag is %d.",
            tilingData.commonTiling.get_y3Flag());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - isWeightTrans is %d.",
            tilingData.commonTiling.get_isWeightTrans());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - activateType is %u.",
            tilingData.commonTiling.get_activateType());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - xShardFlag is %u.",
            tilingData.commonTiling.get_xShardFlag());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - fastGeluBuffer is %u.",
            tilingData.commonTiling.get_fastGeluBuffer());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - totalUbSize is %lu.",
            tilingData.commonTiling.get_totalUbSize());
}

static void PrintSliceTileInfo(AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local E - Tile length is %lu.",
            tilingData.commonTiling.localTileE.get_tileLen());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local E - Tile Count is %lu.",
            tilingData.commonTiling.localTileE.get_tileCnt());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local E - Tail length is %lu.",
            tilingData.commonTiling.localTileE.get_tailLen());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local E - Tail Count is %lu.",
            tilingData.commonTiling.localTileE.get_tailCnt());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local C - Tile length is %lu.",
            tilingData.commonTiling.localTileC.get_tileLen());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local C - Tile Count is %lu.",
            tilingData.commonTiling.localTileC.get_tileCnt());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local C - Tail length is %lu.",
            tilingData.commonTiling.localTileC.get_tailLen());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local C - Tail Count is %lu.",
            tilingData.commonTiling.localTileC.get_tailCnt());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local E - Tile length is %lu.",
            tilingData.commonTiling.domesticTileE.get_tileLen());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local E - Tile Count is %lu.",
            tilingData.commonTiling.domesticTileE.get_tileCnt());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local E - Tail length is %lu.",
            tilingData.commonTiling.domesticTileE.get_tailLen());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local E - Tail Count is %lu.",
            tilingData.commonTiling.domesticTileE.get_tailCnt());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local C - Tile length is %lu.",
            tilingData.commonTiling.domesticTileC.get_tileLen());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local C - Tile Count is %lu.",
            tilingData.commonTiling.domesticTileC.get_tileCnt());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local C - Tail length is %lu.",
            tilingData.commonTiling.domesticTileC.get_tailLen());
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local C - Tail Count is %lu.",
            tilingData.commonTiling.domesticTileC.get_tailCnt());
}

static matmul_tiling::DataType GetMatMulTilingDataType(ge::DataType geDtype)
{
    if (geDtype == ge::DT_BF16) {
        return matmul_tiling::DataType::DT_BFLOAT16;
    } else if (geDtype == ge::DT_FLOAT16) {
        return matmul_tiling::DataType::DT_FLOAT16;
    } else {
        return matmul_tiling::DataType::DT_FLOAT;
    }
}

static uint32_t GetDataSize(ge::DataType geDtype)
{
    if (geDtype == ge::DT_BF16 || geDtype == ge::DT_FLOAT16) {
        return FP16_DATASIZE;
    } else {
        return FP32_DATASIZE;
    }
}

static void InitTileInfoInCommonTiling(AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    tilingData.commonTiling.localTileE.set_tileCnt(1);
    tilingData.commonTiling.localTileE.set_tileLen(tilingData.commonTiling.get_EOverEp());
    tilingData.commonTiling.localTileE.set_tailCnt(0);
    tilingData.commonTiling.localTileE.set_tailLen(0);

    tilingData.commonTiling.domesticTileE.set_tileCnt(1);
    tilingData.commonTiling.domesticTileE.set_tileLen(tilingData.commonTiling.get_EOverEp());
    tilingData.commonTiling.domesticTileE.set_tailCnt(0);
    tilingData.commonTiling.domesticTileE.set_tailLen(0);

    tilingData.commonTiling.localTileC.set_tileCnt(1);
    if (tilingData.commonTiling.get_xShardFlag() == 1) {
        tilingData.commonTiling.localTileC.set_tileLen(tilingData.commonTiling.get_C() /
                                                       tilingData.commonTiling.get_tpGroupSize());
    } else {
        tilingData.commonTiling.localTileC.set_tileLen(tilingData.commonTiling.get_C());
    }
    tilingData.commonTiling.localTileC.set_tailCnt(0);
    tilingData.commonTiling.localTileC.set_tailLen(0);

    tilingData.commonTiling.domesticTileC.set_tileCnt(1);
    if (tilingData.commonTiling.get_xShardFlag() == 1) {
        tilingData.commonTiling.domesticTileC.set_tileLen(tilingData.commonTiling.get_C() /
                                                          tilingData.commonTiling.get_tpGroupSize());
    } else {
        tilingData.commonTiling.domesticTileC.set_tileLen(tilingData.commonTiling.get_C());
    }
    tilingData.commonTiling.domesticTileC.set_tailCnt(0);
    tilingData.commonTiling.domesticTileC.set_tailLen(0);
}

static void UpdateTileInfo(TileInfo &tileInfo, uint64_t tileCnt, uint64_t tileLen, uint64_t tailCnt, uint64_t tailLen)
{
    tileInfo.set_tailCnt(tailCnt);
    tileInfo.set_tailLen(tailLen);
    tileInfo.set_tileCnt(tileCnt);
    tileInfo.set_tileLen(tileLen);
}

static void CheckTransposeUBAndUpdateTileShard(TransposeConfig &config,
                                               AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    uint64_t H = tilingData.commonTiling.get_H();
    uint64_t ep = tilingData.commonTiling.get_epGroupSize();
    uint64_t tp = tilingData.commonTiling.get_tpGroupSize();
    uint64_t coreNum = tilingData.commonTiling.get_aivCoreNum();

    uint64_t ubCapacity = static_cast<uint64_t>(config.ubSize) / config.inputDatasize;
    tilingData.commonTiling.set_ubCapacityForTrans(ubCapacity);

    uint64_t totalDataSize = (config.isLocal == true) ?
                                 tp * config.tileEWhole * config.tileCWhole * H :
                                 static_cast<uint64_t>(tp) * static_cast<uint64_t>(config.tileEWhole) *
                                     (static_cast<uint64_t>(ep) - 1U) * static_cast<uint64_t>(config.tileCWhole) *
                                     static_cast<uint64_t>(H);
    if (tilingData.commonTiling.get_xShardFlag() == 0) {
        totalDataSize = (config.isLocal == true) ?
                            config.tileEWhole * config.tileCWhole * H :
                            static_cast<uint64_t>(config.tileEWhole) * (static_cast<uint64_t>(ep) - 1U) *
                                static_cast<uint64_t>(config.tileCWhole) * static_cast<uint64_t>(H);
    }

    uint64_t tileLen = (totalDataSize + coreNum - 1) / coreNum;
    uint64_t tileCnt = 1;
    uint64_t tailLen = totalDataSize - tileLen * tileCnt;
    uint64_t tailCnt = (tailLen == 0UL) ? 0UL : 1UL;

    TileInfo &tileInfo = config.isLocal ? (config.isTail ? tilingData.commonTiling.localTailUbTranspose :
                                                           tilingData.commonTiling.localUbTranspose) :
                                          (config.isTail ? tilingData.commonTiling.domesticTailUbTranspose :
                                                           tilingData.commonTiling.domesticUbTranspose);
    UpdateTileInfo(tileInfo, tileCnt, tileLen, tailCnt, tailLen);
    return;
}

static void CheckAddActivateUB(ActivationParams &actParams, AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    uint64_t ubCapacity = 1U;
    uint64_t actIn = 2U;
    uint64_t actCast = 0U;
    uint64_t addRight = 0U;
    uint64_t fastGelu = 0U;
    uint64_t actOut = 0U;
    uint64_t coeff = 1U;

    if (tilingData.commonTiling.get_isBias()) {
        if (actParams.xCastFlag == true) {
            actCast = FP32_DATASIZE;
            actOut = FP32_DATASIZE;
        } else {
            actOut = FP16_DATASIZE;
        }
        addRight = tilingData.commonTiling.get_biasDatasize();
    }

    if (actParams.activateType != 0U) {
        if (actParams.xCastFlag == true) {
            actCast = FP32_DATASIZE;
            actOut = FP32_DATASIZE;
        } else {
            actOut = FP16_DATASIZE;
        }
    }

    if ((actParams.activateType == FASTGELU) || (actParams.activateType == GELU)) {
        if (actParams.xCastFlag == true) {
            fastGelu = FP32_DATASIZE;
        } else {
            fastGelu = FP16_DATASIZE;
        }
    }

    coeff = actIn + actCast + actOut + addRight + fastGelu;
    ubCapacity = (actParams.ubSize - FASTGELU_MINSIZE) / static_cast<uint64_t>(coeff);
    ubCapacity = ubCapacity / ALIGN16 * ALIGN16;
    tilingData.commonTiling.set_ubCapacityForAddActivate(ubCapacity);
    tilingData.commonTiling.set_fastGeluBuffer(
        std::max(static_cast<uint64_t>(FASTGELU_MINSIZE), fastGelu * ubCapacity));
}

static void UpdateTileShard(TileShardParams &params, AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    uint64_t ep = tilingData.commonTiling.get_epGroupSize();
    uint64_t tp = tilingData.commonTiling.get_tpGroupSize();
    uint64_t coreNum = tilingData.commonTiling.get_aivCoreNum();

    uint64_t totalDataSize = (params.isLocal == true) ?
                                 static_cast<uint64_t>(tp) * params.tileEWhole * params.tileCWhole * params.m :
                                 static_cast<uint64_t>(tp) * params.tileEWhole * (static_cast<uint64_t>(ep) - 1U) *
                                     static_cast<uint64_t>(params.tileCWhole) * static_cast<uint64_t>(params.m);
    if (tilingData.commonTiling.get_xShardFlag() == 0) {
        totalDataSize = (params.isLocal == true) ?
                            params.tileEWhole * params.tileCWhole * params.m :
                            static_cast<uint64_t>(params.tileEWhole) * (static_cast<uint64_t>(ep) - 1U) *
                                static_cast<uint64_t>(params.tileCWhole) * static_cast<uint64_t>(params.m);
    }

    uint64_t tileLen = (totalDataSize + coreNum - 1U) / static_cast<uint64_t>(coreNum);
    uint64_t tileCnt = 1U;
    uint64_t tailLen = totalDataSize - tileLen * tileCnt;
    uint64_t tailCnt = (tailLen == 0U) ? 0U : 1U;

    TileInfo &tileInfo =
        params.isLocal ?
            (params.isTail ? tilingData.commonTiling.localTailUbAdd : tilingData.commonTiling.localUbAdd) :
            (params.isTail ? tilingData.commonTiling.domesticTailUbAdd : tilingData.commonTiling.domesticUbAdd);
    UpdateTileInfo(tileInfo, tileCnt, tileLen, tailCnt, tailLen);
    return;
}

static void CheckAddActivateUBAndUpdateTileShard(ActivationParams &actParams, TileShardParams &params,
                                                 AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    CheckAddActivateUB(actParams, tilingData);
    UpdateTileShard(params, tilingData);
}

static void UpdateTilingKey(uint64_t &tilingKey, AlltoAllAllGatherBatchMatMulTilingData &tilingData, bool y2Flag,
                            bool y3Flag)
{
    tilingKey += (tilingData.commonTiling.get_xShardFlag() == 1) ? TILINGKEY_X_SHARD : 0;
    tilingKey += (tilingData.commonTiling.get_isWeightTrans() == true) ? TILINGKEY_WEIGHT_TRANSPOSE : 0;
    tilingKey += (tilingData.commonTiling.get_isBias() == true) ? TILINGKEY_IS_BIAS : 0;
    if (y2Flag && y3Flag) {
        tilingKey += TILINGKEY_Y2_Y3;
    } else if (y2Flag) {
        tilingKey += TILINGKEY_Y2_ONLY;
    } else if (y3Flag) {
        tilingKey += TILINGKEY_Y3_ONLY;
    }

    OP_LOGD("AlltoAllAllGatherBatchMatMul", "The final tiling key is %lu.", tilingKey);
    return;
}

static void CompleteBmmStructs(AlltoAllAllGatherBatchInfo &BMMV3BatchInfo, AlltoAllAllGatherMatmulInfo &MMV3ArgsInfo,
                               uint32_t m, uint32_t n, uint32_t k, uint32_t b)
{
    MMV3ArgsInfo.mValue = m;
    MMV3ArgsInfo.nValue = n;
    MMV3ArgsInfo.kValue = k;
    BMMV3BatchInfo.batchA = b;
    BMMV3BatchInfo.batchA3 = b;
    BMMV3BatchInfo.batchB = b;
    BMMV3BatchInfo.batchB3 = b;
    BMMV3BatchInfo.batchC = b;
    BMMV3BatchInfo.batchC3 = b;

    return;
}

static void GetAlltoAllAllGatherFormulateTileCnt(mc2tiling::TilingArgs &formulaicArgs,
                                                 AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    All2AllAllGatherBMM formulaicTiling(formulaicArgs, tilingData.commonTiling.get_epGroupSize(),
                                        tilingData.commonTiling.get_tpGroupSize(),
                                        tilingData.commonTiling.get_EOverEp());
    formulaicTiling.GetTiling();

    tilingData.commonTiling.localTileE.set_tileCnt(formulaicTiling.localCutE.numLongTile);
    tilingData.commonTiling.localTileE.set_tailCnt(formulaicTiling.localCutE.numShortTile);
    tilingData.commonTiling.localTileE.set_tileLen(formulaicTiling.localCutE.longTileLen);
    tilingData.commonTiling.localTileE.set_tailLen(formulaicTiling.localCutE.shortTileLen);

    tilingData.commonTiling.domesticTileE.set_tileCnt(formulaicTiling.cutE.numLongTile);
    tilingData.commonTiling.domesticTileE.set_tailCnt(formulaicTiling.cutE.numShortTile);
    tilingData.commonTiling.domesticTileE.set_tileLen(formulaicTiling.cutE.longTileLen);
    tilingData.commonTiling.domesticTileE.set_tailLen(formulaicTiling.cutE.shortTileLen);

    tilingData.commonTiling.domesticTileC.set_tileCnt(formulaicTiling.tilingC.cutRes.numLongTile);
    tilingData.commonTiling.domesticTileC.set_tileLen(formulaicTiling.tilingC.cutRes.longTileLen /
                                                      tilingData.commonTiling.get_tpGroupSize());
    tilingData.commonTiling.domesticTileC.set_tailCnt(formulaicTiling.tilingC.cutRes.numShortTile);
    tilingData.commonTiling.domesticTileC.set_tailLen(formulaicTiling.tilingC.cutRes.shortTileLen /
                                                      tilingData.commonTiling.get_tpGroupSize());

    return;
}

static void GetAlltoAllAllGatherFormulateTileCntShardH(mc2tiling::TilingArgs &formulaicArgs,
                                                       AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    All2AllAllGatherBMMShardH formulaicTiling(formulaicArgs, tilingData.commonTiling.get_epGroupSize(),
                                              tilingData.commonTiling.get_tpGroupSize(),
                                              tilingData.commonTiling.get_EOverEp());
    formulaicTiling.GetTiling();
    if (formulaicTiling.localCutE.shortTileAtBack) { // local E短块后置
        tilingData.commonTiling.localTileE.set_tileCnt(formulaicTiling.localCutE.numLongTile);
        tilingData.commonTiling.localTileE.set_tailCnt(formulaicTiling.localCutE.numShortTile);
        tilingData.commonTiling.localTileE.set_tileLen(formulaicTiling.localCutE.longTileLen);
        tilingData.commonTiling.localTileE.set_tailLen(formulaicTiling.localCutE.shortTileLen);
    } else { // local E短块前置
        tilingData.commonTiling.localTileE.set_tileCnt(formulaicTiling.localCutE.numShortTile);
        tilingData.commonTiling.localTileE.set_tailCnt(formulaicTiling.localCutE.numLongTile);
        tilingData.commonTiling.localTileE.set_tileLen(formulaicTiling.localCutE.shortTileLen);
        tilingData.commonTiling.localTileE.set_tailLen(formulaicTiling.localCutE.longTileLen);
    }

    if (formulaicTiling.cutE.shortTileAtBack) { // non-local E短块后置
        tilingData.commonTiling.domesticTileE.set_tileCnt(formulaicTiling.cutE.numLongTile);
        tilingData.commonTiling.domesticTileE.set_tailCnt(formulaicTiling.cutE.numShortTile);
        tilingData.commonTiling.domesticTileE.set_tileLen(formulaicTiling.cutE.longTileLen);
        tilingData.commonTiling.domesticTileE.set_tailLen(formulaicTiling.cutE.shortTileLen);
    } else { // non-local E短块前置
        tilingData.commonTiling.domesticTileE.set_tileCnt(formulaicTiling.cutE.numShortTile);
        tilingData.commonTiling.domesticTileE.set_tailCnt(formulaicTiling.cutE.numLongTile);
        tilingData.commonTiling.domesticTileE.set_tileLen(formulaicTiling.cutE.shortTileLen);
        tilingData.commonTiling.domesticTileE.set_tailLen(formulaicTiling.cutE.longTileLen);
    }

    if (formulaicTiling.tilingC.cutRes.shortTileAtBack) { // non-local C短块后置
        tilingData.commonTiling.domesticTileC.set_tileCnt(formulaicTiling.tilingC.cutRes.numLongTile);
        tilingData.commonTiling.domesticTileC.set_tailCnt(formulaicTiling.tilingC.cutRes.numShortTile);
        tilingData.commonTiling.domesticTileC.set_tileLen(formulaicTiling.tilingC.cutRes.longTileLen);
        tilingData.commonTiling.domesticTileC.set_tailLen(formulaicTiling.tilingC.cutRes.shortTileLen);
    } else { // non-local C短块前置
        tilingData.commonTiling.domesticTileC.set_tileCnt(formulaicTiling.tilingC.cutRes.numShortTile);
        tilingData.commonTiling.domesticTileC.set_tailCnt(formulaicTiling.tilingC.cutRes.numLongTile);
        tilingData.commonTiling.domesticTileC.set_tileLen(formulaicTiling.tilingC.cutRes.shortTileLen);
        tilingData.commonTiling.domesticTileC.set_tailLen(formulaicTiling.tilingC.cutRes.longTileLen);
    }

    return;
}

static ge::graphStatus HandleLocalBmmTilingData(gert::TilingContext *context,
                                                AlltoAllAllGatherBatchMatMulTilingData &tilingData,
                                                AlltoAllAllGatherBatchInfo bmmV3BatchInfo,
                                                AlltoAllAllGatherMatmulInfo mmV3ArgsInfo)
{
    // 将切块信息传入BmmV3 tiling，根据后续整改方案更新Bmm所需的相应参数并调用BmmV3
    // Tiling接口，获取local/non-local的BMM tiling信息 Local standard slice BMM tiling
    if (tilingData.commonTiling.get_xShardFlag() == 1U) {
        CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                           tilingData.commonTiling.get_tpGroupSize() * tilingData.commonTiling.localTileC.get_tileLen(),
                           tilingData.commonTiling.get_MOverTp(), tilingData.commonTiling.get_H(),
                           tilingData.commonTiling.localTileE.get_tileLen());
    } else {
        CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo, tilingData.commonTiling.localTileC.get_tileLen(),
                           tilingData.commonTiling.get_MOverTp(), tilingData.commonTiling.get_H(),
                           tilingData.commonTiling.localTileE.get_tileLen());
    }
    AlltoAllAllGatherBatchMatMulTiling bmmTilingLocalStd(context, tilingData.localTiling.bmmTilingData, bmmV3BatchInfo,
                                                         mmV3ArgsInfo);
    OP_TILING_CHECK(
        bmmTilingLocalStd.DoTiling() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Do BmmV3Tiling failed under local standard section."),
        return ge::GRAPH_FAILED);

    // Local tail slice BMM tiling
    if (tilingData.commonTiling.localTileE.get_tailLen() != 0) {
        if (tilingData.commonTiling.get_xShardFlag() == 1U) {
            CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                               tilingData.commonTiling.get_tpGroupSize() *
                                   tilingData.commonTiling.localTileC.get_tailLen(),
                               tilingData.commonTiling.get_MOverTp(), tilingData.commonTiling.get_H(),
                               tilingData.commonTiling.localTileE.get_tileLen());
        } else {
            CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo, tilingData.commonTiling.localTileC.get_tileLen(),
                               tilingData.commonTiling.get_MOverTp(), tilingData.commonTiling.get_H(),
                               tilingData.commonTiling.localTileE.get_tailLen());
        }
        AlltoAllAllGatherBatchMatMulTiling bmmTilingLocalTail(context, tilingData.localTailTiling.bmmTilingData,
                                                              bmmV3BatchInfo, mmV3ArgsInfo);
        OP_TILING_CHECK(
            bmmTilingLocalTail.DoTiling() != ge::GRAPH_SUCCESS,
            VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Do BmmV3Tiling failed under local tail section."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HandleNoneLocalBmmTilingData(gert::TilingContext *context,
                                                    AlltoAllAllGatherBatchMatMulTilingData &tilingData,
                                                    AlltoAllAllGatherBatchInfo bmmV3BatchInfo,
                                                    AlltoAllAllGatherMatmulInfo mmV3ArgsInfo)
{
    // Non-local standard slice BMM tiling
    if (tilingData.commonTiling.get_xShardFlag() == 1U) {
        CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                           (tilingData.commonTiling.get_epGroupSize() - 1) * tilingData.commonTiling.get_tpGroupSize() *
                               tilingData.commonTiling.domesticTileC.get_tileLen(),
                           tilingData.commonTiling.get_MOverTp(), tilingData.commonTiling.get_H(),
                           tilingData.commonTiling.domesticTileE.get_tileLen());
    } else {
        CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                           (tilingData.commonTiling.get_epGroupSize() - 1) *
                               tilingData.commonTiling.domesticTileC.get_tileLen(),
                           tilingData.commonTiling.get_MOverTp(), tilingData.commonTiling.get_H(),
                           tilingData.commonTiling.domesticTileE.get_tileLen());
    }
    AlltoAllAllGatherBatchMatMulTiling bmmTilingDomesticStd(context, tilingData.domesticTiling.bmmTilingData,
                                                            bmmV3BatchInfo, mmV3ArgsInfo);
    OP_TILING_CHECK(bmmTilingDomesticStd.DoTiling() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
                                                    "Do BmmV3Tiling failed under domestic standard section."),
                    return ge::GRAPH_FAILED);

    // Non-local tail slice BMM tiling
    if (tilingData.commonTiling.domesticTileC.get_tailLen() != 0) {
        if (tilingData.commonTiling.get_xShardFlag() == 1U) {
            CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                               (tilingData.commonTiling.get_epGroupSize() - 1) *
                                   tilingData.commonTiling.get_tpGroupSize() *
                                   tilingData.commonTiling.domesticTileC.get_tailLen(),
                               tilingData.commonTiling.get_MOverTp(), tilingData.commonTiling.get_H(),
                               tilingData.commonTiling.domesticTileE.get_tileLen());
        } else {
            CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                               (tilingData.commonTiling.get_epGroupSize() - 1) *
                                   tilingData.commonTiling.domesticTileC.get_tailLen(),
                               tilingData.commonTiling.get_MOverTp(), tilingData.commonTiling.get_H(),
                               tilingData.commonTiling.domesticTileE.get_tileLen());
        }
        AlltoAllAllGatherBatchMatMulTiling bmmTilingDomesticTail(context, tilingData.domesticTailTiling.bmmTilingData,
                                                                 bmmV3BatchInfo, mmV3ArgsInfo);
        OP_TILING_CHECK(bmmTilingDomesticTail.DoTiling() != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
                                                        "Do BmmV3Tiling failed under domestic tail section."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetMatmulTilingAlltoAllAllGatherBatchMatMul(gert::TilingContext *context,
                                                                   AlltoAllAllGatherBatchMatMulTilingData &tilingData,
                                                                   AlltoAllAllGatherBatchInfo BMMV3BatchInfo,
                                                                   AlltoAllAllGatherMatmulInfo MMV3ArgsInfo,
                                                                   mc2tiling::TilingArgs &formulaicArgs)
{
    // 1. 调用公式化tiling接口，获取local和非local块的切分信息并更新commonTiling
    if (tilingData.commonTiling.get_xShardFlag() == 1U) {
        GetAlltoAllAllGatherFormulateTileCnt(formulaicArgs, tilingData);
    } else {
        GetAlltoAllAllGatherFormulateTileCntShardH(formulaicArgs, tilingData);
    }

    OP_TILING_CHECK(HandleLocalBmmTilingData(context, tilingData, BMMV3BatchInfo, MMV3ArgsInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Hanle local bmm tiling data failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(HandleNoneLocalBmmTilingData(context, tilingData, BMMV3BatchInfo, MMV3ArgsInfo) !=
                        ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Hanle none local bmm tiling data failed."),
                    return ge::GRAPH_FAILED);

    // shard-0 Non_local tail E slice BMM tiling
    if ((tilingData.commonTiling.get_xShardFlag() == 0U) &&
        (tilingData.commonTiling.domesticTileC.get_tileLen() == tilingData.commonTiling.get_C()) &&
        (tilingData.commonTiling.domesticTileE.get_tailLen() != 0)) {
        CompleteBmmStructs(BMMV3BatchInfo, MMV3ArgsInfo,
                           (tilingData.commonTiling.get_epGroupSize() - 1) *
                               tilingData.commonTiling.domesticTileC.get_tileLen(),
                           tilingData.commonTiling.get_MOverTp(), tilingData.commonTiling.get_H(),
                           tilingData.commonTiling.domesticTileE.get_tailLen());
        AlltoAllAllGatherBatchMatMulTiling bmmTilingDomesticTailE(context, tilingData.domesticTailETiling.bmmTilingData,
                                                                  BMMV3BatchInfo, MMV3ArgsInfo);
        if (bmmTilingDomesticTailE.DoTiling() != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(), "Do BmmV3Tiling failed under shard-0 domestic tail C section.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

const char *K_INNER_DEBUG = "MC2: AlltoAllAllGatherBmm Tiling Shape Debug";

static bool ActTypeCheck(const char *nodeName, const int64_t actType, const bool y3Flag)
{
    if (std::find(ops::ACT_TYPE_SUPPORT_VEC.begin(), ops::ACT_TYPE_SUPPORT_VEC.end(), actType) ==
        ops::ACT_TYPE_SUPPORT_VEC.end()) {
        OP_LOGE(nodeName, "actType [%ld] is unsupported, support range is [%ld, %ld]", actType,
                static_cast<int64_t>(
                    ops::AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE),
                static_cast<int64_t>(
                    ops::AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU));
        return false;
    }

    if (y3Flag &&
        (actType == static_cast<int64_t>(
                        ops::AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE))) {
        OP_LOGE(nodeName, "actType should be non-none when need_activation_feature is true, but got actType = none.");
        return false;
    }

    return true;
}

static bool CommonCheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape,
                                   const size_t wDimH)
{
    // 检查每个维度: x dim C >= 1，dim E H M 会在后面限制, 这里不再做校验
    if (xShape->GetDim(X_DIM_C) < VALUE_C_MIN) {
        OP_LOGE(nodeName, "The second dim of x should not < %ld, but got x[1] = %ld.", VALUE_C_MIN,
                xShape->GetDim(X_DIM_C));
        return false;
    }
    // x[2]、w[wDimH] 是 H 轴 (reduce 轴)，不能为 0
    if ((xShape->GetDim(X_DIM_H) == 0) || (weightShape->GetDim(wDimH) == 0)) {
        OP_LOGE(nodeName, "The second dim of weight or the last dim of x = 0 is unsupported.");
        return false;
    }

    return true;
}

// GPT 2.2T
static bool XShardCheckTensorShape(const char *nodeName, const int64_t xShard, const gert::Shape *xShape,
                                   const gert::Shape *weightShape, const int64_t epSize, const int64_t tpSize,
                                   const size_t wDimH, const size_t wDimM)
{
    // 检查 shape 维度的范围
    // x[DIM_E] = E, value E should = [2, 512]
    if (((xShape->GetDim(DIM_E) < VALUE_E_MIN) || (xShape->GetDim(DIM_E) > VALUE_E_MAX))) {
        OP_LOGE(nodeName, "Value E should in [%ld, %ld], but got %ld.", VALUE_E_MIN, VALUE_E_MAX,
                xShape->GetDim(DIM_E));
        return false;
    }
    // w[wDimM] = M / Tp, its range should same with H, so it meets M / Tp * H <= 65535 * 65535
    if ((weightShape->GetDim(wDimM) < VALUE_H_MIN) || (weightShape->GetDim(wDimM) > VALUE_H_MAX)) {
        OP_LOGE(nodeName, "Value M / Tp should in [%ld, %ld], but got %ld.", VALUE_H_MIN, VALUE_H_MAX,
                weightShape->GetDim(wDimM));
        return false;
    }
    // x[DIM_E] = E, w[DIM_E] = E / Ep, 所以需要满足 x[DIM_E] = w[DIM_E] * Ep
    if (weightShape->GetDim(DIM_E) * epSize != xShape->GetDim(DIM_E)) {
        OP_LOGE(nodeName,
                "The first dim of weight multi epSize should equal the first dim of x,"
                "but got x[0] = %ld, w[0] = %ld, epSize = %ld",
                xShape->GetDim(DIM_E), weightShape->GetDim(DIM_E), epSize);
        return false;
    }

    if (xShard == 0) {
        // x[X_DIM_H] = H / tp, value H should = [1, 65535]
        if ((xShape->GetDim(X_DIM_H) * tpSize < VALUE_H_MIN) || (xShape->GetDim(X_DIM_H) * tpSize > VALUE_H_MAX)) {
            OP_LOGE(nodeName, "Value H should in [%ld, %ld], but got %ld.", VALUE_H_MIN, VALUE_H_MAX,
                    xShape->GetDim(X_DIM_H) * tpSize);
            return false;
        }
        // x[X_DIM_H] = H / tp, w[wDimH] = H, 所以 x[X_DIM_H] * tp 需要等于 w[wDimH]
        if (xShape->GetDim(X_DIM_H) * tpSize != weightShape->GetDim(wDimH)) {
            OP_LOGE(nodeName,
                    "The last dim of x (H / tp) multi tp should equal the corresponding dim of weight, "
                    "but got x[2] * tp = %ld, w[%lu] = %ld.",
                    xShape->GetDim(X_DIM_H), wDimH, weightShape->GetDim(wDimH));
            return false;
        }
    } else if (xShard == 1) {
        // x[X_DIM_H] = H, value H should = [1, 65535]
        if ((xShape->GetDim(X_DIM_H) < VALUE_H_MIN) || (xShape->GetDim(X_DIM_H) > VALUE_H_MAX)) {
            OP_LOGE(nodeName, "Value H should in [%ld, %ld], but got %ld.", VALUE_H_MIN, VALUE_H_MAX,
                    xShape->GetDim(X_DIM_H));
            return false;
        }
        // x[X_DIM_H] = H, w[wDimH] = H, 所以 x[X_DIM_H] 需要等于 w[wDimH]
        if (xShape->GetDim(X_DIM_H) != weightShape->GetDim(wDimH)) {
            OP_LOGE(nodeName,
                    "The last dim of x should equal the corresponding dim of weight, "
                    "but got x[2] = %ld, w[%lu] = %ld.",
                    xShape->GetDim(X_DIM_H), wDimH, weightShape->GetDim(wDimH));
            return false;
        }
    }

    return true;
}

static bool CheckBiasShape(const char *nodeName, const gert::Shape *weightShape, const gert::Shape *biasShape,
                           const size_t wDimM)
{
    // 检查 dimNum
    if ((biasShape->GetDimNum() != SUPPORT_DIM_NUM) && (biasShape->GetDimNum() != BIAS_SUPPORT_DIM_NUM)) {
        OP_LOGE(nodeName, "Dim of input bias must be the 2 or 3.");
        return false;
    }

    // 检查 shape
    if (biasShape->GetDim(0) != weightShape->GetDim(0)) {
        OP_LOGE(nodeName,
                "The first dim of bias must be equal the first dim of weight, "
                "but got bias[0] = %ld, w[0] = %ld.",
                biasShape->GetDim(0), weightShape->GetDim(0));
        return false;
    }

    size_t biasLastDimIdx = 1U; // 默认 bias 是二维，所以最后一维的 index 是 1
    if (biasShape->GetDimNum() == SUPPORT_DIM_NUM) {
        if (biasShape->GetDim(1) != 1) { // 三维时候，中间维度为 1
            OP_LOGE(nodeName, "The second dim of bias must be 1 when dim num is 3.");
            return false;
        }
        biasLastDimIdx = 2; // 三维时候，bias 的最后一维是 2
    }

    if (biasShape->GetDim(biasLastDimIdx) != weightShape->GetDim(wDimM)) {
        OP_LOGE(nodeName,
                "The last dim of bias must equal the corresponding dim of weight, "
                "but got bias[2] = %ld, w[%lu] = %ld.",
                biasShape->GetDim(biasLastDimIdx), wDimM, weightShape->GetDim(wDimM));
        return false;
    }

    return true;
}

static bool CheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape,
                             const gert::Shape *biasShape, const int64_t epSize, const int64_t tpSize,
                             const size_t wDimH, const size_t wDimM, const int64_t xShard)
{
    if ((xShape == nullptr) || (weightShape == nullptr)) {
        OP_LOGE(nodeName, "xShape or weightShape is nullptr.");
        return false;
    }
    if (!DimNumCheck(nodeName, xShape, weightShape)) {
        OP_LOGE(nodeName, "Dim num check failed.");
        return false;
    }

    if (!CommonCheckTensorShape(nodeName, xShape, weightShape, wDimH)) {
        OP_LOGE(nodeName, "common tensor shape check failed.");
        return false;
    }

    if (xShard == 0 || xShard == 1) {
        if (!XShardCheckTensorShape(nodeName, xShard, xShape, weightShape, epSize, tpSize, wDimH, wDimM)) {
            OP_LOGE(nodeName, "xShard = [%ld] tensor shape check failed.", xShard);
            return false;
        }
    }

    if (biasShape != nullptr) {
        if (!(CheckBiasShape(nodeName, weightShape, biasShape, wDimM))) {
            OP_LOGE(nodeName, "Bias shape check failed.");
            return false;
        }
    }

    OP_LOGI(nodeName, "Tensor shape check success.");
    return true;
}

static bool CheckAttrs(const gert::TilingContext *context, int64_t &epSize, int64_t &tpSize, int64_t &xShard,
                       bool &y2Flag, bool &y3Flag)
{
    const char *nodeName = context->GetNodeName();

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null"), return false);

    // get 只有在 index 超出 attr num 的时候才会返回 nullptr
    const char *groupEp = attrs->GetStr(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_GROUP_EP));
    const char *groupTp = attrs->GetStr(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_GROUP_TP));
    const int64_t *tpWorldSize = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_TP_WORLD_SIZE));
    const int64_t *epWorldSize = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_EP_WORLD_SIZE));
    const int64_t *xShardType = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_X_SHARD_TYPE));
    const int64_t *actTypePtr = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_ACT_TYPE));
    const bool *outputY2Flag = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_OUTPUT_Y2_FLAG));
    const bool *outputY3Flag = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_OUTPUT_Y3_FLAG));

    if ((tpWorldSize == nullptr) || (epWorldSize == nullptr)) {
        OP_LOGE(nodeName, "tpWorldSize or epWorldSize in context is invalid or out of range, attrs got nullptr.");
        return false;
    }

    if ((xShardType == nullptr) || (actTypePtr == nullptr)) {
        OP_LOGE(nodeName, "xShardType or actTypePtr in context is invalid or out of range, attrs got nullptr.");
        return false;
    }

    if ((outputY2Flag == nullptr) || (outputY3Flag == nullptr)) {
        OP_LOGE(nodeName, "outputY2Flag or outputY3Flag in context is invalid or out of range, attrs got nullptr.");
        return false;
    }

    tpSize = *tpWorldSize;
    epSize = *epWorldSize;
    xShard = *xShardType;
    y2Flag = *outputY2Flag;
    y3Flag = *outputY3Flag;
    const int64_t actType = *actTypePtr;

    if (!GroupCheck(nodeName, groupEp, groupTp)) {
        OP_LOGE(nodeName, "group size check failed.");
        return false;
    }

    if (!EpTpSizeCheck(epSize, tpSize)) {
        OP_LOGE(nodeName, "rank size error, tpSize=[%ld], valid=[2/4/8/16/32], epSize=[%ld], valid=[2/4/8/16/32].",
                tpSize, epSize);
        return false;
    }

    if (xShard != 0 && xShard != 1) { // 当前支持 0, 1
        OP_LOGE(nodeName, "x shard type [%ld] is invalid.", xShard);
        return false;
    }

    if (!ActTypeCheck(nodeName, actType, y3Flag)) {
        OP_LOGE(nodeName, "actType check failed.");
        return false;
    }

    OP_LOGI(nodeName,
            "attrs info: groupEp %s, groupTp %s, tpSize %ld, epSize %ld, xShard %ld, "
            "y2Flag %d, y3Flag %d.",
            groupEp, groupTp, tpSize, epSize, xShard, y2Flag, y3Flag);
    return true;
}

// 入参校验
static ge::graphStatus TilingCheckAlltoAllAllGatherBatchMatMul(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter AlltoAllAllGatherBmm tiling check impl.");

    // 检查 shape 是否为空
    const gert::StorageShape *xStorageShape = context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "xShape is null."), return ge::GRAPH_FAILED);
    const gert::StorageShape *weightStorageShape =
        context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_WEIGHT));
    OP_TILING_CHECK(weightStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "weightShape is null."),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape *y1StorageShape =
        context->GetOutputShape(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y1));
    OP_TILING_CHECK(y1StorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "y1Shape is null."), return ge::GRAPH_FAILED);

    // 检查属性
    int64_t epSize = -1;
    int64_t tpSize = -1;
    int64_t xShard = -1;
    bool y2Flag = false;
    bool y3Flag = false;
    if (!CheckAttrs(context, epSize, tpSize, xShard, y2Flag, y3Flag)) {
        OP_LOGE(nodeName, "attrs check failed.");
        return ge::GRAPH_FAILED;
    }

    // 判断可选参数
    if (y2Flag) {
        const gert::StorageShape *y2StorageShape =
            context->GetOutputShape(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y2));
        OP_TILING_CHECK(y2StorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "y2Shape is null."), return ge::GRAPH_FAILED);
    }
    if (y3Flag) {
        const gert::StorageShape *y3StorageShape =
            context->GetOutputShape(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y3));
        OP_TILING_CHECK(y3StorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "y3Shape is null."), return ge::GRAPH_FAILED);
    }

    //  设 w = [E, H, M], dimH 指 H 轴, dimM 指 M 轴
    size_t wDimH = 1UL;
    size_t wDimM = 2UL; // 1 2 分别代表 weight 没有转置时候的维度值, 2: M 轴, 1: H 轴
    auto attrs = context->GetAttrs();
    bool isWeightTrans = *(attrs->GetAttrPointer<bool>(ATTR_IS_WEIGHT_TRANS_INDEX));
    // w_trans = [E, M, H]
    if (isWeightTrans) {
        size_t wDimHTrans = 2UL;
        size_t wDimMTrans = 1UL; // 2 1 分别代表 weight 转置时候的维度值, 2: H 轴, 1: M 轴
        wDimH = wDimHTrans;
        wDimM = wDimMTrans;
    }

    const gert::StorageShape *biasStorageShape =
        context->GetOptionalInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_BIAS));
    const gert::Shape *xShape = &xStorageShape->GetStorageShape();
    const gert::Shape *weightShape = &weightStorageShape->GetStorageShape();
    const gert::Shape *biasShape = (biasStorageShape == nullptr) ? nullptr : &biasStorageShape->GetStorageShape();

    // tensor shape 检查
    if (!CheckTensorShape(nodeName, xShape, weightShape, biasShape, epSize, tpSize, wDimH, wDimM, xShard)) {
        OP_LOGE(nodeName, "tensor shape check failed.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static uint64_t GetCommOutSize(AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    uint64_t commOut = 0UL;
    if (tilingData.commonTiling.get_xShardFlag() == 0) {
        // (E + E/ep) * C * H + E * C * H
        commOut = (static_cast<uint64_t>(tilingData.commonTiling.get_expert()) +
                   static_cast<uint64_t>(tilingData.commonTiling.get_EOverEp())) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_C()) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_H()) +
                  static_cast<uint64_t>(tilingData.commonTiling.get_expert()) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_C()) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_HOverTp());
    } else {
        // (E + E/ep) * C * H + E * C * H/tp
        commOut = (static_cast<uint64_t>(tilingData.commonTiling.get_expert()) +
                   static_cast<uint64_t>(tilingData.commonTiling.get_EOverEp())) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_C()) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_H()) +
                  static_cast<uint64_t>(tilingData.commonTiling.get_expert()) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_COverTp()) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_H()) +
                  static_cast<uint64_t>(tilingData.commonTiling.get_EOverEp()) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_epGroupSize()) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_COverTp()) *
                      static_cast<uint64_t>(tilingData.commonTiling.get_H());
    }
    return commOut * static_cast<uint64_t>(tilingData.commonTiling.get_inputDatasize());
}

static uint64_t GetTransOutSize(AlltoAllAllGatherBatchMatMulTilingData &tilingData, const uint64_t maxLenDomesticTileC,
                                const uint64_t maxLenLocalTileE, const uint64_t maxLenDomesticTileE)
{
    uint64_t transOut = 0UL;
    if (tilingData.commonTiling.get_xShardFlag() == 0) {
        // max of (local_tile_e * local_tile_c * H) and ((ep - 1) * nonLocal_tile_e * nonLocal_tile_c * H)
        transOut =
            std::max(static_cast<uint64_t>(maxLenLocalTileE) *
                         static_cast<uint64_t>(tilingData.commonTiling.localTileC.get_tileLen()) *
                         static_cast<uint64_t>(tilingData.commonTiling.get_H()),
                     static_cast<uint64_t>(tilingData.commonTiling.get_epGroupSize() - 1) *
                         static_cast<uint64_t>(maxLenDomesticTileE) * static_cast<uint64_t>(maxLenDomesticTileC) *
                         static_cast<uint64_t>(tilingData.commonTiling.get_H()));
    } else {
        // max of (local_tile_e * local_tile_c * H * tp) and ((ep - 1) * nonLocal_tile_e * nonLocal_tile_c * H * tp)
        transOut =
            std::max(static_cast<uint64_t>(maxLenLocalTileE) *
                         static_cast<uint64_t>(tilingData.commonTiling.localTileC.get_tileLen()) *
                         static_cast<uint64_t>(tilingData.commonTiling.get_H()) *
                         static_cast<uint64_t>(tilingData.commonTiling.get_tpGroupSize()),
                     static_cast<uint64_t>(tilingData.commonTiling.get_epGroupSize() - 1) *
                         static_cast<uint64_t>(maxLenDomesticTileE) * static_cast<uint64_t>(maxLenDomesticTileC) *
                         static_cast<uint64_t>(tilingData.commonTiling.get_H()) *
                         static_cast<uint64_t>(tilingData.commonTiling.get_tpGroupSize()));
    }
    return transOut * static_cast<uint64_t>(tilingData.commonTiling.get_inputDatasize());
}

static uint64_t GetBmmOutSize(AlltoAllAllGatherBatchMatMulTilingData &tilingData, const uint64_t maxLenDomesticTileC,
                              const uint64_t maxLenLocalTileE, const uint64_t maxLenDomesticTileE)
{
    uint64_t bmmOut = 0UL;
    // max of (local_tile_e * local_tile_c * M/tp) and ((ep - 1) * nonLocal_tile_e * nonLocal_tile_c * M/tp)
    if (tilingData.commonTiling.get_xShardFlag() == 0) {
        bmmOut = std::max(static_cast<uint64_t>(maxLenLocalTileE) *
                              static_cast<uint64_t>(tilingData.commonTiling.localTileC.get_tileLen()) *
                              static_cast<uint64_t>(tilingData.commonTiling.get_MOverTp()),
                          static_cast<uint64_t>(tilingData.commonTiling.get_epGroupSize() - 1) *
                              static_cast<uint64_t>(maxLenDomesticTileE) * static_cast<uint64_t>(maxLenDomesticTileC) *
                              static_cast<uint64_t>(tilingData.commonTiling.get_MOverTp()));
    } else {
        // max of (local_tile_e * local_tile_c * M/tp * tp) and ((ep - 1) * nonLocal_tile_e * nonLocal_tile_c * M/tp *
        // tp)
        bmmOut = std::max(static_cast<uint64_t>(maxLenLocalTileE) *
                              static_cast<uint64_t>(tilingData.commonTiling.localTileC.get_tileLen()) *
                              static_cast<uint64_t>(tilingData.commonTiling.get_MOverTp()) *
                              static_cast<uint64_t>(tilingData.commonTiling.get_tpGroupSize()),
                          static_cast<uint64_t>(tilingData.commonTiling.get_epGroupSize() - 1) *
                              static_cast<uint64_t>(maxLenDomesticTileE) * static_cast<uint64_t>(maxLenDomesticTileC) *
                              static_cast<uint64_t>(tilingData.commonTiling.get_MOverTp()) *
                              static_cast<uint64_t>(tilingData.commonTiling.get_tpGroupSize()));
    }
    return bmmOut * static_cast<uint64_t>(tilingData.commonTiling.get_inputDatasize());
}

// calculate workspace for GPT2.2T scenario
static ge::graphStatus MC2SetWorkspaceShard(gert::TilingContext *context,
                                            AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get workspace failed"),
                    return ge::GRAPH_FAILED);

    const uint64_t commOut = GetCommOutSize(tilingData);

    const uint64_t maxLenDomesticTileC = std::max(tilingData.commonTiling.domesticTileC.get_tileLen(),
                                                  tilingData.commonTiling.domesticTileC.get_tailLen());
    const uint64_t maxLenLocalTileE =
        std::max(tilingData.commonTiling.localTileE.get_tileLen(), tilingData.commonTiling.localTileE.get_tailLen());
    const uint64_t maxLenDomesticTileE = std::max(tilingData.commonTiling.domesticTileE.get_tileLen(),
                                                  tilingData.commonTiling.domesticTileE.get_tailLen());
    const uint64_t transOut = GetTransOutSize(tilingData, maxLenDomesticTileC, maxLenLocalTileE, maxLenDomesticTileE);
    const uint64_t bmmOut = GetBmmOutSize(tilingData, maxLenDomesticTileC, maxLenLocalTileE, maxLenDomesticTileE);

    workspaces[0] = commOut + transOut + bmmOut + 16 * 1024 * 1024; // 16 mb, 1024 * 1024 is 1 mb
    OP_LOGD("AlltoAllAllGatherBatchMatMul",
            "workspaces[0] size is %ld. commOut is %lu, transOut is %lu and"
            " bmmOut is %lu",
            workspaces[0], commOut, transOut, bmmOut);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(gert::TilingContext *context, AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    auto rawTilingData = context->GetRawTilingData();
    OP_TILING_CHECK(rawTilingData == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "GetRawTilingData returned nullptr!"),
                    return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static void InitTilingDataBasicInfo(AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    tilingData.set_version(TWO);
    tilingData.set_hcommCnt(TWO);
    tilingData.commonTiling.set_ubCapacityForTrans(0);
    tilingData.commonTiling.set_ubCapacityForAddActivate(0);
    tilingData.hcommCfgATA.set_skipLocalRankCopy(0);
    tilingData.hcommCfgATA.set_skipBufferWindowCopy(0);
    tilingData.hcommCfgATA.set_stepSize(0);
    tilingData.hcommCfgAG.set_skipLocalRankCopy(0);
    tilingData.hcommCfgAG.set_skipBufferWindowCopy(0);
    tilingData.hcommCfgAG.set_stepSize(0);
    return;
}

static void SetDataSliceInfoInCommmonTilingData(const gert::TilingContext *context,
                                                AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    const gert::StorageShape *xInputShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *weightInputShape = context->GetInputShape(WEIGHT_INDEX);
    auto ep = *(attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX));
    auto tp = *(attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX));
    auto xShard = attrs->GetAttrPointer<int64_t>(ATTR_X_SHARD_TYPE_INDEX);
    bool isWeightTrans = *(attrs->GetAttrPointer<bool>(ATTR_IS_WEIGHT_TRANS_INDEX));
    int64_t E = xInputShape->GetStorageShape().GetDim(0);
    int64_t e = weightInputShape->GetStorageShape().GetDim(0);
    int64_t C =
        (*xShard == 1) ? xInputShape->GetStorageShape().GetDim(1) * tp : xInputShape->GetStorageShape().GetDim(1);
    int64_t c =
        (*xShard == 1) ? xInputShape->GetStorageShape().GetDim(1) : xInputShape->GetStorageShape().GetDim(1) / tp;
    //  设 w = [E, H, M], dimH 指 H 轴, dimM 指 M 轴
    size_t wDimH = 1UL;
    size_t wDimM = 2UL; // 1 2 分别代表 weight 没有转置时候的维度值, 2: M 轴, 1: H 轴
    // w_trans = [E, M, H]
    if (isWeightTrans) {
        size_t wDimHTrans = 2UL;
        size_t wDimMTrans = 1UL; // 2 1 分别代表 weight 转置时候的维度值, 2: H 轴, 1: M 轴
        wDimH = wDimHTrans;
        wDimM = wDimMTrans;
    }
    int64_t dimH = weightInputShape->GetStorageShape().GetDim(wDimH);
    int64_t h = dimH / tp;
    int64_t m = weightInputShape->GetStorageShape().GetDim(wDimM);

    tilingData.commonTiling.set_epGroupSize(ep);
    tilingData.commonTiling.set_tpGroupSize(tp);
    tilingData.commonTiling.set_expert(E);
    tilingData.commonTiling.set_EOverEp(e);
    tilingData.commonTiling.set_C(C);
    tilingData.commonTiling.set_COverTp(c);
    tilingData.commonTiling.set_H(dimH);
    tilingData.commonTiling.set_HOverTp(h);
    tilingData.commonTiling.set_MOverTp(m);
    tilingData.commonTiling.set_isWeightTrans(isWeightTrans);
    tilingData.commonTiling.set_xShardFlag(*xShard);
    return;
}

static void SetDataTypeAndSizeInfoInCommmonTilingData(const gert::TilingContext *context,
                                                      AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    const gert::StorageShape *biasInputShape =
        context->GetOptionalInputTensor(BIAS_INDEX) == nullptr ? nullptr : context->GetOptionalInputShape(BIAS_INDEX);
    auto y2Flag = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_Y2_FLAG_INDEX);
    auto y3Flag = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_Y3_FLAG_INDEX);
    auto activate = attrs->GetAttrPointer<int64_t>(ATTR_ACT_TYPE_INDEX);

    ge::DataType inputDatatype = context->GetInputDesc(X_INDEX)->GetDataType();
    ge::DataType biasDatatype =
        (biasInputShape == nullptr) ? inputDatatype : context->GetOptionalInputDesc(BIAS_INDEX)->GetDataType();
    uint32_t inputDatasize = GetDataSize(inputDatatype);
    uint32_t biasDatasize = GetDataSize(biasDatatype);

    tilingData.commonTiling.set_inputDatasize(inputDatasize);
    tilingData.commonTiling.set_biasDatasize(biasDatasize);
    tilingData.commonTiling.set_isBias(biasInputShape == nullptr ? false : true);
    tilingData.commonTiling.set_y2Flag(*y2Flag);
    tilingData.commonTiling.set_y3Flag(*y3Flag);
    tilingData.commonTiling.set_activateType(*activate);
    return;
}

static void SetCommonTilingData(gert::TilingContext *context, uint64_t ubSize, uint64_t aivNum,
                                AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    SetDataSliceInfoInCommmonTilingData(context, tilingData);
    SetDataTypeAndSizeInfoInCommmonTilingData(context, tilingData);
    InitTileInfoInCommonTiling(tilingData);
    tilingData.commonTiling.set_aivCoreNum(aivNum);
    tilingData.commonTiling.set_totalUbSize(ubSize);
    return;
}

static void GetChipDataFromPlatform(const gert::TilingContext *context, uint32_t &blockDim, uint64_t &ubSize,
                                    uint64_t &aicNum, uint64_t &aivNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aicNum = ascendcPlatform.GetCoreNumAic();
    aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    return;
}

static void SetHcclTilingData(const gert::TilingContext *context, AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    auto epGroup = attrs->GetAttrPointer<char>(ATTR_EP_GROUP_INDEX);
    auto tpGroup = attrs->GetAttrPointer<char>(ATTR_TP_GROUP_INDEX);
    std::string epGroupStr = string(epGroup);
    std::string tpGroupStr = string(tpGroup);
    std::string algConfigATAStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigAGStr = "AllGather=level0:ring";

    std::vector<char> groupNameVecATA(ARR_LENGTH, '\0');
    for (auto ite = epGroupStr.begin(); ite != epGroupStr.end(); ite++) {
        groupNameVecATA[ite - epGroupStr.begin()] = *ite;
    }

    std::vector<char> groupNameVecAG(ARR_LENGTH, '\0');
    for (auto ite = tpGroupStr.begin(); ite != tpGroupStr.end(); ite++) {
        groupNameVecAG[ite - tpGroupStr.begin()] = *ite;
    }

    std::vector<char> algConfigVecATA(ARR_LENGTH, '\0');
    for (auto ite = algConfigATAStr.begin(); ite != algConfigATAStr.end(); ite++) {
        algConfigVecATA[ite - algConfigATAStr.begin()] = *ite;
    }

    std::vector<char> algConfigVecAG(ARR_LENGTH, '\0');
    for (auto ite = algConfigAGStr.begin(); ite != algConfigAGStr.end(); ite++) {
        algConfigVecAG[ite - algConfigAGStr.begin()] = *ite;
    }

    tilingData.hcommCfgATA.set_groupName(groupNameVecATA.data());
    tilingData.hcommCfgATA.set_algConfig(algConfigVecATA.data());
    tilingData.hcommCfgATA.set_opType(OP_TYPE_ALL_TO_ALL); // numeric representation of AlltoAll
    tilingData.hcommCfgAG.set_groupName(groupNameVecAG.data());
    tilingData.hcommCfgAG.set_algConfig(algConfigVecAG.data());
    tilingData.hcommCfgAG.set_opType(OP_TYPE_ALL_GATHER); // numeric representation of AllGather
    return;
}

static ge::graphStatus SetBatchMatMulTilingData(gert::TilingContext *context, uint64_t aicNum,
                                                AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    mc2tiling::TilingArgs formulaicArgs;
    AlltoAllAllGatherBatchInfo BMMV3BatchInfo;
    AlltoAllAllGatherMatmulInfo MMV3ArgsInfo;

    const gert::StorageShape *biasInputShape =
        context->GetOptionalInputTensor(BIAS_INDEX) == nullptr ? nullptr : context->GetOptionalInputShape(BIAS_INDEX);
    ge::DataType inputDatatype = context->GetInputDesc(X_INDEX)->GetDataType();
    ge::DataType biasDatatype =
        (biasInputShape == nullptr) ? inputDatatype : context->GetOptionalInputDesc(BIAS_INDEX)->GetDataType();

    MMV3ArgsInfo.opName = "AlltoAllAllGatherBatchMatMul";
    MMV3ArgsInfo.isWeightTrans = tilingData.commonTiling.get_isWeightTrans();
    MMV3ArgsInfo.isBias = false;
    MMV3ArgsInfo.aType = inputDatatype;
    MMV3ArgsInfo.bType = inputDatatype;
    MMV3ArgsInfo.cType = inputDatatype;
    MMV3ArgsInfo.biasType = biasDatatype;

    BMMV3BatchInfo.biasWithBatch = false;

    formulaicArgs.mValue = tilingData.commonTiling.get_C();
    formulaicArgs.nValue = tilingData.commonTiling.get_MOverTp();
    formulaicArgs.kValue = tilingData.commonTiling.get_H();
    formulaicArgs.inputDtypeSize = GetDataSize(inputDatatype);
    formulaicArgs.outputDtypeSize = GetDataSize(inputDatatype);
    formulaicArgs.aicCoreNum = aicNum;
    formulaicArgs.aType = GetMatMulTilingDataType(inputDatatype);
    formulaicArgs.bType = GetMatMulTilingDataType(inputDatatype);
    formulaicArgs.cType = GetMatMulTilingDataType(inputDatatype);
    formulaicArgs.biasType = GetMatMulTilingDataType(biasDatatype);

    // 待修改，等BMM tiling提供接口和修改方案
    OP_TILING_CHECK(SetMatmulTilingAlltoAllAllGatherBatchMatMul(context, tilingData, BMMV3BatchInfo, MMV3ArgsInfo,
                                                                formulaicArgs) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Set Matmul tiling Failed!"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetUbTilingDataInCommonTiling(const gert::TilingContext *context, uint64_t ubSize,
                                          AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    ge::DataType inputDatatype = context->GetInputDesc(X_INDEX)->GetDataType();
    auto inputDatasize = GetDataSize(inputDatatype);
    auto activateType = tilingData.commonTiling.get_activateType();

    bool xCastFlag = ((inputDatatype == ge::DT_BF16) || (activateType == SILU));
    auto mOverTp = tilingData.commonTiling.get_MOverTp();
    // GPT2.6场景
    TransposeConfig config = {ubSize,
                              tilingData.commonTiling.localTileC.get_tileLen(),
                              tilingData.commonTiling.localTileE.get_tileLen(),
                              inputDatasize,
                              true,
                              false};
    CheckTransposeUBAndUpdateTileShard(config, tilingData);

    config = {ubSize,
              tilingData.commonTiling.localTileC.get_tailLen(),
              tilingData.commonTiling.localTileE.get_tailLen(),
              inputDatasize,
              true,
              true};
    CheckTransposeUBAndUpdateTileShard(config, tilingData);

    config = {ubSize,
              tilingData.commonTiling.domesticTileC.get_tileLen(),
              tilingData.commonTiling.domesticTileE.get_tileLen(),
              inputDatasize,
              false,
              false};
    CheckTransposeUBAndUpdateTileShard(config, tilingData);

    config = {ubSize,
              tilingData.commonTiling.domesticTileC.get_tailLen(),
              tilingData.commonTiling.domesticTileE.get_tailLen(),
              inputDatasize,
              false,
              true};
    CheckTransposeUBAndUpdateTileShard(config, tilingData);

    ActivationParams actParams = {ubSize, xCastFlag, activateType};
    TileShardParams params = {tilingData.commonTiling.localTileC.get_tileLen(),
                              tilingData.commonTiling.localTileE.get_tileLen(), mOverTp, true, false};
    CheckAddActivateUBAndUpdateTileShard(actParams, params, tilingData);
    params = {tilingData.commonTiling.localTileC.get_tailLen(), tilingData.commonTiling.localTileE.get_tailLen(),
              mOverTp, true, true};
    CheckAddActivateUBAndUpdateTileShard(actParams, params, tilingData);
    params = {tilingData.commonTiling.domesticTileC.get_tileLen(), tilingData.commonTiling.domesticTileC.get_tileLen(),
              mOverTp, false, false};
    CheckAddActivateUBAndUpdateTileShard(actParams, params, tilingData);
    params = {tilingData.commonTiling.domesticTileC.get_tailLen(), tilingData.commonTiling.domesticTileC.get_tailLen(),
              mOverTp, false, true};
    CheckAddActivateUBAndUpdateTileShard(actParams, params, tilingData);

    return;
}

static ge::graphStatus SetContextData(gert::TilingContext *context, uint32_t blockDim,
                                      AlltoAllAllGatherBatchMatMulTilingData &tilingData)
{
    uint64_t tilingKey = INIT_TILINGKEY;
    UpdateTilingKey(tilingKey, tilingData, tilingData.commonTiling.get_y2Flag(), tilingData.commonTiling.get_y3Flag());
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDim);
    OP_TILING_CHECK(SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Set Tiling Data Failed!"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllAllGatherBatchMatMulTilingFunc(gert::TilingContext *context)
{
    // tiling校验shape
    OP_TILING_CHECK(TilingCheckAlltoAllAllGatherBatchMatMul(context) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Tiling check shape Failed!"),
                    return ge::GRAPH_FAILED);
    AlltoAllAllGatherBatchMatMulTilingData tilingData;
    InitTilingDataBasicInfo(tilingData);

    uint32_t blockDim = 1U;
    uint64_t ubSize = 0U;
    uint64_t aicNum = 0U;
    uint64_t aivNum = 0U;
    GetChipDataFromPlatform(context, blockDim, ubSize, aicNum, aivNum);

    SetCommonTilingData(context, ubSize, aivNum, tilingData);
    SetHcclTilingData(context, tilingData);
    OP_TILING_CHECK(SetBatchMatMulTilingData(context, aicNum, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Set BatchMatmul tiling failed!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MC2SetWorkspaceShard(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Set workspace Failed!"),
                    return ge::GRAPH_FAILED);
    SetUbTilingDataInCommonTiling(context, ubSize, tilingData);
    OP_TILING_CHECK(SetContextData(context, blockDim, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Set context data failed!"),
                    return ge::GRAPH_FAILED);

    PrintCommonTilingVariables(tilingData);
    PrintSliceTileInfo(tilingData);
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllAllGatherBatchMatMulCompileInfo {};
ge::graphStatus TilingParseForAlltoAllAllGatherBatchMatMul(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AlltoAllAllGatherBatchMatMul)
    .Tiling(AlltoAllAllGatherBatchMatMulTilingFunc)
    .TilingParse<AlltoAllAllGatherBatchMatMulCompileInfo>(TilingParseForAlltoAllAllGatherBatchMatMul);
} // namespace optiling