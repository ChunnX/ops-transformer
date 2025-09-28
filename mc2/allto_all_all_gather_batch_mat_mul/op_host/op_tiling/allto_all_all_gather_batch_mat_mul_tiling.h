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
 * \file all_to_all_all_gather_batch_matmul_tiling.h
 * \brief
 */

#ifndef __ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_TILING_H__
#define __ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_TILING_H__


#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/mc2_tiling_struct.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "mc2_log.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_tiling.h"
#include "batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_base_tiling.h"

namespace optiling {
struct AlltoAllAllGatherBatchInfo {
    uint32_t batchA = 1;
    uint32_t batchA0 = 1;
    uint32_t batchA1 = 1;
    uint32_t batchA2 = 1;
    uint32_t batchA3 = 1;
    uint32_t batchB = 1;
    uint32_t batchB0 = 1;
    uint32_t batchB1 = 1;
    uint32_t batchB2 = 1;
    uint32_t batchB3 = 1;
    uint32_t batchC = 1;
    uint32_t batchC0 = 1;
    uint32_t batchC1 = 1;
    uint32_t batchC2 = 1;
    uint32_t batchC3 = 1;
    bool biasWithBatch = true;
}; // BmmTiling计算时需要用到的结构体

struct AlltoAllAllGatherMatmulInfo {
    const char *opName = nullptr;
    bool isWeightTrans = false;
    bool isBias = false;
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    uint64_t mValue = 0L;
    uint64_t kValue = 0L;
    uint64_t nValue = 0L;
}; // BmmTiling计算时需要用到的结构体


BEGIN_TILING_DATA_DEF(Mc2CommonTiling)
TILING_DATA_FIELD_DEF(uint32_t, epGroupSize); // ep
TILING_DATA_FIELD_DEF(uint32_t, tpGroupSize); // tp
TILING_DATA_FIELD_DEF(uint64_t, expert);      // E
TILING_DATA_FIELD_DEF(uint64_t, EOverEp);     // E/ep
TILING_DATA_FIELD_DEF(uint64_t, C);
TILING_DATA_FIELD_DEF(uint64_t, COverTp); // C/tp
TILING_DATA_FIELD_DEF(uint64_t, H);       // H
TILING_DATA_FIELD_DEF(uint64_t, HOverTp); // H/tp
TILING_DATA_FIELD_DEF(uint64_t, MOverTp); // M/tp
TILING_DATA_FIELD_DEF(uint32_t, aivCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, inputDatasize);
TILING_DATA_FIELD_DEF(uint32_t, biasDatasize);
TILING_DATA_FIELD_DEF(uint64_t, ubCapacityForTrans);
TILING_DATA_FIELD_DEF(uint64_t, ubCapacityForAddActivate);
TILING_DATA_FIELD_DEF(bool, isBias);
TILING_DATA_FIELD_DEF(bool, y2Flag);
TILING_DATA_FIELD_DEF(bool, y3Flag);
TILING_DATA_FIELD_DEF(bool, isWeightTrans);
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, localTileE);    // E 轴本地块切分信息
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, domesticTileE); // E 轴非本地块切分信息
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, localTileC);    // C 轴本地块切分信息
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, domesticTileC); // C 轴非本地块切分信息
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, localUbTranspose);
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, localTailUbTranspose);
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, domesticUbTranspose);
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, domesticTailUbTranspose);
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, localUbAdd);
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, localTailUbAdd);
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, domesticUbAdd);
TILING_DATA_FIELD_DEF_STRUCT(TileInfo, domesticTailUbAdd);
TILING_DATA_FIELD_DEF(uint32_t, activateType); // 激活措施：0为不激活，1为GELU，3为Relu，4为FastGELU
TILING_DATA_FIELD_DEF(uint32_t, xShardFlag);
TILING_DATA_FIELD_DEF(uint32_t, fastGeluBuffer);
TILING_DATA_FIELD_DEF(uint64_t, totalUbSize);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Mc2CommonTilingOp, Mc2CommonTiling)


BEGIN_TILING_DATA_DEF(AlltoAllAllGatherBatchMatMulTilingData)
TILING_DATA_FIELD_DEF(uint32_t, version);                              // 新流程时此处填2
TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);                             // 通信域数量，本算子有allToall和allGather两个
TILING_DATA_FIELD_DEF_STRUCT(Mc2ServerCfg, serverCfg);                 // server端通用参数
TILING_DATA_FIELD_DEF_STRUCT(Mc2HcommCfg, hcommCfgATA);                // 通信域1：allToall
TILING_DATA_FIELD_DEF_STRUCT(Mc2HcommCfg, hcommCfgAG);                 // 通信域2：allGather
TILING_DATA_FIELD_DEF_STRUCT(Mc2CommonTiling, commonTiling);           // kernel侧需要的通用tiling
TILING_DATA_FIELD_DEF_STRUCT(Mc2MatmulTilingData, localTiling);        // local块的matmul tiling数据
TILING_DATA_FIELD_DEF_STRUCT(Mc2MatmulTilingData, domesticTiling);     // 非local块的matmul tiling数据
TILING_DATA_FIELD_DEF_STRUCT(Mc2MatmulTilingData, localTailTiling);    // local尾块的matmul tiling数据
TILING_DATA_FIELD_DEF_STRUCT(Mc2MatmulTilingData, domesticTailTiling); // 非local尾块的matmul tiling数据
TILING_DATA_FIELD_DEF_STRUCT(Mc2MatmulTilingData,
                             domesticTailETiling); // sherd-0切E不切C时非local尾块的matmul tiling数据

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AlltoAllAllGatherBatchMatMul, AlltoAllAllGatherBatchMatMulTilingData);

class AlltoAllAllGatherBatchMatMulTiling : public batch_mat_mul_v3::BatchMatmulV3BaseTiling {
public:
    AlltoAllAllGatherBatchMatMulTiling(gert::TilingContext *context, BatchMatmulTilingData &bmmTilingData,
                                       AlltoAllAllGatherBatchInfo &BMMV3BatchInfo,
                                       AlltoAllAllGatherMatmulInfo &MMV3ArgsInfo)
        : BatchMatmulV3BaseTiling(context, bmmTilingData), BMMV3BatchInfo_(BMMV3BatchInfo), MMV3ArgsInfo_(MMV3ArgsInfo)
    {
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        args_.opName = MMV3ArgsInfo_.opName;
        args_.isBTrans = MMV3ArgsInfo_.isWeightTrans;
        args_.hasBias = MMV3ArgsInfo_.isBias;

        args_.aType = MMV3ArgsInfo_.aType;
        args_.bType = MMV3ArgsInfo_.bType;
        args_.cType = MMV3ArgsInfo_.cType;
        args_.biasType = MMV3ArgsInfo_.biasType;

        args_.aFormat = ge::FORMAT_ND;
        args_.bFormat = ge::FORMAT_ND;
        args_.outFormat = ge::FORMAT_ND;

        args_.mValue = MMV3ArgsInfo_.mValue;
        args_.kValue = MMV3ArgsInfo_.kValue;
        args_.nValue = MMV3ArgsInfo_.nValue;

        OP_LOGD(args_.opName, " args_.mValue %lu", args_.mValue);
        OP_LOGD(args_.opName, " args_.kValue %lu", args_.kValue);
        OP_LOGD(args_.opName, " args_.nValue %lu", args_.nValue);

        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchA %u", BMMV3BatchInfo_.batchA);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchA0 %u", BMMV3BatchInfo_.batchA0);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchA1 %u", BMMV3BatchInfo_.batchA1);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchA2 %u", BMMV3BatchInfo_.batchA2);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchA3 %u", BMMV3BatchInfo_.batchA3);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchB %u", BMMV3BatchInfo_.batchB);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchB0 %u", BMMV3BatchInfo_.batchB0);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchB1 %u", BMMV3BatchInfo_.batchB1);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchB2 %u", BMMV3BatchInfo_.batchB2);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchB3 %u", BMMV3BatchInfo_.batchB3);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchC %u", BMMV3BatchInfo_.batchC);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchC0 %u", BMMV3BatchInfo_.batchC0);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchC1 %u", BMMV3BatchInfo_.batchC1);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchC2 %u", BMMV3BatchInfo_.batchC2);
        OP_LOGD("AlltoAllAllGatherBatchMatMulTiling", " batchC3 %u", BMMV3BatchInfo_.batchC3);

        batchInfo_.batchA = BMMV3BatchInfo_.batchA;
        batchInfo_.batchA0 = BMMV3BatchInfo_.batchA0;
        batchInfo_.batchA1 = BMMV3BatchInfo_.batchA1;
        batchInfo_.batchA2 = BMMV3BatchInfo_.batchA2;
        batchInfo_.batchA3 = BMMV3BatchInfo_.batchA3;
        batchInfo_.batchB = BMMV3BatchInfo_.batchB;
        batchInfo_.batchB0 = BMMV3BatchInfo_.batchB0;
        batchInfo_.batchB1 = BMMV3BatchInfo_.batchB1;
        batchInfo_.batchB2 = BMMV3BatchInfo_.batchB2;
        batchInfo_.batchB3 = BMMV3BatchInfo_.batchB3;
        batchInfo_.batchC = BMMV3BatchInfo_.batchC;
        batchInfo_.batchC0 = BMMV3BatchInfo_.batchC0;
        batchInfo_.batchC1 = BMMV3BatchInfo_.batchC1;
        batchInfo_.batchC2 = BMMV3BatchInfo_.batchC2;
        batchInfo_.batchC3 = BMMV3BatchInfo_.batchC3;
        batchInfo_.biasWithBatch = BMMV3BatchInfo_.biasWithBatch;
        return ge::GRAPH_SUCCESS;
    }

private:
    AlltoAllAllGatherBatchInfo BMMV3BatchInfo_;
    AlltoAllAllGatherMatmulInfo MMV3ArgsInfo_;
};
} // namespace optiling

#endif //__ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_TILING_H__
