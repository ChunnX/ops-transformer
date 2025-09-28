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
* \file mla_prolog_tiling.cpp
* \brief
*/

#include <numeric>
#include <functional>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "mla_prolog_tiling_check.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

const std::unordered_map<ge::DataType, uint32_t> DTYPE_TO_SIZE {
    {ge::DT_BF16, 2},
    {ge::DT_FLOAT16, 2},
    {ge::DT_INT8, 1},
    {ge::DT_INT32, 4}};

const std::unordered_map<ge::DataType, matmul_tiling::DataType> GE_TO_MM_DTYPE {
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT}};

template <typename T>
inline auto CeilDiv(T a, T b) -> T
{
    if (b == 0) {
        return b;
    }
    return (a + b - 1) / b;
}

template <typename T> 
inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

ge::graphStatus MlaPrologTiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

QUANT_MODE MlaPrologTiling::GetQuantizationMode() const
{
    if (context_->tokenX.desc->GetDataType() == ge::DT_INT8) {
        if (context_->kvCache.desc->GetDataType() == ge::DT_INT8) {
            return QUANT_MODE::FULL_QUANT_KV_QUANT;
        } else {
            return QUANT_MODE::FULL_QUANT_KV_NO_QUANT;
        }
    }
    if (context_->weightUqQr.desc->GetDataType() == ge::DT_INT8) {
        if (context_->kvCache.desc->GetDataType() == ge::DT_INT8) {
            return QUANT_MODE::PARTIAL_QUANT_KV_QUANT;
        } else {
            return QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT;
        }
    }
    return QUANT_MODE::NO_QUANT;
}

ge::graphStatus MlaPrologTiling::SetShapeInfo()
{
    if (context_->tokenX.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_3) {
        baseShapeInfo_.bSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
        baseShapeInfo_.s1Size = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
        baseShapeInfo_.heSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_2);
        baseShapeInfo_.tSize = baseShapeInfo_.bSize * baseShapeInfo_.s1Size;
    } else {
        baseShapeInfo_.tSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
        baseShapeInfo_.heSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
    }
    if (context_->weightDq.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_2) {
        baseShapeInfo_.hcqSize = context_->weightDq.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
    } else {
        uint32_t weightDqAxisSize_ = 32U / ge::GetSizeByDataType(context_->weightDq.desc->GetDataType());
        // weightDq: [He, Hcq] -> [Hcq/16, He/16, 16, 16] || [Hcq/32, He/16, 16, 32]
        baseShapeInfo_.hcqSize =
            weightDqAxisSize_ * context_->weightDq.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
    }
    baseShapeInfo_.nSize = context_->weightUk.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
    baseShapeInfo_.drSize =
        context_->ropeCos.shape->GetStorageShape().GetDim(context_->ropeCos.shape->GetStorageShape().GetDimNum() - 1);
    baseShapeInfo_.dSize = context_->weightUk.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
    baseShapeInfo_.headSizeQc = baseShapeInfo_.dSize * baseShapeInfo_.nSize;
    baseShapeInfo_.headSizeQr = baseShapeInfo_.drSize * baseShapeInfo_.nSize;
    baseShapeInfo_.headSizeUqQr = baseShapeInfo_.headSizeQc + baseShapeInfo_.headSizeQr;
    baseShapeInfo_.blockNum = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
    baseShapeInfo_.blockSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
    baseShapeInfo_.nkvSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_2);
    baseShapeInfo_.hckvSize =
        context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_3);
    baseShapeInfo_.s2Size = baseShapeInfo_.nkvSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::SetScenarioInfo()
{
    scenarioInfo_.isV1Flag_ = (strcmp(context_->opType, V1_OP_NAME) == 0);
    scenarioInfo_.batchSeqFusedFlag_ = context_->tokenX.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_2;
    scenarioInfo_.quantMode_ = GetQuantizationMode();
    if (std::strcmp(context_->cacheMode, CACHE_MODE_PA_BSND) == 0) {
        scenarioInfo_.cacheMode_ = CACHE_MODE::PA_BSND;
    } else {
        scenarioInfo_.cacheMode_ = CACHE_MODE::PA_NZ;
    }
    if ((scenarioInfo_.batchSeqFusedFlag_ && baseShapeInfo_.tSize == 0U) ||
        (!scenarioInfo_.batchSeqFusedFlag_ && (baseShapeInfo_.bSize * baseShapeInfo_.s1Size == 0U))) {
        scenarioInfo_.emptyTensorMode_ = EMPTY_TENSOR_MODE::EMPTY_QUERY;
    } else if (baseShapeInfo_.blockNum == 0U) {
        scenarioInfo_.emptyTensorMode_ = EMPTY_TENSOR_MODE::EMPTY_CACHE;
    } else {
        scenarioInfo_.emptyTensorMode_ = EMPTY_TENSOR_MODE::NON_EMPTY;
    }
    return ge::GRAPH_SUCCESS;
}

bool MlaPrologTiling::GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType)
{
    auto mmdt = GE_TO_MM_DTYPE.find(getype);
    if (mmdt != GE_TO_MM_DTYPE.end()) {
        *mmType = mmdt->second;
        return true;
    }
    return false;
}

uint32_t MlaPrologTiling::CalcSingleCoreN(uint32_t n, uint32_t coreNum, uint32_t alignNum) const
{
    return CeilDiv(n, alignNum * coreNum) * alignNum;
}

// mm1.m = stepBatchSize            // 32
// mm1.n = singlecoreHeadSizeCq     // 64
// mm1.k = headSizeX                // 7168
// mm1.baseM = stepBatchSize        // 32
// mm1.baseN = singlecoreHeadSizeCq // 64
// mm1.baseK = 256
ge::graphStatus MlaPrologTiling::FillMatmul1Tiling()
{
    uint32_t M = stepBatchSize_;
    auto dataType = context_->weightDq.desc->GetDataType();
    singlecoreHeadSizeCq_ =
        CalcSingleCoreN(baseShapeInfo_.hcqSize, aicNum_, BLOCK_SIZE / DTYPE_TO_SIZE.at(dataType));
    mm1BlockNum_ = CeilDiv(baseShapeInfo_.hcqSize, singlecoreHeadSizeCq_);

    matmul_tiling::DataType bmm1DataType;

    if (!GetMatmulType(mmDateType_, &bmm1DataType)) {
        OP_LOGE(context_->opName, "get matmul type error");
        return false;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);

    bmm1.SetShape(M, singlecoreHeadSizeCq_, baseShapeInfo_.heSize);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1DataType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, bmm1DataType, false);

    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1DataType);

    bmm1.SetOrgShape(M, singlecoreHeadSizeCq_, baseShapeInfo_.heSize, baseShapeInfo_.heSize);
    bmm1.SetBias(false);

    if (bmm1.GetTiling(*bmm1TilingData_) == -1) {
        OP_LOGE(context_->opName, "bmm1 get tiling fail");
        return false;
    }

    return ge::GRAPH_SUCCESS;
}

// singlecoreHeadSizeCkvKr =  HeadSizeCkvDr / mm2CoreNum // 576 / 9 == 64
// mm2.m = stepBatchSize
// mm2.n = singlecoreHeadSizeCkvKr
// mm2.k = headSizeX // size of He
// mm2.baseN = n
// mm2.baseK = 256
ge::graphStatus MlaPrologTiling::FillMatmul2Tiling()
{
    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t M = stepBatchSize_;
    // 9是经验值
    if (aicNum_ >= 9U) {
        uint32_t baseN = 64U;
        mm2BlockNum_ = (baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) / baseN;
        singlecoreHeadSizeCkvKr_ = baseN;
    } else {
        auto dataType = context_->weightDkvKr.desc->GetDataType();
        singlecoreHeadSizeCkvKr_ = CalcSingleCoreN(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize, aicNum_,
                                                   BLOCK_SIZE / DTYPE_TO_SIZE.at(dataType));
        mm2BlockNum_ = CeilDiv(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize, singlecoreHeadSizeCkvKr_);
    }

    matmul_tiling::DataType bmm2DataType;
    
    if (!GetMatmulType(mmDateType_, &bmm2DataType)) {
        OP_LOGE(context_->opName, "get matmul type error");
        return false;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);

    bmm2.SetShape(M, singlecoreHeadSizeCkvKr_, baseShapeInfo_.heSize);
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2DataType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, bmm2DataType, false);

    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2DataType);

    bmm2.SetOrgShape(M, singlecoreHeadSizeCkvKr_, baseShapeInfo_.heSize, baseShapeInfo_.heSize);
    bmm2.SetBias(false);

    if (bmm2.GetTiling(*bmm2TilingData_) == -1) {
        OP_LOGE(context_->opName, "bmm2 get tiling fail");
        return false;
    }
    return ge::GRAPH_SUCCESS;
}

// singlecoreHeadSizeQcQr = headNum * (dimHeadSizeQc + dimHeadRope) / mm3CoreNum  = 32 * (128 + 64) / 24
// mm3.m = stepBatchSize
// mm3.n = singlecoreHeadSizeQcQr   // 256
// mm3.k = headSizeCq // size of Hcq   1536
// mm3.baseN = 64  //
// mm3.baseK = 256 //
ge::graphStatus MlaPrologTiling::FillMatmul3Tiling()
{
    uint32_t M = stepBatchSize_;
    auto dataType = context_->weightUqQr.desc->GetDataType();
    auto oriM = baseShapeInfo_.nSize * (baseShapeInfo_.dSize + baseShapeInfo_.drSize);
    if (enableGroupComputeOpt_) {
        // 算力分组场景下G=8，dimHeadSizeQc跨8核切，dimHeadSizeQr跨4核切；matmulQc和matmulQr的singleN都取128
        singlecoreHeadSizeQcQr_ =
            CalcSingleCoreN(baseShapeInfo_.nSize * baseShapeInfo_.dSize,
                GROUP_COMPUTE_CUBE_NUM_PER_GROUP, baseShapeInfo_.dSize);
    } else if (enableDequantOpt_) {
        // dequant流水掩盖场景，dimHeadSizeQc + dimHeadRope不跨核
        singlecoreHeadSizeQcQr_ = CalcSingleCoreN(oriM, aicNum_, baseShapeInfo_.dSize + baseShapeInfo_.drSize);
    } else {
        // headnum * (dimHeadSizeQc + dimHeadRope) 合轴切
        singlecoreHeadSizeQcQr_ = CalcSingleCoreN(oriM, aicNum_, BLOCK_SIZE / DTYPE_TO_SIZE.at(dataType));
    }
    mm3BlockNum_ = CeilDiv(oriM, singlecoreHeadSizeQcQr_);

    matmul_tiling::DataType bmm3DataType;
    auto weightUqQrType = context_->weightUqQr.desc->GetDataType();
    if (!GetMatmulType(weightUqQrType, &bmm3DataType)) {
        OP_LOGE(context_->opName, "get matmul type error");
        return false;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm3(ascendcPlatform);

    bmm3.SetShape(M, singlecoreHeadSizeQcQr_, baseShapeInfo_.heSize);
    bmm3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm3DataType, false);
    bmm3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, bmm3DataType, false);

    if (bmm3DataType == matmul_tiling::DataType::DT_BF16) {
        bmm3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm3DataType);
    } else if (bmm3DataType == matmul_tiling::DataType::DT_INT8) {
        bmm3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    } else {
        OP_LOGE(context_->opName, "bmm3DataType only support (DT_BF16, DT_INT8)");
        return ge::GRAPH_FAILED;
    }

    bmm3.SetOrgShape(M, singlecoreHeadSizeQcQr_, baseShapeInfo_.hcqSize, baseShapeInfo_.hcqSize);
    bmm3.SetBias(false);

    if (bmm3.GetTiling(*bmm3TilingData_) == -1) {
        OP_LOGE(context_->opName, "bmm3 get tiling fail");
        return false;
    }

    return ge::GRAPH_SUCCESS;
}

// mm4.m = stepBatchSize
// mm4.n = headSizeCkv  // 512
// mm4.k = dimHeadSizeQc // size of Qc  128
// mm4.baseN = 128  //
// mm4.baseK = 128 //
// mm4.Kstride = dimHeadSizeQc + dimHeadRope
ge::graphStatus MlaPrologTiling::FillMatmul4Tiling()
{
    uint32_t M = stepBatchSize_;
    singlecoreNumHeadSize_ = CeilDiv(baseShapeInfo_.nSize, aicNum_);
    mm4BlockNum_ = CeilDiv(baseShapeInfo_.nSize, singlecoreNumHeadSize_);

    matmul_tiling::DataType bmm4DataType;
    auto weightUkType = context_->weightUk.desc->GetDataType();
    if (!GetMatmulType(weightUkType, &bmm4DataType)) {
        OP_LOGE(context_->opName, "get matmul type error");
        return false;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm4(ascendcPlatform);

    bmm4.SetShape(M, baseShapeInfo_.hckvSize, baseShapeInfo_.dSize);
    bmm4.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm4DataType, false);
    bmm4.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm4DataType, false);

    bmm4.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm4DataType);

    bmm4.SetOrgShape(M, baseShapeInfo_.hckvSize, baseShapeInfo_.dSize, baseShapeInfo_.dSize);
    bmm4.SetBias(false);

    if (bmm4.GetTiling(*bmm4TilingData_) == -1) {
        OP_LOGE(context_->opName, "bmm4 get tiling fail");
        return false;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::ProcessBaseInputs()
{
    reciprocalCq_ = 1.0f / (baseShapeInfo_.hcqSize);
    epsilonCq_ = *context_->rmsNormEspilonCq;
    reciprocalCkv_ = 1.0f / (baseShapeInfo_.hckvSize);
    epsilonCkv_ = *context_->rmsNormEspilonCkv;

    stepBatchSize_ = std::min(128U, baseShapeInfo_.tSize);
    if (baseShapeInfo_.dSize == HIGH_THROUGHPUT__D_SIZE) {
        stepNumHeadDequant_ = std::min(64U, baseShapeInfo_.nSize);
    } else {
        stepNumHeadDequant_ = std::min(16U, baseShapeInfo_.nSize);
    }
    vectorBlockNum_ = std::min(stepBatchSize_, aivNum_);

    // 算力分组开关，仅当半量化场景，BS=1，G=8，可用核数大于等于16时进入分支
    if ((scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT ||
         scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_QUANT) &&
        baseShapeInfo_.tSize == GROUP_COMPUTE_T_SIZE &&
        baseShapeInfo_.nkvSize == GROUP_COMPUTE_NKV_SIZE &&
        aivNum_ >= GROUP_COMPUTE_MIN_AIV_NUM &&
        aicNum_ >= GROUP_COMPUTE_MIN_AIC_NUM) {
        enableGroupComputeOpt_ = true;
        aivNum_ = 32U;
        aicNum_ = 16U;
    } else if (context_->weightUqQr.desc->GetDataType() == ge::DT_INT8 &&
               baseShapeInfo_.nSize >= GROUP_COMPUTE_N_SIZE) {
        // N大于等于8时通过切N处理MM3，MM4之后的操作例如Rope，DynamicQuant等会有性能收益
        enableDequantOpt_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::FillTiling()
{
    baseParams_->set_batchSize(baseShapeInfo_.bSize);
    baseParams_->set_stepBatchSize(stepBatchSize_);
    baseParams_->set_stepNumHeadDequant(stepNumHeadDequant_);
    baseParams_->set_tokenSize(baseShapeInfo_.tSize);
    baseParams_->set_seq1Size(baseShapeInfo_.s1Size);
    baseParams_->set_seq2Size(baseShapeInfo_.s2Size);
    baseParams_->set_headSizeX(baseShapeInfo_.heSize);
    baseParams_->set_headSizeCq(baseShapeInfo_.hcqSize);
    baseParams_->set_headSizeCkv(baseShapeInfo_.hckvSize);
    baseParams_->set_headSizeQc(baseShapeInfo_.headSizeQc);
    baseParams_->set_headSizeQr(baseShapeInfo_.headSizeQr);
    baseParams_->set_headSizeKr(baseShapeInfo_.drSize);
    baseParams_->set_numHeadSize(baseShapeInfo_.nSize);
    baseParams_->set_numHeadKvSize(baseShapeInfo_.nkvSize);
    baseParams_->set_dimHeadSizeQc(baseShapeInfo_.dSize);
    baseParams_->set_dimHeadRope(baseShapeInfo_.drSize);
    baseParams_->set_blockNum(baseShapeInfo_.blockNum);
    baseParams_->set_blockSize(baseShapeInfo_.blockSize);
    baseParams_->set_mm1BlockNum(mm1BlockNum_);
    baseParams_->set_mm2BlockNum(mm2BlockNum_);
    baseParams_->set_mm3BlockNum(mm3BlockNum_);
    baseParams_->set_mm4BlockNum(mm4BlockNum_);
    baseParams_->set_mm1SingleCoreN(singlecoreHeadSizeCq_);
    baseParams_->set_mm2SingleCoreN(singlecoreHeadSizeCkvKr_);
    baseParams_->set_mm3SingleCoreN(singlecoreHeadSizeQcQr_);
    baseParams_->set_mm4SingleCoreBatch(singlecoreNumHeadSize_);
    baseParams_->set_vectorBlockNum(vectorBlockNum_);
    baseParams_->set_reciprocalCq(reciprocalCq_);
    baseParams_->set_epsilonCq(epsilonCq_);
    baseParams_->set_reciprocalCkv(reciprocalCkv_);
    baseParams_->set_epsilonCkv(epsilonCkv_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::CalcWorkSpace()
{
    workspaceSize_ = libapiSize_;
    if (scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_NO_QUANT ||
        scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT) {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.hcqSize) *
                          static_cast<size_t>(NUM_BYTES_INT32);
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.hcqSize) *
                          static_cast<size_t>(NUM_BYTES_BF16);
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
                          static_cast<size_t>(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) *
                          static_cast<size_t>(NUM_BYTES_INT32);
        if (scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT) {
            // 全量化场景mmQnRes输出到workspace, B, S1, N, Hckv, BF16
            workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.nSize) * 
                              static_cast<size_t>(baseShapeInfo_.hckvSize) * static_cast<size_t>(NUM_BYTES_BF16);
        }
    } else {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.hcqSize) *
                          static_cast<size_t>(NUM_BYTES_BF16) * static_cast<size_t>(2);  // 2: double
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
                          static_cast<size_t>(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) *
                          static_cast<size_t>(NUM_BYTES_BF16);
    }
    workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
        static_cast<size_t>(baseShapeInfo_.headSizeQc + baseShapeInfo_.headSizeQr) *
        static_cast<size_t>(NUM_BYTES_INT32);
    workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.nSize) * 
        static_cast<size_t>(baseShapeInfo_.dSize) * static_cast<size_t>(NUM_BYTES_BF16);

    if (enableGroupComputeOpt_ || enableDequantOpt_) {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(BLOCK_SIZE);
    }
    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }
    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::GenTilingKey() const
{
    uint32_t typeValue = 0;
    uint32_t quantType = 0;
    if (scenarioInfo_.quantMode_ == QUANT_MODE::NO_QUANT) {
        typeValue = 1U;
    } else {
        typeValue = 2U;
        // kvCache量化场景，对应tiling key为1(半量化:0 + kv量化:1)或3(全量化:2 + kv量化:1)
        // 全量化场景，对应tiling key为2+0(全量化:2)或2+1（全量化:2+ kv量化:1）
        // 非量化和半量化场景，对应tiling key为0
        quantType = static_cast<uint32_t>(scenarioInfo_.quantMode_);
    }

    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        context_->tilingKey = MLA_PROLOG_TILINGKEY_BASE_OFFSET + static_cast<uint64_t>(scenarioInfo_.emptyTensorMode_) *
                                                                     MLA_PROLOG_EMPTY_TENSOR_MODE_OFFSET;
    } else {
        context_->tilingKey = MLA_PROLOG_TILINGKEY_BASE_OFFSET +
                              static_cast<uint64_t>(scenarioInfo_.cacheMode_) +
                              typeValue * MLA_PROLOG_TYPE_OFFSET +
                              quantType * MLA_PROLOG_QUANT_TYPE_OFFSET +
                              static_cast<uint64_t>(enableDequantOpt_) * MLA_PROLOG_ENABLE_DEQUANT_OPT_OFFSET +
                              static_cast<uint64_t>(enableGroupComputeOpt_) * MLA_PROLOG_ENABLE_GROUP_COMPUTE_OPT_OFFSET +
                              static_cast<uint64_t>(scenarioInfo_.emptyTensorMode_) * MLA_PROLOG_EMPTY_TENSOR_MODE_OFFSET;
    }
    OP_LOGI(context_->opName, "MlaProlog tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::RunBigKernelTiling(MlaPrologContext &context, MlaPrologTilingData &tilingData)
{
    this->context_ = &context;
    this->bmm1TilingData_ = &tilingData.bmm1TilingData;
    this->bmm2TilingData_ = &tilingData.bmm2TilingData;
    this->bmm3TilingData_ = &tilingData.bmm3TilingData;
    this->bmm4TilingData_ = &tilingData.bmm4TilingData;
    this->baseParams_ = &tilingData.baseParams;
    MlaPrologTilingCheck tilingCheck_ {*context_, baseShapeInfo_, scenarioInfo_};

    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs {
        std::bind(&MlaPrologTiling::GetNpuInfo, this),
        std::bind(&MlaPrologTilingCheck::CheckSingleRequiredParam, &tilingCheck_),
        std::bind(&MlaPrologTilingCheck::CheckCacheMode, &tilingCheck_),
        std::bind(&MlaPrologTiling::SetShapeInfo, this),
        std::bind(&MlaPrologTiling::SetScenarioInfo, this),
        std::bind(&MlaPrologTilingCheck::CheckDims, &tilingCheck_),
        std::bind(&MlaPrologTilingCheck::CheckParamByScenario, &tilingCheck_),
        std::bind(&MlaPrologTiling::ProcessBaseInputs, this),
    };
    for (const auto &func: requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        FillTiling();
        if (context_->workSpaces) {
            context_->workSpaces[0] = libapiSize_;
        }
        GenTilingKey();
        context_->blockDim = 1U;
        return ge::GRAPH_SUCCESS;
    }

    std::vector<StatusFunction> optionalTilingFuncs {
        std::bind(&MlaPrologTiling::FillMatmul1Tiling, this),
        std::bind(&MlaPrologTiling::FillMatmul2Tiling, this),
        std::bind(&MlaPrologTiling::FillMatmul3Tiling, this),
        std::bind(&MlaPrologTiling::FillMatmul4Tiling, this),
        std::bind(&MlaPrologTiling::FillTiling, this),
        std::bind(&MlaPrologTiling::CalcWorkSpace, this),
        std::bind(&MlaPrologTiling::GenTilingKey, this)
    };
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::ConvertContext(gert::TilingContext &context, MlaPrologContext &mlaPrologContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE(V1_OP_NAME, "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    mlaPrologContext.opName = context.GetNodeName();
    mlaPrologContext.opType = context.GetNodeType();
    mlaPrologContext.platformInfo = context.GetPlatformInfo();

    ConvertRequiredParams(context, mlaPrologContext);
    ConvertOptionalParams(context, mlaPrologContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    mlaPrologContext.rmsNormEspilonCq = attrs->GetAttrPointer<float>(RMS_NORM_EPSILON_CQ_ATTR_INDEX);
    mlaPrologContext.rmsNormEspilonCkv = attrs->GetAttrPointer<float>(RMS_NORM_EPSILON_CKV_ATTR_INDEX);
    mlaPrologContext.cacheMode = attrs->GetStr(CACHE_MODE_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    mlaPrologContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

void MlaPrologTiling::ConvertRequiredParams(gert::TilingContext &context, MlaPrologContext &mlaPrologContext)
{
    mlaPrologContext.tokenX.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    mlaPrologContext.tokenX.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    mlaPrologContext.weightDq.desc = context.GetRequiredInputDesc(WEIGHT_DQ_INPUT_INDEX);
    mlaPrologContext.weightDq.shape = context.GetRequiredInputShape(WEIGHT_DQ_INPUT_INDEX);
    mlaPrologContext.weightUqQr.desc = context.GetRequiredInputDesc(WEIGHT_UQ_QR_INPUT_INDEX);
    mlaPrologContext.weightUqQr.shape = context.GetRequiredInputShape(WEIGHT_UQ_QR_INPUT_INDEX);
    mlaPrologContext.weightUk.desc = context.GetRequiredInputDesc(WEIGHT_UK_INPUT_INDEX);
    mlaPrologContext.weightUk.shape = context.GetRequiredInputShape(WEIGHT_UK_INPUT_INDEX);
    mlaPrologContext.weightDkvKr.desc = context.GetRequiredInputDesc(WEIGHT_DKV_KR_INPUT_INDEX);
    mlaPrologContext.weightDkvKr.shape = context.GetRequiredInputShape(WEIGHT_DKV_KR_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCq.desc = context.GetRequiredInputDesc(RMSNORM_GAMMA_CQ_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCq.shape = context.GetRequiredInputShape(RMSNORM_GAMMA_CQ_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCkv.desc = context.GetRequiredInputDesc(RMS_NORM_GAMMA_CKV_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCkv.shape = context.GetRequiredInputShape(RMS_NORM_GAMMA_CKV_INPUT_INDEX);
    mlaPrologContext.ropeSin.desc = context.GetRequiredInputDesc(ROPE_SIN_INPUT_INDEX);
    mlaPrologContext.ropeSin.shape = context.GetRequiredInputShape(ROPE_SIN_INPUT_INDEX);
    mlaPrologContext.ropeCos.desc = context.GetRequiredInputDesc(ROPE_COS_INPUT_INDEX);
    mlaPrologContext.ropeCos.shape = context.GetRequiredInputShape(ROPE_COS_INPUT_INDEX);
    mlaPrologContext.cacheIndex.desc = context.GetRequiredInputDesc(CACHE_INDEX_INPUT_INDEX);
    mlaPrologContext.cacheIndex.shape = context.GetRequiredInputShape(CACHE_INDEX_INPUT_INDEX);
    mlaPrologContext.kvCache.desc = context.GetRequiredInputDesc(KV_CACHE_INPUT_INDEX);
    mlaPrologContext.kvCache.shape = context.GetRequiredInputShape(KV_CACHE_INPUT_INDEX);
    mlaPrologContext.krCache.desc = context.GetRequiredInputDesc(KR_CACHE_INPUT_INDEX);
    mlaPrologContext.krCache.shape = context.GetRequiredInputShape(KR_CACHE_INPUT_INDEX);

    mlaPrologContext.query.desc = context.GetOutputDesc(QUERY_OUTPUT_INDEX);
    mlaPrologContext.query.shape = context.GetOutputShape(QUERY_OUTPUT_INDEX);
    mlaPrologContext.queryRope.desc = context.GetOutputDesc(QUERY_ROPE_OUTPUT_INDEX);
    mlaPrologContext.queryRope.shape = context.GetOutputShape(QUERY_ROPE_OUTPUT_INDEX);
    mlaPrologContext.kvCacheOut.desc = context.GetOutputDesc(KV_CACHE_OUT_OUTPUT_INDEX);
    mlaPrologContext.kvCacheOut.shape = context.GetOutputShape(KV_CACHE_OUT_OUTPUT_INDEX);
    mlaPrologContext.krCacheOut.desc = context.GetOutputDesc(KR_CACHE_OUT_OUTPUT_INDEX);
    mlaPrologContext.krCacheOut.shape = context.GetOutputShape(KR_CACHE_OUT_OUTPUT_INDEX);
}

void MlaPrologTiling::ConvertOptionalParams(gert::TilingContext &context, MlaPrologContext &mlaPrologContext)
{
    mlaPrologContext.dequantScaleX.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_X_INDEX);
    mlaPrologContext.dequantScaleX.shape = context.GetOptionalInputShape(DEQUANT_SCALE_X_INDEX);
    mlaPrologContext.dequantScaleWDq.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_W_DQ_INDEX);
    mlaPrologContext.dequantScaleWDq.shape = context.GetOptionalInputShape(DEQUANT_SCALE_W_DQ_INDEX);
    mlaPrologContext.dequantScaleWUqQr.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_W_UQ_QR_INDEX);
    mlaPrologContext.dequantScaleWUqQr.shape = context.GetOptionalInputShape(DEQUANT_SCALE_W_UQ_QR_INDEX);
    mlaPrologContext.dequantScaleWDkvKr.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_W_DKV_KR_INDEX);
    mlaPrologContext.dequantScaleWDkvKr.shape = context.GetOptionalInputShape(DEQUANT_SCALE_W_DKV_KR_INDEX);
    mlaPrologContext.quantScaleCkv.desc = context.GetOptionalInputDesc(QUANT_SCALE_CKV_INDEX);
    mlaPrologContext.quantScaleCkv.shape = context.GetOptionalInputShape(QUANT_SCALE_CKV_INDEX);
    mlaPrologContext.quantScaleCkr.desc = context.GetOptionalInputDesc(QUANT_SCALE_CKR_INDEX);
    mlaPrologContext.quantScaleCkr.shape = context.GetOptionalInputShape(QUANT_SCALE_CKR_INDEX);
    mlaPrologContext.smoothScalesCq.desc = context.GetOptionalInputDesc(SMOOTH_SCALES_CQ_INDEX);
    mlaPrologContext.smoothScalesCq.shape = context.GetOptionalInputShape(SMOOTH_SCALES_CQ_INDEX);

    // only v1 does not support dequantScaleQNope
    if (strcmp(mlaPrologContext.opType, V1_OP_NAME) == 0) {
        mlaPrologContext.dequantScaleQNope.desc = nullptr;
        mlaPrologContext.dequantScaleQNope.shape = nullptr;
    } else {
        mlaPrologContext.dequantScaleQNope.desc = context.GetOutputDesc(DEQUANT_SCALE_Q_NOPE_OUTPUT_INDEX);
        mlaPrologContext.dequantScaleQNope.shape = context.GetOutputShape(DEQUANT_SCALE_Q_NOPE_OUTPUT_INDEX);
    }
}

ge::graphStatus MlaPrologTiling::MlaPrologSetTilingData(gert::TilingContext &context, MlaPrologTilingData &tilingData)
{
    OP_CHECK_IF(context.GetRawTilingData() == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "RawTilingData got from ge context is nullptr."),
               return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context.GetRawTilingData()->GetData(), context.GetRawTilingData()->GetCapacity());
    context.GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

MLA_EXTERN_C ge::graphStatus TilingMlaProlog(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR(V1_OP_NAME, "Context is nullptr."),
               return ge::GRAPH_FAILED);

    MlaPrologContext mlaPrologContext{};
    if (MlaPrologTiling::ConvertContext(*context, mlaPrologContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to MlaProlog context");
        return ge::GRAPH_FAILED;
    }

    MlaPrologTilingData tilingData;
    MlaPrologTiling mlaPrologTiling;
    if (mlaPrologTiling.RunBigKernelTiling(mlaPrologContext, tilingData) == ge::SUCCESS) {
        context->SetTilingKey(mlaPrologContext.tilingKey);
        context->SetBlockDim(mlaPrologContext.blockDim);
        mlaPrologTiling.MlaPrologSetTilingData(*context, tilingData);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}
} // namespace optiling
