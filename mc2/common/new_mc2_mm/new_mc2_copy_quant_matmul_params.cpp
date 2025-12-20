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
 * \file mc2_copy_quant_matmul_params.cpp
 * \brief
 */


#include "new_mc2_copy_quant_matmul_params.h"
namespace optiling {
void NewSetMatmulTilingParams(::TCubeTiling& matmulTilingParams, TCubeTiling& matmulTilingData)
{
    matmulTilingData.set_usedCoreNum(matmulTilingParams.usedCoreNum);
    matmulTilingData.set_M(matmulTilingParams.M);
    matmulTilingData.set_N(matmulTilingParams.N);
    matmulTilingData.set_Ka(matmulTilingParams.Ka);
    matmulTilingData.set_Kb(matmulTilingParams.Kb);
    matmulTilingData.set_singleCoreM(matmulTilingParams.singleCoreM);
    matmulTilingData.set_singleCoreN(matmulTilingParams.singleCoreN);
    matmulTilingData.set_singleCoreK(matmulTilingParams.singleCoreK);
    matmulTilingData.set_baseM(matmulTilingParams.baseM);
    matmulTilingData.set_baseN(matmulTilingParams.baseN);
    matmulTilingData.set_baseK(matmulTilingParams.baseK);
    matmulTilingData.set_depthA1(matmulTilingParams.depthA1);
    matmulTilingData.set_depthB1(matmulTilingParams.depthB1);
    matmulTilingData.set_stepM(matmulTilingParams.stepM);
    matmulTilingData.set_stepN(matmulTilingParams.stepN);
    matmulTilingData.set_isBias(matmulTilingParams.isBias);
    matmulTilingData.set_transLength(matmulTilingParams.transLength);
    matmulTilingData.set_iterateOrder(matmulTilingParams.iterateOrder);
    matmulTilingData.set_stepKa(matmulTilingParams.stepKa);
    matmulTilingData.set_stepKb(matmulTilingParams.stepKb);
    matmulTilingData.set_dbL0A(matmulTilingParams.dbL0A);
    matmulTilingData.set_dbL0B(matmulTilingParams.dbL0B);
    matmulTilingData.set_dbL0C(matmulTilingParams.dbL0C);
    matmulTilingData.set_singleBatchM(matmulTilingParams.singleBatchM);
    matmulTilingData.set_singleBatchN(matmulTilingParams.singleBatchN);
    matmulTilingData.set_batchM(matmulTilingParams.batchM);
    matmulTilingData.set_batchN(matmulTilingParams.batchN);
    matmulTilingData.set_shareMode(matmulTilingParams.shareMode);
    matmulTilingData.set_shareL1Size(matmulTilingParams.shareL1Size);
    matmulTilingData.set_shareL0CSize(matmulTilingParams.shareL0CSize);
    matmulTilingData.set_shareUbSize(matmulTilingParams.shareUbSize);
    matmulTilingData.set_mxTypePara(matmulTilingParams.mxTypePara);
}

void NewSetQuantBatchMatmulV3Params(DequantBmm::Mc2QuantBatchMatmulV3DataParams& params, Mc2QuantBatchMatmulV3Params& dataParams)
{
    dataParams.set_batchA(params.batchA);
    dataParams.set_batchB(params.batchB);
    dataParams.set_batchC(params.batchC);
    dataParams.set_batchA1(params.batchA1);
    dataParams.set_batchA2(params.batchA2);
    dataParams.set_batchA3(params.batchA3);
    dataParams.set_batchA4(params.batchA4);
    dataParams.set_batchB1(params.batchB1);
    dataParams.set_batchB2(params.batchB2);
    dataParams.set_batchB3(params.batchB3);
    dataParams.set_batchB4(params.batchB4);
    dataParams.set_batchC1(params.batchC1);
    dataParams.set_batchC2(params.batchC2);
    dataParams.set_batchC3(params.batchC3);
    dataParams.set_batchC4(params.batchC4);
    dataParams.set_singleCoreBatch(params.singleCoreBatch);
    dataParams.set_isPerTensor(params.isPerTensor);
    dataParams.set_isPertoken(params.isPertoken);
    dataParams.set_isDoubleScale(params.isDoubleScale);
    dataParams.set_biasThreeDim(params.biasThreeDim);
    dataParams.set_ubCalcN(params.ubCalcN);
    dataParams.set_ubCalcM(params.ubCalcM);
    dataParams.set_needUbBuffer(params.needUbBuffer);
    dataParams.set_realSingleCoreM(params.realSingleCoreM);
    dataParams.set_realSingleCoreN(params.realSingleCoreN);
    dataParams.set_biasDtype(params.biasDtype);
    dataParams.set_ubSize(params.ubSize);
    dataParams.set_isMClash(params.isMClash);
    dataParams.set_isNClash(params.isNClash);
    dataParams.set_groupSizeM(params.groupSizeM);
    dataParams.set_groupSizeN(params.groupSizeN);
    dataParams.set_groupSizeK(params.groupSizeK);
}

void NewCopyQuantBatchMatmulParams(DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& quantBatchMatmulParams, 
    Mc2QuantBatchMatmulV3TilingData& quantBmmV3TilingData)
{
    NewSetMatmulTilingParams(quantBatchMatmulParams.matmulTiling, quantBmmV3TilingData.matmulTiling);

    NewSetQuantBatchMatmulV3Params(quantBatchMatmulParams.params, quantBmmV3TilingData.params);

    quantBmmV3TilingData.adaptiveSlidingWin.set_mTailTile(quantBatchMatmulParams.adaptiveSlidingWin.mTailTile);
    quantBmmV3TilingData.adaptiveSlidingWin.set_nTailTile(quantBatchMatmulParams.adaptiveSlidingWin.nTailTile);

    quantBmmV3TilingData.tileL2cacheTiling.set_mTileCntL2(quantBatchMatmulParams.tileL2cacheTiling.mTileCntL2);
    quantBmmV3TilingData.tileL2cacheTiling.set_nTileCntL2(quantBatchMatmulParams.tileL2cacheTiling.nTileCntL2);
    quantBmmV3TilingData.tileL2cacheTiling.set_mTileBlock(quantBatchMatmulParams.tileL2cacheTiling.mTileBlock);
    quantBmmV3TilingData.tileL2cacheTiling.set_nTileBlock(quantBatchMatmulParams.tileL2cacheTiling.nTileBlock);
    quantBmmV3TilingData.tileL2cacheTiling.set_calOrder(quantBatchMatmulParams.tileL2cacheTiling.calOrder);
    quantBmmV3TilingData.tileL2cacheTiling.set_isBasicTiling(quantBatchMatmulParams.tileL2cacheTiling.isBasicTiling);
}

} // namespace Mc2Log
