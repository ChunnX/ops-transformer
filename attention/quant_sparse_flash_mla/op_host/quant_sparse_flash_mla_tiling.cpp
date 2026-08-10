/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file quant_sparse_flash_mla_tiling.cpp
 * \brief QuantSparseFlashMla Tiling entry point and DoOpTiling implementation.
 */

#include "quant_sparse_flash_mla_tiling.h"
#include "qsmla_tiling_info_parser.h"
#include "checkers/qsmla_checker.h"
#include "../op_kernel/quant_sparse_flash_mla_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

struct QSMLACompileInfo {
    int64_t core_num;
};

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForQuantSparseFlashMla(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantSparseFlashMlaTiling类成员函数定义-----------------------
ge::graphStatus QuantSparseFlashMlaTiling::DoOpTiling(QSMLATilingInfo *tilingInfo)
{
    if (tilingInfo->opParamInfo.cmpKv.tensor == nullptr) {
        OP_CHECK_IF(
            tilingInfo->opParamInfo.cmpSparseIndices.tensor != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantSparseFlashMla", "cmp_sparse_indices",
                                                     "cmpSparseIndices must be empty when cmpKv is not provided"),
            return ge::GRAPH_FAILED);
        if (tilingInfo->opParamInfo.oriSparseIndices.tensor != nullptr) {
            perfMode_ = QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE;
        } else {
            perfMode_ = QSMLATemplateMode::SWA_TEMPLATE_MODE;
        }
    } else if (tilingInfo->opParamInfo.cmpSparseIndices.tensor != nullptr) {
        if (tilingInfo->opParamInfo.oriSparseIndices.tensor != nullptr) {
            perfMode_ = QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE;
        } else {
            perfMode_ = QSMLATemplateMode::CSA_TEMPLATE_MODE;
        }
    } else {
        perfMode_ = QSMLATemplateMode::HCA_TEMPLATE_MODE;
    }
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
    constexpr uint32_t M_BASE_SIZE = 64;
    constexpr uint32_t S2_BASE_SIZE = 128;
    constexpr uint32_t D_SIZE = 512;
    constexpr uint32_t VEC_RES_ELEM_SIZE = 2;
    constexpr uint32_t TOPK_MAX_SIZE = 2048;
    constexpr uint32_t UB_SIZE = 248 * 1024;
    constexpr uint32_t SPARSE_BLOCK_ALIGN_NUM = 128;
    uint32_t alignedOriSparseBlockCount = (tilingInfo->oriSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                          SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    uint32_t alignedCmpSparseBlockCount = (tilingInfo->cmpSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                          SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    uint32_t totalBS1 =
        (tilingInfo->qLayout == QSMLALayout::TND) ? tilingInfo->s1Size : (tilingInfo->bSize * tilingInfo->s1Size);

    uint32_t oriBlocksizeFlag = static_cast<uint32_t>((tilingInfo->oriBlockSize & (tilingInfo->oriBlockSize - 1)) == 0);
    uint32_t cmpBlocksizeFlag = static_cast<uint32_t>((tilingInfo->cmpBlockSize & (tilingInfo->cmpBlockSize - 1)) == 0);
    uint32_t blocksizeFlag = (perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) ?
                                 oriBlocksizeFlag :
                                 (oriBlocksizeFlag && cmpBlocksizeFlag);

    uint64_t cmpUbSize = static_cast<uint64_t>(tilingInfo->cmpMaxBlockNumPerBatch) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedCmpSparseBlockCount) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedCmpSparseBlockCount) * sizeof(int64_t);
    uint64_t oriUbSize = static_cast<uint64_t>(tilingInfo->oriMaxBlockNumPerBatch) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedOriSparseBlockCount) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedOriSparseBlockCount) * sizeof(int64_t);
    uint64_t vectorizeUbSize = std::max(cmpUbSize, oriUbSize);

    uint32_t vectorizeFlag = static_cast<uint32_t>((perfMode_ == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                                                    perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                                                    perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) &&
                                                   (vectorizeUbSize <= UB_SIZE) && blocksizeFlag &&
                                                   (tilingInfo->kvLayout == QSMLALayout::PA_BBND));
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (tilingInfo->gSize > 64) {
        workspaceSize += (S2_BASE_SIZE * D_SIZE * VEC_RES_ELEM_SIZE * TRIPLE_BUFFER_NUM * (aicNum >> 1));
    } else {
        workspaceSize += (S2_BASE_SIZE * D_SIZE * VEC_RES_ELEM_SIZE * TRIPLE_BUFFER_NUM * aicNum);
    }
    if (vectorizeFlag) {
        uint64_t oriPhyAddrSize = 0;
        if (perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
            perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            oriPhyAddrSize = static_cast<uint64_t>(totalBS1) * alignedOriSparseBlockCount * sizeof(int64_t);
        }
        uint64_t cmpPhyAddrSize = static_cast<uint64_t>(totalBS1) * alignedCmpSparseBlockCount * sizeof(int64_t);
        workspaceSize += oriPhyAddrSize + cmpPhyAddrSize;
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaces);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.baseParams.set_batchSize(tilingInfo->bSize);
    tilingData_.baseParams.set_kvSeqSize(tilingInfo->s2Size);
    tilingData_.baseParams.set_cmpKvSeqSize(tilingInfo->cmpS2Size);
    tilingData_.baseParams.set_qSeqSize(tilingInfo->s1Size);
    tilingData_.baseParams.set_oriSparseBlockCount(tilingInfo->oriSparseBlockCount);
    tilingData_.baseParams.set_cmpSparseBlockCount(tilingInfo->cmpSparseBlockCount);
    tilingData_.baseParams.set_nNumOfQInOneGroup(tilingInfo->gSize);
    tilingData_.baseParams.set_paOriBlockSize(tilingInfo->oriBlockSize);
    tilingData_.baseParams.set_paCmpBlockSize(tilingInfo->cmpBlockSize);
    tilingData_.baseParams.set_oriMaxBlockNumPerBatch(tilingInfo->oriMaxBlockNumPerBatch);
    tilingData_.baseParams.set_cmpMaxBlockNumPerBatch(tilingInfo->cmpMaxBlockNumPerBatch);

    tilingData_.baseParams.set_softmaxScale(tilingInfo->softmaxScale);
    tilingData_.baseParams.set_oriKvStride(tilingInfo->oriKvStride);
    tilingData_.baseParams.set_cmpKvStride(tilingInfo->cmpKvStride);
    tilingData_.baseParams.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.baseParams.set_oriMaskMode(tilingInfo->oriMaskMode);
    tilingData_.baseParams.set_cmpMaskMode(tilingInfo->cmpMaskMode);
    tilingData_.baseParams.set_oriWinLeft(tilingInfo->oriWinLeft);
    tilingData_.baseParams.set_oriWinRight(tilingInfo->oriWinRight);
    tilingData_.baseParams.set_sparseBlockSize(tilingInfo->sparseBlockSize);
    tilingData_.baseParams.set_dSize(tilingInfo->dSize);
    tilingData_.baseParams.set_dSizeVInput(tilingInfo->dSizeVInput);
    tilingData_.baseParams.set_returnSoftmaxLse(tilingInfo->returnSoftmaxLse);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    uint32_t qLayout = static_cast<uint32_t>(tilingInfo->qLayout);
    uint32_t inputKvLayout = static_cast<uint32_t>(tilingInfo->kvLayout);
    uint32_t tilingKey = GET_TPL_TILING_KEY(0U, qLayout, inputKvLayout, static_cast<uint32_t>(perfMode_),
                                            static_cast<uint32_t>(tilingInfo->gSize > 64), DTYPE_HIF8, vectorizeFlag);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingQuantSparseFlashMla(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("QuantSparseFlashMla", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    QSMLATilingInfo qsmlaInfo;
    QSMLAInfoParser qsmlaInfoParser(context);
    if (qsmlaInfoParser.Parse(qsmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QSMLAChecker qsmlaChecker;
    if (qsmlaChecker.Process(qsmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QuantSparseFlashMlaTiling tiling(context);
    return tiling.DoOpTiling(&qsmlaInfo);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(QuantSparseFlashMla)
    .Tiling(TilingQuantSparseFlashMla)
    .TilingParse<QSMLACompileInfo>(TilingPrepareForQuantSparseFlashMla);

} // namespace optiling
