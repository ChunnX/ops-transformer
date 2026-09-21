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
 * \file qbsa_tiling_context.h
 * \brief Raw Host input collection and the current PA shape decoder.
 */

#ifndef QBSA_TILING_CONTEXT_H_
#define QBSA_TILING_CONTEXT_H_

#include <string>

#include <exe_graph/runtime/tiling_context.h>
#include "qbsa_op_contract.h"

namespace optiling {

struct QBSARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::StorageShape *shape = nullptr;
    const gert::Stride *stride = nullptr;
};

struct QBSAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::Tensor *tensor = nullptr;
};

struct QuantBlockSparseAttnParaInfo {
    QBSARequiredParaInfo query;
    QBSARequiredParaInfo key;
    QBSARequiredParaInfo value;
    QBSARequiredParaInfo qDescale;
    QBSARequiredParaInfo kDescale;
    QBSARequiredParaInfo vDescale;
    QBSARequiredParaInfo pScale;
    QBSARequiredParaInfo sparseIndices;
    QBSARequiredParaInfo sparseSeqLen;
    QBSARequiredParaInfo attenMask;

    QBSAOptionalParaInfo cuSeqlensQ;
    QBSAOptionalParaInfo cuSeqlensKV;
    QBSAOptionalParaInfo seqUsedQ;
    QBSAOptionalParaInfo seqUsedKV;
    QBSAOptionalParaInfo blockTable;
    QBSAOptionalParaInfo metadata;

    QBSARequiredParaInfo attnOut;
    QBSARequiredParaInfo lseOut;

    const float *softmaxScale = nullptr;
    const int64_t *qBlockSize = nullptr;
    const int64_t *kvBlockSize = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKV = nullptr;
    const char *layoutSparseIndices = nullptr;
    const int64_t *quantMode = nullptr;
    const int64_t *maskMode = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

struct QuantBlockSparseAttnTilingInfo {
    QuantBlockSparseAttnParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t qbMax = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t qTokenNum = 0;

    uint32_t sparseCount = 0;
    uint32_t qBlockSizeVal = 0;
    uint32_t kvBlockSizeVal = 0;
    uint32_t paBlockSizeVal = 0;
    uint32_t paBlockStrideVal = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t paBlockNumSum = 0;
    uint32_t gS1OuterSize = 0;

    ge::DataType qDtype = ge::DT_UNDEFINED;
    ge::DataType kvDtype = ge::DT_UNDEFINED;

    float softmaxScaleVal = 1.0F;
    uint32_t maskModeVal = 0;
    uint32_t quantModeVal = 1;
    uint32_t layoutQValue = 0;
    std::string layoutQStr = "TND";
    std::string layoutKVStr = "PA_BNBD";
    std::string layoutSparseIndicesStr = "B_N_Qb_Kb";
    std::string layoutOutStr = "TND";
    bool returnSoftmaxLseVal = false;

    bool isGqa = false;
    bool hasViewStride = true;
};

class QbsaTilingContextParser {
public:
    explicit QbsaTilingContextParser(gert::TilingContext *context);
    ~QbsaTilingContextParser() = default;

    // Collect only stable IR data. Architecture/mode strategies choose their own shape decoder afterwards.
    ge::graphStatus Collect(QuantBlockSparseAttnTilingInfo &tilingInfo);
    ge::graphStatus ParsePaShapeFacts(QuantBlockSparseAttnTilingInfo &tilingInfo);

private:
    ge::graphStatus ParsePaBlockTable(QuantBlockSparseAttnTilingInfo &tilingInfo);
    ge::graphStatus ParseQuery(QuantBlockSparseAttnTilingInfo &tilingInfo, const gert::Shape &queryShape,
                               const gert::Shape &sparseIndicesShape);
    ge::graphStatus ParseKeyValue(QuantBlockSparseAttnTilingInfo &tilingInfo, const gert::Shape &keyShape,
                                  const gert::Shape &valueShape, const gert::Stride *keyStride);
    ge::graphStatus ParseSparseIndices(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                       const gert::Shape &sparseIndicesShape);
    ge::graphStatus ParseOptionalInputs(QuantBlockSparseAttnTilingInfo &tilingInfo);
    ge::graphStatus ParseAttributes(QuantBlockSparseAttnTilingInfo &tilingInfo, const gert::RuntimeAttrs *attrs);

    gert::TilingContext *context_;
};

} // namespace optiling

#endif // QBSA_TILING_CONTEXT_H_
