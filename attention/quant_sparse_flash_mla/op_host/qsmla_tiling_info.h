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
 * \file qsmla_tiling_info.h
 * \brief QSMLAParaInfo struct and QSMLATilingInfo class: parsed tiling parameters.
 */

#ifndef QSMLA_TILING_INFO_H
#define QSMLA_TILING_INFO_H

#include "qsmla_tiling_common.h"

namespace optiling {

struct QSMLAParaInfo {
    QSMLATilingRequiredParaInfo q = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo qDescale = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriKvDescale = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpKvDescale = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriSparseIndices = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpSparseIndices = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriBlockTable = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpBlockTable = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cuSeqLensQ = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cuSeqLensOriKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cuSeqLensCmpKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo seqUsedQ = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo sequsedOriKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo sequsedCmpKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpResidualKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriTopkLength = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpTopkLength = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo sinks = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo metadata = {nullptr, nullptr};
    QSMLATilingRequiredParaInfo attnOut = {nullptr, nullptr};
    QSMLATilingRequiredParaInfo softmaxLse = {nullptr, nullptr};

    const int64_t *quantMode = nullptr;
    const float *softmaxScale = nullptr;
    const int64_t *oriKvStride = nullptr;
    const int64_t *cmpKvStride = nullptr;
    const int64_t *cmpRatio = nullptr;
    const uint32_t *oriMaskMode = nullptr;
    const uint32_t *cmpMaskMode = nullptr;
    const int64_t *oriWinLeft = nullptr;
    const int64_t *oriWinRight = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKv = nullptr;
    const int64_t *topkValueMode = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

class QSMLATilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    QSMLAParaInfo opParamInfo;

    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch = NpuArch::DAV_2201;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t cmpS2Size = 0;
    uint32_t gSize = 0;
    uint32_t qkHeadDim = 0;
    uint32_t qTSize = 0;

    uint32_t maxActualseq = 0;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;

    int64_t quantMode = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dSizeVInput = 0;
    float softmaxScale = 0;
    int64_t oriKvStride = 0;
    int64_t cmpKvStride = 0;
    std::vector<int64_t> oriKvStrides;
    std::vector<int64_t> cmpKvStrides;
    gert::Shape oriKvStorageShape;
    gert::Shape cmpKvStorageShape;
    int64_t cmpRatio = 0;
    uint64_t oriMaskMode = 0;
    uint64_t cmpMaskMode = 0;
    int64_t topkValueMode = 0;
    int64_t oriWinLeft = 0;
    int64_t oriWinRight = 0;
    int64_t sparseBlockSize = 0;
    int64_t oriSparseBlockCount = 0;
    int64_t cmpSparseBlockCount = 0;
    int32_t sparseMode = 0;
    uint32_t sparseCount = 0;

    uint32_t blockTypeSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    int32_t oriBlockSize = 0;
    int32_t cmpBlockSize = 0;
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    ge::DataType qType = ge::DT_FLOAT16;
    ge::DataType oriKvType = ge::DT_FLOAT16;
    ge::DataType cmpKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    QSMLALayout qLayout = QSMLALayout::BSND;
    QSMLALayout kvLayout = QSMLALayout::PA_BBND;
    QSMLALayout outLayout = QSMLALayout::BSND;

    bool returnSoftmaxLse = false;

    QSMLATemplateMode templateMode = QSMLATemplateMode::SWA_TEMPLATE_MODE;
};

} // namespace optiling
#endif // QSMLA_TILING_INFO_H
