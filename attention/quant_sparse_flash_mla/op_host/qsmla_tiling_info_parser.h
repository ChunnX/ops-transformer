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
 * \file qsmla_tiling_info_parser.h
 * \brief QSMLAInfoParser: parses TilingContext into QSMLATilingInfo.
 */

#ifndef QSMLA_TILING_INFO_PARSER_H
#define QSMLA_TILING_INFO_PARSER_H

#include "qsmla_tiling_info.h"

namespace optiling {

class QSMLAInfoParser {
public:
    explicit QSMLAInfoParser(gert::TilingContext *context) : context_(context)
    {
    }
    ~QSMLAInfoParser() = default;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, QSMLALayout &layout,
                                        const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();

    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetKvLayout();
    void SetQSMLAShape();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    ge::graphStatus GetDSizeQ();
    ge::graphStatus GetDSizeKV();
    ge::graphStatus GetKvstride();
    void GenerateInfo(QSMLATilingInfo &qsmlaInfo);
    ge::graphStatus Parse(QSMLATilingInfo &qsmlaInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    QSMLAParaInfo opParamInfo_;

    bool HasAxis(const QSMLAAxis &axis, const QSMLALayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const QSMLAAxis &axis, const QSMLALayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const QSMLAAxis &axis, const QSMLALayout &layout) const;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t cmpS2Size_ = 0;
    uint32_t headDim_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t qkHeadDim_ = 0;
    int64_t sparseBlockSize_ = 0;
    int64_t oriSparseBlockCount_ = 0;
    int64_t cmpSparseBlockCount_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameSeqAllKVTensor_ = true;
    uint32_t dSizeQ_ = 0;
    uint32_t dSizeKV_ = 0;
    uint32_t oriKvStride_ = 0;
    uint32_t cmpKvStride_ = 0;
    std::vector<int64_t> oriKvStridesVec_;
    std::vector<int64_t> cmpKvStridesVec_;
    QSMLALayout qLayout_ = QSMLALayout::BSND;
    QSMLALayout outLayout_ = QSMLALayout::BSND;
    QSMLALayout kvLayout_ = QSMLALayout::PA_BBND;
    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpSparseIndicesType_ = ge::DT_INT32;
    ge::DataType oriBlockTableType_ = ge::DT_INT32;
    ge::DataType cmpBlockTableType_ = ge::DT_INT32;
    ge::DataType cuSeqLensQType_ = ge::DT_INT32;
    ge::DataType seqsedKvType_ = ge::DT_INT32;
    ge::DataType sinksType_ = ge::DT_INT32;
    ge::DataType metadataType_ = ge::DT_INT32;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    gert::Shape qShape_{};
    gert::Shape oriKvShape_{};
    gert::Shape cmpKvShape_{};
    gert::Shape oriSparseIndicesShape_{};
    gert::Shape cmpSparseIndicesShape_{};
};

} // namespace optiling
#endif // QSMLA_TILING_INFO_PARSER_H
