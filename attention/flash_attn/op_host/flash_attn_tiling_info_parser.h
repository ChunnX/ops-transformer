/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_tiling_info_parser.h
 * \brief
 */

#pragma once

#include "fa_tiling_info.h"
#include "fa_tiling_shape.h"

namespace optiling {
constexpr int64_t MASK_MODE_INT_MAX = 2147483647;
class FaInfoParser {
public:
    explicit FaInfoParser(const gert::TilingContext *context) : context_(context)
    {
    }
    ~FaInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetCuSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaMaskInfo();
    void GetOptionalInputParaSeqLengthInfo();
    void GetOptionalInputParaPageAttentionInfo();
    void GetOptionalInputParaSinksInfo();

    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetEmptyTensorFlag();

    void GetInOutDataType();
    ge::graphStatus GetBatchSize();
    void GetQueryTSize();
    void GetKeyTSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetS1Size();
    void GetKvStorageMode();
    
    void SetFaShape();
    ge::graphStatus GetMaxActualSeq(const gert::Tensor *actualSeqLensTensor, FaLayout layout, int64_t &maxActualSeqLen);
    ge::graphStatus GetS2SizeFromCuSeqLens();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();

    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetInAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    void GetUpdateInfo();
    void GetPreNextToken();

    ge::graphStatus GetActualSeqInfo();

    void GenerateAxisInfo(FaTilingInfo &faInfo);
    void GenerateDtypeInfo(FaTilingInfo &faInfo);
    void GenerateFeatureInfo(FaTilingInfo &faInfo);
    void GenerateLayoutInfo(FaTilingInfo &faInfo);
    void GenerateInfo(FaTilingInfo &faInfo);
    ge::graphStatus ParseAxisInfo();
    ge::graphStatus ParseFeatureInfo();
    ge::graphStatus Parse(FaTilingInfo &faInfo);

public:
    const gert::TilingContext *context_ = nullptr;

    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    FAParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t queryTSize_ = 0;
    uint32_t keyTSize_ = 0;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;

    // Layout
    FaLayout layoutQ_ = FaLayout::BSND;
    FaLayout layoutKV_ = FaLayout::BSND;
    FaLayout layoutOut_ = FaLayout::BSND;

    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;

    // NPU信息
    NpuArch npuArch_ = NpuArch::DAV_3510;

    // Dtype信息
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    
    // mask 信息
    bool attnMaskFlag_ = false;
    int64_t winLeft_ = 0;
    int64_t winRight_ = 0;
    int32_t maskMode_ = 0;

    // seqLen信息
    uint32_t seqLenQDims_ = 0;
    uint32_t seqLenKvDims_ = 0;
    uint32_t cuseqLenQDims_ = 0;
    uint32_t cuseqLenKvDims_ = 0;
    uint32_t maxSeqLenQ = 0;
    uint32_t maxSeqLenKv = 0;
    // shape信息
    std::shared_ptr<FaTilingShape> queryShape_ = nullptr;
    std::shared_ptr<FaTilingShape> keyShape_ = nullptr;
    std::shared_ptr<FaTilingShape> valueShape_ = nullptr;
};
} // namespace optiling