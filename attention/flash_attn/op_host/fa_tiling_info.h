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
 * \file fa_tiling_info.h
 * \brief
 */

#ifndef FLASH_ATTN_FA_TILING_INFO_H
#define FLASH_ATTN_FA_TILING_INFO_H

#include <vector>
#include "fa_tiling_base.h"

namespace optiling {
const std::string QUERY_NAME = "q";
const std::string KEY_NAME = "k";
const std::string VALUE_NAME = "v";
const std::string BLOCK_TABLE_NAME = "block_table";
const std::string CU_SEQLENS_Q_NAME = "cu_seqlens_q";
const std::string CU_SEQLENS_KV_NAME = "cu_seqlens_kv";
const std::string SEQUSED_Q_NAME = "seqused_q";
const std::string SEQUSED_KV_NAME = "seqused_kv";
const std::string SINKS_NAME = "sinks";
const std::string METADATA_NAME = "metadata";
const std::string SOFTMAX_SCALE_NAME = "softmax_scale";
const std::string MASK_MODE_NAME = "mask_mode";
const std::string ATTN_MASK_NAME = "attn_mask";
const std::string WIN_LEFT_NAME = "win_left";
const std::string WIN_RIGHT_NAME = "win_right";
const std::string MAX_SEQLEN_Q_NAME = "max_seqlen_q";
const std::string MAX_SEQLEN_KV_NAME = "max_seqlen_kv";
const std::string LAYOUT_Q_NAME = "layout_q";
const std::string LAYOUT_KV_NAME = "layout_kv";
const std::string LAYOUT_OUT_NAME = "layout_out";
const std::string RETURN_SOFTMAX_LSE_NAME = "return_softmax_lse";
const std::string DETERMINISTIC_NAME = "deterministic";
const std::string ATTN_OUT_NAME = "attn_out";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";

enum class MaskMode : int32_t {
    NO_MASK = 0,
    CAUSAL = 3,
    BAND = 4
};

enum class FaLayout : uint32_t {
    BSND = 0,
    BNSD = 1,
    TND = 2,
    PA_BBND = 3,
    PA_BNBD = 4,
    PA_Nz = 5,
    LSE_BNS = 6,
    LSE_NT = 7
};

const std::map<std::string, FaLayout> layoutMap = {{"BSND", FaLayout::BSND},       {"BNSD", FaLayout::BNSD},
                                                   {"TND", FaLayout::TND},         {"PA_BBND", FaLayout::PA_BBND},
                                                   {"PA_BNBD", FaLayout::PA_BNBD}, {"PA_Nz", FaLayout::PA_Nz}};

enum class FaAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    H = 4,
    T = 5,
    D1 = 6,
    D0 = 7,
    S1 = 8,
    S2 = 9,
    Bn = 10,
    Bs = 11,
    CONST = 12
};

enum class FaQuantMode : uint32_t {
    NO_QUANT = 0,
    ANTI_QUANT,
    FULL_QUANT
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    PAGE_ATTENTION = 1
};

enum class FaTilingInOutMode : uint32_t {
    FP16_FP16 = 0,
    BF16_BF16 = 1
};

std::string LayoutToSerialString(FaLayout layout);
std::string AxisToSerialString(FaAxis axis);
std::string QuantModeToSerialString(FaQuantMode faQuantMode);
// std::string SituationToSerialString(RopeMode ropeMode);

struct FARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct FAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct FAParaInfo {
    FARequiredParaInfo query = {nullptr, nullptr};
    FARequiredParaInfo key = {nullptr, nullptr};
    FARequiredParaInfo value = {nullptr, nullptr};

    FAOptionalParaInfo blockTable = {nullptr, nullptr};
    FAOptionalParaInfo cuSeqlensQ = {nullptr, nullptr};
    FAOptionalParaInfo cuSeqlensKv = {nullptr, nullptr};
    FAOptionalParaInfo sequsedQ = {nullptr, nullptr};
    FAOptionalParaInfo sequsedKv = {nullptr, nullptr};
    FAOptionalParaInfo sinks = {nullptr, nullptr};
    FAOptionalParaInfo metadata = {nullptr, nullptr};
    FAOptionalParaInfo attnMask = {nullptr, nullptr};

    const float *softmaxScale = nullptr;
    const int64_t *maskMode = nullptr;
    const int64_t *winLeft = nullptr;
    const int64_t *winRight = nullptr;
    const int64_t *maxSeqlenQ = nullptr;
    const int64_t *maxSeqlenKV = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKV = nullptr;
    const char *layoutOut = nullptr;
    const int64_t *returnSoftMaxLse = nullptr;
    const int64_t *deterministic = nullptr;

    FARequiredParaInfo attnOut = {nullptr, nullptr};
    FARequiredParaInfo lseOut = {nullptr, nullptr};
};

class FaTilingInfo : public TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    FAParaInfo opParamInfo;

    // Base Param
    int64_t bSize = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t qkHeadDim = 0;
    int64_t vHeadDim = 0;
    int64_t gSize = 0;
    int64_t qTSize = 0; // 仅TND/NTD时生效
    int64_t kTSize = 0;
    float softmaxScale = 0;

    uint64_t totalOutputSize = 0;
    uint64_t totalLseSize = 0;

    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    int64_t blockTypeSize = 0; // 计算中间量大小
    int64_t maxBlockNumPerBatch = 0;
    int64_t totalBlockNum = 0;


    // Q seq_lens
    int64_t seqUsedQDims = 0;
    int64_t cuSeqLenQDims = 0;
    int64_t maxSeqQ = 0;

    // Kv seq_lensKv
    int64_t seqUsedKvDims = 0;
    int64_t cuSeqLenKvDims = 0;
    int64_t maxSeqKv = 0;

    // Mask
    int32_t maskMode = 0;
    int64_t winLeft = 0;
    int64_t winRight = 0;

    // Others Flag
    bool batchContinuousFlag = true;
    bool softmaxLseFlag = false;
    bool deterministicFlag = false;
    bool sinksFlag = false;
    bool emptyTensorFlag = false;

    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    // Layout
    FaLayout qLayout = FaLayout::BSND;
    FaLayout outLayout = FaLayout::BSND;
    FaLayout kvLayout = FaLayout::BSND;
};
} // namespace optiling
#endif // FLASH_ATTN_FA_TILING_INFO_H