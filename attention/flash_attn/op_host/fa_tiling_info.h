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
 * \file fa_tiling_info.h
 * \brief
 */

#ifndef FA_TILING_INFO_H
#define FA_TILING_INFO_H

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
const std::string ATTEN_MASK_NAME = "attn_mask";
const std::string WIN_LEFT_NAME = "win_left";
const std::string WIN_RIGHT_NAME = "win_right";
const std::string MAX_SEQLEN_Q_NAME = "max_seqlen_q";
const std::string MAX_SEQLEN_KV_NAME = "max_seqlen_kv";
const std::string LAYOUT_Q_NAME = "layout_q";
const std::string LAYOUT_KV_NAME = "layout_kv";
const std::string LAYOUT_OUT_NAME = "layout_out";
const std::string RETURN_SOFTMAX_LSE_NAME = "return_softmax_lse";
const std::string DETERMINISTIC_NAME = "deterministic";
const std::string ATTEN_OUT_NAME = "attn_out";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";

enum class MaskMode : int32_t {
    NO_MASK = 1,
    CAUSAL = 3,
    BAND = 4
};

enum class FaLayout : uint32_t {
    BSND = 0,
    BNSD = 1,
    TND = 2,
    PA_BBND = 3,
    PA_BNBD = 4,
    PA_Nz = 5
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
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t gSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t qTSize = 0; // 仅TND/NTD时生效
    uint32_t kTSize = 0;
    float softmaxScale = 0;
    int32_t innerPrecise = 0;

    uint64_t l2CacheSize = 0;
    std::vector<gert::StorageShape *> kCache = {};
    std::vector<gert::StorageShape *> vCache = {};

    // empty Tensor
    bool emptyTensorFlag = false;
    uint64_t totalOutputSize = 0;
    uint64_t totalLseSize = 0;

    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t blockTypeSize = 0; // 计算中间量大小
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    // antiquant
    bool antiQuantFlag = false;
    uint32_t msdIterNum = 1;
    uint32_t antiqSeqSize = 0;
    uint32_t antiquantMode = 0;
    uint32_t keyAntiquantMode = 0;
    uint32_t valueAntiquantMode = 0;
    bool antiquantSplit = false;

    // SysTem Prefix
    bool sysPrefixFlag = false;
    uint32_t systemPrefixMaxLen = 0;
    uint32_t systemPrefixLen = 0;

    // Q actual_seq_lens
    uint32_t actualLenQDims = 0;
    int64_t maxActualseq = 0;
    bool isAccumQSeq = false;

    // KV actual_seq_lens
    bool actualSeqLenFlag = false;

    uint32_t actualLenDims = 0;
    std::vector<int64_t> kvListSeqLens{};
    std::vector<int64_t> actualSeqLengthsQ{};
    std::vector<int64_t> actualSeqLengthsKV{};
    bool isAccumKVSeq = false;

    // PSE
    bool pseShiftFlag = false;
    uint32_t pseShiftByBatch = 0U;
    uint32_t pseShiftS1 = 0U;
    uint32_t pseShiftS2 = 0U;
    bool enableAlibiPse = false;

    // Mask
    bool attnMaskFlag = false;
    bool isExistRowInvalid = false;
    uint32_t attnMaskBatchStride = 0;
    uint32_t attnMaskStride = 0;
    int32_t maskMode = 0;

    int64_t winLeft = 0;
    int64_t winRight = 0;

    // PostQuant
    bool isOutQuantPerChnOut = false;
    bool isOutQuantTypeBf16 = false;
    bool isOutQuantEnable = false;

    // Others Flag
    bool batchContinuousFlag = true;
    bool kvPaddingSizeFlag = false;
    bool qPaddingSizeFlag = false;
    bool softmaxLseFlag = false;
    bool quantFlag = false;
    bool isMaxWorkspace = false;
    bool isLegacyIfa = false;
    bool needInit = false;
    bool slidingFlag = false;
    bool sinksFlag = false;
    bool isQKVDDifferent = false;
    // DType
    FaTilingInOutMode inOutMode = FaTilingInOutMode::FP16_FP16;
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;
    ge::DataType inputQRopeType = ge::DT_FLOAT16;
    ge::DataType inputKRopeType = ge::DT_FLOAT16;

    FaLayout qLayout = FaLayout::BSND;
    FaLayout outLayout = FaLayout::BSND;
    FaLayout kvLayout = FaLayout::BSND;
};
} // namespace optiling
#endif // FA_TILING_INFO_H