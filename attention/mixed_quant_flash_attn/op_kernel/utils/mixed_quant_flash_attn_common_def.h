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
 * \file mixed_quant_flash_attn_common_def.h
 * \brief MixedQuantFlashAttn 伪量化算子tiling和kernel共用的相关定义
 */

#ifndef MIXED_QUANT_FLASH_ATTN_COMMON_DEF_H_
#define MIXED_QUANT_FLASH_ATTN_COMMON_DEF_H_

#define ASCENDC_TPL_4_BW 4

#define ASCENDC_TPL_3_BW 3

#define InOutLayoutType_BSND 0
#define InOutLayoutType_BNSD 1
#define InOutLayoutType_TND 2

#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BBH 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

#define AntiquantMode_PER_CHANNEL 0
#define AntiquantMode_PER_TOKEN_HEAD_GROUP 1
#define AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN 2
#define AntiquantMode_PER_TOKEN_HEAD 3
#define AntiquantMode_PER_TOKEN_PAGE_ATTENTION 4
#define AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION 5

enum class inferFaLayOutTypeEnum {
    LAYOUT_BSND = 0,
    LAYOUT_BNSD = 1,
    LAYOUT_TND = 2
};

enum class inferS1TemplateType {
    Aligned64 = 64,
    Aligned128 = 128,
    NotAligned,
};

enum class inferS2TemplateType {
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class inferDTemplateType {
    Aligned128 = 128,
    NotAligned,
};

struct MixedQuantConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
};

static constexpr inferFaLayOutTypeEnum InOutLayoutTypeValue[3][2] = {
    {inferFaLayOutTypeEnum::LAYOUT_BSND, inferFaLayOutTypeEnum::LAYOUT_BSND},
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BNSD},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
};

static constexpr MixedQuantConfigParams MixedQuantConfigValue[] = {
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128},
};

static constexpr float FLOAT_ZERO = 0;
static constexpr float FLOAT_MAX = 3.402823466e+38F;
static constexpr float FLOAT_INF = 3e+99;

static constexpr uint32_t MQFA_CV_RATIO = 2;
static constexpr uint32_t MQFA_BYTE_BLOCK = 32;
static constexpr uint32_t MQFA_METADATA_HEADER_OFFSET = 16U * sizeof(uint32_t);

struct MixedQuantCommonConstInfo {
    uint32_t bSize;
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dBasicBlock;
    uint32_t gSize;
    uint32_t n2Size;
    uint64_t s1Size;
    uint64_t s2Size;
    uint64_t cuSeqLensQSize;
    uint64_t cuSeqLensKVSize;
    uint64_t seqUsedQSize;
    uint64_t seqUsedKvSize;

    uint32_t sparseMode;
    uint32_t attenMaskBatch;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    int64_t preTokens;
    int64_t nextTokens;
    bool isExistRowInvalid;
    float scaleValue;

    uint32_t aicIdx;
    uint32_t aivIdx;
    uint8_t subBlockIdx;
    uint32_t coreNum;

    uint32_t accumOutSize;
    uint32_t logSumExpSize;

    uint32_t outputLayout;
    bool needInitOutput;

    bool isKvContinuous;
};

struct MixedQuantPAConstInfo {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t paLayoutType;
};

struct MixedQuantLseConstInfo {
    bool isSoftmaxLseEnable;
};

struct MixedQuantSinkConstInfo {
    bool learnableSinkFlag = false;
};

struct MixedQuantConstInfoX : MixedQuantCommonConstInfo,
                              MixedQuantPAConstInfo,
                              MixedQuantLseConstInfo,
                              MixedQuantSinkConstInfo {};

#endif // MIXED_QUANT_FLASH_ATTN_COMMON_DEF_H_
