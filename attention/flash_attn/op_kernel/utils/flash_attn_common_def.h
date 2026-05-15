/**
 * Copyright (c) 2026 Huawei Technologies Co.,
    Ltd.
 * This program is free software,
    you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS,
    WITHOUT WARRANTIES OF ANY KIND,
    EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
    MERCHANTABILITY,
    OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_common_def.h
 * \brief
 */


#ifndef FLASH_ATTN_COMMON_DEF_H_
#define FLASH_ATTN_COMMON_DEF_H_

// 提供flash_attn算子tiling和kernel共用的相关定义

// CUBE与VEC核间同步的模式
static constexpr uint32_t FIA_SYNC_MODE2 = 2;
// BUFFER的字节数
static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
// FP32的0值和极大值
static constexpr float FLOAT_ZERO = 0;
static constexpr float FLOAT_MAX = 3.402823466e+38F;
static constexpr float FLOAT_INF = 3e+99;

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_3_BW 3

enum class FA_LAYOUT : uint32_t
{
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class inferFaLayOutTypeEnum { None = 0, LAYOUT_BSH = 1, LAYOUT_SBH = 2, LAYOUT_BNSD = 3, LAYOUT_TND = 4, LAYOUT_NTD_TND = 5, LAYOUT_NTD = 6};
enum class inferPFALayoutTypeEnum { LAYOUT_BSH = 0, LAYOUT_BNSD = 1, LAYOUT_TND = 2, LAYOUT_NTD = 3};
enum class inferDTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    NotAligned,
};

enum class inferS1TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class inferS2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

enum class inferImplModeEnum {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

#define ImplMode_AA_HIGH_PRECISION 0
#define ImplMode_AA_HIGH_PERFORMANCE 1
#define ImplMode_AA_INVALID_LINE_HIGH_PRECISION 2

static constexpr  inferFaLayOutTypeEnum InOutLayoutTypeValue[5][2] ={
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BNSD},
    {inferFaLayOutTypeEnum::LAYOUT_BSH, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_NTD, inferFaLayOutTypeEnum::LAYOUT_NTD},
};

static constexpr  inferPFALayoutTypeEnum InOutLayoutPFATypeValue[5][2] ={
    {inferPFALayoutTypeEnum::LAYOUT_BNSD, inferPFALayoutTypeEnum::LAYOUT_BNSD},
    {inferPFALayoutTypeEnum::LAYOUT_BSH, inferPFALayoutTypeEnum::LAYOUT_BSH},
    {inferPFALayoutTypeEnum::LAYOUT_TND, inferPFALayoutTypeEnum::LAYOUT_TND},
    {inferPFALayoutTypeEnum::LAYOUT_BNSD, inferPFALayoutTypeEnum::LAYOUT_BSH}, // bsndout占位
    {inferPFALayoutTypeEnum::LAYOUT_NTD, inferPFALayoutTypeEnum::LAYOUT_NTD},
};

#define InOutLayoutType_BNSD_BNSD 0
#define InOutLayoutType_BSH_BSH 1 //BSND由于排布与BSH一样，因此两个会编译成相同取值
#define InOutLayoutType_TND_TND 2
#define InOutLayoutType_BNSD_BSND 3
#define InOutLayoutType_NTD_NTD 4

struct ConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
    inferDTemplateType dv;
};

// Config
static constexpr ConfigParams ConfigValue[] ={
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //0
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //1
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //2
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //3
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned192, inferDTemplateType::Aligned128}, //4
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256, inferDTemplateType::Aligned128}, //5
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //6
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //7
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //8
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned576, inferDTemplateType::Aligned512}, //9
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned64, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //10
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned64, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //11
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned1024, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //12
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //13
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //14
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //15
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //16
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //17
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //18
   {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //19
   {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //20
   {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //21
   {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //22
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128, inferDTemplateType::Aligned64}, //23
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64, inferDTemplateType::Aligned128}, //24
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128, inferDTemplateType::Aligned64}, //25
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64, inferDTemplateType::Aligned128}, //26
};

#define Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64 0
#define Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128 1
#define Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64 2
#define Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128 3
#define Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128 4
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128 5
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256 6
#define Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512 7
#define Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64 8
#define Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 9
#define Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256 10
#define Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512 11
#define Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64 12
#define Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128 13
#define Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256 14
#define Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512 15
#define Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64 16
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128 17
#define Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256 18
#define Config_S1Aligned32_S2Aligned512_DAligned64_DVAligned64 19
#define Config_S1Aligned32_S2Aligned512_DAligned128_DVAligned128 20
#define Config_S1Aligned32_S2Aligned256_DAligned256_DVAligned256 21
#define Config_S1Aligned32_S2Aligned128_DAligned512_DVAligned512 22
#define Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned64 23
#define Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned128 24
#define Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned64 25
#define Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned128 26


// PseMode
#define PSE_MODE_PSE_OUTER_MUL_ADD_TYPE 0
#define PSE_MODE_PSE_OUTER_ADD_MUL_TYPE 1
#define PSE_MODE_PSE_INNER_MUL_ADD_TYPE 2
#define PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE 3
#define PSE_MODE_PSE_INVALID_TYPE 4
#define PSE_MODE_PSE_NONE_TYPE 9

// QuantModeEnum
#define AntiquantMode_PER_CHANNEL 0
#define AntiquantMode_PER_TOKEN 1
#define AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN 2
#define AntiquantMode_PER_TOKEN_HEAD 3
#define AntiquantMode_PER_TOKEN_PAGE_ATTENTION 4
#define AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION 5
#define PerBlock 17
#define FULLQUANT_MODE_PER_TOKEN_HEAD 18
#define FullQuantMode 30
#define NoQuantMode 31

// PFAMaskEnum
#define PFAMask_DISABLE_MASK 0
#define PFAMask_ENABLE_MASK_NO_BAND 1
#define PFAMask_ENABLE_MASK_BAND 2

// PFAMatMulTypeEnum
#define PFAMatMulType_MM_PFA 0
#define PFAMatMulType_MM_PA 1
#define PFAMatMulType_MM_IFA_MLA 2
#define PFAMatMulType_MM_IFA_MLA_PA 3
#define PFAMatMulType_MM_PA_D512 4
#define PFAMatMulType_MM_DN 5

// SplitCoreModeEnum
#define SplitCoreMode_SPLIT_NBS_VECTOR 0
#define SplitCoreMode_SPLIT_NBS_CUBE 1
#define SplitCoreMode_SPLIT_ONEN_VECTOR 2
#define SplitCoreMode_SPLIT_ONEN_CUBE 3
#define SplitCoreMode_BALANCE_VECTOR 4
#define SplitCoreMode_BALANCE_CUBE 5

// bool
#define false 0
#define true 1

// kernel stream related struct
struct FDparamsX {
    uint32_t fdCoreEnable;
 	uint32_t fdBN2Idx;
    uint32_t fdMIdx;
    uint32_t fdS2SplitNum;
    uint32_t mStart;
    uint32_t mLen;
    uint32_t fdWorkspaceIdx;
};

enum class FiaKernelType : uint8_t {
    NO_QUANT = 0,
    ANTI_QUANT,
    FULL_QUANT
};

struct RunInfoX {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isChangeBatch = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint64_t actS1Size = 1;   // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1;   // 当前处理head的S2轴实际大小
    uint32_t actMSize = 0;    // GS1方向上的长度
    uint32_t actMSizeAlign32; // GS1 方向上长度对齐
    uint32_t actVecMSize;  // VEC 视角, 基本块GS1方向长度
    uint32_t vecMbaseIdx;    // VEC 对应的M 轴起始位置,V0 为0， V1 为 V0的actVecMSize

    uint32_t actSingleLoopS2Size = 0; // S2方向长度
    uint32_t actSingleLoopS2SizeAlign;
    // uint32_t curS2LoopTimes = 0;
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0;  // FA阶段，S2外切，需要写到workspace时，写出到第几块M*D的GM块

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;

    uint64_t qPaddingBeginOffset = 0;
    uint64_t kvPaddingBeginOffset = 0;
};


struct CommonConstInfo {
    /* 轴长度 */
    uint32_t bSize;
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dBasicBlock;
    uint32_t dSizeRope;
    uint32_t gSize; /* g轴的大小 */
    uint32_t n2Size;
    uint64_t s1Size; /* s1总大小 */
    uint64_t s2Size; /* s2总大小 */
    uint64_t actualSeqLenSize;   /* 用户输入的actualseq的长度 */
    uint64_t actualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */

    /* FA kernel meta */
    uint32_t bN2Start;
    uint32_t bN2End;
    uint32_t gS1OStart;
    uint32_t gS1OEnd;
    uint32_t s2OStart;
    uint32_t s2OEnd;
    uint32_t coreFirstTmpOutWsPos;

    uint32_t sparseMode; // sparse
    uint32_t attenMaskBatch;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    int64_t preTokens;
    int64_t nextTokens;
    bool isRowInvalidOpen;
    bool isExistRowInvalid;
    float scaleValue;

    /* 核信息 */
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint8_t subBlockIdx;
    uint32_t coreNum;

    /* FA中间结果写出workspace信息 */
    uint32_t accumOutSize;
    uint32_t logSumExpSize;

    /* 输出shape */
    FA_LAYOUT outputLayout;
};

struct PAConstInfo {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t paLayoutType;
};

struct LseConstInfo {
    bool isSoftmaxLseEnable;
};

struct SinkConstInfo {
    bool learnableSinkFlag = false;
};

struct PseConstInfo {
    uint32_t pseShiftByBatch;
    int64_t pseS1Size;
    int64_t pseS2Size;
    uint32_t pseStride;
};

struct TensorListConstInfo {
    bool isKvContinuous;     /* 是否为tensorlist */
};

struct PostQuantConstInfo {
    bool isPostQuantPerChnl;
    bool isPostQuantBF16;
    bool isPostQuantOffsetExist;
    float postQuantScaleValue;
    float postQuantOffsetValue;
};

struct LeftPaddingConstInfo {
    bool isQHasLeftPadding;
    bool isKVHasLeftPadding;
    int64_t queryRightPaddingSize;
    int64_t kvRightPaddingSize;
};

struct SysPrefixConstInfo {
    bool isActualSharedPrefixLenNull = true;
    int64_t actualKVPrefixSize = 0; /* 保存prefix实际长度 */
    int64_t kvPrefixSize = 0;       /* 保存prefix shape完整长度 */
    int64_t prefixLoopCount = 0;    /* 保存prefix参与的S2方向循环次数 */
};

template<FiaKernelType>
struct ConstInfo_t;

template <>
struct ConstInfo_t<FiaKernelType::NO_QUANT> :
CommonConstInfo, PAConstInfo, LseConstInfo, SinkConstInfo, PseConstInfo, TensorListConstInfo, PostQuantConstInfo, LeftPaddingConstInfo, SysPrefixConstInfo {
};

#endif  // FLASH_ATTN_COMMON_DEF_H_