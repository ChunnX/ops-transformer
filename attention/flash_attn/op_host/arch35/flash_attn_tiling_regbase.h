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
 * \file flash_attn_tiling_regbase.h
 * \brief FlashAttn arch35 tiling基类
 */

#ifndef ARCH35_FLASH_ATTN_TILING_REGBASE_H_
#define ARCH35_FLASH_ATTN_TILING_REGBASE_H_

#include <numeric>
#include <alog_pub.h>
#include <tiling/tiling_api.h>
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
#include "../../op_kernel/arch35/flash_attn_template_tiling_key.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#include "err/ops_err.h"
#include "platform/soc_spec.h"
#include "../flash_attn_tiling_common.h"

using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace FA {

// 常量定义（参照flash_attn_score_tiling_regbase.h保持一致）
static const int64_t FA_DN_D_64           = 64L;
static const int64_t FA_NUM_64            = 64L;
static const int64_t FA_NUM_128           = 128L;
static const int64_t FA_NUM_192           = 192L;
static const int64_t FA_NUM_256           = 256L;
static const int64_t FA_NUM_512           = 512L;
static const int64_t FA_NUM_768           = 768L;
static const int64_t FA_NUM_1024          = 1024L;
static const int64_t FA_D_TEMPLATE_SPLIT  = 64L;
static const int64_t FA_MIN_D_WORKSPACE   = 128L;
static const int64_t FA_PING_PONG         = 3L;
static const int64_t FA_GM_ALIGN          = 512L;
static const int64_t FA_HEAD_DIM_MAX      = 768L;
static const uint32_t FA_DIM_NUM_2        = 2U;
static const uint32_t FA_DIM_NUM_3        = 3U;
static const uint32_t FA_DIM_NUM_4        = 4U;

// 输入索引（与flash_attn_def.cpp对齐）
static const size_t FA_INPUT_Q_INDEX            = 0UL;
static const size_t FA_INPUT_K_INDEX            = 1UL;
static const size_t FA_INPUT_V_INDEX            = 2UL;
static const size_t FA_INPUT_BLOCK_TABLE_INDEX  = 3UL;
static const size_t FA_INPUT_CU_SEQLENS_Q_INDEX = 4UL;
static const size_t FA_INPUT_CU_SEQLENS_KV_INDEX = 5UL;
static const size_t FA_INPUT_SEQUSED_Q_INDEX    = 6UL;
static const size_t FA_INPUT_SEQUSED_KV_INDEX   = 7UL;
static const size_t FA_INPUT_SINKS_INDEX        = 8UL;
static const size_t FA_INPUT_METADATA_INDEX     = 9UL;

// 输出索引
static const size_t FA_OUTPUT_ATTN_OUT_INDEX    = 0UL;
static const size_t FA_OUTPUT_SOFTMAX_LSE_INDEX = 1UL;

// 属性索引（与flash_attn_def.cpp对齐）
static const size_t FA_ATTR_SOFTMAX_MODE_INDEX    = 0UL;
static const size_t FA_ATTR_MASK_MODE_INDEX       = 1UL;
static const size_t FA_ATTR_WIN_LEFT_INDEX        = 2UL;
static const size_t FA_ATTR_WIN_RIGHT_INDEX       = 3UL;

static const size_t FA_ATTR_MAX_SEQLEN_Q_INDEX    = 4UL;
static const size_t FA_ATTR_MAX_SEQLEN_KV_INDEX   = 5UL;

static const size_t FA_ATTR_LAYOUT_Q_INDEX        = 6UL;
static const size_t FA_ATTR_LAYOUT_KV_INDEX       = 7UL;
static const size_t FA_ATTR_LAYOUT_OUT_INDEX      = 8UL;
static const size_t FA_ATTR_RETURN_SOFTMAX_LSE    = 9UL;
static const size_t FA_ATTR_DETERMINISTIC         = 10UL;

// layout枚举（与flash_attn_score对齐）
enum class FALayoutType : uint8_t {
    NONE  = 0,
    BSND  = 1,
    BNSD  = 3,
    TND   = 4,
};

// KV layout枚举（含PA场景）
enum class FAKVLayoutType : uint8_t {
    BNSD   = 0,
    TND    = 1,
    PA_ND  = 2,
    PA_Nz  = 3,
};

// ImplMode枚举（与flash_attn_score对齐）
enum class FAImplMode : uint8_t {
    HIGH_PRECISION          = 0,
    HIGH_PERFORMANCE        = 1,
    INVALID_LINE_HIGH_PREC  = 2,
};

// DTemplateType枚举（与flash_attn_score对齐）
enum class FADTemplateType : uint32_t {
    NONALIGNED  = 0,
    ALIGNED_64  = 64,
    ALIGNED_128 = 128,
    ALIGNED_192 = 192,
    ALIGNED_256 = 256,
    ALIGNED_768 = 768,
    BOTTOM
};

// STemplateType枚举（与flash_attn_score对齐）
enum class FASTemplateType : uint32_t {
    NONALIGNED  = 0,
    ALIGNED_64  = 64,
    ALIGNED_128 = 128,
    ALIGNED_256 = 256,
    BOTTOM
};

template <typename T>
static auto FA_AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) { return 0; }
    if (num1 < 0) { return -(-num1 / num2) * num2; }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static auto FA_CeilDiv(T num1, T num2) -> T
{
    if (num2 == 0) { return 0; }
    return (num1 + num2 - 1) / num2;
}

// FlashAttn Tiling基类（arch35）
class FlashAttnTilingRegbase : public TilingBaseClass {
public:
    explicit FlashAttnTilingRegbase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~FlashAttnTilingRegbase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Reset();
        TilingBaseClass::Reset(context);
    }

protected:
    void Reset();

    bool IsCapable() override { return true; }

    // 1. 获取平台信息
    ge::graphStatus GetPlatformInfo() override;
    // 2. 解析shape/attr/optional input
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3. 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4. 计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5. 计算TilingKey（纯虚，由子类实现）
    uint64_t GetTilingKey() const override = 0;
    // 6. 计算Workspace大小（纯虚，由子类实现）
    ge::graphStatus GetWorkspaceSize() override = 0;
    // 7. 保存Tiling数据
    ge::graphStatus PostTiling() override;

    // context校验（子类可override）
    virtual ge::graphStatus CheckContext();
    // 解析dtype
    virtual bool AnalyzeDtype();
    // 解析属性
    bool AnalyzeAttrs();
    // 解析layout（Q/KV/Out）
    bool AnalyzeLayout();
    // 解析变长序列参数（cuSeqlens/seqused）
    bool AnalyzeVarLenInput();
    // 解析分页注意力参数（blockTable）
    bool AnalyzePAInput();
    // 解析metadata（预计算tiling）
    bool AnalyzeMetadataInput();

    // S1/S2基本块计算（纯虚，由子类实现）
    virtual void CalcS1S2BasicBlock() = 0;
    // D基本块计算（纯虚，由子类实现）
    virtual void CalcDBasicBlock() = 0;
    // Dv基本块计算
    virtual void CalcDVBasicBlock();
    // 总计算量
    virtual int64_t CalcTotalSize();

    // 平台信息
    uint32_t aivNum = 0U;
    uint32_t aicNum = 0U;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch = NpuArch::DAV_RESV;

    // 数据类型
    ge::DataType inputDtype = ge::DT_FLOAT16;
    int64_t inputDtypeBytes = 2L;
    int64_t calcTypeSize    = 4L;  // 计算精度（高精度时为float大小）
    matmul_tiling::DataType bmmDtype     = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;
    bool isHighPrecision = true;
    DtypeEnum tilingKeyDType = DtypeEnum::FLOAT16;

    // layout
    FALayoutType   tilingKeyLayout    = FALayoutType::NONE;
    FAKVLayoutType tilingKeyKVLayout  = FAKVLayoutType::BNSD;
    FAImplMode     implMode           = FAImplMode::HIGH_PRECISION;
    const char    *inputLayoutQ       = nullptr;
    const char    *inputLayoutKv      = nullptr;
    const char    *inputLayoutOut     = nullptr;

    // shape维度
    int64_t bSize    = 0LL;
    int64_t gSize    = 0LL;
    int64_t dSize    = 0LL;
    int64_t dSizeV   = 0LL;
    int64_t n1Size   = 0LL;  // numHeadsQ（由shape推断）
    int64_t n2Size   = 0LL;  // numHeadsKV
    int64_t s1Size   = 0LL;
    int64_t s2Size   = 0LL;
    int64_t accumS1  = 0LL;  // TND时的累计token数
    int64_t accumS2  = 0LL;
    int64_t realT1Size = 0LL;

    // 属性
    float   softmaxScale      = 0.0f;
    int64_t maskMode          = 0LL;
    int64_t winLeft           = 0LL;
    int64_t winRight          = 0LL;
    int64_t returnSoftmaxLse  = 0LL;
    int64_t deterministic     = 0LL;

    // 可选输入标志
    bool hasAttenMask    = false;  // maskMode != 0时有效
    bool isPA            = false;  // 分页注意力
    bool isTND           = false;  // TND变长layout
    bool hasVarLen       = false;  // 有cuSeqlens或seqused
    bool hasMetadata     = false;  // 预计算tiling

    // PA参数
    int32_t blockSize        = 0;
    int32_t blockTableDim2   = 0;
    int32_t paBlockNumSum    = 0;
    uint8_t paLayoutType     = 0U;

    // 变长序列数据（TND场景）
    std::vector<int64_t> actualSeqLenData;
    std::vector<int64_t> actualSeqLenKvData;

    // 基本块大小（由子类CalcS1S2BasicBlock/CalcDBasicBlock填充）
    int64_t s1BasicBlock = 128LL;
    int64_t s2BasicBlock = 128LL;
    int64_t dBasicBlock  = 128LL;
    int64_t dVBasicBlock = 128LL;

    // DTemplateType
    FADTemplateType dTemplateType  = FADTemplateType::BOTTOM;
    FADTemplateType dVTemplateType = FADTemplateType::BOTTOM;

    // 算子名
    const char *opName = nullptr;

    // TilingData
    FlashAttentionScoreSimplifiedTilingData *tilingData =
        context_->GetTilingData<FlashAttentionScoreSimplifiedTilingData>();
    InputParamsRegbase *inputParamsRegbase_ = &tilingData->inputParamsRegbase;
    MultiCoreParamsRegbase *multiCoreParamsRegbase_ = &tilingData->multiCoreParamsRegbase;
    DropmaskParamsRegbase *dropmaskParamsRegbase_ = &tilingData->dropmaskParamsRegbase;
};

} // namespace FA
} // namespace optiling

#endif // ARCH35_FLASH_ATTN_TILING_REGBASE_H_