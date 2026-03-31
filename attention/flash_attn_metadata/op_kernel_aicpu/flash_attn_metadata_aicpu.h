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
 * \file kv_quant_sparse_attn_sharedkv_metadata_aicpu.h
 * \brief
 */

#ifndef FLASH_ATTN_METADATA_AICPU_H
#define FLASH_ATTN_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "flash_attn_metadata.h"

using namespace optiling;

namespace aicpu {

template <typename T>
static auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
inline auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, bool>::type
GetAttrValue(CpuKernelContext &ctx, const std::string &name, T &value) {
  auto attr = ctx.GetAttr(name);
  if (!attr) {
    KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
    return false;
  }
  value = static_cast<T>(attr->GetInt());
  return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name,
                         std::string &value) {
  auto attr = ctx.GetAttr(name);
  if (!attr) {
    KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
    return false;
  }
  value = attr->GetString();
  return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name,
                         bool &value) {
  auto attr = ctx.GetAttr(name);
  if (!attr) {
    KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
    return false;
  }
  value = attr->GetBool();
  return true;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, void>::type
GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name, T &value) {
  auto attr = ctx.GetAttr(name);
  if (attr != nullptr) {
    value = static_cast<T>(attr->GetInt());
  }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name,
                            std::string &value) {
  auto attr = ctx.GetAttr(name);
  if (attr != nullptr) {
    value = attr->GetString();
  }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name,
                            bool &value) {
  auto attr = ctx.GetAttr(name);
  if (attr != nullptr) {
    value = attr->GetBool();
  }
}

// sparse mode
enum SparseMode {
    NO_MASK = 0, // 未传入attenmask，不做mask操作
    ALL_MASK,
    LEFT_UP_CAUSAL,        // 左上角点划分的三角部分
    RIGHT_DOWN_CAUSAL = 3, // 右下角点划分的下三角部分
    BAND = 4,
    PREFIX = 5,
    PREFIX_COMPRESS = 6,
    RIGHT_DOWN_CASUAL_BAND = 7,
    BAND_LEFT_UP_CASUAL = 8
};

static const int64_t L2_CACHE_SIZE = 128L;
static const int64_t NUM_1024 = 1024L;
static const int64_t NUM_256 = 256L;
static const int64_t NUM_128 = 128L;
static const int64_t NUM_64 = 64L;
static const int64_t D_TEMPLATE_SPLIT_SIZE = 64L;

// s1BasicBlock 
constexpr uint32_t SOUTER_FACTOR_SUB = 32;
constexpr uint32_t SOUTER_FACTOR_DEFAULT = 64;
// s2BasicBlock 
constexpr uint32_t SINNER_FACTOR_DEFAULT = 128;
constexpr uint32_t SINNER_FACTOR_DOUBLE = 256;

constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

// 分核功能模块输出：FA阶段的核间分核信息
struct SplitResult {
    uint32_t usedCoreNum { 0U };        // 使用的核数量

    std::vector<int32_t> s1out {};

    SplitResult(uint32_t aicNum, uint32_t aivNum) :
        usedCoreNum(aicNum),
        s1out(aicNum) {};
};

class FlashAttnMetadataCpuKernel : public CpuKernel {
public:
    FlashAttnMetadataCpuKernel() = default;
    ~FlashAttnMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool BalanceSchedule(SplitResult &splitRes);
    bool GenMetaData(SplitResult &splitRes);
    bool ParamsInit();

    void SetSplitCoreModeParam();
    void CalcS1S2BasicBlock();
    int64_t CalcTotalSize();
    void CalcThresholdForS2Size();
    bool IsUseSpliteCoreMode(SparseMode inputSparseMode);

    void GetPreNextTokensLeftUp(
        int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
    // void FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV,
    //     int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
    // int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV,
    //     uint32_t s1BasicBlock, uint32_t s2BasicBlock, int64_t preTokensLeftUp, int64_t nextTokensLeftUp, bool isAttenMaskUsed);
private:
    // input tensor
    Tensor *actualSeqLengths_ = nullptr;
    Tensor *actualSeqLengthsKV_ = nullptr;
    int32_t* actualSeqLengths;
    int32_t* actualSeqLengthsKV;
    // output tensor
    Tensor *metaData_ = nullptr;
    
    // input attr
    int32_t sparseMode_ = 0U;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 0;
    int32_t headDim_ = 0;
    int32_t ropeHeadDim_ = 0; // to add param
    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = 0;
    int32_t maxSeqlenKv_ = 0;
    int32_t preTokens_ = 2147483647; // to add param
    int32_t nextTokens_ = 2147483647; // to add param

    std::string socVersion_ = "ascend910B";
    int32_t aicCoreNum_ = 0;
    int32_t aivCoreNum_ = 0;

    int32_t maxActualseqQ = 0;
    int32_t maxActualseqKV = 0;

    // output attr
    int32_t coreNum = 0U;
    int32_t totalSize = 0U;
    int32_t firstFullLoadS1OuterIdx = -1;

    SplitCoreMode splitCoreMode = SplitCoreMode::SPLIT_S1OUT_CUBE;

    // params
    uint32_t CV_RATIO = 2U;
    int32_t thresholdS2Size = 0;
    int32_t s1OuterSize = 0;
    int32_t s1BasicBlock = 0;
    int32_t s2BasicBlock = 0;
    bool hasAttenMask = true;
private:
    enum class ParamId : uint32_t {
    // input
    actualSeqLengths = 0,
    actualSeqLengthsKV = 1,
    // output
    metaData = 0,
    };
};
} // namespace aicpu

#endif