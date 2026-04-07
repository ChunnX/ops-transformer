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
#include <limits>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "flash_attn_metadata.h"

using namespace optiling;
using namespace std;

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
inline typename std::enable_if<std::is_integral_v<T>, bool>::type GetAttrValue(CpuKernelContext &ctx,
                                                                               const std::string &name, T &value)
{
    auto attr = ctx.GetAttr(name);
    if (!attr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = static_cast<T>(attr->GetInt());
    return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name, std::string &value)
{
    auto attr = ctx.GetAttr(name);
    if (!attr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = attr->GetString();
    return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name, bool &value)
{
    auto attr = ctx.GetAttr(name);
    if (!attr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = attr->GetBool();
    return true;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, void>::type GetAttrValueOpt(CpuKernelContext &ctx,
                                                                                  const std::string &name, T &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = static_cast<T>(attr->GetInt());
    }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name, std::string &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = attr->GetString();
    }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name, bool &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = attr->GetBool();
    }
}
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t FD_TOLERANCE_RATIO = 2U;
constexpr int64_t V0_COST = 250;

enum BlockType : uint32_t {
    NORMAL_BLOCK = 0,
    TAIL_BLOCK,
    BLOCK_MAX_TYPE
};

// mask mode
enum MaskMode {
    DEFAULT_MASK = 0, // 未传入attenmask，不做mask操作
    ALL_MASK,
    LEFT_UP_CAUSAL,        // 左上角点划分的三角部分
    RIGHT_DOWN_CAUSAL = 3, // 右下角点划分的下三角部分
    BAND = 4,
    SPARSE_BUTT = 5,
    PREFIX_COMPRESS = 6,
    RIGHT_DOWN_CASUAL_BAND = 7,
    BAND_LEFT_UP_CASUAL = 8,
    TREE = 9,
};

template<class T>
using Range = std::pair<T, T>;

template<class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template<typename T>
T Clip(T value, T minValue, T maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

template<typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

// 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
struct FlashDecodeResult {
    uint32_t usedVecNum { 0U };                 // 归约过程中使用的AIV数量
    // 1、归约任务的索引信息
    std::vector<uint32_t> bN2Idx {};            // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> mIdx {};              // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> worksapceIdx {};      // 每个归约任务在workspace中的存放位置
    std::vector<uint32_t> s2SplitNum {};        // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    std::vector<uint32_t> mSize {};             // 每个归约任务的M轴大小，脚标为归约任务的序号
    // 2、FD kernel阶段，归约任务的分核信息
    std::vector<uint32_t> taskIdx {};           // 每个AIV处理的归约任务的对应ID
    std::vector<uint32_t> mStart {};            // 每个AIV处理的归约任务的M轴相对起点
    std::vector<uint32_t> mLen {};              // 每个AIV处理的归约任务的M轴行数

    FlashDecodeResult(uint32_t aicNum, uint32_t aivNum) :
        bN2Idx(aicNum),
        mIdx(aicNum),
        worksapceIdx(aicNum),
        s2SplitNum(aicNum),
        mSize(aicNum),
        taskIdx(aivNum),
        mStart(aivNum),
        mLen(aivNum) {}
};

// 分核功能模块输出：FA阶段的核间分核信息
struct SplitResult {
    uint32_t usedCoreNum { 0U };                            // 使用的核数量
    std::vector<uint32_t> bN2End {};                        // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End {};                        // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End {};                         // 每个核处理数据的S2结束点
    std::vector<uint32_t> firstFdDataWorkspaceIdx {};       // 每个核处理数据的S2结束点
    int64_t maxCost { 0 };                                  // 慢核开销
    uint32_t numOfFdHead { 0U };                            // 归约任务数量
    uint32_t maxS2SplitNum { 0U };                          // 单个归约任务最大分核数量
    FlashDecodeResult fdRes { 0U, 0U };                     // FD信息

    SplitResult(uint32_t aicNum, uint32_t aivNum) :
        bN2End(aicNum),
        gS1End(aicNum),
        s2End(aicNum),
        firstFdDataWorkspaceIdx(aicNum),
        fdRes(aicNum, aivNum) {};
};

// 分核功能模块内部使用：记录切分信息
struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum {};                   // S1G方向，切了多少个基本块
    std::vector<uint32_t> s2BaseNum {};                    // S2方向，切了多少个基本块
    std::vector<uint32_t> s1GTailSize {};                  // S1G方向，尾块size
    std::vector<uint32_t> s2TailSize {};                   // S2方向，尾块size
    bool isKvSeqAllZero { true };

    explicit SplitInfo(uint32_t batchSize) :
        s1GBaseNum(batchSize),
        s2BaseNum(batchSize),
        s1GTailSize(batchSize),
        s2TailSize(batchSize) {}
};

// 分核功能模块内部使用：记录batch的开销信息
struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch {};             // 整个batch的开销
    std::vector<uint32_t> bN2BlockOfEachBatch {};           // 整个batch的开销
    std::vector<int64_t> bN2LastBlockCostOfEachBatch {};    // batch最后一块的开销
    uint32_t totalBlockNum { 0U };
    int64_t totalCost { 0 };

    explicit CostInfo(uint32_t batchSize) :
        bN2CostOfEachBatch(batchSize),
        bN2BlockOfEachBatch(batchSize),
        bN2LastBlockCostOfEachBatch(batchSize) {}
};

// 分核功能模块内部使用：分核过程中，case基本信息的上下文信息，组合以减少接口传参数量
struct SplitContext {
    SplitInfo splitInfo { 0U };
    CostInfo costInfo { 0U };

    explicit SplitContext(uint32_t batchSize) :
        splitInfo(batchSize),
        costInfo(batchSize) {}
};

// 分核功能模块内部使用：记录batch相关的临时信息
struct BatchCache {
    uint32_t bIdx { 0U };
    uint32_t s1Size { 0U };
    uint32_t s2Size { 0U };
    int64_t preTokenLeftUp { 0 };
    int64_t nextTokenLeftUp { 0 };
    BlockCost<int64_t> typeCost {};
};

// 分核功能模块内部使用：记录当前行（S1G）的临时信息
struct S1GCache {
    uint32_t bIdx { 0U };
    uint32_t s1GIdx { 0U };
    uint32_t s2Start { 0U };
    uint32_t s2End { 0U };
    int64_t s1GCost { 0 };
    int64_t s1GLastBlockCost { 0 };
    uint32_t s1GBlock { 0U };
    int64_t s1GNormalBlockCost { 0 };
};

// 分核功能模块内部使用：记录分配过程中，当前核的负载信息
struct CoreCache {
    int64_t costLimit { 0 };  // 负载上限
    int64_t cost { 0 };       // 已分配负载
    uint32_t block { 0U };      // 已分配块数
};

// 分核功能模块内部使用：记录分配过程中的上下文信息
struct AssignContext {
    uint32_t curBIdx { 0U };
    uint32_t curBN2Idx { 0U };
    uint32_t curS1GIdx { 0U };
    uint32_t curS2Idx { 0U };
    uint32_t curCoreIdx { 0U };
    int64_t unassignedCost { 0 };
    uint32_t usedCoreNum { 0U };
    uint32_t curKvSplitPart { 1U };
    uint32_t preFdDataNum { 0U };

    int64_t bN2Cost { 0 };
    uint32_t bN2Block { 0U };
    bool isFinished { false };
    BatchCache batchCache {};
    S1GCache s1GCache {};
    CoreCache coreCache {};
};

// static const int64_t L2_CACHE_SIZE = 128L;
static const int64_t NUM_8192 = 8192L;
static const int64_t NUM_4096 = 4096L;
static const int64_t NUM_2048 = 2048L;
static const int64_t NUM_1024 = 1024L;
static const int64_t NUM_512 = 512L;
static const int64_t NUM_256 = 256L;
static const int64_t NUM_128 = 128L;
static const int64_t NUM_64 = 64L;
static const int64_t NUM_32 = 32L;
// static const int64_t D_TEMPLATE_SPLIT_SIZE = 64L;

// // s1BasicBlock
// constexpr uint32_t SOUTER_FACTOR_SUB = 32;
// constexpr uint32_t SOUTER_FACTOR_DEFAULT = 64;
// // s2BasicBlock
// constexpr uint32_t SINNER_FACTOR_DEFAULT = 128;
// constexpr uint32_t SINNER_FACTOR_DOUBLE = 256;

constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

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

    // util
    uint32_t GetS1SeqSize(uint32_t bIdx);
    uint32_t GetS2SeqSize(uint32_t bIdx);
    uint32_t GetSparseSeqSize(uint32_t bIdx);
    int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    Range<uint32_t> CalcS2Range(uint32_t s1GIdx,const BatchCache &batchCache);
    int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
    BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
        uint32_t s2TailSize);

    // cache calculation
    void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
    void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache);
    void CopyTmpResult(SplitResult &tmpRes, SplitResult &splitRes);
    void ClearTmpResult(SplitResult &tmpRes);

    // preprocess
    void CalcSplitInfo(SplitContext &splitContext);
    void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
    void CalcCostInfo(SplitContext &splitContext);

    // assign
    void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext);
    void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);

    // FD
    bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes);
    void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result);

    // main
    void SplitFD(SplitResult &result);
    void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, SplitResult &result);
    void SplitCore();
    void RollBackCursor(const SplitContext &splitContext, const CostInfo &costInfo, SplitResult &splitRes);
private:
    CpuKernelContext* context_ = nullptr;
    // input tensor
    Tensor *cuSeqlensQ_ = nullptr;
    Tensor *cuSeqlensKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedKv_ = nullptr;
    vector<int64_t> actualSeqLengths = {};
    vector<int64_t> actualSeqLengthsKV = {};
    // output tensor
    Tensor *metaData_ = nullptr;

    // input attr
    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = -1;
    int32_t maxSeqlenKv_ = -1;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 0;
    int32_t headDim_ = 0;
    int32_t maskMode_ = 1;
    int32_t winLeft_ = -1;
    int32_t winRight_ = -1;
    std::string layoutQ_ = "BSND";
    std::string layoutKv_ = "BSND";
    std::string layoutOut_ = "BSND";
    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36U;
    int32_t aivCoreNum_ = 72U;

    // params
    bool hasAttenMask = true;

    // SplitParams
    uint32_t coreNum_ = 36U; // new
    int64_t preToken_ = 0; // new
    int64_t nextToken_ = 0; // new
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = 0;
    uint32_t s2BaseSize_ = 0;
    uint32_t gS1BaseSizeOfFd_ = 0;
    bool isS1G_ = true;

private:
    enum class ParamId : uint32_t {
        // input
        cuSeqlensQ = 0,
        cuSeqlensKv = 1,
        sequsedQ = 2,
        sequsedKv = 3,
        // output
        metaData = 0,
    };
};
} // namespace aicpu

#endif