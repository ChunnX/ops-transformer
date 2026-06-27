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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch35.h
 * \brief AICPU kernel implementation for A5 (arch35)
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_ARCH35_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_ARCH35_H

#include "../sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "log.h"
#include "status.h"
#include "cust_op/cust_cpu_utils.h"
#include "../../../sparse_lightning_indexer_kl_loss_grad/op_kernel/arch35/sparse_lightning_indexer_klloss_grad_metadata_arch35.h"

namespace aicpu {

constexpr int64_t FA_TOLERANCE_RATIO = 2;

enum BlockType : uint32_t {
    NORMAL_BLOCK = 0,
    TAIL_BLOCK,
    BLOCK_MAX_TYPE
};

enum class Arch35SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

enum class ValidSocVersion {
    ASCEND910B = 0,
    ASCEND950,
    RESERVED_VERSION = 99999
};

template<class T>
using Range = std::pair<T, T>;

template<class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template<typename T>
inline T Clip(T value, T minValue, T maxValue)
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

template <typename T>
inline T Arch35CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

struct FlashDecodeResult {
    uint32_t fdUsedVecNum { 0U };
    std::vector<uint32_t> fdBN2Idx {};
    std::vector<uint32_t> fdMIdx {};
    std::vector<uint32_t> fdWorkspaceIdx {};
    std::vector<uint32_t> fdS2SplitNum {};
    std::vector<uint32_t> fdMSize {};
    std::vector<uint32_t> fdIdx {};
    std::vector<uint32_t> fdMStart {};
    std::vector<uint32_t> fdMNum {};

    FlashDecodeResult(uint32_t aicNum, uint32_t aivNum) :
        fdBN2Idx(aicNum),
        fdMIdx(aicNum),
        fdWorkspaceIdx(aicNum),
        fdS2SplitNum(aicNum),
        fdMSize(aicNum),
        fdIdx(aivNum),
        fdMStart(aivNum),
        fdMNum(aivNum) {}
};

struct SplitResult {
    uint32_t usedCoreNum { 0U };
    std::vector<uint32_t> bN2End {};
    std::vector<uint32_t> gS1End {};
    std::vector<uint32_t> s2End {};
    std::vector<uint32_t> firstFdDataWorkspaceIdx {};
    int64_t maxCost { 0 };
    uint32_t numOfFdHead { 0U };
    uint32_t maxS2SplitNum { 0U };
    FlashDecodeResult fdRes { 0U, 0U };

    SplitResult(uint32_t aicNum, uint32_t aivNum) :
        bN2End(aicNum),
        gS1End(aicNum),
        s2End(aicNum),
        firstFdDataWorkspaceIdx(aicNum),
        fdRes(aicNum, aivNum) {};
};

struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum {};
    std::vector<uint32_t> s2BaseNum {};
    std::vector<uint32_t> s1GTailSize {};
    std::vector<uint32_t> s2TailSize {};
    bool isKvSeqAllZero { true };

    explicit SplitInfo(uint32_t batchSize) :
        s1GBaseNum(batchSize),
        s2BaseNum(batchSize),
        s1GTailSize(batchSize),
        s2TailSize(batchSize) {}
};

struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch {};
    std::vector<uint32_t> bN2BlockOfEachBatch {};
    std::vector<int64_t> bN2LastBlockCostOfEachBatch {};
    uint32_t totalBlockNum { 0U };
    int64_t totalCost { 0 };
    uint64_t maxS1GCost { 0 };

    explicit CostInfo(uint32_t batchSize) :
        bN2CostOfEachBatch(batchSize),
        bN2BlockOfEachBatch(batchSize),
        bN2LastBlockCostOfEachBatch(batchSize) {}
};

struct SplitContext {
    SplitInfo splitInfo { 0U };
    CostInfo costInfo { 0U };

    explicit SplitContext(uint32_t batchSize) :
        splitInfo(batchSize),
        costInfo(batchSize) {}
};

struct BatchCache {
    uint32_t bIdx { 0U };
    uint32_t s1Size { 0U };
    uint32_t s2Size { 0U };
    int64_t preTokenLeftUp { 0 };
    int64_t nextTokenLeftUp { 0 };
    BlockCost<int64_t> typeCost {};
};

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

struct CoreCache {
    int64_t costLimit { 0 };
    int64_t cost { 0 };
    uint32_t block { 0U };
};

struct AssignContext {
    uint32_t curBIdx { 0U };
    uint32_t curBN2Idx { 0U };
    uint32_t curS1GIdx { 0U };
    uint32_t curS2Idx { 0U };
    uint32_t curCoreIdx { 0U };
    int64_t unassignedCost { 0 };
    uint32_t curKvSplitPart { 1U };
    uint32_t preFdDataNum { 0U };

    int64_t bN2Cost { 0 };
    uint32_t bN2Block { 0U };
    bool isFinished { false };
    BatchCache batchCache {};
    S1GCache s1GCache {};
    CoreCache coreCache {};
};

class SparseLightningIndexerKLLossGradMetadataCpuKernelArch35 : public CpuKernel {
public:
    SparseLightningIndexerKLLossGradMetadataCpuKernelArch35() = default;
    ~SparseLightningIndexerKLLossGradMetadataCpuKernelArch35() override = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    int32_t GetQueryBatchSize();
    bool ParamsInit();
    bool ParamsCheck();
    bool BalanceSchedule(SplitResult &splitRes);
    bool GenMetadata(SplitResult &splitRes);
    ValidSocVersion ProcessSocVersion();

    uint32_t GetS1SeqSize(uint32_t bIdx);
    uint32_t GetS2SeqSize(uint32_t bIdx);
    int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    Range<int64_t> CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache);
    int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
    BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
        uint32_t s2TailSize);

    void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
    void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache,
        S1GCache &s1GCache);

    void CalcSplitInfo(SplitContext &splitContext);
    void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
    void CalcCostInfo(SplitContext &splitContext);

    void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext);
    void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignBlockToCore(uint32_t coreNum, const SplitContext &splitContext,
        AssignContext &assignContext, SplitResult &result);

    bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes);
    void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result);

    void SplitFD(SplitResult &splitRes);
    void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, SplitResult &result);

    CpuKernelContext* context_ = nullptr;
    Tensor *cuSeqlensQ_ = nullptr;
    Tensor *cuSeqlensK_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedK_ = nullptr;
    Tensor *cmpResidualK_ = nullptr;
    Tensor *metadata_ = nullptr;

    std::string socVersion_ = "";
    bool supportFd_ = false;
    int32_t cmpRatio_ = 4;
    uint32_t aicCoreNum_ = 24U;
    uint32_t aivCoreNum_ = 48U;
    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = 0;
    int32_t maxSeqlenK_ = 0;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsK_ = 0;
    int32_t headDim_ = 0;
    int32_t queryQuantMode_ = 0;
    int32_t keyQuantMode_ = 0;
    int32_t topk_ = 0;
    std::string layoutQ_ = "BSND";
    std::string layoutK_ = "BSND";
    int32_t maskMode_ = 0;
    uint32_t attentionMode_ = 0;
    ValidSocVersion validSocVersion_ = ValidSocVersion::ASCEND910B;

    int64_t  preToken_ = INT64_MAX;
    int64_t  nextToken_ = INT64_MAX;
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = 256;
    uint32_t s2BaseSize_ = 0;
    bool isS1G_ = true;
    uint32_t totalNum = 0;

    enum class ParamId : uint32_t {
        actSeqLenQ = 0,
        actSeqLenK = 1,
        seqUsedQ = 2,
        seqUsedK = 3,
        cmpResidualK = 4,
        metadata = 0,
    };
};

inline uint32_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes {aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::Prepare(CpuKernelContext &ctx)
{
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
    cuSeqlensK_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenK));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedQ));
    sequsedK_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedK));
    cmpResidualK_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpResidualK));
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs = GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                         GetAttrValue(ctx, "aiv_core_num", aivCoreNum_) &&
                         GetAttrValue(ctx, "soc_version", socVersion_) &&
                         GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                         GetAttrValue(ctx, "num_heads_k", numHeadsK_) &&
                         GetAttrValue(ctx, "head_dim", headDim_);
    if (!requiredAttrs) {
        return false;
    }

    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_k", maxSeqlenK_);
    GetAttrValueOpt(ctx, "topk", topk_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_k", layoutK_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);

    return (ParamsCheck() && ParamsInit());
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::ParamsCheck()
{
    // 校验输出 metadata 是否为空
    if (metadata_ == nullptr) {
        KERNEL_LOG_ERROR("Output metadata is nullptr");
        return false;
    } else if (metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("Output metadata data is nullptr");
        return false;
    }
    int32_t batchSize = GetQueryBatchSize();
    // 校验 cu_seqlens_q 元素
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *cuSeqlensQPtr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
            for (int i = 0; i < batchSize + 1; i++) {
                // 校验 cu_seqlens_q 元素非负
                if (cuSeqlensQPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in cu_seqlens_q should be >= 0, but got cu_seqlens_q[%d] = %d",
                        i, cuSeqlensQPtr[i]);
                    return false;
                }
                // 校验 cu_seqlens_q 元素递增
                if (i > 0 && cuSeqlensQPtr[i - 1] > cuSeqlensQPtr[i]) {
                    KERNEL_LOG_ERROR("The elements in cu_seqlens_q must be in ascending order, "
                        "but got cu_seqlens_q[%d] = %d, cu_seqlens_q[%d] = %d",
                        i - 1, cuSeqlensQPtr[i - 1], i, cuSeqlensQPtr[i]);
                    return false;
                }
            }
        }
    }
    // 校验 cu_seqlens_k 元素
    if (layoutK_ == "TND") {
        if (cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) {
            const int32_t *cuSeqlensKPtr = static_cast<const int32_t*>(cuSeqlensK_->GetData());
            for (int i = 0; i < batchSize + 1; i++) {
                // 校验 cu_seqlens_k 元素非负
                if (cuSeqlensKPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in cu_seqlens_k should be >= 0, but got cu_seqlens_k[%d] = %d",
                        i, cuSeqlensKPtr[i]);
                    return false;
                }
                // 校验 cu_seqlens_k 元素递增
                if (i > 0 && cuSeqlensKPtr[i - 1] > cuSeqlensKPtr[i]) {
                    KERNEL_LOG_ERROR("The elements in cu_seqlens_k must be in ascending order, "
                        "but got cu_seqlens_k[%d] = %d, cu_seqlens_k[%d] = %d",
                        i - 1, cuSeqlensKPtr[i - 1], i, cuSeqlensKPtr[i]);
                    return false;
                }
            }
        }
    }
    // 校验 seqused_q 元素非负
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *sequsedQPtr = static_cast<const int32_t*>(sequsedQ_->GetData());
        for (int i = 0; i < batchSize; i++) {
            if (sequsedQPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_q should be >= 0, but got seqused_q[%d] = %d",
                    i, sequsedQPtr[i]);
                return false;
            }
        }
    }
    // 校验 seqused_k 元素非负
    if (sequsedK_ != nullptr && sequsedK_->GetData() != nullptr) {
        const int32_t *sequsedKPtr = static_cast<const int32_t*>(sequsedK_->GetData());
        for (int i = 0; i < batchSize; i++) {
            if (sequsedKPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_k should be >= 0, but got seqused_k[%d] = %d",
                    i, sequsedKPtr[i]);
                return false;
            }
        }
    }
    // 校验 cmp_residual_k 元素非负
    if (cmpResidualK_ != nullptr && cmpResidualK_->GetData() != nullptr) {
        const int32_t *cmpResidualKPtr = static_cast<const int32_t*>(cmpResidualK_->GetData());
        for (int i = 0; i < batchSize; i++) {
            if (cmpResidualKPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in cmp_residual_k should be >= 0, but got cmp_residual_k[%d] = %d",
                    i, cmpResidualKPtr[i]);
                return false;
            }
        }
    }
    return true;
}

inline int32_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::GetQueryBatchSize()
{
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        if (sequsedQ_->GetTensorShape() != nullptr) {
            return sequsedQ_->GetTensorShape()->GetDimSize(0);
        }
    }
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            if (cuSeqlensQ_->GetTensorShape() != nullptr) {
                return cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
            }
        }
    }
    return batchSize_;
}

inline ValidSocVersion SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    } else {
        return ValidSocVersion::ASCEND910B;
    }

    return ValidSocVersion::RESERVED_VERSION;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    auto mode = static_cast<Arch35SparseMode>(maskMode_);
    if (mode == Arch35SparseMode::RIGHT_DOWN_CAUSAL) {
        attentionMode_ = 1;
        preToken_ = INT64_MAX;
    } else if (mode == Arch35SparseMode::DEFAULT_MASK) {
        attentionMode_ = 0;
    } else if (mode == Arch35SparseMode::BAND) {
        attentionMode_ = 1;
    }
    groupSize_ = numHeadsQ_ / numHeadsK_;
    validSocVersion_ = ProcessSocVersion();
    if (validSocVersion_ == ValidSocVersion::ASCEND910B) {
        s2BaseSize_ = 2048U;
    } else if (validSocVersion_ == ValidSocVersion::ASCEND950) {
        s2BaseSize_ = 128U;
    } else {
        s2BaseSize_ = 128U;
    }
    return true;
}

inline uint32_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::GetS1SeqSize(uint32_t bIdx)
{
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t*>(sequsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *s1Ptr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    return static_cast<uint32_t>(maxSeqlenQ_);
}

inline uint32_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::GetS2SeqSize(uint32_t bIdx)
{
    if (sequsedK_ != nullptr && sequsedK_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t*>(sequsedK_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    if (layoutK_ == "TND") {
        if (cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t*>(cuSeqlensK_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    return static_cast<uint32_t>(maxSeqlenK_);
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcSplitInfo(SplitContext &splitContext)
{
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        uint32_t cmpS2Size = GetS2SeqSize(bIdx);
        totalNum += s1Size;
        splitInfo.s1GBaseNum[bIdx] = (s1Size * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (s1Size * groupSize_) % mBaseSize_;
        splitInfo.s2BaseNum[bIdx] = (cmpS2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        splitInfo.s2TailSize[bIdx] = cmpS2Size % s2BaseSize_;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
    return;
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcPreTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<Arch35SparseMode>(maskMode_);
    if (mode == Arch35SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcNextTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<Arch35SparseMode>(maskMode_);
    switch (mode) {
        case Arch35SparseMode::DEFAULT_MASK:
        case Arch35SparseMode::ALL_MASK:
        case Arch35SparseMode::LEFT_UP_CAUSAL:
            return nextToken_;
        case Arch35SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case Arch35SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken_;
        default:
            return nextToken_;
    }
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcCost(
    uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignBasicM = basicM / groupSize_;
    uint32_t alignBasicS2 = 1;
    if (validSocVersion_ == ValidSocVersion::ASCEND910B) {
        if (basicS2 <= 1024) {
            alignBasicS2 = (1536 + basicS2) / 2;
        } else {
            alignBasicS2 = (2560 + basicS2) / 2;
        }
        return static_cast<int64_t>(alignBasicM * alignBasicS2);
    } else {
        uint32_t alignCoefM = 16U;
        uint32_t alignCoefS2 = 64U;
        alignBasicM = (basicM + alignCoefM - 1U) >> 4U;
        alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U;
        return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2);
    }
}

inline BlockCost<int64_t> SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcCostTable(uint32_t s1NormalSize,
    uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

inline Range<int64_t> SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcS2TokenRange(
        uint32_t s1GIdx, const BatchCache &batchCache)
{
    if (!attentionMode_) {
        return std::make_pair(0, static_cast<int64_t>(batchCache.s2Size));
    }
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
        static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) - 1;
    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (isS1G_) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
        s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    return std::make_pair(s2FirstToken, s2LastToken);
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcBatchCache(
    uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.typeCost = CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[bIdx],
        splitInfo.s2TailSize[bIdx]);
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcS1GCache(uint32_t s1GIdx,
    const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0 || splitInfo.s2BaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    auto s2TokenRange = CalcS2TokenRange(s1GIdx, batchCache);
    int64_t s2FirstToken = s2TokenRange.first;
    int64_t s2LastToken = s2TokenRange.second;
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));

    uint32_t s2CmpLength = (s2LastToken - s2FirstToken + 1) / cmpRatio_;
    if (s2CmpLength == 0) {
        s1GCache.s2Start = 0U;
        s1GCache.s2End = 0U;
    } else {
        s1GCache.s2Start = 0U;
        s1GCache.s2End = (s2CmpLength + s2BaseSize_ - 1) / s2BaseSize_;
    }

    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    uint32_t s2TailSize = s2CmpLength % s2BaseSize_;

    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num = s2TailSize != 0 ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;

    BlockCost<int64_t> typeCost = CalcCostTable(mBaseSize_, s2BaseSize_,
                                                splitInfo.s1GTailSize[batchCache.bIdx], s2TailSize);

    if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) &&
               splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost = typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                           typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? typeCost[TAIL_BLOCK][TAIL_BLOCK] :
                                    typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost = typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                           typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? typeCost[NORMAL_BLOCK][TAIL_BLOCK] :
                                    typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcBatchCost(
    uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx) == 0U || GetS2SeqSize(bIdx) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        if (s1GCache.s1GCost > costInfo.maxS1GCost) {
            costInfo.maxS1GCost = s1GCache.s1GCost;
        }

        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcCostInfo(SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * numHeadsK_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * numHeadsK_;
    }
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::UpdateCursor(const SplitContext &splitContext,
    AssignContext &assignContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) {
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    if (assignContext.curBN2Idx == batchSize_ * numHeadsK_) {
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / numHeadsK_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / numHeadsK_;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::AssignByBatch(const SplitContext &splitContext,
    AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 || IsWithinTolerance(assignContext.coreCache.costLimit,
        costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        if (assignContext.curBN2Idx == batchSize_ * numHeadsK_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        if (assignContext.curBN2Idx / numHeadsK_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / numHeadsK_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::AssignByRow(const SplitContext &splitContext,
    AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
        assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                assignContext.bN2Cost - assignContext.s1GCache.s1GCost : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                 assignContext.bN2Block - assignContext.s1GCache.s1GBlock : 0U;
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::AssignByBlock(const SplitContext &splitContext,
    AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
            assignContext.coreCache.cost + curCost)) {
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::ForceAssign(const SplitContext &splitContext,
    AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::IsNeedRecordFDInfo(
    const AssignContext &assignContext, const SplitResult &splitRes)
{
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::RecordFDInfo(const SplitContext &splitContext,
    const AssignContext &assignContext, SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / numHeadsK_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);

    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                            (s1Size * groupSize_ - splitS1GIdx * mBaseSize_) : mBaseSize_;
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    result.fdRes.fdBN2Idx[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdMIdx[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.preFdDataNum;
    result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize / groupSize_;
    result.numOfFdHead++;
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::AssignBlockToCore(uint32_t coreNum,
    const SplitContext &splitContext, AssignContext &assignContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] = assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
    assignContext.coreCache = {};
    assignContext.coreCache.costLimit = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);
    if (!supportFd_) {
        assignContext.coreCache.costLimit = costInfo.maxS1GCost > assignContext.coreCache.costLimit ?
            costInfo.maxS1GCost : assignContext.coreCache.costLimit;
    }
    AssignByBatch(splitContext, assignContext);
    AssignByRow(splitContext, assignContext);
    if (supportFd_) {
        AssignByBlock(splitContext, assignContext);
        if (assignContext.coreCache.block == 0) {
            ForceAssign(splitContext, assignContext);
        }
    }
    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
    result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
    assignContext.unassignedCost -= assignContext.coreCache.cost;
    if (supportFd_ && IsNeedRecordFDInfo(assignContext, result)) {
        RecordFDInfo(splitContext, assignContext, result);
        assignContext.preFdDataNum += assignContext.curKvSplitPart;
        assignContext.curKvSplitPart = 1U;
    }
    if (supportFd_ && assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
        assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
        assignContext.curKvSplitPart++;
    }
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::CalcSplitPlan(uint32_t coreNum,
    int64_t costLimit, const SplitContext &splitContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    if (coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;
    AssignContext assignContext {};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlockToCore(coreNum, splitContext, assignContext, result);
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::SplitFD(SplitResult &splitRes)
{
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_;
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        uint32_t curAveMSize = (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
        curFDVectorNum = (splitRes.fdRes.fdMSize[i] + curAveMSize - 1U) / curAveMSize;
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::BalanceSchedule(SplitResult &splitRes)
{
    SplitContext splitContext(batchSize_);
    CalcSplitInfo(splitContext);
    return true;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch35::GenMetadata(SplitResult &splitRes)
{
    optiling::detail::SlikgMetadata* gradMetadataPtr = (optiling::detail::SlikgMetadata*)metadata_->GetData();
    uint32_t formerCoreProcessNum = Arch35CeilDiv(totalNum, aicCoreNum_);
    uint32_t remainCoreProcessNum = formerCoreProcessNum - 1;
    uint32_t remainCoreNum = formerCoreProcessNum * aicCoreNum_ - totalNum;
    uint32_t usedCoreNum = totalNum < aicCoreNum_ ? totalNum : aicCoreNum_;

    gradMetadataPtr->gradMetadata[optiling::TOTAL_NUM] = totalNum;
    gradMetadataPtr->gradMetadata[optiling::FORMER_CORE_PROCESS_NUM] = formerCoreProcessNum;
    gradMetadataPtr->gradMetadata[optiling::REMAIN_CORE_PROCESS_NUM] = remainCoreProcessNum;
    gradMetadataPtr->gradMetadata[optiling::REMAIN_CORE_NUM] = remainCoreNum;
    gradMetadataPtr->gradMetadata[optiling::USED_CORE_NUM] = usedCoreNum;
    return true;
}

} // namespace aicpu

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_ARCH35_H
