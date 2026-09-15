/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_VALUE_TILING_LEGACY_H
#define FLASH_MLA_VALUE_TILING_LEGACY_H

#include "../flash_mla_value_tiling.h"
#include "../../../fused_infer_attention_score/op_host/fia_tiling_info.h"
#include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling_base.h"
#include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling_struct.h"
#include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling_context.h"

namespace FlashMla {
namespace Arch22 {
// Private value-dependent subset of the V4 tiler. Original method names are
// preserved for comparison with FIA/IFA; entry and methods share the companion .cpp.
class MlaLegacyValueState {
public:
    MlaStatus Compute(const MlaScheduleConfig &config, const MlaPlatformInfo &platform, MlaSeqView seqQ,
                      MlaSeqView seqKv, MlaValuePatch &patch);
    MlaStatus WorkspaceCapacity(const MlaScheduleConfig &config, const MlaPlatformInfo &platform, uint64_t &capacity);

private:
    void Prepare(const MlaScheduleConfig &config, const MlaPlatformInfo &platform,
                 optiling::IncreFlashAttentionContext &context);

    ge::graphStatus CheckTreeSparseMask();
    struct ValidityConfigFD {
        std::vector<int32_t> validBatchSizes;
        std::vector<int32_t> validQSeqSizes;
        int32_t numHeads;
        int32_t numKvHeads;
        int32_t headDim;
        int32_t headDimV;
        int32_t sparseMode;
        int64_t expectedActualSeqLength; // -1 表示范围 [4096, 5120]
    };

    void SetCoreNum();
    bool CheckCommonConditions(const ValidityConfigFD &config) const;
    bool CheckBatchAndQSeqSize(const std::vector<int32_t> &validBatchSizes,
                               const std::vector<int32_t> &validQSeqSizes) const;
    bool CheckHeadDimensions(int32_t numHeads, int32_t numKvHeads, int32_t headDim, int32_t headDimV) const;
    bool CheckQuantizationFlags(int32_t sparseMode) const;
    bool CheckActualSeqLengths(int64_t expectedActualSeqLength) const;
    bool IsValidFlag3B();
    bool IsValidFlag560B();
    bool IsValidFlag();
    void IsFdBalanceCase();
    bool IsBalanceSplitCore();
    ge::graphStatus QKVPreProcess4TND(const std::string layout);
    ge::graphStatus ZeroTensorProcess();
    ge::graphStatus ProcessActualSeqLen();
    ge::graphStatus CheckActualSeqLens();
    ge::graphStatus ParseActualSeqLens();
    ge::graphStatus Split();
    void GetActualSeqInfo(const int64_t *actualSeqKv, optiling::ActualSeqInfo &actualSeqInfo) const;
    void GetSeqTilingInfo(const int64_t *actualSeqKv, const optiling::ActualSeqInfo &actualSeqInfo,
                          optiling::SeqTilingInfo &seqTilingInfo) const;
    void FillBalancedSplitCoreInfo(const optiling::TilingIndexes &tilingIdx,
                                   optiling::BalancedSplitTilingInfo &tilingInfo);
    void EndSplitForCurrentCore(const optiling::TilingIndexes &tilingIdx, const optiling::SeqTilingInfo &seqTilingInfo,
                                uint32_t &currKvSplitPart, optiling::BalancedSplitTilingInfo &tilingInfo);
    void SplitBalancedForEachHeadFd(uint32_t bIdx, const optiling::SeqTilingInfo &seqTilingInfo,
                                    optiling::BalancedSplitTilingInfo &tilingInfo,
                                    std::vector<int64_t> &gS1SplitNumOfFdHead, uint32_t s1);
    void SplitBalancedForEachHead(uint32_t bIdx, const optiling::SeqTilingInfo &seqTilingInfo,
                                  optiling::BalancedSplitTilingInfo &tilingInfo);
    void SplitFDMLa(uint32_t tndFDCoreArrLen, std::vector<int64_t> &gS1SplitNumOfFdHead, uint32_t *s2SplitNumOfFdHead,
                    uint32_t aivCoreNum, optiling::SeqTilingInfo &seqTilingInfo);
    ge::graphStatus SplitBalancedFd();
    ge::graphStatus SplitBalanced();
    ge::graphStatus SplitUnbalanced();
    ge::graphStatus SplitBN();
    void GetEstimatedLoad(int64_t &estimatedLoad) const;
    std::vector<int64_t> InitSparseValidArray(const int64_t *actualLens) const;
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx) const;
    void InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                       const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue) const;
    void SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                           uint32_t *sparseStartIdx, int64_t splitFactorSize) const;
    ge::graphStatus SplitBN_V0();
    ge::graphStatus SplitBNS();
    ge::graphStatus CalcInnerSize(uint32_t seqSize);
    void SetDealBN2Num();
    bool CalcUbBmm();
    std::pair<uint32_t, uint32_t> GetPreLoadNumAndActCoreNum() const;
    void CalcWorkSpaceForBmmAll(const optiling::IfaWorkSpaceSizeParams &params, uint32_t preLoadNum,
                                uint32_t actCoreNum);
    ge::graphStatus CalcWorkSpace();
    void CalcFDWorkSpace(const uint32_t actCoreNum);
    void NormalCalcFDWorkSpace(const uint32_t actCoreNum);
    void MLACalcFDWorkSpace(const uint32_t actCoreNum);
    uint32_t CalcBalanceFDParamNums(const uint32_t actCoreNum) const;
    uint32_t CalcUnbalanceFDParamNums() const;
    void FillTilingSplitKVMla();
    void FillTilingCoreParamsMla();
    void FillTilingSingleCoreParamsMla();
    void FillTilingSingleCoreTensorSizeMla();
    optiling::IfaAmlaMode GetAmlaMode() const;
    ge::graphStatus CheckUbSpace();
    bool IsFlashDecode(uint32_t coreNum, optiling::IfaPerfMode perfMode);
    bool CheckCoreOkFlag(uint32_t coreNum, optiling::IfaPerfMode perfMode) const;
    bool DealSameSeqEachBatch() const;
    bool IsKvZeroBatchSplit(bool needUpdate, uint32_t lastValidBMoreIdx, uint32_t bSize,
                            const std::vector<uint32_t> &s1OuterNum, const std::vector<uint32_t> &s2OuterNum) const;
    uint32_t GetTypeSize(ge::DataType dtype) const
    {
        return ge::GetSizeByDataType(dtype);
    }

    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t nNumOfQInOneGroup_ = 1;
    uint32_t msdIterNum_ = 1;
    uint32_t antiquantMode_ = 0;

    uint32_t headDimRope_ = 64;
    uint32_t headDim_ = 0;
    uint32_t headDimV_ = 0;
    uint32_t seqSize_ = 0;
    uint32_t batchSize_ = 0;
    optiling::IfaLayout inputLayout_ = optiling::IfaLayout::BSH_BSND;
    optiling::IfaLayout inputKvLayout_ = optiling::IfaLayout::BSH_BSND;
    uint32_t sMax_ = 0;
    uint32_t tSeqSize_ = 1;      // TND格式T轴长度
    uint32_t qSeqSize_ = 1;      // 默认S1 = 1
    uint64_t qSeqSquareSum_ = 0; // 给Sparse9 TND下mask大小用
    uint32_t blockTypeSize_ = 0; // 计算中间量大小
    uint32_t kvSplitPart_ = 1;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    uint32_t cvRatio_ = 2; // 2表示CV1:2; 1表示CV 1:1
    optiling::IfaSocVersion socVersion_ = optiling::IfaSocVersion::SOC_ASCEND_910B;
    size_t libapiSize_ = 0;

    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    size_t qPreSizeMla_ = 0;

    bool attenMaskFlag_ = false;
    uint32_t attenMaskTypeSize_ = 0;

    bool antiQuantFlag_ = false;

    bool pageAttentionFlag_ = false;
    uint32_t maxBlockNumPerBatch_ = 0;

    bool batchContinuousFlag_ = true;
    std::vector<int64_t> kvListSeqLens_;

    bool actualSeqLenFlag_ = false;

    bool quantFlag_ = false;

    uint32_t actualLenQDims_ = 0;
    uint32_t actualLenDims_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameActualseq_ = true;
    bool isSameSeqAllKVTensor_ = true;
    bool hasZeroActualseq_ = false;
    uint32_t dealBN2Num_ = optiling::DEAL_BN2_NUM;

    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t headDimAlign_ = 0;
    uint32_t headDimVAlign_ = 0;

    bool splitKVFlag_ = false;
    uint32_t kvSplit_ = 0;

    optiling::IfaPerfMode perfMode_ = optiling::IfaPerfMode::NORMAL;
    optiling::IfaAmlaMode amlaMode_ = optiling::IfaAmlaMode::DISABLE_AMLA;
    size_t workspaceSize_ = 0;

    uint32_t usedCoreNum_ = 0;

    uint32_t startIdxEachCore_[optiling::MAX_CORE_NUM] = {};
    optiling::IncreFlashAttentionContext *ifaContext_ = nullptr;
    bool isWorkspace_ = false;

    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;
    uint32_t groupSplitSize_ = 0;
    uint32_t s1SplitSize_ = 0;
    uint32_t gOuter_ = 1;
    uint32_t s1Outer_ = 1;
    uint32_t sparseMode_ = 3;
    uint32_t gMax_ = 128;

    bool slidingFlag_ = false;

    bool ropeFlag_ = false;
    bool gqaMtpFlag_ = false;
    bool gqaKvNZFlag_ = false;

    bool isSysPrefixTiling_ = false;
    uint32_t batchSizeQ_ = 1;

    MlaLegacyTiling tilingDataMla_{};

    bool balanceModeFlag_ = false;
};

} // namespace Arch22
} // namespace FlashMla
#endif // FLASH_MLA_VALUE_TILING_LEGACY_H
