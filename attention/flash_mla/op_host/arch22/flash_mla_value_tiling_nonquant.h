/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_VALUE_TILING_NONQUANT_H
#define FLASH_MLA_VALUE_TILING_NONQUANT_H

#include "../flash_mla_value_tiling.h"
#include "flash_mla_split_core.h"
#include "../../../fused_infer_attention_score/op_host/fia_tiling_info.h"

namespace FlashMla {
namespace Arch22 {
// Private value-dependent subset of the V4 tiler. Original method names are
// preserved for comparison with FIA/IFA; entry and methods share the companion .cpp.
class MlaValueParser {
    friend class MlaNonQuantValueState;

private:
    ge::graphStatus GetMaxActualSeq(const gert::Tensor *actualSeqLensTensor, optiling::FiaLayout layout,
                                    int64_t &maxActualSeqLen);
    ge::graphStatus GetMaxBlockNumPerBatch();
    void ProcessKVActualSeqLoop(const int64_t *actualLenData, uint32_t loop);
    void ProcessQActualSeqLoop(const int64_t *actualLenQData, uint32_t loop);
    ge::graphStatus GetActualSeqInfo();
    void ComputeQMax(const int64_t *actualSeqQ, uint32_t b);

    const char *opName_ = nullptr;
    optiling::FIAParaInfo opParamInfo_;

    uint32_t bSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    optiling::KvStorageMode kvStorageMode_ = optiling::KvStorageMode::BATCH_CONTINUOUS;
    optiling::FiaQuantMode quantMode_ = optiling::FiaQuantMode::NO_QUANT;

    optiling::FiaLayout qLayout_ = optiling::FiaLayout::BSND;
    optiling::FiaLayout kvLayout_ = optiling::FiaLayout::BSND;

    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    uint32_t blockTableDim2_ = 0;

    std::vector<int32_t> qSize_ = {};
    std::vector<int32_t> kvSize_ = {};

    bool isSameActualseq_ = true;
    bool needInit_ = false;
    bool antiQuantFlag_ = false;
    int64_t maxActualseq_ = 0;
    bool isMaxWorkspace_ = false;

    bool isAccumQSeq_ = false;
    bool isAccumKVSeq_ = false;
    uint32_t actualLenQDims_ = 0;
    uint32_t actualLenDims_ = 0;
    std::vector<int64_t> kvListSeqLens_{};
};

class MlaNonQuantValueState {
public:
    MlaStatus Compute(const MlaScheduleConfig &config, const MlaPlatformInfo &platform, MlaSeqView seqQ,
                      MlaSeqView seqKv, MlaValuePatch &patch);
    MlaStatus WorkspaceCapacity(const MlaScheduleConfig &config, const MlaPlatformInfo &platform, uint64_t &capacity);

private:
    bool IsFlashDecode();
    bool DealSameSeqEachBatch() const;
    void EmptyTensorProcess() const;
    void InitParams();
    void CalcInnerSize(uint32_t seqSize);
    void CalcMBaseSize();
    void CreateSplitInput(BaseInfo &baseInfo, SplitParam &splitParam) const;
    void SetSplitOutput(const SplitResult &res);
    void Split();
    void CalcMmResSize();
    void CalcMaxMmResSize();
    void FillTilingWorkspaceParams();
    uint32_t CalcFlashDecodeParamNums(const uint32_t coreNum) const;
    uint64_t CalcNormalWorkspaceSize(uint32_t coreNum, int64_t mm1ResSize, int64_t mm2ResSize,
                                     uint32_t mBaseSize) const;
    uint64_t CalcFlashDecodeWorkspace(const uint32_t coreNum) const;
    void CalcWorkspaceSize();
    void CalcMaxWorkspaceSize();
    bool splitKVFlag_ = false;

    uint32_t coreNum_ = 0;
    optiling::FiaTemplateId perfMode_ = optiling::FiaTemplateId::GENERAL_GQA;
    uint32_t kvSplitPart_ = 1;
    int64_t mm1ResSize_ = 0;
    int64_t mm2ResSize_ = 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t kvSplit_ = 0;
    uint32_t usedCoreNum_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    uint32_t cvRatio_ = 2; // 2表示CV1:2; 1表示CV 1:1
    size_t libapiSize_ = 0;

    MlaNonQuantTiling tilingData_{};
    uint32_t numBlocks_{0};
    uint64_t workspaceSize_{0};

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 256;
    uint32_t mFdBaseSize_ = 8;

    optiling::FiaTilingInfo *fiaInfo_ = nullptr;
};

class MlaValueSeqCheck {
    friend class MlaNonQuantValueState;

private:
    const char *opName_ = "FlashMlaMetadata";

    ge::graphStatus GetActualSeqLenSize(const gert::Tensor *tensor, const optiling::FiaLayout &layout,
                                        const std::string &actualSeqLenName, const std::string &attrName,
                                        uint32_t &size);
    ge::graphStatus CheckFeatureActualSeqLensQData();
    ge::graphStatus CheckFeatureActualSeqLensKvData();
    optiling::FIAParaInfo opParamInfo_{};
    optiling::FiaLayout qLayout_ = optiling::FiaLayout::BSH;
    optiling::FiaLayout kvLayout_ = optiling::FiaLayout::BSH;
    uint32_t bSize_ = 0;
    uint32_t s1Size_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t kTSize_ = 0;
    int64_t s2Size_ = 0;
    uint32_t actualSeqLengthsQSize_ = 0;
    uint32_t actualSeqLengthsKvSize_ = 0;
    std::vector<int32_t> qSize_;
    std::vector<int32_t> kvSize_;
};
} // namespace Arch22
} // namespace FlashMla
#endif // FLASH_MLA_VALUE_TILING_NONQUANT_H
