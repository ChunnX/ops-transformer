/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file fa_tiling.h
 * \brief GQA non-quant tiling class for flash_attn_example.
 *        Ported from ops-transformer fa_tiling.h, adapted
 *        for AscendOps paradigm (using ContextParamsForTiling and at::Tensor
 *        instead of CANN gert::TilingContext and FaTilingInfo).
 */

#ifndef FA_TILING_H
#define FA_TILING_H

#include <cstdint>
#include <vector>
#include <torch/torch.h>
#include "fa_tiling_public.h"
#include "split_core.h"

namespace ascend_ops {
namespace fa_host {

constexpr uint32_t PRE_LOAD_NUM_GQA_ARCH35 = 3;

#define InOutLayoutType_BNSD_BNSD 0
#define InOutLayoutType_BSH_BSH 1
#define InOutLayoutType_TND_TND 2
#define InOutLayoutType_BNSD_BSND 3
#define InOutLayoutType_NTD_NTD 4

#define PSE_MODE_PSE_NONE_TYPE 9

#define NoQuantMode 31

#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BBH 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

#define GET_TPL_TILING_KEY(...) 0

struct FaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t config = 0;
    bool hasAttnMask = false;
    uint64_t kvLayoutType = 0;
    bool isCombine = false;
    bool isReconstructTemp = false;
};

struct FaPlatFormInfo {
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t cvRatio = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};

enum ScheduleMode {
    BATCH_MODE = 0,
};

class FaTiling {
public:
    FaTiling() {}
    ~FaTiling() = default;

    void DoTiling(ContextParamsForTiling &contextKeyParams);

    void SetPlatMemoryInfo(ContextParamsForTiling &contextKeyParams);
    void InitImplParam(ContextParamsForTiling &contextKeyParams);
    void AdjustSinnerAndSouter();
    void SplitPolicy(ContextParamsForTiling &contextKeyParams);
    void CreateSplitInput(split_core::BaseInfo &baseInfo, split_core::SplitParam &splitParam,
                          ContextParamsForTiling &contextKeyParams);
    void SetSplitOutput(const split_core::FAMetaData &splitRes);
    void FillTiling(ContextParamsForTiling &contextKeyParams);
    void ComputeTilingData(ContextParamsForTiling &contextKeyParams);
    void SetFATilingData(ContextParamsForTiling &contextKeyParams);
    void GenTilingKey(ContextParamsForTiling &contextKeyParams);
    void CalcWorkspaceSize();
    void CalcScheduleMode();
    void CalcNumBlocks(uint32_t coreNum, ContextParamsForTiling &contextKeyParams);
    void UpdateTilingKeyConfig();
    void UpdateTilingKeyLayout(ContextParamsForTiling &contextKeyParams);
    void UpdateTilingKeyPseMode(ContextParamsForTiling &contextKeyParams);
    void UpdateTilingKeyQuantMode();
    void UpdateTilingKeyHasRope();
    void UpdateTilingKeyInfo(ContextParamsForTiling &contextKeyParams);
    void PrintAllTilingData();

    optiling::FlashAttnTilingData tilingData_;
    FaTilingKeyInfo tilingKeyInfo_;
    FaPlatFormInfo platformInfo_;
    uint32_t sOuterFactor_ = SOUTER_64;
    uint32_t sInnerFactor_ = SINNER_128;
    bool isCombine_ = false;
    bool needInit_ = false;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    int32_t numBlocks_ = 0;

    ContextParamsForTiling *contextKeyParamsPtr = nullptr;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint32_t bSize_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    uint32_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    float softmaxScale_ = 0.0f;
    InputLayout inputLayout_ = InputLayout::BSH;
    int32_t sparseModeVal_ = 0;
    bool attnMaskFlag_ = false;
    int32_t preToken_;
    int32_t nextToken_;
};

} // namespace fa_host
} // namespace ascend_ops

#endif