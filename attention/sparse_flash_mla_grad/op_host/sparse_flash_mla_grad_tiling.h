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
 * \file sparse_flash_mla_grad_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "sparse_flash_mla_grad_tiling_common.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SparseFlashMlaGradSplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, singleM);
TILING_DATA_FIELD_DEF(uint32_t, singleN);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseM);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseN);
TILING_DATA_FIELD_DEF(int64_t, s1BasicSize);
TILING_DATA_FIELD_DEF(int64_t, maxGatherSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGradSplitCoreParamsOp, SparseFlashMlaGradSplitCoreParams)

BEGIN_TILING_DATA_DEF(SmlagPostParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, postUbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, qPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, qPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, qPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, oriKvPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, oriKvPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, oriKvPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, oriKvPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, cmpKvPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, cmpKvPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, cmpKvPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, cmpKvPostTailNum);
TILING_DATA_FIELD_DEF(uint64_t, dqWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dkWorkSpaceOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SmlagPostParamsOp, SmlagPostParams)

BEGIN_TILING_DATA_DEF(SparseFlashMlaGradBasicParams)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreProcessNNum);
TILING_DATA_FIELD_DEF(uint32_t, remainCoreProcessNNum);
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, N2);
TILING_DATA_FIELD_DEF(int64_t, S1);
TILING_DATA_FIELD_DEF(int64_t, S2);
TILING_DATA_FIELD_DEF(int64_t, S3);
TILING_DATA_FIELD_DEF(int64_t, G);
TILING_DATA_FIELD_DEF(int64_t, D);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, selectedKWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedVWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm12WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, castUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, additionalWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockSize);
TILING_DATA_FIELD_DEF(int64_t, cmpRatio);
TILING_DATA_FIELD_DEF(int64_t, oriWinLeft);
TILING_DATA_FIELD_DEF(int64_t, oriWinRight);
TILING_DATA_FIELD_DEF(uint32_t, deterministic);        // 0=非确定, 1=确定
TILING_DATA_FIELD_DEF(uint64_t, dsinksDetWsOffset);    // d_sinks 保序 workspace 偏移(B, user相对)
TILING_DATA_FIELD_DEF(uint64_t, cmpL1DetWsOffset);     // cmp_softmax_l1 保序 workspace 偏移(B, user相对)
// dKV 确定性（保序累加）：
//  part(1) SCFA cmp：无新增 workspace，复用 mm4Res/mm5Res，每轮 S1 后 SyncAll + 按核序 scatterAdd
//  part(2) ori & CFA cmp：每核每 singleN 私有区 dkvDetWs[aic][DB][singleN*(Dk+Dv)]，每 singleN 后 SyncAll + 按核序累加
TILING_DATA_FIELD_DEF(uint64_t, dkvDetWsOffset);       // part(2) 每核私有区偏移(B, user相对)
TILING_DATA_FIELD_DEF(uint64_t, dkvDetPerCoreStride);  // part(2) 每核步长(B) = DB * singleN * (Dk+Dv) * 4
TILING_DATA_FIELD_DEF(uint32_t, dkvDetMaxRoundS1);     // part(1) 全核统一 S1 轮数上界(= formerCoreProcessNNum)
TILING_DATA_FIELD_DEF(uint32_t, dkvDetMaxRoundSingleN);// part(2) SWA/CFA 全核统一 singleN(task) 轮数上界
TILING_DATA_FIELD_DEF(uint32_t, dkvDetMaxTasksPerS1Round); // SCFA：每 S1 轮内 task 数上界(用于 ori 保序 SyncAll 对齐)
TILING_DATA_FIELD_DEF(uint64_t, dkvDetScatterMetaOffset); // part(1) 每核每轮 scatter 描述符元数据偏移(B, user相对)
TILING_DATA_FIELD_DEF(uint64_t, dkvDetOriMetaOffset);     // part(2) 每核每 singleN 轮 dkv 描述符元数据偏移(B, user相对)
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGradBasicParamsOp, SparseFlashMlaGradBasicParams)

BEGIN_TILING_DATA_DEF(SparseFlashMlaGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaGradBasicParams, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaGradSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SmlagPostParams, postTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad, SparseFlashMlaGradTilingData)
} // namespace optiling
