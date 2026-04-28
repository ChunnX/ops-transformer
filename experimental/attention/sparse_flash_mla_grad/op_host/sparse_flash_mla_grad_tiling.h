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
TILING_DATA_FIELD_DEF(uint32_t, s2OuterOuter);
TILING_DATA_FIELD_DEF(uint32_t, singleM);
TILING_DATA_FIELD_DEF(uint32_t, singleN);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseM);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseN);
TILING_DATA_FIELD_DEF(uint32_t, rsv);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGradSplitCoreParamsOp, SparseFlashMlaGradSplitCoreParams)

BEGIN_TILING_DATA_DEF(sfmgPostParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, postUbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, postMaxDataSize);
TILING_DATA_FIELD_DEF(uint32_t, nzReservedSize);
TILING_DATA_FIELD_DEF(uint32_t, qPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, qPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, qPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, oriKVPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, oriKVPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, oriKVPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, oriKVPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, cmpKVPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, cmpKVPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, cmpKVPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, cmpKVPostTailNum);
TILING_DATA_FIELD_DEF(uint64_t, qSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, oriKVSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, cmpKVSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, dqWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dOriKVWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dCmpKVWorkSpaceOffset);
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, n2);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, s1);
TILING_DATA_FIELD_DEF(int64_t, s2);
TILING_DATA_FIELD_DEF(int64_t, s3);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, d1);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(sfmgPostParamsOp, sfmgPostParams)

BEGIN_TILING_DATA_DEF(SparseFlashMlaGradEmptyTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaGradSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(sfmgPostParams, postTilingData);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(SparseFlashMlaGradBaseParams)
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
TILING_DATA_FIELD_DEF(int64_t, D);  // qk_dim
TILING_DATA_FIELD_DEF(int64_t, D2); // v_dim
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, selectedKWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedVWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, scatterDkWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, scatterDvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm1WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm2WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm4InputWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm3InputWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, castUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGradBaseParamsOp, SparseFlashMlaGradBaseParams)

//======================== basic template tiling data def start ========================//
BEGIN_TILING_DATA_DEF(SparseFlashMlaGradBaseBasicParams)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreProcessNNum);
TILING_DATA_FIELD_DEF(uint32_t, remainCoreProcessNNum);
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, N2);
TILING_DATA_FIELD_DEF(int64_t, S1);
TILING_DATA_FIELD_DEF(int64_t, S2);
TILING_DATA_FIELD_DEF(int64_t, G);
TILING_DATA_FIELD_DEF(int64_t, D);  // qk_dim
TILING_DATA_FIELD_DEF(int64_t, D2); // v_dim
TILING_DATA_FIELD_DEF(int64_t, ropeD); // rope_dim
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, selectedKWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedVWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm12WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, castUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockSize);
TILING_DATA_FIELD_DEF(bool, deterministic);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGradBaseBasicParamsOp, SparseFlashMlaGradBaseBasicParams)

BEGIN_TILING_DATA_DEF(SparseFlashMlaGradBasicTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaGradBaseBasicParams, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaGradSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(sfmgPostParams, postTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
//======================== basic template tiling data def end ========================//

REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad, SparseFlashMlaGradEmptyTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad_1000, SparseFlashMlaGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad_1100, SparseFlashMlaGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad_1010, SparseFlashMlaGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad_1110, SparseFlashMlaGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad_1001, SparseFlashMlaGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad_1101, SparseFlashMlaGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad_1011, SparseFlashMlaGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashMlaGrad_1111, SparseFlashMlaGradBasicTilingData)
} // namespace optiling
