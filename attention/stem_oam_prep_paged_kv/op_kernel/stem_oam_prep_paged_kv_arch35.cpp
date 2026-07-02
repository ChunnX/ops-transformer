/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_arch35.cpp
 * \brief StemOamPrepPagedKV arch35 kernel entry point
 */
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "arch35/stem_oam_prep_paged_kv_tiling_key.h"
#include "arch35/stem_oam_prep_paged_kv.h"

extern "C" __global__ __aicore__ void stem_oam_prep_paged_kv_apt(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices,
                                                                 GM_ADDR kvSeqLens, GM_ADDR kScaleCache, GM_ADDR vScale,
                                                                 GM_ADDR kFlat, GM_ADDR vBias, GM_ADDR workspace,
                                                                 GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StemPrepKVTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA_WITH_STRUCT(StemPrepKVTilingData, tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        using namespace StemOamPrepPagedKV;
        TPipe pipe;
        KernelStemPrepKV op(&pipe, &tilingData);
        op.Init(kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale, kFlat, vBias, workspace);
        op.Process();
    }
}