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
 * \file bandwidth_test.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "bandwidth_test_tiling_key.h"
#include "bandwidth_test.h"
#include "bandwidth_test_urma.h"
#include "bandwidth_test_urma_shared_jetty.h"

using namespace BandwidthTestImpl;
using namespace BandwidthTestTiling;
using namespace AscendC;

template <uint8_t ArchTag, uint8_t CommMode, uint8_t SharedJetty>
__global__ __aicore__ void bandwidth_test(GM_ADDR context, GM_ADDR x, GM_ADDR dstRankId, GM_ADDR y, GM_ADDR receiveCnt,
                                          GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(BandwidthTestTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", BandwidthTestTilingData);

    TPipe pipe;

    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(BandwidthTestTilingData, tilingData, tilingGM);
        if constexpr (CommMode == TILINGKEY_TPL_URMA) {
            if constexpr (SharedJetty == 1) {
                BandwidthTestUrmaSharedJetty<DTYPE_X, DTYPE_Y> op;
                op.Init(context, x, dstRankId, y, receiveCnt, workspaceGM, &pipe, &tilingData);
                op.Process();
            } else {
                BandwidthTestUrma<DTYPE_X, DTYPE_Y> op;
                op.Init(context, x, dstRankId, y, receiveCnt, workspaceGM, &pipe, &tilingData);
                op.Process();
            }
        } else {
            BandwidthTest<DTYPE_X, DTYPE_Y> op;
            op.Init(context, x, dstRankId, y, receiveCnt, workspaceGM, &pipe, &tilingData);
            op.Process();
        }
    }
}
