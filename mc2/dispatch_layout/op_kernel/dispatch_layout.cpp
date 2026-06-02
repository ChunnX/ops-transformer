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
 * \file dispatch_layout.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "dispatch_layout.h"
#include "dispatch_layout_tiling.h"
#include "dispatch_layout_tiling_key.h"

using namespace AscendC;
using namespace Mc2Kernel;
using namespace Mc2Tiling;

template <typename T>
__aicore__ inline __gm__ T *GetTensorAddrFromTensorList(uint16_t index, GM_ADDR tensorListPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *tensorAddrList = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T *>(*(tensorAddrList + index));
}

template <uint8_t ArchTag>
__global__ __aicore__ void dispatch_layout(GM_ADDR topkIdx, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert,
                                          GM_ADDR isTokenInRank, GM_ADDR ext_handle_data, GM_ADDR workspace,
                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DispatchLayoutTilingData);
    if constexpr ((ArchTag == TILINGKEY_TPL_A3) || (ArchTag == TILINGKEY_TPL_A2)) {
        GET_TILING_DATA_WITH_STRUCT(DispatchLayoutTilingData, tilingData, tiling);

        GM_ADDR notifySendData =
            reinterpret_cast<GM_ADDR>(GetTensorAddrFromTensorList<int32_t>(0, ext_handle_data));
        GM_ADDR sendTokenIdxSmall =
            reinterpret_cast<GM_ADDR>(GetTensorAddrFromTensorList<int32_t>(1, ext_handle_data));

        TPipe pipe;

        MoeDispatchLayout::DispatchLayout<int32_t> op;
        op.Init(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, notifySendData, sendTokenIdxSmall,
                workspace, &pipe, &tilingData);
        op.Process();
    }
}
