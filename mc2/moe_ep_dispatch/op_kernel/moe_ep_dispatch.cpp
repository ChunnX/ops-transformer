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
 * \file moe_ep_dispatch.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_tiling_key.h"
#include "moe_ep_dispatch_tiling.h"
#include "moe_ep_dispatch.h"

#if __has_include("arch35/moe_ep_dispatch_arch35.h")
#include "arch35/moe_ep_dispatch_arch35.h"
#endif

using namespace MoeEpDispatchImpl;
using namespace Mc2Tiling;
using namespace AscendC;

template<uint32_t DO_CPU_SYNC, uint32_t IS_CACHED, uint32_t NETWORK_MODE>
__global__ __aicore__ void moe_ep_dispatch(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
    GM_ADDR cachedSlotIdx,
    GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
    GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
    GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeEpDispatchTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", MoeEpDispatchTilingData);

    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeEpDispatchTilingData, tilingData, tilingGM);
    KernelMoeEpDispatch<DO_CPU_SYNC, IS_CACHED, NETWORK_MODE> op;
    AscendC::printf("start MoeEpDispatch kernel\n");
    op.Init(context, x, topkIdx, topkWeights,
            cachedSlotIdx,
            numRecvPerRankHp, numRecvPerExpertHp,
            dstBufferSlotIdx, recvSrcMetadataInternal,
            workspace, tilingGM, &pipe, &tilingData.moeEpDispatchInfo);
    op.Process();
}
