/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief matmul implementation for single q&k^t base tile
 * This implementation is designed for the following senario:
 * A full q base tile is loaded to L1 from GM at the very beginning,
 * and it remains persistent until each k base tile is dealt
 * A full q*k^t base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_ARCH35_UTILS_HPP
#define GEMM_BLOCK_ARCH35_UTILS_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"

namespace NpuArch::Gemm::Block {

static constexpr uint32_t MAX_L1_STAGES = 3; // 编译期常量，为静态L1Tensor数组开辟准备。取一个buffer份数的极大值
static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16; // 核间同步mode4，AIC侧需要两个flagId分别对应两个AIV

struct Mm1L1TileHelper {
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;

    __aicore__ inline
    Mm1L1TileHelper() {}

    __aicore__ inline
    Mm1L1TileHelper(
        uint32_t m,
        uint32_t n,
        uint32_t kl,
        uint32_t kr,
        uint32_t qbn,
        uint32_t kbn) :
        mm1L1TileM(m),
        mm1L1TileN(n),
        mm1L1TileKLeft(kl),
        mm1L1TileKRight(kr),
        qL1BufNum(qbn),
        kL1BufNum(kbn) {}
};

template <uint32_t MODE, pipe_t PIPE>
__aicore__ inline
void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
{
    // in mode 4, AIC set for 2 AIVs seperately
    if constexpr (MODE == 4U) {
        uint16_t flagIdV0 = crossCoreFlag.id;
        uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
        Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
        Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
    }
}

template <uint32_t MODE, pipe_t PIPE>
__aicore__ inline
void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
{
    // in mode 4, AIC wait for 2 AIVs seperately
    if constexpr (MODE == 4U) {
        uint16_t flagIdV0 = crossCoreFlag.id;
        uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
        Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
        Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
    }
}

__aicore__ inline
uint32_t GetCurLoopCounter(uint32_t outterLoopItr, uint32_t curLoopNum, uint32_t curLoopItr)
{
    return outterLoopItr * curLoopNum + curLoopItr;
}

}
#endif // GEMM_BLOCK_ARCH35_UTILS_HPP