/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ARCH35_UTILS_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ARCH35_UTILS_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"

namespace NpuArch::Epilogue::Block {

struct UBufTileHelper {
    uint32_t qBaseTilePerSubCore;
    uint32_t kvBaseTilePerSubCore;
    uint32_t embedPerSubCore;
    uint32_t sStartOffset;
    uint32_t pStartOffset;
    uint32_t loStartOffset;
    uint32_t diagStartOffset;
    uint32_t goStartOffset;
    uint32_t lmStartOffset;
    uint32_t gmStartOffset;
    uint32_t dmStartOffset;
    uint32_t llStartOffset;
    uint32_t glStartOffset;
    uint32_t lseStartOffset;
    uint32_t maskStartOffset;

    __aicore__ inline
    UBufTileHelper() {}

    __aicore__ inline
    UBufTileHelper(
        uint32_t qs,
        uint32_t kvs,
        uint32_t d,
        uint32_t s,
        uint32_t p,
        uint32_t lo,
        uint32_t go,
        uint32_t lm,
        uint32_t gm,
        uint32_t dm,
        uint32_t ll,
        uint32_t gl,
        uint32_t lse,
        uint32_t mask,
        uint32_t diag = 0) :
        qBaseTilePerSubCore(qs),
        kvBaseTilePerSubCore(kvs),
        embedPerSubCore(d),
        sStartOffset(s),
        pStartOffset(p),
        loStartOffset(lo),
        diagStartOffset(diag),
        goStartOffset(go),
        lmStartOffset(lm),
        gmStartOffset(gm),
        dmStartOffset(dm),
        llStartOffset(ll),
        glStartOffset(gl),
        lseStartOffset(lse),
        maskStartOffset(mask) {}
};

template <uint32_t MODE, pipe_t PIPE>
__aicore__ inline
void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
{
    // in mode 4, AIC set for 2 AIVs seperately
    if constexpr (MODE == 4U) {
        Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
    }
}

template <uint32_t MODE, pipe_t PIPE>
__aicore__ inline
void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
{
    // in mode 4, AIC wait for 2 AIVs seperately
    if constexpr (MODE == 4U) {
        Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
    }
}
}
#endif