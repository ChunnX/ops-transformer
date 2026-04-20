/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <typename ElementOutput>
class BlockEpilogue<EpilogueAscend950FAGPre, ElementOutput>
{
public:
    using DispatchPolicy = EpilogueAscend950FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    AscendC::TPipe *pipe;
    uint32_t cBlockIdx;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in)
    {
        cBlockIdx = AscendC::GetBlockIdx();
        pipe = pipe_in;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOutput> &gDqWorkSpace,
        AscendC::GlobalTensor<ElementOutput> &gDkWorkSpace,
        AscendC::GlobalTensor<ElementOutput> &gDvWorkSpace,
        int64_t qSize, int64_t kvSize, uint32_t coreNum)
    {
        uint32_t qPreBlockFactor = (qSize + coreNum - 1) / coreNum;
        uint32_t qPreBlockTotal = (qSize + qPreBlockFactor - 1) / qPreBlockFactor;
        int64_t qPreTailNumTmp = qSize % qPreBlockFactor;
        uint32_t qPreBlockTail = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

        uint32_t kvPreBlockFactor = (kvSize + coreNum - 1) / coreNum;
        uint32_t kvPreBlockTotal = (kvSize + kvPreBlockFactor - 1) / kvPreBlockFactor;
        int64_t kvPreTailNumTmp = kvSize % kvPreBlockFactor;
        uint32_t kvPreBlockTail = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

        int64_t dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
        int64_t dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
        int64_t initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        int64_t initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;

        if (g_coreType == AscendC::AIV && cBlockIdx < qPreBlockTotal) {
            AscendC::InitOutput<float>(gDqWorkSpace[dqOffset], initdqSize, 0);
        }

        if (g_coreType == AscendC::AIV && cBlockIdx < kvPreBlockTotal) {
            AscendC::InitOutput<float>(gDkWorkSpace[dkvOffset], initdkSize, 0);
            AscendC::InitOutput<float>(gDvWorkSpace[dkvOffset], initdkSize, 0);
        }
    }

};

}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
