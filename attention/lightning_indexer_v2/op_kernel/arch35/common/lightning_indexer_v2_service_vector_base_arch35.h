/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_SERVICE_VECTOR_BASE_ARCH35_H
#define LIGHTNING_INDEXER_V2_SERVICE_VECTOR_BASE_ARCH35_H

#include "lightning_indexer_v2_base_arch35.h"

namespace LIV2Common {

template <typename Derived>
class LightningIndexerV2ServiceVectorBase {
public:
    template <typename Score>
    __aicore__ inline void InitVecWorkspaceTensor(AscendC::GlobalTensor<Score> scoreGm,
        AscendC::GlobalTensor<Score> ldScoreGm, AscendC::GlobalTensor<int32_t> ldIndexGm)
    {
        auto &service = static_cast<Derived &>(*this);
        service.scoreGm = scoreGm;
        service.ldScoreGm = ldScoreGm;
        service.ldIndexGm = ldIndexGm;
    }

protected:
    __aicore__ inline void CleanInvalidIndices(int64_t offset, uint64_t size)
    {
        auto &service = static_cast<Derived &>(*this);
        AscendC::GlobalTensor<int32_t> output = service.indiceOutGm[offset];
        AscendC::InitGlobalMemory(output, size, service.constInfo_.INVALID_IDX);
    }

    template <typename ValueBits, typename RunInfoT>
    __aicore__ inline void PadTndOutputs(const RunInfoT &runInfo, uint64_t topk, bool returnValue,
                                       ValueBits negativeInfinity)
    {
        auto &service = static_cast<Derived &>(*this);
        uint32_t paddingLen = runInfo.curCuSeqlensQ - runInfo.curSequsedQ;
        uint64_t paddingOffset = runInfo.indiceOutOffset + runInfo.curSequsedQ * service.constInfo_.kHeadNum * topk;
        uint64_t dealSize = paddingLen * service.constInfo_.kHeadNum * topk;
        CleanInvalidIndices(paddingOffset, dealSize);
        if (returnValue) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Derived::MTE3_V_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Derived::MTE3_V_EVENT);
            AscendC::GlobalTensor<ValueBits> valueOutGmTmp;
            valueOutGmTmp.SetGlobalBuffer((__gm__ ValueBits *)service.valueOutGm.GetPhyAddr());
            AscendC::GlobalTensor<ValueBits> valueOut = valueOutGmTmp[paddingOffset];
            AscendC::InitGlobalMemory(valueOut, dealSize, negativeInfinity);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Derived::MTE3_V_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Derived::MTE3_V_EVENT);
        }
    }

    __aicore__ inline uint64_t GetLdGmOffset(uint32_t row, uint32_t block, uint32_t workspaceCount)
    {
        auto &service = static_cast<Derived &>(*this);
        return service.ldInfo_.workspaceIdx * service.s1BaseSize_ * service.topkCountAlign16_ +
               service.topkCountAlign16_ * (service.ldInfo_.mStart + row) +
               block * workspaceCount * service.s1BaseSize_ * service.topkCountAlign16_;
    }

    __aicore__ inline void AllocVectorOutputEvents()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Derived::VEC1_MTE3_V_EVENT + 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Derived::VEC1_MTE3_V_EVENT + 1);
    }

    __aicore__ inline void AllocTopkEvents()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(Derived::TOPK_V_MTE2_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Derived::TOPK_MTE3_V_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(Derived::V_MTE2_EVENT1);
    }

    __aicore__ inline void FreeVectorOutputEvents()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Derived::VEC1_MTE3_V_EVENT + 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Derived::VEC1_MTE3_V_EVENT + 1);
    }

    __aicore__ inline void FreeTopkEvents()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(Derived::TOPK_V_MTE2_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Derived::TOPK_MTE3_V_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(Derived::V_MTE2_EVENT1);
    }

};

} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_SERVICE_VECTOR_BASE_ARCH35_H
