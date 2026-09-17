// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

#include "catlass/catlass.hpp"
#include "catccos/comm/mte_transport_params.hpp"
#include "shmem.h"

namespace Catccos::Comm {

// Each lane writes its own cache line. A leader can publish a peer completion
// only after every participating lane has finished MTE3, including empty lanes.
class MtePeerCompletion {
public:
    CATLASS_DEVICE explicit MtePeerCompletion(GM_ADDR control) : control_(control) {}

    CATLASS_DEVICE void Reset(uint32_t ranks, uint32_t workers)
    {
        for (uint32_t peer = 0; peer < ranks; ++peer) {
            for (uint32_t lane = 0; lane < workers; ++lane) {
                auto ptr = Flag(peer, lane);
                *ptr = 0;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ptr));
            }
        }
    }

    CATLASS_DEVICE void MarkDone(uint32_t peer, uint32_t lane, int32_t generation)
    {
        // MTE3_MTE2 drains a reusable payload slot. Publishing scalar state
        // additionally requires an explicit MTE3 -> Scalar dependency.
        constexpr auto event = OpsTransformer::CatccosMte::kMteCompletionEvent;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event);
        auto ptr = Flag(peer, lane);
        *ptr = generation;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ptr));
    }

    CATLASS_DEVICE void Wait(uint32_t peer, uint32_t workers, int32_t generation)
    {
        for (uint32_t lane = 0; lane < workers; ++lane) {
            aclshmem_signal_wait_until(Flag(peer, lane), ACLSHMEM_CMP_EQ, generation);
        }
    }

private:
    CATLASS_DEVICE __gm__ int32_t *Flag(uint32_t peer, uint32_t lane)
    {
        using namespace OpsTransformer::CatccosMte;
        return reinterpret_cast<__gm__ int32_t *>(control_ + kMteWorkerFlagsOffset +
            (uint64_t{peer} * kMteMaxPeerWorkers + lane) * kMteWorkerFlagStride);
    }
    GM_ADDR control_;
};

} // namespace Catccos::Comm
