// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

#include "catlass/arch/cross_core_sync.hpp"
#include "catccos/comm/mte_transport_params.hpp"
#include "shmem.h"

namespace Catccos::Comm {

// Runtime initializes this line once. Per-invocation signal/worker resets and
// the initialization probe must leave it intact, including across empty calls.
inline constexpr uint64_t kMteBarrierGenerationOffset =
    OpsTransformer::CatccosMte::kMteOperatorControlBytes - 64;
static_assert(kMteBarrierGenerationOffset % 64 == 0);
static_assert(OpsTransformer::CatccosMte::kMteWorkerFlagsOffset +
                  uint64_t{OpsTransformer::CatccosMte::kMteMaxRanks} *
                      OpsTransformer::CatccosMte::kMteMaxPeerWorkers *
                      OpsTransformer::CatccosMte::kMteWorkerFlagStride <=
              kMteBarrierGenerationOffset);

// Every AIV, including inactive communication workers, calls this in the same
// order on every rank. The dedicated AIV barrier uses hardware flag 8, leaving
// stage flags 0/1 untouched and avoiding SHMEM's fixed-size software core pool.
CATLASS_DEVICE void MteAllRankBarrier(GM_ADDR control, uint32_t ranks)
{
    Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
    if (AscendC::GetBlockIdx() == 0) {
        auto local = reinterpret_cast<__gm__ uint32_t *>(control + kMteBarrierGenerationOffset);
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(local));
        uint32_t generation = *reinterpret_cast<__gm__ volatile uint32_t *>(local) + 1U;
        *local = generation;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(local));
        for (uint32_t peer = 0; peer < ranks; ++peer) {
            auto remote = static_cast<__gm__ uint32_t *>(aclshmem_ptr(local, peer));
            uint32_t observed;
            do {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(remote));
                observed = *reinterpret_cast<__gm__ volatile uint32_t *>(remote);
                // A fast rank can publish the next barrier before this rank
                // finishes polling. Modular >= accepts that and uint32 wrap.
                // Barrier progress bounds rank skew to one, well below 2^31.
            } while (static_cast<int32_t>(observed - generation) < 0);
        }
    }
    Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
}

} // namespace Catccos::Comm
