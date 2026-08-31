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
 * \file moe_gmm_sync_policy.h
 * \brief GMM1 SyncPolicy classes — encapsulate MoE-specific synchronization logic.
 *
 * Each class implements 6 methods:
 *   WaitUpstreamReady(mLoc)        — wait for upstream (Dispatch) data ready
 *   NotifyAivTileDone(loopIdx)    — AIC notifies AIV that tile is done
 *   WaitAivTileDone(loopIdx)      — AIC waits for AIV to finish reading
 *   WaitAicTileDone(loopIdx)      — AIV waits for AIC to finish writing
 *   NotifyAicTileDone(loopIdx)    — AIV notifies AIC that reading is done
 *   NotifyDownstreamReady(mLoc)   — notify downstream (GMM2) that data is ready
 *
 * Kernel loops call sync.Method() without knowing the underlying sync mechanism
 * (CrossCore flag, GM flag, or GM sequence number). The TopkWeightsPrefetch template
 * parameter selects between UB pingpong (non-prefetch) and GM flag (prefetch) paths.
 */

#ifndef MOE_GMM_SYNC_POLICY_H
#define MOE_GMM_SYNC_POLICY_H

#include "common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

// ===========================================================================
// Gmm1GenericSyncPolicy — Generic Kernel sync
//   TopkWeightsPrefetch=false: AIC writes UB, CrossCore flag pingpong (Gmm1UbSyncPolicy)
//   TopkWeightsPrefetch=true:  AIC writes GM, GM per-tile flag (Gmm1GmSyncPolicy)
// ===========================================================================
template <bool TopkWeightsPrefetch, bool IsShared, bool IsWaveFlagGrained = false, bool IsGmm1Interleaved = false>
class Gmm1GenericSyncPolicy {
public:
    __aicore__ inline Gmm1GenericSyncPolicy(__gm__ int32_t *dispatchToGmm1Flag, __gm__ int32_t *activationToGmm2Flag,
                                            __gm__ int32_t *gmm1TileStatus, uint32_t expertIdx,
                                            int32_t &vecSetSyncCom, uint16_t &pingpongIdx,
                                            uint32_t m, uint32_t tileM)
        : dispatchToGmm1Flag_(dispatchToGmm1Flag), activationToGmm2Flag_(activationToGmm2Flag),
          gmm1TileStatus_(gmm1TileStatus), expertIdx_(expertIdx),
          vecSetSyncCom_(vecSetSyncCom), pingpongIdx_(pingpongIdx), m_(m), tileM_(tileM) {}

    __aicore__ inline void WaitUpstreamReady(uint32_t mLoc)
    {
        if constexpr (IsShared) {
            return;
        }
        uint32_t waveIdx = mLoc / tileM_;
        if (waveIdx == lastWaveWaited_) {
            return;
        }
        uint32_t targetValue = (mLoc + tileM_ > m_) ? (m_ - mLoc) : tileM_;
        uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
        __gm__ int32_t *flagValueAddr = dispatchToGmm1Flag_ + flagOffset;
        WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetValue));
        lastWaveWaited_ = waveIdx;
    }

    __aicore__ inline void NotifyAivTileDone(uint32_t loopIdx)
    {
        if constexpr (TopkWeightsPrefetch) {
            __gm__ int32_t *statusAddr = gmm1TileStatus_ + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
            AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(expertIdx_ + 1));
        } else {
            if constexpr (IsGmm1Interleaved) {
                NotifyVector(pingpongIdx_);
                vecSetSyncCom_++;
                pingpongIdx_ = 1U - pingpongIdx_;
            } else {
                NotifyVector();
                vecSetSyncCom_ = 1;
            }
        }
    }

    __aicore__ inline void WaitAivTileDone(uint32_t loopIdx)
    {
        if constexpr (TopkWeightsPrefetch) {
            return;
        }
        if constexpr (IsGmm1Interleaved) {
            if (vecSetSyncCom_ >= 2) {
                WaitForVector(pingpongIdx_);
            }
        } else if (vecSetSyncCom_ != 0) {
            WaitForVector();
        }
    }

    __aicore__ inline void WaitAicTileDone(uint32_t loopIdx)
    {
        if constexpr (TopkWeightsPrefetch) {
            __gm__ int32_t *statusAddr = gmm1TileStatus_ + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
            int32_t roundTag = static_cast<int32_t>(expertIdx_ + 1);
            WaitUntilGmFlagEquals(statusAddr, roundTag);
        } else {
            if constexpr (IsGmm1Interleaved) {
                WaitForCube(pingpongIdx_);
            } else {
                WaitForCube();
            }
        }
    }

    __aicore__ inline void NotifyAicTileDone(uint32_t loopIdx)
    {
        if constexpr (TopkWeightsPrefetch) {
            return;
        }
        if constexpr (IsGmm1Interleaved) {
            NotifyCube(pingpongIdx_);
            pingpongIdx_ = 1U - pingpongIdx_;
        } else {
            NotifyCube();
        }
    }

    __aicore__ inline void NotifyDownstreamReady(uint32_t mLoc)
    {
        if (activationToGmm2Flag_ == nullptr) {
            return;
        }
        __gm__ int32_t *flagAddr = activationToGmm2Flag_;
        if constexpr (IsWaveFlagGrained) {
            flagAddr += static_cast<uint64_t>(mLoc / L1_TILE_M_256) * INT_CACHELINE;
        }
        AscendC::AtomicAdd(flagAddr, 1);
    }

private:
    __gm__ int32_t *dispatchToGmm1Flag_;
    __gm__ int32_t *activationToGmm2Flag_;
    __gm__ int32_t *gmm1TileStatus_;
    uint32_t expertIdx_;
    int32_t &vecSetSyncCom_;
    uint16_t &pingpongIdx_;
    uint32_t lastWaveWaited_ = static_cast<uint32_t>(-1);
    uint32_t m_;
    uint32_t tileM_;
};

// ===========================================================================
// Gmm1A8W4SyncPolicy — A8W4 Kernel sync
//   TopkWeightsPrefetch=false: AIC writes GM, GM sequence number
//   TopkWeightsPrefetch=true:  AIC writes GM, GM per-tile flag
// ===========================================================================
template <bool TopkWeightsPrefetch, bool IsShared, bool IsWaveFlagGrained = false>
class Gmm1A8W4SyncPolicy {
public:
    __aicore__ inline Gmm1A8W4SyncPolicy(__gm__ int32_t *dispatchToGmm1Flag, __gm__ int32_t *activationToGmm2Flag,
                                         __gm__ int32_t *gmm1TileStatus, uint32_t expertIdx,
                                         __gm__ int32_t *gmmToEpilogueFlag, int32_t &gmTileSequence,
                                         uint32_t m, uint32_t tileM)
        : dispatchToGmm1Flag_(dispatchToGmm1Flag), activationToGmm2Flag_(activationToGmm2Flag),
          gmm1TileStatus_(gmm1TileStatus), expertIdx_(expertIdx),
          gmmToEpilogueFlag_(gmmToEpilogueFlag), gmTileSequence_(gmTileSequence), m_(m), tileM_(tileM) {}

    __aicore__ inline void WaitUpstreamReady(uint32_t mLoc)
    {
        if constexpr (IsShared) {
            return;
        }
        uint32_t waveIdx = mLoc / tileM_;
        if (waveIdx == lastWaveWaited_) {
            return;
        }
        uint32_t targetValue = (mLoc + tileM_ > m_) ? (m_ - mLoc) : tileM_;
        uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
        __gm__ int32_t *flagValueAddr = dispatchToGmm1Flag_ + flagOffset;
        WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetValue));
        lastWaveWaited_ = waveIdx;
    }

    __aicore__ inline void NotifyAivTileDone(uint32_t loopIdx)
    {
        if constexpr (TopkWeightsPrefetch) {
            __gm__ int32_t *statusAddr = gmm1TileStatus_ + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
            AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(expertIdx_ + 1));
        } else {
            AscendC::WriteGmByPassDCache(gmmToEpilogueFlag_, ++gmTileSequence_);
        }
    }

    __aicore__ inline void WaitAivTileDone(uint32_t loopIdx) {}

    __aicore__ inline void WaitAicTileDone(uint32_t loopIdx)
    {
        if constexpr (TopkWeightsPrefetch) {
            __gm__ int32_t *statusAddr = gmm1TileStatus_ + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
            int32_t roundTag = static_cast<int32_t>(expertIdx_ + 1);
            WaitUntilGmFlagEquals(statusAddr, roundTag);
        } else {
            int32_t expectedReadySequence = gmTileSequence_ + 1;
            WaitUntilGmFlagAtLeast(gmmToEpilogueFlag_, expectedReadySequence);
            gmTileSequence_ = expectedReadySequence;
        }
    }

    __aicore__ inline void NotifyAicTileDone(uint32_t loopIdx) {}

    __aicore__ inline void NotifyDownstreamReady(uint32_t mLoc)
    {
        if (activationToGmm2Flag_ == nullptr) {
            return;
        }
        __gm__ int32_t *flagAddr = activationToGmm2Flag_;
        if constexpr (IsWaveFlagGrained) {
            flagAddr += static_cast<uint64_t>(mLoc / L1_TILE_M_256) * INT_CACHELINE;
        }
        AscendC::AtomicAdd(flagAddr, 1);
    }

private:
    __gm__ int32_t *dispatchToGmm1Flag_;
    __gm__ int32_t *activationToGmm2Flag_;
    __gm__ int32_t *gmm1TileStatus_;
    uint32_t expertIdx_;
    __gm__ int32_t *gmmToEpilogueFlag_;
    int32_t &gmTileSequence_;
    uint32_t lastWaveWaited_ = static_cast<uint32_t>(-1);
    uint32_t m_;
    uint32_t tileM_;
};

// ===========================================================================
// Gmm2UbSyncPolicy — GMM2 Generic (Combine notification via AtomicAdd)
// ===========================================================================
template <bool IsShared, uint8_t CombineQuantMode, bool IsLayered = false, bool IsGmm1Interleaved = false,
        bool IsWaveFlagGrained = false>
class Gmm2UbSyncPolicy {
public:
    __aicore__ inline Gmm2UbSyncPolicy(__gm__ int32_t *activationToGmm2Flag,
                                       __gm__ int32_t *gmm2CombineSyncCounter,
                                       __gm__ int32_t *sharedExpertGmm2TileCounter,
                                       const ProblemShape &problemShape,
                                       uint32_t activationTileM, uint32_t blockNum)
        : activationToGmm2Flag_(activationToGmm2Flag),
          gmm2CombineSyncCounter_(gmm2CombineSyncCounter),
          sharedExpertGmm2TileCounter_(sharedExpertGmm2TileCounter),
          m_(static_cast<uint32_t>(Get<M_VALUE>(problemShape))),
          k_(static_cast<uint32_t>(Get<K_VALUE>(problemShape))),
          activationTileM_(activationTileM), blockNum_(blockNum)
    {
        if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
            uint32_t logicalCoreCount = blockNum;
            if constexpr (!IsWaveFlagGrained) {
                logicalCoreCount *= 2U;
            }
            groupSyncSlotLayout_ = CalcGroupSyncSlotLayout(m_, logicalCoreCount);
        }
    }

    __aicore__ inline void WaitUpstreamReady(uint32_t mLoc)
    {
        if constexpr (IsShared) {
            return;
        }
        if constexpr (IsWaveFlagGrained) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx == lastWaveWaited_) {
                return;
            }
            uint32_t waveStart = waveIdx * L1_TILE_M_256;
            uint32_t waveM = waveStart + L1_TILE_M_256 > m_ ? m_ - waveStart : L1_TILE_M_256;
            constexpr uint32_t sourceNFactor = IsGmm1Interleaved ? ACTIVATION_N_HALF : 1U;
            uint32_t targetLoops =
                Ops::Base::CeilDiv(waveM, activationTileM_) * Ops::Base::CeilDiv(k_ * sourceNFactor, L1_TILE_N);
            uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
            __gm__ int32_t *flagValueAddr = activationToGmm2Flag_ + flagOffset;
            WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetLoops));
            lastWaveWaited_ = waveIdx;
        } else {
            if (upstreamReady_) {
                return;
            }
            uint32_t targetLoops =
                Ops::Base::CeilDiv(m_, activationTileM_) * Ops::Base::CeilDiv(k_, L1_TILE_N);
            WaitUntilGmFlagEquals(activationToGmm2Flag_, static_cast<int32_t>(targetLoops));
            upstreamReady_ = true;
        }
    }

    __aicore__ inline void NotifyAivTileDone(uint32_t loopIdx) {}
    __aicore__ inline void WaitAivTileDone(uint32_t loopIdx) {}
    __aicore__ inline void WaitAicTileDone(uint32_t loopIdx) {}
    __aicore__ inline void NotifyAicTileDone(uint32_t loopIdx) {}

    __aicore__ inline void NotifyDownstreamReady(uint32_t mLoc)
    {
        if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            uint32_t tokenGroupIndex = mLoc / COMBINE_TOKEN_GROUP_SIZE;
            uint32_t firstSyncSlot = 0;
            uint32_t syncSlotCount = 0;
            GetGroupSyncSlotRange(tokenGroupIndex, groupSyncSlotLayout_, firstSyncSlot, syncSlotCount);
            for (uint32_t syncSlot = firstSyncSlot; syncSlot < firstSyncSlot + syncSlotCount; ++syncSlot) {
                AscendC::AtomicAdd(GetCombineSyncCounterAddress(gmm2CombineSyncCounter_, syncSlot), int32_t(1));
            }
        } else if constexpr (IsShared && !IsLayered) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            uint32_t tokenGroupIndex = mLoc / COMBINE_TOKEN_GROUP_SIZE;
            AscendC::AtomicAdd(GetCombineSyncCounterAddress(sharedExpertGmm2TileCounter_, tokenGroupIndex), int32_t(1));
        }
    }

private:
    __gm__ int32_t *activationToGmm2Flag_;
    __gm__ int32_t *gmm2CombineSyncCounter_;
    __gm__ int32_t *sharedExpertGmm2TileCounter_;
    GroupSyncSlotLayout groupSyncSlotLayout_{};
    uint32_t lastWaveWaited_ = static_cast<uint32_t>(-1);
    bool upstreamReady_ = false;
    uint32_t m_;
    uint32_t k_;
    uint32_t activationTileM_;
    uint32_t blockNum_;
};

// ===========================================================================
// Gmm2A8W4SyncPolicy — GMM2 A8W4 (NO_QUANT: GM 序号; 量化: Combine 通知)
// ===========================================================================
template <bool IsShared, uint8_t CombineQuantMode, bool IsLayered = false, bool IsGmm1Interleaved = false,
            bool IsWaveFlagGrained = false>
class Gmm2A8W4SyncPolicy {
public:
    __aicore__ inline Gmm2A8W4SyncPolicy(__gm__ int32_t *activationToGmm2Flag,
                                        __gm__ int32_t *gmmToEpilogueFlag,
                                        __gm__ int32_t *gmm2CombineSyncCounter,
                                        __gm__ int32_t *sharedExpertGmm2TileCounter,
                                        int32_t &gmTileSequence,
                                        const ProblemShape &problemShape,
                                        uint32_t activationTileM, uint32_t blockNum,
                                        uint32_t expertTokenCount, uint32_t rowOffsetInExpert)
        : activationToGmm2Flag_(activationToGmm2Flag),
          gmmToEpilogueFlag_(gmmToEpilogueFlag),
          gmm2CombineSyncCounter_(gmm2CombineSyncCounter),
          sharedExpertGmm2TileCounter_(sharedExpertGmm2TileCounter),
          gmTileSequence_(gmTileSequence),
          m_(static_cast<uint32_t>(Get<M_VALUE>(problemShape))),
          k_(static_cast<uint32_t>(Get<K_VALUE>(problemShape))),
          activationTileM_(activationTileM), blockNum_(blockNum),
          rowOffsetInExpert_(rowOffsetInExpert)
    {
        if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
            uint32_t logicalCoreCount = blockNum;
            groupSyncSlotLayout_ = CalcGroupSyncSlotLayout(expertTokenCount, logicalCoreCount);
        }
    }

    __aicore__ inline void WaitUpstreamReady(uint32_t mLoc)
    {
        if constexpr (IsShared) {
            return;
        }
        if constexpr (IsWaveFlagGrained) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx == lastWaveWaited_) {
                return;
            }
            uint32_t waveStart = waveIdx * L1_TILE_M_256;
            uint32_t waveM = waveStart + L1_TILE_M_256 > m_ ? m_ - waveStart : L1_TILE_M_256;
            constexpr uint32_t sourceNFactor = IsGmm1Interleaved ? ACTIVATION_N_HALF : 1U;
            uint32_t targetLoops =
                Ops::Base::CeilDiv(waveM, activationTileM_) * Ops::Base::CeilDiv(k_ * sourceNFactor, L1_TILE_N);
            uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
            __gm__ int32_t *flagValueAddr = activationToGmm2Flag_ + flagOffset;
            WaitUntilGmFlagEquals(flagValueAddr , static_cast<int32_t>(targetLoops));
            lastWaveWaited_ = waveIdx;
        } else {
            if (upstreamReady_) {
                return;
            }
            uint32_t targetLoops =
                Ops::Base::CeilDiv(m_, activationTileM_) * Ops::Base::CeilDiv(k_, L1_TILE_N);
            WaitUntilGmFlagEquals(activationToGmm2Flag_, static_cast<int32_t>(targetLoops));
            upstreamReady_ = true;
        }
    }

    __aicore__ inline void NotifyAivTileDone(uint32_t loopIdx)
    {
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmToEpilogueFlag_, ++gmTileSequence_);
        }
    }

    __aicore__ inline void WaitAivTileDone(uint32_t loopIdx) {}

    __aicore__ inline void WaitAicTileDone(uint32_t loopIdx)
    {
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
            int32_t expectedReadySequence = gmTileSequence_ + 1;
            WaitUntilGmFlagAtLeast(gmmToEpilogueFlag_, expectedReadySequence);
            gmTileSequence_ = expectedReadySequence;
        }
    }

    __aicore__ inline void NotifyAicTileDone(uint32_t loopIdx) {}

    __aicore__ inline void NotifyDownstreamReady(uint32_t mLoc)
    {
        if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            uint32_t tokenGroupIndex = (rowOffsetInExpert_ + mLoc) / COMBINE_TOKEN_GROUP_SIZE;
            uint32_t firstSyncSlot = 0;
            uint32_t syncSlotCount = 0;
            GetGroupSyncSlotRange(tokenGroupIndex, groupSyncSlotLayout_, firstSyncSlot, syncSlotCount);
            for (uint32_t syncSlot = firstSyncSlot; syncSlot < firstSyncSlot + syncSlotCount; ++syncSlot) {
                AscendC::AtomicAdd(GetCombineSyncCounterAddress(gmm2CombineSyncCounter_, syncSlot), int32_t(1));
            }
        } else if constexpr (IsShared) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            uint32_t tokenGroupIndex = mLoc / COMBINE_TOKEN_GROUP_SIZE;
            AscendC::AtomicAdd(GetCombineSyncCounterAddress(sharedExpertGmm2TileCounter_, tokenGroupIndex), int32_t(1));
        }
    }

private:
    __gm__ int32_t *activationToGmm2Flag_;
    __gm__ int32_t *gmmToEpilogueFlag_;
    __gm__ int32_t *gmm2CombineSyncCounter_;
    __gm__ int32_t *sharedExpertGmm2TileCounter_;
    int32_t &gmTileSequence_;
    GroupSyncSlotLayout groupSyncSlotLayout_{};
    uint32_t lastWaveWaited_ = static_cast<uint32_t>(-1);
    bool upstreamReady_ = false;
    uint32_t m_;
    uint32_t k_;
    uint32_t activationTileM_;
    uint32_t blockNum_;
    uint32_t rowOffsetInExpert_;
};

} // namespace MegaMoeImpl

#endif // MOE_GMM_SYNC_POLICY_H
