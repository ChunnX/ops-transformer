/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_GMM_SYNC_POLICY_H
#define MOE_GMM_SYNC_POLICY_H

#include "common/mega_moe_utils.h"
#include "common/mega_moe_gmm_common.h"

namespace MegaMoeImpl {

/*
 * GMM1 同步策略：封装 GMM1 三条执行路径的 MoE 专属同步逻辑。
 *
 * 每个策略类实现统一 6 方法接口（duck-typing，无 trait 检测）：
 *   WaitUpstreamReady(mLoc)       AIC 等待 Dispatch 输出就绪（含 wave 去重）
 *   NotifyAivTileDone(loopIdx)    AIC 通知 AIV/AIV1 tile 完成可消费
 *   WaitAivTileDone(loopIdx)      AIC 等待 AIV 读完 UB（防覆盖，仅 UB ping-pong 路径）
 *   WaitAicTileDone(loopIdx)      AIV/AIV1 等待 AIC tile 就绪
 *   NotifyAicTileDone(loopIdx)    AIV 通知 AIC 可复用 UB（仅 UB ping-pong 路径）
 *   NotifyDownstreamReady(mLoc)   AIV 通知 GMM2 输入就绪
 */

template <bool IsShared, bool IsGmm1Interleaved, bool IsWaveFlagGrained = false,
          bool TopkWeightsPrefetch = false>
class Gmm1GenericSyncPolicy {
public:
    __aicore__ inline Gmm1GenericSyncPolicy(const GMMAddrInfo &gmmAddrInfo, uint32_t m, uint32_t tileM,
                                            uint32_t expertIdx, int32_t &vecSetSyncCom, uint16_t &pingpongIdx)
        : dispatchToGmm1Flag_(gmmAddrInfo.dispatchToGmm1Flag),
          activationToGmm2Flag_(gmmAddrInfo.activationToGmm2Flag),
          gmm1TileStatus_(gmmAddrInfo.gmm1TileStatus),
          expertIdx_(expertIdx),
          vecSetSyncCom_(vecSetSyncCom),
          pingpongIdx_(pingpongIdx),
          m_(m),
          tileM_(tileM)
    {
    }

    __aicore__ inline void WaitUpstreamReady(uint32_t mLoc)
    {
        uint32_t waveIdx = mLoc / tileM_;
        if (waveIdx == lastWaveWaited_) {
            return;
        }
        if constexpr (!IsShared) {
            uint32_t targetValue = (mLoc + tileM_ > m_) ? (m_ - mLoc) : tileM_;
            uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
            __gm__ int32_t *flagValueAddr = dispatchToGmm1Flag_ + flagOffset;
            WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetValue));
        }
        lastWaveWaited_ = waveIdx;
    }

    __aicore__ inline void WaitAivTileDone(uint32_t)
    {
        if constexpr (!TopkWeightsPrefetch) {
            if constexpr (IsGmm1Interleaved) {
                if (vecSetSyncCom_ >= 2) {
                    WaitForVector(pingpongIdx_);
                }
            } else if (vecSetSyncCom_ != 0) {
                WaitForVector();
            }
        }
    }

    __aicore__ inline uint16_t GetPingpongIdx() const
    {
        return pingpongIdx_;
    }

    __aicore__ inline void NotifyAivTileDone(uint32_t loopIdx)
    {
        if constexpr (TopkWeightsPrefetch) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
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

    __aicore__ inline void NotifyAicTileDone(uint32_t)
    {
        if constexpr (!TopkWeightsPrefetch) {
            if constexpr (IsGmm1Interleaved) {
                NotifyCube(pingpongIdx_);
                pingpongIdx_ = 1U - pingpongIdx_;
            } else {
                NotifyCube();
            }
        }
    }

    __aicore__ inline void NotifyDownstreamReady(uint32_t mLoc)
    {
        __gm__ int32_t *flagAddr = activationToGmm2Flag_;
        if (flagAddr == nullptr) {
            return;
        }
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
    uint32_t m_;
    uint32_t tileM_;
    uint32_t lastWaveWaited_ = static_cast<uint32_t>(-1);
};

// 场景：A8W4 全路径
template <bool IsShared, bool IsWaveFlagGrained = false, bool TopkWeightsPrefetch = false>
class Gmm1A8W4SyncPolicy {
public:
    __aicore__ inline Gmm1A8W4SyncPolicy(const GMMAddrInfo &gmmAddrInfo, uint32_t m, uint32_t tileM, uint32_t expertIdx,
                                         int32_t &gmm1TileReadySequence)
        : dispatchToGmm1Flag_(gmmAddrInfo.dispatchToGmm1Flag),
          activationToGmm2Flag_(gmmAddrInfo.activationToGmm2Flag),
          gmmToEpilogueFlag_(gmmAddrInfo.gmmToEpilogueFlag),
          gmm1TileStatus_(gmmAddrInfo.gmm1TileStatus),
          expertIdx_(expertIdx),
          gmm1TileReadySequence_(gmm1TileReadySequence),
          m_(m),
          tileM_(tileM)
    {
    }

    __aicore__ inline void WaitUpstreamReady(uint32_t mLoc)
    {
        uint32_t waveIdx = mLoc / tileM_;
        if (waveIdx == lastWaveWaited_) {
            return;
        }
        if constexpr (!IsShared) {
            uint32_t targetValue = (mLoc + tileM_ > m_) ? (m_ - mLoc) : tileM_;
            uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
            __gm__ int32_t *flagValueAddr = dispatchToGmm1Flag_ + flagOffset;
            WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetValue));
        }
        lastWaveWaited_ = waveIdx;
    }

    __aicore__ inline void NotifyAivTileDone(uint32_t loopIdx)
    {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        if constexpr (TopkWeightsPrefetch) {
            __gm__ int32_t *statusAddr = gmm1TileStatus_ + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
            AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(expertIdx_ + 1));
        } else {
            AscendC::WriteGmByPassDCache(gmmToEpilogueFlag_, ++gmm1TileReadySequence_);
        }
    }

    __aicore__ inline void WaitAivTileDone(uint32_t) {}

    __aicore__ inline void WaitAicTileDone(uint32_t loopIdx)
    {
        if constexpr (TopkWeightsPrefetch) {
            __gm__ int32_t *statusAddr = gmm1TileStatus_ + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
            int32_t roundTag = static_cast<int32_t>(expertIdx_ + 1);
            WaitUntilGmFlagEquals(statusAddr, roundTag);
        } else {
            int32_t expectedReadySequence = gmm1TileReadySequence_ + 1;
            WaitUntilGmFlagAtLeast(gmmToEpilogueFlag_, expectedReadySequence);
            gmm1TileReadySequence_ = expectedReadySequence;
        }
    }

    __aicore__ inline void NotifyAicTileDone(uint32_t) {}

    __aicore__ inline void NotifyDownstreamReady(uint32_t mLoc)
    {
        __gm__ int32_t *flagAddr = activationToGmm2Flag_;
        if (flagAddr == nullptr) {
            return;
        }
        if constexpr (IsWaveFlagGrained) {
            flagAddr += static_cast<uint64_t>(mLoc / L1_TILE_M_256) * INT_CACHELINE;
        }
        AscendC::AtomicAdd(flagAddr, 1);
    }

private:
    __gm__ int32_t *dispatchToGmm1Flag_;
    __gm__ int32_t *activationToGmm2Flag_;
    __gm__ int32_t *gmmToEpilogueFlag_;
    __gm__ int32_t *gmm1TileStatus_;
    uint32_t expertIdx_;
    int32_t &gmm1TileReadySequence_;
    uint32_t m_;
    uint32_t tileM_;
    uint32_t lastWaveWaited_ = static_cast<uint32_t>(-1);
};

// GMM2 同步策略：封装 GMM2 两条执行路径的 MoE 专属同步逻辑。
template <bool IsLayered, bool IsShared, bool IsWaveFlagGrained, bool IsGmm1Interleaved,
          bool NotifyCombineTileReady = false>
class Gmm2GenericSyncPolicy {
    static_assert(!IsShared || !NotifyCombineTileReady, "Shared expert GMM2 has its own completion counter");
    static_assert(!IsLayered || !NotifyCombineTileReady, "Layered GMM2 selects its synchronization through IsLayered");
public:
    __aicore__ inline Gmm2GenericSyncPolicy(const GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape,
                                            uint32_t activationTileM, uint32_t blockNum,
                                            int32_t *gmTileSequence = nullptr)
        : activationToGmm2Flag_(gmmAddrInfo.activationToGmm2Flag),
          gmm2CombineSyncCounter_(gmmAddrInfo.gmm2CombineSyncCounter),
          sharedExpertGmm2TileCounter_(gmmAddrInfo.sharedExpertGmm2TileCounter),
          gmmToEpilogueFlag_(gmmAddrInfo.gmmToEpilogueFlag),
          gmTileSequence_(gmTileSequence),
          m_(static_cast<uint32_t>(Get<M_VALUE>(problemShape))),
          k_(static_cast<uint32_t>(Get<N_VALUE>(problemShape)) / ACTIVATION_N_HALF),
          n_(static_cast<uint32_t>(Get<K_VALUE>(problemShape))),
          activationTileM_(activationTileM),
          blockNum_(blockNum)
    {
        if constexpr (IsLayered && !IsShared) {
            uint32_t logicalCoreCount = blockNum_;
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
            GmmKernel::BlockScheduler gmmBlockScheduler(
                {m_, k_, n_}, GmmKernel::BlockScheduler::Params{
                                  Te::MakeCoord(static_cast<int64_t>(activationTileM_), static_cast<int64_t>(L1_TILE_N))});
            uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
            WaitUntilGmFlagEquals(activationToGmm2Flag_, static_cast<int32_t>(targetLoops));
            upstreamReady_ = true;
        }
    }

    __aicore__ inline void NotifyAivTileDone(uint32_t mLoc)
    {
        if constexpr (NotifyCombineTileReady && !IsShared) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmToEpilogueFlag_, ++(*gmTileSequence_));
        } else if constexpr (IsLayered && !IsShared) {
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

    __aicore__ inline void WaitUpstreamTileDone()
    {
        int32_t expectedReadySequence = *gmTileSequence_ + 1;
        WaitUntilGmFlagAtLeast(gmmToEpilogueFlag_, expectedReadySequence);
        *gmTileSequence_ = expectedReadySequence;
    }

    __aicore__ inline void WaitAivTileDone(uint32_t) {}
    __aicore__ inline void WaitAicTileDone(uint32_t) {}
    __aicore__ inline void NotifyAicTileDone(uint32_t) {}
    __aicore__ inline void NotifyDownstreamReady(uint32_t) {}

private:
    __gm__ int32_t *activationToGmm2Flag_;
    __gm__ int32_t *gmm2CombineSyncCounter_;
    __gm__ int32_t *sharedExpertGmm2TileCounter_;
    __gm__ int32_t *gmmToEpilogueFlag_;
    int32_t *gmTileSequence_;
    GroupSyncSlotLayout groupSyncSlotLayout_{};
    uint32_t m_;
    uint32_t k_;
    uint32_t n_;
    uint32_t activationTileM_;
    uint32_t blockNum_;
    uint32_t lastWaveWaited_ = static_cast<uint32_t>(-1);
    bool upstreamReady_ = false;
};

// 场景：GMM2 A8W4
template <bool IsLayered, bool IsShared, bool IsWaveFlagGrained, bool IsGmm1Interleaved,
          bool NotifyCombineTileReady = false>
class Gmm2A8W4SyncPolicy {
    static_assert(!IsShared || !NotifyCombineTileReady, "Shared expert GMM2 has its own completion counter");
    static_assert(!IsLayered || !NotifyCombineTileReady, "Layered GMM2 selects its synchronization through IsLayered");
public:
    __aicore__ inline Gmm2A8W4SyncPolicy(const GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape,
                                         uint32_t activationTileM, uint32_t blockNum, uint32_t expertTokenCount,
                                         uint32_t rowOffsetInExpert, int32_t *gmTileSequence = nullptr)
        : activationToGmm2Flag_(gmmAddrInfo.activationToGmm2Flag),
          gmm2CombineSyncCounter_(gmmAddrInfo.gmm2CombineSyncCounter),
          sharedExpertGmm2TileCounter_(gmmAddrInfo.sharedExpertGmm2TileCounter),
          gmmToEpilogueFlag_(gmmAddrInfo.gmmToEpilogueFlag),
          gmTileSequence_(gmTileSequence),
          rowOffsetInExpert_(rowOffsetInExpert),
          m_(static_cast<uint32_t>(Get<M_VALUE>(problemShape))),
          k_(static_cast<uint32_t>(Get<N_VALUE>(problemShape)) / ACTIVATION_N_HALF),
          n_(static_cast<uint32_t>(Get<K_VALUE>(problemShape))),
          activationTileM_(activationTileM),
          blockNum_(blockNum)
    {
        if constexpr (IsLayered && !IsShared) {
            groupSyncSlotLayout_ = CalcGroupSyncSlotLayout(expertTokenCount, blockNum_);
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
            GmmKernel::BlockScheduler gmmBlockScheduler(
                {m_, k_, n_}, GmmKernel::BlockScheduler::Params{
                                  Te::MakeCoord(static_cast<int64_t>(activationTileM_), static_cast<int64_t>(L1_TILE_N))});
            uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
            WaitUntilGmFlagEquals(activationToGmm2Flag_, static_cast<int32_t>(targetLoops));
            upstreamReady_ = true;
        }
    }

    __aicore__ inline void NotifyAivTileDone(uint32_t mLoc)
    {
        if constexpr (NotifyCombineTileReady && !IsShared) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmToEpilogueFlag_, ++(*gmTileSequence_));
        } else if constexpr (IsLayered && !IsShared) {
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

    __aicore__ inline void WaitUpstreamTileDone()
    {
        int32_t expectedReadySequence = *gmTileSequence_ + 1;
        WaitUntilGmFlagAtLeast(gmmToEpilogueFlag_, expectedReadySequence);
        *gmTileSequence_ = expectedReadySequence;
    }

    __aicore__ inline void WaitAivTileDone(uint32_t) {}
    __aicore__ inline void WaitAicTileDone(uint32_t) {}
    __aicore__ inline void NotifyAicTileDone(uint32_t) {}
    __aicore__ inline void NotifyDownstreamReady(uint32_t) {}

private:
    __gm__ int32_t *activationToGmm2Flag_;
    __gm__ int32_t *gmm2CombineSyncCounter_;
    __gm__ int32_t *sharedExpertGmm2TileCounter_;
    __gm__ int32_t *gmmToEpilogueFlag_;
    int32_t *gmTileSequence_;
    GroupSyncSlotLayout groupSyncSlotLayout_{};
    uint32_t rowOffsetInExpert_;
    uint32_t m_;
    uint32_t k_;
    uint32_t n_;
    uint32_t activationTileM_;
    uint32_t blockNum_;
    uint32_t lastWaveWaited_ = static_cast<uint32_t>(-1);
    bool upstreamReady_ = false;
};

} // namespace MegaMoeImpl

#endif // MOE_GMM_SYNC_POLICY_H
