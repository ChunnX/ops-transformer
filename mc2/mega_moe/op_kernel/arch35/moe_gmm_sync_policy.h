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
 *
 * 语义以 mega_moe_gmm1_activation.h 现有实现为唯一真值逐行移植；
 * 各方法注释标注了被封装的原始函数与行号，接线（wiring）后原函数删除。
 *
 * 状态成员说明：
 *   vecSetSyncCom / pingpongIdx / gmm1TileReadySequence 以引用持有——
 *   它们是调用方（GmmRuntimeState / wave 局部）的跨专家流水状态，
 *   且 wave 层 EndSync 在 GMM 返回后仍消费这些值，必须原地更新。
 *   lastWaveWaited 为单个专家循环内的局部去重状态，构造时重置。
 */
template <bool IsShared, bool IsGmm1Interleaved, bool IsWaveFlagGrained = false,
          bool TopkWeightsPrefetch = false>
class Gmm1GenericSyncPolicy {
public:
    template <typename Config>
    __aicore__ inline Gmm1GenericSyncPolicy(const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t expertIdx,
                                            int32_t &vecSetSyncCom, uint16_t &pingpongIdx)
        : dispatchToGmm1Flag_(gmmAddrInfo.dispatchToGmm1Flag),
          activationToGmm2Flag_(gmmAddrInfo.activationToGmm2Flag),
          gmm1TileStatus_(gmmAddrInfo.gmm1TileStatus),
          expertIdx_(expertIdx),
          vecSetSyncCom_(vecSetSyncCom),
          pingpongIdx_(pingpongIdx),
          m_(config.m),
          tileM_(config.tileM)
    {
    }

    // 原始逻辑：两条 AIC 循环共用的 wave 去重 + WaitForGmm1InputReady<IsShared>
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

    // 原始逻辑：按 TopkWeightsPrefetch 分叉。
    // UB 模式（非 Prefetch）：Gmm1AicMmadGeneric 的 AIC 侧 UB 防覆盖等待
    //   （gmm1_activation.h:246-252；交织时第 3 个 tile 起等 AIV，非交织时第 2 个 tile 起等）。
    // GM 模式（Prefetch）：写 GM 无需等待（原 Gmm1AicPrefetchMmadGeneric 无此逻辑），编译为空。
    // 调用方在循环固定位置（计算前）统一调用，模式差异由本方法内部吸收。
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

    // 原始逻辑：Gmm1AicMmadGeneric 传给 Gmm1AicMmadTileToUbGeneric 的 pingpongIdx
    __aicore__ inline uint16_t GetPingpongIdx() const
    {
        return pingpongIdx_;
    }

    // 原始逻辑：按 TopkWeightsPrefetch 分叉。
    // UB 模式（非 Prefetch）：AIC 侧通知 + 计数/乒乓递推（gmm1_activation.h:258-265）。
    // GM 模式（Prefetch）：FIX_S 落盘 + NotifyGmm1TileStatus（gmm1_activation.h:204-206；
    //   FIX_S 与通知紧耦合——先等 MMAD 结果落 GM 再发布状态，故整体封装）。
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

    // 原始逻辑：按 TopkWeightsPrefetch 分叉。
    // UB 模式（非 Prefetch）：Gmm1Aiv0EpilogueGeneric 的 AIV0 等待 AIC 写完 UB
    //   （gmm1_activation.h:637-638 / 644）。
    // GM 模式（Prefetch）：WaitGmm1TileStatus（gmm1_activation.h:592-597，
    //   由 Gmm1Aiv0PrefetchEpilogueGeneric 调用，等 expertIdx+1 轮次标记）。
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

    // 原始逻辑：按 TopkWeightsPrefetch 分叉。
    // UB 模式（非 Prefetch）：Gmm1Aiv0EpilogueGeneric 的 AIV0 通知 AIC 可复用 UB + 乒乓递推
    //   （gmm1_activation.h:641-642 / 647）。
    // GM 模式（Prefetch）：写 GM 无需 AIV 通知 AIC（原 Gmm1Aiv0PrefetchEpilogueGeneric 无此逻辑）。
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

    // 原始逻辑：NotifyGmm2InputReady<IsWaveFlagGrained>（gmm1_activation.h:60-70，
    // 由各 epilogue tile 函数调用，slotIdx = mLoc / L1_TILE_M_256；两模式同源，无分叉）。
    // 注意：偏移在局部拷贝上计算——原函数 flagBase 按值传入，每次从基地址重算；
    // epilogue 子 tile 循环会带不同 mLoc 多次调用，成员不可累积偏移。
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

// 场景：A8W4 全路径（AIC 写 GM，AIV1 消费）。
// TopkWeightsPrefetch 选择通知通道（还原原始 Gmm1AicMmadA8W4 的函数内分叉，gmm1_activation.h:298-302）：
//   true  → gmm1TileStatus[loopIdx] = expertIdx+1（与 Gmm1GmSyncPolicy 同源，两通道各自冻结、互不演化）
//   false → gmmToEpilogueFlag 序号递增 + AIV1 WaitUntilGmFlagAtLeast
// 构造同时接收两通道参数（expertIdx / gmm1TileReadySequence），各分支 if constexpr 择用。
template <bool IsShared, bool IsWaveFlagGrained = false, bool TopkWeightsPrefetch = false>
class Gmm1A8W4SyncPolicy {
public:
    template <typename Config>
    __aicore__ inline Gmm1A8W4SyncPolicy(const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t expertIdx,
                                         int32_t &gmm1TileReadySequence)
        : dispatchToGmm1Flag_(gmmAddrInfo.dispatchToGmm1Flag),
          activationToGmm2Flag_(gmmAddrInfo.activationToGmm2Flag),
          gmmToEpilogueFlag_(gmmAddrInfo.gmmToEpilogueFlag),
          gmm1TileStatus_(gmmAddrInfo.gmm1TileStatus),
          expertIdx_(expertIdx),
          gmm1TileReadySequence_(gmm1TileReadySequence),
          m_(config.m),
          tileM_(config.tileM)
    {
    }

    // 原始逻辑：Gmm1AicMmadA8W4 内 wave 去重 + WaitForGmm1InputReady<IsShared>
    //（gmm1_activation.h:287-291 / 33-43）。
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

    // 原始逻辑：FIX_S 落盘 + 按 TopkWeightsPrefetch 分叉通知（gmm1_activation.h:296-302）。
    // Prefetch → NotifyGmm1TileStatus（gmm1TileStatus = expertIdx+1，与 Gmm1GmSyncPolicy 同源）；
    // 非 Prefetch → WriteGmByPassDCache(gmmToEpilogueFlag, ++序号)。
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

    // 写 GM 路径无需 AIC 等待 AIV。
    __aicore__ inline void WaitAivTileDone(uint32_t) {}

    // 原始逻辑：按 TopkWeightsPrefetch 分叉等待（gmm1_activation.h:592-597 / 765-769）。
    // Prefetch → WaitGmm1TileStatus（等 expertIdx+1）；非 Prefetch → 序号 WaitUntilGmFlagAtLeast + 推进。
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

    // 写 GM 路径无需 AIV 通知 AIC。
    __aicore__ inline void NotifyAicTileDone(uint32_t) {}

    // 原始逻辑：NotifyGmm2InputReady（原 Gmm1UbSyncPolicy 同源，gmm1_activation.h:60-70；
    // 偏移在局部拷贝上计算，成员不可累积偏移）。
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

/*
 * GMM2 同步策略：封装 GMM2 两条执行路径的 MoE 专属同步逻辑。
 * 语义以 mega_moe_gmm2_combine.h 现有实现为唯一真值逐行移植，方法注释标注原始行号。
 *
 * 与 GMM1 的关键差异（2920 真值，勿按 0827 改写）：
 * 1. 无 AIV 内联 epilogue（2920 已删除，Combine 由 wave 层独立调度）——
 *    故无需 gm 序号同步，Wait/NotifyAic/Aiv 的 AIV 侧方法全部 noop。
 * 2. 通知条件不含 CombineQuantMode（wave 量化的 combine 计数已移交 wave 层）：
 *    layered 非共享走 per-group AtomicAdd，仅共享专家走 shared 计数器。
 * 3. 非 wave-grained 等待：原始仅在 loopIdx == startLoopIdx 首迭代等待一次，
 *    且 targetLoops 用 BlockScheduler tile 数、与 mLoc 无关——
 *    首调用锁存（upstreamReady_）与原始严格等价。
 */

// 场景：GMM2 Generic（AIC 写 GM；layered/共享专家/普通 wave 均走此路径）。
template <bool IsLayered, bool IsShared, bool IsWaveFlagGrained, bool IsGmm1Interleaved>
class Gmm2UbSyncPolicy {
public:
    template <typename Config>
    __aicore__ inline Gmm2UbSyncPolicy(const GMMAddrInfo &gmmAddrInfo, const Config &config)
        : activationToGmm2Flag_(gmmAddrInfo.activationToGmm2Flag),
          gmm2CombineSyncCounter_(gmmAddrInfo.gmm2CombineSyncCounter),
          sharedExpertGmm2TileCounter_(gmmAddrInfo.sharedExpertGmm2TileCounter),
          m_(config.m),
          k_(config.k),
          n_(config.n),
          activationTileM_(config.activationTileM),
          blockNum_(config.blockNum)
    {
        // 原始逻辑：Gmm2AicMmadGeneric 循环外的 layout 计算（gmm2_combine.h:549-556）。
        // 非 wave-grained 时 logicalCoreCount *= 2 是 2920 真实语义（:552-554），勿删。
        if constexpr (IsLayered && !IsShared) {
            uint32_t logicalCoreCount = blockNum_;
            if constexpr (!IsWaveFlagGrained) {
                logicalCoreCount *= 2U;
            }
            groupSyncSlotLayout_ = CalcGroupSyncSlotLayout(m_, logicalCoreCount);
        }
    }

    // 原始逻辑：WaitForGmm2InputReady<IsShared>（gmm2_combine.h:487-511）。
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
            // 原始：仅 loopIdx == startLoopIdx 首迭代等待一次；targetLoops 与 mLoc 无关。
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

    // 原始逻辑：NotifyCombineConsumersOfTileCompletion / NotifySharedExpertTileCompletion
    //（gmm2_combine.h:514-528 / 531-538）。FIX_S 与通知紧耦合（先落盘再计数），整体封装。
    __aicore__ inline void NotifyAivTileDone(uint32_t mLoc)
    {
        if constexpr (IsLayered && !IsShared) {
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

    __aicore__ inline void WaitAivTileDone(uint32_t) {}
    __aicore__ inline void WaitAicTileDone(uint32_t) {}
    __aicore__ inline void NotifyAicTileDone(uint32_t) {}
    __aicore__ inline void NotifyDownstreamReady(uint32_t) {}

private:
    __gm__ int32_t *activationToGmm2Flag_;
    __gm__ int32_t *gmm2CombineSyncCounter_;
    __gm__ int32_t *sharedExpertGmm2TileCounter_;
    GroupSyncSlotLayout groupSyncSlotLayout_{};
    uint32_t m_;
    uint32_t k_;
    uint32_t n_;
    uint32_t activationTileM_;
    uint32_t blockNum_;
    uint32_t lastWaveWaited_ = static_cast<uint32_t>(-1);
    bool upstreamReady_ = false;
};

// 场景：GMM2 A8W4（AIC 写 GM；Combine 由调用路径独立调度）。
// 与 Generic 的三点差异（均为 2920 真值，逐字保留）：
// 1. layout 用 expertTokenCount 而非 config.m，且无 ×2 分支（gmm2_combine.h:611-614）；
// 2. layered 通知实参为 rowOffsetInExpert + mLoc（:642）——紧凑 token 按累计行偏移寻址；
// 3. shared 通知无 !IsLayered 守卫（:644，与 Generic 的 :595 不对称，原样保留）。
template <bool IsLayered, bool IsShared, bool IsWaveFlagGrained, bool IsGmm1Interleaved>
class Gmm2A8W4SyncPolicy {
public:
    template <typename Config>
    __aicore__ inline Gmm2A8W4SyncPolicy(const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                         uint32_t expertTokenCount, uint32_t rowOffsetInExpert)
        : activationToGmm2Flag_(gmmAddrInfo.activationToGmm2Flag),
          gmm2CombineSyncCounter_(gmmAddrInfo.gmm2CombineSyncCounter),
          sharedExpertGmm2TileCounter_(gmmAddrInfo.sharedExpertGmm2TileCounter),
          rowOffsetInExpert_(rowOffsetInExpert),
          m_(config.m),
          k_(config.k),
          n_(config.n),
          activationTileM_(config.activationTileM),
          blockNum_(config.blockNum)
    {
        if constexpr (IsLayered && !IsShared) {
            groupSyncSlotLayout_ = CalcGroupSyncSlotLayout(expertTokenCount, blockNum_);
        }
    }

    // 原始逻辑：WaitForGmm2InputReady<IsShared>（gmm2_combine.h:487-511，与 Generic 同源）。
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

    // 原始逻辑：NotifyCombineConsumersOfTileCompletion / NotifySharedExpertTileCompletion
    //（gmm2_combine.h:641-646）。注意与 Generic 的两点差异见类注释。
    __aicore__ inline void NotifyAivTileDone(uint32_t mLoc)
    {
        if constexpr (IsLayered && !IsShared) {
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

    __aicore__ inline void WaitAivTileDone(uint32_t) {}
    __aicore__ inline void WaitAicTileDone(uint32_t) {}
    __aicore__ inline void NotifyAicTileDone(uint32_t) {}
    __aicore__ inline void NotifyDownstreamReady(uint32_t) {}

private:
    __gm__ int32_t *activationToGmm2Flag_;
    __gm__ int32_t *gmm2CombineSyncCounter_;
    __gm__ int32_t *sharedExpertGmm2TileCounter_;
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
