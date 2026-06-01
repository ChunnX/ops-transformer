#ifndef MOE_EP_DISPATCH_H
#define MOE_EP_DISPATCH_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "moe_ep_dispatch_tiling_data.h"

namespace MoeEpDispatchImpl {

using namespace AscendC;

static constexpr uint32_t MAX_TOPK = 8;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;

template <uint32_t DO_CPU_SYNC, uint32_t IS_CACHED, uint32_t NETWORK_MODE>
class KernelMoeEpDispatch {
public:
    __aicore__ inline KernelMoeEpDispatch() {};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
        GM_ADDR cachedSlotIdx,
        GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchInfo* tilingData)
    {
        tpipe_ = pipe;
        td_ = tilingData;
        aivId_ = GetBlockIdx();
        blockIdx_ = GetBlockIdx();
        actualA_ = 0;
        workspace_ = workspace;

        xGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x, td_->cfg.numTokens * td_->cfg.hidden);
        topkIdxGm_.SetGlobalBuffer((__gm__ int32_t*)topkIdx, td_->cfg.numTokens * td_->cfg.topK);
        topkWeightsGm_.SetGlobalBuffer((__gm__ float*)topkWeights, td_->cfg.numTokens * td_->cfg.topK);
        dstSlotIdxGm_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx, td_->cfg.numTokens * td_->cfg.topK);
        recvSrcMetaInternalGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadataInternal,
                                                 td_->cfg.epWorldSize * td_->cfg.numMaxTokensPerRank * 3);
        numRecvPerRankHpGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp, td_->cfg.epWorldSize);
        numRecvPerExpertHpGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp, td_->cfg.numLocalExperts);
        if constexpr (IS_CACHED == 1) {
            cachedSlotIdxGm_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx,
                                               td_->cfg.numTokens * td_->cfg.topK);
        }

        winContext_ = reinterpret_cast<__gm__ Mc2Kernel::HcclOpParam*>(workspace);

        uint32_t ubXInBytes    = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), 32);
        uint32_t ubSlotBytes   = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), 32);
        uint32_t ubMetaBytes   = 256;

        tpipe_->InitBuffer(ubXBuf_,      ubXInBytes);
        tpipe_->InitBuffer(ubSlotBuf_[0], ubSlotBytes);
        tpipe_->InitBuffer(ubSlotBuf_[1], ubSlotBytes);
        tpipe_->InitBuffer(ubMetaBuf_,    ubMetaBytes);

        ubXIn_     = ubXBuf_.Get<bfloat16_t>();
        ubSlot_[0] = ubSlotBuf_[0].Get<bfloat16_t>();
        ubSlot_[1] = ubSlotBuf_[1].Get<bfloat16_t>();
        ubMeta_    = ubMetaBuf_.Get<int32_t>();
    }

    __aicore__ inline void Process()
    {
        if constexpr (NETWORK_MODE == 0) {
            Process_Direct();
        } else {
            Process_Hybrid();
        }
    }

private:
    __aicore__ inline void Process_Direct()
    {
        ResetCounters();
        SyncAll<true>();

        if constexpr (IS_CACHED == 0) {
            SendPhaseFirstTime();
        } else {
            SendPhaseCached();
        }

        HcclBarrier();

        RecvPhase();

        if constexpr (DO_CPU_SYNC == 1) {
            if (blockIdx_ == 0 && aivId_ == 0) {
                WriteActualAToHostPinned();
            }
        }
    }

    __aicore__ inline void Process_Hybrid()
    {
        const uint32_t aiv = aivId_;
        const uint32_t s1End = td_->numAivStage1;
        const uint32_t s2End = s1End + td_->numAivStage2;

        if (aiv < s1End) {
            if constexpr (IS_CACHED == 0) {
                SendPhaseScaleoutFirstTime();
            } else {
                SendPhaseScaleoutCached();
            }
        }
        else if (aiv < s2End) {
            SendPhaseScaleupFanout();
        }
        else {
            RecvPhase();
        }

        SyncAll<true>();
        HcclBarrier();

        if constexpr (DO_CPU_SYNC == 1) {
            if (blockIdx_ == 0 && aivId_ == 0) {
                WriteActualAToHostPinned();
            }
        }
    }

    __aicore__ inline void ResetCounters()
    {
        GM_ADDR counterBase = workspace_ + sizeof(__gm__ Mc2Kernel::HcclOpParam);
        GM_ADDR actualABase = counterBase + td_->cfg.epWorldSize * sizeof(int32_t);

        for (uint32_t i = aivId_; i < td_->cfg.epWorldSize; i += td_->aivNum) {
            *reinterpret_cast<__gm__ int32_t*>(counterBase + i * sizeof(int32_t)) = 0;
        }
        if (aivId_ == 0) {
            *reinterpret_cast<__gm__ int32_t*>(actualABase) = 0;
        }

        if constexpr (NETWORK_MODE == 1) {
            GM_ADDR scaleoutCounterBase = counterBase + td_->cfg.epWorldSize * sizeof(int32_t) + sizeof(int32_t);
            for (uint32_t i = aivId_; i < td_->numScaleoutRanks; i += td_->aivNum) {
                *reinterpret_cast<__gm__ int32_t*>(scaleoutCounterBase + i * sizeof(int32_t)) = 0;
            }
        }
    }

    __aicore__ inline void SendPhaseFirstTime()
    {
        uint32_t tStart = ComputePerAivTokenStart();
        uint32_t tEnd   = ComputePerAivTokenEnd();

        for (uint32_t t = tStart; t < tEnd; ++t) {
            DedupAndSendDirect(t);
        }
    }

    __aicore__ inline void DedupAndSendDirect(uint32_t t)
    {
        int32_t  seen_dst[MAX_TOPK];
        int32_t  seen_slot[MAX_TOPK];
        uint32_t n_seen = 0;

        DataCopyPad(ubXIn_, xGm_[t * td_->cfg.hidden], {td_->cfg.hidden, 0, 0, 0, false});

        for (uint32_t k = 0; k < td_->cfg.topK; ++k) {
            int32_t expert = topkIdxGm_.GetValue(t * td_->cfg.topK + k);
            if (expert == -1) {
                dstSlotIdxGm_.SetValue(t * td_->cfg.topK + k, -1);
                continue;
            }

            uint32_t dst_rank = expert / td_->cfg.numLocalExperts;

            int32_t hit = -1;
            for (uint32_t j = 0; j < n_seen; ++j) {
                if (seen_dst[j] == (int32_t)dst_rank) { hit = j; break; }
            }
            if (hit >= 0) {
                dstSlotIdxGm_.SetValue(t * td_->cfg.topK + k, seen_slot[hit]);
                continue;
            }

            uint32_t slot = SetAtomicAdd<int32_t>(
                reinterpret_cast<__gm__ int32_t*>(
                    workspace_ + sizeof(__gm__ Mc2Kernel::HcclOpParam) + dst_rank * sizeof(int32_t)),
                1);
            seen_dst[n_seen]  = dst_rank;
            seen_slot[n_seen] = slot;
            n_seen++;
            dstSlotIdxGm_.SetValue(t * td_->cfg.topK + k, slot);

            WriteSlotToRemoteWindow(dst_rank, slot, t);
        }
    }

    __aicore__ inline void WriteSlotToRemoteWindow(uint32_t dst_rank, uint32_t slot, uint32_t t)
    {
        uint32_t my_rank = Mc2Kernel::GetRankId(winContext_);
        GM_ADDR dstWinBase = Mc2Kernel::GetBaseWindAddrByRankId(winContext_, dst_rank, my_rank);
        GM_ADDR slotAddr = dstWinBase + (my_rank * td_->cfg.numMaxTokensPerRank + slot) * td_->cfg.perSlotBytes;

        DataCopyPad(reinterpret_cast<GlobalTensor<bfloat16_t>&>(
            *reinterpret_cast<__gm__ bfloat16_t*>(slotAddr)),
            ubXIn_, {td_->cfg.hidden, 0, 0, 0, false});
        PipeBarrier<PIPE_ALL>();

        uint32_t metaOffset = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), WIN_ADDR_ALIGN);
        __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(slotAddr + metaOffset);

        for (uint32_t kk = 0; kk < td_->cfg.topK; ++kk) {
            ubMeta_.SetValue(kk, topkIdxGm_.GetValue(t * td_->cfg.topK + kk));
        }
        for (uint32_t kk = 0; kk < td_->cfg.topK; ++kk) {
            float w = topkWeightsGm_.GetValue(t * td_->cfg.topK + kk);
            ubMeta_.SetValue(td_->cfg.topK + kk, *reinterpret_cast<int32_t*>(&w));
        }
        ubMeta_.SetValue(td_->cfg.topK * 2, my_rank);
        ubMeta_.SetValue(td_->cfg.topK * 2 + 1, t);

        DataCopy(reinterpret_cast<GlobalTensor<int32_t>&>(*metaPtr), ubMeta_,
                 td_->cfg.topK * 2 + 2);
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t*>(metaPtr),
                          (td_->cfg.topK * 2 + 2) * sizeof(int32_t));
    }

    __aicore__ inline void SendPhaseCached()
    {
        uint32_t tStart = ComputePerAivTokenStart();
        uint32_t tEnd   = ComputePerAivTokenEnd();
        uint32_t my_rank = Mc2Kernel::GetRankId(winContext_);

        for (uint32_t t = tStart; t < tEnd; ++t) {
            DataCopyPad(ubXIn_, xGm_[t * td_->cfg.hidden], {td_->cfg.hidden, 0, 0, 0, false});

            for (uint32_t k = 0; k < td_->cfg.topK; ++k) {
                int32_t slot = cachedSlotIdxGm_.GetValue(t * td_->cfg.topK + k);
                if (slot == -1) continue;

                int32_t expert = topkIdxGm_.GetValue(t * td_->cfg.topK + k);
                uint32_t dst_rank = expert / td_->cfg.numLocalExperts;

                WriteSlotToRemoteWindow(dst_rank, slot, t);
            }
        }
    }

    __aicore__ inline void RecvPhase()
    {
        uint32_t my_rank = Mc2Kernel::GetRankId(winContext_);
        actualA_ = 0;

        uint32_t srStart = aivId_ * td_->cfg.epWorldSize / td_->aivNum;
        uint32_t srEnd   = (aivId_ + 1) * td_->cfg.epWorldSize / td_->aivNum;

        GM_ADDR actualABase = workspace_ + sizeof(__gm__ Mc2Kernel::HcclOpParam)
                              + td_->cfg.epWorldSize * sizeof(int32_t);
        GM_ADDR counterBase = workspace_ + sizeof(__gm__ Mc2Kernel::HcclOpParam);
        GM_ADDR localWinBase = Mc2Kernel::GetBaseWindAddrByRankId(winContext_, my_rank, my_rank);

        for (uint32_t sr = srStart; sr < srEnd; ++sr) {
            int32_t numRecvFromSrc = *reinterpret_cast<__gm__ int32_t*>(
                counterBase + sr * sizeof(int32_t));

            numRecvPerRankHpGm_.SetValue(sr, numRecvFromSrc);

            GM_ADDR srcRankBase = localWinBase + sr * td_->cfg.numMaxTokensPerRank * td_->cfg.perSlotBytes;

            for (int32_t slotIdx = 0; slotIdx < numRecvFromSrc; ++slotIdx) {
                uint32_t a = SetAtomicAdd<int32_t>(
                    reinterpret_cast<__gm__ int32_t*>(actualABase), 1);
                actualA_ = a + 1;

                GM_ADDR slotAddr = srcRankBase + slotIdx * td_->cfg.perSlotBytes;

                DataCopyPad(ubSlot_[slotIdx % 2],
                    reinterpret_cast<GlobalTensor<bfloat16_t>&>(
                        *reinterpret_cast<__gm__ bfloat16_t*>(slotAddr)),
                    {td_->cfg.hidden, 0, 0, 0, false});

                uint32_t metaOffset = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), WIN_ADDR_ALIGN);
                __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(slotAddr + metaOffset);
                DataCopy(ubMeta_, reinterpret_cast<GlobalTensor<int32_t>&>(*metaPtr),
                         td_->cfg.topK * 2 + 2);

                int32_t src_rank_id    = ubMeta_.GetValue(td_->cfg.topK * 2);
                int32_t src_token_idx  = ubMeta_.GetValue(td_->cfg.topK * 2 + 1);
                int32_t src_token_global = src_rank_id * td_->cfg.numMaxTokensPerRank + src_token_idx;

                recvSrcMetaInternalGm_.SetValue(a * 3 + 0, src_token_global);
                recvSrcMetaInternalGm_.SetValue(a * 3 + 1, src_rank_id);
                recvSrcMetaInternalGm_.SetValue(a * 3 + 2, slotIdx);

                int32_t expert_local_0 = ubMeta_.GetValue(0) % td_->cfg.numLocalExperts;
                SetAtomicAdd<int64_t>(
                    reinterpret_cast<__gm__ int64_t*>(&numRecvPerExpertHpGm_[expert_local_0]), 1);
            }
        }

        SyncAll<true>();
    }

    __aicore__ inline void WriteActualAToHostPinned()
    {
        __gm__ int32_t* hpPtr = reinterpret_cast<__gm__ int32_t*>(td_->hostPinnedCounterAddr);
        *hpPtr = static_cast<int32_t>(actualA_);
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t*>(hpPtr), sizeof(int32_t));
    }

    __aicore__ inline uint32_t ComputePerAivTokenStart()
    {
        return aivId_ * td_->cfg.numTokens / td_->aivNum;
    }

    __aicore__ inline uint32_t ComputePerAivTokenEnd()
    {
        return (aivId_ + 1) * td_->cfg.numTokens / td_->aivNum;
    }

    __aicore__ inline void SendPhaseScaleoutFirstTime()
    {
        uint32_t tStart = ComputePerAivTokenStart();
        uint32_t tEnd   = ComputePerAivTokenEnd();

        for (uint32_t t = tStart; t < tEnd; ++t) {
            DedupAndSend_Hybrid_Stage1(t);
        }
    }

    __aicore__ inline void DedupAndSend_Hybrid_Stage1(uint32_t t)
    {
        int32_t  seen_so[MAX_TOPK];
        int32_t  seen_so_slot[MAX_TOPK];
        uint32_t target_scaleup_mask[MAX_TOPK] = {0};
        uint32_t n_so = 0;

        DataCopyPad(ubXIn_, xGm_[t * td_->cfg.hidden], {td_->cfg.hidden, 0, 0, 0, false});

        for (uint32_t k = 0; k < td_->cfg.topK; ++k) {
            int32_t expert = topkIdxGm_.GetValue(t * td_->cfg.topK + k);
            if (expert == -1) {
                dstSlotIdxGm_.SetValue(t * td_->cfg.topK + k, -1);
                continue;
            }

            uint32_t dst_rank     = expert / td_->cfg.numLocalExperts;
            uint32_t dst_scaleout = dst_rank / td_->numScaleupRanks;
            uint32_t dst_scaleup  = dst_rank % td_->numScaleupRanks;

            int32_t hit = -1;
            for (uint32_t j = 0; j < n_so; ++j) {
                if (seen_so[j] == (int32_t)dst_scaleout) { hit = j; break; }
            }
            if (hit >= 0) {
                target_scaleup_mask[hit] |= (1u << dst_scaleup);
                dstSlotIdxGm_.SetValue(t * td_->cfg.topK + k, seen_so_slot[hit]);
                continue;
            }

            GM_ADDR scaleoutCounterBase = workspace_ + sizeof(__gm__ Mc2Kernel::HcclOpParam)
                                           + td_->cfg.epWorldSize * sizeof(int32_t)
                                           + sizeof(int32_t);
            uint32_t slot = SetAtomicAdd<int32_t>(
                reinterpret_cast<__gm__ int32_t*>(scaleoutCounterBase + dst_scaleout * sizeof(int32_t)),
                1);
            seen_so[n_so]          = dst_scaleout;
            seen_so_slot[n_so]     = slot;
            target_scaleup_mask[n_so] = (1u << dst_scaleup);
            dstSlotIdxGm_.SetValue(t * td_->cfg.topK + k, slot);
            n_so++;

            uint32_t proxy_rank = dst_scaleout * td_->numScaleupRanks + (Mc2Kernel::GetRankId(winContext_) % td_->numScaleupRanks);
            WriteSlotToRemoteScaleoutWindow(proxy_rank, slot, t, target_scaleup_mask[n_so - 1]);
        }
    }

    __aicore__ inline void WriteSlotToRemoteScaleoutWindow(uint32_t proxy_rank, uint32_t slot, uint32_t t, uint32_t mask)
    {
        uint32_t my_rank = Mc2Kernel::GetRankId(winContext_);
        GM_ADDR dstWinBase = Mc2Kernel::GetBaseWindAddrByRankId(winContext_, proxy_rank, my_rank);
        GM_ADDR slotAddr = dstWinBase + (my_rank * td_->cfg.numMaxTokensPerRank + slot) * td_->cfg.perSlotBytes;

        DataCopyPad(reinterpret_cast<GlobalTensor<bfloat16_t>&>(
            *reinterpret_cast<__gm__ bfloat16_t*>(slotAddr)),
            ubXIn_, {td_->cfg.hidden, 0, 0, 0, false});
        PipeBarrier<PIPE_ALL>();

        uint32_t metaOffset = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), WIN_ADDR_ALIGN);
        __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(slotAddr + metaOffset);

        for (uint32_t kk = 0; kk < td_->cfg.topK; ++kk) {
            ubMeta_.SetValue(kk, topkIdxGm_.GetValue(t * td_->cfg.topK + kk));
        }
        for (uint32_t kk = 0; kk < td_->cfg.topK; ++kk) {
            float w = topkWeightsGm_.GetValue(t * td_->cfg.topK + kk);
            ubMeta_.SetValue(td_->cfg.topK + kk, *reinterpret_cast<int32_t*>(&w));
        }
        ubMeta_.SetValue(td_->cfg.topK * 2, my_rank);
        ubMeta_.SetValue(td_->cfg.topK * 2 + 1, t);
        ubMeta_.SetValue(td_->cfg.topK * 2 + 2, mask);

        DataCopy(reinterpret_cast<GlobalTensor<int32_t>&>(*metaPtr), ubMeta_,
                 td_->cfg.topK * 2 + 3);
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t*>(metaPtr),
                          (td_->cfg.topK * 2 + 3) * sizeof(int32_t));
    }

    __aicore__ inline void SendPhaseScaleoutCached()
    {
        uint32_t tStart = ComputePerAivTokenStart();
        uint32_t tEnd   = ComputePerAivTokenEnd();
        uint32_t my_rank = Mc2Kernel::GetRankId(winContext_);

        for (uint32_t t = tStart; t < tEnd; ++t) {
            DataCopyPad(ubXIn_, xGm_[t * td_->cfg.hidden], {td_->cfg.hidden, 0, 0, 0, false});

            for (uint32_t k = 0; k < td_->cfg.topK; ++k) {
                int32_t slot = cachedSlotIdxGm_.GetValue(t * td_->cfg.topK + k);
                if (slot == -1) continue;

                int32_t expert = topkIdxGm_.GetValue(t * td_->cfg.topK + k);
                uint32_t dst_rank = expert / td_->cfg.numLocalExperts;
                uint32_t dst_scaleout = dst_rank / td_->numScaleupRanks;
                uint32_t proxy_rank = dst_scaleout * td_->numScaleupRanks + (my_rank % td_->numScaleupRanks);

                WriteSlotToRemoteWindow(proxy_rank, slot, t);
            }
        }
    }

    __aicore__ inline void SendPhaseScaleupFanout()
    {
        uint32_t my_rank = Mc2Kernel::GetRankId(winContext_);
        uint32_t my_scaleup_rank = my_rank % td_->numScaleupRanks;
        GM_ADDR localWinBase = Mc2Kernel::GetBaseWindAddrByRankId(winContext_, my_rank, my_rank);

        for (uint32_t src_so = 0; src_so < td_->numScaleoutRanks; ++src_so) {
            GM_ADDR scaleoutRecvBase = localWinBase
                + td_->cfg.epWorldSize * td_->cfg.numMaxTokensPerRank * td_->cfg.perSlotBytes
                + td_->cfg.numMaxTokensPerRank * td_->cfg.perSlotBytes
                + src_so * (td_->cfg.numMaxTokensPerRank + 64) * td_->cfg.perSlotBytes;

            GM_ADDR scaleoutCounterBase = workspace_ + sizeof(__gm__ Mc2Kernel::HcclOpParam)
                                           + td_->cfg.epWorldSize * sizeof(int32_t)
                                           + sizeof(int32_t);
            int32_t count = *reinterpret_cast<__gm__ int32_t*>(scaleoutCounterBase + src_so * sizeof(int32_t));

            for (int32_t slotIdx = 0; slotIdx < count; ++slotIdx) {
                GM_ADDR slotAddr = scaleoutRecvBase + slotIdx * td_->cfg.perSlotBytes;

                uint32_t metaOffset = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), WIN_ADDR_ALIGN);
                __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(slotAddr + metaOffset);

                DataCopy(ubMeta_, reinterpret_cast<GlobalTensor<int32_t>&>(*metaPtr),
                         td_->cfg.topK * 2 + 3);

                uint32_t mask = ubMeta_.GetValue(td_->cfg.topK * 2 + 2);

                while (mask) {
                    uint32_t dst_su = __builtin_ctz(mask);
                    mask &= mask - 1;

                    uint32_t su_slot = SetAtomicAdd<int32_t>(
                        reinterpret_cast<__gm__ int32_t*>(
                            workspace_ + sizeof(__gm__ Mc2Kernel::HcclOpParam) + dst_su * sizeof(int32_t)),
                        1);

                    GM_ADDR scaleupRecvBase = localWinBase;
                    GM_ADDR scaleupDstAddr = scaleupRecvBase
                        + (dst_su * td_->numScaleoutRanks * td_->cfg.numMaxTokensPerRank + su_slot)
                        * td_->cfg.perSlotBytes;

                    DataCopyPad(ubSlot_[slotIdx % 2],
                        reinterpret_cast<GlobalTensor<bfloat16_t>&>(
                            *reinterpret_cast<__gm__ bfloat16_t*>(slotAddr)),
                        {td_->cfg.hidden, 0, 0, 0, false});
                    DataCopyPad(reinterpret_cast<GlobalTensor<bfloat16_t>&>(
                        *reinterpret_cast<__gm__ bfloat16_t*>(scaleupDstAddr)),
                        ubSlot_[slotIdx % 2], {td_->cfg.hidden, 0, 0, 0, false});
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

    TPipe* tpipe_{nullptr};
    const MoeEpDispatchInfo* td_{nullptr};
    __gm__ Mc2Kernel::HcclOpParam* winContext_{nullptr};

    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<int32_t> topkIdxGm_;
    GlobalTensor<float> topkWeightsGm_;
    GlobalTensor<int32_t> dstSlotIdxGm_;
    GlobalTensor<int32_t> recvSrcMetaInternalGm_;
    GlobalTensor<int32_t> numRecvPerRankHpGm_;
    GlobalTensor<int64_t> numRecvPerExpertHpGm_;
    GlobalTensor<int32_t> cachedSlotIdxGm_;

    LocalTensor<bfloat16_t> ubXIn_;
    LocalTensor<bfloat16_t> ubSlot_[2];
    LocalTensor<int32_t> ubMeta_;

    TBuf<QuePosition::VECIN> ubXBuf_;
    TBuf<QuePosition::VECIN> ubSlotBuf_[2];
    TBuf<QuePosition::VECIN> ubMetaBuf_;

    GM_ADDR workspace_{nullptr};
    uint32_t aivId_{0};
    uint32_t actualA_{0};
    uint32_t blockIdx_{0};
};

}  // namespace MoeEpDispatchImpl

#endif  // MOE_EP_DISPATCH_H