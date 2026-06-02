#ifndef MOE_EP_COMBINE_H
#define MOE_EP_COMBINE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_combine_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "mc2_moe_context.h"
#include "moe_ep_combine_tiling.h"

namespace MoeEpCombineImpl {

using namespace AscendC;
using Mc2Aclnn::Mc2MoeContext;
using Mc2Aclnn::HCCL_MAX_RANK_SIZE;

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];

__aicore__ inline GM_ADDR GetBaseWinAddrByRankId(uint32_t rankId)
{
    return (GM_ADDR)(winRankAddr_[rankId] + Mc2Kernel::A5_MTE_STATE_WIN_SIZE);
}

static constexpr uint32_t MAX_TOPK = 8;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;

template <uint32_t HAS_TOPK_WEIGHTS>
class KernelMoeEpCombine {
public:
    __aicore__ inline KernelMoeEpCombine() {};

__aicore__ inline void Init(
        GM_ADDR context, GM_ADDR x, GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadata,
        GM_ADDR numRecvPerRank, GM_ADDR topkWeights,
        GM_ADDR bias0, GM_ADDR bias1,
        GM_ADDR combinedX, GM_ADDR combinedTopkWeightsOptional,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpCombineInfo* tilingData)
    {
        tpipe_ = pipe;
        td_ = tilingData;
        aivId_ = GetBlockIdx();
        workspace_ = workspace;

        mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
        rankId_ = mc2Context_->epRankId;
        for (uint32_t i = 0; i < td_->cfg.epWorldSize; ++i) {
            winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
        }

        uint32_t A = td_->cfg.epWorldSize * td_->cfg.numMaxTokensPerRank;

        xGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x, A * td_->cfg.hidden);
        dstSlotIdxGm_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx,
                                        td_->cfg.numTokens * td_->cfg.topK);
        recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata, A * 3);
        numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank,
                                            td_->cfg.epWorldSize);
        combinedXGm_.SetGlobalBuffer((__gm__ bfloat16_t*)combinedX,
                                      td_->cfg.numTokens * td_->cfg.hidden);

        if constexpr (HAS_TOPK_WEIGHTS == 1) {
            topkWeightsGm_.SetGlobalBuffer((__gm__ float*)topkWeights,
                                             td_->cfg.numTokens * td_->cfg.topK);
            combinedTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)combinedTopkWeightsOptional,
                                                     td_->cfg.numTokens * td_->cfg.topK);
        }

        uint32_t ubXBytes      = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), 32);
        uint32_t ubAccBytes    = td_->cfg.hidden * sizeof(float);
        uint32_t ubTmpBytes    = td_->cfg.hidden * sizeof(float);
        uint32_t ubWeightBytes = td_->cfg.hidden * sizeof(float);

        tpipe_->InitBuffer(ubXBuf_[0],      ubXBytes);
        tpipe_->InitBuffer(ubXBuf_[1],      ubXBytes);
        tpipe_->InitBuffer(ubAccFp32Buf_,   ubAccBytes);
        tpipe_->InitBuffer(ubTmpFp32Buf_,   ubTmpBytes);
        if constexpr (HAS_TOPK_WEIGHTS == 1) {
            tpipe_->InitBuffer(ubWeightedBuf_, ubWeightBytes);
        }

        ubX_[0]      = ubXBuf_[0].Get<bfloat16_t>();
        ubX_[1]      = ubXBuf_[1].Get<bfloat16_t>();
        ubAccFp32_   = ubAccFp32Buf_.Get<float>();
        ubTmpFp32_   = ubTmpFp32Buf_.Get<float>();
        if constexpr (HAS_TOPK_WEIGHTS == 1) {
            ubWeighted_ = ubWeightedBuf_.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        SendPhaseExpertToToken();
        HcclBarrier();

        RecvPhaseStashing();

        RecvPhaseReduce();
    }

private:
    __aicore__ inline void SendPhaseExpertToToken()
    {
        uint32_t A = td_->cfg.epWorldSize * td_->cfg.numMaxTokensPerRank;

        uint32_t aStart = aivId_ * A / td_->aivNum;
        uint32_t aEnd   = (aivId_ + 1) * A / td_->aivNum;

        for (uint32_t a = aStart; a < aEnd; ++a) {
            int32_t src_token_global = recvSrcMetadataGm_.GetValue(a * 3 + 0);
            int32_t src_rank         = recvSrcMetadataGm_.GetValue(a * 3 + 1);
            int32_t src_slot         = recvSrcMetadataGm_.GetValue(a * 3 + 2);

            if (src_rank < 0) continue;

            GM_ADDR dstWinBase = GetBaseWinAddrByRankId(src_rank);
            GM_ADDR slotAddr = dstWinBase
                + (rankId_ * td_->cfg.numMaxTokensPerRank + src_slot)
                * td_->cfg.perSlotBytes;

            DataCopyPad(ubX_[0], xGm_[a * td_->cfg.hidden], {td_->cfg.hidden, 0, 0, 0, false});

            DataCopyPad(
                reinterpret_cast<GlobalTensor<bfloat16_t>&>(
                    *reinterpret_cast<__gm__ bfloat16_t*>(slotAddr)),
                ubX_[0], {td_->cfg.hidden, 0, 0, 0, false});
            PipeBarrier<PIPE_ALL>();

            if constexpr (HAS_TOPK_WEIGHTS == 1) {
                cacheWriteThrough(reinterpret_cast<__gm__ uint8_t*>(slotAddr),
                                  td_->cfg.hidden * sizeof(bfloat16_t));
            } else {
                cacheWriteThrough(reinterpret_cast<__gm__ uint8_t*>(slotAddr),
                                  td_->cfg.hidden * sizeof(bfloat16_t));
            }
        }
    }

    __aicore__ inline void RecvPhaseStashing()
    {
        uint32_t tStart = aivId_ * td_->cfg.numTokens / td_->aivNum;
        uint32_t tEnd   = (aivId_ + 1) * td_->cfg.numTokens / td_->aivNum;

        GM_ADDR localWinBase = GetBaseWinAddrByRankId(rankId_);
        GM_ADDR stashBase = workspace_;

        for (uint32_t t = tStart; t < tEnd; ++t) {
            for (uint32_t k = 0; k < td_->cfg.topK; ++k) {
                int32_t slot = dstSlotIdxGm_.GetValue(t * td_->cfg.topK + k);
                if (slot < 0) continue;

                uint32_t numValidRanks = 0;
                GM_ADDR slotAddr = nullptr;

                for (uint32_t src_r = 0; src_r < td_->cfg.epWorldSize; ++src_r) {
                    int32_t cnt = numRecvPerRankGm_.GetValue(src_r);
                    if (slot < cnt) {
                        slotAddr = localWinBase
                            + src_r * td_->cfg.numMaxTokensPerRank * td_->cfg.perSlotBytes
                            + slot * td_->cfg.perSlotBytes;
                        break;
                    }
                }

                if (slotAddr == nullptr) continue;

                DataCopyPad(ubX_[k % 2],
                    reinterpret_cast<GlobalTensor<bfloat16_t>&>(
                        *reinterpret_cast<__gm__ bfloat16_t*>(slotAddr)),
                    {td_->cfg.hidden, 0, 0, 0, false});

                GM_ADDR stashAddr = stashBase
                    + ((t - tStart) * td_->cfg.topK + k) * td_->cfg.hidden * sizeof(bfloat16_t);
                DataCopyPad(
                    reinterpret_cast<GlobalTensor<bfloat16_t>&>(
                        *reinterpret_cast<__gm__ bfloat16_t*>(stashAddr)),
                    ubX_[k % 2], {td_->cfg.hidden, 0, 0, 0, false});
            }
        }

        SyncAll<true>();
    }

    __aicore__ inline void RecvPhaseReduce()
    {
        uint32_t tStart = aivId_ * td_->cfg.numTokens / td_->aivNum;
        uint32_t tEnd   = (aivId_ + 1) * td_->cfg.numTokens / td_->aivNum;

        GM_ADDR stashBase = workspace_;

        for (uint32_t t = tStart; t < tEnd; ++t) {
            Init(ubAccFp32_, (float)0, td_->cfg.hidden);

            for (uint32_t k = 0; k < td_->cfg.topK; ++k) {
                int32_t slot = dstSlotIdxGm_.GetValue(t * td_->cfg.topK + k);
                if (slot < 0) continue;

                GM_ADDR stashAddr = stashBase
                    + ((t - tStart) * td_->cfg.topK + k) * td_->cfg.hidden * sizeof(bfloat16_t);
                DataCopyPad(ubX_[0],
                    reinterpret_cast<GlobalTensor<bfloat16_t>&>(
                        *reinterpret_cast<__gm__ bfloat16_t*>(stashAddr)),
                    {td_->cfg.hidden, 0, 0, 0, false});

                Cast(ubTmpFp32_, ubX_[0], RoundMode::CAST_NONE, td_->cfg.hidden);

                if constexpr (HAS_TOPK_WEIGHTS == 0) {
                    Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_, td_->cfg.hidden);
                } else {
                    float w = topkWeightsGm_.GetValue(t * td_->cfg.topK + k);
                    Muls(ubWeighted_, ubTmpFp32_, w, td_->cfg.hidden);
                    Add(ubAccFp32_, ubAccFp32_, ubWeighted_, td_->cfg.hidden);
                    combinedTopkWeightsGm_.SetValue(t * td_->cfg.topK + k, w);
                }
            }

            Cast(combinedXGm_[t * td_->cfg.hidden], ubAccFp32_,
                 RoundMode::CAST_RINT, td_->cfg.hidden);
        }
    }

    TPipe* tpipe_{nullptr};
    const MoeEpCombineInfo* td_{nullptr};
    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    uint32_t rankId_{0};

    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<int32_t> dstSlotIdxGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<float> topkWeightsGm_;

    GlobalTensor<bfloat16_t> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<bfloat16_t> ubX_[2];
    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<float> ubWeighted_;

    TBuf<QuePosition::VECIN> ubXBuf_[2];
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<QuePosition::VECIN> ubWeightedBuf_;

    GM_ADDR workspace_{nullptr};
    uint32_t aivId_{0};
};

}  // namespace MoeEpCombineImpl

#endif  // MOE_EP_COMBINE_H