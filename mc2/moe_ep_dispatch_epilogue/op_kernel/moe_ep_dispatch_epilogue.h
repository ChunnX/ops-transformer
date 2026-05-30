#ifndef MOE_EP_DISPATCH_EPILOGUE_H
#define MOE_EP_DISPATCH_EPILOGUE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_epilogue_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "moe_ep_dispatch_epilogue_tiling_data.h"

namespace MoeEpDispatchEpilogueImpl {

using namespace AscendC;

static constexpr uint32_t WIN_ADDR_ALIGN = 512;

class KernelDispatchEpilogue {
public:
    __aicore__ inline KernelDispatchEpilogue() {};

    __aicore__ inline void Init(
        GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
        GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData)
    {
        tpipe_ = pipe;
        td_ = tilingData;
        aivId_ = GetBlockIdx();
        workspace_ = workspace;

        dstSlotIdxGm_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx,
                                        td_->cfg.numTokens * td_->cfg.topK);
        recvSrcMetaInternalGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadataInternal,
                                                 td_->cfg.epWorldSize * td_->cfg.numMaxTokensPerRank * 3);
        numRecvPerRankHpGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp,
                                               td_->cfg.epWorldSize);
        numRecvPerExpertHpGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp,
                                                td_->numLocalExperts);
        recvXGm_.SetGlobalBuffer((__gm__ bfloat16_t*)recvX,
                                   td_->aAlloc * td_->cfg.hidden);
        recvTopkIdxGm_.SetGlobalBuffer((__gm__ int32_t*)recvTopkIdx,
                                         td_->aAlloc * td_->cfg.topK);
        recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)recvTopkWeights,
                                             td_->aAlloc * td_->cfg.topK);
        recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata,
                                             td_->aAlloc * 3);
        numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank,
                                            td_->cfg.epWorldSize);
        numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert,
                                              td_->numLocalExperts);

        winContext_ = reinterpret_cast<__gm__ Mc2Kernel::HcclOpParam*>(workspace);

        uint32_t ubTokenBytes  = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), 32);
        uint32_t ubMetaBytes   = 256;
        uint32_t ubPfxBytes    = (td_->cfg.epWorldSize + 1) * sizeof(int32_t);

        tpipe_->InitBuffer(ubTokenBuf_[0], ubTokenBytes);
        tpipe_->InitBuffer(ubTokenBuf_[1], ubTokenBytes);
        tpipe_->InitBuffer(ubMetaBuf_,     ubMetaBytes);
        tpipe_->InitBuffer(ubPrefixSumBuf_, ubPfxBytes);

        ubToken_[0]  = ubTokenBuf_[0].Get<bfloat16_t>();
        ubToken_[1]  = ubTokenBuf_[1].Get<bfloat16_t>();
        ubMeta_      = ubMetaBuf_.Get<int32_t>();
        ubPrefixSum_ = ubPrefixSumBuf_.Get<int32_t>();
    }

    __aicore__ inline void Process()
    {
        ComputePrefixSum();
        SyncAll<true>();

        CopyFromWindow();
        SyncAll<true>();

        if (aivId_ == 0) {
            CopyHostPinnedToDevice();
        }
    }

private:
    __aicore__ inline void ComputePrefixSum()
    {
        __gm__ int32_t* hpRankPtr = reinterpret_cast<__gm__ int32_t*>(td_->numRecvPerRankAddr);

        for (uint32_t i = 0; i < td_->cfg.epWorldSize; ++i) {
            ubPrefixSum_.SetValue(i, *reinterpret_cast<__gm__ int32_t*>(hpRankPtr + i));
        }

        int32_t acc = 0;
        for (uint32_t i = 0; i < td_->cfg.epWorldSize; ++i) {
            int32_t val = ubPrefixSum_.GetValue(i);
            acc += val;
            ubPrefixSum_.SetValue(i, acc);
        }

        GM_ADDR pfxWorkspace = workspace_ + sizeof(__gm__ Mc2Kernel::HcclOpParam);
        for (uint32_t i = 0; i < td_->cfg.epWorldSize; ++i) {
            *reinterpret_cast<__gm__ int32_t*>(pfxWorkspace + i * sizeof(int32_t))
                = ubPrefixSum_.GetValue(i);
        }
    }

    __aicore__ inline void CopyFromWindow()
    {
        uint32_t my_rank = Mc2Kernel::GetRankId(winContext_);
        GM_ADDR pfxWorkspace = workspace_ + sizeof(__gm__ Mc2Kernel::HcclOpParam);

        uint32_t srStart = aivId_ * td_->cfg.epWorldSize / td_->aivNum;
        uint32_t srEnd   = (aivId_ + 1) * td_->cfg.epWorldSize / td_->aivNum;

        GM_ADDR localWinBase = Mc2Kernel::GetBaseWindAddrByRankId(winContext_, my_rank, my_rank);

        for (uint32_t sr = srStart; sr < srEnd; ++sr) {
            int32_t base = *reinterpret_cast<__gm__ int32_t*>(pfxWorkspace + sr * sizeof(int32_t));
            __gm__ int32_t* hpRankPtr = reinterpret_cast<__gm__ int32_t*>(td_->numRecvPerRankAddr);
            int32_t cnt = *reinterpret_cast<__gm__ int32_t*>(hpRankPtr + sr);

            GM_ADDR srcRankWinBase = localWinBase + sr * td_->cfg.numMaxTokensPerRank
                                     * td_->cfg.perSlotBytes;

            for (int32_t slotIdx = 0; slotIdx < cnt; ++slotIdx) {
                uint32_t a = base + slotIdx;
                if (a >= td_->aAlloc) break;

                GM_ADDR slotAddr = srcRankWinBase + slotIdx * td_->cfg.perSlotBytes;

                DataCopyPad(ubToken_[slotIdx % 2],
                    reinterpret_cast<GlobalTensor<bfloat16_t>&>(
                        *reinterpret_cast<__gm__ bfloat16_t*>(slotAddr)),
                    {td_->cfg.hidden, 0, 0, 0, false});
                DataCopyPad(recvXGm_[a * td_->cfg.hidden], ubToken_[slotIdx % 2],
                            {td_->cfg.hidden, 0, 0, 0, false});

                uint32_t metaOffset = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), WIN_ADDR_ALIGN);
                __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(slotAddr + metaOffset);
                DataCopy(ubMeta_, reinterpret_cast<GlobalTensor<int32_t>&>(*metaPtr),
                         td_->cfg.topK * 2 + 2);

                for (uint32_t kk = 0; kk < td_->cfg.topK; ++kk) {
                    recvTopkIdxGm_.SetValue(a * td_->cfg.topK + kk, ubMeta_.GetValue(kk));
                }

                __gm__ float* weightsOutPtr = reinterpret_cast<__gm__ float*>(
                    &recvTopkWeightsGm_[a * td_->cfg.topK]);
                for (uint32_t kk = 0; kk < td_->cfg.topK; ++kk) {
                    int32_t rawBits = ubMeta_.GetValue(td_->cfg.topK + kk);
                    *reinterpret_cast<__gm__ int32_t*>(&weightsOutPtr[kk]) = rawBits;
                }

                int32_t src_rank_id   = ubMeta_.GetValue(td_->cfg.topK * 2);
                int32_t src_token_idx = ubMeta_.GetValue(td_->cfg.topK * 2 + 1);
                int32_t src_token_global = src_rank_id * td_->cfg.numMaxTokensPerRank + src_token_idx;

                recvSrcMetadataGm_.SetValue(a * 3 + 0, src_token_global);
                recvSrcMetadataGm_.SetValue(a * 3 + 1, src_rank_id);
                recvSrcMetadataGm_.SetValue(a * 3 + 2, slotIdx);
            }
        }
    }

    __aicore__ inline void CopyHostPinnedToDevice()
    {
        __gm__ int32_t* hpRankSrc = reinterpret_cast<__gm__ int32_t*>(td_->numRecvPerRankAddr);
        for (uint32_t i = 0; i < td_->cfg.epWorldSize; ++i) {
            numRecvPerRankGm_.SetValue(i, *reinterpret_cast<__gm__ int32_t*>(hpRankSrc + i));
        }

        __gm__ int64_t* hpExpertSrc = reinterpret_cast<__gm__ int64_t*>(td_->numRecvPerExpertAddr);
        for (uint32_t i = 0; i < td_->numLocalExperts; ++i) {
            numRecvPerExpertGm_.SetValue(i, *reinterpret_cast<__gm__ int64_t*>(hpExpertSrc + i));
        }
    }

    TPipe* tpipe_{nullptr};
    const MoeEpDispatchEpilogueInfo* td_{nullptr};
    __gm__ Mc2Kernel::HcclOpParam* winContext_{nullptr};

    GlobalTensor<int32_t> dstSlotIdxGm_;
    GlobalTensor<int32_t> recvSrcMetaInternalGm_;
    GlobalTensor<int32_t> numRecvPerRankHpGm_;
    GlobalTensor<int64_t> numRecvPerExpertHpGm_;

    GlobalTensor<bfloat16_t> recvXGm_;
    GlobalTensor<int32_t> recvTopkIdxGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;

    LocalTensor<bfloat16_t> ubToken_[2];
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<int32_t> ubPrefixSum_;

    TBuf<QuePosition::VECIN> ubTokenBuf_[2];
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TBuf<QuePosition::VECIN> ubPrefixSumBuf_;

    GM_ADDR workspace_{nullptr};
    uint32_t aivId_{0};
};

}  // namespace MoeEpDispatchEpilogueImpl

#endif  // MOE_EP_DISPATCH_EPILOGUE_H