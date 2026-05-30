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

// ─── Epilogue 核心职责 ───────────────────────────────────────────────
// 纯本地操作 (无跨 rank 通信, 无 DO_CPU_SYNC/IS_CACHED/NETWORK_MODE 分支):
//   1. Prefix sum: numRecvPerRank[ep] → offsets[ep+1]
//   2. CopyFromWindow: 从 Window scatter-read → 紧凑 recv_x 等
//   3. CopyHostPinnedToDevice: DMA host-pinned → device tensor

class KernelDispatchEpilogue {
public:
    __aicore__ inline KernelDispatchEpilogue() {};

    // ─── Init ──────────────────────────────────────────────────────
    // Pseudocode:
    //   tpipe_ = pipe; td_ = tilingData; aivId_ = GetBlockIdx()
    //   绑定 GM tensor:
    //     输入: dstSlotIdxGm_, recvSrcMetaInternalGm_, numRecvPerRankHpGm_, numRecvPerExpertHpGm_
    //     输出: recvXGm_[aAlloc*hidden], recvTopkIdxGm_[aAlloc*topK],
    //           recvTopkWeightsGm_[aAlloc*topK], recvSrcMetadataGm_[aAlloc*3],
    //           numRecvPerRankGm_[ep], numRecvPerExpertGm_[numLocalExperts]
    //   读取 HCCL Window context: winContext_ = reinterpret_cast<HcclOpParam*>(workspace)
    //   分配 UB 缓冲:
    //     ubToken_[2]: ALIGN_UP(hidden×2, 32) × 2 — 双缓冲 (Window读 / recv_x写 ping-pong)
    //     ubMeta_:     256B — metadata 中转
    //     ubPrefixSum_: (ep+1) × 4B — prefix sum 输出
    __aicore__ inline void Init(
        GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
        GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData);

    // ─── Process ──────────────────────────────────────────────────
    // Pseudocode:
    //   1. ComputePrefixSum() → SyncAll<true>()
    //   2. CopyFromWindow() → SyncAll<true>()
    //   3. aivId==0: CopyHostPinnedToDevice()
    __aicore__ inline void Process();

private:
    // ─── ComputePrefixSum ──────────────────────────────────────────
    // Pseudocode:
    //   读 host-pinned (device-mapped) numRecvPerRank[ep]: hpRankPtr = td_->numRecvPerRankAddr
    //   for i in [0, ep): ubPrefixSum_[i] = *(hpRankPtr + i)
    //   InclusiveScan: acc=0; for i in [0, ep): acc += val; ubPrefixSum_[i] = acc
    //   写 workspace 中 prefixSum 供跨 AIV 共享:
    //     pfxWorkspace = workspace + sizeof(HcclOpParam)
    //     for i in [0, scanLen): *(pfxWorkspace + i*4) = ubPrefixSum_[i]
    __aicore__ inline void ComputePrefixSum();

    // ─── CopyFromWindow ──────────────────────────────────────────
    // Pseudocode:
    //   分核: 每 AIV 处理 src_rank ∈ [srStart, srEnd)
    //   localWinBase = GetBaseWindAddrByRankId(winContext_, my_rank, my_rank)
    //   for sr in [srStart, srEnd):
    //     base = prefixSum[sr] (从 workspace 读)
    //     cnt  = numRecvPerRank[sr] (从 host-pinned 读)
    //     srcRankWinBase = localWinBase + sr * nmt * perSlotBytes
    //     for slot in [0, cnt):
    //       a = base + slot  ← recv_x 目的行号
    //       if a >= aAlloc: break (越界保护)
    //       slotAddr = srcRankWinBase + slot * perSlotBytes
    //       读 token data: DataCopyPad(ubToken_[slot%2], slotAddr, hidden)
    //       写 recv_x[a]: DataCopyPad(recvXGm_[a*hidden], ubToken_[slot%2], hidden)
    //       读 metadata: DataCopy(ubMeta_, metaPtr, topK*2+2)
    //       写 recv_topk_idx[a, topK]: ubMeta_[0..K-1]
    //       写 recv_topk_weights[a, topK]: float raw bits from ubMeta_[K..2K-1]
    //       写 recv_src_metadata[a, 3]:
    //         src_rank_id   = ubMeta_[2K]
    //         src_token_idx = ubMeta_[2K+1]
    //         src_token_global = src_rank_id * nmt + src_token_idx  ← 全局ID
    //         recvSrcMetadataGm_[a*3+0] = src_token_global
    //         recvSrcMetadataGm_[a*3+1] = src_rank_id
    //         recvSrcMetadataGm_[a*3+2] = slot
    __aicore__ inline void CopyFromWindow();

    // ─── CopyHostPinnedToDevice ──────────────────────────────────
    // Pseudocode:
    //   DMA: host-pinned numRecvPerRank[ep] → device tensor (handle字段)
    //   DMA: host-pinned numRecvPerExpert[numLocalExperts] → device tensor (handle字段)
    //   hpRankSrc  = td_->numRecvPerRankAddr;   DataCopy(numRecvPerRankGm_, hpRankSrc, ep)
    //   hpExpertSrc = td_->numRecvPerExpertAddr; DataCopy(numRecvPerExpertGm_, hpExpertSrc, numLocalExperts)
    __aicore__ inline void CopyHostPinnedToDevice();

    // ─── 成员变量 ────────────────────────────────────────────────
    TPipe* tpipe_{nullptr};
    const MoeEpDispatchEpilogueInfo* td_{nullptr};
    __gm__ Mc2Kernel::HcclOpParam* winContext_{nullptr};

    // GM tensor: 输入 (来自 Dispatch)
    GlobalTensor<int32_t> dstSlotIdxGm_;
    GlobalTensor<int32_t> recvSrcMetaInternalGm_;
    GlobalTensor<int32_t> numRecvPerRankHpGm_;
    GlobalTensor<int64_t> numRecvPerExpertHpGm_;

    // GM tensor: 输出 (组成 EPHandle)
    GlobalTensor<bfloat16_t> recvXGm_;
    GlobalTensor<int32_t> recvTopkIdxGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;

    // UB buffer
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