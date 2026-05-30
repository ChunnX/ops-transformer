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

#include "moe_ep_combine_tiling_data.h"

namespace MoeEpCombineImpl {

using namespace AscendC;

static constexpr uint32_t MAX_TOPK = 8;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;

// ─── Combine 两个模板变体 ────────────────────────────────────────────
// C-A (HAS_TOPK_WEIGHTS=0): 纯 sum, 用于 forward (用户已在 FFN 后乘过权重)
//   combined_x[t] = Σ_k Cast(stash[t,k], fp32) → Cast→bf16
// C-B (HAS_TOPK_WEIGHTS=1): 加权归约, 用于 dispatch backward
//   combined_x[t] = Σ_k stash[t,k] × weight[t,k]
//   额外输出 combined_topk_weights[t,k]

template <uint32_t HAS_TOPK_WEIGHTS>
class KernelMoeEpCombine {
public:
    __aicore__ inline KernelMoeEpCombine() {};

    // ─── Init ──────────────────────────────────────────────────────
    // Pseudocode:
    //   tpipe_ = pipe; td_ = tilingData; aivId_ = GetBlockIdx()
    //   A = ep × nmt (上界)
    //   绑定 GM tensor: xGm_[A*hidden], dstSlotIdxGm_[numTokens*topK],
    //                    recvSrcMetadataGm_[A*3], numRecvPerRankGm_[ep],
    //                    combinedXGm_[numTokens*hidden]
    //   HAS_TOPK_WEIGHTS==1:
    //     topkWeightsGm_[numTokens*topK], combinedTopkWeightsGm_[numTokens*topK]
    //   winContext_ = reinterpret_cast<HcclOpParam*>(workspace)
    //   分配 UB 缓冲:
    //     ubX_[2]:       ALIGN_UP(hidden×2, 32) × 2 — 双缓冲 (Window读 / stash写)
    //     ubAccFp32_:    hidden × 4 — FP32 累加器
    //     ubTmpFp32_:    hidden × 4 — FP32 临时
    //     ubWeighted_:   hidden × 4 — 仅 C-B, 权重×activation 结果
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadata,
        GM_ADDR numRecvPerRank, GM_ADDR topkWeights,
        GM_ADDR bias0, GM_ADDR bias1,
        GM_ADDR combinedX, GM_ADDR combinedTopkWeightsOptional,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpCombineInfo* tilingData);

    // ─── Process ──────────────────────────────────────────────────
    // Pseudocode:
    //   1. SendPhaseExpertToToken()  ← 反向路由 RDMA 写回原始 rank
    //   2. HcclBarrier()             ← 所有 rank 发送完成
    //   3. RecvPhaseStashing()       ← Window → workspace stash
    //   4. RecvPhaseReduce()         ← FP32 累加 → combined_x
    __aicore__ inline void Process();

private:
    // ─── SendPhaseExpertToToken ──────────────────────────────────
    // Pseudocode:
    //   反向路由: 按 recv_src_metadata[a] 把 x[a] 写回原始 rank 的 Window slot
    //   my_rank = GetRankId(winContext_)
    //   A = ep × nmt
    //   分核: 每 AIV 处理 [aStart, aEnd) 行
    //   for a in [aStart, aEnd):
    //     src_token_global = recvSrcMetadata[a*3+0]
    //     src_rank         = recvSrcMetadata[a*3+1]
    //     src_slot         = recvSrcMetadata[a*3+2]
    //     if src_rank < 0: continue (上界空行跳过)
    //     dstWinBase = GetBaseWindAddrByRankId(winContext_, src_rank, my_rank)
    //     slotAddr = dstWinBase + (my_rank * nmt + src_slot) * perSlotBytes
    //     读 x[a] → ubX_[0]
    //     写 token data 到目标 Window: DataCopyPad(slotAddr, ubX_[0], hidden)
    //     PipeBarrier<PIPE_ALL>
    //     C-B: DataCacheCleanAndInvalid flush metadata 区域
    __aicore__ inline void SendPhaseExpertToToken();

    // ─── RecvPhaseStashing ──────────────────────────────────────
    // Pseudocode:
    //   从本地 Window 收集到达的 expert 输出, 按 (t,k) stash 到 workspace
    //   my_rank = GetRankId(winContext_)
    //   localWinBase = GetBaseWindAddrByRankId(winContext_, my_rank, my_rank)
    //   stashBase = workspace + sizeof(HcclOpParam)
    //   分核: 每 AIV 处理 [tStart, tEnd) tokens
    //   for t in [tStart, tEnd):
    //     for k in [0, topK):
    //       slot = dstSlotIdxGm_[t*topK+k]; if slot<0: continue
    //       从 Window 读 slot 数据 → ubX_[k%2]
    //       stashAddr = stashBase + ((t-tStart)*topK+k) * hidden * sizeof(bf16)
    //       写 stash[t-tStart][k] = ubX_[k%2]
    __aicore__ inline void RecvPhaseStashing();

    // ─── RecvPhaseReduce ────────────────────────────────────────
    // Pseudocode:
    //   FP32 累加: combined_x[t] = Σ_k Cast(stash[t,k], fp32) → Cast→bf16
    //   C-B: 额外 combined_topk_weights[t,k] 归约
    //   分核: 每 AIV 处理 [tStart, tEnd) tokens
    //   stashBase = workspace + sizeof(HcclOpParam)
    //   for t in [tStart, tEnd):
    //     Init(ubAccFp32_, 0, hidden)  ← FP32 累加器清零
    //     for k in [0, topK):
    //       slot = dstSlotIdxGm_[t*topK+k]; if slot<0: continue
    //       读 stash[t-tStart][k] → ubX_[0] (bf16)
    //       Cast(ubTmpFp32_, ubX_[0], CAST_NONE)  ← bf16→fp32
    //       C-A: Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_)  ← 纯Σ累加
    //       C-B: Muls(ubWeighted_, ubTmpFp32_, w, hidden) ← 乘权重
    //            Add(ubAccFp32_, ubAccFp32_, ubWeighted_)  ← 加权累加
    //            combinedTopkWeightsGm_[t*topK+k] = w
    //     Cast(combinedXGm_[t*hidden], ubAccFp32_, CAST_RINT)  ← fp32→bf16
    __aicore__ inline void RecvPhaseReduce();

    // ─── 成员变量 ────────────────────────────────────────────────
    TPipe* tpipe_{nullptr};
    const MoeEpCombineInfo* td_{nullptr};
    __gm__ Mc2Kernel::HcclOpParam* winContext_{nullptr};

    // GM tensor: 输入
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<int32_t> dstSlotIdxGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<float> topkWeightsGm_;

    // GM tensor: 输出
    GlobalTensor<bfloat16_t> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    // UB buffer
    LocalTensor<bfloat16_t> ubX_[2];
    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<float> ubWeighted_;             // 仅 C-B

    TBuf<QuePosition::VECIN> ubXBuf_[2];
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<QuePosition::VECIN> ubWeightedBuf_;     // 仅 C-B

    GM_ADDR workspace_{nullptr};
    uint32_t aivId_{0};
};

}  // namespace MoeEpCombineImpl

#endif  // MOE_EP_COMBINE_H