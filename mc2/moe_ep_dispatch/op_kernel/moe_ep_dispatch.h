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

// ─── Window Slot 内存布局 ─────────────────────────────────────────────
// 每个 slot 按 perSlotBytes 对齐布局：
//   Offset 0                        : token_data    (hidden × sizeof(bf16),   ALIGN 16/32)
//   Offset hidden_bytes_aligned      : scale_factors (仅 FP8 路径, bf16=0,   ALIGN 16)
//   Offset sf_bytes_aligned          : metadata      (见下方字段,             ALIGN 16)
//   Offset metadata_bytes_aligned    : mbarrier      (8B, hybrid PC用,        ALIGN 16→16B)
//   total = perSlotBytes
//
// metadata 字段：
//   topk_idx[K]           : K × 4B   (bf16训练 K=8 → 32B)
//   topk_weights[K]       : K × 4B   (32B)
//   src_rank_id           : 4B
//   src_token_idx         : 4B
//   target_scaleup_mask   : 4B       (仅 hybrid, bf16训练 ALIGN 后 metadata=80B)
//
// ─── Window 累计寻址 ─────────────────────────────────────────────────
// Direct 模式：
//   slot_addr = Window_base + (src_rank × nmt + slot_idx) × perSlotBytes
// Hybrid 模式（三段）：
//   scaleup_recv:  [src_scaleup_rank][src_scaleout_rank × nmt + slot] × perSlotBytes
//   scaleout_send: 本 NPU Stage 1 staging
//   scaleout_recv: [src_scaleout_rank][slot] × perSlotBytes, 每 slot 带 mbarrier

template <uint32_t DO_CPU_SYNC, uint32_t IS_CACHED, uint32_t NETWORK_MODE>
class KernelMoeEpDispatch {
public:
    __aicore__ inline KernelMoeEpDispatch() {};

    // ─── Init ──────────────────────────────────────────────────────────
    // Pseudocode:
    //   tpipe_ = pipe; td_ = tilingData; aivId_ = GetBlockIdx(); actualA_ = 0
    //   绑定 GM tensor: xGm_, topkIdxGm_, topkWeightsGm_, dstSlotIdxGm_,
    //                    recvSrcMetaInternalGm_, numRecvPerRankHpGm_, numRecvPerExpertHpGm_
    //   IS_CACHED=1: 绑定 cachedSlotIdxGm_
    //   读取 HCCL Window context: winContext_ = reinterpret_cast<HcclOpParam*>(workspace)
    //   分配 UB 缓冲:
    //     ubXIn_:    ALIGN_UP(hidden × 2, 32)     — 单 token 输入 bf16
    //     ubSlot_[2]: ALIGN_UP(hidden × 2, 32) × 2 — RecvPhase 双缓冲
    //     ubMeta_:   256B                           — metadata 中转
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
        GM_ADDR cachedNumPerExpert, GM_ADDR cachedSlotIdx, GM_ADDR cachedSrcMeta,
        GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchInfo* tilingData);

    // ─── Process ──────────────────────────────────────────────────────
    // Pseudocode:
    //   NETWORK_MODE==0 (Direct, ep≤128):
    //     1. ResetCounters() → SyncAll<true>()
    //     2. IS_CACHED==0: SendPhaseFirstTime() | IS_CACHED==1: SendPhaseCached()
    //     3. HcclBarrier()
    //     4. RecvPhase()
    //     5. DO_CPU_SYNC==1: WriteActualAToHostPinned() (blockIdx=0, aivId=0)
    //
    //   NETWORK_MODE==1 (Hybrid, ep>128, AIV三组并发):
    //     Group A [0, s1End):   Stage1 跨超节点 URMA
    //       IS_CACHED==0: DedupAndSend_Hybrid_Stage1(t)
    //       IS_CACHED==1: SendPhaseScaleoutCached
    //     Group B [s1End, s2End): Stage2 本超节点 fan-out (proxy poll mbarrier → mask扇出)
    //     Group C [s2End, total):  RecvPhase + metadata填表 + 计数累加
    //     SyncAll<true>() + HcclBarrier()
    //     DO_CPU_SYNC==1: WriteActualAToHostPinned() (单写者)
    __aicore__ inline void Process();

private:
    // ─── ResetCounters ──────────────────────────────────────────────
    // Pseudocode:
    //   清零 workspace 中 scaleup_counter[0..ep-1] + actualA 全局累加器
    //   for i in [aivId, ep, step=aivNum]: counter[i] = 0
    //   aivId==0: actualABase = 0
    __aicore__ inline void ResetCounters();

    // ─── SendPhaseFirstTime ────────────────────────────────────────
    // Pseudocode:
    //   tStart = aivId * numTokens / aivNum; tEnd = (aivId+1) * numTokens / aivNum
    //   for t in [tStart, tEnd): DedupAndSendDirect(t)
    __aicore__ inline void SendPhaseFirstTime();

    // ─── SendPhaseCached ──────────────────────────────────────────
    // Pseudocode:
    //   for t in [tStart, tEnd):
    //     读 x[t] → ubXIn_
    //     for k in [0, topK):
    //       slot = cachedSlotIdx[t,k]; if slot==-1: continue
    //       dst_rank = expert / numLocalExperts
    //       跳过 atomic! slot 已由 handle 固定
    //       WriteSlotToRemoteWindow(dst_rank, my_rank, slot, t, k)
    __aicore__ inline void SendPhaseCached();

    // ─── DedupAndSendDirect ────────────────────────────────────────
    // Pseudocode:
    //   核心: 按 dst_rank 去重 (scalar register seen-list, O(topK²))
    //   同一 token 发往同一 dst_rank 只发1份 → Window 不乘 topK
    //   读 x[t] → ubXIn_
    //   for k in [0, topK):
    //     expert = topkIdx[t,k]; if expert==-1: dstSlotIdx=-1; continue
    //     dst_rank = expert / numLocalExperts
    //     search seen_dst → hit?
    //       YES: dstSlotIdx[t,k] = seen_slot[hit] (复用, 不发新副本)
    //       NO:  slot = SetAtomicAdd(&counter[dst_rank], 1)
    //            seen_dst[n_seen] = dst_rank; seen_slot[n_seen] = slot; n_seen++
    //            dstSlotIdx[t,k] = slot
    //            WriteSlotToRemoteWindow(dst_rank, my_rank, slot, t, k)
    //
    //   WriteSlotToRemoteWindow 内部:
    //     dstWinBase = GetBaseWindAddrByRankId(winContext_, dst_rank, my_rank)
    //     slotAddr = dstWinBase + (my_rank * nmt + slot) * perSlotBytes
    //     写 token_data: DataCopyPad → PipeBarrier<PIPE_ALL>
    //     写 metadata: topk_idx[K], topk_weights[K](float raw bits),
    //                  src_rank_id=my_rank, src_token_idx=t
    //     DataCacheCleanAndInvalid flush L2
    __aicore__ inline void DedupAndSendDirect(uint32_t t);

    // ─── RecvPhase (Direct 模式) ──────────────────────────────────
    // Pseudocode:
    //   actualA_ = 0
    //   分核: AIV[i] 处理 src_rank ∈ [i*ep/aivNum, (i+1)*ep/aivNum)
    //   for sr in [srStart, srEnd):
    //     numRecvFromSrc = counter[sr]
    //     numRecvPerRankHp[sr] = numRecvFromSrc
    //     srcRankWinBase = localWinBase + sr * nmt * perSlotBytes
    //     for slot in [0, numRecvFromSrc):
    //       a = SetAtomicAdd(&actualABase, 1)  ← 全局累加分配 A 轴位置
    //       slotAddr = srcRankWinBase + slot * perSlotBytes
    //       读 token data: DataCopyPad(ubSlot_[0], slotAddr, hidden)
    //       读 metadata: DataCopy(ubMeta_, metaPtr, topK*2+2)
    //       src_rank_id    = ubMeta_[topK*2]
    //       src_token_idx  = ubMeta_[topK*2+1]
    //       src_token_global = src_rank_id * nmt + src_token_idx
    //       recv_src_metadata_internal[a*3+0] = src_token_global  ← 全局ID
    //       recv_src_metadata_internal[a*3+1] = src_rank_id
    //       recv_src_metadata_internal[a*3+2] = slot
    //       AtomicAdd(numRecvPerExpert[local_expert_id], 1)
    //   SyncAll<true>()
    __aicore__ inline void RecvPhase();

    // ─── WriteActualAToHostPinned ─────────────────────────────────
    // Pseudocode:
    //   单写者 (blockIdx=0, aivId=0)
    //   hpPtr = td_->hostPinnedCounterAddr
    //   *hpPtr = actualA_
    //   DataCacheCleanAndInvalid<int32_t> bypass L2 写入 host-pinned dev-mapped 地址
    __aicore__ inline void WriteActualAToHostPinned();

    // ─── AIV 分核计算 ──────────────────────────────────────────────
    // Pseudocode:
    //   ComputePerAivTokenStart: return aivId_ * numTokens / aivNum
    //   ComputePerAivTokenEnd:   return (aivId_+1) * numTokens / aivNum
    __aicore__ inline uint32_t ComputePerAivTokenStart();
    __aicore__ inline uint32_t ComputePerAivTokenEnd();

    // ─── Hybrid Stage 1: DedupAndSend_Hybrid_Stage1 ────────────────────
    // 按 dst_scaleout_rank 去重, 每 slot 带 target_scaleup_mask 位图
    // Pseudocode:
    //   for k in [0, topK):
    //     dst_rank = expert / numLocalExperts
    //     dst_scaleout = dst_rank / numScaleupRanks  ← 超节点ID
    //     dst_scaleup  = dst_rank % numScaleupRanks  ← 超节点内rank
    //     search seen_so list → hit?
    //       YES: mask |= (1<<dst_scaleup); dstSlotIdx = seen_so_slot[hit]
    //       NO:  slot = SetAtomicAdd(&scaleout_counter[dst_scaleout], 1)
    //            mask = (1<<dst_scaleup); n_so++
    //   for j in [0, n_so):
    //     URMA write-with-notify: token+meta+mask+notify 一体发送

    // ─── Hybrid Stage 2: proxy fan-out ──────────────────────────────────
    // Pseudocode:
    //   poll scaleout_recv 每 slot 的 mbarrier (细粒度 producer-consumer)
    //   读 target_scaleup_mask → 按 mask 位图扇出:
    //     while (mask) { dst_su = CTZ(mask); mask &= mask-1;
    //       su_slot = SetAtomicAdd(&scaleup_counter[dst_su], 1)
    //       HCCS write: scaleout_recv slot → 本 SN dst_su 的 scaleup_recv }

    // ─── proxy_rank 计算 ────────────────────────────────────────────────
    // proxy_rank(so) = so × num_scaleup_ranks + my_scaleup_rank_id

    // ─── GM tensor: 输入/输出 ──────────────────────────────────────────
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

    // ─── UB buffer ─────────────────────────────────────────────────────
    LocalTensor<bfloat16_t> ubXIn_;
    LocalTensor<bfloat16_t> ubSlot_[2];
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<float> ubAccFp32_;

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