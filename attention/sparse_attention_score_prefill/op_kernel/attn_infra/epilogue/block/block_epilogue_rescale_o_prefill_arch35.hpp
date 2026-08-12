/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Phase2 combine epilogue for prefill (Arch35).
 *
 * Workspace layout matches IFA FlashDecode (separate buffers):
 *   accumOut  [totalQ, kvHead, topK, groupSize, D]
 *   softmaxMax / softmaxSum [totalQ, kvHead, topK, groupSize] (compact float)
 *
 * GM max/sum are compact; CopyLseIn expands to 32B-aligned UB for ComputeScaleValue_VF.
 * Unlike the IFA library, this operator's Phase1 leaves accumOut unnormalized (no /sum),
 * so Phase2 must divide each partial by its original per-row rowSum. That divide is
 * embedded as a regbase MicroAPI::Div (broadcast rowSum via DIST_BRC_B32) inside the
 * local copy of ReduceFinalRes_VF (arch35/vector_api/vf_flash_decode_msa.h). The per-row
 * rowSum is not re-read from GM: it reuses the compact rowSum already DataCopyPad'd into
 * broadCastTmp_ by CopyLseIn (lseSumTmpBuf = broadCastTmpBuf_[alignedLseCount],
 * layout [topK, dealRowCount]), sliced per split j. CopyAccumOutIn only copies accumOut.
 *
 * Pipeline: the per buf-group g loop is double-buffered so the MTE2 load of group g+1
 * overlaps the V reduce of group g (mirrors IFA fdMm2ResBuf1/2 at coarser grain). Each
 * group loads splitsPerBuf_ (or the tail) contiguous splits in ONE DataCopyPad
 * (CopyAccumOutIn), then the V reduce walks those splits from UB — the per-split j
 * MTE2 scatter reads (16x 8KB) were the Phase2 MTE2 bottleneck (39.5% of the op at
 * 24 GB/s); batching to ~splitsPerBuf_*8KB raises effective BW and drops per-transaction
 * overhead. accumOut is managed like the kernel's ubSTensor[ubSBufId]: a 2-deep array
 * accumOutBuf[bufId], bufId = g % ACCUM_OUT_STAGES, with two strided stages carved from
 * the free [192,256)KiB region, and event ids ACCUM_OUT_FLAG_BASE + bufId for both
 * MTE2_V and V_MTE2. Slot validity (left-contiguous NEG_INF_LSE) is checked in the
 * inner split loop; the first invalid slot breaks the whole combine. CopyLseIn
 * (MTE2_V EVENT_ID0) and CopyFinalResOut(V_MTE3 EVENT_ID0) stay per-chunk blocking
 * (MTE3 overlap out of scope: production single chunk).
 */
// =============================================================================
// 中文导读 (FlashDecode Phase2 combine 流程)
// =============================================================================
// 本文件实现 Phase2 的 combine 逻辑 (FlashDecode 路径), 用于将 topK 个 split 的
// 部分输出 O_partial[k] 加权融合为最终注意力输出 O。
//
// 核心公式:
//   O = Σ scale[k] * (O_partial[k] / rowSum[k])
// 其中 scale[k] 由各 split 的 rowMax / rowSum 经 softmax 归一化得到 (见
// ComputeScaleValue)。
//
// 与 IFA (Incremental Flash Attention) 库的区别:
//   - IFA 的 Phase1 阶段已经对 O_partial 做了 /rowSum 归一化, Phase2 只需加权求和。
//   - 本算子的 Phase1 不做 O/rowSum, 而是将除法推迟到 Phase2 的 CopyAccumOutIn 之后,
//     由 ReduceFinalRes 中嵌入的 regbase MicroAPI::Div 完成。
//   - 这样 Phase1 减少一次除法, 但 Phase2 需要额外读取 rowSum 并在归约时做逐行除法。
// =============================================================================

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"

using namespace AscendC;

#include "../../../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
// Local copy of the library ReduceFinalRes_VF family with the per-row regbase Div
// embedded (this operator's Phase1 leaves accumOut unnormalized). FP32-only path
// (<T, Tacc> with Tacc=float; the bf16 O_partial path uses the separate header
// below, do NOT add bf16 branches here — keeps the verified fp32 binary identical).
// Must follow the vf_flash_decode.h include above (relies on FLT_ZERO / MicroAPI).
#include "../../../arch35/vector_api/vf_flash_decode_msa.h"

namespace NpuArch::Epilogue::Block {

template <class InDtype, class Tws>
class BlockEpilogue<
    EpilogueRescaleOPrefillArch35,
    InDtype,
    Tws>
{
public:
    using DispatchPolicy = EpilogueRescaleOPrefillArch35;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = InDtype;
    using T = float;            // lse / rowSum / dst / cast compute (always fp32)
    // O_partial (workspace accumOut) dtype: float (fp32 path) or bfloat16_t (innerPrecise==1
    // bf16 path: PV fixpipe F322BF16 writes bf16, Phase2 MTE2 reads bf16, regbase-cast to fp32).
    using TwsO = Tws;

    static constexpr uint32_t FP32_ONE_BLOCK_SIZE = 8U;
    static constexpr uint32_t BYTE_BLOCK = 32U;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768U;
    static constexpr uint32_t BUF_32K_FLOATS = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192U;
    static constexpr uint32_t BUF_8K_FLOATS = BUFFER_SIZE_BYTE_8K / sizeof(T);
    static constexpr float NEG_INF_LSE = -3.4028235e38f;

    // accumOut ping-pong (mirrors the kernel ubSTensor[ubSBufId] style): a 2-deep array of
    // UB tensors indexed by bufId = g % ACCUM_OUT_STAGES (one ping-pong slot per buf-group,
    // each holding splitsPerBuf_ contiguous splits), each a strided stage, with event
    // ids derived as ACCUM_OUT_FLAG_BASE + bufId for both MTE2_V and V_MTE2 (distinct
    // HardEvent types share an id safely, per the IFA fdMm2ResBuf1/2 pattern).
    static constexpr uint32_t ACCUM_OUT_STAGES = 2U;
    static constexpr uint32_t ACCUM_OUT_FLAG_BASE = 3U;     // MTE2_V/V_MTE2 id = base + bufId

    __aicore__ inline BlockEpilogue() {}

    // ---------------------------------------------------------------------------
    // InitFDBuffers: 初始化 FlashDecode 所需的 UB buffer 布局
    // ---------------------------------------------------------------------------
    // 从传入的 ubBase 起始地址依次切分以下连续 buffer (无空洞):
    //   lseMaxBuf       (8K float)  —— 存放 broadcast 后的 rowMax, 布局 [topK, dealRow, 8]
    //   lseSumBuf       (8K float)  —— 存放 broadcast 后的 rowSum/scale, 布局同上
    //   dstBuf          (32K slot)  —— 最终融合结果 O 的 UB 暂存 (TwsO 类型)
    //   broadCastTmpBuf (32K slot)  —— Broadcast scratch + compact lse 暂存区
    //                                   (lseSumTmpBuf = broadCastTmpBuf_[alignedLseCount])
    //   castOutBuf      (32K slot)  —— fp32→bf16 out-of-place Cast 的独立输出缓冲
    //   accumOutBuf[2]  (32K each)  —— O_partial ping-pong 双缓冲, 每个槽存一组 split
    //
    // splitsPerBuf_: 每个 ping-pong slot 读取的 split 数 (批量连续读)。
    //   - 目标是让 topK 个 split 分成约 2 组 (buf-group), 使 2 级 ping-pong
    //     仍能重叠 MTE2(g+1) || V(g)。
    //   - cap = 32K / (groupSize * headDimAlignFp32), 即一个 32K 槽最多容纳的 split 数。
    //   - half = (topK+1)/2, splitsPerBuf_ = min(cap, half), 下限为 1。
    // ---------------------------------------------------------------------------
    __aicore__ inline
    void InitFDBuffers(AscendC::LocalTensor<T> &ubBase,
                       uint32_t D,
                       uint32_t groupSize,
                       uint32_t topK,
                       uint32_t kvHeads)
    {
        D_ = D;
        groupSize_ = groupSize;
        topK_ = topK;
        kvHeads_ = kvHeads;
        headDimAlignFp32_ = RoundUp(D_, FP32_ONE_BLOCK_SIZE);

        // E: batched accumOut read. One 32K-float UB slot holds cap splits
        // (split = groupSize * headDimAlignFp32 floats). Supported range
        // (groupSize<=16, D<=512 => split<=8192 floats) keeps cap>=4. Aim for ~2
        // buf-groups so the 2-stage ping-pong still overlaps MTE2(g+1) || V(g) at
        // coarser grain (was per-split). Capped by cap, floored at 1.
        uint32_t cap = BUF_32K_FLOATS / (groupSize_ * headDimAlignFp32_);
        if (cap == 0U) { cap = 1U; }
        uint32_t half = (topK_ + 1U) / 2U;
        splitsPerBuf_ = (half < 1U) ? 1U : ((cap < half) ? cap : half);

        // UB layout: contiguous, no holes. lse buffers capped at 8K (the [topK, dealRowCount, 8]
        // broadcast layout fits in 8KiB for topK, groupSize <= 16); the rest are 32K slots.
        lseMaxBuf_       = ubBase;                       // 8K
        lseSumBuf_       = ubBase[BUF_8K_FLOATS];        // 8K
        uint32_t off = BUF_8K_FLOATS * 2U;
        // dst slot carved as a 32K-float (32KB) stride; for the bf16 path TwsO=bf16, so the
        // raw float carve is reinterpreted (bf16 dst uses <=4KB of the 32KB — slack is free).
        dstBuf_          = ubBase[off].template ReinterpretCast<TwsO>();
        off += BUF_32K_FLOATS;   // 32K
        broadCastTmpBuf_ = ubBase[off];
        off += BUF_32K_FLOATS;  // 32K (Broadcast scratch + lse staging)
        // Separate bf16 cast-output buffer so CopyFinalResOut's Cast is out-of-place.
        castOutBuf_      = ubBase[off];
        off += BUF_32K_FLOATS;  // 32K
        // accumOut ping-pong (kernel ubSTensor style): 2 stages, uniform 32K-slot stride.
        // For the bf16 path the stage is ReinterpretCast<TwsO> over the same 32K-float carve
        // (bf16 stage holds <=16KB of data, the 32KB carve leaves slack — UB only gets looser).
        for (uint32_t i = 0U; i < ACCUM_OUT_STAGES; ++i) {
            accumOutBuf[i] = ubBase[off + BUF_32K_FLOATS * i].template ReinterpretCast<TwsO>();
        }
    }

    // ---------------------------------------------------------------------------
    // FlashDecodeCompute: Phase2 主入口, 每个 task 处理 1 个 (qToken, kvHeadIdx)
    // ---------------------------------------------------------------------------
    // task 编码 (与 IFA 一致):
    //   qToken    = tmpBlockIdx / kvHeads   (query token 在 totalQ 维的索引)
    //   kvHeadIdx = tmpBlockIdx % kvHeads   (KV head 在 kvHeads 维的索引)
    //
    // attenOutOffset 计算 (GM O 的写入起始偏移):
    //   = qToken * kvHeads * groupSize * D + kvHeadIdx * groupSize * D
    // 即在 GM 输出布局 [totalQ, kvHeads, groupSize, D] 中定位到当前 task 对应的
    // 连续 groupSize*D 区域。后续 CombineSplitKVRes 负责将融合结果写回该偏移处。
    //
    // actualCombineLoopSize_ = topK_: 遍历 topK 个 split 做加权融合。
    // ---------------------------------------------------------------------------
    __aicore__ inline
    void FlashDecodeCompute(uint32_t tmpBlockIdx,
                            uint32_t totalTaskNumP2,
                            AscendC::GlobalTensor<ElementO> &gmO,
                            AscendC::GlobalTensor<TwsO> &gmAccumOut,
                            AscendC::GlobalTensor<T> &gmSoftmaxMax,
                            AscendC::GlobalTensor<T> &gmSoftmaxSum,
                            uint32_t numHeads,
                            uint32_t kvHeads)
    {
        if (tmpBlockIdx >= totalTaskNumP2) {
            return;
        }
        (void)numHeads;

        uint32_t qToken = tmpBlockIdx / kvHeads;
        uint32_t kvHeadIdx = tmpBlockIdx % kvHeads;
        // IFA: bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim
        uint64_t attenOutOffset = static_cast<uint64_t>(qToken) * kvHeads * groupSize_ * D_
            + static_cast<uint64_t>(kvHeadIdx) * groupSize_ * D_;

        actualCombineLoopSize_ = topK_;
        CombineSplitKVRes(attenOutOffset, qToken, kvHeadIdx, gmO, gmAccumOut, gmSoftmaxMax, gmSoftmaxSum);
    }

private:
    AscendC::LocalTensor<T> lseMaxBuf_;
    AscendC::LocalTensor<T> lseSumBuf_;
    // dst (combined O) dtype = TwsO: float for the fp32 path, bfloat16_t for the
    // innerPrecise==1 bf16 path (regbase-fused reduce stores bf16 directly here;
    // CopyFinalResOut then plain-copies bf16 to GM O, no cast).
    AscendC::LocalTensor<TwsO> dstBuf_;
    // O_partial ping-pong slot dtype = TwsO (float for fp32 path, bf16 for innerPrecise==1).
    AscendC::LocalTensor<TwsO> accumOutBuf[ACCUM_OUT_STAGES];
    AscendC::LocalTensor<T> broadCastTmpBuf_;
    AscendC::LocalTensor<T> castOutBuf_;

    uint32_t D_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t topK_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t headDimAlignFp32_ = 0;
    uint32_t actualCombineLoopSize_ = 0;
    uint32_t splitsPerBuf_ = 0;   // E: splits read per ping-pong slot (batched contiguous read)

    __aicore__ inline
    static uint32_t RoundUp(uint32_t a, uint32_t b)
    {
        return (a + b - 1U) / b * b;
    }

    __aicore__ inline
    uint64_t TaskStatBase(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return (static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx) * topK_ * groupSize_;
    }

    __aicore__ inline
    uint64_t TaskAccumBase(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return (static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx) * topK_ * groupSize_ * D_;
    }

    // ---------------------------------------------------------------------------
    // CopyLseIn: 从 GM 读 compact max/sum, Broadcast 展开为 IFA 所需的 [topK, dealRow, 8] 布局
    // ---------------------------------------------------------------------------
    // GM 中的 softmaxMax / softmaxSum 是紧凑存储 (compact), 布局 [topK, groupSize],
    // 每个 split 仅 1 个标量。但后续 ComputeScaleValue_VF 需要 32B 对齐的
    // [topK, dealRowCount, 8] 布局 (每标量复制 8 份), 因此需要 Broadcast 展开。
    //
    // 流程:
    //   1. DataCopyPad 将 compact max/sum 从 GM 读入 broadCastTmpBuf_ 的前半区
    //      (lseMaxTmpBuf 和 lseSumTmpBuf, 各 alignedLseCount 个 float)。
    //   2. MTE2→V 同步 (SetFlag/WaitFlag EVENT_ID0)。
    //   3. Broadcast 将 compact 数据展开为 [lseCount, 8] 写入 lseMaxBuf_ / lseSumBuf_。
    //
    // Reciprocal 优化: 预计算 1/rowSum, 将后续 ReduceFinalRes 中的 Div 转为 Mul。
    //   - 对 compact lseSumTmpBuf 原地执行 Reciprocal (而非 broadcast 后的 lseSumBuf_)。
    //   - lseSumBuf_ (8x broadcast) 是独立 buffer, 原始 sum 值不受影响, 仍供
    //     ComputeScaleValue 计算 scale 使用。
    //   - 无效 slot 的 rowSum==0 → reciprocal=+inf, 但 combineDone 会在归约到达前跳出,
    //     永远不会被使用。
    //   - GM max/sum compact [topK, groupSize] -> UB IFA layout [topK, dealRow, 8].
    // ---------------------------------------------------------------------------
    __aicore__ inline
    void CopyLseIn(uint32_t qToken,
                   uint32_t kvHeadIdx,
                   uint32_t dealRowCount,
                   AscendC::GlobalTensor<T> &gmSoftmaxMax,
                   AscendC::GlobalTensor<T> &gmSoftmaxSum,
                   AscendC::LocalTensor<T> &lseMaxLocal,
                   AscendC::LocalTensor<T> &lseSumLocal)
    {
        uint32_t ubRowStride = dealRowCount * FP32_ONE_BLOCK_SIZE;
        auto lseCount = dealRowCount * topK_, alignedLseCount = RoundUp(lseCount, FP32_ONE_BLOCK_SIZE);

        AscendC::LocalTensor<T> lseMaxTmpBuf = broadCastTmpBuf_;
        AscendC::LocalTensor<T> lseSumTmpBuf = broadCastTmpBuf_[alignedLseCount];
        AscendC::LocalTensor<uint8_t> broadCastTmpBuf = broadCastTmpBuf_[alignedLseCount * 2U].template ReinterpretCast<uint8_t>();
        uint64_t taskBase = TaskStatBase(qToken, kvHeadIdx);
        DataCopyParams copyParams {
            1,                                      // blockCount
            uint16_t(lseCount * sizeof(float)),     // blockLen
            0,                                      // srcStride
            0,                                      // dstStride
        };
        DataCopyPadParams copyPadParams {
            false,                 // isPad
            0,                     // leftPadding
            0,                     // rightPadding
            0                      // paddingValue
        };
        AscendC::DataCopyPad(lseMaxTmpBuf, gmSoftmaxMax[taskBase], copyParams, copyPadParams);
        AscendC::DataCopyPad(lseSumTmpBuf, gmSoftmaxSum[taskBase], copyParams, copyPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        const uint32_t srcShape[2] = {lseCount, 1};
        const uint32_t dstShape[2] = {lseCount, FP32_ONE_BLOCK_SIZE};
        AscendC::Broadcast<T, 2, 1>(lseMaxLocal, lseMaxTmpBuf, dstShape, srcShape, broadCastTmpBuf);
        AscendC::Broadcast<T, 2, 1>(lseSumLocal, lseSumTmpBuf, dstShape, srcShape, broadCastTmpBuf);
        // Div->Mul: precompute 1/rowSum in place on the compact lseSumTmpBuf. The 8x
        // broadcast lseSumBuf_ (read by ComputeScaleValue for the per-split scale) was
        // just written above and is a SEPARATE buffer, so the original sum values feeding
        // scale stay intact; only the compact per-row rowSum that ReduceFinalRes_VF reads
        // as rowSumLocal becomes its reciprocal, turning the per-(row,split,z) Div into a
        // Mul. Broadcast (V) reads lseSumTmpBuf before this in-place Reciprocal (same V
        // pipe, in-order). Invalid slots have rowSum==0 (WS_ROWSUM_INIT) -> reciprocal=+inf,
        // never used (combineDone breaks before the reduce reaches them).
        AscendC::Reciprocal<T>(lseSumTmpBuf, lseSumTmpBuf, static_cast<int32_t>(alignedLseCount));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---------------------------------------------------------------------------
    // ComputeScaleValue: 计算 scale 权重 (各 split 在最终融合中的占比)
    // ---------------------------------------------------------------------------
    // 公式:
    //   scale[k] = rowSum[k] * exp(rowMax[k] - max_global) / Σ scale[j]
    // 其中 max_global = max(rowMax[0..topK-1]) 是所有 split 的全局最大值。
    //
    // softmaxLseFlag = false: 不使用 LSE (log-sum-exp) 路径, 直接用 max/sum 计算。
    //   - 本算子的 Phase1 存储的是原始 max 和 sum (非 log 域), 故走 false 分支。
    //   - learnableSinkFlag = false: 不使用可学习 sink (learnable sink) 功能。
    //
    // 调用 FaVectorApi::ComputeScaleValue_VF, 结果写回 lseSumLocal (即 lseSumBuf_),
    // 后续 ReduceFinalRes 读取该 scale 对各 split 的 O_partial 做加权。
    // ---------------------------------------------------------------------------
    __aicore__ inline
    void ComputeScaleValue(AscendC::LocalTensor<T> &lseMaxLocal,
                           AscendC::LocalTensor<T> &lseSumLocal,
                           uint32_t dealRowCount)
    {
        AscendC::LocalTensor<bfloat16_t> tmpSinkUb;
        bool softmaxLseFlag = false;
        bool learnableSinkFlag = false;
        //AscendC::DumpTensor(lseMaxLocal, 123, 64);
        //AscendC::DumpTensor(lseSumLocal, 456, 64);
        FaVectorApi::ComputeScaleValue_VF(
            tmpSinkUb, lseMaxLocal, lseSumLocal, accumOutBuf[0].template ReinterpretCast<T>(),
            dealRowCount, actualCombineLoopSize_, softmaxLseFlag, learnableSinkFlag);
        //AscendC::DumpTensor(lseSumLocal, 789, 64);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---------------------------------------------------------------------------
    // CopyAccumOutIn: 批量连续读 nSplits 个 split 的 O_partial 到 UB
    // ---------------------------------------------------------------------------
    // GM 布局: accumOut [totalQ, kvHead, topK, groupSize, D], 其中 topK 维连续
    // (topK 为次慢维, D 为最快维), 因此 [jStart..jStart+nSplits) 范围内的数据在
    // GM 中是连续存放的, 可以用一次 DataCopyPad 读取 (srcStride=0, 行与行紧挨)。
    //
    // 优化点 (E: batched contiguous read):
    //   - 旧方案: 每个 split 单独发一次 8KB DataCopyPad (topK 次), per-transaction
    //     开销高, 是 Phase2 MTE2 瓶颈 (占算子 39.5%, 仅 24 GB/s)。
    //   - 新方案: 一次读取 nSplits * dealRowCount 行 (blockCount = nSplits * dealRowCount),
    //     事务数从 topK 降至 numGroups = ceil(topK / splitsPerBuf_), 有效带宽提升。
    //
    // blockElems: 32B 块对应的元素数 (fp32=8, bf16=16), 用于计算 dstStride 和 rightPadding。
    // oOffset: 在 GM accumOut 中定位到 [qToken, kvHead, jStart] 的起始偏移。
    // ---------------------------------------------------------------------------
    __aicore__ inline
    void CopyAccumOutIn(uint32_t qToken,
                        uint32_t kvHeadIdx,
                        uint32_t jStart,
                        uint32_t nSplits,
                        uint32_t dealRowCount,
                        AscendC::GlobalTensor<TwsO> &gmAccumOut,
                        AscendC::LocalTensor<TwsO> &accumOutLocal)
    {
        // E: batched contiguous read of nSplits splits in one DataCopyPad (was one
        // 8KB DataCopyPad per split). GM layout [totalQ, kvHead, topK, groupSize, D]
        // is contiguous over [jStart..jStart+nSplits) (topK slowest, D fastest,
        // srcStride 0 => rows back-to-back), so the whole span is one transaction.
        // O_partial dtype = TwsO (float fp32 path, bf16 innerPrecise==1 path). blockElems
        // is the 32B-block element count (8 fp32 = FP32_ONE_BLOCK_SIZE, 16 bf16); for the
        // fp32 path blockElems==FP32_ONE_BLOCK_SIZE so the params are byte-identical to prior.
        constexpr uint32_t blockElems = BYTE_BLOCK / sizeof(TwsO);
        AscendC::DataCopyExtParams copyInParams;
        AscendC::DataCopyPadExtParams<TwsO> copyInPadParams;
        copyInParams.blockCount = nSplits * dealRowCount;
        copyInParams.blockLen = static_cast<uint32_t>(D_) * sizeof(TwsO);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (headDimAlignFp32_ - D_) / blockElems;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (headDimAlignFp32_ - D_) % blockElems;
        copyInPadParams.paddingValue = 0;

        uint64_t oOffset = TaskAccumBase(qToken, kvHeadIdx)
            + static_cast<uint64_t>(jStart) * groupSize_ * D_;
        AscendC::DataCopyPad(accumOutLocal, gmAccumOut[oOffset], copyInParams, copyInPadParams);
    }

    // ---------------------------------------------------------------------------
    // ReduceFinalRes: 遍历 topK 个 split, 累加 scale[k] * O_norm[k] 到 dst
    // ---------------------------------------------------------------------------
    // 核心逻辑: 对每个 split j, 执行
    //   dst += scale[j] * (O_partial[j] / rowSum[j])
    // 其中 O_partial[j] 来自 accumOutBuf (由 CopyAccumOutIn 批量读入 UB),
    // scale[j] 来自 lseSumBuf_ (由 ComputeScaleValue 计算),
    // rowSum[j] 来自 broadCastTmpBuf_ 的 compact 区 (由 CopyLseIn 读入并取倒数)。
    //
    // 无效 slot 检查 (invalid slot check):
    //   - lseMaxBuf_[maxOff].GetValue(0) <= NEG_INF_LSE → 该 slot 无效, 跳过。
    //   - 无效 slot 的 rowMax 被 InitWorkspaceStats 预填为 -FLT_MAX (NEG_INF_LSE)。
    //   - 有效 slot 是左连续的 (Phase1 只写 slotK=0..valid_k-1), 所以第一个无效
    //     slot 出现后, 后续全部无效, 直接 break 跳出整个 combine。
    //
    // 双缓冲 ping-pong (double-buffered):
    //   - accumOutBuf[2] 双槽, curId = g % ACCUM_OUT_STAGES。
    //   - MTE2 加载 group g+1 的数据 与 V 归约 group g 重叠执行:
    //     Wait<V_MTE2>(g) → CopyAccumOutIn(g) → Set<MTE2_V>(g) → Wait<MTE2_V>(g) → V reduce(g)
    //   - 每个 buf-group 一次性加载 splitsPerBuf_ 个 split (CopyAccumOutIn),
    //     然后内层 j 循环逐 split 做 V 归约。
    //
    // bf16 vs fp32 路径分发:
    //   - bf16 路径 (TwsO=bfloat16_t): 调用 ReduceFinalRes_VF_BF16, 在 regbase 中
    //     融合 Cast bf16→fp32 → fp32 Mul/Add → Cast fp32→bf16, 直接存 bf16 到 dst。
    //   - fp32 路径 (TwsO=float): 调用 ReduceFinalRes_VF, 全程 fp32 计算。
    //   - 两条路径的 j==0 初始化 vs 加载已有 dst 的分发在 VF 内部处理。
    // ---------------------------------------------------------------------------
    __aicore__ inline
    void ReduceFinalRes(uint32_t qToken,
                        uint32_t kvHeadIdx,
                        AscendC::LocalTensor<TwsO> &dst,
                        AscendC::LocalTensor<T> &scaleLocal,
                        uint32_t dealRowCount,
                        AscendC::GlobalTensor<TwsO> &gmAccumOut)
    {
        // Slot validity check from UB (no GM round-trip). After CopyLseIn +
        // ComputeScaleValue, lseMaxBuf_ holds the per-slot rowMax in broadcast
        // layout [topK, dealRowCount, FP32_ONE_BLOCK_SIZE] (topK outer, each
        // scalar replicated 8x). ComputeScaleValue_8_VF reads lseMax but does not
        // overwrite it, so the values are intact. InitWorkspaceStats prefills
        // -FLT_MAX (NEG_INF_LSE) for slots Phase1 never wrote; a valid slot's
        // rowMax is a finite score-derived value, so firstRowMax <= NEG_INF_LSE
        // discriminates the whole slot. Valid slots are left-contiguous in the
        // workspace (Phase1 writes slotK = 0..valid_k-1 per qToken/kvHead, since
        // build_k2q_csr assigns slot_flat = arange(topk) only to valid entries),
        // so the first invalid slot means every later slot is invalid too: break
        // once instead of continue. Skipping CopyAccumOutIn + ReduceFinalRes_VF
        // for invalid splits avoids Divs-by-zero (rowSum==0) and stale-UB
        // accumulation; scale[j]==0 from ComputeScaleValue makes them 0 anyway.
        // The per-row rowSum for the regbase Div is reused from the compact rowSum
        // already staged by CopyLseIn into broadCastTmp_ (lseSumTmpBuf =
        // broadCastTmpBuf_[alignedLseCount], layout [topK, dealRowCount]); no GM
        // re-read or SetValue here.
        uint32_t lseCount = dealRowCount * topK_;
        uint32_t alignedLseCount = RoundUp(lseCount, FP32_ONE_BLOCK_SIZE);

        // Double-buffered (ping-pong) accumOut, managed like the kernel's ubSTensor[ubSBufId]:
        // a 2-deep array accumOutBuf[curId], curId = g % ACCUM_OUT_STAGES, with event ids
        // ACCUM_OUT_FLAG_BASE + curId for both MTE2_V (forward) and V_MTE2 (reverse) — distinct
        // HardEvent types share an id safely (IFA fdMm2ResBuf1/2 does the same).
        //
        // E: the ping-pong grain moved from per-split j to per buf-group g. Each group
        // loads splitsPerBuf_ (or the tail) splits in ONE contiguous DataCopyPad
        // (CopyAccumOutIn), then the V reduce walks those splits from UB. The MTE2 load
        // of group g+1 overlaps the V reduce of group g (still 2-stage, still overlapped);
        // only the transaction size grew (8KB -> ~splitsPerBuf_*8KB) so per-transaction
        // overhead drops ~splitsPerBuf_-fold and effective MTE2 BW rises. For topK=1 this
        // degenerates to a single load (no overlap, nothing to overlap with).
        //
        // Validity is checked in the inner split loop (one scalar GetValue per split,
        // issued alongside the V reduce so it hides under the V pipe, not pacing it).
        // Valid slots are left-contiguous (Phase1 writes slotK = 0..valid_k-1; slots
        // beyond are prefilled -FLT_MAX = NEG_INF_LSE by InitWorkspaceStats, whose
        // WS_ROWSUM_INIT=0 makes an unreduced invalid split a Div-by-zero NaN risk), so
        // the first invalid slot means every later slot is invalid too: break out of the
        // whole combine there. The batch load already pulled this group's splits (incl.
        // the rare invalid tail straddling valid_k); we simply don't reduce past the first
        // invalid. The extra copied data is <= one partial group and only when valid_k <
        // topK (causal head / sparse tail) — rare in prefill, accepted.

        // Prime each stage's V_MTE2: a stage's first reuse has no prior V reduce to drain.
        for (uint32_t s = 0U; s < ACCUM_OUT_STAGES; ++s) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + s);
        }

        uint32_t numGroups = (topK_ + splitsPerBuf_ - 1U) / splitsPerBuf_;
        bool combineDone = false;
        for (uint32_t g = 0U; (g < numGroups) && !combineDone; ++g) {
            uint32_t jStart = g * splitsPerBuf_;
            uint32_t nSplits = splitsPerBuf_;
            uint32_t remain = topK_ - jStart;
            if (nSplits > remain) { nSplits = remain; }
            uint32_t curId = g % ACCUM_OUT_STAGES;
            // stage drained (reduce g-2, or primed) -> MTE2 may overwrite it.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + curId);
            CopyAccumOutIn(qToken, kvHeadIdx, jStart, nSplits, dealRowCount, gmAccumOut,
                           accumOutBuf[curId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ACCUM_OUT_FLAG_BASE + curId);    // load g || V reduce g-1
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ACCUM_OUT_FLAG_BASE + curId);  // load g done
            // V reduce each split in this group: accumOut base is the in-buf offset
            // (j-jStart)*groupSize*headDimAlignFp32; j stays the global split index so
            // lse (lseSumBuf_) and rowSum (broadCastTmpBuf_[alignedLseCount+j*dealRowCount])
            // and the _0/_Rest (j==0 init vs load prior dst) dispatches are unchanged.
            // First invalid (left-contiguous) breaks the whole combine.
            for (uint32_t j = jStart; j < jStart + nSplits; ++j) {
                uint32_t maxOff = j * dealRowCount * FP32_ONE_BLOCK_SIZE;
                if (lseMaxBuf_[maxOff].GetValue(0) <= NEG_INF_LSE) {
                    combineDone = true;
                    break;
                }
                uint64_t localOff = static_cast<uint64_t>(j - jStart) * groupSize_
                    * headDimAlignFp32_;
                uint64_t rowSumOff = static_cast<uint64_t>(alignedLseCount)
                    + static_cast<uint64_t>(j) * dealRowCount;
                AscendC::LocalTensor<TwsO> accumSplit = accumOutBuf[curId][localOff];
                AscendC::LocalTensor<T> rowSumLocal = broadCastTmpBuf_[rowSumOff];
                // bf16 path: regbase-fused reduce (vf_flash_decode_msa.h::ReduceFinalRes_VF_BF16).
                // Reads bf16 accumSplit straight into a reg, fuses Cast bf16->fp32 (ZERO/ONE),
                // fp32 Mul/Mul/Add, Cast back to bf16 (ZERO/ONE), Or-merge, vsstb store bf16
                // directly to dst (TwsO=bf16). No vector Cast staging pass / no castOutBuf_ staging
                // here (castOutBuf_ stays free for the fp32 path's CopyFinalResOut). dst is bf16 so
                // the cross-split accumulation is bf16-rounded per split (innerPrecise==1 INNER_LOW).
                // fp32 path calls the fp32 reduce directly (byte-identical to the verified binary).
                if constexpr (AscendC::IsSameType<TwsO, bfloat16_t>::value) {
                    FaVectorApiPrefill::ReduceFinalRes_VF_BF16<TwsO>(
                        dst, scaleLocal, accumSplit, rowSumLocal, dealRowCount,
                        static_cast<uint64_t>(headDimAlignFp32_), j);
                } else {
                    FaVectorApiPrefill::ReduceFinalRes_VF<T>(
                        dst, scaleLocal, accumSplit, rowSumLocal, dealRowCount,
                        static_cast<uint64_t>(headDimAlignFp32_), j);
                }
            }
            // V done reading accumOutBuf[curId] (and dst store committed; SetFlag<V_MTE2> is a V
            // completion barrier, ordering dst V-store->V-load across iterations) -> stage reusable.
            // ALWAYS executed, even when the inner loop broke: this group already did
            // Wait<V_MTE2> above, so it must replenish with a Set or the trailing drain
            // would underflow (the broken group's Wait consumed a primed/prior Set).
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + curId);
        }
        // Trailing drain: consume the dangling V_MTE2 per stage (last reduce's set + the primed
        // twin that no later load reached) so it doesn't stale-signal the next chunk/task. MTE2_V
        // is set+waited 1:1 within each iter, so no MTE2_V trailing drain.
        for (uint32_t s = 0U; s < ACCUM_OUT_STAGES; ++s) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + s);
        }
    }

    // ---------------------------------------------------------------------------
    // CopyFinalResOut: 将最终结果从 UB 写回 GM
    // ---------------------------------------------------------------------------
    // 将 ReduceFinalRes 产出的 dst (在 dstBuf_ 中) 写回 GM O 的 attenOutOffset 位置。
    // 根据中间精度路径分两条:
    //
    // bf16 路径 (TwsO=bfloat16_t):
    //   - dst 已是 bf16 (regbase reduce 直接存 bf16, 无需额外 Cast)。
    //   - 直接 DataCopyPad 从 dstBuf_ 拷贝到 GM O, 无 Cast, 无 castOutBuf_ 暂存。
    //
    // fp32 路径 (TwsO=float):
    //   - dst 是 fp32, 需要转成 bf16 (ElementO=bfloat16_t) 再写回 GM。
    //   - 先 Cast fp32→bf16 out-of-place 到 castOutBuf_ (独立 buffer, 避免 aliasing
    //     dstBuf_ 导致数据损坏), 再 DataCopyPad 从 castOutBuf_ 拷贝到 GM O。
    //   - V→MTE3 同步 (SetFlag/WaitFlag EVENT_ID0) 确保 Cast 完成后再搬运。
    // ---------------------------------------------------------------------------
    __aicore__ inline
    void CopyFinalResOut(uint64_t attenOutOffset,
                         AscendC::GlobalTensor<ElementO> &gmO,
                         AscendC::LocalTensor<TwsO> &accumOutLocal,
                         uint32_t dealRowCount)
    {
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<TwsO, bfloat16_t>::value) {
            // bf16 path: dst (dstBuf_) is already bf16 (the regbase reduce stored bf16
            // directly). Plain copy to GM O — no Cast, no castOutBuf_ staging.
            uint32_t shapeArray[] = {dealRowCount, D_};
            accumOutLocal.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = dealRowCount;
            dataCopyParams.blockLen = static_cast<uint32_t>(D_) * sizeof(ElementO);
            dataCopyParams.srcStride = (headDimAlignFp32_ - D_) / (BYTE_BLOCK / sizeof(ElementO));
            dataCopyParams.dstStride = 0;
            AscendC::DataCopyPad(gmO[attenOutOffset], accumOutLocal, dataCopyParams);
        } else {
            // fp32 path (byte-identical to the verified binary): cast fp32 dst -> bf16
            // out-of-place into castOutBuf_ (separate buffer; aliasing dstBuf_ corrupts),
            // then copy. Identical for the fp32 path only.
            AscendC::LocalTensor<ElementO> castBuf = castOutBuf_.template ReinterpretCast<ElementO>();
            uint32_t shapeArray[] = {dealRowCount, D_};
            castBuf.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
            AscendC::Cast(
                castBuf, accumOutLocal,
                AscendC::RoundMode::CAST_ROUND,
                dealRowCount * headDimAlignFp32_);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            AscendC::DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = dealRowCount;
            dataCopyParams.blockLen = static_cast<uint32_t>(D_) * sizeof(ElementO);
            dataCopyParams.srcStride = (headDimAlignFp32_ - D_) / (BYTE_BLOCK / sizeof(ElementO));
            dataCopyParams.dstStride = 0;
            AscendC::DataCopyPad(gmO[attenOutOffset], castBuf, dataCopyParams);
        }
    }

    // ---------------------------------------------------------------------------
    // CombineSplitKVRes: 组合所有步骤的顶层函数 (Phase2 combine 主流程)
    // ---------------------------------------------------------------------------
    // 将 topK 个 split 的部分输出融合为最终 O, 依次执行以下四步:
    //
    //   1. CopyLseIn      — 从 GM 读 compact max/sum, Broadcast 展开为 [topK, dealRow, 8],
    //                        并预计算 1/rowSum (Reciprocal 优化)。
    //   2. ComputeScaleValue — 计算 scale 权重: scale[k] = rowSum[k]*exp(max[k]-max_global)/Σ
    //   3. ReduceFinalRes   — 双缓冲 ping-pong 遍历 topK 个 split, 累加
    //                        scale[k] * (O_partial[k] / rowSum[k]) 到 dst。
    //   4. CopyFinalResOut  — 将 dst 从 UB 写回 GM O (bf16 直拷 / fp32 先 Cast 再拷)。
    //
    // groupSize <= 16 时所有 UB buffer 均可容纳 (lse 8K cap, dst/accumOut/castOut 32K),
    // topK <= 16, D <= 512, 单次 chunk 即可覆盖整个 group, 无需分块。
    // ---------------------------------------------------------------------------
    __aicore__ inline
    void CombineSplitKVRes(uint64_t attenOutOffset,
                           uint32_t qToken,
                           uint32_t kvHeadIdx,
                           AscendC::GlobalTensor<ElementO> &gmO,
                           AscendC::GlobalTensor<TwsO> &gmAccumOut,
                           AscendC::GlobalTensor<T> &gmSoftmaxMax,
                           AscendC::GlobalTensor<T> &gmSoftmaxSum)
    {
        // groupSize <= 16 fits every UB buffer (lse 8K cap, dst/accumOut/castOut 32K slots)
        // for topK <= 16, D <= 512, so a single chunk covers the whole group.
        CopyLseIn(qToken, kvHeadIdx, groupSize_,
                  gmSoftmaxMax, gmSoftmaxSum, lseMaxBuf_, lseSumBuf_);
        ComputeScaleValue(lseMaxBuf_, lseSumBuf_, groupSize_);
        AscendC::LocalTensor<TwsO> dst = dstBuf_;
        ReduceFinalRes(qToken, kvHeadIdx, dst, lseSumBuf_, groupSize_, gmAccumOut);
        CopyFinalResOut(attenOutOffset, gmO, dst, groupSize_);
    }
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP
