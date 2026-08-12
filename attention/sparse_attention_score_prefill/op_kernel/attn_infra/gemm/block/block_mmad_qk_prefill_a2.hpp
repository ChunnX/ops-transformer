/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * =========================================================================================
 * BlockMmadQKPrefillArch35 —— A5 (Arch35) 平台 Prefill QK 矩阵乘 Block
 * =========================================================================================
 *
 * 【一句话概括】
 *   计算 Attention Score: S = Q × K^T (Q 乘 K 的转置)
 *   这是稀疏注意力 Prefill 阶段的第一个 Cube (矩阵乘) 组件。
 *
 * 【在整体架构中的位置】
 *   Phase1 三级流水线的第一级 (QK):
 *     QK (本文件) → softmax (Vector) → PV (另一个 Cube Block)
 *   本组件负责: 从 GM 加载 Q 和 K → L1 → L0A/L0B → MMAD 矩阵乘 → L0C → Fixpipe 到 UB
 *   结果 S (注意力分数) 写入 UB, 供 Vector 侧 softmax 组件读取。
 *
 * 【KV-Gather-Q 适配】
 *   在 KV-centric 遍历中, K 以 KV block 为单位加载, 每个 task 只加载一次 K 并常驻 L1/L0B。
 *   Q 则按 CSR 反向索引 "gather" (收集): 多个不连续的 Q token 从 GM 拼装到 L1 连续空间,
 *   然后一次 MMAD 完成所有 Q token 与该 K block 的乘法 (批量计算, 减少 MMAD 启动开销)。
 *
 * 【A5 硬件特性利用】
 *   - L0C→UB: A5 支持 Fixpipe 直接搬运 (无需 GM 中转, A2 才需要)
 *   - 双 AIV 分割: S 结果按 "group" (Q head group) 分配到 2 个 AIV 子核, 每个 AIV 处理自己
 *     的 group, 避免一个 group 跨越两个子核 (与 softmax/ScatterBatchStats 的分组一致)
 *
 * 【FixPipe / 跨核同步】
 *   与 block_mmad_qk_arch35_ABf16_C_to_UB.hpp 对齐: S 结果按 group 拆分到两个 AIV 的 UB buffer。
 *   跨核同步使用 MODE_4 (1 AIC : 2 AIV), 同时 Set/Wait 两个 flag (V0 + V1, 偏移 16)。
 *
 * 【数据流图】
 *   GM(K) ──MTE2──→ L1(K) ──MTE1──→ L0B(K) ──┐
 *                                              ├── MMAD ──→ L0C(S) ──Fixpipe──→ UB(S)
 *   GM(Q) ──MTE2──→ L1(Q) ──MTE1──→ L0A(Q) ──┘                    ↓
 *                                                          跨核 flag 通知 VEC
 * =========================================================================================
 */
#ifndef GEMM_BLOCK_MMAD_QK_PREFILL_A2_HPP
#define GEMM_BLOCK_MMAD_QK_PREFILL_A2_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/arch/cross_core_sync.hpp"
#include "../../../attn_infra/coord.hpp"
#include "../../../attn_infra/gemm/dispatch_policy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_mmad.hpp"
#include "../../../attn_infra/layout/layout.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Block {

// =========================================================================================
// BlockMmadQKPrefillArch35: A5 平台 Prefill QK 矩阵乘组件
//
// 模板参数:
//   ElementQ_ — Q 输入元素类型 (通常 bf16)
//   ElementK_ — K 输入元素类型 (通常 bf16)
//   ElementS_ — S = Q×K^T 输出元素类型 (bf16, fixpipe 时从 fp32 累加结果转换)
// =========================================================================================
template <class ElementQ_, class ElementK_, class ElementS_>
struct BlockMmadQKPrefillA2 {
    using DispatchPolicy = MmadAtlasA2PrefillQK;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementQ = ElementQ_;
    using ElementK = ElementK_;
    using ElementS = ElementS_;
    using ElementAccumulator = float;

    using LayoutTagQ = layout::RowMajor;
    using LayoutTagK = layout::ColumnMajor;
    using LayoutTagS = layout::RowMajor;

    // A2 uses PackedTileCopyTla (CopyL0CToDst = CopyL0CToGmTla), NOT
    // PackedTileCopyTlaToUB (which is A5-only and targets UB directly).
    using TileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK,
        ElementS, LayoutTagS, void>;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, LayoutTagL1A>;

    static constexpr uint32_t L0_STAGES = 2;
    static constexpr uint32_t L0_TILE_M = 64;
    static constexpr uint32_t L0_TILE_N = 128;
    static constexpr uint32_t L0_TILE_K = 128;
    static constexpr uint32_t L0A_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_BUF_SIZE = L0C_HALF_SIZE / L0_STAGES;
    static constexpr uint32_t L1_Q_STAGES = 1;
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;
    static constexpr uint32_t COPY_GRANULARITY = 2;
    static constexpr uint32_t C0_SIZE = 16;

    AscendC::LocalTensor<ElementK> l1KTensor_;
    AscendC::LocalTensor<ElementQ> l1QTensor_[L1_Q_STAGES];
    AscendC::LocalTensor<ElementQ> l0ATensor_[L0_STAGES];
    AscendC::LocalTensor<ElementK> l0BTensor_[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_[L0_STAGES];

    // GM workspace for L0C→GM→UB flow (A2 has no direct L0C→UB fixpipe).
    // Double-buffered by bi%2; each stage holds one L0_TILE_M×blockSize tile.
    // gmSStageElems_ is in ELEMENTS (GlobalTensor::operator[] indexes by element,
    // not byte). Total GM bytes = 2 * gmSStageElems_ * sizeof(ElementS).
    AscendC::GlobalTensor<ElementS> gmSWorkspace_;
    uint32_t gmSStageElems_;

    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    TileMmad tileMmad;

    uint32_t l1KBufBytes_;

    __aicore__ inline
    BlockMmadQKPrefillA2() {}

    // A2: MODE_2 (0x2) cross-core sync — the proven A2 pattern from infer code.
    // SET: CrossCoreSetFlag<0x2, PIPE_FIX> (CUBE's fixpipe pipe, after L0C→GM).
    // WAIT: CrossCoreWaitFlag (non-templated, MODE=0/PIPE_S — generic sync pipe).
    // MODE_4 with PIPE_FIX↔PIPE_V pairing does NOT work on A2 hardware.
    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 0x2U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 0x2U) {
            // A2 CUBE: must use MODE=0x2 with a CUBE-available pipe (PIPE_FIX etc).
            // Arch::CrossCoreWaitFlag defaults to MODE=0/PIPE_S; PIPE_S is not
            // available on AIC, causing VEC→CUBE cross-core Wait to silently fail.
            AscendC::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag.id);
        }
    }

    __aicore__ inline
    void Init(Arch::Resource<ArchTag> &resource, uint32_t blockSize, uint32_t D,
              AscendC::GlobalTensor<ElementS> &gmSWorkspace)
    {
        uint32_t kBufBytes = blockSize * D * sizeof(ElementK);
        uint32_t qBufBytes = L0_TILE_M * D * sizeof(ElementQ);
        l1KBufBytes_ = kBufBytes;

        l1KTensor_ = resource.l1Buf.template GetBufferByByte<ElementK>(0);
        for (uint32_t i = 0; i < L1_Q_STAGES; i++) {
            l1QTensor_[i] = resource.l1Buf.template GetBufferByByte<ElementQ>(
                kBufBytes + qBufBytes * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor_[i] = resource.l0ABuf.template GetBufferByByte<ElementQ>(L0A_BUF_SIZE * i);
            l0BTensor_[i] = resource.l0BBuf.template GetBufferByByte<ElementK>(L0B_BUF_SIZE * i);
            l0CTensor_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_BUF_SIZE * i);
        }

        gmSWorkspace_ = gmSWorkspace;
        gmSStageElems_ = L0_TILE_M * blockSize;
    }

    template <class TensorK>
    __aicore__ inline
    void LoadKResident(TensorK &gmKTensor, uint32_t validSize, uint32_t D)
    {
        using CopyGmToL1K = typename TileCopy::template CopyGmToL1B<TensorK>;
        CopyGmToL1K copyGmToL1K;

        auto l1KLayout = tla::MakeLayout<ElementK, LayoutTagL1B>(D, validSize);
        auto l1KTensorTla = tla::MakeTensor(l1KTensor_, l1KLayout, Arch::PositionL1{});
        auto l1KTile = GetTile(l1KTensorTla,
            tla::MakeCoord(0, 0), tla::MakeShape(D, validSize));
        auto gmKTile = GetTile(gmKTensor,
            tla::MakeCoord(0, 0), tla::MakeShape(D, validSize));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1K(l1KTile, gmKTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        // The full K matrix remains in L1.  A2 has only 32 KiB per L0B
        // stage, so D=256 is loaded one 128-column K tile at a time below.
    }

    // Batched QK: M = groupCount * groupRows (up to 128 = 8 groups). groupCount scattered
    // gmQ slabs [groupRows, K] are gathered into contiguous L1 Q rows [g*groupRows] (same total
    // MTE2 volume as the old per-qToken path, now feeding one M=128 matmul instead of 8 M=16
    // matmuls). qBase is computed on demand from qToken (== qToken*qHeads*embed + qHeadStart*embed
    // = qToken*perQTokenStrideQ + qHeadStartBase); the kernel precomputes perQTokenStrideQ
    // (=qHeads*embed) and qHeadStartBase (=qHeadStart*embed), constant within a kvHead. The
    // matmul/fixpipe below are unchanged and already tile M via mL0Num.
    //
    // A2 fixpipe flow: L0C → GM(workspace) only. CUBE fixpipes L0C→GM, signals VEC on
    // PIPE_FIX. VEC then DataCopyPad GM→UB (MTE3, valid on VEC) before softmax. Unlike A5
    // which fixpipes L0C→UB directly (CopyL0CToUBTla with subBlockId), A2 has no L0C→UB path.
    // CUBE has no MTE3 pipe — DataCopyPad and PIPE_MTE3 are CUBE-invalid.
    template <class TensorQ, class TensorS>
    __aicore__ inline
    void operator()(TensorQ &gQ, const uint32_t *qTokens,
                    uint32_t groupCount, uint32_t groupRows,
                    TensorS &ubSTensor,
                    uint32_t validSize, uint32_t D,
                    uint64_t perQTokenStrideQ, uint64_t qHeadStartBase,
                    uint32_t numBatches, uint32_t bi,
                    Arch::CrossCoreFlag &mm1ToSmFlag)
    {
        uint32_t M = groupCount * groupRows;
        uint32_t N = validSize;
        uint32_t K = D;

        auto gmQLayout = tla::MakeLayout<ElementQ, LayoutTagQ>(groupRows, K);
        auto gmQEx = tla::MakeTensor(gQ, gmQLayout, Arch::PositionGM{});
        using CopyGmToL1Q = typename TileCopy::template CopyGmToL1A<decltype(gmQEx)>;
        CopyGmToL1Q copyGmToL1Q;

        // Instantiate CopyL0CToGm for the GM workspace tensor type (RowMajor, ElementS).
        auto gmSLayoutEx = tla::MakeLayout<ElementS, LayoutTagS>(L0_TILE_M, L0_TILE_N);
        auto gmSEx = tla::MakeTensor(gmSWorkspace_, gmSLayoutEx, Arch::PositionGM{});
        using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<decltype(gmSEx)>;
        CopyL0CToGm copyL0CToGmSub0;

        auto l1QLayout = tla::MakeLayout<ElementQ, LayoutTagL1A>(M, K);
        auto l1QTensorTla = tla::MakeTensor(l1QTensor_[0], l1QLayout, Arch::PositionL1{});
        auto l1KLayout = tla::MakeLayout<ElementK, LayoutTagL1B>(K, N);
        auto l1KTensorTla = tla::MakeTensor(l1KTensor_, l1KLayout, Arch::PositionL1{});

        // === 阶段 1: Q gather — 从 GM 收集不连续的 Q tile 到 L1 连续空间 ===
        // MTE1_MTE2(EVENT_ID1) 同步: 等上一个 batch 的 L1→L0A MTE1 完成, L1 Q 空间可覆盖
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        // 每次搬 2 个 group (COPY_GRANULARITY=2): 利用 ndNum=2 的 DataCopy 一次搬两个不连续的 Q tile
        for (uint32_t g = 0; g < groupCount; g += COPY_GRANULARITY) {
            // qBase: 当前 group 的 Q token 在 GM 中的起始地址
            //   qBase = qToken × (qHeads × D) + (qHeadStart × D) = qToken × perQTokenStrideQ + qHeadStartBase
            uint64_t qBase = static_cast<uint64_t>(qTokens[g]) * perQTokenStrideQ + qHeadStartBase;
            uint64_t srcNdMatrixStride = 0;   // GM 中两个 group 的间距 (ndNum=2 时用)
            uint64_t dstNzMatrixStride = 0;   // L1 中两个 group 的间距 (NZ 格式 stride)
            uint32_t ndNum = 1;               // 搬运 group 数: 1 或 2
            // 如果还有下一个 group, 尝试 ndNum=2 批量搬运 (减少 MTE2 启动开销)
            if (g + 1 < groupCount) {
                uint32_t nextQ = qTokens[g + 1];
                // 950平台，srcNdMatrixStride不限制小于65536
                if (nextQ > qTokens[g]) {
                    srcNdMatrixStride = static_cast<uint64_t>(nextQ - qTokens[g]) * perQTokenStrideQ;
                } else {
                    // nextQ < qTokens[g]: Q token 逆序, 取较小者作为 qBase, stride 取绝对值
                    srcNdMatrixStride = static_cast<uint64_t>(qTokens[g] - nextQ) * perQTokenStrideQ;
                    qBase = static_cast<uint64_t>(nextQ) * perQTokenStrideQ + qHeadStartBase;
                }
                dstNzMatrixStride = groupRows * C0_SIZE;  // L1 NZ 格式: 每个 group 间隔 = groupRows × 16
                ndNum = COPY_GRANULARITY;                  // 一次搬 2 个 group
            }
            auto gmQTensor = tla::MakeTensor(gQ[qBase], gmQLayout, Arch::PositionGM{});
            auto gmQTile = GetTile(gmQTensor,
                tla::MakeCoord(0, 0), tla::MakeShape(groupRows, K));
            // L1 目标: 按 group 顺序连续排列 [g*groupRows, 0]
            auto l1QTile = GetTile(l1QTensorTla,
                tla::MakeCoord(g * groupRows, 0), tla::MakeShape(groupRows, K));
            copyGmToL1Q(l1QTile, gmQTile, ndNum, srcNdMatrixStride, dstNzMatrixStride);
        }
        // MTE2→MTE1 同步: 等 GM→L1 完成, L1 Q 数据就绪可供 L1→L0A 读取
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
#if KERNEL_DUMP
        if (AscendC::GetBlockIdx() == 1U && bi == 0U) {
            AscendC::DumpTensor(l1QTensor_[0], 990, M * K);
        }
#endif

        // === 阶段 2: L0C 乒乓选择 + MMAD 三重循环 ===
        // L0 tile 切分: M/N/K 各按 L0_TILE 切分 (实际 N=validSize≤128=L0_TILE_N, nL0Num=1)
        uint32_t nL0Num = CeilDiv(N, L0_TILE_N);
        uint32_t mL0Num = CeilDiv(M, L0_TILE_M);
        uint32_t kL0Num = CeilDiv(K, L0_TILE_K);

        for (uint32_t nL0Itr = 0; nL0Itr < nL0Num; nL0Itr++) {
            uint32_t nAct = (nL0Itr == nL0Num - 1) ? (N - nL0Itr * L0_TILE_N) : L0_TILE_N;
            // L0C 乒乓策略 (Plan B): 按 batch bi 选择 L0C stage (bi % 2)
            //   bi=0 → L0C[0], bi=1 → L0C[1], bi=2 → L0C[0]...
            //   这样 MMA(bi) 写 L0C[bi%2] 与 Fixpipe(bi-1) 读 L0C[(bi-1)%2] 可以并行 (不同 buffer)
            //   同一 stage 的复用 (bi vs bi+2) 通过 Wait<FIX_M>(bi%2) 串行化, 保证 UB S 安全
            //   InitSyncFlags 预置 FIX_M(0..3) 使前两个 batch 不死锁
            uint32_t l0CBufId = bi % L0_STAGES;
            uint32_t l0CEventId = l0CBufId;

            auto l0CLayout = tla::MakeLayoutL0C(M, nAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor_[l0CBufId], l0CLayout, Arch::PositionL0C{});

            // === M 维循环: 按 L0_TILE_M 切分 Q 行 ===
            for (uint32_t mL0Itr = 0; mL0Itr < mL0Num; mL0Itr++) {
                uint32_t mAct = (mL0Itr == mL0Num - 1) ? (M - mL0Itr * L0_TILE_M) : L0_TILE_M;
                auto l0CTile = GetTile(l0CTensorTla,
                    tla::MakeCoord(mL0Itr * L0_TILE_M, 0), tla::MakeShape(mAct, nAct));

                // === K 维循环: 按 L0_TILE_K 切分 D 维, 累加到 L0C ===
                // initMmad=true 时清空 L0C, 后续 kL0Itr 累加到已有结果
                for (uint32_t kL0Itr = 0; kL0Itr < kL0Num; kL0Itr++) {
                    uint32_t kAct = (kL0Itr == kL0Num - 1) ? (K - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    // L0A/L0B 按 K 维乒乓: kL0Itr%2 选择 stage
                    uint32_t l0ABufId = kL0Itr % L0_STAGES;
                    uint32_t l0BBufId = 0U;
                    uint32_t l0AEventId = l0ABufId;           // L0A flag: 0/1
                    uint32_t l0BEventId = l0BBufId + 2;       // L0B flag: 2/3 (与 L0A 错开, 因 K 常驻 L0B[0])

                    auto l1QSubTile = GetTile(l1QTensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                        tla::MakeShape(mAct, kAct));
                    auto l0ALayout = tla::MakeLayout<ElementQ, LayoutTagL0A>(mAct, kAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor_[l0ABufId], l0ALayout, Arch::PositionL0A{});

                    // --- L1→L0A: 搬运 Q tile 到 L0A ---
                    // M_MTE1/MTE1_M 同步: L0A 的 MMAD ↔ MTE1 乒乓
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);  // 等 L0A 空闲 (上一次 MMAD 读完)
                    copyL1ToL0A(l0ATensorTla, l1QSubTile);                       // L1 Q → L0A
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);   // L0A 就绪, 通知 MMAD 可读

                    auto l0BLayout = tla::MakeLayout<ElementK, LayoutTagL0B>(kAct, nAct);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor_[l0BBufId], l0BLayout, Arch::PositionL0B{});
                    auto l1KSubTile = GetTile(l1KTensorTla,
                        tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                        tla::MakeShape(kAct, nAct));
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    copyL1ToL0B(l0BTensorTla, l1KSubTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                    bool initMmad = (kL0Itr == 0);
                    uint32_t mAligned = RoundUp(mAct, 16);

                    // --- MMAD 执行: S += Q_tile × K_tile ---
                    // MTE1_M: 等 L0A/L0B 数据就绪
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    if (mL0Itr == 0 && kL0Itr == 0) {
                        // 首个 K tile: 等 L0C 空闲 (上一个同 stage batch 的 Fixpipe 完成)
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    // 执行矩阵乘: L0C[mAct, nAct] += L0A[mAct, kAct] × L0B[kAct, nAct]
                    // initMmad=true (kL0Itr==0) 时清空 L0C, false 时累加到已有结果
                    tileMmad(l0CTile, l0ATensorTla, l0BTensorTla,
                             mAligned, nAct, kAct, initMmad);
                    // 释放 L0A 给下一轮 MTE1 加载
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                }
            }

            // === A2 fixpipe: L0C → GM(workspace) ===
            if (nL0Itr == 0) {
                if (bi == 0U && AscendC::GetBlockIdx() == 0U) {
#if 0
                    AscendC::printf("[QK] before WaitCrossCoreSync bi=%u flagId=%u\n", bi, mm1ToSmFlag.id);
#endif
                }
                WaitCrossCoreSync<0x2U, PIPE_FIX>(mm1ToSmFlag);
                if (bi == 0U && AscendC::GetBlockIdx() == 0U) {
#if 0
                    AscendC::printf("[QK] after WaitCrossCoreSync bi=%u\n", bi);
#endif
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);

            // Fixpipe L0C → GM workspace (double-buffered by bi%2)
            // A2 Fixpipe requires mSize aligned to 16 (fractal size). The matmul
            // already computes RoundUp(M,16) rows in L0C, so we fixpipe all of them.
            // VEC reads only M rows via DataCopyPad; extra rows are harmless padding.
            // TODO 搬运形状和坐标存疑
            uint32_t gmStageOffset = (bi % 2U) * gmSStageElems_;
            uint32_t mFixAligned = RoundUp(M, 16U);
            auto gmSStageTensor = tla::MakeTensor(
                gmSWorkspace_[gmStageOffset], gmSLayoutEx, Arch::PositionGM{});
            auto gmSTile = GetTile(gmSStageTensor,
                tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mFixAligned, nAct));
            auto l0CFullTile = GetTile(l0CTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(mFixAligned, nAct));
            copyL0CToGmSub0(gmSTile, l0CFullTile);
#if KERNEL_DUMP
            if (AscendC::GetBlockIdx() == 1U && bi == 0U) {
                AscendC::DumpTensor(gmSWorkspace_, 991, M * N);
            }
#endif

            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }

        // Signal VEC that S is ready in GM workspace.
        SetCrossCoreSync<0x2U, PIPE_FIX>(mm1ToSmFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    }
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_MMAD_QK_PREFILL_A2_HPP
