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
#ifndef GEMM_BLOCK_MMAD_QK_PREFILL_ARCH35_HPP
#define GEMM_BLOCK_MMAD_QK_PREFILL_ARCH35_HPP

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
struct BlockMmadQKPrefillArch35 {
    // === 调度策略与架构标签 ===
    using DispatchPolicy = MmadAtlasA5PrefillQK;              // A5 Prefill QK 专用调度策略
    using ArchTag = typename DispatchPolicy::ArchTag;          // 架构标签 (AtlasA5), 用于 Resource/TileCopy 派发

    // === 数据类型定义 ===
    using ElementQ = ElementQ_;                                // Q 输入类型 (bf16, 从 GM 读取)
    using ElementK = ElementK_;                                // K 输入类型 (bf16, 从 GM 读取到 L1 常驻)
    using ElementS = ElementS_;                                // S 输出类型 (bf16, fixpipe 到 UB)
    using ElementAccumulator = float;                          // L0C 累加类型 (fp32, MMAD 内部累加精度)

    // === 内存布局标签 ===
    // Q 按行存储 (RowMajor): [num_q_tokens, D], 每行一个 Q token
    // K 按列存储 (ColumnMajor/转置): [D, blockSize], 列方向是 KV token — 这样 Q×K 就是 Q RowMajor × K ColumnMajor = 标准矩阵乘
    // S 按行存储 (RowMajor): [M, N], M=Q行数, N=KV token数
    using LayoutTagQ = layout::RowMajor;
    using LayoutTagK = layout::ColumnMajor;
    using LayoutTagS = layout::RowMajor;

    // === TileCopy: 数据搬运组件工厂 ===
    // PackedTileCopyTlaToUB 封装了 GM→L1, L1→L0A/L0B, L0C→UB( Fixpipe) 的搬运逻辑
    // CopyL0CToUBMode::NO_SPLIT: Fixpipe 不拆分 (S 整块写到 UB, 后续手动按 group 分给 2 个 AIV)
    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK,
        ElementS, LayoutTagS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;

    // 从 TileCopy 推导各搬运组件和 L1/L0 布局
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;       // L1→L0A (Q tile 搬运)
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;       // L1→L0B (K tile 搬运)
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;     // L1 中 Q 的布局 (NZ 格式)
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;     // L1 中 K 的布局 (NZ 格式)
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;     // L0A 中 Q 的布局
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;     // L0B 中 K 的布局

    // TileMmad: Cube 矩阵乘 (MMAD) 组件, 执行 L0A × L0B → L0C
    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, LayoutTagL1A>;

    // === L0 / L1 缓冲参数 ===
    static constexpr uint32_t L0_STAGES = 2;                   // L0A/L0B/L0C 双缓冲 (乒乓交替)
    static constexpr uint32_t L0_TILE_M = 128;                 // L0 M 维 tile 大小 (Q token 行数, 最多 8 groups × 16 = 128)
    static constexpr uint32_t L0_TILE_N = 128;                 // L0 N 维 tile 大小 (KV token 列数, = blockSize)
    static constexpr uint32_t L0_TILE_K = 128;                 // L0 K 维 tile 大小 (head 维度 D)
    // 双缓冲: L0A/L0B/L0C 各分成 2 级, 一级做 MMAD 时另一级可加载下一批数据
    static constexpr uint32_t L0A_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;  // 每级 L0A 大小
    static constexpr uint32_t L0B_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;  // 每级 L0B 大小
    static constexpr uint32_t L0C_HALF_SIZE = ArchTag::L0C_SIZE / 2;         // L0C 分两半 (QK/PV 各一半)
    static constexpr uint32_t L0C_BUF_SIZE = L0C_HALF_SIZE / L0_STAGES;      // 每级 L0C 大小
    static constexpr uint32_t L1_Q_STAGES = 1;                 // L1 Q 单缓冲 (每次 batch 重新加载)
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;       // V0/V1 flag id 偏移 (MODE_4 双 AIV)
    static constexpr uint32_t COPY_GRANULARITY = 2;            // Q gather 粒度: 每次 2 个 group
    static constexpr uint32_t C0_SIZE = 16;                    // Cube 基本块大小 (16×16)

    // === L1/L0 缓冲 Tensor ===
    AscendC::LocalTensor<ElementK> l1KTensor_;                 // L1: K 常驻 buffer (每 task 加载一次)
    AscendC::LocalTensor<ElementQ> l1QTensor_[L1_Q_STAGES];   // L1: Q buffer (每 batch gather 重新加载)
    AscendC::LocalTensor<ElementQ> l0ATensor_[L0_STAGES];     // L0A: Q tile (双缓冲, 乒乓 MMAD)
    AscendC::LocalTensor<ElementK> l0BTensor_[L0_STAGES];     // L0B: K tile (双缓冲, K 常驻在 stage 0)
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_[L0_STAGES]; // L0C: S 累加结果 (双缓冲, 按 batch 乒乓)

    // === 搬运/计算组件实例 ===
    CopyL1ToL0A copyL1ToL0A;                                   // L1→L0A 搬运器
    CopyL1ToL0B copyL1ToL0B;                                   // L1→L0B 搬运器
    TileMmad tileMmad;                                         // MMAD 矩阵乘器

    uint32_t l1KBufBytes_;                                     // L1 K buffer 占用字节数 (用于 Q buffer 偏移)

    __aicore__ inline
    BlockMmadQKPrefillArch35() {}

    // 跨核同步: Set flag (CUBE → VEC 方向)
    //
    // 【作用】CUBE 完成 QK (Fixpipe S 到 UB) 后, 通知 VEC 可以读取 S 做 softmax
    //
    // 【MODE_4 双 AIV】
    //   1 AIC : 2 AIV 混合核模式下, 一次通知要同时发给 2 个 AIV 子核:
    //     flagIdV0: AIV0 的 flag (原 id)
    //     flagIdV1: AIV1 的 flag (原 id + 16, 即 V0_V1_FLAG_ID_OFFSET)
    //   两个 AIV 各自 Wait 自己的 flag, 都收到后才能开始 softmax
    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    // 跨核同步: Wait flag (CUBE 侧等待 VEC 的信号)
    //
    // 【作用】CUBE 等待 VEC 完成 softmax (UB S buffer 安全可覆盖)
    //   在 Fixpipe 前调用: 确保上一个 batch 的 VEC softmax 已读完 UB S, 不会被新 S 覆盖
    //
    // 【MODE_4 双 AIV】同时等待 2 个 AIV 的 flag (V0 + V1), 都到齐才继续
    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    // 初始化: 在 L1/L0 中分配 buffer
    //
    // 【L1 布局】 (K 和 Q 共享 L1, K 在前 Q 在后)
    //   ┌──────────────────────────────────────────┐
    //   │ K buffer  [blockSize × D × sizeof(K)]     │ ← l1KTensor_, 常驻 (每 task 加载一次)
    //   ├──────────────────────────────────────────┤
    //   │ Q buffer  [L0_TILE_M × D × sizeof(Q)]    │ ← l1QTensor_[0], 每 batch 重新 gather
    //   └──────────────────────────────────────────┘
    //
    // 【L0 布局】 (双缓冲, 每级大小 = 总大小 / L0_STAGES)
    //   L0A[0], L0A[1]: Q tile 双缓冲 (乒乓: 一级 MMAD, 另一级加载)
    //   L0B[0], L0B[1]: K tile 双缓冲 (K 常驻在 L0B[0], L0B[1] 给 PV 的 V)
    //   L0C[0], L0C[1]: S 累加结果双缓冲 (按 batch 乒乓, bi%2 选择)
    __aicore__ inline
    void Init(Arch::Resource<ArchTag> &resource, uint32_t blockSize, uint32_t D)
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
    }

    // 加载 K 到 L1 并常驻到 L0B[0] (每个 task 只调用一次)
    //
    // 【为什么 K 要常驻?】
    //   KV-centric 遍历中, 一个 task 处理 1 个 KV block × 多个 Q token:
    //     - K (该 KV block) 只需加载一次, 后续所有 batch 的 QK 都复用这份 K
    //     - Q 则每 batch 不同 (gather 不同的 Q token), 需要 re-load
    //   K 常驻在 L0B[0], L0B[1] 留给 PV 的 V (QK/PV 共享 L0B 但用不同 stage, 无冲突)
    //
    // 【数据流】
    //   GM(K) ──MTE2──→ L1(K) ──MTE1──→ L0B[0](K)
    //                     ↑ co-located (GM→L1 和 L1→L0B 在同一函数内完成)
    //
    // 【Flag 同步】
    //   MTE1_MTE2 (EVENT_ID0): GM→L1 加载与上一个 task L1→L0B 的 MTE1 同步
    //     Wait: 等 MTE1 (上 task 的 L1→L0B) 完成, L1 空间可覆盖
    //     Set+Wait MTE2_MTE1: 等 GM→L1 完成, 才能开始 L1→L0B
    //   M_MTE1 (flag 2): L0B[0] 的 MMAD ↔ MTE1 同步
    //     Wait<M_MTE1>(2): 等 L0B[0] 从上 task 最后一个 batch 的 MMAD 释放 (InitSyncFlags 预置首个 task)
    //     Set<MTE1_M>(2): L0B[0] 已加载完成, 通知首个 batch 的 MMAD 可以读取
    //     最后一个 batch 的 Set<M_MTE1>(2): 释放 L0B[0] 给下一个 task 的 LoadKResident
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

        // K resident in L0B[0] (stage 0, flag 2): copy L1->L0B once per task, co-located with
        // the GM->L1 load (above). De-aliased from PV's V (L0B[1], flag 3). Wait<M_MTE1>(2):
        // L0B[0] free from the prev task's last QK batch drain (InitSyncFlags primes first
        // task). Set<MTE1_M>(2): L0B[0] ready; operator()'s first QK batch (bi==0) Wait<MTE1_M>(2)
        // consumes it before MMA; the last QK batch's Set<M_MTE1>(2) drains for the next task.
        auto l0BKLayout = tla::MakeLayout<ElementK, LayoutTagL0B>(D, validSize);
        auto l0BKTensor = tla::MakeTensor(l0BTensor_[0], l0BKLayout, Arch::PositionL0B{});
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2U);
        copyL1ToL0B(l0BKTensor, l1KTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2U);
    }

    // =====================================================================================
    // operator(): 执行一次 batch 的 QK 矩阵乘 — S = Q × K^T
    //
    // 【批量 QK (Batched QK) 设计】
    //   将选择同一 KV block 的多个 Q token 按 batchGroupsMax 分批:
    //     M = groupCount × groupRows (最多 L0_TILE_M=128 行)
    //     每个 group 对应一个 Q token, groupRows = groupSize (GQA 分组大小)
    //   Q token 在 GM 中不连续 (按 CSR 索引 scatter 分布), 需要 "gather" (收集) 到 L1 连续空间:
    //     GM: Q[qToken0], Q[qToken1], Q[qToken2]... (分散)
    //     L1: [Q[qToken0] rows][Q[qToken1] rows][Q[qToken2] rows]... (连续, 一次 MMAD)
    //   优化: 用 ndNum=2 的 DataCopy 一次搬 2 个 group, 减少 MTE2 启动次数
    //
    // 【参数说明】
    //   gQ               — GM 中的 Q tensor
    //   qTokens          — 当前 batch 的 Q token id 数组 (从 CSR gather 的结果)
    //   groupCount       — 当前 batch 的 group 数 (= Q token 数)
    //   groupRows        — 每个 group 的行数 (= groupSize, GQA 分组)
    //   ubSTensor        — UB 中的 S 输出 tensor (fixpipe 目标)
    //   validSize        — KV block 有效 token 数 (非尾块=blockSize, 尾块可能更少)
    //   D                — head 维度 (K 维)
    //   perQTokenStrideQ — Q token stride in GM (= qHeads × D), 用于计算 qBase
    //   qHeadStartBase   — Q head 起始偏移 (= qHeadStart × D), 用于 GQA
    //   numBatches       — 总 batch 数 (用于判断首/末 batch)
    //   bi               — 当前 batch 索引 (用于 L0C 乒乓选择)
    //   mm1ToSmFlag      — 跨核同步 flag: CUBE(QK) → VEC(softmax)
    // =====================================================================================
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
        (void)gmQEx;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorS>;
        CopyL0CToDst copyL0CToDstSub0;
        CopyL0CToDst copyL0CToDstSub1;

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
                // A5 (950) 平台: srcNdMatrixStride 不限制小于 65536 (A2 有限制)
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
                    uint32_t l0BBufId = kL0Itr % L0_STAGES;
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
                    // K 常驻 L0B[0] 策略: L1→L0B 只在 LoadKResident 做一次 (每 task), 此处仅 gate/drain:
                    //   firstQKBatch (bi==0): Wait<MTE1_M>(2) 等 LoadKResident 的 Set<MTE1_M>(2) 完成
                    //   lastQKBatch (bi+1>=numBatches): Set<M_MTE1>(2) 释放 L0B[0] 给下一个 task
                    //   中间 batch: K 已常驻, 直接读取, 无需 gate/drain
                    //   L0B[1] 给 PV 的 V, flag 3, QK/PV 之间通过 L0A flag 0 同步 (Q/P 按 batch 复用 L0A)
                    bool firstQKBatch = (bi == 0U);
                    bool lastQKBatch = (bi + 1U >= numBatches);

                    bool initMmad = (kL0Itr == 0);
                    uint32_t mAligned = RoundUp(mAct, 16);

                    // --- MMAD 执行: S += Q_tile × K_tile ---
                    // MTE1_M: 等 L0A/L0B 数据就绪
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    if (firstQKBatch) {
                        // 首个 batch: 等 K 加载到 L0B[0] 完成
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    }
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
                    if (lastQKBatch) {
                        // 末个 batch: 释放 L0B[0] 给下一个 task 的 LoadKResident
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    }
                }
            }

            // === 阶段 3: Fixpipe — L0C(fp32) → UB(bf16), 按 group 分割到 2 个 AIV ===
            // 跨核同步: 等 VEC 完成 softmax (UB S buffer 安全可覆盖)
            if (nL0Itr == 0) {
                WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
            }
            // M_FIX / FIX_M 同步: MMAD → Fixpipe 的 L0C 读写同步
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);   // MMAD 完成, L0C 可 Fixpipe
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);  // 等 Fixpipe 就绪

            uint32_t nFixPAligned16 = RoundUp(nAct, 16U);
            // S 按 GROUP (不是 M/2) 分割到 2 个 AIV:
            //   AIV0: groups [0, gSplit)     → mSub0 = gSplit × groupRows
            //   AIV1: groups [gSplit, end)   → mSub1 = (groupCount - gSplit) × groupRows
            //   按 group 分割保证一个 group 不会跨越两个 AIV (与 softmax/ScatterBatchStats 一致)
            //   groupCount 为奇数时: AIV0 取 ceil(groupCount/2), AIV1 取剩余 (可能为 0)
            uint32_t gSplit = CeilDiv(groupCount, 2U);
            uint32_t mSub0 = gSplit * groupRows;
            uint32_t mSub1 = (groupCount - gSplit) * groupRows;
            // Fixpipe AIV0 的部分: L0C[0:mSub0] → UB S[0:mSub0]
            auto ubSTileSub0 = GetTile(ubSTensor,
                tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mSub0, nFixPAligned16));
            auto l0CTileSub0 = GetTile(l0CTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(mSub0, nAct));
            copyL0CToDstSub0(ubSTileSub0, l0CTileSub0, false);  // false = AIV0
            // Fixpipe AIV1 的部分: L0C[mSub0:end] → UB S (同一 UB 起始, 不同 AIV)
            if (mSub1 > 0U) {
                auto ubSTileSub1 = GetTile(ubSTensor,
                    tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mSub1, nFixPAligned16));
                auto l0CTileSub1 = GetTile(l0CTensorTla,
                    tla::MakeCoord(mSub0, 0), tla::MakeShape(mSub1, nAct));
                copyL0CToDstSub1(ubSTileSub1, l0CTileSub1, true);  // true = AIV1
            }
            // 释放 L0C 给下一个同 stage batch 的 MMAD
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
        // 通知 VEC: 本 batch 的 S 已 Fixpipe 到 UB, 可以开始 softmax
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
        // 释放 L1 Q 空间给下一个 batch 的 Q gather
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    }
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_MMAD_QK_PREFILL_ARCH35_HPP
