/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * =========================================================================================
 * BlockMmadPVPrefillArch35 —— A5 (Arch35) 平台 Prefill PV 矩阵乘 Block
 * =========================================================================================
 *
 * 【一句话概括】
 *   计算注意力加权值: O_partial = P × V (softmax 概率乘以 Value)
 *   这是稀疏注意力 Prefill 阶段的第二个 Cube (矩阵乘) 组件。
 *
 * 【在整体架构中的位置】
 *   Phase1 三级流水线的第三级 (PV):
 *     QK (Cube) → softmax (Vector) → PV (本文件)
 *   本组件滞后 QK 两个 batch (PRE_LAUNCH=2), 等 softmax 生成 P 后才执行:
 *     batch bi: QK 计算 bi, softmax 计算 bi-1, PV 计算 bi-2
 *   输入: P (softmax 概率, 从 L1 读取) × V (Value, 常驻 L0B[1])
 *   输出: O_partial (未归一化的部分结果, Fixpipe 到 GM workspace)
 *
 * 【与 QK Block 的关键差异】
 *   ┌──────────────┬────────────────────────────┬─────────────────────────────────────┐
 *   │              │ QK Block                   │ PV Block (本文件)                   │
 *   ├──────────────┼────────────────────────────┼─────────────────────────────────────┤
 *   │ 计算         │ S = Q × K^T                │ O_partial = P × V                   │
 *   │ 左操作数     │ Q (每 batch 重新 gather)    │ P (softmax 输出, 从 L1 读取)        │
 *   │ 右操作数     │ K (常驻 L0B[0])            │ V (常驻 L0B[1])                     │
 *   │ K 常驻 L0B   │ L0B[0], flag 2             │ L0B[1], flag 3                      │
 *   │ L0C 区域     │ 下半区 (L0C[0..1])         │ 上半区 (L0C_HALF_SIZE + [0..1])     │
 *   │ 输出目标     │ UB (Fixpipe, 供 VEC softmax)│ GM workspace (NZ2ND, 供 Phase2 合并)│
 *   │ 跨核方向     │ CUBE → VEC (通知 softmax)  │ VEC → CUBE (等待 softmax 完成)      │
 *   │ P 布局       │ N/A                        │ zN (Cube 专用 NZ 格式, 按分形存储)   │
 *   └──────────────┴────────────────────────────┴─────────────────────────────────────┘
 *
 * 【V 常驻策略】
 *   与 QK 的 K 常驻类似: V 以 KV block 为单位加载, 每个 task 只加载一次到 L1/L0B[1]。
 *   后续所有 batch 的 PV 都复用这份 V, 避免 GM 重复加载。
 *
 * 【O_partial Scatter 到 GM】
 *   L0C 的 O_partial 结果 [batchM, D] 需要按 qToken scatter (分散写入) 到 GM workspace:
 *     - 每个 qToken 有自己的 slot (由 qToken × kvHeads × topK × slotOElems 定位)
 *     - 用 NZ2ND Fixpipe (L0C→GM) 将 Cube 格式结果转成 RowMajor 写到 GM
 *     - ndNum=2 优化: 一次写 2 个 qToken 的 slot, 减少 MTE3 启动开销
 *
 * 【数据流图】
 *   L1(P) ←──── VEC softmax 写入 (跨核 flag 同步)
 *      │ MTE1
 *      ↓
 *   L0A(P) ──┐
 *            ├── MMAD ──→ L0C(O_partial) ──NZ2ND Fixpipe──→ GM workspace (scatter)
 *   L0B(V) ──┘                                           (每个 qToken 各一个 slot)
 *      ↑
 *   V 常驻 (LoadVResident, 每 task 一次)
 * =========================================================================================
 */
#ifndef GEMM_BLOCK_MMAD_PV_PREFILL_ARCH35_HPP
#define GEMM_BLOCK_MMAD_PV_PREFILL_ARCH35_HPP

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
// BlockMmadPVPrefillArch35: A5 平台 Prefill PV 矩阵乘组件
//
// 模板参数:
//   ElementP_ — P 输入元素类型 (softmax 输出, 通常 bf16, zN 布局)
//   ElementV_ — V 输入元素类型 (通常 bf16, 从 GM 读取到 L1 常驻)
//   ElementO_ — O_partial 输出类型 (fp32 或 bf16, 取决于 innerPrecise, 写到 GM workspace)
// =========================================================================================
template <class ElementP_, class ElementV_, class ElementO_>
struct BlockMmadPVPrefillArch35 {
    // === 调度策略与架构标签 ===
    using DispatchPolicy = MmadAtlasA5PrefillPV;               // A5 Prefill PV 专用调度策略
    using ArchTag = typename DispatchPolicy::ArchTag;          // 架构标签 (AtlasA5)

    // === 数据类型定义 ===
    using ElementP = ElementP_;                                // P 类型 (bf16, softmax 概率输出)
    using ElementV = ElementV_;                                // V 类型 (bf16, Value 输入)
    using ElementO = ElementO_;                                // O_partial 类型 (fp32 或 bf16, 写到 GM)
    using ElementAccumulator = float;                          // L0C 累加类型 (fp32, MMAD 内部累加精度)

    // === 内存布局标签 ===
    // P 使用 zN 布局: Cube 专用的 NZ (N-Z fractal) 格式, 按 16×16 分形存储
    //   zN = 按列方向分形, 适合 Cube L0A 的读取模式 (与 QK 的 RowMajor Q 不同)
    //   softmax 组件将 P 以 zN 格式写入 L1, 本组件直接从 L1 读取
    // V 按行存储 (RowMajor): [blockSize, D], 行方向是 KV token
    // O 按行存储 (RowMajor): [groupRows, D], 写到 GM workspace
    using LayoutTagP = layout::zN;
    using LayoutTagV = layout::RowMajor;
    using LayoutTagO = layout::RowMajor;

    // === TileCopy: 数据搬运组件工厂 ===
    // PackedTileCopyTla (不含 ToUB): 输出到 GM 而非 UB, 使用 NZ2ND Fixpipe (L0C→GM)
    using TileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementP, LayoutTagP, ElementV, LayoutTagV,
        ElementO, LayoutTagO, void>;

    // 从 TileCopy 推导各搬运组件和布局
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;       // L1→L0A (P tile 搬运)
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;       // L1→L0B (V tile 搬运)
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;     // L1 中 P 的布局 (zN)
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;     // L1 中 V 的布局 (NZ)
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;     // L0A 中 P 的布局
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;     // L0B 中 V 的布局

    // TileMmad: Cube 矩阵乘 (MMAD) 组件, 执行 L0A(P) × L0B(V) → L0C(O_partial)
    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementP, LayoutTagL1A>;

    // === L0 / L1 缓冲参数 (与 QK Block 相同) ===
    static constexpr uint32_t L0_STAGES = 2;                   // L0A/L0B/L0C 双缓冲
    static constexpr uint32_t L0_TILE_M = 128;                 // L0 M 维 tile (Q token 行数)
    static constexpr uint32_t L0_TILE_N = 128;                 // L0 N 维 tile (head 维度 D)
    static constexpr uint32_t L0_TILE_K = 128;                 // L0 K 维 tile (KV token 数)
    static constexpr uint32_t L0A_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    // L0C 分上下两半: QK 用下半 (offset 0), PV 用上半 (offset L0C_HALF_SIZE)
    static constexpr uint32_t L0C_HALF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_BUF_SIZE = L0C_HALF_SIZE / L0_STAGES;
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;       // V0/V1 flag id 偏移 (MODE_4 双 AIV)
    static constexpr uint32_t COPY_GRANULARITY = 2;            // Scatter 粒度: 每次 2 个 group
    static constexpr uint32_t C0_SIZE = 16;                    // Cube 基本块大小

    // === L1/L0 缓冲 Tensor ===
    AscendC::LocalTensor<ElementV> l1VTensor_;                 // L1: V 常驻 buffer (每 task 加载一次)
    AscendC::LocalTensor<ElementP> l1PTensor_;                 // L1: P buffer (softmax 写入, 按 stage 轮转)
    AscendC::LocalTensor<ElementP> l0ATensor_[L0_STAGES];     // L0A: P tile (双缓冲)
    AscendC::LocalTensor<ElementV> l0BTensor_[L0_STAGES];     // L0B: V tile (V 常驻在 L0B[1], L0B[0] 给 QK 的 K)
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_[L0_STAGES]; // L0C: O_partial 累加结果 (上半区, 双缓冲)

    // === 搬运/计算组件实例 ===
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    TileMmad tileMmad;

    // === L1 管理状态 ===
    Arch::Resource<ArchTag> *resourcePtr_;                     // Resource 指针 (用于 SetL1PBuf 重新绑定)
    uint32_t l1BaseAddr_;                                      // L1 起始偏移 (V buffer 起点, 在 QK 的 K/Q 之后)
    uint32_t vBufBytes_;                                       // V buffer 字节数
    uint32_t l1PStageBytes_;                                   // 每个 P stage 的字节数 (用于多级 P buffer 轮转)
    // V 常驻的 K 维 (block valid size):
    //   operator() 的 L1 V layout 必须用这个 K, 而非 gemm 的 causalValidLen (≤ block valid size)。
    //   zN 的 N-fractal stride = RoundUp(K,16)*16 依赖这个 K; 如果 layout 用 causalValidLen
    //   而 L1 数据按 block valid size 排列, stride 不一致 → N≥16 的 O_partial 损坏。
    //   (QK 不存在此问题, 因为 QK 的 K=D 是常驻维, 与 gemm K 相同)
    uint32_t residentValidSize_ = 0;

    __aicore__ inline
    BlockMmadPVPrefillArch35()
        : resourcePtr_(nullptr), l1BaseAddr_(0), vBufBytes_(0), l1PStageBytes_(0) {}

    // 跨核同步: Set flag (CUBE → VEC 方向)
    //
    // 【作用】PV 的最后一个 K tile 完成后, 通知 VEC 可以覆写 L1 P buffer
    //   (PV 读 P 完成后, softmax 可以写下一个 batch 的 P 到同一 L1 区域)
    //
    // 【MODE_4 双 AIV】同时 Set 2 个 flag (V0 + V1, 偏移 16), 通知两个 AIV
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

    // 跨核同步: Wait flag (CUBE 等待 VEC 的信号)
    //
    // 【作用】PV 开始前等待 VEC softmax 完成 — P 已写入 L1, 可以读取做 PV matmul
    //   这是三级流水线的 softmax→PV 同步: softmax(bi-1) 完成 → PV(bi) 可开始
    //
    // 【MODE_4 双 AIV】同时 Wait 2 个 flag (V0 + V1), 两个 AIV 的 softmax 都完成才继续
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
    // 【L1 布局】 (V 和 P 共享 L1, V 在前 P 在后, l1StartAddr 接在 QK 的 K/Q 之后)
    //   ┌──────────────────────────────────────────────────────────────┐
    //   │ V buffer  [blockSize × D × sizeof(V)]    ← l1VTensor_         │ 常驻 (每 task 加载一次)
    //   ├──────────────────────────────────────────────────────────────┤
    //   │ P stage 0 [RoundUp(groupSize,16) × RoundUp(blockSize,16)    │ ← l1PTensor_ (softmax 写入)
    //   │           × sizeof(P)]                                       │   按 stage 轮转 (SetL1PBuf 切换)
    //   ├──────────────────────────────────────────────────────────────┤
    //   │ P stage 1 ...                                                │   (多级 P buffer, 流水线用)
    //   └──────────────────────────────────────────────────────────────┘
    //
    // 【L0 布局】 (双缓冲, PV 用 L0C 上半区)
    //   L0A[0], L0A[1]: P tile 双缓冲 (与 QK 共享 L0A, 按 batch 乒乓)
    //   L0B[1]:         V 常驻 (L0B[0] 给 QK 的 K, PV/ QK 各用不同 stage 不冲突)
    //   L0C[0], L0C[1]: O_partial 双缓冲 (上半区: L0C_HALF_SIZE + offset, 与 QK 下半区不冲突)
    __aicore__ inline
    void Init(Arch::Resource<ArchTag> &resource, uint32_t l1StartAddr,
              uint32_t blockSize, uint32_t D, uint32_t groupSize)
    {
        vBufBytes_ = blockSize * D * sizeof(ElementV);
        l1BaseAddr_ = l1StartAddr;
        l1PStageBytes_ = RoundUp(groupSize, 16U) * RoundUp(blockSize, 16U) * sizeof(ElementP);
        resourcePtr_ = &resource;
        l1VTensor_ = resource.l1Buf.template GetBufferByByte<ElementV>(l1StartAddr);
        l1PTensor_ = resource.l1Buf.template GetBufferByByte<ElementP>(l1StartAddr + vBufBytes_);

        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor_[i] = resource.l0ABuf.template GetBufferByByte<ElementP>(L0A_BUF_SIZE * i);
            l0BTensor_[i] = resource.l0BBuf.template GetBufferByByte<ElementV>(L0B_BUF_SIZE * i);
            l0CTensor_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_HALF_SIZE + L0C_BUF_SIZE * i);
        }
    }

    // 切换 L1 P buffer 到指定 stage (流水线轮转)
    //
    // 【作用】三级流水线中, softmax 和 PV 需要多级 P buffer:
    //   - softmax(bi) 写 P 到 stage A
    //   - PV(bi) 读 P 从 stage B (滞后 PRE_LAUNCH=2 个 batch)
    //   通过 SetL1PBuf 在每 batch 切换 P buffer, 使读写不冲突
    __aicore__ inline
    void SetL1PBuf(uint32_t l1PBufId)
    {
        l1PTensor_ = resourcePtr_->l1Buf.template GetBufferByByte<ElementP>(
            l1BaseAddr_ + vBufBytes_ + l1PStageBytes_ * l1PBufId);
    }

    // 加载 V 到 L1 并常驻到 L0B[1] (每个 task 只调用一次)
    //
    // 【为什么 V 要常驻?】
    //   与 QK 的 K 常驻同理: KV-centric 遍历中, 一个 task 处理 1 个 KV block × 多个 Q token:
    //     - V (该 KV block) 只需加载一次, 后续所有 batch 的 PV 都复用这份 V
    //     - P 则每 batch 不同 (softmax 重新生成), 需要从 L1 按 stage 轮转读取
    //   V 常驻在 L0B[1], L0B[0] 给 QK 的 K (两个 Cube Block 共享 L0B 但用不同 stage, 无冲突)
    //
    // 【数据流】
    //   GM(V) ──MTE2──→ L1(V) ──MTE1──→ L0B[1](V)
    //                    ↑ co-located (GM→L1 和 L1→L0B 在同一函数内完成)
    //
    // 【residentValidSize_ 记录】
    //   保存 V 的 K 维 (block valid size), 供 operator() 构造 L1 V layout 使用
    //   必须用这个 K 而非 causalValidLen, 否则 zN stride 不一致 (见成员变量注释)
    //
    // 【Flag 同步】
    //   MTE1_MTE2 (EVENT_ID2): GM→L1 加载与上 task L1→L0B 的 MTE1 同步
    //   M_MTE1 (flag 3): L0B[1] 的 MMAD ↔ MTE1 同步
    //     Wait<M_MTE1>(3): 等 L0B[1] 从上 task 最后 batch 的 MMAD 释放 (InitSyncFlags 预置首个 task)
    //     Set<MTE1_M>(3): L0B[1] 已加载完成, 通知首个 batch 的 MMAD 可读
    //     末个 batch Set<M_MTE1>(3): 释放 L0B[1] 给下一个 task
    template <class TensorV>
    __aicore__ inline
    void LoadVResident(TensorV &gmVTensor, uint32_t validSize, uint32_t D)
    {
        residentValidSize_ = validSize;
        using CopyGmToL1V = typename TileCopy::template CopyGmToL1B<TensorV>;
        CopyGmToL1V copyGmToL1V;

        auto l1VLayout = tla::MakeLayout<ElementV, LayoutTagL1B>(validSize, D);
        auto l1VTensorTla = tla::MakeTensor(l1VTensor_, l1VLayout, Arch::PositionL1{});
        auto l1VTile = GetTile(l1VTensorTla,
            tla::MakeCoord(0, 0), tla::MakeShape(validSize, D));
        auto gmVTile = GetTile(gmVTensor,
            tla::MakeCoord(0, 0), tla::MakeShape(validSize, D));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        copyGmToL1V(l1VTile, gmVTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);

        // V resident in L0B[1] (stage 1, flag 3): copy L1->L0B once per task, co-located with
        // the GM->L1 load (above). De-aliased from QK's K (L0B[0], flag 2). Wait<M_MTE1>(3):
        // L0B[1] free from the prev task's last PV batch drain (InitSyncFlags primes first
        // task). Set<MTE1_M>(3): L0B[1] ready; operator()'s first PV batch (bDe==0) Wait<MTE1_M>(3)
        // consumes it before MMA; the last PV batch's Set<M_MTE1>(3) drains for the next task.
        auto l0BVLayout = tla::MakeLayout<ElementV, LayoutTagL0B>(validSize, D);
        auto l0BVTensor = tla::MakeTensor(l0BTensor_[1], l0BVLayout, Arch::PositionL0B{});
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3U);
        copyL1ToL0B(l0BVTensor, l1VTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(3U);
    }

    // =====================================================================================
    // operator(): 执行一次 batch 的 PV 矩阵乘 — O_partial = P × V
    //
    // 【批量 PV (Batched PV) 设计】
    //   M = groupCount × groupRows (最多 L0_TILE_M=128 行, 与 QK 的 M 相同)
    //   K = validSize (KV block 有效 token 数, 由 V 常驻决定 = residentValidSize_)
    //   N = D (head 维度)
    //
    //   P 是 [batchM, validSize] 的矩阵, 其中每个 group 的 causal 尾部 [cvl[g], validSize)
    //   被 softmax 预置为 0, 所以对 K=validSize 整体做 reduce 等价于有效 exp + 零尾部 (无副作用)。
    //
    //   K = residentValidSize_ (V 常驻的 K), P/V 的 K stride 天然一致 (softmax 已 zero-pad 到 validSize)。
    //
    // 【O_partial Scatter 到 GM】
    //   一次 matmul 产出 O_partial[batchM, D] 在 L0C, 然后按 qToken scatter 到 GM workspace:
    //     每个 qToken 有自己的 slot: wsOOff = qToken × perQTokenStride + kvHeadBase + slotK × slotOElems
    //       perQTokenStride = kvHeads × topK × slotOElems (kernel 预计算, kvHead 内不变)
    //       kvHeadBase       = kvHeadIdx × topK × slotOElems
    //       slotOElems       = groupSize × D
    //   无需传递 wsOOff 数组, 从 qTokens/slotKs 数组按需计算。
    //
    // 【参数说明】
    //   gAccumOut       — GM workspace accumOut tensor
    //   qTokens         — 当前 batch 的 Q token id 数组
    //   slotKs          — 当前 batch 的 slotK 编号数组 (每个 Q token 的 topK slot)
    //   groupCount      — 当前 batch 的 group 数
    //   groupRows       — 每个 group 的行数 (= groupSize)
    //   D               — head 维度 (N 维)
    //   perQTokenStride — Q token stride in workspace (= kvHeads × topK × slotOElems)
    //   kvHeadBase      — KV head 基偏移 (= kvHeadIdx × topK × slotOElems)
    //   slotOElems      — 每 slot 元素数 (= groupSize × D)
    //   numBatches      — 总 batch 数 (判断首/末 batch)
    //   bDe              — 当前 PV batch 索引 (滞后 QK 的 bi 两个 batch, 用于 L0C 乒乓)
    //   smToMm2Flag     — 跨核同步 flag: VEC(softmax) → CUBE(PV)
    // =====================================================================================
    template <class TensorO>
    __aicore__ inline
    void operator()(TensorO &gAccumOut, const uint32_t *qTokens, const uint32_t *slotKs,
                    uint32_t groupCount, uint32_t groupRows,
                    uint32_t D,
                    uint64_t perQTokenStride, uint64_t kvHeadBase, uint64_t slotOElems,
                    uint32_t numBatches, uint32_t bDe,
                    Arch::CrossCoreFlag &smToMm2Flag)
    {
        uint32_t M = groupCount * groupRows;
        uint32_t N = D;
        uint32_t K = residentValidSize_;
        // 等待 VEC softmax 完成: P 已写入 L1, 可以开始 PV matmul
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);

        // GM O_partial slot 布局: [groupRows, N] per slot, RowMajor
        auto gmOLayout = tla::MakeLayout<ElementO, LayoutTagO>(groupRows, N);
        auto gmOSlotEx = tla::MakeTensor(gAccumOut, gmOLayout, Arch::PositionGM{});
        using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<decltype(gmOSlotEx)>;
        CopyL0CToGm copyL0CToGm;
        (void)gmOSlotEx;

        // L1 P layout: [M, K] zN 格式 (softmax 写入, Cube 直接读取)
        auto l1PLayout = tla::MakeLayout<ElementP, LayoutTagL1A>(M, K);
        auto l1PTensorTla = tla::MakeTensor(l1PTensor_, l1PLayout, Arch::PositionL1{});
        // L1 V layout: 必须用 residentValidSize_ 作为 K (而非 causalValidLen)
        //   zN 的 N-fractal stride = RoundUp(K,16)*16 依赖 K, 不一致会导致 N≥16 损坏
        auto l1VLayout = tla::MakeLayout<ElementV, LayoutTagL1B>(residentValidSize_, N);
        auto l1VTensorTla = tla::MakeTensor(l1VTensor_, l1VLayout, Arch::PositionL1{});

        // === MMAD 三重循环: N(外) × M(中) × K(内) ===
        uint32_t nL0Num = CeilDiv(N, L0_TILE_N);
        uint32_t mL0Num = CeilDiv(M, L0_TILE_M);
        uint32_t kL0Num = CeilDiv(K, L0_TILE_K);
        for (uint32_t nL0Itr = 0; nL0Itr < nL0Num; nL0Itr++) {
            uint32_t nAct = (nL0Itr == nL0Num - 1) ? (N - nL0Itr * L0_TILE_N) : L0_TILE_N;
            // L0C 乒乓策略: 按 PV batch bDe 选择 L0C stage (bDe % 2)
            //   bDe=0 → L0C[0] (上半区), bDe=1 → L0C[1], bDe=2 → L0C[0]...
            //   MMA(bDe) 写 L0C[bDe%2] 与 Fixpipe(bDe-1) 读 L0C[(bDe-1)%2] 并行
            //   l0CEventId = bDe%2 + 2: PV 的 L0C flag 偏移 +2 (QK 用 0/1, PV 用 2/3)
            uint32_t l0CBufId = bDe % L0_STAGES;
            uint32_t l0CEventId = l0CBufId + 2;

            auto l0CLayout = tla::MakeLayoutL0C(M, nAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor_[l0CBufId], l0CLayout, Arch::PositionL0C{});
            for (uint32_t mL0Itr = 0; mL0Itr < mL0Num; mL0Itr++) {
                uint32_t mAct = (mL0Itr == mL0Num - 1) ? (M - mL0Itr * L0_TILE_M) : L0_TILE_M;
                auto l0CTile = GetTile(l0CTensorTla,
                    tla::MakeCoord(mL0Itr * L0_TILE_M, 0), tla::MakeShape(mAct, nAct));
                for (uint32_t kL0Itr = 0; kL0Itr < kL0Num; kL0Itr++) {
                    uint32_t kAct = (kL0Itr == kL0Num - 1) ? (K - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    // V 常驻 L0B[1] (stage 1, flag 3):
                    //   L0B[0] 给 QK 的 K (flag 2), L0B[1] 给 PV 的 V (flag 3), 互不冲突
                    //   固定 stage 1, 不按 kL0Itr 乒乓 (kL0Num=1, K=validSize≤128=L0_TILE_K)
                    uint32_t l0BBufId = 1U;
                    uint32_t l0AEventId = l0ABufId;
                    uint32_t l0BEventId = l0BBufId + 2;   // flag 3

                    auto l0BLayout = tla::MakeLayout<ElementV, LayoutTagL0B>(kAct, nAct);
                    auto l0BTensorTla = tla::MakeTensor(
                        l0BTensor_[l0BBufId], l0BLayout, Arch::PositionL0B{});
                    // V 常驻 L0B[1] 的 MMA gate/drain:
                    //   L1→L0B 在 LoadVResident 中完成 (每 task 一次), 此处仅需 gate/drain:
                    //   firstPVBatch (bDe==0): Wait<MTE1_M>(3) 等 V 加载完成
                    //   lastPVBatch (bDe+1>=numBatches): Set<M_MTE1>(3) 释放 L0B[1] 给下一 task
                    //   中间 batch: V 已常驻, 直接读取
                    bool firstPVBatch = (bDe == 0U);
                    bool lastPVBatch = (bDe + 1U >= numBatches);

                    auto l1PSubTile = GetTile(l1PTensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                        tla::MakeShape(mAct, kAct));
                    auto l0ALayout = tla::MakeLayout<ElementP, LayoutTagL0A>(mAct, kAct);
                    auto l0ATensorTla = tla::MakeTensor(
                        l0ATensor_[l0ABufId], l0ALayout, Arch::PositionL0A{});
                    // --- L1→L0A: 搬运 P tile 到 L0A ---
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);  // 等 L0A 空闲
                    copyL1ToL0A(l0ATensorTla, l1PSubTile);                       // L1 P → L0A
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);   // L0A 就绪

                    // 最后一个 K tile: 通知 VEC 可以覆写 L1 P buffer (softmax 写下一个 batch)
                    if ((mL0Itr == mL0Num - 1U) && (nL0Itr == nL0Num - 1U) &&
                        (kL0Itr == kL0Num - 1U)) {
                        SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                    }

                    bool initMmad = (kL0Itr == 0);               // 首个 K tile: 清空 L0C
                    uint32_t mAligned = RoundUp(mAct, 16);
                    // --- MMAD: O_partial += P_tile × V_tile ---
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);  // 等 P 加载完成
                    if (firstPVBatch) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);  // 首batch: 等 V 加载
                    }
                    if (mL0Itr == 0 && kL0Itr == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);  // 等 L0C 空闲
                    }
                    // L0C[mAct, nAct] += L0A[mAct, kAct] × L0B[kAct, nAct]
                    tileMmad(l0CTile, l0ATensorTla, l0BTensorTla,
                             mAligned, nAct, kAct, initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);  // 释放 L0A
                    if (lastPVBatch) {
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);  // 末batch: 释放 L0B
                    }
                }
            }
            // === NZ2ND Fixpipe + Scatter: L0C(O_partial) → GM workspace (按 qToken 分散写入) ===
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);   // MMAD 完成, L0C 可 Fixpipe
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);  // 等 Fixpipe 就绪
            uint32_t mFixPAligned8Grp = RoundUp(groupRows, 8U);        // M 维对齐 (Fixpipe 要求 8 对齐)
            uint32_t nFixPAligned8 = RoundUp(nAct, 8U);                // N 维对齐
            // 将 L0C[batchM, nAct] scatter (分散写入) 到每个 qToken 的 GM slot:
            //   wsOOff = qToken × perQTokenStride + kvHeadBase + slotK × slotOElems
            //   GM slot 布局: [groupRows, D], 只写有效 groupRows 行 (不写对齐 padding)
            //   ndNum=2 优化: 一次写 2 个 qToken 的 slot (dstNdStride 计算两 slot 间距)
            for (uint32_t g = 0; g < groupCount; g += COPY_GRANULARITY) {
                uint64_t wsOOff = static_cast<uint64_t>(qTokens[g]) * perQTokenStride + kvHeadBase
                                + static_cast<uint64_t>(slotKs[g]) * slotOElems;
                uint16_t ndNum = 1;
                uint16_t srcNdStride = 1;
                uint32_t dstNdStride = 1;
                if (g + 1 < groupCount) {
                    uint32_t nextQ = qTokens[g + 1];
                    uint32_t nextSlot = slotKs[g + 1];
                    if (nextQ > qTokens[g]) {
                        dstNdStride = static_cast<int32_t>(nextQ - qTokens[g]) * perQTokenStride + static_cast<int32_t>(nextSlot - slotKs[g]) * static_cast<int32_t>(slotOElems);
                    } else {
                        dstNdStride = static_cast<int32_t>(qTokens[g] - nextQ) * perQTokenStride + static_cast<int32_t>(slotKs[g] - nextSlot) * static_cast<int32_t>(slotOElems);
                        wsOOff = static_cast<uint64_t>(nextQ) * perQTokenStride + kvHeadBase
                                + static_cast<uint64_t>(nextSlot) * slotOElems;
                    }
                    srcNdStride = groupRows;
                    ndNum = COPY_GRANULARITY;
                }
                auto gmOSlot = tla::MakeTensor(gAccumOut[wsOOff], gmOLayout, Arch::PositionGM{});
                auto oTile = GetTile(gmOSlot,
                    tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(groupRows, nFixPAligned8));
                auto l0CTile = GetTile(l0CTensorTla,
                    tla::MakeCoord(g * groupRows, 0), tla::MakeShape(mFixPAligned8Grp, nAct));
                copyL0CToGm(oTile, l0CTile, ndNum, srcNdStride, dstNdStride);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
    }
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_MMAD_PV_PREFILL_ARCH35_HPP
