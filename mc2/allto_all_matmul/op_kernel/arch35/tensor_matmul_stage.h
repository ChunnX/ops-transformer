/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * 基于 Tensor API 的矩阵乘计算阶段
 * 替换 MC2FpMatmul，提供 Init / GetContextPtr / Process / End 接口
 * 核内使用 tensor_api 的 Copy / Mad 完成 GM→L1→L0→MMAD→GM 流水
 *
 * [MODIFY] 新算子需修改的位置：
 *   1. AType / BType / CType — 输入输出数据类型
 *   2. BASE_N / BASE_K     — 基础切分块大小
 *   3. 如需 bias，在 L0C→GM fixpipe 前添加 bias 加法
 */

#ifndef TENSOR_MATMUL_STAGE_H
#define TENSOR_MATMUL_STAGE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../../common/op_kernel/mc2_templates/computation/matmul/matmul_base.h"
#include "tensor_matmul_tiling_data.h"

// ──── Tensor API 层头文件 ────
#include "tensor.h"

// ──── 硬件/算法常量 ────
static constexpr int64_t TENSOR_L0A_SIZE = 64 * 1024;
static constexpr int64_t TENSOR_L0B_SIZE = 64 * 1024;
static constexpr int64_t TENSOR_L0C_SIZE = 256 * 1024;
static constexpr int64_t TENSOR_TOTAL_L1_SIZE = 512 * 1024;
static constexpr int64_t TENSOR_BT_SIZE = 4096;

static constexpr uint64_t TENSOR_DOUBLE_BUFFER_COUNT = 2UL;
static constexpr uint64_t TENSOR_HALF_L0_SIZE = TENSOR_L0A_SIZE / TENSOR_DOUBLE_BUFFER_COUNT;
static constexpr uint64_t TENSOR_HALF_L0C_SIZE = TENSOR_L0C_SIZE / TENSOR_DOUBLE_BUFFER_COUNT / sizeof(float);

// Cube 粒度: fp16/bf16 = 16, fp8 = 32
static constexpr uint64_t TENSOR_BLOCK_CUBE = 16UL;

// MMAD 累加模式: 最后一次累加用 FINAL_ACCUMULATION=3, 中间用 NON_FINAL_ACCUMULATION=2
static constexpr uint32_t TENSOR_FINAL_ACCUMULATION = 3;
static constexpr uint32_t TENSOR_NON_FINAL_ACCUMULATION = 2;

// 函数式辅助
template <typename T>
__aicore__ inline T TensorCeilDiv(T a, T b)
{
    if (b == 0) { return a; }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T TensorCeilAlign(T a, T b)
{
    if (b == 0) { return a; }
    return (a + b - 1) / b * b;
}

// ═══════════════════════════════════════════════════════════════════
// TensorMatmulStage — 单任务矩阵乘阶段
// ═══════════════════════════════════════════════════════════════════

namespace Mc2Kernel {

template <typename TilingType>
class TensorMatmulStage {
public:
    // ── 数据类型（可从模板参数实例化时覆盖） ──
    using AType = DTYPE_X1;      // 来自 MC2 模板系统的 DTYPE_X1 (half / bfloat16_t)
    using BType = DTYPE_X2;      // DTYPE_X2
    using CType = DTYPE_Y;       // DTYPE_Y

    // ── 基础切分块大小 ──
    static constexpr uint32_t BASE_N = 128;
    static constexpr uint32_t BASE_K = 128;

    // ── L1 Buffer 相关 ──
    static constexpr uint64_t L1_BUFFER_NUM = 2UL;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1UL;
    static constexpr uint64_t L1_HALF_SIZE = TENSOR_TOTAL_L1_SIZE >> 1;

    __aicore__ inline TensorMatmulStage(AscendC::TPipe *tPipe)
        : tPipePtr_(tPipe) {}

    __aicore__ inline void Init() {}

    __aicore__ inline MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::FpMMAdditionalData, TilingType> *GetContextPtr()
    {
        return &MMcontext_;
    }

    __aicore__ inline void Process(uint32_t taskIndex);

    __aicore__ inline void End() {}

private:
    __aicore__ inline void ComputeSingleBlock(
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
        uint32_t m_, uint32_t k_, uint32_t n_);

private:
    MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::FpMMAdditionalData, TilingType> MMcontext_;
    AscendC::TPipe *tPipePtr_;
};

// ═══════════════════════════════════════════════════════════════════
// Process(taskIndex) — 从 Context 获取 GM 地址，调用 ComputeSingleBlock
// ═══════════════════════════════════════════════════════════════════

template <typename TilingType>
__aicore__ inline void TensorMatmulStage<TilingType>::Process(uint32_t taskIndex)
{
    if ASCEND_IS_AIV {
        return;
    }

    auto& base = MMcontext_.baseData;
    GM_ADDR aGM = base.aGM + taskIndex * base.aOffset;
    GM_ADDR bGM = base.bGM + taskIndex * base.bOffset;
    GM_ADDR cGM = base.cGM + taskIndex * base.cOffset;

    if (taskIndex == 0) {
        AscendC::printf("[TENSOR-MATMUL-FP] Process enter task=%u\n", taskIndex);
    }

    tPipePtr_->Reset();

    // 从 tiling 获取维度信息 (由 ProcessTile/ProcessTail 填入)
    auto* tiling = MMcontext_.tilingDataPtr;
    uint32_t m_ = tiling->tileM;
    uint32_t k_ = tiling->fullK;
    uint32_t n_ = tiling->fullN;

    ComputeSingleBlock(aGM, bGM, cGM, m_, k_, n_);
}

// ═══════════════════════════════════════════════════════════════════
// ComputeSingleBlock — 单块 (m, k, n) 矩阵乘，含 N-tiling + K-tiling
// ═══════════════════════════════════════════════════════════════════

template <typename TilingType>
__aicore__ inline void TensorMatmulStage<TilingType>::ComputeSingleBlock(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
    uint32_t m_, uint32_t k_, uint32_t n_)
{
    // ── 预置 MTE 事件 ──
    #pragma unroll
    for (uint8_t i = 0; i < L1_BUFFER_NUM; ++i) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
    }

    // ── 创建 GM 张量视图 (RowMajor = ND) ──
    auto layoutA = AscendC::Te::MakeNDLayout<AType>(m_, k_);
    auto layoutB = AscendC::Te::MakeNDLayout<BType>(k_, n_);
    auto layoutC = AscendC::Te::MakeNDLayout<CType>(m_, n_);

    auto gmA = AscendC::Te::MakeTensor(
        AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ AType*>(aGM)), layoutA);
    auto gmB = AscendC::Te::MakeTensor(
        AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ BType*>(bGM)), layoutB);
    auto gmC = AscendC::Te::MakeTensor(
        AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ CType*>(cGM)), layoutC);

    // ── 计算单次 L1 可容纳的 K 长度 ──
    // A(NZ): m_ * kL1 * sizeof(AType)
    // B(ZN): kL1 * baseNCur * sizeof(BType)
    // 两者平分一半 L1
    uint32_t kL1 = BASE_K;
    // 动态约束：确保 m_*kL1 和 kL1*BASE_N 不超 half-L1
    while (kL1 > 16) {
        uint64_t aL1Bytes = (uint64_t)m_ * kL1 * sizeof(AType);
        uint64_t bL1Bytes = (uint64_t)kL1 * BASE_N * sizeof(BType);
        if (aL1Bytes <= L1_HALF_SIZE && bL1Bytes <= L1_HALF_SIZE) {
            break;
        }
        kL1 >>= 1;
    }

    uint64_t aL1OneBuffer = (uint64_t)m_ * kL1 * sizeof(AType);
    uint64_t bL1OneBuffer = (uint64_t)BASE_N * kL1 * sizeof(BType);

    // L1 ping-pong 偏移: [A0|B0] | [A1|B1]
    uint64_t l1BufferAOffset[L1_BUFFER_NUM];
    uint64_t l1BufferBOffset[L1_BUFFER_NUM];
    for (uint64_t bufId = 0; bufId < L1_BUFFER_NUM; ++bufId) {
        uint64_t group = bufId >> 1;
        uint64_t halfOff = (bufId & 1UL) * L1_HALF_SIZE;
        l1BufferAOffset[bufId] = halfOff + group * aL1OneBuffer;
        l1BufferBOffset[bufId] = halfOff + (L1_BUFFER_NUM >> 1) * aL1OneBuffer +
            group * bL1OneBuffer;
    }

    // ── L1 布局工厂 ──
    using MakeLayoutAL1 = AscendC::Te::NzLayoutFormat<AType>;
    using MakeLayoutBL1 = AscendC::Te::ZnLayoutFormat<BType>;

    uint64_t abL1LoopCnt = 0;

    // ── N-tiling 外循环 ──
    for (uint32_t nIdx = 0; nIdx < n_; nIdx += BASE_N) {
        uint32_t curN = (nIdx + BASE_N <= n_) ? BASE_N : (n_ - nIdx);

        // gmC 子视图: (m_, curN) 列偏移
        auto gmBlockC = gmC(
            AscendC::Te::MakeCoord(0, nIdx),
            AscendC::Te::MakeShape(m_, curN));

        // L0C 累加器
        auto layoutL0C = AscendC::Te::MakeL0CLayout(m_, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeL0CmemPtr<float>(0), layoutL0C);

        abL1LoopCnt = 0;

        // ── K-tiling 外循环 (L1 级别) ──
        uint32_t kL1Iter = TensorCeilDiv(k_, kL1);
        for (uint32_t iterK = 0; iterK < kL1Iter; ++iterK) {
            uint64_t l1BufId = abL1LoopCnt & L1_BUFFER_MASK;
            uint32_t kL1Offset = iterK * kL1;
            uint32_t curKL1 = (kL1Offset + kL1 <= k_) ? kL1 : (k_ - kL1Offset);

            // ── GM → L1 搬运 A, B ──
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

            // A: (m_, curKL1) 从 GM 读，NZ 格式写入 L1
            auto layoutAL1 = MakeLayoutAL1{}(m_, curKL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset[l1BufId]), layoutAL1);
            auto gmBlockA = gmA(
                AscendC::Te::MakeCoord(0, kL1Offset),
                AscendC::Te::MakeShape(m_, curKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);

            // B: (curKL1, curN) 从 GM 读，ZN 格式写入 L1
            auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr<BType>(l1BufferBOffset[l1BufId]), layoutBL1);
            auto gmBlockB = gmB(
                AscendC::Te::MakeCoord(kL1Offset, nIdx),
                AscendC::Te::MakeShape(curKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            // ── K-tiling 内循环 (baseK 级别, L1 → L0A/L0B → MMAD) ──
            uint32_t kL0Iter = TensorCeilDiv(curKL1, BASE_K);
            uint64_t l0PingPong = 0;
            for (uint32_t iterK0 = 0; iterK0 < kL0Iter; ++iterK0) {
                uint32_t kL0Offset = iterK0 * BASE_K;
                uint32_t curKL0 = (kL0Offset + BASE_K <= curKL1) ? BASE_K : (curKL1 - kL0Offset);
                uint64_t l0BufId = l0PingPong & 0x1;
                uint64_t l0Offset = TENSOR_HALF_L0_SIZE * l0BufId;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);

                auto CopyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});

                // L1 → L0A (NZ 格式)
                auto layoutAL0 = AscendC::Te::MakeNzLayout<AType>(m_, curKL0);
                auto tensorAL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL0AmemPtr<AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1 = tensorAL1(
                    AscendC::Te::MakeCoord(0, kL0Offset),
                    AscendC::Te::MakeShape(m_, curKL0));
                AscendC::Te::Copy(CopyL12L0, tensorAL0, tensorBlockAL1);

                // L1 → L0B (ZN 格式)
                auto layoutBL0 = AscendC::Te::MakeZnLayout<BType>(curKL0, curN);
                auto tensorBL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL0BmemPtr<BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 = tensorBL1(
                    AscendC::Te::MakeCoord(kL0Offset, 0),
                    AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(CopyL12L0, tensorBL0, tensorBlockBL1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

                // ── MMAD 计算 ──
                bool isFinal = (iterK + 1 == kL1Iter && iterK0 + 1 == kL0Iter);
                bool initL0c = (iterK == 0 && iterK0 == 0);
                uint8_t unitFlag = isFinal ? TENSOR_FINAL_ACCUMULATION : TENSOR_NON_FINAL_ACCUMULATION;

                AscendC::Te::MmadParams mmadParams{
                    static_cast<uint16_t>(m_),
                    static_cast<uint16_t>(curN),
                    static_cast<uint16_t>(TensorCeilAlign<uint32_t>(curKL0, TENSOR_BLOCK_CUBE)),
                    unitFlag,
                    initL0c};
                AscendC::Te::Mad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{},
                    tensorL0C, tensorAL0, tensorBL0, mmadParams);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong++;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt++;
        }

        // ── L0C → GM (fixpipe, fp32 → CType 量化) ──
        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(CopyL0C2GM, gmBlockC, tensorL0C,
                          AscendC::Te::FixpipeParams{TENSOR_FINAL_ACCUMULATION});
    }
}

} // namespace Mc2Kernel

#endif // TENSOR_MATMUL_STAGE_H
