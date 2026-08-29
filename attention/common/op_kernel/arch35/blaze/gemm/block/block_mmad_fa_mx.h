/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_fa_mx.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/tile/pad_mx_kl1.h"
#include "blaze/gemm/tile/copy_gm_to_l1.h"

namespace Blaze {
namespace Gemm {
namespace Block {
template <
    class KernelSchedule_, class QType_, class LayoutQ_,
    class KType_, class LayoutK_, class VType_, class LayoutV_, class CType_, class LayoutC_>
class BlockMmad<
    FaBlockMmadMx<KernelSchedule_>, QType_, LayoutQ_, KType_, LayoutK_,
    VType_, LayoutV_, CType_, LayoutC_> {
public:
    using QType = QType_;
    using KType = KType_;
    using VType = VType_;
    using CType = CType_;
    using PType = VType;
    using LayoutQ = LayoutQ_;
    using LayoutK = LayoutK_;
    using LayoutV = LayoutV_;
    using LayoutC = LayoutC_;
    using DispatchPolicy = FaBlockMmadMx<KernelSchedule_>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ScaleType = AscendC::fp8_e8m0_t;

    struct Params {
        GM_ADDR queryAddr{nullptr};
        GM_ADDR keyAddr{nullptr};
        GM_ADDR valueAddr{nullptr};
        GM_ADDR queryScaleAddr{nullptr};
        GM_ADDR keyScaleAddr{nullptr};
        GM_ADDR valueScaleAddr{nullptr};
        GM_ADDR actualSeqlenQAddr{nullptr};  // 存放实际的不同batch的s1值
        GM_ADDR actualSeqlenKVAddr{nullptr}; // 存放实际的不同batch的s2值
        GM_ADDR blockTableAddr{nullptr};
        GM_ADDR cAddr{nullptr};
    };

    __aicore__ inline BlockMmad()
    {
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<Gemm::SYNC_MODE_4, PIPE_FIX>(7);
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(7 + SYNC_FLAG_ID_MAX);

            CrossCoreWaitFlag<Gemm::SYNC_MODE_4, PIPE_FIX>(9);
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(9 + SYNC_FLAG_ID_MAX);

            CrossCoreWaitFlag<Gemm::SYNC_MODE_4, PIPE_FIX>(5);
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(5 + SYNC_FLAG_ID_MAX);
        }        
        AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out
    }

    __aicore__ inline void Init()
    {
        InitBuffer();
    }

    // ================= 流水化拆分（while 软件流水使用） =================
    // 原来 MM1(Q·K) 和 MM2(P·V) 焊在 operator() 里同步执行，无法错开 2 拍。
    // 现在拆成两个独立函数：
    //   - mm1Process : 只做第一轮 matmul Q·K，结束后把 QK 结果写到 UB（mm1Res）供 AIV 的 softmax 用
    //   - mm2Process : 只做第二轮 matmul P·V，开头必须等 AIV softmax 完成（P 就绪）才开始
    //   - operator() : 兼容入口，等价 mm1Process + mm2Process 顺序调用（供 for 版本的 kernel 用）
    template <
        typename TensorQ, typename TensorK, typename TensorV, typename TensorScaleQ, typename TensorScaleK, typename TensorScaleV>
    __aicore__ inline void operator()(
        const TensorQ& tensorQ, const TensorK& tensorK, const TensorV& tensorV, const TensorScaleQ& tensorScaleQ,
        const TensorScaleK& tensorScaleK, const TensorScaleV& tensorScaleV, const BlockCoord& s1gCoord, uint32_t s1gSize, uint32_t s2Size)
    {
        mm1Process(tensorQ, tensorK, tensorScaleQ, tensorScaleK, s1gCoord, s1gSize, s2Size);
        mm2Process(tensorV, tensorScaleV, s1gSize, s2Size);
    }

    // ---- MM1 = Q · K^T ----
    // 输入：已经从 GM 切好并传进来的 Q/K 小块及其 scale。
    // 流程：把 Q、K 从 GM 拷到 L1 -> 再拷到 L0A/L0B -> 用 cube 算 Q·K 得 L0C ->
    //       把 L0C 结果拷回 UB（qkOutUbIdx_ 双缓冲交替），并 set 跨核标志通知 AIV 可以取走做 softmax。
    template <typename TensorQ, typename TensorK, typename TensorScaleQ, typename TensorScaleK>
    __aicore__ inline void mm1Process(const TensorQ& tensorQ, const TensorK& tensorK, const TensorScaleQ& tensorScaleQ,
        const TensorScaleK& tensorScaleK, const BlockCoord& s1gCoord, uint32_t s1gSize, uint32_t s2Size) 
    {
        // TODO s2循环时，Q不需要多次拷贝
        auto tensorQL1 = CopyQGm2L1(tensorQ, s1gCoord, s1gSize);          // Q: GM -> L1
        auto tensorScaleQL1 = CopyScaleQGm2L1(tensorScaleQ, s1gSize);     // QScale: GM -> L1
        constexpr uint32_t s2TileSize = 256;
        uint32_t nLoopNum = CeilDiv(s2Size, s2TileSize);
        // s2Size按256切分，分2轮循环计算（512 的 KV 块分两次算）
        for (uint32_t loop = 0; loop < nLoopNum; ++loop) {
            auto s2CurSize = (loop + 1 == nLoopNum) ? (s2Size - loop * s2TileSize) : s2TileSize;
            // k是64对齐，所以不需要填0
            auto tensorKL1 = CopyKGm2L1(tensorK, s2CurSize, loop * s2TileSize);        // K 256 一段: GM -> L1
            auto tensorScaleKL1 = CopyScaleKGm2L1(tensorScaleK, s2CurSize, loop * s2TileSize);
            auto tensorQL0a = CopyQL12L0a(tensorQL1, tensorScaleQL1, s1gSize);         // Q: L1 -> L0A
            auto tensorKL0b = CopyKL12L0b(tensorKL1, tensorScaleKL1, s2CurSize);       // K: L1 -> L0B
            auto tensorL0c = ComputeQK(tensorQL0a, tensorKL0b, s1gSize, s2CurSize);    // cube: Q·K -> L0C

            // 跨核同步：等 AIV 把上一轮(或上一次)的 QK 结果取走后，才能复用 qkOut 的 UB 槽
            auto syncId = 6 + (qkOutUbIdx_ & 0x1) * 2;
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(syncId + 1); // 7, 9
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(syncId + 1 + SYNC_FLAG_ID_MAX);

            CopyQKOut2UB(tensorL0c, s1gSize, s2CurSize);   // L0C -> UB（写到 qkOutUbIdx_ 对应的槽）

            // 通知 AIV：这个 QK 结果已经在 UB 上，可以取走做 softmax 了
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(syncId); // 6, 8
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(syncId + SYNC_FLAG_ID_MAX);
            kvBufferIdx_++;
            l0abBufferIdx_++;
            l0cBufferIdx_++;
            qkOutUbIdx_++;   // UB 双缓冲轮替
        }
        qBufferIdx_++;
    }

    // ---- MM2 = P · V ----
    // 输入：V 及 VScale 小块；P 是 AIV 做过 softmax 后写回 L1（p1）的结果。
    // 开头先 wait：必须等 AIV 的 softmax 完成（P 就绪）才能开始，因为 MM2 要用 P。
    template <typename TensorV, typename TensorScaleV>
    __aicore__ inline void mm2Process(const TensorV& tensorV, const TensorScaleV& tensorScaleV, uint32_t s1gSize, uint32_t s2Size) 
    {
        // 等待 AIV softmax 完成（P 结果就绪后，mm2 才能开始）
        auto mm2WaitFlag = mm2Index_ % 1;
        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(mm2WaitFlag); // 0, 1, 2
        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(mm2WaitFlag + SYNC_FLAG_ID_MAX);

        constexpr uint32_t s2TileSize = 256;
        auto kLoopNum = CeilDiv(s2Size, s2TileSize);
        auto slotL0c = l0c_[l0cBufferIdx_ & 0x1];
        auto layoutL0c = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::_16>{}(s1gSize, dSize);
        auto tensorL0c =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(slotL0c.Addr()), layoutL0c);
        for (uint32_t loop = 0; loop < kLoopNum; ++loop) {
            auto s2CurSize = (loop + 1 == kLoopNum) ? (s2Size - loop * s2TileSize) : s2TileSize;
            auto tensorVL1 = CopyVGm2L1(tensorV, s2CurSize, loop * s2TileSize);        // V 256 一段: GM -> L1
            auto tensorScaleVL1 = CopyScaleVGm2L1(tensorScaleV, s2CurSize, loop * s2TileSize);
            auto tensorPL0a = CopyPL12L0a(s1gSize, s2Size, s2CurSize, loop);            // P(softmax结果): L1 -> L0A
            auto tensorVL0b = CopyVL12L0b(tensorVL1, tensorScaleVL1, s2CurSize);        // V: L1 -> L0B
            ComputePV(tensorL0c, tensorPL0a, tensorVL0b, s1gSize, s2CurSize, loop, kLoopNum); // cube: P·V -> L0C
            kvBufferIdx_++;
            l0abBufferIdx_++;
        }
        l0cBufferIdx_++;
        // 等上一步的 PV 结果被取走后，再做这次的 L0C -> UB 搬出（跨核同步）
        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(5);
        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(5 + SYNC_FLAG_ID_MAX);
        CopyPVOut2UB(tensorL0c, s1gSize, dSize);    // L0C -> UB（mm2Res 供 AIV 的 vec2 使用）
        // 通知 AIV：MM2 结果 PV 已写到 UB，可以取走做最终输出
        CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(4);
        CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(4 + SYNC_FLAG_ID_MAX);
        mm2Index_++;    // 独立自推进，保证下次 MM2 用下一组 p1/L1 环形缓冲
    }

private:
    template <typename TensorL0C>
    __aicore__ inline void CopyPVOut2UB(const TensorL0C& tensorL0c, uint32_t s1gSize, uint32_t dSize)
    {
        auto ubSize = mBaseSize / 2 * s2BaseSize / 2 * sizeof(float);
        // TODO m要32对齐，不然PScale的ZZ的16无法给出，是否可以在L1的时候，s1gSize直接32对齐？
        // mSize可以配置的比L0C中的m大，当前API能力无法支持，unitflag可能无法使用？
        auto shapeM = CeilAlign(s1gSize, 32U);
        // 内轴32B对齐
        auto shapeN = CeilAlign(dSize * sizeof(float), 32UL) / sizeof(float);
        
        auto layoutUB = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDLayoutPtn, AscendC::Te::_16>{}(shapeM, shapeN);;        
        auto tensorUB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubSize * 2), layoutUB);
        
        auto CopyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{}, Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
        AscendC::Te::Copy(CopyL0C2UB.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), tensorUB, tensorL0c);
    }
    
    template <typename TensorL0C, typename TensorPL0, typename TensorVL0>
    __aicore__ inline void ComputePV(
        TensorL0C& tensorL0c, const TensorPL0& tensorPL0, const TensorVL0& tensorVL0, uint32_t s1gSize, uint32_t s2CurSize, uint32_t loop, uint32_t kLoopNum)
    {
        AscendC::Te::MmadParams params;
        params.m = static_cast<uint16_t>(s1gSize);
        params.k = static_cast<uint16_t>(s2CurSize);
        params.n = static_cast<uint16_t>(dSize);
        params.unitFlag = (loop + 1 == kLoopNum) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
        params.cmatrixInitVal = (loop == 0);
        {
            auto slotL0 = l0ab_[l0abBufferIdx_ & 0x1];
            auto lockL0 = slotL0.LockM();
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0c, tensorPL0, tensorVL0);
        }
    }
    
    template <typename TensorVL1, typename TensorScaleVL1>
    __aicore__ inline auto CopyVL12L0b(
        const TensorVL1& tensorVL1, const TensorScaleVL1& tensorScaleVL1, uint32_t s2CurSize)
    {
        auto slotL0b = l0ab_[l0abBufferIdx_ & 0x1];
        auto kPadSize = Align64(s2CurSize);
        auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<VType>>>(
            kPadSize, dSize);
        auto tensorBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, VType>(slotL0b.Addr()), layoutBL0);
        auto CopyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        {
            auto slotL1 = kvL1_[kvBufferIdx_ & 0x3];
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0b.LockMte1();
            AscendC::Te::Copy(CopyL12L0B, tensorBL0, tensorVL1);
        }

        uint64_t scaleKL0Len = (Align64(s2CurSize) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        auto layoutScaleBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            scaleKL0Len, dSize);
        auto tensorScaleBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, ScaleType>(slotL0b.Addr() >> 4), layoutScaleBL0);
        auto CopyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});

        {
            auto slotL1 = kvScaleL1_[kvBufferIdx_ & 0x3];
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0b.LockMte1();
            AscendC::Te::Copy(CopyL12L0ScaleB, tensorScaleBL0, tensorScaleVL1);
        }

        return tensorBL0;
    }
    
    __aicore__ inline auto CopyPL12L0a(uint32_t s1gSize, uint32_t s2Size, uint32_t s2CurSize, uint32_t loop)
    {
        constexpr uint32_t s2TileSize = 256;
        auto slotL0a = l0ab_[l0abBufferIdx_ & 0x1];
        auto slotL1 = pL1_[mm2Index_ % 3];
        auto kPadSize = Align64(s2Size);
        auto layoutAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<PType>>>(
            mBaseSize, Align64(s2Size)); // m方向按128分配空间
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, PType>(slotL1.Addr()), layoutAL1);
        
        auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<PType>>>(
            s1gSize, Align64(s2CurSize));
        auto tensorAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, PType>(slotL0a.Addr()), layoutAL0);
        auto CopyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});

        {
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0a.LockMte1();
            AscendC::Te::Copy(CopyL12L0A, tensorAL0, 
                tensorAL1.Slice(AscendC::Te::MakeCoord(0, loop * s2TileSize), AscendC::Te::MakeShape(s1gSize, Align64(s2CurSize))));
        }

        uint64_t scaleKL0Len = (Align64(s2CurSize) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        // (m1, k1/2, m0, 2) 连续存储，k1方向分了2个loop，每个loop是连续一块数据
        auto kOffset = loop * (Align64(s2TileSize) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE * CeilAlign(s1gSize, 16U);
        auto slotScaleL1 = pScaleL1_[mm2Index_ % 3];
        auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            s1gSize, scaleKL0Len);
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slotScaleL1.Addr() + kOffset), layoutScaleAL1);
        
        auto layoutScaleAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            s1gSize, scaleKL0Len);
        auto tensorScaleAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, ScaleType>(slotL0a.Addr() >> 4), layoutScaleAL0);
        auto CopyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});

        {
            auto lockL1 = slotScaleL1.LockMte1();
            auto lockL0 = slotL0a.LockMte1();
            AscendC::Te::Copy(CopyL12L0ScaleA, tensorScaleAL0, tensorScaleAL1);
        }
        
        return tensorAL0;
    }

    template <typename TensorScaleV>
    __aicore__ inline auto CopyScaleVGm2L1(const TensorScaleV& tensorScaleV, uint32_t s2Size, uint32_t offset)
    {
        // (s2Size, dSize)
        auto slot = kvScaleL1_[kvBufferIdx_ & 0x3];
        auto scaleKSize = Blaze::Gemm::CeilDiv(s2Size, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        auto layoutScaleL1 = MakeLayoutKVScaleL1{}(scaleKSize, dSize);
        auto tensorScaleL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slot.Addr()), layoutScaleL1);

        auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            auto scaleOffset = Blaze::Gemm::CeilDiv(offset, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            AscendC::Te::Copy(
                CopyScaleGM2L1, tensorScaleL1,
                tensorScaleV.Slice(AscendC::Te::MakeCoord(scaleOffset, 0), AscendC::Te::MakeShape(scaleKSize, dSize)));
        }
        return tensorScaleL1;
    }
    
    template <typename TensorV>
    __aicore__ inline auto CopyVGm2L1(const TensorV& tensorV, uint32_t s2Size, uint32_t offset)
    {
        // k方向必须64对齐，按需填0
        auto kPadSize = Align64(s2Size);
        auto slot = kvL1_[kvBufferIdx_ & 0x3];
        auto layoutVL1 = MakeLayoutVL1{}(kPadSize, dSize);
        auto tensorVL1 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, VType>(slot.Addr()), layoutVL1);
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto subTensorV = tensorV.Slice(AscendC::Te::MakeCoord(offset, 0), AscendC::Te::MakeShape(s2Size, dSize));
        Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorVL1, subTensorV);
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(copyGM2L1, tensorVL1, subTensorV);
        }
        return tensorVL1;
    }
    
    template <typename TensorL0C>
    __aicore__ inline void CopyQKOut2UB(const TensorL0C& tensorL0c, uint32_t s1gSize, uint32_t s2CurSize)
    {
        auto ubSize = mBaseSize / 2 * s2BaseSize / 2 * sizeof(float);
        // TODO m要32对齐，不然PScale的ZZ的16无法给出，是否可以在L1的时候，s1gSize直接32对齐？
        // mSize可以配置的比L0C中的m大，当前API能力无法支持，unitflag是否能使用？
        auto shapeM = CeilAlign(s1gSize, 32U);
        auto shapeN = CeilAlign(s2CurSize * sizeof(float), 32UL);
        auto layoutUB = AscendC::Te::MakePatternLayout<AscendC::Te::NDLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(
            AscendC::Te::MakeShape(shapeM, shapeN), AscendC::Te::MakeStride(s2BaseSize / 2, AscendC::Te::_1{}));        
        auto tensorUB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubSize * (qkOutUbIdx_ & 0x1)), layoutUB);
        
        auto CopyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{}, Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
        AscendC::Te::Copy(CopyL0C2UB.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), tensorUB, tensorL0c);      
    }
    
    template <typename TensorQL0, typename TensorKL0>
    __aicore__ inline auto ComputeQK(
        const TensorQL0& tensorQL0, const TensorKL0& tensorKL0, uint32_t s1gSize, uint32_t s2CurSize)
    {
        auto slotL0c = l0c_[l0cBufferIdx_ & 0x1];
        auto layoutL0c = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::_16>{}(s1gSize, s2CurSize);
        auto tensorL0c =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(slotL0c.Addr()), layoutL0c);
        AscendC::Te::MmadParams params;
        params.m = static_cast<uint16_t>(s1gSize);
        params.k = static_cast<uint16_t>(dSize);
        params.n = static_cast<uint16_t>(s2CurSize);
        params.unitFlag = FINAL_ACCUMULATION;
        params.cmatrixInitVal = 1;
        {
            auto slotL0 = l0ab_[l0abBufferIdx_ & 0x1];
            auto lockL0 = slotL0.LockM();
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0c, tensorQL0, tensorKL0);            
        }

        return tensorL0c;
    }
    
    template <typename TensorKL1, typename TensorScaleKL1>
    __aicore__ inline auto CopyKL12L0b(
        const TensorKL1& tensorKL1, const TensorScaleKL1& tensorScaleKL1, uint32_t s2CurSize)
    {
        auto slotL0b = l0ab_[l0abBufferIdx_ & 0x1];
        auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<KType>>>(
            dSize, s2CurSize);
        auto tensorBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, KType>(slotL0b.Addr()), layoutBL0);
        auto CopyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        {
            auto slotL1 = kvL1_[kvBufferIdx_ & 0x3];
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0b.LockMte1();
            AscendC::Te::Copy(CopyL12L0B, tensorBL0, tensorKL1);
        }

        uint64_t scaleKL0Len = (Align64(dSize) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        auto layoutScaleBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            scaleKL0Len, s2CurSize);
        auto tensorScaleBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, ScaleType>(slotL0b.Addr() >> 4), layoutScaleBL0);
        auto CopyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});
        {
            auto slotL1 = kvScaleL1_[kvBufferIdx_ & 0x3];
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0b.LockMte1();
            AscendC::Te::Copy(CopyL12L0ScaleB, tensorScaleBL0, tensorScaleKL1);
        }

        return tensorBL0;    
    }
    
    template <typename TensorQL1, typename TensorScaleQL1>
    __aicore__ inline auto CopyQL12L0a(const TensorQL1& tensorQL1, const TensorScaleQL1& tensorScaleQL1, uint32_t s1gSize)
    {
        auto slotL0a = l0ab_[l0abBufferIdx_ & 0x1];
        auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<QType>>>(
            s1gSize, dSize);
        auto tensorAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, QType>(slotL0a.Addr()), layoutAL0);
        auto CopyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        {
            auto slotL1 = qL1_[qBufferIdx_ & 0x1];
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0a.LockMte1();
            AscendC::Te::Copy(CopyL12L0A, tensorAL0, tensorQL1);
        }
        
        uint64_t scaleKL0Len = (Align64(dSize) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        auto layoutScaleAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            s1gSize, scaleKL0Len);
        auto tensorScaleAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, ScaleType>(slotL0a.Addr() >> 4), layoutScaleAL0);
        auto CopyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});
        {
            auto slotL1 = qScaleL1_[qBufferIdx_ & 0x1];
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0a.LockMte1();
            AscendC::Te::Copy(CopyL12L0ScaleA, tensorScaleAL0, tensorScaleQL1);
        }
        
        return tensorAL0;
    }
    
    template <typename TensorScaleK>
    __aicore__ inline auto CopyScaleKGm2L1(const TensorScaleK& tensorScaleK, uint32_t s2Size, uint32_t offset)
    {
        // (dSize, s2Size)
        auto slot = kvScaleL1_[kvBufferIdx_ & 0x3];
        const auto scaleDSize = Blaze::Gemm::CeilDiv(dSize, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        auto layoutScaleL1 = MakeLayoutKVScaleL1{}(scaleDSize, s2Size);
        auto tensorScaleL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slot.Addr()), layoutScaleL1);
        auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(
                CopyScaleGM2L1, tensorScaleL1,
                tensorScaleK.Slice(AscendC::Te::MakeCoord(0, offset), AscendC::Te::MakeShape(dSize, s2Size)));
        }

        return tensorScaleL1;
    }

    template <typename TensorK>
    __aicore__ inline auto CopyKGm2L1(const TensorK& tensorK, uint32_t s2Size, uint32_t offset)
    {
        // k方向是128，64对齐，所以不需要填0
        auto slot = kvL1_[kvBufferIdx_ & 0x3];
        auto layoutKL1 = MakeLayoutKL1{}(dSize, s2Size);
        auto tensorKL1 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, KType>(slot.Addr()), layoutKL1);
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(copyGM2L1, tensorKL1, 
                tensorK.Slice(AscendC::Te::MakeCoord(0, offset), AscendC::Te::MakeShape(dSize, s2Size)));
        }
        
        return tensorKL1;
    }

    __aicore__ inline void InitBuffer()
    {
        constexpr uint32_t s1BaseSize = 128;
        constexpr uint32_t s2BaseSize = 512;
        constexpr uint32_t dSize = 128;
        constexpr uint32_t qOneSize = s1BaseSize * dSize;
        constexpr uint32_t qScaleOneSize = qOneSize / MXFP_DIVISOR_SIZE * 2;
        constexpr uint32_t pOneSize = s1BaseSize * s2BaseSize;
        constexpr uint32_t pScaleOneSize = pOneSize / MXFP_DIVISOR_SIZE * 2;
        constexpr uint32_t kvOneSize = s2BaseSize / 2 * dSize;
        constexpr uint32_t kvScaleOneSize = kvOneSize / MXFP_DIVISOR_SIZE * 2;

        uint8_t eventId = 0;
        for (auto i = 0; i < P_BUFFER_NUMS; ++i) {
            auto base = pOneSize + pScaleOneSize;
            pL1_[i] = {base * i, eventId};
            pScaleL1_[i] = {base * i + pOneSize, eventId};
            eventId++;
        }
        for (auto i = 0; i < Q_BUFFER_NUMS; ++i) {
            auto base = (pOneSize + pScaleOneSize) * P_BUFFER_NUMS;
            qL1_[i] = {base + (qOneSize + qScaleOneSize) * i, eventId};
            qScaleL1_[i] = {base + (qOneSize + qScaleOneSize) * i + qOneSize, eventId};
            eventId++;
        }
        for (auto i = 0; i < KV_BUFFER_NUMS; ++i) {
            auto base = (pOneSize + pScaleOneSize) * P_BUFFER_NUMS + (qOneSize + qScaleOneSize) * Q_BUFFER_NUMS;
            kvL1_[i] = {base + (kvOneSize + kvScaleOneSize) * i, eventId};
            kvScaleL1_[i] = {base + kvOneSize + (kvOneSize + kvScaleOneSize) * i, eventId};
            eventId++;
        }
        for (auto i = 0; i < 2; ++i) {
            l0ab_[i] = {(AscendC::TOTAL_L0A_SIZE / 2) * i, eventId};
            eventId++;
            l0c_[i] = {(AscendC::TOTAL_L0C_SIZE / 2) * i, eventId};
            eventId++;
        }
    }
    
    template <typename TensorScaleQ>
    __aicore__ inline auto CopyScaleQGm2L1(const TensorScaleQ& tensorScaleQ, uint32_t s1gSize)
    {
        auto slot = qScaleL1_[qBufferIdx_ & 0x1];
        const auto scaleDSize = Blaze::Gemm::CeilDiv(dSize, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        auto layoutScaleL1 = MakeLayoutQScaleL1{}(s1gSize, scaleDSize);
        auto tensorScaleL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slot.Addr()), layoutScaleL1);

        // TODO 输入是3维shape的tensor，需要重新创建2维shape的tensor，复用已有搬运，也可增加自定义copy方法
        auto layoutGM = MakeLayoutQScale{}(s1gSize, scaleDSize);
        auto scaleGM =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(tensorScaleQ.Data().Get()), layoutGM);

        // K轴是64的倍数，所以不需要填充0
        auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleL1, scaleGM);
        }

        return tensorScaleL1;
    }
    
    template <typename TensorQ>
    __aicore__ inline auto CopyQGm2L1(const TensorQ& tensorQ, const BlockCoord& s1gCoord, uint32_t s1gSize)
    {
        // TODO L1的shape必须写成3维，当g不是16的倍数或者因子时，4维获取正确偏移地址有点复杂，可能要切分m0
        // Q矩阵：[S1, G, D], L1:((row0, row1), (col0, col1))
        auto slot = qL1_[qBufferIdx_ & 0x1];
        auto layoutQL1 = MakeLayoutQL1{}(s1gSize, dSize);
        auto tensorQL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, QType>(slot.Addr()), layoutQL1);

        // [s1Start, gStart] -> [s1Eng, gEnd] 头、中间、尾巴
        auto s1Start = AscendC::Te::Get<0>(s1gCoord);
        auto s1End = AscendC::Te::Get<1>(s1gCoord);
        auto gStart = AscendC::Te::Get<2>(s1gCoord);
        auto gEnd = AscendC::Te::Get<3>(s1gCoord);
        auto copyGM2L1 = AscendC::Te::MakeCopy(Gemm::Tile::CopySliceGM2L1{});

        auto lock = slot.LockMte2();
        // 头  coord = [s1Start, gStart] shape = [1, gSize - gStart]
        auto mHead = (gSize - gStart > mBaseSize) ? mBaseSize : (gSize - gStart);
        auto tensorQHead = tensorQ.Slice(
            AscendC::Te::MakeCoord(s1Start, AscendC::Te::MakeCoord(gStart, 0)),
            AscendC::Te::MakeShape(AscendC::Te::_1{}, AscendC::Te::MakeShape(mHead, dSize)));
        AscendC::Te::Copy(copyGM2L1, tensorQL1, tensorQHead);
        // 中间  coord = [s1Start + 1, 0] shape = [s1End - s1Start - 1, gSize]
        if (s1End - s1Start - 1 > 0) {
            auto tensorQMid = tensorQ.Slice(
                AscendC::Te::MakeCoord(s1Start + 1, AscendC::Te::MakeCoord(0, 0)),
                AscendC::Te::MakeShape(s1End - s1Start - 1, AscendC::Te::MakeShape(gSize, dSize)));
            auto qL1Mid = tensorQL1.Slice(AscendC::Te::MakeCoord(mHead, 0), AscendC::Te::MakeShape((s1End - s1Start - 1) * gSize, dSize));
            AscendC::Te::Copy(copyGM2L1, qL1Mid, tensorQMid);
        }
        // 尾巴  coord = [s1End, 0] shape = [1, gEnd]  [k1, m, k0]
        if (gEnd > 0 && s1End != s1Start) {
            auto tensorQEnd = tensorQ.Slice(
                AscendC::Te::MakeCoord(s1End, AscendC::Te::MakeCoord(0, 0)),
                AscendC::Te::MakeShape(1, AscendC::Te::MakeShape(gEnd, dSize)));
            auto totalOffset = (s1End - s1Start - 1) * gSize + mHead;
            auto qL1End = tensorQL1.Slice(AscendC::Te::MakeCoord(totalOffset, 0), AscendC::Te::MakeShape(gEnd, dSize));
            AscendC::Te::Copy(copyGM2L1, qL1End, tensorQEnd);            
        }

        return tensorQL1;
    }
    
private:
    using MakeLayoutQScale = AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutQL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<QType>>>;
    using MakeLayoutQScaleL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutKL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<KType>>>;
    using MakeLayoutKVScaleL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutVL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<VType>>>;
    // tiling获取
    uint32_t dSize = 128;
    uint32_t gSize = 8;
    uint32_t mBaseSize = 128;
    uint32_t s2BaseSize = 512;

    uint64_t qBufferIdx_{0};
    uint64_t kvBufferIdx_{0};
    uint64_t l0abBufferIdx_{0};
    uint64_t l0cBufferIdx_{0};
    uint64_t qkOutUbIdx_{0};
    uint64_t mm2Index_{0};

    static constexpr uint32_t P_BUFFER_NUMS = 3;
    static constexpr uint32_t Q_BUFFER_NUMS = 2;
    static constexpr uint32_t KV_BUFFER_NUMS = 4;
    BufferSlot pL1_[P_BUFFER_NUMS];
    BufferSlot pScaleL1_[P_BUFFER_NUMS];
    BufferSlot qL1_[Q_BUFFER_NUMS];
    BufferSlot qScaleL1_[Q_BUFFER_NUMS];
    BufferSlot kvL1_[KV_BUFFER_NUMS];
    BufferSlot kvScaleL1_[KV_BUFFER_NUMS];
    BufferSlot l0ab_[2];
    BufferSlot l0c_[2];
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
