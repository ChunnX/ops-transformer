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
 * \brief AIC 侧 FA-MX cube block，按 fia_block_cube_fullquant_mx.h 的大框架重写：
 *        IterateBmm1/IterateBmm2 接口、单 subLoop 调用、set_intra_block/wait_intra_block 跨核同步。
 *        底层搬运/计算复用 Blaze tensor API（AscendC::Te::Copy / Mmad / BufferSlot + Lock）。
 *        拉取版备份见 block_mmad_fa_mx_pulled_backup.h，原始 for 版本见 block_mmad_fa_mx_orig.h。
 *
 * 框架对齐 FIA：
 *   1. operator()/mm1Process/mm2Process  →  IterateBmm1/IterateBmm2（单 subLoop，kernel while 循环驱动）
 *   2. CrossCoreSetFlag/CrossCoreWaitFlag →  set_intra_block/wait_intra_block（+0 / +16 双 subblock）
 *   3. p1(P+PScale) 不再是本类成员，由 kernel 分配并通过 IterateBmm2 参数传入
 *   4. GM 指针在 Init 时存储，IterateBmm1/2 内部按 bIdx/n2Idx/gS1Idx/s2Idx 计算 offset 并建 tensor
 *   5. 形状参数由运行时成员改为 static constexpr（与 FIA/epilogue 对齐，标 //修改 处需按实际调整）
 *
 * 写法对齐 Blaze：
 *   - GM→L1：AscendC::Te::Copy(CopyGM2L1{}) + NDSliceLayoutPtn / ZNLayoutPtn / NZLayoutPtn
 *   - L1→L0：AscendC::Te::Copy(CopyL12L0A/L0B/ScaleA/ScaleB{}) + BufferSlot LockMte1
 *   - Compute：AscendC::Te::Mmad + BufferSlot LockM
 *   - L0C→UB：AscendC::Te::Copy(CopyL0C2UB{}, CopyL0C2UBTraitSplitM{}) + FixpipeParams
 *   - L0C M↔FIX 同步：显式 set_flag/wait_flag（FIA 风格，BufferSlot 无 FIX 锁）
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
        GM_ADDR actualSeqlenQAddr{nullptr};
        GM_ADDR actualSeqlenKVAddr{nullptr};
        GM_ADDR blockTableAddr{nullptr};
        GM_ADDR cAddr{nullptr};
    };

    // ===================== static constexpr 形状参数（与 FIA/epilogue 对齐，//修改 处需按实际调整）=====================
    static constexpr uint32_t mBaseSize = (uint32_t)128;
    static constexpr uint32_t s2BaseSize = (uint32_t)512;
    static constexpr uint32_t dBaseSize = (uint32_t)128;
    static constexpr uint32_t dVBaseSize = (uint32_t)128;
    static constexpr uint32_t s2SplitSize = (uint32_t)256;
    static constexpr uint32_t blockSize = (uint32_t)512;
    static constexpr uint32_t bSize = (uint32_t)1;              //修改
    static constexpr uint32_t n2Size = (uint32_t)8;
    static constexpr uint32_t gSize = (uint32_t)8;
    static constexpr uint32_t s1Size = (uint32_t)4;
    static constexpr uint32_t realGSize = (uint32_t)8;
    static constexpr uint32_t maxBlockNumPerBatch = (uint32_t)2;//修改
    static constexpr uint32_t actMSize = (uint32_t)32;
    static constexpr uint32_t actMSizeAlign32 = (uint32_t)32;
    static constexpr uint32_t FirstS2Loop = (uint32_t)0;
    static constexpr uint32_t LastS2Loop = (uint32_t)2;         //修改
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2U;

    static constexpr uint32_t Q_BUFFER_NUMS = 2;
    static constexpr uint32_t KV_BUFFER_NUMS = 4;

    __aicore__ inline BlockMmad()
    {
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_AIC {
            // FIA 风格：析构时等待最后几个未消费的跨核标志
            wait_intra_block(PIPE_FIX, 7);
            wait_intra_block(PIPE_FIX, 7 + 16);
            wait_intra_block(PIPE_FIX, 9);
            wait_intra_block(PIPE_FIX, 9 + 16);
            wait_intra_block(PIPE_FIX, 5);
            wait_intra_block(PIPE_FIX, 5 + 16);
        }
        AscendC::SetMMLayoutTransform(false);
    }

    // =================================================================================
    //  Init：存储 GM 指针（对应 FIA InitCubeInput）
    // =================================================================================
    __aicore__ inline void Init(const Params &params)
    {
        queryGm_ = reinterpret_cast<__gm__ uint8_t *>(params.queryAddr);
        qScaleGm_ = reinterpret_cast<__gm__ uint8_t *>(params.queryScaleAddr);
        kScaleGm_ = reinterpret_cast<__gm__ uint8_t *>(params.keyScaleAddr);
        vScaleGm_ = reinterpret_cast<__gm__ uint8_t *>(params.valueScaleAddr);
        blockTableGm_ = reinterpret_cast<__gm__ int32_t *>(params.blockTableAddr);
        actualSeqLengthsGmQ_ = reinterpret_cast<__gm__ uint64_t *>(params.actualSeqlenQAddr);

        // K/V 使用 ListTensorDesc（paged attention）
        ListTensorDesc keyListTensorDesc(reinterpret_cast<__gm__ void *>(params.keyAddr));
        keyGm_ = reinterpret_cast<__gm__ uint8_t *>(keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0));
        ListTensorDesc valueListTensorDesc(reinterpret_cast<__gm__ void *>(params.valueAddr));
        valueGm_ = reinterpret_cast<__gm__ uint8_t *>(valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0));
    }

    // =================================================================================
    //  InitBuffers：分配 L1/L0 缓冲（FIA 布局，Blaze BufferSlot）
    //  p1(P+PScale) 由 kernel 分配，占据 L1 [0, mm2LeftSize*3)；
    //  q1(Q+QScale) 紧随其后；kv1(K/V+Scale) 紧随 q1。
    // =================================================================================
    __aicore__ inline void InitBuffers()
    {
        // ---- L1 布局（与 FIA InitBuffers 一致）----
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize + mBaseSize * s2BaseSize / MXFP_GROUP_SIZE;
        constexpr uint32_t mm1QSize = mBaseSize * dBaseSize;
        constexpr uint32_t mm1QScaleSize = mm1QSize / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;
        constexpr uint32_t mm1KVSize = s2SplitSize * dBaseSize;       // 256*128
        constexpr uint32_t mm1KVScaleSize = mm1KVSize / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;

        uint8_t eventId = 0;
        // q1[2]：Q + QScale，紧跟 p1 区域之后
        for (uint32_t i = 0; i < Q_BUFFER_NUMS; ++i) {
            uint32_t base = mm2LeftSize * 3 + (mm1QSize + mm1QScaleSize) * i;
            q1_[i] = {base, eventId};
            eventId++;
        }
        // kv1[4]：K/V + K/VScale，紧跟 q1 之后
        for (uint32_t i = 0; i < KV_BUFFER_NUMS; ++i) {
            uint32_t base = mm2LeftSize * 3 + (mm1QSize + mm1QScaleSize) * Q_BUFFER_NUMS
                            + (mm1KVSize + mm1KVScaleSize) * i;
            kv1_[i] = {base, eventId};
            eventId++;
        }
        // l0ab[2]：L0A + L0B（同一偏移，不同物理内存）
        for (uint32_t i = 0; i < 2; ++i) {
            l0ab_[i] = {(AscendC::TOTAL_L0A_SIZE / 2) * i, eventId};
            eventId++;
        }
        // l0c[2]：L0C
        for (uint32_t i = 0; i < 2; ++i) {
            l0c_[i] = {(AscendC::TOTAL_L0C_SIZE / 2) * i, eventId};
            eventId++;
        }
    }

    // =================================================================================
    //  IterateBmm1：单 subLoop 的 Q·K 计算（对应 FIA IterateBmm1）
    //    参数与 FIA 完全一致：
    //      mm1Res     : UB 中 MM1 结果地址（由 kernel 分配）
    //      bmm1_id    : MM1 编号（用于跨核同步 id）
    //      bIdx/n2Idx : batch / n2 索引
    //      gS1Idx     : gS1 起始偏移
    //      s2Idx      : s2 起始偏移
    //      curS2Loop  : 当前 s2 循环编号
    //      subLoop    : s2Split 内 0/1（256 列一轮，共 2 轮）
    // =================================================================================
    __aicore__ inline void IterateBmm1(__ubuf__ void *mm1Res, int bmm1_id,
                                        uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx,
                                        uint32_t s2Idx, uint32_t curS2Loop, uint32_t subLoop)
    {
        // ---- Q: GM → L1（仅首个 s2Loop 的首个 subLoop 拷贝，后续 subLoop 复用）----
        if (curS2Loop == FirstS2Loop && subLoop == 0) {
            iq_++;
            int q1use = iq_ % 2;
            iquse_ = 4 + q1use;
            CopyQGm2L1(bIdx, n2Idx, gS1Idx, q1use);
            CopyQScaleGm2L1(bIdx, n2Idx, gS1Idx, q1use);
        }

        // ---- K: GM → L1 ----
        ikv_++;
        int ivkuse = ikv_ % 4;
        CopyKGm2L1(s2Idx, bIdx, n2Idx, subLoop, ivkuse);
        CopyKScaleGm2L1(s2Idx, bIdx, n2Idx, subLoop, ivkuse);

        // ---- L0C 槽分配：等待上一次 fixpipe 释放 ----
        l0c_id_++;
        int l0c_iduse = l0c_id_ % 2;
        wait_flag(PIPE_FIX, PIPE_M, l0c_iduse);  // FIA 风格：等 L0C 空闲

        // ---- Q: L1 → L0A, K: L1 → L0B ----
        l0a_id_++;
        int l0a_iduse = l0a_id_ % 2;
        CopyQL12L0a(iq_ % 2, l0a_iduse);
        CopyKL12L0b(ivkuse, l0a_iduse);

        // ---- Compute: Q · K → L0C ----
        ComputeQK(l0c_iduse, l0a_iduse);

        // ---- L0C → UB (fixpipe) ----
        set_flag(PIPE_M, PIPE_FIX, l0c_iduse);   // 通知 L0C 就绪
        wait_flag(PIPE_M, PIPE_FIX, l0c_iduse);  // 等待 L0C 计算完成

        // 跨核：等待 AIV 取走上一次 QK 结果（UB 槽可复用）
        int bmm1_idused = 6 + bmm1_id * 2;
        wait_intra_block(PIPE_FIX, 0 + bmm1_idused + 1);
        wait_intra_block(PIPE_FIX, 16 + bmm1_idused + 1);

        CopyQKOut2UB(mm1Res, l0c_iduse);

        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);   // L0C 空闲，供下次 compute

        // 跨核：通知 AIV QK 结果已就绪
        set_intra_block(PIPE_FIX, 0 + bmm1_idused);
        set_intra_block(PIPE_FIX, 16 + bmm1_idused);
    }

    // =================================================================================
    //  IterateBmm2：P·V 计算（对应 FIA IterateBmm2），内部 k<2 循环
    //    参数与 FIA 完全一致：
    //      ipuse   : P 缓冲槽编号（loop % 3）
    //      mm2Res  : UB 中 MM2 结果地址
    //      p1      : L1 中 P+PScale 缓冲（3 槽之一，由 kernel 分配）
    //      bmm2_id : MM2 编号（用于跨核同步 id）
    //      bIdx/n2Idx/s2Idx : 任务坐标
    // =================================================================================
    __aicore__ inline void IterateBmm2(int ipuse, __ubuf__ void *mm2Res,
                                        __cbuf__ void *p1, int bmm2_id,
                                        uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx)
    {
        // 跨核：等待 AIV softmax 完成（P 就绪）
        wait_intra_block(PIPE_MTE1, ipuse);
        wait_intra_block(PIPE_MTE1, ipuse + 16);

        // ---- L0C 槽分配 ----
        l0c_id_++;
        int l0c_iduse = l0c_id_ % 2;
        wait_flag(PIPE_FIX, PIPE_M, l0c_iduse);

        // ---- 内部 k<2 循环：每次 256 列 V，累加到 L0C ----
        for (uint32_t k = 0; k < 2; k++) {
            // V: GM → L1
            ikv_++;
            int ivkuse = ikv_ % 4;
            CopyVGm2L1(s2Idx, bIdx, n2Idx, k, ivkuse);
            CopyVScaleGm2L1(s2Idx, bIdx, n2Idx, k, ivkuse);

            // P: L1 → L0A, V: L1 → L0B
            l0a_id_++;
            int l0a_iduse = l0a_id_ % 2;
            CopyPL12L0a(p1, k, l0a_iduse);
            CopyVL12L0b(ivkuse, l0a_iduse);

            // Compute: P · V → L0C（k==0 初始化，k==1 累加）
            ComputePV(l0c_iduse, l0a_iduse, k);
        }

        // ---- L0C → UB (fixpipe) ----
        set_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        wait_flag(PIPE_M, PIPE_FIX, l0c_iduse);

        // 跨核：等待 AIV 取走上一次 PV 结果
        wait_intra_block(PIPE_FIX, bmm2_id + 1);
        wait_intra_block(PIPE_FIX, bmm2_id + 1 + 16);

        CopyPVOut2UB(mm2Res, l0c_iduse);

        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);

        // 跨核：通知 AIV PV 结果已就绪
        set_intra_block(PIPE_FIX, bmm2_id);
        set_intra_block(PIPE_FIX, bmm2_id + 16);
    }

private:
    // =================================================================================
    //  Offset 计算辅助（对应 FIA GetTBaseQ_MX / GetTBaseQsacle_MX）
    // =================================================================================
    __aicore__ inline uint64_t GetTBaseQ(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        return actualSeqLengthsGmQ_[bIdx - 1];
    }

    __aicore__ inline uint64_t GetTBaseQScale(uint32_t bIdx) const
    {
        return static_cast<uint64_t>(bIdx) * s1Size;
    }

    // =================================================================================
    //  GM → L1 搬运（Blaze tensor API + FIA offset 计算）
    // =================================================================================

    // ---- Q: GM → L1 ----
    // Q 在 GM 中布局 [S1, G, D]，stride [N2*G*D, D, 1]；
    // 一次拷贝 s1Size 个 S1 行，每行 gSize*D 连续。
    __aicore__ inline void CopyQGm2L1(uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx, int q1use)
    {
        uint64_t tIdx = GetTBaseQ(bIdx) + gS1Idx / gSize;
        uint64_t gmQOffset = tIdx * n2Size * gSize * dBaseSize + n2Idx * gSize * dBaseSize;

        auto slot = q1_[q1use];
        auto layoutL1 = MakeLayoutQL1{}(actMSize, dBaseSize);
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, QType>(slot.Addr()), layoutL1);

        // GM 3D [s1Size, gSize, dBaseSize]，stride [N2*G*D, D, 1]
        auto layoutGM = AscendC::Te::MakePatternLayout<
            AscendC::Te::NDSliceLayoutPtn, AscendC::Te::LayoutTraitDefault<>>(
            AscendC::Te::MakeShape(s1Size, AscendC::Te::MakeShape(gSize, dBaseSize)),
            AscendC::Te::MakeStride(n2Size * gSize * dBaseSize,
                                    AscendC::Te::MakeStride(dBaseSize, AscendC::Te::_1{})));
        auto tensorGM = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ QType *>(queryGm_ + gmQOffset)),
            layoutGM);

        auto copy = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(copy, tensorL1, tensorGM);
        }
    }

    // ---- QScale: GM → L1 ----
    __aicore__ inline void CopyQScaleGm2L1(uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx, int q1use)
    {
        constexpr uint32_t scaleDSize = dBaseSize / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;
        uint32_t s1IdxStart = gS1Idx / gSize;
        uint32_t gIdxStart = gS1Idx % gSize;
        uint64_t tIdx = GetTBaseQScale(bIdx) + s1IdxStart;
        uint64_t gmQScaleOffset = tIdx * scaleDSize * gSize
                                  + n2Idx * bSize * s1Size * scaleDSize * gSize
                                  + gIdxStart * scaleDSize;

        uint32_t qScaleOffset = actMSizeAlign32 * dBaseSize;  // QScale 在 q1 中的偏移
        auto slot = q1_[q1use];

        auto layoutL1 = MakeLayoutQScaleL1{}(actMSize, scaleDSize);
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slot.Addr() + qScaleOffset),
            layoutL1);

        auto layoutGM = MakeLayoutQScale{}(actMSize, scaleDSize);
        auto tensorGM = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ ScaleType *>(qScaleGm_ + gmQScaleOffset)),
            layoutGM);

        auto copy = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(copy, tensorL1, tensorGM);
        }
    }

    // ---- K: GM → L1 ----
    // K 在 GM 中按 blockTable 分页存储，ZN 布局 [D, s2SplitSize]。
    __aicore__ inline void CopyKGm2L1(uint32_t s2Idx, uint32_t bIdx, uint32_t n2Idx,
                                       uint32_t subLoop, int ivkuse)
    {
        uint32_t s2Offset = (subLoop % 2 == 0) ? s2Idx : s2Idx + s2SplitSize;
        uint64_t blockIdxInBatch = s2Offset / blockSize;
        uint64_t bsIdx = s2Offset % blockSize;
        uint64_t blockId = static_cast<uint64_t>(blockTableGm_[bIdx * maxBlockNumPerBatch + blockIdxInBatch]);
        uint64_t gmKOffset = blockId * n2Size * (dBaseSize / 32) * blockSize * 32
                             + n2Idx * (dBaseSize / 32) * blockSize * 32
                             + bsIdx * 32;

        auto slot = kv1_[ivkuse];
        auto layoutL1 = MakeLayoutKL1{}(dBaseSize, s2SplitSize);
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, KType>(slot.Addr()), layoutL1);

        auto layoutGM = MakeLayoutKL1{}(dBaseSize, s2SplitSize);
        auto tensorGM = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ KType *>(keyGm_ + gmKOffset)),
            layoutGM);

        // k 方向 64 对齐，按需填 0
        Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorL1, tensorGM);

        auto copy = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(copy, tensorL1, tensorGM);
        }
    }

    // ---- KScale: GM → L1 ----
    __aicore__ inline void CopyKScaleGm2L1(uint32_t s2Idx, uint32_t bIdx, uint32_t n2Idx,
                                            uint32_t subLoop, int ivkuse)
    {
        constexpr uint32_t scaleDSize = dBaseSize / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;
        uint32_t s2Offset = (subLoop % 2 == 0) ? s2Idx : s2Idx + s2SplitSize;
        uint64_t bsIdx = s2Offset % blockSize;
        uint64_t blockKScaleId = static_cast<uint64_t>(
            blockTableGm_[bIdx * maxBlockNumPerBatch + s2Offset / blockSize]);
        uint64_t gmKScaleOffset = blockKScaleId * n2Size * (blockSize / 32) * scaleDSize
                                  + n2Idx * (blockSize / 32) * scaleDSize
                                  + bsIdx / 16 * scaleDSize + bsIdx % 16;

        uint32_t kScaleOffset = s2SplitSize * dBaseSize;  // KScale 在 kv1 中的偏移
        auto slot = kv1_[ivkuse];

        auto scaleKSize = Blaze::Gemm::CeilDiv(s2SplitSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto layoutL1 = MakeLayoutKVScaleL1{}(scaleDSize, scaleKSize);
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slot.Addr() + kScaleOffset),
            layoutL1);

        auto layoutGM = MakeLayoutKVScaleL1{}(scaleDSize, scaleKSize);
        auto tensorGM = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ ScaleType *>(kScaleGm_ + gmKScaleOffset)),
            layoutGM);

        auto copy = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(copy, tensorL1, tensorGM);
        }
    }

    // ---- V: GM → L1 ----
    // V 在 GM 中按 blockTable 分页存储，NZ 布局 [s2SplitSize, D]。
    __aicore__ inline void CopyVGm2L1(uint32_t s2Idx, uint32_t bIdx, uint32_t n2Idx,
                                       uint32_t k, int ivkuse)
    {
        uint32_t s2Offset = s2Idx + k * s2SplitSize;
        uint64_t blockIdxInBatch = s2Offset / blockSize;
        uint64_t bsIdx = s2Offset % blockSize;
        uint64_t blockId = static_cast<uint64_t>(
            blockTableGm_[bIdx * maxBlockNumPerBatch + blockIdxInBatch]);
        uint64_t gmVOffset = blockId * n2Size * (dVBaseSize / 32) * blockSize * 32
                             + n2Idx * (dVBaseSize / 32) * blockSize * 32
                             + bsIdx * 32;

        auto slot = kv1_[ivkuse];
        auto kPadSize = Align64(s2SplitSize);
        auto layoutL1 = MakeLayoutVL1{}(kPadSize, dVBaseSize);
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, VType>(slot.Addr()), layoutL1);

        auto layoutGM = MakeLayoutVL1{}(kPadSize, dVBaseSize);
        auto tensorGM = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ VType *>(valueGm_ + gmVOffset)),
            layoutGM);

        Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorL1, tensorGM);

        auto copy = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(copy, tensorL1, tensorGM);
        }
    }

    // ---- VScale: GM → L1 ----
    __aicore__ inline void CopyVScaleGm2L1(uint32_t s2Idx, uint32_t bIdx, uint32_t n2Idx,
                                            uint32_t k, int ivkuse)
    {
        constexpr uint32_t scaleDSize = dVBaseSize / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;
        uint32_t s2VscaleScaleOffset = (s2Idx + k * s2SplitSize) / MXFP_DIVISOR_SIZE;
        uint64_t blockVscaleIdxInBatch = s2VscaleScaleOffset / (dVBaseSize * MXFP_MULTI_BASE_SIZE / 32);
        uint64_t bsVscaleIdx = s2VscaleScaleOffset % (dVBaseSize * MXFP_MULTI_BASE_SIZE / 32);
        uint64_t blockVscaleId = static_cast<uint64_t>(
            blockTableGm_[bIdx * maxBlockNumPerBatch + blockVscaleIdxInBatch]);
        uint64_t gmVScaleOffset = blockVscaleId * n2Size * dVBaseSize * MXFP_MULTI_BASE_SIZE / 32
                                  * blockSize / MXFP_DIVISOR_SIZE * 32
                                  + n2Idx * dVBaseSize * MXFP_MULTI_BASE_SIZE / 32
                                  * blockSize / MXFP_DIVISOR_SIZE * 32
                                  + bsVscaleIdx * 32;

        uint32_t vScaleOffset = s2SplitSize * dVBaseSize;  // VScale 在 kv1 中的偏移
        auto slot = kv1_[ivkuse];

        auto scaleVSize = Blaze::Gemm::CeilDiv(s2SplitSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto layoutL1 = MakeLayoutKVScaleL1{}(scaleDSize, scaleVSize);
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slot.Addr() + vScaleOffset),
            layoutL1);

        auto layoutGM = MakeLayoutKVScaleL1{}(scaleDSize, scaleVSize);
        auto tensorGM = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ ScaleType *>(vScaleGm_ + gmVScaleOffset)),
            layoutGM);

        auto copy = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        {
            auto lock = slot.LockMte2();
            AscendC::Te::Copy(copy, tensorL1, tensorGM);
        }
    }

    // =================================================================================
    //  L1 → L0 搬运（Blaze tensor API + BufferSlot LockMte1）
    // =================================================================================

    // ---- Q: L1 → L0A + QScale: L1 → L0ScaleA ----
    __aicore__ inline void CopyQL12L0a(int q1use, int l0a_iduse)
    {
        auto slotL0a = l0ab_[l0a_iduse];
        auto slotL1 = q1_[q1use];

        // Q: L1 → L0A
        auto layoutL0 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<QType>>>(
            actMSize, dBaseSize);
        auto tensorL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, QType>(slotL0a.Addr()), layoutL0);

        auto layoutL1 = MakeLayoutQL1{}(actMSize, dBaseSize);
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, QType>(slotL1.Addr()), layoutL1);

        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        {
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0a.LockMte1();
            AscendC::Te::Copy(copyL12L0A, tensorL0, tensorL1);
        }

        // QScale: L1 → L0ScaleA
        constexpr uint32_t scaleDSize = dBaseSize / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;
        uint32_t qScaleOffset = actMSizeAlign32 * dBaseSize;

        auto layoutScaleL0 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(actMSize, scaleDSize);
        auto tensorScaleL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, ScaleType>(slotL0a.Addr() >> 4),
            layoutScaleL0);

        auto layoutScaleL1 = MakeLayoutQScaleL1{}(actMSize, scaleDSize);
        auto tensorScaleL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slotL1.Addr() + qScaleOffset),
            layoutScaleL1);

        auto copyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});
        {
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0a.LockMte1();
            AscendC::Te::Copy(copyL12L0ScaleA, tensorScaleL0, tensorScaleL1);
        }
    }

    // ---- K: L1 → L0B + KScale: L1 → L0ScaleB ----
    __aicore__ inline void CopyKL12L0b(int ivkuse, int l0a_iduse)
    {
        auto slotL0b = l0ab_[l0a_iduse];
        auto slotL1 = kv1_[ivkuse];

        // K: L1 → L0B
        auto layoutL0 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<KType>>>(
            dBaseSize, s2SplitSize);
        auto tensorL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, KType>(slotL0b.Addr()), layoutL0);

        auto layoutL1 = MakeLayoutKL1{}(dBaseSize, s2SplitSize);
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, KType>(slotL1.Addr()), layoutL1);

        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        {
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0b.LockMte1();
            AscendC::Te::Copy(copyL12L0B, tensorL0, tensorL1);
        }

        // KScale: L1 → L0ScaleB
        constexpr uint32_t scaleDSize = dBaseSize / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;
        uint32_t kScaleOffset = s2SplitSize * dBaseSize;
        auto scaleKL0Len = (Align64(dBaseSize) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;

        auto layoutScaleL0 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(scaleKL0Len, s2SplitSize);
        auto tensorScaleL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, ScaleType>(slotL0b.Addr() >> 4),
            layoutScaleL0);

        auto layoutScaleL1 = MakeLayoutKVScaleL1{}(scaleKL0Len, s2SplitSize);
        auto tensorScaleL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slotL1.Addr() + kScaleOffset),
            layoutScaleL1);

        auto copyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});
        {
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0b.LockMte1();
            AscendC::Te::Copy(copyL12L0ScaleB, tensorScaleL0, tensorScaleL1);
        }
    }

    // ---- P: L1 → L0A + PScale: L1 → L0ScaleA ----
    // p1 由 kernel 传入（3 槽之一），P 在 p1，PScale 在 p1 + mBaseSize*s2BaseSize
    __aicore__ inline void CopyPL12L0a(__cbuf__ void *p1, uint32_t k, int l0a_iduse)
    {
        auto slotL0a = l0ab_[l0a_iduse];

        // P: L1 → L0A（第 k 个 256 列段）
        auto layoutL0 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<PType>>>(
            actMSize, Align64(s2SplitSize));
        auto tensorL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, PType>(slotL0a.Addr()), layoutL0);

        auto layoutL1 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<PType>>>(
            mBaseSize, Align64(s2BaseSize));
        auto pL1Ptr = reinterpret_cast<__cbuf__ PType *>(p1) + k * s2SplitSize * mBaseSize;
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, PType>(pL1Ptr), layoutL1);

        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        {
            auto lockL0 = slotL0a.LockMte1();
            AscendC::Te::Copy(copyL12L0A, tensorL0,
                tensorL1.Slice(AscendC::Te::MakeCoord(0, 0),
                               AscendC::Te::MakeShape(actMSize, Align64(s2SplitSize))));
        }

        // PScale: L1 → L0ScaleA
        uint32_t pScaleOffset = mBaseSize * s2BaseSize;  // PScale 在 p1 中的偏移
        uint64_t l1ScaleOffset = k * (Align64(s2SplitSize) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE
                                 * actMSizeAlign32;
        auto scaleKL0Len = (Align64(s2SplitSize) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;

        auto layoutScaleL0 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(actMSize, scaleKL0Len);
        auto tensorScaleL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, ScaleType>(slotL0a.Addr() >> 4),
            layoutScaleL0);

        auto pScaleL1Ptr = reinterpret_cast<__cbuf__ ScaleType *>(
            reinterpret_cast<__cbuf__ uint8_t *>(p1) + pScaleOffset + l1ScaleOffset);
        auto layoutScaleL1 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(actMSize, scaleKL0Len);
        auto tensorScaleL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(pScaleL1Ptr), layoutScaleL1);

        auto copyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});
        {
            auto lockL0 = slotL0a.LockMte1();
            AscendC::Te::Copy(copyL12L0ScaleA, tensorScaleL0, tensorScaleL1);
        }
    }

    // ---- V: L1 → L0B + VScale: L1 → L0ScaleB ----
    __aicore__ inline void CopyVL12L0b(int ivkuse, int l0a_iduse)
    {
        auto slotL0b = l0ab_[l0a_iduse];
        auto slotL1 = kv1_[ivkuse];

        // V: L1 → L0B（转置）
        auto kPadSize = Align64(s2SplitSize);
        auto layoutL0 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<VType>>>(
            kPadSize, dVBaseSize);
        auto tensorL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, VType>(slotL0b.Addr()), layoutL0);

        auto layoutL1 = MakeLayoutVL1{}(kPadSize, dVBaseSize);
        auto tensorL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, VType>(slotL1.Addr()), layoutL1);

        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        {
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0b.LockMte1();
            AscendC::Te::Copy(copyL12L0B, tensorL0, tensorL1);
        }

        // VScale: L1 → L0ScaleB
        constexpr uint32_t scaleDSize = dVBaseSize / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;
        uint32_t vScaleOffset = s2SplitSize * dVBaseSize;
        auto scaleVL0Len = (Align64(s2SplitSize) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;

        auto layoutScaleL0 = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(scaleVL0Len, dVBaseSize);
        auto tensorScaleL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, ScaleType>(slotL0b.Addr() >> 4),
            layoutScaleL0);

        auto layoutScaleL1 = MakeLayoutKVScaleL1{}(scaleDSize, scaleVL0Len);
        auto tensorScaleL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ScaleType>(slotL1.Addr() + vScaleOffset),
            layoutScaleL1);

        auto copyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});
        {
            auto lockL1 = slotL1.LockMte1();
            auto lockL0 = slotL0b.LockMte1();
            AscendC::Te::Copy(copyL12L0ScaleB, tensorScaleL0, tensorScaleL1);
        }
    }

    // =================================================================================
    //  Cube 计算（Blaze tensor API + BufferSlot LockM）
    // =================================================================================

    // ---- ComputeQK: Q · K → L0C ----
    __aicore__ inline void ComputeQK(int l0c_iduse, int l0a_iduse)
    {
        auto slotL0c = l0c_[l0c_iduse];
        auto layoutL0c = AscendC::Te::FrameLayoutFormat<
            AscendC::Te::NZLayoutPtn, AscendC::Te::_16>{}(actMSize, s2SplitSize);
        auto tensorL0c = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(slotL0c.Addr()), layoutL0c);

        AscendC::Te::MmadParams params;
        params.m = static_cast<uint16_t>(actMSize);
        params.k = static_cast<uint16_t>(dBaseSize);
        params.n = static_cast<uint16_t>(s2SplitSize);
        params.unitFlag = FINAL_ACCUMULATION;
        params.cmatrixInitVal = 1;
        {
            auto slotL0 = l0ab_[l0a_iduse];
            auto lockL0 = slotL0.LockM();
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<
                    AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0c,
                AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, QType>(slotL0.Addr()),
                    AscendC::Te::MakeFrameLayout<
                        AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<QType>>>(
                        actMSize, dBaseSize)),
                AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, KType>(slotL0.Addr()),
                    AscendC::Te::MakeFrameLayout<
                        AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<KType>>>(
                        dBaseSize, s2SplitSize)));
        }
    }

    // ---- ComputePV: P · V → L0C（k==0 初始化，k==1 累加）----
    __aicore__ inline void ComputePV(int l0c_iduse, int l0a_iduse, uint32_t k)
    {
        auto slotL0c = l0c_[l0c_iduse];
        auto layoutL0c = AscendC::Te::FrameLayoutFormat<
            AscendC::Te::NZLayoutPtn, AscendC::Te::_16>{}(actMSize, dVBaseSize);
        auto tensorL0c = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(slotL0c.Addr()), layoutL0c);

        AscendC::Te::MmadParams params;
        params.m = static_cast<uint16_t>(actMSize);
        params.k = static_cast<uint16_t>(s2SplitSize);
        params.n = static_cast<uint16_t>(dVBaseSize);
        params.unitFlag = FINAL_ACCUMULATION;
        params.cmatrixInitVal = (k == 0) ? 1 : 0;
        {
            auto slotL0 = l0ab_[l0a_iduse];
            auto lockL0 = slotL0.LockM();
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<
                    AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0c,
                AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, PType>(slotL0.Addr()),
                    AscendC::Te::MakeFrameLayout<
                        AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<PType>>>(
                        actMSize, Align64(s2SplitSize))),
                AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, VType>(slotL0.Addr()),
                    AscendC::Te::MakeFrameLayout<
                        AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<VType>>>(
                        Align64(s2SplitSize), dVBaseSize)));
        }
    }

    // =================================================================================
    //  L0C → UB 搬运（Blaze tensor API + FixpipeParams）
    // =================================================================================

    // ---- CopyQKOut2UB: L0C → mm1Res ----
    __aicore__ inline void CopyQKOut2UB(__ubuf__ void *mm1Res, int l0c_iduse)
    {
        auto shapeM = CeilAlign(actMSize, 32U);
        auto shapeN = CeilAlign(s2SplitSize * sizeof(float), 32UL);
        auto layoutUB = AscendC::Te::MakePatternLayout<
            AscendC::Te::NDLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(
            AscendC::Te::MakeShape(shapeM, shapeN),
            AscendC::Te::MakeStride(s2SplitSize, AscendC::Te::_1{}));
        auto tensorUB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(mm1Res), layoutUB);

        auto slotL0c = l0c_[l0c_iduse];
        auto layoutL0c = AscendC::Te::FrameLayoutFormat<
            AscendC::Te::NZLayoutPtn, AscendC::Te::_16>{}(actMSize, s2SplitSize);
        auto tensorL0c = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(slotL0c.Addr()), layoutL0c);

        auto copyL0C2UB = AscendC::Te::MakeCopy(
            AscendC::Te::CopyL0C2UB{}, Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
        AscendC::Te::Copy(copyL0C2UB.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)),
                          tensorUB, tensorL0c);
    }

    // ---- CopyPVOut2UB: L0C → mm2Res ----
    __aicore__ inline void CopyPVOut2UB(__ubuf__ void *mm2Res, int l0c_iduse)
    {
        auto shapeM = CeilAlign(actMSize, 32U);
        auto shapeN = CeilAlign(dVBaseSize * sizeof(float), 32UL) / sizeof(float);
        auto layoutUB = AscendC::Te::FrameLayoutFormat<
            AscendC::Te::NDLayoutPtn, AscendC::Te::_16>{}(shapeM, shapeN);
        auto tensorUB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(mm2Res), layoutUB);

        auto slotL0c = l0c_[l0c_iduse];
        auto layoutL0c = AscendC::Te::FrameLayoutFormat<
            AscendC::Te::NZLayoutPtn, AscendC::Te::_16>{}(actMSize, dVBaseSize);
        auto tensorL0c = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(slotL0c.Addr()), layoutL0c);

        auto copyL0C2UB = AscendC::Te::MakeCopy(
            AscendC::Te::CopyL0C2UB{}, Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
        AscendC::Te::Copy(copyL0C2UB.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)),
                          tensorUB, tensorL0c);
    }

    // =================================================================================
    //  成员变量
    // =================================================================================

    // ---- layout 类型别名（Blaze 写法）----
    using MakeLayoutQScale = AscendC::Te::FrameLayoutFormat<
        AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutQL1 = AscendC::Te::FrameLayoutFormat<
        AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<QType>>>;
    using MakeLayoutQScaleL1 = AscendC::Te::FrameLayoutFormat<
        AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutKL1 = AscendC::Te::FrameLayoutFormat<
        AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<KType>>>;
    using MakeLayoutKVScaleL1 = AscendC::Te::FrameLayoutFormat<
        AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutVL1 = AscendC::Te::FrameLayoutFormat<
        AscendC::Te::NZLayoutPtn, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<VType>>>;

    // ---- GM 指针（FIA 风格，Init 时存储）----
    __gm__ uint8_t *queryGm_{nullptr};
    __gm__ uint8_t *keyGm_{nullptr};
    __gm__ uint8_t *valueGm_{nullptr};
    __gm__ uint8_t *qScaleGm_{nullptr};
    __gm__ uint8_t *kScaleGm_{nullptr};
    __gm__ uint8_t *vScaleGm_{nullptr};
    __gm__ int32_t *blockTableGm_{nullptr};
    __gm__ uint64_t *actualSeqLengthsGmQ_{nullptr};

    // ---- L1/L0 缓冲槽（Blaze BufferSlot，FIA 布局）----
    BufferSlot q1_[Q_BUFFER_NUMS];     // Q + QScale in L1
    BufferSlot kv1_[KV_BUFFER_NUMS];   // K/V + K/VScale in L1
    BufferSlot l0ab_[2];               // L0A + L0B（共享偏移）
    BufferSlot l0c_[2];                // L0C

    // ---- 索引计数器（FIA 命名）----
    int iq_{-1};
    int iquse_{-1};
    int ikv_{-1};
    int l0a_id_{-1};
    int l0c_id_{-1};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
