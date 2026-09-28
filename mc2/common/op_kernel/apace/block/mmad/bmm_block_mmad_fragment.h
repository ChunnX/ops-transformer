/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bmm_block_mmad_fragment.h
 * \brief 非量化（FP16/BF16）Matmul 的 mmad fragment 通路（block/mmad 共享组件层，
 *        与量化版 qmm_mx_block_mmad_fragment.h 并排）
 *
 * 底本 apace/block/mmad/qmm_mx_block_mmad_fragment.h（去 scale 通路）：
 *   - 去 FragScaleA/scaleB 全通路（CopyScalesInL1、scale L1 分区、SCALE_BUFFER_FLAG 4/5、
 *     l0scalea/l0scaleb 加载、MmadTraitMX）；
 *   - L1 布局简化为 A|B|bias 多 buffer（偏移表按非 scale 分区重排）；
 *   - Mmad trait 换非量化默认 trait（对齐 Blaze block_mmad_matmul_basic.h Compute 的
 *     asc::te::mmad_trait_default；params.k 按 baseK 尾块直传，去 MXFP_DIVISOR 64 对齐）；
 *   - bias 通路：gmBias(BiasType=x1 dtype) → L1(make_mem_ptr<l1, BiasType>) →
 *     copy_l1_to_biastable（conv_control: half/bf16→float，硬件转换）→ mmad(bias table float)。
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
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class AType_, class LayoutA_, class BType_, class LayoutB_,
          class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BmmBlockMmadFragment;

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)

template <uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class AType_, class LayoutA_, class BType_, class LayoutB_,
          class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BmmBlockMmadFragment {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    // 非量化 policy（底本 MatmulWithScaleMx → MatmulMultiBlockBasic，官方 UT mat_mul_basic.h 同款）
    using DispatchPolicy =
        Blaze::Gemm::MatmulMultiBlockBasic<A_FULL_LOAD_MODE, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, 0>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    static constexpr bool weightNz = Blaze::Gemm::IsWeightNz<LayoutB>::value;
    static constexpr bool transA = Blaze::Gemm::IsTrans<LayoutA>::value;
    static constexpr bool transB = Blaze::Gemm::IsTrans<LayoutB>::value;

    constexpr static uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t BIAS_BUFFER_NUM = 2; // bias L1/BIAS table 双缓冲轮转

    // L1 布局 c0 跟随 dtype trait（底本 FP8 用 Int<C0_SIZE_B8>；非量化按 layout_trait_default，
    // 与 Blaze block_mmad_matmul_basic.h MakeLayoutAL1/BL1 一致）
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<BType>>>;

    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t splitKNum_{1};
    bool enableL0cPingPong_{false};

    struct TileL1L0Param {
        uint64_t curM = 0;
        uint64_t curN = 0;
        uint64_t curGmAKL1 = 0;
        uint64_t curGmBKL1 = 0;
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline BmmBlockMmadFragment()
    {
        // INPUT_BUFFER_FLAG_0..3 全量预置为 Blaze 4-buffer 惯例（底本同款）：本实例化 l1BufNum=2 运行期
        // 仅消费 FLAG_0/1（l1BufId = abL1LoopCnt_ & 1），FLAG_2/3 为无害冗余预置——析构侧全量 WaitFlag
        // 收尾闭合（配对完整，无 flag 泄漏），可安全保留。
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Blaze::Gemm::INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Blaze::Gemm::INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Blaze::Gemm::INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Blaze::Gemm::INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(Blaze::Gemm::INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(Blaze::Gemm::INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(Blaze::Gemm::M_MTE1_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(Blaze::Gemm::M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~BmmBlockMmadFragment()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Blaze::Gemm::INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Blaze::Gemm::INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Blaze::Gemm::INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Blaze::Gemm::INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(Blaze::Gemm::INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(Blaze::Gemm::INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(Blaze::Gemm::M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(Blaze::Gemm::M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out
    }

public:
    __aicore__ inline void Init(const ProblemShape &problemShape, const BlockShape &l0TileShape,
                                const L1Params &l1Params, bool isBias, bool dbL0C, uint64_t splitKNum = 1)
    {
        k_ = asc::te::get<Blaze::Gemm::IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        splitKNum_ = splitKNum;
        baseM_ = asc::te::get<Blaze::Gemm::IDX_M_IDX>(l0TileShape);
        baseN_ = asc::te::get<Blaze::Gemm::IDX_N_IDX>(l0TileShape);
        baseK_ = asc::te::get<Blaze::Gemm::IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        // L1 分区（去 scale）：2 buffer: A0|B0|bias0|...|A1|B1|bias1|...；
        // 4 buffer: A0A2|B0B2|bias0|...|A1A3|B1B3|bias1|...（底本非 scale 分区逻辑重排）
        aL1OneBuffer_ = baseM_ * Blaze::Gemm::CeilAlign(kL1_, asc::te::c0_element<AType>) * sizeof(AType);
        bL1OneBuffer_ = baseN_ * Blaze::Gemm::CeilAlign(kL1_, asc::te::c0_element<BType>) * sizeof(BType);
        biasL1OneBuffer_ = isBias_ ? baseN_ * sizeof(BiasType) : 0;
        // l1BufNum 约束：本类分区/轮转协议仅支持 2（本实例化 nBufferNum=2 由 tiling 侧固定，
        // N4 FillApaceMmTile 常量赋值）。l1BufNum=1 时 (l1BufNum_>>1)==0 使 A/B 槽 offset 重叠；
        // l1BufNum=4 时 bias 双缓冲槽（biasBufId_=abL1LoopCnt_&0x1）与 l1Buf flag 域
        // （l1BufId=abL1LoopCnt_&0x3）轮转不同步，存在理论 RAW——如需扩展须同步重排 bias 槽域
        // 与 HardEvent flag 配对（l1BufNum 为运行时参数，此处以约束注释限定，不改行为）。
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == 0) {
            for (uint64_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
                l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
                l1BufferBOffset_[bufferId] =
                    l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            // bias 双缓冲槽位紧跟 B 区之后（底本 scaleB 位置）
            for (uint64_t bufferId = 0; bufferId < BIAS_BUFFER_NUM; bufferId++) {
                l1BufferBiasOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
            }
        } else {
            // FULL_LOAD：A 整块驻留 L1 尾部，B|bias 滚动（底本分区逻辑去 scale 重排）
            uint64_t mAlign =
                Blaze::Gemm::CeilAlign(baseM_, transA ? Blaze::Gemm::C0_SIZE_B8 : Blaze::Gemm::BLOCK_CUBE);
            uint64_t kAlign = Blaze::Gemm::CeilAlign(k_, asc::te::c0_element<AType>);
            aL1OneBuffer_ = mAlign * kAlign * sizeof(AType);
            l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + biasL1OneBuffer_;
            uint64_t b1Offset = l1BufferAOffset_[0] + aL1OneBuffer_ >= (AscendC::TOTAL_L1_SIZE >> 1) ?
                                    l1BufferAOffset_[0] + aL1OneBuffer_ :
                                    (AscendC::TOTAL_L1_SIZE >> 1);
            for (uint64_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (uint64_t bufferId = 0; bufferId < BIAS_BUFFER_NUM; bufferId++) {
                l1BufferBiasOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
            }
        }
        kL1Iter_ = Blaze::Gemm::CeilDiv(k_, kL1_);
    }

    template <typename TensorA>
    __aicore__ inline auto CopyAInL1(TensorA const &gmA, TileL1L0Param &tileL1L0Param, uint64_t l1BufId, uint64_t iter0)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == 0) {
            auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curGmAKL1);
            auto tensorAL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
            auto blockA = gmA.Slice(asc::te::make_coord(0, iter0 * kL1_),
                                     asc::te::make_shape(tileL1L0Param.curM, tileL1L0Param.curGmAKL1));
            Apace::Basic::FragmentSliceCopy<false>(copyGM2L1, tensorAL1, blockA);
            return tensorAL1;
        } else {
            auto layoutAL1 =
                MakeLayoutAL1{}(tileL1L0Param.curM, Blaze::Gemm::CeilAlign(k_, asc::te::c0_element<AType>));
            auto tensorTotalAL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1BufferAOffset_[0]), layoutAL1);
            auto tensorAL1 = tensorTotalAL1.slice(asc::te::make_coord(0, iter0 * kL1_),
                                                  asc::te::make_shape(tileL1L0Param.curM, tileL1L0Param.curGmAKL1));
            if (abL1LoopCnt_ < kL1Iter_) {
                auto blockA = gmA.Slice(asc::te::make_coord(0, iter0 * kL1_),
                                         asc::te::make_shape(tileL1L0Param.curM, tileL1L0Param.curGmAKL1));
                Apace::Basic::FragmentSliceCopy<false>(copyGM2L1, tensorAL1, blockA);
            }
            return tensorAL1;
        }
    }

    template <typename TensorAL1, typename TensorBL1, typename TensorBiasL1, typename TensorL0C>
    __aicore__ inline void Iterate(TileL1L0Param &tileL1L0Param, uint64_t iter0, TensorAL1 &tensorAL1,
                                   TensorBL1 &tensorBL1, TensorBiasL1 &tensorBiasL1, TensorL0C &tensorL0C,
                                   uint64_t splitKIdx)
    {
        uint64_t kL0Iter = Blaze::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            uint64_t curKL0 = (iter1 * baseK_ + baseK_ > tileL1L0Param.curGmBKL1) ?
                                  (tileL1L0Param.curGmBKL1 - iter1 * baseK_) :
                                  baseK_;
            // Load data to L0 and open DB
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            uint16_t mte1WaitMFlag = (l0PingPong_ & 0x1) + Blaze::Gemm::M_MTE1_FLAG_0;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            // A L1->L0A
            auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
                tileL1L0Param.curM, curKL0);
            auto tensorAL0 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset), layoutAL0);
            auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, iter1 * baseK_),
                                                  asc::te::make_shape(tileL1L0Param.curM, curKL0));
            auto CopyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
            asc::te::copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

            // bias L1->BT（conv_control: half/bf16→float，硬件转换；仅首 K 块注入）
            auto layoutBt = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
                1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, Blaze::Gemm::BLOCK_CUBE));
            auto tensorBt = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::bias, float>(baseN_ * biasBufId_ * sizeof(float)), layoutBt);
            if (NeedBias(iter0, iter1, splitKIdx)) {
                auto CopyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
                asc::te::copy(CopyL12BT, tensorBt, tensorBiasL1);
            }

            // B L1->L0B
            auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(
                curKL0, tileL1L0Param.curN);
            auto tensorBL0 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset), layoutBL0);
            auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(iter1 * baseK_, 0),
                                                  asc::te::make_shape(curKL0, tileL1L0Param.curN));
            auto CopyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
            asc::te::copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            Mmad(tileL1L0Param, iter0, iter1, kL0Iter, curKL0, tensorL0C, tensorAL0, tensorBL0, tensorBt, splitKIdx);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC = int>
    __aicore__ inline void operator()(TensorA const &gmA, TensorB const &gmB, TensorBias const &gmBias,
                                      GM_ADDR *cFragAddrs, uint64_t mPerRank, uint64_t tileM, uint64_t tileCnt,
                                      uint64_t tailM, uint64_t remoteRankCnt, int64_t cRowStride,
                                      BlockShape const &singleShape, int64_t mPosGlobal, int64_t nPosGlobal,
                                      uint64_t splitKIdx = 0, TensorC const &gmC = 0)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = asc::te::get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = asc::te::get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C =
            asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<Blaze::Gemm::C0_SIZE_L0C>>{}(
                tileL1L0Param.curM, tileL1L0Param.curN);
        auto tensorL0C =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset), layoutL0C);
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            // bias 双缓冲槽位随 ab 循环轮转（Blaze basic btBufId 同款）
            biasBufId_ = abL1LoopCnt_ & 0x1;
            tileL1L0Param.curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
            tileL1L0Param.curGmAKL1 = tileL1L0Param.curGmBKL1;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

            // A GM->L1
            auto tensorAL1 = CopyAInL1(gmA, tileL1L0Param, l1BufId, iter0);

            // bias GM->L1
            auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            auto layoutBiasL1 = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>{}(
                1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, Blaze::Gemm::BLOCK_CUBE));
            auto tensorBiasL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(l1BufferBiasOffset_[biasBufId_]),
                layoutBiasL1);
            if (isBias_ && iter0 == 0 && splitKIdx == 0) {
                asc::te::copy(copyGM2L1, tensorBiasL1, gmBias);
            }

            // B GM->L1; 先slice再copy
            auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curGmBKL1, tileL1L0Param.curN);
            auto tensorBL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB.slice(asc::te::make_coord(iter0 * kL1_, 0),
                                      asc::te::make_shape(tileL1L0Param.curGmBKL1, tileL1L0Param.curN));
            asc::te::copy(copyGM2L1, tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            Iterate(tileL1L0Param, iter0, tensorAL1, tensorBL1, tensorBiasL1, tensorL0C, splitKIdx);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
        if (splitKIdx == splitKNum_ - 1) {
            ScatterL0C2GM(gmC, tensorL0C);
        }
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void ScatterL0C2GM(TensorC const &gmC, TensorL0C &tensorL0C)
    {
        auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        Apace::Basic::FragmentSliceCopy<true>(
            copyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), tensorL0C, gmC);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1, uint64_t kIter2)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0 && kIter2 == 0;
    }

    template <typename TensorL0CType, typename TensorAL0, typename TensorBL0, typename TensorBT>
    __aicore__ inline void Mmad(TileL1L0Param &tileL1L0Param, uint64_t iter0, uint64_t iter1, uint64_t kL0Iter,
                                uint64_t curKL0, TensorL0CType &tensorL0C, TensorAL0 const &tensorAL0,
                                TensorBL0 const &tensorBL0, TensorBT const &tensorBt, uint64_t splitKIdx)
    {
        // 非量化默认 trait（对齐 Blaze block_mmad_matmul_basic.h Compute 的 mmad_trait_default）
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        asc::te::mmad_params params;
        params.m = static_cast<uint16_t>(tileL1L0Param.curM);
        params.n = static_cast<uint16_t>(tileL1L0Param.curN);
        params.k = static_cast<uint16_t>(curKL0);
        params.unit_flag = (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter && (splitKIdx == splitKNum_ - 1)) ?
                               asc::te::unit_flag_mode::enable_update :
                               asc::te::unit_flag_mode::enable_keep;
        params.init_with_zero = (iter0 == 0 && iter1 == 0 && splitKIdx == 0 && !isBias_);
        if (NeedBias(iter0, iter1, splitKIdx)) {
            asc::te::mmad(mmadAtom.with(params), tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            asc::te::mmad(mmadAtom.with(params), tensorL0C, tensorAL0, tensorBL0);
        }
    }

    uint16_t biasBufId_ = 0;
    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL};    // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0UL};    // default 4 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL}; // default 2 buffer
};
#endif
} // namespace Block
} // namespace Gemm
} // namespace Blaze
