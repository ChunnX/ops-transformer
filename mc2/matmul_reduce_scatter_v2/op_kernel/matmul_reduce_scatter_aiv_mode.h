/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_aiv_mode.h
 * \brief
 */

 #ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_H
 #define MATMUL_REDUCE_SCATTER_AIV_MODE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"
#include "../common/inc/kernel/moe_distribute_base.h"
#include "matmul_reduce_scatter_aiv_mode_util.h"
#include "../3rd/template_linear_algebra/include/template_linear_algebra/catlass.hpp"
#include "../3rd/template_linear_algebra/include/template_linear_algebra/arch/arch.hpp"
#include "../3rd/template_linear_algebra/include/template_linear_algebra/layout/layout.hpp"
#include "../3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad.hpp"
#include "../3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_swizzle.hpp"
#include "../3rd/template_linear_algebra/include/template_linear_algebra/gemm/dispatch_policy.hpp"
#include "../3rd/template_linear_algebra/include/template_linear_algebra/gemm/gemm_type.hpp"
#include "../3rd/template_linear_algebra/include/template_linear_algebra/gemm_coord.hpp"
#include "matmul.hpp"
#include "matmul_reduce_scatter_aiv_mode_padding.h"
#include "matmul_reduce_scatter_aiv_mode_dequant.h"

using namespace Catlass;
using namespace AscendC;
using namespace matmulReduceScatterV2_aivmode_tiling;
using namespace matmulReduceScatterV2_util;
using namespace dequant;
using namespace padding;
namespace MatmulReduceScatterV2Impl {

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// MMA2A : MatmulAllToAll
#define TemplateMMReduceScatterV2Class typename AType, typename BType, typename biasType, typename x2ScaleType, typename cType, bool weight_nz, bool TA, bool TB
#define TemplateMMReduceScatterV2Func AType, BType, biasType, x2ScaleType, cType, weight_nz, TA, TB

template <TemplateMMReduceScatterV2Class>
class MatmulReduceScatterAivMode : public CommBase{
    static constexpr bool quantFlag = (std::is_same<AType, int8_t>::value) && (std::is_same<BType, int8_t>::value);
public:
    __aicore__ inline MatmulReduceScatterAivMode() {};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR perTokenScale, GM_ADDR perChannelScale, GM_ADDR cGM,
                                GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AIVInit();
    __aicore__ inline void AICInit();
    __aicore__ inline void CatlassMatmul();
    __aicore__ inline void Padding();
    __aicore__ inline void Dequant(int32_t calIdx, uint64_t flagIdx);
    __aicore__ inline void StartBeforeFisrtStep(bool needAivDequant);
    __aicore__ inline void EndFirstStep(bool needAivDequant);
    __aicore__ inline void FirstStepInOut(int32_t data_size_remain, __gm__ cType *input,
    int32_t gm_offset, int32_t move_offset, int32_t loop_idx_st);
    __aicore__ inline void FirstStepInOutWithSplit(int32_t rank_total, int32_t rank_offset,
                                                     int32_t loop_idx_st, int32_t data_loop_idx);

private:
    bool aligned_a;
    bool aligned_b;
    int32_t core_loop;
    int32_t cal_count;
    int32_t m_per_rank;
    int32_t m_align;
    int64_t k_align;
    int32_t n_align;

    __gm__ cType* gm_peer_mem;
    GM_ADDR gm_a_align;
    GM_ADDR gm_b_align;
    __gm__ AType* gm_a_src;
    __gm__ BType* gm_b_src;
    __gm__ int32_t* gm_accum;
    bool needAivDequant;
    DequantRunner<cType> dequant_runner;
    Arch::Resource<Arch::AtlasA2> resource;
};

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR perTokenScale,
                                                                                  GM_ADDR perChannelScale, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulReduceScatterV2AivModeTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    perChannelScaleGM_ = perChannelScale;
    perTokenScaleGM_ = perTokenScale;
    CommBase::SetArgs(tilingData);
    hasBAlign = weight_nz ? false : hasBAlign;
    gm_a_align = reinterpret_cast<GM_ADDR>(hasAAlign ? workspaceGM  : 0);
    gm_b_align = reinterpret_cast<GM_ADDR>(hasBAlign ? workspaceGM + aAlignSize : 0);
    gm_accum = reinterpret_cast<__gm__ int32_t *>(quantFlag ? workspaceGM + aAlignSize + bAlignSize : 0);
    gm_a_src = reinterpret_cast<__gm__ AType *>(hasAAlign ? gm_a_align : aGM_);
    gm_b_src = reinterpret_cast<__gm__ BType *>(hasBAlign ? gm_b_align : bGM_);

    m_align = Block512B<AType>::AlignUp(m);
    k_align = Block512B<AType>::AlignUp(k);
    n_align = Block512B<AType>::AlignUp(n);
    aligned_a = hasAAlign;
    aligned_b = hasBAlign;
    //仅做perChannel量化，且x2Scale为INT64类型、输出为FP16时，通过fixPipe方式，不走AIV。
    needAivDequant = quantFlag && !(dequant_type == DequantType::PER_CHANNEL && isX2ScaleTypeInt64 && std::is_same<cType, float16_t>::value);
    MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::AICInit();
    MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::AIVInit();
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::AICInit()
{
    if ASCEND_IS_AIC {
        SetLoadDataPaddingValue(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        gm_peer_mem = reinterpret_cast<__gm__ cType*>(buff[rank]);
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::AIVInit()
{
    if ASCEND_IS_AIV {
        SetAtomicNone();
        SetMaskNormImpl();
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);

        max_ub_ping_pong_size = max_ub_single_dma_size / 2; //double buffer的方式
        max_ub_ping_pong_size = max_ub_ping_pong_size / n0 * n0;
        core_loop = m_loop * n_loop;
        cal_count = (core_loop + loop_num_per_comm - 1) / loop_num_per_comm;
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        bool need_fixpipe = quantFlag && isX2ScaleTypeInt64 && std::is_same<cType, half>::value;
        int32_t peer_mem_m = m0 * loop_num_per_comm * MAX_BLOCK_COUNT;
        uint32_t layout_b_row = (TB && !weight_nz) ? static_cast<uint32_t>(k_align) : static_cast<uint32_t>(k);
        uint32_t layout_b_col = (TB || weight_nz) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(n_align);
        using ArchTag = Arch::AtlasA2;
        constexpr bool ENABLE_UNIT_FLAG = false;
        constexpr bool ENABLE_SHUFFLE_K = false;
        using ElementA = AType;
        using ElementB = BType;
        using ElementC = typename std::conditional<quantFlag, int32_t, cType>::type;

        using LayoutA = typename std::conditional<TA, layout::ColumnMajor, layout::RowMajor>::type;
        using LayoutC = layout::RowMajor;
        using LayoutScale = layout::VectorLayout;

        LayoutA layoutA{TA ? static_cast<uint32_t>(m_align) : static_cast<uint32_t>(m),
                        TA ? static_cast<uint32_t>(k) : static_cast<uint32_t>(k_align)};
        LayoutC layoutC{static_cast<uint32_t>(m / rank_size), static_cast<uint32_t>(n)};
        LayoutC layoutPeerMem{static_cast<uint32_t>(peer_mem_m), static_cast<uint32_t>(n0)};
        LayoutScale layoutScale{static_cast<uint32_t>(n)};
        GemmCoord processSize{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

        using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;
        using AType_ = Gemm::GemmType<ElementA, LayoutA>;
        using CType_ = Gemm::GemmType<ElementC, LayoutC>;

        if (weight_nz) {
            // B矩阵NZ格式
            using LayoutNZ = typename std::conditional<TB, layout::nZ, layout::zN>::type;
            using BType_ = Gemm::GemmType<ElementB, LayoutNZ>;
            LayoutNZ layoutBNZ = LayoutNZ::template MakeLayout<ElementB>(layout_b_row, layout_b_col);

            struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, void> {
                using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, void>;
                using ElementA = typename Base::ElementA;
                using ElementB = typename Base::ElementB;
                using ElementAccumulator = typename Base::ElementAccumulator;
                using CopyGmToL1A = typename Base::CopyGmToL1A;
                using CopyGmToL1B = typename Base::CopyGmToL1B;

                using CopyL1ToL0A = typename Base::CopyL1ToL0A;
                using CopyL1ToL0B = typename Base::CopyL1ToL0B;

                using CopyL0CToGm = typename Base::CopyL0CToGm;
            };
            using TileCopy = TileCopyOpt;

            if (m0 == TILE_SHAPE_128) {
                using L1TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, TILE_SHAPE_256>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, TILE_SHAPE_64>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy,
                                 L1TileShape, L0TileShape, AType_, BType_, CType_, void, TileCopy>;
                using MatmulKernel = Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt>;
                typename MatmulKernel::Params params{processSize,
                                                 reinterpret_cast<GM_ADDR>(gm_a_src), layoutA,
                                                 reinterpret_cast<GM_ADDR>(gm_b_src), layoutBNZ,
                                                 reinterpret_cast<GM_ADDR>(cGM_), layoutC,
                                                 reinterpret_cast<GM_ADDR>(perChannelScaleGM_), layoutScale,
                                                 reinterpret_cast<GM_ADDR>(gm_peer_mem), layoutPeerMem,
                                                 reinterpret_cast<GM_ADDR>(gm_accum),
                                                 p_value, swizzl_count, swizzl_direct, dequant_type, rank,
                                                 rank_size, need_fixpipe};
                MatmulKernel matmul_op;
                matmul_op(params);                
            } else {
                using L1TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, TILE_SHAPE_256>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, TILE_SHAPE_64>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy,
                                    L1TileShape, L0TileShape, AType_, BType_, CType_, void, TileCopy>;
                using MatmulKernel = Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt>;
                typename MatmulKernel::Params params{processSize,
                                                 reinterpret_cast<GM_ADDR>(gm_a_src), layoutA,
                                                 reinterpret_cast<GM_ADDR>(gm_b_src), layoutBNZ,
                                                 reinterpret_cast<GM_ADDR>(cGM_), layoutC,
                                                 reinterpret_cast<GM_ADDR>(perChannelScaleGM_), layoutScale,
                                                 reinterpret_cast<GM_ADDR>(gm_peer_mem), layoutPeerMem,
                                                 reinterpret_cast<GM_ADDR>(gm_accum),
                                                 p_value, swizzl_count, swizzl_direct, dequant_type, rank,
                                                 rank_size, need_fixpipe};
                MatmulKernel matmul_op;
                matmul_op(params);    
            }
        } else {
            // B矩阵ND格式
            using LayoutB = typename std::conditional<TB, layout::ColumnMajor, layout::RowMajor>::type;
            LayoutB layoutB{layout_b_row, layout_b_col};
            using BType_ = Gemm::GemmType<ElementB, LayoutB>;

            struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, void> {
                using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, void>;
                using ElementA = typename Base::ElementA;
                using ElementB = typename Base::ElementB;
                using ElementAccumulator = typename Base::ElementAccumulator;

                // When matrix A is row-major, if the number of rows in matrix A is less than 16,
                // using the CopyGmToL1IntervalDataCopy method can improve the transfer efficiency.
                // The situation is similar for matrix B. If the above conditions are met,
                // please uncomment the following and comment out the original matrix A transfer method

                using CopyGmToL1A = typename Base::CopyGmToL1A;
                using CopyGmToL1B = typename Base::CopyGmToL1B;
                using CopyL1ToL0A = typename Base::CopyL1ToL0A;
                using CopyL1ToL0B = typename Base::CopyL1ToL0B;
                using CopyL0CToGm = typename Base::CopyL0CToGm;           
            };
            using TileCopy = TileCopyOpt;
            if (m0 == TILE_SHAPE_128) {
                using L1TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, TILE_SHAPE_256>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, TILE_SHAPE_64>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy,
                                        L1TileShape, L0TileShape, AType_, BType_, CType_, void, TileCopy>;
                using MatmulKernel = Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt>;
                typename MatmulKernel::Params params{processSize,
                                                 reinterpret_cast<GM_ADDR>(gm_a_src), layoutA,
                                                 reinterpret_cast<GM_ADDR>(gm_b_src), layoutB,
                                                 reinterpret_cast<GM_ADDR>(cGM_), layoutC,
                                                 reinterpret_cast<GM_ADDR>(perChannelScaleGM_), layoutScale,
                                                 reinterpret_cast<GM_ADDR>(gm_peer_mem), layoutPeerMem,
                                                 reinterpret_cast<GM_ADDR>(gm_accum),
                                                 p_value, swizzl_count, swizzl_direct, dequant_type, rank,
                                                 rank_size, need_fixpipe};
                MatmulKernel matmul_op;
                matmul_op(params);
            } else {
                using L1TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, TILE_SHAPE_256>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, TILE_SHAPE_64>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy,
                                    L1TileShape, L0TileShape, AType_, BType_, CType_, void, TileCopy>;
                using MatmulKernel = Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt>;
                typename MatmulKernel::Params params{processSize,
                                                 reinterpret_cast<GM_ADDR>(gm_a_src), layoutA,
                                                 reinterpret_cast<GM_ADDR>(gm_b_src), layoutB,
                                                 reinterpret_cast<GM_ADDR>(cGM_), layoutC,
                                                 reinterpret_cast<GM_ADDR>(perChannelScaleGM_), layoutScale,
                                                 reinterpret_cast<GM_ADDR>(gm_peer_mem), layoutPeerMem,
                                                 reinterpret_cast<GM_ADDR>(gm_accum),
                                                 p_value, swizzl_count, swizzl_direct, dequant_type, rank,
                                                 rank_size, need_fixpipe};
                MatmulKernel matmul_op;
                matmul_op(params);           
            }
        }
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::Dequant(int32_t calIdx, uint64_t flagIdx)
{
    bool needPerChannel = quantFlag && !(isX2ScaleTypeInt64 && std::is_same<cType, float16_t>::value); //仅在x2ScaleType为INT64,输出为FP16时走fixPipe，其余场景均需要AIV做反量化    
    bool needPerToken =  quantFlag && dequant_type == DequantType::PER_TOKEN; //可简化为只校验dequant_type；host侧校验，只有输入为quant场景，dequantType才能传有效值。
    if (!needPerChannel && !needPerToken) {
        return;
    }

    SetAndWaitAivSync(flagIdx);
    uint32_t pingpongSt = flagIdx * gm_c_pingpong_size;

    uint32_t rowNum = m;
    uint32_t colNum = n;
    uint32_t tileM0 = m0;
    uint32_t tileN0 = n0;
    uint32_t coreIdx = core_idx;
    uint32_t coreNum = core_num;
    uint32_t rankSize = rank_size;
    uint32_t pValue = p_value;
    uint32_t swizzlDirect = swizzl_direct;
    uint32_t swizzlCount = swizzl_count;
    __gm__ float32_t *perChannelScale = needPerChannel ?
        reinterpret_cast<__gm__ float32_t *>(perChannelScaleGM_) : nullptr;
    __gm__ float32_t *perTokenScale = needPerToken ?
        reinterpret_cast<__gm__ float32_t *>(perTokenScaleGM_) : nullptr;
    __gm__ int32_t *workspace = needPerChannel ?
        reinterpret_cast<__gm__ int32_t *>(gm_accum) + pingpongSt : nullptr;
    __gm__ cType *output = reinterpret_cast<__gm__ cType *>(buff[rank]) + pingpongSt;
    dequant_runner.RunMatmulReduceScatter(DEQUANT_ARGS_CALL());
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::Padding()
{
    if (!aligned_a && !aligned_b) {
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreFlag flagAivFinishPadding{AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID};
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        return;
    }
    bool transA = TA; //当前暂未支持A矩阵转置
    bool transB = TB;
    bool alignedA = aligned_a;
    bool alignedB = aligned_b;
    uint32_t matrixAM = m;
    uint32_t matrixAK = k;
    uint32_t matrixBK = k;
    uint32_t matrixBN = n;
    uint32_t matrixAMAlign = static_cast<uint32_t>(m_align);
    uint32_t matrixAKAlign = static_cast<uint32_t>(k_align);
    uint32_t matrixBKAlign = static_cast<uint32_t>(k_align);
    uint32_t matrixBNAlign = static_cast<uint32_t>(n_align);
    GM_ADDR gmA = reinterpret_cast<GM_ADDR>(aGM_);
    GM_ADDR gmB = reinterpret_cast<GM_ADDR>(bGM_);
    GM_ADDR gmAAlign = reinterpret_cast<GM_ADDR>(gm_a_align);
    GM_ADDR gmBAlign = reinterpret_cast<GM_ADDR>(gm_b_align);
    PaddingRunner<AType, BType> padding_runner;
    padding_runner.Run(PADDING_ARGS_CALL());
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::StartBeforeFisrtStep(bool needAivDequant)
{
    // 量化流程的原子加在从buff搬运数据时开启
    if (!needAivDequant) {
        SetAtomicAdd<cType>();
        PipeBarrier<PIPE_ALL>();
    }

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::EndFirstStep(bool needAivDequant)
{
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
    if (!needAivDequant) {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0); // Scalar等MTE3
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::FirstStepInOut(int32_t data_size_remain, __gm__ cType *input,
    int32_t gm_offset, int32_t move_offset, int32_t loop_idx_st)
{
    auto ub_offset = USED_UB_SIZE / 2 / sizeof(cType);
    LocalTensor<cType> ubTensor = uBuf_.AllocTensor<cType>();
    LocalTensor<cType> copyTensor0 = ubTensor;
    LocalTensor<cType> copyTensor1 = ubTensor[ub_offset];
    int32_t ping_pong_move_count = (data_size_remain + max_ub_ping_pong_size - 1) / max_ub_ping_pong_size;      // max_ub_ping_pong_size一定是N0的倍数，但不一定是M0*N0的倍数
    for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
        int32_t actual_move_size = max_ub_ping_pong_size;
        if (move_idx == ping_pong_move_count - 1) {
            actual_move_size = data_size_remain - move_idx * max_ub_ping_pong_size;
        }
        auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
        auto ub_buff_st = (move_idx & 1) ? copyTensor0 : copyTensor1;
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        // 读的matrix是多个小的m0*n0块顺序排布，写的时候需要重排
        CopyGmToUbuf(ub_buff_st, input + gm_offset + move_idx * max_ub_ping_pong_size, 1,
            actual_move_size * sizeof(cType) / 32, 0, 0);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        int32_t move_num_offset = move_offset + move_idx * max_ub_ping_pong_size;
        auto ub_buff = ub_buff_st;
        int64_t ub_buff_offset = 0;
        int32_t left_m = actual_move_size / n0;
        while (left_m > 0) {
            int32_t loop_idx = loop_idx_st + (move_num_offset / (m0 * n0)) * rank_size;
            int64_t batch_idx = loop_idx / (m_loop * n_loop);
            int32_t in_batch_idx = loop_idx % (m_loop * n_loop);
            int32_t in_rank_idx = in_batch_idx / rank_size;
            int64_t m_idx, n_idx;
            GetBlockIdx(in_rank_idx, m_loop / rank_size, n_loop, swizzl_direct, swizzl_count, m_idx, n_idx);
            int32_t actual_m = (m_idx == (m_loop / rank_size - 1)) ? (m / rank_size - m_idx * m0) : m0;
            int32_t actual_n = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
            int32_t m_offset = (move_num_offset % (m0 * n0)) / n0; // 当前一块起点对应的m，在当前块的位置
            int32_t actual_move_m = m0 < m_offset + left_m ? m0 - m_offset : left_m;
            // m0 - m_offset表示当前块剩下的一小段，跳过；
            if (m_offset < actual_m) {
                actual_move_m = actual_m < m_offset + left_m ? actual_m - m_offset : left_m;
                // left_m较大，则该块copy完，下次再copy下一块；
                // left_m较小，则只copy left_m的部分
                int64_t out_buff_offset = batch_idx * m * n / rank_size + (m_idx * m0 + m_offset) * n + n_idx * n0;
                CopyUbufToGmUnknown(nAlign16, reinterpret_cast<__gm__ cType*>(cGM_) + out_buff_offset, ub_buff[ub_buff_offset], actual_move_m,
                                    actual_n * sizeof(cType), (n0 - actual_n) * sizeof(cType) / 32, (n - actual_n) * sizeof(cType));
            }
            left_m -= actual_move_m;
            move_num_offset += actual_move_m * n0;
            ub_buff_offset += actual_move_m * n0;
        }
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::FirstStepInOutWithSplit(int32_t rank_total, int32_t rank_offset,
                                                     int32_t loop_idx_st, int32_t data_loop_idx)
{
    int32_t rank_per_core = rank_size / comm_npu_split;
    int32_t before_core_offset = data_loop_idx * comm_data_split * len_per_loop;
    int32_t core_rank_offset = (core_idx / comm_data_split) * rank_per_core;
    int32_t core_offset = core_idx % comm_data_split * len_per_loop;
    int32_t loop_total = rank_total - before_core_offset;

    int32_t rank_buff_offset = rank_offset + before_core_offset + core_offset;

    int32_t m_in_core = (core_offset >= loop_total) ? 0 :
        ((core_offset + len_per_loop) > loop_total ?
        loop_total - core_offset : len_per_loop);
    for (int32_t rank_idx = 0; rank_idx < rank_per_core; rank_idx++) {
        // 由于有些服务器gm地址初始为脏数据，reduceScatter perToken量化场景中matmul数据全部写到了peerMem
        // aiv写gm地址的时候均为atomic add，会导致在脏数据上进行累加，结果精度错误
        // 故perToken量化场景，此处第一次搬运不做累加，做覆盖搬运，从第二次开始做累加
        if (needAivDequant && rank_idx == 1) {
            SetAtomicAdd<cType>();
            PipeBarrier<PIPE_ALL>();
        }
        int32_t rank_idx_rot = (rank_idx + core_idx) % rank_per_core;
        int32_t real_rank_idx = core_rank_offset + rank_idx_rot;
        if (real_rank_idx == rank && !needAivDequant) {
            continue;
        }
        FirstStepInOut(m_in_core, reinterpret_cast<__gm__ cType*>(buff[real_rank_idx]), rank_buff_offset, before_core_offset + core_offset, loop_idx_st);
    }

    if (needAivDequant) {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0); // Scalar等MTE3
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();        
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func>::Process()
{
    CatlassMatmul();
    if ASCEND_IS_AIV {
        Padding();
        ResetIpcFlags(2);
        PipeBarrier<PIPE_ALL>();
        // 初始化通知aic共享内存是空闲的
        int32_t max_flag_id = cal_count < MAX_BLOCK_COUNT? cal_count: MAX_BLOCK_COUNT;
        for (int64_t cal_idx = 0; cal_idx < max_flag_id; ++cal_idx) {
            if (cal_idx * loop_num_per_comm + core_idx < core_loop) {
                SetAicSync(cal_idx);
            }
        }
        int32_t gm_c_block_size = gm_c_pingpong_size / rank_size;
        for (int32_t cal_idx = 0; cal_idx < cal_count; ++cal_idx) {
            uint64_t flag_idx = cal_idx % MAX_BLOCK_COUNT;
            int32_t actual_loop_num =
                (cal_idx == cal_count - 1) ? (core_loop - cal_idx * loop_num_per_comm) : loop_num_per_comm;
            m_per_rank = actual_loop_num * m0 / rank_size;
            if (core_idx < actual_loop_num) {
                WaitEvent(flag_idx);
            }
            Dequant(cal_idx, flag_idx);
            // aiv之间同步
            SetAndWaitAivSync(flag_idx);
            CrossRankSyncV1(FLAG_ZERO_IDX, cal_idx + 1);
            SetAndWaitAivSync(flag_idx);

            StartBeforeFisrtStep(needAivDequant);
            int32_t m_per_core = (m_per_rank * n0) / comm_data_split;
            int32_t data_split_num = DivCeil(m_per_core, len_per_loop);
            int32_t rank_offset = flag_idx * gm_c_pingpong_size + rank * gm_c_block_size;
            for (int32_t loop_idx = 0; loop_idx < data_split_num; loop_idx++) {
                if (aiv_idx == 0 && core_idx < comm_npu_split * comm_data_split) {
                    FirstStepInOutWithSplit(m_per_rank * n0, rank_offset, cal_idx * loop_num_per_comm, loop_idx);
                }
            }
            EndFirstStep(needAivDequant);

            SetAndWaitAivSync(flag_idx);
            CrossRankSyncV1(FLAG_ONE_IDX, cal_idx + 1);
            // aiv之间同步
            SetAndWaitAivSync(flag_idx);
            // 发送aic同步
            SetAicSync(flag_idx);
        }
        ResetIpcFlags(1);
        if (aiv_idx == 1 && core_idx < rank_size) {
            CheckBuffFlag((__gm__ int32_t *)buff[other_rank] + FLAG_OFFSET + FLAG_ZERO_IDX, 0);
        }
        PipeBarrier<PIPE_ALL>();
    }
}
} // MatmulReduceScatterV2Impl
#endif // MATMUL_REDUCE_SCATTER_V2_H