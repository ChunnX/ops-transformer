/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_aiv_mode.h
 * \brief MatmulReduceScatterV2算子的AIV模式核心实现（大M场景）
 *
 * 本文件实现了Matmul+ReduceScatter融合算子的主体逻辑，采用AIC+AIV混合核编程模型：
 *   - AIC核心：负责矩阵乘法（Matmul）计算，使用Catlass模板线性代数库
 *   - AIV核心：负责数据对齐padding、反量化（Dequant）、跨卡通信（ReduceScatter）
 *
 * 整体执行流水线（Pipeline）：
 *   1. Init阶段：初始化tiling参数、workspace地址映射、AIC/AIV各自初始化
 *   2. Padding阶段（AIV）：对不满足512B对齐的A/B矩阵进行padding补零
 *   3. CatlassMatmul阶段（AIC）：分块矩阵乘法，结果写入peermem或workspace
 *   4. Dequant阶段（AIV）：对int32累加结果做per-channel/per-token反量化
 *   5. ReduceScatter通信阶段（AIV）：从其他rank的peermem读取数据并原子累加到本地输出
 *
 * 关键设计：
 *   - 双缓冲（Ping-Pong）：AIC计算和AIV通信交替执行，隐藏通信延迟
 *   - Swizzle分块调度：改善L1 Cache命中率
 *   - CRTP模式：通过Derived模板参数支持SmallM子类的多态调用
 */

#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"
#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#endif
#include "matmul_reduce_scatter_aiv_mode_util.h"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/catlass.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/arch.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/layout.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/block_mmad.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/block_swizzle.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/dispatch_policy.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/gemm_type.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm_coord.hpp"
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

// MMA2A : MatmulAllToAll
// 模板参数宏定义：用于简化类模板声明的重复书写
// AType: A矩阵元素类型, BType: B矩阵元素类型, biasType: bias类型,
// x2ScaleType: 反量化scale类型, cType: 输出类型,
// weight_nz: B矩阵是否为NZ格式, TA: A矩阵是否转置, TB: B矩阵是否转置
#define TemplateMMReduceScatterV2Class                                                                                 \
    typename AType, typename BType, typename biasType, typename x2ScaleType, typename cType, bool hasBias,             \
    bool weight_nz, bool TA, bool TB
#define TemplateMMReduceScatterV2Func AType, BType, biasType, x2ScaleType, cType, hasBias, weight_nz, TA, TB

// MatmulReduceScatterAivMode: 大M场景的Matmul+ReduceScatter融合算子核心类
// 采用CRTP（Curiously Recurring Template Pattern）设计模式：
//   - Derived参数允许子类（如SmallM）覆写父类的虚函数行为，避免虚函数开销
//   - 继承自CommBase，复用通信基础设施（peermem管理、跨rank同步、flag机制等）
template <TemplateMMReduceScatterV2Class, typename Derived>
class MatmulReduceScatterAivMode : public CommBase {
public:
    __aicore__ inline MatmulReduceScatterAivMode(){};
    // 初始化函数：解析tiling数据、设置workspace地址映射、分别初始化AIC和AIV核心
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR perTokenScale,
                                GM_ADDR perChannelScale, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    // 主执行函数：根据Derived类型分发到具体子类的CatlassMatmul和AIVComm
    __aicore__ inline void Process();

protected:
    // 量化标志：当A和B均为int8时为true，表示需要反量化
    static constexpr bool quantFlag = (std::is_same<AType, int8_t>::value) && (std::is_same<BType, int8_t>::value);
    // 对A/B矩阵进行512B对齐padding（AIV执行），补零到满足DMA对齐要求
    __aicore__ inline void Padding();
    // 通信前准备：设置原子加模式、初始化双缓冲event flag
    __aicore__ inline void StartBeforeFirstStep();
    // 通信后收尾：等待DMA完成、清除原子加模式
    __aicore__ inline void EndFirstStep();

private:
    // AIV核心初始化：设置向量计算掩码、计算双缓冲参数和循环次数
    __aicore__ inline void AIVInit();
    // AIC核心初始化：设置数据搬运padding值、原子模式、NZ转ND标志
    __aicore__ inline void AICInit();
    // AIC矩阵乘法：使用Catlass库执行分块Matmul，结果写入peermem或workspace
    __aicore__ inline void CatlassMatmul();
    // AIV通信主循环：协调Padding→Dequant→ReduceScatter的完整流水线
    __aicore__ inline void AIVComm();
    // 反量化：对int32累加结果做per-channel和/或per-token反量化，输出到peermem
    __aicore__ inline void Dequant(int32_t calIdx, uint64_t flagIdx);
    // 单次数据搬入搬出：从远端peermem读取数据到UB，再从UB写入本地输出GM（含数据重排）
    __aicore__ inline void FirstStepInOut(int32_t actual_move_size, __gm__ cType *input, int32_t move_offset,
                                          int32_t loop_idx_st);
    // 带NPU切分的数据搬入搬出：将通信任务按comm_npu_split和comm_data_split分配给不同AIV核心
    __aicore__ inline void FirstStepInOutWithSplit(int32_t rank_data_offset, int32_t loop_idx_st,
                                                   int32_t data_loop_idx);

protected:
    int32_t m_align;       // M维度对齐到512B后的值
    int64_t k_align;       // K维度对齐到512B后的值
    int32_t n_align;       // N维度对齐到512B后的值
    int32_t core_loop;     // 总计算循环次数 = m_loop * n_loop
    int32_t cal_count;     // 通信轮次数 = ceil(core_loop / loop_num_per_comm)

    __gm__ cType *gm_peer_mem;   // 本rank的peermem指针，用于AIC写入和AIV通信
    __gm__ AType *gm_a_src;      // A矩阵源地址（可能指向对齐后的workspace）
    __gm__ BType *gm_b_src;      // B矩阵源地址（可能指向对齐后的workspace）
    __gm__ int32_t *gm_accum;    // int32累加结果workspace地址（仅量化场景使用）

private:
    bool aligned_a;              // A矩阵是否需要padding对齐
    bool aligned_b;              // B矩阵是否需要padding对齐
    int32_t len_per_rank;        // 每个rank在当前通信轮次中需要处理的数据量
    int32_t pingpong_idx;        // 双缓冲索引（0或1交替）
    int32_t tile_block_size;     // 单个分块大小 = m0 * n0

    GM_ADDR gm_a_align;
    GM_ADDR gm_b_align;
    DequantRunner<biasType, cType> dequant_runner;
    Arch::Resource<Arch::AtlasA2> resource;
};

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR perTokenScale, GM_ADDR perChannelScale, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // 注册tiling数据结构并获取tiling指针
    REGISTER_TILING_DEFAULT(MatmulReduceScatterV2AivModeTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    // 保存各输入输出的GM地址
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    perChannelScaleGM_ = perChannelScale;
    perTokenScaleGM_ = perTokenScale;
    // 从CommBase继承的SetArgs：解析rank/peermem/分块参数等
    CommBase::SetArgs(tilingData);
    // NZ格式的B矩阵不做padding对齐（NZ格式本身已按block对齐）
    hasBAlign = weight_nz ? false : hasBAlign;
    // workspace布局：[A对齐区 | B对齐区 | int32累加区（仅量化场景）]
    gm_a_align = reinterpret_cast<GM_ADDR>(hasAAlign ? workspaceGM : 0);
    gm_b_align = reinterpret_cast<GM_ADDR>(hasBAlign ? workspaceGM + aAlignSize : 0);
    gm_accum = reinterpret_cast<__gm__ int32_t *>(quantFlag ? workspaceGM + aAlignSize + bAlignSize : 0);
    // 根据是否需要对齐，选择实际读取的A/B矩阵源地址
    gm_a_src = reinterpret_cast<__gm__ AType *>(hasAAlign ? gm_a_align : aGM_);
    gm_b_src = reinterpret_cast<__gm__ BType *>(hasBAlign ? gm_b_align : bGM_);

    // SmallM子类需要重新计算m_loop（因为m_loop在小M场景下不按rank_size拆分）
    if constexpr (!std::is_same_v<Derived, void>) {
        m_loop = CeilDev(m, m0);
    }

    // 计算M/K/N维度的512B对齐值，用于Catlass Matmul的布局参数
    m_align = Block512B<AType>::AlignUp(m);
    k_align = Block512B<AType>::AlignUp(k);
    n_align = Block512B<AType>::AlignUp(n);
    aligned_a = hasAAlign;
    aligned_b = hasBAlign;
    // 分别初始化AIC和AIV核心
    MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AICInit();
    MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AIVInit();
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AICInit()
{
    if ASCEND_IS_AIC {
        // 设置L1数据搬运时padding值为0（不足一个block的尾部补零）
        SetLoadDataPaddingValue(0);
        // 清除原子操作模式
        SetAtomicNone();
        // 启用NZ到ND格式转换标志（fixpipe输出时NZ→ND）
        SetFixpipeNz2ndFlag(1, 0, 0);
        // 本rank的peermem地址，AIC将Matmul结果写入此区域供其他rank读取
        gm_peer_mem = reinterpret_cast<__gm__ cType *>(buff[rank]);
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AIVInit()
{
    if ASCEND_IS_AIV {
        // 清除原子操作模式
        SetAtomicNone();
        // 设置向量计算掩码为标准模式
        SetMaskNormImpl();
        // 设置全量掩码（所有位有效），用于向量运算
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);

        // 双缓冲：将UB空间一分为二，交替使用ping/pong两个区域
        max_ub_ping_pong_size = max_ub_single_dma_size / 2; // double buffer的方式
        // 确保ping/pong大小对齐到n0的整数倍
        max_ub_ping_pong_size = max_ub_ping_pong_size / n0 * n0;
        // 总计算循环次数 = M方向分块数 × N方向分块数
        core_loop = m_loop * n_loop;
        // 通信轮次数 = ceil(core_loop / loop_num_per_comm)
        cal_count = (core_loop + loop_num_per_comm - 1) / loop_num_per_comm;
        // 单个分块的元素数 = m0 × n0
        tile_block_size = m0 * n0;
        // 初始化pingpong索引为0
        pingpong_idx = 0;
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        // 判断是否需要fixpipe：量化场景 + x2Scale为INT64 + 输出为FP16时，AIC可直接做反量化输出
        bool need_fixpipe = quantFlag && isX2ScaleTypeInt64 && std::is_same<cType, half>::value;
        // peermem的M维度大小：m0 × loop_num_per_comm × MAX_BLOCK_COUNT（双缓冲）
        int32_t peer_mem_m = m0 * loop_num_per_comm * MAX_BLOCK_COUNT;
        // B矩阵布局的行列数：根据转置和NZ格式确定
        uint32_t layout_b_row = (TB && !weight_nz) ? static_cast<uint32_t>(k_align) : static_cast<uint32_t>(k);
        uint32_t layout_b_col = (TB || weight_nz) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(n_align);
        using ArchTag = Arch::AtlasA2;
        constexpr bool ENABLE_UNIT_FLAG = false;
        constexpr bool ENABLE_SHUFFLE_K = true;
        constexpr bool aicCalBias = !quantFlag && hasBias; // 如果计算量化后的矩阵乘，bias不由CatlassMatmul负责
        using ElementA = AType;
        using ElementB = BType;
        // 量化场景下C元素为int32（累加器），非量化场景直接为输出类型
        using ElementC = typename std::conditional<quantFlag, int32_t, cType>::type;
        using ElementBias = biasType;

        using LayoutA = typename std::conditional<TA, layout::ColumnMajor, layout::RowMajor>::type;
        using LayoutC = layout::RowMajor;
        using LayoutScale = layout::VectorLayout;
        using LayoutBias = layout::VectorLayout;

        // 构建A矩阵布局：根据转置方向设置行/列维度
        LayoutA layoutA{TA ? static_cast<uint32_t>(m_align) : static_cast<uint32_t>(m),
                        TA ? static_cast<uint32_t>(k) : static_cast<uint32_t>(k_align)};
        // C矩阵布局：每个rank只负责M/rank_size行
        LayoutC layoutC{static_cast<uint32_t>(m / rank_size), static_cast<uint32_t>(n)};
        // peermem布局：按双缓冲组织，M维度为peer_mem_m，N维度为n0
        LayoutPeerMem layoutPeerMem{static_cast<uint32_t>(peer_mem_m), static_cast<uint32_t>(n0)};
        // scale布局：一维向量，长度为N
        LayoutScale layoutScale{static_cast<uint32_t>(n)};
        LayoutBias layoutBias{static_cast<uint32_t>(n)};
        GemmCoord processSize{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

        // 量化场景使用更大的L1/L0分块（512/128），非量化场景使用较小的分块（256/64）
        constexpr int32_t L1TileShapeK = quantFlag ? TILE_SHAPE_512 : TILE_SHAPE_256;
        constexpr int32_t L0TileShapeK = quantFlag ? TILE_SHAPE_128 : TILE_SHAPE_64;
        using DispatchPolicy = std::conditional_t<aicCalBias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                                    Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;
        using AType_ = Gemm::GemmType<ElementA, LayoutA>;
        using CType_ = Gemm::GemmType<ElementC, LayoutC>;
        using BiasType_ = std::conditional_t<aicCalBias, Gemm::GemmType<ElementBias, LayoutBias>, void>;
        if (weight_nz) {
            // B矩阵NZ格式：根据转置方向选择nZ或zN布局
            using LayoutNZ = typename std::conditional<TB, layout::nZ, layout::zN>::type;
            using BType_ = Gemm::GemmType<ElementB, LayoutNZ>;
            // 构建NZ格式的B矩阵布局
            LayoutNZ layoutBNZ = LayoutNZ::template MakeLayout<ElementB>(layout_b_row, layout_b_col);

            struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
                using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
                using ElementA = typename Base::ElementA;
                using ElementB = typename Base::ElementB;
                using ElementAccumulator = typename Base::ElementAccumulator;
                using CopyGmToL1A = typename Base::CopyGmToL1A;
                using CopyGmToL1B = typename Base::CopyGmToL1B;

                using CopyL1ToL0A = typename Base::CopyL1ToL0A;
                using CopyL1ToL0B = typename Base::CopyL1ToL0B;

                using CopyL0CToGm = typename Base::CopyL0CToGm;
                using BiasTypeSelector = typename Base::BiasTypeSelector;
                using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
                using CopyL1ToBT = typename Base::CopyL1ToBT;
            };
            using TileCopy = TileCopyOpt;

            if (m0 == TILE_SHAPE_128) {
                // m0=128的L1/L0分块配置：128×256×K
                using L1TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy,
                                 L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
                using MatmulKernel =
                    Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt, void, void, aicCalBias>;
                typename MatmulKernel::Params params{processSize,   reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,       reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutBNZ,     biasGM_, reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,       reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                                     layoutScale,   reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem, reinterpret_cast<GM_ADDR>(gm_accum),
                                                     p_value,       swizzl_count,
                                                     swizzl_direct, dequant_type,
                                                     rank,          rank_size,
                                                     need_fixpipe};
                MatmulKernel matmul_op;
                matmul_op(params);
            } else {
                // m0=256的L1/L0分块配置：256×128×K
                using L1TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy,
                                    L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
                using MatmulKernel =
                    Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt, void, void, aicCalBias>;
                typename MatmulKernel::Params params{processSize,   reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,       reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutBNZ,     biasGM_, reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,       reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                                     layoutScale,   reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem, reinterpret_cast<GM_ADDR>(gm_accum),
                                                     p_value,       swizzl_count,
                                                     swizzl_direct, dequant_type,
                                                     rank,          rank_size,
                                                     need_fixpipe};
                MatmulKernel matmul_op;
                matmul_op(params);
            }
        } else {
            // B矩阵ND格式：根据转置方向选择行主序或列主序
            using LayoutB = typename std::conditional<TB, layout::ColumnMajor, layout::RowMajor>::type;
            LayoutB layoutB{layout_b_row, layout_b_col};
            using BType_ = Gemm::GemmType<ElementB, LayoutB>;

            struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
                using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
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
                using BiasTypeSelector = typename Base::BiasTypeSelector;
                using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
                using CopyL1ToBT = typename Base::CopyL1ToBT;
            };
            using TileCopy = TileCopyOpt;
            if (m0 == TILE_SHAPE_128) {
                // m0=128的L1/L0分块配置：128×256×K
                using L1TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy,
                                        L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
                using MatmulKernel =
                    Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt, void, void, aicCalBias>;
                typename MatmulKernel::Params colMajorParams{processSize,   reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,       reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutB,       reinterpret_cast<GM_ADDR>(biasGM_),
                                                     reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,       reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                                     layoutScale,   reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem, reinterpret_cast<GM_ADDR>(gm_accum),
                                                     p_value,       swizzl_count,
                                                     swizzl_direct, dequant_type,
                                                     rank,          rank_size,
                                                     need_fixpipe};
                MatmulKernel colMajorMatmulOp;
                colMajorMatmulOp(colMajorParams);
            } else {
                // m0=256的L1/L0分块配置：256×128×K
                using L1TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy,
                                    L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
                using MatmulKernel =
                    Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt, void, void, aicCalBias>;
                typename MatmulKernel::Params colMajorParams{processSize,   reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,       reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutB,       reinterpret_cast<GM_ADDR>(biasGM_),
                                                     reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,       reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                                     layoutScale,   reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem, reinterpret_cast<GM_ADDR>(gm_accum),
                                                     p_value,       swizzl_count,
                                                     swizzl_direct, dequant_type,
                                                     rank,          rank_size,
                                                     need_fixpipe};
                MatmulKernel colMajorMatmulOp;
                colMajorMatmulOp(colMajorParams);
            }
        }
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::Dequant(int32_t calIdx,
                                                                                                   uint64_t flagIdx)
{
    // 仅在x2ScaleType为INT64,输出为FP16时走fixPipe，其余场景均需要AIV做反量化
    bool needPerChannel = quantFlag && !(isX2ScaleTypeInt64 && std::is_same<cType, float16_t>::value);
    // 可简化为只校验dequant_type；host侧校验，只有输入为quant场景，dequantType才能传有效值。
    bool needPerToken =  quantFlag && dequant_type == DequantType::PER_TOKEN; 
    // 如果既不需要per-channel也不需要per-token反量化，直接返回
    if (!needPerChannel && !needPerToken) {
        return;
    }

    // 计算双缓冲的偏移量：当前通信轮次的pingpong起始位置
    uint32_t pingpongSt = flagIdx * gm_c_pingpong_size;

    // 准备反量化参数
    uint32_t rowNum = m;
    uint32_t colNum = n;
    uint32_t tileM0 = m0;
    uint32_t tileN0 = n0;
    uint32_t coreIdx = core_idx;
    uint32_t coreNum = core_num;
    uint32_t rankIdx = rank;
    uint32_t rankSize = rank_size;
    uint32_t pValue = p_value;
    uint32_t swizzlDirect = swizzl_direct;
    uint32_t swizzlCount = swizzl_count;
    // 根据反量化类型选择性地传入scale和workspace指针
    __gm__ float32_t *perChannelScale = needPerChannel ?
        reinterpret_cast<__gm__ float32_t *>(perChannelScaleGM_) : nullptr;
    __gm__ float32_t *perTokenScale = needPerToken ?
        reinterpret_cast<__gm__ float32_t *>(perTokenScaleGM_) : nullptr;
    __gm__ int32_t *workspace = needPerChannel ?
        reinterpret_cast<__gm__ int32_t *>(gm_accum) + pingpongSt : nullptr;
    // peermem和输出地址
    __gm__ cType *peerMem = reinterpret_cast<__gm__ cType *>(buff[rank]) + pingpongSt;
    __gm__ cType *output = reinterpret_cast<__gm__ cType *>(cGM_);
    __gm__ biasType *biasptr = reinterpret_cast<__gm__ biasType *>(biasGM_);

    dequant_runner.RunMatmulReduceScatter(DEQUANT_ARGS_CALL());
    // AIV内部同步：确保反量化完成后AIC才能继续写入该缓冲区
    SetAndWaitAivSync(flagIdx);
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::Padding()
{
    // 如果A和B都不需要padding对齐，直接设置AIC同步flag并返回
    if (!aligned_a && !aligned_b) {
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreFlag flagAivFinishPadding{AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID};
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        return;
    }
    // 准备padding参数：转置标志、对齐标志、矩阵维度和对齐维度
    bool transA = TA; // 当前暂未支持A矩阵转置
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

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::StartBeforeFirstStep()
{
    // 设置原子加模式：ReduceScatter通信时多个rank的数据累加到同一输出地址
    SetAtomicAdd<cType>();
    PipeBarrier<PIPE_ALL>();
    // 初始化双缓冲的event flag，确保第一次DMA前MTE2等待MTE3完成
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::EndFirstStep()
{
    // 等待所有DMA搬运完成
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
    // 等待Scalar侧确认所有原子加操作完成
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);     // Scalar等MTE3
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    // 清除原子加模式，恢复正常写入
    SetAtomicNone();
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::FirstStepInOut(
    int32_t actual_move_size, __gm__ cType *input, int32_t move_offset, int32_t loop_idx_st)
{
    // 将UB空间分为两个ping/pong区域，用于双缓冲DMA搬运
    auto ub_offset = USED_UB_SIZE / 2 / sizeof(cType);
    LocalTensor<cType> ubTensor = uBuf_.AllocTensor<cType>();
    LocalTensor<cType> copyTensor0 = ubTensor;
    LocalTensor<cType> copyTensor1 = ubTensor[ub_offset];

    // 根据pingpong索引选择当前使用的UB区域和event
    auto event_id = (pingpong_idx & 1) ? EVENT_ID0 : EVENT_ID1;
    auto ub_buff_st = (pingpong_idx & 1) ? copyTensor0 : copyTensor1;
    // 等待上一次DMA写入GM完成
    WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    // 从远端peermem读取数据到UB（连续读取，无需重排）
    CopyGmToUbuf(ub_buff_st, input, 1, actual_move_size * sizeof(cType) / 32, 0, 0);
    SetFlag<HardEvent::MTE2_MTE3>(event_id);
    // 等待UB数据就绪
    WaitFlag<HardEvent::MTE2_MTE3>(event_id);

    // 以下循环将UB中的数据重排后写入本地输出GM
    // peermem中的数据按m0×n0小块顺序排布，输出GM需要按行主序排列
    int32_t move_num_offset = move_offset;
    auto ub_buff = ub_buff_st;
    int64_t ub_buff_offset = 0;
    int32_t left_m = actual_move_size / n0;  // 剩余需要搬运的行数
    while (left_m > 0) {
        // 计算当前数据属于哪个m0×n0分块
        int32_t loop_idx = loop_idx_st + (move_num_offset / tile_block_size) * rank_size;
        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int32_t in_batch_idx = loop_idx % (m_loop * n_loop);
        int32_t in_rank_idx = in_batch_idx / rank_size;
        int64_t m_idx, n_idx;
        // 使用Swizzle映射获取分块的M/N索引
        GetSwizzledBlockIdx(in_rank_idx, m_loop / rank_size, n_loop, swizzl_direct, swizzl_count, m_idx, n_idx);
        // 计算当前分块的实际行列数（最后一个分块可能不满）
        int32_t actual_m = (m_idx == (m_loop / rank_size - 1)) ? (m / rank_size - m_idx * m0) : m0;
        int32_t actual_n = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
        // 当前一块起点对应的m偏移，在当前块内的位置
        int32_t m_offset = (move_num_offset % tile_block_size) / n0;
        int32_t actual_move_m = m0 < m_offset + left_m ? m0 - m_offset : left_m;
        // m0 - m_offset表示当前块剩下的一小段，跳过；
        if (m_offset < actual_m) {
            actual_move_m = actual_m < m_offset + left_m ? actual_m - m_offset : left_m;
            // left_m较大，则该块copy完，下次再copy下一块；
            // left_m较小，则只copy left_m的部分
            // 计算输出GM的偏移：batch × (m × n/rank_size) + 行偏移 × n + 列偏移
            int64_t out_buff_offset = batch_idx * m * n / rank_size + (m_idx * m0 + m_offset) * n + n_idx * n0;
            // 将UB数据写入GM，处理N方向不对齐16字节的情况
            CopyUbufToGmUnknown(nAlign16, reinterpret_cast<__gm__ cType *>(cGM_) + out_buff_offset,
                                ub_buff[ub_buff_offset], actual_move_m, actual_n * sizeof(cType),
                                (n0 - actual_n) * sizeof(cType) / 32, (n - actual_n) * sizeof(cType));
        }
        left_m -= actual_move_m;
        move_num_offset += actual_move_m * n0;
        ub_buff_offset += actual_move_m * n0;
    }
    // 设置MTE3完成flag，允许下一次DMA使用该UB区域
    SetFlag<HardEvent::MTE3_MTE2>(event_id);
    // 切换pingpong索引
    pingpong_idx = 1 - pingpong_idx;
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::FirstStepInOutWithSplit(
    int32_t rank_data_offset, int32_t loop_idx_st, int32_t data_loop_idx)
{
    // 计算每个AIV核心负责的rank数
    int32_t rank_per_core = rank_size / comm_npu_split;
    // data_loop_idx：data方向的第几个loop；计算当前data偏移
    int32_t data_offset = (data_loop_idx * comm_data_split + core_idx % comm_data_split) * max_ub_ping_pong_size;
    // 当前AIV核心负责的rank起始偏移
    int32_t core_rank_offset = (core_idx / comm_data_split) * rank_per_core;
    // 在peermem中的偏移 = rank数据偏移 + data偏移
    int32_t rank_buff_offset = rank_data_offset + data_offset;

    // 计算实际需要搬运的数据量（可能不足一个ping_pong_size）
    int32_t actual_move_num = (data_offset >= len_per_rank) ? 0 :
                ((data_offset + max_ub_ping_pong_size) > len_per_rank ?
                len_per_rank - data_offset : max_ub_ping_pong_size);
    // 遍历当前AIV核心负责的所有rank（跳过本rank，因为本rank的数据已在本地）
    for (int32_t rank_idx = 0; rank_idx < rank_per_core; rank_idx++) {
        int32_t real_rank_idx = core_rank_offset + (rank_idx + core_idx) % rank_per_core;
        if (real_rank_idx == rank) {
            continue;  // 跳过本rank
        }
        FirstStepInOut(actual_move_num,
                       reinterpret_cast<__gm__ cType*>(buff[real_rank_idx]) + rank_buff_offset,
                       data_offset, loop_idx_st);
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AIVComm()
{
    if ASCEND_IS_AIV {
        // 阶段1：Padding（AIV对A/B矩阵进行512B对齐补零）
        Padding();

        // 阶段2：重置IPC通信flag
        ResetIpcFlags(2);
        PipeBarrier<PIPE_ALL>();

        // 阶段3：主通信循环 - 逐轮执行 Dequant → CrossRankSync → ReduceScatter
        int32_t gm_c_block_size = gm_c_pingpong_size / rank_size;
        for (int32_t cal_idx = 0; cal_idx < cal_count; ++cal_idx) {
            uint64_t flag_idx = cal_idx % MAX_BLOCK_COUNT;
            int32_t loop_st = cal_idx * loop_num_per_comm;
            int32_t actual_loop_num = cal_idx == cal_count - 1 ? core_loop - loop_st : loop_num_per_comm;

            len_per_rank = actual_loop_num * tile_block_size / rank_size;
            // 等待AIC完成当前轮次的Matmul计算
            WaitEvent(flag_idx);

            // AIV内部同步：确保所有AIV核心都到达此处
            SetAndWaitAivSync(flag_idx);
            // 执行反量化（如果需要）
            Dequant(cal_idx, flag_idx);
            // 跨rank同步V1：通知其他rank本rank的数据已准备好
            CrossRankSyncV1(FLAG_ZERO_IDX, cal_idx + 1);
            // AIV内部同步
            SetAndWaitAivSync(flag_idx);

            // 执行ReduceScatter通信：从其他rank的peermem读取数据并原子累加到本地输出
            if (aiv_idx == 0 && core_idx < comm_npu_split * comm_data_split) {
                StartBeforeFirstStep();
                // 3层循环从外到内：data_loops*comm_data_split*max_ub_pingpong_size
                int32_t data_loops = DivCeil(len_per_rank, max_ub_ping_pong_size * comm_data_split);
                int32_t rank_offset = flag_idx * gm_c_pingpong_size + rank * gm_c_block_size;
                for (int32_t loop_idx = 0; loop_idx < data_loops; loop_idx++) {
                    FirstStepInOutWithSplit(rank_offset, loop_st, loop_idx);
                }
                EndFirstStep();
            }

            SetAndWaitAivSync(flag_idx);
            CrossRankSyncV2(FLAG_ONE_IDX, cal_idx + 1);
            SetAndWaitAivSync(flag_idx);
            if (cal_idx < cal_count - MAX_BLOCK_COUNT) { // 最后两次不进行同步
                // 发送aic同步
                SetAicSync(flag_idx);
            }
        }
        // 阶段4：通信收尾 - 重置flag、等待对端rank完成
        ResetIpcFlags(1);
        if (aiv_idx == 1 && core_idx < rank_size) {
            CheckBuffFlag((__gm__ int32_t *)buff[other_rank] + FLAG_OFFSET + FLAG_ZERO_IDX, 0);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::Process()
{
    // CRTP分发：如果Derived为void则直接调用本类方法，否则调用子类覆写的方法
    // 这允许SmallM子类覆写CatlassMatmul和AIVComm的行为
    if constexpr (std::is_same_v<Derived, void>) {
        CatlassMatmul();
        AIVComm();
    } else {
        static_cast<Derived *>(this)->CatlassMatmul();
        static_cast<Derived *>(this)->AIVComm();
    }
}
} // namespace MatmulReduceScatterV2Impl
#endif // MATMUL_REDUCE_SCATTER_V2_H