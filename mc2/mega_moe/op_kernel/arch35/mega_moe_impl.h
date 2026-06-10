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
 * \file mega_moe_impl.h
 * \brief
 */

#ifndef MEGA_MOE_IMPL_H
#define MEGA_MOE_IMPL_H

#include "kernel_operator.h"

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_base.h"

#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_swizzle.h"

#include "mega_moe_combine_send.h"

namespace MegaMoeImpl {
using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>;  // 3: SwizzleOffset
constexpr uint32_t ALIGN32 = 32U;
constexpr uint32_t L1_TILE_M_256 = 256;
constexpr uint32_t L1_TILE_M_128 = 128;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t L0_TILE_K = 128;
constexpr uint32_t SCALE_K_L1_RATE = 2;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN_256_256 = 256 * 256;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = (MAX_SINGLE_MN_256_256 + 31U) / ALIGN32 * ALIGN32;
constexpr uint32_t MAX_SINGLE_MN_128_256 = 128 * 256;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = (MAX_SINGLE_MN_128_256 + 31U) / ALIGN32 * ALIGN32;

// GroupMatmulSwigluQuant - GMM1 + SwiGLU + Quant
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& epilogueOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom)
{
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<N_VALUE>(problemShape);
    uint32_t k = Get<K_VALUE>(problemShape);
    uint32_t outputN = n / SWIGLU_N_HALF;

    epilogueOp.UpdateNextProblem({m, outputN, k, 0});

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx() / GetTaskRation();

    GlobalTensor<int32_t> groupFlagListGm2;
    groupFlagListGm2.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList2);

    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    constexpr bool transA = false;
    constexpr bool transB = true; // 后续要改为从模板参数获取转置属性

    constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    constexpr uint32_t c0SizeC = AuxGetC0Size<ElementC>();
    constexpr uint32_t c0SizeScale = 2U;

    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleK);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n);
    auto layoutBias = Te::MakeFrameLayout<LayoutBias>(1u, n); // block mmad需要传入bias
    auto layoutC = MakeLayoutC{}(L1_TILE_M_256, L1_TILE_N);

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias>;
    BlockMmad blockMmad;
    bool enableL0CPingPong = false;
    typename BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K,
        .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE,
        .l1BufNum = 2 // 2: double buffer
    };
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M_256, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmad::ProblemShape matmulShape{m, n, k, 0};
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, enableL0CPingPong); // 当前固定无bias

    int64_t ubOffset = 0;
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    ubOffset += MAX_SINGLE_MN_ALIGN32_NUM_256 * sizeof(ElementC);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layoutScaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias);

    BlockScheduler scheduler(
        {m, outputN, k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_256), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

    // wave-grain dispatch flag: 每 wave (L1_TILE_M_256 行) 一个槽,dispatch 完成的行数累加到该槽。
    // 目标值 = 该 wave 实际行数 = min(L1_TILE_M_256, m - mLoc) (尾 wave 可能 < L1_TILE_M_256)。
    // 仅在 wave 切换时等待;跨 nLoc 不同的同 mLoc tile 复用上次等待结果。
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        if constexpr (g_coreType == AscendC::AIC) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                uint32_t targetValue = (mLoc + L1_TILE_M_256 > m) ? (m - mLoc) : L1_TILE_M_256;
                __gm__ int32_t* flagValueAddr =
                    (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr() + waveIdx;
                while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {
                    }
                }
                lastWaveWaited = waveIdx;
            }

            if (vecSetSyncCom) {
                WaitForVector();
            }

            auto gmBlockA = gmA.Slice(
                Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
            auto gmBlockScaleA = gmScaleA.Slice(
                Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

            auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

            typename BlockMmad::BlockShape singleShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto gmBlockB = gmB.Slice(
                    Te::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                    Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto gmBlockScaleB = gmScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                    Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
                blockMmad(
                    gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                    weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
            }

            NotifyVector();
        }

        vecSetSyncCom = 1;

        if constexpr (g_coreType == AscendC::AIV) {
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                mLoc * outputN + nLoc,
                mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0, 0, 0, 0};
            WaitForCube();
            AscendC::SetCtrlSpr<60, 60>(0);
            epilogueOp(epilogueShape, epilogueOffset);
            NotifyCube();
        }
    }
    startBlockIdx = (startBlockIdx + tileNum) % blockNum;
}

// ============================================================================
// GroupMatmul2Combine - MoE第二轮分组矩阵乘法(GMM2) + Combine合并发送
// ============================================================================
// 整体流程:
//   1. Cube核(AIC): 执行MX量化矩阵乘法 A * B2 = C2，输出写入UB
//   2. Vector核(AIV): 从UB读取C2结果，根据triple信息(toRank, tokenIdx, topkIdx)
//      将每个token的输出直接Combine发送到目标卡的peermem窗口
//
// Cube与Vector通过pingpong双缓冲机制交替工作:
//   - pingpongIdx=0: Cube写First缓冲, Vector读Second缓冲
//   - pingpongIdx=1: Cube写Second缓冲, Vector读First缓冲
//   每轮循环结束后pingpongIdx翻转(1-pingpongIdx)
//
// 同步机制:
//   - Cube: NotifyVector -> Vector: WaitForCube (Cube通知Vector可读取)
//   - Vector: NotifyCube -> Cube: WaitForVector (Vector通知Cube可写入)
//   - vecSetSyncCom2: 记录Cube已完成的迭代数，>=2时Cube才需要WaitForVector
//     (第一轮Cube无需等待Vector，第二轮起需等待上一轮Vector完成读取)
//
// 参数说明:
//   - ElementA/ElementB: 矩阵A/B的数据类型(MX量化输入)
//   - ElementC: 矩阵乘输出数据类型
//   - ElementMxScaleA/ElementMxScaleB: A/B的MX量化scale类型
//   - problemShape: GMM2的矩阵形状, 注意GMM2中M=m, N=n, K=k
//     但problemShape传入的是{m, n, k_swiglu}格式，需做映射
//   - gmmAddrInfo: 包含A/B/scale的GM全局地址及groupFlag同步标志地址
//   - startBlockIdx: 多核循环起始偏移，用于将tile均匀分配给各核
//   - vecSetSyncCom2: Cube-Vector同步计数器，跟踪已完成的迭代轮数
//   - groupCnt: 当前expert分组索引，用于定位triple信息偏移
//   - pingpongIdx: pingpong缓冲索引(0或1)
// ============================================================================
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul2Combine(
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom2,
    uint32_t groupCnt, uint16_t& pingpongIdx)
{
    // GMM2单个tile块的元素总数: 128(M方向) * 256(N方向)
    constexpr uint32_t GMM2_BLOCK_SIZE = L1_TILE_M_128 * L1_TILE_N;

    // 从problemShape提取矩阵维度:
    // GMM2计算 A[m,k] * B2[k,n] = C2[m,n]
    // 注意: problemShape传入的是{m, n_swiglu, k_swiglu}格式
    //   - M维度: 直接取M_VALUE作为m(行数/token数)
    //   - N维度: 取K_VALUE作为n(列数/输出隐藏维度)
    //   - K维度: 取N_VALUE/SWIGLU_N_HALF作为k(内积维度)
    //     因为SwiGLU将n维度翻倍了(N_VALUE = k * SWIGLU_N_HALF = k*2)
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<K_VALUE>(problemShape);
    uint32_t k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;

    // 多核调度参数: blockNum为总核数, blockIdx为当前核逻辑编号
    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx() / GetTaskRation();

    // groupFlagList: 用于Cube核在首轮迭代时等待前置GMM1+SwiGLU阶段完成
    // 该flag记录了GMM1阶段的tile完成数，Cube需确认GMM1全部完成后才启动GMM2
    GlobalTensor<int32_t> groupFlagListGm;
    groupFlagListGm.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList);

    // MX量化scale的K维度长度:
    // 每MXFP_DIVISOR_SIZE(64)个元素共享1个e8m0 scale(2字节)
    // scaleK = CeilDiv(k, 64) * 2，即K方向上scale的总字节宽度
    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    // 矩阵转置属性:
    // transA=false: A矩阵为行主序(ND布局)
    // transB=true:  B矩阵为列主序(DN布局)，即B在内存中按列存储
    constexpr bool transA = false;
    constexpr bool transB = true;

    // C0尺寸: 对应硬件Cube核L0缓存中每种数据类型的最小存储单元大小
    // 用于确定Tensor Layout的帧格式(frame format)
    constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    constexpr uint32_t c0SizeC = AuxGetC0Size<ElementC>();
    constexpr uint32_t c0SizeScale = 2U; // MX scale为e8m0格式, 占2字节

    // 根据转置属性选择Layout模式:
    //   NDExtLayoutPtn: 非转置(行主序)布局
    //   DNExtLayoutPtn: 转置(列主序)布局
    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    // FrameLayoutFormat: 将Layout模式与C0尺寸组合，生成具体的内存排布格式
    // Scale的Layout根据对应矩阵的转置属性选择不同的scale排布模式:
    //   ScaleAND/ScaleADN: A矩阵scale，跟随A的转置属性
    //   ScaleBND/ScaleBDN: B矩阵scale，跟随B的转置属性
    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>;

    // 实例化Layout对象: 指定各矩阵/张量的全局维度大小
    // A: [m, k], B: [k, n], ScaleA: [m, scaleK], ScaleB: [scaleK, n]
    // Bias: [1, n] (实际不使用bias, 传空指针)
    // C: [L1_TILE_M_128, L1_TILE_N] (输出tile块的局部维度)
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleK);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n);
    auto layoutBias = Te::MakeFrameLayout<LayoutBias>(1U, n);
    auto layoutC = MakeLayoutC{}(L1_TILE_M_128, L1_TILE_N);

    // BlockMmad: MX量化矩阵乘法的块级计算引擎
    // DispatchPolicy: MatmulWithScaleMx 指定使用MX量化scale的分发策略
    // BiasType=float但实际不使用bias(传入空指针)
    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias>;
    BlockMmad blockMmad;

    // enableL0CPingPong=false: 不启用L0 C矩阵的pingpong双缓冲
    bool enableL0CPingPong = false;

    // L1缓冲参数:
    //   kL1=256: K方向L1分块大小
    //   scaleKL1=256*2=512: Scale K方向L1分块大小(scale比数据宽2倍)
    //   l1BufNum=2: L1使用双缓冲,提高数据搬运与计算的并行度
    typename BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K,
        .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE,
        .l1BufNum = 2
    };
    // L0 tile形状: [128, 256, 128], 即M=128, N=256, K=128
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M_128, L1_TILE_N, L0_TILE_K, 0};
    // 整体矩阵乘形状: [m, n, k]
    typename BlockMmad::TupleShape matmulShape{m, n, k};
    // 初始化BlockMmad引擎: 当前固定无bias(第4参数=false)
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, enableL0CPingPong);

    // ---- UB缓冲区分配 (pingpong双缓冲) ----
    // 在Vector核的UB上分配两块缓冲区，用于pingpong交替存储GMM2的矩阵乘输出
    // First缓冲: 存放pingpongIdx=0时的输出结果
    int64_t ubOffset = 0;
    LocalTensor<ElementC> l0cOutUbGMM2First = LocalTensor<ElementC>(TPosition::VECIN,
        ubOffset, L1_TILE_M_128 * L1_TILE_N);
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    // Second缓冲: 存放pingpongIdx=1时的输出结果
    // 偏移量 = First缓冲大小(MAX_SINGLE_MN_ALIGN32_NUM_128 * sizeof(ElementC))
    // MAX_SINGLE_MN_ALIGN32_NUM_128 = CeilAlign(128*256, 32), 32字节对齐后的元素数
    ubOffset += MAX_SINGLE_MN_ALIGN32_NUM_128 * sizeof(ElementC);
    LocalTensor<ElementC> l0cOutUbGMM2Second = LocalTensor<ElementC>(TPosition::VECIN,
        ubOffset, L1_TILE_M_128 * L1_TILE_N);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    // ---- GM全局张量构造 ----
    // 将GM上的原始地址封装为Te::Tensor对象，绑定对应的Layout
    // gmA/gmB: 矩阵A/B的全局张量
    // gmScaleA/gmScaleB: MX量化scale的全局张量
    // gmBias: bias张量(地址为0, 实际不使用)
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layoutScaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias);

    // ---- BlockScheduler: 块级迭代调度器 ----
    // 将大矩阵[m,n,k]按tile[L1_TILE_M_128, L1_TILE_N]分块
    // Swizzle偏移=3, 交换维度=1, 优化L1缓存局部性
    BlockScheduler scheduler(
        {m, n, k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_128), static_cast<int64_t>(L1_TILE_N))});

    // 多核任务分配:
    // 各核从startBlockIdx开始, 按blockIdx偏移轮流处理tile
    // startLoopIdx = 当前核在本轮循环中的起始tile索引
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;
    uint32_t tileNum = scheduler.GetTileNum(); // 总tile数

    // ---- 主循环: 逐tile执行GMM2 + Combine ----
    // 每个核每隔blockNum个tile取一个, 实现多核均匀分配
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += blockNum) {
        // 获取当前tile的坐标(mLoc, nLoc, kLoc)和实际形状(最后一个tile可能不满块)
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord); // 当前tile在M方向的全局起始行号
        uint32_t nLoc = Get<N_VALUE>(blockCoord); // 当前tile在N方向的全局起始列号
        uint32_t kLoc = Get<K_VALUE>(blockCoord); // 当前tile在K方向的全局起始偏移

        // pingpong缓冲选择:
        // tensorUb: Te::Tensor视图(用于BlockMmad写入)
        // l0cOutUbGMM2: LocalTensor视图(用于Vector核CombineTokens读取)
        auto tensorUb = pingpongIdx == 0 ? l0cOutUbFirst : l0cOutUbSecond;
        auto l0cOutUbGMM2 = pingpongIdx == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;

        // ========== Cube核(AIC)分支: 执行MX量化矩阵乘法 ==========
        if constexpr (g_coreType == AscendC::AIC) {
            // 首轮迭代时, 等待前置GMM1+SwiGLU阶段完成
            // 通过轮询groupFlagList, 确认GMM1的所有tile已计算完毕
            // targetLoops = GMM1阶段的总tile数(使用L1_TILE_M_256是因为GMM1的M分块为256)
            // 当groupFlag值等于targetLoops时, 表示GMM1全部完成, Cube可以启动GMM2
            if (loopIdx == startLoopIdx) {
                BlockScheduler gmmBlockScheduler(
                    {m, k, n},
                    BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_256),
                        static_cast<int64_t>(L1_TILE_N))});
                uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm.GetPhyAddr();
                while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {
                    }
                }
            }

            // Cube-Vector同步: 从第二轮起(vecSetSyncCom2>=2), Cube需等待
            // Vector核完成上一轮Combine读取后才能写入当前pingpong缓冲
            // 第一轮时Vector尚未启动, 无需等待
            if (vecSetSyncCom2 >= 2) {
                WaitForVector(pingpongIdx);
            }

            // ---- 切片提取当前tile的GM数据 ----
            // gmBlockA: 矩阵A当前tile [mLoc:mLoc+actualM, kLoc:kLoc+actualK]
            // gmBlockScaleA: A的scale当前tile, K坐标需除以MXFP_SCALE_GROUP_NUM(32)
            //   因为每32个scale元素对应一组MX量化参数
            auto gmBlockA = gmA.Slice(
                Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
            auto gmBlockScaleA = gmScaleA.Slice(
                Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM), Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

            // tensorBlockUb: UB输出缓冲当前tile的切片视图
            //   只取actualShape大小的区域, 处理尾部tile不满块的情况
            auto tensorBlockUb = tensorUb.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

            // gmBlockB: 矩阵B当前tile [kLoc:kLoc+actualK, nLoc:nLoc+actualN]
            // gmBlockScaleB: B的scale当前tile, K坐标同样除以MXFP_SCALE_GROUP_NUM
            auto gmBlockB = gmB.Slice(
                Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.Slice(
                Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));

            // singleShape: 当前tile的实际形状[actualM, actualN, actualK]
            // 尾部tile的维度可能小于标准tile大小(L1_TILE_M_128/L1_TILE_N)
            typename BlockMmad::BlockShape singleShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};

            // 执行MX量化矩阵乘法: A * B2 (带scale反量化) = C2
            // BlockMmad内部完成: GM搬运->L1->L0->MMAD计算->UB输出
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockUb, singleShape);

            // 通知Vector核: 当前pingpong缓冲的矩阵乘结果已就绪, Vector可以开始Combine
            NotifyVector(pingpongIdx);
        }

        // 同步计数器递增: 记录Cube已完成的迭代数
        // 当>=2时, 下一轮Cube需WaitForVector等待Vector读完
        vecSetSyncCom2++;

        // ========== Vector核(AIV)分支: 执行Combine合并发送 ==========
        if constexpr (g_coreType == AscendC::AIV) {
            // 等待Cube核完成当前pingpong缓冲的矩阵乘计算
            WaitForCube(pingpongIdx);

            // V_MTE2事件同步: 确保后续DataCopy(GM->UB)的源数据已就绪
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);

            // ---- 从GM读取triple信息 ----
            // triple信息: 每个token对应8个int32, 包含:
            //   [0] toRankId  - 目标卡号(Combine发送目标)
            //   [1] tokenIdx  - token在源卡中的索引
            //   [2] topkIdx   - top-K选择索引
            //   [3-7] 预留/其他信息
            // triple在workspace中的偏移 = (groupCnt + mLoc) * 32字节
            //   groupCnt: 当前expert分组索引
            //   mLoc: 当前tile的M方向起始行号
            //   32字节对齐(INT_CACHELINE * sizeof(int32_t))
            AscendC::GlobalTensor<int32_t> tripleGm;
            int32_t lenTile = Get<M_VALUE>(actualShape); // 当前tile的行数(token数)
            // tripleTensor: VECCALC上的局部缓冲, 大小=lenTile*8个int32
            // 偏移200*1024字节, 避免与UB缓冲冲突
            LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECCALC, 200 * 1024, lenTile * 8);
            tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.tripleInfoPtr +
                (groupCnt + mLoc) * 32));
            // 将triple信息从GM搬运到VECCALC局部缓冲
            AscendC::DataCopy(tripleTensor, tripleGm, lenTile * 8);

            // ---- CombineTokens: 将矩阵乘结果按triple信息发送到目标卡 ----
            // 遍历当前tile的每个token, 读取其(toRank, tokenIdx, topkIdx)
            // 将该token的C2输出行直接写入目标卡的peermem窗口对应位置
            // 实现MoE的跨卡Combine合并
            MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
                mLoc, nLoc, n, tripleTensor, l0cOutUbGMM2, actualShape, params);

            // MTE3_V事件同步: 确保DataCopy(UB->GM)的数据发送完成
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);

            // 通知Cube核: 当前pingpong缓冲的Combine读取已完成, Cube可以写入下一轮数据
            NotifyCube(pingpongIdx);
        }

        // pingpong索引翻转: 0->1, 1->0, 下一轮使用另一个缓冲区
        pingpongIdx = 1 - pingpongIdx;
    }

    // 更新startBlockIdx: 下一组expert的tile分配起始偏移
    // 取模确保循环分配: 当所有核处理完当前tileNum后, 回到开头继续分配
startBlockIdx = (startBlockIdx + tileNum) % blockNum;
}
} // namespace MegaMoeImpl

#endif