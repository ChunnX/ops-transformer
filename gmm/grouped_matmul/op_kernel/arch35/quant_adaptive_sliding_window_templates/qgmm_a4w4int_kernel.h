/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qgmm_a4w4int_kernel.h
 * \brief GroupMatmulA4W4 桥接(transformer 仓 host tiling -> ops-tensor Blaze A4W4 kernel)。
 *
 * 对接 ops-tensor 的 S8S4 风格 A4W4 kernel (kernel_gmm_a4w4.h::GemmUniversal):
 *   AIV 预处理(int4 weight->int8 + int4 x->int8) -> SyncAll -> AIC FixPipe matmul
 *   (block_mmad_a8w8_fixpipe_quant.h, 硬件施加 weight scale) -> AIV epilogue
 *   (fixpipeResult × perTokenScale -> y, withOffset=false 对称量化)。
 *
 * 桥接层职责: 把 host 的 GMMBaseParamsA4W4 + TCubeTiling 转换为 Blaze GMMTiling + Params。
 *   不修改 GMMBaseParamsA4W4, 所有转换在本文件内完成。
 *
 * 关键转换(详见各处注释):
 *   - kAL1/kBL1: host TCubeTiling 用"份数"(depthA1/stepKa), Blaze 用"元素数"(kAL1)。
 *     kAL1 = stepKa * baseK (= depthA1*baseK/2), kBL1 = stepKb * baseK。
 *   - quantMode/quantGroupSize: host 只有 quantGroupNum, 桥接推断 quantMode 并算 quantGroupSize。
 *   - nBufferNum/withOffset: host 无, 桥接补全(双缓冲=2, A4W4 对称=0)。
 *   - workspace 偏移: host 布局 [SYS 16MB][int8WeightWs][int8XWs][mmOutWs], 必须通过GetUserWorkspace获取用户基地址。
 *
 * 硬约束: 完整 Blaze 桥接需 ops-tensor 侧 tensor_api 等环境就位。用 A4W4_BLAZE_ENABLED 宏守护:
 *   - 未定义(默认): 安全桩, 保证 transformer 仓编译通过。
 *   - 已定义: 启用真正 Blaze 桥接(对接 kernel_gmm_a4w4.h)。
 */

#ifndef QGMM_A4W4_KERNEL_H
#define QGMM_A4W4_KERNEL_H

#include "../grouped_matmul_tiling_data_apt.h"
#include "tensor_api/tensor.h"

// 使能 Blaze 桥接: 走 ops-tensor kernel_gmm_a4w4.h 真分支(非安全桩)。
// 默认代码层开启; 可由编译选项 -UA4W4_BLAZE_ENABLED 关闭。
// 前置条件(任一不满足则桥接不被编译调用, 见 grouped_matmul_apt.cpp:39-41):
//   IS_BLAZE=true(CANN devkit>9.0) && V310_GMM_QUANT_A4W4 &&
//   KERNEL_TYPE==GMM_A4W4_MIX。A4W4 Blaze kernel
#ifndef A4W4_BLAZE_ENABLED
#define A4W4_BLAZE_ENABLED
#endif

#ifdef A4W4_BLAZE_ENABLED
#include "blaze/gemm/kernel/kernel_gmm_a4w4.h"  // S8S4 风格 A4W4 kernel (ops-tensor)
#include "../../grouped_matmul_utils.h"           // GROUPED_MATMUL::GetTensorAddr
#endif

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType,
          class xLayout, class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmA4W4Kernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                      GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                      GM_ADDR user1,
                                      const GroupedMatmulTilingData::GMMBaseParamsA4W4* gmmBaseParams,
                                      const TCubeTiling* mmTilingData,
                                      AscendC::TPipe* que)
{
#ifdef A4W4_BLAZE_ENABLED
    // A5 cube 只支持 int8: x/weight 经 AIV 预处理(int4->int8) 后, AIC 按 int8×int8 计算。
    using AType = int8_t;         // cube 输入 A (int4 x -> int8 workspace 后)
    using BType = int8_t;         // cube 输入 B (int4 weight -> int8 workspace 后)
    using CType = half;           // mmOut workspace 类型(AIC FixPipe 写 / AIV epilogue 读)
    using BiasType = int32_t;     // BlockMmad 模板形参占位(A4W4 无 bias, isBias=false)
    using ScaleType = uint64_t;   // weight pergroup/perchannel scale (X2Scale)
    using YType = yType;          // 输出 half/bfloat16
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;  // (M, N, K, pad)

    // ---- Blaze 类型装配(与 kernel_gmm_a4w4.h::GmmA4W4PerChannel 同款) ----
    // BTypeTuple = (BType, X2ScaleType): BlockMmad 从 tuple 取 BType=int8 与 X2ScaleType=uint64。
    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    // DispatchPolicy 用 FixPipeQuant + Grouped schedule(匹配 block_mmad_a8w8_fixpipe_quant.h 偏特化)。
    using DispatchPolicy =
        Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false, Blaze::Gemm::KernelGroupedMmadWithScaleFixpipeQuant>;
    using QgmmBlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC, BiasType, LayoutC>;
    // A4W4 复用 S8S4 epilogue(withOffset=false 分支: 只做 fixpipeResult×perTokenScale, 无 offset/rowSum)。
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGmmS8S4<YType, CType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using QgmmKernel =
        Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, QgmmBlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QgmmKernel::Params;
    using GMMTiling = typename QgmmKernel::GMMTiling;

    // ============================================================
    // 1) host tiling -> Blaze GMMTiling (值转换, 不改 GMMBaseParamsA4W4)
    // ============================================================
    const uint32_t baseK = gmmBaseParams->baseK;
    const bool isPerGroup = gmmBaseParams->quantGroupNum > 1U;  // 1=perchannel, >1=pergroup

    // kAL1/kBL1 转换(关键):
    //   host TCubeTiling 用"份数": depthA1=L1A 双缓冲总 K 份数, stepKa=depthA1/2=单缓冲 K 份数。
    //   Blaze block_mmad 用"元素数": kAL1=L1A 单缓冲 K 元素数, aL1OneBuffer=baseM*kAL1, kAL1Iter_=CeilDiv(K,kAL1)。
    //   关系: kAL1 = stepKa * baseK (= depthA1*baseK/2)。@950 baseK=128, stepKa=8 -> kAL1=1024。
    //   若误用 depthA1(=16) 当 kAL1, 则 kAL1<baseK, L0 取 baseK 时 L1 缓存不足, K 迭代/L1 分块错乱。
    const uint32_t kAL1 = static_cast<uint32_t>(mmTilingData->stepKa) * baseK;
    const uint32_t kBL1 = static_cast<uint32_t>(mmTilingData->stepKb) * baseK;

    // quantGroupSize: pergroup = K/quantGroupNum(host 校验 K%quantGroupNum==0, 整除);
    //   perchannel 填 0, kernel Init 内部自取 problemK。
    const uint32_t quantGroupSize = isPerGroup
        ? static_cast<uint32_t>(gmmBaseParams->k / gmmBaseParams->quantGroupNum)
        : 0U;
    const uint32_t quantMode = isPerGroup
        ? static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERGROUP_MODE)
        : static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERCHANNEL_MODE);

    GMMTiling gmmParams{};
    gmmParams.groupNum = gmmBaseParams->groupNum;
    gmmParams.m = static_cast<int64_t>(gmmBaseParams->m);
    gmmParams.n = static_cast<int64_t>(gmmBaseParams->n);
    gmmParams.k = static_cast<int64_t>(gmmBaseParams->k);
    gmmParams.baseM = gmmBaseParams->baseM;
    gmmParams.baseN = gmmBaseParams->baseN;
    gmmParams.baseK = baseK;
    gmmParams.quantGroupSize = quantGroupSize;
    gmmParams.quantMode = quantMode;
    gmmParams.kAL1 = kAL1;
    gmmParams.kBL1 = kBL1;
    gmmParams.nBufferNum = 2U;  // L1 双缓冲(对应 host dbL0A=dbL0B=2)
    gmmParams.dbL0C = static_cast<uint8_t>(mmTilingData->dbL0C);
    gmmParams.withOffset = 0U;  // A4W4 对称量化(offset=null, 无 xRowSum 修正)
    gmmParams.groupListType = gmmBaseParams->groupListType;
    gmmParams.isPerTokenQuant = static_cast<uint8_t>(gmmBaseParams->isPerTokenQuant);  // F3: Run prologue 据此填全 1

    // ============================================================
    // 2) 用户 workspace 偏移([int8WeightWs][int8XWs][mmOutWs])
    //   host 未把 int8WeightWs/int8XWs 偏移下发到 GMMBaseParamsA4W4, 桥接层按 host
    //   CalWorkspaceA4W4 同款公式重算。DAV3510 必须通过 GetUserWorkspace 取得跳过
    //   16MB 系统保留区后的用户段基址，不能直接对原始 workspace 形参加偏移。
    // ============================================================
    constexpr uint64_t A4W4_MMOUT_PIPELINE = 4UL;  // 与 host A4W4_MMOUT_PIPELINE 一致
    // reserved: bit0=weightIsNz, bit1=transB (host PostTiling 下发)。
    const uint64_t weightFmtFlags = gmmBaseParams->reserved;
    const bool weightIsNz = (weightFmtFlags & 1ULL) != 0;
    const bool transB = (weightFmtFlags & 2ULL) != 0;
    // int4->int8 展开后体积: int4(0.5B/元素) -> int8(1B/元素)。weight[E,K,N], x[M,K]。
    // NZ 展开后的 INT8 物理布局按 K/16、N/32 补齐，必须与 weight preprocess 的
    // nzOutputNarrowBlockCount/nzOutputWideBlockCount 完全一致，避免后续 X workspace 重叠。
    constexpr uint64_t INT8_NZ_K_ALIGN = 16UL;
    constexpr uint64_t INT8_NZ_N_ALIGN = 32UL;
    const uint64_t int8WeightK = weightIsNz
        ? (static_cast<uint64_t>(gmmBaseParams->k) + INT8_NZ_K_ALIGN - 1UL) / INT8_NZ_K_ALIGN * INT8_NZ_K_ALIGN
        : static_cast<uint64_t>(gmmBaseParams->k);
    const uint64_t int8WeightN = weightIsNz
        ? (static_cast<uint64_t>(gmmBaseParams->n) + INT8_NZ_N_ALIGN - 1UL) / INT8_NZ_N_ALIGN * INT8_NZ_N_ALIGN
        : static_cast<uint64_t>(gmmBaseParams->n);
    const uint64_t int8WeightWs = static_cast<uint64_t>(gmmBaseParams->groupNum) *
                                  int8WeightK * int8WeightN;  // int8 字节
    const uint64_t int8XWs = static_cast<uint64_t>(gmmBaseParams->m) *
                             static_cast<uint64_t>(gmmBaseParams->k);  // int8 字节
    const uint64_t mmOutWs = A4W4_MMOUT_PIPELINE * gmmBaseParams->baseM * gmmBaseParams->baseN *
                             gmmBaseParams->coreNum * sizeof(uint16_t);
    GM_ADDR int8WeightWsAddr = user1;
    GM_ADDR int8XWsAddr = int8WeightWsAddr + int8WeightWs;
    GM_ADDR mmOutWsAddr = int8XWsAddr + int8XWs;  // cube FixPipe 输出 / AIV epilogue 读
    // F3: isPerTokenQuant=0 时 perTokenScale 兜底段 (mmOut 之后, totalM·float)。
    //     桥接仅切换 perTokenScaleGmAddr 指向此段; 全 1 填充在 kernel_gmm_a4w4.h::Run prologue。
    GM_ADDR perTokenScaleFillWsAddr = mmOutWsAddr + mmOutWs;

    // ============================================================
    // 3) GM 地址透传(single x/w/y)
    // ============================================================
    // 原始 int4 x/weight GM 地址(预处理输入); cube 读 int8 workspace, 不读原始 int4。
    GM_ADDR xGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<int4b_t>(0, x));
    GM_ADDR weightGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<int4b_t>(0, weight));
    GM_ADDR yGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    GM_ADDR scaleGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleType>(0, scale));
    // F3: isPerTokenQuant=0 (无 perTokenScale 输入) 时用 workspace 全 1 段, epilogue 正常 mul (乘 1)。
    //     保持 S8S4 epilogue 逐字同步不改。全 1 填充见下方 §3.5 (AIV, gmm(params) 前)。
    GM_ADDR perTokenScaleGmAddr = (gmmBaseParams->isPerTokenQuant == 0U)
        ? perTokenScaleFillWsAddr : perTokenScale;

    // ============================================================
    // 4) 组装 Params 各子结构
    // ============================================================
    // WeightPreprocess: int4 weight -> int8 workspace(AIV)。
    typename QgmmKernel::WeightPreprocessParams weightPreprocessParams{};
    weightPreprocessParams.weightGmAddr = weightGmAddr;
    weightPreprocessParams.workspaceGmAddr = int8WeightWsAddr;
    weightPreprocessParams.groupNum = gmmBaseParams->groupNum;
    weightPreprocessParams.n = gmmBaseParams->n;
    weightPreprocessParams.k = gmmBaseParams->k;
    // inputTransposedNz: weight 为 NZ 且 transB 时, weightPreprocessOp_ 走 ConvertTransposedNzTile
    // ([E,N,K] int4 NZ -> [E,K,N] int8 NZ); 否则走标准 NZ/ND 路径。
    // host tiling 通过 GMMBaseParamsA4W4.reserved 下发: bit0=weightIsNz, bit1=transB。
    weightPreprocessParams.inputTransposedNz = weightIsNz && transB;

    // XPreprocess: int4 x -> int8 workspace(AIV, A4W4 独有)。
    typename QgmmKernel::XPreprocessParams xPreprocessParams{};
    xPreprocessParams.srcInt4GmAddr = xGmAddr;
    xPreprocessParams.dstInt8GmAddr = int8XWsAddr;
    xPreprocessParams.groupNum = 1UL;                  // X 单 tensor，groupNum=1
    xPreprocessParams.outerDim = gmmBaseParams->m;      // M
    xPreprocessParams.innerDim = gmmBaseParams->k;      // K
    xPreprocessParams.inputTransposedNz = false;        // X 为 ND，无 NZ 转置

    // BlockMmad: cube A=int8 x ws, B=int8 weight ws, C=mmOut ws, scaleB=weight scale。
    typename QgmmBlockMmad::Params mmadParams{};
    mmadParams.aGmAddr = int8XWsAddr;       // A 读 int8 x workspace(Stage0 后)
    mmadParams.bGmAddr = int8WeightWsAddr;  // B 读 int8 weight workspace
    mmadParams.cGmAddr = mmOutWsAddr;       // C 写 mmOut workspace(供 AIV epilogue 读)
    mmadParams.biasGmAddr = nullptr;        // A4W4 无 bias
    mmadParams.scaleAGmAddr = nullptr;      // A4W4 无 activation scale(perTokenScale 在 epilogue)
    mmadParams.scaleBGmAddr = scaleGmAddr;  // weight scale(uint64, FixPipe 硬件施加)

    // BlockEpilogue: fixpipeResult(half) × perTokenScale -> YType, withOffset=false。
    typename BlockEpilogue::Params epilogueParams{};
    epilogueParams.workspaceGmAddr = mmOutWsAddr;         // FixPipe 结果(half)
    epilogueParams.perTokenScaleGmAddr = perTokenScaleGmAddr;
    epilogueParams.offsetGmAddr = nullptr;                // A4W4 无 offset
    epilogueParams.xRowSumGmAddr = nullptr;               // A4W4 无 rowSum(对称量化)
    epilogueParams.outGmAddr = yGmAddr;
    epilogueParams.n = gmmBaseParams->n;
    epilogueParams.baseM = gmmBaseParams->baseM;
    epilogueParams.baseN = gmmBaseParams->baseN;
    epilogueParams.withOffset = false;  // A4W4 对称(withOffset=false, offset/rowSum 不用)

    Params params{};
    params.problemShape = {gmmParams.m, gmmParams.n, gmmParams.k, static_cast<int64_t>(1)};
    params.weightPreprocessParams = weightPreprocessParams;
    params.xPreprocessParams = xPreprocessParams;
    params.mmadParams = mmadParams;
    params.epilogueParams = epilogueParams;
    params.groupListGmAddr = groupList;
    params.gmmParams = gmmParams;

    // F3: perTokenScale 兜底全 1 填充已移入 kernel_gmm_a4w4.h::Run prologue (x preprocess 后,
    //     复用入口透传的同一 pipe, 避免 TPipe 实例冲突)。桥接仅负责 workspace 段预留 +
    //     perTokenScaleGmAddr 切换 (isPerTokenQuant=0 时指向 fillWsAddr) + isPerTokenQuant 透传。
    //     pipe 透传: 入口 grouped_matmul_apt.cpp 的 tPipe 经 que 传入 kernel, 与 S8S4 一致,
    //     不在 kernel 内重复创建 TPipe (否则双 TPipe 实例导致 SyncAll 等核内资源管理异常)。

    QgmmKernel gmm;
    gmm(params, que);
#else
    // 安全桩: A4W4_BLAZE_ENABLED 未定义。ops-tensor 侧 tensor_api 等环境未就位时,
    // 实例化 Blaze kernel 链会触发底层头编译问题。回退桩保证 transformer 仓编译通过。
    // 启用方式: ops-tensor 环境就位后, 在编译选项定义 A4W4_BLAZE_ENABLED。
    (void)x; (void)weight; (void)bias; (void)scale;
    (void)groupList; (void)perTokenScale; (void)y; (void)user1;
    (void)gmmBaseParams; (void)mmTilingData; (void)que;
#endif
}

#endif  // QGMM_A4W4_KERNEL_H
