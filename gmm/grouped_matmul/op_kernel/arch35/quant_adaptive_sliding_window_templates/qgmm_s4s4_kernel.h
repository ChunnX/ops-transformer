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
 * \file qgmm_s4s4_kernel.h
 * \brief Adapter from GroupedMatmul S4S4 tiling data to the ops-tensor S4S4 kernel.
 */

#ifndef QGMM_S4S4_KERNEL_H
#define QGMM_S4S4_KERNEL_H

#include "../grouped_matmul_tiling_data_apt.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/kernel/kernel_gmm_a4w4.h"
#include "../int4_tensor_to_int8_preprocess.h"
#include "../int4_weight_to_int8_preprocess.h"
#include "../../grouped_matmul_utils.h"

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType,
          class xLayout, class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmS4S4Kernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                      GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                      GM_ADDR user1,
                                      const GroupedMatmulTilingData::GMMBaseParamsS4S4* gmmBaseParams,
                                      const TCubeTiling* mmTilingData,
                                      AscendC::TPipe* que)
{
    // A5 cube 只支持 int8: x/weight 经 AIV 预处理(int4->int8) 后, AIC 按 int8×int8 计算。
    using AType = int8_t;         // cube 输入 A (int4 x -> int8 workspace 后)
    using BType = int8_t;         // cube 输入 B (int4 weight -> int8 workspace 后)
    using CType = half;           // mmOut workspace 类型(AIC FixPipe 写 / AIV epilogue 读)
    using BiasType = int32_t;     // BlockMmad 模板形参占位(S4S4 无 bias, isBias=false)
    using ScaleType = uint64_t;   // weight pergroup/perchannel scale (X2Scale)
    using YType = yType;          // 输出 half/bfloat16
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;  // (M, N, K, pad)

    // ---- Blaze 类型装配(与 kernel_gmm_a4w4.h::GmmS4S4PerChannel 同款) ----
    // BTypeTuple = (BType, X2ScaleType): BlockMmad 从 tuple 取 BType=int8 与 X2ScaleType=uint64。
    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    // DispatchPolicy 用 FixPipeQuant + Grouped schedule(匹配 block_mmad_a8w8_fixpipe_quant.h 偏特化)。
    using DispatchPolicy =
        Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false, Blaze::Gemm::KernelGroupedMmadWithScaleFixpipeQuant>;
    using QgmmBlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC, BiasType, LayoutC>;
    // S4S4 复用 S8S4 epilogue(withOffset=false 分支: 只做 fixpipeResult×perTokenScale, 无 offset/rowSum)。
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGmmS8S4<YType, CType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using QgmmKernel =
        Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, QgmmBlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QgmmKernel::Params;
    using GMMTiling = typename QgmmKernel::GMMTiling;

    const uint32_t baseK = gmmBaseParams->baseK;
    const bool isPerGroup = gmmBaseParams->quantGroupNum > 1U;  // 1=perchannel, >1=pergroup

    // Blaze expects the K-element count of one L1 stage; TCubeTiling stepK is the number of baseK tiles.
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
    gmmParams.withOffset = 0U;  // S4S4 对称量化(offset=null, 无 xRowSum 修正)
    gmmParams.groupListType = gmmBaseParams->groupListType;

    // The user workspace follows the runtime-reserved system workspace.
    constexpr uint64_t S4S4_MMOUT_PIPELINE = 4UL;  // 与 host S4S4_MMOUT_PIPELINE 一致
    // reserved: bit0=weightIsNz, bit1=transB (host PostTiling 下发)。
    const uint64_t weightFmtFlags = gmmBaseParams->reserved;
    const bool weightIsNz = (weightFmtFlags & 1ULL) != 0;
    const bool transB = (weightFmtFlags & 2ULL) != 0;
    const bool weightNzC032 = (weightFmtFlags & 4ULL) != 0;
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
    const uint64_t mmOutWs = S4S4_MMOUT_PIPELINE * gmmBaseParams->baseM * gmmBaseParams->baseN *
                             gmmBaseParams->coreNum * sizeof(uint16_t);
    GM_ADDR int8WeightWsAddr = user1;
    GM_ADDR int8XWsAddr = int8WeightWsAddr + int8WeightWs;
    GM_ADDR mmOutWsAddr = int8XWsAddr + int8XWs;  // cube FixPipe 输出 / AIV epilogue 读
    // F3: isPerTokenQuant=0 时 perTokenScale 兜底段 (mmOut 之后, totalM·float)。
    GM_ADDR perTokenScaleFillWsAddr = mmOutWsAddr + mmOutWs;

    // 原始 int4 x/weight GM 地址(预处理输入); cube 读 int8 workspace, 不读原始 int4。
    GM_ADDR xGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<int4b_t>(0, x));
    GM_ADDR weightGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<int4b_t>(0, weight));
    GM_ADDR yGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    GM_ADDR scaleGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleType>(0, scale));
    // F3: isPerTokenQuant=0 时 epilogue 使用 host 预留的 perTokenScale 兜底段。
    GM_ADDR perTokenScaleGmAddr = (gmmBaseParams->isPerTokenQuant == 0U)
        ? perTokenScaleFillWsAddr : perTokenScale;

    // Operator-owned preprocessing: int4 weight/X -> int8 workspaces on AIV.
    using WeightPreprocess =
        GROUPED_MATMUL::INT4_PREPROCESS::Int4WeightToInt8Preprocess<LayoutB>;
    typename WeightPreprocess::Params weightPreprocessParams{};
    weightPreprocessParams.weightGmAddr = weightGmAddr;
    weightPreprocessParams.workspaceGmAddr = int8WeightWsAddr;
    weightPreprocessParams.groupNum = gmmBaseParams->groupNum;
    weightPreprocessParams.n = gmmBaseParams->n;
    weightPreprocessParams.k = gmmBaseParams->k;
    // inputTransposedNz: weight 为 NZ 且 transB 时走 ConvertTransposedNzTile
    // ([E,N,K] int4 NZ -> [E,K,N] int8 NZ); 否则走标准 NZ/ND 路径。
    // host tiling 通过 GMMBaseParamsA4W4.reserved 下发:
    // bit0=weightIsNz, bit1=transB, bit2=input NZ C0=32。
    weightPreprocessParams.inputTransposedNz = weightIsNz && transB;
    weightPreprocessParams.inputNzC032 = weightIsNz && weightNzC032;

    // XPreprocess: int4 x -> int8 workspace(AIV, A4W4 独有)。
    using XPreprocess =
        GROUPED_MATMUL::INT4_PREPROCESS::Int4TensorToInt8Preprocess<LayoutA>;
    typename XPreprocess::Params xPreprocessParams{};
    xPreprocessParams.srcInt4GmAddr = xGmAddr;
    xPreprocessParams.dstInt8GmAddr = int8XWsAddr;
    xPreprocessParams.groupNum = 1UL;                  // X 单 tensor，groupNum=1
    xPreprocessParams.outerDim = gmmBaseParams->m;      // M
    xPreprocessParams.innerDim = gmmBaseParams->k;      // K
    xPreprocessParams.inputTransposedNz = false;        // X 为 ND，无 NZ 转置

    {
        WeightPreprocess weightPreprocess;
        weightPreprocess(weightPreprocessParams, que);
        XPreprocess xPreprocess;
        xPreprocess(xPreprocessParams, que);

        // Publish both expanded workspaces before releasing preprocessing UB
        // and entering the compute-only Tensor API kernel.
        AscendC::SyncAll<false>();
        que->Reset();
    }

    // BlockMmad: cube A=int8 x ws, B=int8 weight ws, C=mmOut ws, scaleB=weight scale。
    typename QgmmBlockMmad::Params mmadParams{};
    mmadParams.aGmAddr = int8XWsAddr;       // A 读 int8 x workspace(Stage0 后)
    mmadParams.bGmAddr = int8WeightWsAddr;  // B 读 int8 weight workspace
    mmadParams.cGmAddr = mmOutWsAddr;       // C 写 mmOut workspace(供 AIV epilogue 读)
    mmadParams.biasGmAddr = nullptr;        // S4S4 无 bias
    mmadParams.scaleAGmAddr = nullptr;      // S4S4 无 activation scale(perTokenScale 在 epilogue)
    mmadParams.scaleBGmAddr = scaleGmAddr;  // weight scale(uint64, FixPipe 硬件施加)

    // BlockEpilogue: fixpipeResult(half) × perTokenScale -> YType, withOffset=false。
    typename BlockEpilogue::Params epilogueParams{};
    epilogueParams.workspaceGmAddr = mmOutWsAddr;         // FixPipe 结果(half)
    epilogueParams.perTokenScaleGmAddr = perTokenScaleGmAddr;
    epilogueParams.offsetGmAddr = nullptr;                // S4S4 无 offset
    epilogueParams.xRowSumGmAddr = nullptr;               // S4S4 无 rowSum(对称量化)
    epilogueParams.outGmAddr = yGmAddr;
    epilogueParams.n = gmmBaseParams->n;
    epilogueParams.baseM = gmmBaseParams->baseM;
    epilogueParams.baseN = gmmBaseParams->baseN;
    epilogueParams.withOffset = false;  // S4S4 对称(withOffset=false, offset/rowSum 不用)

    Params params{};
    params.mmadParams = mmadParams;
    params.epilogueParams = epilogueParams;
    params.groupListGmAddr = groupList;
    params.gmmParams = gmmParams;

    QgmmKernel gmm;
    gmm(params, que);
}

#endif  // QGMM_S4S4_KERNEL_H
