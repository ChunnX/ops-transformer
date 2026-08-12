/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND OF EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * =========================================================================================
 * Kernel 接口层 —— 模板类型选择与 GEMM/Epilogue 组件组装
 * =========================================================================================
 *
 * 【本文件职责】
 *   将 kernel 入口传入的 GM 指针打包为 params 结构体, 并根据模板参数组装
 *   GEMM Block (QK/PV matmul) 和 Epilogue (softmax/combine) 组件,
 *   最终实例化 SasaPrefillRegularKernelArch35 并调用其 operator().
 *
 * 【模板参数说明】
 *   InDtype  — 输入数据类型 (Q/K/V), 当前固定为 bfloat16_t
 *   SMDtype  — score 矩阵 S 的数据类型 (中间精度), 当前固定为 bfloat16_t
 *   REDtype  — PV fixpipe 输出 (O_partial) 的数据类型:
 *              float      → fp32 路径 (innerPrecise!=1, TilingKey=20001)
 *              bfloat16_t → bf16 路径 (innerPrecise==1, TilingKey=20002)
 *
 * 【组件组装关系】
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │                    SasaPrefillRegularKernelArch35                   │
 *   │                                                                     │
 *   │  Phase 1:                                                           │
 *   │    BlockMmadQK    — Cube 侧 QK matmul: S = Q × K^T                  │
 *   │    EpilogueSoftmax — Vector 侧 online softmax: P = exp(S - rowMax)  │
 *   │    BlockMmadPV    — Cube 侧 PV matmul: O_partial = P × V            │
 *   │                                                                     │
 *   │  Phase 2:                                                           │
 *   │    EpilogueRescaleO — Vector 侧 combine: O = Σ scale[k]·O_norm[k]  │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 *   数据流: Q,K → BlockMmadQK → S(UB) → EpilogueSoftmax → P(L1) → BlockMmadPV → O_partial(GM)
 *          O_partial(GM) + max/sum(GM) → EpilogueRescaleO → attentionOut(GM)
 *
 * 【GQA 映射】
 *   groupSize = numHeads / kvHeads
 *   同一个 KV head 的 K/V 只加载一次, groupSize 个 Q head 共享该 K/V block
 * =========================================================================================
 */

#include "kernel_operator.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/sparse_attention_score_prefill_kernel_arch35.h"
#endif
#if (__CCE_AICORE__ == 220)
// A2 使用独立的 kernel 类 (SasaPrefillRegularKernelA2),
// 包含 L0C→GM→UB 等 A2 特有逻辑
#include "arch22/sparse_attention_score_prefill_kernel_a2.h"
#endif

#if (__CCE_AICORE__ == 310)

using namespace NpuArch;
using namespace SasaPrefillKernelArch35;

// 模板化 kernel 接口: 根据入口 tiling key 派发不同 REDtype 实例化
//   InDtype  = bfloat16_t (输入 bf16)
//   SMDtype  = bfloat16_t (score 中间 bf16, Cube fixpipe 到 UB)
//   REDtype  = float / bfloat16_t (O_partial 精度, 由 innerPrecise 决定)
template <class InDtype, class SMDtype, class REDtype>
__global__ __aicore__ void SasaPrefillInferIntf(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR blockTable,
    GM_ADDR k2qRowPtr, GM_ADDR k2qQIndices,
    GM_ADDR k2qSlotIndices,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
{
    // === Phase 1 组件: QK matmul (Cube) ===
    // BlockMmadQKPrefillArch35: Q[bf16] × K^T[bf16] → S[bf16], K 常驻 L1
    using KernelBlockMmadQK = Gemm::Block::BlockMmadQKPrefillArch35<InDtype, InDtype, bfloat16_t>;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using PType = Gemm::GemmType<InDtype, LayoutPDummy>;
    using SType = Gemm::GemmType<bfloat16_t, LayoutS>;

    // === Phase 1 组件: Online Softmax (Vector) ===
    // 对 S 做 online softmax: 计算 rowMax/rowSum, 生成 P=exp(S-rowMax) cast bf16 写 L1
    // EpilogueOnlineSoftmaxBsa: BSA (Block Sparse Attention) dispatch policy
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsa;
    using KernelEpilogueSoftmax = Epilogue::Block::BlockEpilogue<
        DispatchPolicyOnlineSoftmax, PType, SType>;

    // === Phase 1 组件: PV matmul (Cube) ===
    // BlockMmadPVPrefillArch35: P[bf16] × V[bf16] → O_partial[REDtype], V 常驻 L1
    // REDtype=float → fixpipe F3222F32; REDtype=bf16 → fixpipe F3222BF16
    using KernelBlockMmadPV = Gemm::Block::BlockMmadPVPrefillArch35<InDtype, InDtype, REDtype>;

    // === Phase 2 组件: Rescale & Combine (Vector) ===
    // EpilogueRescaleOPrefillArch35: IFA FlashDecode 路径
    //   CopyLseIn → ComputeScaleValue → CopyAccumOutIn(O/rowSum) → ReduceFinalRes → CopyFinalResOut
    using DispatchPolicyRescaleO = Epilogue::EpilogueRescaleOPrefillArch35;
    using KernelEpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, InDtype, REDtype>;

    // === 组装最终 Kernel ===
    // 将 4 个组件作为模板参数传入, kernel 内部按 Phase1/Phase2 分别调用
    using SasaPrefillKernel = SasaPrefillRegularKernelArch35<
        KernelBlockMmadQK,
        KernelEpilogueSoftmax,
        KernelBlockMmadPV,
        KernelEpilogueRescaleO>;

    // 打包所有 GM 指针为 params 结构体
    SasaPrefillKernelParamsArch35 params{
        q, k, v, blockTable,
        k2qRowPtr, k2qQIndices, k2qSlotIndices,
        actualQseqlen, actualKvseqlen,
        o, workspace, tiling};
    // 实例化并执行 kernel
    SasaPrefillKernel sasaPrefillKernel;
    sasaPrefillKernel(params);
}

#endif

#if (__CCE_AICORE__ == 220)

// ============================================================================
// A2 (AtlasA2) kernel 接口 — 与 A5 结构一致, 仅替换组件为 A2 版本
// ============================================================================
using namespace NpuArch;
using namespace SasaPrefillKernelA2;

template <class InDtype, class SMDtype, class REDtype>
__global__ __aicore__ void SasaPrefillInferIntf(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR blockTable,
    GM_ADDR k2qRowPtr, GM_ADDR k2qQIndices,
    GM_ADDR k2qSlotIndices,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
{
    // === Phase 1 组件: QK matmul (Cube) — A2 版本 ===
    using KernelBlockMmadQK = Gemm::Block::BlockMmadQKPrefillA2<InDtype, InDtype, bfloat16_t>;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using PType = Gemm::GemmType<InDtype, LayoutPDummy>;
    using SType = Gemm::GemmType<bfloat16_t, LayoutS>;

    // === Phase 1 组件: Online Softmax (Vector) — A2 版本 ===
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxPrefillA2;
    using KernelEpilogueSoftmax = Epilogue::Block::BlockEpilogue<
        DispatchPolicyOnlineSoftmax, PType, SType>;

    // === Phase 1 组件: PV matmul (Cube) — A2 版本 ===
    using KernelBlockMmadPV = Gemm::Block::BlockMmadPVPrefillA2<InDtype, InDtype, REDtype>;

    // === Phase 2 组件: Rescale & Combine (Vector) — A2 版本 ===
    using DispatchPolicyRescaleO = Epilogue::EpilogueRescaleOPrefillA2;
    using KernelEpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, InDtype, REDtype>;

    // === 组装最终 Kernel (A2 独立 kernel 类) ===
    using SasaPrefillKernel = SasaPrefillRegularKernelA2<
        KernelBlockMmadQK,
        KernelEpilogueSoftmax,
        KernelBlockMmadPV,
        KernelEpilogueRescaleO>;

    SasaPrefillKernelParamsA2 params{
        q, k, v, blockTable,
        k2qRowPtr, k2qQIndices, k2qSlotIndices,
        actualQseqlen, actualKvseqlen,
        o, workspace, tiling};
    SasaPrefillKernel sasaPrefillKernel;
    sasaPrefillKernel(params);
}

#endif
