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
 * \file moe_token_permute_with_ep.cpp
 * \brief
 */
#include "moe_mrgsort_out_with_ep.h"
#include "moe_mrgsort_with_ep.h"
#include "moe_sort_multi_core_with_ep.h"
#include "moe_sort_one_core_with_ep.h"
#include "moe_index_copy_with_ep.h"
#include "moe_index_copy_spilt_d_with_ep.h"
#if !defined(DTYPE_TOKENS)
#define DTYPE_TOKENS bfloat16_t
#endif
#if !defined(DTYPE_INDICES)
#define DTYPE_INDICES int64_t
#endif
#if !defined(DTYPE_PROBS)
#define DTYPE_PROBS bfloat16_t
#endif
using namespace AscendC;
using namespace MoeTokenPermuteWithEp;
#define GENERAL_OP_IMPL(sortClass1, sortClass2, indexCopyClass, ...)                                       \
    do {                                                                                                   \
        sortClass1<DTYPE_INDICES> op;                                                                      \
        op.Init(indices, sortedIndices, userWS, t, &sortPipe);                                             \
        op.Process();                                                                                      \
        sortPipe.Destroy();                                                                                \
        sortClass2<int32_t> op2;                                                                           \
        TPipe sortPipe2;                                                                                   \
        op2.Init(sortedIndices, sortedIndices, userWS, t, &sortPipe2);                                     \
        op2.Process();                                                                                     \
        sortPipe2.Destroy();                                                                               \
        TPipe MoeindexCopyPipe;                                                                            \
        indexCopyClass<__VA_ARGS__> indexCopyOp;                                                           \
        indexCopyOp.Init(tokens, probs, sortedIndices, permuteTokens, permuteProbs, t, &MoeindexCopyPipe); \
        indexCopyOp.Process();                                                                             \
    } while (0)

extern "C" __global__ __aicore__ void moe_token_permute_with_ep(
    GM_ADDR tokens, GM_ADDR indices, GM_ADDR probs, GM_ADDR permuteTokens, GM_ADDR sortedIndices, GM_ADDR permuteProbs,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    // 空输入保护: n*topK==0 时三个输出均为空 tensor，语义上无任何工作，在 kernel 总入口统一短路。
    // 后果: 不做保护时 tiling 产生 CoreLoop=0，两个搬运类 Process() 尾部无条件调用的
    // CopyInIndices(CoreLoop-1=-1) 负偏移访问 GM——未定义行为，950 实测设备侧任务失败
    // （NO_OUTPUT，TTK bb_l1_0/1/50），且随运行时内存布局可能无症状漂移，不可依赖。
    // 选择入口防护而非下游修补: 一处覆盖排序/搬运全部阶段及全部 tilingKey 路径（sort 类
    // 同为"主循环+尾部单独处理"模式，同样依赖 CoreLoop>=1）；替代方案（各 Process 加守卫
    // 改动面大易漏 / tiling 钳位 CoreLoop>=1 语义扭曲 / tiling 层报错违反"支持空 tensor"
    // 文档承诺）均不如入口短路。
    if (tilingData.n * tilingData.topK == 0) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    auto t = &tilingData;
    TPipe sortPipe;
    if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopyOp, DTYPE_TOKENS, DTYPE_PROBS, false);
    } else if (TILING_KEY_IS(3)) {
        GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopySpiltDOp, DTYPE_TOKENS, DTYPE_PROBS, false);
    } else if (TILING_KEY_IS(2)) {
        GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopyOp, DTYPE_TOKENS, DTYPE_PROBS, false);
    } else if (TILING_KEY_IS(4)) {
        GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySpiltDOp, DTYPE_TOKENS, DTYPE_PROBS, false);
    } else if (TILING_KEY_IS(5)) {
        GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopyOp, DTYPE_TOKENS, DTYPE_PROBS, true);
    } else if (TILING_KEY_IS(7)) {
        GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopySpiltDOp, DTYPE_TOKENS, DTYPE_PROBS, true);
    } else if (TILING_KEY_IS(6)) {
        GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopyOp, DTYPE_TOKENS, DTYPE_PROBS, true);
    } else if (TILING_KEY_IS(8)) {
        GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySpiltDOp, DTYPE_TOKENS, DTYPE_PROBS, true);
    }
}
