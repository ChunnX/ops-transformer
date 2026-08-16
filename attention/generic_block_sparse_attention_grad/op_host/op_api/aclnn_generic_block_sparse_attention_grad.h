/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_H
#define ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGenericBlockSparseAttentionGrad的第一段接口,根据具体的计算流程,计算workspace大小
 * @domain aclnn_ops_infer
 *
 * @param [in] query: Query tensor输入
 * @param [in] key: Key tensor输入
 * @param [in] value: Value tensor输入
 * @param [in] dout: attention out 梯度输入
 * @param [in] out: attention out tensor输入
 * @param [in] lse: softmax tensor输入
 * @param [in] sparseBlockIdx: K_OUT 使用 KV2Q 有序稀疏块索引
 * @param [in] sparseBlockCount: 每个 KV block 行的有效索引数
 * @param [in] metadataOptional: metadata tensor（本轮仅透传）
 * @param [in] attenMaskOptional: Attention mask (只支持传入nullptr)
 * @param [in] cuSeqLengthsQOptional: Q 前缀和 INT64 tensor
 * @param [in] cuSeqLengthsKvOptional: KV 前缀和 INT64 tensor
 * @param [in] sequsedQOptional: Q 实际长度 INT32 tensor
 * @param [in] sequsedKvOptional: KV 实际长度 INT32 tensor
 * @param [in] blockShape: 稀疏块形状数组 [blockShapeX, blockShapeY]
 * @param [in] isPackedGQA: GQA 稀疏 pattern 共享标志
 * @param [in] layoutQ: Query输入layout ("TND", "BNSD", "BSND")
 * @param [in] layoutKv: KV输入layout ("TND", "BNSD", "BSND")
 * @param [in] maskType: Mask类型
 * @param [in] scaleValue: 缩放因子 (double类型)
 * @param [in] softmaxPrecision: Softmax精度级别
 * @param [in] winLeft: 左窗口，仅支持-1
 * @param [in] winRight: 右窗口，仅支持-1
 * @param [out] dq: Attention输出tensor
 * @param [out] dk: Attention输出tensor
 * @param [out] dv: Attention输出tensor
 * @param [out] workspaceSize: 返回计算所需workspace大小
 * @param [out] executor: 返回op执行器
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *dout,
    const aclTensor *out,
    const aclTensor *lse,
    const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount,
    const aclTensor *metadataOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    char *layoutQ,
    char *layoutKv,
    double scaleValue,
    int64_t maskType,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    aclTensor *dq,
    aclTensor *dk,
    aclTensor *dv,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnGenericBlockSparseAttentionGrad的第二段接口,用于执行计算
 *
 * @param [in] workspace: 工作空间地址
 * @param [in] workspaceSize: 工作空间大小
 * @param [in] executor: op执行器
 * @param [in] stream: acl stream流
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_H
