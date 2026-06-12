/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SPARSE_FLASH_ATTENTION_GRAD_H
#define ACLNN_SPARSE_FLASH_ATTENTION_GRAD_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSparseFlashAttentionGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] query: 表示attention结构的输入Q。支持BSND布局(B,S1,N1,D)或TND布局(T1,N1,D)，数据类型支持FLOAT16、BFLOAT16。
 * @param [in] key: 表示attention结构的输入K。支持BSND布局(B,S2,N2,D)或TND布局(T2,N2,D)，数据类型和query保持一致。
 * @param [in] value: 表示attention结构的输入V。普通场景下shape和key保持一致；KV merge场景下可传入空指针，内部按value=key处理。
 * @param [in] sparseIndices: 稀疏场景下选择的注意力索引，shape为(B,S1,N2,K)或(T1,N2,K)，数据类型支持INT32。
 * @param [in] dOut: 表示attention输出矩阵的梯度，shape和query保持一致，数据类型和query保持一致。
 * @param [in] out: 表示attention输出矩阵，shape和query保持一致，数据类型和query保持一致。
 * @param [in] softmaxMax: 表示attention正向计算的softmax max中间结果，shape为(B,N2,S1,G)或(N2,T1,G)，数据类型支持FLOAT32。
 * @param [in] softmaxSum: 表示attention正向计算的softmax sum中间结果，shape和softmaxMax保持一致，数据类型支持FLOAT32。
 * @param [in] actualSeqLengthsQueryOptional: 表示每个batch中Query的有效token数。TND布局下需要传入，长度与batch size保持一致。
 * @param [in] actualSeqLengthsKvOptional: 表示每个batch中Key/Value的有效token数。TND布局下需要传入，长度与batch size保持一致。
 * @param [in] queryRopeOptional: MLA rope部分Query位置编码的输出，shape为(B,S1,N1,Dr)或(T1,N1,Dr)，可选输入。
 * @param [in] keyRopeOptional: MLA rope部分Key位置编码的输出，shape为(B,S2,N2,Dr)或(T2,N2,Dr)，可选输入。
 * @param [in] scaleValue: 表示缩放系数，建议取值为head dimension开根号的倒数。
 * @param [in] sparseBlockSize: 表示每个被选取KV block包含的token数量。
 * @param [in] layoutOptional: 表示输入数据排布，当前支持"BSND"和"TND"。
 * @param [in] sparseMode: 表示sparse模式，当前支持0和3。
 * @param [in] preTokens: 表示attention中S矩阵滑窗的起始位置，当前仅支持2147483647。
 * @param [in] nextTokens: 表示attention中S矩阵滑窗的终止位置，当前仅支持2147483647。
 * @param [in] deterministic: 表示是否开启确定性计算，与整网确定性参数use_deterministic_algorithms保持一致。
 * @param [out] dQueryOut: 表示query的梯度输出，shape和query保持一致，数据类型和query保持一致。
 * @param [out] dKeyOut: 表示key的梯度输出。KV merge场景下表示key和value的合并梯度，即dK+dV。
 * @param [out] dValueOut: 表示value的梯度输出。KV merge场景下需传入空指针，dValue不单独输出。
 * @param [out] dQueryRopeOutOptional: 表示queryRope的梯度输出。当queryRopeOptional存在时，该参数才输出。
 * @param [out] dKeyRopeOutOptional: 表示keyRope的梯度输出。当keyRopeOptional存在时，该参数才输出。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionGradGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *dOut,
    const aclTensor *out,
    const aclTensor *softmaxMax,
    const aclTensor *softmaxSum,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    double scaleValue,
    int64_t sparseBlockSize,
    char *layoutOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    bool deterministic,
    const aclTensor *dQueryOut,
    const aclTensor *dKeyOut,
    const aclTensor *dValueOut,
    const aclTensor *dQueryRopeOutOptional,
    const aclTensor *dKeyRopeOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnSparseFlashAttentionGrad的第二段接口，用于执行计算。
 *
 * 算子功能：根据sparseIndices对key和value选取数据并重排，计算SparseFlashAttentionGrad反向输出。
 * KV merge场景下，value为空指针，dKeyOut返回dK+dV的合并梯度。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSparseFlashAttentionGradGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionGrad(void *workspace,
                                                                                 uint64_t workspaceSize,
                                                                                 aclOpExecutor *executor,
                                                                                 const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_FLASH_ATTENTION_GRAD_H
