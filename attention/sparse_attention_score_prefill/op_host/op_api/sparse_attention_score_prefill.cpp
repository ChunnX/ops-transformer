/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

/*
 * 本文件是 SparseAttentionScorePrefill 算子的 L0 层算子实现。
 * L0 层负责：创建算子执行器（executor）、分配输出 tensor、调用 InferShape 推断输出形状、
 * 并将算子加入 launcher 列表以便后续在流上启动。
 */

#include "sparse_attention_score_prefill.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

// 注册算子类型，便于框架按名称查找。
OP_TYPE_REGISTER(SparseAttentionScorePrefill);

// SparseAttentionScorePrefill: L0 算子实现入口。
// 功能：根据 query 的数据类型分配输出 tensor（attentionOutTensor），
//       调用 INFER_SHAPE 推断输出形状，
//       通过 ADD_TO_LAUNCHER_LIST_AICORE 将算子及其输入/输出/属性注册到 launcher 列表，
//       供后续在 stream 上启动执行。
// 返回值：分配得到的输出 tensor 指针；若 InferShape 失败则返回 nullptr。
const aclTensor *SparseAttentionScorePrefill(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockTable,
    const aclTensor *k2qRowPtr,
    const aclTensor *k2qQIndices,
    const aclTensor *k2qSlotIndices,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t blockSize,
    int64_t topK,
    int64_t innerPrecise,
    aclOpExecutor *executor)
{
    L0_DFX(SparseAttentionScorePrefill, query, key, value, blockTable,
           k2qRowPtr, k2qQIndices, k2qSlotIndices,
           actualSeqLengthsOptional, actualSeqLengthsKvOptional,
           numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise);

    DataType attentionOutDtype = query->GetDataType();
    auto attentionOutTensor = executor->AllocTensor(attentionOutDtype,
        Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(SparseAttentionScorePrefill,
                           OP_INPUT(query, key, value, blockTable,
                                    k2qRowPtr, k2qQIndices, k2qSlotIndices,
                                    actualSeqLengthsOptional, actualSeqLengthsKvOptional),
                           OP_OUTPUT(attentionOutTensor),
                           OP_ATTR(static_cast<int64_t>(numKeyValueHeads),
                                   static_cast<float>(scaleValue),
                                   static_cast<int64_t>(blockSize),
                                   static_cast<int64_t>(topK),
                                   static_cast<int64_t>(innerPrecise)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "SparseAttentionScorePrefill infer shape failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(SparseAttentionScorePrefill,
                                OP_INPUT(query, key, value, blockTable,
                                         k2qRowPtr, k2qQIndices, k2qSlotIndices,
                                         actualSeqLengthsOptional, actualSeqLengthsKvOptional),
                                OP_OUTPUT(attentionOutTensor),
                                OP_ATTR(static_cast<int64_t>(numKeyValueHeads),
                                        static_cast<float>(scaleValue),
                                        static_cast<int64_t>(blockSize),
                                        static_cast<int64_t>(topK),
                                        static_cast<int64_t>(innerPrecise)));

    return attentionOutTensor;
}

} // namespace l0op
