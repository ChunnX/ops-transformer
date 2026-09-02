#include "generic_block_sparse_attention_grad_metadata.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GenericBlockSparseAttentionGradMetadata);

const aclTensor *GenericBlockSparseAttentionGradMetadata(
    const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    const aclTensor *blockShape, int64_t isPackedGQA, char *layoutQ, char *layoutKv, int64_t maskType,
    int64_t softmaxPrecision, int64_t winLeft, int64_t winRight, const aclTensor *metadata, int64_t aicCoreNum,
    int64_t aivCoreNum, const char *socVersion, aclOpExecutor *executor)
{
    L0_DFX(GenericBlockSparseAttentionGradMetadata, sparseBlockIdx, sparseBlockCount, cuSeqLengthsQOptional,
           cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads,
           headDim, blockShape, isPackedGQA, layoutQ, layoutKv, maskType, softmaxPrecision, winLeft, winRight, metadata,
           aicCoreNum, aivCoreNum, socVersion);

    static internal::AicpuTaskSpace space("GenericBlockSparseAttentionGradMetadata");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        GenericBlockSparseAttentionGradMetadata,
        OP_ATTR_NAMES({"max_q_seqlen", "max_kv_seqlen", "num_q_heads", "num_kv_heads", "head_dim", "is_packed_gqa",
                       "layout_q", "layout_kv", "mask_type", "softmax_precision", "win_left", "win_right",
                       "aic_core_num", "aiv_core_num", "soc_version"}),
        OP_INPUT(sparseBlockIdx, sparseBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional,
                 sequsedKvOptional, blockShape),
        OP_OUTPUT(metadata),
        OP_ATTR(maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads, headDim, isPackedGQA, layoutQ, layoutKv, maskType,
                softmaxPrecision, winLeft, winRight, aicCoreNum, aivCoreNum, socVersion));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GenericBlockSparseAttentionGradMetadata AICPU launch failed."),
             return nullptr);
    return metadata;
}
} // namespace l0op
