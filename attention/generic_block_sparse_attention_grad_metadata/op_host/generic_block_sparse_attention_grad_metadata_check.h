#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_CHECK_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_CHECK_H

#include <cstdint>
#include <cstring>

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"

// L2 validation shared by the metadata ACLNN entry point.  Constraints follow
// docs/aclnnGenericBlockSparseAttentionGrad.md and the metadata output ABI in
// 2_tiling下沉/metadata_abi.md (contiguous int32 metadata buffer whose length is
// a shape/attr-derived upper bound, NOT a fixed [1024]).
class GenericBlockSparseAttentionGradMetadataCheck {
public:
    static constexpr int64_t HEAD_NUM_MAX = 128;
    static constexpr int64_t HEAD_DIM_SUPPORTED = 128;
    static constexpr int64_t BLOCK_SHAPE_X_SUPPORTED = 1;
    static constexpr int64_t BLOCK_SHAPE_Y_MIN = 128;
    static constexpr int64_t BLOCK_SHAPE_Y_ALIGN = 64;
    // metadata 布局常量（与 op_kernel_aicpu 的 metadata_layout.h 逐字对齐）。
    static constexpr int64_t METADATA_HEADER_WORDS = 32;
    static constexpr int64_t METADATA_KTASK_WORDS = 8;
    static constexpr int64_t METADATA_MAX_AIC_CORE_NUM = 64;

    static aclnnStatus ParamsCheck(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                   const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional,
                                   const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                                   int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads,
                                   int64_t headDim, const aclIntArray *blockShape, int64_t isPackedGQA,
                                   const char *layoutQ, const char *layoutKv, int64_t maskType,
                                   int64_t softmaxPrecision, int64_t winLeft, int64_t winRight,
                                   const aclTensor *metadata)
    {
        // ---- nullptr checks ----
        CHECK_COND(sparseBlockIdx != nullptr && sparseBlockCount != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "sparseBlockIdx and sparseBlockCount must not be null");
        CHECK_COND(blockShape != nullptr, ACLNN_ERR_PARAM_INVALID, "blockShape must not be null");
        CHECK_COND(metadata != nullptr, ACLNN_ERR_PARAM_INVALID, "metadata must not be null");
        CHECK_COND(layoutQ != nullptr && layoutKv != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "layoutQ and layoutKv must not be null");

        // ---- layout checks ----
        CHECK_COND(IsSupportedLayout(layoutQ), ACLNN_ERR_PARAM_INVALID,
                   "layoutQ only supports TND, BNSD or BSND, but got %s", layoutQ);
        CHECK_COND(IsSupportedLayout(layoutKv), ACLNN_ERR_PARAM_INVALID,
                   "layoutKv only supports TND, BNSD or BSND, but got %s", layoutKv);

        // ---- scalar attribute checks ----
        CHECK_COND(maxQSeqlen >= 0, ACLNN_ERR_PARAM_INVALID, "maxQSeqlen must be non-negative, but got %ld",
                   maxQSeqlen);
        CHECK_COND(maxKvSeqlen >= 0, ACLNN_ERR_PARAM_INVALID, "maxKvSeqlen must be non-negative, but got %ld",
                   maxKvSeqlen);
        CHECK_COND(numQHeads >= 1 && numQHeads <= HEAD_NUM_MAX, ACLNN_ERR_PARAM_INVALID,
                   "numQHeads must be in [1, %ld], but got %ld", HEAD_NUM_MAX, numQHeads);
        CHECK_COND(numKvHeads >= 1 && numKvHeads <= HEAD_NUM_MAX, ACLNN_ERR_PARAM_INVALID,
                   "numKvHeads must be in [1, %ld], but got %ld", HEAD_NUM_MAX, numKvHeads);
        CHECK_COND(numQHeads % numKvHeads == 0, ACLNN_ERR_PARAM_INVALID,
                   "numQHeads(%ld) must be divisible by numKvHeads(%ld)", numQHeads, numKvHeads);
        CHECK_COND(headDim == HEAD_DIM_SUPPORTED, ACLNN_ERR_PARAM_INVALID, "headDim only supports %ld, but got %ld",
                   HEAD_DIM_SUPPORTED, headDim);
        CHECK_COND(isPackedGQA == 1, ACLNN_ERR_PARAM_INVALID, "isPackedGQA only supports 1, but got %ld", isPackedGQA);
        CHECK_COND(maskType == 1, ACLNN_ERR_PARAM_INVALID, "maskType only supports 1 (causal mask), but got %ld",
                   maskType);
        CHECK_COND(softmaxPrecision == 0 || softmaxPrecision == 1, ACLNN_ERR_PARAM_INVALID,
                   "softmaxPrecision only supports 0 or 1, but got %ld", softmaxPrecision);
        if (softmaxPrecision != 0) {
            OP_LOGW("softmaxPrecision %ld is not fully supported yet, high precision mode 0 is recommended.",
                    softmaxPrecision);
        }
        // Only causal mask is supported currently, so sliding window must stay disabled.
        CHECK_COND(winLeft == -1 && winRight == -1, ACLNN_ERR_PARAM_INVALID,
                   "winLeft and winRight must be -1 when sliding window is not enabled, but got %ld and %ld", winLeft,
                   winRight);

        // ---- blockShape checks ----
        CHECK_COND(blockShape->Size() == 2, ACLNN_ERR_PARAM_INVALID,
                   "blockShape must contain exactly 2 elements, but got %lu", blockShape->Size());
        const int64_t *blockShapeData = blockShape->GetData();
        CHECK_COND(blockShapeData != nullptr, ACLNN_ERR_PARAM_INVALID, "blockShape data must not be null");
        CHECK_COND(blockShapeData[0] == BLOCK_SHAPE_X_SUPPORTED, ACLNN_ERR_PARAM_INVALID,
                   "blockShapeX only supports %ld, but got %ld", BLOCK_SHAPE_X_SUPPORTED, blockShapeData[0]);
        CHECK_COND(blockShapeData[1] >= BLOCK_SHAPE_Y_MIN && blockShapeData[1] % BLOCK_SHAPE_Y_ALIGN == 0,
                   ACLNN_ERR_PARAM_INVALID, "blockShapeY must be 64-aligned and no smaller than %ld, but got %ld",
                   BLOCK_SHAPE_Y_MIN, blockShapeData[1]);

        // ---- sparseBlockIdx / sparseBlockCount checks ----
        // idx:   INT32, rank 4, [B, N2, ceilDiv(maxKvSeqlen, blockShapeY), maxQSeqlen]
        // count: INT32, rank 3, [B, N2, ceilDiv(maxKvSeqlen, blockShapeY)]
        CHECK_COND(sparseBlockIdx->GetDataType() == op::DataType::DT_INT32, ACLNN_ERR_PARAM_INVALID,
                   "sparseBlockIdx must be INT32, but got %s", op::ToString(sparseBlockIdx->GetDataType()).GetString());
        CHECK_COND(sparseBlockCount->GetDataType() == op::DataType::DT_INT32, ACLNN_ERR_PARAM_INVALID,
                   "sparseBlockCount must be INT32, but got %s",
                   op::ToString(sparseBlockCount->GetDataType()).GetString());
        const int64_t idxDimNum = static_cast<int64_t>(sparseBlockIdx->GetViewShape().GetDimNum());
        const int64_t countDimNum = static_cast<int64_t>(sparseBlockCount->GetViewShape().GetDimNum());
        CHECK_COND(idxDimNum == 4, ACLNN_ERR_PARAM_INVALID, "the dim num of sparseBlockIdx must be 4, but got %ld",
                   idxDimNum);
        CHECK_COND(countDimNum == 3, ACLNN_ERR_PARAM_INVALID, "the dim num of sparseBlockCount must be 3, but got %ld",
                   countDimNum);

        const int64_t batchSize = sparseBlockIdx->GetViewShape().GetDim(0);
        const int64_t kvHeadNum = sparseBlockIdx->GetViewShape().GetDim(1);
        const int64_t kvBlockNum = CeilDiv(maxKvSeqlen, blockShapeData[1]);
        CHECK_COND(batchSize > 0, ACLNN_ERR_PARAM_INVALID,
                   "the batch dim of sparseBlockIdx must be positive, but got %ld", batchSize);
        CHECK_COND(kvHeadNum == numKvHeads, ACLNN_ERR_PARAM_INVALID,
                   "the head dim of sparseBlockIdx must equal numKvHeads(%ld), but got %ld", numKvHeads, kvHeadNum);
        CHECK_COND(sparseBlockIdx->GetViewShape().GetDim(2) == kvBlockNum, ACLNN_ERR_PARAM_INVALID,
                   "the kv block dim of sparseBlockIdx must be ceilDiv(maxKvSeqlen, blockShapeY) = %ld, but got %ld",
                   kvBlockNum, sparseBlockIdx->GetViewShape().GetDim(2));
        CHECK_COND(sparseBlockIdx->GetViewShape().GetDim(3) == maxQSeqlen, ACLNN_ERR_PARAM_INVALID,
                   "the last dim of sparseBlockIdx must be maxQSeqlen(%ld), but got %ld", maxQSeqlen,
                   sparseBlockIdx->GetViewShape().GetDim(3));
        CHECK_COND(sparseBlockCount->GetViewShape().GetDim(0) == batchSize &&
                       sparseBlockCount->GetViewShape().GetDim(1) == numKvHeads &&
                       sparseBlockCount->GetViewShape().GetDim(2) == kvBlockNum,
                   ACLNN_ERR_PARAM_INVALID,
                   "sparseBlockCount must have shape [B, numKvHeads, kvBlockNum] = [%ld, %ld, %ld], "
                   "but got [%ld, %ld, %ld]",
                   batchSize, numKvHeads, kvBlockNum, sparseBlockCount->GetViewShape().GetDim(0),
                   sparseBlockCount->GetViewShape().GetDim(1), sparseBlockCount->GetViewShape().GetDim(2));

        // ---- cuSeqLengths / seqused optional tensor checks ----
        if (std::strcmp(layoutQ, "TND") == 0) {
            CHECK_COND(cuSeqLengthsQOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                       "cuSeqLengthsQOptional is required when layoutQ is TND");
        } else if (cuSeqLengthsQOptional != nullptr) {
            OP_LOGW("cuSeqLengthsQOptional is ignored for layoutQ %s, it is only consumed by TND layout", layoutQ);
        }
        if (std::strcmp(layoutKv, "TND") == 0) {
            CHECK_COND(cuSeqLengthsKvOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                       "cuSeqLengthsKvOptional is required when layoutKv is TND");
        } else if (cuSeqLengthsKvOptional != nullptr) {
            OP_LOGW("cuSeqLengthsKvOptional is ignored for layoutKv %s, it is only consumed by TND layout", layoutKv);
        }
        CHECK_COND(CheckCuSeqLengths(cuSeqLengthsQOptional, batchSize, "cuSeqLengthsQOptional") == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "cuSeqLengthsQOptional check failed");
        CHECK_COND(CheckCuSeqLengths(cuSeqLengthsKvOptional, batchSize, "cuSeqLengthsKvOptional") == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "cuSeqLengthsKvOptional check failed");
        CHECK_COND(CheckSeqUsed(sequsedQOptional, batchSize, "sequsedQOptional") == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "sequsedQOptional check failed");
        CHECK_COND(CheckSeqUsed(sequsedKvOptional, batchSize, "sequsedKvOptional") == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "sequsedKvOptional check failed");

        // ---- metadata output checks: rank-1 INT64, fixed capacity for max AIC cores ----
        CHECK_COND(metadata->GetDataType() == op::DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
                   "metadata must be INT64, but got %s", op::ToString(metadata->GetDataType()).GetString());
        CHECK_COND(static_cast<int64_t>(metadata->GetViewShape().GetDimNum()) == 1, ACLNN_ERR_PARAM_INVALID,
                   "the dim num of metadata must be 1, but got %ld",
                   static_cast<int64_t>(metadata->GetViewShape().GetDimNum()));
        const int64_t metadataCapacity = METADATA_HEADER_WORDS + METADATA_MAX_AIC_CORE_NUM * METADATA_KTASK_WORDS;
        CHECK_COND(metadata->GetViewShape().GetDim(0) == metadataCapacity, ACLNN_ERR_PARAM_INVALID,
                   "metadata length must be fixed capacity %ld words, but got %ld", metadataCapacity,
                   metadata->GetViewShape().GetDim(0));
        return ACLNN_SUCCESS;
    }

private:
    static bool IsSupportedLayout(const char *layout)
    {
        return std::strcmp(layout, "TND") == 0 || std::strcmp(layout, "BNSD") == 0 || std::strcmp(layout, "BSND") == 0;
    }

    static int64_t CeilDiv(int64_t value, int64_t divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    // cuSeqLengths*: INT64, rank 1, [batchSize + 1] (prefix sums).
    static aclnnStatus CheckCuSeqLengths(const aclTensor *tensor, int64_t batchSize, const char *name)
    {
        if (tensor == nullptr) {
            return ACLNN_SUCCESS;
        }
        CHECK_COND(tensor->GetDataType() == op::DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
                   "%s must be INT64, but got %s", name, op::ToString(tensor->GetDataType()).GetString());
        CHECK_COND(static_cast<int64_t>(tensor->GetViewShape().GetDimNum()) == 1, ACLNN_ERR_PARAM_INVALID,
                   "the dim num of %s must be 1, but got %ld", name,
                   static_cast<int64_t>(tensor->GetViewShape().GetDimNum()));
        CHECK_COND(tensor->GetViewShape().GetDim(0) == batchSize + 1, ACLNN_ERR_PARAM_INVALID,
                   "the length of %s must be batchSize + 1 = %ld, but got %ld", name, batchSize + 1,
                   tensor->GetViewShape().GetDim(0));
        return ACLNN_SUCCESS;
    }

    // seqused*: INT32, rank 1, [batchSize].
    static aclnnStatus CheckSeqUsed(const aclTensor *tensor, int64_t batchSize, const char *name)
    {
        if (tensor == nullptr) {
            return ACLNN_SUCCESS;
        }
        CHECK_COND(tensor->GetDataType() == op::DataType::DT_INT32, ACLNN_ERR_PARAM_INVALID,
                   "%s must be INT32, but got %s", name, op::ToString(tensor->GetDataType()).GetString());
        CHECK_COND(static_cast<int64_t>(tensor->GetViewShape().GetDimNum()) == 1, ACLNN_ERR_PARAM_INVALID,
                   "the dim num of %s must be 1, but got %ld", name,
                   static_cast<int64_t>(tensor->GetViewShape().GetDimNum()));
        CHECK_COND(tensor->GetViewShape().GetDim(0) == batchSize, ACLNN_ERR_PARAM_INVALID,
                   "the length of %s must be batchSize = %ld, but got %ld", name, batchSize,
                   tensor->GetViewShape().GetDim(0));
        return ACLNN_SUCCESS;
    }
};

#endif
