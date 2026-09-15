/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"

#include <limits>

#include "../../flash_mla/op_host/flash_mla_sequence_policy.h"
#include "../op_host/flash_mla_metadata_config.h"
#include "acl/acl_rt.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

namespace FlashMla {
namespace {
bool IsMlaDeviceTensor(const aclTensor *tensor)
{
    return tensor != nullptr && gert::TensorPlacementUtils::IsOnDevice(tensor->GetPlacement());
}

aclnnStatus CheckMlaMetadataTensor(const aclTensor *tensor)
{
    OP_CHECK_NULL(tensor, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_DTYPE_NOT_MATCH(tensor, op::DataType::DT_INT32, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(tensor->GetViewFormat() == op::Format::FORMAT_ND && tensor->GetStorageFormat() == op::Format::FORMAT_ND &&
                 tensor->GetViewShape() == op::Shape{kMlaMetadataElements} && tensor->GetStorageOffset() == 0 &&
                 tensor->GetViewOffset() == 0 && op::IsContiguous(tensor) && IsMlaDeviceTensor(tensor) &&
                 tensor->GetData() != nullptr &&
                 (reinterpret_cast<uintptr_t>(tensor->GetData()) & (kMlaMetadataAlignment - 1U)) == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "metadata must be a 64-byte aligned device INT32[480] ND tensor with zero offset."),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMlaSeqTensor(const aclTensor *tensor, int64_t batchSize, bool exactBatchCount, const char *name,
                              bool checkBatchCount = true)
{
    if (tensor == nullptr) {
        if (exactBatchCount && batchSize != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s is required for TND.", name);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
    OP_CHECK(tensor->GetDataType() == op::DataType::DT_INT64 && tensor->GetViewFormat() == op::Format::FORMAT_ND &&
                 tensor->GetViewShape().GetDimNum() == 1 && op::IsContiguous(tensor) && IsMlaDeviceTensor(tensor),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be a contiguous one-dimensional device INT64 ND tensor.", name),
             return ACLNN_ERR_PARAM_INVALID);
    const int64_t count = tensor->GetViewShape().GetDim(0);
    OP_CHECK(count >= 0 && (exactBatchCount ? count == batchSize :
                                              (!checkBatchCount || count == 0 || count == 1 || count >= batchSize)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s has descriptor length %ld, incompatible with batchSize %ld.", name,
                     count, batchSize),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

struct MlaApiPlatform {
    uint32_t aic = 0;
    uint32_t aiv = 0;
};

bool IsMlaSupportedSocVersion(op::SocVersion soc)
{
    return soc == op::SocVersion::ASCEND910B || soc == op::SocVersion::ASCEND910_93;
}

aclnnStatus GetMlaApiPlatform(MlaApiPlatform &out)
{
    const auto &current = op::GetCurrentPlatformInfo();
    OP_CHECK(IsMlaSupportedSocVersion(current.GetSocVersion()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashMla only supports Ascend910B and Ascend910_93."),
             return ACLNN_ERR_PARAM_INVALID);
    auto infos = current.GetPlatformInfos();
    OP_CHECK(infos != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FlashMla platform information is null."),
             return ACLNN_ERR_INNER_NULLPTR);
    out = MlaApiPlatform{};
    // Match the ACLNN phase-1 context: stream resources bound to this thread
    // override the physical core counts cached in PlatformInfo.
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &out.aic) != ACL_SUCCESS) {
        out.aic = current.GetCubeCoreNum();
    }
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &out.aiv) != ACL_SUCCESS) {
        out.aiv = current.GetVectorCoreNum();
    }
    OP_CHECK(out.aic != 0 && out.aiv != 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashMla requires nonzero AIC/AIV core counts."),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// Called after CheckMlaSeqTensor. AICPU serializes OriginalShape, which must
// describe the logical sequence length even for a view of a larger storage.
const aclTensor *NormalizeMlaSeqTensor(const aclTensor *input, aclOpExecutor *executor)
{
    if (input == nullptr) {
        return nullptr;
    }
    const auto &shape = input->GetViewShape();
    if (input->GetStorageShape() == shape && input->GetOriginalShape() == shape &&
        input->GetStorageFormat() == op::Format::FORMAT_ND && input->GetOriginalFormat() == op::Format::FORMAT_ND) {
        return input;
    }
    // CreateView shares storage and preserves its offset/address binding. It
    // creates only a descriptor; no sequence values are read or copied.
    auto view = executor->CreateView(input, shape, input->GetViewOffset());
    if (view != nullptr) {
        view->SetStorageFormat(op::Format::FORMAT_ND);
        view->SetOriginalFormat(op::Format::FORMAT_ND);
        view->SetViewFormat(op::Format::FORMAT_ND);
    }
    return view;
}
} // namespace
} // namespace FlashMla

namespace FlashMlaMetadata {
namespace {
OP_TYPE_REGISTER(FlashMlaMetadata);

bool ResolveBatchSize(const aclTensor *actualSeqLengthsQ, const char *inputLayer, int64_t &batchSize)
{
    if (batchSize != FlashMla::kMlaInferDimension) {
        return true;
    }
    std::string queryLayout, outputLayout;
    OP_CHECK(FlashMla::SplitMlaLayout(inputLayer, queryLayout, outputLayout) && queryLayout == "TND" &&
                 actualSeqLengthsQ != nullptr && actualSeqLengthsQ->GetViewShape().GetDimNum() == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Omitted batchSize requires TND and a one-dimensional Query sequence tensor."), return false);
    // Only the descriptor is read here; device sequence values stay on AICPU.
    batchSize = actualSeqLengthsQ->GetViewShape().GetDim(0);
    return true;
}

bool CheckAttrs(int64_t numHeads, int64_t batchSize, int64_t qSeqSize, int64_t kvSeqSize, const char *inputLayer,
                int64_t blockSize, int64_t sparseMode, int64_t quantMode, FlashMla::MlaScheduleConfig &config)
{
    OP_CHECK(numHeads > 0 && numHeads <= 128,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numHeads must be in [1, 128], but got %ld.", numHeads), return false);
    OP_CHECK(batchSize >= 0 && batchSize <= 65536,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Logical batchSize must be in [0, 65536], but got %ld.", batchSize),
             return false);
    OP_CHECK(blockSize >= 0 && blockSize <= std::numeric_limits<int32_t>::max(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockSize must be in [0, INT32_MAX], but got %ld.", blockSize),
             return false);
    OP_CHECK(quantMode == FlashMla::kMlaNoQuant || quantMode == FlashMla::kMlaInt8FullQuant,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode must be 0 (nonquant) or 1 (INT8 full quantization)."),
             return false);
    config.blockSize = blockSize;
    OP_CHECK(FlashMla::ConfigureMlaMetadata(inputLayer, quantMode, config),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported inputLayer: %s.", inputLayer), return false);
    const bool canInferTotal = config.layoutQ == FlashMla::MlaLayout::TND && batchSize > 0;
    OP_CHECK((qSeqSize >= 0 && qSeqSize <= std::numeric_limits<uint32_t>::max()) ||
                 (qSeqSize == FlashMla::kMlaInferDimension && canInferTotal),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "qSeqSize must be in [0, UINT32_MAX]; omission requires TND with a nonempty batch."), return false);
    OP_CHECK((kvSeqSize >= 0 && kvSeqSize <= std::numeric_limits<int32_t>::max()) ||
                 (kvSeqSize == FlashMla::kMlaInferDimension && canInferTotal && blockSize == 0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "kvSeqSize must be in [0, INT32_MAX]; omission requires dense TND with a nonempty batch."),
             return false);
    OP_CHECK((sparseMode >= 0 && sparseMode <= 4) || sparseMode == 9,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparseMode must be 0, 1, 2, 3, 4 or 9, but got %ld.", sparseMode),
             return false);
    // Metadata is independently callable and must establish the original
    // split-RoPE tiler's Q/T bounds before its sequence-dependent arithmetic.
    OP_CHECK(quantMode != FlashMla::kMlaInt8FullQuant || qSeqSize <= (config.layoutQ == FlashMla::MlaLayout::TND ? 1024 * 1024 : 32),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "INT8 qSeqSize exceeds the original MLA layout limit."), return false);
    OP_CHECK(quantMode != FlashMla::kMlaInt8FullQuant || config.layoutQ != FlashMla::MlaLayout::BNSD,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "INT8 query layout must be BSH, BSND or TND."), return false);
    return true;
}

bool CheckPagedAttention(const aclTensor *actualSeqLengthsKvOptional, int64_t numHeads, int64_t batchSize,
                         FlashMla::MlaLayout layoutQ, int64_t blockSize, int64_t quantMode)
{
    OP_CHECK(blockSize == 0 || (blockSize <= 1024 && blockSize % 16 == 0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Nonzero blockSize must be a multiple of 16 no greater than 1024."),
             return false);
    if (quantMode == FlashMla::kMlaInt8FullQuant) {
        OP_CHECK(
            blockSize == 128 && (numHeads & (numHeads - 1)) == 0,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "INT8 requires blockSize 128 and a power-of-two numHeads; the consumer validates NZ KV."),
            return false);
    }
    if ((blockSize != 0 || layoutQ == FlashMla::MlaLayout::TND) && batchSize != 0) {
        OP_CHECK(actualSeqLengthsKvOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "actualSeqLengthsKvOptional is required for paged attention or TND query."),
                 return false);
    }
    return true;
}

aclnnStatus CheckParams(const aclTensor *actualSeqLengthsQOptional, const aclTensor *actualSeqLengthsKvOptional,
                        int64_t numHeads, int64_t &batchSize, int64_t qSeqSize, int64_t kvSeqSize, const char *inputLayer,
                        int64_t blockSize, int64_t sparseMode, int64_t preTokens,
                        int64_t nextTokens, bool hasAttenMask, int64_t quantMode, const aclTensor *metadataOut)
{
    OP_CHECK_NULL(inputLayer, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(ResolveBatchSize(actualSeqLengthsQOptional, inputLayer, batchSize), ACLNN_ERR_PARAM_INVALID);
    FlashMla::MlaScheduleConfig tokenConfig;
    CHECK_RET(
        CheckAttrs(numHeads, batchSize, qSeqSize, kvSeqSize, inputLayer, blockSize, sparseMode, quantMode, tokenConfig),
        ACLNN_ERR_PARAM_INVALID);
    // Only TND may defer qSeqSize; its sparse checks do not use the stored total.
    // AICPU resolves and validates the actual total before computing the schedule.
    tokenConfig.qSeqSize = qSeqSize == FlashMla::kMlaInferDimension ? 0 : qSeqSize;
    tokenConfig.sparseMode = sparseMode;
    tokenConfig.preTokens = preTokens;
    tokenConfig.nextTokens = nextTokens;
    tokenConfig.hasAttenMask = hasAttenMask;
    OP_CHECK(FlashMla::NormalizeMlaScheduleConfig(tokenConfig) == FlashMla::MlaStatus::SUCCESS,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid preTokens/nextTokens for the effective sparse mode."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(FlashMla::ValidateMlaSparseConfig(tokenConfig) == FlashMla::MlaStatus::SUCCESS,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported sparse mode/mask/query length combination for MLA."),
             return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckPagedAttention(actualSeqLengthsKvOptional, numHeads, batchSize, tokenConfig.layoutQ, blockSize,
                                  quantMode),
              ACLNN_ERR_PARAM_INVALID);
    auto ret = FlashMla::CheckMlaMetadataTensor(metadataOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = FlashMla::CheckMlaSeqTensor(
        actualSeqLengthsQOptional, batchSize, tokenConfig.layoutQ == FlashMla::MlaLayout::TND, "seq_q",
        !FlashMla::IsMlaQuerySeqIgnored(tokenConfig.queryDataType, tokenConfig.layoutQ == FlashMla::MlaLayout::TND));
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // TND queries require exactly B KV lengths even for PA, whose KV values
    // remain absolute rather than cumulative.
    ret = FlashMla::CheckMlaSeqTensor(actualSeqLengthsKvOptional, batchSize,
                                      tokenConfig.layoutQ == FlashMla::MlaLayout::TND, "seq_kv");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}
} // namespace
} // namespace FlashMlaMetadata

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnInnerFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *actualSeqLengthsQOptional, const aclTensor *actualSeqLengthsKvOptional, int64_t numHeads,
    int64_t batchSize, int64_t qSeqSize, int64_t kvSeqSize, const char *inputLayer,
    int64_t blockSize, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, bool hasAttenMask,
    int64_t quantMode, const aclTensor *metadataOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(
        aclnnInnerFlashMlaMetadata,
        DFX_IN(actualSeqLengthsQOptional, actualSeqLengthsKvOptional, numHeads, batchSize, qSeqSize, kvSeqSize, inputLayer,
               blockSize, sparseMode, preTokens, nextTokens, hasAttenMask, quantMode),
        DFX_OUT(metadataOut));

    auto ret = FlashMlaMetadata::CheckParams(actualSeqLengthsQOptional, actualSeqLengthsKvOptional, numHeads, batchSize,
                                             qSeqSize, kvSeqSize, inputLayer, blockSize, sparseMode, preTokens,
                                             nextTokens, hasAttenMask, quantMode, metadataOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    FlashMla::MlaApiPlatform platform;
    ret = FlashMla::GetMlaApiPlatform(platform);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // The AICPU launcher leaves null optional slots' extended shape bytes
    // unspecified. An explicit INT64[0] descriptor preserves absent-seq
    // semantics without reading or copying any device sequence values.
    const auto seqQ = actualSeqLengthsQOptional == nullptr ?
                          uniqueExecutor->AllocTensor(op::Shape{0}, op::DataType::DT_INT64, op::Format::FORMAT_ND) :
                          FlashMla::NormalizeMlaSeqTensor(actualSeqLengthsQOptional, uniqueExecutor.get());
    CHECK_RET(seqQ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const auto seqKv = actualSeqLengthsKvOptional == nullptr ?
                           uniqueExecutor->AllocTensor(op::Shape{0}, op::DataType::DT_INT64, op::Format::FORMAT_ND) :
                           FlashMla::NormalizeMlaSeqTensor(actualSeqLengthsKvOptional, uniqueExecutor.get());
    CHECK_RET(seqKv != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // The AICPU task reads device seq and writes the caller's aligned metadata
    // directly. Even an empty batch needs a valid metadata header.
    static op::internal::AicpuTaskSpace space("FlashMlaMetadata");
    auto opArgs = op::GetOpArgContext(
        OP_INPUT(seqQ, seqKv), OP_OUTPUT(metadataOut),
        OP_ATTR(numHeads, batchSize, qSeqSize, kvSeqSize, inputLayer, blockSize, sparseMode, preTokens, nextTokens,
                hasAttenMask, quantMode, static_cast<int64_t>(platform.aic), static_cast<int64_t>(platform.aiv)));
    ret = op::internal::CreatAicpuKernelLauncher(
        FlashMlaMetadata::FlashMlaMetadataOpTypeId(), space, uniqueExecutor.get(),
        OP_ATTR_NAMES({"num_heads", "batch_size", "q_seq_size", "kv_seq_size", "input_layer", "block_size",
                       "sparse_mode", "pre_tokens", "next_tokens", "has_atten_mask", "quant_mode", "aic_core_num",
                       "aiv_core_num"}),
        opArgs);
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to create FlashMlaMetadata AICPU launcher."),
             return ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnInnerFlashMlaMetadata(void *workspace, uint64_t workspaceSize,
                                                                              aclOpExecutor *executor,
                                                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInnerFlashMlaMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
