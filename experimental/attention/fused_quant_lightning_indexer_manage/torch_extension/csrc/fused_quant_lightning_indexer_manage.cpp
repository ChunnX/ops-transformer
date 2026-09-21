/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_quant_lightning_indexer_manage.cpp
 * \brief Torch bridge for aclnnFusedQuantLightningIndexerManage.
 */

#include <torch/extension.h>

#include <tuple>

#include "aclnn_common.h"

// Queue-safe launch for this stateful operator. Keep both caller-owned tensors
// and the temporary workspace alive until the queued ACLNN handler is submitted.
#define FQLIM_ACLNN_CMD_ORDERED(aclnn_api, tensor_keepalive, ...) \
    do { \
        auto device = DecodeDevice(__VA_ARGS__); \
        const c10::OptionalDeviceGuard device_guard(device); \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize"); \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api); \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal"); \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal"); \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem"); \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ", \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(), ", or ", \
                    GetTransformerOpApiLibName(), "not found."); \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false); \
        if (c10_npu::check_enqueue_need_use(acl_stream)) { \
            aclrtUseStreamResInCurrentThread(acl_stream); \
        } \
        uint64_t workspace_size = 0; \
        uint64_t *workspace_size_addr = &workspace_size; \
        aclOpExecutor *executor = nullptr; \
        aclOpExecutor **executor_addr = &executor; \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr); \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr); \
        if (initMemFunc) { \
            initMemFunc(nullptr, false); \
        } \
        ApplyDeterministicConfig(); \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr); \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr); \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params); \
        if (workspace_status != 0) { \
            ReleaseConvertTypes(converted_params); \
            TORCH_CHECK(false, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
        } \
        at::Tensor workspace_tensor; \
        void *workspace_addr = nullptr; \
        if (workspace_size != 0) { \
            at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type()); \
            workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte)); \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data()); \
        } \
        auto ordered_tensor_keepalive = tensor_keepalive; \
        auto acl_call = [converted_params, workspace_tensor, workspace_addr, workspace_size, acl_stream, executor, \
                         ordered_tensor_keepalive]() -> int { \
            (void)workspace_tensor; \
            (void)ordered_tensor_keepalive; \
            if (c10_npu::check_enqueue_need_use(acl_stream)) { \
                aclrtUseStreamResInCurrentThread(acl_stream); \
            } \
            typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream); \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr); \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream); \
            ReleaseConvertTypes(converted_params); \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr); \
            if (releaseMemFunc) { \
                releaseMemFunc(nullptr, false); \
            } \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
            return api_ret; \
        }; \
        at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call); \
        if (unInitMemFunc) { \
            unInitMemFunc(nullptr, false); \
        } \
    } while (false)

namespace op_api {
namespace {
constexpr int64_t TOPK = 2048;
constexpr int64_t MISS_CAPACITY = 32768;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t MAX_SOURCE_CAPACITY = 1 << 18;

void CheckBatchVector(const at::Tensor &tensor, int64_t batch, const char *name)
{
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == batch, name, " must be int32 [B].");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32.");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void CheckInputs(const at::Tensor &indexWeights, const at::Tensor &queryDequantScale, const at::Tensor &query,
                 const at::Tensor &indexKeyDequantScale, const at::Tensor &indexKeyCache,
                 const at::Tensor &indexBlockTable, const at::Tensor &actualSeqLengthsQuery,
                 const at::Tensor &actualSeqLengthsKey, const at::Tensor &offloadSeqLengthsKey,
                 const at::Tensor &numCacheTokens, const at::Tensor &requestState, const at::Tensor &reqPoolEntries,
                 const at::Tensor &cacheSlotsPool, const at::Tensor &topkSrcIds, const at::Tensor &topkDstSlots,
                 const at::Tensor &topkMissCounts, const at::Tensor &missSrcIds, const at::Tensor &missDstSlots,
                 const at::Tensor &missCounts)
{
    TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && (query.size(1) == 32 || query.size(1) == 64) &&
                    query.size(2) == 128 && query.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "query must be float8_e4m3fn [T,32|64,128].");
    const int64_t totalQueries = query.size(0);
    const int64_t heads = query.size(1);
    TORCH_CHECK(indexWeights.dim() == 2 && indexWeights.size(0) == totalQueries && indexWeights.size(1) == heads &&
                    indexWeights.scalar_type() == at::kBFloat16 && indexWeights.stride(1) == 1 &&
                    indexWeights.stride(0) >= heads,
                "index_weights must be bfloat16 [T,H] with only T-axis padding stride.");
    TORCH_CHECK(queryDequantScale.dim() == 2 && queryDequantScale.size(0) == totalQueries &&
                    queryDequantScale.size(1) == heads && queryDequantScale.scalar_type() == at::kFloat &&
                    queryDequantScale.is_contiguous(),
                "query_dequant_scale must be contiguous float32 [T,H].");
    TORCH_CHECK(indexKeyCache.dim() == 4 && indexKeyCache.size(0) > 0 && indexKeyCache.size(1) == BLOCK_SIZE &&
                    indexKeyCache.size(2) == 1 && indexKeyCache.size(3) == 128 &&
                    indexKeyCache.scalar_type() == at::ScalarType::Float8_e4m3fn && indexKeyCache.stride(1) == 128 &&
                    indexKeyCache.stride(2) == 128 && indexKeyCache.stride(3) == 1,
                "index_key_cache must be float8_e4m3fn [K,128,1,128] with only block-axis padding stride.");
    TORCH_CHECK(indexKeyDequantScale.dim() == 3 && indexKeyDequantScale.size(0) == indexKeyCache.size(0) &&
                    indexKeyDequantScale.size(1) == BLOCK_SIZE && indexKeyDequantScale.size(2) == 1 &&
                    indexKeyDequantScale.scalar_type() == at::kFloat && indexKeyDequantScale.stride(1) == 1 &&
                    indexKeyDequantScale.stride(2) == 1,
                "index_key_dequant_scale must be float32 [K,128,1] with only block-axis padding stride.");
    TORCH_CHECK(indexBlockTable.dim() == 2 && indexBlockTable.size(0) > 0 && indexBlockTable.size(1) > 0 &&
                    indexBlockTable.size(1) * BLOCK_SIZE <= MAX_SOURCE_CAPACITY &&
                    indexBlockTable.scalar_type() == at::kInt && indexBlockTable.is_contiguous(),
                "index_block_table must be contiguous int32 [B,S/128] with S <= 262144.");

    const int64_t batch = indexBlockTable.size(0);
    const int64_t sourceCapacity = indexBlockTable.size(1) * BLOCK_SIZE;
    CheckBatchVector(actualSeqLengthsQuery, batch, "actual_seq_lengths_query");
    CheckBatchVector(actualSeqLengthsKey, batch, "actual_seq_lengths_key");
    CheckBatchVector(offloadSeqLengthsKey, batch, "offload_seq_lengths_key");
    CheckBatchVector(numCacheTokens, batch, "num_cache_tokens");
    CheckBatchVector(requestState, batch, "request_state");
    CheckBatchVector(reqPoolEntries, batch, "req_pool_entries");
    TORCH_CHECK(totalQueries >= batch && totalQueries <= 4 * batch,
                "the operator supports 1..4 query rows per request.");
    TORCH_CHECK(cacheSlotsPool.dim() == 2 && cacheSlotsPool.size(0) > 0 &&
                    cacheSlotsPool.size(1) == sourceCapacity && cacheSlotsPool.scalar_type() == at::kInt &&
                    cacheSlotsPool.is_contiguous(),
                "cache_slots_pool must be contiguous int32 [POOL_SIZE,S].");
    TORCH_CHECK(topkSrcIds.dim() == 3 && topkSrcIds.size(0) == totalQueries && topkSrcIds.size(1) == 1 &&
                    topkSrcIds.size(2) == TOPK && topkDstSlots.sizes() == topkSrcIds.sizes(),
                "topk outputs must be int32 [T,1,2048].");
    TORCH_CHECK(topkMissCounts.dim() == 1 && topkMissCounts.size(0) == totalQueries,
                "topk_miss_counts must be int32 [T].");
    TORCH_CHECK(missSrcIds.dim() == 2 && missSrcIds.size(0) == batch && missSrcIds.size(1) == MISS_CAPACITY &&
                    missDstSlots.sizes() == missSrcIds.sizes(),
                "miss outputs must be int32 [B,32768].");
    CheckBatchVector(missCounts, batch, "miss_counts");

    const at::Tensor outputTensors[] = {topkSrcIds, topkDstSlots, topkMissCounts, missSrcIds, missDstSlots, missCounts};
    for (const auto &tensor : outputTensors) {
        TORCH_CHECK(tensor.scalar_type() == at::kInt, "all output tensors must be int32.");
        TORCH_CHECK(tensor.is_contiguous(), "all output tensors must be contiguous.");
    }

    const auto device = query.device();
    const at::Tensor tensors[] = {indexWeights,
                                  queryDequantScale,
                                  query,
                                  indexKeyDequantScale,
                                  indexKeyCache,
                                  indexBlockTable,
                                  actualSeqLengthsQuery,
                                  actualSeqLengthsKey,
                                  offloadSeqLengthsKey,
                                  numCacheTokens,
                                  requestState,
                                  reqPoolEntries,
                                  cacheSlotsPool,
                                  topkSrcIds,
                                  topkDstSlots,
                                  topkMissCounts,
                                  missSrcIds,
                                  missDstSlots,
                                  missCounts};
    for (const auto &tensor : tensors) {
        TORCH_CHECK(tensor.device() == device, "all tensors must be on the same NPU device.");
    }
}
} // namespace

void
FusedQuantLightningIndexerManage(const at::Tensor &indexWeights, const at::Tensor &queryDequantScale,
                                 const at::Tensor &query, const at::Tensor &indexKeyDequantScale,
                                 const at::Tensor &indexKeyCache, const at::Tensor &indexBlockTable,
                                 const at::Tensor &actualSeqLengthsQuery, const at::Tensor &actualSeqLengthsKey,
                                 const at::Tensor &offloadSeqLengthsKey, const at::Tensor &numCacheTokens,
                                 const at::Tensor &requestState, const at::Tensor &reqPoolEntries,
                                 at::Tensor &cacheSlotsPool, at::Tensor &topkSrcIds, at::Tensor &topkDstSlots,
                                 at::Tensor &topkMissCounts, at::Tensor &missSrcIds, at::Tensor &missDstSlots,
                                 at::Tensor &missCounts)
{
    CheckInputs(indexWeights, queryDequantScale, query, indexKeyDequantScale, indexKeyCache, indexBlockTable,
                actualSeqLengthsQuery, actualSeqLengthsKey, offloadSeqLengthsKey, numCacheTokens, requestState,
                reqPoolEntries, cacheSlotsPool, topkSrcIds, topkDstSlots, topkMissCounts, missSrcIds, missDstSlots,
                missCounts);

    const c10::OptionalDeviceGuard deviceGuard(query.device());
    const int64_t keyStride = indexKeyCache.stride(0);
    const int64_t scaleStride = indexKeyDequantScale.stride(0);
    const int64_t weightStride = indexWeights.stride(0);
    auto keepalive = std::make_tuple(indexWeights, queryDequantScale, query, indexKeyDequantScale, indexKeyCache,
                                     indexBlockTable, actualSeqLengthsQuery, actualSeqLengthsKey, offloadSeqLengthsKey,
                                     numCacheTokens, requestState, reqPoolEntries, cacheSlotsPool, topkSrcIds,
                                     topkDstSlots, topkMissCounts, missSrcIds, missDstSlots, missCounts);
    FQLIM_ACLNN_CMD_ORDERED(aclnnFusedQuantLightningIndexerManage, keepalive, indexWeights, queryDequantScale, query,
                            indexKeyDequantScale, indexKeyCache, indexBlockTable, actualSeqLengthsQuery,
                            actualSeqLengthsKey, offloadSeqLengthsKey, numCacheTokens, requestState, reqPoolEntries,
                            cacheSlotsPool, keyStride, scaleStride, weightStride, topkSrcIds, topkDstSlots,
                            topkMissCounts, missSrcIds, missDstSlots, missCounts, cacheSlotsPool);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_quant_lightning_indexer_manage", &FusedQuantLightningIndexerManage,
          "fused quant Lightning Indexer cache manager");
}
} // namespace op_api
