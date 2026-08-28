/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#include "flash_kda_l0.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "aclnn_flash_kda.h"
#include "opdev/framework_op.h"
#include "cannbotdsl_provider_adapter.h"
#include "cannbotdsl_provider_registry.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/provider_kernel.h"
#include "opdev/tensor_view_utils.h"

namespace {

constexpr int64_t kHeadDim = 128;

static_assert(op::PROVIDER_KERNEL_SPEC_ABI_VERSION == 1U,
              "FlashKDA requires opbase Provider kernel ABI v1");
static_assert(op::PROVIDER_ACLNN_REGISTRY_ABI_VERSION == 1U,
              "FlashKDA requires opbase Provider registry ABI v1");

const op::ProviderAclnnRegistrySpec kFlashKdaRegistry = {
    sizeof(op::ProviderAclnnRegistrySpec),
    op::PROVIDER_ACLNN_REGISTRY_ABI_VERSION,
    nullptr,
    cannbotdsl::generated::kProviderAclnnVariants,
    "../../op_impl/cannbotdsl",
    "ops-transformer::attention/flash_kda",
    cannbotdsl::generated::kProviderAclnnVariants,
    cannbotdsl::generated::kProviderAclnnVariantCount,
};

const op::ProviderKernelSpec kFlashKdaProviderSpec =
    op::MakeProviderKernelSpec(&kFlashKdaRegistry);

int64_t Dim(const aclTensor* tensor, size_t axis)
{
    return tensor->GetViewShape().GetDim(axis);
}

bool HasShape(const aclTensor* tensor, std::initializer_list<int64_t> expected)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() != expected.size()) {
        return false;
    }
    size_t axis = 0U;
    for (const int64_t dimension : expected) {
        if (Dim(tensor, axis++) != dimension) {
            return false;
        }
    }
    return true;
}

bool CheckTensorContract(
    const aclTensor* k, const aclTensor* v, const aclTensor* q,
    const aclTensor* beta, const aclTensor* g, const aclTensor* aLog,
    const aclTensor* dtBias, const aclTensor* initialState,
    const aclTensor* out, const aclTensor* finalState)
{
    if (k == nullptr || v == nullptr || q == nullptr || beta == nullptr ||
        g == nullptr || aLog == nullptr || dtBias == nullptr ||
        initialState == nullptr || out == nullptr || finalState == nullptr) {
        return false;
    }
    if (v->GetViewShape().GetDimNum() != 4U) {
        return false;
    }
    const int64_t batch = Dim(v, 0U);
    const int64_t heads = Dim(v, 1U);
    const int64_t sequence = Dim(v, 2U);
    if (batch <= 0 || heads <= 0 || sequence <= 0 ||
        sequence % 64 != 0 || Dim(v, 3U) != kHeadDim) {
        return false;
    }
    if (!HasShape(k, {batch, heads, sequence, kHeadDim}) ||
        !HasShape(q, {batch, heads, sequence, kHeadDim}) ||
        !HasShape(beta, {batch, heads, sequence, 1}) ||
        !HasShape(g, {batch, heads, sequence, kHeadDim}) ||
        !HasShape(aLog, {heads}) || !HasShape(dtBias, {heads, kHeadDim}) ||
        !HasShape(initialState, {batch, heads, kHeadDim, kHeadDim}) ||
        !HasShape(out, {batch, heads, sequence, kHeadDim}) ||
        !HasShape(finalState, {batch, heads, kHeadDim, kHeadDim})) {
        return false;
    }
    for (const aclTensor* tensor : {k, v, q, beta, g, out}) {
        if (tensor->GetDataType() != op::DataType::DT_BF16 ||
            !op::IsContiguous(tensor)) {
            return false;
        }
    }
    for (const aclTensor* tensor : {aLog, dtBias, initialState, finalState}) {
        if (tensor->GetDataType() != op::DataType::DT_FLOAT ||
            !op::IsContiguous(tensor)) {
            return false;
        }
    }
    return true;
}

}  // namespace

namespace l0op {

OP_TYPE_REGISTER(FlashKdaProvider);

aclnnStatus FlashKda(
    const aclTensor* k, const aclTensor* v, const aclTensor* q,
    const aclTensor* beta, const aclTensor* g, const aclTensor* aLog,
    const aclTensor* dtBias, const aclTensor* initialState, float scale,
    float lowerBound, const aclTensor* out, const aclTensor* finalState,
    aclOpExecutor* executor)
{
    L0_DFX(FlashKda, k, v, q, beta, g, aLog, dtBias, initialState,
           scale, lowerBound, out, finalState);
    if (executor == nullptr ||
        !CheckTensorContract(k, v, q, beta, g, aLog, dtBias, initialState,
                             out, finalState) ||
        !std::isfinite(scale) || !std::isfinite(lowerBound) ||
        lowerBound < -5.0F || lowerBound > 0.0F) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t batch = Dim(v, 0U);
    const int64_t heads = Dim(v, 1U);
    if (batch <= 0 || heads <= 0 ||
        batch > std::numeric_limits<int64_t>::max() / heads) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    cannbotdsl::generated::FlashKdaEntryArguments entryArguments;
    entryArguments.K = k;
    entryArguments.V = v;
    entryArguments.Q = q;
    entryArguments.beta = beta;
    entryArguments.g = g;
    entryArguments.A_log = aLog;
    entryArguments.dt_bias = dtBias;
    entryArguments.st_in = initialState;
    entryArguments.scale = scale;
    entryArguments.lower_bound = lowerBound;
    entryArguments.o = out;
    entryArguments.st_out = finalState;
    cannbotdsl::generated::FlashKdaProviderArguments providerArguments;
    aclnnStatus status =
        cannbotdsl::generated::PrepareFlashKdaProviderArguments(
            executor, entryArguments, providerArguments);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    op::OpArgContext* args =
        cannbotdsl::generated::MakeFlashKdaProviderOpArgContext(providerArguments);
    if (args == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    status = cannbotdsl::generated::SetFlashKdaProviderWorkspaceTensors(
        executor, args, providerArguments);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return op::AddProviderKernelLauncher(
        "FlashKdaProvider", FlashKdaProviderOpTypeId(),
        &kFlashKdaProviderSpec, executor, args);
}

}  // namespace l0op

extern "C" aclnnStatus aclnnFlashKdaGetWorkspaceSize(
    const aclTensor* k, const aclTensor* v, const aclTensor* q,
    const aclTensor* beta, const aclTensor* g, const aclTensor* aLog,
    const aclTensor* dtBias, const aclTensor* initialState, double scale,
    double lowerBound, const aclTensor* out, const aclTensor* finalState,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (workspaceSize != nullptr) {
        *workspaceSize = 0U;
    }
    if (executor != nullptr) {
        *executor = nullptr;
    }
    if (workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(scale) || !std::isfinite(lowerBound) ||
        lowerBound < -5.0 || lowerBound > 0.0 ||
        scale > std::numeric_limits<float>::max() ||
        scale < -std::numeric_limits<float>::max() ||
        lowerBound > std::numeric_limits<float>::max() ||
        lowerBound < -std::numeric_limits<float>::max()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    const aclnnStatus status = l0op::FlashKda(
        k, v, q, beta, g, aLog, dtBias, initialState,
        static_cast<float>(scale), static_cast<float>(lowerBound), out,
        finalState, uniqueExecutor.get());
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnFlashKda(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
