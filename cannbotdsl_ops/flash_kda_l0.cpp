/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#include "flash_kda_l0.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "aclnn_flash_kda.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "cannbotdsl_provider_registry.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/provider_kernel.h"
#include "opdev/tensor_view_utils.h"

namespace {

constexpr int64_t kChunkSize = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kHalfHeadDim = 64;
constexpr int64_t kBlockNum = 32;
constexpr uint64_t kWorkspaceAlignment = 512U;

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

bool IsSupportedVariant(int64_t batch, int64_t heads, int64_t sequence)
{
    if (sequence <= 0 || sequence % kChunkSize != 0) {
        return false;
    }
    if (batch == 1 && heads == 3) {
        return sequence == 8192 || (sequence >= 32768 && sequence <= 131072);
    }
    if (batch == 4 && heads == 3) {
        return sequence >= 8192 && sequence <= 131072;
    }
    if (heads != 96) {
        return false;
    }
    if (batch == 1 || batch == 4) {
        return sequence >= 1024 && sequence <= 65536;
    }
    return batch == 2 && sequence >= 10240 && sequence <= 65536;
}

int64_t ResolveGroup(int64_t bnTotal, int64_t sequence)
{
    int64_t group = 1;
    if (bnTotal <= 1) {
        group = 512;
    } else if (bnTotal <= 3) {
        group = 256;
    } else if (bnTotal <= 6) {
        group = 128;
    } else if (bnTotal <= 13) {
        group = 64;
    } else if (bnTotal <= 26) {
        group = 32;
    } else if (bnTotal <= 53) {
        group = 16;
    } else if (bnTotal <= 107) {
        group = 8;
    } else if (bnTotal <= 215) {
        group = 4;
    } else if (bnTotal <= 430) {
        group = 2;
    }
    const int64_t chunks = sequence / kChunkSize;
    group = group < chunks ? group : chunks;
    while (bnTotal * group < kBlockNum && group < chunks) {
        group *= 2;
    }
    return group > 0 ? group : 1;
}

bool CheckedMultiply(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs <= 0 || rhs <= 0 ||
        lhs > std::numeric_limits<int64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

op::Shape MakeShape(std::initializer_list<int64_t> dimensions)
{
    op::Shape shape;
    for (const int64_t dimension : dimensions) {
        shape.AppendDim(dimension);
    }
    return shape;
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
    if (!IsSupportedVariant(batch, heads, sequence) || Dim(v, 3U) != kHeadDim) {
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

const aclTensor* BuildIdentity16(aclOpExecutor* executor)
{
    std::array<op::fp16_t, 16U * 16U> values{};
    for (size_t index = 0U; index < 16U; ++index) {
        values[index * 16U + index] = op::fp16_t(1.0);
    }
    const aclTensor* host = executor->ConvertToTensor(
        values.data(), values.size(), op::DataType::DT_FLOAT16);
    if (host == nullptr) {
        return nullptr;
    }
    const aclTensor* device = l0op::Contiguous(host, executor);
    if (device == nullptr) {
        return nullptr;
    }
    return l0op::Reshape(device, MakeShape({16, 16}), executor);
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
        !std::isfinite(scale) || !std::isfinite(lowerBound)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t batch = Dim(v, 0U);
    const int64_t heads = Dim(v, 1U);
    const int64_t sequence = Dim(v, 2U);
    int64_t bnTotal = 0;
    if (!CheckedMultiply(batch, heads, bnTotal)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t group = ResolveGroup(bnTotal, sequence);
    int64_t workspaceTokens = 0;
    int64_t groupTokens = 0;
    if (!CheckedMultiply(bnTotal, group, groupTokens) ||
        !CheckedMultiply(groupTokens, kChunkSize, workspaceTokens)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    aclTensor* qkExchange = op::AllocProviderWorkspaceTensor(
        executor, MakeShape({kBlockNum, 2, kChunkSize, kHalfHeadDim}),
        op::DataType::DT_FLOAT, kWorkspaceAlignment);
    aclTensor* kRestored = op::AllocProviderWorkspaceTensor(
        executor, MakeShape({workspaceTokens, kHeadDim}),
        op::DataType::DT_BF16, kWorkspaceAlignment);
    aclTensor* gamma = op::AllocProviderWorkspaceTensor(
        executor, MakeShape({groupTokens * kHeadDim, 1}),
        op::DataType::DT_FLOAT, kWorkspaceAlignment);
    aclTensor* qDecayed = op::AllocProviderWorkspaceTensor(
        executor, MakeShape({workspaceTokens, kHeadDim}),
        op::DataType::DT_BF16, kWorkspaceAlignment);
    aclTensor* w = op::AllocProviderWorkspaceTensor(
        executor, MakeShape({workspaceTokens, kHeadDim}),
        op::DataType::DT_BF16, kWorkspaceAlignment);
    aclTensor* mqk = op::AllocProviderWorkspaceTensor(
        executor, MakeShape({workspaceTokens, kChunkSize}),
        op::DataType::DT_BF16, kWorkspaceAlignment);
    aclTensor* u = op::AllocProviderWorkspaceTensor(
        executor, MakeShape({workspaceTokens, kHeadDim}),
        op::DataType::DT_BF16, kWorkspaceAlignment);
    if (qkExchange == nullptr || kRestored == nullptr || gamma == nullptr ||
        qDecayed == nullptr || w == nullptr || mqk == nullptr || u == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclTensor* identity = BuildIdentity16(executor);
    if (identity == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    op::OpArgContext* args = op::GetOpArgContext(
        OP_INPUT(k, v, q, beta, g, qkExchange, aLog, dtBias, kRestored,
                 gamma, qDecayed, w, mqk, u, initialState, identity),
        OP_OUTPUT(out, finalState),
        OP_ATTR(bnTotal, sequence, scale, lowerBound));
    if (args == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclTensor* workspaceTensors[] = {
        qkExchange, kRestored, gamma, qDecayed, w, mqk, u,
    };
    aclnnStatus status = op::SetProviderWorkspaceTensors(
        executor, args, workspaceTensors,
        sizeof(workspaceTensors) / sizeof(workspaceTensors[0]));
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
    if (workspaceSize == nullptr || executor == nullptr ||
        !std::isfinite(scale) || !std::isfinite(lowerBound) ||
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
