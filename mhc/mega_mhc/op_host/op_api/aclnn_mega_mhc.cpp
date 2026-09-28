/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_mega_mhc.h"
#include "mega_mhc.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

namespace {
constexpr size_t INPUT_COUNT = 11;
constexpr size_t OUTPUT_COUNT = 5;
constexpr size_t H_PRE_INDEX = 4;
constexpr size_t MID_INDEX = 10;
constexpr size_t PRE_MIX_INDEX = 8;
constexpr int64_t HEADS = 4;
constexpr int64_t PROJECTION = 24;
using Inputs = std::array<const aclTensor *, INPUT_COUNT>;
using Outputs = std::array<const aclTensor *, OUTPUT_COUNT>;

/** Validate one public ND tensor against its exact shape and dtype before making it contiguous. */
bool CheckTensor(const aclTensor *tensor, const char *name, const std::vector<int64_t> &shape, op::DataType dtype)
{
    if (tensor == nullptr || tensor->GetDataType() != dtype || op::IsPrivateFormat(tensor->GetViewFormat()) ||
        tensor->GetViewShape().GetDimNum() != shape.size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be non-null, public format, and have the expected rank/dtype", name);
        return false;
    }
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (tensor->GetViewShape().GetDim(axis) != shape[axis]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s shape mismatch at axis %zu", name, axis);
            return false;
        }
    }
    return true;
}

/** Check N=4, aligned D in [2560,7168], optional Mid, four active outputs and reserved h_pre; M is the product of
 * leading axes. */
aclnnStatus CheckContract(const Inputs &inputs, const Outputs &outputs, double normEps, double mhcPreNormEps,
                          double hcEps, int64_t mode)
{
    for (size_t i = 0; i < MID_INDEX; ++i) {
        if (i != PRE_MIX_INDEX) {
            CHECK_RET(inputs[i] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        }
    }
    if (inputs[PRE_MIX_INDEX] != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pre_mix is reserved and must be null in this version");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (outputs[H_PRE_INDEX] != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_pre is reserved and must be null in this version");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t i = 0; i < H_PRE_INDEX; ++i) {
        CHECK_RET(outputs[i] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    const float norm = static_cast<float>(normEps);
    const float preNorm = static_cast<float>(mhcPreNormEps);
    const float hc = static_cast<float>(hcEps);
    CHECK_RET(std::isfinite(norm) && norm > 0 && std::isfinite(preNorm) && preNorm > 0 && std::isfinite(hc) &&
                  hcEps >= 0 && (mode == 0 || mode == 1),
              ACLNN_ERR_PARAM_INVALID);
    const auto &shape = inputs[0]->GetViewShape();
    const auto rank = shape.GetDimNum();
    CHECK_RET(rank == 2 || rank == 3, ACLNN_ERR_PARAM_INVALID);
    const int64_t d = shape.GetDim(rank - 1);
    CHECK_RET(d >= 2560 && d <= 7168 && d % 256 == 0, ACLNN_ERR_PARAM_INVALID);
    const auto dtype = inputs[0]->GetDataType();
    CHECK_RET(dtype == op::DataType::DT_FLOAT16 || dtype == op::DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID);
    std::vector<int64_t> leading;
    uint64_t rows = 1;
    for (size_t axis = 0; axis + 1 < rank; ++axis) {
        const int64_t dim = shape.GetDim(axis);
        CHECK_RET(dim > 0 && rows <= UINT32_MAX / static_cast<uint64_t>(dim), ACLNN_ERR_PARAM_INVALID);
        rows *= dim;
        leading.push_back(dim);
    }
    auto xShape = leading;
    xShape.push_back(d);
    auto postShape = leading;
    postShape.push_back(HEADS);
    auto resShape = postShape;
    resShape.push_back(HEADS);
    auto x2Shape = postShape;
    x2Shape.push_back(d);
    const std::array<std::vector<int64_t>, INPUT_COUNT> shapes = {xShape,
                                                                  x2Shape,
                                                                  postShape,
                                                                  resShape,
                                                                  std::vector<int64_t>{PROJECTION, HEADS * d},
                                                                  std::vector<int64_t>{3},
                                                                  std::vector<int64_t>{PROJECTION},
                                                                  std::vector<int64_t>{d},
                                                                  postShape,
                                                                  std::vector<int64_t>{d},
                                                                  std::vector<int64_t>{HEADS * d}};
    const char *names[] = {"hOut", "residual", "hPost",  "hRes",   "phi",   "alpha",
                           "bias", "gamma1",   "preMix", "gamma0", "gamma2"};
    for (size_t i = 0; i < INPUT_COUNT; ++i) {
        if (i == PRE_MIX_INDEX || (i == MID_INDEX && inputs[i] == nullptr)) {
            continue;
        }
        CHECK_RET(CheckTensor(inputs[i], names[i], shapes[i], i < 2 ? dtype : op::DataType::DT_FLOAT),
                  ACLNN_ERR_PARAM_INVALID);
    }
    const size_t outputSources[] = {0, 2, 3, 1, 2};
    const char *outputNames[] = {"hInPrime", "hPostNew", "hResNew", "x2Out", "hPre"};
    for (size_t i = 0; i < H_PRE_INDEX; ++i) {
        CHECK_RET(CheckTensor(outputs[i], outputNames[i], shapes[outputSources[i]],
                              i == 0 || i == 3 ? dtype : op::DataType::DT_FLOAT),
                  ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

/** Materialize strided inputs, launch to contiguous outputs, then ViewCopy back to the caller's output views. */
aclnnStatus BuildExecution(Inputs inputs, const Outputs &outputs, double normEps, double mhcPreNormEps, double hcEps,
                           int64_t mode, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    const auto status = CheckContract(inputs, outputs, normEps, mhcPreNormEps, hcEps, mode);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    auto owner = CREATE_EXECUTOR();
    CHECK_RET(owner.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    for (auto &input : inputs) {
        if (input != nullptr) {
            input = l0op::Contiguous(input, owner.get());
            CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    Outputs contiguousOutputs{};
    for (size_t i = 0; i < H_PRE_INDEX; ++i) {
        contiguousOutputs[i] =
            owner->AllocTensor(outputs[i]->GetViewShape(), outputs[i]->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(contiguousOutputs[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // CANN-0830 drops null output slots when packing kernel arguments. An empty tensor preserves
    // the reserved h_pre slot without allocating output storage; the public argument remains null.
    contiguousOutputs[H_PRE_INDEX] = owner->AllocTensor(op::Shape{0}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    CHECK_RET(contiguousOutputs[H_PRE_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const bool launched = MegaMhcl0op::MegaMhc(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9],
        inputs[10], normEps, mhcPreNormEps, hcEps, mode, contiguousOutputs[0], contiguousOutputs[1],
        contiguousOutputs[2], contiguousOutputs[3], contiguousOutputs[4], owner.get());
    CHECK_RET(launched, ACLNN_ERR_INNER_NULLPTR);
    for (size_t i = 0; i < H_PRE_INDEX; ++i) {
        CHECK_RET(l0op::ViewCopy(contiguousOutputs[i], outputs[i], owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

/** Build the execution graph and report its workspace; no kernel executes in this phase. */
aclnnStatus aclnnMegaMhcGetWorkspaceSize(const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost,
                                         const aclTensor *hRes, const aclTensor *phi, const aclTensor *alpha,
                                         const aclTensor *bias, const aclTensor *gamma1,
                                         const aclTensor *preMixOptional, const aclTensor *gamma0Optional,
                                         const aclTensor *gamma2Optional, double normEps, double mhcPreNormEps,
                                         double hcEps, int64_t computeMode, const aclTensor *hInPrimeOut,
                                         const aclTensor *hPostNewOut, const aclTensor *hResNewOut,
                                         const aclTensor *x2Out, const aclTensor *hPreOut, uint64_t *workspaceSize,
                                         aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMegaMhc,
                   DFX_IN(hOut, residual, hPost, hRes, phi, alpha, bias, gamma1, preMixOptional, gamma0Optional,
                          gamma2Optional, normEps, mhcPreNormEps, hcEps, computeMode),
                   DFX_OUT(hInPrimeOut, hPostNewOut, hResNewOut, x2Out, hPreOut));
    return BuildExecution(
        {hOut, residual, hPost, hRes, phi, alpha, bias, gamma1, preMixOptional, gamma0Optional, gamma2Optional},
        {hInPrimeOut, hPostNewOut, hResNewOut, x2Out, hPreOut}, normEps, mhcPreNormEps, hcEps, computeMode,
        workspaceSize, executor);
}

/** Execute the prepared graph on the caller's stream using its allocated workspace. */
aclnnStatus aclnnMegaMhc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMegaMhc);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
