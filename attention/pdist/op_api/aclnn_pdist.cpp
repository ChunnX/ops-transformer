/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "aclnn_pdist.h"
#include "pdist.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

#include <cmath>

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
// N/M bounds mirror the pypto kernel contract (op_kernel/pdist.py, DESIGN frozen spec):
// 2 <= N <= 65535 (N=65535 keeps y within ~2.1e9 fp32 elements), 1 <= M <= 65535.
constexpr int64_t PDIST_MIN_ROWS = 2;
constexpr int64_t PDIST_MAX_ROWS = 65535;
constexpr int64_t PDIST_MIN_COLS = 1;
constexpr int64_t PDIST_MAX_COLS = 65535;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

inline static bool CheckNotNull(const aclTensor *x, const aclTensor *out) {
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

// The PyPTO opc binary targets the reg-base (Ascend 950 family) instruction set only.
inline static aclnnStatus CheckArchSupport() {
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_3510 && npuArch != NpuArch::DAV_5102) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Pdist only supports Ascend 950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

inline static bool CheckDtypeValid(const aclTensor *x, const aclTensor *out) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(x, out, return false);
    return true;
}

inline static bool CheckTensorFormat(const aclTensor *tensor, const char *tensorName) {
    const auto originalFormat = tensor->GetOriginalFormat();
    const auto storageFormat = tensor->GetStorageFormat();
    OP_CHECK(originalFormat == Format::FORMAT_ND && storageFormat == Format::FORMAT_ND,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Pdist: %s requires ND format, but got original=%s, storage=%s.", tensorName,
                     op::ToString(originalFormat).GetString(), op::ToString(storageFormat).GetString()),
             return false);
    return true;
}

inline static bool CheckFormat(const aclTensor *x, const aclTensor *out) {
    if (!CheckTensorFormat(x, "x")) {
        return false;
    }
    return CheckTensorFormat(out, "out");
}

inline static aclnnStatus CheckParamsLogic(const aclTensor *x, const aclTensor *out) {
    OP_CHECK_WRONG_DIMENSION(x, 2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(out, 1, return ACLNN_ERR_PARAM_INVALID);

    const int64_t nSize = x->GetViewShape().GetDim(0);
    const int64_t mSize = x->GetViewShape().GetDim(1);
    if (nSize < PDIST_MIN_ROWS || nSize > PDIST_MAX_ROWS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Pdist requires N in [%ld, %ld], but got %ld.", PDIST_MIN_ROWS,
                PDIST_MAX_ROWS, nSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (mSize < PDIST_MIN_COLS || mSize > PDIST_MAX_COLS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Pdist requires M in [%ld, %ld], but got %ld.", PDIST_MIN_COLS,
                PDIST_MAX_COLS, mSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    op::Shape expectOutShape = {nSize * (nSize - 1) / 2};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectOutShape, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

inline static bool CheckTensorContiguous(const aclTensor *tensor, const char *tensorName) {
    OP_CHECK(IsContiguous(tensor),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Pdist: %s must be contiguous, but got shape %s and strides %s.",
                     tensorName, op::ToString(tensor->GetViewShape()).GetString(),
                     op::ToString(tensor->GetViewStrides()).GetString()),
             return false);
    return true;
}

inline static bool CheckContiguous(const aclTensor *x, const aclTensor *out) {
    if (!CheckTensorContiguous(x, "x")) {
        return false;
    }
    return CheckTensorContiguous(out, "out");
}

inline static aclnnStatus CheckPValue(double p) {
    // p >= 0 or +inf; NaN and negative values are parameter errors (matches the tiling-side check).
    if (std::isnan(p) || p < 0.0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Pdist only supports p >= 0 or +inf, but got %lf.", p);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckParams(const aclTensor *x, double p, const aclTensor *out) {
    // 1. null pointers
    CHECK_COND(CheckNotNull(x, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");
    // 2. arch support (reg-base / Ascend 950 family only)
    const aclnnStatus archStatus = CheckArchSupport();
    CHECK_RET(archStatus == ACLNN_SUCCESS, archStatus);
    // 3. dtype
    CHECK_COND(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");
    // 4. format
    CHECK_COND(CheckFormat(x, out), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed!");
    // 5. shape / ranges / output length
    CHECK_RET(CheckParamsLogic(x, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    // 6. contiguity
    CHECK_COND(CheckContiguous(x, out), ACLNN_ERR_PARAM_INVALID, "CheckContiguous failed!");
    // 7. norm order p
    CHECK_RET(CheckPValue(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// Torch tensors may use a flattened storage shape. The operator requires storage and original
// shapes to match the logical view shape, so normalize the descriptors before invoking the L0 op.
inline static void PreProcess(const aclTensor *x, const aclTensor *out) {
    const aclTensor *tensors[] = {x, out};
    for (const aclTensor *tensor : tensors) {
        tensor->SetStorageShape(tensor->GetViewShape());
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
}
} // namespace

aclnnStatus aclnnPdistGetWorkspaceSize(const aclTensor *x, double p, const aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnPdist, DFX_IN(x, p), DFX_OUT(out));

    auto ret = CheckParams(x, p, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    PreProcess(x, out);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclTensor *pdistOutRet = l0op::Pdist(x, static_cast<float>(p), uniqueExecutor.get());
    CHECK_RET(pdistOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyOut = l0op::ViewCopy(pdistOutRet, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPdist(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnPdist);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
