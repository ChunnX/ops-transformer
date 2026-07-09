/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_mhc_pre.h"
#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include <initializer_list>
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mhc_pre.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr size_t DIM_NUM_1 = 1UL;
constexpr size_t DIM_NUM_2 = 2UL;
constexpr size_t DIM_NUM_3 = 3UL;
constexpr size_t DIM_NUM_4 = 4UL;

constexpr int64_t DIM_IDX_0 = 0;
constexpr int64_t DIM_IDX_1 = 1;
constexpr int64_t DIM_IDX_2 = 2;
constexpr int64_t DIM_IDX_3 = 3;

constexpr int64_t N_VALID_VALUES[] = {4, 6, 8};
constexpr int64_t D_ALIGNMENT = 16;
constexpr int64_t ALPHA_DIM_SIZE_3 = 3;
constexpr int64_t ALPHA_DIM_SIZE_2 = 2;
constexpr int64_t PHI_DIM_OFFSET = 2;

static bool CheckAlphaShape(const aclTensor *alphaTensor);
static bool ValidateNDParams(int64_t n, int64_t d);
static bool CheckPhiShape(const aclTensor *phiTensor, int64_t n2Plus2n, int64_t nD);
static bool CheckBiasShape(const aclTensor *biasTensor, int64_t n2Plus2n);
static bool CheckGammaShape(const aclTensor *gammaOptional, int64_t n, int64_t d);
static bool CheckTensorShape(const aclTensor *tensor, std::initializer_list<int64_t> expectedShape, const char *name);

struct MhcParamsBase {
    const aclTensor *x = nullptr;
    const aclTensor *phi = nullptr;
    const aclTensor *alpha = nullptr;
    const aclTensor *bias = nullptr;
    const aclTensor *gammaOptional = nullptr;
    double normEps;
    double hcEps;
    aclTensor *hIn = nullptr;
    aclTensor *hPost = nullptr;
    aclTensor *hRes = nullptr;
    aclTensor *invRmsOptional = nullptr;
    aclTensor *hMixOptional = nullptr;
    aclTensor *hPreOptional = nullptr;
    bool hasResi = true;

    // 用于存储转换后的连续tensor（在ConvertDataContiguous中使用）
    const aclTensor *xContiguous = nullptr;
    const aclTensor *phiContiguous = nullptr;
    const aclTensor *alphaContiguous = nullptr;
    const aclTensor *biasContiguous = nullptr;
    const aclTensor *gammaOptionalContiguous = nullptr;
};

class MhcBuilder {
public:
    static MhcBuilder Create()
    {
        MhcBuilder obj;

        return obj;
    }

    MhcBuilder &SetInput(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                         const aclTensor *gammaOptional)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.bias = bias;
        obj_.gammaOptional = gammaOptional;
        return *this;
    }

    MhcBuilder &SetAttr(double normEps, double hcEps)
    {
        obj_.normEps = normEps;
        obj_.hcEps = hcEps;
        return *this;
    }

    MhcBuilder &SetOutput(aclTensor *hIn, aclTensor *hPost, aclTensor *hRes)
    {
        obj_.hIn = hIn;
        obj_.hPost = hPost;
        obj_.hRes = hRes;
        return *this;
    }

    MhcBuilder &SetOptionalOutput(aclTensor *invRmsOptional, aclTensor *hMixOptional, aclTensor *hPreOptional)
    {
        obj_.invRmsOptional = invRmsOptional;
        obj_.hMixOptional = hMixOptional;
        obj_.hPreOptional = hPreOptional;

        return *this;
    }

    MhcParamsBase Build() const
    {
        return obj_;
    }

private:
    MhcParamsBase obj_;
};

static bool CheckNotNull(const MhcParamsBase &params)
{
    if (params.x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "X tensor is nullptr");
        return false;
    }
    if (params.phi == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Phi tensor is nullptr");
        return false;
    }
    if (params.alpha == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Alpha tensor is nullptr");
        return false;
    }
    if (params.bias == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Bias tensor is nullptr");
        return false;
    }
    if (params.hIn == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HIn tensor is nullptr");
        return false;
    }
    if (params.hPost == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HPost tensor is nullptr");
        return false;
    }
    int64_t alphaSize = params.alpha->GetViewShape().GetDim(0);
    if (alphaSize == ALPHA_DIM_SIZE_3 && params.hRes == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HRes tensor is nullptr when alpha size is 3");
        return false;
    }
    return true;
}

static bool CheckEmptyTensor(const MhcParamsBase &params)
{
    if (params.x->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X tensor is empty");
        return false;
    }
    if (params.phi->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor is empty");
        return false;
    }
    if (params.alpha->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha tensor is empty");
        return false;
    }
    if (params.bias->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor is empty");
        return false;
    }
    return true;
}

static bool CheckInputDims(const MhcParamsBase &params)
{
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    if (xDimNum != DIM_NUM_3 && xDimNum != DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X tensor dim num must be 3 or 4, but got %zu", xDimNum);
        return false;
    }
    if (params.phi->GetViewShape().GetDimNum() != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor dim num must be 2");
        return false;
    }
    if (params.alpha->GetViewShape().GetDimNum() != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha tensor dim num must be 1");
        return false;
    }
    if (params.bias->GetViewShape().GetDimNum() != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor dim num must be 1");
        return false;
    }
    if (params.gammaOptional != nullptr && params.gammaOptional->GetViewShape().GetDimNum() != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GammaOptional tensor dim num must be 2");
        return false;
    }
    return true;
}

static bool CheckOutputDims(const MhcParamsBase &params)
{
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    size_t outputDimNum = xDimNum == DIM_NUM_4 ? DIM_NUM_3 : DIM_NUM_2;
    if (params.hIn->GetViewShape().GetDimNum() != outputDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "HIn tensor dim num must be %zu", outputDimNum);
        return false;
    }
    if (params.hPost->GetViewShape().GetDimNum() != outputDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "HPost tensor dim num must be %zu", outputDimNum);
        return false;
    }
    int64_t alphaSize = params.alpha->GetViewShape().GetDim(0);
    if (alphaSize == ALPHA_DIM_SIZE_3 && params.hRes->GetViewShape().GetDimNum() != xDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "HRes tensor dim num must be %zu", xDimNum);
        return false;
    }
    return true;
}

static bool CheckOptionalOutputGroup(const MhcParamsBase &params)
{
    bool hasAnyOptional =
        params.invRmsOptional != nullptr || params.hMixOptional != nullptr || params.hPreOptional != nullptr;
    bool hasAllOptional =
        params.invRmsOptional != nullptr && params.hMixOptional != nullptr && params.hPreOptional != nullptr;
    if (hasAnyOptional && !hasAllOptional) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "InvRmsOptional, HMixOptional and HPreOptional must be all nullptr or all not nullptr");
        return false;
    }
    return true;
}


static bool CheckOptionalOutputDims(const MhcParamsBase &params)
{
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    size_t outputDimNum = xDimNum == DIM_NUM_4 ? DIM_NUM_3 : DIM_NUM_2;
    size_t invRmsDimNum = xDimNum == DIM_NUM_4 ? DIM_NUM_2 : DIM_NUM_1;
    if (params.invRmsOptional->GetViewShape().GetDimNum() != invRmsDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InvRmsOptional tensor dim num must be %zu", invRmsDimNum);
        return false;
    }
    if (params.hMixOptional->GetViewShape().GetDimNum() != outputDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "HMixOptional tensor dim num must be %zu", outputDimNum);
        return false;
    }
    if (params.hPreOptional->GetViewShape().GetDimNum() != outputDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "HPreOptional tensor dim num must be %zu", outputDimNum);
        return false;
    }
    return true;
}

static bool CheckInputOutDims(const MhcParamsBase &params)
{
    return CheckInputDims(params) && CheckOutputDims(params) && CheckOptionalOutputGroup(params) &&
           CheckOptionalOutputDims(params);
}

static void GetXShapeInfo(const MhcParamsBase &params, int64_t &n, int64_t &d, int64_t &nD)
{
    auto xShape = params.x->GetViewShape();
    auto xDimNum = xShape.GetDimNum();
    n = xDimNum == DIM_NUM_4 ? xShape.GetDim(DIM_IDX_2) : xShape.GetDim(DIM_IDX_1);
    d = xDimNum == DIM_NUM_4 ? xShape.GetDim(DIM_IDX_3) : xShape.GetDim(DIM_IDX_2);
    nD = n * d;
}

static void GetExpectedParamShape(int64_t alphaSize, int64_t n, int64_t &expectedPhiRows, int64_t &expectedBiasSize)
{
    expectedPhiRows = alphaSize == ALPHA_DIM_SIZE_2 ? 2 * n : n * n + 2 * n;
    expectedBiasSize = expectedPhiRows;
}

static bool CheckParamShapes(const MhcParamsBase &params, int64_t n, int64_t d, int64_t nD, int64_t expectedPhiRows,
                             int64_t expectedBiasSize)
{
    if (!ValidateNDParams(n, d)) {
        return false;
    }
    if (!CheckPhiShape(params.phi, expectedPhiRows, nD)) {
        return false;
    }
    if (!CheckBiasShape(params.bias, expectedBiasSize)) {
        return false;
    }
    return CheckGammaShape(params.gammaOptional, n, d);
}

static bool CheckOutputShape4D(const MhcParamsBase &params, int64_t n, int64_t d, int64_t expectedPhiRows)
{
    auto xShape = params.x->GetViewShape();
    int64_t alphaSize = params.alpha->GetViewShape().GetDim(0);
    int64_t b = xShape.GetDim(DIM_IDX_0);
    int64_t s = xShape.GetDim(DIM_IDX_1);
    if (!CheckTensorShape(params.hIn, {b, s, d}, "HIn") || !CheckTensorShape(params.hPost, {b, s, n}, "HPost")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && !CheckTensorShape(params.hRes, {b, s, n, n}, "HRes")) {
        return false;
    }
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    return CheckTensorShape(params.invRmsOptional, {b, s}, "InvRmsOptional") &&
           CheckTensorShape(params.hMixOptional, {b, s, expectedPhiRows}, "HMixOptional") &&
           CheckTensorShape(params.hPreOptional, {b, s, n}, "HPreOptional");
}

static bool CheckOutputShape3D(const MhcParamsBase &params, int64_t n, int64_t d, int64_t expectedPhiRows)
{
    auto xShape = params.x->GetViewShape();
    int64_t alphaSize = params.alpha->GetViewShape().GetDim(0);
    int64_t t = xShape.GetDim(DIM_IDX_0);
    if (!CheckTensorShape(params.hIn, {t, d}, "HIn") || !CheckTensorShape(params.hPost, {t, n}, "HPost")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && !CheckTensorShape(params.hRes, {t, n, n}, "HRes")) {
        return false;
    }
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    return CheckTensorShape(params.invRmsOptional, {t}, "InvRmsOptional") &&
           CheckTensorShape(params.hMixOptional, {t, expectedPhiRows}, "HMixOptional") &&
           CheckTensorShape(params.hPreOptional, {t, n}, "HPreOptional");
}

static bool CheckInputOutShape(const MhcParamsBase &params)
{
    if (!CheckAlphaShape(params.alpha)) {
        return false;
    }
    int64_t n = 0;
    int64_t d = 0;
    int64_t nD = 0;
    GetXShapeInfo(params, n, d, nD);
    int64_t expectedPhiRows = 0;
    int64_t expectedBiasSize = 0;
    GetExpectedParamShape(params.alpha->GetViewShape().GetDim(0), n, expectedPhiRows, expectedBiasSize);
    if (!CheckParamShapes(params, n, d, nD, expectedPhiRows, expectedBiasSize)) {
        return false;
    }
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    if (xDimNum == DIM_NUM_4) {
        return CheckOutputShape4D(params, n, d, expectedPhiRows);
    }
    return CheckOutputShape3D(params, n, d, expectedPhiRows);
}

static bool CheckTensorShape(const aclTensor *tensor, std::initializer_list<int64_t> expectedShape, const char *name)
{
    auto tensorShape = tensor->GetViewShape();
    size_t dimIndex = 0;
    for (const auto expectedDim : expectedShape) {
        int64_t actualDim = tensorShape.GetDim(dimIndex);
        if (actualDim != expectedDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor dim[%zu] must be %ld, but got %ld", name, dimIndex, expectedDim,
                    actualDim);
            return false;
        }
        ++dimIndex;
    }
    return true;
}

static bool CheckAlphaShape(const aclTensor *alphaTensor)
{
    auto alphaShape = alphaTensor->GetViewShape();
    int64_t alphaSize = alphaShape.GetDim(0);
    if (alphaSize != ALPHA_DIM_SIZE_2 && alphaSize != ALPHA_DIM_SIZE_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha tensor shape must be (2) or (3), but got (%ld)", alphaSize);
        return false;
    }
    return true;
}

static bool ValidateNDParams(int64_t n, int64_t d)
{
    if (n <= 0 || d <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid X tensor shape: n=%ld, d=%ld", n, d);
        return false;
    }

    bool isValidN = false;
    for (size_t i = 0; i < sizeof(N_VALID_VALUES) / sizeof(N_VALID_VALUES[0]); ++i) {
        if (n == N_VALID_VALUES[i]) {
            isValidN = true;
            break;
        }
    }
    if (!isValidN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "N must be 4/6/8, but got n=%ld", n);
        return false;
    }

    if (d % D_ALIGNMENT != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "D must be aligned to %u elements, but got d=%ld", D_ALIGNMENT, d);
        return false;
    }

    return true;
}

static bool CheckPhiShape(const aclTensor *phiTensor, int64_t n2Plus2n, int64_t nD)
{
    auto phiShape = phiTensor->GetViewShape();
    if (phiShape.GetDim(0) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor first dim must be %ld, but got %ld", n2Plus2n, phiShape.GetDim(0));
        return false;
    }
    if (phiShape.GetDim(1) != nD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor second dim must be nD=%ld, but got %ld", nD, phiShape.GetDim(1));
        return false;
    }
    return true;
}

static bool CheckBiasShape(const aclTensor *biasTensor, int64_t n2Plus2n)
{
    auto biasShape = biasTensor->GetViewShape();
    if (biasShape.GetDim(0) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor dim must be %ld, but got %ld", n2Plus2n, biasShape.GetDim(0));
        return false;
    }
    return true;
}

static bool CheckGammaShape(const aclTensor *gammaOptional, int64_t n, int64_t d)
{
    if (gammaOptional != nullptr) {
        auto gammaShape = gammaOptional->GetViewShape();
        if (gammaShape.GetDim(0) != n) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GammaOptional tensor first dim must be n=%ld, but got %ld", n,
                    gammaShape.GetDim(0));
            return false;
        }
        if (gammaShape.GetDim(1) != d) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GammaOptional tensor second dim must be D=%ld, but got %ld", d,
                    gammaShape.GetDim(1));
            return false;
        }
    }
    return true;
}

static bool CheckInputDtype(const MhcParamsBase &params)
{
    const std::initializer_list<DataType> X_SUPPORT_DTYPE_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16};
    auto xDtype = params.x->GetDataType();
    bool xDtypeValid = false;
    for (const auto &dtype : X_SUPPORT_DTYPE_LIST) {
        if (xDtype == dtype) {
            xDtypeValid = true;
            break;
        }
    }
    if (!xDtypeValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X tensor dtype must be BF16 or FP16");
        return false;
    }
    if (params.phi->GetDataType() != DataType::DT_FLOAT || params.alpha->GetDataType() != DataType::DT_FLOAT ||
        params.bias->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi, Alpha and Bias tensor dtype must be FP32");
        return false;
    }
    return params.gammaOptional == nullptr || params.gammaOptional->GetDataType() == DataType::DT_FLOAT;
}

static bool CheckOutputDtype(const MhcParamsBase &params)
{
    if (params.hIn->GetDataType() != params.x->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "HIn tensor dtype must be same as X tensor dtype");
        return false;
    }
    if (params.hPost->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "HPost tensor dtype must be FP32");
        return false;
    }
    if (params.hRes != nullptr && params.hRes->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "HRes tensor dtype must be FP32");
        return false;
    }
    return true;
}

static bool CheckOptionalOutputDtype(const MhcParamsBase &params)
{
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    if (params.invRmsOptional->GetDataType() != DataType::DT_FLOAT ||
        params.hMixOptional->GetDataType() != DataType::DT_FLOAT ||
        params.hPreOptional->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Optional output tensor dtype must be FP32");
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const MhcParamsBase &params)
{
    return CheckInputDtype(params) && CheckOutputDtype(params) && CheckOptionalOutputDtype(params);
}

static bool IsPrivateFormat(ge::Format format)
{
    if (format == ge::FORMAT_NC1HWC0 || format == ge::FORMAT_FRACTAL_Z || format == ge::FORMAT_NDC1HWC0 ||
        format == ge::FORMAT_FRACTAL_Z_3D || format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_NC1HWC0_C04) {
        return true;
    }

    return false;
}

static bool CheckTensorFormat(const aclTensor *tensor, const char *name)
{
    if (IsPrivateFormat(tensor->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor format must be ND", name);
        return false;
    }
    return true;
}

static bool CheckInputFormat(const MhcParamsBase &params)
{
    if (!CheckTensorFormat(params.x, "X") || !CheckTensorFormat(params.phi, "Phi") ||
        !CheckTensorFormat(params.alpha, "Alpha") || !CheckTensorFormat(params.bias, "Bias")) {
        return false;
    }
    return params.gammaOptional == nullptr || CheckTensorFormat(params.gammaOptional, "GammaOptional");
}

static bool CheckOutputFormat(const MhcParamsBase &params)
{
    if (!CheckTensorFormat(params.hIn, "HIn") || !CheckTensorFormat(params.hPost, "HPost")) {
        return false;
    }
    return params.hRes == nullptr || CheckTensorFormat(params.hRes, "HRes");
}

static bool CheckOptionalOutputFormat(const MhcParamsBase &params)
{
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    return CheckTensorFormat(params.invRmsOptional, "InvRmsOptional") &&
           CheckTensorFormat(params.hMixOptional, "HMixOptional") &&
           CheckTensorFormat(params.hPreOptional, "HPreOptional");
}

static bool CheckFormat(const MhcParamsBase &params)
{
    return CheckInputFormat(params) && CheckOutputFormat(params) && CheckOptionalOutputFormat(params);
}
static aclnnStatus CheckParams(const MhcParamsBase &params)
{
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckEmptyTensor(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInputOutDims(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInputOutShape(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ConvertDataContiguous(MhcParamsBase &params, aclOpExecutor *executor)
{
    // 将输入tensor转换为连续格�?    params.xContiguous = l0op::Contiguous(params.x, executor);
    CHECK_RET(params.xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.phiContiguous = l0op::Contiguous(params.phi, executor);
    CHECK_RET(params.phiContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.alphaContiguous = l0op::Contiguous(params.alpha, executor);
    CHECK_RET(params.alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.biasContiguous = l0op::Contiguous(params.bias, executor);
    CHECK_RET(params.biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (params.gammaOptional != nullptr) {
        params.gammaOptionalContiguous = l0op::Contiguous(params.gammaOptional, executor);
        CHECK_RET(params.gammaOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CopyRequiredOutputs(const aclTensor *hInOut, const aclTensor *hPostOut, const aclTensor *hResOut,
                                       MhcParamsBase &params, aclOpExecutor *executor)
{
    auto ret0 = l0op::ViewCopy(hInOut, params.hIn, executor);
    CHECK_RET(ret0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ret1 = l0op::ViewCopy(hPostOut, params.hPost, executor);
    CHECK_RET(ret1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (params.hasResi && params.hRes != nullptr) {
        auto ret2 = l0op::ViewCopy(hResOut, params.hRes, executor);
        CHECK_RET(ret2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CopyOptionalOutputs(const aclTensor *invRmsOut, const aclTensor *hMixOut, const aclTensor *hPreOut,
                                       MhcParamsBase &params, aclOpExecutor *executor)
{
    if (params.invRmsOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto ret3 = l0op::ViewCopy(invRmsOut, params.invRmsOptional, executor);
    CHECK_RET(ret3 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ret4 = l0op::ViewCopy(hMixOut, params.hMixOptional, executor);
    CHECK_RET(ret4 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ret5 = l0op::ViewCopy(hPreOut, params.hPreOptional, executor);
    CHECK_RET(ret5 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus mHCPreCommonProcess(MhcParamsBase &params, aclOpExecutor *executor)
{
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ConvertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    int64_t alphaSize = params.alphaContiguous->GetViewShape().GetDim(0);
    params.hasResi = (alphaSize == ALPHA_DIM_SIZE_3);
    int64_t outFlag = params.invRmsOptional != nullptr ? 1 : 0;
    auto outParams =
        l0op::MhcPre(params.xContiguous, params.phiContiguous, params.alphaContiguous, params.biasContiguous,
                     params.gammaOptionalContiguous, outFlag, params.normEps, params.hcEps, executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
    ret = CopyRequiredOutputs(std::get<0>(outParams), std::get<1>(outParams), std::get<2>(outParams), params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CopyOptionalOutputs(std::get<3>(outParams), std::get<4>(outParams), std::get<5>(outParams), params,
                               executor);
}

aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                        const aclTensor *bias, const aclTensor *gammaOptional, double normEps,
                                        double hcEps, aclTensor *hIn, aclTensor *hPost, aclTensor *hRes,
                                        aclTensor *invRmsOptional, aclTensor *hMixOptional, aclTensor *hPreOptional,
                                        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPre, DFX_IN(x, phi, alpha, bias, gammaOptional, normEps, hcEps),
                   DFX_OUT(hIn, hPost, hRes, invRmsOptional, hMixOptional, hPreOptional));
    auto uniqueExecutor = CREATE_EXECUTOR();

    MhcParamsBase params = MhcBuilder::Create()
                               .SetInput(x, phi, alpha, bias, gammaOptional)
                               .SetAttr(normEps, hcEps)
                               .SetOutput(hIn, hPost, hRes)
                               .SetOptionalOutput(invRmsOptional, hMixOptional, hPreOptional)
                               .Build();

    auto ret = mHCPreCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPre(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPre);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace
#ifdef __cplusplus
}
#endif
