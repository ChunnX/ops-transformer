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
 * \file aclnn_mhc_sinkhorn.cpp
 * \brief aclnn_mhc_sinkhorn
 */

#include "aclnn_mhc_sinkhorn.h"
#include "mhc_sinkhorn.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT};

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;

static constexpr size_t MIN_NUMITERS = 0;
static constexpr size_t MAX_NUMITERS = 100;

static constexpr size_t SUPPORT_DIM_NUM_3 = 3;
static constexpr size_t SUPPORT_DIM_NUM_4 = 4;

static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;

static constexpr int64_t N_ALIGN = 8;
static constexpr int64_t DOUBLE_SIZE = 2;

/**
 * 判断可选辅助输出是否表示"无效输出"。
 *
 * nullptr 或 shape={1} 均认为该辅助输出无效，对应 outFlag=0。
 */
static bool IsOptionalOutputEmpty(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return true;
    }

    const auto &shape = tensor->GetViewShape();
    return shape.GetDimNum() == 1 && shape.GetShapeSize() == 1;
}

/**
 * 空指针检查。
 *
 * x/output 始终必选。
 * outFlag=1 时 normOut/sumOut 必选。
 */
static bool CheckNotNull(
    const aclTensor *x,
    int64_t outFlag,
    const aclTensor *output,
    const aclTensor *normOut,
    const aclTensor *sumOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(output, return false);

    if (outFlag) {
        OP_CHECK_NULL(normOut, return false);
        OP_CHECK_NULL(sumOut, return false);
    }

    return true;
}

/**
 * Dtype 合法性检查。
 */
static bool CheckDtypeValid(
    const aclTensor *x,
    int64_t outFlag,
    const aclTensor *output,
    const aclTensor *normOut,
    const aclTensor *sumOut)
{
    // x仅支持float32
    OP_CHECK_DTYPE_NOT_SUPPORT(
        x,
        DTYPE_SUPPORT_LIST,
        return false);

    // output仅支持float32
    OP_CHECK_DTYPE_NOT_SUPPORT(
        output,
        DTYPE_SUPPORT_LIST,
        return false);

    if (outFlag) {
        // outFlag=1时辅助输出也必须为float32
        OP_CHECK_DTYPE_NOT_SUPPORT(
            normOut,
            DTYPE_SUPPORT_LIST,
            return false);

        OP_CHECK_DTYPE_NOT_SUPPORT(
            sumOut,
            DTYPE_SUPPORT_LIST,
            return false);
    }

    return true;
}

/**
 * Format 合法性检查。
 */
static bool CheckFormat(
    const aclTensor *x,
    int64_t outFlag,
    const aclTensor *output,
    const aclTensor *normOut,
    const aclTensor *sumOut)
{
    // 当前规格要求主输入输出format一致
    if (x->GetStorageFormat() != output->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Format of input and output should be same. "
            "x [%s], output [%s].",
            ToString(x->GetStorageFormat()).GetString(),
            ToString(output->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

/**
 * Shape 和属性合法性检查。
 */
static bool CheckShape(
    const aclTensor *x,
    int64_t outFlag,
    const aclTensor *output,
    const aclTensor *normOut,
    const aclTensor *sumOut,
    int64_t numIters)
{
    // 主输出shape必须与输入x完全一致
    OP_CHECK_SHAPE_NOT_EQUAL(
        x,
        output,
        return false);

    // numIters范围：[1, 100]
    if (numIters <= MIN_NUMITERS || numIters > MAX_NUMITERS) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "numIters value error, numIters must in 1 to 100, "
            "but got numIters = %ld.",
            numIters);
        return false;
    }

    // outFlag只能为0或1
    if (outFlag != 0 && outFlag != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "outFlag value error, outFlag must be 0 or 1, "
            "but got outFlag = %ld.",
            outFlag);
        return false;
    }

    auto xShape = x->GetViewShape();
    auto xDim = xShape.GetDimNum();

    // x维数只能是3D或4D
    if (xDim != SUPPORT_DIM_NUM_3 &&
        xDim != SUPPORT_DIM_NUM_4) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dim value error, input dim must be 3 or 4, "
            "but got Dim = %ld.",
            xDim);
        return false;
    }

    // 获取最后两个矩阵维度n0/n1
    int64_t n0 = 0;
    int64_t n1 = 0;

    if (xDim == SUPPORT_DIM_NUM_3) {
        n0 = xShape.GetDim(DIM_ONE);
        n1 = xShape.GetDim(DIM_TWO);
    } else {
        n0 = xShape.GetDim(DIM_TWO);
        n1 = xShape.GetDim(DIM_THREE);
    }

    // Sinkhorn矩阵必须为方阵，且n仅支持4/6/8
    if ((n0 != n1) ||
        (n0 != N_VALID_4 &&
         n0 != N_VALID_6 &&
         n0 != N_VALID_8)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "n0 must equal n1, and n must be %ld or %ld or %ld, "
            "but got n0 = %ld, n1 = %ld.",
            N_VALID_4,
            N_VALID_6,
            N_VALID_8,
            n0,
            n1);
        return false;
    }

    if (outFlag) {
        /*
         * outFlag=1：
         *
         * normOut和sumOut必须是真实有效输出。
         * 根据已确认规格，两者均为1D Tensor。
         */

        int64_t T = xShape.GetDim(DIM_ZERO);

        if (xDim == SUPPORT_DIM_NUM_4) {
            T =
                xShape.GetDim(DIM_ZERO) *
                xShape.GetDim(DIM_ONE);
        }

        int64_t n = n0;

        int64_t expectNormSize =
            DOUBLE_SIZE *
            numIters *
            T *
            n *
            N_ALIGN;

        int64_t expectSumSize =
            DOUBLE_SIZE *
            numIters *
            T *
            N_ALIGN;

        // normOut必须是1D，且元素数严格匹配
        if (normOut->GetViewShape().GetDimNum() != 1 ||
            normOut->GetViewShape().GetShapeSize() != expectNormSize) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "normOut shape must be 1-dim and size %ld, "
                "but got dim=%ld size=%ld.",
                expectNormSize,
                normOut->GetViewShape().GetDimNum(),
                normOut->GetViewShape().GetShapeSize());
            return false;
        }

        // sumOut必须是1D，且元素数严格匹配
        if (sumOut->GetViewShape().GetDimNum() != 1 ||
            sumOut->GetViewShape().GetShapeSize() != expectSumSize) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "sumOut shape must be 1-dim and size %ld, "
                "but got dim=%ld size=%ld.",
                expectSumSize,
                sumOut->GetViewShape().GetDimNum(),
                sumOut->GetViewShape().GetShapeSize());
            return false;
        }
    } else {
        /*
         * outFlag=0：
         *
         * normOut/sumOut不参与实际输出，
         * 内部统一使用shape={1}的占位Tensor。
         */

        if (normOut->GetViewShape().GetDimNum() != 1 ||
            normOut->GetViewShape().GetShapeSize() != 1) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "outFlag is 0, normOut shape must be {1}, "
                "but got dim=%ld size=%ld.",
                normOut->GetViewShape().GetDimNum(),
                normOut->GetViewShape().GetShapeSize());
            return false;
        }

        if (sumOut->GetViewShape().GetDimNum() != 1 ||
            sumOut->GetViewShape().GetShapeSize() != 1) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "outFlag is 0, sumOut shape must be {1}, "
                "but got dim=%ld size=%ld.",
                sumOut->GetViewShape().GetDimNum(),
                sumOut->GetViewShape().GetShapeSize());
            return false;
        }
    }

    return true;
}

/**
 * ACLNN公共参数检查。
 */
static inline aclnnStatus CheckParams(
    const aclTensor *x,
    int64_t outFlag,
    float eps,
    int64_t numIters,
    const aclTensor *output,
    const aclTensor *normOut,
    const aclTensor *sumOut)
{
    // 当前接口中eps暂无额外范围约束
    (void)eps;

    // 1. 空指针检查
    CHECK_RET(
        CheckNotNull(
            x,
            outFlag,
            output,
            normOut,
            sumOut),
        ACLNN_ERR_PARAM_NULLPTR);

    // 2. Dtype检查
    CHECK_RET(
        CheckDtypeValid(
            x,
            outFlag,
            output,
            normOut,
            sumOut),
        ACLNN_ERR_PARAM_INVALID);

    // 3. Shape/属性检查
    CHECK_RET(
        CheckShape(
            x,
            outFlag,
            output,
            normOut,
            sumOut,
            numIters),
        ACLNN_ERR_PARAM_INVALID);

    // 4. Format检查
    CHECK_RET(
        CheckFormat(
            x,
            outFlag,
            output,
            normOut,
            sumOut),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(
    const aclTensor *x,
    float eps,
    int64_t numIters,
    aclTensor *output,
    aclTensor *normOut,
    aclTensor *sumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    /*
     * ==============================
     * 关键修复：
     * 必选参数必须在任何解引用之前判空。
     *
     * 不能等到CheckParams()才检查，
     * 因为后面的DFX、GetDataType等逻辑
     * 都有可能提前访问Tensor对象。
     * ==============================
     */
    OP_CHECK_NULL(
        x,
        return ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK_NULL(
        output,
        return ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK_NULL(
        workspaceSize,
        return ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK_NULL(
        executor,
        return ACLNN_ERR_PARAM_NULLPTR);

    /*
     * 必选指针确认有效后再进入DFX。
     */
    L2_DFX_PHASE_1(
        aclnnMhcSinkhorn,
        DFX_IN(x, eps, numIters),
        DFX_OUT(output, normOut, sumOut));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();

    CHECK_RET(
        uniqueExecutor.get() != nullptr,
        ACLNN_ERR_INNER_CREATE_EXECUTOR);

    /*
     * 根据两个辅助输出判断outFlag。
     *
     * 规则：
     *
     * normOut/sumOut同时为空或shape={1}
     *     -> outFlag=0
     *
     * normOut/sumOut同时为真实有效输出
     *     -> outFlag=1
     *
     * 一个有效、一个无效
     *     -> 非法参数，Host直接拦截
     */
    const bool normEmpty =
        IsOptionalOutputEmpty(normOut);

    const bool sumEmpty =
        IsOptionalOutputEmpty(sumOut);

    int64_t outFlag = 1;

    if (normEmpty && sumEmpty) {
        /*
         * outFlag=0。
         *
         * 如果用户传nullptr，
         * 为L0/Tiling创建shape={1}占位Tensor。
         */
        Shape emptyShape({1});

        if (normOut == nullptr) {
            normOut =
                uniqueExecutor.get()->AllocTensor(
                    emptyShape,
                    x->GetDataType(),
                    Format::FORMAT_ND);

            CHECK_RET(
                normOut != nullptr,
                ACLNN_ERR_INNER_NULLPTR);
        }

        if (sumOut == nullptr) {
            sumOut =
                uniqueExecutor.get()->AllocTensor(
                    emptyShape,
                    x->GetDataType(),
                    Format::FORMAT_ND);

            CHECK_RET(
                sumOut != nullptr,
                ACLNN_ERR_INNER_NULLPTR);
        }

        outFlag = 0;
    } else if (!normEmpty && !sumEmpty) {
        // 两个辅助输出同时有效
        outFlag = 1;
    } else {
        /*
         * 一个有效、一个无效属于非法组合。
         *
         * 必须Host直接返回错误，
         * 不能继续进入Tiling/Kernel。
         */
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "normOut and sumOut must both be valid or both be empty. "
            "normOut=%s, sumOut=%s.",
            normEmpty ? "empty" : "valid",
            sumEmpty ? "empty" : "valid");

        return ACLNN_ERR_PARAM_INVALID;
    }

    /*
     * 完整参数校验。
     *
     * 到这里：
     * x/output一定非nullptr；
     * normOut/sumOut也一定已经有Tensor对象。
     */
    auto ret = CheckParams(
        x,
        outFlag,
        eps,
        numIters,
        output,
        normOut,
        sumOut);

    CHECK_RET(
        ret == ACLNN_SUCCESS,
        ret);

    // 不支持空输入Tensor
    if (x->IsEmpty()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Do not support empty input tensor.");

        return ACLNN_ERR_PARAM_INVALID;
    }

    // 将输入x转换为连续Tensor
    const aclTensor *xContiguous =
        l0op::Contiguous(
            x,
            uniqueExecutor.get());

    CHECK_RET(
        xContiguous != nullptr,
        ACLNN_ERR_INNER_NULLPTR);

    // 主输出转换为连续Tensor
    aclTensor *outputContiguous =
        const_cast<aclTensor *>(
            l0op::Contiguous(
                output,
                uniqueExecutor.get()));

    CHECK_RET(
        outputContiguous != nullptr,
        ACLNN_ERR_INNER_NULLPTR);

    /*
     * 调用L0 MhcSinkhorn。
     *
     * outFlag会作为属性继续传入Tiling，
     * 决定使用outFlag=0还是outFlag=1模板。
     */
    const aclTensor *kernelOut =
        l0op::MhcSinkhorn(
            xContiguous,
            outFlag,
            eps,
            numIters,
            outputContiguous,
            normOut,
            sumOut,
            uniqueExecutor.get());

    CHECK_RET(
        kernelOut != nullptr,
        ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝回用户output
    auto viewCopyResult =
        l0op::ViewCopy(
            kernelOut,
            output,
            uniqueExecutor.get());

    CHECK_RET(
        viewCopyResult != nullptr,
        ACLNN_ERR_INNER_NULLPTR);

    // 获取workspace大小
    *workspaceSize =
        uniqueExecutor->GetWorkspaceSize();

    // 将executor所有权转移给外部
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcSinkhorn(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcSinkhorn);

    // 调用框架执行算子
    return CommonOpExecutorRun(
        workspace,
        workspaceSize,
        executor,
        stream);
}

#ifdef __cplusplus
}
#endif
