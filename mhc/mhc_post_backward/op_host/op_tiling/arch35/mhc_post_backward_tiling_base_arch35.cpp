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
 * \file mhc_post_backward_tiling_base_arch35.cpp
 * \brief
 */

#include <cmath>
#include <algorithm>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_base.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "../mhc_post_backward_tiling.h"
#include "../../../op_kernel/arch35/mhc_post_backward_tiling_data_arch35.h"
#include "../../../op_kernel/arch35/mhc_post_backward_tiling_key_arch35.h"

namespace optiling {

// Ceiling division
inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

// Align value up to the nearest multiple of align
inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

// Align value down to the nearest multiple of align
inline uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return (value / align) * align;
}

// Constants for memory alignment and tiling configuration
constexpr uint64_t TILING_KEY_GENERALIZED = 0;
constexpr uint32_t BF16_FP16_ALIGN_SIZE = 16;  // 16 elements = 32 bytes for bf16/fp16
constexpr uint32_t FLOAT32_ALIGN_SIZE = 8;     // 8 elements = 32 bytes for float32

// Input indices - 按照OpDef定义的顺序
// Input: grad_output, x, h_res, h_out, h_post
const static int64_t GRAD_OUTPUT_INPUT_INDEX = 0;
const static int64_t X_INPUT_INDEX = 1;
const static int64_t H_RES_INPUT_INDEX = 2;
const static int64_t H_OUT_INPUT_INDEX = 3;
const static int64_t H_POST_INPUT_INDEX = 4;

// Output indices
const static int64_t GRAD_X_OUTPUT_INDEX = 0;
const static int64_t GRAD_H_RES_OUTPUT_INDEX = 1;
const static int64_t GRAD_H_OUT_OUTPUT_INDEX = 2;
const static int64_t GRAD_H_POST_OUTPUT_INDEX = 3;

class MhcPostBackwardTilingBaseArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPostBackwardTilingBaseArch35(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~MhcPostBackwardTilingBaseArch35() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    // Check functions
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShapeAllPositive(int64_t idx) const;
    ge::graphStatus CheckShapeAllPositive();
    ge::graphStatus CheckDataType();
    ge::graphStatus CheckShapeConsistency();
    ge::graphStatus CheckSpecConstraints();
    ge::graphStatus CheckParam();
    ge::graphStatus ComputeTiling();

    const gert::Shape *gradOutputShape_ = nullptr;

    uint32_t B_ = 0;
    uint32_t S_ = 0;
    uint32_t n_ = 0;
    uint32_t D_ = 0;
    uint32_t totalItems_ = 0;
    uint32_t usedCores_ = 0;
    uint32_t itemsPerCore_ = 0;
    uint32_t remainderItems_ = 0;
    uint32_t tileD_ = 0;
    uint32_t nTilesD_ = 0;
    uint32_t usedAic_ = 0;
    uint32_t itemsPerAic_ = 0;
    uint32_t remainderItemsAic_ = 0;
    uint32_t isNAligned_ = 0;
    uint32_t isNNAligned_ = 0;
    uint32_t isDAligned_ = 0;

    const char *opName_ = "";
    ge::DataType dtype_ = ge::DT_UNDEFINED;
};

ge::graphStatus MhcPostBackwardTilingBaseArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_, "fail to get platform info");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();

    // 获取grad_output shape信息: (B, S, n, D)
    auto gradOutputShapePtr = context_->GetInputShape(GRAD_OUTPUT_INPUT_INDEX);
    if (gradOutputShapePtr == nullptr) {
        OP_LOGE(context_, "grad_output shape is null");
        return ge::GRAPH_FAILED;
    }
    gradOutputShape_ = &gradOutputShapePtr->GetStorageShape();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckNullptr()
{
    // Check all input desc and shape
    for (int64_t i = GRAD_OUTPUT_INPUT_INDEX; i <= H_POST_INPUT_INDEX; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr,
                    OP_LOGE(context_, "input %ld desc is nullptr", i),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr,
                    OP_LOGE(context_, "input %ld shape is nullptr", i),
                    return ge::GRAPH_FAILED);
    }

    // Check all output desc and shape
    for (int64_t i = GRAD_X_OUTPUT_INDEX; i <= GRAD_H_POST_OUTPUT_INDEX; i++) {
        auto desc = context_->GetOutputDesc(i);
        OP_CHECK_IF(desc == nullptr,
                    OP_LOGE(context_, "output %ld desc is nullptr", i),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetOutputShape(i);
        OP_CHECK_IF(shape == nullptr,
                    OP_LOGE(context_, "output %ld shape is nullptr", i),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckShapeAllPositive(int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE(context_, "input %ld has non-positive shape, dim %lu actual %ld",
                            idx, i, shape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckShapeAllPositive()
{
    // Check all inputs
    for (int64_t i = GRAD_OUTPUT_INPUT_INDEX; i <= H_POST_INPUT_INDEX; i++) {
        OP_CHECK_IF(CheckShapeAllPositive(i) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "input %ld has non-positive shape", i),
                    return ge::GRAPH_FAILED);
    }

    // Check all outputs
    for (int64_t i = GRAD_X_OUTPUT_INDEX; i <= GRAD_H_POST_OUTPUT_INDEX; i++) {
        auto shape = context_->GetOutputShape(i)->GetStorageShape();
        for (size_t j = 0; j < shape.GetDimNum(); j++) {
            OP_CHECK_IF(shape.GetDim(j) <= 0,
                        OP_LOGE(context_, "output %ld has non-positive shape, dim %lu actual %ld",
                                i, j, shape.GetDim(j)),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckDataType()
{
    // Get grad_output dtype as reference
    dtype_ = context_->GetInputDesc(GRAD_OUTPUT_INPUT_INDEX)->GetDataType();

    // Check supported dtype
    const std::vector<ge::DataType> supportedDtype = {ge::DT_BF16, ge::DT_FLOAT16};
    OP_CHECK_IF(std::find(supportedDtype.begin(), supportedDtype.end(), dtype_) == supportedDtype.end(),
                OP_LOGE(context_, "Only support BF16 and FP16 dtype, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str()),
                return ge::GRAPH_FAILED);

    // Check grad_output, x, h_out have same dtype (bf16/fp16)
    auto xType = context_->GetInputDesc(X_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(xType != dtype_,
                OP_LOGE(context_, "x datatype expect %s, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(xType).c_str()),
                return ge::GRAPH_FAILED);

    auto hOutType = context_->GetInputDesc(H_OUT_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hOutType != dtype_,
                OP_LOGE(context_, "h_out datatype expect %s, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(hOutType).c_str()),
                return ge::GRAPH_FAILED);

    // Check h_res and h_post are float32
    auto hResType = context_->GetInputDesc(H_RES_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hResType != ge::DT_FLOAT,
                OP_LOGE(context_, "h_res datatype must be float32, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(hResType).c_str()),
                return ge::GRAPH_FAILED);

    auto hPostType = context_->GetInputDesc(H_POST_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hPostType != ge::DT_FLOAT,
                OP_LOGE(context_, "h_post datatype must be float32, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(hPostType).c_str()),
                return ge::GRAPH_FAILED);

    // Check output dtypes match input dtypes
    auto gradXType = context_->GetOutputDesc(GRAD_X_OUTPUT_INDEX)->GetDataType();
    OP_CHECK_IF(gradXType != dtype_,
                OP_LOGE(context_, "grad_x datatype expect %s, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradXType).c_str()),
                return ge::GRAPH_FAILED);

    auto gradHOutType = context_->GetOutputDesc(GRAD_H_OUT_OUTPUT_INDEX)->GetDataType();
    OP_CHECK_IF(gradHOutType != dtype_,
                OP_LOGE(context_, "grad_h_out datatype expect %s, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradHOutType).c_str()),
                return ge::GRAPH_FAILED);

    auto gradHResType = context_->GetOutputDesc(GRAD_H_RES_OUTPUT_INDEX)->GetDataType();
    OP_CHECK_IF(gradHResType != ge::DT_FLOAT,
                OP_LOGE(context_, "grad_h_res datatype must be float32, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(gradHResType).c_str()),
                return ge::GRAPH_FAILED);

    auto gradHPostType = context_->GetOutputDesc(GRAD_H_POST_OUTPUT_INDEX)->GetDataType();
    OP_CHECK_IF(gradHPostType != ge::DT_FLOAT,
                OP_LOGE(context_, "grad_h_post datatype must be float32, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(gradHPostType).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckShapeConsistency()
{
    OP_CHECK_IF(gradOutputShape_ == nullptr,
                OP_LOGE(context_, "grad_output shape is null"),
                return ge::GRAPH_FAILED);

    // Support both BSND (4D) and TND (3D) formats
    // BSND: (B, S, n, D) -> totalItems = B * S
    // TND:  (T, n, D)    -> totalItems = T
    uint32_t dimNum = gradOutputShape_->GetDimNum();
    if (dimNum == 4) {
        // BSND format: (B, S, n, D)
        int64_t B_int = gradOutputShape_->GetDim(0);
        int64_t S_int = gradOutputShape_->GetDim(1);
        int64_t n_int = gradOutputShape_->GetDim(2);
        int64_t D_int = gradOutputShape_->GetDim(3);

        B_ = static_cast<uint32_t>(B_int);
        S_ = static_cast<uint32_t>(S_int);
        n_ = static_cast<uint32_t>(n_int);
        D_ = static_cast<uint32_t>(D_int);
        totalItems_ = B_ * S_;
        OP_LOGI(context_, "BSND format: B=%u, S=%u, n=%u, D=%u, totalItems=%u", B_, S_, n_, D_, totalItems_);
    } else if (dimNum == 3) {
        // TND format: (T, n, D)
        int64_t T_int = gradOutputShape_->GetDim(0);
        int64_t n_int = gradOutputShape_->GetDim(1);
        int64_t D_int = gradOutputShape_->GetDim(2);

        B_ = 1;  // Not used in TND format
        S_ = 1;  // Not used in TND format
        totalItems_ = static_cast<uint32_t>(T_int);
        n_ = static_cast<uint32_t>(n_int);
        D_ = static_cast<uint32_t>(D_int);
        OP_LOGI(context_, "TND format: T=%u, n=%u, D=%u", totalItems_, n_, D_);
    } else {
        OP_LOGE(context_, "Unsupported input dimension: %u (expected 3 for TND or 4 for BSND)", dimNum);
        return ge::GRAPH_FAILED;
    }

    // Cross-validate all input shapes to ensure consistency
    // Input: grad_output(0), x(1), h_res(2), h_out(3), h_post(4)
    // Expected shapes:
    //   BSND (4D): grad_output(B,S,n,D), x(B,S,n,D), h_res(B,S,n,n), h_out(B,S,D), h_post(B,S,n)
    //   TND (3D):  grad_output(T,n,D),   x(T,n,D),   h_res(T,n,n),   h_out(T,D),   h_post(T,n)

    if (dimNum == 4) {
        // BSND format validation
        int64_t B = static_cast<int64_t>(B_);
        int64_t S = static_cast<int64_t>(S_);
        int64_t n = static_cast<int64_t>(n_);
        int64_t D = static_cast<int64_t>(D_);

        // Validate x: (B, S, n, D)
        auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
        const gert::Shape* xShape = &xShapePtr->GetStorageShape();
        OP_CHECK_IF(xShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "x has %u dimensions, expected %u (format mismatch)",
                            xShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(xShape->GetDim(0) != B || xShape->GetDim(1) != S ||
                    xShape->GetDim(2) != n || xShape->GetDim(3) != D,
                    OP_LOGE(context_, "x shape (%ld,%ld,%ld,%ld) != expected (%ld,%ld,%ld,%ld)",
                            xShape->GetDim(0), xShape->GetDim(1), xShape->GetDim(2), xShape->GetDim(3),
                            B, S, n, D),
                    return ge::GRAPH_FAILED);

        // Validate h_res: (B, S, n, n)
        auto hResShapePtr = context_->GetInputShape(H_RES_INPUT_INDEX);
        const gert::Shape* hResShape = &hResShapePtr->GetStorageShape();
        OP_CHECK_IF(hResShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "h_res has %u dimensions, expected %u (format mismatch)",
                            hResShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hResShape->GetDim(0) != B || hResShape->GetDim(1) != S ||
                    hResShape->GetDim(2) != n || hResShape->GetDim(3) != n,
                    OP_LOGE(context_, "h_res shape (%ld,%ld,%ld,%ld) != expected (%ld,%ld,%ld,%ld)",
                            hResShape->GetDim(0), hResShape->GetDim(1), hResShape->GetDim(2), hResShape->GetDim(3),
                            B, S, n, n),
                    return ge::GRAPH_FAILED);

        // Validate h_out: (B, S, D)
        auto hOutShapePtr = context_->GetInputShape(H_OUT_INPUT_INDEX);
        const gert::Shape* hOutShape = &hOutShapePtr->GetStorageShape();
        OP_CHECK_IF(hOutShape->GetDimNum() != 3,
                    OP_LOGE(context_, "h_out has %u dimensions, expected 3",
                            hOutShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hOutShape->GetDim(0) != B || hOutShape->GetDim(1) != S || hOutShape->GetDim(2) != D,
                    OP_LOGE(context_, "h_out shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            hOutShape->GetDim(0), hOutShape->GetDim(1), hOutShape->GetDim(2),
                            B, S, D),
                    return ge::GRAPH_FAILED);

        // Validate h_post: (B, S, n)
        auto hPostShapePtr = context_->GetInputShape(H_POST_INPUT_INDEX);
        const gert::Shape* hPostShape = &hPostShapePtr->GetStorageShape();
        OP_CHECK_IF(hPostShape->GetDimNum() != 3,
                    OP_LOGE(context_, "h_post has %u dimensions, expected 3",
                            hPostShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hPostShape->GetDim(0) != B || hPostShape->GetDim(1) != S || hPostShape->GetDim(2) != n,
                    OP_LOGE(context_, "h_post shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            hPostShape->GetDim(0), hPostShape->GetDim(1), hPostShape->GetDim(2),
                            B, S, n),
                    return ge::GRAPH_FAILED);

        // Validate output shapes for BSND format
        // Output: grad_x(0), grad_h_res(1), grad_h_out(2), grad_h_post(3)
        // Expected shapes: grad_x(B,S,n,D), grad_h_res(B,S,n,n), grad_h_out(B,S,D), grad_h_post(B,S,n)

        // Validate grad_x: (B, S, n, D)
        auto gradXShapePtr = context_->GetOutputShape(GRAD_X_OUTPUT_INDEX);
        const gert::Shape* gradXShape = &gradXShapePtr->GetStorageShape();
        OP_CHECK_IF(gradXShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "grad_x has %u dimensions, expected %u (format mismatch)",
                            gradXShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradXShape->GetDim(0) != B || gradXShape->GetDim(1) != S ||
                    gradXShape->GetDim(2) != n || gradXShape->GetDim(3) != D,
                    OP_LOGE(context_, "grad_x shape (%ld,%ld,%ld,%ld) != expected (%ld,%ld,%ld,%ld)",
                            gradXShape->GetDim(0), gradXShape->GetDim(1), gradXShape->GetDim(2), gradXShape->GetDim(3),
                            B, S, n, D),
                    return ge::GRAPH_FAILED);

        // Validate grad_h_res: (B, S, n, n)
        auto gradHResShapePtr = context_->GetOutputShape(GRAD_H_RES_OUTPUT_INDEX);
        const gert::Shape* gradHResShape = &gradHResShapePtr->GetStorageShape();
        OP_CHECK_IF(gradHResShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "grad_h_res has %u dimensions, expected %u (format mismatch)",
                            gradHResShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradHResShape->GetDim(0) != B || gradHResShape->GetDim(1) != S ||
                    gradHResShape->GetDim(2) != n || gradHResShape->GetDim(3) != n,
                    OP_LOGE(context_, "grad_h_res shape (%ld,%ld,%ld,%ld) != expected (%ld,%ld,%ld,%ld)",
                            gradHResShape->GetDim(0), gradHResShape->GetDim(1), gradHResShape->GetDim(2),
                            gradHResShape->GetDim(3), B, S, n, n),
                    return ge::GRAPH_FAILED);

        // Validate grad_h_out: (B, S, D)
        auto gradHOutShapePtr = context_->GetOutputShape(GRAD_H_OUT_OUTPUT_INDEX);
        const gert::Shape* gradHOutShape = &gradHOutShapePtr->GetStorageShape();
        OP_CHECK_IF(gradHOutShape->GetDimNum() != 3,
                    OP_LOGE(context_, "grad_h_out has %u dimensions, expected 3",
                            gradHOutShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradHOutShape->GetDim(0) != B || gradHOutShape->GetDim(1) != S || gradHOutShape->GetDim(2) != D,
                    OP_LOGE(context_, "grad_h_out shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            gradHOutShape->GetDim(0), gradHOutShape->GetDim(1), gradHOutShape->GetDim(2),
                            B, S, D),
                    return ge::GRAPH_FAILED);

        // Validate grad_h_post: (B, S, n)
        auto gradHPostShapePtr = context_->GetOutputShape(GRAD_H_POST_OUTPUT_INDEX);
        const gert::Shape* gradHPostShape = &gradHPostShapePtr->GetStorageShape();
        OP_CHECK_IF(gradHPostShape->GetDimNum() != 3,
                    OP_LOGE(context_, "grad_h_post has %u dimensions, expected 3",
                            gradHPostShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradHPostShape->GetDim(0) != B || gradHPostShape->GetDim(1) != S || gradHPostShape->GetDim(2) != n,
                    OP_LOGE(context_, "grad_h_post shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            gradHPostShape->GetDim(0), gradHPostShape->GetDim(1), gradHPostShape->GetDim(2),
                            B, S, n),
                    return ge::GRAPH_FAILED);
    } else {  // dimNum == 3
        // TND format validation
        int64_t T = static_cast<int64_t>(totalItems_);
        int64_t n = static_cast<int64_t>(n_);
        int64_t D = static_cast<int64_t>(D_);

        // Validate x: (T, n, D)
        auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
        const gert::Shape* xShape = &xShapePtr->GetStorageShape();
        OP_CHECK_IF(xShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "x has %u dimensions, expected %u (format mismatch)",
                            xShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(xShape->GetDim(0) != T || xShape->GetDim(1) != n || xShape->GetDim(2) != D,
                    OP_LOGE(context_, "x shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            xShape->GetDim(0), xShape->GetDim(1), xShape->GetDim(2),
                            T, n, D),
                    return ge::GRAPH_FAILED);

        // Validate h_res: (T, n, n)
        auto hResShapePtr = context_->GetInputShape(H_RES_INPUT_INDEX);
        const gert::Shape* hResShape = &hResShapePtr->GetStorageShape();
        OP_CHECK_IF(hResShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "h_res has %u dimensions, expected %u (format mismatch)",
                            hResShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hResShape->GetDim(0) != T || hResShape->GetDim(1) != n || hResShape->GetDim(2) != n,
                    OP_LOGE(context_, "h_res shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            hResShape->GetDim(0), hResShape->GetDim(1), hResShape->GetDim(2),
                            T, n, n),
                    return ge::GRAPH_FAILED);

        // Validate h_out: (T, D)
        auto hOutShapePtr = context_->GetInputShape(H_OUT_INPUT_INDEX);
        const gert::Shape* hOutShape = &hOutShapePtr->GetStorageShape();
        OP_CHECK_IF(hOutShape->GetDimNum() != 2,
                    OP_LOGE(context_, "h_out has %u dimensions, expected 2",
                            hOutShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hOutShape->GetDim(0) != T || hOutShape->GetDim(1) != D,
                    OP_LOGE(context_, "h_out shape (%ld,%ld) != expected (%ld,%ld)",
                            hOutShape->GetDim(0), hOutShape->GetDim(1),
                            T, D),
                    return ge::GRAPH_FAILED);

        // Validate h_post: (T, n)
        auto hPostShapePtr = context_->GetInputShape(H_POST_INPUT_INDEX);
        const gert::Shape* hPostShape = &hPostShapePtr->GetStorageShape();
        OP_CHECK_IF(hPostShape->GetDimNum() != 2,
                    OP_LOGE(context_, "h_post has %u dimensions, expected 2",
                            hPostShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hPostShape->GetDim(0) != T || hPostShape->GetDim(1) != n,
                    OP_LOGE(context_, "h_post shape (%ld,%ld) != expected (%ld,%ld)",
                            hPostShape->GetDim(0), hPostShape->GetDim(1),
                            T, n),
                    return ge::GRAPH_FAILED);

        // Validate output shapes for TND format
        // Output: grad_x(0), grad_h_res(1), grad_h_out(2), grad_h_post(3)
        // Expected shapes: grad_x(T,n,D), grad_h_res(T,n,n), grad_h_out(T,D), grad_h_post(T,n)

        // Validate grad_x: (T, n, D)
        auto gradXShapePtr = context_->GetOutputShape(GRAD_X_OUTPUT_INDEX);
        const gert::Shape* gradXShape = &gradXShapePtr->GetStorageShape();
        OP_CHECK_IF(gradXShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "grad_x has %u dimensions, expected %u (format mismatch)",
                            gradXShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradXShape->GetDim(0) != T || gradXShape->GetDim(1) != n || gradXShape->GetDim(2) != D,
                    OP_LOGE(context_, "grad_x shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            gradXShape->GetDim(0), gradXShape->GetDim(1), gradXShape->GetDim(2),
                            T, n, D),
                    return ge::GRAPH_FAILED);

        // Validate grad_h_res: (T, n, n)
        auto gradHResShapePtr = context_->GetOutputShape(GRAD_H_RES_OUTPUT_INDEX);
        const gert::Shape* gradHResShape = &gradHResShapePtr->GetStorageShape();
        OP_CHECK_IF(gradHResShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "grad_h_res has %u dimensions, expected %u (format mismatch)",
                            gradHResShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradHResShape->GetDim(0) != T || gradHResShape->GetDim(1) != n || gradHResShape->GetDim(2) != n,
                    OP_LOGE(context_, "grad_h_res shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            gradHResShape->GetDim(0), gradHResShape->GetDim(1), gradHResShape->GetDim(2),
                            T, n, n),
                    return ge::GRAPH_FAILED);

        // Validate grad_h_out: (T, D)
        auto gradHOutShapePtr = context_->GetOutputShape(GRAD_H_OUT_OUTPUT_INDEX);
        const gert::Shape* gradHOutShape = &gradHOutShapePtr->GetStorageShape();
        OP_CHECK_IF(gradHOutShape->GetDimNum() != 2,
                    OP_LOGE(context_, "grad_h_out has %u dimensions, expected 2",
                            gradHOutShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradHOutShape->GetDim(0) != T || gradHOutShape->GetDim(1) != D,
                    OP_LOGE(context_, "grad_h_out shape (%ld,%ld) != expected (%ld,%ld)",
                            gradHOutShape->GetDim(0), gradHOutShape->GetDim(1),
                            T, D),
                    return ge::GRAPH_FAILED);

        // Validate grad_h_post: (T, n)
        auto gradHPostShapePtr = context_->GetOutputShape(GRAD_H_POST_OUTPUT_INDEX);
        const gert::Shape* gradHPostShape = &gradHPostShapePtr->GetStorageShape();
        OP_CHECK_IF(gradHPostShape->GetDimNum() != 2,
                    OP_LOGE(context_, "grad_h_post has %u dimensions, expected 2",
                            gradHPostShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradHPostShape->GetDim(0) != T || gradHPostShape->GetDim(1) != n,
                    OP_LOGE(context_, "grad_h_post shape (%ld,%ld) != expected (%ld,%ld)",
                            gradHPostShape->GetDim(0), gradHPostShape->GetDim(1),
                            T, n),
                    return ge::GRAPH_FAILED);
    }

    OP_LOGI(context_, "All input and output shapes validated successfully");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckSpecConstraints()
{
    OP_CHECK_IF(totalItems_ == 0,
                OP_LOGE(context_, "totalItems cannot be zero"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(n_ != 4 && n_ != 6 && n_ != 8,
                OP_LOGE(context_, "n (%u) must be 4, 6, or 8", n_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckParam()
{
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckNullptr failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDataType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckDataType failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeConsistency() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckShapeConsistency failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckSpecConstraints() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckSpecConstraints failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeAllPositive() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckShapeAllPositive failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::ComputeTiling()
{
    // Core Partitioning - handle remainder properly
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    usedCores_ = (totalItems_ < (coreNum * 2)) ? totalItems_ : (coreNum * 2);
    itemsPerCore_ = totalItems_ / usedCores_;
    remainderItems_ = totalItems_ % usedCores_;
    usedAic_ = (totalItems_ < coreNum) ? totalItems_ : coreNum;
    itemsPerAic_ = totalItems_ / usedAic_;
    remainderItemsAic_ = totalItems_ % usedAic_;

    // Dynamic Tiling Strategy - maximize tileD based on UB size
    // UB usage formula (generalized for any n):
    // TQue (bf16): (2n+2) * tileD * 2 bytes
    //   - gradOutTile: n*tileD, hOutTile: tileD
    //   - gradHOutTile: tileD, gradXTile: n*tileD
    // TQue (f32): (2n+n*n) * sizeof(float) bytes (small, n-related buffers)
    //   - hPost: n, hRes: n*n, gradHPost: n,
    // TBuf (f32): n * tileD * 4 bytes
    //   - gradOutF32: n*tileD
    // Simplified formula: bytesPerTileD = (2n+2)*2 + n*4 = 4n+4 + 4n = 8n+4
    // More accurate: include n-related small buffers separately
    const uint32_t UB_SIZE = static_cast<uint32_t>(aicoreParams_.ubSize);

    // Calculate aligned n and n*n for float32 vector operations
    uint32_t alignedN = AlignUp(n_, FLOAT32_ALIGN_SIZE);
    uint32_t alignedNN = AlignUp(n_ * n_, FLOAT32_ALIGN_SIZE);

    // Calculate bytes per tileD element based on actual n
    // TQue bf16: (2n+2) * 2 bytes (gradOut:n, hOut:1, gradHOut:1, gradX:n)
    // TBuf f32:  n * 4 bytes (gradOutF32:n)
    // Small buffers (fixed, not per-tileD):
    //   - TQue f32: (2*alignedN + alignedNN) * 4 for hPost, hRes, gradHPost
    uint32_t bytesPerTileD = (2 * n_ + 2) * 2 + (n_ + 3) * 4;  // = 8n + 4
    uint32_t smallBufferBytes = (3 * alignedN + 2 * alignedNN) * 4;  // aligned sizes

    // Reserve space for small buffers, then calculate max tileD
    uint32_t availableUB = UB_SIZE - smallBufferBytes;
    uint32_t maxTileD = availableUB / bytesPerTileD;
    maxTileD = AlignDown(maxTileD, BF16_FP16_ALIGN_SIZE);  // Align to 16 elements

    // Align D to BF16_FP16_ALIGN_SIZE for proper memory access
    uint32_t alignedD = AlignUp(D_, BF16_FP16_ALIGN_SIZE);
    // Determine optimal tileD:
    if (alignedD <= maxTileD) {
        tileD_ = alignedD;
        nTilesD_ = 1;
    } else {
        // 直接使用 maxTileD，允许最后一个 tile 较小
        tileD_ = maxTileD;
        nTilesD_ = CeilDiv(alignedD, tileD_);
    }

    // Calculate last tile size
    uint32_t lastTileD = alignedD - (nTilesD_ - 1) * tileD_;

    // Calculate fast path flags
    // For n: only n=8 (32 bytes) can use DataCopy directly
    // n=4 (16 bytes) and n=6 (24 bytes) need padding to 32 bytes
    isNAligned_ = (n_ == 8) ? 1 : 0;
    // For n*n: 16 (n=4) and 64 (n=8) are 8-aligned, 36 (n=6) is not
    isNNAligned_ = ((n_ * n_) % FLOAT32_ALIGN_SIZE == 0) ? 1 : 0;
    // D is aligned if D % 16 == 0 and single tile
    isDAligned_ = ((D_ % BF16_FP16_ALIGN_SIZE == 0) && (nTilesD_ == 1)) ? 1 : 0;

    MhcPostBackwardTilingDataArch35 *tilingData_ = context_->GetTilingData<MhcPostBackwardTilingDataArch35>();
    // Set tiling data
    tilingData_->totalItems = totalItems_;
    tilingData_->itemsPerCore = itemsPerCore_;
    tilingData_->remainderItems = remainderItems_;
    tilingData_->usedCores = usedCores_;
    tilingData_->S = S_;
    tilingData_->n = n_;
    tilingData_->D = D_;
    tilingData_->tileD = tileD_;
    tilingData_->nTilesD = nTilesD_;
    tilingData_->alignedD = alignedD;
    tilingData_->lastTileD = lastTileD;
    tilingData_->alignedN = alignedN;
    tilingData_->alignedNN = alignedNN;
    tilingData_->itemsPerAic = itemsPerAic_;
    tilingData_->remainderItemsAic = remainderItemsAic_;
    auto xDtype = context_->GetInputDesc(X_INPUT_INDEX)->GetDataType();
    matmul_tiling::MultiCoreMatmulTiling mm(ascendcPlatform);
    matmul_tiling::DataType halfDtype = matmul_tiling::DataType::DT_FLOAT16;
    if (xDtype == ge::DT_BF16) {
        halfDtype = matmul_tiling::DataType::DT_BFLOAT16;
    }
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype, true);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetShape(n_, n_, D_);
    mm.SetOrgShape(n_, n_, D_);
    mm.EnableBias(false);
    mm.SetBufferSpace(-1, -1, -1);
    if (mm.GetTiling(tilingData_->matmulTiling) == -1) {
        OP_LOGE(context_, "fail to get matmul tiling");
        return ge::GRAPH_FAILED;
    }
    tilingData_->matmulTiling.usedCoreNum = usedAic_;
    OP_LOGI(context_,
        "Tiling: n=%u, D=%u, alignedD=%u, tileD=%u, lastTileD=%u, nTilesD=%u",
        n_, D_, alignedD, tileD_, lastTileD, nTilesD_);
    OP_LOGI(context_,
        "Tiling: alignedN=%u, alignedNN=%u, isNAligned=%u, isNNAligned=%u, isDAligned=%u",
        alignedN, alignedNN, isNAligned_, isNNAligned_, isDAligned_);
    OP_LOGI(context_,
        "Tiling: usedCores=%u, itemsPerCore=%u, remainderItems=%u, UB=%u, bytesPerTileD=%u",
        usedCores_, itemsPerCore_, remainderItems_, UB_SIZE, bytesPerTileD);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::DoOpTiling()
{
    auto ret = CheckParam();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ComputeTiling();
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_, "fail to get platform info");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGI(context_, "Workspace size: %ld bytes (%.2f MB)",
            workspaceSize_, workspaceSize_ / (1024.0 * 1024.0));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::PostTiling()
{
    context_->SetBlockDim(usedAic_);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcPostBackwardTilingBaseArch35::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(1);
}

void MhcPostBackwardTilingBaseArch35::Reset()
{
    opName_ = nullptr;
    gradOutputShape_ = nullptr;
    B_ = 0;
    S_ = 0;
    n_ = 0;
    D_ = 0;
    totalItems_ = 0;
    usedCores_ = 0;
    itemsPerCore_ = 0;
    remainderItems_ = 0;
    tileD_ = 0;
    nTilesD_ = 0;
    usedAic_ = 0;
    itemsPerAic_ = 0;
    remainderItemsAic_ = 0;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(MhcPostBackward, MhcPostBackwardTilingBaseArch35,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 0);
} // namespace optiling
