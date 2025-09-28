/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_legacy_api.cpp
 * \brief
 */

#include "opdev/op_executor.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/add.h"
#include "level0/axpy.h"
#include "level0/broadcast_to.h"
#include "level0/dot.h"
#include "level0/fill.h"
#include "level0/mul.h"
#include "level0/muls.h"
#include "level0/reduce_mean.h"
#include "level0/padv3.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "level0/adaptive_avg_pool2d_assist_matrix.h"
#include "level0/max_pool3d_with_argmax_v2.h"
#include "level0/max_pool3d_grad_with_argmax.h"
#include "level0/max_pool_grad_with_argmax_v1.h"
#include "level0/max_pool_grad_with_argmax_v3.h"
namespace l0op {
const aclTensor* Cast(const aclTensor* /*self*/, op::DataType /*dstDtype*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* MaxPoolGradWithArgmaxV1(
    const aclTensor* /*gradOutput*/, const aclTensor* /*self*/, const aclTensor* /*indices*/, const aclIntArray* /*kernelSize*/,
    const aclIntArray* /*stride*/, const aclIntArray* /*padding*/, const aclIntArray* /*dilation*/, bool /*ceilMode*/,
    aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* MaxPoolGradWithArgmaxV3(
    const aclTensor* /*gradOutput*/, const aclTensor* /*self*/, const aclTensor* /*indices*/, const aclIntArray* /*kernelSize*/,
    const aclIntArray* /*stride*/, const aclIntArray* /*padding*/, const ge::DataType /*dtype*/, const aclIntArray* /*dilation*/,
    bool /*ceilMode*/, std::string& /*dataFormat*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* CastOnlyForConvBackward(
    const aclTensor* /*self*/, op::DataType /*dstDtype*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* Contiguous(const aclTensor* /*x*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* ViewCopy(const aclTensor* /*x*/, const aclTensor* /*y*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* PickViewAsContiguous(const aclTensor* /*x*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* ReViewToOut(const aclTensor* /*x*/, const aclTensor* /*y*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

bool CanOptimizeContiguous(
    const op::Shape& /*viewShape*/, const op::Strides& /*strides*/, int64_t /*offset*/, int64_t /*storageSize*/,
    ContiguousParam& /*param*/)
{
    return true;
}

bool CanOptimizeView(
    const op::Shape& /*viewShape*/, const op::Strides& /*strides*/, int64_t /*offset*/, ContiguousParam& /*param*/)
{
    return true;
}

const aclTensor* Pad(const aclTensor* /*self*/, const aclTensor* /*paddings*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* Reshape(const aclTensor* /*x*/, const op::Shape& /*shape*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* Reshape(const aclTensor* /*x*/, const aclIntArray* /*shape*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* Slice(
    const aclTensor* /*x*/, const aclTensor* /*y*/, const aclTensor* /*offset*/, const aclTensor* /*size*/,
    aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* Slice(
    const aclTensor* /*x*/, const aclIntArray* /*offsets*/, const aclIntArray* /*size*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* ReFormat(const aclTensor* /*x*/, const op::Format& /*format*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* TransData(
    const aclTensor* /*x*/, op::Format /*dstPrimaryFormat*/, int64_t /*groups*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* TransDataSpecial(
    const aclTensor* /*x*/, op::Format /*dstPrimaryFormat*/, int64_t /*groups*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* Transpose(
    const aclTensor* /*x*/, const aclTensor* /*y*/, const aclTensor* /*perm*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* Transpose(const aclTensor* /*x*/, const aclIntArray* /*perm*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* Add(const aclTensor* /*self*/, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* Axpy(
    const aclTensor* /*self*/, const aclTensor* /*other*/, float /*alpha*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* BroadcastTo(
    const aclTensor* /*x*/, const aclTensor* /*y*/, const aclTensor* /*shape*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* BroadcastTo(const aclTensor* /*x*/, const aclIntArray* /*shape*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* Dot(const aclTensor* /*self*/, const aclTensor* /*tensor*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* Fill(
    const aclTensor* /*dims*/, const aclTensor* /*value*/, const aclIntArray* /*outShape*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* Mul(const aclTensor* /*self*/, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* Muls(const aclTensor* /*self*/, float /*alpha*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* ReduceMean(const aclTensor* /*self*/, const aclIntArray* /*dim*/, bool /*keepDim*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* ReduceMean(
    const aclTensor* /*self*/, const aclIntArray* /*dim*/, bool /*keepDim*/, bool /*noopWithEmptyAxes*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* Shape_op(const aclTensor* /*x*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const std::array<aclTensor*, 3> AdaptiveAvgPool2dAssistMatrix(
    const aclTensor* /*input*/, const aclTensor* /*origin_input*/, const aclIntArray* /*output_size*/, aclOpExecutor* /*executor*/)
{
    return {nullptr, nullptr, nullptr};
}
const aclTensor* SqueezeNd(const aclTensor* /*x*/, const aclIntArray* /*dim*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* SqueezeNd(const aclTensor* /*x*/, int64_t /*dim*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const std::tuple<const aclTensor*, const aclTensor*> MaxPool3DWithArgmaxV2Ncdhw(
    const aclTensor* /*self*/, const aclIntArray* /*kernelSize*/, const aclIntArray* /*stride*/, const aclIntArray* /*padding*/,
    const aclIntArray* /*dilation*/, bool /*ceilMode*/, std::string /*dataFormat*/, aclOpExecutor* /*executor*/)
{
    return std::make_tuple(nullptr, nullptr);
}
const aclTensor* MaxPool3DGradWithArgmax(
    const aclTensor* /*gradOutput*/, const aclTensor* /*self*/, const aclTensor* /*indices*/, const aclIntArray* /*kernelSize*/,
    const aclIntArray* /*stride*/, const aclIntArray* /*padding*/, const aclIntArray* /*dilation*/, bool /*ceilMode*/,
    aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* PadV3(
    const aclTensor* /*self*/, const aclTensor* /*paddings*/, const aclTensor* /*constant_values*/,
    const std::string& /*mode*/, const bool /*paddingsContiguous*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}
const aclTensor* UnsqueezeNd(const aclTensor* /*x*/, const aclIntArray* /*dim*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* UnsqueezeNd(const aclTensor* /*x*/, int64_t /*dim*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* ReduceSumOp(
    const aclTensor* /*x*/, const aclIntArray* /*axes*/, bool /*keep_dims*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

} // namespace l0op