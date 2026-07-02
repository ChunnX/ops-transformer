/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_types.h
 * \brief Core type definitions, enums, constants, and basic utilities for ACLNN operator wrappers.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_TYPES_H
#define CANN_OPS_TRANSFORMER_ACLNN_TYPES_H

#include <vector>
#include <string>
#include <unordered_map>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/interface/EnvVariables.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#if __has_include("torch_npu/csrc/flopcount/FlopCount.h")
#include "torch_npu/csrc/flopcount/FlopCount.h"
#endif

#define NPU_NAME_SPACE at_npu::native

using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
using aclScalar = struct aclScalar;
using aclIntArray = struct aclIntArray;
using aclFloatArray = struct aclFloatArray;
using aclBoolArray = struct aclBoolArray;
using aclTensorList = struct aclTensorList;

using _aclCreateTensor = aclTensor *(*)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
                                        const int64_t *stride, int64_t offset, aclFormat format, const int64_t *storage_dims, uint64_t storage_dims_num,
    void *tensor_data);
using _aclCreateScalar = aclScalar *(*)(void *value, aclDataType data_type);
using _aclCreateIntArray = aclIntArray *(*)(const int64_t *value, uint64_t size);
using _aclCreateFloatArray = aclFloatArray *(*)(const float *value, uint64_t size);
using _aclCreateBoolArray = aclBoolArray *(*)(const bool *value, uint64_t size);
using _aclCreateTensorList = aclTensorList *(*)(const aclTensor *const *value, uint64_t size);

using _aclDestroyTensor = int (*)(const aclTensor *tensor);
using _aclDestroyScalar = int (*)(const aclScalar *scalar);
using _aclDestroyIntArray = int (*)(const aclIntArray *array);
using _aclDestroyFloatArray = int (*)(const aclFloatArray *array);
using _aclDestroyBoolArray = int (*)(const aclBoolArray *array);
using _aclDestroyTensorList = int (*)(const aclTensorList *array);

constexpr int64_t MAX_DIM_NUM = 5;
constexpr int64_t FP4_IN_INT8 = 2;
constexpr int64_t PENULTIMATE_DIM = 2;
const int g_toAclOffset = 256;

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)                                                                    \
    _(at::ScalarType::Byte, ACL_UINT8)                                                                                 \
    _(at::ScalarType::Char, ACL_INT8)                                                                                  \
    _(at::ScalarType::Short, ACL_INT16)                                                                                \
    _(at::ScalarType::Int, ACL_INT32)                                                                                  \
    _(at::ScalarType::Long, ACL_INT64)                                                                                 \
    _(at::ScalarType::Half, ACL_FLOAT16)                                                                               \
    _(at::ScalarType::Float, ACL_FLOAT)                                                                                \
    _(at::ScalarType::Double, ACL_DOUBLE)                                                                              \
    _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)                                                                      \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)                                                                     \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)                                                                   \
    _(at::ScalarType::Bool, ACL_BOOL)                                                                                  \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::BFloat16, ACL_BF16)                                                                              \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::Float8_e5m2, ACL_FLOAT8_E5M2)                                                                    \
    _(at::ScalarType::Float8_e4m3fn, ACL_FLOAT8_E4M3FN)                                                                \
    _(at::ScalarType::Float8_e5m2fnuz, ACL_DT_UNDEFINED)                                                               \
    _(at::ScalarType::Float8_e4m3fnuz, ACL_DT_UNDEFINED)                                                               \
    _(at::ScalarType::UInt16, ACL_UINT16)                                                                              \
    _(at::ScalarType::UInt32, ACL_UINT32)                                                                              \
    _(at::ScalarType::UInt64, ACL_UINT64)                                                                              \
    _(at::ScalarType::UInt1, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt2, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt3, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt4, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt5, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt6, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt7, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::Int1, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int2, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int3, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int4, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int5, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int6, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int7, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Float8_e8m0fnu, ACL_FLOAT8_E8M0)                                                                \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)                                                                     \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

aclDataType ConvertToAclDataType(const at::ScalarType &data_type)
{
    int64_t dtype_index = static_cast<int64_t>(data_type);
    TORCH_CHECK(dtype_index >= 0 && dtype_index < static_cast<int64_t>(at::ScalarType::NumOptions) + 1,
                "data_type enum value (",
                dtype_index,
                ") is out of range: [0, ",
                static_cast<int64_t>(at::ScalarType::NumOptions),
                "]")
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[dtype_index];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
                std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}

#define ENUM_OFFSET(new_name, old_name) new_name = static_cast<int>(old_name) + g_toAclOffset,

enum class DType {
    UNDEFINED = -1,
    ENUM_OFFSET(FLOAT, ACL_FLOAT)
    ENUM_OFFSET(FLOAT16, ACL_FLOAT16)
    ENUM_OFFSET(INT8, ACL_INT8)
    ENUM_OFFSET(INT32, ACL_INT32)
    ENUM_OFFSET(UINT8, ACL_UINT8)
    ENUM_OFFSET(INT16, ACL_INT16)
    ENUM_OFFSET(UINT16, ACL_UINT16)
    ENUM_OFFSET(UINT32, ACL_UINT32)
    ENUM_OFFSET(INT64, ACL_INT64)
    ENUM_OFFSET(UINT64, ACL_UINT64)
    ENUM_OFFSET(DOUBLE, ACL_DOUBLE)
    ENUM_OFFSET(BOOL, ACL_BOOL)
    ENUM_OFFSET(STRING, ACL_STRING)
    ENUM_OFFSET(COMPLEX64, ACL_COMPLEX64)
    ENUM_OFFSET(COMPLEX128, ACL_COMPLEX128)
    ENUM_OFFSET(BF16, ACL_BF16)
    ENUM_OFFSET(INT4, ACL_INT4)
    ENUM_OFFSET(UINT1, ACL_UINT1)
    ENUM_OFFSET(COMPLEX32, ACL_COMPLEX32)
    ENUM_OFFSET(HIFLOAT8, ACL_HIFLOAT8)
    ENUM_OFFSET(FLOAT8_E5M2, ACL_FLOAT8_E5M2)
    ENUM_OFFSET(FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN)
    ENUM_OFFSET(FLOAT8_E8M0, ACL_FLOAT8_E8M0)
    ENUM_OFFSET(FLOAT6_E3M2, ACL_FLOAT6_E3M2)
    ENUM_OFFSET(FLOAT6_E2M3, ACL_FLOAT6_E2M3)
    ENUM_OFFSET(FLOAT4_E2M1, ACL_FLOAT4_E2M1)
    ENUM_OFFSET(FLOAT4_E1M2, ACL_FLOAT4_E1M2)
};

namespace op_infer {
const int N = 32;
const int SIZE = 8;

inline c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }
    return shape_small_vec;
}
}

typedef struct {
    const at::Tensor& tensor_;
    aclDataType dtype;
} TensorWrapper;

typedef struct {
    const at::TensorList& tensor_list_;
    aclDataType dtype;
} TensorListWrapper;

inline bool Is4BitDtype(const aclDataType acl_data_type)
{
    return acl_data_type == ACL_FLOAT4_E2M1 || acl_data_type == ACL_FLOAT4_E1M2 || acl_data_type == ACL_INT4;
}

static std::unordered_map<aclFormat, aclFormat> FORMAT_FAKE_TO_REAL {
    { ACL_FORMAT_FRACTAL_NZ_C0_16, ACL_FORMAT_FRACTAL_NZ_C0_32 },
    { ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ }
};

static inline bool IsOpInputBaseFormat(const at::Tensor &at_tensor)
{
    if (!torch_npu::utils::is_npu(at_tensor)) {
        return true;
    }
    const auto format = static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
    return (format == ACL_FORMAT_ND) || (format == ACL_FORMAT_NCHW) ||
           (format == ACL_FORMAT_NHWC) || (format == ACL_FORMAT_NCDHW);
}

inline void CollectB4ShapeInfo(const at::Tensor &at_tensor,
                               c10::SmallVector<int64_t, MAX_DIM_NUM>& wrapperStride,
                               c10::SmallVector<int64_t, MAX_DIM_NUM>& wrapperShape)
{
    int64_t nDim = at_tensor.sizes().size();
    if (nDim == 1) {
        wrapperShape[0] = wrapperShape[0] * FP4_IN_INT8;
    } else if (nDim > 1) {
        if (wrapperStride[nDim - 1] == 1 && wrapperStride[nDim - PENULTIMATE_DIM] == 1) {
            if (wrapperShape[nDim - PENULTIMATE_DIM] == 1) {
                wrapperStride[nDim - 1] = wrapperStride[nDim - 1] * FP4_IN_INT8;
                wrapperShape[nDim - PENULTIMATE_DIM] = wrapperShape[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
            } else if (wrapperShape[nDim - 1] == 1) {
                wrapperStride[nDim - PENULTIMATE_DIM] = wrapperStride[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
                wrapperShape[nDim - 1] = wrapperShape[nDim - 1] * FP4_IN_INT8;
            }
        } else if (wrapperStride[nDim - 1] == 1) {
            wrapperStride[nDim - PENULTIMATE_DIM] =
                wrapperStride[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
            wrapperShape[nDim - 1] = wrapperShape[nDim - 1] * FP4_IN_INT8;
        } else if (wrapperStride[nDim - PENULTIMATE_DIM] == 1) {
            wrapperStride[nDim - 1] =
                wrapperStride[nDim - 1] * FP4_IN_INT8;
            wrapperShape[nDim - PENULTIMATE_DIM] =
                wrapperShape[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
        }

        for (auto i = 0; i < nDim - PENULTIMATE_DIM; i++) {
            wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
        }
    } else {
        TORCH_CHECK(false, "unsupported tensor size() in 4-bit dtype.");
    }
}

enum QuantMode {
    QUANT_MODE_NO_QUANT = 0,
    QUANT_MODE_STATIC = 1,
    QUANT_MODE_PERTOKEN = 2,
    QUANT_MODE_PERGROUP = 3,
    QUANT_MODE_MX = 4,
};

inline aclDataType GetAclDataType(int64_t t)
{
    if (t >= g_toAclOffset) {
        return static_cast<aclDataType>(t - g_toAclOffset);
    }
    return ConvertToAclDataType(
        static_cast<at::ScalarType>(t));
}

#endif  // CANN_OPS_TRANSFORMER_ACLNN_TYPES_H