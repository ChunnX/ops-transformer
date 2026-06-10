/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_api/aclnn_stem_oam_prep_varlen_q.h"

namespace ascend_kernel {

static constexpr int64_t DIM_QK = 128;

static aclTensor* CreateAclTensor(const at::Tensor& tensor, aclDataType dtype) {
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    int64_t storageOffset = tensor.storage_offset();
    void* dataPtr = tensor.mutable_data_ptr();

    std::vector<int64_t> aclSizes(sizes.begin(), sizes.end());
    std::vector<int64_t> aclStrides(strides.begin(), strides.end());

    return aclCreateTensor(
        aclSizes.data(), aclSizes.size(), dtype, ACL_FORMAT_ND,
        aclStrides.data(), aclStrides.size(), storageOffset, dataPtr);
}

static void DestroyAclTensor(aclTensor* tensor) {
    if (tensor != nullptr) {
        void* dataPtr = aclGetTensorDataPtr(tensor);
        if (dataPtr != nullptr) {
            aclrtFree(dataPtr);
        }
        aclDestroyTensor(tensor);
    }
}

at::Tensor stem_oam_prep_varlen_q_torch(
    const at::Tensor& q,
    const at::Tensor& qSeqLens,
    const at::Tensor& cuSeqLensQ,
    const c10::optional<at::Tensor>& qScaleOpt,
    const c10::optional<at::Tensor>& kScaleOpt,
    int64_t stemBlockSize,
    int64_t stemStride)
{
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "only FP8_e4m3fn supported");
    TORCH_CHECK(q.is_privateuseone(), "q must be on NPU");
    TORCH_CHECK(qSeqLens.is_privateuseone(), "qSeqLens must be on NPU");
    TORCH_CHECK(cuSeqLensQ.is_privateuseone(), "cuSeqLensQ must be on NPU");
    TORCH_CHECK(qScaleOpt.has_value(), "qScale must be provided for FP8 input");
    TORCH_CHECK(kScaleOpt.has_value(), "kScale must be provided for FP8 input");
    TORCH_CHECK(stemBlockSize % 32 == 0, "stemBlockSize must be multiple of 32");
    TORCH_CHECK(stemStride % 16 == 0, "stemStride must be multiple of 16");
    TORCH_CHECK(stemStride <= stemBlockSize, "stemStride must be <= stemBlockSize");
    TORCH_CHECK(stemBlockSize <= 256, "stemBlockSize must be <= 256");
    TORCH_CHECK(stemStride <= 64, "stemStride must be <= 64");

    at::Tensor qSeqLensNpu  = qSeqLens.to(at::kInt).npu();
    at::Tensor cuSeqLensNpu = cuSeqLensQ.to(at::kInt).npu();

    int64_t H_q         = q.size(1);
    int64_t D            = q.size(2);
    int64_t batch        = qSeqLens.size(0);
    TORCH_CHECK(D == DIM_QK, "q last dim must be 128");

    int64_t maxQLen = qSeqLens.max().item<int32_t>();
    int64_t maxQPadded = ((maxQLen + stemBlockSize - 1) / stemBlockSize) * stemBlockSize;
    int64_t maxQb = maxQPadded / stemBlockSize;

    auto& qScale = qScaleOpt.value();
    TORCH_CHECK(qScale.is_privateuseone(), "qScale must be on NPU");
    TORCH_CHECK(qScale.dim() == 3, "qScale must be 3D [batch, H_q, max_seqlen_pad_stemBlockSize]");
    TORCH_CHECK(qScale.size(0) == batch, "qScale dim 0 must equal batch");
    TORCH_CHECK(qScale.size(1) == H_q, "qScale dim 1 must equal H_q");
    TORCH_CHECK(qScale.size(2) == maxQPadded,
        "qScale last dim must be max_seqlen_pad_stemBlockSize (", maxQPadded, "), got ", qScale.size(2));
    TORCH_CHECK(qScale.scalar_type() == at::kFloat, "qScale must be FP32");

    auto& kScale = kScaleOpt.value();
    TORCH_CHECK(kScale.is_privateuseone(), "kScale must be on NPU");
    TORCH_CHECK(kScale.dim() == 1, "kScale must be 1D [H_q]");
    TORCH_CHECK(kScale.size(0) == H_q, "kScale size must equal H_q (", H_q, "), got ", kScale.size(0));
    TORCH_CHECK(kScale.scalar_type() == at::kFloat, "kScale must be FP32");

    int64_t qflat_dim = stemStride * DIM_QK;
    at::Tensor qFlat = at::empty({batch, H_q, maxQb, qflat_dim},
                                  q.options().dtype(at::kBFloat16));

    auto aclStream = c10_npu::getCurrentNPUStream().stream();

    aclTensor* qAcl = CreateAclTensor(q, ACL_DT_FLOAT8_E4M3FN);
    aclTensor* qSeqLensAcl = CreateAclTensor(qSeqLensNpu, ACL_DT_INT32);
    aclTensor* cuSeqLensAcl = CreateAclTensor(cuSeqLensNpu, ACL_DT_INT32);
    aclTensor* qFlatAcl = CreateAclTensor(qFlat, ACL_DT_BFLOAT16);
    aclTensor* qScaleAcl = CreateAclTensor(qScale, ACL_DT_FLOAT);
    aclTensor* kScaleAcl = CreateAclTensor(kScale, ACL_DT_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnStemOamPrepVarlenQGetWorkspaceSize(
        qAcl, qSeqLensAcl, cuSeqLensAcl, qScaleAcl, kScaleAcl,
        stemBlockSize, stemStride, qFlatAcl,
        &workspaceSize, &executor);
    TORCH_CHECK(ret == ACLNN_SUCCESS, "aclnn GetWorkspaceSize failed: ", static_cast<int>(ret));

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    ret = aclnnStemOamPrepVarlenQ(workspace, workspaceSize, executor, aclStream);
    TORCH_CHECK(ret == ACLNN_SUCCESS, "aclnn Execute failed: ", static_cast<int>(ret));

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    DestroyAclTensor(qAcl);
    DestroyAclTensor(qSeqLensAcl);
    DestroyAclTensor(cuSeqLensAcl);
    DestroyAclTensor(qFlatAcl);
    DestroyAclTensor(qScaleAcl);
    DestroyAclTensor(kScaleAcl);

    return qFlat;
}

}  // namespace ascend_kernel