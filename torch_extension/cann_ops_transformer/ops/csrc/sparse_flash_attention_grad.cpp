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
 * \file sparse_flash_attention_grad.cpp
 * \brief PTA wrapper of aclnnSparseFlashAttentionGrad (SFAG, MLA OSS Sink).
 */

#include <cctype>
#include <string>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t SFAG_EMPTY_NUMEL = 0;
constexpr int64_t SFAG_MIN_SPARSE_BLOCK_SIZE = 1;
constexpr int64_t SFAG_MAX_SPARSE_BLOCK_SIZE = 64;

bool SfagIsAscend950()
{
    const char *socName = aclrtGetSocName();
    return socName != nullptr && std::strstr(socName, "Ascend950") != nullptr;
}

bool SfagLayoutValid(const std::string &layout)
{
    std::string layoutUpper = layout;
    for (auto &c : layoutUpper) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return layoutUpper == "BSND" || layoutUpper == "TND";
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
SparseFlashAttentionGrad(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparseIndices, const at::Tensor &dOut, const at::Tensor &out,
    const at::Tensor &softmaxMax, const at::Tensor &softmaxSum,
    const c10::optional<at::Tensor> &sinks,
    double scaleValue, int64_t sparseBlockSize,
    const c10::optional<at::Tensor> &queryRope, const c10::optional<at::Tensor> &keyRope,
    const c10::optional<at::Tensor> &actualSeqLenQ, const c10::optional<at::Tensor> &actualSeqLenKv,
    std::string layout, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, bool deterministic)
{
    // ---- 入参校验 ----
    TORCH_CHECK((query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf),
                "query should be bfloat16 or float16, current dtype is: ", query.scalar_type());
    TORCH_CHECK((key.scalar_type() == query.scalar_type() && value.scalar_type() == query.scalar_type() &&
                 dOut.scalar_type() == query.scalar_type() && out.scalar_type() == query.scalar_type()),
                "query/key/value/d_out/out should have the same dtype, current query dtype: ",
                query.scalar_type());
    TORCH_CHECK((sparseIndices.scalar_type() == at::kInt),
                "sparse_indices should be int32, current dtype is: ", sparseIndices.scalar_type());
    TORCH_CHECK((softmaxMax.scalar_type() == at::kFloat && softmaxSum.scalar_type() == at::kFloat),
                "softmax_max/softmax_sum should be float32, current dtype is: ", softmaxMax.scalar_type());
    TORCH_CHECK(SfagLayoutValid(layout), "layout should be BSND or TND, current is: ", layout);
    TORCH_CHECK((sparseBlockSize == SFAG_MIN_SPARSE_BLOCK_SIZE || sparseBlockSize == 8 ||
                 sparseBlockSize == 16 || sparseBlockSize == 32 ||
                 sparseBlockSize == SFAG_MAX_SPARSE_BLOCK_SIZE),
                "sparse_block_size should be in {1,8,16,32,64}, current is: ", sparseBlockSize);

    const at::Tensor &sinksConst = sinks.value_or(at::Tensor());
    if (sinksConst.defined()) {
        TORCH_CHECK((sinksConst.scalar_type() == at::kFloat),
                    "sinks should be float32, current dtype is: ", sinksConst.scalar_type());
    }
    if (queryRope.has_value()) {
        TORCH_CHECK((queryRope->scalar_type() == query.scalar_type()),
                    "query_rope should have the same dtype as query, current dtype is: ", queryRope->scalar_type());
    }
    if (keyRope.has_value()) {
        TORCH_CHECK((keyRope->scalar_type() == query.scalar_type()),
                    "key_rope should have the same dtype as query, current dtype is: ", keyRope->scalar_type());
    }

    const at::Tensor &queryRopeConst = queryRope.value_or(at::Tensor());
    const at::Tensor &keyRopeConst = keyRope.value_or(at::Tensor());
    const at::Tensor &actualSeqLenQConst = actualSeqLenQ.value_or(at::Tensor());
    const at::Tensor &actualSeqLenKvConst = actualSeqLenKv.value_or(at::Tensor());

    // ---- 申请输出张量（DeviceGuard 必须包住输出申请） ----
    at::Tensor dq{nullptr};
    at::Tensor dk{nullptr};
    at::Tensor dv{nullptr};
    at::Tensor dqRope{nullptr};
    at::Tensor dkRope{nullptr};
    at::Tensor dSinks{nullptr};
    {
        auto localDevice = c10::Device(query.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);

        dq = at::empty_like(query);
        dk = at::empty_like(key);
        dv = at::empty_like(value);
        dqRope = queryRopeConst.defined() ? at::empty_like(queryRopeConst)
                                          : at::empty({SFAG_EMPTY_NUMEL}, query.options());
        dkRope = keyRopeConst.defined() ? at::empty_like(keyRopeConst)
                                        : at::empty({SFAG_EMPTY_NUMEL}, query.options());
        dSinks = sinksConst.defined() ? at::empty_like(sinksConst)
                                      : at::empty({SFAG_EMPTY_NUMEL}, query.options().dtype(at::kFloat));
    }

    char *layoutPtr = const_cast<char *>(layout.c_str());
    ACLNN_CMD(aclnnSparseFlashAttentionGrad, query, key, value, sparseIndices, dOut, out, softmaxMax, softmaxSum,
              sinksConst, actualSeqLenQConst, actualSeqLenKvConst, queryRopeConst, keyRopeConst, scaleValue,
              sparseBlockSize, layoutPtr, sparseMode, preTokens, nextTokens, deterministic, dq, dk, dv, dqRope, dkRope,
              dSinks);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(dq, dk, dv, dqRope,
                                                                                               dkRope, dSinks);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_flash_attention_grad", &SparseFlashAttentionGrad, "sparse_flash_attention_grad");
}
} // namespace op_api
