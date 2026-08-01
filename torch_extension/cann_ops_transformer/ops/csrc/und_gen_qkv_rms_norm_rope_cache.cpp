/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

const int64_t HEAD_DIM = 128;
const int64_t MROPE_AXIS_NUM = 3;

at::Tensor UndGenQkvRmsNormRopeCache(const at::Tensor &undQkv, const at::Tensor &undWeightsQ,
                                     const at::Tensor &undWeightsK, const at::Tensor &cosSinCache,
                                     const at::Tensor &kCache, const at::Tensor &vCache,
                                     const at::Tensor &slotMapping, const at::Tensor &positions,
                                     const c10::optional<at::Tensor> &genQkv,
                                     const c10::optional<at::Tensor> &genWeightsQ,
                                     const c10::optional<at::Tensor> &genWeightsK,
                                     const c10::optional<at::Tensor> &catIndices, int64_t numHeadsQ,
                                     int64_t numHeadsK, int64_t numHeadsV, double normEps,
                                     const at::IntArrayRef &mropeSection)
{
    TORCH_CHECK(undQkv.dim() == 3, "und_qkv must be 3D [und_len, N, D], got ", undQkv.dim(), "D");
    TORCH_CHECK(undQkv.size(2) == HEAD_DIM, "head_dim must be ", HEAD_DIM, ", got ", undQkv.size(2));
    TORCH_CHECK(numHeadsQ > 0 && numHeadsK > 0 && numHeadsV > 0, "num_heads_q/k/v must be positive");
    TORCH_CHECK(numHeadsQ + numHeadsK + numHeadsV == undQkv.size(1),
                "num_heads_q + num_heads_k + num_heads_v must equal und_qkv N dim, got ",
                numHeadsQ + numHeadsK + numHeadsV, " vs ", undQkv.size(1));
    TORCH_CHECK(kCache.dim() == 4 && vCache.dim() == 4, "k_cache/v_cache must be 4D [Bn, Bs, N, D]");
    TORCH_CHECK(kCache.is_contiguous() && vCache.is_contiguous(), "k_cache/v_cache must be contiguous (BBND)");
    TORCH_CHECK(positions.dim() == 2 && positions.size(0) == MROPE_AXIS_NUM, "positions must be [3, T]");
    TORCH_CHECK(mropeSection.size() == 0 || mropeSection.size() == static_cast<size_t>(MROPE_AXIS_NUM),
                "mrope_section must be empty or have 3 elements, got ", mropeSection.size());
    if (genQkv.has_value()) {
        TORCH_CHECK(genWeightsQ.has_value() && genWeightsK.has_value(),
                    "gen_weights_q/gen_weights_k are required when gen_qkv is provided");
    }

    int64_t undLen = undQkv.size(0);
    int64_t genLen = genQkv.has_value() ? genQkv.value().size(0) : 0;
    int64_t total = undLen + genLen;
    TORCH_CHECK(slotMapping.numel() == total, "slot_mapping length must be und_len + gen_len (", total, "), got ",
                slotMapping.numel());
    TORCH_CHECK(positions.size(1) == total, "positions T dim must be ", total, ", got ", positions.size(1));

    at::Tensor q = at::empty({total, numHeadsQ, HEAD_DIM}, undQkv.options());

    StorageShapeTensor undQkvWrapped{undQkv};
    StorageShapeTensor undWeightsQWrapped{undWeightsQ};
    StorageShapeTensor undWeightsKWrapped{undWeightsK};
    StorageShapeTensor cosSinCacheWrapped{cosSinCache};
    StorageShapeTensor kCacheWrapped{kCache};
    StorageShapeTensor vCacheWrapped{vCache};
    StorageShapeTensor slotMappingWrapped{slotMapping};
    StorageShapeTensor positionsWrapped{positions};
    StorageShapeTensor qWrapped{q};

    ACLNN_CMD(aclnnUndGenQkvRmsNormRopeCache, undQkvWrapped, undWeightsQWrapped, undWeightsKWrapped,
              cosSinCacheWrapped, kCacheWrapped, vCacheWrapped, slotMappingWrapped, positionsWrapped, genQkv,
              genWeightsQ, genWeightsK, catIndices, numHeadsQ, numHeadsK, numHeadsV, normEps, mropeSection, qWrapped,
              kCacheWrapped, vCacheWrapped);

    return q;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("und_gen_qkv_rms_norm_rope_cache", &UndGenQkvRmsNormRopeCache, "und_gen_qkv_rms_norm_rope_cache");
}
} // namespace op_api
