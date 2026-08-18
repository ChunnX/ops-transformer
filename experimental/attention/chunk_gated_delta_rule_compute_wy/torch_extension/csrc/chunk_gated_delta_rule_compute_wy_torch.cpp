/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include "aclnn_common.h"

namespace ascend_ops {
namespace ChunkGatedDeltaRule {

using namespace at;

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> chunk_gated_delta_rule_compute_wy(const Tensor &q, const Tensor &k,
                                                                                     const Tensor &v, const Tensor &g,
                                                                                     const Tensor &beta,
                                                                                     int64_t chunkSize = 64)
{
    TORCH_CHECK(q.defined() && k.defined() && v.defined() && g.defined() && beta.defined(),
                "Inputs must be defined tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4-D tensors");
    TORCH_CHECK(g.dim() == 3 && beta.dim() == 3, "g/beta must be 3-D tensors");

    int64_t B = q.size(0);
    int64_t Nq = q.size(2);
    int64_t Nv = v.size(2);
    int64_t D = q.size(3);

    Tensor qKernel = at::empty({B, Nq, D, D}, q.options());
    Tensor kKernel = at::empty({B, Nq, D, D}, k.options());
    Tensor wKernel = at::empty({B, Nv, D, D}, v.options());
    Tensor uKernel = at::empty({B, Nv, D, D}, v.options());
    Tensor gKernel = at::empty({B, Nv, D}, g.options());

    ACLNN_CMD(aclnnChunkGatedDeltaRuleComputeWy, q, k, v, g, beta, chunkSize, qKernel, kKernel, wKernel, uKernel,
              gKernel);

    return std::make_tuple(qKernel, kKernel, wKernel, uKernel, gKernel);
}

TORCH_LIBRARY(ascend_ops, m)
{
    m.def(
        R"(chunk_gated_delta_rule_compute_wy(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, int chunkSize=64) -> (Tensor, Tensor, Tensor, Tensor, Tensor))");
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("chunk_gated_delta_rule_compute_wy", chunk_gated_delta_rule_compute_wy);
}

} // namespace ChunkGatedDeltaRule
} // namespace ascend_ops

extern "C" {
PyObject *PyInit__C(void)
{
    static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL};
    return PyModule_Create(&module_def);
}
}
