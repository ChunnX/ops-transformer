/*
 * torch.ops.npu registration for compressor_grad
 */

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("compressor_grad(Tensor x, Tensor wkv, Tensor wgate, Tensor d_cmp_kv, "
          "Tensor softmax_score, Tensor kv, int cmp_ratio, int coff) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("compressor_grad", TORCH_FN(ascend_kernel::compressor_grad));
}

}  // namespace
