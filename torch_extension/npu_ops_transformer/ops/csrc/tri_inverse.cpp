/*!
 * \file tri_inverse.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t TRI_INVERSE_DIM_NUM = 3;
constexpr int64_t CHUNK_SIZE = 64;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;

void CheckInputs(const at::Tensor &cc_in)
{
    TORCH_CHECK(cc_in.dim() == TRI_INVERSE_DIM_NUM, "cc_in should be 3D, but got ", cc_in.dim(), "D.");
    TORCH_CHECK(cc_in.scalar_type() == at::kBFloat16, "cc_in dtype should be BFloat16.");
    TORCH_CHECK(cc_in.size(DIM_1) == CHUNK_SIZE && cc_in.size(DIM_2) == CHUNK_SIZE,
                "cc_in shape should be [tile_num, 64, 64].");
}
} // namespace

at::Tensor npu_tri_inverse(const at::Tensor &cc_in)
{
    CheckInputs(cc_in);
    auto cc_out = at::empty(cc_in.sizes(), at::TensorOptions().dtype(cc_in.scalar_type())
                                             .device(cc_in.device())
                                             .memory_format(c10::MemoryFormat::Contiguous));

    ACLNN_CMD(aclnnTriInverse, cc_in, cc_out);
    return cc_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_tri_inverse", &npu_tri_inverse, "tri_inverse");
}
} // namespace op_api
