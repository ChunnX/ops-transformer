/*!
 * \file chunk_gated_delta_rule_c128.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using tensor_pair = std::tuple<at::Tensor, at::Tensor>;

namespace {
constexpr int64_t QKV_DIM_NUM = 3;
constexpr int64_t STATE_DIM_NUM = 4;
constexpr int64_t SEQ_LENS_DIM_NUM = 1;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

void CheckInputs(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &beta,
                 const at::Tensor &initial_state, const at::Tensor &actual_seq_lengths,
                 const c10::optional<at::Tensor> &g)
{
    TORCH_CHECK(query.dim() == QKV_DIM_NUM, "query should be 3D, but got ", query.dim(), "D.");
    TORCH_CHECK(key.dim() == QKV_DIM_NUM, "key should be 3D, but got ", key.dim(), "D.");
    TORCH_CHECK(value.dim() == QKV_DIM_NUM, "value should be 3D, but got ", value.dim(), "D.");
    TORCH_CHECK(beta.dim() == 2, "beta should be 2D, but got ", beta.dim(), "D.");
    TORCH_CHECK(initial_state.dim() == STATE_DIM_NUM, "initial_state should be 4D, but got ",
                initial_state.dim(), "D.");
    TORCH_CHECK(actual_seq_lengths.dim() == SEQ_LENS_DIM_NUM, "actual_seq_lengths should be 1D, but got ",
                actual_seq_lengths.dim(), "D.");

    TORCH_CHECK(query.scalar_type() == at::kBFloat16, "query dtype should be BFloat16.");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16, "key dtype should be BFloat16.");
    TORCH_CHECK(value.scalar_type() == at::kBFloat16, "value dtype should be BFloat16.");
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16, "beta dtype should be BFloat16.");
    TORCH_CHECK(initial_state.scalar_type() == at::kBFloat16, "initial_state dtype should be BFloat16.");
    TORCH_CHECK(actual_seq_lengths.scalar_type() == at::kInt, "actual_seq_lengths dtype should be Int32.");

    TORCH_CHECK(query.size(DIM_0) == key.size(DIM_0), "query and key T should be the same.");
    TORCH_CHECK(query.size(DIM_0) == value.size(DIM_0), "query and value T should be the same.");
    TORCH_CHECK(query.size(DIM_1) == key.size(DIM_1), "query and key Nk should be the same.");
    TORCH_CHECK(query.size(DIM_2) == key.size(DIM_2), "query and key Dk should be the same.");
    TORCH_CHECK(beta.size(DIM_0) == value.size(DIM_0), "beta T should equal value T.");
    TORCH_CHECK(beta.size(DIM_1) == value.size(DIM_1), "beta Nv should equal value Nv.");
    TORCH_CHECK(initial_state.size(DIM_0) == actual_seq_lengths.size(DIM_0),
                "initial_state B should equal actual_seq_lengths length.");
    TORCH_CHECK(initial_state.size(DIM_1) == value.size(DIM_1), "initial_state Nv should equal value Nv.");
    TORCH_CHECK(initial_state.size(DIM_2) == value.size(DIM_2), "initial_state Dv should equal value Dv.");
    TORCH_CHECK(initial_state.size(DIM_3) == query.size(DIM_2), "initial_state Dk should equal query Dk.");

    if (g.has_value() && g.value().defined()) {
        TORCH_CHECK(g.value().dim() == 2, "g should be 2D, but got ", g.value().dim(), "D.");
        TORCH_CHECK(g.value().scalar_type() == at::kFloat, "g dtype should be Float32.");
        TORCH_CHECK(g.value().size(DIM_0) == value.size(DIM_0), "g T should equal value T.");
        TORCH_CHECK(g.value().size(DIM_1) == value.size(DIM_1), "g Nv should equal value Nv.");
    }
}
} // namespace

tensor_pair npu_chunk_gated_delta_rule_C128(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                            const at::Tensor &beta, const at::Tensor &initial_state,
                                            const at::Tensor &actual_seq_lengths,
                                            const c10::optional<at::Tensor> &g, double scale)
{
    CheckInputs(query, key, value, beta, initial_state, actual_seq_lengths, g);

    auto out = at::empty(value.sizes(), at::TensorOptions().dtype(value.scalar_type())
                                           .device(value.device())
                                           .memory_format(c10::MemoryFormat::Contiguous));
    auto final_state = at::empty(initial_state.sizes(), at::TensorOptions().dtype(initial_state.scalar_type())
                                                        .device(initial_state.device())
                                                        .memory_format(c10::MemoryFormat::Contiguous));
    float scale_value = static_cast<float>(scale);

    ACLNN_CMD(aclnnChunkGatedDeltaRuleC128, query, key, value, beta, initial_state, actual_seq_lengths, g,
              scale_value, out, final_state);

    return std::make_tuple(out, final_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_chunk_gated_delta_rule_C128", &npu_chunk_gated_delta_rule_C128, "chunk_gated_delta_rule_C128");
}
} // namespace op_api
