#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using OptionalTensor = c10::optional<at::Tensor>;
using OptionalInts = c10::optional<std::vector<int64_t>>;

std::vector<OptionalTensor> chunk_kda_fwd(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &g, const at::Tensor &beta, double scale,
    const OptionalTensor &initial_state, bool output_final_state,
    bool return_intermediate_states, bool disable_recompute, bool safe_gate,
    double lower_bound, bool use_gate_in_kernel, const OptionalTensor &a_log,
    const OptionalTensor &dt_bias, int64_t chunk_size, const std::string &layout,
    bool state_v_first, const OptionalInts &cu_seqlens, const OptionalInts &chunk_indices)
{
    TORCH_CHECK(layout == "BSND" || layout == "BNSD" || layout == "TND" || layout == "NTD",
                "chunk_kda_fwd: invalid layout");
    const bool rank3 = layout == "TND" || layout == "NTD";
    const bool head_first = layout == "NTD" || layout == "BNSD";
    TORCH_CHECK(q.dim() == (rank3 ? 3 : 4) && v.dim() == q.dim(), "input rank/layout mismatch");
    TORCH_CHECK(chunk_size == 64 || chunk_size == 128, "chunk_size must be 64 or 128");
    const int64_t batch = rank3 ? 1 : q.size(0);
    const int64_t seqlen = q.size(head_first ? -2 : -3);
    const int64_t hv = v.size(head_first ? -3 : -2), kd = q.size(-1), vd = v.size(-1);
    TORCH_CHECK(!cu_seqlens || cu_seqlens->size() >= 2, "cu_seqlens must contain >= 2 entries");
    const int64_t nseq = cu_seqlens ? cu_seqlens->size() - 1 : batch;
    int64_t chunks = (seqlen + chunk_size - 1) / chunk_size;
    if (cu_seqlens) {
        chunks = 0;
        for (size_t i = 1; i < cu_seqlens->size(); ++i) {
            const auto length = (*cu_seqlens)[i] - (*cu_seqlens)[i - 1];
            TORCH_CHECK(length >= 0, "cu_seqlens must be nondecreasing");
            chunks += (length + chunk_size - 1) / chunk_size;
        }
    }
    const c10::OptionalDeviceGuard guard(q.device());
    const int64_t state0 = state_v_first ? vd : kd, state1 = state_v_first ? kd : vd;
    auto empty = [&](std::vector<int64_t> dims, at::ScalarType dtype) {
        return at::empty(dims, q.options().dtype(dtype));
    };
    auto shaped = [&](std::initializer_list<int64_t> dims, at::ScalarType dtype) {
        std::vector<int64_t> shape;
        if (!rank3) shape.push_back(batch);
        shape.insert(shape.end(), dims.begin(), dims.end());
        return empty(shape, dtype);
    };
    std::vector<OptionalTensor> out(12);
    out[0] = shaped({seqlen, hv, vd}, q.scalar_type());
    if (output_final_state) out[1] = empty({nseq, hv, state0, state1}, at::kFloat);
    out[3] = shaped({hv, seqlen, chunk_size}, q.scalar_type());
    out[4] = shaped({hv, seqlen, chunk_size}, q.scalar_type());
    if (disable_recompute) {
        out[2] = shaped({hv, seqlen, kd}, at::kFloat);
        for (const int i : {5, 7, 8}) out[i] = shaped({hv, seqlen, kd}, q.scalar_type());
        for (const int i : {6, 9}) out[i] = shaped({hv, seqlen, vd}, q.scalar_type());
    }
    if (return_intermediate_states || disable_recompute)
        out[10] = shaped({chunks, hv, state0, state1}, q.scalar_type());
    // 由仓库公共扩展桥管理 Tensor 描述符、workspace、PyTorch NPU stream 和生命周期。
    c10::optional<at::IntArrayRef> cu, indices;
    if (cu_seqlens) cu = at::IntArrayRef(*cu_seqlens);
    if (chunk_indices) indices = at::IntArrayRef(*chunk_indices);
    const char *layout_ptr = layout.c_str();
    ACLNN_CMD(aclnnChunkKdaFwd, q, k, v, g, beta, a_log, dt_bias, initial_state,
              cu, indices, layout_ptr, scale, chunk_size, safe_gate, lower_bound,
              use_gate_in_kernel, state_v_first, out[0], out[1], out[2], out[3],
              out[4], out[5], out[6], out[7], out[8], out[9], out[10]);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("chunk_kda_fwd", &chunk_kda_fwd, "ChunkKdaFwd");
}
} // namespace op_api
