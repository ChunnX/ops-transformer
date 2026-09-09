// 独立 CPU FP64 逐 token 递推。只并行互不依赖的 batch/head，不调用 NPU 算子。
#include <torch/extension.h>
#include <cmath>
#include <vector>

std::vector<at::Tensor> reference(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor g, at::Tensor beta,
    at::Tensor initial, const std::vector<int64_t>& lengths, double scale, bool raw_gate)
{
    for (const auto& t : {q, k, v, g, beta, initial}) {
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kFloat && t.is_contiguous(),
                    "CPU reference requires contiguous float32 inputs");
    }
    const auto tokens = v.size(0), hv = v.size(1), vd = v.size(2);
    const auto hk = q.size(1), kd = q.size(2), batch = initial.size(0);
    TORCH_CHECK(static_cast<int64_t>(lengths.size()) == batch && hv % hk == 0,
                "invalid sequence/head dimensions");
    std::vector<int64_t> offsets(batch + 1, 0);
    for (int64_t b = 0; b < batch; ++b) offsets[b + 1] = offsets[b] + lengths[b];
    TORCH_CHECK(offsets.back() == tokens, "sequence lengths do not sum to T");
    auto out = at::empty(v.sizes(), v.options().dtype(at::kDouble));
    auto final = at::empty(initial.sizes(), initial.options().dtype(at::kDouble));
    const auto* qp = q.data_ptr<float>();
    const auto* kp = k.data_ptr<float>();
    const auto* vp = v.data_ptr<float>();
    const auto* gp = g.data_ptr<float>();
    const auto* bp = beta.data_ptr<float>();
    const auto* sp = initial.data_ptr<float>();
    auto* op = out.data_ptr<double>();
    auto* fp = final.data_ptr<double>();
    #pragma omp parallel for schedule(dynamic) num_threads(16)
    for (int64_t item = 0; item < batch * hv; ++item) {
        const int64_t b = item / hv, head = item % hv, qhead = head / (hv / hk);
        std::vector<double> state(vd * kd), decay(kd);
        for (int64_t i = 0; i < vd * kd; ++i) state[i] = sp[item * vd * kd + i];
        for (int64_t t = offsets[b]; t < offsets[b + 1]; ++t) {
            const float* key = kp + (t * hk + qhead) * kd;
            const float* query = qp + (t * hk + qhead) * kd;
            for (int64_t d = 0; d < kd; ++d) {
                const double gate = gp[(t * hv + head) * kd + d];
                const double gamma = raw_gate ? -5.0 / (1.0 + std::exp(-gate)) : gate;
                decay[d] = std::exp(gamma);
            }
            for (int64_t e = 0; e < vd; ++e) {
                double* row = state.data() + e * kd;
                double prediction = 0.0;
                #pragma omp simd reduction(+:prediction)
                for (int64_t d = 0; d < kd; ++d) {
                    row[d] *= decay[d];
                    prediction += row[d] * key[d];
                }
                const double delta = static_cast<double>(bp[t * hv + head]) *
                                     (vp[(t * hv + head) * vd + e] - prediction);
                double value = 0.0;
                #pragma omp simd reduction(+:value)
                for (int64_t d = 0; d < kd; ++d) {
                    row[d] += delta * key[d];
                    value += row[d] * query[d];
                }
                op[(t * hv + head) * vd + e] = value * scale;
            }
        }
        for (int64_t i = 0; i < vd * kd; ++i) fp[item * vd * kd + i] = state[i];
    }
    return {out, final};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reference", &reference, "CPU FP64 KDA recurrence");
}
