#include <cmath>
#include <vector>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

namespace {
constexpr int64_t NZ_BORDER = 8;
constexpr int64_t NZ_BLOCK = 16;
constexpr int64_t COMPLEX = 2;
constexpr int64_t BACKWARD = 1;
constexpr int64_t FORWARD = 2;
constexpr int64_t ORTHO = 3;
constexpr int64_t DFT_BORDER_VALUE = 4096;

// 与 CANN acl_rfft1d.cpp 中 Rfft1DDftGen 一致，用于 n <= 4096
std::vector<float> Rfft1DDftGen(int64_t fftLength, int64_t norm) {
    std::vector<float> out;
    double normParam = 1.0;
    if (norm != BACKWARD) {
        normParam = (norm == FORWARD) ? (normParam / static_cast<double>(fftLength))
                                     : (normParam / std::sqrt(static_cast<double>(fftLength)));
    }
    size_t fftLenPad = fftLength + (NZ_BORDER - fftLength % NZ_BORDER);
    size_t fftLenPadRow = fftLength + (NZ_BLOCK - fftLength % NZ_BLOCK);
    size_t fftOut = fftLength / COMPLEX + 1;
    size_t k = 0;
    if ((fftLength % NZ_BLOCK != 0) && (fftLength % NZ_BLOCK <= NZ_BORDER)) {
        out.resize(fftLenPad * fftLenPadRow, 0.f);
        for (int64_t i = 0; i < fftLength; ++i) {
            for (size_t j = 0; j < fftOut; ++j) {
                double param = -2.0 * M_PI * i * j / static_cast<double>(fftLength);
                out[k++] = static_cast<float>(normParam * std::cos(param));
                out[k++] = static_cast<float>(normParam * std::sin(param));
            }
        }
        return out;
    }
    std::vector<float> dftNd(fftLenPad * fftLenPadRow, 0.f);
    for (int64_t i = 0; i < fftLength; ++i) {
        for (size_t j = 0; j < fftOut; ++j) {
            double param = -2.0 * M_PI * i * j / static_cast<double>(fftLength);
            dftNd[k++] = static_cast<float>(normParam * std::cos(param));
            dftNd[k++] = static_cast<float>(normParam * std::sin(param));
        }
        for (size_t j = 0; j + COMPLEX * fftOut < fftLenPad; ++j)
            dftNd[k++] = 0.f;
    }
    out.clear();
    out.reserve(fftLenPad * fftLenPadRow);
    for (size_t n = 0; n < fftLenPad / NZ_BORDER; ++n) {
        for (size_t i = 0; i < fftLenPadRow; ++i) {
            for (size_t blockIdx = 0; blockIdx < NZ_BORDER; ++blockIdx)
                out.push_back(dftNd[i * fftLenPad + NZ_BORDER * n + blockIdx]);
        }
    }
    return out;
}
}  // namespace

/**
 * libcust_opapi.so 由 opbuild 根据 def 生成，接口为 (x, dft, n, norm, y)。
 * 此处在 CPU 上生成 dft 矩阵后拷到 NPU，再调用 aclnnRfft1D。
 */
at::Tensor rfft1_d(at::Tensor& input, c10::optional<int64_t> n, int64_t norm)
{
    int64_t n_val = n.has_value() ? n.value() : input.size(-1);
    TORCH_CHECK(n_val > 0 && n_val <= DFT_BORDER_VALUE, "rfft1_d: n must be in (0, 4096] for dft gen");

    int64_t out_last = (n_val / 2 + 1) * 2;
    std::vector<int64_t> out_shape;
    if (input.dim() == 1) {
        out_shape = {out_last};
    } else {
        out_shape = input.sizes().vec();
        out_shape.back() = out_last;
    }
    at::Tensor result = torch::empty(out_shape, input.options());

    std::vector<float> dft_cpu = Rfft1DDftGen(n_val, norm);
    at::Tensor dft_cpu_tensor = torch::from_blob(dft_cpu.data(), {static_cast<int64_t>(dft_cpu.size())},
                                                 torch::dtype(torch::kFloat32));
    at::Tensor dft_tensor = dft_cpu_tensor.clone().to(input.device());

    EXEC_NPU_CMD(aclnnRfft1D, input, dft_tensor, n_val, norm, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("rfft1_d", &rfft1_d, "");
}
