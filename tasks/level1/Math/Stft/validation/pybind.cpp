/**
 * STFT: 参考 Rfft1D，在 C++ 内生成与 CANN acl_stft.cpp GenerateDftMatrix 完全一致的 DFT plan，
 * 再调用 aclnnSTFT，保证与 kernel 约定一致。
 */
#include <cmath>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

namespace {
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t PACKAGE_SIZE = 128;  // 与 stft_tiling_generalized.cpp 一致
constexpr double K2PI = 6.2831853071795864769252867665590057683943388;
constexpr unsigned int QUADRANT_ONE = 1;
constexpr unsigned int QUADRANT_TWO = 2;
constexpr unsigned int QUADRANT_FOUR = 4;

// 与 acl_stft.cpp nFftToAlign 一致
int64_t NfftToAlign(int64_t nfft, int alignBytes) {
    int alignNum = alignBytes / FP32_BYTES;
    return (nfft + alignNum - 1) / alignNum * alignNum;
}

// 与 acl_stft.cpp NfftAlignBytes 一致（一般走 PACKAGE_SIZE）
int NfftAlignBytes(int64_t /*nfft*/, int64_t /*hopLength*/, bool /*normalized*/, bool /*onesided*/,
                   bool /*returnComplex*/) {
    return static_cast<int>(PACKAGE_SIZE);
}

// 与 acl_stft.cpp CalcRealAndImag 一致，保证与 kernel 数值一致
void CalcRealAndImag(int m, int n, float* out) {
    int m0 = m;
    int n0 = n;
    float theta, c, s, t;
    unsigned int octant = 0;
    int size = n0;

    m0 = m0 % n0;
    n0 += n0;
    n0 += n0;
    m0 += m0;
    m0 += m0;

    if (m0 < 0) {
        m0 += n0;
    }
    if (m0 > n0 - m0) {
        m0 = n0 - m0;
        octant |= QUADRANT_FOUR;
    }
    if (m0 > size) {
        m0 = m0 - size;
        octant |= QUADRANT_TWO;
    }
    if (m0 > size - m0) {
        m0 = size - m0;
        octant |= QUADRANT_ONE;
    }

    theta = static_cast<float>((K2PI * m0) / n0);
    c = cosf(theta);
    s = sinf(theta);

    if ((octant & QUADRANT_ONE) != 0U) {
        t = c;
        c = s;
        s = t;
    }
    if ((octant & QUADRANT_TWO) != 0U) {
        t = c;
        c = -s;
        s = t;
    }
    if ((octant & QUADRANT_FOUR) != 0U) {
        s = -s;
    }
    out[0] = c;
    out[1] = s;
}

// 与 acl_stft GenerateDftMatrix 布局一致：按行交错 [real_row0, imag_row0, real_row1, imag_row1, ...]
at::Tensor GenerateStftPlan(int64_t n_fft, bool onesided, at::Device device) {
    int alignBytes = NfftAlignBytes(n_fft, 0, false, onesided, true);
    int64_t rowSize = onesided ? (n_fft / 2 + 1) : n_fft;
    int64_t colSizeAlign = NfftToAlign(n_fft, alignBytes);

    int64_t total = rowSize * colSizeAlign * 2;
    at::Tensor plan = torch::empty({total}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    float* ptr = plan.data_ptr<float>();
    float out[2];

    for (int64_t i = 0; i < rowSize; i++) {
        float* addrReal = ptr + i * (2 * colSizeAlign);
        float* addrImag = addrReal + colSizeAlign;
        for (int64_t j = 0; j < colSizeAlign; j++) {
            if (j < n_fft) {
                CalcRealAndImag(-1 * i * j, static_cast<int>(n_fft), out);
                *addrReal = out[0];
                *addrImag = out[1];
            } else {
                *addrReal = 0.f;
                *addrImag = 0.f;
            }
            addrReal++;
            addrImag++;
        }
    }

    return plan.to(device);
}
}  // namespace

/**
 * 接口与 Rfft1D 类似：不要求 Python 传 plan，在 C++ 内生成 plan 后调用 aclnnSTFT。
 */
at::Tensor stft(at::Tensor& x, at::Tensor& window, int64_t n_fft, int64_t hop_length, int64_t win_length,
                bool normalized, bool onesided, bool return_complex) {
    int64_t batch = (x.dim() == 1) ? 1 : x.size(0);
    int64_t len = (x.dim() == 1) ? x.size(0) : x.size(1);
    int64_t n_frames = 1 + (len - n_fft) / hop_length;
    int64_t n_dim = onesided ? (n_fft / 2 + 1) : n_fft;

    c10::SmallVector<int64_t, 4> out_shape;
    if (batch == 1 && x.dim() == 1 && return_complex) {
        out_shape = {n_dim, n_frames};
    } else if (batch == 1 && x.dim() == 1 && !return_complex) {
        out_shape = {n_dim, n_frames, 2};
    } else if (return_complex) {
        out_shape = {batch, n_dim, n_frames};
    } else {
        out_shape = {batch, n_dim, n_frames, 2};
    }

    at::TensorOptions opts = x.options();
    if (return_complex && x.dtype() == torch::kFloat32) {
        opts = opts.dtype(torch::kComplexFloat);
    } else if (return_complex && x.dtype() == torch::kFloat64) {
        opts = opts.dtype(torch::kComplexDouble);
    } else if (!return_complex) {
        opts = opts.dtype(x.dtype() == torch::kComplexFloat ? torch::kFloat32 : torch::kFloat64);
    }
    at::Tensor result = torch::empty(out_shape, opts);

    at::Tensor plan = GenerateStftPlan(n_fft, onesided, x.device());

    EXEC_NPU_CMD(aclnnSTFT, x, plan, window, hop_length, win_length, normalized, onesided, return_complex, n_fft,
                 result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("stft", &stft, "");
}
