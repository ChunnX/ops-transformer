#include <cmath>
#include <vector>
#include <numeric>
#include <complex>
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

// 与 tiling 一致的常量，用于 n > 4096 时计算 dft 缓冲区大小
constexpr uint32_t MAX_FACTORS_LEN = 3;
constexpr uint32_t LAST_FACTOR = 64;
constexpr uint32_t COMPLEX_PART = 2;
constexpr uint32_t ROW_PAD = 16;
constexpr uint32_t COL_PAD = 8;
constexpr uint32_t TWIDDLE_MATRICES_AMOUNT = 2;

// 与 op_host rfft1_d_tiling_align.cpp 一致：n > 4096 时计算 Cooley-Tukey 因子
static void CalcColleyTukeyFactors(uint32_t factors[], std::vector<uint32_t>& availableFactors, uint32_t len) {
    std::vector<uint32_t> factorsTmp;
    int curFactorIndex = static_cast<int>(availableFactors.size()) - 1;
    uint32_t tmpN = len;

    if (tmpN == LAST_FACTOR * LAST_FACTOR * LAST_FACTOR * COMPLEX_PART) {
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = LAST_FACTOR;
        }
        factors[0] = LAST_FACTOR * COMPLEX_PART;
    } else if (tmpN > static_cast<uint32_t>(DFT_BORDER_VALUE)) {
        while (curFactorIndex >= 0) {
            uint32_t curFactor = availableFactors[curFactorIndex];
            while (tmpN % curFactor == 0) {
                tmpN /= curFactor;
                factorsTmp.emplace_back(curFactor);
            }
            curFactorIndex -= 1;
        }
        while (factorsTmp.size() < MAX_FACTORS_LEN) {
            factorsTmp.emplace_back(1);
        }
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = factorsTmp[i];
        }
    }
}

// 与 op_host 一致：Bluestein 时对 pow2 分解因子
static void CalcBluesteinFactors(uint32_t factors[], std::vector<uint32_t>& availableFactors, uint32_t pow2) {
    if (pow2 == LAST_FACTOR * LAST_FACTOR * LAST_FACTOR * COMPLEX_PART) {
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = LAST_FACTOR;
        }
        factors[0] = LAST_FACTOR * COMPLEX_PART;
    } else {
        std::vector<uint32_t> factorsTmpBluestein;
        uint32_t tmpN = pow2;
        int curFactorIndex = static_cast<int>(availableFactors.size()) - 1;
        while (curFactorIndex >= 0) {
            uint32_t curFactor = availableFactors[curFactorIndex];
            while (tmpN % curFactor == 0) {
                tmpN /= curFactor;
                factorsTmpBluestein.emplace_back(curFactor);
            }
            curFactorIndex -= 1;
        }
        while (factorsTmpBluestein.size() < MAX_FACTORS_LEN) {
            factorsTmpBluestein.emplace_back(1);
        }
        for (size_t i = 0; i < MAX_FACTORS_LEN; ++i) {
            factors[i] = factorsTmpBluestein[i];
        }
    }
}

// 与 tiling 一致的 dft/twiddle 尺寸与偏移（n > 4096）
struct LongPathSizes {
    uint32_t factors[MAX_FACTORS_LEN];
    bool isBluestein;
    uint32_t lengthPad;
    uint32_t dftRealOverallSize;
    uint32_t dftImagOverallSize;
    uint32_t twiddleOverallSize;
    uint32_t fftMatrOverallSize;
    uint32_t dftRealOffsets[MAX_FACTORS_LEN];
    uint32_t dftImagOffsets[MAX_FACTORS_LEN];
    uint32_t twiddleOffsets[MAX_FACTORS_LEN];
    uint32_t dftCurRows[MAX_FACTORS_LEN];
    uint32_t dftCurCols[MAX_FACTORS_LEN];
    uint32_t twiddleCurSize[MAX_FACTORS_LEN];
    uint32_t prevRadices[MAX_FACTORS_LEN];
};

static void ComputeLongPathSizes(uint32_t len, LongPathSizes* out) {
    uint32_t factors[MAX_FACTORS_LEN] = {1, 1, 1};
    std::vector<uint32_t> availableFactors(LAST_FACTOR - 1);
    std::iota(availableFactors.begin(), availableFactors.end(), COMPLEX_PART);

    CalcColleyTukeyFactors(factors, availableFactors, len);

    out->isBluestein = (len % LAST_FACTOR != 0) || (factors[0] * factors[1] * factors[2] != len);
    out->lengthPad = COMPLEX_PART * static_cast<uint32_t>(std::pow(2, std::ceil(std::log2(static_cast<double>(len)))));
    if (out->isBluestein) {
        CalcBluesteinFactors(factors, availableFactors, out->lengthPad);
    }
    for (size_t i = 0; i < MAX_FACTORS_LEN; ++i) {
        out->factors[i] = factors[i];
    }

    uint32_t dftRealOffsets[3] = {0, 1, 1};
    uint32_t dftImagOffsets[3] = {0, 1, 1};
    uint32_t twiddleOffsets[3] = {0, 0, 1};
    uint32_t dftRealOverallSize = 0;
    uint32_t dftImagOverallSize = 0;
    uint32_t twiddleOverallSize = 0;
    size_t prevFactors = 1;

    for (size_t curIndex = 0; curIndex < MAX_FACTORS_LEN; ++curIndex) {
        size_t curFactor = factors[curIndex];
        size_t rowsNum = curFactor * (1 + (curIndex == 0 && out->isBluestein ? 1u : 0u));
        size_t colsNum = curFactor * (curIndex != 0 ? 1 : 2);
        size_t dftCurSize = rowsNum * colsNum;
        size_t twiddleCurSize = rowsNum * prevFactors * COMPLEX_PART;

        if (curIndex != 0) {
            dftRealOffsets[curIndex] = dftRealOverallSize;
            dftImagOffsets[curIndex] = dftImagOverallSize;
            if (curIndex != 1) {
                twiddleOffsets[curIndex] = twiddleOverallSize;
            }
        }

        out->dftCurRows[curIndex] = static_cast<uint32_t>(rowsNum);
        out->dftCurCols[curIndex] = static_cast<uint32_t>(colsNum);
        out->twiddleCurSize[curIndex] = static_cast<uint32_t>(twiddleCurSize);
        out->prevRadices[curIndex] = static_cast<uint32_t>(prevFactors);

        dftRealOverallSize += static_cast<uint32_t>(dftCurSize);
        if (curIndex != 0) {
            dftImagOverallSize += static_cast<uint32_t>(dftCurSize);
            twiddleOverallSize += static_cast<uint32_t>(twiddleCurSize);
        }
        prevFactors *= curFactor;
    }

    out->dftRealOverallSize = dftRealOverallSize;
    out->dftImagOverallSize = dftImagOverallSize;
    out->twiddleOverallSize = twiddleOverallSize;
    out->fftMatrOverallSize = dftRealOverallSize + dftImagOverallSize + TWIDDLE_MATRICES_AMOUNT * twiddleOverallSize;
    for (int i = 0; i < 3; ++i) {
        out->dftRealOffsets[i] = dftRealOffsets[i];
        out->dftImagOffsets[i] = dftImagOffsets[i];
        out->twiddleOffsets[i] = twiddleOffsets[i];
    }
}

// 计算 n > 4096 时 dft 缓冲区总大小（含 Bluestein 的 alpha/beta 区）
// Bluestein 时 kernel 期望 bluesteinPointer = dft + COMPLEX*fftMatrOverallSize，即前 2*fftMatrOverallSize 为矩阵区（正+Back）
static uint64_t CalcDftBufferSizeLongPath(uint32_t len) {
    LongPathSizes s;
    ComputeLongPathSizes(len, &s);
    uint64_t total = s.fftMatrOverallSize;
    if (s.isBluestein) {
        total += s.fftMatrOverallSize;  // 第二份：dftBack + twiddleBack，与 CANN FinalCalculation 一致
        total += static_cast<uint64_t>(COMPLEX_PART) * len;           // alpha: 2*fftLength
        total += static_cast<uint64_t>(COMPLEX_PART) * 4 * s.lengthPad;  // conjBeta real/imag, ctBeta real/imag: 4 * 2*lengthPad
    }
    return total;
}

// 第一块：与 CANN CalculationDft curIndex==0 一致，real/imag 交错 (cos, sin)
static void FillDftBlockInterleaved(float* dft, uint32_t rows, uint32_t colsComplex, uint32_t r, double normFactor) {
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < colsComplex; ++j) {
            double angle = -2.0 * M_PI * i * j / static_cast<double>(r);
            dft[i * (2 * colsComplex) + 2 * j] = static_cast<float>(normFactor * std::cos(angle));
            dft[i * (2 * colsComplex) + 2 * j + 1] = static_cast<float>(normFactor * std::sin(angle));
        }
    }
}

// 后续块：与 CANN CalculationDft 一致，real=cos, imag=sin (exp(-i*param) 的实虚部 CANN 用 sin)
static void FillDftBlockSplit(float* dftReal, float* dftImag, uint32_t rows, uint32_t cols, uint32_t r, double normFactor) {
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            double angle = -2.0 * M_PI * i * j / static_cast<double>(r);
            dftReal[i * cols + j] = static_cast<float>(normFactor * std::cos(angle));
            dftImag[i * cols + j] = static_cast<float>(normFactor * std::sin(angle));
        }
    }
}

// 与 CANN 一致的递归 FFT（N 为 2 的幂）
static void FftRec(std::vector<std::complex<double>>& x, int N) {
    if (N <= 1) return;
    constexpr int COMPLEX_FFT = 2;
    std::vector<std::complex<double>> even(N / COMPLEX_FFT), odd(N / COMPLEX_FFT);
    for (int i = 0; i < N / COMPLEX_FFT; ++i) {
        even[i] = x[i * COMPLEX_FFT];
        odd[i] = x[i * COMPLEX_FFT + 1];
    }
    FftRec(even, N / COMPLEX_FFT);
    FftRec(odd, N / COMPLEX_FFT);
    for (int k = 0; k < N / COMPLEX_FFT; ++k) {
        std::complex<double> t = std::exp(std::complex<double>(0, -2.0 * M_PI * k / N)) * odd[k];
        x[k] = even[k] + t;
        x[N / COMPLEX_FFT + k] = even[k] - t;
    }
}

// 与 CANN DoubleRI 一致：对 FFT 后的 beta 写 (real,real)、(-imag,imag) 到 ctBeta
static void DoubleRI(std::vector<double>& ret, const std::vector<std::complex<double>>& betas) {
    for (int k = 0; k < 2; ++k) {
        for (size_t i = 0; i < betas.size(); ++i) {
            if (k == 0) {
                ret.push_back(betas[i].real());
                ret.push_back(betas[i].real());
            } else {
                ret.push_back(-betas[i].imag());
                ret.push_back(betas[i].imag());
            }
        }
    }
}

// 与 CANN 一致的 Bluestein alpha/beta 生成（conjBeta 含 reverse，ctBeta 为 FFT(beta) 的 DoubleRI）
static void FillBluesteinAlphaBeta(uint32_t len, uint32_t lenPow2, float* bluesteinBase) {
    double param = -2.0 * M_PI / static_cast<double>(len);
    std::complex<double> w(std::cos(param), std::sin(param));
    std::complex<double> B = std::pow(w, -0.5);

    // Alpha: 2*len
    for (uint32_t k = 0; k < len; ++k) {
        std::complex<double> curAlpha = 1.0 / std::pow(B, static_cast<int64_t>(k) * k);
        bluesteinBase[2 * k] = static_cast<float>(curAlpha.real());
        bluesteinBase[2 * k + 1] = static_cast<float>(curAlpha.imag());
    }
    uint64_t off = static_cast<uint64_t>(COMPLEX_PART) * len;

    // conjBeta / conjBetaImag: 与 CANN CalculateBeta 一致，含 [startIndex, lenPow2) 的 reverse 填充
    const uint64_t startIndex = len + (lenPow2 - COMPLEX_PART * len + 1);
    std::vector<double> betaConjR(lenPow2 * 2, 0.0), betaConjI(lenPow2 * 2, 0.0);
    std::vector<double> betaRevR(lenPow2 * 2, 0.0), betaRevI(lenPow2 * 2, 0.0);

    for (int64_t i = 0; i < static_cast<int64_t>(len); ++i) {
        std::complex<double> curBeta = std::pow(B, static_cast<int64_t>(i) * i);
        betaConjR[2 * i] = betaConjR[2 * i + 1] = curBeta.real();
        betaConjI[2 * i] = curBeta.imag();
        betaConjI[2 * i + 1] = -curBeta.imag();
    }
    std::copy(betaConjR.begin(), betaConjR.begin() + 2 * len, betaRevR.begin());
    std::reverse(betaRevR.begin() + 2, betaRevR.begin() + 2 * len);
    std::copy(betaConjI.begin(), betaConjI.begin() + 2 * len, betaRevI.begin());
    std::reverse(betaRevI.begin() + 2, betaRevI.begin() + 2 * len);

    for (uint64_t i = startIndex; i < lenPow2; ++i) {
        uint64_t src = 2 + 2 * i - 2 * startIndex + 1;
        if (src < 2 * len) {
            double r = betaRevR[src];
            betaConjR[2 * i] = betaConjR[2 * i + 1] = r;
            betaConjI[2 * i] = betaRevI[src];
            betaConjI[2 * i + 1] = -betaRevI[src];
        }
    }

    for (uint64_t i = 0; i < lenPow2 * 2; ++i) {
        bluesteinBase[off + i] = static_cast<float>(betaConjR[i]);
        bluesteinBase[off + lenPow2 * 2 + i] = static_cast<float>(betaConjI[i]);
    }
    off += lenPow2 * 2 * 2;

    // ctBeta: beta 序列 [0,len) + [startIndex,lenPow2) 反转，再 FFT，再 DoubleRI
    std::vector<std::complex<double>> betaComplex(lenPow2, 0.0);
    for (int64_t i = 0; i < static_cast<int64_t>(len); ++i) {
        std::complex<double> curBeta = std::pow(B, static_cast<int64_t>(i) * i);
        betaComplex[i] = curBeta;
    }
    std::vector<double> betaRT(len * 2, 0.0);
    for (int64_t i = 0; i < static_cast<int64_t>(len); ++i) {
        betaRT[2 * i] = betaComplex[i].real();
        betaRT[2 * i + 1] = betaComplex[i].imag();
    }
    std::reverse(betaRT.begin() + 2, betaRT.begin() + 2 * len);
    for (uint64_t i = startIndex; i < lenPow2; ++i) {
        uint64_t srcReal = 2 + 2 * i - 2 * startIndex + 1;
        uint64_t srcImag = srcReal - 1;
        if (srcReal < 2 * len && srcImag < 2 * len) {
            betaComplex[i] = std::complex<double>(betaRT[srcReal], betaRT[srcImag]);
        }
    }
    FftRec(betaComplex, static_cast<int>(lenPow2));
    std::vector<double> ctRI;
    ctRI.reserve(lenPow2 * 4);
    DoubleRI(ctRI, betaComplex);
    for (size_t i = 0; i < ctRI.size(); ++i)
        bluesteinBase[off + i] = static_cast<float>(ctRI[i]);
}

// 填充 twiddle 块：与 CANN CalculationMatricesValues 一致，param = -2*pi*i*j/(curFactor*prevFactors)，每 (i,j) 写 2 个 real、2 个 imag
static void FillTwiddleBlock(float* twiddleReal, float* twiddleImag, uint32_t curRadix, uint32_t prevRadices) {
    double denom = static_cast<double>(curRadix) * static_cast<double>(prevRadices);
    for (uint32_t i = 0; i < curRadix; ++i) {
        for (uint32_t j = 0; j < prevRadices; ++j) {
            double angle = -2.0 * M_PI * i * j / denom;
            uint32_t base = (i * prevRadices + j) * 2;
            twiddleReal[base] = static_cast<float>(std::cos(angle));
            twiddleReal[base + 1] = static_cast<float>(std::cos(angle));
            twiddleImag[base] = static_cast<float>(-std::sin(angle));
            twiddleImag[base + 1] = static_cast<float>(std::sin(angle));
        }
    }
}

// 填充 n > 4096 时的完整 dft 缓冲区（dft+twiddle，以及 Bluestein 的 alpha/beta）
static void FillDftBufferLongPath(uint32_t len, int64_t norm, float* buf, uint64_t bufSize) {
    LongPathSizes s;
    ComputeLongPathSizes(len, &s);

    double normBase = 1.0;
    if (norm != BACKWARD) {
        normBase = (norm == FORWARD) ? (1.0 / len) : (1.0 / std::sqrt(static_cast<double>(len)));
    }

    float* dftRealBase = buf;
    float* dftImagBase = buf + s.dftRealOverallSize;
    for (size_t curIndex = 0; curIndex < MAX_FACTORS_LEN; ++curIndex) {
        uint32_t r = s.factors[curIndex];
        if (r == 1) continue;
        uint32_t rows = s.dftCurRows[curIndex];
        uint32_t cols = s.dftCurCols[curIndex];
        double stageNorm = (curIndex == 0) ? normBase : 1.0;
        if (curIndex == 0) {
            FillDftBlockInterleaved(dftRealBase, rows, cols / 2, r, stageNorm);
        } else {
            uint32_t realOff = s.dftRealOffsets[curIndex];
            uint32_t imagOff = s.dftImagOffsets[curIndex];
            FillDftBlockSplit(dftRealBase + realOff, dftImagBase + imagOff, rows, cols, r, stageNorm);
        }
    }

    float* twiddleRealBase = buf + s.dftRealOverallSize + s.dftImagOverallSize;
    float* twiddleImagBase = twiddleRealBase + s.twiddleOverallSize;
    uint32_t twiddleOff = 0;
    for (size_t curIndex = 1; curIndex < MAX_FACTORS_LEN; ++curIndex) {
        uint32_t r = s.factors[curIndex];
        if (r == 1) continue;
        uint32_t prevR = s.prevRadices[curIndex];
        FillTwiddleBlock(twiddleRealBase + twiddleOff, twiddleImagBase + twiddleOff, r, prevR);
        twiddleOff += s.twiddleCurSize[curIndex];
    }

    if (s.isBluestein) {
        // 与 kernel 一致：bluesteinPointer = dft + COMPLEX*fftMatrOverallSize，第二段为 dftBack+twiddleBack
        float* backBase = buf + s.fftMatrOverallSize;
        // dftBack: dftReal 同，dftImag 取负（CANN dftImagBackCurVal = -sin）
        for (uint64_t i = 0; i < s.dftRealOverallSize; ++i)
            backBase[i] = dftRealBase[i];
        for (uint64_t i = 0; i < s.dftImagOverallSize; ++i)
            backBase[s.dftRealOverallSize + i] = -dftImagBase[i];
        // twiddleBack: twiddleReal 同，twiddleImagBack = -twiddleImag（CANN twiddleImagBack 为 sin/-sin）
        float* twiddleBackR = backBase + s.dftRealOverallSize + s.dftImagOverallSize;
        float* twiddleBackI = twiddleBackR + s.twiddleOverallSize;
        for (uint64_t i = 0; i < s.twiddleOverallSize; ++i) {
            twiddleBackR[i] = twiddleRealBase[i];
            twiddleBackI[i] = -twiddleImagBase[i];
        }
        uint64_t bluesteinBase = 2 * s.fftMatrOverallSize;
        FillBluesteinAlphaBeta(len, s.lengthPad, buf + bluesteinBase);
    }
}

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
 * n <= 4096：在 CPU 上生成 dft 矩阵后拷到 NPU，再调用 aclnnRfft1D（FastDFT 分支）。
 * n > 4096：按 tiling 逻辑计算 dft 缓冲区大小，分配后调用 aclnnRfft1D（Cooley-Tukey/Bluestein 分支）。
 */
at::Tensor rfft1_d(at::Tensor& input, c10::optional<int64_t> n, int64_t norm)
{
    int64_t n_val = n.has_value() ? n.value() : input.size(-1);
    TORCH_CHECK(n_val > 0, "rfft1_d: n must be positive");

    int64_t out_last = (n_val / 2 + 1) * 2;
    std::vector<int64_t> out_shape;
    if (input.dim() == 1) {
        out_shape = {out_last};
    } else {
        out_shape = input.sizes().vec();
        out_shape.back() = out_last;
    }
    at::Tensor result = torch::empty(out_shape, input.options());

    at::Tensor dft_tensor;
    if (n_val <= DFT_BORDER_VALUE) {
        std::vector<float> dft_cpu = Rfft1DDftGen(n_val, norm);
        at::Tensor dft_cpu_tensor = torch::from_blob(dft_cpu.data(), {static_cast<int64_t>(dft_cpu.size())},
                                                     torch::dtype(torch::kFloat32));
        dft_tensor = dft_cpu_tensor.clone().to(input.device());
    } else {
        uint64_t dft_size = CalcDftBufferSizeLongPath(static_cast<uint32_t>(n_val));
        dft_tensor = torch::empty({static_cast<int64_t>(dft_size)}, torch::dtype(torch::kFloat32).device(torch::kCPU));
        FillDftBufferLongPath(static_cast<uint32_t>(n_val), norm, dft_tensor.data_ptr<float>(), dft_size);
        dft_tensor = dft_tensor.to(input.device());
    }

    EXEC_NPU_CMD(aclnnRfft1D, input, dft_tensor, n_val, norm, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("rfft1_d", &rfft1_d, "");
}
