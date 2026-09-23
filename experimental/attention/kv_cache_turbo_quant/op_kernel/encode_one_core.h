/**
 * EncodeOne 算法真源（与任务书 golden.py / cpu_encode_twin 同语义）。
 * Host 孪生与 Device Kernel 共用，避免漂移。
 *
 * 约束：freestanding（禁 std::vector）。
 * 矩阵按 (j,i) 加载器读取；Kernel 侧勿在栈上展开 128×128 整表。
 * Sqrt 由调用方注入（Host=std::sqrt；Device=Babylonian / AscendC::Sqrt）。
 */
#ifndef ENCODE_ONE_CORE_H_
#define ENCODE_ONE_CORE_H_

#include <cstdint>

#ifndef KV_ENC_INLINE
#define KV_ENC_INLINE inline
#endif

namespace KvEncodeCore {

constexpr int kHeadDim = 128;

constexpr float kCentroids2[4] = {-0.1335033178f, -0.04002048075f, 0.04002048075f, 0.1335033178f};
constexpr float kCentroids3[8] = {-0.19020693f,   -0.1187859178f, -0.06682205945f, -0.02166347019f,
                                  0.02166347019f, 0.06682205945f, 0.1187859178f,   0.19020693f};
constexpr float kCentroids4[16] = {
    -0.2414890379f, -0.1828317791f, -0.1429702938f, -0.1109927073f, -0.08325428516f, -0.05802082643f,
    -0.03428063914f, -0.01134236995f, 0.01134236995f, 0.03428063914f, 0.05802082643f, 0.08325428516f,
    0.1109927073f, 0.1429702938f, 0.1828317791f, 0.2414890379f};

KV_ENC_INLINE const float *Centroids(int bits, int *nOut)
{
    if (bits == 2) {
        *nOut = 4;
        return kCentroids2;
    }
    if (bits == 3) {
        *nOut = 8;
        return kCentroids3;
    }
    *nOut = 16;
    return kCentroids4;
}

/** 与 golden.py `_pack_bits` 相同：每 8 个码并成 bit_width 个字节。nValues≤128。 */
KV_ENC_INLINE void PackBits(const uint8_t *values, int nValues, int bitWidth, uint8_t *out)
{
    const int groups = (nValues + 7) / 8;
    uint8_t padded[128];
    for (int i = 0; i < groups * 8; ++i) {
        padded[i] = 0;
    }
    for (int i = 0; i < nValues; ++i) {
        padded[i] = values[i];
    }
    for (int g = 0; g < groups; ++g) {
        uint64_t word = 0;
        for (int lane = 0; lane < 8; ++lane) {
            word |= static_cast<uint64_t>(padded[g * 8 + lane]) << (lane * bitWidth);
        }
        if (bitWidth == 1) {
            out[g] = static_cast<uint8_t>(word & 0xFFu);
        } else {
            for (int b = 0; b < bitWidth; ++b) {
                out[g * bitWidth + b] = static_cast<uint8_t>((word >> (8 * b)) & 0xFFu);
            }
        }
    }
}

/** Host：row-major 指针加载器 */
struct PtrMat {
    const float *base;
    KV_ENC_INLINE float operator()(int j, int i) const
    {
        return base[j * kHeadDim + i];
    }
};

/**
 * Encode 单向量（fp32 工作精度）。
 * rot(j,i) / qjl(j,i) = H[j,i] / S[j,i]；rotated[j]=sum_i unit[i]*H[j,i]（unit @ H.T）。
 * sqrtFn(float)->float 注入平方根实现。
 */
template <typename RotLoad, typename QjlLoad, typename SqrtFn>
KV_ENC_INLINE void EncodeOne(const float *x, RotLoad rot, QjlLoad qjlMat, SqrtFn sqrtFn, int mseBits,
                             uint8_t *idxOut, uint8_t *qjlOut, float *normOut, float *gammaOut)
{
    float sumSq = 0.f;
    for (int i = 0; i < kHeadDim; ++i) {
        sumSq += x[i] * x[i];
    }
    const float norm = sqrtFn(sumSq);
    *normOut = norm;

    float unit[kHeadDim];
    if (norm > 0.f) {
        const float inv = 1.f / (norm < 1e-30f ? 1e-30f : norm);
        for (int i = 0; i < kHeadDim; ++i) {
            unit[i] = x[i] * inv;
        }
    } else {
        for (int i = 0; i < kHeadDim; ++i) {
            unit[i] = 0.f;
        }
    }

    float rotated[kHeadDim];
    for (int j = 0; j < kHeadDim; ++j) {
        float acc = 0.f;
        for (int i = 0; i < kHeadDim; ++i) {
            acc += unit[i] * rot(j, i);
        }
        rotated[j] = acc;
    }

    int nCen = 0;
    const float *cen = Centroids(mseBits, &nCen);
    uint8_t indices[kHeadDim];
    float primary[kHeadDim];
    for (int j = 0; j < kHeadDim; ++j) {
        int idx = 0;
        for (int c = 0; c < nCen - 1; ++c) {
            const float boundary = 0.5f * (cen[c] + cen[c + 1]);
            if (rotated[j] > boundary) {
                ++idx;
            }
        }
        indices[j] = static_cast<uint8_t>(idx);
        primary[j] = cen[idx];
    }

    float residual[kHeadDim];
    float resSumSq = 0.f;
    for (int j = 0; j < kHeadDim; ++j) {
        residual[j] = rotated[j] - primary[j];
        resSumSq += residual[j] * residual[j];
    }
    const float resNorm = sqrtFn(resSumSq);
    *gammaOut = norm * resNorm;

    float resUnit[kHeadDim];
    if (resNorm > 0.f) {
        const float inv = 1.f / (resNorm < 1e-30f ? 1e-30f : resNorm);
        for (int j = 0; j < kHeadDim; ++j) {
            resUnit[j] = residual[j] * inv;
        }
    } else {
        for (int j = 0; j < kHeadDim; ++j) {
            resUnit[j] = 0.f;
        }
    }

    uint8_t signs[kHeadDim];
    for (int j = 0; j < kHeadDim; ++j) {
        float acc = 0.f;
        for (int i = 0; i < kHeadDim; ++i) {
            acc += resUnit[i] * qjlMat(j, i);
        }
        signs[j] = (acc >= 0.f) ? 1 : 0;
    }

    PackBits(indices, kHeadDim, mseBits, idxOut);
    PackBits(signs, kHeadDim, 1, qjlOut);
}

/** Host 便捷：rotation / qjl 为 row-major HxH（勿与模板 EncodeOne 重载混淆） */
template <typename SqrtFn>
KV_ENC_INLINE void EncodeOneHost(const float *x, const float *rotation, const float *qjl, SqrtFn sqrtFn, int mseBits,
                                 uint8_t *idxOut, uint8_t *qjlOut, float *normOut, float *gammaOut)
{
    EncodeOne(x, PtrMat{rotation}, PtrMat{qjl}, sqrtFn, mseBits, idxOut, qjlOut, normOut, gammaOut);
}

} // namespace KvEncodeCore

#endif // ENCODE_ONE_CORE_H_
