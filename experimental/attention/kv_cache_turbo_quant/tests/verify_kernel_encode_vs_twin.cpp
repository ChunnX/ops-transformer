/**
 * 中间验证：Kernel 迁入的 encode_one_core 与孪生字节对拍（不依赖 ops-transformer 整仓编译）。
 *
 * 覆盖：
 *  1) Host EncodeOne(std::sqrt) ≡ twin smoke 落盘
 *  2) DeviceSqrt（与 Kernel 同公式）全路径 idx/qjl/norm/gamma ≡ std::sqrt 路径（smoke）
 *  3) ProcessOne 语义：PtrMat 加载器 + DeviceSqrt ≡ 上式
 *  4) soft-bf16 输入路径可跑通（与 Kernel bf16→fp32 同序）
 *
 * 编译/运行（cwd=tests/）：
 *   ./cpu_encode_twin   # 先产 twin_smoke_out.bin
 *   g++ -O2 -std=c++17 -o verify_kernel_encode_vs_twin verify_kernel_encode_vs_twin.cpp
 *   ./verify_kernel_encode_vs_twin
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../op_kernel/encode_one_core.h"

namespace {

using KvEncodeCore::EncodeOne;
using KvEncodeCore::EncodeOneHost;
using KvEncodeCore::PackBits;
using KvEncodeCore::PtrMat;
using KvEncodeCore::kHeadDim;

float StdSqrt(float x)
{
    return std::sqrt(x);
}

/** 与 kernel DeviceSqrt 同公式 */
struct DeviceSqrt {
    float operator()(float x) const
    {
        if (x <= 0.0f) {
            return 0.0f;
        }
        float y = x;
        if (y < 1.0f) {
            y = 1.0f;
        }
        for (int k = 0; k < 8; ++k) {
            y = 0.5f * (y + x / y);
        }
        return y;
    }
};

float SoftBf16RoundTrip(float x)
{
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    u &= 0xFFFF0000u;
    float y = 0.f;
    std::memcpy(&y, &u, sizeof(y));
    return y;
}

bool LoadBin(const std::string &path, std::vector<float> *out, size_t expectFloats)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return false;
    }
    out->assign(expectFloats, 0.f);
    ifs.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(expectFloats * 4));
    return static_cast<bool>(ifs) || ifs.eof();
}

bool BytesEq(const uint8_t *a, const uint8_t *b, int n)
{
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int IdxBytes(int mseBits)
{
    return (kHeadDim * mseBits + 7) / 8;
}

} // namespace

int main()
{
    std::vector<float> rotation;
    std::vector<float> qjl;
    if (!LoadBin("data/rotation_matrix.bin", &rotation, 128ull * 128ull) ||
        !LoadBin("data/qjl_matrix.bin", &qjl, 128ull * 128ull)) {
        std::printf("FAIL load matrices（请先 python3 gen_selftest_matrices.py）\n");
        return 1;
    }

    uint8_t idxVals[8];
    for (int i = 0; i < 8; ++i) {
        idxVals[i] = 3;
    }
    uint8_t packed[3] = {0, 0, 0};
    PackBits(idxVals, 8, 3, packed);
    if (packed[0] != 219 || packed[1] != 182 || packed[2] != 109) {
        std::printf("FAIL pack known\n");
        return 1;
    }
    std::printf("PASS pack known\n");

    const int mseBits = 3;
    const int idxBytes = IdxBytes(mseBits);
    uint8_t idxA[64] = {0};
    uint8_t idxB[64] = {0};
    uint8_t qjA[16] = {0};
    uint8_t qjB[16] = {0};
    float normA = 0.f;
    float normB = 0.f;
    float gammaA = 0.f;
    float gammaB = 0.f;

    float zero[kHeadDim] = {0};
    EncodeOneHost(zero, rotation.data(), qjl.data(), StdSqrt, mseBits, idxA, qjA, &normA, &gammaA);
    if (normA != 0.f || gammaA != 0.f || idxA[0] != 219 || idxA[1] != 182 || idxA[2] != 109) {
        std::printf("FAIL zero_vector core\n");
        return 1;
    }
    std::printf("PASS zero_vector encode_one_core\n");

    float x[kHeadDim];
    for (int i = 0; i < kHeadDim; ++i) {
        x[i] = static_cast<float>((i % 17) - 8) * 0.1f;
    }
    EncodeOneHost(x, rotation.data(), qjl.data(), StdSqrt, mseBits, idxA, qjA, &normA, &gammaA);

    std::ifstream twinIn("data/twin_smoke_out.bin", std::ios::binary);
    if (!twinIn) {
        std::printf("FAIL missing twin_smoke_out.bin（请先跑 ./cpu_encode_twin）\n");
        return 1;
    }
    uint8_t twinIdx[48];
    uint8_t twinQj[16];
    float twinNorm = 0.f;
    float twinGamma = 0.f;
    twinIn.read(reinterpret_cast<char *>(twinIdx), 48);
    twinIn.read(reinterpret_cast<char *>(twinQj), 16);
    twinIn.read(reinterpret_cast<char *>(&twinNorm), 4);
    twinIn.read(reinterpret_cast<char *>(&twinGamma), 4);
    if (!BytesEq(idxA, twinIdx, idxBytes) || !BytesEq(qjA, twinQj, 16) || normA != twinNorm ||
        gammaA != twinGamma) {
        std::printf("FAIL core vs twin smoke bytes\n");
        return 1;
    }
    std::printf("PASS encode_one_core ↔ twin smoke byte match\n");

    // Kernel 同款 DeviceSqrt + PtrMat（= ProcessOne 去掉 GM 的语义）
    EncodeOne(x, PtrMat{rotation.data()}, PtrMat{qjl.data()}, DeviceSqrt{}, mseBits, idxB, qjB, &normB, &gammaB);
    if (!BytesEq(idxA, idxB, idxBytes) || !BytesEq(qjA, qjB, 16)) {
        std::printf("FAIL DeviceSqrt path idx/qjl vs std::sqrt twin\n");
        return 1;
    }
    if (std::fabs(normA - normB) > 1e-5f || std::fabs(gammaA - gammaB) > 1e-5f) {
        std::printf("FAIL DeviceSqrt norm/gamma %f/%f %f/%f\n", normA, normB, gammaA, gammaB);
        return 1;
    }
    std::printf("PASS ProcessOne语义 DeviceSqrt ↔ twin idx/qjl 字节一致（norm/gamma≤1e-5）\n");

    // soft-bf16 输入（模拟 Kernel 读 bf16）
    float xBf[kHeadDim];
    for (int i = 0; i < kHeadDim; ++i) {
        xBf[i] = SoftBf16RoundTrip(x[i]);
    }
    EncodeOne(xBf, PtrMat{rotation.data()}, PtrMat{qjl.data()}, DeviceSqrt{}, mseBits, idxB, qjB, &normB, &gammaB);
    EncodeOneHost(xBf, rotation.data(), qjl.data(), StdSqrt, mseBits, idxA, qjA, &normA, &gammaA);
    if (!BytesEq(idxA, idxB, idxBytes) || !BytesEq(qjA, qjB, 16)) {
        std::printf("FAIL bf16-roundtrip DeviceSqrt vs StdSqrt idx/qjl\n");
        return 1;
    }
    std::printf("PASS soft-bf16 input DeviceSqrt ↔ StdSqrt idx/qjl\n");

    for (int bits : {2, 3, 4}) {
        EncodeOneHost(x, rotation.data(), qjl.data(), StdSqrt, bits, idxA, qjA, &normA, &gammaA);
        EncodeOne(x, PtrMat{rotation.data()}, PtrMat{qjl.data()}, DeviceSqrt{}, bits, idxB, qjB, &normB, &gammaB);
        const int nb = IdxBytes(bits);
        if (!BytesEq(idxA, idxB, nb) || !BytesEq(qjA, qjB, 16)) {
            std::printf("FAIL mse_bits=%d DeviceSqrt vs StdSqrt\n", bits);
            return 1;
        }
    }
    std::printf("PASS mse_bits={2,3,4} DeviceSqrt ↔ StdSqrt\n");

    std::printf("ALL PASS verify_kernel_encode_vs_twin\n");
    return 0;
}
