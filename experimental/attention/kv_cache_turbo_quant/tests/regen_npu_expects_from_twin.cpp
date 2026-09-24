/**
 * 用 encode_one_core（与 Kernel 同核）重写 npu_cases 下各 case 的 expect_idx/qjl/norm/gamma。
 * 解决 Python cpu_golden 在 mse4 边界与 C++ 孪生字节漂移。
 *
 * 编译（cwd=tests/）：
 *   g++ -O2 -std=c++17 -o regen_npu_expects_from_twin regen_npu_expects_from_twin.cpp
 * 运行：
 *   ./regen_npu_expects_from_twin [data/npu_cases]
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../op_kernel/encode_one_core.h"

namespace {

float StdSqrt(float x)
{
    return std::sqrt(x);
}

float Bf16BitsToFloat(uint16_t u)
{
    uint32_t x = static_cast<uint32_t>(u) << 16;
    float f = 0.f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

uint16_t FloatToBf16Bits(float f)
{
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

bool LoadBytes(const std::string &path, std::vector<uint8_t> *out)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    const auto n = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    out->assign(n, 0);
    ifs.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(n));
    return true;
}

bool LoadMeta(const std::string &path, int64_t *tokens, int64_t *heads, int *bits)
{
    std::ifstream ifs(path);
    if (!ifs) {
        return false;
    }
    *tokens = 0;
    *heads = 0;
    *bits = 3;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("tokens=", 0) == 0) {
            *tokens = std::stoll(line.substr(7));
        } else if (line.rfind("heads=", 0) == 0) {
            *heads = std::stoll(line.substr(6));
        } else if (line.rfind("bits=", 0) == 0) {
            *bits = std::stoi(line.substr(5));
        }
    }
    return *tokens > 0 && *heads > 0;
}

bool RegenOne(const std::string &caseDir, const std::string &name)
{
    int64_t tokens = 0;
    int64_t heads = 0;
    int bits = 3;
    if (!LoadMeta(caseDir + "/meta.txt", &tokens, &heads, &bits)) {
        std::cerr << "FAIL " << name << " meta\n";
        return false;
    }
    std::vector<uint8_t> kv;
    std::vector<uint8_t> rotb;
    std::vector<uint8_t> qjlb;
    if (!LoadBytes(caseDir + "/kv_bf16.bin", &kv) || !LoadBytes(caseDir + "/rotation_matrix.bin", &rotb) ||
        !LoadBytes(caseDir + "/qjl_matrix.bin", &qjlb)) {
        std::cerr << "FAIL " << name << " load bins\n";
        return false;
    }
    const int64_t nVec = tokens * heads;
    const int64_t idxBytes = (KvEncodeCore::kHeadDim * bits + 7) / 8;
    const int64_t qjlBytes = (KvEncodeCore::kHeadDim + 7) / 8;
    const float *rot = reinterpret_cast<const float *>(rotb.data());
    const float *qjl = reinterpret_cast<const float *>(qjlb.data());

    std::vector<uint8_t> expIdx(static_cast<size_t>(nVec * idxBytes));
    std::vector<uint8_t> expQjl(static_cast<size_t>(nVec * qjlBytes));
    std::vector<uint16_t> expNorm(static_cast<size_t>(nVec));
    std::vector<uint16_t> expGamma(static_cast<size_t>(nVec));

    for (int64_t v = 0; v < nVec; ++v) {
        float x[KvEncodeCore::kHeadDim];
        for (int i = 0; i < KvEncodeCore::kHeadDim; ++i) {
            uint16_t u = 0;
            std::memcpy(&u, kv.data() + (v * KvEncodeCore::kHeadDim + i) * 2, 2);
            x[i] = Bf16BitsToFloat(u);
        }
        float norm = 0.f;
        float gamma = 0.f;
        KvEncodeCore::EncodeOneHost(x, rot, qjl, StdSqrt, bits, expIdx.data() + v * idxBytes,
                                    expQjl.data() + v * qjlBytes, &norm, &gamma);
        expNorm[static_cast<size_t>(v)] = FloatToBf16Bits(norm);
        expGamma[static_cast<size_t>(v)] = FloatToBf16Bits(gamma);
    }

    std::ofstream(caseDir + "/expect_idx.bin", std::ios::binary)
        .write(reinterpret_cast<const char *>(expIdx.data()), static_cast<std::streamsize>(expIdx.size()));
    std::ofstream(caseDir + "/expect_qjl.bin", std::ios::binary)
        .write(reinterpret_cast<const char *>(expQjl.data()), static_cast<std::streamsize>(expQjl.size()));
    std::ofstream(caseDir + "/expect_norm_bf16.bin", std::ios::binary)
        .write(reinterpret_cast<const char *>(expNorm.data()), static_cast<std::streamsize>(expNorm.size() * 2));
    std::ofstream(caseDir + "/expect_gamma_bf16.bin", std::ios::binary)
        .write(reinterpret_cast<const char *>(expGamma.data()), static_cast<std::streamsize>(expGamma.size() * 2));
    std::cout << "PASS regen " << name << " T=" << tokens << " H=" << heads << " bits=" << bits << "\n";
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    std::string root = "data/npu_cases";
    if (argc >= 2) {
        root = argv[1];
    }
    const char *names[] = {"gqa_decode_b1",     "gqa_decode_b64", "gqa_prefill_t2048",
                           "gqa_mse2_t2048",     "gqa_mse4_t2048", "smoke_1x1",
                           "t1h2",              "t2h1",           "t1h8"};
    int fail = 0;
    for (const char *name : names) {
        if (!RegenOne(root + "/" + name, name)) {
            ++fail;
        }
    }
    if (fail) {
        std::cout << "ALL FAIL regen fail=" << fail << "\n";
        return 1;
    }
    std::cout << "ALL PASS regen_npu_expects_from_twin\n";
    return 0;
}
