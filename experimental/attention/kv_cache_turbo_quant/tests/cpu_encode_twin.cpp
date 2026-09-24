/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cpu_encode_twin.cpp
 * \brief
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

using KvEncodeCore::EncodeOneHost;
using KvEncodeCore::PackBits;
using KvEncodeCore::kHeadDim;

float StdSqrt(float x)
{
    return std::sqrt(x);
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

} // namespace

int main()
{
    uint8_t idxVals[8];
    for (int i = 0; i < 8; ++i) {
        idxVals[i] = 3;
    }
    uint8_t packed[3] = {0, 0, 0};
    PackBits(idxVals, 8, 3, packed);
    if (packed[0] != 219 || packed[1] != 182 || packed[2] != 109) {
        std::printf("FAIL pack known: %u %u %u\n", packed[0], packed[1], packed[2]);
        return 1;
    }
    std::printf("PASS pack known bytes 219,182,109\n");

    std::vector<float> rotation;
    std::vector<float> qjl;
    if (!LoadBin("data/rotation_matrix.bin", &rotation, 128ull * 128ull) ||
        !LoadBin("data/qjl_matrix.bin", &qjl, 128ull * 128ull)) {
        std::printf("FAIL load bin under tests/data（请先 python3 gen_selftest_matrices.py）\n");
        return 1;
    }

    float zero[kHeadDim] = {0};
    uint8_t idx[48] = {0};
    uint8_t qj[16] = {0};
    float norm = -1.f;
    float gamma = -1.f;
    EncodeOneHost(zero, rotation.data(), qjl.data(), StdSqrt, 3, idx, qj, &norm, &gamma);
    if (norm != 0.f || gamma != 0.f) {
        std::printf("FAIL zero norm/gamma %f %f\n", norm, gamma);
        return 1;
    }
    if (idx[0] != 219 || idx[1] != 182 || idx[2] != 109) {
        std::printf("FAIL zero idx head %u %u %u\n", idx[0], idx[1], idx[2]);
        return 1;
    }
    std::printf("PASS zero_vector mse3\n");

    float x[kHeadDim];
    for (int i = 0; i < kHeadDim; ++i) {
        x[i] = static_cast<float>((i % 17) - 8) * 0.1f;
    }
    EncodeOneHost(x, rotation.data(), qjl.data(), StdSqrt, 3, idx, qj, &norm, &gamma);
    std::printf("PASS encode_one mse3 norm=%f gamma=%f idx0=%u qjl0=%u\n", norm, gamma, idx[0], qj[0]);

    std::ofstream ofs("data/twin_smoke_out.bin", std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(idx), 48);
    ofs.write(reinterpret_cast<const char *>(qj), 16);
    ofs.write(reinterpret_cast<const char *>(&norm), 4);
    ofs.write(reinterpret_cast<const char *>(&gamma), 4);
    ofs.write(reinterpret_cast<const char *>(x), static_cast<std::streamsize>(sizeof(x)));
    std::printf("ALL PASS cpu_encode_twin\n");
    return 0;
}
