/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
// Standalone CPU check: c++ -std=c++17 tiling_validation_test.cpp -o tiling_test
#include "../tiling_validation.h"
#include <array>
#include <cstdio>
#include <cstdlib>

namespace V = OpsTransformer::CatccosUdma::TilingValidation;

static void Check(bool value, const char *message)
{
    if (!value) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main()
{
    const std::array<std::array<int64_t, 3>, 3> shapes{{
        {{128, 256, 256}}, {{256, 128, 256}}, {{256, 256, 256}}}};
    V::AllToAllV a{};
    for (auto shape : shapes) {
        for (int64_t stages : {2, 3}) {
            for (int64_t split : {1, 2, 4}) {
                const int64_t values[]{shape[0], shape[1], shape[2], 64,
                    2048, stages, 32, split, 16, 8};
                Check(V::ParseAllToAllV(values, 10, true, a) == nullptr,
                      "every compiled export instance must be selectable");
                Check(V::ValidateAllToAllV(a, 131072, 2048, 8, 32, 262144) == nullptr,
                      "compiled export instance must fit the representative workload");
            }
        }
    }
    int64_t values[]{128, 256, 256, 64, 1280, 2, 32, 1, 0, 1};
    Check(V::ParseAllToAllV(values, 9, false, a) != nullptr, "reject wrong field count");
    values[0] = 0;
    Check(V::ParseAllToAllV(values, 10, false, a) != nullptr, "reject zero tile");
    values[0] = 64;
    Check(V::ParseAllToAllV(values, 10, false, a) != nullptr, "reject uncompiled shape");
    values[0] = 128;
    values[5] = 3;
    Check(V::ParseAllToAllV(values, 10, false, a) != nullptr, "reject non-export stage3");
    values[5] = 2;
    values[7] = 2;
    Check(V::ParseAllToAllV(values, 10, false, a) != nullptr, "reject non-export split2");
    values[7] = 1;
    values[4] = 1281;
    Check(V::ParseAllToAllV(values, 10, false, a) != nullptr, "reject unaligned stage");
    values[4] = 1280;
    Check(V::ParseAllToAllV(values, 10, false, a) == nullptr, "valid default override");
    Check(V::ValidateAllToAllV(a, 131072, 2048, 8, 7, 262144) != nullptr,
          "reject insufficient peer workers");
    Check(V::ValidateAllToAllV(a, 131072, 2048, 8, 32, 0) != nullptr,
          "reject unknown UB capacity");
    a.selfCopyCores = 26;
    Check(V::ValidateAllToAllV(a, 131072, 2048, 8, 32, 262144) != nullptr,
          "reject self workers beyond available pool");
    a.selfCopyCores = 1;
    a.localCopyTileRows = 255;
    Check(V::ValidateAllToAllV(a, 131072, 2048, 8, 32, 262144) == nullptr,
          "UB boundary below reserved scratch remains valid");
    a.localCopyTileRows = 256;
    Check(V::ValidateAllToAllV(a, 131072, 2048, 8, 32, 262144) != nullptr,
          "reject UB tile aliasing the scratch region");
    a.localCopyTileRows = 32;
    a.stageRows = 16384;
    Check(V::ValidateAllToAllV(a, 131072, 2048, 8, 32, 262144) != nullptr,
          "reject workspace overflow");
    a.stageRows = 128;
    a.commBlockM = 1;
    Check(V::ValidateAllToAllV(a, 400000000, 256, 8, 32, 262144) != nullptr,
          "reject int32 completion counter overflow");
    Check(V::ValidateAllToAllVGrid(a, 1024, 8, 32) == nullptr,
          "ordinary MMAD grid remains valid");
    Check(V::ValidateAllToAllVGrid(a, 4294967295, 8, 32) != nullptr,
          "reject N ceiling overflow");
    a.stageRows = 1048576;
    Check(V::ValidateAllToAllVGrid(a, 16777216, 8, 32) != nullptr,
          "reject MMAD task count overflow");

    V::GroupedMatMul g{};
    int64_t grouped[]{128, 256, 256, 14, 31};
    Check(V::ParseGroupedMatMul(grouped, 5, g) == nullptr, "valid GMM wire");
    Check(V::ValidateGroupedMatMul(g, 2048, 8, 32, 262144) == nullptr,
          "valid GMM profile");
    g.localCopyTileRows = 32;
    Check(V::ValidateGroupedMatMul(g, 2048, 8, 32, 262144) != nullptr,
          "explicit GMM tiling must reject UB overflow rather than clamp");
    g.localCopyTileRows = 1;
    g.commInterval = 1;
    Check(V::ValidateGroupedMatMul(g, 2048, 8, 32, 262144) != nullptr,
          "reject too few tiles to fill an N group");
    g.commInterval = 3;
    Check(V::ValidateGroupedMatMul(g, 256, 8, 30, 262144) != nullptr,
          "reject non-divisible rank schedule");
    g.commInterval = 2147483647;
    Check(V::ValidateGroupedMatMul(g, 256, 8, 32, 262144) != nullptr,
          "reject uint32 block schedule overflow");
    g.commInterval = 100000;
    Check(V::ValidateGroupedMatMul(g, 256, 8, 32, 262144) != nullptr,
          "reject GMM workspace overflow");
    grouped[0] = 256;
    Check(V::ParseGroupedMatMul(grouped, 5, g) != nullptr, "reject uncompiled GMM square shape");
    grouped[1] = 128;
    grouped[4] = 8;
    Check(V::ParseGroupedMatMul(grouped, 5, g) == nullptr, "select GMM second shape");
    Check(V::ValidateGroupedMatMul(g, 2048, 8, 32, 262144) == nullptr,
          "valid second GMM shape");
    std::puts("Native tiling validation: compiled variants and boundary checks passed");
    return 0;
}
