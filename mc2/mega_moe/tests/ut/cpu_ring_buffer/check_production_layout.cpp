/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

// Compile the production header directly. The include stubs only replace
// unavailable toolkit arithmetic headers and the missing Windows UCRT.
#include "../../../op_kernel/arch35/common/mega_moe_workspace.h"

using namespace MegaMoeImpl;

// A freestanding Windows entry point allows actual executable checks with an
// otherwise incomplete compiler installation. No CRT/device APIs are called.
extern "C" void *memset(void *target, int value, uint64_t size)
{
    volatile uint8_t *bytes = static_cast<volatile uint8_t *>(target);
    for (uint64_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<uint8_t>(value);
    }
    return target;
}
extern "C" int _fltused = 0;

static int64_t Align512(int64_t value)
{
    return ((value + 511) / 512) * 512;
}

static int CheckOneLayout(MegaMoeTilingData &tiling)
{
    const bool expectedRing = tiling.topoType == 0 && tiling.dedupMode == 0 &&
                              tiling.combineQuantMode == 0 && tiling.mGroupsPerWave != 0;
    if (IsMteRingBufferEnabled(&tiling) != expectedRing) {
        return 11;
    }
    const int64_t rows = expectedRing ? 3LL * 256 * tiling.mGroupsPerWave : tiling.maxOutputSize;
    if (CalcMteRingRowCapacity(&tiling) != rows) {
        return 12;
    }

    const WorkspaceLayout layout(&tiling);
    const bool fp4 = tiling.moeGmmMode == GMM_MODE_A4W4_ND || tiling.moeGmmMode == GMM_MODE_A4W4_NZ;
    const int64_t inputBytes = rows * tiling.h / (fp4 ? 2 : 1);
    const int64_t inputScaleBytes = rows * ((tiling.h + 63) / 64) * 2;
    const int64_t activationBytes = rows * tiling.hiddenDim / 2;
    const int64_t activationScaleBytes = rows * ((tiling.hiddenDim / 2 + 63) / 64) * 2;
    if (layout.dispatchRevScaleOffset - layout.dispatchRevDataOffset != Align512(inputBytes)) {
        return 13;
    }
    const int64_t dispatchNext = layout.dispatchRevWeightsOffset >= 0 ? layout.dispatchRevWeightsOffset :
                                layout.activationQuantDataOffset;
    if (dispatchNext - layout.dispatchRevScaleOffset != Align512(inputScaleBytes)) {
        return 14;
    }
    if (layout.activationQuantScaleOffset - layout.activationQuantDataOffset != Align512(activationBytes)) {
        return 15;
    }
    if (layout.expertRecvTokenCountOffset - layout.activationQuantScaleOffset != Align512(activationScaleBytes)) {
        return 16;
    }
    const bool gmGmm1 = tiling.moeGmmMode == GMM_MODE_A8W4_NZ || tiling.topkWeightsPrefetch == 1;
    if ((layout.gmm1MmadResOffset >= 0) != gmGmm1) {
        return 17;
    }
    if (gmGmm1) {
        int64_t expectedGmm2Offset = layout.gmm1MmadResOffset + rows * tiling.hiddenDim * 2;
        if (tiling.topoType == 0) {
            expectedGmm2Offset = Align512(expectedGmm2Offset);
        }
        if (layout.gmm2MmadResOffset != expectedGmm2Offset) {
            return 18;
        }
    }
    const int64_t gmm2End = layout.gmm2MmadResOffset + Align512(rows * tiling.h * 2);
    if (gmm2End > layout.workspaceSize) {
        return 19;
    }
    if (tiling.topoType == 0 && tiling.dedupMode == 0 && tiling.sharedExpertNum == 0 &&
        tiling.topkWeightsPrefetch == 0 && gmm2End != layout.workspaceSize) {
        return 20;
    }
    if ((layout.ringProgressOffset >= 0) != expectedRing) {
        return 21;
    }
    // Metadata must still address all received logical rows, independently of
    // the much smaller (or very small-input larger) physical data rings.
    if (layout.flagActivationToGmm2Offset - layout.metaInfoOffset != Align512(tiling.maxOutputSize * 32LL)) {
        return 22;
    }
    return 0;
}

static int CheckPaddingAndWrap()
{
    const uint64_t counts[] = {0, 1, 100, 255, 256, 257, 300, 513, 1024};
    for (uint32_t i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
        const uint64_t expected = ((counts[i] + 255) / 256) * 256;
        if (CalcMtePaddedRowCount(counts[i]) != expected) {
            return 31;
        }
    }
    for (uint32_t budget = 1; budget <= 33; ++budget) {
        const uint64_t ringRows = 3ULL * budget * 256;
        for (uint64_t group = 0; group < 20ULL * budget; ++group) {
            for (uint32_t row = 0; row < 256; ++row) {
                const uint64_t physical = CalcMteRingPhysicalRow(group * 256 + row, ringRows);
                const uint64_t expected = (group % (3 * budget)) * 256 + row;
                if (physical != expected || physical >= ringRows) {
                    return 32;
                }
            }
        }
    }
    return CalcMteRingPhysicalRow(12345, 0) == 12345 ? 0 : 33;
}

static int RunChecks()
{
    int result = CheckPaddingAndWrap();
    if (result != 0) {
        return result;
    }
    const uint32_t hidden[] = {1024, 4096, 8192};
    const uint32_t intermediate[] = {256, 2048, 4096};
    const uint32_t budgets[] = {0, 1, 7, 16};
    const uint32_t routeCaps[] = {1, 128, 65536};
    for (uint32_t h : hidden) {
        for (uint32_t intermediateHidden : intermediate) {
            for (uint32_t budget : budgets) {
                for (uint32_t routeCap : routeCaps) {
                    for (uint8_t mode = 0; mode <= GMM_MODE_A4W4_NZ; ++mode) {
                        for (int32_t dedup = 0; dedup < 4; ++dedup) {
                            for (int64_t quant = 0; quant < 2; ++quant) {
                                for (int64_t topology = 0; topology < 2; ++topology) {
                                    for (int32_t prefetch = 0; prefetch < 2; ++prefetch) {
                                        for (uint32_t shared = 0; shared < 2; ++shared) {
                                            MegaMoeTilingData tiling{};
                                            tiling.h = h;
                                            tiling.hiddenDim = intermediateHidden * 2;
                                            tiling.mGroupsPerWave = budget;
                                            tiling.maxOutputSize = routeCap;
                                            tiling.moeExpertPerRank = 4;
                                            tiling.epWorldSize = 8;
                                            tiling.rankNumPerServer = 4;
                                            tiling.topK = 8;
                                            tiling.bs = 128;
                                            tiling.numMaxTokensPerRank = 128;
                                            tiling.aicNum = 32;
                                            tiling.moeGmmMode = mode;
                                            tiling.sharedGmmMode = mode;
                                            tiling.dedupMode = dedup;
                                            tiling.combineQuantMode = quant;
                                            tiling.topoType = topology;
                                            tiling.topkWeightsPrefetch = prefetch;
                                            tiling.sharedExpertNum = shared;
                                            tiling.maxTilesPerExpert = 1024;
                                            result = CheckOneLayout(tiling);
                                            if (result != 0) {
                                                return result;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

#if !defined(CPU_RING_EMBEDDED_MAIN)
extern "C" int main()
{
#if defined(CPU_RING_FORCE_FAILURE)
    return 91; // The runner checks that the freestanding process preserves exit status.
#else
    return RunChecks();
#endif
}
#endif
