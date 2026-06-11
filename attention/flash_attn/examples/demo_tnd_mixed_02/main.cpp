#include <cstdio>
#include <cstdint>
#include "flash_attn_def.h"
#include "tiling_key.h"

int main()
{
    CaseParams params;
    params.bSize = 8;
    params.n1Size = 4;
    params.n2Size = 4;
    params.s1Size = 118;
    params.s2Size = 118;
    params.qkHeadDim = 128;
    params.vHeadDim = 128;
    params.maskMode = 3;
    params.winLeft = -1;
    params.winRight = -1;
    params.qLayout = FaLayout::TND;
    params.kvLayout = FaLayout::TND;
    params.outLayout = FaLayout::TND;

    FaTilingKeyInfo keyInfo;
    uint32_t sOuter = 0;
    uint32_t sInner = 0;
    uint64_t tilingKey = GenTilingKey(params, keyInfo, sOuter, sInner);

    printf("=== Flash Attention TilingKey Demo: TND_MIXED_02 ===\n\n");
    printf("Case Parameters:\n");
    printf("  B=%ld, N1=%ld, N2=%ld\n", params.bSize, params.n1Size, params.n2Size);
    printf("  s1Size=%ld, s2Size=%ld\n", params.s1Size, params.s2Size);
    printf("  vHeadDim=%ld, maskMode=%ld (CAUSAL)\n", params.vHeadDim, params.maskMode);
    printf("  layout_q=TND, layout_kv=TND, layout_out=TND\n\n");

    printf("SplitPolicy Result:\n");
    printf("  sOuterFactor=%u, sInnerFactor=%u\n\n", sOuter, sInner);

    printf("TilingKey Fields:\n");
    printf("  InOutLayoutType = %lu (TND=2)\n", keyInfo.inputLayout);
    printf("  KvLayoutType    = %lu (continuous=0)\n", keyInfo.kvLayoutType);
    printf("  HasAttenMask    = %d (causal=1)\n", keyInfo.hasAttenMask);
    printf("  Config          = %lu (sOuter=64 -> config=0)\n\n", keyInfo.config);

    printf("TilingKey Computation:\n");
    printf("  key = %lu | (%lu << 8) | (%d << 16) | (%lu << 17)\n", keyInfo.inputLayout, keyInfo.kvLayoutType,
           keyInfo.hasAttenMask, keyInfo.config);
    printf("      = %lu + %lu + %lu + %lu\n", keyInfo.inputLayout, keyInfo.kvLayoutType << 8,
           static_cast<uint64_t>(keyInfo.hasAttenMask) << 16, keyInfo.config << 17);
    printf("      = %lu\n\n", tilingKey);

    printf("Binary:  0b");
    for (int i = 17; i >= 0; --i) {
        printf("%lu", (tilingKey >> i) & 1);
        if (i == 17 || i == 16 || i == 8)
            printf("_");
    }
    printf("\n");
    printf("Hex:     0x%lx\n\n", tilingKey);

    const uint64_t expectedKey = 65538;
    if (tilingKey == expectedKey) {
        printf("[PASS] TilingKey = %lu == %lu (expected)\n", tilingKey, expectedKey);
        return 0;
    } else {
        printf("[FAIL] TilingKey = %lu != %lu (expected)\n", tilingKey, expectedKey);
        return 1;
    }
}
