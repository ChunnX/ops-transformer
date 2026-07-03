// -----------------------------------------------------------------------------------------------------------
// Verification: CalcLeastCclBufferSizeA2A3 (mega_moe.cpp)
//              vs tiling_arch22.cpp CalcLeastCclBufferSize
//
// Build and run on A2 / A3 machine:
//   g++ -std=c++17 -O0 -o test_verify_a2a3 test_verify_a2a3.cpp && ./test_verify_a2a3
// -----------------------------------------------------------------------------------------------------------
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int64_t RESERVED_SPACE_SIZE = 10LL * 1024 * 1024;

int64_t CeilAlign(int64_t val, int64_t align)
{
    return (val + align - 1) / align * align;
}

// ========== tiling_arch22.cpp CalcLeastCclBufferSize (original) ==========
static int64_t CalcLeastCclBufferSize_Tiling(int64_t maxRecvTokenNum, int64_t h,
    int64_t epWorldSize, int64_t expertPerRank,
    bool isQuantRouting, bool /*isW4A8*/, bool isA3,
    int64_t bs, int64_t topK)
{
    int64_t offsetTokenPerExpert = epWorldSize *
        CeilAlign(epWorldSize * expertPerRank + 1, ALIGN_128) *
        static_cast<int64_t>(sizeof(int32_t));

    int64_t offsetAAfterDispatch = 0;
    if (isA3) {
        offsetAAfterDispatch = bs * topK *
            (isQuantRouting ? (h + ALIGN_512) : h * sizeof(int16_t));
    } else {
        offsetAAfterDispatch = maxRecvTokenNum *
            (isQuantRouting ? (h + ALIGN_512) : h * sizeof(int16_t));
    }
    int64_t offsetD = bs * topK * h * sizeof(int16_t);
    int64_t winInTensorSize = offsetAAfterDispatch + offsetD;

    int64_t winOutTensorSize = 0;
    if (!isA3) {
        int64_t offsetA = bs * topK *
            (!isQuantRouting ? h * sizeof(int16_t) : (h + ALIGN_512));
        int64_t offsetC = maxRecvTokenNum * h * sizeof(int16_t);
        winOutTensorSize = offsetA + offsetC;
    }
    int64_t offsetTensor = std::max(winInTensorSize, winOutTensorSize);
    if (isQuantRouting) {
        offsetTensor += (isA3 ? bs * topK : maxRecvTokenNum) * sizeof(float);
    }

    int64_t offsetFlag = epWorldSize * ALIGN_512;
    if (!isA3) {
        int64_t dispatchFlag = epWorldSize * expertPerRank * 64;
        int64_t allgatherFlag = isQuantRouting ? static_cast<int64_t>(0) : epWorldSize * 64;
        offsetFlag += dispatchFlag + allgatherFlag;
    }
    return offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE;
}

// ========== mega_moe.cpp CalcLeastCclBufferSizeA2A3 (new) ==========
static int64_t CalcLeastCclBufferSizeA2A3_New(int64_t maxRecvTokenNum, int64_t h,
    int64_t epWorldSize, int64_t expertPerRank,
    bool isQuantRouting, bool isA3,
    int64_t bs, int64_t topK)
{
    int64_t offsetTokenPerExpert = epWorldSize *
        CeilAlign(epWorldSize * expertPerRank + 1, ALIGN_128) *
        static_cast<int64_t>(sizeof(int32_t));

    int64_t offsetAAfterDispatch;
    if (isA3) {
        offsetAAfterDispatch = bs * topK *
            (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    } else {
        offsetAAfterDispatch = maxRecvTokenNum *
            (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    }
    int64_t offsetD = bs * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winInTensorSize = offsetAAfterDispatch + offsetD;

    int64_t winOutTensorSize = 0;
    if (!isA3) {
        int64_t offsetA = bs * topK *
            (!isQuantRouting ? h * static_cast<int64_t>(sizeof(int16_t)) : (h + ALIGN_512));
        int64_t offsetC = maxRecvTokenNum * h * static_cast<int64_t>(sizeof(int16_t));
        winOutTensorSize = offsetA + offsetC;
    }
    int64_t offsetTensor = std::max(winInTensorSize, winOutTensorSize);
    if (isQuantRouting) {
        offsetTensor += (isA3 ? bs * topK : maxRecvTokenNum) * static_cast<int64_t>(sizeof(float));
    }

    int64_t offsetFlag = epWorldSize * ALIGN_512;
    if (!isA3) {
        offsetFlag += epWorldSize * expertPerRank * 64LL;
        if (!isQuantRouting) {
            offsetFlag += epWorldSize * 64LL;
        }
    }

    return offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE;
}

// ========== test harness ==========
struct TestCase {
    int64_t maxRecvTokenNum;
    int64_t h;
    int64_t epWorldSize;
    int64_t expertPerRank;
    bool    isQuantRouting;
    bool    isA3;
    int64_t bs;
    int64_t topK;
};

int main()
{
    TestCase cases[] = {
        // === A2 (isA3=false) ===
        // BF16, no quant
        {256, 4096, 8, 4,  false, false, 256, 6},
        {256, 4096, 8, 4,  false, false, 256, 8},
        {128, 5120, 4, 2,  false, false, 128, 6},
        {512, 7168, 16, 8, false, false, 512, 8},
        {512, 4096, 32, 4, false, false, 512, 6},
        {64,  4096, 2,  2,  false, false, 64,  6},
        // MX quant
        {256, 4096, 8, 4,  true,  false, 256, 6},
        {256, 4096, 8, 4,  true,  false, 256, 8},
        {128, 5120, 4, 2,  true,  false, 128, 6},
        {512, 7168, 16, 8, true,  false, 512, 8},
        {64,  4096, 2,  2,  true,  false, 64,  6},

        // === A3 (isA3=true) ===
        // BF16, no quant
        {256, 4096, 8, 4,  false, true,  256, 6},
        {256, 4096, 8, 4,  false, true,  256, 8},
        {128, 5120, 4, 2,  false, true,  128, 6},
        {512, 7168, 16, 8, false, true,  512, 8},
        {64,  4096, 2,  2,  false, true,  64,  6},
        // MX quant
        {256, 4096, 8, 4,  true,  true,  256, 6},
        {128, 5120, 4, 2,  true,  true,  128, 6},
        {512, 7168, 16, 8, true,  true,  512, 8},
        {64,  4096, 2,  2,  true,  true,  64,  6},

        // edge cases
        {1,   4096, 2,  1,  false, false, 1,   6},
        {1,   4096, 2,  1,  false, true,  1,   6},
        {1,   4096, 2,  1,  true,  false, 1,   6},
        {1,   4096, 2,  1,  true,  true,  1,   6},
    };

    int passed = 0;
    int failed = 0;

    for (const auto &c : cases) {
        int64_t tilingBytes = CalcLeastCclBufferSize_Tiling(
            c.maxRecvTokenNum, c.h, c.epWorldSize, c.expertPerRank,
            c.isQuantRouting, false /*isW4A8 unused*/, c.isA3,
            c.bs, c.topK);

        int64_t newBytes = CalcLeastCclBufferSizeA2A3_New(
            c.maxRecvTokenNum, c.h, c.epWorldSize, c.expertPerRank,
            c.isQuantRouting, c.isA3,
            c.bs, c.topK);

        // Also verify suggestion values match
        constexpr int64_t MB_SIZE = 1024LL * 1024LL;
        int64_t tilingSuggestion = (tilingBytes + MB_SIZE) / MB_SIZE;
        int64_t newSuggestion = (newBytes + MB_SIZE) / MB_SIZE;

        bool bytesOk = (tilingBytes == newBytes);
        bool suggestOk = (tilingSuggestion == newSuggestion);

        printf("[%s] A%d quant=%d BS=%ld EP=%ld E=%ld H=%ld K=%ld maxRecv=%ld\n",
               (bytesOk && suggestOk) ? "PASS" : "FAIL",
               c.isA3 ? 3 : 2, c.isQuantRouting,
               c.bs, c.epWorldSize, c.expertPerRank, c.h, c.topK, c.maxRecvTokenNum);
        printf("  tiling: %ld bytes -> suggestion %ld MB\n", tilingBytes, tilingSuggestion);
        if (!bytesOk) {
            printf("  NEW:    %ld bytes  *** MISMATCH ***\n", newBytes);
            failed++;
        } else if (!suggestOk) {
            printf("  NEW suggestion %ld MB  *** MISMATCH ***\n", newSuggestion);
            failed++;
        } else {
            passed++;
        }
    }

    printf("\n=== Result: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
