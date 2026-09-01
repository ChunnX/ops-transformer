/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SageAttention2 chained host driver: MM1(INT8 QK^T) -> softmax -> MM2(FP8 PV).
 *
 * 多 kernel 串联方案（任务书 §4 Host+Kernel 融合）：
 *   1. MM1: q_int8 @ k_int8^T (int32 累加 + per-group/per-block scale 反量化)  -> S [Sq, Sk] fp16
 *   2. Host softmax: P = softmax(S)（S 已含 q_scale*k_scale*sm_scale）
 *   3. P 按 MX 规则量化 FP8 E4M3（每 32 元素一组 e8m0 scale）
 *   4. MM2: P_fp8 @ V_fp8 (FP8 E4M3 + MX scale) -> O [Sq, D] fp32
 *   5. rescale O（乘 v_scale 在后续 Host 管线；本程序输出原始 O 验证通路）
 *
 * 编译：bash scripts/build.sh sageattn_chain -DCATLASS_ARCH=3510
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#include "catlass/gemm/kernel/quant_matmul_per_group_per_block_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------
struct ChainOptions {
    uint32_t sq{0};   // query seqlen
    uint32_t sk{0};   // key seqlen
    uint32_t d{0};    // head dim
    uint32_t deviceId{0};

    int Parse(int argc, const char** argv)
    {
        if (argc < 4) {
            printf("Usage: %s sq sk d [--device DEVICE_ID]\n", argv[0]);
            return -1;
        }
        sq = atoi(argv[1]);
        sk = atoi(argv[2]);
        d = atoi(argv[3]);
        for (int i = 4; i < argc; ++i) {
            std::string flag(argv[i]);
            if (flag == "--device") { deviceId = atoi(argv[++i]); }
        }
        return 0;
    }
};

// ---------------------------------------------------------------------------
// FP8 E4M3 encode: return raw byte (sign|exp4|mant3), OCP MX grid.
// ---------------------------------------------------------------------------
static uint8_t EncodeE4m3(float v)
{
    constexpr float kMax = 448.0f;
    v = std::clamp(v, -kMax, kMax);
    if (v == 0.0f) return 0x00;
    uint8_t sign = (v < 0.0f) ? 0x80 : 0x00;
    float a = std::fabs(v);
    int e = static_cast<int>(std::floor(std::log2(a))) + 7;  // E4M3 bias=7
    e = std::clamp(e, 0, 15);
    float base = std::exp2(static_cast<float>(e - 7));
    int m = static_cast<int>(std::lround((a / base - 1.0f) * 8.0f));
    m = std::clamp(m, 0, 8);
    if (m == 8) {
        m = 0;
        if (e < 15) ++e;
    }
    return static_cast<uint8_t>(sign | ((e & 0xF) << 3) | (m & 0x7));
}

/**
 * Host MX-FP8 quantize (per-32-elem e8m0 scale).
 * src: contiguous float data; blocks of 32 consecutive elements share one scale.
 * fp8Out: E4M3 raw bytes; scaleOut: e8m0 scale bytes (E = exp + 127, OCP MX).
 */
static void HostFp8Quantize(
    const std::vector<float>& src, size_t n, std::vector<int8_t>& fp8Out, std::vector<int8_t>& scaleOut)
{
    constexpr uint32_t BLK = 32;
    size_t numBlocks = CeilDiv(n, BLK);
    fp8Out.assign(n, 0);
    scaleOut.assign(numBlocks, 0);
    for (size_t b = 0; b < numBlocks; ++b) {
        float maxAbs = 0.0f;
        for (size_t i = b * BLK; i < std::min((b + 1) * BLK, n); ++i) {
            maxAbs = std::max(maxAbs, std::fabs(src[i]));
        }
        // e8m0 exponent: scale = 2^exp, emax(E4M3)=8 -> exp = floor(log2(maxAbs)) - 8
        int32_t exp = 0;
        if (maxAbs > 1e-30f) {
            exp = static_cast<int32_t>(std::floor(std::log2(maxAbs))) - 8;
        }
        exp = std::clamp(exp, -128, 127);
        // OCP MX e8m0 byte: E = exp + 127 (8-bit unsigned, stored in int8 buffer)
        scaleOut[b] = static_cast<int8_t>(static_cast<uint8_t>(exp + 127));
        float scale = std::exp2(static_cast<float>(exp));
        for (size_t i = b * BLK; i < std::min((b + 1) * BLK, n); ++i) {
            fp8Out[i] = static_cast<int8_t>(EncodeE4m3(src[i] / scale));
        }
    }
}

/** Host softmax over rows of a [Sq, Sk] fp16 S matrix -> P float [Sq, Sk]. */
static void HostSoftmax(const std::vector<half>& s, uint32_t sq, uint32_t sk, std::vector<float>& p)
{
    p.assign((size_t)sq * sk, 0.0f);
    for (uint32_t r = 0; r < sq; ++r) {
        float maxV = -1e30f;
        for (uint32_t c = 0; c < sk; ++c) {
            maxV = std::max(maxV, (float)s[(size_t)r * sk + c]);
        }
        float sum = 0.0f;
        for (uint32_t c = 0; c < sk; ++c) {
            float e = std::exp((float)s[(size_t)r * sk + c] - maxV);
            p[(size_t)r * sk + c] = e;
            sum += e;
        }
        for (uint32_t c = 0; c < sk; ++c) {
            p[(size_t)r * sk + c] /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// MM1: INT8 QK^T (based on 51 pattern) —— S [Sq, Sk] fp16
// ---------------------------------------------------------------------------
struct Mm1Result {
    std::vector<half> s;   // [Sq*Sk] fp16
};

static void RunMm1(
    const std::vector<int8_t>& qInt8, const std::vector<int8_t>& kInt8,
    const std::vector<float>& qScale, const std::vector<float>& kScale,
    uint32_t m, uint32_t n, uint32_t k, uint32_t gs, uint32_t bs,
    aclrtStream stream, Mm1Result& out)
{
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using Elementx1scale = float;
    using Elementx2scale = float;
    using ElementY = half;

    size_t lenA = m * k;
    size_t lenB = k * n;
    size_t lenX1scale = m * CeilDiv(k, gs);
    size_t lenX2scale = CeilDiv(k, bs) * CeilDiv(n, bs);
    size_t lenC = m * n;

    // Upload
    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceX1Scale{nullptr};
    uint8_t* deviceX2Scale{nullptr};
    uint8_t* deviceC{nullptr};
    uint8_t* deviceWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), lenA * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), lenB * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX1Scale), lenX1scale * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX2Scale), lenX2scale * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), lenC * sizeof(fp16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, lenA * sizeof(int8_t), qInt8.data(), lenA * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, lenB * sizeof(int8_t), kInt8.data(), lenB * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceX1Scale, lenX1scale * sizeof(float), qScale.data(), lenX1scale * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceX2Scale, lenX2scale * sizeof(float), kScale.data(), lenX2scale * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::Ascend950;
    using DispatchPolicy = Gemm::MmadPingpongPertile<ArchTag, false>;
    using L1TileShape = Shape<Int<256>, Int<128>, Int<128>>;
    using L0TileShape = Shape<Int<256>, Int<128>, Int<128>>;
    using ProblemShape = Catlass::GemmCoord;

    auto layoutA = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, layout::RowMajor>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(m, n);

    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementA, layout::RowMajor, ElementB, layout::RowMajor, ElementC, layout::RowMajor, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using EpilogueDispatchPolicy = Epilogue::BlockEpiloguePertile;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, L0TileShape, ElementY, ElementC, void, Elementx1scale, Elementx2scale,
        layout::RowMajor, layout::RowMajor>;

    uint32_t taskNum = CeilDiv(m, tla::get<0>(L1TileShape{})) * CeilDiv(n, tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);
    using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape>;

    using MatmulKernel =
        Gemm::Kernel::QuantMatmulPerGroupPerBlockTla<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulKernel::Arguments arguments{
        ProblemShape(m, n, k), deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, nullptr,
        {deviceC, deviceX2Scale, deviceX1Scale, nullptr, tla::get<0>(L0TileShape{}), tla::get<1>(L0TileShape{}),
         tla::get<2>(L0TileShape{})}};

    MatmulAdapter matmulOp;
    matmulOp.CanImplement(arguments);
    size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreUsed);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    out.s.assign(lenC, half(0));
    ACL_CHECK(aclrtMemcpy(out.s.data(), lenC * sizeof(fp16_t), deviceC, lenC * sizeof(fp16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceX1Scale));
    ACL_CHECK(aclrtFree(deviceX2Scale));
    ACL_CHECK(aclrtFree(deviceC));
    if (sizeWorkspace > 0) ACL_CHECK(aclrtFree(deviceWorkspace));
}

// ---------------------------------------------------------------------------
// MM2: FP8 PV (based on 53 pattern) —— O [Sq, D] fp32
// ---------------------------------------------------------------------------
static void RunMm2(
    const std::vector<int8_t>& pFp8, const std::vector<int8_t>& pScale,
    const std::vector<int8_t>& vFp8, const std::vector<int8_t>& vScale,
    uint32_t m, uint32_t n, uint32_t k, aclrtStream stream, std::vector<float>& outO)
{
    using ElementA = float8_e4m3_t;
    using ElementB = float8_e4m3_t;
    using ElementMxScale = float8_e8m0_t;
    using ElementC = float;

    size_t lenA = m * k;   // P [Sq, Sk]
    size_t lenB = k * n;   // V [Sk, D]
    size_t lenC = m * n;
    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);
    size_t lenMxScaleA = m * mxScaleAlignedK;
    size_t lenMxScaleB = mxScaleAlignedK * n;

    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceMxScaleA{nullptr};
    uint8_t* deviceMxScaleB{nullptr};
    uint8_t* deviceC{nullptr};
    uint8_t* deviceWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), lenA * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), lenB * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceMxScaleA), lenMxScaleA * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceMxScaleB), lenMxScaleB * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), lenC * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, lenA * sizeof(int8_t), pFp8.data(), lenA * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, lenB * sizeof(int8_t), vFp8.data(), lenB * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleA, lenMxScaleA * sizeof(int8_t), pScale.data(), lenMxScaleA * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleB, lenMxScaleB * sizeof(int8_t), vScale.data(), lenMxScaleB * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::Ascend950;
    using DispatchPolicy = Gemm::MmadMx<ArchTag, true, 16>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<256>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<128>>;

    // P: RowMajor [Sq, Sk]; V: ColumnMajor [Sk, D]（与 53 一致，B 按列主序）
    auto layoutA = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, layout::ColumnMajor>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, layout::RowMajor, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, layout::ColumnMajor, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(m, n);

    using TileCopy = Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, layout::RowMajor, ElementB, layout::ColumnMajor, ElementMxScale, decltype(layoutMxScaleA),
        ElementMxScale, decltype(layoutMxScaleB), ElementC, layout::RowMajor, void>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;

    uint32_t taskNum = CeilDiv(m, tla::get<0>(L1TileShape{})) * CeilDiv(n, tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    using MatmulKernel = Gemm::Kernel::MxMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulKernel::Arguments arguments{
        Catlass::GemmCoord(m, n, k), deviceA, layoutA, deviceB, layoutB, deviceMxScaleA,
        layoutMxScaleA, deviceMxScaleB, layoutMxScaleB, deviceC, layoutC, nullptr};

    MatmulAdapter matmulOp;
    matmulOp.CanImplement(arguments);
    size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreUsed);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    outO.assign(lenC, 0.0f);
    ACL_CHECK(aclrtMemcpy(outO.data(), lenC * sizeof(float), deviceC, lenC * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceMxScaleA));
    ACL_CHECK(aclrtFree(deviceMxScaleB));
    ACL_CHECK(aclrtFree(deviceC));
    if (sizeWorkspace > 0) ACL_CHECK(aclrtFree(deviceWorkspace));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
static void Run(const ChainOptions& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t sq = options.sq;
    uint32_t sk = options.sk;
    uint32_t d = options.d;
    constexpr uint32_t gs = 128;  // Q 的 groupsize（沿 D，D<=128 -> 每行一组）
    constexpr uint32_t bs = 128;  // K^T 的 blocksize
    constexpr uint32_t B = 1;

    // ---- 1. Host 侧量化输入: q/k int8 + scale 随机；V 用 float 随机后 FP8 量化 ----
    size_t lenQ = (size_t)sq * d;
    size_t lenK = (size_t)sk * d;
    size_t lenV = (size_t)sk * d;
    std::vector<int8_t> qInt8(lenQ), kInt8(lenK);
    std::vector<float> vFloat(lenV);
    std::vector<float> qScale(sq * CeilDiv(d, gs));
    std::vector<float> kScale(CeilDiv(d, bs) * CeilDiv(sk, bs));
    std::vector<float> vScale(B * d);   // per-channel（未实际用于 MM2，占位）
    (void)B;

    // 随机 int8 数据（-16..16），V float [-4,4]，scale 随机 0.05..1.0
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> idist(-16, 16);
    for (size_t i = 0; i < lenQ; ++i) qInt8[i] = (int8_t)idist(rng);
    for (size_t i = 0; i < lenK; ++i) kInt8[i] = (int8_t)idist(rng);
    std::uniform_real_distribution<float> fdist(-4.0f, 4.0f);
    for (size_t i = 0; i < lenV; ++i) vFloat[i] = fdist(rng);
    std::uniform_real_distribution<float> sdist(0.05f, 1.0f);
    for (size_t i = 0; i < qScale.size(); ++i) qScale[i] = sdist(rng);
    for (size_t i = 0; i < kScale.size(); ++i) kScale[i] = sdist(rng);
    for (size_t i = 0; i < vScale.size(); ++i) vScale[i] = sdist(rng);

    // ---- 2. MM1: INT8 QK^T -> S ----
    std::cout << "[chain] MM1 INT8 QK^T (" << sq << ", " << sk << ", " << d << ")..." << std::endl;
    Mm1Result mm1;
    RunMm1(qInt8, kInt8, qScale, kScale, sq, sk, d, gs, bs, stream, mm1);
    {
        float sMax = 0.0f;
        bool sFinite = true;
        for (size_t i = 0; i < mm1.s.size(); ++i) {
            float v = (float)mm1.s[i];
            if (!std::isfinite(v)) { sFinite = false; break; }
            sMax = std::max(sMax, std::fabs(v));
        }
        std::cout << "[chain] S size=" << mm1.s.size() << " maxAbs=" << sMax
                  << " finite=" << (sFinite ? "OK" : "FAIL")
                  << " s[0]=" << (float)mm1.s[0] << " s[1]=" << (float)mm1.s[1] << std::endl;
    }

    // ---- 3. Host softmax -> P ----
    std::vector<float> p;
    HostSoftmax(mm1.s, sq, sk, p);
    {
        float rowSum = 0.0f;
        for (uint32_t c = 0; c < sk; ++c) rowSum += p[c];  // 第一行
        std::cout << "[chain] P row0 sum=" << rowSum << " p[0]=" << p[0] << " p[1]=" << p[1] << std::endl;
    }

    // ---- 4. P quantize FP8 (MX per-32) ----
    std::vector<int8_t> pFp8, pMxScale;
    std::vector<float> pFlat(sq * sk);
    for (size_t r = 0; r < sq; ++r) {
        for (size_t c = 0; c < sk; ++c) {
            pFlat[r * sk + c] = p[r * sk + c];
        }
    }
    // 4a. 转置 P [Sq, Sk] -> 按 RowMajor 存（MM2 A 要求 [m=Sq, k=Sk]）
    HostFp8Quantize(pFlat, sq * sk, pFp8, pMxScale);
    // 4b. V: float [Sk, D] 行主序 -> B 存储序 [D, Sk]（列主序展平，连续32=同一列32行=32个K元素）
    std::vector<int8_t> vFp8CM;
    std::vector<int8_t> vMxScale;
    std::vector<float> vFlat(lenV);
    for (size_t c = 0; c < d; ++c) {
        for (size_t r = 0; r < sk; ++r) {
            vFlat[c * sk + r] = vFloat[r * d + c];
        }
    }
    HostFp8Quantize(vFlat, lenV, vFp8CM, vMxScale);

    // ---- 5. MM2: FP8 P @ V -> O ----
    std::cout << "[chain] MM2 FP8 PV (" << sq << ", " << d << ", " << sk << ")..." << std::endl;
    std::vector<float> o;
    RunMm2(pFp8, pMxScale, vFp8CM, vMxScale, sq, d, sk, stream, o);

    // ---- 自检: 无 NaN / 有限值 ----
    bool finiteAll = true;
    float maxAbs = 0.0f;
    for (size_t i = 0; i < o.size(); ++i) {
        if (!std::isfinite(o[i])) { finiteAll = false; break; }
        maxAbs = std::max(maxAbs, std::fabs(o[i]));
    }
    std::cout << "[chain] O size=" << o.size() << " maxAbs=" << maxAbs << (finiteAll ? " finite=OK" : " finite=FAIL")
              << std::endl;
    if (!finiteAll) {
        std::cerr << "[chain] OUTPUT CONTAINS NON-FINITE VALUES" << std::endl;
    } else {
        std::cout << "[chain] Chain pipeline OK." << std::endl;
    }

    // ---- 6. FP32 golden 精度对比（L-B 标准：max<=2 / mean<=1.2 / RMSE<=1.2）----
    // golden O = P_float @ V_float（host softmax 输出 P 乘原始 V，FP32 全精度）
    {
        double maxAbsErr = 0.0, sumAbsErr = 0.0, sumSqErr = 0.0;
        for (uint32_t r = 0; r < sq; ++r) {
            for (uint32_t c = 0; c < d; ++c) {
                double golden = 0.0;
                for (uint32_t t = 0; t < sk; ++t) {
                    golden += (double)p[r * sk + t] * (double)vFloat[t * d + c];
                }
                double err = std::fabs((double)o[r * d + c] - golden);
                maxAbsErr = std::max(maxAbsErr, err);
                sumAbsErr += err;
                sumSqErr += err * err;
            }
        }
        size_t n = (size_t)sq * d;
        double meanAbsErr = sumAbsErr / n;
        double rmse = std::sqrt(sumSqErr / n);
        bool lbPass = (maxAbsErr <= 2.0 && meanAbsErr <= 1.2 && rmse <= 1.2);
        std::cout << "[L-B] vs FP32 golden: maxAbsErr=" << maxAbsErr << " meanAbsErr=" << meanAbsErr
                  << " rmse=" << rmse << (lbPass ? " -> PASS (L-B)" : " -> FAIL (L-B)") << std::endl;
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    ChainOptions options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}