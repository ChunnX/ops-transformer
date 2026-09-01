/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SageAttention2 MM1: standalone INT8 QK^T matmul kernel (Ascend950).
 *
 * Based on examples/51_ascend950_quant_matmul_per_group_per_block_tla pattern:
 *   - A = q_int8 [Sq, D] (RowMajor int8)
 *   - B = k_int8^T [D, Sk] (RowMajor int8)
 *   - C = int32 accumulator (hardware INT8 matmul, bypasses Fixpipe float limitation)
 *   - x1scale = q_scale (per-row-group), x2scale = k_scale (per-col-block), sm_scale folded into scales
 *   - Y = S [Sq, Sk] = (Q @ K^T) * q_scale * k_scale * sm_scale
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/quant_matmul_per_group_per_block_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
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
struct Mm1Options {
    uint32_t sq{0};      // query seqlen (M)
    uint32_t sk{0};      // key seqlen (N)
    uint32_t d{0};       // head dim (K)
    uint32_t deviceId{0};
    std::string dataPath = "../../examples/51_ascend950_quant_matmul_per_group_per_block_tla/data";

    int Parse(int argc, const char** argv)
    {
        if (argc < 4) {
            printf("Usage: %s sq sk d [--device DEVICE_ID] [--datapath PATH]\n", argv[0]);
            return -1;
        }
        sq = atoi(argv[1]);
        sk = atoi(argv[2]);
        d = atoi(argv[3]);
        for (int i = 4; i < argc; ++i) {
            std::string flag(argv[i]);
            if (flag == "--device") { deviceId = atoi(argv[++i]); }
            else if (flag == "--datapath") { dataPath = std::string(argv[++i]); }
        }
        return 0;
    }
};

static void Run(const Mm1Options& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.sq;
    uint32_t n = options.sk;
    uint32_t k = options.d;

    uint32_t gs = 128; // groupsize of matrix A (Q) along K
    uint32_t bs = 128; // blocksize of matrix B (K^T) along N

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using Elementx1scale = float;
    using Elementx2scale = float;
    using ElementY = half;
    using ElementBias = void;

    using ElementBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementY>(m, n);
    LayoutTagA tagScalex1 = LayoutTagC::MakeLayout<Elementx1scale>(m, CeilDiv(k, gs));
    LayoutTagB tagScalex2 = LayoutTagC::MakeLayout<Elementx2scale>(CeilDiv(k, bs), CeilDiv(n, bs));

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();
    size_t lenX1scale = static_cast<size_t>(m) * (CeilDiv(k, gs));
    size_t lenX2scale = static_cast<size_t>(CeilDiv(k, bs)) * (CeilDiv(n, bs));
    size_t lenC = tagC.Capacity();
    size_t lenBias = static_cast<size_t>(n);

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeC = lenC * sizeof(fp16_t);
    size_t sizeX1Scale = lenX1scale * sizeof(float);
    size_t sizeX2Scale = lenX2scale * sizeof(float);
    size_t sizeBias = lenBias * sizeof(ElementBiasType);
    size_t sizeWorkspace;

    std::vector<ElementA> hostA(lenA);
    std::vector<ElementB> hostB(lenB);
    std::vector<float> hostX1Scale(lenX1scale);
    std::vector<float> hostX2Scale(lenX2scale);
    std::vector<ElementBiasType> hostBias(lenBias);
    golden::FillRandomData<ElementA>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<ElementB>(hostB, -5.0f, 5.0f);
    golden::FillRandomData<float>(hostX1Scale, 0.0, 1.0);
    golden::FillRandomData<float>(hostX2Scale, 0.0, 1.0);

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceX1Scale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX1Scale), sizeX1Scale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX1Scale, sizeX1Scale, hostX1Scale.data(), sizeX1Scale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceX2Scale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX2Scale), sizeX2Scale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX2Scale, sizeX2Scale, hostX2Scale.data(), sizeX2Scale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceBias{nullptr};
    if constexpr (!std::is_void_v<ElementBias>) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint8_t* deviceWorkspace{nullptr};

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = false;
    constexpr bool useHF32 = false;
    constexpr bool enableL1Resident = false;
    constexpr uint32_t l0CStages = 2;
    constexpr uint32_t l1AStages = 2;
    constexpr uint32_t l1BStages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    using DispatchPolicy = Gemm::MmadPingpongPertile<ArchTag, enableUnitFlag>;
    using L1TileShape = Shape<Int<256>, Int<128>, Int<128>>;
    using L0TileShape = Shape<Int<256>, Int<128>, Int<128>>;
    using ProblemShape = Catlass::GemmCoord;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using EpilogueDispatchPolicy = Epilogue::BlockEpiloguePertile;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, L0TileShape, ElementY, ElementC, ElementBias, Elementx1scale, Elementx2scale,
        LayoutTagA, LayoutTagB>;

    uint32_t taskNum = CeilDiv(options.sq, tla::get<0>(L1TileShape{})) *
                       CeilDiv(options.sk, tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);

    // Swizzle offset is 3 and direction is 1.
    using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape>;

    // kernel level
    using MatmulKernel =
        Gemm::Kernel::QuantMatmulPerGroupPerBlockTla<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulKernel::Arguments arguments{
        Catlass::GemmCoord(m, n, k),
        deviceA,
        layoutA,
        deviceB,
        layoutB,
        deviceC,
        layoutC,
        deviceBias,
        {deviceC, deviceX2Scale, deviceX1Scale, deviceBias, tla::get<0>(L0TileShape{}), tla::get<1>(L0TileShape{}),
         tla::get<2>(L0TileShape{})}};

    MatmulAdapter matmulOp;
    matmulOp.CanImplement(arguments);
    sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreUsed);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::QuantMatmulPergroupPerBlockDequant(
        Catlass::GemmCoord(m, n, k), hostA, tagA, hostB, tagB, hostX1Scale, tagScalex1, hostX2Scale, tagScalex2,
        hostGolden, tagC, gs, bs);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "MM1 INT8 QK^T Compare success." << std::endl;
    } else {
        std::cerr << "MM1 INT8 QK^T Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceX1Scale));
    ACL_CHECK(aclrtFree(deviceX2Scale));
    ACL_CHECK(aclrtFree(deviceC));
    if constexpr (!std::is_void_v<ElementBias>) {
        ACL_CHECK(aclrtFree(deviceBias));
    }
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    Mm1Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}