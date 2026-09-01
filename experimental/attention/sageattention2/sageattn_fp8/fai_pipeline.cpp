/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SageAttention2 完整 FA kernel：INT8 QK^T + online softmax + FP8 PV + rescale，全在 ONE kernel。
 * Host 驱动：准备 int8 q/k + q_scale/k_scale + fp8 v + mxScaleV，launch FAInferInt8QK，回读 O。
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <unistd.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "fai_int8_qk_kernel.h"
#include "fai_int8_qk_tiling.h"

using namespace Catlass;

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------
struct Options {
    uint32_t sq{0};
    uint32_t sk{0};
    uint32_t d{0};
    uint32_t deviceId{0};

    int Parse(int argc, const char** argv) {
        if (argc < 4) {
            printf("Usage: %s sq sk d [--device ID]\n", argv[0]);
            return -1;
        }
        sq = atoi(argv[1]);
        sk = atoi(argv[2]);
        d = atoi(argv[3]);
        for (int i = 4; i < argc; ++i) {
            std::string f(argv[i]);
            if (f == "--device") deviceId = atoi(argv[++i]);
        }
        return 0;
    }
};

// Host FP8 MX quantize (per-32 group, e8m0 scale)
static void HostFp8Quantize(const std::vector<float>& src, std::vector<int8_t>& fp8Data,
                             std::vector<int8_t>& mxScale) {
    constexpr uint32_t GROUP = 32;
    size_t numGroups = (src.size() + GROUP - 1) / GROUP;
    fp8Data.assign(src.size(), 0);
    mxScale.assign(numGroups, 127);  // neutral scale = 1.0 (e8m0 exp=127)
    for (size_t g = 0; g < numGroups; ++g) {
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < GROUP && g * GROUP + i < src.size(); ++i) {
            maxAbs = std::max(maxAbs, std::fabs(src[g * GROUP + i]));
        }
        if (maxAbs < 1e-30f) continue;
        int exp = static_cast<int>(std::floor(std::log2(maxAbs))) - 8;
        exp = std::max(-127, std::min(127, exp));
        mxScale[g] = static_cast<int8_t>(exp & 0xFF);
        float scale = std::pow(2.0f, exp);
        float invScale = 1.0f / scale;
        for (uint32_t i = 0; i < GROUP && g * GROUP + i < src.size(); ++i) {
            float v = src[g * GROUP + i] * invScale;
            v = std::max(-448.0f, std::min(448.0f, v));
            fp8Data[g * GROUP + i] = static_cast<int8_t>(static_cast<int>(v));
        }
    }
}

// Host softmax for golden comparison (dynamic sm_scale)
static void HostSoftmax(const std::vector<int32_t>& s, uint32_t sq, uint32_t sk,
                        std::vector<float>& p) {
    float sMaxGlobal = 0.0f;
    for (size_t i = 0; i < s.size(); ++i) {
        float v = std::fabs(static_cast<float>(s[i]));
        if (v > sMaxGlobal) sMaxGlobal = v;
    }
    float smScale = (sMaxGlobal > 1e-6f) ? (10.0f / sMaxGlobal) : (1.0f / std::sqrt(static_cast<float>(sk)));
    p.assign(static_cast<size_t>(sq) * sk, 0.0f);
    for (uint32_t r = 0; r < sq; ++r) {
        float maxV = -1e30f;
        for (uint32_t c = 0; c < sk; ++c) {
            maxV = std::max(maxV, static_cast<float>(s[r * sk + c]) * smScale);
        }
        float sum = 0.0f;
        for (uint32_t c = 0; c < sk; ++c) {
            float e = std::exp(static_cast<float>(s[r * sk + c]) * smScale - maxV);
            p[r * sk + c] = e;
            sum += e;
        }
        for (uint32_t c = 0; c < sk; ++c) p[r * sk + c] /= sum;
    }
}

static void Run(const Options& options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t batch = 1;
    uint32_t numHeads = 1;    // Hq
    uint32_t kvHeads = 1;    // Hk
    uint32_t groupSize = numHeads / kvHeads;  // GQA ratio
    uint32_t qSeqlen = options.sq;
    uint32_t kvSeqlen = options.sk;
    uint32_t embed = options.d;
    uint32_t blockSize = 128;
    uint32_t gs = 128;  // Q scale groupsize
    uint32_t bs = 128;  // K scale blocksize

    if (embed > 128) {
        std::cerr << "[ERROR] embed must be <= 128" << std::endl;
        return;
    }

    uint32_t maxKvSeqlen = kvSeqlen;
    uint32_t numBlocks = batch * ((maxKvSeqlen + blockSize - 1) / blockSize);

    // Sizes
    uint64_t qSize = static_cast<uint64_t>(batch) * qSeqlen * numHeads * embed * sizeof(int8_t);
    uint64_t kSize = static_cast<uint64_t>(numBlocks) * blockSize * kvHeads * embed * sizeof(int8_t);
    uint64_t vSize = static_cast<uint64_t>(numBlocks) * blockSize * kvHeads * embed * sizeof(int8_t);
    uint64_t qScaleSize = static_cast<uint64_t>(batch) * qSeqlen * numHeads * CeilDiv(embed, gs) * sizeof(float);
    uint64_t kScaleSize = static_cast<uint64_t>(CeilDiv(kvHeads * embed, bs)) * CeilDiv(batch * kvSeqlen, bs) * sizeof(float);
    uint64_t vScaleSize = static_cast<uint64_t>(numBlocks) * blockSize * kvHeads * embed / 32 * sizeof(int8_t);
    uint64_t oSize = static_cast<uint64_t>(batch) * qSeqlen * numHeads * embed * sizeof(fp16_t);
    uint64_t maskSize = 1;  // dummy
    uint64_t blockTableSize = static_cast<uint64_t>(numBlocks) * sizeof(int32_t);
    uint32_t tilingSize = sizeof(FATilingData);

    // Host data: random int8 q/k, float v, random scales
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> idist(-16, 16);
    std::uniform_real_distribution<float> fdist(-4.0f, 4.0f);
    std::uniform_real_distribution<float> sdist(0.05f, 1.0f);

    std::vector<int8_t> hostQ(qSize);
    std::vector<int8_t> hostK(kSize);
    for (auto& v : hostQ) v = static_cast<int8_t>(idist(rng));
    for (auto& v : hostK) v = static_cast<int8_t>(idist(rng));

    // V: float → FP8 MX
    std::vector<float> vFloat(vSize / sizeof(int8_t));
    for (auto& v : vFloat) v = fdist(rng);
    std::vector<int8_t> hostV, hostVScale;
    HostFp8Quantize(vFloat, hostV, hostVScale);

    // q_scale / k_scale (float, random)
    std::vector<float> hostQScale(qScaleSize / sizeof(float));
    std::vector<float> hostKScale(kScaleSize / sizeof(float));
    for (auto& v : hostQScale) v = sdist(rng);
    for (auto& v : hostKScale) v = sdist(rng);

    // block table (identity)
    std::vector<int32_t> hostBlockTable(numBlocks);
    for (uint32_t i = 0; i < numBlocks; ++i) hostBlockTable[i] = i;

    // Device alloc
    uint8_t *qDevice, *kDevice, *vDevice, *qScaleDevice, *kScaleDevice, *vScaleDevice;
    uint8_t *oDevice, *blockTableDevice, *maskDevice, *tilingDevice;
    ACL_CHECK(aclrtMalloc((void**)&qDevice, qSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&kDevice, kSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&vDevice, vSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&qScaleDevice, qScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&kScaleDevice, kScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&vScaleDevice, vScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&oDevice, oSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&blockTableDevice, blockTableSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&maskDevice, maskSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // H2D
    ACL_CHECK(aclrtMemcpy(qDevice, qSize, hostQ.data(), qSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(kDevice, kSize, hostK.data(), kSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(vDevice, vSize, hostV.data(), vSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(qScaleDevice, qScaleSize, hostQScale.data(), qScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(kScaleDevice, kScaleSize, hostKScale.data(), kScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(vScaleDevice, vScaleSize, hostVScale.data(), vScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(blockTableDevice, blockTableSize, hostBlockTable.data(), blockTableSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(oDevice, oSize, 0, oSize));

    // Tiling
    uint32_t blockDim = aicCoreNum;  // [第五十三轮-方案A] 与官方 fai.cpp L269 一致：纯 AIC
    // 获取 AIV 核数
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    std::cout << "[fai] aicCoreNum=" << aicCoreNum << " aivCoreNum=" << aivCoreNum << std::endl;
    FAInferTiling::FAInfo faInfo;
    faInfo.batchSize = batch;
    faInfo.numOfHeads = numHeads;
    faInfo.numOfKVHeads = kvHeads;
    faInfo.seqSize = qSeqlen;
    faInfo.seqInnerSize = kvSeqlen;
    faInfo.headSize = embed;
    // Dynamic sm_scale: scan int8 Q/K to estimate S maxAbs
    float sMaxEst = static_cast<float>(numHeads * embed) * 16.0f * 16.0f * 0.25f;  // rough estimate
    faInfo.scaleValue = (sMaxEst > 1e-6f) ? (10.0f / sMaxEst) : (1.0f / std::sqrt(static_cast<float>(embed)));
    faInfo.blockSize = blockSize;
    faInfo.maxBlockNumPerBatch = (kvSeqlen + blockSize - 1) / blockSize;
    faInfo.maskType = 0;  // no mask

    FATilingData faTilingData;
    if (FAInferTiling::GetFATilingParam(faInfo, blockDim, faTilingData) != 0) {
        std::cerr << "[ERROR] GetFATilingParam failed" << std::endl;
        return;
    }
    // [方案 A] 测试用例太小 tiling 只分 1 核，强制 coreNum 让 AIC 核参与 QK/bmm2
    // [第六十一轮-路径D] 关键证据：官方 AIC核=1 被调度（coreNum=1 实测），本机 AIC核=0（coreNum=84 强制）
    // 实验：coreNum 改回与官方一致（=1），验证 AIC 核是否被调度、QK 是否执行
    faTilingData.multiCoreParamsRegbase.coreNum = 1;
    // [方案 B] 打印 tiling 的 bnAxisStartIdx/sparseStartIdx 数组实际值（前 8 项 + coreNum）
    {
        auto &mc = faTilingData.multiCoreParamsRegbase;
        std::cout << "[R51] coreNum=" << mc.coreNum
                  << " bnAxisStart[0..7]=" << mc.bnAxisStartIdx[0] << "," << mc.bnAxisStartIdx[1]
                  << "," << mc.bnAxisStartIdx[2] << "," << mc.bnAxisStartIdx[3]
                  << "," << mc.bnAxisStartIdx[4] << "," << mc.bnAxisStartIdx[5]
                  << "," << mc.bnAxisStartIdx[6] << "," << mc.bnAxisStartIdx[7]
                  << " sparseStart[0..7]=" << mc.sparseStartIdx[0] << "," << mc.sparseStartIdx[1]
                  << "," << mc.sparseStartIdx[2] << "," << mc.sparseStartIdx[3]
                  << "," << mc.sparseStartIdx[4] << "," << mc.sparseStartIdx[5]
                  << "," << mc.sparseStartIdx[6] << "," << mc.sparseStartIdx[7] << std::endl;
    }
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, &faTilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // [第四十一轮] 分配调试 buffer（pScale 参数，用于 kernel 哨兵写入）
    uint8_t* mm2DbgDevice{nullptr};
    ACL_CHECK(aclrtMalloc((void**)&mm2DbgDevice, 4096, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(mm2DbgDevice, 4096, 0x7f, 4096));

    std::cout << "[fai] Launch FAInferInt8QK (blockDim=" << blockDim
              << " batch=" << batch << " sq=" << qSeqlen << " sk=" << kvSeqlen
              << " d=" << embed << " scaleValue=" << faInfo.scaleValue << ")" << std::endl;

    // Launch: FAInferInt8QK<InDtype=float8_e4m3_t, OutDtype=half, mask=false, pa=false, pscale=false>
    FAInferInt8QK<float8_e4m3_t, half, false, false, false><<<blockDim, nullptr, stream>>>(
        qDevice, kDevice, vDevice,
        qScaleDevice, kScaleDevice,
        maskDevice, blockTableDevice, oDevice,
        nullptr, nullptr,  // actualQSeqlen, actualKvSeqlen (null = full seq)
        vScaleDevice, mm2DbgDevice, tilingDevice  // mxScaleV, pScale(=mm2DbgDevice), tiling
    );

    // [第四十二轮] 改用带超时的非阻塞回读：kernel 若 hang 也不阻塞，先回读哨兵确认执行到哪一步
    // aclrtSynchronizeStreamWithTimeout 带超时等待（毫秒），超时返回则继续回读哨兵
    {
        aclError ts = aclrtSynchronizeStreamWithTimeout(stream, 10000);  // 10 秒超时
        if (ts != ACL_SUCCESS) {
            std::cout << "[R41] WARN: kernel 10s 内未完成(可能hang, err=" << ts << ")，继续回读哨兵" << std::endl;
        }
    }

    // [第四十一轮] 回读哨兵 + 关键指标（GM[80..95]）
    {
        int32_t d[16];
        ACL_CHECK(aclrtMemcpy(d, 64, mm2DbgDevice + 320, 64, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[80..95]
        std::cout << "[R42] AIC_Init(80)=" << std::hex << d[0] << std::dec << std::endl;
        std::cout << "[R42] AIV_Init(81)=" << std::hex << d[1] << std::dec << std::endl;
        std::cout << "[R41] 主循环入口(82)=" << std::hex << d[2] << std::endl;
        std::cout << "[R41] softmax分支(90)=" << std::hex << d[10] << std::endl;
        std::cout << "[R41] bmm2输出(94)=" << std::hex << d[14] << std::endl;
        // [第四十二轮] 启动检查：Init 入口哨兵 GM[70]（确认 kernel 是否被调度）
        {
            int32_t s0;
            ACL_CHECK(aclrtMemcpy(&s0, 4, mm2DbgDevice + 280, 4, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[70] offset=280
            std::cout << "[R41] Init入口(70)=" << std::hex << s0 << std::endl;
        }
        // [第四十二轮] 等待点哨兵回读（GM[96-111]）
        {
            int32_t w[8];
            ACL_CHECK(aclrtMemcpy(w, 32, mm2DbgDevice + 384, 32, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[96..103]
            std::cout << "[R41] 等待点1(96)=" << std::hex << w[0] << " sentinel(97)=" << w[1] << std::endl;
            std::cout << "[R41] 等待点2(98)=" << std::hex << w[2] << " sentinel(99)=" << w[3] << std::endl;
            std::cout << "[R41] 等待点3(100)=" << std::hex << w[4] << " sentinel(101)=" << w[5] << std::endl;
        // [第四十二轮] bmm2 分支入口哨兵回读（GM[110-111]）：taskId/notLast 实际值
        {
            int32_t t[2];
            ACL_CHECK(aclrtMemcpy(t, 8, mm2DbgDevice + 440, 8, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[110..111] offset=440
            std::cout << "[R41] bmm2入口: taskId=" << std::dec << t[0] << " notLast=" << t[1] << std::endl;
        // [第四十二轮] BlockMmad 构造前哨兵回读（GM[106-109]）
        {
            int32_t b[4];
            ACL_CHECK(aclrtMemcpy(b, 16, mm2DbgDevice + 424, 16, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[106..109] offset=424
            std::cout << "[R41] QK构造前(106)=" << std::hex << b[0] << " sentinel(107)=" << b[1]
                      << " PV构造前(108)=" << b[2] << " sentinel(109)=" << b[3] << std::endl;
        // [第四十二轮] bmm2 内部细分哨兵回读（GM[112-115]）
        {
            int32_t m[4];
            ACL_CHECK(aclrtMemcpy(m, 16, mm2DbgDevice + 448, 16, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[112..115] offset=448
            std::cout << "[R41] bmm2内部1(112)=" << std::hex << m[0] << " sentinel(113)=" << m[1]
                      << " bmm2内部2(114)=" << m[2] << " sentinel(115)=" << m[3] << std::endl;
        }
        // [第四十七轮] 决定性哨兵回读：谁通过了 blockIdx 过滤 + QK 分支入口（GM[115-119]）
        {
            int32_t d[6];
            ACL_CHECK(aclrtMemcpy(d, 24, mm2DbgDevice + 460, 24, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[115..120] offset=460
            std::cout << "[R47] AIC通过过滤(115)=" << std::hex << d[0] << " AIC_blockIdx(116)=" << d[1] << std::endl;
            std::cout << "[R47] AIV通过过滤(117)=" << std::hex << d[2] << " AIV_blockIdx(118)=" << d[3] << std::endl;
            std::cout << "[R47] QK分支入口(119)=" << std::hex << d[4] << std::endl;
        }
        // [第五十一轮] 过滤前哨兵回读：constInfo.coreNum / GetBlockIdx / blockIdx（GM[120-122]）
        {
            int32_t e[3];
            ACL_CHECK(aclrtMemcpy(e, 12, mm2DbgDevice + 480, 12, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[120..122] offset=480
            std::cout << "[R51] coreNum(120)=" << std::dec << e[0] << " GetBlockIdx(121)=" << e[1]
                      << " blockIdx(122)=" << e[2] << std::endl;
        }
        // [第五十一轮] AIC/AIV 分离过滤前哨兵回读（GM[120-124]）
        {
            int32_t g[5];
            ACL_CHECK(aclrtMemcpy(g, 20, mm2DbgDevice + 480, 20, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[120..124] offset=480
            std::cout << "[R51] coreNum=" << std::dec << g[0]
                      << " AIC_GetBlockIdx(121)=" << g[1] << " AIC_blockIdx(122)=" << g[2]
                      << " AIV_GetBlockIdx(123)=" << g[3] << " AIV_blockIdx(124)=" << g[4] << std::endl;
        }
        // [第五十一轮] bnAxis 循环边界回读（GM[125-128]）
        {
            int32_t b[4];
            ACL_CHECK(aclrtMemcpy(b, 16, mm2DbgDevice + 500, 16, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[125..128] offset=500
            std::cout << "[R51] AIC_bnStart(125)=" << std::dec << b[0] << " AIC_bnEnd(126)=" << b[1]
                      << " AIV_bnStart(127)=" << b[2] << " AIV_bnEnd(128)=" << b[3] << std::endl;
        }
        // [第五十一轮] blockIdx==0 专用哨兵回读（GM[130-131]）
        {
            int32_t z[2];
            ACL_CHECK(aclrtMemcpy(z, 8, mm2DbgDevice + 520, 8, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[130..131] offset=520
            std::cout << "[R51] blockIdx0到达(130)=" << std::hex << z[0] << " blockIdx0通过(131)=" << z[1] << std::endl;
        }
        // [第五十一轮] operator() 入口 AIC/AIV 独立哨兵回读（GM[132-133]）
        {
            int32_t a[2];
            ACL_CHECK(aclrtMemcpy(a, 8, mm2DbgDevice + 528, 8, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[132..133] offset=528
            std::cout << "[R51] AIC进op(132)=" << std::hex << a[0] << " AIV进op(133)=" << a[1] << std::endl;
        }
        // [第五十一轮] g_coreType 原始值回读（GM[134]）
        {
            int32_t ct;
            ACL_CHECK(aclrtMemcpy(&ct, 4, mm2DbgDevice + 536, 4, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[134] offset=536
            std::cout << "[R51] g_coreType(134)=" << std::dec << ct << " (AIC=1, AIV=2)" << std::endl;
        }
        // [第五十七轮] constInfo 关键值回读（GM[128..139]，mm2DbgDevice offset 512）
        {
            int32_t ci[12];
            ACL_CHECK(aclrtMemcpy(ci, 48, mm2DbgDevice + 512, 48, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[128..139]
            std::cout << "[R57] batch=" << ci[0] << " qHeads=" << ci[1] << " kvHeads=" << ci[2]
                      << " qSeqlen=" << ci[3] << " kvSeqlen=" << ci[4] << " embed=" << ci[5] << std::endl;
            std::cout << "[R57] coreNum=" << ci[6] << " bnAxisStartIdx=" << ci[7]
                      << " bnAxisEndIdx=" << ci[8] << " mcInnerOffset=" << ci[9]
                      << " mcInnerLimit=" << ci[10] << " blockIdx=" << ci[11] << std::endl;
        }
        // [第五十八轮] 每核 slot 调度分布回读（mm2DbgDevice offset 1024：GM[256..]=类型 / GM[384..]=blockIdx / GM[512..]=到达标记）
        {
            int32_t d[384];
            ACL_CHECK(aclrtMemcpy(d, 1536, mm2DbgDevice + 1024, 1536, ACL_MEMCPY_DEVICE_TO_HOST));
            int aicCount = 0, aivCount = 0, arrivedCount = 0;
            std::cout << "[R58] 核调度分布（slot=GetBlockIdx()%128）:" << std::endl;
            for (int i = 0; i < 128; ++i) {
                if (d[i] == 1) { aicCount++; std::cout << "  slot" << i << ": AIC(type=1) blockIdx=" << d[128 + i] << std::endl; }
                else if (d[i] == 2) { aivCount++; std::cout << "  slot" << i << ": AIV(type=2) blockIdx=" << d[128 + i] << std::endl; }
                if (d[256 + i] == (int32_t)0x77777777) arrivedCount++;
            }
            std::cout << "[R58] 汇总: AIC核=" << aicCount << " AIV核=" << aivCount
                      << " 到达operator()=" << arrivedCount << " (blockDim=" << aicCoreNum << ")" << std::endl;
        }
        // [第六十一轮-路径D] g_coreType 运行时原始值分布（GM[640..767]，offset 2560）
        {
            int32_t ct[128];
            ACL_CHECK(aclrtMemcpy(ct, 512, mm2DbgDevice + 2560, 512, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[640..767]
            int aicGt = 0, aivGt = 0;
            for (int i = 0; i < 128; ++i) {
                if (ct[i] == 1) aicGt++;
                else if (ct[i] == 2) aivGt++;
            }
            std::cout << "[R61] g_coreType运行时分布: AIC核=" << aicGt << " AIV核=" << aivGt
                      << " (slot=GetBlockIdx()%128)" << std::endl;
        }
        // [第六十一轮-路径D] AIC 断链点哨兵回读（GM[140-143]）
        {
            int32_t s[4];
            ACL_CHECK(aclrtMemcpy(s, 16, mm2DbgDevice + 560, 16, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[140..143] offset=560
            std::cout << "[R61] AIC断链: qSeqAxis入口(140)=" << std::hex << s[0]
                      << " notLastThreeLoop(141)=" << s[1] << " kvSeqLoop(142)=" << s[2]
                      << " 进AIC块前(143)=" << s[3] << std::endl;
        }
        }
        }
        }
    }
    // [R43] bmm2 输出哨兵回读
    {
        int32_t d[4];
        ACL_CHECK(aclrtMemcpy(d, 16, mm2DbgDevice + 376, 16, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[94..97]
        std::cout << "[R43] bmm2输出哨兵(94)=" << std::hex << d[0] << std::dec << std::endl;
    }
    // [Step2] 关键指标回读（P90, bmm2_94, rs98）
    {
        int32_t d[10];
        ACL_CHECK(aclrtMemcpy(d, 40, mm2DbgDevice + 360, 40, ACL_MEMCPY_DEVICE_TO_HOST));  // GM[90..99]
        std::cout << "[Step2] P90=" << std::hex << d[0] << " bmm2_94=" << d[4] << " rs98=" << d[8] << std::dec
                  << " (P=" << (d[0]==(int32_t)0x10101010?"✅":"❌")
                  << " bmm2=" << (d[4]!=0?"✓":"0")
                  << " rs=" << (d[8]==(int32_t)0x55555555?"✅":"❌") << ")" << std::endl;
    }

    // Copy O back
    std::vector<fp16_t> hostO(oSize / sizeof(fp16_t));
    ACL_CHECK(aclrtMemcpy(hostO.data(), oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Self-check
    bool finiteAll = true;
    float maxAbs = 0.0f;
    for (size_t i = 0; i < hostO.size(); ++i) {
        float v = (float)hostO[i];
        if (!std::isfinite(v)) { finiteAll = false; break; }
        maxAbs = std::max(maxAbs, std::fabs(v));
    }
    std::cout << "[fai] O size=" << hostO.size() << " maxAbs=" << maxAbs
              << (finiteAll ? " finite=OK" : " finite=FAIL") << std::endl;
    if (maxAbs > 1e-6f && finiteAll) {
        std::cout << "[fai] ✓ Kernel output non-zero, no hang (PASS)" << std::endl;
    } else {
        std::cout << "[fai] ✗ Output zero or NaN (FAIL)" << std::endl;
    }

    // Cleanup
    ACL_CHECK(aclrtFree(qDevice));
    ACL_CHECK(aclrtFree(kDevice));
    ACL_CHECK(aclrtFree(vDevice));
    ACL_CHECK(aclrtFree(qScaleDevice));
    ACL_CHECK(aclrtFree(kScaleDevice));
    ACL_CHECK(aclrtFree(vScaleDevice));
    ACL_CHECK(aclrtFree(oDevice));
    ACL_CHECK(aclrtFree(blockTableDevice));
    ACL_CHECK(aclrtFree(maskDevice));
    ACL_CHECK(aclrtFree(tilingDevice));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) return -1;
    Run(options);
    return 0;
}
