#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_sage_attention2.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v5.h"

struct CaseConfig {
    const char *caseId;
    int64_t b, hq, hk, s, d;
    bool isBf16;
    bool isCausal;
};

static const std::vector<CaseConfig> BENCHMARK_CASES = {
    {"P-01", 1, 32, 32, 4096,  128, false, false},
    {"P-02", 1, 32, 32, 8192,  128, false, false},
    {"P-03", 1, 32, 32, 8192,  128, false, true },
    {"P-04", 2, 32, 32, 8192,  128, false, false},
    {"P-05", 1, 32, 8,  4096,  128, false, false},
    {"P-06", 1, 32, 32, 4096,  128, true,  false},
    {"P-07", 1, 32, 32, 16384, 128, false, false},
};

#define GATE_SPEEDUP 1.6f

#define REQUIRE_EXEC(exec, impl)                                                       \
    do {                                                                               \
        if ((exec) == nullptr) {                                                       \
            std::cerr << "[ERROR] " << (impl) << " GetWorkspaceSize failed" << std::endl; \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

int main(int argc, char **argv) {
    int warmup = 50;
    int repeats = 100;
    const char *outJsonPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) outJsonPath = argv[++i];
    }

    int32_t deviceId = 0;
    aclrtStream stream;
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclInit failed" << std::endl;
        return 1;
    }
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice failed" << std::endl;
        return 1;
    }
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed" << std::endl;
        return 1;
    }

    std::cout << "==========================================================================" << std::endl;
    std::cout << "   SageAttention2 C++ aclnn Benchmark Suite (P-01 ~ P-07)" << std::endl;
    std::cout << "   Warmup: " << warmup << ", Repeats: " << repeats << std::endl;
    std::cout << "==========================================================================" << std::endl;
    std::cout << std::left << std::setw(8) << "Case"
              << std::setw(16) << "Shape(HND)"
              << std::setw(14) << "FIA-aclnn(ms)"
              << std::setw(15) << "Ours-aclnn(ms)"
              << std::setw(16) << "aclnn Speedup"
              << std::setw(10) << "Gate >=1.6x" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    std::string jsonContent = "{\n  \"aclnn_benchmarks\": {\n";

    for (size_t idx = 0; idx < BENCHMARK_CASES.size(); ++idx) {
        const auto &cfg = BENCHMARK_CASES[idx];
        int64_t b = cfg.b, hq = cfg.hq, hk = cfg.hk, s = cfg.s, d = cfg.d;
        aclDataType dtype = cfg.isBf16 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
        size_t elemSize = 2;

        std::vector<int64_t> qShape = {b, hq, s, d};
        std::vector<int64_t> kShape = {b, hk, s, d};
        std::vector<int64_t> vShape = {b, hk, s, d};
        std::vector<int64_t> outShape = {b, hq, s, d};

        size_t qBytes = b * hq * s * d * elemSize;
        size_t kBytes = b * hk * s * d * elemSize;
        size_t vBytes = b * hk * s * d * elemSize;
        size_t outBytes = b * hq * s * d * elemSize;

        void *qDev = nullptr, *kDev = nullptr, *vDev = nullptr;
        void *outSageDev = nullptr, *outFiaDev = nullptr;
        aclrtMalloc(&qDev, qBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&kDev, kBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&vDev, vBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outSageDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outFiaDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

        std::vector<int64_t> qStrides = {hq * s * d, s * d, d, 1};
        std::vector<int64_t> kStrides = {hk * s * d, s * d, d, 1};
        std::vector<int64_t> vStrides = {hk * s * d, s * d, d, 1};
        std::vector<int64_t> outStrides = {hq * s * d, s * d, d, 1};

        aclTensor *qTensor = aclCreateTensor(qShape.data(), 4, dtype, qStrides.data(), 0, aclFormat::ACL_FORMAT_ND, qShape.data(), 4, qDev);
        aclTensor *kTensor = aclCreateTensor(kShape.data(), 4, dtype, kStrides.data(), 0, aclFormat::ACL_FORMAT_ND, kShape.data(), 4, kDev);
        aclTensor *vTensor = aclCreateTensor(vShape.data(), 4, dtype, vStrides.data(), 0, aclFormat::ACL_FORMAT_ND, vShape.data(), 4, vDev);
        aclTensor *outSageTensor = aclCreateTensor(outShape.data(), 4, dtype, outStrides.data(), 0, aclFormat::ACL_FORMAT_ND, outShape.data(), 4, outSageDev);
        aclTensor *outFiaTensor = aclCreateTensor(outShape.data(), 4, dtype, outStrides.data(), 0, aclFormat::ACL_FORMAT_ND, outShape.data(), 4, outFiaDev);

        aclTensor *kListArr[1] = {kTensor};
        aclTensor *vListArr[1] = {vTensor};
        aclTensorList *kList = aclCreateTensorList(kListArr, 1);
        aclTensorList *vList = aclCreateTensorList(vListArr, 1);

        double smScale = 1.0 / std::sqrt(static_cast<double>(d));
        char layout[] = "BNSD";
        int64_t sparseMode = cfg.isCausal ? 2 : 0;
        int64_t preTokens = cfg.isCausal ? 0 : 2147483647;
        int64_t nextTokens = cfg.isCausal ? 0 : 2147483647;

        aclTensor *maskTensor = nullptr;
        void *maskDev = nullptr;
        if (cfg.isCausal) {
            size_t maskBytes = 2048 * 2048 * sizeof(uint8_t);
            aclrtMalloc(&maskDev, maskBytes, ACL_MEM_MALLOC_HUGE_FIRST);
            std::vector<uint8_t> hostMask(2048 * 2048, 0);
            for (int r = 0; r < 2048; ++r) {
                for (int c = 0; c < 2048; ++c) {
                    if (c > r) hostMask[r * 2048 + c] = 1;
                }
            }
            aclrtMemcpy(maskDev, maskBytes, hostMask.data(), maskBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            std::vector<int64_t> mShape = {2048, 2048};
            std::vector<int64_t> mStrides = {2048, 1};
            maskTensor = aclCreateTensor(mShape.data(), 2, aclDataType::ACL_BOOL, mStrides.data(), 0, aclFormat::ACL_FORMAT_ND, mShape.data(), 2, maskDev);
        }

        uint64_t fiaWsSize = 0;
        aclOpExecutor *fiaExec = nullptr;
        aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
            qTensor, kList, vList, nullptr, maskTensor, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr,
            hq, smScale, preTokens, nextTokens, layout, hk, sparseMode, 0, 0, 0, false, 0, 0, 0, 0,
            outFiaTensor, nullptr, &fiaWsSize, &fiaExec
        );
        void *fiaWs = nullptr;
        if (fiaWsSize > 0) aclrtMalloc(&fiaWs, fiaWsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        for (int i = 0; i < warmup; ++i) {
            aclOpExecutor *exec = nullptr;
            aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
                qTensor, kList, vList, nullptr, maskTensor, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr,
                hq, smScale, preTokens, nextTokens, layout, hk, sparseMode, 0, 0, 0, false, 0, 0, 0, 0,
                outFiaTensor, nullptr, &fiaWsSize, &exec
            );
            REQUIRE_EXEC(exec, "aclnnFusedInferAttentionScoreV5");
            aclnnFusedInferAttentionScoreV5(fiaWs, fiaWsSize, exec, stream);
        }
        aclrtSynchronizeStream(stream);

        aclrtEvent startFia, endFia;
        aclrtCreateEvent(&startFia);
        aclrtCreateEvent(&endFia);
        aclrtRecordEvent(startFia, stream);
        for (int i = 0; i < repeats; ++i) {
            aclOpExecutor *exec = nullptr;
            aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
                qTensor, kList, vList, nullptr, maskTensor, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr,
                hq, smScale, preTokens, nextTokens, layout, hk, sparseMode, 0, 0, 0, false, 0, 0, 0, 0,
                outFiaTensor, nullptr, &fiaWsSize, &exec
            );
            REQUIRE_EXEC(exec, "aclnnFusedInferAttentionScoreV5");
            aclnnFusedInferAttentionScoreV5(fiaWs, fiaWsSize, exec, stream);
        }
        aclrtRecordEvent(endFia, stream);
        aclrtSynchronizeStream(stream);
        float fiaTimeMs = 0;
        aclrtEventElapsedTime(&fiaTimeMs, startFia, endFia);
        fiaTimeMs /= repeats;
        aclrtDestroyEvent(startFia);
        aclrtDestroyEvent(endFia);

        uint64_t sageWsSize = 0;
        aclOpExecutor *sageExec = nullptr;
        aclnnSageAttention2GetWorkspaceSize(
            qTensor, kTensor, vTensor, "HND", cfg.isCausal, "per_thread", smScale, "fp32+fp16", true, false, false,
            outSageTensor, nullptr, &sageWsSize, &sageExec
        );
        REQUIRE_EXEC(sageExec, "aclnnSageAttention2");
        void *sageWs = nullptr;
        if (sageWsSize > 0) aclrtMalloc(&sageWs, sageWsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        for (int i = 0; i < warmup; ++i) {
            aclOpExecutor *exec = nullptr;
            aclnnSageAttention2GetWorkspaceSize(
                qTensor, kTensor, vTensor, "HND", cfg.isCausal, "per_thread", smScale, "fp32+fp16", true, false, false,
                outSageTensor, nullptr, &sageWsSize, &exec
            );
            REQUIRE_EXEC(exec, "aclnnSageAttention2");
            aclnnSageAttention2(sageWs, sageWsSize, exec, stream);
        }
        aclrtSynchronizeStream(stream);

        aclrtEvent startSage, endSage;
        aclrtCreateEvent(&startSage);
        aclrtCreateEvent(&endSage);
        aclrtRecordEvent(startSage, stream);
        for (int i = 0; i < repeats; ++i) {
            aclOpExecutor *exec = nullptr;
            aclnnSageAttention2GetWorkspaceSize(
                qTensor, kTensor, vTensor, "HND", cfg.isCausal, "per_thread", smScale, "fp32+fp16", true, false, false,
                outSageTensor, nullptr, &sageWsSize, &exec
            );
            REQUIRE_EXEC(exec, "aclnnSageAttention2");
            aclnnSageAttention2(sageWs, sageWsSize, exec, stream);
        }
        aclrtRecordEvent(endSage, stream);
        aclrtSynchronizeStream(stream);
        float sageTimeMs = 0;
        aclrtEventElapsedTime(&sageTimeMs, startSage, endSage);
        sageTimeMs /= repeats;
        aclrtDestroyEvent(startSage);
        aclrtDestroyEvent(endSage);

        float speedup = sageTimeMs > 0.0f ? fiaTimeMs / sageTimeMs : 0.0f;
        bool pass = speedup >= GATE_SPEEDUP;

        std::string shapeStr = "[" + std::to_string(b) + "," + std::to_string(hq) + "," + std::to_string(s) + "," + std::to_string(d) + "]";
        std::cout << std::left << std::setw(8) << cfg.caseId
                  << std::setw(16) << shapeStr
                  << std::setw(14) << std::fixed << std::setprecision(3) << fiaTimeMs
                  << std::setw(15) << std::fixed << std::setprecision(3) << sageTimeMs
                  << std::setw(16) << (std::to_string(speedup).substr(0, 4) + "x")
                  << std::setw(10) << (pass ? "PASS" : "FAIL") << std::endl;

        jsonContent += "    \"" + std::string(cfg.caseId) + "\": {\n";
        jsonContent += "      \"fia_time_ms\": " + std::to_string(fiaTimeMs) + ",\n";
        jsonContent += "      \"sage_time_ms\": " + std::to_string(sageTimeMs) + ",\n";
        jsonContent += "      \"speedup\": " + std::to_string(speedup) + ",\n";
        jsonContent += "      \"gate_passed\": " + std::string(pass ? "true" : "false") + "\n";
        jsonContent += "    }" + std::string(idx + 1 < BENCHMARK_CASES.size() ? "," : "") + "\n";

        if (fiaWs) aclrtFree(fiaWs);
        if (sageWs) aclrtFree(sageWs);
        if (maskDev) aclrtFree(maskDev);
        if (maskTensor) aclDestroyTensor(maskTensor);
        aclDestroyTensor(qTensor);
        aclDestroyTensor(kTensor);
        aclDestroyTensor(vTensor);
        aclDestroyTensor(outSageTensor);
        aclDestroyTensor(outFiaTensor);
        aclDestroyTensorList(kList);
        aclDestroyTensorList(vList);
        aclrtFree(qDev); aclrtFree(kDev); aclrtFree(vDev); aclrtFree(outSageDev); aclrtFree(outFiaDev);

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    jsonContent += "  }\n}\n";
    if (outJsonPath) {
        std::ofstream ofs(outJsonPath);
        if (ofs.is_open()) {
            ofs << jsonContent;
            ofs.close();
            std::cout << "\n[SUCCESS] aclnn benchmark results saved to: " << outJsonPath << std::endl;
        }
    }

    std::cout << "==========================================================================" << std::endl;
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
