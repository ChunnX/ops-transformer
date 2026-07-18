#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "acl/acl.h"
#include "aclnnop/aclnn_rotary_position_embedding3d.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg()); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg()); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg()); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg()); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg()); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// CPU 参考：RoPE half 模式
static void RefRoPE(int64_t batch, int64_t seqLen, int64_t headDim,
                    const float* x1, const float* x2, float* y)
{
    int64_t halfD = headDim / 2;
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seqLen; s++) {
            int64_t base = (b * seqLen + s) * headDim;
            for (int64_t k = 0; k < halfD; k++) {
                float xl = x1[base + k];
                float xr = x1[base + halfD + k];
                float cos = x2[base + k];
                float sin = x2[base + halfD + k];
                y[base + k]           = xl * cos - xr * sin;
                y[base + halfD + k]   = xl * sin + xr * cos;
            }
        }
    }
}

// ========== 动态加载 aclnn API ==========
int RunTest(int64_t batch, int64_t seqLen, int64_t headDim,
            int testIdx, aclrtStream stream, bool benchmark)
{
    int64_t total = batch * seqLen * headDim;
    std::vector<int64_t> shape = {batch, seqLen, headDim};
    int ret;

    std::vector<float> x1Data(total);
    for (int64_t i = 0; i < total; i++) {
        x1Data[i] = std::sin(static_cast<float>(i) * 0.73f) * 2.0f;
    }

    int64_t halfD = headDim / 2;
    std::vector<float> x2Data(total);
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seqLen; s++) {
            int64_t base = (b * seqLen + s) * headDim;
            float pos = static_cast<float>(s);
            for (int64_t k = 0; k < halfD; k++) {
                float theta = pos * std::pow(0.0001f, static_cast<float>(2 * k) / static_cast<float>(headDim));
                x2Data[base + k]           = std::cos(theta);
                x2Data[base + halfD + k]   = std::sin(theta);
            }
        }
    }

    std::vector<float> yRef(total, 0);
    RefRoPE(batch, seqLen, headDim, x1Data.data(), x2Data.data(), yRef.data());

    int printN = std::min(4, (int)total);
    LOG_PRINT("[Test %d] X1[0..%d]: ", testIdx, printN - 1);
    for (int i = 0; i < printN; i++) LOG_PRINT("%.4f ", x1Data[i]);
    LOG_PRINT("\n  X2[0..%d]: ", printN - 1);
    for (int i = 0; i < printN; i++) LOG_PRINT("%.4f ", x2Data[i]);
    LOG_PRINT("\n");

    aclTensor* x1 = nullptr; void* x1Dev = nullptr;
    ret = CreateAclTensor(x1Data, shape, &x1Dev, ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* x2 = nullptr; void* x2Dev = nullptr;
    ret = CreateAclTensor(x2Data, shape, &x2Dev, ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* out = nullptr; void* outDev = nullptr;
    std::vector<float> outInit(total, 0);
    ret = CreateAclTensor(outInit, shape, &outDev, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    ret = aclnnRotaryPositionEmbedding3dGetWorkspaceSize(x1, x2, out, &wsSize, &exec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d] GetWorkspaceSize failed. %d\n", testIdx, ret); return ret);

    void* ws = nullptr;
    if (wsSize > 0) { aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST); }

    // 正确性验证（1次）
    ret = aclnnRotaryPositionEmbedding3d(ws, wsSize, exec, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d] Execute failed. %d\n", testIdx, ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d] Sync failed. %d\n%s", testIdx, ret, aclGetRecentErrMsg()); return ret);

    std::vector<float> outData(total, 0);
    aclrtMemcpy(outData.data(), total * sizeof(float), outDev, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    const float eps = 1e-4f;
    bool allOk = true;
    for (int64_t i = 0; i < total; i++) {
        if (std::fabs(outData[i] - yRef[i]) > eps) {
            if (allOk) {
                LOG_PRINT("  y[0..%d]: ", printN - 1);
                for (int j = 0; j < printN; j++) LOG_PRINT("%.4f ", outData[j]);
                LOG_PRINT("\n  yRef[0..%d]: ", printN - 1);
                for (int j = 0; j < printN; j++) LOG_PRINT("%.4f ", yRef[j]);
                LOG_PRINT("\n");
                allOk = false;
            }
            LOG_PRINT("[Test %d] y[%ld] mismatch: exp %.6f got %.6f (diff %.6f)\n",
                      testIdx, i, yRef[i], outData[i], std::fabs(outData[i] - yRef[i]));
        }
    }

    // 性能测试（仅 benchmark 模式跑）
    double avgMs = 0.0;
    if (benchmark && allOk) {
        const int kWarmup = 5;
        const int kIter = 20;
        // 每次调用需要重建 executor（one-shot）
        for (int i = 0; i < kWarmup; i++) {
            exec = nullptr;
            ret = aclnnRotaryPositionEmbedding3dGetWorkspaceSize(x1, x2, out, &wsSize, &exec);
            if (ret != ACL_SUCCESS) break;
            ret = aclnnRotaryPositionEmbedding3d(ws, wsSize, exec, stream);
            aclrtSynchronizeStream(stream);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kIter; i++) {
            exec = nullptr;
            aclnnRotaryPositionEmbedding3dGetWorkspaceSize(x1, x2, out, &wsSize, &exec);
            aclnnRotaryPositionEmbedding3d(ws, wsSize, exec, stream);
            aclrtSynchronizeStream(stream);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        avgMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kIter;
    }

    if (!allOk) {
        aclDestroyTensor(x1); aclrtFree(x1Dev);
        aclDestroyTensor(x2); aclrtFree(x2Dev);
        aclDestroyTensor(out); aclrtFree(outDev);
        if (wsSize > 0) aclrtFree(ws);
        return -1;
    }

    LOG_PRINT("[Test %d] PASSED (batch=%ld seq=%ld D=%ld)", testIdx, batch, seqLen, headDim);
    if (avgMs > 0) {
        LOG_PRINT("  %.3f ms\n", avgMs);
    } else {
        LOG_PRINT("\n");
    }
    LOG_PRINT("  y[0..%d]: ", printN - 1);
    for (int i = 0; i < printN; i++) LOG_PRINT("%.4f ", outData[i]);
    LOG_PRINT("\n");

    aclDestroyTensor(x1); aclrtFree(x1Dev);
    aclDestroyTensor(x2); aclrtFree(x2Dev);
    aclDestroyTensor(out); aclrtFree(outDev);
    if (wsSize > 0) aclrtFree(ws);
    return 0;
}

int main()
{
    int32_t deviceId = 0; aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. %d\n", ret); return ret);

    struct Case { int64_t b, s, d; const char* desc; };
    Case tests[] = {
        // 小规模
        {1, 16, 16,  "M=16  D=16"},
        {2, 16, 32,  "M=32  D=32"},
        {4, 8,  64,  "M=32  D=64"},
        // 中规模
        {1, 64, 128, "M=64  D=128"},
        {4, 32, 128, "M=128 D=128"},
        {8, 16, 256, "M=128 D=256"},
        // 大规模
        {2, 128, 512,"M=256 D=512"},
        {4, 128, 256,"M=512 D=256"},
        {8, 128, 128,"M=1024 D=128"},
    };
    int numTests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0, failed = 0;
    LOG_PRINT("%-25s %-12s %-8s\n", "Case", "Time(ms)", "Result");
    LOG_PRINT("------------------------------------------\n");
    for (int i = 0; i < numTests; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ret = RunTest(tests[i].b, tests[i].s, tests[i].d, i+1, stream, numTests <= 10);
        auto t1 = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ret) {
            LOG_PRINT("=== Test %d FAILED ===\n", i+1); failed++;
            LOG_PRINT("%-25s %-12.3f %-8s\n", tests[i].desc, totalMs, "FAIL");
        } else {
            passed++;
        }
    }
    LOG_PRINT("------------------------------------------\n");
    LOG_PRINT("=== %d/%d tests PASSED, %d FAILED ===\n", passed, numTests, failed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failed ? -1 : 0;
}
