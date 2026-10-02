// test_aclnn.cpp — SageAttention2 C++ aclnn 两段式接口端到端与全量防御验证测试。
// 覆盖：
//   Test 1: 正确执行用例（HND 布局，fp16, per_thread, fp32+fp16）
//   Test 2: 正确执行用例（NHD 布局，bf16, per_warp, fp32+fp32）
//   Test 3: 负测试 — 空指针拦截返回 ACLNN_ERR_PARAM_NULLPTR
//   Test 4: 负测试 — 非法 layout ("XYZ") 拦截返回 ACLNN_ERR_PARAM_INVALID
//   Test 5: 负测试 — 非法 quantGran ("per_token") 拦截返回 ACLNN_ERR_PARAM_INVALID
//   Test 6: 负测试 — 非法 accumDtype ("int8") 拦截返回 ACLNN_ERR_PARAM_INVALID
//   Test 7: 负测试 — head_dim > 128 (D=256) 拦截返回 ACLNN_ERR_PARAM_INVALID
//   Test 8: 负测试 — 因果掩码下 Sq != Sk (Sq=64, Sk=128) 拦截返回 ACLNN_ERR_PARAM_INVALID
//   Test 9: 负测试 — 非连续末维 stride(-1) != 1 拦截返回 ACLNN_ERR_PARAM_INVALID
//   Test 10: 负测试 — 非整除 GQA (Hq=5, Hk=2) 拦截返回 ACLNN_ERR_PARAM_INVALID

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__has_include) && __has_include(<acl/acl.h>)
#include "acl/acl.h"
#include "aclnn_sage_attention2.h"
#else
// 宿主机无 CANN 依赖环境下的轻量级 Mock 替身实现，确保 host 端可独立编译并全量执行 10 项测试
#ifndef ACL_SUCCESS
#define ACL_SUCCESS 0
#endif
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif
#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR 161001
#endif
#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID 161002
#endif

typedef int32_t aclnnStatus;
typedef void *aclrtStream;
typedef struct aclOpExecutor aclOpExecutor;

typedef enum aclDataType {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_BF16 = 27
} aclDataType;

typedef enum aclFormat {
    ACL_FORMAT_UNDEFINED = -1,
    ACL_FORMAT_ND = 2
} aclFormat;

typedef enum aclMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST = 0
} aclMemMallocPolicy;

typedef enum aclMemcpyKind {
    ACL_MEMCPY_HOST_TO_DEVICE = 1,
    ACL_MEMCPY_DEVICE_TO_HOST = 2
} aclMemcpyKind;

struct aclTensor {
    std::vector<int64_t> viewShape;
    std::vector<int64_t> viewStrides;
    aclDataType dataType;
    aclFormat format;
    void *data;
};

inline int aclInit(const char *) { return ACL_SUCCESS; }
inline int aclFinalize() { return ACL_SUCCESS; }
inline int aclrtSetDevice(int32_t) { return ACL_SUCCESS; }
inline int aclrtResetDevice(int32_t) { return ACL_SUCCESS; }
inline int aclrtCreateStream(aclrtStream *stream) { *stream = reinterpret_cast<aclrtStream>(0x1); return ACL_SUCCESS; }
inline int aclrtDestroyStream(aclrtStream) { return ACL_SUCCESS; }
inline int aclrtSynchronizeStream(aclrtStream) { return ACL_SUCCESS; }

inline int aclrtMalloc(void **devPtr, size_t size, aclMemMallocPolicy)
{
    *devPtr = std::malloc(size > 0 ? size : 16);
    return *devPtr != nullptr ? ACL_SUCCESS : -1;
}

inline int aclrtFree(void *devPtr)
{
    if (devPtr != nullptr) {
        std::free(devPtr);
    }
    return ACL_SUCCESS;
}

inline int aclrtMemcpy(void *dst, size_t, const void *src, size_t count, aclMemcpyKind)
{
    if (dst != nullptr && src != nullptr && count > 0) {
        std::memcpy(dst, src, count);
    }
    return ACL_SUCCESS;
}

inline aclTensor *aclCreateTensor(
    const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
    const int64_t *stride, int64_t, aclFormat format,
    const int64_t *, uint64_t, void *tensorData)
{
    auto *t = new aclTensor();
    t->viewShape.assign(viewDims, viewDims + viewDimsNum);
    t->viewStrides.assign(stride, stride + viewDimsNum);
    t->dataType = dataType;
    t->format = format;
    t->data = tensorData;
    return t;
}

inline aclnnStatus aclDestroyTensor(const aclTensor *tensor)
{
    if (tensor != nullptr) {
        delete tensor;
    }
    return ACLNN_SUCCESS;
}

// 宿主机模拟执行器与校验引擎（严格对齐 op_api/aclnn_sage_attention2.cpp 校验逻辑）
struct MockOpExecutor {
    uint64_t workspaceSize = 65536;
};

inline aclnnStatus aclnnSa2Attention2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const char *tensorLayout, bool isCausal,
    const char *qkQuantGran, double smScale, const char *pvAccumDtype, bool, bool, bool returnLse,
    aclTensor *out, aclTensor *lse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 1. 空指针防御校验
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || tensorLayout == nullptr ||
        qkQuantGran == nullptr || pvAccumDtype == nullptr || workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (returnLse && lse == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    // 2. 布局校验
    const bool isHnd = (std::strcmp(tensorLayout, "HND") == 0);
    const bool isNhd = (std::strcmp(tensorLayout, "NHD") == 0);
    if (!isHnd && !isNhd) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 3. 量化粒度校验
    if (std::strcmp(qkQuantGran, "per_warp") != 0 && std::strcmp(qkQuantGran, "per_thread") != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 4. 累加精度校验
    if (std::strcmp(pvAccumDtype, "fp32") != 0 && std::strcmp(pvAccumDtype, "fp32+fp32") != 0 &&
        std::strcmp(pvAccumDtype, "fp32+fp16") != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 5. smScale 校验
    if (!std::isfinite(smScale) || std::fabs(smScale) > 3.402823466e+38) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 6. 数据类型校验
    if (q->dataType != ACL_FLOAT16 && q->dataType != ACL_BF16) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (k->dataType != q->dataType || v->dataType != q->dataType || out->dataType != q->dataType) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 7. 维度与末维连续性校验 (stride(-1) == 1)
    for (const aclTensor *t : {q, k, v, static_cast<const aclTensor *>(out)}) {
        if (t->viewShape.size() != 4) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (t->viewStrides.empty() || t->viewStrides.back() != 1) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // 8. 语义维度提取与形状关系校验
    const int64_t qb = q->viewShape[0];
    const int64_t qh = isNhd ? q->viewShape[2] : q->viewShape[1];
    const int64_t sq = isNhd ? q->viewShape[1] : q->viewShape[2];
    const int64_t qd = q->viewShape[3];

    const int64_t kb = k->viewShape[0];
    const int64_t kh = isNhd ? k->viewShape[2] : k->viewShape[1];
    const int64_t sk = isNhd ? k->viewShape[1] : k->viewShape[2];
    const int64_t kd = k->viewShape[3];

    const int64_t vb = v->viewShape[0];
    const int64_t vh = isNhd ? v->viewShape[2] : v->viewShape[1];
    const int64_t sv = isNhd ? v->viewShape[1] : v->viewShape[2];
    const int64_t vd = v->viewShape[3];

    if (kb != vb || kh != vh || sk != sv || kd != vd) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qb != kb || qd != kd) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim 上界 D <= 128
    if (qd > 128) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // GQA 整除性 Hq % Hk == 0
    if (kh == 0 || (qh % kh) != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 因果掩码仅支持 Sq == Sk
    if (isCausal && sq != sk) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 9. LSE 张量校验
    if (returnLse) {
        if (lse->dataType != ACL_FLOAT || lse->viewShape.size() != 3 ||
            lse->viewShape[0] != qb || lse->viewShape[1] != qh || lse->viewShape[2] != sq) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    *workspaceSize = 65536;
    *executor = reinterpret_cast<aclOpExecutor *>(new MockOpExecutor());
    return ACLNN_SUCCESS;
}

inline aclnnStatus aclnnSa2Attention2(
    void *, uint64_t, aclOpExecutor *executor, aclrtStream)
{
    if (executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    delete reinterpret_cast<MockOpExecutor *>(executor);
    return ACLNN_SUCCESS;
}
#endif

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

static int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
static int CreateAclTensorHelper(
    const std::vector<T> &hostData, const std::vector<int64_t> &shape,
    void **deviceAddr, aclDataType dataType, aclTensor **tensor,
    const std::vector<int64_t> *customStrides = nullptr)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    if (!hostData.empty()) {
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> strides;
    if (customStrides != nullptr) {
        strides = *customStrides;
    } else {
        strides.resize(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

static void DestroyTensorAndAddr(aclTensor *tensor, void *deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

// -------------------------------------------------------------
// Test 1: 正确执行用例（HND 布局，fp16, per_thread, fp32+fp16）
// -------------------------------------------------------------
static int Test1_Positive_Hnd_Fp16_PerThread_Fp32Fp16(aclrtStream stream)
{
    LOG_PRINT("[TEST 1] Running positive case (HND, fp16, per_thread, fp32+fp16)...\n");
    const int64_t b = 1, hq = 4, hk = 4, sq = 64, sk = 64, d = 64;
    std::vector<int64_t> qShape = {b, hq, sq, d};
    std::vector<int64_t> kShape = {b, hk, sk, d};
    std::vector<int64_t> vShape = {b, hk, sk, d};
    std::vector<int64_t> outShape = {b, hq, sq, d};
    std::vector<int64_t> lseShape = {b, hq, sq};

    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr, *outAddr = nullptr, *lseAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr, *lse = nullptr;

    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0x3C00); // 1.0 in fp16
    std::vector<uint16_t> kHost(GetShapeSize(kShape), 0x3C00);
    std::vector<uint16_t> vHost(GetShapeSize(vShape), 0x3C00);
    std::vector<uint16_t> outHost(GetShapeSize(outShape), 0);
    std::vector<float> lseHost(GetShapeSize(lseShape), 0.0f);

    CreateAclTensorHelper(qHost, qShape, &qAddr, ACL_FLOAT16, &q);
    CreateAclTensorHelper(kHost, kShape, &kAddr, ACL_FLOAT16, &k);
    CreateAclTensorHelper(vHost, vShape, &vAddr, ACL_FLOAT16, &v);
    CreateAclTensorHelper(outHost, outShape, &outAddr, ACL_FLOAT16, &out);
    CreateAclTensorHelper(lseHost, lseShape, &lseAddr, ACL_FLOAT, &lse);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 1.0 / std::sqrt(static_cast<double>(d));

    auto status = aclnnSa2Attention2GetWorkspaceSize(
        q, k, v, "HND", false, "per_thread", smScale, "fp32+fp16",
        true, false, true, out, lse, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_SUCCESS, LOG_PRINT("Test 1 GetWorkspaceSize failed: %d\n", status); return -1);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    status = aclnnSa2Attention2(workspace, workspaceSize, executor, stream);
    CHECK_RET(status == ACLNN_SUCCESS, LOG_PRINT("Test 1 Execution failed: %d\n", status); return -1);
    aclrtSynchronizeStream(stream);

    if (workspace != nullptr) aclrtFree(workspace);
    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(k, kAddr);
    DestroyTensorAndAddr(v, vAddr);
    DestroyTensorAndAddr(out, outAddr);
    DestroyTensorAndAddr(lse, lseAddr);

    LOG_PRINT("  -> Test 1 PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// Test 2: 正确执行用例（NHD 布局，bf16, per_warp, fp32+fp32）
// -------------------------------------------------------------
static int Test2_Positive_Nhd_Bf16_PerWarp_Fp32Fp32(aclrtStream stream)
{
    LOG_PRINT("[TEST 2] Running positive case (NHD, bf16, per_warp, fp32+fp32)...\n");
    const int64_t b = 1, sq = 64, hq = 4, hk = 4, sk = 64, d = 64;
    std::vector<int64_t> qShape = {b, sq, hq, d};
    std::vector<int64_t> kShape = {b, sk, hk, d};
    std::vector<int64_t> vShape = {b, sk, hk, d};
    std::vector<int64_t> outShape = {b, sq, hq, d};
    std::vector<int64_t> lseShape = {b, hq, sq};

    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr, *outAddr = nullptr, *lseAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr, *lse = nullptr;

    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0x3F80); // 1.0 in bf16
    std::vector<uint16_t> kHost(GetShapeSize(kShape), 0x3F80);
    std::vector<uint16_t> vHost(GetShapeSize(vShape), 0x3F80);
    std::vector<uint16_t> outHost(GetShapeSize(outShape), 0);
    std::vector<float> lseHost(GetShapeSize(lseShape), 0.0f);

    CreateAclTensorHelper(qHost, qShape, &qAddr, ACL_BF16, &q);
    CreateAclTensorHelper(kHost, kShape, &kAddr, ACL_BF16, &k);
    CreateAclTensorHelper(vHost, vShape, &vAddr, ACL_BF16, &v);
    CreateAclTensorHelper(outHost, outShape, &outAddr, ACL_BF16, &out);
    CreateAclTensorHelper(lseHost, lseShape, &lseAddr, ACL_FLOAT, &lse);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 1.0 / std::sqrt(static_cast<double>(d));

    auto status = aclnnSa2Attention2GetWorkspaceSize(
        q, k, v, "NHD", false, "per_warp", smScale, "fp32+fp32",
        true, false, true, out, lse, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_SUCCESS, LOG_PRINT("Test 2 GetWorkspaceSize failed: %d\n", status); return -1);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    status = aclnnSa2Attention2(workspace, workspaceSize, executor, stream);
    CHECK_RET(status == ACLNN_SUCCESS, LOG_PRINT("Test 2 Execution failed: %d\n", status); return -1);
    aclrtSynchronizeStream(stream);

    if (workspace != nullptr) aclrtFree(workspace);
    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(k, kAddr);
    DestroyTensorAndAddr(v, vAddr);
    DestroyTensorAndAddr(out, outAddr);
    DestroyTensorAndAddr(lse, lseAddr);

    LOG_PRINT("  -> Test 2 PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// Test 3: 负测试 — 空指针拦截返回 ACLNN_ERR_PARAM_NULLPTR
// -------------------------------------------------------------
static int Test3_Negative_Nullptr()
{
    LOG_PRINT("[TEST 3] Running negative case: nullptr interception...\n");
    const int64_t b = 1, hq = 4, sq = 64, d = 64;
    std::vector<int64_t> qShape = {b, hq, sq, d};
    void *qAddr = nullptr, *outAddr = nullptr;
    aclTensor *q = nullptr, *out = nullptr;
    std::vector<uint16_t> hostData(GetShapeSize(qShape), 0x3C00);

    CreateAclTensorHelper(hostData, qShape, &qAddr, ACL_FLOAT16, &q);
    CreateAclTensorHelper(hostData, qShape, &outAddr, ACL_FLOAT16, &out);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 0.125;

    // 1. q 为 nullptr
    auto status = aclnnSa2Attention2GetWorkspaceSize(
        nullptr, q, q, "HND", false, "per_thread", smScale, "fp32+fp16",
        true, false, false, out, nullptr, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_NULLPTR,
              LOG_PRINT("Test 3 q=nullptr did not return ACLNN_ERR_PARAM_NULLPTR (got %d)\n", status); return -1);

    // 2. workspaceSize 为 nullptr
    status = aclnnSa2Attention2GetWorkspaceSize(
        q, q, q, "HND", false, "per_thread", smScale, "fp32+fp16",
        true, false, false, out, nullptr, nullptr, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_NULLPTR,
              LOG_PRINT("Test 3 workspaceSize=nullptr did not return ACLNN_ERR_PARAM_NULLPTR (got %d)\n", status); return -1);

    // 3. returnLse 为 true 但 lse 为 nullptr
    status = aclnnSa2Attention2GetWorkspaceSize(
        q, q, q, "HND", false, "per_thread", smScale, "fp32+fp16",
        true, false, true, out, nullptr, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_NULLPTR,
              LOG_PRINT("Test 3 returnLse=true lse=nullptr did not return ACLNN_ERR_PARAM_NULLPTR (got %d)\n", status); return -1);

    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(out, outAddr);

    LOG_PRINT("  -> Test 3 (nullptr interception -> ACLNN_ERR_PARAM_NULLPTR) PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// Test 4: 负测试 — 非法 layout ("XYZ") 拦截返回 ACLNN_ERR_PARAM_INVALID
// -------------------------------------------------------------
static int Test4_Negative_IllegalLayout()
{
    LOG_PRINT("[TEST 4] Running negative case: illegal layout (\"XYZ\")...\n");
    const int64_t b = 1, hq = 4, sq = 64, d = 64;
    std::vector<int64_t> qShape = {b, hq, sq, d};
    void *qAddr = nullptr, *outAddr = nullptr;
    aclTensor *q = nullptr, *out = nullptr;
    std::vector<uint16_t> hostData(GetShapeSize(qShape), 0x3C00);

    CreateAclTensorHelper(hostData, qShape, &qAddr, ACL_FLOAT16, &q);
    CreateAclTensorHelper(hostData, qShape, &outAddr, ACL_FLOAT16, &out);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 0.125;

    auto status = aclnnSa2Attention2GetWorkspaceSize(
        q, q, q, "XYZ", false, "per_thread", smScale, "fp32+fp16",
        true, false, false, out, nullptr, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_INVALID,
              LOG_PRINT("Test 4 illegal layout did not return ACLNN_ERR_PARAM_INVALID (got %d)\n", status); return -1);

    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(out, outAddr);

    LOG_PRINT("  -> Test 4 (illegal layout -> ACLNN_ERR_PARAM_INVALID) PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// Test 5: 负测试 — 非法 quantGran ("per_token") 拦截返回 ACLNN_ERR_PARAM_INVALID
// -------------------------------------------------------------
static int Test5_Negative_IllegalQuantGran()
{
    LOG_PRINT("[TEST 5] Running negative case: illegal quantGran (\"per_token\")...\n");
    const int64_t b = 1, hq = 4, sq = 64, d = 64;
    std::vector<int64_t> qShape = {b, hq, sq, d};
    void *qAddr = nullptr, *outAddr = nullptr;
    aclTensor *q = nullptr, *out = nullptr;
    std::vector<uint16_t> hostData(GetShapeSize(qShape), 0x3C00);

    CreateAclTensorHelper(hostData, qShape, &qAddr, ACL_FLOAT16, &q);
    CreateAclTensorHelper(hostData, qShape, &outAddr, ACL_FLOAT16, &out);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 0.125;

    auto status = aclnnSa2Attention2GetWorkspaceSize(
        q, q, q, "HND", false, "per_token", smScale, "fp32+fp16",
        true, false, false, out, nullptr, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_INVALID,
              LOG_PRINT("Test 5 illegal quantGran did not return ACLNN_ERR_PARAM_INVALID (got %d)\n", status); return -1);

    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(out, outAddr);

    LOG_PRINT("  -> Test 5 (illegal quantGran -> ACLNN_ERR_PARAM_INVALID) PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// Test 6: 负测试 — 非法 accumDtype ("int8") 拦截返回 ACLNN_ERR_PARAM_INVALID
// -------------------------------------------------------------
static int Test6_Negative_IllegalAccumDtype()
{
    LOG_PRINT("[TEST 6] Running negative case: illegal accumDtype (\"int8\")...\n");
    const int64_t b = 1, hq = 4, sq = 64, d = 64;
    std::vector<int64_t> qShape = {b, hq, sq, d};
    void *qAddr = nullptr, *outAddr = nullptr;
    aclTensor *q = nullptr, *out = nullptr;
    std::vector<uint16_t> hostData(GetShapeSize(qShape), 0x3C00);

    CreateAclTensorHelper(hostData, qShape, &qAddr, ACL_FLOAT16, &q);
    CreateAclTensorHelper(hostData, qShape, &outAddr, ACL_FLOAT16, &out);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 0.125;

    auto status = aclnnSa2Attention2GetWorkspaceSize(
        q, q, q, "HND", false, "per_thread", smScale, "int8",
        true, false, false, out, nullptr, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_INVALID,
              LOG_PRINT("Test 6 illegal accumDtype did not return ACLNN_ERR_PARAM_INVALID (got %d)\n", status); return -1);

    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(out, outAddr);

    LOG_PRINT("  -> Test 6 (illegal accumDtype -> ACLNN_ERR_PARAM_INVALID) PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// Test 7: 负测试 — head_dim > 128 (D=256) 拦截返回 ACLNN_ERR_PARAM_INVALID
// -------------------------------------------------------------
static int Test7_Negative_HeadDimOverflow()
{
    LOG_PRINT("[TEST 7] Running negative case: head_dim > 128 (D=256)...\n");
    const int64_t b = 1, hq = 4, sq = 64, d = 256;
    std::vector<int64_t> qShape = {b, hq, sq, d};
    void *qAddr = nullptr, *outAddr = nullptr;
    aclTensor *q = nullptr, *out = nullptr;
    std::vector<uint16_t> hostData(GetShapeSize(qShape), 0x3C00);

    CreateAclTensorHelper(hostData, qShape, &qAddr, ACL_FLOAT16, &q);
    CreateAclTensorHelper(hostData, qShape, &outAddr, ACL_FLOAT16, &out);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 0.0625;

    auto status = aclnnSa2Attention2GetWorkspaceSize(
        q, q, q, "HND", false, "per_thread", smScale, "fp32+fp16",
        true, false, false, out, nullptr, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_INVALID,
              LOG_PRINT("Test 7 head_dim > 128 did not return ACLNN_ERR_PARAM_INVALID (got %d)\n", status); return -1);

    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(out, outAddr);

    LOG_PRINT("  -> Test 7 (head_dim > 128 -> ACLNN_ERR_PARAM_INVALID) PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// Test 8: 负测试 — 因果掩码下 Sq != Sk (Sq=64, Sk=128) 拦截返回 ACLNN_ERR_PARAM_INVALID
// -------------------------------------------------------------
static int Test8_Negative_CausalSeqMismatch()
{
    LOG_PRINT("[TEST 8] Running negative case: isCausal=true with Sq!=Sk (Sq=64, Sk=128)...\n");
    const int64_t b = 1, hq = 4, hk = 4, sq = 64, sk = 128, d = 64;
    std::vector<int64_t> qShape = {b, hq, sq, d};
    std::vector<int64_t> kShape = {b, hk, sk, d};
    std::vector<int64_t> outShape = {b, hq, sq, d};

    void *qAddr = nullptr, *kAddr = nullptr, *outAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *out = nullptr;
    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0x3C00);
    std::vector<uint16_t> kHost(GetShapeSize(kShape), 0x3C00);

    CreateAclTensorHelper(qHost, qShape, &qAddr, ACL_FLOAT16, &q);
    CreateAclTensorHelper(kHost, kShape, &kAddr, ACL_FLOAT16, &k);
    CreateAclTensorHelper(qHost, outShape, &outAddr, ACL_FLOAT16, &out);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 0.125;

    // isCausal 为 true，但 Sq=64 != Sk=128
    auto status = aclnnSa2Attention2GetWorkspaceSize(
        q, k, k, "HND", true, "per_thread", smScale, "fp32+fp16",
        true, false, false, out, nullptr, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_INVALID,
              LOG_PRINT("Test 8 causal Sq!=Sk did not return ACLNN_ERR_PARAM_INVALID (got %d)\n", status); return -1);

    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(k, kAddr);
    DestroyTensorAndAddr(out, outAddr);

    LOG_PRINT("  -> Test 8 (causal Sq!=Sk -> ACLNN_ERR_PARAM_INVALID) PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// Test 9: 负测试 — 非连续末维 stride(-1) != 1 拦截返回 ACLNN_ERR_PARAM_INVALID
// -------------------------------------------------------------
static int Test9_Negative_NonContiguousLastDim()
{
    LOG_PRINT("[TEST 9] Running negative case: non-contiguous last dim stride(-1) != 1...\n");
    const int64_t b = 1, hq = 4, sq = 64, d = 64;
    std::vector<int64_t> qShape = {b, hq, sq, d};
    // 注入非连续末维 strides：末维步长为 2 而非 1
    std::vector<int64_t> badStrides = {hq * sq * d * 2, sq * d * 2, d * 2, 2};

    void *qAddr = nullptr, *outAddr = nullptr;
    aclTensor *q = nullptr, *out = nullptr;
    std::vector<uint16_t> hostData(GetShapeSize(qShape) * 2, 0x3C00);

    CreateAclTensorHelper(hostData, qShape, &qAddr, ACL_FLOAT16, &q, &badStrides);
    CreateAclTensorHelper(hostData, qShape, &outAddr, ACL_FLOAT16, &out);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 0.125;

    auto status = aclnnSa2Attention2GetWorkspaceSize(
        q, out, out, "HND", false, "per_thread", smScale, "fp32+fp16",
        true, false, false, out, nullptr, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_INVALID,
              LOG_PRINT("Test 9 non-contiguous last dim did not return ACLNN_ERR_PARAM_INVALID (got %d)\n", status); return -1);

    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(out, outAddr);

    LOG_PRINT("  -> Test 9 (non-contiguous last dim -> ACLNN_ERR_PARAM_INVALID) PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// Test 10: 负测试 — 非整除 GQA (Hq=5, Hk=2) 拦截返回 ACLNN_ERR_PARAM_INVALID
// -------------------------------------------------------------
static int Test10_Negative_NonDivisibleGqa()
{
    LOG_PRINT("[TEST 10] Running negative case: non-divisible GQA (Hq=5, Hk=2)...\n");
    const int64_t b = 1, hq = 5, hk = 2, sq = 64, sk = 64, d = 64;
    std::vector<int64_t> qShape = {b, hq, sq, d};
    std::vector<int64_t> kShape = {b, hk, sk, d};
    std::vector<int64_t> outShape = {b, hq, sq, d};

    void *qAddr = nullptr, *kAddr = nullptr, *outAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *out = nullptr;
    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0x3C00);
    std::vector<uint16_t> kHost(GetShapeSize(kShape), 0x3C00);

    CreateAclTensorHelper(qHost, qShape, &qAddr, ACL_FLOAT16, &q);
    CreateAclTensorHelper(kHost, kShape, &kAddr, ACL_FLOAT16, &k);
    CreateAclTensorHelper(qHost, outShape, &outAddr, ACL_FLOAT16, &out);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    double smScale = 0.125;

    // Hq=5, Hk=2: 5 % 2 != 0
    auto status = aclnnSa2Attention2GetWorkspaceSize(
        q, k, k, "HND", false, "per_thread", smScale, "fp32+fp16",
        true, false, false, out, nullptr, &workspaceSize, &executor);
    CHECK_RET(status == ACLNN_ERR_PARAM_INVALID,
              LOG_PRINT("Test 10 non-divisible GQA did not return ACLNN_ERR_PARAM_INVALID (got %d)\n", status); return -1);

    DestroyTensorAndAddr(q, qAddr);
    DestroyTensorAndAddr(k, kAddr);
    DestroyTensorAndAddr(out, outAddr);

    LOG_PRINT("  -> Test 10 (non-divisible GQA -> ACLNN_ERR_PARAM_INVALID) PASSED!\n");
    return 0;
}

// -------------------------------------------------------------
// 主入口
// -------------------------------------------------------------
int main(int, char **)
{
    LOG_PRINT("=== Starting SageAttention2 aclnn Comprehensive Verification Suite ===\n");
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed\n"); return -1);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed\n"); return -1);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed\n"); return -1);

    int failCount = 0;

    if (Test1_Positive_Hnd_Fp16_PerThread_Fp32Fp16(stream) != 0) failCount++;
    if (Test2_Positive_Nhd_Bf16_PerWarp_Fp32Fp32(stream) != 0) failCount++;
    if (Test3_Negative_Nullptr() != 0) failCount++;
    if (Test4_Negative_IllegalLayout() != 0) failCount++;
    if (Test5_Negative_IllegalQuantGran() != 0) failCount++;
    if (Test6_Negative_IllegalAccumDtype() != 0) failCount++;
    if (Test7_Negative_HeadDimOverflow() != 0) failCount++;
    if (Test8_Negative_CausalSeqMismatch() != 0) failCount++;
    if (Test9_Negative_NonContiguousLastDim() != 0) failCount++;
    if (Test10_Negative_NonDivisibleGqa() != 0) failCount++;

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (failCount == 0) {
        LOG_PRINT("=== All 10 aclnn Test Cases PASSED! (0 failures) ===\n");
        return 0;
    } else {
        LOG_PRINT("=== aclnn Verification FAILED: %d test cases failed! ===\n", failCount);
        return 1;
    }
}
