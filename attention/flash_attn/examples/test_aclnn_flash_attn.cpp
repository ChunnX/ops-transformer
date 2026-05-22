/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_flash_attn.cpp
 * \brief FlashAttn + FlashAttnMetadata 算子调用示例
 *
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <cstdio>
#include <string>
#include <sstream>
#include "acl/acl.h"
#include "../op_api/aclnn_flash_attn.h"

namespace {

static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (mant << 13);  // zero / denormal → treat as zero here
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (mant << 13);  // inf / nan
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

#define CHECK_RET(cond)  ((cond) ? true : (false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// 直接在 device 上分配并 memset 为 0，不需要 host 侧大数组
// 用于 K/V 这类大 tensor，避免主机内存 bad_alloc
static int CreateAclTensorDeviceZero(const std::vector<int64_t> &shape, void **deviceAddr,
                                     aclDataType dataType, size_t elemSize, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * (int64_t)elemSize;
    auto ret = aclrtMalloc(deviceAddr, (size_t)size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc (device zero) failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemset(*deviceAddr, (size_t)size, 0, (size_t)size);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemset failed. ERROR: %d\n", ret);
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

static uint16_t float_to_fp16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ffu;
    if (exp <= 0)  return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | 0x7c00u);
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | mant);
}

// maskMode: 0=全量注意力  1=RIGHT_DOWN_CAUSAL(下三角因果)  其他暂不支持CPU参考
static void cpu_flash_attention(
    const std::vector<float> &Q,   // BNSD [B, N_q, S_q, D]
    const std::vector<float> &K,   // BNSD [B, N_kv, S_kv, D]
    const std::vector<float> &V,   // BNSD [B, N_kv, S_kv, D]
    std::vector<float>       &Out, // BNSD [B, N_q, S_q, D]
    int64_t B, int64_t N_q, int64_t N_kv,
    int64_t S_q, int64_t S_kv, int64_t D,
    int64_t maskMode = 0)
{
    float scale = 1.0f / std::sqrt((float)D);
    // 因果偏移: position (s1, s2) 有效当且仅当 s2 <= s1 + causalOffset
    // RIGHT_DOWN_CAUSAL: causalOffset = S_kv - S_q
    int64_t causalOffset = (maskMode == 1) ? (S_kv - S_q) : (S_kv - 1);
    std::vector<float> score(S_q * S_kv);
    for (int64_t b = 0; b < B; b++) {
        for (int64_t nq = 0; nq < N_q; nq++) {
            int64_t nkv = nq * N_kv / N_q;  // GQA head mapping
            // 1. QK^T * scale（含因果掩码）
            for (int64_t s1 = 0; s1 < S_q; s1++) {
                for (int64_t s2 = 0; s2 < S_kv; s2++) {
                    if (maskMode == 1 && s2 > s1 + causalOffset) {
                        score[s1 * S_kv + s2] = -1e9f;  // masked
                        continue;
                    }
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; d++) {
                        dot += Q[((b * N_q  + nq)  * S_q  + s1) * D + d] *
                               K[((b * N_kv + nkv) * S_kv + s2) * D + d];
                    }
                    score[s1 * S_kv + s2] = dot * scale;
                }
            }
            // 2. row-wise softmax
            for (int64_t s1 = 0; s1 < S_q; s1++) {
                float mx = *std::max_element(score.data() + s1 * S_kv,
                                             score.data() + s1 * S_kv + S_kv);
                float sm = 0.0f;
                for (int64_t s2 = 0; s2 < S_kv; s2++) {
                    score[s1 * S_kv + s2] = std::exp(score[s1 * S_kv + s2] - mx);
                    sm += score[s1 * S_kv + s2];
                }
                for (int64_t s2 = 0; s2 < S_kv; s2++)
                    score[s1 * S_kv + s2] /= sm;
            }
            // 3. softmax * V
            for (int64_t s1 = 0; s1 < S_q; s1++) {
                for (int64_t d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (int64_t s2 = 0; s2 < S_kv; s2++) {
                        acc += score[s1 * S_kv + s2] *
                               V[((b * N_kv + nkv) * S_kv + s2) * D + d];
                    }
                    Out[((b * N_q + nq) * S_q + s1) * D + d] = acc;
                }
            }
        }
    }
}

// 分块随机填充 device tensor（避免大 tensor 一次性分配主机内存）
// floatRef: 若非 null，存储前 refElems 个元素的 float 值（fp16 round-trip）
static int CreateAclTensorRandom(const std::vector<int64_t> &shape, void **deviceAddr,
                                 aclDataType dataType, aclTensor **tensor,
                                 std::mt19937 &rng,
                                 float randMin = 0.0f, float randMax = 1.0f,
                                 std::vector<float> *floatRef = nullptr,
                                 int64_t refElems = 0)
{
    const int64_t CHUNK = 1 << 20;  // 1M elements per chunk (~2MB fp16)
    int64_t total = GetShapeSize(shape);
    auto ret = aclrtMalloc(deviceAddr, (size_t)total * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { LOG_PRINT("aclrtMalloc random failed. ERROR: %d\n", ret); return ret; }
    if (floatRef && refElems > 0) floatRef->resize(refElems);
    std::uniform_real_distribution<float> dist(randMin, randMax);
    std::vector<uint16_t> chunk(std::min(CHUNK, total));
    int64_t offset = 0;
    while (offset < total) {
        int64_t count = std::min(CHUNK, total - offset);
        for (int64_t i = 0; i < count; ++i) {
            float v = dist(rng);
            uint16_t h = float_to_fp16(v);
            chunk[i] = h;
            if (floatRef && offset + i < refElems)
                (*floatRef)[offset + i] = fp16_to_float(h);  // fp16 round-trip
        }
        ret = aclrtMemcpy(reinterpret_cast<uint8_t*>(*deviceAddr) + offset * sizeof(uint16_t),
                          (size_t)count * sizeof(uint16_t), chunk.data(),
                          (size_t)count * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        if (!CHECK_RET(ret == ACL_SUCCESS)) { LOG_PRINT("aclrtMemcpy chunk failed. ERROR: %d\n", ret); return ret; }
        offset += count;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; --i)
        strides[i] = shape[i + 1] * strides[i + 1];
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// ─── 元数据 tensor 分配（uint32，预清零，作为 flash_attn_metadata 输出）───
// 布局（uint32 元素，参见 flash_attn_metadata.h）：
//   [0]       sectionNum
//   [1]       mBaseSize
//   [2]       s2BaseSize
//   [3..15]   reserved
//   FA 段: [16 + sec*36*16 + core*16 + field]  (7 字段有效)
//   FD 段: [16 + sectionNum*36*16 + sec*72*16 + core*16 + field]  (6 字段有效)
static constexpr int64_t METADATA_MAX_ELEMS = 8192;  // uint32；支持最多 ~4 section
static constexpr uint32_t META_AIC = 36, META_AIV = 72;
static constexpr uint32_t META_FA_SLOT = 16, META_FD_SLOT = 16, META_HDR = 16;

static int CreateMetadataTensor(void **deviceAddr, aclTensor **tensor)
{
    size_t size = METADATA_MAX_ELEMS * sizeof(uint32_t);
    int ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtMalloc metadata failed. ERROR: %d\n", ret); return ret; }
    ret = aclrtMemset(*deviceAddr, size, 0, size);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtMemset metadata failed. ERROR: %d\n", ret); return ret; }
    int64_t shape[1] = {METADATA_MAX_ELEMS};
    int64_t strides[1] = {1};
    *tensor = aclCreateTensor(shape, 1, aclDataType::ACL_INT32, strides, 0,
                              aclFormat::ACL_FORMAT_ND, shape, 1, *deviceAddr);
    return ACL_SUCCESS;
}

// ─── 形象打印 metadata buffer（对标 npu_impl.py::print_metadata）───────────
static void PrintMetadata(const uint32_t *meta, int64_t totalElems)
{
    if (totalElems < (int64_t)META_HDR) {
        LOG_PRINT("[PrintMetadata] buffer 太小 (%ld 元素)\n", totalElems);
        return;
    }
    uint32_t sectionNum = meta[0];
    uint32_t mBaseSize  = meta[1];
    uint32_t s2BaseSize = meta[2];

    static const char *FA_FIELDS[7] = {
        "BN2_START", "M_START",  "S2_START",
        "BN2_END",   "M_END",    "S2_END",   "FIRST_FD_WS"
    };
    static const char *FD_FIELDS[6] = {
        "BN2_IDX", "M_IDX", "WS_IDX", "WS_NUM", "M_START", "M_NUM"
    };

    const char *SEP  = "==============================================================================";
    const char *DASH = "------------------------------------------------------------------------------";

    LOG_PRINT("\n%s\n", SEP);
    LOG_PRINT("  [Metadata Header]\n");
    LOG_PRINT("  sectionNum  = %u\n", sectionNum);
    LOG_PRINT("  mBaseSize   = %u\n", mBaseSize);
    LOG_PRINT("  s2BaseSize  = %u\n", s2BaseSize);

    // ── FA 段 ──────────────────────────────────────────────────────────────────
    for (uint32_t sec = 0; sec < sectionNum; sec++) {
        LOG_PRINT("\n%s\n", DASH);
        LOG_PRINT("  [Section %u] FA Metadata — AIC cores  (36 × 16 slots, 7 字段有效)\n", sec);
        LOG_PRINT("%s\n", DASH);

        // 表头
        LOG_PRINT("  %-8s", "Core");
        for (int f = 0; f < 7; f++) LOG_PRINT("%14s", FA_FIELDS[f]);
        LOG_PRINT("\n  %-8s", "");
        for (int f = 0; f < 7; f++) {
            char buf[8]; snprintf(buf, sizeof(buf), "[%d]", f);
            LOG_PRINT("%14s", buf);
        }
        LOG_PRINT("\n  %s\n", "--------" "--------------------------------------------------------------"
                              "----------------------------------------------------------");

        // 每核一行
        for (uint32_t core = 0; core < META_AIC; core++) {
            uint32_t off = META_HDR + sec * META_AIC * META_FA_SLOT + core * META_FA_SLOT;
            uint32_t vals[7];
            bool allZero = true;
            for (int f = 0; f < 7; f++) {
                vals[f] = meta[off + f];
                if (vals[f]) allZero = false;
            }
            LOG_PRINT("  AIC%02u   ", core);
            for (int f = 0; f < 7; f++) LOG_PRINT("%14u", vals[f]);
            if (allZero) LOG_PRINT("  (inactive)");
            LOG_PRINT("\n");
        }
    }

    // ── FD 段（只打印活跃核，M_NUM > 0）──────────────────────────────────────
    for (uint32_t sec = 0; sec < sectionNum; sec++) {
        LOG_PRINT("\n%s\n", DASH);
        LOG_PRINT("  [Section %u] FD Metadata — 活跃 AIV cores  (M_NUM>0, 72 × 16 slots, 6 字段有效)\n", sec);
        LOG_PRINT("%s\n", DASH);

        LOG_PRINT("  %-8s", "Core");
        for (int f = 0; f < 6; f++) LOG_PRINT("%12s", FD_FIELDS[f]);
        LOG_PRINT("\n  %-8s", "");
        for (int f = 0; f < 6; f++) {
            char buf[8]; snprintf(buf, sizeof(buf), "[%d]", f);
            LOG_PRINT("%12s", buf);
        }
        LOG_PRINT("\n  %s\n",
                  "--------" "------------------------------------------------------------"
                  "----------------------------------");

        uint32_t fdBase = META_HDR + sectionNum * META_AIC * META_FA_SLOT;
        int active = 0;
        for (uint32_t core = 0; core < META_AIV; core++) {
            uint32_t off = fdBase + sec * META_AIV * META_FD_SLOT + core * META_FD_SLOT;
            if (meta[off + 5] == 0) continue;  // M_NUM == 0 → 非活跃核
            LOG_PRINT("  AIV%02u   ", core);
            for (int f = 0; f < 6; f++) LOG_PRINT("%12u", meta[off + f]);
            LOG_PRINT("\n");
            ++active;
        }
        if (active == 0) LOG_PRINT("  (no active FD cores)\n");
    }
    LOG_PRINT("%s\n\n", SEP);
}

// ─── 保存浮点向量到 txt 文件（对标 save_tensor_to_txt）─────────────────────
static void SaveFloatToTxt(const char *path, const float *data, int64_t n,
                           const char *header = nullptr)
{
    FILE *f = fopen(path, "w");
    if (!f) { LOG_PRINT("[WARN] 无法创建文件: %s\n", path); return; }
    if (header) fprintf(f, "# %s\n", header);
    for (int64_t i = 0; i < n; ++i) fprintf(f, "%.8f\n", data[i]);
    fclose(f);
}

// 从 pytest --dump_metadata 生成的 .bin 文件加载 metadata tensor 到 device
static int LoadMetadataFromBin(const char *path, void **deviceAddr, aclTensor **tensor)
{
    FILE *f = fopen(path, "rb");
    if (!f) { LOG_PRINT("[ERROR] Cannot open metadata bin: %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    if (fsize <= 0 || fsize % (long)sizeof(int32_t) != 0)
        { LOG_PRINT("[ERROR] Invalid metadata file size: %ld\n", fsize); fclose(f); return -1; }
    int64_t nElems = (int64_t)(fsize / sizeof(int32_t));
    std::vector<int32_t> host(nElems);
    fread(host.data(), sizeof(int32_t), (size_t)nElems, f);
    fclose(f);
    size_t byteSize = (size_t)nElems * sizeof(int32_t);
    int ret = aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtMalloc metadata failed. ERROR: %d\n", ret); return ret; }
    ret = aclrtMemcpy(*deviceAddr, byteSize, host.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtMemcpy metadata failed. ERROR: %d\n", ret); return ret; }
    int64_t shape[1]   = {nElems};
    int64_t strides[1] = {1};
    *tensor = aclCreateTensor(shape, 1, aclDataType::ACL_INT32, strides, 0,
                              aclFormat::ACL_FORMAT_ND, shape, 1, *deviceAddr);
    const uint32_t *m = reinterpret_cast<const uint32_t *>(host.data());
    LOG_PRINT("[INFO] Loaded metadata: %s  (%ld int32)  sectionNum=%u mBase=%u s2Base=%u\n",
              path, nElems, nElems >= 1 ? m[0] : 0u, nElems >= 2 ? m[1] : 0u, nElems >= 3 ? m[2] : 0u);
    return ACL_SUCCESS;
}

// 从 host int32 数组创建 device int32 tensor
static int CreateInt32Tensor(const int32_t *hostData, int64_t n,
                             void **deviceAddr, aclTensor **tensor)
{
    size_t byteSize = (size_t)n * sizeof(int32_t);
    int ret = aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtMalloc int32 failed. ERROR: %d\n", ret); return ret; }
    ret = aclrtMemcpy(*deviceAddr, byteSize, hostData, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtMemcpy int32 failed. ERROR: %d\n", ret); return ret; }
    int64_t shape[1]   = {n};
    int64_t strides[1] = {1};
    *tensor = aclCreateTensor(shape, 1, aclDataType::ACL_INT32, strides, 0,
                              aclFormat::ACL_FORMAT_ND, shape, 1, *deviceAddr);
    return ACL_SUCCESS;
}

} // namespace

// ─── 测试参数结构（对标 test_case.py 各字段）────────────────────────────────
struct TestCase {
    const char *name;
    int64_t B;         // batch_size
    int64_t N_q;       // query head 数（N1）
    int64_t N_kv;      // kv head 数（N2，支持 GQA/MQA，=N_q 为 MHA）
    int64_t S_q;       // query 序列长度（S1）；S_q=1 触发 FlashDecode 模式
    int64_t S_kv;      // key/value 序列长度（S2）
    int64_t D;         // head dim
    int64_t maskMode;  // 0=全量注意力  1=因果下三角(RIGHT_DOWN_CAUSAL)
    int64_t winLeft;   // 对应 pre_tokens；-1 表示无限制（INT_MAX）
    int64_t winRight;  // 对应 next_tokens；-1 表示无限制（INT_MAX）
    float qMin, qMax;  // Q tensor 均匀随机值域
    float kMin, kMax;  // K tensor 均匀随机值域
    float vMin, vMax;  // V tensor 均匀随机值域
};

// ─── 单用例运行（精度比较阈值对标 pytests RATIO_THRESHOLD/FAIL_RATIO_LIMIT）──
static constexpr float RATIO_THRESHOLD = 0.005f;   // 单元素相对误差阈值 0.5%
static constexpr float FAIL_RATIO_LIMIT = 0.005f;  // 超阈值元素占比上限 0.5%

static bool RunTestCase(const TestCase &tc, aclrtStream stream,
                        bool dumpOutput, bool verbose, uint32_t seed)
{
    const char *SEP64 = "──────────────────────────────────────────────────────────────────";

    LOG_PRINT("\n┌%s┐\n", SEP64);
    LOG_PRINT("│  Case: %-10s  B=%ld N_q=%ld N_kv=%ld S_q=%ld S_kv=%ld D=%ld  "
              "mask=%ld\n",
              tc.name, tc.B, tc.N_q, tc.N_kv, tc.S_q, tc.S_kv, tc.D, tc.maskMode);
    LOG_PRINT("└%s┘\n", SEP64);

    int64_t B = tc.B, N_q = tc.N_q, N_kv = tc.N_kv;
    int64_t S_q = tc.S_q, S_kv = tc.S_kv, D = tc.D;

    // ── 张量形状 ────────────────────────────────────────────────────────────
    std::vector<int64_t> qShape   = {B, N_q,  S_q,  D};
    std::vector<int64_t> kShape   = {B, N_kv, S_kv, D};
    std::vector<int64_t> vShape   = {B, N_kv, S_kv, D};
    std::vector<int64_t> outShape = {B, N_q,  S_q,  D};
    int64_t qElem = GetShapeSize(qShape), kElem = GetShapeSize(kShape);
    int64_t outElem = GetShapeSize(outShape); (void)GetShapeSize(vShape);
    int64_t qRefElems  = N_q  * S_q  * D;   // batch 0 CPU参考
    int64_t kvRefElems = N_kv * S_kv * D;

    LOG_PRINT("  Tensors: Q[%ld,%ld,%ld,%ld]=%.2fMB  K/V[%ld,%ld,%ld,%ld]=%.2fMB\n",
              B,N_q,S_q,D, qElem*2.0/1e6, B,N_kv,S_kv,D, kElem*2.0/1e6);

    std::mt19937 rng(seed ? seed : static_cast<uint32_t>(std::random_device{}()));
    std::vector<float> qRef, kRef, vRef;

    // ── 分配设备张量 ────────────────────────────────────────────────────────
    void *qDev=nullptr, *kDev=nullptr, *vDev=nullptr, *outDev=nullptr;
    aclTensor *qT=nullptr, *kT=nullptr, *vT=nullptr, *outT=nullptr;

    auto cleanup = [&]() {
        if (qT)   aclDestroyTensor(qT);
        if (kT)   aclDestroyTensor(kT);
        if (vT)   aclDestroyTensor(vT);
        if (outT) aclDestroyTensor(outT);
        if (qDev)   aclrtFree(qDev);
        if (kDev)   aclrtFree(kDev);
        if (vDev)   aclrtFree(vDev);
        if (outDev) aclrtFree(outDev);
    };

    // Q
    {
        std::uniform_real_distribution<float> dist(tc.qMin, tc.qMax);
        std::vector<uint16_t> h(qElem);
        qRef.resize(qRefElems);
        for (int64_t i = 0; i < qElem; ++i) {
            uint16_t fp = float_to_fp16(dist(rng));
            h[i] = fp;
            if (i < qRefElems) qRef[i] = fp16_to_float(fp);
        }
        if (CreateAclTensor(h, qShape, &qDev, aclDataType::ACL_FLOAT16, &qT) != ACL_SUCCESS)
            { cleanup(); return false; }
    }
    // K
    if (CreateAclTensorRandom(kShape, &kDev, aclDataType::ACL_FLOAT16, &kT,
                              rng, tc.kMin, tc.kMax, &kRef, kvRefElems) != ACL_SUCCESS)
        { cleanup(); return false; }
    // V
    if (CreateAclTensorRandom(vShape, &vDev, aclDataType::ACL_FLOAT16, &vT,
                              rng, tc.vMin, tc.vMax, &vRef, kvRefElems) != ACL_SUCCESS)
        { cleanup(); return false; }
    // Out (清零)
    if (CreateAclTensorDeviceZero(outShape, &outDev, aclDataType::ACL_FLOAT16,
                                  sizeof(uint16_t), &outT) != ACL_SUCCESS)
        { cleanup(); return false; }

    double softmaxScale = 1.0 / std::sqrt((double)D);
    int64_t winLeft     = tc.winLeft;
    int64_t winRight    = tc.winRight;

    // ────────────────────────────────────────────────────────────────────────
    //  调用 flash_attn 算子（metadata 传 nullptr，由 tiling 自动计算）
    // ────────────────────────────────────────────────────────────────────────
    LOG_PRINT("\n  [A] 调用 flash_attn 算子...\n");
    {
        uint64_t wsSize = 0;
        aclOpExecutor *exec = nullptr;
        int ret = aclnnFlashAttnGetWorkspaceSize(
            qT, kT, vT,
            nullptr,    // blockTable
            nullptr,    // cuSeqlensQ
            nullptr,    // cuSeqlensKv
            nullptr,    // sequsedQ
            nullptr,    // sequsedKv
            nullptr,    // sinks
            nullptr,    // attnMask
            nullptr,    // metadata (nullptr → tiling 自动计算)
            softmaxScale,
            tc.maskMode, winLeft, winRight,
            S_q, S_kv,
            "BNSD", "BNSD", "BNSD",
<<<<<<< HEAD
            (int32_t)0, // returnSoftmaxLse
=======
            (int64_t)0, // returnSoftmaxLse
            (int64_t)0, // deterministic
>>>>>>> 94104abcd (性能优化)
            outT,
            nullptr,    // softmaxLse
            &wsSize, &exec);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnFlashAttnGetWorkspaceSize FAILED. ret=%d\n", ret);
            cleanup(); return false;
        }
        void *ws = nullptr;
        if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnFlashAttn(ws, wsSize, exec, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnFlashAttn FAILED. ret=%d\n", ret);
            if (ws) aclrtFree(ws);
            cleanup(); return false;
        }
        aclrtSynchronizeStream(stream);
        if (ws) aclrtFree(ws);
        LOG_PRINT("  flash_attn 执行完毕。\n");
    }

    // ── 回读 NPU 输出 ───────────────────────────────────────────────────────
    std::vector<uint16_t> outHost(outElem, 0);
    aclrtMemcpy(outHost.data(), (size_t)outElem * sizeof(uint16_t),
                outDev, (size_t)outElem * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);

    // ── CPU 参考计算（仅 batch 0）──────────────────────────────────────────
    LOG_PRINT("\n  [C] CPU 参考计算（batch 0, mask=%ld）...\n", tc.maskMode);
    std::vector<float> cpuOut(qRefElems, 0.0f);
    cpu_flash_attention(qRef, kRef, vRef, cpuOut, 1, N_q, N_kv, S_q, S_kv, D, tc.maskMode);

    // ── 精度对比（对标 check_result）──────────────────────────────────────
    double maxAbsErr=0, sumAbsErr=0, maxRelErr=0, sumRelErr=0;
    int64_t failCount = 0;
    for (int64_t i = 0; i < qRefElems; ++i) {
        float npu = fp16_to_float(outHost[i]);
        float cpu = cpuOut[i];
        float ae  = std::abs(npu - cpu);
        float ref = std::abs(cpu) + 1e-6f;
        float re  = ae / ref;
        // 动态阈值：max(ref * RATIO_THRESHOLD, 0.000025)
        float thr = std::max(ref * RATIO_THRESHOLD, 0.000025f);
        if (ae > maxAbsErr) maxAbsErr = ae;
        if (re > maxRelErr) maxRelErr = re;
        sumAbsErr += ae;
        sumRelErr += re;
        if (ae > thr) ++failCount;
    }
    double meanAbsErr  = sumAbsErr / (double)qRefElems;
    double meanRelErr  = sumRelErr / (double)qRefElems;
    double failRatio   = (double)failCount / (double)qRefElems;
    bool   passed      = (failRatio <= (double)FAIL_RATIO_LIMIT);

    LOG_PRINT("\n┌%s┐\n", SEP64);
    LOG_PRINT("│  精度报告: %-10s  (batch 0, %ld 元素)\n", tc.name, qRefElems);
    LOG_PRINT("├%s┤\n", SEP64);
    LOG_PRINT("│  Shape       : [%ld,%ld,%ld,%ld]\n", 1L, N_q, S_q, D);
    LOG_PRINT("│  MaxAbsErr   : %.8f\n", maxAbsErr);
    LOG_PRINT("│  MeanAbsErr  : %.8f\n", meanAbsErr);
    LOG_PRINT("│  MaxRelErr   : %.8f\n", maxRelErr);
    LOG_PRINT("│  MeanRelErr  : %.8f\n", meanRelErr);
    LOG_PRINT("│  FailElems   : %ld / %ld  (%.4f%%)\n",
              failCount, qRefElems, failRatio * 100.0);
    LOG_PRINT("│  Threshold   : elemDynamic  failRatio<=%.2f%%\n",
              FAIL_RATIO_LIMIT * 100.0f);
    LOG_PRINT("│  结论        : %s\n", passed ? "✓ PASS" : "✗ FAIL");

    // verbose: 打印所有超阈值元素
    if (failCount > 0) {
        LOG_PRINT("├%s┤\n", SEP64);
        int64_t printMax = verbose ? failCount : std::min(failCount, (int64_t)10);
        LOG_PRINT("│  %s%ld 个超阈值元素:\n", verbose ? "全部 " : "前", printMax);
        LOG_PRINT("│  %8s  %14s  %14s  %12s  %12s\n",
                  "idx", "NPU", "CPU", "absErr", "relErr");
        int64_t printed = 0;
        for (int64_t i = 0; i < qRefElems && printed < printMax; ++i) {
            float npu = fp16_to_float(outHost[i]);
            float cpu = cpuOut[i];
            float ae  = std::abs(npu - cpu);
            float ref = std::abs(cpu) + 1e-6f;
            float thr = std::max(ref * RATIO_THRESHOLD, 0.000025f);
            if (ae > thr) {
                LOG_PRINT("│  %8ld  %+14.8f  %+14.8f  %12.8f  %12.6f\n",
                          i, (double)npu, (double)cpu, (double)ae, (double)(ae/ref));
                ++printed;
            }
        }
    }
    LOG_PRINT("└%s┘\n", SEP64);

    // ── dump 选项：保存 txt ──────────────────────────────────────────────────
    if (dumpOutput) {
        char buf[256];
        // NPU output (batch 0)
        std::vector<float> npuFloat(qRefElems);
        for (int64_t i = 0; i < qRefElems; ++i) npuFloat[i] = fp16_to_float(outHost[i]);
        snprintf(buf, sizeof(buf), "fa_npu_out_%s.txt", tc.name);
        char hdr[256];
        snprintf(hdr, sizeof(hdr), "shape=[1,%ld,%ld,%ld]  case=%s  elems=%ld",
                 N_q, S_q, D, tc.name, qRefElems);
        SaveFloatToTxt(buf, npuFloat.data(), qRefElems, hdr);
        LOG_PRINT("  NPU output saved: %s\n", buf);

        snprintf(buf, sizeof(buf), "fa_cpu_ref_%s.txt", tc.name);
        SaveFloatToTxt(buf, cpuOut.data(), qRefElems, hdr);
        LOG_PRINT("  CPU ref  saved: %s\n", buf);

        // precision report
        snprintf(buf, sizeof(buf), "fa_prec_cmp_%s.txt", tc.name);
        FILE *f = fopen(buf, "w");
        if (f) {
            fprintf(f, "case=%s  B=%ld N_q=%ld N_kv=%ld S_q=%ld S_kv=%ld D=%ld mask=%ld\n",
                    tc.name, B, N_q, N_kv, S_q, S_kv, D, tc.maskMode);
            fprintf(f, "maxAbsErr=%.8f  meanAbsErr=%.8f  maxRelErr=%.8f  "
                    "failElems=%ld/%ld  %s\n",
                    maxAbsErr, meanAbsErr, maxRelErr,
                    failCount, qRefElems, passed ? "PASS" : "FAIL");
            fprintf(f, "%-8s  %-14s  %-14s  %-14s  %-12s  %s\n",
                    "idx", "npu", "cpu", "absErr", "relErr", "result");
            for (int64_t i = 0; i < qRefElems; ++i) {
                float npu = fp16_to_float(outHost[i]);
                float ae  = std::abs(npu - cpuOut[i]);
                float ref = std::abs(cpuOut[i]) + 1e-6f;
                float thr = std::max(ref * RATIO_THRESHOLD, 0.000025f);
                fprintf(f, "%-8ld  %+.8f    %+.8f    %.8f    %.6f    %s\n",
                        i, (double)npu, (double)cpuOut[i], (double)ae,
                        (double)(ae/ref), ae > thr ? "FAIL" : "pass");
            }
            fclose(f);
            LOG_PRINT("  Report saved: %s\n", buf);
        }
    }

    cleanup();
    return passed;
}

// ─── TND_MIXED_02 专用测试函数（直接读取 metadata.bin，跳过 metadata 算子）───
static bool RunTndMixed02(aclrtStream stream, const char *metaBinPath,
                          bool dumpOutput, bool verbose, uint32_t seed)
{
    // test_case.py TND_MIXED_02 硬编码参数
    static const int64_t  B       = 8;
    static const int64_t  N_q    = 4;
    static const int64_t  N_kv   = 4;
    static const int64_t  D      = 128;
    static const int64_t  T_q    = 944;   // sum(cu_seqlens_q)
    static const int64_t  T_kv   = 944;
    static const int64_t  maskMode = 3;
    static const int64_t  winLeft  = -1;  // sparse_mode=3 → pre_tokens=-1
    static const int64_t  winRight = -1;
    static const float    qMin = -10.0f, qMax = 10.0f;
    static const float    kMin = -10.0f, kMax = 10.0f;
    static const float    vMin = -10.0f, vMax = 10.0f;

    static const int32_t CU_SEQLENS_Q[9]  = {0,118,236,354,472,590,708,826,944};
    static const int32_t CU_SEQLENS_KV[9] = {0,118,236,354,472,590,708,826,944};
    static const int32_t SEQUSED_Q[8]     = {118,118,118,118,118,118,118,118};
    static const int32_t SEQUSED_KV[8]    = {118,118,118,118,118,118,118,118};

    const int64_t S_q0  = SEQUSED_Q[0];   // 118
    const int64_t S_kv0 = SEQUSED_KV[0];  // 118
    const char *SEP = "----------------------------------------------------------------";

    LOG_PRINT("\n+%s+\n", SEP);
    LOG_PRINT("| TND_MIXED_02: B=%ld N_q=%ld N_kv=%ld T=%ld D=%ld mask=%ld\n",
              B, N_q, N_kv, T_q, D, maskMode);
    LOG_PRINT("+%s+\n", SEP);

    std::vector<int64_t> qShape   = {T_q,  N_q,  D};
    std::vector<int64_t> kShape   = {T_kv, N_kv, D};
    std::vector<int64_t> vShape   = {T_kv, N_kv, D};
    std::vector<int64_t> outShape = {T_q,  N_q,  D};
    int64_t qElem  = T_q  * N_q  * D;
    int64_t kvElem = T_kv * N_kv * D;
    int64_t outElem = qElem;
    int64_t qRef0  = S_q0  * N_q  * D;  // batch 0 comparison elements

    std::mt19937 rng(seed);

    void *qDev=nullptr, *kDev=nullptr, *vDev=nullptr, *outDev=nullptr;
    void *cuQDev=nullptr, *cuKvDev=nullptr, *seqQDev=nullptr, *seqKvDev=nullptr;
    void *metaDev=nullptr, *maskDev=nullptr;
    aclTensor *qT=nullptr, *kT=nullptr, *vT=nullptr, *outT=nullptr;
    aclTensor *cuQT=nullptr, *cuKvT=nullptr, *seqQT=nullptr, *seqKvT=nullptr;
    aclTensor *metaT=nullptr, *maskT=nullptr;

    auto cleanup = [&]() {
        aclDestroyTensor(qT);  aclDestroyTensor(kT);  aclDestroyTensor(vT);
        aclDestroyTensor(outT); aclDestroyTensor(cuQT); aclDestroyTensor(cuKvT);
        aclDestroyTensor(seqQT); aclDestroyTensor(seqKvT);
        aclDestroyTensor(metaT); aclDestroyTensor(maskT);
        aclrtFree(qDev); aclrtFree(kDev); aclrtFree(vDev); aclrtFree(outDev);
        aclrtFree(cuQDev); aclrtFree(cuKvDev); aclrtFree(seqQDev); aclrtFree(seqKvDev);
        aclrtFree(metaDev); aclrtFree(maskDev);
    };

    // ── 创建 Q/K/V，同时提取 batch 0 数据供 CPU 参考 ────────────────────────
    // TND 格式 [T, N, D]：token t, head n, dim d → index t*N*D + n*D + d
    // batch 0 tokens: t in [CU_SEQLENS_Q[0], CU_SEQLENS_Q[0]+S_q0)
    // CPU ref 使用 BNSD-like [N_q, S_q0, D]：index n*S_q0*D + s*D + d
    std::vector<float> qRef, kRef, vRef;
    (void)dumpOutput;  // dump not implemented for TND variant
    auto extractBatch0 = [&](const std::vector<uint16_t> &h, int64_t N,
                             int32_t cu_start, int64_t seq_len, std::vector<float> &ref) {
        ref.resize(seq_len * N * D);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t s = 0; s < seq_len; ++s)
                for (int64_t d = 0; d < D; ++d) {
                    int64_t tndIdx = ((int64_t)cu_start + s) * N * D + n * D + d;
                    ref[n * seq_len * D + s * D + d] = fp16_to_float(h[tndIdx]);
                }
    };
    {
        std::uniform_real_distribution<float> dist(qMin, qMax);
        std::vector<uint16_t> h(qElem);
        for (auto &v : h) v = float_to_fp16(dist(rng));
        extractBatch0(h, N_q, CU_SEQLENS_Q[0], S_q0, qRef);
        if (CreateAclTensor(h, qShape, &qDev, aclDataType::ACL_FLOAT16, &qT) != ACL_SUCCESS)
            { cleanup(); return false; }
    }
    {
        std::uniform_real_distribution<float> dist(kMin, kMax);
        std::vector<uint16_t> h(kvElem);
        for (auto &v : h) v = float_to_fp16(dist(rng));
        extractBatch0(h, N_kv, CU_SEQLENS_KV[0], S_kv0, kRef);
        if (CreateAclTensor(h, kShape, &kDev, aclDataType::ACL_FLOAT16, &kT) != ACL_SUCCESS)
            { cleanup(); return false; }
    }
    {
        std::uniform_real_distribution<float> dist(vMin, vMax);
        std::vector<uint16_t> h(kvElem);
        for (auto &v : h) v = float_to_fp16(dist(rng));
        extractBatch0(h, N_kv, CU_SEQLENS_KV[0], S_kv0, vRef);
        if (CreateAclTensor(h, vShape, &vDev, aclDataType::ACL_FLOAT16, &vT) != ACL_SUCCESS)
            { cleanup(); return false; }
    }
    if (CreateAclTensorDeviceZero(outShape, &outDev, aclDataType::ACL_FLOAT16,
                                  sizeof(uint16_t), &outT) != ACL_SUCCESS)
        { cleanup(); return false; }

    // ── 创建 cu_seqlens / seqused 张量 ───────────────────────────────────────
    if (CreateInt32Tensor(CU_SEQLENS_Q,  B+1, &cuQDev,  &cuQT)  != ACL_SUCCESS) { cleanup(); return false; }
    if (CreateInt32Tensor(CU_SEQLENS_KV, B+1, &cuKvDev, &cuKvT) != ACL_SUCCESS) { cleanup(); return false; }
    if (CreateInt32Tensor(SEQUSED_Q,     B,   &seqQDev, &seqQT) != ACL_SUCCESS) { cleanup(); return false; }
    if (CreateInt32Tensor(SEQUSED_KV,    B,   &seqKvDev,&seqKvT)!= ACL_SUCCESS) { cleanup(); return false; }

    // ── 从 .bin 文件加载 metadata ─────────────────────────────────────────────
    if (LoadMetadataFromBin(metaBinPath, &metaDev, &metaT) != ACL_SUCCESS)
        { cleanup(); return false; }

    // ── 构造因果 attn_mask [2048, 2048] INT8（mask_mode=3 必须提供）─────────
    // mask[i][j] = 1(屏蔽) if j > i（上三角），0(允许) if j <= i（下三角）
    {
        constexpr int64_t MSZ = 2048;
        std::vector<int8_t> hMask((size_t)(MSZ * MSZ), 0);
        for (int64_t i = 0; i < MSZ; ++i)
            for (int64_t j = i + 1; j < MSZ; ++j)
                hMask[i * MSZ + j] = 1;
        size_t byteSize = (size_t)(MSZ * MSZ);
        if (aclrtMalloc(&maskDev, byteSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMemcpy(maskDev, byteSize, hMask.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
            { cleanup(); return false; }
        int64_t mShape[2]   = {MSZ, MSZ};
        int64_t mStrides[2] = {MSZ, 1};
        maskT = aclCreateTensor(mShape, 2, aclDataType::ACL_INT8, mStrides, 0,
                                aclFormat::ACL_FORMAT_ND, mShape, 2, maskDev);
    }

    // ── 调用 flash_attn（TND 布局，maxSeqlen=-1 由 cu_seqlens 决定）──────────
    LOG_PRINT("\n  [A] Calling aclnnFlashAttn (TND, metadata from file)...\n");
    double softmaxScale = 1.0 / std::sqrt((double)D);
    {
        uint64_t wsSize = 0;
        aclOpExecutor *exec = nullptr;
        int ret = aclnnFlashAttnGetWorkspaceSize(
            qT, kT, vT,
            nullptr,  // blockTable
            cuQT,     // cuSeqlensQ
            cuKvT,    // cuSeqlensKv
            seqQT,    // sequsedQ
            seqKvT,   // sequsedKv
            nullptr,  // sinks
            maskT,    // attnMask: causal mask [2048,2048] INT8 (mask_mode=3 required)
            metaT,    // metadata from .bin
            softmaxScale,
            maskMode, winLeft, winRight,
            (int64_t)-1, (int64_t)-1,  // maxSeqlenQ/KV=-1 for TND (kernel uses cu_seqlens)
            "TND", "TND", "TND",
            (int64_t)0,  // returnSoftmaxLse
            (int64_t)0,  // deterministic
            outT,
            nullptr,  // softmaxLse
            &wsSize, &exec);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnFlashAttnGetWorkspaceSize FAILED. ret=%d\n", ret);
            cleanup(); return false;
        }
        void *ws = nullptr;
        if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnFlashAttn(ws, wsSize, exec, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnFlashAttn FAILED. ret=%d\n", ret);
            if (ws) aclrtFree(ws);
            cleanup(); return false;
        }
        aclrtSynchronizeStream(stream);
        if (ws) aclrtFree(ws);
        LOG_PRINT("  flash_attn done.\n");
    }

    // ── 回读 NPU 输出 ──────────────────────────────────────────────────────────
    std::vector<uint16_t> outHost(outElem, 0);
    aclrtMemcpy(outHost.data(), (size_t)outElem * sizeof(uint16_t),
                outDev, (size_t)outElem * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);

    // ── CPU 参考（batch 0，因果注意力：mask_mode=3 = CAUSAL）──────────────────
    LOG_PRINT("\n  [B] CPU reference (batch 0: S_q=%ld S_kv=%ld, causal)...\n", S_q0, S_kv0);
    std::vector<float> cpuOut(qRef0, 0.0f);
    cpu_flash_attention(qRef, kRef, vRef, cpuOut, 1, N_q, N_kv, S_q0, S_kv0, D, /*maskMode=*/1);

    // ── 精度对比（batch 0）───────────────────────────────────────────────────
    // NPU 输出 TND [T_q, N_q, D]，batch 0: token [CU_SEQLENS_Q[0], +S_q0)
    double maxAbsErr=0, sumAbsErr=0, maxRelErr=0, sumRelErr=0;
    int64_t failCount=0;
    for (int64_t n = 0; n < N_q; ++n) {
        for (int64_t s = 0; s < S_q0; ++s) {
            for (int64_t d = 0; d < D; ++d) {
                int64_t tndIdx = ((int64_t)CU_SEQLENS_Q[0] + s) * N_q * D + n * D + d;
                int64_t refIdx = n * S_q0 * D + s * D + d;
                float npu = fp16_to_float(outHost[tndIdx]);
                float cpu = cpuOut[refIdx];
                float ae  = std::abs(npu - cpu);
                float ref = std::abs(cpu) + 1e-6f;
                float thr = std::max(ref * RATIO_THRESHOLD, 0.000025f);
                if (ae > maxAbsErr) maxAbsErr = ae;
                if ((double)(ae/ref) > maxRelErr) maxRelErr = ae/ref;
                sumAbsErr += ae;
                sumRelErr += ae/ref;
                if (ae > thr) ++failCount;
            }
        }
    }
    double meanAbsErr = sumAbsErr / qRef0;
    double failRatio  = (double)failCount / qRef0;
    bool   passed     = (failRatio <= (double)FAIL_RATIO_LIMIT);

    LOG_PRINT("\n+%s+\n", SEP);
    LOG_PRINT("| Accuracy: TND_MIXED_02 (batch 0, %ld elements)\n", qRef0);
    LOG_PRINT("|  MaxAbsErr : %.8f\n", maxAbsErr);
    LOG_PRINT("|  MeanAbsErr: %.8f\n", meanAbsErr);
    LOG_PRINT("|  MaxRelErr : %.8f\n", maxRelErr);
    LOG_PRINT("|  FailElems : %ld / %ld  (%.4f%%)\n",
              failCount, qRef0, failRatio * 100.0);
    LOG_PRINT("|  Result    : %s\n", passed ? "PASS" : "FAIL");
    if (failCount > 0) {
        int64_t printMax = verbose ? failCount : std::min(failCount, (int64_t)10);
        int64_t printed  = 0;
        for (int64_t n = 0; n < N_q && printed < printMax; ++n)
            for (int64_t s = 0; s < S_q0 && printed < printMax; ++s)
                for (int64_t d = 0; d < D && printed < printMax; ++d) {
                    int64_t ti = ((int64_t)CU_SEQLENS_Q[0]+s)*N_q*D + n*D + d;
                    int64_t ri = n*S_q0*D + s*D + d;
                    float npu = fp16_to_float(outHost[ti]);
                    float ae  = std::abs(npu - cpuOut[ri]);
                    float thr = std::max((std::abs(cpuOut[ri])+1e-6f)*RATIO_THRESHOLD, 0.000025f);
                    if (ae > thr) {
                        LOG_PRINT("|  [n=%ld s=%ld d=%ld] npu=%.6f cpu=%.6f ae=%.6f\n",
                                  n, s, d, (double)npu, (double)cpuOut[ri], (double)ae);
                        ++printed;
                    }
                }
    }
    LOG_PRINT("+%s+\n\n", SEP);

    cleanup();
    return passed;
}

// ─── 打印用法帮助 ────────────────────────────────────────────────────────────
static void PrintUsage(const char *prog)
{
    printf(
        "用法: %s [选项]\n"
        "\n"
        "形状参数（对标 test_case.py 各字段）:\n"
        "  --B     <int>   batch_size                        (默认 1)\n"
        "  --N1    <int>   query head 数                     (默认 1)\n"
        "  --N2    <int>   kv head 数；N2<N1 为 GQA/MQA     (默认 = N1)\n"
        "  --S1    <int>   query 序列长度；S1=1 触发 FD 模式 (默认 64)\n"
        "  --S2    <int>   key/value 序列长度                (默认 256)\n"
        "  --D     <int>   head_dim                          (默认 128)\n"
        "\n"
        "掩码参数:\n"
        "  --mask  <int>   mask_mode: 0=全量 1=因果下三角    (默认 1)\n"
        "  --wL    <int>   win_left / pre_tokens；-1=无限制  (默认 -1)\n"
        "  --wR    <int>   win_right / next_tokens；-1=无限制(默认 -1)\n"
        "\n"
        "QKV 值域参数（fp16 均匀随机分布）:\n"
        "  --qMin  <float> Q 下界  (默认  0.0)\n"
        "  --qMax  <float> Q 上界  (默认  1.0)\n"
        "  --kMin  <float> K 下界  (默认  0.0)\n"
        "  --kMax  <float> K 上界  (默认  1.0)\n"
        "  --vMin  <float> V 下界  (默认  0.0)\n"
        "  --vMax  <float> V 上界  (默认  1.0)\n"
        "\n"
        "其他选项:\n"
        "  --name    <str> 输出文件名前缀                    (默认 \"test\")\n"
        "  --seed    <int> 随机种子；0=每次随机              (默认 42)\n"
        "  --device  <int> NPU device id                     (默认 0)\n"
        "  --dump          保存 npu_out/cpu_ref/prec_cmp.txt\n"
        "  --verbose       打印全部超阈值点（而非仅前 10 个）\n"
        "  --help          打印此帮助\n"
        "\n"
        "示例:\n"
        "  %s                                    # 默认参数运行\n"
        "  %s --B 1 --N1 8 --N2 2 --S1 1 --S2 512 --D 128  # FD+GQA\n"
        "  %s --B 4 --N1 16 --N2 4 --S1 256 --S2 256 --D 128 --mask 1\n"
        "  %s --S1 64 --S2 256 --qMin 10 --qMax 10 --kMin 10 --kMax 10\n",
        prog, prog, prog, prog, prog);
}

// ─── 修改此路径以指向 pytest 生成的 metadata.bin ──────────────────────────────
#define METADATA_BIN_PATH  "./metadata_TND_MIXED_02.bin"

int main(int argc, char *argv[])
{
    (void)argc; (void)argv;
    const char *metaBinPath = METADATA_BIN_PATH;
    const uint32_t seed     = 42u;
    const int32_t  deviceId = 0;
    const bool     dumpOutput = false;
    const bool     verbose    = false;

    aclrtStream stream;
    if (Init(deviceId, &stream) != ACL_SUCCESS) { LOG_PRINT("Init ACL failed.\n"); return 1; }

    bool passed = false;
    try {
        passed = RunTndMixed02(stream, metaBinPath, dumpOutput, verbose, seed);
    } catch (const std::exception &e) {
        LOG_PRINT("[ERROR] Exception: %s\n", e.what());
    } catch (...) {
        LOG_PRINT("[ERROR] Unknown exception\n");
    }

    const char *SEP = "================================================================";
    LOG_PRINT("\n%s\n  Final Result: %s\n%s\n\n",
              SEP, passed ? "PASS" : "FAIL", SEP);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return passed ? 0 : 1;
}