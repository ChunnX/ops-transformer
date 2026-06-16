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
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <sys/wait.h>
#if defined(__linux__)
#include <unistd.h>
#endif
#include <string>
#include <sstream>
#include "acl/acl.h"
#include "../op_api/aclnn_flash_attn.h"

namespace {

// examples 目录下的默认文件名（与 test_aclnn_flash_attn.cpp 同级）
static const char *kDefaultMetadataPtName   = "metadata.pt";
static const char *kDefaultLoaderScriptName = "load_metadata_pt.py";

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

static uint16_t float_to_bf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static float bf16_to_float(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static float half_to_float(uint16_t h, aclDataType dt)
{
    return (dt == aclDataType::ACL_BF16) ? bf16_to_float(h) : fp16_to_float(h);
}

static uint16_t float_to_half(float f, aclDataType dt)
{
    return (dt == aclDataType::ACL_BF16) ? float_to_bf16(f) : float_to_fp16(f);
}

static bool IsBsndLayout(const char *layout)
{
    return layout && (std::strcmp(layout, "BSND") == 0 || std::strcmp(layout, "bsnd") == 0);
}

static bool IsTndLayout(const char *layout)
{
    return layout && (std::strcmp(layout, "TND") == 0 || std::strcmp(layout, "tnd") == 0);
}

static void GetMaxSeqlenAttrs(const char *layoutQ, const char *layoutKv,
                              int64_t S_q, int64_t S_kv,
                              int64_t &maxSeqlenQ, int64_t &maxSeqlenKv)
{
    // 与 pytests/core/data.py 一致：仅 TND layout 传实际 max_seqlen，BSND/BNSD 传 -1
    maxSeqlenQ  = IsTndLayout(layoutQ)  ? S_q  : -1;
    maxSeqlenKv = IsTndLayout(layoutKv) ? S_kv : -1;
}

static void BuildShapes(const char *layoutQ, const char *layoutOut,
                        int64_t B, int64_t N_q, int64_t N_kv,
                        int64_t S_q, int64_t S_kv, int64_t D,
                        std::vector<int64_t> &qShape,
                        std::vector<int64_t> &kShape,
                        std::vector<int64_t> &vShape,
                        std::vector<int64_t> &outShape)
{
    if (IsBsndLayout(layoutQ)) {
        qShape   = {B, S_q,  N_q,  D};
        kShape   = {B, S_kv, N_kv, D};
        vShape   = {B, S_kv, N_kv, D};
    } else {
        qShape   = {B, N_q,  S_q,  D};
        kShape   = {B, N_kv, S_kv, D};
        vShape   = {B, N_kv, S_kv, D};
    }
    if (IsBsndLayout(layoutOut)) {
        outShape = {B, S_q, N_q, D};
    } else {
        outShape = {B, N_q, S_q, D};
    }
}

// maskMode: 0=全量  1=RIGHT_DOWN_CAUSAL(下三角)  3=rightDownCausal(同 pytests)
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
    int64_t causalOffset = (maskMode == 1) ? (S_kv - S_q) : (S_kv - 1);
    std::vector<float> score(S_q * S_kv);
    for (int64_t b = 0; b < B; b++) {
        for (int64_t nq = 0; nq < N_q; nq++) {
            int64_t nkv = nq * N_kv / N_q;  // GQA head mapping
            for (int64_t s1 = 0; s1 < S_q; s1++) {
                for (int64_t s2 = 0; s2 < S_kv; s2++) {
                    if (maskMode == 1 && s2 > s1 + causalOffset) {
                        score[s1 * S_kv + s2] = -1e9f;
                        continue;
                    }
                    if (maskMode == 3 && s2 > s1 + S_kv - S_q) {
                        score[s1 * S_kv + s2] = -1e9f;
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
    const int64_t CHUNK = 1 << 20;
    int64_t total = GetShapeSize(shape);
    size_t elemSize = (dataType == aclDataType::ACL_BF16 ||
                       dataType == aclDataType::ACL_FLOAT16) ? 2U : 4U;
    auto ret = aclrtMalloc(deviceAddr, (size_t)total * elemSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { LOG_PRINT("aclrtMalloc random failed. ERROR: %d\n", ret); return ret; }
    if (floatRef && refElems > 0) floatRef->resize(refElems);
    std::uniform_real_distribution<float> dist(randMin, randMax);
    std::vector<uint16_t> chunk(std::min(CHUNK, total));
    int64_t offset = 0;
    while (offset < total) {
        int64_t count = std::min(CHUNK, total - offset);
        for (int64_t i = 0; i < count; ++i) {
            float v = dist(rng);
            uint16_t h = float_to_half(v, dataType);
            chunk[i] = h;
            if (floatRef && offset + i < refElems)
                (*floatRef)[offset + i] = half_to_float(h, dataType);
        }
        ret = aclrtMemcpy(reinterpret_cast<uint8_t*>(*deviceAddr) + offset * elemSize,
                          (size_t)count * elemSize, chunk.data(),
                          (size_t)count * elemSize, ACL_MEMCPY_HOST_TO_DEVICE);
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

static int CreateMetadataTensor(void **deviceAddr, aclTensor **tensor, int64_t numElems = METADATA_MAX_ELEMS)
{
    size_t size = (size_t)numElems * sizeof(uint32_t);
    int ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtMalloc metadata failed. ERROR: %d\n", ret); return ret; }
    ret = aclrtMemset(*deviceAddr, size, 0, size);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtMemset metadata failed. ERROR: %d\n", ret); return ret; }
    int64_t shape[1] = {numElems};
    int64_t strides[1] = {1};
    *tensor = aclCreateTensor(shape, 1, aclDataType::ACL_INT32, strides, 0,
                              aclFormat::ACL_FORMAT_ND, shape, 1, *deviceAddr);
    return ACL_SUCCESS;
}

static int UploadMetadataTensor(const std::vector<uint32_t> &hostMeta,
                                void **deviceAddr, aclTensor **tensor)
{
    if (hostMeta.empty()) {
        LOG_PRINT("[LoadMetadata] metadata 为空\n");
        return -1;
    }
    int64_t numElems = (int64_t)hostMeta.size();
    int ret = CreateMetadataTensor(deviceAddr, tensor, numElems);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtMemcpy(*deviceAddr, hostMeta.size() * sizeof(uint32_t),
                      hostMeta.data(), hostMeta.size() * sizeof(uint32_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[LoadMetadata] H2D memcpy failed. ERROR: %d\n", ret);
        return ret;
    }
    return ACL_SUCCESS;
}

static bool LoadMetadataRawBin(const std::string &path, std::vector<uint32_t> &out)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        LOG_PRINT("[LoadMetadata] 无法打开文件: %s\n", path.c_str());
        return false;
    }
    auto fileSize = ifs.tellg();
    if (fileSize <= 0 || (fileSize % (std::streamoff)sizeof(uint32_t)) != 0) {
        LOG_PRINT("[LoadMetadata] 非法 bin 大小 (%lld bytes): %s\n",
                  (long long)fileSize, path.c_str());
        return false;
    }
    out.resize((size_t)fileSize / sizeof(uint32_t));
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(out.data()), fileSize);
    return ifs.good();
}

static std::string ShellQuote(const std::string &s)
{
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static void PrintPythonLoadError(const std::string &errPath)
{
    std::ifstream err(errPath);
    if (!err.good()) return;
    std::string line;
    while (std::getline(err, line)) {
        LOG_PRINT("[LoadMetadata] python: %s\n", line.c_str());
    }
}

static std::string DirnameOfPath(const std::string &path)
{
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return "./";
    return path.substr(0, pos + 1);
}

static bool FileExists(const std::string &path)
{
    std::ifstream f(path);
    return f.good();
}

static std::string GetExecutableDir(const char *argv0)
{
#if defined(__linux__)
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        return DirnameOfPath(buf);
    }
#endif
    if (argv0 != nullptr && argv0[0] != '\0') {
        return DirnameOfPath(argv0);
    }
    return "./";
}

// 在 example 源码目录 / 可执行文件目录 / 当前工作目录 中查找 assets
static std::string ResolveExampleDir(const char *argv0)
{
    const std::string candidates[] = {
        DirnameOfPath(__FILE__),
        GetExecutableDir(argv0),
        std::string("./"),
    };
    for (const auto &dir : candidates) {
        if (FileExists(dir + kDefaultLoaderScriptName) || FileExists(dir + kDefaultMetadataPtName)) {
            return dir;
        }
    }
    return DirnameOfPath(__FILE__);
}

static std::string ResolveMetadataPtPath(const std::string &exampleDir)
{
    const char *names[] = {kDefaultMetadataPtName, "case_test_001_metadata.pt"};
    for (const char *name : names) {
        std::string path = exampleDir + name;
        if (FileExists(path)) return path;
    }
    return exampleDir + kDefaultMetadataPtName;
}

static std::string ResolveMetadataLoaderScript(const std::string &exampleDir)
{
    return exampleDir + kDefaultLoaderScriptName;
}

static bool LoadMetadataFromPt(const std::string &path, const std::string &loaderScript,
                               std::vector<uint32_t> &out)
{
    const std::string binPath = "/tmp/flash_attn_metadata.bin";
    const std::string errPath = "/tmp/flash_attn_metadata_pt_err.txt";
    if (!FileExists(loaderScript)) {
        LOG_PRINT("[LoadMetadata] 未找到脚本: %s\n", loaderScript.c_str());
        return false;
    }
    std::string cmd = std::string("python3 ") + ShellQuote(loaderScript) + " " +
                      ShellQuote(path) + " " + ShellQuote(binPath) + " 2>" + ShellQuote(errPath);
    LOG_PRINT("[LoadMetadata] 执行: python3 %s %s\n", loaderScript.c_str(), path.c_str());
    int status = system(cmd.c_str());
    bool exitOk = (status != -1) && WIFEXITED(status) && (WEXITSTATUS(status) == 0);
    if (!exitOk) {
        LOG_PRINT("[LoadMetadata] python 退出失败 (status=%d, script=%s)\n",
                  status, loaderScript.c_str());
        PrintPythonLoadError(errPath);
        out.clear();
        return false;
    }
    if (!LoadMetadataRawBin(binPath, out)) {
        PrintPythonLoadError(errPath);
        return false;
    }
    LOG_PRINT("[LoadMetadata] 从 .pt 加载 %zu 个 uint32 元素: %s\n", out.size(), path.c_str());
    return true;
}

static bool LoadMetadataFromFile(const std::string &path, const std::string &loaderScript,
                                 std::vector<uint32_t> &out)
{
    if (path.size() >= 3 && path.substr(path.size() - 3) == ".pt") {
        return LoadMetadataFromPt(path, loaderScript, out);
    }
    return LoadMetadataRawBin(path, out);
}

// mask_mode=3 需要 attn_mask: INT8 [2048,2048]，与 pytests 一致 triu(diagonal=1)
static constexpr int64_t CAUSAL_ATTN_MASK_SIZE = 2048;

static int CreateCausalAttnMask2048(void **deviceAddr, aclTensor **tensor)
{
    const int64_t M = CAUSAL_ATTN_MASK_SIZE;
    std::vector<int8_t> host(M * M, 0);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = i + 1; j < M; ++j) {
            host[i * M + j] = 1;
        }
    }
    std::vector<int64_t> shape = {M, M};
    return CreateAclTensor(host, shape, deviceAddr, aclDataType::ACL_INT8, tensor);
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

} // namespace

// ─── 测试参数结构（对标 test_case.py 各字段）────────────────────────────────
struct TestCase {
    const char *name;
    int64_t B;
    int64_t N_q;
    int64_t N_kv;
    int64_t S_q;
    int64_t S_kv;
    int64_t D;
    int64_t maskMode;
    int64_t winLeft;
    int64_t winRight;
    float qMin, qMax;
    float kMin, kMax;
    float vMin, vMax;
    const char *layoutQ;
    const char *layoutKv;
    const char *layoutOut;
    aclDataType dtype;
    bool skipCpuRef;
};

static bool RunTestCase(const TestCase &tc, aclrtStream stream,
                        bool dumpOutput, bool verbose, uint32_t seed,
                        const std::string &metaFilePath,
                        const std::string &loaderScript)
{
    static constexpr float RATIO_THRESHOLD = 0.005f;
    static constexpr float FAIL_RATIO_LIMIT = 0.005f;

    const char *SEP64 = "──────────────────────────────────────────────────────────────────";

    LOG_PRINT("\n┌%s┐\n", SEP64);
    LOG_PRINT("│  Case: %-14s  B=%ld N_q=%ld N_kv=%ld S_q=%ld S_kv=%ld D=%ld  "
              "mask=%ld\n",
              tc.name, tc.B, tc.N_q, tc.N_kv, tc.S_q, tc.S_kv, tc.D, tc.maskMode);
    LOG_PRINT("│  layout Q/Kv/Out = %s / %s / %s   dtype=%s\n",
              tc.layoutQ, tc.layoutKv, tc.layoutOut,
              tc.dtype == aclDataType::ACL_BF16 ? "bf16" : "fp16");
    LOG_PRINT("└%s┘\n", SEP64);

    int64_t B = tc.B, N_q = tc.N_q, N_kv = tc.N_kv;
    int64_t S_q = tc.S_q, S_kv = tc.S_kv, D = tc.D;
    size_t elemSize = 2;

    std::vector<int64_t> qShape, kShape, vShape, outShape;
    BuildShapes(tc.layoutQ, tc.layoutOut, B, N_q, N_kv, S_q, S_kv, D,
                qShape, kShape, vShape, outShape);
    int64_t qElem = GetShapeSize(qShape), kElem = GetShapeSize(kShape);
    int64_t outElem = GetShapeSize(outShape); (void)GetShapeSize(vShape);
    int64_t qRefElems  = N_q  * S_q  * D;   // batch 0 CPU参考
    int64_t kvRefElems = N_kv * S_kv * D;

    LOG_PRINT("  Tensors: Q shape=[%ld,%ld,%ld,%ld] (%.2fMB)  K/V shape=[%ld,%ld,%ld,%ld] (%.2fMB)\n",
              qShape[0], qShape[1], qShape[2], qShape[3], qElem * (double)elemSize / 1e6,
              kShape[0], kShape[1], kShape[2], kShape[3], kElem * (double)elemSize / 1e6);

    std::mt19937 rng(seed ? seed : static_cast<uint32_t>(std::random_device{}()));
    std::vector<float> qRef, kRef, vRef;

    // ── 分配设备张量 ────────────────────────────────────────────────────────
    void *qDev=nullptr, *kDev=nullptr, *vDev=nullptr, *outDev=nullptr;
    void *metaDev = nullptr;
    void *maskDev = nullptr;
    aclTensor *qT=nullptr, *kT=nullptr, *vT=nullptr, *outT=nullptr, *metaT=nullptr;
    aclTensor *maskT = nullptr;

    auto cleanup = [&]() {
        if (qT)    aclDestroyTensor(qT);
        if (kT)    aclDestroyTensor(kT);
        if (vT)    aclDestroyTensor(vT);
        if (outT)  aclDestroyTensor(outT);
        if (metaT) aclDestroyTensor(metaT);
        if (maskT) aclDestroyTensor(maskT);
        if (qDev)    aclrtFree(qDev);
        if (kDev)    aclrtFree(kDev);
        if (vDev)    aclrtFree(vDev);
        if (outDev)  aclrtFree(outDev);
        if (metaDev) aclrtFree(metaDev);
        if (maskDev) aclrtFree(maskDev);
    };

    // Q
    {
        std::uniform_real_distribution<float> dist(tc.qMin, tc.qMax);
        std::vector<uint16_t> h(qElem);
        qRef.resize(qRefElems);
        for (int64_t i = 0; i < qElem; ++i) {
            uint16_t fp = float_to_half(dist(rng), tc.dtype);
            h[i] = fp;
            if (i < qRefElems) qRef[i] = half_to_float(fp, tc.dtype);
        }
        if (CreateAclTensor(h, qShape, &qDev, tc.dtype, &qT) != ACL_SUCCESS)
            { cleanup(); return false; }
    }
    // K
    if (CreateAclTensorRandom(kShape, &kDev, tc.dtype, &kT,
                              rng, tc.kMin, tc.kMax, &kRef, kvRefElems) != ACL_SUCCESS)
        { cleanup(); return false; }
    // V
    if (CreateAclTensorRandom(vShape, &vDev, tc.dtype, &vT,
                              rng, tc.vMin, tc.vMax, &vRef, kvRefElems) != ACL_SUCCESS)
        { cleanup(); return false; }
    // Out (清零)
    if (CreateAclTensorDeviceZero(outShape, &outDev, tc.dtype, elemSize, &outT) != ACL_SUCCESS)
        { cleanup(); return false; }

    std::vector<uint32_t> metaHost;
    LOG_PRINT("\n  [A] 从 .pt 加载 metadata: %s\n", metaFilePath.c_str());
    if (!LoadMetadataFromFile(metaFilePath, loaderScript, metaHost)) {
        cleanup(); return false;
    }
    if (UploadMetadataTensor(metaHost, &metaDev, &metaT) != ACL_SUCCESS) {
        cleanup(); return false;
    }

    // mask_mode=3 (causal) 必须提供 attn_mask [2048,2048] INT8
    if (tc.maskMode == 3) {
        if (CreateCausalAttnMask2048(&maskDev, &maskT) != ACL_SUCCESS) {
            cleanup(); return false;
        }
        LOG_PRINT("  attn_mask: INT8 [%ld,%ld] (causal / mask_mode=3)\n",
                  CAUSAL_ATTN_MASK_SIZE, CAUSAL_ATTN_MASK_SIZE);
    }

    double softmaxScale = 1.0 / std::sqrt((double)D);
    int64_t winLeft     = tc.winLeft;
    int64_t winRight    = tc.winRight;
    if (tc.maskMode == 0 || tc.maskMode == 3) {
        winLeft = -1;
        winRight = -1;
    }

    int64_t metaPrintElems = (int64_t)metaHost.size();
    LOG_PRINT("\n  ▼ metadata（sectionNum=%u  mBase=%u  s2Base=%u）\n",
              metaHost[0], metaHost[1], metaHost[2]);
    PrintMetadata(metaHost.data(), metaPrintElems);

    // ────────────────────────────────────────────────────────────────────────
    //  步骤 B：调用 flash_attn 算子（传入 metadata）
    // ────────────────────────────────────────────────────────────────────────
    LOG_PRINT("  [B] 调用 flash_attn 算子（使用 .pt metadata）...\n");
    {
        int64_t maxSeqlenQ = -1, maxSeqlenKv = -1;
        GetMaxSeqlenAttrs(tc.layoutQ, tc.layoutKv, S_q, S_kv, maxSeqlenQ, maxSeqlenKv);
        uint64_t wsSize = 0;
        aclOpExecutor *exec = nullptr;
        int ret = aclnnFlashAttnGetWorkspaceSize(
            qT, kT, vT,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, maskT,
            metaT,
            softmaxScale,
            (int32_t)tc.maskMode, (int32_t)winLeft, (int32_t)winRight,
            maxSeqlenQ, maxSeqlenKv,
            tc.layoutQ, tc.layoutKv, tc.layoutOut,
            (int32_t)0,
            outT, nullptr,
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
    aclrtMemcpy(outHost.data(), (size_t)outElem * elemSize,
                outDev, (size_t)outElem * elemSize, ACL_MEMCPY_DEVICE_TO_HOST);

    if (tc.skipCpuRef) {
        LOG_PRINT("\n  [C] 跳过 CPU 参考（大 shape 或 --skipCpuRef）\n");
        cleanup();
        return true;
    }

    // ── CPU 参考计算（仅 batch 0）──────────────────────────────────────────
    LOG_PRINT("\n  [C] CPU 参考计算（batch 0, mask=%ld）...\n", tc.maskMode);
    std::vector<float> cpuOut(qRefElems, 0.0f);
    cpu_flash_attention(qRef, kRef, vRef, cpuOut, 1, N_q, N_kv, S_q, S_kv, D, tc.maskMode);

    // ── 精度对比（对标 check_result）──────────────────────────────────────
    double maxAbsErr=0, sumAbsErr=0, maxRelErr=0, sumRelErr=0;
    int64_t failCount = 0;
    for (int64_t i = 0; i < qRefElems; ++i) {
        float npu = half_to_float(outHost[i], tc.dtype);
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
            float npu = half_to_float(outHost[i], tc.dtype);
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
        for (int64_t i = 0; i < qRefElems; ++i) npuFloat[i] = half_to_float(outHost[i], tc.dtype);
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
                float npu = half_to_float(outHost[i], tc.dtype);
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

// ─── 打印用法帮助 ────────────────────────────────────────────────────────────
static void PrintUsage(const char *prog)
{
    printf(
        "用法: %s [选项]\n"
        "\n"
        "形状参数（对标 test_case.py 各字段）:\n"
        "  --B     <int>   batch_size                        (默认 1)\n"
        "  --N1    <int>   query head 数                     (默认 32)\n"
        "  --N2    <int>   kv head 数；N2<N1 为 GQA/MQA     (默认 2)\n"
        "  --S1    <int>   query 序列长度                    (默认 8192)\n"
        "  --S2    <int>   key/value 序列长度                (默认 8192)\n"
        "  --D     <int>   head_dim                          (默认 128)\n"
        "\n"
        "掩码参数:\n"
        "  --mask  <int>   mask_mode: 0=全量 3=rightDownCausal (默认 3)\n"
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
        "布局 / 数据类型:\n"
        "  --layoutQ   <str>  query layout: BNSD/BSND       (默认 BSND)\n"
        "  --layoutKv  <str>  kv layout                     (默认 BSND)\n"
        "  --layoutOut <str>  output layout                 (默认 BSND)\n"
        "  --dtype     <str>  fp16 / bf16                   (默认 bf16)\n"
        "\n"
        "  --metaPt  <path> 指定 metadata.pt（默认自动查找 example 同级目录）\n"
        "\n"
        "其他选项:\n"
        "  --name    <str>  case 名称                       (默认 case_test_001)\n"
        "  --seed    <int>  随机种子                        (默认 42)\n"
        "  --device  <int>  NPU device id                   (默认 0)\n"
        "  --skipCpuRef     跳过 CPU 参考（大 shape 默认开启）\n"
        "  --cpuRef         强制开启 CPU 参考\n"
        "  --dump           保存 npu_out/cpu_ref/prec_cmp.txt\n"
        "  --verbose        打印全部超阈值点\n"
        "  --help           打印此帮助\n"
        "\n"
        "说明: 默认在 example 同级目录查找 load_metadata_pt.py / metadata.pt\n"
        "示例: %s --device 0\n",
        prog, prog);
}

int main(int argc, char *argv[])
{
    std::string exampleDir = ResolveExampleDir(argv[0]);
    std::string metaPtPath = ResolveMetadataPtPath(exampleDir);
    std::string loaderScript = ResolveMetadataLoaderScript(exampleDir);

    // ─── 默认参数：case_test_001 ───────────────────────────────────────────
    std::string caseName = "case_test_001";
    int64_t B        = 1;
    int64_t N_q      = 32;
    int64_t N_kv     = 2;
    int64_t S_q      = 8192;
    int64_t S_kv     = 8192;
    int64_t D        = 128;
    int64_t maskMode = 3;
    int64_t winLeft  = -1;
    int64_t winRight = -1;
    float   qMin = -10.0f, qMax = 10.0f;
    float   kMin = -10.0f, kMax = 10.0f;
    float   vMin = -10.0f, vMax = 10.0f;
    std::string layoutQ   = "BSND";
    std::string layoutKv  = "BSND";
    std::string layoutOut = "BSND";
    std::string dtypeStr  = "bf16";
    uint32_t seed    = 42u;
    int32_t deviceId = 0;
    bool dumpOutput  = false;
    bool verbose     = false;
    bool skipCpuRef  = true;   // S1*S2 较大时默认跳过
    bool forceCpuRef = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nextStr = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if      (a == "--help")      { PrintUsage(argv[0]); return 0; }
        else if (a == "--dump")      { dumpOutput = true; }
        else if (a == "--verbose")   { verbose    = true; }
        else if (a == "--skipCpuRef"){ skipCpuRef = true; }
        else if (a == "--cpuRef")    { forceCpuRef = true; skipCpuRef = false; }
        else if (a == "--name")      { caseName   = nextStr(); }
        else if (a == "--seed")      { seed       = (uint32_t)std::stoul(nextStr()); }
        else if (a == "--device")    { deviceId   = (int32_t) std::stoi(nextStr()); }
        else if (a == "--metaPt")    { metaPtPath = nextStr(); }
        else if (a == "--layoutQ")   { layoutQ    = nextStr(); }
        else if (a == "--layoutKv")  { layoutKv   = nextStr(); }
        else if (a == "--layoutOut") { layoutOut  = nextStr(); }
        else if (a == "--dtype")     { dtypeStr   = nextStr(); }
        else if (a == "--B")         { B          = std::stoll(nextStr()); }
        else if (a == "--N1")        { N_q        = std::stoll(nextStr()); }
        else if (a == "--N2")        { N_kv       = std::stoll(nextStr()); }
        else if (a == "--S1")        { S_q        = std::stoll(nextStr()); }
        else if (a == "--S2")        { S_kv       = std::stoll(nextStr()); }
        else if (a == "--D")         { D          = std::stoll(nextStr()); }
        else if (a == "--mask")      { maskMode   = std::stoll(nextStr()); }
        else if (a == "--wL")        { winLeft    = std::stoll(nextStr()); }
        else if (a == "--wR")        { winRight   = std::stoll(nextStr()); }
        else if (a == "--qMin")      { qMin       = std::stof(nextStr()); }
        else if (a == "--qMax")      { qMax       = std::stof(nextStr()); }
        else if (a == "--kMin")      { kMin       = std::stof(nextStr()); }
        else if (a == "--kMax")      { kMax       = std::stof(nextStr()); }
        else if (a == "--vMin")      { vMin       = std::stof(nextStr()); }
        else if (a == "--vMax")      { vMax       = std::stof(nextStr()); }
        else { LOG_PRINT("[WARN] 未知参数: %s，已忽略（--help 查看用法）\n", a.c_str()); }
    }
    if (N_kv < 0) N_kv = N_q;
    if (!forceCpuRef && S_q * S_kv > (1LL << 20)) skipCpuRef = true;

    // loader 优先与 metadata.pt 同目录，否则用 example 目录
    {
        std::string besidePt = DirnameOfPath(metaPtPath) + kDefaultLoaderScriptName;
        if (FileExists(besidePt)) {
            loaderScript = besidePt;
        } else if (!FileExists(loaderScript)) {
            loaderScript = exampleDir + kDefaultLoaderScriptName;
        }
    }

    aclDataType dtype = aclDataType::ACL_FLOAT16;
    if (dtypeStr == "bf16" || dtypeStr == "BF16") {
        dtype = aclDataType::ACL_BF16;
    }

    TestCase tc;
    tc.name      = caseName.c_str();
    tc.B         = B;
    tc.N_q       = N_q;
    tc.N_kv      = N_kv;
    tc.S_q       = S_q;
    tc.S_kv      = S_kv;
    tc.D         = D;
    tc.maskMode  = maskMode;
    tc.winLeft   = winLeft;
    tc.winRight  = winRight;
    tc.qMin      = qMin;  tc.qMax = qMax;
    tc.kMin      = kMin;  tc.kMax = kMax;
    tc.vMin      = vMin;  tc.vMax = vMax;
    tc.layoutQ   = layoutQ.c_str();
    tc.layoutKv  = layoutKv.c_str();
    tc.layoutOut = layoutOut.c_str();
    tc.dtype     = dtype;
    tc.skipCpuRef = skipCpuRef;

    // ─── 打印当前参数配置 ────────────────────────────────────────────────────
    const char *SEP = "══════════════════════════════════════════════════════════════════";
    LOG_PRINT("\n%s\n", SEP);
    LOG_PRINT("  FlashAttn 测试参数配置\n");
    LOG_PRINT("%s\n", SEP);
    LOG_PRINT("  B=%ld  N1(N_q)=%ld  N2(N_kv)=%ld  S1(S_q)=%ld  S2(S_kv)=%ld  D=%ld\n",
              B, N_q, N_kv, S_q, S_kv, D);
    LOG_PRINT("  layout Q/Kv/Out = %s / %s / %s   dtype=%s\n",
              layoutQ.c_str(), layoutKv.c_str(), layoutOut.c_str(), dtypeStr.c_str());
    LOG_PRINT("  maskMode=%ld  winLeft=%ld  winRight=%ld\n",
              maskMode, winLeft, winRight);
    LOG_PRINT("  Q∈[%.3g, %.3g]  K∈[%.3g, %.3g]  V∈[%.3g, %.3g]\n",
              (double)qMin, (double)qMax, (double)kMin, (double)kMax, (double)vMin, (double)vMax);
    LOG_PRINT("  exampleDir=%s\n", exampleDir.c_str());
    LOG_PRINT("  seed=%u  device=%d  metaPt=%s  loader=%s  skipCpuRef=%s  dump=%s  verbose=%s\n",
              seed, deviceId, metaPtPath.c_str(), loaderScript.c_str(),
              skipCpuRef ? "yes" : "no",
              dumpOutput ? "yes" : "no", verbose ? "yes" : "no");
    LOG_PRINT("%s\n\n", SEP);

    // ─── device / stream 初始化 ─────────────────────────────────────────────
    aclrtStream stream;
    if (Init(deviceId, &stream) != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed.\n");
        return 1;
    }

    // ─── 运行 ────────────────────────────────────────────────────────────────
    bool passed = false;
    try {
        passed = RunTestCase(tc, stream, dumpOutput, verbose, seed, metaPtPath, loaderScript);
    } catch (const std::exception &e) {
        LOG_PRINT("[ERROR] 运行异常: %s\n", e.what());
    } catch (...) {
        LOG_PRINT("[ERROR] 未知异常\n");
    }

    LOG_PRINT("\n%s\n  最终结论: %s\n%s\n\n",
              SEP, passed ? "✓ PASS" : "✗ FAIL", SEP);

    // ─── 资源释放 ────────────────────────────────────────────────────────────
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return passed ? 0 : 1;
}