/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Description: 非量化 FP16/BF16 AllGatherMatmulV2 example（参数化版，tests/sim 通道 A 契约）。
 * 环境变量参数化（契约权威源：tests/sim/README.md「与参数化 example 的接口契约」）：
 *   AGM_V2_CASE_ID        用例号（meta.json.case_id），默认 "default"
 *   AGM_V2_DUMP_DIR       dump 输出目录（未设置则不落盘）
 *   AGM_V2_DEV_NUM        rank 数（兼容键，默认 2，取值 2/4/8/...）
 *   AGM_V2_RANK_SIZE      rank 数（契约主键，设置时优先于 AGM_V2_DEV_NUM）
 *   AGM_V2_COMM_MODE      通信模式，默认 ccu（ccu / ai_cpu）
 *   AGM_V2_DTYPE          数据类型，默认 fp16（fp16 / bf16）
 *   AGM_V2_M/K/N          矩阵形状，默认 256 / 256 / 128（K 须在 [256, 65535)；M=0 / N=0 合法）
 *   AGM_V2_BIAS           bias 开关，默认 0（1 时按 dtype 同 x1 构造 (n,) bias 并传入）
 *   AGM_V2_TRANS_B        x2 转置实参开关，默认 0（已冻结口径：x2 实参 = view shape (k,n)
 *                         + strides (1,k)，底层 storage 为 (n,k) 行主序，即 W.t() 连续）
 *   AGM_V2_GATHER_OUT     gatherOut 输出开关，默认 1（0 时传空 tensor 触发 IsGatherOut=false）
 *   AGM_V2_VALUE_SPEC     输入数值语义，默认 clean
 *                         （clean / signed_zero / inf_nan / magnitude_bands，语义与
 *                          tests/sim/selftest_compare.py::gen_values 一致）
 *   AGM_V2_VALUE_SCALE    输入幅值缩放系数，乘在 VALUE_SPEC 生成值上，默认 1.0
 *                         （取值 (0,1]；P1 处置 2026-09-20：clean ±5→±0.5 压回
 *                          §2.3 固定 abs 上限适用域；±0/±inf/nan 位形不变——
 *                          IEEE 乘法保持：-0*s=-0、±inf*s=±inf、nan*s=nan）
 *   AGM_V2_REPEAT_LAUNCH  确定性重复 launch 次数，默认 1（>1 时逐 run dump output）
 *   AGM_V2_SEED           确定性伪随机种子（per-rank 派生区分）；未设置且 VALUE_SPEC 未设置
 *                         时走内置确定性公式（与旧版完全一致）
 * 未设置任何新键时行为与旧版种子完全一致（内置公式数据、无 bias、无转置、gatherOut 有效、
 * 单次 launch、不落盘）。
 * 计时：每次 launch（含同步）后打印一行 "AGM_V2_TIMING_US: <微秒>"
 * （tests/sim/perf/collect_perf.sh 采集、perf_report.py 解析的契约格式）。
 * dump 落盘（AGM_V2_DUMP_DIR 设置时，按 compare_dump.py 期望的命名与格式）：
 *   meta.json / x1_rank{r}.bin / x2_rank{r}.bin（完整 storage 字节）/
 *   bias_rank{r}.bin（BIAS=1 时）/ output_rank{r}.bin /
 *   gather_out_rank{r}.bin（GATHER_OUT=1 时）/ output_rank{r}_run{i}.bin（REPEAT>1 时 i>=1）
 * 判定（不变 + VALUE_SPEC=inf_nan 特例；P2 处置 2026-09-20 内嵌校验对齐 §2.3）：
 *   gatherOut 与各 rank 的 x1 严格位级比对（通信为纯搬运，含 ±0 符号位与 NaN 位形）；
 *   output 与 host 侧 FP32 累加 golden 按 §2.3 口径比对（与 compare_dump.py::
 *   PRECISION_CRITERIA 对齐：fp16 atol=rtol=2^-9 且 max_abs_error<=0.1；
 *   bf16 atol=rtol=2^-6 且 max_abs_error<=1.0；matched_ratio>=0.99）；
 *   inf_nan 规格：output 的 ±inf/nan 位置按传播语义类别核对（信息性，不判失败，
 *   与 §2.3 口径一致——权威判定在 compare_dump.py）；
 *   有限域仍按容差判定。
 *   全 rank 通过进程退出码为 0。
 */

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_all_gather_matmul_v2.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
        fflush(stdout); \
    } while (0)

namespace {
// ---------------- 环境参数（契约键，见文件头注释） ----------------
int g_devNum = 2;
const char *g_commMode = "ccu";
bool g_isBf16 = false;
int64_t g_m = 256;
int64_t g_k = 256;
int64_t g_n = 128;
bool g_bias = false;        // AGM_V2_BIAS，默认 0
bool g_transB = false;      // AGM_V2_TRANS_B，默认 0
bool g_gatherOut = true;    // AGM_V2_GATHER_OUT，默认 1
int64_t g_repeatLaunch = 1; // AGM_V2_REPEAT_LAUNCH，默认 1
bool g_seedSet = false;     // AGM_V2_SEED 是否显式设置
uint64_t g_seed = 20260919; // 默认种子（仅显式设置 VALUE_SPEC 时参与生成）
bool g_dumpEnabled = false; // AGM_V2_DUMP_DIR 是否设置
std::string g_dumpDir;
std::string g_caseId = "default"; // AGM_V2_CASE_ID

// 输入数值语义（与 selftest_compare.py::gen_values 一致）
enum ValueSpec {
    SPEC_CLEAN = 0,
    SPEC_SIGNED_ZERO = 1,
    SPEC_INF_NAN = 2,
    SPEC_MAG_BANDS = 3,
};
ValueSpec g_valueSpec = SPEC_CLEAN;
bool g_valueSpecSet = false;
float g_valueScale = 1.0f; // AGM_V2_VALUE_SCALE，默认 1.0（行为不变）

const char *ValueSpecName(ValueSpec spec)
{
    switch (spec) {
        case SPEC_SIGNED_ZERO:
            return "signed_zero";
        case SPEC_INF_NAN:
            return "inf_nan";
        case SPEC_MAG_BANDS:
            return "magnitude_bands";
        case SPEC_CLEAN:
        default:
            return "clean";
    }
}

// 内置公式数据（未设置 SEED/VALUE_SPEC 时使用，保持旧版行为完全一致）
bool UseBuiltinFormula()
{
    return !g_seedSet && !g_valueSpecSet;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto d : shape) {
        shapeSize *= d;
    }
    return shapeSize;
}

int ParseIntEnv(const char *key, int64_t defVal, int64_t minVal, int64_t maxVal, bool &ok)
{
    const char *raw = getenv(key);
    if (raw == nullptr || raw[0] == '\0') {
        return static_cast<int>(defVal);
    }
    char *end = nullptr;
    long v = std::strtol(raw, &end, 10);
    if (end == nullptr || *end != '\0' || v < minVal || v > maxVal) {
        LOG_PRINT("[ERROR] env %s='%s' invalid, require [%ld, %ld].\n", key, raw, minVal, maxVal);
        ok = false;
        return static_cast<int>(defVal);
    }
    return static_cast<int>(v);
}

bool ParseBoolEnv(const char *key, bool defVal, bool &ok)
{
    const char *raw = getenv(key);
    if (raw == nullptr || raw[0] == '\0') {
        return defVal;
    }
    if (std::strcmp(raw, "0") == 0) {
        return false;
    }
    if (std::strcmp(raw, "1") == 0) {
        return true;
    }
    LOG_PRINT("[ERROR] env %s='%s' invalid, require 0/1.\n", key, raw);
    ok = false;
    return defVal;
}

bool ParseGlobalParams()
{
    bool ok = true;
    // rank 数：RANK_SIZE（契约主键）优先，DEV_NUM（兼容键）兜底，均未设置默认 2
    const char *rankRaw = getenv("AGM_V2_RANK_SIZE");
    const char *devRaw = getenv("AGM_V2_DEV_NUM");
    if (rankRaw != nullptr && rankRaw[0] != '\0') {
        g_devNum = ParseIntEnv("AGM_V2_RANK_SIZE", 2, 1, 64, ok);
        if (devRaw != nullptr && devRaw[0] != '\0' && std::strcmp(rankRaw, devRaw) != 0) {
            LOG_PRINT("[WARN] AGM_V2_RANK_SIZE=%s and AGM_V2_DEV_NUM=%s differ, using RANK_SIZE.\n", rankRaw, devRaw);
        }
    } else {
        g_devNum = ParseIntEnv("AGM_V2_DEV_NUM", 2, 1, 64, ok);
    }
    int64_t m = ParseIntEnv("AGM_V2_M", 256, 0, 1 << 20, ok); // m=0 合法（x1 空 tensor 用例）
    int64_t k = ParseIntEnv("AGM_V2_K", 256, 256, 65534, ok);
    int64_t n = ParseIntEnv("AGM_V2_N", 128, 0, 1 << 20, ok); // n=0 合法（只通信分支用例）
    g_m = m;
    g_k = k;
    g_n = n;
    const char *comm = getenv("AGM_V2_COMM_MODE");
    if (comm != nullptr && comm[0] != '\0') {
        if (std::strcmp(comm, "ccu") != 0 && std::strcmp(comm, "ai_cpu") != 0) {
            LOG_PRINT("[ERROR] AGM_V2_COMM_MODE='%s' invalid, require ccu/ai_cpu.\n", comm);
            ok = false;
        } else {
            g_commMode = comm;
        }
    }
    const char *dtype = getenv("AGM_V2_DTYPE");
    if (dtype != nullptr && dtype[0] != '\0') {
        if (std::strcmp(dtype, "fp16") != 0 && std::strcmp(dtype, "bf16") != 0) {
            LOG_PRINT("[ERROR] AGM_V2_DTYPE='%s' invalid, require fp16/bf16.\n", dtype);
            ok = false;
        } else {
            g_isBf16 = (std::strcmp(dtype, "bf16") == 0);
        }
    }
    g_bias = ParseBoolEnv("AGM_V2_BIAS", false, ok);
    g_transB = ParseBoolEnv("AGM_V2_TRANS_B", false, ok);
    g_gatherOut = ParseBoolEnv("AGM_V2_GATHER_OUT", true, ok);
    g_repeatLaunch = ParseIntEnv("AGM_V2_REPEAT_LAUNCH", 1, 1, 1000, ok);
    const char *spec = getenv("AGM_V2_VALUE_SPEC");
    if (spec != nullptr && spec[0] != '\0') {
        g_valueSpecSet = true;
        if (std::strcmp(spec, "clean") == 0) {
            g_valueSpec = SPEC_CLEAN;
        } else if (std::strcmp(spec, "signed_zero") == 0) {
            g_valueSpec = SPEC_SIGNED_ZERO;
        } else if (std::strcmp(spec, "inf_nan") == 0) {
            g_valueSpec = SPEC_INF_NAN;
        } else if (std::strcmp(spec, "magnitude_bands") == 0) {
            g_valueSpec = SPEC_MAG_BANDS;
        } else {
            LOG_PRINT("[ERROR] AGM_V2_VALUE_SPEC='%s' invalid, "
                      "require clean/signed_zero/inf_nan/magnitude_bands.\n",
                      spec);
            ok = false;
        }
    }
    const char *seedRaw = getenv("AGM_V2_SEED");
    if (seedRaw != nullptr && seedRaw[0] != '\0') {
        char *end = nullptr;
        if (seedRaw[0] < '0' || seedRaw[0] > '9') {
            LOG_PRINT("[ERROR] env AGM_V2_SEED='%s' invalid, require non-negative integer.\n", seedRaw);
            ok = false;
        } else {
            unsigned long long v = std::strtoull(seedRaw, &end, 10);
            if (end == nullptr || *end != '\0') {
                LOG_PRINT("[ERROR] env AGM_V2_SEED='%s' invalid, require non-negative integer.\n", seedRaw);
                ok = false;
            } else {
                g_seed = static_cast<uint64_t>(v);
                g_seedSet = true;
            }
        }
    }
    const char *scaleRaw = getenv("AGM_V2_VALUE_SCALE");
    if (scaleRaw != nullptr && scaleRaw[0] != '\0') {
        char *end = nullptr;
        double v = std::strtod(scaleRaw, &end);
        if (end == nullptr || *end != '\0' || !(v > 0.0) || v > 1.0) {
            LOG_PRINT("[ERROR] env AGM_V2_VALUE_SCALE='%s' invalid, require (0, 1].\n", scaleRaw);
            ok = false;
        } else {
            g_valueScale = static_cast<float>(v);
        }
    }
    const char *dumpRaw = getenv("AGM_V2_DUMP_DIR");
    if (dumpRaw != nullptr && dumpRaw[0] != '\0') {
        g_dumpDir = dumpRaw;
        g_dumpEnabled = true;
    }
    const char *caseRaw = getenv("AGM_V2_CASE_ID");
    if (caseRaw != nullptr && caseRaw[0] != '\0') {
        g_caseId = caseRaw;
    }
    return ok;
}

// ---------------- 确定性伪随机（splitmix64，语义对齐 selftest gen_values） ----------------
struct SplitMix64Rng {
    uint64_t state_;
    explicit SplitMix64Rng(uint64_t seed)
        : state_(seed)
    {}
    uint64_t NextU64()
    {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // [0, 1) 的 53-bit 均匀双精度
    double NextDouble()
    {
        return static_cast<double>(NextU64() >> 11) * (1.0 / 9007199254740992.0);
    }
};

// per-rank 种子派生（README 契约：确定性种子 per-rank 派生区分）
uint64_t RankSeed(uint64_t seed, int rank)
{
    return seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(rank + 1);
}

float UniformRange(SplitMix64Rng &rng, float lo, float hi)
{
    return lo + static_cast<float>(rng.NextDouble()) * (hi - lo);
}

float GenSpecValueRaw(ValueSpec spec, SplitMix64Rng &rng)
{
    switch (spec) {
        case SPEC_CLEAN: {
            // 80% uniform(-5,5) + 20% uniform(-1,1)（小值带分层，含负镜像）
            float big = UniformRange(rng, -5.0f, 5.0f);
            float small = UniformRange(rng, -1.0f, 1.0f);
            bool takeSmall = rng.NextDouble() < 0.2;
            return takeSmall ? small : big;
        }
        case SPEC_SIGNED_ZERO: {
            // 约 20% 元素为 +0/-0（符号位随机）
            float v = UniformRange(rng, -5.0f, 5.0f);
            if (rng.NextDouble() < 0.2) {
                return (rng.NextDouble() < 0.5f) ? -0.0f : +0.0f;
            }
            return v;
        }
        case SPEC_INF_NAN: {
            // 约 0.1% +inf、0.1% -inf、0.3% nan
            float v = UniformRange(rng, -5.0f, 5.0f);
            double u = rng.NextDouble();
            if (u < 0.001) {
                return std::numeric_limits<float>::infinity();
            }
            if (u < 0.002) {
                return -std::numeric_limits<float>::infinity();
            }
            if (u < 0.005) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return v;
        }
        case SPEC_MAG_BANDS: {
            // [1,5]/[1e-2,1e-1]/[1e-4,1e-3]/[0,1e-6] 四带等分，符号随机
            static const float kBandLo[4] = {1.0f, 1e-2f, 1e-4f, 0.0f};
            static const float kBandHi[4] = {5.0f, 1e-1f, 1e-3f, 1e-6f};
            int band = static_cast<int>(rng.NextDouble() * 4.0);
            if (band > 3) {
                band = 3;
            }
            float v = UniformRange(rng, kBandLo[band], kBandHi[band]);
            return (rng.NextDouble() < 0.5f) ? v : -v;
        }
        default:
            return UniformRange(rng, -5.0f, 5.0f);
    }
}

// P1 处置（2026-09-20）：VALUE_SPEC 生成值统一乘幅值缩放系数 g_valueScale
// （默认 1.0 时 v*1.0f 逐位不变）。单点应用——x1/x2/bias 生成与全 rank x1
// 重放共用本函数，缩放后输入仍为同一（缩放）量级语义；±0/±inf/nan 位形经
// IEEE 乘法保持（-0*s=-0、±inf*s=±inf、nan*s=nan）。
float GenSpecValue(ValueSpec spec, SplitMix64Rng &rng)
{
    return GenSpecValueRaw(spec, rng) * g_valueScale;
}

// ---------------- host 侧 dtype 比特工具 ----------------
// bf16 host 侧位级转换（round-to-nearest-even）；不引 bfloat16.h 以兼容 cust 模式 include 路径
uint16_t FloatToBf16Bits(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    uint32_t rounded = u + 0x7FFFU + ((u >> 16U) & 1U);
    return static_cast<uint16_t>(rounded >> 16U);
}

float Bf16BitsToFloat(uint16_t bits)
{
    uint32_t u = static_cast<uint32_t>(bits) << 16U;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t FloatToBits(float v)
{
    if (g_isBf16) {
        return FloatToBf16Bits(v);
    }
    op::fp16_t h;
    h = v;
    return h.val;
}

float BitsToFloat(uint16_t bits)
{
    if (g_isBf16) {
        return Bf16BitsToFloat(bits);
    }
    op::fp16_t h(bits);
    return h.toFloat();
}

// ---------------- 内置确定性输入（未设置 SEED/VALUE_SPEC 时，与旧版一致） ----------------
float X1RawValue(int rank, int64_t row, int64_t col)
{
    int v = (rank * 131 + static_cast<int>(row) * 7 + static_cast<int>(col) * 3) % 1000;
    return static_cast<float>(v) / 1000.0f - 0.5f; // [-0.5, 0.499]
}

float X2RawValue(int64_t row, int64_t col)
{
    int v = (static_cast<int>(row) * 17 + static_cast<int>(col) * 29) % 1000;
    return (static_cast<float>(v) / 1000.0f - 0.5f) * 0.5f; // [-0.25, 0.2495]
}

float BiasRawValue(int rank, int64_t col)
{
    int v = (rank * 53 + static_cast<int>(col) * 11) % 1000;
    return (static_cast<float>(v) / 1000.0f - 0.5f) * 0.25f; // [-0.125, 0.12475]
}

// ---------------- ACL tensor 构造 ----------------
std::vector<int64_t> RowMajorStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

// 通用构造：hostBits 须与 view 形状元素数一致（空 tensor 传空 vector）。
// 空 tensor（numel==0）时兜底申请一段守卫内存，规避 aclrtMalloc(0) 的边界行为。
int CreateAclTensorFromBits(const std::vector<uint16_t> &hostBits, const std::vector<int64_t> &viewShape,
                            const std::vector<int64_t> &viewStrides, const std::vector<int64_t> &storageShape,
                            void **deviceAddr, aclDataType dataType, aclTensor **tensor)
{
    // 调用方不变量：host 数据量须与 view 元素数一致（空 tensor 两者均为 0）
    CHECK_RET(static_cast<size_t>(GetShapeSize(viewShape)) == hostBits.size(),
              LOG_PRINT("[ERROR] tensor create numel mismatch: view %ld vs host %zu.\n",
                        static_cast<long>(GetShapeSize(viewShape)), hostBits.size());
              return -1);
    const size_t size = hostBits.size() * sizeof(uint16_t);
    const size_t allocSize = (size > 0) ? size : 32; // 空 tensor 守卫分配
    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    if (size > 0) {
        ret = aclrtMemcpy(*deviceAddr, size, hostBits.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    }
    *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, viewStrides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensorRowMajor(const std::vector<uint16_t> &hostBits, const std::vector<int64_t> &shape, void **deviceAddr,
                            aclDataType dataType, aclTensor **tensor)
{
    return CreateAclTensorFromBits(hostBits, shape, RowMajorStrides(shape), shape, deviceAddr, dataType, tensor);
}

// ---------------- dump 落盘（compare_dump.py 契约格式） ----------------
bool MakeDirs(const std::string &path)
{
    if (path.empty()) {
        return false;
    }
    std::string cur;
    cur.reserve(path.size());
    for (size_t i = 0; i < path.size(); i++) {
        cur.push_back(path[i]);
        if (path[i] == '/' && cur.size() > 1) {
            mkdir(cur.substr(0, cur.size() - 1).c_str(), 0755); // 中间目录，忽略已存在
        }
    }
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
    }
    return true;
}

void DumpBytes(const std::string &path, const void *data, size_t len)
{
    FILE *fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        LOG_PRINT("[ERROR] dump open '%s' failed (errno=%d).\n", path.c_str(), errno);
        return;
    }
    if (len > 0 && std::fwrite(data, 1, len, fp) != len) {
        LOG_PRINT("[ERROR] dump write '%s' failed.\n", path.c_str());
    }
    std::fclose(fp);
}

void DumpTensor(const char *name, int rank, const std::vector<uint16_t> &bits)
{
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s_rank%d.bin", g_dumpDir.c_str(), name, rank);
    if (n < 0 || static_cast<size_t>(n) >= sizeof(path)) {
        LOG_PRINT("[ERROR] dump path too long for %s rank %d.\n", name, rank);
        return;
    }
    DumpBytes(path, bits.data(), bits.size() * sizeof(uint16_t));
}

void DumpTensorRun(const char *name, int rank, int64_t run, const std::vector<uint16_t> &bits)
{
    char path[1024];
    int n =
        snprintf(path, sizeof(path), "%s/%s_rank%d_run%ld.bin", g_dumpDir.c_str(), name, rank, static_cast<long>(run));
    if (n < 0 || static_cast<size_t>(n) >= sizeof(path)) {
        LOG_PRINT("[ERROR] dump path too long for %s rank %d run %ld.\n", name, rank, static_cast<long>(run));
        return;
    }
    DumpBytes(path, bits.data(), bits.size() * sizeof(uint16_t));
}

std::string JsonEscape(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '\"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// meta.json：case 参数 + x2 视图 shape/strides（元素单位），字段与 compare_dump.py 装载逻辑对齐
void WriteMetaJson()
{
    // transB 契约（已冻结口径）：x2 实参 view shape (k,n) + strides (1,k)，
    // storage 为 (n,k) 行主序（W.t() 连续）；非 transB 为 (k,n) 行主序
    std::vector<int64_t> x2ViewShape = {g_k, g_n};
    std::vector<int64_t> x2ViewStrides;
    if (g_transB) {
        x2ViewStrides = {1, g_k};
    } else {
        x2ViewStrides = RowMajorStrides({g_k, g_n});
    }
    std::string desc = "AGM_V2 parameterized example (dtype=" + std::string(g_isBf16 ? "bf16" : "fp16") +
                       ", m=" + std::to_string(g_m) + ", k=" + std::to_string(g_k) + ", n=" + std::to_string(g_n) +
                       ", R=" + std::to_string(g_devNum) + ", comm=" + std::string(g_commMode) +
                       ", bias=" + std::to_string(g_bias ? 1 : 0) + ", transB=" + std::to_string(g_transB ? 1 : 0) +
                       ", gatherOut=" + std::to_string(g_gatherOut ? 1 : 0) + ", spec=" + ValueSpecName(g_valueSpec) +
                       ", scale=" + std::to_string(g_valueScale) + ", repeat=" + std::to_string(g_repeatLaunch) +
                       ", seedMode=" + std::string(UseBuiltinFormula() ? "builtin" : "seeded") + ")";
    std::string json = "{\n";
    json += " \"case_id\": \"" + JsonEscape(g_caseId) + "\",\n";
    json += " \"desc\": \"" + JsonEscape(desc) + "\",\n";
    json += std::string(" \"dtype\": \"") + (g_isBf16 ? "bf16" : "fp16") + "\",\n";
    json += " \"m\": " + std::to_string(g_m) + ",\n";
    json += " \"k\": " + std::to_string(g_k) + ",\n";
    json += " \"n\": " + std::to_string(g_n) + ",\n";
    json += " \"rank_size\": " + std::to_string(g_devNum) + ",\n";
    json += std::string(" \"comm_mode\": \"") + g_commMode + "\",\n";
    json += std::string(" \"bias\": ") + (g_bias ? "true" : "false") + ",\n";
    json += std::string(" \"trans_b\": ") + (g_transB ? "true" : "false") + ",\n";
    json += std::string(" \"gather_out\": ") + (g_gatherOut ? "true" : "false") + ",\n";
    json += std::string(" \"value_spec\": \"") + ValueSpecName(g_valueSpec) + "\",\n";
    json += " \"value_scale\": " + std::to_string(g_valueScale) + ",\n";
    json += " \"repeat_launch\": " + std::to_string(g_repeatLaunch) + ",\n";
    json += " \"tensors\": {\n";
    json += "  \"x2\": {\n";
    json += "   \"shape\": [" + std::to_string(x2ViewShape[0]) + ", " + std::to_string(x2ViewShape[1]) + "],\n";
    json += "   \"strides\": [" + std::to_string(x2ViewStrides[0]) + ", " + std::to_string(x2ViewStrides[1]) + "]\n";
    json += "  }\n";
    json += " }\n";
    json += "}\n";
    DumpBytes(g_dumpDir + "/meta.json", json.data(), json.size());
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    int result = -1; // 0 = pass
};

// 秩序收尾（失败路径也要保证通信域/流/上下文释放，避免仿真进程挂死在收尾）
void FinalizeRank(Args &args)
{
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtResetDevice(args.rankId);
    aclrtDestroyContext(args.context);
}

// ---------------- 单 rank 执行 + 内嵌 golden 比对 ----------------
int LaunchOneThreadAllGatherMmV2(Args &args)
{
    const int rankId = args.rankId;
    const int rankDim = g_devNum;
    const int64_t m = g_m;
    const int64_t k = g_k;
    const int64_t n = g_n;
    const aclDataType dtype = g_isBf16 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
    // §2.3 口径（P2 处置 2026-09-20：内嵌校验对齐 compare_dump.py::
    // PRECISION_CRITERIA——fp16 rtol=atol=2^-9 且 max_abs_error<=0.1；
    // bf16 rtol=atol=2^-6 且 max_abs_error<=1.0；matched_ratio>=0.99）
    const float atol = g_isBf16 ? 0.015625f : 0.001953125f; // 2^-6 / 2^-9
    const float rtol = g_isBf16 ? 0.015625f : 0.001953125f;
    const float maxAbsErrLimit = g_isBf16 ? 1.0f : 0.1f;
    const double matchedRatioMin = 0.99;

    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, devNum = %d, commMode = %s, dtype = %s, m = %ld, k = %ld, "
              "n = %ld, bias = %d, transB = %d, gatherOut = %d, valueSpec = %s, valueScale = %.4g, "
              "repeat = %ld, seed = %s\n",
              rankId, hcomName, rankDim, g_commMode, g_isBf16 ? "bf16" : "fp16", static_cast<long>(m),
              static_cast<long>(k), static_cast<long>(n), g_bias ? 1 : 0, g_transB ? 1 : 0, g_gatherOut ? 1 : 0,
              ValueSpecName(g_valueSpec), static_cast<double>(g_valueScale), static_cast<long>(g_repeatLaunch),
              UseBuiltinFormula() ? "builtin" : "seeded");

    std::vector<int64_t> x1Shape = {m, k};
    std::vector<int64_t> x2ViewShape = {k, n};
    std::vector<int64_t> outShape = {m * rankDim, n};
    std::vector<int64_t> gatherOutShape = {m * rankDim, k};

    // 确定性生成：本 rank 的 x1 / 逻辑权重 W(k,n) / bias（golden 需要全 rank x1 数据）
    std::vector<uint16_t> x1Bits;
    std::vector<float> x1Quant;
    std::vector<uint16_t> x2Bits; // 逻辑权重 W (k,n) 行主序
    std::vector<float> x2Quant;   // 逻辑权重 W 的量化值（golden 用）
    std::vector<uint16_t> biasBits;
    std::vector<float> biasQuant;
    // 全 rank x1 量化值（gatherOut golden 与 output golden 共用）
    std::vector<float> allX1Quant(static_cast<size_t>(m) * k * rankDim);
    std::vector<uint16_t> allX1Bits(static_cast<size_t>(m) * k * rankDim);

    if (UseBuiltinFormula()) {
        // 内置公式（旧版行为，逐位一致）
        x1Bits.resize(static_cast<size_t>(m) * k);
        x1Quant.resize(static_cast<size_t>(m) * k);
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < k; j++) {
                uint16_t bits = FloatToBits(X1RawValue(rankId, i, j));
                x1Bits[static_cast<size_t>(i) * k + j] = bits;
                x1Quant[static_cast<size_t>(i) * k + j] = BitsToFloat(bits);
            }
        }
        x2Bits.resize(static_cast<size_t>(k) * n);
        x2Quant.resize(static_cast<size_t>(k) * n);
        for (int64_t i = 0; i < k; i++) {
            for (int64_t j = 0; j < n; j++) {
                uint16_t bits = FloatToBits(X2RawValue(i, j));
                x2Bits[static_cast<size_t>(i) * n + j] = bits;
                x2Quant[static_cast<size_t>(i) * n + j] = BitsToFloat(bits);
            }
        }
        if (g_bias) {
            // bias：clean 语义小数值、(n,) 形状、本 rank 独立构造（口径与种子化分支一致，dtype 同 x1）
            biasBits.resize(static_cast<size_t>(n));
            biasQuant.resize(static_cast<size_t>(n));
            for (int64_t j = 0; j < n; j++) {
                uint16_t bits = FloatToBits(BiasRawValue(rankId, j));
                biasBits[static_cast<size_t>(j)] = bits;
                biasQuant[static_cast<size_t>(j)] = BitsToFloat(bits);
            }
        }
        for (int r = 0; r < rankDim; r++) {
            for (int64_t i = 0; i < m; i++) {
                for (int64_t j = 0; j < k; j++) {
                    uint16_t bits = FloatToBits(X1RawValue(r, i, j));
                    allX1Bits[(static_cast<size_t>(r) * m + i) * k + j] = bits;
                    allX1Quant[(static_cast<size_t>(r) * m + i) * k + j] = BitsToFloat(bits);
                }
            }
        }
    } else {
        // 种子化生成（契约 value_spec 语义；x1 → x2 → bias 按序抽取，per-rank 种子派生）
        SplitMix64Rng rng(RankSeed(g_seed, rankId));
        auto genTensor = [&rng](size_t numel, ValueSpec spec, std::vector<uint16_t> &bits, std::vector<float> &quant) {
            bits.resize(numel);
            quant.resize(numel);
            for (size_t i = 0; i < numel; i++) {
                uint16_t b = FloatToBits(GenSpecValue(spec, rng));
                bits[i] = b;
                quant[i] = BitsToFloat(b);
            }
        };
        genTensor(static_cast<size_t>(m) * k, g_valueSpec, x1Bits, x1Quant);
        genTensor(static_cast<size_t>(k) * n, g_valueSpec, x2Bits, x2Quant);
        if (g_bias) {
            genTensor(static_cast<size_t>(n), SPEC_CLEAN, biasBits, biasQuant); // bias 用 clean 语义
        }
        // 全 rank x1 重建：x1 为各 rank 随机流的首段抽取，独立重放即可复现
        for (int r = 0; r < rankDim; r++) {
            SplitMix64Rng replay(RankSeed(g_seed, r));
            for (int64_t i = 0; i < m; i++) {
                for (int64_t j = 0; j < k; j++) {
                    uint16_t bits = FloatToBits(GenSpecValue(g_valueSpec, replay));
                    allX1Bits[(static_cast<size_t>(r) * m + i) * k + j] = bits;
                    allX1Quant[(static_cast<size_t>(r) * m + i) * k + j] = BitsToFloat(bits);
                }
            }
        }
    }

    // transB 契约：x2 API 实参 = view (k,n) + strides (1,k)，storage 为 W.t() 连续（(n,k) 行主序）
    std::vector<uint16_t> x2DumpBits; // 落盘 = API 实参完整 storage 字节
    if (g_transB) {
        x2DumpBits.assign(static_cast<size_t>(k) * n, 0);
        for (int64_t i = 0; i < k; i++) {
            for (int64_t j = 0; j < n; j++) {
                x2DumpBits[static_cast<size_t>(j) * k + i] = x2Bits[static_cast<size_t>(i) * n + j];
            }
        }
    } else {
        x2DumpBits = x2Bits;
    }

    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *gatherOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *biasTensor = nullptr;
    aclTensor *out = nullptr;
    aclTensor *gatherOut = nullptr;

    ret = CreateAclTensorRowMajor(x1Bits, x1Shape, &x1DeviceAddr, dtype, &x1);
    CHECK_RET(ret == 0, return ret);
    if (g_transB) {
        // 转置视图：view (k,n)、strides (1,k)、storage (n,k) 行主序
        std::vector<int64_t> x2StorageShape = {n, k};
        std::vector<int64_t> x2Strides = {1, k};
        ret = CreateAclTensorFromBits(x2DumpBits, x2ViewShape, x2Strides, x2StorageShape, &x2DeviceAddr, dtype, &x2);
        CHECK_RET(ret == 0, return ret);
    } else {
        ret = CreateAclTensorRowMajor(x2Bits, x2ViewShape, &x2DeviceAddr, dtype, &x2);
        CHECK_RET(ret == 0, return ret);
    }
    if (g_bias) {
        // 非量化 bias：dtype 同 x1，shape (n,)，本 rank 独立构造
        std::vector<int64_t> biasShape = {n};
        ret = CreateAclTensorRowMajor(biasBits, biasShape, &biasDeviceAddr, dtype, &biasTensor);
        CHECK_RET(ret == 0, return ret);
    }
    std::vector<uint16_t> outInit(static_cast<size_t>(m) * rankDim * n, 0);
    ret = CreateAclTensorRowMajor(outInit, outShape, &outDeviceAddr, dtype, &out);
    CHECK_RET(ret == 0, return ret);
    // gatherOut：GATHER_OUT=0 时传空 tensor（view (0,k)）触发 IsGatherOut=false 路径
    std::vector<uint16_t> gatherInit;
    if (g_gatherOut) {
        gatherInit.assign(static_cast<size_t>(m) * rankDim * k, 0);
    }
    std::vector<int64_t> gatherShape = g_gatherOut ? gatherOutShape : std::vector<int64_t>{0, k};
    ret = CreateAclTensorRowMajor(gatherInit, gatherShape, &gatherOutDeviceAddr, dtype, &gatherOut);
    CHECK_RET(ret == 0, return ret);

    // ---------------- 输入侧 dump（API 实参字节） ----------------
    if (g_dumpEnabled) {
        DumpTensor("x1", rankId, x1Bits);
        DumpTensor("x2", rankId, x2DumpBits); // 完整 storage 字节
        if (g_bias) {
            DumpTensor("bias", rankId, biasBits);
        }
    }

    // ---------------- 重复 launch（REPEAT_LAUNCH 次，每次完整 API 调用周期） ----------------
    int checkRet = 0;
    const size_t outElemCnt = static_cast<size_t>(m) * rankDim * n;
    const size_t gatherElemCnt = g_gatherOut ? static_cast<size_t>(m) * rankDim * k : 0;
    const size_t outBytes = outElemCnt * sizeof(uint16_t);
    const size_t gatherBytes = gatherElemCnt * sizeof(uint16_t);
    std::vector<uint16_t> outDev(outElemCnt, 0);
    std::vector<uint16_t> gatherDev(gatherElemCnt, 0);

    for (int64_t run = 0; run < g_repeatLaunch; run++) {
        const int64_t gatherIndex = 0;
        const int64_t commTurn = 0;
        const int64_t streamMode = 1;
        const int64_t blockSize = 0;
        const int64_t groupSize = 0;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        // 每次 run 重建 executor（完整 API 调用周期，确定性用例语义）
        ret = aclnnAllGatherMatmulV2GetWorkspaceSize(x1, x2, g_bias ? biasTensor : nullptr, nullptr, nullptr, nullptr,
                                                     blockSize, hcomName, gatherIndex, commTurn, streamMode, groupSize,
                                                     g_commMode, out, gatherOut, nullptr, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR][rank %d][run %ld] "
                                                "aclnnAllGatherMatmulV2GetWorkspaceSize failed. ret = %d\n",
                                                rankId, static_cast<long>(run), ret);
                  return ret);
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret: %d \n", ret);
                      return ret);
        }
        // run 间清零输出缓冲，保证 run 间 bitwise 一致性判定非平凡（非残留数据）
        if (outBytes > 0) {
            ret = aclrtMemset(outDeviceAddr, outBytes, 0, outBytes);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemset out failed. ret: %d\n", ret); return ret);
        }
        if (g_gatherOut && gatherBytes > 0) {
            ret = aclrtMemset(gatherOutDeviceAddr, gatherBytes, 0, gatherBytes);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemset gatherOut failed. ret: %d\n", ret);
                      return ret);
        }
        LOG_PRINT("[INFO][rank %d] run %ld/%ld GetWorkspaceSize success, workspaceSize = %lu.\n", rankId,
                  static_cast<long>(run + 1), static_cast<long>(g_repeatLaunch),
                  static_cast<unsigned long>(workspaceSize));

        // 阶段二 + 计时（launch 至同步完成的墙钟，微秒）
        auto t0 = std::chrono::steady_clock::now();
        ret = aclnnAllGatherMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR][rank %d][run %ld] aclnnAllGatherMatmulV2 failed. "
                                                "ret = %d\n",
                                                rankId, static_cast<long>(run), ret);
                  return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 1800000);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR][rank %d][run %ld] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", rankId,
                            static_cast<long>(run), ret);
                  return ret);
        auto t1 = std::chrono::steady_clock::now();
        double timingUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
        LOG_PRINT("AGM_V2_TIMING_US: %.3f\n", timingUs);
        LOG_PRINT("[INFO][rank %d][run %ld] aclnnAllGatherMatmulV2 execute done.\n", rankId, static_cast<long>(run));

        // ---------------- 回读 + dump + golden 比对 ----------------
        if (outBytes > 0) {
            ret = aclrtMemcpy(outDev.data(), outBytes, outDeviceAddr, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR][rank %d] memcpy out failed. ret: %d\n", rankId, ret);
                      return ret);
        }
        if (g_gatherOut && gatherBytes > 0) {
            ret =
                aclrtMemcpy(gatherDev.data(), gatherBytes, gatherOutDeviceAddr, gatherBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR][rank %d] memcpy gatherOut failed. ret: %d\n", rankId, ret);
                      return ret);
        }

        if (g_dumpEnabled) {
            if (run == 0) {
                DumpTensor("output", rankId, outDev);
                if (g_gatherOut) {
                    DumpTensor("gather_out", rankId, gatherDev);
                }
            } else {
                DumpTensorRun("output", rankId, run, outDev); // 确定性 run 追加（i>=1）
            }
        }

        // 1) gatherOut：纯通信搬运，严格位级比对（±0 符号位与 NaN 位形均检出）
        if (g_gatherOut) {
            size_t gatherMismatch = 0;
            for (size_t idx = 0; idx < gatherElemCnt; idx++) {
                if (gatherDev[idx] != allX1Bits[idx]) {
                    if (gatherMismatch < 4) {
                        LOG_PRINT("[ERROR][rank %d][run %ld] gatherOut mismatch at flat %zu (row %zu col %zu): "
                                  "expect 0x%04x got 0x%04x\n",
                                  rankId, static_cast<long>(run), idx, idx / static_cast<size_t>(k),
                                  idx % static_cast<size_t>(k), allX1Bits[idx], gatherDev[idx]);
                    }
                    gatherMismatch++;
                }
            }
            if (gatherMismatch != 0) {
                checkRet = 1;
            }
            LOG_PRINT("[INFO][rank %d][run %ld] gatherOut bitwise compare: %zu/%zu mismatch.\n", rankId,
                      static_cast<long>(run), gatherMismatch, gatherElemCnt);
        }

        // 2) output：FP32 累加 golden，混合容差；±inf/nan 位置按传播语义类别核对
        //    （VALUE_SPEC=inf_nan 时类别失配为信息性，与 §2.3 口径一致，权威判定在 compare_dump.py）
        size_t outMismatch = 0;
        size_t outSpecialInfo = 0;
        float maxAbsErr = 0.0f;
        float maxRelErr = 0.0f;
        for (int64_t row = 0; row < m * rankDim; row++) {
            const float *aRow = &allX1Quant[static_cast<size_t>(row) * k];
            for (int64_t col = 0; col < n; col++) {
                float golden = 0.0f;
                for (int64_t t = 0; t < k; t++) {
                    golden += aRow[t] * x2Quant[static_cast<size_t>(t) * n + col];
                }
                if (g_bias) {
                    golden += biasQuant[static_cast<size_t>(col)];
                }
                float dev = BitsToFloat(outDev[static_cast<size_t>(row) * n + col]);
                bool goldenNan = std::isnan(golden);
                bool devNan = std::isnan(dev);
                if (goldenNan || devNan) {
                    if (!(goldenNan && devNan)) {
                        if (g_valueSpec == SPEC_INF_NAN) {
                            outSpecialInfo++;
                        } else {
                            outMismatch++;
                        }
                    }
                    continue;
                }
                bool goldenInf = std::isinf(golden);
                bool devInf = std::isinf(dev);
                if (goldenInf || devInf) {
                    if (!(goldenInf && devInf && (std::signbit(golden) == std::signbit(dev)))) {
                        if (g_valueSpec == SPEC_INF_NAN) {
                            outSpecialInfo++;
                        } else {
                            outMismatch++;
                        }
                    }
                    continue;
                }
                float absErr = std::fabs(dev - golden);
                float relErr = (std::fabs(golden) > 1e-6f) ? absErr / std::fabs(golden) : 0.0f;
                if (absErr > maxAbsErr) {
                    maxAbsErr = absErr;
                }
                if (relErr > maxRelErr) {
                    maxRelErr = relErr;
                }
                if (absErr > atol + rtol * std::fabs(golden)) {
                    if (outMismatch < 4) {
                        LOG_PRINT("[ERROR][rank %d][run %ld] output mismatch at (row %ld, col %ld): "
                                  "golden %.6f dev %.6f absErr %.6f\n",
                                  rankId, static_cast<long>(run), static_cast<long>(row), static_cast<long>(col),
                                  golden, dev, absErr);
                    }
                    outMismatch++;
                }
            }
        }
        // §2.3 判定（P2 处置）：matched_ratio>=0.99 且 maxAbsErr<=固定上限；
        // 0-mismatch 旧口径严于 §2.3（L1-112 14/35072 误杀），已对齐权威口径。
        // 空 output（m=0/n=0 分支）matched_ratio 定义为 1.0（按通过处理）。
        const double matchedRatio =
            (outElemCnt > 0) ? 1.0 - static_cast<double>(outMismatch) / static_cast<double>(outElemCnt) : 1.0;
        const bool sec23Ok = (matchedRatio >= matchedRatioMin) && (maxAbsErr <= maxAbsErrLimit);
        LOG_PRINT("[INFO][rank %d][run %ld] output compare(§2.3: atol=rtol=%.4g, maxAbsLimit=%.3g, "
                  "matched_ratio_min=%.2f): %zu/%zu mismatch, specialClass(info)=%zu, maxAbsErr %.6e, "
                  "maxRelErr %.6e, matched_ratio=%.6f -> %s.\n",
                  rankId, static_cast<long>(run), static_cast<double>(atol), static_cast<double>(maxAbsErrLimit),
                  matchedRatioMin, outMismatch, outElemCnt, outSpecialInfo, maxAbsErr, maxRelErr, matchedRatio,
                  sec23Ok ? "PASS" : "FAIL");
        if (!sec23Ok) {
            checkRet = 1;
        }
        if (outSpecialInfo != 0) {
            LOG_PRINT("[WARN][rank %d][run %ld] %zu special-value class mismatches (informational for "
                      "value_spec=inf_nan, authoritative check is compare_dump.py).\n",
                      rankId, static_cast<long>(run), outSpecialInfo);
        }

        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
    }

    LOG_PRINT("[INFO][rank %d] %s: %s.\n", rankId, checkRet == 0 ? "PASS" : "FAIL",
              checkRet == 0 ? "all runs passed" : "check failed");
    if (g_dumpEnabled) {
        LOG_PRINT("[INFO][rank %d] dump written to %s (runs=%ld).\n", rankId, g_dumpDir.c_str(),
                  static_cast<long>(g_repeatLaunch));
    }

    // ---------------- 资源释放 ----------------
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (biasTensor != nullptr) {
        aclDestroyTensor(biasTensor);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (gatherOut != nullptr) {
        aclDestroyTensor(gatherOut);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (gatherOutDeviceAddr != nullptr) {
        aclrtFree(gatherOutDeviceAddr);
    }
    return checkRet;
}
} // namespace

int main(int argc, char *argv[])
{
    class EnvGuard {
    public:
        EnvGuard(const char *key, const char *val, bool enable = true)
            : key_(enable ? key : nullptr)
        {
            if (!enable) {
                return;
            }
            const char *old = getenv(key);
            if (old != nullptr) {
                strncpy(saved_, old, sizeof(saved_) - 1);
            }
            setenv(key, val, 1);
        }
        ~EnvGuard()
        {
            if (key_ == nullptr) {
                return;
            }
            if (saved_[0] != '\0') {
                setenv(key_, saved_, 1);
            } else {
                unsetenv(key_);
            }
        }

    private:
        const char *key_;
        char saved_[256] = {0};
    };

    if (!ParseGlobalParams()) {
        return 1;
    }
    if (g_dumpEnabled && !MakeDirs(g_dumpDir)) {
        LOG_PRINT("[ERROR] cannot create dump dir '%s' (errno=%d).\n", g_dumpDir.c_str(), errno);
        return 1;
    }
    if (g_dumpEnabled) {
        WriteMetaJson(); // case 参数 + x2 视图描述（落盘一次，主线程写出）
    }
    EnvGuard envGuard("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", std::strcmp(g_commMode, "ccu") == 0);

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret: %d\n", ret); return ret);
    aclrtStream stream[g_devNum];
    aclrtContext context[g_devNum];
    for (int rankId = 0; rankId < g_devNum; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice(%d) failed. ret: %d\n", rankId, ret);
                  return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret: %d\n", ret); return ret);
    }
    int32_t devices[g_devNum];
    for (int i = 0; i < g_devNum; i++) {
        devices[i] = i;
    }
    HcclComm comms[g_devNum];
    ret = HcclCommInitAll(g_devNum, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll(devNum=%d) failed. ret: %d\n", g_devNum, ret);
              return ret);

    std::vector<Args> args(g_devNum);
    std::vector<std::unique_ptr<std::thread>> threads(g_devNum);
    for (int rankId = 0; rankId < g_devNum; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread([rankId, &args]() {
            int r = LaunchOneThreadAllGatherMmV2(args[rankId]);
            args[rankId].result = r;
            FinalizeRank(args[rankId]);
        }));
    }
    for (int rankId = 0; rankId < g_devNum; rankId++) {
        threads[rankId]->join();
    }
    int finalRet = 0;
    for (int rankId = 0; rankId < g_devNum; rankId++) {
        if (args[rankId].result != 0) {
            LOG_PRINT("[RESULT][rank %d] FAIL (ret = %d)\n", rankId, args[rankId].result);
            finalRet = 1;
        } else {
            LOG_PRINT("[RESULT][rank %d] PASS\n", rankId);
        }
    }
    aclFinalize();
    LOG_PRINT("[RESULT] all_gather_matmul_v2 fp16 example %s (devNum=%d, commMode=%s, dtype=%s, m=%ld, k=%ld, "
              "n=%ld, bias=%d, transB=%d, gatherOut=%d, valueSpec=%s, repeat=%ld)\n",
              finalRet == 0 ? "PASS" : "FAIL", g_devNum, g_commMode, g_isBf16 ? "bf16" : "fp16", static_cast<long>(g_m),
              static_cast<long>(g_k), static_cast<long>(g_n), g_bias ? 1 : 0, g_transB ? 1 : 0, g_gatherOut ? 1 : 0,
              ValueSpecName(g_valueSpec), static_cast<long>(g_repeatLaunch));
    return finalRet;
}
