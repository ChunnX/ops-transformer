/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License in the root directory.
 */

/*
 * CSV-driven numerical test for FLOAT4_E2M1 mode-6 PageAttention antiquant.
 *
 * The source KV tensor is written in the selected physical layout (NZ, BNBD or
 * BBH), but the CPU oracle reads a separate canonical [P, N, Bs, D] code tensor.
 * This separation makes layout/packing errors observable instead of allowing the
 * producer and oracle to share a mistaken packed-address calculation.
 */

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v5.h"

namespace {

constexpr int64_t D0 = 16;
constexpr int64_t GROUP_SIZE = 32;
constexpr std::array<float, 16> FP4_E2M1_VALUES = {
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
    -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
};

enum class QType : uint8_t { FP16, BF16 };
enum class KvLayout : uint8_t { PA_NZ, PA_BNBD, PA_BBH };
enum class RandomProfile : uint8_t { LEGACY, STABLE, WIDE };

struct CaseConfig {
    std::string name;
    KvLayout kvLayout;
    QType qType;
    bool withOffset;
    int64_t b;
    int64_t n; // KV head count
    int64_t qHeads;
    int64_t s1;
    int64_t d;
    int64_t blockSize;
    int64_t totalBlockNum;
    std::vector<int64_t> actualKvLens;
    std::vector<int32_t> blockTable;
    uint64_t seed = 0;
    RandomProfile randomProfile = RandomProfile::LEGACY;
};

int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

float Fp16ToFloat(uint16_t value)
{
    const uint32_t sign = (value >> 15U) & 1U;
    uint32_t exponent = (value >> 10U) & 0x1FU;
    uint32_t mantissa = value & 0x3FFU;
    uint32_t bits = 0;
    if (exponent == 0x1FU) {
        bits = (sign << 31U) | (0xFFU << 23U) | (mantissa == 0 ? 0U : 0x7FFFFFU);
    } else if (exponent == 0U) {
        if (mantissa != 0U) {
            exponent = 113U;
            while ((mantissa & 0x400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x3FFU;
            bits = (sign << 31U) | (exponent << 23U) | (mantissa << 13U);
        } else {
            bits = sign << 31U;
        }
    } else {
        bits = (sign << 31U) | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U); // round-to-nearest-even
    return static_cast<uint16_t>(bits >> 16U);
}

float Bf16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t EncodeQ(float value, QType type)
{
    return type == QType::FP16 ? static_cast<uint16_t>(aclFloatToFloat16(value)) : FloatToBf16(value);
}

float DecodeQ(uint16_t value, QType type)
{
    return type == QType::FP16 ? Fp16ToFloat(value) : Bf16ToFloat(value);
}

float RoundToBf16(float value)
{
    return Bf16ToFloat(FloatToBf16(value));
}

aclDataType ToAclType(QType type)
{
    return type == QType::FP16 ? aclDataType::ACL_FLOAT16 : aclDataType::ACL_BF16;
}

const char *LayoutName(KvLayout layout)
{
    return layout == KvLayout::PA_NZ ? "PA_NZ" : layout == KvLayout::PA_BNBD ? "PA_BNBD" : "PA_BBH";
}

const char *QTypeName(QType type)
{
    return type == QType::FP16 ? "fp16" : "bf16";
}

const char *RandomProfileName(RandomProfile profile)
{
    return profile == RandomProfile::WIDE ? "wide" : profile == RandomProfile::STABLE ? "stable" : "legacy";
}

float E8M0ToFloat(uint8_t exponent)
{
    return std::ldexp(1.0F, static_cast<int>(exponent) - 127);
}

uint64_t Mix64(uint64_t value)
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

uint64_t SampleHash(uint64_t seed, uint64_t stream, int64_t p, int64_t n, int64_t token, int64_t index,
                    int64_t salt)
{
    uint64_t value = Mix64(seed ^ stream);
    value = Mix64(value ^ static_cast<uint64_t>(p));
    value = Mix64(value ^ static_cast<uint64_t>(n));
    value = Mix64(value ^ static_cast<uint64_t>(token));
    value = Mix64(value ^ static_cast<uint64_t>(index));
    return Mix64(value ^ static_cast<uint64_t>(salt));
}

uint8_t Fp4Code(const CaseConfig &config, int64_t p, int64_t n, int64_t token, int64_t d, int64_t salt)
{
    if (config.seed != 0) {
        // Every 16-element segment is a seeded random permutation of all E2M1
        // codes. This is less regular than the legacy affine pattern while
        // retaining guaranteed coverage of every code in every full segment.
        const uint64_t sample = SampleHash(config.seed, 0x465034ULL, p, n, token, d / 16, salt);
        const uint8_t multiplier = static_cast<uint8_t>(((sample >> 8U) & 0x7U) * 2U + 1U);
        return static_cast<uint8_t>((multiplier * static_cast<uint8_t>(d & 0xFU) + (sample & 0xFU)) & 0xFU);
    }
    // Coefficients are odd modulo 16. For D >= 16 every token covers all 16
    // E2M1 codes, while P/N/token still perturb the sequence independently.
    return static_cast<uint8_t>((11 * p + 7 * n + 5 * token + 3 * d + salt) & 0xFU);
}

uint8_t ScaleCode(const CaseConfig &config, int64_t p, int64_t n, int64_t group, int64_t token, int64_t salt)
{
    if (config.seed != 0) {
        // The stable profile preserves the established functional range while
        // varying every scale position. The wide profile is used only by a
        // single-token antiquant test, where softmax is identically one.
        constexpr std::array<uint8_t, 4> STABLE_SCALE_CODES = {125, 126, 127, 128};
        constexpr std::array<uint8_t, 6> WIDE_SCALE_CODES = {123, 125, 126, 127, 129, 131};
        const uint64_t sample = SampleHash(config.seed, 0x5343414C45ULL, p, n, token, group, salt);
        if (config.randomProfile == RandomProfile::WIDE) {
            return WIDE_SCALE_CODES[static_cast<size_t>(sample % WIDE_SCALE_CODES.size())];
        }
        return STABLE_SCALE_CODES[static_cast<size_t>(sample % STABLE_SCALE_CODES.size())];
    }
    // E8M0 exponents 125..128 correspond to 0.25, 0.5, 1.0 and 2.0.
    return static_cast<uint8_t>(125 + ((5 * p + 3 * n + 7 * group + token + salt) & 0x3U));
}

float OffsetValue(const CaseConfig &config, int64_t p, int64_t n, int64_t group, int64_t token, int64_t salt)
{
    if (config.seed != 0) {
        // Both ranges are exactly representable in FP16 and BF16. The wide
        // range is isolated to the single-token antiquant test.
        const uint64_t sample = SampleHash(config.seed, 0x4F4646534554ULL, p, n, token, group, salt);
        if (config.randomProfile == RandomProfile::WIDE) {
            return 0.125F * static_cast<float>(static_cast<int64_t>(sample % 33U) - 16);
        }
        return 0.0625F * static_cast<float>(static_cast<int64_t>(sample % 17U) - 8);
    }
    // Binary fractions stay exactly representable in both FP16 and BF16.
    return 0.03125F * static_cast<float>(((7 * p + 5 * n + 3 * group + token + salt) % 17) - 8);
}

float QueryValue(const CaseConfig &config, int64_t b, int64_t n, int64_t s1, int64_t d)
{
    if (config.seed != 0) {
        const uint64_t sample = SampleHash(config.seed, 0x5155455259ULL, b, n, s1, d, 0);
        return 0.0625F * static_cast<float>(static_cast<int64_t>(sample % 31U) - 15);
    }
    return 0.03125F * static_cast<float>(((5 * b + 3 * n + 7 * s1 + d) % 15) - 7);
}

int32_t OrderedLowPrecisionBits(uint16_t bits)
{
    // Monotonic encoding for finite IEEE FP16/BF16 values. +0 and -0 map to
    // the same position, so ULP distance is meaningful for output comparison.
    const int32_t magnitude = static_cast<int32_t>(bits & 0x7FFFU);
    return (bits & 0x8000U) == 0 ? 0x8000 + magnitude : 0x8000 - magnitude;
}

float DequantizeReference(uint8_t fp4Code, uint8_t scaleCode, uint16_t offset, bool withOffset, QType qType)
{
    // This is an independent scalar E2M1 decode table. Arithmetic deliberately
    // mirrors the public W4 contract: BF16 add/multiply, then Q_T storage.
    float value = RoundToBf16(FP4_E2M1_VALUES[fp4Code]);
    if (withOffset) {
        value = RoundToBf16(value + RoundToBf16(DecodeQ(offset, qType)));
    }
    value = RoundToBf16(value * RoundToBf16(E8M0ToFloat(scaleCode)));
    return DecodeQ(EncodeQ(value, qType), qType);
}

int CreateTensor(const std::vector<uint16_t> &host, const std::vector<int64_t> &shape, aclDataType type,
                 void **device, aclTensor **tensor)
{
    const size_t bytes = host.size() * sizeof(uint16_t);
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = shape[index + 1] * strides[index + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *device);
    return *tensor == nullptr ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

int CreateByteTensor(const std::vector<uint8_t> &host, const std::vector<int64_t> &shape, aclDataType type,
                     void **device, aclTensor **tensor)
{
    const size_t bytes = host.size();
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = shape[index + 1] * strides[index + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *device);
    return *tensor == nullptr ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

int CreateBlockTableTensor(const std::vector<int32_t> &host, const std::vector<int64_t> &shape, void **device,
                           aclTensor **tensor)
{
    const size_t bytes = host.size() * sizeof(int32_t);
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ret;
    }
    const int64_t strides[] = {shape[1], 1};
    *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, strides, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *device);
    return *tensor == nullptr ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

void DestroyTensor(aclTensor *tensor, void *device)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (device != nullptr) {
        aclrtFree(device);
    }
}

std::string Trim(const std::string &text)
{
    const size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::vector<std::string> Split(const std::string &text, char delimiter)
{
    std::vector<std::string> values;
    std::stringstream stream(text);
    std::string value;
    while (std::getline(stream, value, delimiter)) {
        values.push_back(Trim(value));
    }
    return values;
}

bool ParseInt64(const std::string &text, int64_t *value)
{
    errno = 0;
    char *end = nullptr;
    const long long parsed = std::strtoll(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        return false;
    }
    *value = static_cast<int64_t>(parsed);
    return true;
}

bool ParseInt32List(const std::string &text, std::vector<int32_t> *values)
{
    values->clear();
    for (const std::string &token : Split(text, ';')) {
        int64_t value = 0;
        if (!ParseInt64(token, &value) || value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        values->push_back(static_cast<int32_t>(value));
    }
    return !values->empty();
}

bool ParseInt64List(const std::string &text, std::vector<int64_t> *values)
{
    values->clear();
    for (const std::string &token : Split(text, ';')) {
        int64_t value = 0;
        if (!ParseInt64(token, &value)) {
            return false;
        }
        values->push_back(value);
    }
    return !values->empty();
}

bool ParseLayout(const std::string &text, KvLayout *layout)
{
    if (text == "nz") {
        *layout = KvLayout::PA_NZ;
        return true;
    }
    if (text == "bnbd") {
        *layout = KvLayout::PA_BNBD;
        return true;
    }
    if (text == "bbh") {
        *layout = KvLayout::PA_BBH;
        return true;
    }
    return false;
}

bool ParseQType(const std::string &text, QType *type)
{
    if (text == "fp16") {
        *type = QType::FP16;
        return true;
    }
    if (text == "bf16") {
        *type = QType::BF16;
        return true;
    }
    return false;
}

bool ParseCase(const std::vector<std::string> &fields, CaseConfig *config)
{
    constexpr size_t FIELD_COUNT = 12;
    constexpr size_t SEEDED_FIELD_COUNT = 13;
    constexpr size_t GQA_FIELD_COUNT = 14;
    constexpr size_t PROFILE_FIELD_COUNT = 15;
    if ((fields.size() != FIELD_COUNT && fields.size() != SEEDED_FIELD_COUNT && fields.size() != GQA_FIELD_COUNT &&
         fields.size() != PROFILE_FIELD_COUNT) ||
        !ParseLayout(fields[1], &config->kvLayout) ||
        !ParseQType(fields[2], &config->qType) || (fields[3] != "0" && fields[3] != "1")) {
        return false;
    }
    config->name = fields[0];
    config->withOffset = fields[3] == "1";
    if (!ParseInt64(fields[4], &config->b) || !ParseInt64(fields[5], &config->n) ||
        !ParseInt64(fields[6], &config->s1) || !ParseInt64(fields[7], &config->d) ||
        !ParseInt64(fields[8], &config->blockSize) || !ParseInt64(fields[9], &config->totalBlockNum) ||
        !ParseInt64List(fields[10], &config->actualKvLens) || !ParseInt32List(fields[11], &config->blockTable)) {
        return false;
    }
    config->seed = 0;
    config->qHeads = config->n;
    config->randomProfile = RandomProfile::LEGACY;
    if (fields.size() >= SEEDED_FIELD_COUNT) {
        int64_t parsedSeed = 0;
        if (!ParseInt64(fields[12], &parsedSeed) || parsedSeed <= 0) {
            return false;
        }
        config->seed = static_cast<uint64_t>(parsedSeed);
        config->randomProfile = RandomProfile::STABLE;
    }
    if (fields.size() >= GQA_FIELD_COUNT &&
        (!ParseInt64(fields[13], &config->qHeads) || config->qHeads <= 0 || config->qHeads % config->n != 0)) {
        return false;
    }
    if (fields.size() == PROFILE_FIELD_COUNT) {
        if (fields[14] != "wide") {
            return false;
        }
        config->randomProfile = RandomProfile::WIDE;
    }
    return true;
}

bool LoadCasesFromFile(const char *path, std::vector<CaseConfig> *cases)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    cases->clear();
    std::string line;
    uint32_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        const std::vector<std::string> fields = Split(trimmed, ',');
        if (!fields.empty() && fields[0] == "name") {
            continue;
        }
        CaseConfig config;
        if (!ParseCase(fields, &config)) {
            std::printf("[MXFP4_PA] invalid CSV row %u in %s\n", lineNumber, path);
            return false;
        }
        cases->push_back(std::move(config));
    }
    if (cases->empty()) {
        std::printf("[MXFP4_PA] no cases found in %s\n", path);
        return false;
    }
    std::printf("[MXFP4_PA] loaded %zu cases from %s\n", cases->size(), path);
    return true;
}

bool LoadCases(std::vector<CaseConfig> *cases)
{
    const char *fromEnv = std::getenv("FIA_CASE_FILE");
    if (fromEnv != nullptr) {
        if (!LoadCasesFromFile(fromEnv, cases)) {
            std::printf("[MXFP4_PA] FIA_CASE_FILE could not be loaded: %s\n", fromEnv);
            return false;
        }
        return true;
    }
    constexpr std::array<const char *, 2> DEFAULT_PATHS = {
        "attention/fused_infer_attention_score/examples/arch35/data/fused_infer_attention_score_mxfp4_pa_cases.csv",
        "../attention/fused_infer_attention_score/examples/arch35/data/fused_infer_attention_score_mxfp4_pa_cases.csv",
    };
    for (const char *path : DEFAULT_PATHS) {
        if (LoadCasesFromFile(path, cases)) {
            return true;
        }
    }
    std::printf("[MXFP4_PA] set FIA_CASE_FILE to the MXFP4 PA CSV path\n");
    return false;
}

int64_t CanonicalIndex(const CaseConfig &config, int64_t p, int64_t n, int64_t token, int64_t d)
{
    return (((p * config.n + n) * config.blockSize + token) * config.d + d);
}

int64_t KvLogicalIndex(const CaseConfig &config, int64_t p, int64_t n, int64_t token, int64_t d)
{
    if (config.kvLayout == KvLayout::PA_NZ) {
        return ((((p * config.n + n) * (config.d / D0) + d / D0) * config.blockSize + token) * D0 + d % D0);
    }
    if (config.kvLayout == KvLayout::PA_BNBD) {
        return (((p * config.n + n) * config.blockSize + token) * config.d + d);
    }
    return (((p * config.blockSize + token) * config.n + n) * config.d + d);
}

int64_t ParamIndex(const CaseConfig &config, int64_t p, int64_t n, int64_t group, int64_t token)
{
    const int64_t groupNum = config.d / GROUP_SIZE;
    if (config.kvLayout == KvLayout::PA_NZ) {
        return ((p * config.n + n) * groupNum + group) * config.blockSize + token;
    }
    if (config.kvLayout == KvLayout::PA_BNBD) {
        return ((p * config.n + n) * config.blockSize + token) * groupNum + group;
    }
    return ((p * config.blockSize + token) * config.n + n) * groupNum + group;
}

void SetPackedFp4(std::vector<uint8_t> *packed, int64_t logicalIndex, uint8_t code)
{
    uint8_t &byte = (*packed)[static_cast<size_t>(logicalIndex / 2)];
    const uint8_t shift = static_cast<uint8_t>((logicalIndex & 1) * 4);
    byte = static_cast<uint8_t>((byte & static_cast<uint8_t>(~(0xFU << shift))) | ((code & 0xFU) << shift));
}

bool ValidateCase(const CaseConfig &config, int64_t *blocksPerBatch)
{
    if (config.b <= 0 || config.n <= 0 || config.qHeads <= 0 || config.qHeads % config.n != 0 || config.s1 <= 0 ||
        config.d <= 0 || config.blockSize <= 0 ||
        config.totalBlockNum <= 0 || config.d % GROUP_SIZE != 0 || config.d % D0 != 0) {
        return false;
    }
    if (config.actualKvLens.size() != static_cast<size_t>(config.b) ||
        config.blockTable.size() % static_cast<size_t>(config.b) != 0) {
        return false;
    }
    *blocksPerBatch = static_cast<int64_t>(config.blockTable.size() / static_cast<size_t>(config.b));
    if (*blocksPerBatch <= 0) {
        return false;
    }
    for (int64_t length : config.actualKvLens) {
        if (length <= 0 || length > *blocksPerBatch * config.blockSize) {
            return false;
        }
    }
    for (int32_t physicalBlock : config.blockTable) {
        if (physicalBlock < 0 || physicalBlock >= config.totalBlockNum) {
            return false;
        }
    }
    return true;
}

bool RunCase(aclrtStream stream, const CaseConfig &config)
{
    int64_t blocksPerBatch = 0;
    if (!ValidateCase(config, &blocksPerBatch)) {
        std::printf("[MXFP4_PA][%s] invalid case configuration\n", config.name.c_str());
        return false;
    }
    const int64_t groupNum = config.d / GROUP_SIZE;
    const std::vector<int64_t> queryShape = {config.b, config.qHeads, config.s1, config.d};
    const std::vector<int64_t> kvShape = config.kvLayout == KvLayout::PA_NZ
        ? std::vector<int64_t>{config.totalBlockNum, config.n, config.d / D0, config.blockSize, D0}
        : config.kvLayout == KvLayout::PA_BNBD
            ? std::vector<int64_t>{config.totalBlockNum, config.n, config.blockSize, config.d}
            : std::vector<int64_t>{config.totalBlockNum, config.blockSize, config.n * config.d};
    const std::vector<int64_t> antiqParamShape = config.kvLayout == KvLayout::PA_NZ
        ? std::vector<int64_t>{config.totalBlockNum, config.n, groupNum, config.blockSize}
        : config.kvLayout == KvLayout::PA_BNBD
            ? std::vector<int64_t>{config.totalBlockNum, config.n, config.blockSize, groupNum}
            : std::vector<int64_t>{config.totalBlockNum, config.blockSize, config.n * groupNum};
    const std::vector<int64_t> blockTableShape = {config.b, blocksPerBatch};
    const std::vector<int64_t> actualQLens(static_cast<size_t>(config.b), config.s1);

    std::printf("[MXFP4_PA][%s][%s] begin qType=%s offset=%s B=%ld QH=%ld KVH=%ld S1=%ld D=%ld G=%ld Bs=%ld P=%ld seed=%llu profile=%s\n",
                LayoutName(config.kvLayout), config.name.c_str(), QTypeName(config.qType),
                config.withOffset ? "yes" : "no", config.b, config.qHeads, config.n, config.s1, config.d, groupNum,
                config.blockSize, config.totalBlockNum, static_cast<unsigned long long>(config.seed),
                RandomProfileName(config.randomProfile));

    std::vector<uint16_t> query(static_cast<size_t>(ShapeSize(queryShape)));
    for (int64_t b = 0; b < config.b; ++b) {
        for (int64_t n = 0; n < config.qHeads; ++n) {
            for (int64_t s1 = 0; s1 < config.s1; ++s1) {
                for (int64_t d = 0; d < config.d; ++d) {
                    const int64_t index = ((b * config.qHeads + n) * config.s1 + s1) * config.d + d;
                    query[static_cast<size_t>(index)] = EncodeQ(QueryValue(config, b, n, s1, d), config.qType);
                }
            }
        }
    }
    std::vector<uint16_t> output(static_cast<size_t>(ShapeSize(queryShape)), EncodeQ(0.0F, config.qType));
    std::vector<uint8_t> canonicalKey(static_cast<size_t>(config.totalBlockNum * config.n * config.blockSize * config.d));
    std::vector<uint8_t> canonicalValue(canonicalKey.size());
    std::vector<uint8_t> key(static_cast<size_t>(ShapeSize(kvShape) / 2), 0);
    std::vector<uint8_t> value(key.size(), 0);
    for (int64_t p = 0; p < config.totalBlockNum; ++p) {
        for (int64_t n = 0; n < config.n; ++n) {
            for (int64_t token = 0; token < config.blockSize; ++token) {
                for (int64_t d = 0; d < config.d; ++d) {
                    const int64_t canonicalIndex = CanonicalIndex(config, p, n, token, d);
                    const int64_t kvIndex = KvLogicalIndex(config, p, n, token, d);
                    canonicalKey[static_cast<size_t>(canonicalIndex)] = Fp4Code(config, p, n, token, d, 0);
                    canonicalValue[static_cast<size_t>(canonicalIndex)] = Fp4Code(config, p, n, token, d, 9);
                    SetPackedFp4(&key, kvIndex, canonicalKey[static_cast<size_t>(canonicalIndex)]);
                    SetPackedFp4(&value, kvIndex, canonicalValue[static_cast<size_t>(canonicalIndex)]);
                }
            }
        }
    }

    std::vector<uint8_t> keyScale(static_cast<size_t>(ShapeSize(antiqParamShape)));
    std::vector<uint8_t> valueScale(keyScale.size());
    std::vector<uint16_t> keyOffset(keyScale.size(), EncodeQ(0.0F, config.qType));
    std::vector<uint16_t> valueOffset(keyScale.size(), EncodeQ(0.0F, config.qType));
    for (int64_t p = 0; p < config.totalBlockNum; ++p) {
        for (int64_t n = 0; n < config.n; ++n) {
            for (int64_t group = 0; group < groupNum; ++group) {
                for (int64_t token = 0; token < config.blockSize; ++token) {
                    const int64_t index = ParamIndex(config, p, n, group, token);
                    keyScale[static_cast<size_t>(index)] = ScaleCode(config, p, n, group, token, 0);
                    valueScale[static_cast<size_t>(index)] = ScaleCode(config, p, n, group, token, 1);
                    keyOffset[static_cast<size_t>(index)] = EncodeQ(OffsetValue(config, p, n, group, token, 0), config.qType);
                    valueOffset[static_cast<size_t>(index)] = EncodeQ(OffsetValue(config, p, n, group, token, 1), config.qType);
                }
            }
        }
    }

    void *queryDevice = nullptr;
    void *keyDevice = nullptr;
    void *valueDevice = nullptr;
    void *keyScaleDevice = nullptr;
    void *valueScaleDevice = nullptr;
    void *keyOffsetDevice = nullptr;
    void *valueOffsetDevice = nullptr;
    void *blockTableDevice = nullptr;
    void *outputDevice = nullptr;
    void *workspace = nullptr;
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *keyScaleTensor = nullptr;
    aclTensor *valueScaleTensor = nullptr;
    aclTensor *keyOffsetTensor = nullptr;
    aclTensor *valueOffsetTensor = nullptr;
    aclTensor *blockTableTensor = nullptr;
    aclTensor *outputTensor = nullptr;
    aclTensorList *keyList = nullptr;
    aclTensorList *valueList = nullptr;
    aclIntArray *actualQArray = nullptr;
    aclIntArray *actualKvArray = nullptr;
    aclOpExecutor *executor = nullptr;

    const auto cleanup = [&]() {
        if (keyList != nullptr) aclDestroyTensorList(keyList);
        if (valueList != nullptr) aclDestroyTensorList(valueList);
        if (actualQArray != nullptr) aclDestroyIntArray(actualQArray);
        if (actualKvArray != nullptr) aclDestroyIntArray(actualKvArray);
        DestroyTensor(queryTensor, queryDevice);
        DestroyTensor(keyTensor, keyDevice);
        DestroyTensor(valueTensor, valueDevice);
        DestroyTensor(keyScaleTensor, keyScaleDevice);
        DestroyTensor(valueScaleTensor, valueScaleDevice);
        DestroyTensor(keyOffsetTensor, keyOffsetDevice);
        DestroyTensor(valueOffsetTensor, valueOffsetDevice);
        DestroyTensor(blockTableTensor, blockTableDevice);
        DestroyTensor(outputTensor, outputDevice);
        if (workspace != nullptr) aclrtFree(workspace);
    };

    int ret = CreateTensor(query, queryShape, ToAclType(config.qType), &queryDevice, &queryTensor);
    if (ret == ACL_SUCCESS) ret = CreateByteTensor(key, kvShape, aclDataType::ACL_FLOAT4_E2M1, &keyDevice, &keyTensor);
    if (ret == ACL_SUCCESS) ret = CreateByteTensor(value, kvShape, aclDataType::ACL_FLOAT4_E2M1, &valueDevice, &valueTensor);
    if (ret == ACL_SUCCESS) ret = CreateByteTensor(keyScale, antiqParamShape, aclDataType::ACL_FLOAT8_E8M0,
                                                     &keyScaleDevice, &keyScaleTensor);
    if (ret == ACL_SUCCESS) ret = CreateByteTensor(valueScale, antiqParamShape, aclDataType::ACL_FLOAT8_E8M0,
                                                     &valueScaleDevice, &valueScaleTensor);
    if (ret == ACL_SUCCESS && config.withOffset) ret = CreateTensor(keyOffset, antiqParamShape, ToAclType(config.qType),
                                                                      &keyOffsetDevice, &keyOffsetTensor);
    if (ret == ACL_SUCCESS && config.withOffset) ret = CreateTensor(valueOffset, antiqParamShape, ToAclType(config.qType),
                                                                      &valueOffsetDevice, &valueOffsetTensor);
    if (ret == ACL_SUCCESS) ret = CreateBlockTableTensor(config.blockTable, blockTableShape, &blockTableDevice,
                                                          &blockTableTensor);
    if (ret == ACL_SUCCESS) ret = CreateTensor(output, queryShape, ToAclType(config.qType), &outputDevice, &outputTensor);
    if (ret != ACL_SUCCESS) {
        std::printf("[MXFP4_PA][%s] create input tensor failed: %d\n", config.name.c_str(), ret);
        cleanup();
        return false;
    }

    keyList = aclCreateTensorList(&keyTensor, 1);
    valueList = aclCreateTensorList(&valueTensor, 1);
    actualQArray = aclCreateIntArray(actualQLens.data(), actualQLens.size());
    actualKvArray = aclCreateIntArray(config.actualKvLens.data(), config.actualKvLens.size());
    if (keyList == nullptr || valueList == nullptr || actualQArray == nullptr || actualKvArray == nullptr) {
        std::printf("[MXFP4_PA][%s] create tensor list or sequence-length array failed\n", config.name.c_str());
        cleanup();
        return false;
    }

    uint64_t workspaceSize = 0;
    char inputLayout[] = "BNSD";
    std::printf("[MXFP4_PA][%s] before GetWorkspaceSize (mode=6, layout=%s)\n", config.name.c_str(), inputLayout);
    ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
        queryTensor, keyList, valueList, nullptr, nullptr, actualQArray, actualKvArray,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        blockTableTensor, nullptr, nullptr,
        keyScaleTensor, config.withOffset ? keyOffsetTensor : nullptr,
        valueScaleTensor, config.withOffset ? valueOffsetTensor : nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        config.qHeads, 1.0 / std::sqrt(static_cast<double>(config.d)), 65536, 65536, inputLayout, config.n, 0, 1,
        config.blockSize, 0, false, 6, 6, 0, 0, outputTensor, nullptr, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::printf("[MXFP4_PA][%s] GetWorkspaceSize failed: %d\n", config.name.c_str(), ret);
        cleanup();
        return false;
    }
    std::printf("[MXFP4_PA][%s] GetWorkspaceSize succeeded workspace=%lu executor=%p\n", config.name.c_str(),
                workspaceSize, static_cast<void *>(executor));
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::printf("[MXFP4_PA][%s] workspace allocation failed: %d\n", config.name.c_str(), ret);
            cleanup();
            return false;
        }
    }
    ret = aclnnFusedInferAttentionScoreV5(workspace, workspaceSize, executor, stream);
    std::printf("[MXFP4_PA][%s] kernel launch returned %d\n", config.name.c_str(), ret);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
        std::printf("[MXFP4_PA][%s] stream synchronize returned %d\n", config.name.c_str(), ret);
    }
    if (ret != ACL_SUCCESS) {
        std::printf("[MXFP4_PA][%s] device execution failed: %d\n", config.name.c_str(), ret);
        cleanup();
        return false;
    }
    ret = aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), outputDevice,
                      output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::printf("[MXFP4_PA][%s] output D2H copy failed: %d\n", config.name.c_str(), ret);
        cleanup();
        return false;
    }

    float maxError = 0.0F;
    float maxReferenceOutputCastDelta = 0.0F;
    int64_t bitwiseMismatchCount = 0;
    int64_t overOneUlpCount = 0;
    int32_t maxUlpDistance = 0;
    int64_t maxErrorB = -1;
    int64_t maxErrorQHead = -1;
    int64_t maxErrorS1 = -1;
    int64_t maxErrorD = -1;
    float maxErrorActual = 0.0F;
    float maxErrorExpectedAccumulated = 0.0F;
    float maxErrorExpectedOutput = 0.0F;
    uint16_t maxErrorActualBits = 0;
    uint16_t maxErrorExpectedBits = 0;
    for (int64_t b = 0; b < config.b; ++b) {
        const int64_t validS = config.actualKvLens[static_cast<size_t>(b)];
        for (int64_t n = 0; n < config.qHeads; ++n) {
            const int64_t kvHead = n / (config.qHeads / config.n);
            for (int64_t s1 = 0; s1 < config.s1; ++s1) {
                std::vector<float> expScores(static_cast<size_t>(validS));
                float maxScore = -INFINITY;
                for (int64_t s2 = 0; s2 < validS; ++s2) {
                    const int64_t p = config.blockTable[static_cast<size_t>(b * blocksPerBatch + s2 / config.blockSize)];
                    const int64_t token = s2 % config.blockSize;
                    float score = 0.0F;
                    for (int64_t d = 0; d < config.d; ++d) {
                        const int64_t group = d / GROUP_SIZE;
                        const int64_t paramIndex = ParamIndex(config, p, kvHead, group, token);
                        const uint16_t q = query[static_cast<size_t>(((b * config.qHeads + n) * config.s1 + s1) * config.d + d)];
                        const uint8_t kCode = canonicalKey[static_cast<size_t>(CanonicalIndex(config, p, kvHead, token, d))];
                        const float keyValue = DequantizeReference(kCode, keyScale[static_cast<size_t>(paramIndex)],
                                                                   keyOffset[static_cast<size_t>(paramIndex)],
                                                                   config.withOffset, config.qType);
                        score += DecodeQ(q, config.qType) * keyValue;
                    }
                    expScores[static_cast<size_t>(s2)] = score / std::sqrt(static_cast<float>(config.d));
                    maxScore = std::max(maxScore, expScores[static_cast<size_t>(s2)]);
                }
                float sumExp = 0.0F;
                for (float &score : expScores) {
                    score = std::exp(score - maxScore);
                    sumExp += score;
                }
                for (int64_t d = 0; d < config.d; ++d) {
                    const int64_t group = d / GROUP_SIZE;
                    float expected = 0.0F;
                    for (int64_t s2 = 0; s2 < validS; ++s2) {
                        const int64_t p = config.blockTable[static_cast<size_t>(b * blocksPerBatch + s2 / config.blockSize)];
                        const int64_t token = s2 % config.blockSize;
                        const int64_t paramIndex = ParamIndex(config, p, kvHead, group, token);
                        const uint8_t vCode = canonicalValue[static_cast<size_t>(CanonicalIndex(config, p, kvHead, token, d))];
                        const float valueValue = DequantizeReference(vCode, valueScale[static_cast<size_t>(paramIndex)],
                                                                     valueOffset[static_cast<size_t>(paramIndex)],
                                                                     config.withOffset, config.qType);
                        expected += (expScores[static_cast<size_t>(s2)] / sumExp) * valueValue;
                    }
                    // outputTensor has Q_T dtype. Compare to the reference after
                    // its required final output conversion, rather than to the
                    // higher-precision host accumulator.
                    const float expectedOutput = DecodeQ(EncodeQ(expected, config.qType), config.qType);
                    maxReferenceOutputCastDelta = std::max(maxReferenceOutputCastDelta,
                                                           std::fabs(expectedOutput - expected));
                    const int64_t outIndex = ((b * config.qHeads + n) * config.s1 + s1) * config.d + d;
                    const uint16_t actualBits = output[static_cast<size_t>(outIndex)];
                    const float actual = DecodeQ(actualBits, config.qType);
                    const float error = std::fabs(actual - expectedOutput);
                    const uint16_t expectedBits = EncodeQ(expected, config.qType);
                    const int32_t ulpDistance = std::abs(OrderedLowPrecisionBits(actualBits) -
                                                         OrderedLowPrecisionBits(expectedBits));
                    maxUlpDistance = std::max(maxUlpDistance, ulpDistance);
                    if (actualBits != expectedBits) {
                        ++bitwiseMismatchCount;
                    }
                    if (ulpDistance > 1) {
                        ++overOneUlpCount;
                    }
                    if (error > maxError) {
                        maxError = error;
                        maxErrorB = b;
                        maxErrorQHead = n;
                        maxErrorS1 = s1;
                        maxErrorD = d;
                        maxErrorActual = actual;
                        maxErrorExpectedAccumulated = expected;
                        maxErrorExpectedOutput = expectedOutput;
                        maxErrorActualBits = actualBits;
                        maxErrorExpectedBits = expectedBits;
                    }
                }
            }
        }
    }
    const float strictAbsoluteTolerance = config.qType == QType::FP16 ? 0.04F : 0.08F;
    std::printf("[MXFP4_PA][%s] max_abs_error=%f strict_absolute_tolerance=%f "
                "reference_output_cast_delta=%f\n",
                config.name.c_str(), maxError, strictAbsoluteTolerance, maxReferenceOutputCastDelta);
    std::printf("[MXFP4_PA][%s] bitwise_mismatch=%ld max_ulp_distance=%d over_one_ulp=%ld\n",
                config.name.c_str(), bitwiseMismatchCount, maxUlpDistance, overOneUlpCount);
    std::printf("[MXFP4_PA][%s] worst_error b=%ld qh=%ld s1=%ld d=%ld actual=%f (0x%04x) "
                "expected_accumulated=%f expected_output=%f (0x%04x)\n",
                config.name.c_str(), maxErrorB, maxErrorQHead, maxErrorS1, maxErrorD, maxErrorActual,
                maxErrorActualBits, maxErrorExpectedAccumulated, maxErrorExpectedOutput, maxErrorExpectedBits);
    cleanup();
    return maxError <= strictAbsoluteTolerance;
}

} // namespace

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<CaseConfig> cases;
    if (!LoadCases(&cases)) {
        return 2;
    }
    const char *caseFilter = std::getenv("FIA_CASE");
    if (caseFilter != nullptr) {
        std::printf("[MXFP4_PA] FIA_CASE=%s\n", caseFilter);
    }

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::printf("[MXFP4_PA] aclInit failed: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        std::printf("[MXFP4_PA] aclrtSetDevice failed: %d\n", ret);
        aclFinalize();
        return ret;
    }
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::printf("[MXFP4_PA] aclrtCreateStream failed: %d\n", ret);
        aclrtResetDevice(0);
        aclFinalize();
        return ret;
    }

    bool passed = true;
    bool matched = false;
    for (const CaseConfig &config : cases) {
        if (caseFilter != nullptr && config.name != caseFilter) {
            continue;
        }
        matched = true;
        passed = RunCase(stream, config) && passed;
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    if (!matched) {
        std::printf("[MXFP4_PA] no case matched FIA_CASE\n");
        return 2;
    }
    std::printf("MXFP4 mode-6 PageAttention numerical example %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
