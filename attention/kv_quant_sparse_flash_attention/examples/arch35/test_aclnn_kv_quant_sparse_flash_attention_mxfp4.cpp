/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_kv_quant_sparse_flash_attention.h"

namespace {

constexpr int64_t kNopeDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kQueryDim = kNopeDim + kRopeDim;
constexpr int64_t kMxfp4RecordBytes = 400;
constexpr int64_t kLegacyRecordBytes = 656;
constexpr int64_t kFp4Bytes = 256;
constexpr int64_t kMxfp4RopeOffset = 256;
constexpr int64_t kMxfp4ScaleOffset = 384;
constexpr int64_t kLegacyRopeOffset = 512;
constexpr int64_t kLegacyScaleOffset = 640;
constexpr int64_t kScaleCount = 16;
constexpr int64_t kMxGroupSize = 32;
constexpr int64_t kPaBlockSize = 16;

enum class QKind { BF16, FP16 };
enum class KvLayout { BSND, TND, PA_BSND };
enum class RecordKind { MXFP4, LEGACY_INT8, RAW };

struct CaseSpec {
    std::string name;
    QKind qKind;
    KvLayout layout;
    RecordKind recordKind;
    bool queryTnd;
    int64_t n1;
    std::vector<int64_t> kvLengths;
    std::vector<std::vector<int32_t>> sparseRows;
    int64_t storageOffsetElements;
    bool edgeScales;
    bool bitExact;
    bool expectWorkspaceFailure;
    aclDataType keyDtype;
    int64_t keyDim;
};

struct ResourceSet {
    std::vector<void *> deviceBuffers;
    std::vector<aclTensor *> tensors;

    ~ResourceSet()
    {
        for (auto *tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto *buffer : deviceBuffers) {
            aclrtFree(buffer);
        }
    }
};

struct PreparedCase {
    std::vector<uint16_t> query;
    std::vector<int8_t> key;
    std::vector<int8_t> value;
    std::vector<int32_t> sparseIndices;
    std::vector<int32_t> actualSeqQ;
    std::vector<int32_t> actualSeqKv;
    std::vector<int32_t> blockTable;
    std::vector<uint16_t> output;
    std::vector<uint16_t> expected;
    std::vector<int64_t> queryShape;
    std::vector<int64_t> keyViewShape;
    std::vector<int64_t> keyStorageShape;
    std::vector<int64_t> valueViewShape;
    std::vector<int64_t> valueStorageShape;
    std::vector<int64_t> sparseShape;
    std::vector<int64_t> outputShape;
    std::vector<std::vector<std::vector<int8_t>>> logicalRecords;
};

int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

uint32_t FloatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float BitsFloat(uint32_t bits)
{
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = FloatBits(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float Bf16ToFloat(uint16_t value)
{
    return BitsFloat(static_cast<uint32_t>(value) << 16U);
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = FloatBits(value);
    uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    uint32_t exponentBits = (bits >> 23U) & 0xFFU;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponentBits == 0xFFU) {
        return static_cast<uint16_t>(sign | (mantissa == 0U ? 0x7C00U : 0x7E00U));
    }
    int32_t exponent = static_cast<int32_t>(exponentBits) - 127;
    if (exponent > 15) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (exponent >= -14) {
        uint32_t rounded = mantissa + 0xFFFU + ((mantissa >> 13U) & 1U);
        if ((rounded & 0x800000U) != 0U) {
            rounded = 0U;
            ++exponent;
            if (exponent > 15) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent + 15) << 10U) |
                                     static_cast<uint16_t>(rounded >> 13U));
    }
    if (exponent < -24) {
        return sign;
    }
    mantissa |= 0x800000U;
    uint32_t shift = static_cast<uint32_t>(-exponent - 1);
    uint32_t roundBias = (1U << (shift - 1U)) - 1U + ((mantissa >> shift) & 1U);
    return static_cast<uint16_t>(sign | static_cast<uint16_t>((mantissa + roundBias) >> shift));
}

float HalfToFloat(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1FU;
    uint32_t mantissa = value & 0x3FFU;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            return BitsFloat(sign);
        }
        int32_t unbiased = -14;
        while ((mantissa & 0x400U) == 0U) {
            mantissa <<= 1U;
            --unbiased;
        }
        mantissa &= 0x3FFU;
        return BitsFloat(sign | (static_cast<uint32_t>(unbiased + 127) << 23U) | (mantissa << 13U));
    }
    if (exponent == 0x1FU) {
        return BitsFloat(sign | 0x7F800000U | (mantissa << 13U));
    }
    return BitsFloat(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

uint16_t EncodeQ(float value, QKind kind)
{
    return kind == QKind::BF16 ? FloatToBf16(value) : FloatToHalf(value);
}

float DecodeQ(uint16_t value, QKind kind)
{
    return kind == QKind::BF16 ? Bf16ToFloat(value) : HalfToFloat(value);
}

float RoundBf16(float value)
{
    return Bf16ToFloat(FloatToBf16(value));
}

float RoundQ(float value, QKind kind)
{
    return DecodeQ(EncodeQ(value, kind), kind);
}

float CubeInputValue(float value, QKind kind)
{
    if (kind != QKind::BF16) {
        return value;
    }
    uint16_t bits = FloatToBf16(value);
    if ((bits & 0x7F80U) == 0U) {
        bits &= 0x8000U;
    }
    return Bf16ToFloat(bits);
}

float Fp4E2M1Value(uint8_t code)
{
    float sign = (code & 0x8U) == 0U ? 1.0F : -1.0F;
    uint8_t exponent = (code >> 1U) & 0x3U;
    uint8_t mantissa = code & 0x1U;
    if (exponent == 0U) {
        return sign * static_cast<float>(mantissa) * 0.5F;
    }
    return sign * std::ldexp(1.0F + static_cast<float>(mantissa) * 0.5F,
                             static_cast<int>(exponent) - 1);
}

float E8M0Value(uint8_t code)
{
    if (code == 0xFFU) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::ldexp(1.0F, static_cast<int>(code) - 127);
}

void StoreQ(std::vector<int8_t> &record, int64_t offset, float value, QKind kind)
{
    uint16_t bits = EncodeQ(value, kind);
    record[offset] = static_cast<int8_t>(bits & 0xFFU);
    record[offset + 1] = static_cast<int8_t>(bits >> 8U);
}

float LoadQ(const int8_t *record, int64_t offset, QKind kind)
{
    uint16_t bits = static_cast<uint8_t>(record[offset]);
    bits |= static_cast<uint16_t>(static_cast<uint8_t>(record[offset + 1])) << 8U;
    return DecodeQ(bits, kind);
}

std::vector<int8_t> MakeMxfp4Record(int64_t token, QKind kind, bool edgeScales)
{
    std::vector<int8_t> record(kMxfp4RecordBytes, 0);
    for (int64_t d = 0; d < kNopeDim; d += 2) {
        uint8_t low = edgeScales ? 0x2U : static_cast<uint8_t>((token + d) % 16);
        uint8_t high = edgeScales ? 0x2U : static_cast<uint8_t>((token + d + 1) % 16);
        record[d / 2] = static_cast<int8_t>(low | static_cast<uint8_t>(high << 4U));
    }
    for (int64_t d = 0; d < kRopeDim; ++d) {
        float value = edgeScales ? 0.0F
                                 : 0.015625F * static_cast<float>(token + 1) * static_cast<float>((d % 9) - 4);
        StoreQ(record, kMxfp4RopeOffset + d * 2, value, kind);
    }
    for (int64_t group = 0; group < kScaleCount; ++group) {
        uint8_t scale = static_cast<uint8_t>(126 + ((token + group) % 4));
        if (edgeScales) {
            scale = group == 0 ? 0U : (group == 1 ? 254U : 127U);
        }
        record[kMxfp4ScaleOffset + group] = static_cast<int8_t>(scale);
    }
    return record;
}

std::vector<int8_t> MakeLegacyRecord(int64_t token, QKind kind)
{
    std::vector<int8_t> record(kLegacyRecordBytes, 0);
    for (int64_t d = 0; d < kNopeDim; ++d) {
        record[d] = static_cast<int8_t>(((token + d) % 15) - 7);
    }
    for (int64_t d = 0; d < kRopeDim; ++d) {
        StoreQ(record, kLegacyRopeOffset + d * 2,
               0.015625F * static_cast<float>(token + 1) * static_cast<float>((d % 7) - 3), kind);
    }
    for (int64_t tile = 0; tile < 4; ++tile) {
        float scale = 0.5F + 0.25F * static_cast<float>((token + tile) % 4);
        std::memcpy(record.data() + kLegacyScaleOffset + tile * sizeof(float), &scale, sizeof(float));
    }
    return record;
}

std::vector<int8_t> MakeRawRecord(int64_t token, int64_t bytes)
{
    std::vector<int8_t> record(bytes);
    for (int64_t i = 0; i < bytes; ++i) {
        record[i] = static_cast<int8_t>((token * 17 + i) & 0x7F);
    }
    return record;
}

std::vector<float> DecodeMxfp4Record(const int8_t *record, QKind kind)
{
    std::vector<float> decoded(kQueryDim, 0.0F);
    for (int64_t byte = 0; byte < kFp4Bytes; ++byte) {
        uint8_t packed = static_cast<uint8_t>(record[byte]);
        int64_t group = (byte * 2) / kMxGroupSize;
        float scale = RoundBf16(E8M0Value(static_cast<uint8_t>(record[kMxfp4ScaleOffset + group])));
        float low = RoundBf16(Fp4E2M1Value(packed & 0xFU) * scale);
        float high = RoundBf16(Fp4E2M1Value(packed >> 4U) * scale);
        decoded[byte * 2] = kind == QKind::BF16 ? low : RoundQ(low, QKind::FP16);
        decoded[byte * 2 + 1] = kind == QKind::BF16 ? high : RoundQ(high, QKind::FP16);
    }
    for (int64_t d = 0; d < kRopeDim; ++d) {
        decoded[kNopeDim + d] = LoadQ(record, kMxfp4RopeOffset + d * 2, kind);
    }
    return decoded;
}

std::vector<float> DecodeLegacyRecord(const int8_t *record, QKind kind)
{
    std::vector<float> decoded(kQueryDim, 0.0F);
    for (int64_t d = 0; d < kNopeDim; ++d) {
        float scale = 0.0F;
        std::memcpy(&scale, record + kLegacyScaleOffset + (d / 128) * sizeof(float), sizeof(float));
        decoded[d] = RoundQ(static_cast<float>(record[d]) * scale, kind);
    }
    for (int64_t d = 0; d < kRopeDim; ++d) {
        decoded[kNopeDim + d] = LoadQ(record, kLegacyRopeOffset + d * 2, kind);
    }
    return decoded;
}

std::vector<float> DecodeRecord(const std::vector<int8_t> &record, const CaseSpec &spec)
{
    return spec.recordKind == RecordKind::MXFP4 ? DecodeMxfp4Record(record.data(), spec.qKind)
                                                : DecodeLegacyRecord(record.data(), spec.qKind);
}

std::vector<int32_t> RangeSparse(int64_t count, bool appendEnd)
{
    std::vector<int32_t> result(count + (appendEnd ? 1 : 0));
    std::iota(result.begin(), result.begin() + count, 0);
    if (appendEnd) {
        result.back() = -1;
    }
    return result;
}

std::vector<CaseSpec> BuildCases()
{
    return {
        {"mx_bs_bf16_g1_full16_offset32", QKind::BF16, KvLayout::BSND, RecordKind::MXFP4, false, 1, {16},
         {RangeSparse(16, false)}, 32, false, false, false, ACL_INT8, 400},
        {"mx_bs_fp16_g1_tail3_offset0", QKind::FP16, KvLayout::BSND, RecordKind::MXFP4, false, 1, {3},
         {{0, 1, 2, -1}}, 0, false, false, false, ACL_INT8, 400},
        {"mx_bs_bf16_g1_s2_33_offset1", QKind::BF16, KvLayout::BSND, RecordKind::MXFP4, false, 1, {33},
         {RangeSparse(33, false)}, 1, false, false, false, ACL_INT8, 400},
        {"mx_bs_bf16_g128_split_full16", QKind::BF16, KvLayout::BSND, RecordKind::MXFP4, false, 128, {16},
        {RangeSparse(16, false)}, 0, false, false, false, ACL_INT8, 400},
        {"mx_bs_fp16_g128_split_full16", QKind::FP16, KvLayout::BSND, RecordKind::MXFP4, false, 128, {16},
         {RangeSparse(16, false)}, 0, false, false, false, ACL_INT8, 400},
        {"mx_bs_bf16_sparse_repeat_desc", QKind::BF16, KvLayout::BSND, RecordKind::MXFP4, false, 1, {8},
         {{7, 3, 7, 1, 0, 2}}, 0, false, false, false, ACL_INT8, 400},
        {"mx_tnd_bf16_b2_prefix", QKind::BF16, KvLayout::TND, RecordKind::MXFP4, true, 1, {3, 4},
         {{0, 1, 2, -1}, {0, 1, 3, 2}}, 32, false, false, false, ACL_INT8, 400},
        {"mx_tnd_fp16_b2_prefix", QKind::FP16, KvLayout::TND, RecordKind::MXFP4, true, 1, {3, 4},
         {{0, 1, 2, -1}, {0, 1, 3, 2}}, 32, false, false, false, ACL_INT8, 400},
        {"mx_tnd_bf16_g128_split", QKind::BF16, KvLayout::TND, RecordKind::MXFP4, true, 128, {16},
         {RangeSparse(16, false)}, 0, false, false, false, ACL_INT8, 400},
        {"mx_pa_bf16_block_table", QKind::BF16, KvLayout::PA_BSND, RecordKind::MXFP4, false, 1, {3},
         {{0, 1, 2, -1}}, 0, false, false, false, ACL_INT8, 400},
        {"mx_pa_fp16_block_table", QKind::FP16, KvLayout::PA_BSND, RecordKind::MXFP4, false, 1, {3},
         {{0, 1, 2, -1}}, 0, false, false, false, ACL_INT8, 400},
        {"mx_pa_bf16_g128_split", QKind::BF16, KvLayout::PA_BSND, RecordKind::MXFP4, false, 128, {3},
         {{0, 1, 2, -1}}, 0, false, false, false, ACL_INT8, 400},
        {"mx_tnd_pa_bf16_g1", QKind::BF16, KvLayout::PA_BSND, RecordKind::MXFP4, true, 1, {3},
         {{0, 1, 2, -1}}, 0, false, false, false, ACL_INT8, 400},
        {"mx_tnd_pa_bf16_g128_split", QKind::BF16, KvLayout::PA_BSND, RecordKind::MXFP4, true, 128, {3},
         {{0, 1, 2, -1}}, 0, false, false, false, ACL_INT8, 400},
        {"mx_bs_bf16_e8m0_edges_s2_1", QKind::BF16, KvLayout::BSND, RecordKind::MXFP4, false, 1, {1},
         {{0, -1}}, 0, true, true, false, ACL_INT8, 400},
        {"legacy_bs_bf16_d656", QKind::BF16, KvLayout::BSND, RecordKind::LEGACY_INT8, false, 1, {3},
         {{0, 1, 2, -1}}, 0, false, false, false, ACL_INT8, 656},
        {"reject_d400_fp8", QKind::BF16, KvLayout::BSND, RecordKind::RAW, false, 1, {1}, {{0, -1}}, 0, false,
         false, true, ACL_FLOAT8_E4M3FN, 400},
        {"reject_d399_int8", QKind::BF16, KvLayout::BSND, RecordKind::RAW, false, 1, {1}, {{0, -1}}, 0, false, false,
         true, ACL_INT8, 399},
    };
}

std::vector<int8_t> MakeRecordForCase(const CaseSpec &spec, int64_t token)
{
    if (spec.recordKind == RecordKind::MXFP4) {
        return MakeMxfp4Record(spec.edgeScales ? 0 : token, spec.qKind, spec.edgeScales);
    }
    if (spec.recordKind == RecordKind::LEGACY_INT8) {
        return MakeLegacyRecord(token, spec.qKind);
    }
    return MakeRawRecord(token, spec.keyDim);
}

PreparedCase PrepareCase(const CaseSpec &spec)
{
    PreparedCase data;
    int64_t batchCount = static_cast<int64_t>(spec.kvLengths.size());
    data.query.resize(batchCount * spec.n1 * kQueryDim);
    for (int64_t b = 0; b < batchCount; ++b) {
        for (int64_t head = 0; head < spec.n1; ++head) {
            for (int64_t d = 0; d < kQueryDim; ++d) {
                float value = spec.edgeScales
                                  ? 0.0F
                                  : 0.00390625F * static_cast<float>(((d * 3 + head * 5 + b * 7) % 29) - 14);
                data.query[(b * spec.n1 + head) * kQueryDim + d] = EncodeQ(value, spec.qKind);
            }
        }
    }
    int64_t sparseWidth = static_cast<int64_t>(spec.sparseRows.front().size());
    for (const auto &row : spec.sparseRows) {
        if (static_cast<int64_t>(row.size()) != sparseWidth) {
            throw std::runtime_error("all sparse rows must have the same width");
        }
        data.sparseIndices.insert(data.sparseIndices.end(), row.begin(), row.end());
    }
    data.logicalRecords.resize(batchCount);
    int64_t logicalTokenBase = 0;
    for (int64_t b = 0; b < batchCount; ++b) {
        for (int64_t token = 0; token < spec.kvLengths[b]; ++token) {
            data.logicalRecords[b].push_back(MakeRecordForCase(spec, logicalTokenBase + token));
        }
        logicalTokenBase += spec.kvLengths[b] + 17;
    }
    if (spec.layout == KvLayout::BSND) {
        int64_t s2 = spec.kvLengths[0];
        for (const auto &record : data.logicalRecords[0]) {
            data.key.insert(data.key.end(), record.begin(), record.end());
        }
        data.keyViewShape = {1, s2, 1, spec.keyDim};
        data.keyStorageShape = {1, s2 + (spec.storageOffsetElements > 0 ? 1 : 0), 1, spec.keyDim};
        data.valueViewShape = {1, s2, 1, kNopeDim};
        data.valueStorageShape = {1, s2 + (spec.storageOffsetElements > 0 ? 1 : 0), 1, kNopeDim};
        data.actualSeqKv = {static_cast<int32_t>(s2)};
    } else if (spec.layout == KvLayout::TND) {
        int64_t totalKv = 0;
        int32_t kvPrefix = 0;
        for (int64_t b = 0; b < batchCount; ++b) {
            for (const auto &record : data.logicalRecords[b]) {
                data.key.insert(data.key.end(), record.begin(), record.end());
            }
            totalKv += spec.kvLengths[b];
            kvPrefix += static_cast<int32_t>(spec.kvLengths[b]);
            data.actualSeqKv.push_back(kvPrefix);
        }
        data.keyViewShape = {totalKv, 1, spec.keyDim};
        data.keyStorageShape = {totalKv + (spec.storageOffsetElements > 0 ? 1 : 0), 1, spec.keyDim};
        data.valueViewShape = {totalKv, 1, kNopeDim};
        data.valueStorageShape = {totalKv + (spec.storageOffsetElements > 0 ? 1 : 0), 1, kNopeDim};
    } else {
        data.key.resize(2 * kPaBlockSize * spec.keyDim);
        auto sentinel = MakeRecordForCase(spec, 113);
        for (int64_t token = 0; token < kPaBlockSize; ++token) {
            std::copy(sentinel.begin(), sentinel.end(), data.key.begin() + token * spec.keyDim);
        }
        for (int64_t token = 0; token < kPaBlockSize; ++token) {
            const auto &record = token < spec.kvLengths[0] ? data.logicalRecords[0][token] : sentinel;
            std::copy(record.begin(), record.end(), data.key.begin() + (kPaBlockSize + token) * spec.keyDim);
        }
        data.keyViewShape = {2, kPaBlockSize, 1, spec.keyDim};
        data.keyStorageShape = data.keyViewShape;
        data.valueViewShape = {2, kPaBlockSize, 1, kNopeDim};
        data.valueStorageShape = data.valueViewShape;
        data.actualSeqKv = {static_cast<int32_t>(spec.kvLengths[0])};
        data.blockTable = {1};
    }
    if (spec.queryTnd) {
        data.sparseShape = {batchCount, 1, sparseWidth};
        data.queryShape = {batchCount, spec.n1, kQueryDim};
        data.outputShape = {batchCount, spec.n1, kNopeDim};
        int32_t qPrefix = 0;
        for (int64_t b = 0; b < batchCount; ++b) {
            data.actualSeqQ.push_back(++qPrefix);
        }
    } else {
        data.sparseShape = {batchCount, 1, 1, sparseWidth};
        data.queryShape = {batchCount, 1, spec.n1, kQueryDim};
        data.outputShape = {batchCount, 1, spec.n1, kNopeDim};
        data.actualSeqQ.assign(batchCount, 1);
    }
    data.value.resize(ShapeSize(data.valueViewShape), 0);
    data.output.resize(ShapeSize(data.outputShape), 0);
    return data;
}

std::vector<uint16_t> CpuReference(const CaseSpec &spec, const PreparedCase &data)
{
    int64_t batchCount = static_cast<int64_t>(spec.kvLengths.size());
    std::vector<uint16_t> expected(ShapeSize(data.outputShape), 0);
    for (int64_t b = 0; b < batchCount; ++b) {
        std::vector<std::vector<float>> decoded;
        for (const auto &record : data.logicalRecords[b]) {
            decoded.push_back(DecodeRecord(record, spec));
        }
        std::vector<int32_t> selected;
        for (int32_t index : spec.sparseRows[b]) {
            if (index == -1) {
                break;
            }
            if (index < 0 || index >= spec.kvLengths[b]) {
                throw std::runtime_error("sparse index is outside actual sequence length");
            }
            selected.push_back(index);
        }
        if (selected.empty()) {
            throw std::runtime_error("positive case must select at least one KV token");
        }
        for (int64_t head = 0; head < spec.n1; ++head) {
            const uint16_t *query = data.query.data() + (b * spec.n1 + head) * kQueryDim;
            std::vector<float> scores(selected.size(), 0.0F);
            for (size_t i = 0; i < selected.size(); ++i) {
                for (int64_t d = 0; d < kQueryDim; ++d) {
                    scores[i] += CubeInputValue(DecodeQ(query[d], spec.qKind), spec.qKind) *
                                 CubeInputValue(decoded[selected[i]][d], spec.qKind);
                }
                scores[i] /= std::sqrt(static_cast<float>(kQueryDim));
            }
            float maxScore = *std::max_element(scores.begin(), scores.end());
            float sum = 0.0F;
            for (auto &score : scores) {
                score = std::exp(score - maxScore);
                sum += score;
            }
            for (auto &score : scores) {
                score /= sum;
            }
            for (int64_t d = 0; d < kNopeDim; ++d) {
                float value = 0.0F;
                for (size_t i = 0; i < selected.size(); ++i) {
                    value += scores[i] * CubeInputValue(decoded[selected[i]][d], spec.qKind);
                }
                expected[(b * spec.n1 + head) * kNopeDim + d] = EncodeQ(value, spec.qKind);
            }
        }
    }
    return expected;
}

template <typename T>
aclTensor *CreateTensor(const std::vector<T> &hostData, const std::vector<int64_t> &viewShape,
                        const std::vector<int64_t> &storageShape, aclDataType dtype, int64_t offsetElements,
                        ResourceSet &resources, void **deviceOut = nullptr)
{
    if (static_cast<int64_t>(hostData.size()) != ShapeSize(viewShape) || viewShape.size() != storageShape.size()) {
        return nullptr;
    }
    std::vector<int64_t> strides(storageShape.size(), 1);
    for (int64_t i = static_cast<int64_t>(storageShape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * storageShape[i + 1];
    }
    int64_t lastElement = offsetElements;
    for (size_t i = 0; i < viewShape.size(); ++i) {
        lastElement += (viewShape[i] - 1) * strides[i];
    }
    if (lastElement >= ShapeSize(storageShape)) {
        return nullptr;
    }
    size_t storageBytes = static_cast<size_t>(ShapeSize(storageShape)) * sizeof(T);
    void *device = nullptr;
    if (aclrtMalloc(&device, storageBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return nullptr;
    }
    resources.deviceBuffers.push_back(device);
    if (aclrtMemset(device, storageBytes, 0, storageBytes) != ACL_SUCCESS) {
        return nullptr;
    }
    size_t hostBytes = hostData.size() * sizeof(T);
    void *viewDevice = static_cast<void *>(static_cast<uint8_t *>(device) + offsetElements * sizeof(T));
    if (aclrtMemcpy(viewDevice, hostBytes, hostData.data(), hostBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return nullptr;
    }
    aclTensor *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dtype, strides.data(), offsetElements,
                                        ACL_FORMAT_ND, storageShape.data(), storageShape.size(), device);
    if (tensor != nullptr) {
        resources.tensors.push_back(tensor);
    }
    if (deviceOut != nullptr) {
        *deviceOut = device;
    }
    return tensor;
}

template <typename T>
aclTensor *CreateContiguousTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape,
                                  aclDataType dtype, ResourceSet &resources, void **deviceOut = nullptr)
{
    return CreateTensor(hostData, shape, shape, dtype, 0, resources, deviceOut);
}

const char *LayoutName(KvLayout layout)
{
    if (layout == KvLayout::TND) {
        return "TND";
    }
    return layout == KvLayout::PA_BSND ? "PA_BSND" : "BSND";
}

aclDataType QAclType(QKind kind)
{
    return kind == QKind::BF16 ? ACL_BF16 : ACL_FLOAT16;
}

int CompareOutput(const CaseSpec &spec, const PreparedCase &data, const std::vector<uint16_t> &actual)
{
    if (actual.size() != data.expected.size()) {
        std::printf("[FAIL] %s output size mismatch\n", spec.name.c_str());
        return 1;
    }
    if (spec.bitExact) {
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != data.expected[i]) {
                std::printf("[FAIL] %s bit mismatch at %zu: actual=0x%04x expected=0x%04x\n", spec.name.c_str(), i,
                            actual[i], data.expected[i]);
                return 1;
            }
        }
        std::printf("[PASS] %s bit-exact, elements=%zu\n", spec.name.c_str(), actual.size());
        return 0;
    }
    int64_t closeCount = 0;
    float maxAbsError = 0.0F;
    for (size_t i = 0; i < actual.size(); ++i) {
        float actualValue = DecodeQ(actual[i], spec.qKind);
        float expectedValue = DecodeQ(data.expected[i], spec.qKind);
        float error = std::abs(actualValue - expectedValue);
        float tolerance = 0.05F + 0.02F * std::abs(expectedValue);
        closeCount += error <= tolerance;
        maxAbsError = std::max(maxAbsError, error);
    }
    double ratio = static_cast<double>(closeCount) / static_cast<double>(actual.size());
    std::printf("[%s] %s close_ratio=%.6f max_abs_error=%.6f\n", ratio >= 0.99 ? "PASS" : "FAIL",
                spec.name.c_str(), ratio, maxAbsError);
    return ratio >= 0.99 ? 0 : 1;
}

int RunCase(const CaseSpec &spec, aclrtStream stream)
{
    PreparedCase data = PrepareCase(spec);
    if (!spec.expectWorkspaceFailure) {
        data.expected = CpuReference(spec, data);
    }
    ResourceSet resources;
    void *outputDevice = nullptr;
    aclTensor *queryTensor =
        CreateContiguousTensor(data.query, data.queryShape, QAclType(spec.qKind), resources);
    aclTensor *keyTensor = CreateTensor(data.key, data.keyViewShape, data.keyStorageShape, spec.keyDtype,
                                        spec.storageOffsetElements, resources);
    aclTensor *valueTensor = CreateTensor(data.value, data.valueViewShape, data.valueStorageShape, spec.keyDtype,
                                          spec.storageOffsetElements, resources);
    aclTensor *sparseTensor =
        CreateContiguousTensor(data.sparseIndices, data.sparseShape, ACL_INT32, resources);
    std::vector<float> dummyScale{1.0F};
    aclTensor *keyScaleTensor = CreateContiguousTensor(dummyScale, {1}, ACL_FLOAT, resources);
    aclTensor *valueScaleTensor = CreateContiguousTensor(dummyScale, {1}, ACL_FLOAT, resources);
    aclTensor *actualQTensor = CreateContiguousTensor(data.actualSeqQ, {static_cast<int64_t>(data.actualSeqQ.size())},
                                                       ACL_INT32, resources);
    aclTensor *actualKvTensor = CreateContiguousTensor(
        data.actualSeqKv, {static_cast<int64_t>(data.actualSeqKv.size())}, ACL_INT32, resources);
    aclTensor *blockTableTensor = nullptr;
    if (!data.blockTable.empty()) {
        blockTableTensor = CreateContiguousTensor(data.blockTable, {1, 1}, ACL_INT32, resources);
    }
    aclTensor *outputTensor = CreateContiguousTensor(data.output, data.outputShape, QAclType(spec.qKind), resources,
                                                      &outputDevice);
    if (queryTensor == nullptr || keyTensor == nullptr || valueTensor == nullptr || sparseTensor == nullptr ||
        keyScaleTensor == nullptr || valueScaleTensor == nullptr || actualQTensor == nullptr ||
        actualKvTensor == nullptr || outputTensor == nullptr ||
        (spec.layout == KvLayout::PA_BSND && blockTableTensor == nullptr)) {
        std::printf("[FAIL] %s tensor creation failed\n", spec.name.c_str());
        return 1;
    }
    char layoutQuery[5] = "BSND";
    if (spec.queryTnd) {
        std::strcpy(layoutQuery, "TND");
    }
    char layoutKv[8] = {0};
    std::strcpy(layoutKv, LayoutName(spec.layout));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = aclnnKvQuantSparseFlashAttentionGetWorkspaceSize(
        queryTensor, keyTensor, valueTensor, sparseTensor, keyScaleTensor, valueScaleTensor, blockTableTensor,
        actualQTensor, actualKvTensor, 1.0 / std::sqrt(static_cast<double>(kQueryDim)), 2, 2, 1, layoutQuery, layoutKv,
        3, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), 2, 1, 128, kRopeDim,
        outputTensor, &workspaceSize, &executor);
    if (spec.expectWorkspaceFailure) {
        if (status == ACL_SUCCESS) {
            std::printf("[FAIL] %s unexpectedly passed GetWorkspaceSize\n", spec.name.c_str());
            return 1;
        }
        std::printf("[PASS] %s rejected as expected, status=%d\n", spec.name.c_str(), status);
        return 0;
    }
    if (status != ACL_SUCCESS) {
        std::printf("[FAIL] %s GetWorkspaceSize status=%d\n", spec.name.c_str(), status);
        return status;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0U) {
        if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            std::printf("[FAIL] %s workspace allocation failed\n", spec.name.c_str());
            return 1;
        }
        resources.deviceBuffers.push_back(workspace);
    }
    status = aclnnKvQuantSparseFlashAttention(workspace, workspaceSize, executor, stream);
    if (status == ACL_SUCCESS) {
        status = aclrtSynchronizeStream(stream);
    }
    if (status != ACL_SUCCESS) {
        std::printf("[FAIL] %s execution status=%d\n", spec.name.c_str(), status);
        return status;
    }
    std::vector<uint16_t> actual(data.output.size());
    size_t outputBytes = actual.size() * sizeof(uint16_t);
    if (aclrtMemcpy(actual.data(), outputBytes, outputDevice, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::printf("[FAIL] %s output copy failed\n", spec.name.c_str());
        return 1;
    }
    return CompareOutput(spec, data, actual);
}

void PrintCases(const std::vector<CaseSpec> &cases)
{
    for (const auto &spec : cases) {
        std::printf("%s\n", spec.name.c_str());
    }
}

int RunCpuOnly(const CaseSpec &spec)
{
    PreparedCase data = PrepareCase(spec);
    if (spec.expectWorkspaceFailure) {
        std::printf("[SKIP] %s requires aclnn GetWorkspaceSize\n", spec.name.c_str());
        return 0;
    }
    data.expected = CpuReference(spec, data);
    if (data.expected.size() != static_cast<size_t>(ShapeSize(data.outputShape))) {
        std::printf("[FAIL] %s CPU output size mismatch\n", spec.name.c_str());
        return 1;
    }
    for (uint16_t bits : data.expected) {
        if (!std::isfinite(DecodeQ(bits, spec.qKind))) {
            std::printf("[FAIL] %s CPU output contains NaN/Inf\n", spec.name.c_str());
            return 1;
        }
    }
    std::printf("[PASS] %s CPU record/oracle, elements=%zu\n", spec.name.c_str(), data.expected.size());
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    std::string selectedCase;
    int32_t deviceId = 0;
    bool listOnly = false;
    bool cpuOnly = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--list") {
            listOnly = true;
        } else if (arg == "--cpu-only") {
            cpuOnly = true;
        } else if (arg == "--case" && i + 1 < argc) {
            selectedCase = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            deviceId = static_cast<int32_t>(std::stoi(argv[++i]));
        } else {
            std::printf("usage: %s [--list] [--cpu-only] [--case NAME] [--device ID]\n", argv[0]);
            return 2;
        }
    }
    auto cases = BuildCases();
    if (listOnly) {
        PrintCases(cases);
        return 0;
    }
    if (!selectedCase.empty() &&
        std::none_of(cases.begin(), cases.end(), [&](const CaseSpec &spec) { return spec.name == selectedCase; })) {
        std::printf("unknown case: %s\n", selectedCase.c_str());
        PrintCases(cases);
        return 2;
    }
    if (cpuOnly) {
        int failed = 0;
        int executed = 0;
        int skipped = 0;
        for (const auto &spec : cases) {
            if (!selectedCase.empty() && spec.name != selectedCase) {
                continue;
            }
            ++executed;
            if (spec.expectWorkspaceFailure) {
                RunCpuOnly(spec);
                ++skipped;
                continue;
            }
            try {
                failed += RunCpuOnly(spec) != 0;
            } catch (const std::exception &error) {
                std::printf("[FAIL] %s exception: %s\n", spec.name.c_str(), error.what());
                ++failed;
            }
        }
        std::printf("CPU summary: executed=%d passed=%d skipped=%d failed=%d\n", executed,
                    executed - skipped - failed, skipped, failed);
        return failed == 0 ? 0 : 1;
    }
    if (deviceId < 0) {
        std::printf("invalid device id: %d\n", deviceId);
        return 2;
    }
    aclError initStatus = aclInit(nullptr);
    if (initStatus != ACL_SUCCESS) {
        std::printf("aclInit failed, status=%d\n", initStatus);
        return initStatus;
    }
    aclError setDeviceStatus = aclrtSetDevice(deviceId);
    if (setDeviceStatus != ACL_SUCCESS) {
        std::printf("aclrtSetDevice(%d) failed, status=%d\n", deviceId, setDeviceStatus);
        aclFinalize();
        return setDeviceStatus;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::printf("stream creation failed\n");
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
    int failed = 0;
    int executed = 0;
    for (const auto &spec : cases) {
        if (!selectedCase.empty() && spec.name != selectedCase) {
            continue;
        }
        ++executed;
        try {
            failed += RunCase(spec, stream) != 0;
        } catch (const std::exception &error) {
            std::printf("[FAIL] %s exception: %s\n", spec.name.c_str(), error.what());
            ++failed;
        }
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    std::printf("summary: executed=%d passed=%d failed=%d\n", executed, executed - failed, failed);
    return failed == 0 ? 0 : 1;
}
