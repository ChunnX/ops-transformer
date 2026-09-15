/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * This program is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file test_moe_tutel_combine_x_utils.h
 * \brief CSV-driven kernel-UT helpers for MoeTutelCombineX (arch35 / Ascend950).
 *
 * 迁移自插件工作区 moe-tutel-combine-x 的内核级测试（白盒 branch-1/2/3 设计用例集 +
 * 黑盒 L0 边界），按本仓 block_attn_res_prepare 的 CSV 参数化 kernel-UT 范式重组：
 *   - key=0 (normal-no-batch) / key=2 (normal-batch) / key=1 (empty) 三分支
 *   - float32 / float16 / bfloat16 三 dtype（float32 走真实 extern "C" 入口，
 *     16b 走测试 TU 内的模板包装器，见 arch35/test_moe_tutel_combine_x.cpp）
 *   - 有效路由 / -1 容量丢弃 / locations>=C 溢出 / E>EC 全零 / E=1 单专家 /
 *     capacity=eC 哨兵（kernel 侧重扫）/ H 维多 chunk + 尾块 / 多核 big-core 切分 /
 *     空张量四退化（EC=0 零填充、S=0 / H=0 / B=0 空转）
 *
 * 输入为确定性生成（seed 驱动），且全部取值在 fp32/fp16/bf16 下可精确表示
 * （k/8 与 (k1/8)*(k2/8)=k1*k2/64，|k1*k2|<=24，mantissa 位数充足），参考值按
 * spec 公式（fp64 累加）计算后与 kernel 输出逐点比对。
 */

#ifndef TEST_MOE_TUTEL_COMBINE_X_UTILS_H
#define TEST_MOE_TUTEL_COMBINE_X_UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gmm_csv_parse_utils.h"
#include "gtest/gtest.h"

// TilingData struct + tiling keys (shared host/kernel definition).
#include "../../../op_kernel/arch35/moe_tutel_combine_x_tiling_struct.h"

namespace MoeTutelCombineXUt {

// 与 op_host/moe_tutel_combine_x_tiling_arch35.cpp 一致的 UB 预算常量（tiling 契约）。
constexpr int64_t UB_BUDGET_CAP = 245760;
constexpr int64_t UB_SPLIT_P = 3;

enum class Dtype {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2
};

struct KernelTestParam {
    std::string socVersion;
    std::string caseName;
    std::string kernelUtTarget;
    Dtype dtype = Dtype::FP32;
    int64_t tilingKey = 0; // 0 / 1 / 2
    int64_t batchSize = 1;
    int64_t samples = 0;
    int64_t hidden = 0;
    int64_t eC = 0;
    int64_t expertNum = 0; // 0 => 全部 indices=-1（全丢弃）
    uint32_t seed = 0;
    bool injectInvalid = false;
    bool sentinelCapacity = false;
    int64_t hChunkOverride = 0; // 0 => 按 tiling 契约公式取值
    uint32_t numBlocks = 1;     // 供 usedCoreNum=min(numBlocks, tokenTotal) 使用的核数上限
};

struct KernelCsvLoadResult {
    std::vector<KernelTestParam> params;
    std::vector<std::string> errors;
};

inline size_t DtypeSize(Dtype dtype)
{
    return dtype == Dtype::FP32 ? 4UL : 2UL;
}

inline KernelCsvLoadResult LoadParams(const std::string &socVersion, const std::string &testSuite)
{
    KernelCsvLoadResult result;
    const std::string csvPath =
        ops::ut::ResolveCsvPath("test_moe_tutel_combine_x.csv", "moe/moe_tutel_combine_x/tests/ut/op_kernel", __FILE__);
    std::ifstream csvData(csvPath, std::ios::in);
    if (!csvData.is_open()) {
        result.errors.emplace_back("Cannot open kernel case file: " + csvPath);
        return result;
    }

    std::string line;
    size_t lineNo = 0UL;
    while (std::getline(csvData, line)) {
        ++lineNo;
        if (ops::ut::Trim(line).empty()) {
            continue;
        }
        std::vector<std::string> fields;
        ops::ut::SplitStr2Vec(line, ",", fields);
        if (fields.empty() || ops::ut::Trim(fields[0]) == "socVersion") {
            continue;
        }
        const std::string caseName = fields.size() > 1UL ? ops::ut::Trim(fields[1]) : "";
        try {
            constexpr size_t REQUIRED_COLUMN_COUNT = 16UL;
            if (fields.size() < REQUIRED_COLUMN_COUNT) {
                throw std::runtime_error("invalid CSV column count: " + std::to_string(fields.size()));
            }
            KernelTestParam param;
            size_t index = 0UL;
            param.socVersion = ops::ut::Trim(fields[index++]);
            param.caseName = ops::ut::Trim(fields[index++]);
            param.kernelUtTarget = ops::ut::Trim(fields[index++]);
            const std::string dtypeStr = ops::ut::ToLower(ops::ut::Trim(fields[index++]));
            if (dtypeStr == "float32") {
                param.dtype = Dtype::FP32;
            } else if (dtypeStr == "float16") {
                param.dtype = Dtype::FP16;
            } else if (dtypeStr == "bfloat16") {
                param.dtype = Dtype::BF16;
            } else {
                throw std::runtime_error("unknown dtype: " + dtypeStr);
            }
            param.tilingKey = std::stoll(fields[index++]);
            param.batchSize = std::stoll(fields[index++]);
            param.samples = std::stoll(fields[index++]);
            param.hidden = std::stoll(fields[index++]);
            param.eC = std::stoll(fields[index++]);
            param.expertNum = std::stoll(fields[index++]);
            param.seed = static_cast<uint32_t>(std::stoul(fields[index++]));
            param.injectInvalid = ops::ut::ParseBool(fields[index++]);
            param.sentinelCapacity = ops::ut::ParseBool(fields[index++]);
            param.hChunkOverride = std::stoll(fields[index++]);
            param.numBlocks = static_cast<uint32_t>(std::stoul(fields[index++]));

            if (param.socVersion != socVersion) {
                continue;
            }
            if (param.kernelUtTarget != testSuite) {
                continue;
            }
            result.params.push_back(param);
        } catch (const std::exception &error) {
            result.errors.emplace_back(ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error));
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// 参数合法性校验（tiling 契约 + kernel 分支前提）。
// ---------------------------------------------------------------------------
inline void ValidateParam(const KernelTestParam &param)
{
    if (param.tilingKey != TILING_KEY_NORMAL_NO_BATCH && param.tilingKey != TILING_KEY_EMPTY &&
        param.tilingKey != TILING_KEY_NORMAL_BATCH) {
        throw std::runtime_error("tilingKey must be 0/1/2, got " + std::to_string(param.tilingKey));
    }
    if (param.tilingKey == TILING_KEY_NORMAL_NO_BATCH && param.batchSize != 1) {
        throw std::runtime_error("key=0 requires batchSize==1");
    }
    if (param.tilingKey != TILING_KEY_EMPTY) {
        if (param.batchSize <= 0 || param.samples <= 0 || param.hidden <= 0 || param.eC <= 0) {
            throw std::runtime_error("key=0/2 requires B/S/H/EC > 0");
        }
        if (param.expertNum < 0) {
            throw std::runtime_error("expertNum must be >= 0 (0 = all dropped)");
        }
    }
    if (param.numBlocks == 0UL) {
        throw std::runtime_error("numBlocks must be >= 1");
    }
    const int64_t perBufBytes = (UB_BUDGET_CAP / UB_SPLIT_P) & ~31LL; // 81920
    const int64_t budget = param.tilingKey == TILING_KEY_NORMAL_BATCH ?
                               perBufBytes / static_cast<int64_t>(DtypeSize(param.dtype)) :
                               perBufBytes / 4LL;
    if (param.hChunkOverride < 0 || param.hChunkOverride > budget) {
        throw std::runtime_error("hChunkOverride out of tiling budget [0, " + std::to_string(budget) + "]");
    }
}

// ---------------------------------------------------------------------------
// 确定性输入 + fp64 参考（spec 公式：x[b,t,:] = valid ? gates[b,t] * y[b, src, :] : 0，
// src = indices*C + locations，C = EC/E，E = max(indices)+1（全 -1 记 1））。
// ---------------------------------------------------------------------------
struct CaseData {
    std::vector<float> yGrad;       // B*EC*H（fp32 精确表示的生成值）
    std::vector<float> gates;       // B*S
    std::vector<int32_t> indices;   // B*S
    std::vector<int32_t> locations; // B*S
    std::vector<float> expected;    // B*S*H（fp32 域参考值）
    MoetutelcombinexTilingData tiling{};
    uint32_t launchBlocks = 1;
};

inline CaseData BuildCase(const KernelTestParam &param)
{
    ValidateParam(param);
    CaseData data;
    const int64_t b = param.batchSize;
    const int64_t s = param.samples;
    const int64_t h = param.hidden;
    const int64_t ec = param.eC;

    // -- 确定性输入生成（全部值在三种 dtype 下均可精确表示）--
    data.yGrad.resize(static_cast<size_t>(std::max<int64_t>(0, b * ec * h)));
    for (size_t i = 0UL; i < data.yGrad.size(); ++i) {
        const int32_t k = static_cast<int32_t>((i * 7UL + param.seed) % 13UL) - 6;
        data.yGrad[i] = static_cast<float>(k) * 0.125F;
    }
    data.gates.resize(static_cast<size_t>(std::max<int64_t>(0, b * s)));
    data.indices.assign(static_cast<size_t>(std::max<int64_t>(0, b * s)), -1);
    data.locations.assign(static_cast<size_t>(std::max<int64_t>(0, b * s)), 0);
    if (b * s > 0) {
        // expertNum==0 是 CSV 在用取值(0 = 全部 indices=-1 的全丢弃用例),此处三元把
        // 0 映射为 E=1(与 kernel 侧重扫的"全负记 E=1"同口径),勿删守卫:删掉后下方
        // ec/eGen 会对 0 除(宿主侧 SIGFPE 崩掉整个 UT 进程)。
        const int64_t eGen = (param.expertNum == 0) ? 1 : param.expertNum;
        const int64_t cGen = ec / eGen;
        for (int64_t t = 0; t < b * s; ++t) {
            const size_t i = static_cast<size_t>(t);
            data.gates[i] = static_cast<float>(static_cast<int32_t>((i * 5UL + param.seed) % 9UL) - 4) * 0.125F;
            if (param.expertNum == 0) {
                data.indices[i] = -1; // 全丢弃
            } else if (t == 0) {
                data.indices[i] = static_cast<int32_t>(param.expertNum - 1); // 锚定 max(indices)=E-1
            } else {
                data.indices[i] = (param.injectInvalid && (t % 5 == 2)) ?
                                      -1 :
                                      static_cast<int32_t>((t * 3 + param.seed) % param.expertNum);
            }
            const int64_t locMod = (cGen > 0) ? cGen : 1;
            data.locations[i] = (param.injectInvalid && (t % 7 == 3)) ?
                                    static_cast<int32_t>(cGen) // 溢出容量（>=C 无效）
                                    :
                                    static_cast<int32_t>((t * 5 + param.seed) % locMod);
        }
    }

    // -- E/C 派生（与 host DeriveCapacity / kernel 侧重扫同口径）--
    int32_t maxIdx = -1;
    for (int32_t v : data.indices) {
        maxIdx = std::max(maxIdx, v);
    }
    const int64_t eRef = (maxIdx >= 0) ? (static_cast<int64_t>(maxIdx) + 1) : 1;
    const int64_t cRef = (ec > 0) ? (ec / eRef) : 0;

    // -- tiling 构造（镜像 tiling 契约公式；override 用于压小 chunk 触发多 chunk 路径）--
    if (param.tilingKey == TILING_KEY_EMPTY) {
        std::memset(&data.tiling, 0, sizeof(data.tiling));
        data.tiling.batchSize = b;
        data.tiling.samples = s;
        data.tiling.hidden = h;
        data.launchBlocks = 1U;
        data.expected.assign(static_cast<size_t>(std::max<int64_t>(0, b * s * h)), 0.0F); // 全零 / 空
        return data;
    }

    const int64_t tokenTotal = b * s;
    const int64_t usedCoreNum = std::min<int64_t>(param.numBlocks, tokenTotal);
    const int64_t coreTokenBase = tokenTotal / usedCoreNum;
    const int32_t bigCoreCnt = static_cast<int32_t>(tokenTotal % usedCoreNum);
    const int64_t perBufBytes = (UB_BUDGET_CAP / UB_SPLIT_P) & ~31LL;
    const int64_t budget = (param.tilingKey == TILING_KEY_NORMAL_BATCH) ?
                               perBufBytes / static_cast<int64_t>(DtypeSize(param.dtype)) :
                               perBufBytes / 4LL;
    const int64_t hChunk =
        (param.hChunkOverride > 0) ? param.hChunkOverride : std::min(h, std::max<int64_t>(1, budget));
    const int64_t hChunkCnt = (h + hChunk - 1) / hChunk;
    const int64_t hChunkTail = h - (hChunkCnt - 1) * hChunk;
    const int64_t capacity = param.sentinelCapacity ? ec : cRef;

    data.tiling.batchSize = b;
    data.tiling.samples = s;
    data.tiling.hidden = h;
    data.tiling.capacity = capacity;
    data.tiling.eC = ec;
    data.tiling.tokenTotal = tokenTotal;
    data.tiling.coreTokenBase = coreTokenBase;
    data.tiling.bigCoreCnt = bigCoreCnt;
    data.tiling.hChunk = hChunk;
    data.tiling.hChunkCnt = hChunkCnt;
    data.tiling.hChunkTail = hChunkTail;
    data.launchBlocks = static_cast<uint32_t>(usedCoreNum);

    // -- fp64 参考（哨兵/重扫语义与 kernel 一致：capacity==eC 且 eC>0 时按重扫 C=EC/E_ref）--
    const int64_t cKernel = (capacity == ec && ec > 0) ? (ec / eRef) : capacity;
    data.expected.assign(static_cast<size_t>(b * s * h), 0.0F);
    for (int64_t t = 0; t < tokenTotal; ++t) {
        const size_t i = static_cast<size_t>(t);
        const int32_t idx0 = data.indices[i];
        const int32_t idx1 = data.locations[i];
        const double gate = static_cast<double>(data.gates[i]);
        const int64_t srcRow = static_cast<int64_t>(idx0) * cKernel + idx1;
        const bool valid = (idx0 >= 0) && (idx1 >= 0) && (idx1 < cKernel) && (srcRow >= 0) && (srcRow < ec);
        const int64_t batch = (param.tilingKey == TILING_KEY_NORMAL_BATCH) ? (t / s) : 0;
        if (!valid) {
            continue; // 无效条目零贡献（输出已置零）
        }
        const size_t yBase = static_cast<size_t>(batch * ec * h + srcRow * h);
        for (int64_t hh = 0; hh < h; ++hh) {
            const double got = gate * static_cast<double>(data.yGrad[yBase + static_cast<size_t>(hh)]);
            data.expected[static_cast<size_t>(t * h + hh)] = static_cast<float>(got);
        }
    }
    return data;
}

// ---------------------------------------------------------------------------
// dtype 位级转换（宿主侧不依赖 half / bfloat16_t 的宿主 API）。
// 生成值与参考值均落在 16b 可精确表示域内，转换是无损的。
// ---------------------------------------------------------------------------
inline uint16_t FloatToFp16Bits(float value)
{
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    uint32_t mant = (bits & 0x007FFFFFU) >> 13U;
    if (exp <= 0) { // 本测试取值域不触及；防御性置零
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10U) | mant);
}

inline float Fp16BitsToFloat(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
    const uint32_t exp = (bits >> 10U) & 0x1FU;
    const uint32_t mant = bits & 0x03FFU;
    if (exp == 0U) {
        return 0.0F;
    }
    if (exp == 31U) {
        // IEEE-754 保真:mant==0 => ±Inf,否则 NaN。若按普通指数换算,0xFFFF
        // (输出毒化标记)会被转成有限大数,NaN 语义信息丢失。
        const uint32_t out = sign | 0x7F800000U | (mant << 13U);
        float value = 0.0F;
        std::memcpy(&value, &out, sizeof(value));
        return value;
    }
    const uint32_t out = sign | ((exp - 15U + 127U) << 23U) | (mant << 13U);
    float value = 0.0F;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

inline uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16U) & 1U;
    uint32_t out = (bits >> 16U) + lsb; // round-to-nearest-even（取值域内无损）
    return static_cast<uint16_t>(out);
}

inline float Bf16BitsToFloat(uint16_t bits)
{
    const uint32_t out = static_cast<uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

inline void WriteDtypeBuffer(uint8_t *gm, const std::vector<float> &values, Dtype dtype)
{
    if (dtype == Dtype::FP32) {
        std::memcpy(gm, values.data(), values.size() * sizeof(float));
    } else if (dtype == Dtype::FP16) {
        auto *out = reinterpret_cast<uint16_t *>(gm);
        for (size_t i = 0UL; i < values.size(); ++i) {
            out[i] = FloatToFp16Bits(values[i]);
        }
    } else {
        auto *out = reinterpret_cast<uint16_t *>(gm);
        for (size_t i = 0UL; i < values.size(); ++i) {
            out[i] = FloatToBf16Bits(values[i]);
        }
    }
}

inline std::vector<float> ReadDtypeBuffer(const uint8_t *gm, size_t count, Dtype dtype)
{
    std::vector<float> values(count, 0.0F);
    if (dtype == Dtype::FP32) {
        std::memcpy(values.data(), gm, count * sizeof(float));
    } else if (dtype == Dtype::FP16) {
        const auto *in = reinterpret_cast<const uint16_t *>(gm);
        for (size_t i = 0UL; i < count; ++i) {
            values[i] = Fp16BitsToFloat(in[i]);
        }
    } else {
        const auto *in = reinterpret_cast<const uint16_t *>(gm);
        for (size_t i = 0UL; i < count; ++i) {
            values[i] = Bf16BitsToFloat(in[i]);
        }
    }
    return values;
}

} // namespace MoeTutelCombineXUt

#endif // TEST_MOE_TUTEL_COMBINE_X_UTILS_H
