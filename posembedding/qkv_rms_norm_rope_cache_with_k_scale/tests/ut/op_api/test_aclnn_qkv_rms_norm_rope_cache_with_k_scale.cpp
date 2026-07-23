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
 * \file test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp
 * \brief CSV driven QkvRmsNormRopeCacheWithKScale aclnn op_api UT.
 */

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

namespace {
using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kLegacyCsvColumnCount = 54;
constexpr size_t kCsvColumnCount = 61;
constexpr int64_t kDefaultT = 128;
constexpr int64_t kDefaultNq = 16;
constexpr int64_t kDefaultNk = 2;
constexpr int64_t kDefaultNv = 2;
constexpr int64_t kDefaultD = 128;
constexpr int64_t kDefaultBlockNum = 32;
constexpr int64_t kDefaultBlockSize = 16;
constexpr int64_t kDefaultBatch = 1;
constexpr int64_t kDefaultMaxSeqLen = 128;

vector<int64_t> ParseDims(const string &value)
{
    return ops::ut::ParseDims(value);
}

vector<int64_t> ParseI64List(const string &value)
{
    return ops::ut::ParseI64List(value);
}

aclDataType ParseDtype(const string &dtype)
{
    return ops::ut::ParseAclDtype(dtype);
}

aclnnStatus ParseStatus(const string &status)
{
    static const map<string, aclnnStatus> statusMap = {
        {"SUCCESS", ACLNN_SUCCESS},
        {"ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
        {"ERR_PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
    };
    const auto it = statusMap.find(Trim(status));
    return it == statusMap.end() ? ACLNN_SUCCESS : it->second;
}

TensorDesc MakeTensorDesc(const vector<int64_t> &shape, aclDataType dtype, bool useRange = true)
{
    auto desc = TensorDesc(shape, dtype, ACL_FORMAT_ND);
    if (useRange) {
        desc.ValueRange(-1, 1);
    }
    return desc;
}

struct QkvRmsNormRopeCacheWithKScaleCase {
    void Run() const
    {
        TensorDesc qkv = MakeTensorDesc(ParseDims(qkvShape), ParseDtype(qkvDtype));
        TensorDesc qGamma = MakeTensorDesc(ParseDims(qGammaShape), ParseDtype(qGammaDtype));
        TensorDesc kGamma = MakeTensorDesc(ParseDims(kGammaShape), ParseDtype(kGammaDtype));
        TensorDesc cosSin = MakeTensorDesc(ParseDims(cosSinShape), ParseDtype(cosSinDtype));
        TensorDesc slotMapping = MakeTensorDesc(ParseDims(slotMappingShape), ParseDtype(slotMappingDtype));
        TensorDesc kCache = MakeTensorDesc(ParseDims(kCacheShape), ParseDtype(kCacheDtype));
        TensorDesc vCache = MakeTensorDesc(ParseDims(vCacheShape), ParseDtype(vCacheDtype));
        TensorDesc kScaleCache = MakeTensorDesc(ParseDims(kScaleCacheShape), ParseDtype(kScaleCacheDtype));
        TensorDesc queryStartLoc = MakeTensorDesc(ParseDims(queryStartLocShape), ParseDtype(queryStartLocDtype));
        TensorDesc seqLens = MakeTensorDesc(ParseDims(seqLensShape), ParseDtype(seqLensDtype));
        TensorDesc rotationOptional =
            MakeTensorDesc(ParseDims(rotationOptionalShape), ParseDtype(rotationOptionalDtype));
        TensorDesc vScaleOptional = MakeTensorDesc(ParseDims(vScaleOptionalShape), ParseDtype(vScaleOptionalDtype));
        TensorDesc qOut = MakeTensorDesc(ParseDims(qOutShape), ParseDtype(qOutDtype), false);
        TensorDesc qScale = MakeTensorDesc(ParseDims(qScaleShape), ParseDtype(qScaleDtype), false);
        IntArrayDesc headNums(ParseI64List(headNumsValue));

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACLNN_SUCCESS;
        const bool hasHeadNumsValue = ParseBool(hasHeadNums);
        const bool hasLayoutQkvValue = ParseBool(hasLayoutQkv);
        const bool hasLayoutQOutValue = ParseBool(hasLayoutQOut);
        const string layoutQkvStorage = layoutQkv == "<empty>" ? "" : layoutQkv;
        const string layoutQOutStorage = layoutQOut == "<empty>" ? "" : layoutQOut;
        const char *layoutQkvArg = hasLayoutQkvValue ? layoutQkvStorage.c_str() : nullptr;
        const char *layoutQOutArg = hasLayoutQOutValue ? layoutQOutStorage.c_str() : nullptr;

        if (!ParseBool(hasQkv)) {
            auto ut =
                OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                          INPUT(nullptr, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                epsilon, nullptr, "PerTokenPerHead"),
                          OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasQGamma)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, nullptr, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasKGamma)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, nullptr, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasCosSin)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, nullptr, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasSlotMapping)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, nullptr, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasKCache)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, nullptr, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasVCache)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, nullptr, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasKScaleCache)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, nullptr, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasQueryStartLoc)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, nullptr,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasSeqLens)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      nullptr, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasOptionalRotation)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, nullptr, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasOptionalVScale)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, nullptr, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!hasHeadNumsValue) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, nullptr, layoutQkvArg,
                                      layoutQOutArg, epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!hasLayoutQkvValue) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, nullptr, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!hasLayoutQOutValue) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, nullptr,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasQOut)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(nullptr, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasQScale)) {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, nullptr));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            auto ut = OP_API_UT(aclnnQkvRmsNormRopeCacheWithKScale,
                                INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc,
                                      seqLens, rotationOptional, vScaleOptional, nullptr, headNums, layoutQkvArg, layoutQOutArg,
                                      epsilon, nullptr, "PerTokenPerHead"),
                                OUTPUT(qOut, qScale));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }

        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, ParseStatus(expectRet)) << "caseName=" << caseName;
        }
    }

    string caseName;
    int64_t T = kDefaultT;
    int64_t Nq = kDefaultNq;
    int64_t Nk = kDefaultNk;
    int64_t Nv = kDefaultNv;
    int64_t D = kDefaultD;
    int64_t Batch = kDefaultBatch;
    int64_t MaxSeqLen = kDefaultMaxSeqLen;
    int64_t BlockNum = kDefaultBlockNum;
    int64_t BlockSize = kDefaultBlockSize;
    string qkvShape;
    string qkvDtype;
    string hasQkv;
    string qGammaShape;
    string qGammaDtype;
    string hasQGamma;
    string kGammaShape;
    string kGammaDtype;
    string hasKGamma;
    string cosSinShape;
    string cosSinDtype;
    string hasCosSin;
    string slotMappingShape;
    string slotMappingDtype;
    string hasSlotMapping;
    string kCacheShape;
    string kCacheDtype;
    string hasKCache;
    string vCacheShape;
    string vCacheDtype;
    string hasVCache;
    string kScaleCacheShape;
    string kScaleCacheDtype;
    string hasKScaleCache;
    string queryStartLocShape;
    string queryStartLocDtype;
    string hasQueryStartLoc;
    string seqLensShape;
    string seqLensDtype;
    string hasSeqLens;
    string rotationOptionalShape;
    string rotationOptionalDtype;
    string hasOptionalRotation;
    string vScaleOptionalShape;
    string vScaleOptionalDtype;
    string hasOptionalVScale;
    string headNumsValue;
    string hasHeadNums;
    string layoutQkv;
    string hasLayoutQkv;
    string layoutQOut;
    string hasLayoutQOut;
    float epsilon = 1e-6f;
    string qOutShape;
    string qOutDtype;
    string hasQOut;
    string qScaleShape;
    string qScaleDtype;
    string hasQScale;
    string expectRet;
    string checkRet;
};

vector<QkvRmsNormRopeCacheWithKScaleCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;

    vector<QkvRmsNormRopeCacheWithKScaleCase> cases;
    string line;
    size_t lineNo = 0U;
    while (getline(in, line)) {
        ++lineNo;
        const string trimmedLine = Trim(line);
        if (trimmedLine.empty() || trimmedLine[0] == '#') {
            continue;
        }

        vector<string> cols;
        SplitStr2Vec(trimmedLine, ",", cols);
        if (cols.empty() || cols[0] == "caseName") {
            continue;
        }
        const bool legacyCsv = cols.size() == kLegacyCsvColumnCount;
        if (cols.size() != kCsvColumnCount && !legacyCsv) {
            ADD_FAILURE() << "Bad csv row column count in " << csvFilePath << ": " << trimmedLine;
            continue;
        }

        const string caseName = Trim(cols[0]);
        try {
            QkvRmsNormRopeCacheWithKScaleCase c;
            size_t i = 0;
            c.caseName = Trim(cols[i++]);
            c.T = stoll(Trim(cols[i++]));
            c.Nq = stoll(Trim(cols[i++]));
            c.Nk = stoll(Trim(cols[i++]));
            c.Nv = stoll(Trim(cols[i++]));
            c.D = stoll(Trim(cols[i++]));
            c.Batch = stoll(Trim(cols[i++]));
            c.MaxSeqLen = stoll(Trim(cols[i++]));
            c.BlockNum = stoll(Trim(cols[i++]));
            c.BlockSize = stoll(Trim(cols[i++]));
            c.qkvShape = Trim(cols[i++]);
            c.qkvDtype = Trim(cols[i++]);
            c.hasQkv = Trim(cols[i++]);
            c.qGammaShape = Trim(cols[i++]);
            c.qGammaDtype = Trim(cols[i++]);
            c.hasQGamma = Trim(cols[i++]);
            c.kGammaShape = Trim(cols[i++]);
            c.kGammaDtype = Trim(cols[i++]);
            c.hasKGamma = Trim(cols[i++]);
            c.cosSinShape = Trim(cols[i++]);
            c.cosSinDtype = Trim(cols[i++]);
            c.hasCosSin = Trim(cols[i++]);
            c.slotMappingShape = Trim(cols[i++]);
            c.slotMappingDtype = Trim(cols[i++]);
            c.hasSlotMapping = Trim(cols[i++]);
            c.kCacheShape = Trim(cols[i++]);
            c.kCacheDtype = Trim(cols[i++]);
            c.hasKCache = Trim(cols[i++]);
            c.vCacheShape = Trim(cols[i++]);
            c.vCacheDtype = Trim(cols[i++]);
            c.hasVCache = Trim(cols[i++]);
            c.kScaleCacheShape = Trim(cols[i++]);
            c.kScaleCacheDtype = Trim(cols[i++]);
            c.hasKScaleCache = Trim(cols[i++]);
            c.queryStartLocShape = Trim(cols[i++]);
            c.queryStartLocDtype = Trim(cols[i++]);
            c.hasQueryStartLoc = Trim(cols[i++]);
            if (legacyCsv) {
                c.seqLensShape = std::to_string(c.Batch);
                c.seqLensDtype = "INT32";
                c.hasSeqLens = "true";
            } else {
                c.seqLensShape = Trim(cols[i++]);
                c.seqLensDtype = Trim(cols[i++]);
                c.hasSeqLens = Trim(cols[i++]);
            }
            c.rotationOptionalShape = Trim(cols[i++]);
            c.rotationOptionalDtype = Trim(cols[i++]);
            c.hasOptionalRotation = Trim(cols[i++]);
            c.vScaleOptionalShape = Trim(cols[i++]);
            c.vScaleOptionalDtype = Trim(cols[i++]);
            c.hasOptionalVScale = Trim(cols[i++]);
            c.headNumsValue = Trim(cols[i++]);
            c.hasHeadNums = Trim(cols[i++]);
            if (legacyCsv) {
                c.layoutQkv = "NTD";
                c.hasLayoutQkv = "true";
                c.layoutQOut = "NTD";
                c.hasLayoutQOut = "true";
            } else {
                c.layoutQkv = Trim(cols[i++]);
                c.hasLayoutQkv = Trim(cols[i++]);
                c.layoutQOut = Trim(cols[i++]);
                c.hasLayoutQOut = Trim(cols[i++]);
            }
            c.epsilon = stof(Trim(cols[i++]));
            c.qOutShape = Trim(cols[i++]);
            c.qOutDtype = Trim(cols[i++]);
            c.hasQOut = Trim(cols[i++]);
            c.qScaleShape = Trim(cols[i++]);
            c.qScaleDtype = Trim(cols[i++]);
            c.hasQScale = Trim(cols[i++]);
            c.expectRet = Trim(cols[i++]);
            c.checkRet = Trim(cols[i++]);
            cases.emplace_back(c);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

const vector<QkvRmsNormRopeCacheWithKScaleCase> &GetCases()
{
    static const auto cases =
        LoadCases(ops::ut::ResolveCsvPath("test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.csv",
                                          "attention/qkv_rms_norm_rope_cache_with_k_scale/tests/ut/op_api", __FILE__));
    return cases;
}

string MakeParamName(const testing::TestParamInfo<QkvRmsNormRopeCacheWithKScaleCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class qkv_rms_norm_rope_cache_with_k_scale_csv_test : public testing::TestWithParam<QkvRmsNormRopeCacheWithKScaleCase> {
};

TEST_P(qkv_rms_norm_rope_cache_with_k_scale_csv_test, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CSV, qkv_rms_norm_rope_cache_with_k_scale_csv_test,
                         testing::ValuesIn(GetCases()), MakeParamName);

aclnnStatus RunCompleteMropeScene(const vector<int64_t> &section, const char *layoutQkv = "TND",
                                  const char *layoutQOut = "TND", aclDataType kCacheDtype = ACL_INT8,
                                  aclDataType qOutDtype = ACL_BF16, const char *qQuantMode = "NoQuant",
                                  bool hasQScale = false)
{
    TensorDesc qkv = MakeTensorDesc({5, 3, 128}, ACL_BF16);
    TensorDesc qGamma = MakeTensorDesc({128}, ACL_FLOAT);
    TensorDesc kGamma = MakeTensorDesc({128}, ACL_FLOAT);
    TensorDesc cosSin = MakeTensorDesc({128, 128}, ACL_FLOAT);
    TensorDesc slotMapping = MakeTensorDesc({5}, ACL_INT32);
    TensorDesc kCache = MakeTensorDesc({1, 1, 8, 128}, kCacheDtype);
    TensorDesc vCache = MakeTensorDesc({1, 1, 8, 128}, ACL_FLOAT8_E4M3FN);
    TensorDesc kScaleCache = MakeTensorDesc({1, 1, 8, 1}, ACL_FLOAT);
    TensorDesc rotation = MakeTensorDesc({128, 128}, ACL_BF16);
    TensorDesc vScale = MakeTensorDesc({1}, ACL_FLOAT);
    TensorDesc mropePosition = MakeTensorDesc({5, 3}, ACL_INT32);
    TensorDesc qOut = MakeTensorDesc({5, 1, 128}, qOutDtype, false);
    TensorDesc qScale = MakeTensorDesc({5, 1}, ACL_FLOAT, false);
    IntArrayDesc headNums(vector<int64_t>{1, 1, 1});
    IntArrayDesc mropeSection(section);

    uint64_t workspaceSize = 0;
    if (hasQScale) {
        auto ut = OP_API_UT(
            aclnnQkvRmsNormRopeCacheWithKScale,
            INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, nullptr, nullptr,
                  rotation, vScale, mropePosition, headNums, layoutQkv, layoutQOut, 1e-6f, mropeSection, qQuantMode),
            OUTPUT(qOut, qScale));
        return ut.TestGetWorkspaceSize(&workspaceSize);
    }
    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, nullptr, nullptr, rotation,
              vScale, mropePosition, headNums, layoutQkv, layoutQOut, 1e-6f, mropeSection, qQuantMode),
        OUTPUT(qOut, nullptr));
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

aclnnStatus RunMropeSectionScene(const vector<int64_t> &section)
{
    return RunCompleteMropeScene(section);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeSectionWrongLength)
{
    const vector<vector<int64_t>> invalidSections = {{}, {22, 21}, {22, 21, 21, 0}};
    for (const auto &section : invalidSections) {
        EXPECT_EQ(RunMropeSectionScene(section), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsNegativeMropeSectionValue)
{
    const vector<vector<int64_t>> invalidSections = {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
    for (const auto &section : invalidSections) {
        EXPECT_EQ(RunMropeSectionScene(section), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeSectionSumGreaterThanHalfD)
{
    EXPECT_EQ(RunMropeSectionScene({22, 21, 22}), ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeSectionAxisUpperBound)
{
    // Keep each sum at or below 64 so the per-axis checks are reached independently.
    const vector<vector<int64_t>> invalidSections = {{23, 20, 20}, {20, 22, 20}, {20, 20, 22}};
    for (const auto &section : invalidSections) {
        EXPECT_EQ(RunMropeSectionScene(section), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, acceptsMropeSectionBoundary)
{
    EXPECT_EQ(RunMropeSectionScene({22, 21, 21}), ACLNN_SUCCESS);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropePositionSectionMismatch)
{
    TensorDesc queryStartLoc = MakeTensorDesc({1}, ACL_INT32);
    TensorDesc seqLens = MakeTensorDesc({1}, ACL_INT32);
    TensorDesc mropePosition = MakeTensorDesc({1, 3}, ACL_INT32);

    // Scene validation runs before the regular tensor checker, so only scene arguments are needed here.
    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, queryStartLoc, seqLens, nullptr,
              nullptr, mropePosition, nullptr, "TND", "NTD", 1e-6f, nullptr, "PerTokenPerHead"),
        OUTPUT(nullptr, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropePositionAxisMajorShape)
{
    TensorDesc qkv = MakeTensorDesc({5, 1, 128}, ACL_BF16);
    TensorDesc mropePosition = MakeTensorDesc({3, 5}, ACL_INT32);
    TensorDesc qOut = MakeTensorDesc({5, 1, 128}, ACL_BF16, false);
    IntArrayDesc mropeSection(vector<int64_t>{22, 12, 10});

    // M-RoPE positions are token-major: [T, 3], with T/H/W in the columns.
    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(qkv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, mropePosition, nullptr, "TND", "TND", 1e-6f, mropeSection, "NoQuant"),
        OUTPUT(qOut, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropePositionTokenCountMismatch)
{
    TensorDesc qkv = MakeTensorDesc({5, 1, 128}, ACL_BF16);
    TensorDesc mropePosition = MakeTensorDesc({4, 3}, ACL_INT32);
    TensorDesc qOut = MakeTensorDesc({5, 1, 128}, ACL_BF16, false);
    IntArrayDesc mropeSection(vector<int64_t>{22, 12, 10});

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(qkv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, mropePosition, nullptr, "TND", "TND", 1e-6f, mropeSection, "NoQuant"),
        OUTPUT(qOut, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropePositionTokenCountMismatchForNtdQkv)
{
    TensorDesc qkv = MakeTensorDesc({1, 5, 128}, ACL_BF16);
    TensorDesc mropePosition = MakeTensorDesc({4, 3}, ACL_INT32);
    TensorDesc qOut = MakeTensorDesc({1, 5, 128}, ACL_BF16, false);
    IntArrayDesc mropeSection(vector<int64_t>{22, 12, 10});

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(qkv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, mropePosition, nullptr, "NTD", "NTD", 1e-6f, mropeSection, "NoQuant"),
        OUTPUT(qOut, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, acceptsTokenMajorMropePosition)
{
    EXPECT_EQ(RunCompleteMropeScene({22, 12, 10}), ACLNN_SUCCESS);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeWithRopePositionSource)
{
    TensorDesc queryStartLoc = MakeTensorDesc({2}, ACL_INT32);
    TensorDesc seqLens = MakeTensorDesc({1}, ACL_INT32);
    TensorDesc mropePosition = MakeTensorDesc({1, 3}, ACL_INT32);
    IntArrayDesc mropeSection(vector<int64_t>{22, 12, 10});
    TensorDesc qOut = MakeTensorDesc({1, 1, 128}, ACL_BF16, false);

    // A valid M-RoPE pair must not be combined with the RoPE position source.
    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, queryStartLoc, seqLens, nullptr,
              nullptr, mropePosition, nullptr, "TND", "NTD", 1e-6f, mropeSection, "NoQuant"),
        OUTPUT(qOut, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeWithQueryStartLoc)
{
    TensorDesc queryStartLoc = MakeTensorDesc({2}, ACL_INT32);
    TensorDesc mropePosition = MakeTensorDesc({1, 3}, ACL_INT32);
    IntArrayDesc mropeSection(vector<int64_t>{22, 12, 10});
    TensorDesc qOut = MakeTensorDesc({1, 1, 128}, ACL_BF16, false);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, queryStartLoc, nullptr, nullptr,
              nullptr, mropePosition, nullptr, "TND", "NTD", 1e-6f, mropeSection, "NoQuant"),
        OUTPUT(qOut, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeWithSeqLens)
{
    TensorDesc seqLens = MakeTensorDesc({1}, ACL_INT32);
    TensorDesc mropePosition = MakeTensorDesc({1, 3}, ACL_INT32);
    IntArrayDesc mropeSection(vector<int64_t>{22, 12, 10});
    TensorDesc qOut = MakeTensorDesc({1, 1, 128}, ACL_BF16, false);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, seqLens, nullptr,
              nullptr, mropePosition, nullptr, "TND", "NTD", 1e-6f, mropeSection, "NoQuant"),
        OUTPUT(qOut, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsUnsupportedMropeLayouts)
{
    EXPECT_EQ(RunCompleteMropeScene({22, 12, 10}, "NTD", "NTD"), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(RunCompleteMropeScene({22, 12, 10}, "TND", "NTD"), ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeKCacheDtype)
{
    EXPECT_EQ(RunCompleteMropeScene({22, 12, 10}, "TND", "TND", ACL_FLOAT8_E4M3FN),
              ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeQOutDtype)
{
    EXPECT_EQ(RunCompleteMropeScene({22, 12, 10}, "TND", "TND", ACL_INT8, ACL_FLOAT8_E4M3FN),
              ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeQScale)
{
    EXPECT_EQ(RunCompleteMropeScene({22, 12, 10}, "TND", "TND", ACL_INT8, ACL_BF16, "NoQuant", true),
              ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsMropeQQuantMode)
{
    EXPECT_EQ(RunCompleteMropeScene({22, 12, 10}, "TND", "TND", ACL_INT8, ACL_BF16, "PerTokenPerHead"),
              ACLNN_ERR_PARAM_INVALID);
}

TEST(qkv_rms_norm_rope_cache_with_k_scale_scene_test, rejectsRopeDtypeReadFromQOut)
{
    TensorDesc qOut = MakeTensorDesc({1, 1, 128}, ACL_BF16, false);

    // RoPE requires FP8; ACLNN reads the effective dtype from qOut itself.
    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, "TND", "NTD", 1e-6f, nullptr, "PerTokenPerHead"),
        OUTPUT(qOut, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
} // namespace
