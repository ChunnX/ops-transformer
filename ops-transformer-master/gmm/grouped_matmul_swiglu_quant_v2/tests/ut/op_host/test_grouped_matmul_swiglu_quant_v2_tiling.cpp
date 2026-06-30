/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_matmul_swiglu_quant_v2_tiling.cpp
 * \brief CSV-driven unit tests for grouped_matmul_swiglu_quant_v2 tiling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v2_base_tiling.h"
#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v2_fusion_tiling.h"
#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v2_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "gmm_csv_ge_parse_utils.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;
using namespace optiling;


namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

struct TensorDescParam {
    vector<int64_t> originDims;
    vector<int64_t> storageDims;
    ge::DataType dtype = ge::DT_UNDEFINED;
    ge::Format format = ge::FORMAT_ND;
};

constexpr size_t kInputTensorCount = 8U;
constexpr size_t kOutputTensorCount = 2U;
constexpr size_t kTensorFieldCount = 4U;
constexpr size_t kScalarFieldCount = 19U;
constexpr size_t kCsvColumnCount = kScalarFieldCount + (kInputTensorCount + kOutputTensorCount) * kTensorFieldCount;

struct GroupedMatmulSwigluQuantV2TilingCase {
    void Run() const
    {
        if (!enable) {
            return;
        }
        auto compileInfoForRun = compileInfo;
        vector<int64_t> tuningConfigVec = ops::ut::ParseDims(tuningConfig, {});
        if (tuningConfigVec.empty()) {
            tuningConfigVec = {0};
        }
        gert::TilingContextPara tilingContextPara(
            "GroupedMatmulSwigluQuantV2", BuildTensorDescs(inputs), BuildTensorDescs(outputs),
            {
                {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dequantMode)},
                {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dequantDtype)},
                {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantDtype)},
                {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tuningConfigVec)},
            },
            &compileInfoForRun);
        if (expectSuccess) {
            TilingInfo tilingInfo;
            ExecuteTiling(tilingContextPara, tilingInfo);
            EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
            return;
        } else {
            ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
        }
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    bool expectSuccess = false;
    int64_t expectTilingKey = 0;
    optiling::GMMSwigluV2CompileInfo compileInfo;
    vector<TensorDescParam> inputs;
    vector<TensorDescParam> outputs;
    int64_t dequantMode = 0;
    float dequantDtype = 0.0F;
    int64_t quantMode = 0;
    int64_t quantDtype = 0;
    bool transposeWeight = false;
    int64_t groupListType = 0;
    string tuningConfig;

private:
    static vector<gert::TilingContextPara::TensorDescription> BuildTensorDescs(const vector<TensorDescParam> &tensors)
    {
        vector<gert::TilingContextPara::TensorDescription> descs;
        descs.reserve(tensors.size());
        for (const auto &tensor : tensors) {
            descs.emplace_back(ops::ut::MakeGertStorageShape(tensor.originDims, tensor.storageDims), tensor.dtype,
                               tensor.format);
        }
        return descs;
    }
};

vector<GroupedMatmulSwigluQuantV2TilingCase> LoadCases(const string &socVersion)
{
    vector<GroupedMatmulSwigluQuantV2TilingCase> cases;
    string csvPath = ops::ut::ResolveCsvPath("test_grouped_matmul_swiglu_quant_v2_tiling.csv",
                                             "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_host", __FILE__);
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << endl;
        return cases;
    }

    string line;
    size_t lineNo = 0U;
    while (getline(csvData, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "socVersion" || items.size() < kCsvColumnCount) {
            continue;
        }

        const string caseName = items.size() > 1U ? Trim(items[1]) : "";
        try {
            size_t idx = 0;
            GroupedMatmulSwigluQuantV2TilingCase tc;
            tc.socVersion = Trim(items[idx++]);
            if (tc.socVersion != socVersion) {
                continue;
            }
            tc.caseName = Trim(items[idx++]);
            tc.enable = ParseBool(Trim(items[idx++]));
            if (!tc.enable) {
                continue;
            }
            tc.prefix = Trim(items[idx++]);
            tc.expectSuccess = ParseBool(Trim(items[idx++]));
            tc.expectTilingKey = stoll(Trim(items[idx++]));

            tc.compileInfo = {
                static_cast<uint64_t>(stoull(Trim(items[idx++]))), static_cast<uint32_t>(stoul(Trim(items[idx++]))),
                static_cast<uint32_t>(stoul(Trim(items[idx++]))),  static_cast<uint32_t>(stoul(Trim(items[idx++]))),
                static_cast<uint32_t>(stoul(Trim(items[idx++]))),  ParseBool(Trim(items[idx++])),
            };

            for (size_t tensorIdx = 0; tensorIdx < kInputTensorCount; ++tensorIdx) {
                tc.inputs.push_back({ops::ut::ParseDims(Trim(items[idx++])), ops::ut::ParseDims(Trim(items[idx++])),
                                     ops::ut::ParseGeDtype(Trim(items[idx++])),
                                     ops::ut::ParseGeFormat(Trim(items[idx++]))});
            }
            for (size_t tensorIdx = 0; tensorIdx < kOutputTensorCount; ++tensorIdx) {
                tc.outputs.push_back({ops::ut::ParseDims(Trim(items[idx++])), ops::ut::ParseDims(Trim(items[idx++])),
                                      ops::ut::ParseGeDtype(Trim(items[idx++])),
                                      ops::ut::ParseGeFormat(Trim(items[idx++]))});
            }

            tc.dequantMode = stoll(Trim(items[idx++]));
            tc.dequantDtype = stof(Trim(items[idx++]));
            tc.quantMode = stoll(Trim(items[idx++]));
            tc.quantDtype = stoll(Trim(items[idx++]));
            tc.transposeWeight = ParseBool(Trim(items[idx++]));
            tc.groupListType = stoll(Trim(items[idx++]));
            tc.tuningConfig = Trim(items[idx++]);
            cases.push_back(tc);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulSwigluQuantV2TilingCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedMatmulSwigluQuantV2UT {

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestGroupedMatmulSwigluQuantV2Tiling950);


const vector<GroupedMatmulSwigluQuantV2TilingCase> &GetAscend950Cases()
{
    static const vector<GroupedMatmulSwigluQuantV2TilingCase> cases = LoadCases("Ascend950");
    return cases;
}

const vector<GroupedMatmulSwigluQuantV2TilingCase> &GetAscend910BCases()
{
    static const vector<GroupedMatmulSwigluQuantV2TilingCase> cases = LoadCases("Ascend910B");
    return cases;
}

class TestGroupedMatmulSwigluQuantV2Tiling950 : public testing::TestWithParam<GroupedMatmulSwigluQuantV2TilingCase> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantV2Tiling950 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantV2Tiling950 TearDown" << std::endl;
    }
};

TEST_P(TestGroupedMatmulSwigluQuantV2Tiling950, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMSQ_V2_TILING_950, TestGroupedMatmulSwigluQuantV2Tiling950,
                         testing::ValuesIn(GetAscend950Cases()), MakeParamName);

class TestGroupedMatmulSwigluQuantV2Tiling910B : public testing::TestWithParam<GroupedMatmulSwigluQuantV2TilingCase> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantV2Tiling910B SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantV2Tiling910B TearDown" << std::endl;
    }
};

TEST_P(TestGroupedMatmulSwigluQuantV2Tiling910B, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMSQ_V2_TILING_910B, TestGroupedMatmulSwigluQuantV2Tiling910B,
                         testing::ValuesIn(GetAscend910BCases()), MakeParamName);

class TestTilingParseForGroupedMatmulSwigluQuantV2 : public testing::Test {
protected:
};

TEST_F(TestTilingParseForGroupedMatmulSwigluQuantV2, TilingPrepareSuccess)
{
    optiling::GMMSwigluV2CompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> socInfos = {
        {"ai_core_cnt", "24"}, {"l2_size", "33554432"}, {"core_type_list", "AICore"}};
    std::map<std::string, std::string> aicoreSpec = {{"ub_size", "201326592"},
                                                     {"l1_size", "524288"},
                                                     {"l0_a_size", "65536"},
                                                     {"l0_b_size", "65536"},
                                                     {"l0_c_size", "131072"}};
    std::map<std::string, std::string> intrinsics = {{"Intrinsic_data_move_l12ub", "float16"},
                                                     {"Intrinsic_data_move_l0c2ub", "float16"},
                                                     {"Intrinsic_fix_pipe_l0c2out", "float16"},
                                                     {"Intrinsic_data_move_out2l1_nd2nz", "float16"},
                                                     {"Intrinsic_data_move_l12bt", "bf16"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    const char *compileJsonStr = "{}";

    gert::OpTilingParseContextBuilder builder;
    builder.OpType("GroupedMatmulSwigluQuantV2")
        .OpName("GroupedMatmulSwigluQuantV2")
        .IONum(2, 1)
        .CompiledJson(compileJsonStr)
        .CompiledInfo(reinterpret_cast<void *>(&compileInfo))
        .PlatformInfo(reinterpret_cast<void *>(&platformInfo));
    auto holder = builder.Build();
    auto parseContext = holder.GetContext();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("GroupedMatmulSwigluQuantV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TestTilingParseForGroupedMatmulSwigluQuantV2, TilingDataFieldCoverage)
{
    optiling::GMMSwigluQuantV2TilingData tilingData;
    auto &bp = tilingData.gmmSwigluQuantV2BaseParams;
    bp.set_groupNum(1);
    EXPECT_EQ(bp.get_groupNum(), 1);
    bp.set_coreNum(24);
    EXPECT_EQ(bp.get_coreNum(), 24);
    bp.set_K(2048);
    EXPECT_EQ(bp.get_K(), 2048);
    bp.set_N(256);
    EXPECT_EQ(bp.get_N(), 256);
    bp.set_M(1024);
    EXPECT_EQ(bp.get_M(), 1024);
    bp.set_baseM(128);
    EXPECT_EQ(bp.get_baseM(), 128);
    bp.set_baseN(256);
    EXPECT_EQ(bp.get_baseN(), 256);
    bp.set_mLimit(32768);
    EXPECT_EQ(bp.get_mLimit(), 32768);
    bp.set_workSpaceOffset1(0);
    EXPECT_EQ(bp.get_workSpaceOffset1(), 0);
    bp.set_workSpaceOffset2(0);
    EXPECT_EQ(bp.get_workSpaceOffset2(), 0);
    bp.set_quantGroupNum(1);
    EXPECT_EQ(bp.get_quantGroupNum(), 1);
    bp.set_isSingleTensor(1);
    EXPECT_EQ(bp.get_isSingleTensor(), 1);
    bp.set_groupListType(0);
    EXPECT_EQ(bp.get_groupListType(), 0);
    bp.set_smoothScaleDimNum(0);
    EXPECT_EQ(bp.get_smoothScaleDimNum(), 0);
    bp.set_singleN(256);
    EXPECT_EQ(bp.get_singleN(), 256);

    auto &sq = tilingData.gmmSwigluQuantV2;
    sq.set_maxProcessRowNum(100);
    EXPECT_EQ(sq.get_maxProcessRowNum(), 100);
    sq.set_groupListLen(16);
    EXPECT_EQ(sq.get_groupListLen(), 16);
    sq.set_tokenLen(256);
    EXPECT_EQ(sq.get_tokenLen(), 256);
}

TEST_F(TestTilingParseForGroupedMatmulSwigluQuantV2, FusionTilingDataFieldCoverage)
{
    optiling::GMMSwigluQuantV2TilingFusionData fusionData;
    fusionData.set_cubeBlockDim(24);
    EXPECT_EQ(fusionData.get_cubeBlockDim(), 24);
    fusionData.set_vectorBlockDim(48);
    EXPECT_EQ(fusionData.get_vectorBlockDim(), 48);
    fusionData.set_groupNum(16);
    EXPECT_EQ(fusionData.get_groupNum(), 16);
    fusionData.set_K(2048);
    EXPECT_EQ(fusionData.get_K(), 2048);
    fusionData.set_N(7168);
    EXPECT_EQ(fusionData.get_N(), 7168);
    fusionData.set_M(1024);
    EXPECT_EQ(fusionData.get_M(), 1024);
    fusionData.set_ubFactorDimx(2);
    EXPECT_EQ(fusionData.get_ubFactorDimx(), 2);
    fusionData.set_ubFactorDimy(3584);
    EXPECT_EQ(fusionData.get_ubFactorDimy(), 3584);
    fusionData.set_actRight(1);
    EXPECT_EQ(fusionData.get_actRight(), 1);
    fusionData.set_groupListType(0);
    EXPECT_EQ(fusionData.get_groupListType(), 0);
    fusionData.set_isSingleTensor(1);
    EXPECT_EQ(fusionData.get_isSingleTensor(), 1);
}

} // namespace GroupedMatmulSwigluQuantV2UT
