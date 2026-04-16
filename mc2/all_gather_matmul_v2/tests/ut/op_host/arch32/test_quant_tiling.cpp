/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <map>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tiling_context_faker.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topology_mocker.h"
#include "kernel_tiling/kernel_tiling.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace all_gather_matmul_v2_ut {

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;
using TilingFunc = std::function<ge::graphStatus(gert::TilingContext *)>;

const TensorDesc TD_EMPTY = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};

static TilingFunc GetTilingFunc()
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("AllGatherMatmulV2");
    return opImpl->tiling;
}

static void ParsePlatFormInfos(const std::string &compileInfoStr, std::map<std::string, std::string> &socInfos,
                               std::map<std::string, std::string> &aicoreSpec,
                               std::map<std::string, std::string> &intrinsics)
{
    nlohmann::json compileInfoJson = nlohmann::json::parse(compileInfoStr);
    std::map<std::string, std::string> socInfoKeys = {{"ai_core_cnt", "CORE_NUM"},
                                                      {"l2_size", "L2_SIZE"},
                                                      {"cube_core_cnt", "cube_core_cnt"},
                                                      {"vector_core_cnt", "vector_core_cnt"},
                                                      {"core_type_list", "core_type_list"}};
    for (auto &t : socInfoKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            auto &objJson = compileInfoJson["hardware_info"][t.second];
            if (objJson.is_number_integer()) {
                socInfos[t.first] = std::to_string(objJson.get<uint32_t>());
            } else if (objJson.is_string()) {
                socInfos[t.first] = objJson;
            }
        }
    }
    if (socInfos.find("cube_core_cnt") != socInfos.end() && socInfos.find("vector_core_cnt") != socInfos.end()) {
        socInfos["core_type_list"] = "CubeCore,VectorCore";
    } else {
        socInfos["core_type_list"] = "AICore";
    }
    std::map<std::string, std::string> aicoreSpecKeys = {{"ub_size", "UB_SIZE"},
                                                         {"l0_a_size", "L0A_SIZE"},
                                                         {"l0_b_size", "L0B_SIZE"},
                                                         {"l0_c_size", "L0C_SIZE"},
                                                         {"l1_size", "L1_SIZE"},
                                                         {"bt_size", "BT_SIZE"},
                                                         {"load3d_constraints", "load3d_constraints"}};
    aicoreSpec["cube_freq"] = "cube_freq";
    for (auto &t : aicoreSpecKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            if (t.second == "load3d_constraints") {
                aicoreSpec[t.first] = compileInfoJson["hardware_info"][t.second].get<std::string>();
            } else {
                aicoreSpec[t.first] = std::to_string(compileInfoJson["hardware_info"][t.second].get<uint32_t>());
            }
        }
    }
    std::vector<std::string> intrinsicsKeys = {"Intrinsic_data_move_l12ub",  "Intrinsic_data_move_l0c2ub",
                                               "Intrinsic_fix_pipe_l0c2out", "Intrinsic_data_move_out2l1_nd2nz",
                                               "Intrinsic_matmul_ub_to_ub",  "Intrinsic_conv_ub_to_ub",
                                               "Intrinsic_data_move_l12bt"};
    for (auto &key : intrinsicsKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(key) &&
            compileInfoJson["hardware_info"][key].get<bool>()) {
            intrinsics[key] = "float16";
            if (key.find("Intrinsic_data_move_l12bt") != std::string::npos) {
                intrinsics[key] = "bf16";
            }
        }
    }
}

class QuantTilingTest : public ::testing::Test {
protected:
public:
    static TilingFunc s_tilingFunc;

    static void SetUpTestCase()
    {
        s_tilingFunc = GetTilingFunc();
        ASSERT_NE(s_tilingFunc, nullptr);
    }
};

TilingFunc QuantTilingTest::s_tilingFunc = nullptr;

static gert::StorageShape S(std::initializer_list<int64_t> dims)
{
    return gert::StorageShape(dims, dims);
}

static ge::graphStatus RunTiling(const std::vector<TensorDesc> &inputs, const std::vector<TensorDesc> &outputs,
                                 const std::vector<OpAttr> &attrs, const std::string &socVersion, uint64_t coreNum,
                                 uint64_t ubSize, int64_t mockRankNum = 8)
{
    Mc2Hcom::MockValues hcomMockValues{{"rankNum", mockRankNum}};
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomMockValues);

    gert::TilingContextFaker contextFaker;
    std::vector<uint32_t> inputIrInstance;
    std::vector<uint32_t> outputIrInstance;
    std::vector<gert::Tensor *> inputTensors;
    std::vector<gert::Tensor *> outputTensors;
    std::vector<std::unique_ptr<gert::Tensor>> inputKeepAlive;
    std::vector<std::unique_ptr<gert::Tensor>> outputKeepAlive;

    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i].shape_.GetStorageShape().GetDimNum() == 0) {
            inputIrInstance.push_back(0);
        } else {
            inputIrInstance.push_back(1);
            auto t = std::make_unique<gert::Tensor>(
                inputs[i].shape_, gert::StorageFormat(inputs[i].format_, inputs[i].format_, gert::ExpandDimsType()),
                gert::TensorPlacement::kOnHost, inputs[i].dtype_, nullptr);
            inputTensors.push_back(t.get());
            inputKeepAlive.push_back(std::move(t));
        }
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        if (outputs[i].shape_.GetStorageShape().GetDimNum() == 0) {
            outputIrInstance.push_back(0);
        } else {
            outputIrInstance.push_back(1);
            auto t = std::make_unique<gert::Tensor>(
                outputs[i].shape_, gert::StorageFormat(outputs[i].format_, outputs[i].format_, gert::ExpandDimsType()),
                gert::TensorPlacement::kOnHost, outputs[i].dtype_, nullptr);
            outputTensors.push_back(t.get());
            outputKeepAlive.push_back(std::move(t));
        }
    }
    contextFaker.IrInstanceNum(inputIrInstance, outputIrInstance);
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors);

    for (auto &a : attrs) {
        switch (a.attr_.type_) {
            case Ops::Transformer::AnyValue::VT_BOOL:
                contextFaker.Attr(a.attrName_, *reinterpret_cast<bool *>(a.attr_.valuePtr_.get()));
                break;
            case Ops::Transformer::AnyValue::VT_INT:
                contextFaker.Attr(a.attrName_, *reinterpret_cast<int64_t *>(a.attr_.valuePtr_.get()));
                break;
            case Ops::Transformer::AnyValue::VT_STRING:
                contextFaker.Attr(a.attrName_,
                                  ge::AscendString(reinterpret_cast<std::string *>(a.attr_.valuePtr_.get())->c_str()));
                break;
            default:
                break;
        }
    }

    struct CompileInfo {};
    CompileInfo compileInfo;
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);

    auto contextHolder = contextFaker.SetOpType("AllGatherMatmulV2")
                             .CompileInfo(&compileInfo)
                             .PlatformInfo(&platformInfo)
                             .TilingData(tilingData.get())
                             .Workspace(reinterpret_cast<gert::ContinuousVector *>(wsHolder.get()))
                             .Build();

    std::string ciStr = "{\"hardware_info\": {\"BT_SIZE\": 0, \"load3d_constraints\": \"1\", "
                        "\"Intrinsic_fix_pipe_l0c2out\": false, \"Intrinsic_data_move_l12ub\": true, "
                        "\"Intrinsic_data_move_l0c2ub\": true, \"Intrinsic_data_move_out2l1_nd2nz\": false, "
                        "\"UB_SIZE\": " +
                        std::to_string(ubSize) +
                        ", \"L2_SIZE\": 33554432, \"L1_SIZE\": 524288, "
                        "\"L0A_SIZE\": 65536, \"L0B_SIZE\": 65536, \"L0C_SIZE\": 131072, "
                        "\"CORE_NUM\": " +
                        std::to_string(coreNum) + ", \"socVersion\":\"" + socVersion + "\"} }";

    std::map<std::string, std::string> socInfos, aicoreSpec, intrinsics;
    ParsePlatFormInfos(ciStr, socInfos, aicoreSpec, intrinsics);

    std::map<std::string, std::string> socToArch = {{"Ascend910B", "2201"}, {"Ascend910_93", "2201"}};
    std::map<std::string, std::string> versions = {{"NpuArch", socToArch[socVersion]},
                                                   {"Short_SoC_version", socVersion}};

    auto ctx = contextHolder.GetContext();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    ctx->GetPlatformInfo()->SetPlatformRes("version", versions);

    auto result = QuantTilingTest::s_tilingFunc(ctx);

    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    return result;
}

static std::vector<OpAttr> MakeAttrs(const std::string &group, bool is_trans_a, bool is_trans_b, int64_t gather_index,
                                     int64_t comm_turn, int64_t rank_size, int64_t block_size, int64_t group_size,
                                     bool is_gather_out, bool is_amax_out, int64_t y_dtype,
                                     const std::string &comm_mode)
{
    return {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(is_trans_a)},
        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(is_trans_b)},
        {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(gather_index)},
        {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(comm_turn)},
        {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rank_size)},
        {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(block_size)},
        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(group_size)},
        {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(is_gather_out)},
        {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(is_amax_out)},
        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(y_dtype)},
        {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(comm_mode)},
    };
}

#define QUANT_TEST(test_name, soc, rk, a_dtype, b_dtype, m, k, n, x1s, x2s)                                            \
    TEST_F(QuantTilingTest, test_name)                                                                                 \
    {                                                                                                                  \
        auto attrs = MakeAttrs("hcclCom", false, false, 0, 0, rk, 0, 0, true, false, 0, "aiv");                        \
        auto result = RunTiling({TensorDesc(S({m, k}), a_dtype, ge::FORMAT_ND),                                        \
                                 TensorDesc(S({k, n}), b_dtype, ge::FORMAT_ND), TD_EMPTY, x1s, x2s, TD_EMPTY},         \
                                {TensorDesc(S({m * rk, n}), ge::DT_FLOAT16, ge::FORMAT_ND),                            \
                                 TensorDesc(S({m * rk, k}), a_dtype, ge::FORMAT_ND), TD_EMPTY},                        \
                                attrs, soc, 30, 262144, rk);                                                           \
        EXPECT_EQ(result, ge::GRAPH_FAILED);                                                                           \
    }

QUANT_TEST(int8_rank2_910b, "Ascend910B", 2, ge::DT_INT8, ge::DT_INT8, 256, 512, 1024,
           TensorDesc(S({1}), ge::DT_FLOAT, ge::FORMAT_ND), TensorDesc(S({1024}), ge::DT_FLOAT, ge::FORMAT_ND))

QUANT_TEST(int8_rank4_910b, "Ascend910B", 4, ge::DT_INT8, ge::DT_INT8, 256, 512, 1024,
           TensorDesc(S({1}), ge::DT_FLOAT, ge::FORMAT_ND), TensorDesc(S({1024}), ge::DT_FLOAT, ge::FORMAT_ND))

QUANT_TEST(int8_rank8_910b, "Ascend910B", 8, ge::DT_INT8, ge::DT_INT8, 256, 512, 1024,
           TensorDesc(S({1}), ge::DT_FLOAT, ge::FORMAT_ND), TensorDesc(S({1024}), ge::DT_FLOAT, ge::FORMAT_ND))

QUANT_TEST(int8_rank8_per_channel_910b, "Ascend910B", 8, ge::DT_INT8, ge::DT_INT8, 256, 512, 1024, TD_EMPTY,
           TensorDesc(S({1024}), ge::DT_FLOAT, ge::FORMAT_ND))

QUANT_TEST(int4_rank8_910b, "Ascend910B", 8, ge::DT_INT4, ge::DT_INT4, 256, 512, 1024, TD_EMPTY,
           TensorDesc(S({1024}), ge::DT_FLOAT, ge::FORMAT_ND))

QUANT_TEST(int8_rank4_91093, "Ascend910_93", 4, ge::DT_INT8, ge::DT_INT8, 256, 512, 1024,
           TensorDesc(S({1}), ge::DT_FLOAT, ge::FORMAT_ND), TensorDesc(S({1024}), ge::DT_FLOAT, ge::FORMAT_ND))

QUANT_TEST(int8_rank8_91093, "Ascend910_93", 8, ge::DT_INT8, ge::DT_INT8, 256, 512, 1024,
           TensorDesc(S({1}), ge::DT_FLOAT, ge::FORMAT_ND), TensorDesc(S({1024}), ge::DT_FLOAT, ge::FORMAT_ND))

TEST_F(QuantTilingTest, fp16_rank2_910b)
{
    auto attrs = MakeAttrs("hcclCom", false, false, 0, 0, 2, 0, 0, true, false, 0, "aiv");
    auto result =
        RunTiling({TensorDesc(S({512, 8192}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({8192, 1280}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY, TD_EMPTY, TD_EMPTY, TD_EMPTY},
                  {TensorDesc(S({1024, 1280}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({1024, 8192}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY},
                  attrs, "Ascend910B", 30, 262144, 2);
    EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}

TEST_F(QuantTilingTest, fp16_unaligned_m_910b)
{
    auto attrs = MakeAttrs("hcclCom", false, false, 0, 0, 8, 0, 0, true, false, 0, "aiv");
    auto result =
        RunTiling({TensorDesc(S({513, 8192}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({8192, 1280}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY, TD_EMPTY, TD_EMPTY, TD_EMPTY},
                  {TensorDesc(S({4104, 1280}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({4104, 8192}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY},
                  attrs, "Ascend910B", 30, 262144);
    EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}

TEST_F(QuantTilingTest, fp16_unaligned_k_910b)
{
    auto attrs = MakeAttrs("hcclCom", false, false, 0, 0, 8, 0, 0, true, false, 0, "aiv");
    auto result =
        RunTiling({TensorDesc(S({512, 8193}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({8193, 1280}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY, TD_EMPTY, TD_EMPTY, TD_EMPTY},
                  {TensorDesc(S({4096, 1280}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({4096, 8193}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY},
                  attrs, "Ascend910B", 30, 262144);
    EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}

TEST_F(QuantTilingTest, fp16_unaligned_n_910b)
{
    auto attrs = MakeAttrs("hcclCom", false, false, 0, 0, 8, 0, 0, true, false, 0, "aiv");
    auto result =
        RunTiling({TensorDesc(S({512, 8192}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({8192, 1281}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY, TD_EMPTY, TD_EMPTY, TD_EMPTY},
                  {TensorDesc(S({4096, 1281}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({4096, 8192}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY},
                  attrs, "Ascend910B", 30, 262144);
    EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}

TEST_F(QuantTilingTest, fp16_trans_b_shape_mismatch_910b)
{
    auto attrs = MakeAttrs("hcclCom", false, true, 0, 0, 8, 0, 0, true, false, 0, "aiv");
    auto result =
        RunTiling({TensorDesc(S({512, 8192}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({1280, 8192}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY, TD_EMPTY, TD_EMPTY, TD_EMPTY},
                  {TensorDesc(S({4096, 1280}), ge::DT_FLOAT16, ge::FORMAT_ND),
                   TensorDesc(S({4096, 8192}), ge::DT_FLOAT16, ge::FORMAT_ND), TD_EMPTY},
                  attrs, "Ascend910B", 30, 262144);
    EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}

TEST_F(QuantTilingTest, bf16_rank2_910b)
{
    auto attrs = MakeAttrs("hcclCom", false, false, 0, 0, 2, 0, 0, true, false, 0, "aiv");
    auto result =
        RunTiling({TensorDesc(S({512, 8192}), ge::DT_BF16, ge::FORMAT_ND),
                   TensorDesc(S({8192, 1280}), ge::DT_BF16, ge::FORMAT_ND), TD_EMPTY, TD_EMPTY, TD_EMPTY, TD_EMPTY},
                  {TensorDesc(S({1024, 1280}), ge::DT_BF16, ge::FORMAT_ND),
                   TensorDesc(S({1024, 8192}), ge::DT_BF16, ge::FORMAT_ND), TD_EMPTY},
                  attrs, "Ascend910B", 30, 262144, 2);
    EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}

TEST_F(QuantTilingTest, bf16_rank4_91093)
{
    auto attrs = MakeAttrs("hcclCom", false, false, 0, 0, 4, 0, 0, true, false, 0, "aiv");
    auto result =
        RunTiling({TensorDesc(S({512, 8192}), ge::DT_BF16, ge::FORMAT_ND),
                   TensorDesc(S({8192, 1280}), ge::DT_BF16, ge::FORMAT_ND), TD_EMPTY, TD_EMPTY, TD_EMPTY, TD_EMPTY},
                  {TensorDesc(S({2048, 1280}), ge::DT_BF16, ge::FORMAT_ND),
                   TensorDesc(S({2048, 8192}), ge::DT_BF16, ge::FORMAT_ND), TD_EMPTY},
                  attrs, "Ascend910_93", 30, 262144, 4);
    EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}

} // namespace all_gather_matmul_v2_ut
