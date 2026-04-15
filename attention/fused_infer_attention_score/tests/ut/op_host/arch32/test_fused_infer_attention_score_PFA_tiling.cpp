/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cctype>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "../fused_infer_attention_score_param.h"
#include "../../../../op_host/fused_infer_attention_score_tiling.h"
#include "platform/platform_infos_def.h"
#include "tiling_case_executor.h"
#include "softmax_tiling_mocker.h"

namespace FusedInferAttentionScoreUT {
namespace {

const char *A2_SOC_INFO =
    "{\n"
    "  \"hardware_info\": {\n"
    "    \"BT_SIZE\": 0,\n"
    "    \"load3d_constraints\": \"1\",\n"
    "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
    "    \"Intrinsic_data_move_l12ub\": true,\n"
    "    \"Intrinsic_data_move_l0c2ub\": true,\n"
    "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
    "    \"UB_SIZE\": 196608,\n"
    "    \"L2_SIZE\": 201326592,\n"
    "    \"L1_SIZE\": 524288,\n"
    "    \"L0A_SIZE\": 65536,\n"
    "    \"L0B_SIZE\": 65536,\n"
    "    \"L0C_SIZE\": 131072,\n"
    "    \"vector_core_cnt\": 40,\n"
    "    \"cube_core_cnt\": 20,\n"
    "    \"socVersion\": \"Ascend910_B3\"\n"
    "  }\n"
    "}";

gert::TilingContextPara MakeFiaContext(
    const std::vector<gert::TilingContextPara::TensorDescription> &inputs,
    const std::vector<gert::TilingContextPara::TensorDescription> &outputs,
    const std::vector<gert::TilingContextPara::OpAttr> &attrs)
{
    static struct EmptyCompileInfo {} compileInfo;
    return gert::TilingContextPara(
        "FusedInferAttentionScore",
        inputs,
        outputs,
        attrs,
        &compileInfo,
        "Ascend910B",
        40,
        196608,
        4096,
        A2_SOC_INFO);
}

std::vector<gert::TilingContextPara::OpAttr> MakeCommonAttrs(
    const std::string &layout,
    int64_t sparseMode,
    int64_t blockSize,
    int64_t keyAntiquantMode,
    int64_t valueAntiquantMode,
    int64_t innerPrecise = 1,
    int64_t numHeads = 8,
    int64_t numKvHeads = 8)
{
    return {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeads)},
        {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
        {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
        {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numKvHeads)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
        {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(innerPrecise)},
        {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize)},
        {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(keyAntiquantMode)},
        {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(valueAntiquantMode)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
}

class FusedInferAttentionScorePfaTiling : public testing::Test {
protected:
    void SetUp() override
    {
        SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B");
    }

    void TearDown() override
    {
        SoftmaxTilingMocker::GetInstance().Reset();
    }
};

bool IsNpuArchString(const std::string &version)
{
    if (version.empty()) {
        return false;
    }
    for (char c : version) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

void GetPlatFormInfosLocal(const char *compileInfoStr,
    std::map<std::string, std::string> &socInfos,
    std::map<std::string, std::string> &aicoreSpec,
    std::map<std::string, std::string> &intrinsics)
{
    std::string defaultHardwareInfo = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432,
                          "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
                          "CORE_NUM": 32}})";
    nlohmann::json compileInfoJson = nlohmann::json::parse(compileInfoStr);
    if (compileInfoJson.type() != nlohmann::json::value_t::object) {
        compileInfoJson = nlohmann::json::parse(defaultHardwareInfo.c_str());
    }

    std::map<std::string, std::string> socInfoKeys = {
        {"ai_core_cnt", "CORE_NUM"},
        {"l2_size", "L2_SIZE"},
        {"cube_core_cnt", "cube_core_cnt"},
        {"vector_core_cnt", "vector_core_cnt"},
        {"core_type_list", "core_type_list"}};
    for (auto &item : socInfoKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(item.second)) {
            auto &objJson = compileInfoJson["hardware_info"][item.second];
            if (objJson.is_number_integer()) {
                socInfos[item.first] = std::to_string(objJson.get<uint32_t>());
            } else if (objJson.is_string()) {
                socInfos[item.first] = objJson;
            }
        }
    }

    if (socInfos.find("cube_core_cnt") != socInfos.end() && socInfos.find("vector_core_cnt") != socInfos.end()) {
        socInfos["core_type_list"] = "CubeCore,VectorCore";
    } else {
        socInfos["core_type_list"] = "AICore";
    }

    std::map<std::string, std::string> aicoreSpecKeys = {
        {"ub_size", "UB_SIZE"},
        {"l0_a_size", "L0A_SIZE"},
        {"l0_b_size", "L0B_SIZE"},
        {"l0_c_size", "L0C_SIZE"},
        {"l1_size", "L1_SIZE"},
        {"bt_size", "BT_SIZE"},
        {"load3d_constraints", "load3d_constraints"}};
    aicoreSpec["cube_freq"] = "cube_freq";
    for (auto &item : aicoreSpecKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(item.second)) {
            if (item.second == "load3d_constraints") {
                aicoreSpec[item.first] = compileInfoJson["hardware_info"][item.second].get<std::string>();
            } else {
                aicoreSpec[item.first] = std::to_string(compileInfoJson["hardware_info"][item.second].get<uint32_t>());
            }
        }
    }

    const std::string intrinsicKeys[] = {
        "Intrinsic_data_move_l12ub",
        "Intrinsic_data_move_l0c2ub",
        "Intrinsic_fix_pipe_l0c2out",
        "Intrinsic_data_move_out2l1_nd2nz",
        "Intrinsic_matmul_ub_to_ub",
        "Intrinsic_conv_ub_to_ub",
        "Intrinsic_data_move_l12bt"};
    for (const auto &key : intrinsicKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(key) &&
            compileInfoJson["hardware_info"][key].get<bool>()) {
            intrinsics[key] = key.find("Intrinsic_data_move_l12bt") != std::string::npos ? "bf16" : "float16";
        }
    }
}

void ExecuteDirectTiling(const gert::TilingContextPara &tilingContextPara,
    ge::graphStatus (*tilingFunc)(gert::TilingContext *),
    ge::graphStatus expectResult)
{
    auto contextFaker = gert::TilingContextFaker();
    size_t inputNum = tilingContextPara.inputTensorDesc_.size();
    size_t outputNum = tilingContextPara.outputTensorDesc_.size();
    std::vector<uint32_t> inputIrInstance;
    std::vector<uint32_t> outputIrInstance;
    std::vector<gert::Tensor *> inputTensors;
    std::vector<gert::Tensor *> outputTensors;
    std::vector<std::unique_ptr<gert::Tensor>> inputTensorsKeepAlive;
    std::vector<std::unique_ptr<gert::Tensor>> outputTensorsKeepAlive;

    for (size_t index = 0; index < inputNum; ++index) {
        const auto &desc = tilingContextPara.inputTensorDesc_[index];
        if (desc.shape_.GetStorageShape().GetDimNum() == 0) {
            inputIrInstance.push_back(0);
            continue;
        }
        inputIrInstance.push_back(1);
        auto tensor = std::make_unique<gert::Tensor>(
            desc.shape_,
            gert::StorageFormat(desc.format_, desc.format_, gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost,
            desc.dtype_,
            desc.isConst_ ? desc.constValue_ : nullptr);
        inputTensors.push_back(tensor.get());
        inputTensorsKeepAlive.push_back(std::move(tensor));
    }

    for (size_t index = 0; index < outputNum; ++index) {
        const auto &desc = tilingContextPara.outputTensorDesc_[index];
        if (desc.shape_.GetStorageShape().GetDimNum() == 0) {
            outputIrInstance.push_back(0);
            continue;
        }
        outputIrInstance.push_back(1);
        auto tensor = std::make_unique<gert::Tensor>(
            desc.shape_,
            gert::StorageFormat(desc.format_, desc.format_, gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost,
            desc.dtype_,
            desc.isConst_ ? desc.constValue_ : nullptr);
        outputTensors.push_back(tensor.get());
        outputTensorsKeepAlive.push_back(std::move(tensor));
    }

    if (!tilingContextPara.inputInstanceNum_.empty() || !tilingContextPara.outputInstanceNum_.empty()) {
        contextFaker.IrInstanceNum(tilingContextPara.inputInstanceNum_, tilingContextPara.outputInstanceNum_);
    } else {
        contextFaker.IrInstanceNum(inputIrInstance, outputIrInstance);
    }
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors);
    contextFaker.DeterministicInfo(tilingContextPara.deterministicInfo_);

    for (auto &attrInfo : tilingContextPara.attrs_) {
        switch (attrInfo.attr_.type_) {
            case Ops::Transformer::AnyValue::ValueType::VT_BOOL:
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool *>(attrInfo.attr_.valuePtr_.get()));
                break;
            case Ops::Transformer::AnyValue::ValueType::VT_INT:
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<int64_t *>(attrInfo.attr_.valuePtr_.get()));
                break;
            case Ops::Transformer::AnyValue::ValueType::VT_FLOAT:
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float *>(attrInfo.attr_.valuePtr_.get()));
                break;
            case Ops::Transformer::AnyValue::ValueType::VT_STRING:
                contextFaker.Attr(
                    attrInfo.attrName_,
                    ge::AscendString(reinterpret_cast<std::string *>(attrInfo.attr_.valuePtr_.get())->c_str()));
                break;
            default:
                FAIL() << "unsupported attr type " << attrInfo.attr_.type_;
        }
    }

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingData = gert::TilingData::CreateCap(tilingContextPara.tilingDataSize_);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    auto contextHolder = contextFaker.SetOpType(tilingContextPara.opName_.c_str())
                             .CompileInfo(tilingContextPara.compileInfo_)
                             .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                             .TilingData(tilingData.get())
                             .Workspace(workspace)
                             .Build();

    std::string compileInfoString =
        std::string("{\"hardware_info\": {\"BT_SIZE\": 0, \"load3d_constraints\": \"1\", ")
        + "\"Intrinsic_fix_pipe_l0c2out\": false, \"Intrinsic_data_move_l12ub\": true, "
        + "\"Intrinsic_data_move_l0c2ub\": true, \"Intrinsic_data_move_out2l1_nd2nz\": false, \"UB_SIZE\": "
        + std::to_string(tilingContextPara.ubSize_)
        + ", \"L2_SIZE\": 33554432, \"L1_SIZE\": 524288, \"L0A_SIZE\": 65536, \"L0B_SIZE\": 65536, "
        + "\"L0C_SIZE\": 131072, \"CORE_NUM\": "
        + std::to_string(tilingContextPara.coreNum_)
        + ", \"socVersion\":\"" + tilingContextPara.socVersion_ + "\"} }";
    if (!tilingContextPara.socInfoString_.empty()) {
        compileInfoString = tilingContextPara.socInfoString_;
    }

    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> versions;
    if (IsNpuArchString(tilingContextPara.socVersion_)) {
        versions = {{"NpuArch", tilingContextPara.socVersion_}, {"Short_SoC_version", "Ascend950"}};
    } else {
        std::map<std::string, std::string> socToArch = {
            {"Ascend310P", "2002"},
            {"Ascend910B", "2201"},
            {"Ascend910_93", "2201"},
            {"Ascend950", "3510"}};
        versions = {{"NpuArch", socToArch[tilingContextPara.socVersion_]},
                    {"Short_SoC_version", tilingContextPara.socVersion_}};
    }
    GetPlatFormInfosLocal(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    auto tilingContext = contextHolder.GetContext();
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", versions);

    ASSERT_EQ(tilingFunc(tilingContext), expectResult);
}

} // namespace

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_0)
{
    int64_t actualSeqKv[] = {256};
    auto tilingContextPara = MakeFiaContext(
        {
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeCommonAttrs("BNSD", 0, 127, 0, 0));
    ExecuteDirectTiling(tilingContextPara, optiling::DeviceDoOpTilingFusedInferAttentionScore, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_1)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    auto tilingContextPara = MakeFiaContext(
        {
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 128, 1024}, {16, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeCommonAttrs("TND", 1, 128, 0, 0));
    ExecuteDirectTiling(tilingContextPara, optiling::DeviceDoOpTilingFusedInferAttentionScore, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_2)
{
    int64_t actualSeqKv[] = {256};
    auto tilingContextPara = MakeFiaContext(
        {
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            TD_DEFAULT,
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeCommonAttrs("BSH", 3, 128, 1, 1));
    ExecuteDirectTiling(tilingContextPara, optiling::DeviceDoOpTilingFusedInferAttentionScore, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_3)
{
    int64_t actualSeqKv[] = {256};
    auto tilingContextPara = MakeFiaContext(
        {
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            TD_DEFAULT,
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeCommonAttrs("BSH", 0, 128, 0, 0));
    ExecuteDirectTiling(tilingContextPara, optiling::DeviceDoOpTilingFusedInferAttentionScore, ge::GRAPH_FAILED);
}

} // namespace FusedInferAttentionScoreUT
