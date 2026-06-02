/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <limits>
#include <map>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v3_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "platform/platform_infos_def.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include <nlohmann/json.hpp>

using namespace std;

namespace {
void InitParsePlatformInfo(fe::PlatFormInfos &platformInfo)
{
    platformInfo.Init();
    const char *compileJson =
        R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "BT_SIZE": 0, "load3d_constraints": "1", "socVersion":"Ascend910B"}})";
    nlohmann::json compileInfoJson = nlohmann::json::parse(compileJson);
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    socInfos["ai_core_cnt"] = to_string(compileInfoJson["hardware_info"]["CORE_NUM"].get<uint32_t>());
    socInfos["l2_size"] = to_string(compileInfoJson["hardware_info"]["L2_SIZE"].get<uint32_t>());
    socInfos["core_type_list"] = "AICore";
    aicoreSpec["ub_size"] = to_string(compileInfoJson["hardware_info"]["UB_SIZE"].get<uint32_t>());
    aicoreSpec["l1_size"] = to_string(compileInfoJson["hardware_info"]["L1_SIZE"].get<uint32_t>());
    map<string, string> versions = {{"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("version", versions);
}
} // namespace

constexpr int64_t QUANT_MODE_NONE = -1;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t ROW_IDX_TYPE_DROPPAD = 0;
constexpr int64_t ROW_IDX_TYPE_DROPLESS = 1;
constexpr int64_t EXPERT_NUM = 256;
constexpr uint64_t SKIP_TILING_KEY_VALIDATION = std::numeric_limits<uint64_t>::max();
constexpr uint64_t EMPTY_TENSOR_TILINGKEY = 3000000ULL;
constexpr size_t EMPTY_TENSOR_WORKSPACE = 16ULL * 1024ULL * 1024ULL;
constexpr int64_t EXPERT_TOKENS_TYPE_COUNT = 1;
constexpr int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2;

gert::StorageShape MakeShape(const std::vector<int64_t> &dims)
{
    switch (dims.size()) {
        case 0:
            return gert::StorageShape({}, {});
        case 1:
            return gert::StorageShape({dims[0]}, {dims[0]});
        case 2:
            return gert::StorageShape({dims[0], dims[1]}, {dims[0], dims[1]});
        case 3:
            return gert::StorageShape({dims[0], dims[1], dims[2]}, {dims[0], dims[1], dims[2]});
        default:
            return gert::StorageShape({}, {});
    }
}

gert::TilingContextPara MakeExtendedTilingContextPara(
    const std::vector<gert::TilingContextPara::TensorDescription> &inputs,
    const std::vector<gert::TilingContextPara::TensorDescription> &outputs,
    const std::vector<gert::TilingContextPara::OpAttr> &attrs,
    optiling::MoeInitRoutingV3CompileInfo *compileInfo)
{
    return gert::TilingContextPara("MoeInitRoutingV3", inputs, outputs, attrs, compileInfo);
}

void RunExtendedTilingCase(const gert::TilingContextPara &tilingContextPara,
                           ge::graphStatus expectResult = ge::GRAPH_SUCCESS,
                           uint64_t expectTilingKey = SKIP_TILING_KEY_VALIDATION,
                           const std::vector<size_t> &expectWorkspaces = {})
{
    ExecuteTestCase(tilingContextPara, expectResult, expectTilingKey, "", expectWorkspaces);
}

ge::DataType GetLargeColsExpandedXDtype(int64_t quantMode, ge::DataType xDataType)
{
    if (quantMode == QUANT_MODE_NONE) {
        return xDataType;
    }
    return ge::DT_INT8;
}

std::vector<int64_t> GetLargeColsExpertTokensShape(int64_t expertTokensNumType, int64_t expertNum, int64_t expertRange)
{
    if (expertTokensNumType == EXPERT_TOKENS_TYPE_KEY_VALUE) {
        return {expertNum, 2};
    }
    return {expertRange};
}

std::vector<gert::TilingContextPara::TensorDescription> BuildLargeColsInputs(
    int64_t n, int64_t h, int64_t k, ge::DataType xDataType,
    const std::vector<gert::TilingContextPara::TensorDescription> &extraInputs)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.emplace_back(gert::StorageShape({n, h}, {n, h}), xDataType, ge::FORMAT_ND);
    inputs.emplace_back(gert::StorageShape({n, k}, {n, k}), ge::DT_INT32, ge::FORMAT_ND);
    inputs.insert(inputs.end(), extraInputs.begin(), extraInputs.end());
    return inputs;
}

std::vector<gert::TilingContextPara::TensorDescription> BuildLargeColsOutputs(
    int64_t totalLength, int64_t h, ge::DataType expandedXDtype, const std::vector<int64_t> &expertTokensShape,
    const std::vector<gert::TilingContextPara::TensorDescription> &extraOutputs)
{
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(gert::StorageShape({totalLength, h}, {totalLength, h}), expandedXDtype, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({totalLength}, {totalLength}), ge::DT_INT32, ge::FORMAT_ND);
    outputs.emplace_back(MakeShape(expertTokensShape), ge::DT_INT64, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({totalLength}, {totalLength}), ge::DT_FLOAT, ge::FORMAT_ND);
    outputs.insert(outputs.end(), extraOutputs.begin(), extraOutputs.end());
    return outputs;
}

std::vector<gert::TilingContextPara::OpAttr> BuildLargeColsAttrs(
    int64_t totalLength, int64_t dropPadMode, int64_t expertTokensNumType, bool expertTokensNumFlag, int64_t quantMode,
    const std::vector<int64_t> &aciveExpertRange, int64_t rowIdxType)
{
    return {
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(totalLength)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0LL)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
        {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertTokensNumType)},
        {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(expertTokensNumFlag)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
        {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
        {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
    };
}

void RunLargeColsCase(int64_t N, int64_t H, int64_t K, int64_t quantMode, int64_t dropPadMode,
                      int64_t expertTokensNumType, bool expertTokensNumFlag, ge::DataType xDataType,
                      std::vector<int64_t> aciveExpertRange, int64_t rowIdxType, uint64_t ubSize,
                      const std::vector<gert::TilingContextPara::TensorDescription> &extraInputs,
                      const std::vector<gert::TilingContextPara::TensorDescription> &extraOutputs)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, ubSize};
    int64_t expertNum = EXPERT_NUM;
    int64_t expertRange = aciveExpertRange.empty() ? expertNum : (aciveExpertRange[1] - aciveExpertRange[0]);
    int64_t totalLength = N * K;
    ge::DataType expandedXDtype = GetLargeColsExpandedXDtype(quantMode, xDataType);
    std::vector<int64_t> expertTokensShape =
        GetLargeColsExpertTokensShape(expertTokensNumType, expertNum, expertRange);
    auto inputs = BuildLargeColsInputs(N, H, K, xDataType, extraInputs);
    auto outputs = BuildLargeColsOutputs(totalLength, H, expandedXDtype, expertTokensShape, extraOutputs);
    auto attrs = BuildLargeColsAttrs(totalLength, dropPadMode, expertTokensNumType, expertTokensNumFlag, quantMode,
                                     aciveExpertRange, rowIdxType);
    RunExtendedTilingCase(MakeExtendedTilingContextPara(inputs, outputs, attrs, &compileInfo));
}

class MoeInitRoutingV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling TearDown" << std::endl;
    }
};

void RunNormalCase(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    ge::DataType expandedx_type;
    if (quantMode == 0) {
        expandedx_type = xDataType;
    }
    else {
        expandedx_type = ge::DT_INT8;
    }
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, expandedx_type, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseDropless(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    ge::DataType expandedx_type;
    if (quantMode == 0) {
        expandedx_type = xDataType;
    }
    else {
        expandedx_type = ge::DT_INT8;
    }
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, expandedx_type, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseNoQuant(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseNoQuantDroppad(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = EXPERT_NUM;
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{expert_num, C, H}, {expert_num, C, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{expert_num * C}, {expert_num * C}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseKeyValue(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    ge::DataType expandedx_type;
    if (quantMode == 0) {
        expandedx_type = xDataType;
    }
    else {
        expandedx_type = ge::DT_INT8;
    }
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, expandedx_type, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{expert_num, 2}, {expert_num, 2}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseFullload(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E, H}, {E, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{expert_num, 2}, {expert_num, 2}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseStaticQuant(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseStaticQuantDroppad(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = EXPERT_NUM;
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{expert_num, C, H}, {expert_num, C, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{expert_num * C}, {expert_num * C}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseDynamicQuant1H(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1, H}, {1, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseDynamicQuantEH(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E, H}, {E, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseDynamicQuantDroppad(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = EXPERT_NUM;
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E, H}, {E, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{expert_num, C, H}, {expert_num, C, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{expert_num * C}, {expert_num * C}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 通用模板
// 单核 + not quant + active + gather + scale not None float32  1000000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_01)
{   
    string expectTilingData = "40 1859 2880 1 0 256 256 -1 0 1 0 256 1 0 0 0 859 0 0 0 0 1 1859 1 1859 1859 1859 1 1859 1859 1984 0 1504 40 47 26 1 47 47 1 26 26 40 47 26 1 47 47 1 26 26 1 2880 2880 0 40 47 47 47 26 26 26 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 47 47 47 26 26 26 1 1 ";
    std::vector<size_t> expectWorkspaces = {16832884};
    RunNormalCaseNoQuant(1859, 2880, 1, 0, 859, 0, 1, false, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1000000, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + dropless + scatter + consum + scale None bfloat16  1001000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_02)
{
    string expectTilingData = "40 1811 2880 1 63 158 95 -1 1 0 0 256 0 1 0 1 1811 0 0 0 0 1 1811 1 1811 1811 1811 1 1811 1811 1984 0 1504 40 46 17 1 46 46 1 17 17 40 46 17 1 46 46 1 17 17 1 2880 2880 0 40 46 46 46 17 17 17 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 46 46 46 17 17 17 1 1 ";
    std::vector<size_t> expectWorkspaces = {16830896};
    RunNormalCaseDropless(1811, 2880, 1, 0, 0, 0, 0, true, QUANT_MODE_NONE, 0, ge::DT_BF16, {63, 158}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1001000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + dropless + gather + keyvalue + scale None int8  1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_03)
{
    string expectTilingData = "40 160 96 1450 0 256 256 -1 0 0 0 256 2 1 0 0 232000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23276832};
    RunNormalCaseKeyValue(160, 96, 1450, 0, 0, 0, 2, true, QUANT_MODE_NONE, 0, ge::DT_INT8, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1100000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + active + scatter + count + scale None float16  1101000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_04)
{
    string expectTilingData = "40 160 96 1450 180 192 12 -1 1 0 0 256 1 1 0 1 230000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23275856};
    RunNormalCase(160, 96, 1450, 0, 230000, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT16, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1101000, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + droppad + gather + scale not None float32  1000100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_05)
{   
    string expectTilingData = "40 89 2880 8 0 256 256 -1 0 1 0 256 1 0 0 0 712 1 0 0 13 1 712 1 712 712 712 1 712 712 1984 0 1504 40 18 10 1 18 18 1 10 10 40 18 10 1 18 18 1 10 10 1 2880 2880 0 40 18 18 18 10 10 10 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 18 18 18 10 10 10 1 1 ";
    std::vector<size_t> expectWorkspaces = {16801088};
    RunNormalCaseNoQuantDroppad(89, 2880, 8, 13, 712, 1, 1, false, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1000100, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + droppad + gather + count + scale not None int8  1100100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_06)
{
    string expectTilingData = "40 926 2880 8 0 256 256 -1 0 1 0 256 1 1 0 0 7408 1 0 0 78 4 1856 1 1856 1856 1840 1 1856 1840 1984 0 1504 40 186 154 1 186 186 1 154 154 40 186 154 1 186 186 1 154 154 1 2880 2880 0 40 186 186 186 154 154 154 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 186 186 186 154 154 154 1 1 ";
    std::vector<size_t> expectWorkspaces = {16988576};
    RunNormalCaseNoQuantDroppad(926, 2880, 8, 78, 7408, 1, 1, true, QUANT_MODE_NONE, 1, ge::DT_INT8, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1100100, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + active + gather float32  1010000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_07)
{
    string expectTilingData = "40 1859 2880 1 0 256 256 0 0 1 1 256 1 0 0 0 633 0 0 0 0 1 1859 1 1859 1859 1859 1 1859 1859 1984 0 1504 40 47 26 1 47 47 1 26 26 40 47 26 1 47 47 1 26 26 1 2880 2880 0 40 47 47 47 26 26 26 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 47 47 47 26 26 26 1 1 ";
    std::vector<size_t> expectWorkspaces = {16832884};
    RunNormalCaseStaticQuant(1859, 2880, 1, 0, 633, 0, 1, false, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1010000, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + active + scatter + count float16  1011000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_08)
{
    string expectTilingData = "40 1859 2880 1 180 192 12 0 1 1 1 256 1 1 1 1 633 0 0 0 0 1 1859 1 1859 1859 1859 1 1859 1859 1984 0 1504 40 47 26 1 47 47 1 26 26 40 47 26 1 47 47 1 26 26 1 2880 2880 0 40 47 47 47 26 26 26 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 47 47 47 26 26 26 1 1 ";
    std::vector<size_t> expectWorkspaces = {16831908};
    RunNormalCaseStaticQuant(1859, 2880, 1, 0, 633, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT16, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1011000, expectTilingData, expectWorkspaces);
}

// 多核 + static quant + dropless + gather + consum bfloat16  1110000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_09)
{
    string expectTilingData = "40 479 2880 5 0 256 256 0 0 1 1 256 0 1 0 0 2395 0 0 0 0 4 608 1 608 608 571 1 576 571 1984 0 1504 40 60 55 1 60 60 1 55 55 40 60 55 1 60 60 1 55 55 1 2880 2880 0 40 60 60 60 55 55 55 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 60 60 60 55 55 55 1 1 ";
    std::vector<size_t> expectWorkspaces = {16847892};
    RunNormalCaseStaticQuant(479, 2880, 5, 0, 2395, 0, 0, true, QUANT_MODE_STATIC, 1, ge::DT_BF16, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1110000, expectTilingData, expectWorkspaces);
}

// 多核 + static quant + active + scatter + count float32  1111000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_10)
{
    string expectTilingData = "40 479 2880 5 180 192 12 0 1 1 1 256 1 1 0 1 2395 0 0 0 0 4 608 1 608 608 571 1 576 571 1984 0 1504 40 60 55 1 60 60 1 55 55 40 60 55 1 60 60 1 55 55 1 2880 2880 0 40 60 60 60 55 55 55 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 60 60 60 55 55 55 1 1 ";
    std::vector<size_t> expectWorkspaces = {16846916};
    RunNormalCaseStaticQuant(479, 2880, 5, 0, 2395, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1111000, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + droppad + gather bfloat16  1010100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_11)
{
    string expectTilingData = "40 76 2880 8 0 256 256 0 0 1 1 256 1 0 0 0 608 1 0 0 9 1 608 1 608 608 608 1 608 608 1984 0 1504 38 16 16 1 16 16 1 16 16 38 16 16 1 16 16 1 16 16 1 2880 2880 0 38 16 16 16 16 16 16 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 38 16 16 16 16 16 16 1 1 ";
    std::vector<size_t> expectWorkspaces = {16798176};
    RunNormalCaseStaticQuantDroppad(76, 2880, 8, 9, 608, 1, 1, false, QUANT_MODE_STATIC, 1, ge::DT_BF16, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1010100, expectTilingData, expectWorkspaces);
}

// 多核 + static quant + droppad + gather + bfloat16 1110100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_12)
{
    string expectTilingData = "40 771 2880 8 0 256 256 0 0 1 1 256 1 1 0 0 6168 1 0 0 28 4 1536 1 1536 1536 1560 1 1568 1560 1984 0 1504 40 155 123 1 155 155 1 123 123 40 155 123 1 155 155 1 123 123 1 2880 2880 0 40 155 155 155 123 123 123 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 155 155 155 123 123 123 1 1 ";
    std::vector<size_t> expectWorkspaces = {16953856};
    RunNormalCaseStaticQuantDroppad(771, 2880, 8, 28, 6168, 1, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1110100, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + active + gather + count + scale not None 1H float32  1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_13)
{
    string expectTilingData = "40 1859 2880 1 0 256 256 1 0 1 0 256 1 1 0 0 633 0 1 0 0 1 1859 1 1859 1859 1859 1 1859 1859 1984 0 1504 40 47 26 1 47 47 1 26 26 40 47 26 1 47 47 1 26 26 1 2880 2880 0 40 47 47 47 26 26 26 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 47 47 47 26 26 26 1 1 ";
    std::vector<size_t> expectWorkspaces = {17293684};
    RunNormalCaseDynamicQuant1H(1859, 2880, 1, 0, 633, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + active + gather + scale None bfloat16  1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_14)
{
    string expectTilingData = "40 1859 2880 1 180 192 12 1 0 0 0 256 1 0 1 1 633 0 0 0 0 1 1859 1 1859 1859 1859 1 1859 1859 1984 0 1504 40 47 26 1 47 47 1 26 26 40 47 26 1 47 47 1 26 26 1 2880 2880 0 40 47 47 47 26 26 26 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 47 47 47 26 26 26 1 1 ";
    std::vector<size_t> expectWorkspaces = {17292708};
    RunNormalCase(1859, 2880, 1, 0, 633, 0, 1, false, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + active + scatter + cosum + scale not None EH  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_15)
{
    string expectTilingData = "40 1859 2880 1 0 100 100 1 1 1 0 256 0 1 0 1 633 0 2 0 0 1 1859 1 1859 1859 1859 1 1859 1859 1984 0 1504 40 47 26 1 47 47 1 26 26 40 47 26 1 47 47 1 26 26 1 2880 2880 0 40 47 47 47 26 26 26 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 47 47 47 26 26 26 1 1 ";
    std::vector<size_t> expectWorkspaces = {17293060};
    RunNormalCaseDynamicQuantEH(1859, 2880, 1, 0, 633, 0, 0, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + dropless + scatter + count + scale None float16  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_16)
{
    string expectTilingData = "40 1859 2880 1 0 100 100 1 1 0 0 256 1 1 0 1 1859 0 0 0 0 1 1859 1 1859 1859 1859 1 1859 1859 1984 0 1504 40 47 26 1 47 47 1 26 26 40 47 26 1 47 47 1 26 26 1 2880 2880 0 40 47 47 47 26 26 26 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 47 47 47 26 26 26 1 1 ";
    std::vector<size_t> expectWorkspaces = {17293060};
    RunNormalCaseDropless(1859, 2880, 1, 0, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + dropless + gather + scale not None EH  1120000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_17)
{
    string expectTilingData = "40 160 96 1450 0 256 256 1 0 1 0 256 1 1 0 0 202000 0 2 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23292192};
    RunNormalCaseDynamicQuantEH(160, 96, 1450, 0, 202000, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + droppless + gather + keyvalue + scale None  1120000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_18)
{
    string expectTilingData = "40 160 96 1450 180 192 12 1 0 0 0 256 2 1 0 1 232000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291216};
    RunNormalCaseKeyValue(160, 96, 1450, 0, 0, 0, 2, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + active + scatter + count + scale not None 1H  1121000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_19)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 1 0 256 1 1 0 1 202000 0 1 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCaseDynamicQuant1H(160, 96, 1450, 0, 202000, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + active + scatter + count + scale None  1121000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_20)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 1 202000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCase(160, 96, 1450, 0, 202000, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + active + scatter + scale None bfloat16  1121000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_21)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 1 202000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCase(160, 96, 1450, 0, 202000, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + dropless + scatter + scale None float16  1121000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_22)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCaseDropless(160, 96, 1450, 0, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + droppad + gather + scale not None EH  1020100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_23)
{
    string expectTilingData = "40 55 2880 8 0 256 256 1 0 1 0 256 1 0 0 0 440 1 2 0 8 1 440 1 440 440 440 1 440 440 1984 0 1504 40 11 11 1 11 11 1 11 11 40 11 11 1 11 11 1 11 11 1 2880 2880 0 0 0 0 0 0 0 0 0 0 0 0 0 40 11 11 11 11 11 11 1 1 2880 2880 1 40 11 11 11 11 11 11 1 1 ";
    std::vector<size_t> expectWorkspaces = {17254272};
    RunNormalCaseDynamicQuantDroppad(55, 2880, 8, 8, 440, 1, 1, false, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020100, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + droppad + gather + count + scale None float16  1120100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_24)
{
    string expectTilingData = "40 766 880 8 0 256 256 1 0 1 0 256 1 1 0 0 6128 1 2 0 191 4 1536 1 1536 1536 1520 1 1536 1520 1984 0 1504 40 154 122 1 154 154 1 122 122 40 154 122 1 154 154 1 122 122 1 880 880 0 0 0 0 0 0 0 0 0 0 0 0 0 40 154 154 154 122 122 122 1 1 880 880 1 40 154 154 154 122 122 122 1 1 ";
    std::vector<size_t> expectWorkspaces = {17093536};
    RunNormalCaseDynamicQuantDroppad(766, 880, 8, 191, 6128, 1, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT16, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120100, expectTilingData, expectWorkspaces);
}

// full load
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_25)
{
    string expectTilingData = "40 1 7168 8 0 256 256 1 1 1 0 256 2 1 0 0 8 0 2 0 0 1 8 1 8 8 8 1 8 8 2016 0 1504 8 1 1 1 1 1 1 1 1 8 1 1 1 1 1 1 1 1 4 1792 1792 0 8 1 1 1 1 1 1 1 1 7168 7168 1 0 0 0 0 0 0 0 0 0 0 0 0 8 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {17010432};
    RunNormalCaseFullload(1, 7168, 8, 0, 8, 0, 2, true, QUANT_MODE_DYNAMIC, 1, ge::DT_BF16, {0, 256}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2000000, expectTilingData, expectWorkspaces);
}

// performance 单核gather
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_26)
{
    string expectTilingData = "40 1920 7168 8 0 8 8 -1 1 0 0 256 1 1 0 1 15360 0 0 0 0 16 960 1 960 960 960 1 960 960 1984 4 1504 40 384 384 1 384 384 1 384 384 40 384 384 1 384 384 1 384 384 1 7168 7168 0 40 384 384 384 384 384 384 1 1 7168 7168 1 0 0 0 0 0 0 0 0 0 0 0 0 40 384 384 384 384 384 384 1 1 ";
    std::vector<size_t> expectWorkspaces = {17209920};
    RunNormalCase(1920, 7168, 8, 0, 15360, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1201000, expectTilingData, expectWorkspaces);
}

// performance 多核gather
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_27)
{
    string expectTilingData = "40 4608 7168 8 0 8 8 -1 1 0 0 256 1 1 0 1 36864 0 0 0 0 40 928 1 928 928 672 1 672 672 1984 10 1504 40 922 906 1 922 922 1 906 906 40 922 906 1 922 922 1 906 906 1 7168 7168 0 40 922 922 922 906 906 906 1 1 7168 7168 1 0 0 0 0 0 0 0 0 0 0 0 0 40 922 922 922 906 906 906 1 1 ";
    std::vector<size_t> expectWorkspaces = {17812032};
    RunNormalCase(4608, 7168, 8, 0, 36864, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1301000, expectTilingData, expectWorkspaces);
}

// 多核排序
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_28)
{
    string expectTilingData = "40 4608 7168 10 0 8 8 -1 1 0 0 256 1 1 0 1 46080 0 0 0 0 40 1152 1 1152 1152 1152 1 1152 1152 1984 10 1504 40 1152 1152 1 1152 1152 1 1152 1152 40 1152 1152 2 1016 136 2 1016 136 1 7168 7168 0 40 1152 1152 1152 1152 1152 1152 1 1 7168 7168 1 0 0 0 0 0 0 0 0 0 0 0 0 40 1152 1152 1152 1152 1152 1152 1 1 ";
    std::vector<size_t> expectWorkspaces = {18070080};
    RunNormalCase(4608, 7168, 10, 0, 46080, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1101000, expectTilingData, expectWorkspaces);
}

// v3性能收编新增UT
// 非量化 fullload + scale None + count  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_29)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCase(1, 83, 27, 0, 27, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale not None + consum  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_30)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 1 0 256 0 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseNoQuant(1, 83, 27, 0, 27, 0, 0, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale None + key_value  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_31)
{
    string expectTilingData = "40 1 83 27 10 22 12 -1 0 0 0 256 2 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseKeyValue(1, 83, 27, 0, 27, 0, 2, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {10, 22}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale None + GatherFirst  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_32)
{
    string expectTilingData = "40 100 2880 4 180 192 12 -1 0 0 0 256 1 1 1 1 400 0 0 0 0 1 400 1 400 400 400 1 400 400 1984 0 1504 40 10 10 1 10 10 1 10 10 40 10 10 1 10 10 1 10 10 1 2880 2880 0 40 10 10 10 10 10 10 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 10 10 10 10 10 10 1 1 ";
    std::vector<size_t> expectWorkspaces = {16791056};
    RunNormalCase(100, 2880, 4, 0, 400, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale None + scatter  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_33)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 1 0 0 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCase(1, 83, 27, 0, 27, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 静态量化 fullload + gather + count  2200000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_34)
{
    string expectTilingData = "40 1 83 27 180 192 12 0 0 1 1 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseStaticQuant(1, 83, 27, 0, 27, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2200000, expectTilingData, expectWorkspaces);
}

// 静态量化 fullload + scatter + consum  2200000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_35)
{
    string expectTilingData = "40 1 83 27 180 192 12 0 1 1 1 256 0 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseStaticQuant(1, 83, 27, 0, 27, 0, 0, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2200000, expectTilingData, expectWorkspaces);
}

// 动态量化 fullload + gather + consum + scale None  2300000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_36)
{
    string expectTilingData = "40 77 1024 4 0 256 256 1 0 0 0 256 0 1 0 0 308 0 0 0 0 1 308 1 308 308 308 1 308 308 1984 0 1504 39 8 4 1 8 8 1 4 4 39 8 4 1 8 8 1 4 4 1 1024 1024 0 39 8 8 8 4 4 4 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 39 8 8 8 4 4 4 1 1 ";
    std::vector<size_t> expectWorkspaces = {16953296};
    RunNormalCase(77, 1024, 4, 0, 308, 0, 0, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2300000, expectTilingData, expectWorkspaces);
}

// 动态量化 EPFULLLOAD fullload + scatter + count + scale None  2310000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_37)
{
    string expectTilingData = "40 120 1024 4 180 192 12 1 1 0 0 256 1 1 1 1 480 0 0 0 0 1 480 1 480 480 480 1 480 480 1984 0 1504 40 12 12 1 12 12 1 12 12 40 12 12 1 12 12 1 12 12 1 1024 1024 0 40 12 12 12 12 12 12 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 40 12 12 12 12 12 12 1 1 ";
    std::vector<size_t> expectWorkspaces = {16957136};
    RunNormalCase(120, 1024, 4, 0, 480, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2310000, expectTilingData, expectWorkspaces);
}

// 动态量化 SMOOTHTYPE 1H fullload + gather + consum  2301000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_38)
{
    string expectTilingData = "40 77 1024 4 0 256 256 1 0 1 0 256 0 1 0 0 308 0 1 0 0 1 308 1 308 308 308 1 308 308 1984 0 1504 39 8 4 1 8 8 1 4 4 39 8 4 1 8 8 1 4 4 1 1024 1024 0 39 8 8 8 4 4 4 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 39 8 8 8 4 4 4 1 1 ";
    std::vector<size_t> expectWorkspaces = {16953296};
    RunNormalCaseDynamicQuant1H(77, 1024, 4, 0, 308, 0, 0, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2301000, expectTilingData, expectWorkspaces);
}

// 动态量化 SMOOTHTYPE EH fullload + gather + consum  2302000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_39)
{
    string expectTilingData = "40 77 1024 4 0 256 256 1 0 1 0 256 0 1 0 0 308 0 2 0 0 1 308 1 308 308 308 1 308 308 1984 0 1504 39 8 4 1 8 8 1 4 4 39 8 4 1 8 8 1 4 4 1 1024 1024 0 39 8 8 8 4 4 4 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 39 8 8 8 4 4 4 1 1 ";
    std::vector<size_t> expectWorkspaces = {16953296};
    RunNormalCaseDynamicQuantEH(77, 1024, 4, 0, 308, 0, 0, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2302000, expectTilingData, expectWorkspaces);
}

// 动态量化 EPFULLLOAD  + SMOOTHTYPE 1H fullload + scatter + count  2311000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_40)
{
    string expectTilingData = "40 120 1024 4 180 192 12 1 1 1 0 256 1 1 1 1 480 0 1 0 0 1 480 1 480 480 480 1 480 480 1984 0 1504 40 12 12 1 12 12 1 12 12 40 12 12 1 12 12 1 12 12 1 1024 1024 0 40 12 12 12 12 12 12 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 40 12 12 12 12 12 12 1 1 ";
    std::vector<size_t> expectWorkspaces = {16957136};
    RunNormalCaseDynamicQuant1H(120, 1024, 4, 0, 480, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2311000, expectTilingData, expectWorkspaces);
}

// 动态量化 EPFULLLOAD  + SMOOTHTYPE EH fullload + scatter + count  2312000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_41)
{
    string expectTilingData = "40 120 1024 4 180 192 12 1 1 1 0 256 1 1 1 1 480 0 2 0 0 1 480 1 480 480 480 1 480 480 1984 0 1504 40 12 12 1 12 12 1 12 12 40 12 12 1 12 12 1 12 12 1 1024 1024 0 40 12 12 12 12 12 12 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 40 12 12 12 12 12 12 1 1 ";
    std::vector<size_t> expectWorkspaces = {16957136};
    RunNormalCaseDynamicQuantEH(120, 1024, 4, 0, 480, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2312000, expectTilingData, expectWorkspaces);
}

// 空 tensor + count 模式
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_empty_tensor_count)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.emplace_back(gert::StorageShape({0, 128}, {0, 128}), ge::DT_FLOAT, ge::FORMAT_ND);
    inputs.emplace_back(gert::StorageShape({0, 8}, {0, 8}), ge::DT_INT32, ge::FORMAT_ND);
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(gert::StorageShape({0, 128}, {0, 128}), ge::DT_FLOAT, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({0}, {0}), ge::DT_INT32, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({expert_num}, {expert_num}), ge::DT_INT64, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({0}, {0}), ge::DT_FLOAT, ge::FORMAT_ND);
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0LL)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0LL)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0LL)},
        {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_TOKENS_TYPE_COUNT)},
        {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
        {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{})},
        {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
    };
    RunExtendedTilingCase(MakeExtendedTilingContextPara(inputs, outputs, attrs, &compileInfo), ge::GRAPH_SUCCESS,
                          EMPTY_TENSOR_TILINGKEY, {EMPTY_TENSOR_WORKSPACE});
}

// 空 tensor + key_value 模式
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_empty_tensor_keyvalue)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.emplace_back(gert::StorageShape({0, 64}, {0, 64}), ge::DT_FLOAT16, ge::FORMAT_ND);
    inputs.emplace_back(gert::StorageShape({0, 4}, {0, 4}), ge::DT_INT32, ge::FORMAT_ND);
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(gert::StorageShape({0, 64}, {0, 64}), ge::DT_FLOAT16, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({0}, {0}), ge::DT_INT32, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({expert_num, 2}, {expert_num, 2}), ge::DT_INT64, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({0}, {0}), ge::DT_FLOAT, ge::FORMAT_ND);
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1LL)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0LL)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0LL)},
        {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_TOKENS_TYPE_KEY_VALUE)},
        {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
        {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{})},
        {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPLESS)},
    };
    RunExtendedTilingCase(MakeExtendedTilingContextPara(inputs, outputs, attrs, &compileInfo), ge::GRAPH_SUCCESS,
                          EMPTY_TENSOR_TILINGKEY, {EMPTY_TENSOR_WORKSPACE});
}

// 未指定 active_expert_range，默认 [0, expert_num)
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_default_expert_range)
{
    RunLargeColsCase(1, 128, 8, QUANT_MODE_NONE, 0, EXPERT_TOKENS_TYPE_COUNT, true, ge::DT_FLOAT, {}, ROW_IDX_TYPE_DROPPAD,
                     65536, {}, {});
}

// 大 cols 触发 SrcToDstDropPad 切分（非 dynamic droppad）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_large_cols_drop_pad_split)
{
    // ubSize must leave headroom after MAX_COLS_ONE_LOOP column buffer; 65536 makes basePerLoopMaxRows=0 (FPE).
    RunLargeColsCase(4, 25000, 8, QUANT_MODE_NONE, 0, EXPERT_TOKENS_TYPE_COUNT, true, ge::DT_FLOAT, {0, 256},
                     ROW_IDX_TYPE_DROPPAD, 262144, {}, {});
}

// 大 cols + dynamic droppad 触发 Dynamic 切分
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_large_cols_dynamic_droppad_split)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t N = 8;
    int64_t H = 25000;
    int64_t K = 4;
    int64_t C = 8;
    int64_t expert_num = EXPERT_NUM;
    int64_t totalLength = N * K;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.emplace_back(gert::StorageShape({N, H}, {N, H}), ge::DT_FLOAT, ge::FORMAT_ND);
    inputs.emplace_back(gert::StorageShape({N, K}, {N, K}), ge::DT_INT32, ge::FORMAT_ND);
    inputs.emplace_back(gert::StorageShape({expert_num, H}, {expert_num, H}), ge::DT_FLOAT, ge::FORMAT_ND);
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(gert::StorageShape({expert_num, C, H}, {expert_num, C, H}), ge::DT_INT8, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({totalLength}, {totalLength}), ge::DT_INT32, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({expert_num}, {expert_num}), ge::DT_INT64, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({expert_num * C}, {expert_num * C}), ge::DT_FLOAT, ge::FORMAT_ND);
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(totalLength)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1LL)},
        {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_TOKENS_TYPE_COUNT)},
        {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_DYNAMIC)},
        {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 256})},
        {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
    };
    RunExtendedTilingCase(MakeExtendedTilingContextPara(inputs, outputs, attrs, &compileInfo));
}

// 小 UB + 大 cols 触发 static quant gather 切分
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_small_ub_static_quant_split)
{
    std::vector<gert::TilingContextPara::TensorDescription> extraInputs;
    extraInputs.emplace_back(gert::StorageShape({1}, {1}), ge::DT_FLOAT, ge::FORMAT_ND);
    extraInputs.emplace_back(gert::StorageShape({1}, {1}), ge::DT_FLOAT, ge::FORMAT_ND);
    RunLargeColsCase(8, 18000, 4, QUANT_MODE_STATIC, 0, EXPERT_TOKENS_TYPE_COUNT, true, ge::DT_FLOAT, {0, 256},
                     ROW_IDX_TYPE_DROPPAD, 8192, extraInputs, {});
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_parse_success)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {0, 0};
    fe::PlatFormInfos platformInfo;
    InitParsePlatformInfo(platformInfo);
    const char *compileJson =
        R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion":"Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeInitRoutingV3"))
                      .OpName(ge::AscendString("MoeInitRoutingV3"))
                      .IOInstanceNum({1, 1, 1, 1}, {1, 1, 1, 1})
                      .InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .Build();
    auto *parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeInitRoutingV3");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_GT(compileInfo.aivNum, 0);
    EXPECT_GT(compileInfo.ubSize, 0);
}