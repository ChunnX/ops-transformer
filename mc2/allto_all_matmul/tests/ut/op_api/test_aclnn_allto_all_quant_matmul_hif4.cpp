/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.openhisilicon.com
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_aclnn_allto_all_quant_matmul_hif4.cpp
 * \brief aclnnAlltoAllQuantMatmul HiF4 全量化场景 op_api UT(CSV 参数化)
 *
 * 用例来源: mc2/allto_all_matmul/docs/hif4/HiF4_UT_Case_Design.md §3
 * 用例数据: test_aclnn_allto_all_quant_matmul_hif4.csv(soc=Ascend960DT, 18 条)
 *
 * HiF4 仅在 A6(ascend960dt/DAV_9201) 支持。运行门控(满足前整体 GTEST_SKIP):
 *   1. A6 devkit: ge::DT_HIFLOAT4_SCALE(43) 落位, 即 ge::DT_MAX > 43;
 *   2. UT 框架 SOC_VERSION 表补齐 "Ascend960DT"。
 * 框架 ACL_DTYPE 表补齐 HIFLOAT4/HIFLOAT4_SCALE 前, 参数层做 dtype 桥接
 * (ACL_HIFLOAT4=42 / ACL_HIFLOAT4_SCALE=43, 与 ge 枚举同值, devkit 落位后核对)。
 */

#include <gtest/gtest.h>
#include <string>

#include "graph/types.h"
#include "test_allto_all_matmul_api_ut_param.h"
#include "../../../op_api/aclnn_allto_all_quant_matmul.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

namespace AlltoAllMatmulHif4ApiUT {

// 门控 1: A6 devkit 落位 C_DT_HIFLOAT4_SCALE=43 后 ge::DT_MAX>=44(A5 包为 43)
static bool Hif4DtypeReady()
{
    return static_cast<int32_t>(ge::DT_MAX) > 43;
}

// HiF4 aclDataType 桥接(A5 包 ACL 枚举止于 ACL_FLOAT4_E1M2=41)
static aclDataType ParseHif4AclDtype(const std::string &dtypeStr)
{
    if (dtypeStr == "HIFLOAT4") {
        return static_cast<aclDataType>(42); // ACL_HIFLOAT4(计划值, 与 ge::DT_HIFLOAT4 同值)
    }
    if (dtypeStr == "HIFLOAT4_SCALE") {
        return static_cast<aclDataType>(43); // ACL_HIFLOAT4_SCALE(计划值, 与 ge 同值)
    }
    return ACL_DT_UNDEFINED;
}

struct AlltoAllMatmulHif4ApiUtParam : public AlltoAllMatmulUT::AlltoAllMatmulApiUtParam {
    std::string socRaw;

    explicit AlltoAllMatmulHif4ApiUtParam(const csv_map &csvMap)
        : AlltoAllMatmulUT::AlltoAllMatmulApiUtParam(csvMap), socRaw(ReadMap(csvMap, "soc"))
    {
        Fixup(ReadMap(csvMap, "x1_dtype"), this->x1);
        Fixup(ReadMap(csvMap, "x2_dtype"), this->x2);
        Fixup(ReadMap(csvMap, "bias_dtype"), this->bias);
        Fixup(ReadMap(csvMap, "x1_scale_dtype"), this->x1Scale);
        Fixup(ReadMap(csvMap, "x2_scale_dtype"), this->x2Scale);
        Fixup(ReadMap(csvMap, "output_dtype"), this->output);
        Fixup(ReadMap(csvMap, "allto_all_out_dtype"), this->alltoAllOut);
    }

    // 框架解析为 ACL_DT_UNDEFINED 的 HiF4 dtype 用桥接值重建(保持原 view dims, ND 格式)
    static void Fixup(const std::string &dtypeStr, TensorDesc &desc)
    {
        aclDataType dtype = ParseHif4AclDtype(dtypeStr);
        if (dtype != ACL_DT_UNDEFINED) {
            desc = TensorDesc(desc.GetViewDims(), dtype, ACL_FORMAT_ND);
        }
    }

    // 门控 2: 框架 SOC_VERSION 未收录 Ascend960DT 时解析回退为 ASCEND910B
    bool SocResolved() const
    {
        return !(socRaw == "Ascend960DT" && soc == op::SocVersion::ASCEND910B);
    }
};

class AclnnAlltoAllQuantMatmulHif4Test : public testing::TestWithParam<AlltoAllMatmulHif4ApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulHif4ApiUT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulHif4ApiUT TearDown" << std::endl;
    }
};

TEST_P(AclnnAlltoAllQuantMatmulHif4Test, param)
{
    auto param = GetParam();
    if (!Hif4DtypeReady()) {
        GTEST_SKIP() << "HiF4 requires A6 devkit (ge::DT_HIFLOAT4_SCALE), skip on current package";
    }
    if (!param.SocResolved()) {
        GTEST_SKIP() << "framework SOC_VERSION lacks 'Ascend960DT', add it before running HiF4 cases";
    }
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    op::SetPlatformSocVersion(param.soc);
    aclTensor *bias_opt = param.bias.GetViewDims().empty() ? nullptr : param.bias.ToAclTypeRawPtr();
    aclTensor *x1_scale_opt = param.x1Scale.GetViewDims().empty() ? nullptr : param.x1Scale.ToAclTypeRawPtr();
    aclTensor *allto_all_out_opt =
        param.alltoAllOut.GetViewDims().empty() ? nullptr : param.alltoAllOut.ToAclTypeRawPtr();
    aclTensor *x1 = param.x1.GetViewDims().empty() ? nullptr : param.x1.ToAclTypeRawPtr();
    aclTensor *x2 = param.x2.GetViewDims().empty() ? nullptr : param.x2.ToAclTypeRawPtr();
    aclTensor *x2_scale = param.x2Scale.GetViewDims().empty() ? nullptr : param.x2Scale.ToAclTypeRawPtr();
    aclTensor *output = param.output.GetViewDims().empty() ? nullptr : param.output.ToAclTypeRawPtr();
    aclTensor *comm_scale = nullptr;
    aclTensor *x1_offset = nullptr;
    aclTensor *x2_offset = nullptr;
    int64_t comm_quant_mode = 0;
    int64_t comm_quant_dtype = -1;
    auto ut =
        OP_API_UT(aclnnAlltoAllQuantMatmul,
                  INPUT(x1, x2, bias_opt, x1_scale_opt, x2_scale, comm_scale, x1_offset, x2_offset, param.group.c_str(),
                        param.alltoAllAxes, param.x1QuantMode, param.x2QuantMode, comm_quant_mode, comm_quant_dtype,
                        param.x1QuantDtype, param.groupSize, param.transposeX1, param.transposeX2),
                  OUTPUT(output, allto_all_out_opt));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AlltoAllMatmulHif4, AclnnAlltoAllQuantMatmulHif4Test,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulHif4ApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllMatmulHif4ApiUtParam>);

} // namespace AlltoAllMatmulHif4ApiUT
