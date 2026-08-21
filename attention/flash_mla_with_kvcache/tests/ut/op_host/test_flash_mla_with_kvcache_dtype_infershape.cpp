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
#include "flash_mla_with_kvcache_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace FlashMlaWithKvcacheUT {

class FlashMlaWithKvcacheInferDTypeTest : public testing::TestWithParam<FlashMlaWithKvcacheInferDTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashMlaWithKvcache InferDTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashMlaWithKvcache InferDTypeTest TearDown" << std::endl;
    }
};

TEST_P(FlashMlaWithKvcacheInferDTypeTest, param)
{
    auto param = GetParam();

    std::vector<void *> inputDataTypes;
    inputDataTypes.emplace_back(&param.q_dtype);
    inputDataTypes.emplace_back(&param.k_cache_dtype);

    ge::DataType attn_out_dtype_init = ge::DT_UNDEFINED;
    ge::DataType softmax_lse_dtype_init = ge::DT_UNDEFINED;
    std::vector<void *> outputDataTypes;
    outputDataTypes.emplace_back(&attn_out_dtype_init);
    outputDataTypes.emplace_back(&softmax_lse_dtype_init);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("FlashMlaWithKvcache")
            .IrInstanceNum({1, 1}, {1, 1})
            .InputDataTypes(inputDataTypes)
            .OutputDataTypes(outputDataTypes)
            .NodeAttrs({
                {"head_dim_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.head_dim_v)},
                {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
                {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
                {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_q)},
                {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_kv)},
                {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
                {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
                {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
                {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.return_softmax_lse)},
            })
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("FlashMlaWithKvcache")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0),
                  param.expect_attn_out_dtype);
        if (param.expect_softmax_lse_dtype != ge::DT_UNDEFINED) {
            EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1),
                      param.expect_softmax_lse_dtype);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    FlashMlaWithKvcache, FlashMlaWithKvcacheInferDTypeTest,
    testing::ValuesIn(GetCasesFromCsv<FlashMlaWithKvcacheInferDTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<FlashMlaWithKvcacheInferDTypeUtParam>);

} // namespace FlashMlaWithKvcacheUT