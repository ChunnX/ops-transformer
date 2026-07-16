/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include "stem_oam_prep_varlen_q_param.h"
#include "infer_shape_case_executor.h"

namespace StemOamPrepVarlenQUT {

class StemPrepQInferShapeTest : public testing::TestWithParam<StemPrepQInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StemOamPrepVarlenQ InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StemOamPrepVarlenQ InferShapeTest TearDown" << std::endl;
    }
};

TEST_P(StemPrepQInferShapeTest, param)
{
    auto param = GetParam();

    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1)
        inputTensorDesc.emplace_back(param.q);
    if (param.inputInstance[1] == 1)
        inputTensorDesc.emplace_back(param.qSeqLens);
    if (param.inputInstance[2] == 1)
        inputTensorDesc.emplace_back(param.cuSeqLensQ);
    if (param.inputInstance.size() > 3 && param.inputInstance[3] == 1)
        inputTensorDesc.emplace_back(param.qScale);

    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc;
    if (param.outputInstance[0] == 1)
        outputTensorDesc.emplace_back(param.qFlat);

    gert::InfershapeContextPara infershapeContextPara(
        "StemOamPrepVarlenQ", inputTensorDesc, outputTensorDesc,
        {
            {"stemBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.stemBlockSize)},
            {"stemStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.stem_stride)},
        },
        param.inputInstance, param.outputInstance);

    ExecuteTestCase(infershapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    StemOamPrepVarlenQ, StemPrepQInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<StemPrepQInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StemPrepQInferShapeUtParam>);

} // namespace StemOamPrepVarlenQUT
