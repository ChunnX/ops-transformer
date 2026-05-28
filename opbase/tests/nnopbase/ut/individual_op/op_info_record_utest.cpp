/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <fstream>
#include "mockcpp/mockcpp.hpp"
#include "acl/acl_op.h"
#include "register/op_impl_kernel_registry.h"
#include "graph/types.h"
#include "graph/operator_factory.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "op_info_serialize.h"
#include "ini_parse.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "lib_path.h"
#include "nlohmann/json.hpp"
class OpInfoRecordUtest: public testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(OpInfoRecordUtest, Utest_OpInfoEmptyBinInfo)
{
    std::string path = "../../../../tests/nnopbase/common/depends/";
    MOCKER(mmAccess2).stubs().will(returnValue(EN_OK));
    MOCKER_CPP(&aclnnOpInfoRecord::LibPath::GetInstallParentPath).stubs().will(returnValue(aclnnOpInfoRecord::Path(path)));
    aclnnOpInfoRecord::OpCompilerOption opt("", 0);
    aclnnOpInfoRecord::OpKernelInfo kernelInfo("", 0);
    EXPECT_EQ(kernelInfo.isMc2, false);
    gert::TilingContext ctx;
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoSerialize(&ctx, opt, &kernelInfo), -1);
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoDump(), 0);
}

TEST_F(OpInfoRecordUtest, Utest_OpInfoSerialize)
{
    std::string path = "../../../../tests/nnopbase/common/depends/";
    MOCKER(mmAccess2).stubs().will(returnValue(EN_OK));
    MOCKER_CPP(&aclnnOpInfoRecord::LibPath::GetInstallParentPath).stubs().will(returnValue(aclnnOpInfoRecord::Path(path)));
    aclnnOpInfoRecord::OpCompilerOption opt("", 0);
    aclnnOpInfoRecord::OpKernelInfo kernelInfo("../../../../tests/nnopbase/mock/built-in/op_impl/ai_core/tbe/kernel/ascend910/add_n/AddN_8b4ec10fbb39c5c0a65192c606400b29_high_performance.json", 0);
    gert::TilingContext ctx;
    const ge::char_t *nodeName = "Utest_OpInfoSerialize";
    MOCKER_CPP(&gert::TilingContext::GetComputeNodeInfo).stubs().will(returnValue((const gert::ComputeNodeInfo *)nodeName));
    MOCKER_CPP(&gert::ComputeNodeInfo::GetNodeName).stubs().will(returnValue(nodeName));
    const ge::char_t *nodeType = "Add";
    MOCKER_CPP(&gert::ComputeNodeInfo::GetNodeType).stubs().will(returnValue(nodeType));
    MOCKER_CPP(&gert::ComputeNodeInfo::GetAttrs).stubs().will(returnValue((const gert::RuntimeAttrs *)nodeType));

    // mock for attr_to_json
    MOCKER_CPP(&gert::RuntimeAttrs::GetAttrNum).stubs().will(returnValue((size_t)1));
    MOCKER_CPP(&gert::RuntimeAttrs::GetStr).stubs().will(returnValue((const char *)nodeName));
    bool stubGetBool = false;
    MOCKER_CPP(&gert::RuntimeAttrs::GetBool).stubs().will(returnValue((const bool *)&stubGetBool));
    MOCKER_CPP(&gert::RuntimeAttrs::GetInt).stubs().will(returnValue((const int64_t *)nodeName));
    MOCKER_CPP(&gert::RuntimeAttrs::GetFloat).stubs().will(returnValue((const float *)nodeName));

    MOCKER_CPP(&gert::RuntimeAttrs::GetListInt).stubs().will(returnValue((const gert::TypedContinuousVector<int64_t> *)nodeName));
    MOCKER_CPP(&gert::TypedContinuousVector<int64_t>::GetData).stubs().will(returnValue((const int64_t *)nodeName));
    MOCKER_CPP(&gert::TypedContinuousVector<int64_t>::GetSize).stubs().will(returnValue((size_t)1));
    MOCKER_CPP(&gert::RuntimeAttrs::GetListFloat).stubs().will(returnValue((const gert::TypedContinuousVector<float> *)nodeName));
    MOCKER_CPP(&gert::TypedContinuousVector<float>::GetData).stubs().will(returnValue((const float *)nodeName));
    MOCKER_CPP(&gert::TypedContinuousVector<float>::GetSize).stubs().will(returnValue((size_t)1));

    MOCKER_CPP(&gert::ComputeNodeInfo::GetInputTdInfo).stubs().will(returnValue((const gert::CompileTimeTensorDesc *)nodeType));
    MOCKER_CPP(&gert::TilingContext::GetInputShape).stubs().will(returnValue((const gert::StorageShape *)nodeType));
    // mock for stride
    static gert::Stride fakeStride({16, 4});
    MOCKER_CPP(&gert::TilingContext::InputIsView).stubs().will(returnValue(true));
    MOCKER_CPP(&gert::TilingContext::OutputIsView).stubs().will(returnValue(true));
    MOCKER_CPP(&gert::TilingContext::GetInputStride).stubs()
        .will(returnValue(static_cast<const gert::Stride*>(&fakeStride)));
    MOCKER_CPP(&gert::TilingContext::GetOutputStride).stubs()
        .will(returnValue(static_cast<const gert::Stride*>(&fakeStride)));
    MOCKER_CPP(&gert::Stride::GetDimNum).stubs().will(returnValue((size_t)2));
    MOCKER_CPP(&gert::Stride::GetStride).stubs()
        .will(returnValue((int64_t)16))
        .then(returnValue((int64_t)4));
    MOCKER_CPP(&gert::TilingContext::GetInputOffset).stubs().will(returnValue((int64_t)128));
    MOCKER_CPP(&gert::TilingContext::GetOutputOffset).stubs().will(returnValue((int64_t)256));

    // mock for shape
    MOCKER_CPP(&gert::Shape::GetDimNum).stubs().will(returnValue((size_t)1));
    MOCKER_CPP(&gert::StorageFormat::GetStorageFormat).stubs().will(returnValue((ge::Format)2));
    MOCKER_CPP(&gert::Shape::GetDim).stubs().will(returnValue((int64_t)2));
    MOCKER_CPP(&gert::StorageFormat::GetOriginFormat).stubs().will(returnValue((ge::Format)2));

    MOCKER_CPP(&gert::OpImplKernelRegistry::OpImplFunctions::IsInputDataDependency).stubs().will(returnValue(false)).then(returnValue(true));
    MOCKER_CPP(&gert::OpImplKernelRegistry::OpImplFunctions::IsTilingInputDataDependency).stubs()
        .will(returnValue(false)).then(returnValue(true));
    MOCKER_CPP(&gert::Tensor::GetDataType).stubs()
        .will(returnValue(ge::DT_INT8))
        .then(returnValue(ge::DT_UINT8))
        .then(returnValue(ge::DT_INT16))
        .then(returnValue(ge::DT_UINT16))
        .then(returnValue(ge::DT_INT32))
        .then(returnValue(ge::DT_UINT32))
        .then(returnValue(ge::DT_INT64))
        .then(returnValue(ge::DT_UINT64))
        .then(returnValue(ge::DT_FLOAT))
        .then(returnValue(ge::DT_DOUBLE));
    MOCKER_CPP(&gert::TensorData::GetSize).stubs().will(returnValue((size_t)10));
    MOCKER_CPP(&gert::Tensor::GetAddr, const void *(gert::Tensor::*)() const).stubs().will(returnValue((const void *)nodeName));
    MOCKER_CPP(&gert::TilingContext::GetInputTensor).stubs().will(returnValue((const gert::Tensor *)nodeName));
    MOCKER_CPP(&gert::ComputeNodeInfo::GetInputInstanceInfo).stubs().will(returnValue((const gert::AnchorInstanceInfo *)nodeName));
    MOCKER_CPP(&gert::CompileTimeTensorDesc::GetDataType).stubs().will(returnValue(ge::DT_INT8));
    MOCKER_CPP(&gert::AnchorInstanceInfo::GetInstanceNum).stubs().will(returnValue((size_t)2));
    MOCKER_CPP(&ge::TypeUtils::DataTypeToSerialString).stubs().will(returnValue(std::string("DT_FLOAT16")));

    EXPECT_EQ(aclnnOpInfoRecord::OpInfoSerialize(&ctx, opt, &kernelInfo), 0);
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoDump(), -1);
}

TEST_F(OpInfoRecordUtest, Utest_OpInfoSerialize_without_supportInfo)
{
    std::string path = "../../../../tests/nnopbase/common/depends/";
    MOCKER(mmAccess2).stubs().will(returnValue(EN_OK));
    MOCKER_CPP(&aclnnOpInfoRecord::LibPath::GetInstallParentPath).stubs().will(returnValue(aclnnOpInfoRecord::Path(path)));
    aclnnOpInfoRecord::OpCompilerOption opt("", 0);
    aclnnOpInfoRecord::OpKernelInfo kernelInfo("../../../../tests/nnopbase/mock/built-in/op_impl/ai_core/tbe/kernel/config/ascend910/add_n.json", 0);
    gert::TilingContext ctx;
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoSerialize(&ctx, opt, &kernelInfo), -1);
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoDump(), 0);
}

TEST_F(OpInfoRecordUtest, Utest_OpInfoPathEqual)
{
    std::string dependPath = "../../../../tests/nnopbase/common/depends/";
    aclnnOpInfoRecord::Path pathA = aclnnOpInfoRecord::Path(dependPath);
    aclnnOpInfoRecord::Path pathB = aclnnOpInfoRecord::Path(dependPath);
    std::string configPath = "../../../../tests/nnopbase/mock/built-in/op_impl/ai_core/tbe/kernel/config/";
    aclnnOpInfoRecord::Path pathC = aclnnOpInfoRecord::Path(configPath);
    EXPECT_EQ(pathA == pathB, true);
    EXPECT_EQ(pathA == pathC, false);
}