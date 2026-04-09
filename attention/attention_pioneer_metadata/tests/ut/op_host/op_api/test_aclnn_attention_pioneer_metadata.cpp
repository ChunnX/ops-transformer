/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cstdio>
#include <vector>
#include <fstream>
#include "gtest/gtest.h"

#include "aclnn_attention_pioneer_metadata.h"
#include "common/op_api_def.h"

#include "op_api_ut_common/inner/rts_interface.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "ut_stub.h"

using namespace std;
using namespace op;

class l2_attention_pioneer_metadata_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2_attention_pioneer_metadata_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "l2_attention_pioneer_metadata_test TearDown" << endl;
  }
};

TEST_F(l2_attention_pioneer_metadata_test, normal_pside_success) {
  auto actualSeqLengthsDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsRef = DescToAclContainer(actualSeqLengthsDesc);
  auto actualSeqLengthsKvDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsKvRef = DescToAclContainer(actualSeqLengthsKvDesc);
  auto metaDataDesc = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND);
  auto metaDataRef = DescToAclContainer(metaDataDesc);

  char inputLayout[] = "TND";
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus aclRet = aclnnAttentionPioneerMetadataGetWorkspaceSize(
      actualSeqLengthsRef, actualSeqLengthsKvRef,
      /* numHeadsQ */ 32, /* numHeadsKv */ 8,
      /* headDimQk */ 128, /* headDimV */ 128,
      /* batchSizeOptional */ 4, /* sparseModeOptional */ 2,
      /* preTokensOptional */ 2147483647, /* nextTokensOptional */ 0,
      inputLayout, /* kSinkNumOptional */ 0,
      /* ropeHeadDimOptional */ 0, /* blockSizeOptional */ 0,
      /* aicCoreNumOptional */ 36, /* aivCoreNumOptional */ 72,
      metaDataRef, &workspace_size, &executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  unique_ptr<void, decltype(&FreeDeviceMemory)> workspace_ptr(MallocDeviceMemory(workspace_size), FreeDeviceMemory);
  aclRet = aclnnAttentionPioneerMetadata(workspace_ptr.get(), workspace_size, executor, nullptr);
  SynchronizeStream();
  if (executor != nullptr) {
    delete executor;
  }
}

TEST_F(l2_attention_pioneer_metadata_test, invalid_sparse_mode) {
  auto actualSeqLengthsDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsRef = DescToAclContainer(actualSeqLengthsDesc);
  auto actualSeqLengthsKvDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsKvRef = DescToAclContainer(actualSeqLengthsKvDesc);
  auto metaDataDesc = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND);
  auto metaDataRef = DescToAclContainer(metaDataDesc);

  char inputLayout[] = "TND";
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  // sparse_mode=0 is not supported (only 2 and 4)
  aclnnStatus aclRet = aclnnAttentionPioneerMetadataGetWorkspaceSize(
      actualSeqLengthsRef, actualSeqLengthsKvRef,
      32, 8, 128, 128, 4, /* sparseModeOptional */ 0,
      2147483647, 0, inputLayout, 0, 0, 0, 36, 72,
      metaDataRef, &workspace_size, &executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  if (executor != nullptr) {
    delete executor;
  }
}

TEST_F(l2_attention_pioneer_metadata_test, invalid_head_num_not_divisible) {
  auto actualSeqLengthsDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsRef = DescToAclContainer(actualSeqLengthsDesc);
  auto actualSeqLengthsKvDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsKvRef = DescToAclContainer(actualSeqLengthsKvDesc);
  auto metaDataDesc = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND);
  auto metaDataRef = DescToAclContainer(metaDataDesc);

  char inputLayout[] = "TND";
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  // numHeadsQ=32 % numHeadsKv=5 != 0
  aclnnStatus aclRet = aclnnAttentionPioneerMetadataGetWorkspaceSize(
      actualSeqLengthsRef, actualSeqLengthsKvRef,
      32, /* numHeadsKv */ 5, 128, 128, 4, 2,
      2147483647, 0, inputLayout, 0, 0, 0, 36, 72,
      metaDataRef, &workspace_size, &executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  if (executor != nullptr) {
    delete executor;
  }
}

TEST_F(l2_attention_pioneer_metadata_test, invalid_head_dim_zero) {
  auto actualSeqLengthsDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsRef = DescToAclContainer(actualSeqLengthsDesc);
  auto actualSeqLengthsKvDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsKvRef = DescToAclContainer(actualSeqLengthsKvDesc);
  auto metaDataDesc = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND);
  auto metaDataRef = DescToAclContainer(metaDataDesc);

  char inputLayout[] = "TND";
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  // headDimQk=0 is invalid
  aclnnStatus aclRet = aclnnAttentionPioneerMetadataGetWorkspaceSize(
      actualSeqLengthsRef, actualSeqLengthsKvRef,
      32, 8, /* headDimQk */ 0, 128, 4, 2,
      2147483647, 0, inputLayout, 0, 0, 0, 36, 72,
      metaDataRef, &workspace_size, &executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  if (executor != nullptr) {
    delete executor;
  }
}

TEST_F(l2_attention_pioneer_metadata_test, invalid_k_sink_num) {
  auto actualSeqLengthsDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsRef = DescToAclContainer(actualSeqLengthsDesc);
  auto actualSeqLengthsKvDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsKvRef = DescToAclContainer(actualSeqLengthsKvDesc);
  auto metaDataDesc = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND);
  auto metaDataRef = DescToAclContainer(metaDataDesc);

  char inputLayout[] = "TND";
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  // kSinkNum=64 is invalid (only 0 or 128)
  aclnnStatus aclRet = aclnnAttentionPioneerMetadataGetWorkspaceSize(
      actualSeqLengthsRef, actualSeqLengthsKvRef,
      32, 8, 128, 128, 4, 2,
      2147483647, 0, inputLayout, /* kSinkNumOptional */ 64, 0, 0, 36, 72,
      metaDataRef, &workspace_size, &executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  if (executor != nullptr) {
    delete executor;
  }
}

TEST_F(l2_attention_pioneer_metadata_test, invalid_input_layout) {
  auto actualSeqLengthsDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsRef = DescToAclContainer(actualSeqLengthsDesc);
  auto actualSeqLengthsKvDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto actualSeqLengthsKvRef = DescToAclContainer(actualSeqLengthsKvDesc);
  auto metaDataDesc = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND);
  auto metaDataRef = DescToAclContainer(metaDataDesc);

  char inputLayout[] = "BSH";
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  // inputLayout="BSH" is not supported
  aclnnStatus aclRet = aclnnAttentionPioneerMetadataGetWorkspaceSize(
      actualSeqLengthsRef, actualSeqLengthsKvRef,
      32, 8, 128, 128, 4, 2,
      2147483647, 0, inputLayout, 0, 0, 0, 36, 72,
      metaDataRef, &workspace_size, &executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  if (executor != nullptr) {
    delete executor;
  }
}
