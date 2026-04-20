/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "l0_attention_pioneer_metadata.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>

#include "ut_stub.h"
#include "opdev/make_op_executor.h"

using namespace op;
using namespace std;

const int64_t METADATA_SIZE = 1024;
const int64_t SEQ_LEN_SIZE = 4;

class AttentionPioneerMetadata : public ::testing::Test {
 public:
  AttentionPioneerMetadata() : exe(nullptr) {
  }

  aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype) {
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           data);
  }

  void Clear() {
    exe->kernelLaunchObjList_.clear();
  }

  void SetUp() override {
    auto executor = &exe;
    auto unique_executor = CREATE_EXECUTOR();
    unique_executor.ReleaseTo(executor);
  }

  void TearDown() override {
    delete exe;
  }

 public:
  aclOpExecutor* exe;
  int64_t data[METADATA_SIZE] = {0};
};

TEST_F(AttentionPioneerMetadata, AttentionPioneerMetadata_PSIDE_SUCC) {
  auto actualSeqLengths = CreateAclTensor({SEQ_LEN_SIZE}, ACL_INT64);
  auto actualSeqLengthsKv = CreateAclTensor({SEQ_LEN_SIZE}, ACL_INT64);
  auto metaData = CreateAclTensor({METADATA_SIZE}, ACL_INT32);
  char inputLayout[] = "TND";
  auto res = l0op::AttentionPioneerMetadata(
      actualSeqLengths, actualSeqLengthsKv,
      /* numHeadsQ */ 32, /* numHeadsKv */ 8,
      /* headDimQk */ 128, /* headDimV */ 128,
      /* batchSizeOptional */ 4, /* sparseModeOptional */ 2,
      /* preTokensOptional */ 2147483647, /* nextTokensOptional */ 0,
      inputLayout, /* kSinkNumOptional */ 0,
      /* ropeHeadDimOptional */ 0, /* blockSizeOptional */ 0,
      /* socVersion */ "ascend950",
      /* aicCoreNum */ 36, /* aivCoreNum */ 72,
      metaData, exe);
  EXPECT_NE(res, nullptr);
}

TEST_F(AttentionPioneerMetadata, AttentionPioneerMetadata_DSIDE_SUCC) {
  auto actualSeqLengths = CreateAclTensor({SEQ_LEN_SIZE}, ACL_INT64);
  auto actualSeqLengthsKv = CreateAclTensor({SEQ_LEN_SIZE}, ACL_INT64);
  auto metaData = CreateAclTensor({METADATA_SIZE}, ACL_INT32);
  char inputLayout[] = "TND";
  auto res = l0op::AttentionPioneerMetadata(
      actualSeqLengths, actualSeqLengthsKv,
      /* numHeadsQ */ 128, /* numHeadsKv */ 1,
      /* headDimQk */ 512, /* headDimV */ 512,
      /* batchSizeOptional */ 4, /* sparseModeOptional */ 2,
      /* preTokensOptional */ 2147483647, /* nextTokensOptional */ 0,
      inputLayout, /* kSinkNumOptional */ 0,
      /* ropeHeadDimOptional */ 64, /* blockSizeOptional */ 0,
      /* socVersion */ "ascend950",
      /* aicCoreNum */ 36, /* aivCoreNum */ 72,
      metaData, exe);
  EXPECT_NE(res, nullptr);
}

TEST_F(AttentionPioneerMetadata, AttentionPioneerMetadata_BAND_SUCC) {
  auto actualSeqLengths = CreateAclTensor({SEQ_LEN_SIZE}, ACL_INT64);
  auto actualSeqLengthsKv = CreateAclTensor({SEQ_LEN_SIZE}, ACL_INT64);
  auto metaData = CreateAclTensor({METADATA_SIZE}, ACL_INT32);
  char inputLayout[] = "TND";
  auto res = l0op::AttentionPioneerMetadata(
      actualSeqLengths, actualSeqLengthsKv,
      /* numHeadsQ */ 32, /* numHeadsKv */ 8,
      /* headDimQk */ 128, /* headDimV */ 128,
      /* batchSizeOptional */ 4, /* sparseModeOptional */ 4,
      /* preTokensOptional */ 512, /* nextTokensOptional */ 512,
      inputLayout, /* kSinkNumOptional */ 0,
      /* ropeHeadDimOptional */ 0, /* blockSizeOptional */ 0,
      /* socVersion */ "ascend950",
      /* aicCoreNum */ 36, /* aivCoreNum */ 72,
      metaData, exe);
  EXPECT_NE(res, nullptr);
}
