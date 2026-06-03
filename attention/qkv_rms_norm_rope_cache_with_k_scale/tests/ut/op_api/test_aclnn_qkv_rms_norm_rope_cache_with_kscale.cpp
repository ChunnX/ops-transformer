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
 * \file test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
 * \brief QkvRmsNormRopeCacheWithKScale aclnn op_api UT.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
constexpr int64_t kT = 128;
constexpr int64_t kNq = 16;
constexpr int64_t kNk = 2;
constexpr int64_t kNv = 2;
constexpr int64_t kD = 128;
constexpr int64_t kBlockNum = 32;
constexpr int64_t kBlockSize = 16;
constexpr int64_t kBatch = 1;
constexpr int64_t kMaxSeqLen = 128;
}  // namespace

class qkv_rms_norm_rope_cache_with_kscale_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "qkv_rms_norm_rope_cache_with_kscale_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "qkv_rms_norm_rope_cache_with_kscale_opapi_ut TearDown" << endl;
    }
};

TEST_F(qkv_rms_norm_rope_cache_with_kscale_opapi_ut, normal_case)
{
    auto tensorQkv = TensorDesc({kT, kNq + kNk + kNv, kD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorQGamma = TensorDesc({kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto tensorKGamma = TensorDesc({kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto tensorCosSin = TensorDesc({kMaxSeqLen, kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorSlotMapping = TensorDesc({kT}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), kBlockNum * kBlockSize - 1);
    auto tensorKCache = TensorDesc({kBlockNum, kNk, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorVCache = TensorDesc({kBlockNum, kNv, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorKScaleCache = TensorDesc({kBlockNum, kNk, kBlockSize, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 2.0);
    auto tensorCuSeqLen = TensorDesc({kBatch + 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), kT);
    auto tensorRotation = TensorDesc({kD, kD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorVScale = TensorDesc({kNv}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);

    auto headNums = IntArrayDesc(vector<int64_t>{kNq, kNk, kNv});
    const float epsilon = 1e-6f;

    auto tensorQOut = TensorDesc({kT, kNq, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorQScale = TensorDesc({kT, kNq}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKCacheOut = TensorDesc({kBlockNum, kNk, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorVCacheOut = TensorDesc({kBlockNum, kNv, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorKScaleCacheOut = TensorDesc({kBlockNum, kNk, kBlockSize, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(
            tensorQkv,
            tensorQGamma,
            tensorKGamma,
            tensorCosSin,
            tensorSlotMapping,
            tensorKCache,
            tensorVCache,
            tensorKScaleCache,
            tensorCuSeqLen,
            tensorRotation,
            tensorVScale,
            headNums,
            epsilon
        ),
        OUTPUT(
            tensorQOut,
            tensorQScale,
            tensorKCacheOut,
            tensorVCacheOut,
            tensorKScaleCacheOut
        )
    );

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(qkv_rms_norm_rope_cache_with_kscale_opapi_ut, null_qkv)
{
    auto tensorQGamma = TensorDesc({kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto tensorKGamma = TensorDesc({kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto tensorCosSin = TensorDesc({kMaxSeqLen, kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorSlotMapping = TensorDesc({kT}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), kBlockNum * kBlockSize - 1);
    auto tensorKCache = TensorDesc({kBlockNum, kNk, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorVCache = TensorDesc({kBlockNum, kNv, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorKScaleCache = TensorDesc({kBlockNum, kNk, kBlockSize, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 2.0);
    auto tensorCuSeqLen = TensorDesc({kBatch + 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), kT);
    auto tensorRotation = TensorDesc({kD, kD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorVScale = TensorDesc({kNv}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);

    auto headNums = IntArrayDesc(vector<int64_t>{kNq, kNk, kNv});
    const float epsilon = 1e-6f;

    auto tensorQOut = TensorDesc({kT, kNq, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorQScale = TensorDesc({kT, kNq}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKCacheOut = TensorDesc({kBlockNum, kNk, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorVCacheOut = TensorDesc({kBlockNum, kNv, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorKScaleCacheOut = TensorDesc({kBlockNum, kNk, kBlockSize, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(
            nullptr,
            tensorQGamma,
            tensorKGamma,
            tensorCosSin,
            tensorSlotMapping,
            tensorKCache,
            tensorVCache,
            tensorKScaleCache,
            tensorCuSeqLen,
            tensorRotation,
            tensorVScale,
            headNums,
            epsilon
        ),
        OUTPUT(
            tensorQOut,
            tensorQScale,
            tensorKCacheOut,
            tensorVCacheOut,
            tensorKScaleCacheOut
        )
    );

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(qkv_rms_norm_rope_cache_with_kscale_opapi_ut, null_rotation)
{
    auto tensorQkv = TensorDesc({kT, kNq + kNk + kNv, kD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorQGamma = TensorDesc({kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto tensorKGamma = TensorDesc({kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto tensorCosSin = TensorDesc({kMaxSeqLen, kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorSlotMapping = TensorDesc({kT}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), kBlockNum * kBlockSize - 1);
    auto tensorKCache = TensorDesc({kBlockNum, kNk, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorVCache = TensorDesc({kBlockNum, kNv, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorKScaleCache = TensorDesc({kBlockNum, kNk, kBlockSize, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 2.0);
    auto tensorCuSeqLen = TensorDesc({kBatch + 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), kT);
    auto tensorVScale = TensorDesc({kNv}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);

    auto headNums = IntArrayDesc(vector<int64_t>{kNq, kNk, kNv});
    const float epsilon = 1e-6f;

    auto tensorQOut = TensorDesc({kT, kNq, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorQScale = TensorDesc({kT, kNq}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKCacheOut = TensorDesc({kBlockNum, kNk, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorVCacheOut = TensorDesc({kBlockNum, kNv, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorKScaleCacheOut = TensorDesc({kBlockNum, kNk, kBlockSize, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(
            tensorQkv,
            tensorQGamma,
            tensorKGamma,
            tensorCosSin,
            tensorSlotMapping,
            tensorKCache,
            tensorVCache,
            tensorKScaleCache,
            tensorCuSeqLen,
            nullptr,
            tensorVScale,
            headNums,
            epsilon
        ),
        OUTPUT(
            tensorQOut,
            tensorQScale,
            tensorKCacheOut,
            tensorVCacheOut,
            tensorKScaleCacheOut
        )
    );

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(qkv_rms_norm_rope_cache_with_kscale_opapi_ut, wrong_dtype)
{
    auto tensorQkv = TensorDesc({kT, kNq + kNk + kNv, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorQGamma = TensorDesc({kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto tensorKGamma = TensorDesc({kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto tensorCosSin = TensorDesc({kMaxSeqLen, kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorSlotMapping = TensorDesc({kT}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), kBlockNum * kBlockSize - 1);
    auto tensorKCache = TensorDesc({kBlockNum, kNk, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorVCache = TensorDesc({kBlockNum, kNv, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorKScaleCache = TensorDesc({kBlockNum, kNk, kBlockSize, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 2.0);
    auto tensorCuSeqLen = TensorDesc({kBatch + 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), kT);
    auto tensorRotation = TensorDesc({kD, kD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorVScale = TensorDesc({kNv}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);

    auto headNums = IntArrayDesc(vector<int64_t>{kNq, kNk, kNv});
    const float epsilon = 1e-6f;

    auto tensorQOut = TensorDesc({kT, kNq, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorQScale = TensorDesc({kT, kNq}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKCacheOut = TensorDesc({kBlockNum, kNk, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorVCacheOut = TensorDesc({kBlockNum, kNv, kBlockSize, kD}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensorKScaleCacheOut = TensorDesc({kBlockNum, kNk, kBlockSize, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(
            tensorQkv,
            tensorQGamma,
            tensorKGamma,
            tensorCosSin,
            tensorSlotMapping,
            tensorKCache,
            tensorVCache,
            tensorKScaleCache,
            tensorCuSeqLen,
            tensorRotation,
            tensorVScale,
            headNums,
            epsilon
        ),
        OUTPUT(
            tensorQOut,
            tensorQScale,
            tensorKCacheOut,
            tensorVCacheOut,
            tensorKScaleCacheOut
        )
    );

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
