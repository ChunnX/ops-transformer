/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ts_fas_tc_empty_tensor.cpp
 * \brief FlashAttentionScore 正向用例.
 */

#include "ts_fas.h"
#include "tiling/fa/tiling_data.h"

void SetBasicEmptyInfo(FasCase &emptyCase) {
    emptyCase.mParam.b = 2;
    emptyCase.mParam.n2 = 16;
    emptyCase.mParam.g = 1;
    emptyCase.mParam.s1 = 2048;
    emptyCase.mParam.s2 = 0;
    emptyCase.mParam.d = 32;
    emptyCase.mParam.layoutType = LayoutType::BSH;
    emptyCase.mParam.scale = 1.0f;
}

TEST_F(Ts_Fas_Ascend910B3, Tc_fas_EmptyTensor90_001)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase emptyCase;
    SetBasicEmptyInfo(emptyCase);
    emptyCase.mParam.dtype = ge::DataType::DT_FLOAT16;

    // 用例 期望信息
    emptyCase.mForward.mExp.mTilingKeys[(uint64_t)Platform::SocVersion::Ascend910B3] = 90UL;

    /**
     * 运行用例
     */
    ASSERT_TRUE(emptyCase.Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_TRUE(
        emptyCase.mForwardCtx.SetTilingDataMaxSize(ops::adv::tests::utils::Context::kDefaultTilingDataMaxSize * 2));
    ASSERT_TRUE(emptyCase.Run());
}

TEST_F(Ts_Fas_Ascend910B3, Tc_fas_EmptyTensor92_002)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase emptyCase;
    SetBasicEmptyInfo(emptyCase);
    emptyCase.mParam.dtype = ge::DataType::DT_FLOAT;

    // 用例 期望信息
    emptyCase.mForward.mExp.mTilingKeys[(uint64_t)Platform::SocVersion::Ascend910B3] = 92UL;

    /**
     * 运行用例
     */
    ASSERT_TRUE(emptyCase.Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_TRUE(
        emptyCase.mForwardCtx.SetTilingDataMaxSize(ops::adv::tests::utils::Context::kDefaultTilingDataMaxSize * 2));
    ASSERT_TRUE(emptyCase.Run());
}

TEST_F(Ts_Fas_Ascend910B3, Tc_fas_EmptyTensor94_003)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase emptyCase;
    SetBasicEmptyInfo(emptyCase);
    emptyCase.mParam.dtype = ge::DataType::DT_BF16;

    // 用例 期望信息
    emptyCase.mForward.mExp.mTilingKeys[(uint64_t)Platform::SocVersion::Ascend910B3] = 94UL;

    /**
     * 运行用例
     */
    ASSERT_TRUE(emptyCase.Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_TRUE(
        emptyCase.mForwardCtx.SetTilingDataMaxSize(ops::adv::tests::utils::Context::kDefaultTilingDataMaxSize * 2));
    ASSERT_TRUE(emptyCase.Run());
}
