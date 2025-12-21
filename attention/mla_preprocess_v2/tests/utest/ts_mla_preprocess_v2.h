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
 * \file ts_mla_preprocess.h
 * \brief MlaPreprocessV2 UTest 相关基类定义.
 */

#include "tests/utest/ts.h"
#include "mla_preprocess_v2_case.h"

using MlaPreprocessV2Case = ops::adv::tests::mla_preprocess::MlaPreprocessV2Case;

class Ts_MlaPreprocessV2 : public Ts<MlaPreprocessV2Case> {};
class Ts_MlaPreprocessV2_Ascend910B2 : public Ts_Ascend910B2<MlaPreprocessV2Case> {};

class Ts_MlaPreprocessV2_WithParam : public Ts_WithParam<MlaPreprocessV2Case> {};
class Ts_MlaPreprocessV2_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<MlaPreprocessV2Case> {};
