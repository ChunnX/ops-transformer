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
 * \file ts_mla_prolog_v2.h
 * \brief MlaPrologV2 UTest 相关基类定义.
 */

#ifndef TS_MLA_PROLOG_V2_H
#define TS_MLA_PROLOG_V2_H

#include "tests/utest/ts.h"
#include "mla_prolog_v2_case.h"

using MlaPrologV2Param = ops::adv::tests::MlaPrologV2::MlaPrologV2Param;
using CacheModeType = ops::adv::tests::MlaPrologV2::MlaPrologV2Param::CacheModeType;
using MlaPrologV2Case = ops::adv::tests::MlaPrologV2::MlaPrologV2Case;

class Ts_MlaPrologV2 : public Ts<MlaPrologV2Case> {};
class Ts_MlaPrologV2_Ascend910B1 : public Ts_Ascend910B1<MlaPrologV2Case> {};
class Ts_MlaPrologV2_Ascend910B2 : public Ts_Ascend910B2<MlaPrologV2Case> {};
class Ts_MlaPrologV2_Ascend910B3 : public Ts_Ascend910B3<MlaPrologV2Case> {};

class Ts_MlaPrologV2_WithParam : public Ts_WithParam<MlaPrologV2Case> {};
class Ts_MlaPrologV2_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<MlaPrologV2Case> {};
class Ts_MlaPrologV2_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<MlaPrologV2Case> {};
class Ts_MlaPrologV2_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<MlaPrologV2Case> {};

#endif