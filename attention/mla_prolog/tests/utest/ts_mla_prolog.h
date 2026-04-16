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
 * \file ts_mla_prolog.h
 * \brief MlaProlog UTest 相关基类定义.
 */

#ifndef TS_MLA_PROLOG_H
#define TS_MLA_PROLOG_H

#include "tests/utest/ts.h"
#include "mla_prolog_case.h"

using MlaPrologParam = ops::adv::tests::MlaProlog::MlaPrologParam;
using CacheModeType = ops::adv::tests::MlaProlog::MlaPrologParam::CacheModeType;
using MlaPrologCase = ops::adv::tests::MlaProlog::MlaPrologCase;

class Ts_MlaProlog : public Ts<MlaPrologCase> {};
class Ts_MlaProlog_Ascend910B1 : public Ts_Ascend910B1<MlaPrologCase> {};
class Ts_MlaProlog_Ascend910B2 : public Ts_Ascend910B2<MlaPrologCase> {};
class Ts_MlaProlog_Ascend910B3 : public Ts_Ascend910B3<MlaPrologCase> {};

class Ts_MlaProlog_WithParam : public Ts_WithParam<MlaPrologCase> {};
class Ts_MlaProlog_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<MlaPrologCase> {};
class Ts_MlaProlog_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<MlaPrologCase> {};
class Ts_MlaProlog_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<MlaPrologCase> {};

#endif