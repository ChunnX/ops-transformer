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
 * \file aclnn_fas_case.cpp
 * \brief
 */

#include "aclnn_fas_case.h"
#include <utility>

using namespace ops::adv::tests::fas;

AclnnFasCase::AclnnFasCase() : AclnnFasCase("Undefined", true, "", OpInfoWithSocversion(), AclnnFaParam())
{
}

AclnnFasCase::AclnnFasCase(const char *name, bool enable, const char *dbgInfo, OpInfoWithSocversion forward, AclnnFaParam param)
    : AclnnFaCase(name, enable, dbgInfo, std::move(forward), OpInfoWithSocversion(), std::move(param),
                  kTilingTemplatePriority_Invalid)
{
}

bool AclnnFasCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mForward.ProcessTiling(mName, this->socVersion)) {
        return false;
    }
    if (!mForward.ProcessKernel(mName)) {
        return false;
    }
    return true;
}
