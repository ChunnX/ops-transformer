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
 * \file a16w4_msd_tool.h
 * \brief A16W4 MSD tool functions for grouped matmul
 */
#ifndef GROUPED_MATMUL_A16W4_MSD_TOOL_H
#define GROUPED_MATMUL_A16W4_MSD_TOOL_H

#include "gmm/op_kernel/common/gmm_tool_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::GetUserWorkspace;

namespace GROUPED_MATMUL::A16W4Msd {
using GmmOpKernelCommon::CeilAlign;
using GmmOpKernelCommon::CeilDivide;
using GmmOpKernelCommon::DataCopyPad2D;
using GmmOpKernelCommon::GetKBUnit;

// A16W4Msd特有工具函数可在此定义

} // namespace GROUPED_MATMUL::A16W4Msd
#endif
