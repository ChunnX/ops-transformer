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
 * \file aclnn_common.h
 * \brief Aggregation header — includes all sub-headers for backward compatibility.
 *
 * This file has been split into focused sub-headers:
 *   - aclnn_types.h       : Core type definitions, enums, constants, basic utilities
 *   - aclnn_api_loader.h  : Dynamic library loading (dlopen/dlsym) and API function resolution
 *   - aclnn_type_convert.h: ConvertType series, Release, Copy utilities, DecodeDevice
 *   - aclnn_cmd.h         : ACLNN_CMD macro, hash buffer utilities, memory management
 *
 * Existing code that includes "aclnn_common.h" continues to work unchanged.
 * New code may include specific sub-headers when only a subset of functionality is needed.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_COMMON_H
#define CANN_OPS_TRANSFORMER_ACLNN_COMMON_H

#include "aclnn_types.h"
#include "aclnn_api_loader.h"
#include "aclnn_type_convert.h"
#include "aclnn_cmd.h"

#endif  // CANN_OPS_TRANSFORMER_ACLNN_COMMON_H