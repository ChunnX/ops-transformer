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
 * \brief Aggregator header — includes all sub-headers for backward compatibility.
 *
 * New code should include specific sub-headers directly:
 *   - aclnn_common_types.h  — type definitions, enums, constants (lightweight)
 *   - aclnn_opapi_loader.h  — dlopen/dlsym, GetOpApiFuncAddr
 *   - aclnn_type_convert.h  — ConvertType functions
 *   - aclnn_hash_utils.h    — AddParamToBuf, CalcHashId
 *   - aclnn_release_utils.h — Release, ConvertTypes, get_valid_tensor
 *   - aclnn_device_utils.h  — DecodeDevice, is_at_tensor
 *   - aclnn_cmd_macros.h    — ACLNN_CMD macro (includes all above)
 *
 * For operators using ACLNN_CMD, just include aclnn_cmd_macros.h.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_COMMON_H
#define CANN_OPS_TRANSFORMER_ACLNN_COMMON_H

#include "aclnn_common_types.h"
#include "aclnn_opapi_loader.h"
#include "aclnn_type_convert.h"
#include "aclnn_hash_utils.h"
#include "aclnn_release_utils.h"
#include "aclnn_device_utils.h"
#include "aclnn_cmd_macros.h"

#endif  // CANN_OPS_TRANSFORMER_ACLNN_COMMON_H
