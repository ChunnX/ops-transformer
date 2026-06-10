/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_SCORE_LOG_H_
#define FLASH_ATTENTION_SCORE_LOG_H_

#include "op_common/log/log.h"

#include <string>
#include <vector>

#ifdef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#undef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#endif

#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entityName, paramName, reason)                         \
    do {                                                                                                \
        std::string _safe_entityName_(entityName);                                                      \
        std::string _safe_paramName_(paramName);                                                        \
        std::string _safe_reason_(reason);                                                              \
        OP_LOGE_LIBOPAPI_REPORT(                                                                        \
            _safe_entityName_.c_str(), "Parameter %s of %s is invalid. Reason: %s.",                    \
            _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_reason_.c_str());                \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "reason"};                    \
        const std::vector<const char*> msgvalue = {                                                     \
            _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_reason_.c_str()};                \
        REPORT_PREDEFINED_ERR_MSG("EZ0037", msgKey, msgvalue);                                          \
    } while (0)

#endif // FLASH_ATTENTION_SCORE_LOG_H_
