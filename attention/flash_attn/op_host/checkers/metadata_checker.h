/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file metadata_checker.h
 * \brief Checker for metadata parameter (文档: INT32, shape=(max_schedule_size,))
 */

#ifndef METADATA_CHECKER_H
#define METADATA_CHECKER_H

#include "./base_checker.h"

namespace optiling {
class MetadataChecker : public FABaseChecker {
public:
    MetadataChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : FABaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~MetadataChecker() = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
};
} // namespace optiling
#endif // METADATA_CHECKER_H