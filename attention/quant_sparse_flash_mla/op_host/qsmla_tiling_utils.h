/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file qsmla_tiling_utils.h
 * \brief Free utility functions for QSMLA: DataType/Layout serialization and shape string.
 */

#ifndef QSMLA_TILING_UTILS_H
#define QSMLA_TILING_UTILS_H

#include "qsmla_tiling_common.h"

namespace optiling {

std::string QSMLADataTypeToSerialString(ge::DataType type);
std::string QSMLALayoutToSerialString(QSMLALayout layout);
std::string QSMLAGetShapeStr(gert::Shape shape);

} // namespace optiling
#endif // QSMLA_TILING_UTILS_H
