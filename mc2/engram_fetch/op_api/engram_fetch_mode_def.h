/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_fetch_mode_def.h
 * \brief fetch_mode definitions for engram_fetch, shared by op_api and op_host/tiling.
 * \note The PTA side (mc2/common/torch_extension/{elastic_buffer.py, csrc/elastic_buffer.cpp}) and its Python
 *       constants (FETCH_MODE_SLICED/LOCAL) must keep the same values; update both together.
 */

#ifndef ENGRAM_FETCH_MODE_DEF_H
#define ENGRAM_FETCH_MODE_DEF_H

#include <cstdint>

constexpr int64_t ENGRAM_FETCH_MODE_SLICED = 0; // sliced sharding + cross-NPU comm (default, train/infer)
constexpr int64_t ENGRAM_FETCH_MODE_LOCAL = 1;  // full table replicated per NPU, local gather only (infer only)

#endif // ENGRAM_FETCH_MODE_DEF_H
