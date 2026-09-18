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
 * \brief engram_fetch 取数模式定义(op_api 与 op_host/tiling 共享)
 * 注意: PTA 侧 mc2/common/torch_extension/{elastic_buffer.py, csrc/elastic_buffer.cpp}
 * 与 Python 侧常量(FETCH_MODE_SLICED/LOCAL)与本文件保持相同的值契约, 修改时需同步
 */

#ifndef ENGRAM_FETCH_MODE_DEF_H
#define ENGRAM_FETCH_MODE_DEF_H

#include <cstdint>

constexpr int64_t ENGRAM_FETCH_MODE_SLICED = 0; // 切片分存 + 跨NPU通信(默认, 推理/训练)
constexpr int64_t ENGRAM_FETCH_MODE_LOCAL = 1;  // 全量表冗余到每个NPU, 取数纯本地(仅推理)

#endif // ENGRAM_FETCH_MODE_DEF_H
