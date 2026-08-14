/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPR_RW_H
#define SPR_RW_H

#include <cstdint>

// SPR 读写测试 demo 入口
//   sprVaddr      : 目标核 SPR vaddr (halResMap 返回，用于 host 侧记录/回显)
//   blockDim      : kernel block 数 (默认 1)
//   stream        : ACL 流
//   sprVaddrTable : GM 地址，存放所有核的 SPR vaddr 表 (host 通过 halResMap 填充)
//   resultAddr    : GM 地址，存放结果 (5 个 uint64_t)
//   targetCoreId  : 要访问的目标核 id (res_id)
void spr_rw_demo(uint64_t sprVaddr, uint32_t blockDim, void *stream,
                 uint8_t *sprVaddrTable, uint8_t *resultAddr, uint32_t targetCoreId);

#endif // SPR_RW_H
