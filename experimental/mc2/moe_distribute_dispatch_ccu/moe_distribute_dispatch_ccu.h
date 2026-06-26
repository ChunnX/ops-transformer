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
 * \file moe_distribute_dispatch_ccu.h
 * \brief teardown(AICore) <<<>>> 启动包装声明 —— 实现在 moe_distribute_dispatch_ccu.cpp（ascendc 编译）。
 *        host 侧按普通函数调用此包装；<<<>>> 语法不能出现在 host 文件中。
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_CCU_H
#define MOE_DISTRIBUTE_DISPATCH_CCU_H

#include <cstdint>

/**
 * @brief 启动 dispatch teardown AICore 核函数（<<<>>> 包装）
 * @param blockDim     核函数 block 数
 * @param stream       AICore 主算路 stream（stream1）
 * @param localBuffer  本卡 HCCL Buffer（CCU setup 把数据 / 收讫 flag 写到这里）
 * @param expandXOut   [out] 本卡整理后的 token 输出
 * @param tilingGM     device 侧 DispatchCcuTilingData 指针
 */
void DispatchTeardownLaunch(uint32_t blockDim, void* stream,
                            uint8_t* localBuffer, uint8_t* expandXOut, uint8_t* tilingGM);

#endif // MOE_DISTRIBUTE_DISPATCH_CCU_H
