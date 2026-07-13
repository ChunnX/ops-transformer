/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file const_common.h
 * \brief 公共常量和宏定义
 */

#pragma once

// 计数器/标志位大小（字节）
#define BARRIER_FLAG_SIZE 8

// 内存对齐宏
#define ALIGN_UP(val, al) (((val) + ((al) - 1)) & ~((al) - 1))
