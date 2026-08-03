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
 * \file broadcast_nddma_bench.cpp
 * \brief BroadcastNddmaBench 默认架构 kernel 入口。
 *
 * 本算子仅注册 ascend950（arch35），OpDef 中 ExtendCfgInfo("opFile.value",
 * "broadcast_nddma_bench_apt") 将 ascend950 的编译路由到 *_apt.cpp。
 * 该默认文件仅为满足构建框架对默认 kernel 源文件的存在性要求而提供，直接复用
 * arch35 实现（NDDMA 依赖 __NPU_ARCH__==3510，仅在 Ascend950PR_9599 下实体化）。
 */

#include "broadcast_nddma_bench_apt.cpp"
