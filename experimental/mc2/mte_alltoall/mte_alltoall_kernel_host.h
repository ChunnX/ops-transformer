/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

// ============================================================================
// MTE AllToAll 算子 - Kernel 入口头文件
// ============================================================================

#ifndef MTE_ALLTOALL_KERNEL_HOST_H
#define MTE_ALLTOALL_KERNEL_HOST_H

void MteAllToAllDemo(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* output,
                         uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM);

#endif
