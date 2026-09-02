/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_ARCH_SELECT_H_
#define QUANT_BLOCK_SPARSE_ATTN_ARCH_SELECT_H_

// DAV_9201 and DAV_9202 use the independently compiled arch-9102r1 sources.
#if defined(DAV_9201) || defined(DAV_9202) || defined(__DAV_9201__) || defined(__DAV_9202__) || \
    (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
#define QBSA_ARCH_9102R1 1
#else
#define QBSA_ARCH_9102R1 0
#endif

#endif // QUANT_BLOCK_SPARSE_ATTN_ARCH_SELECT_H_
