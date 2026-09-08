/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_vselr_buf.h
 * \brief 静态Tensor范式的 vselrIndexesBuf wrapper：替代 TBuf<> 传入 common VF 接口
 */

#ifndef QUANT_FLASH_ATTN_VSELR_BUF_H_
#define QUANT_FLASH_ATTN_VSELR_BUF_H_

// 静态Tensor范式的 vselrIndexesBuf wrapper：替代 TBuf<> 传入 common VF 接口
// （VF 侧通过尾部默认模板参数 VselrBufT 兼容 TBuf<> 与本 wrapper）
struct VselrIndexesBuffer {
    LocalTensor<uint8_t> tensor;
    template <typename T>
    __aicore__ inline LocalTensor<T> Get() const
    {
        return tensor.template ReinterpretCast<T>();
    }
};

#endif // QUANT_FLASH_ATTN_VSELR_BUF_H_
