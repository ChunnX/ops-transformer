/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fag_copy_in_buffer_base.h
 * \brief
 */

#ifndef FAG_COPY_IN_BUFFER_BASE_KERNEL_H_
#define FAG_COPY_IN_BUFFER_BASE_KERNEL_H_

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FagCubeInBuffer {
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);

    using SrcT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline FagCubeInBuffer() {}
    __aicore__ inline ~FagCubeInBuffer() {}

    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheSize, int32_t reduceAxisCnt) {}

    __aicore__ inline void Destroy()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            MATMUL_MODULE(Context)->needAllocA = true;
            MATMUL_MODULE(Context)->needFreeA = true;
            MATMUL_MODULE(Context)->isFirstBaseBlockA = true;
            MATMUL_MODULE(Context)->isLastBaseBlockA = false;
            tscmGlobal[MATMUL_MODULE(Context)->tscmIndexA].cacheSize = 0; // IterateAll 结束，cacheSize清零
            tscmGlobal[MATMUL_MODULE(Context)->tscmIndexA].localQue.FreeAllEvent();
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            MATMUL_MODULE(Context)->needAllocB = true;
            MATMUL_MODULE(Context)->needFreeB = true;
            MATMUL_MODULE(Context)->isFirstBaseBlockB = true;
            MATMUL_MODULE(Context)->isLastBaseBlockB = false;
            tscmGlobal[MATMUL_MODULE(Context)->tscmIndexB].cacheSize = 0;
            tscmGlobal[MATMUL_MODULE(Context)->tscmIndexB].localQue.FreeAllEvent();
        }
    }

    __aicore__ inline int32_t GetIterIndex(int32_t curRow, int32_t curCol)
    {
        // 获取当前base块在A/B矩阵中的位置
        int32_t iterIndex = 0;
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            iterIndex = curRow * MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() + curCol;
        }
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            iterIndex = curCol * MATMUL_MODULE(CopyCubeInParams)->GetTotalRow() + curRow;
        }
        return iterIndex;
    }

    __aicore__ inline int32_t GetTransIterIndex(int32_t curRow, int32_t curCol)
    {
        int32_t iterIndex = 0;
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            iterIndex = curCol * MATMUL_MODULE(CopyCubeInParams)->GetTotalRow() + curRow;
        }
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            iterIndex = curRow * MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() + curCol;
        }
        return iterIndex;
    }

    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1)
    {
        return false;
    }

    __aicore__ inline LocalTensor<SrcT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        return LocalTensor<SrcT>{};
    }

    __aicore__ inline LocalTensor<SrcT> AllocTensor(int32_t bufferPos = -1)
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            LocalTensor<SrcT> aL1 =
                tscmGlobal[MATMUL_MODULE(Context)->tscmIndexA].localQue.template AllocTensor<SrcT>();
            return aL1[bufferPos * MATMUL_MODULE(Context)->baseBlockSizeA];
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            LocalTensor<SrcT> bL1 =
                tscmGlobal[MATMUL_MODULE(Context)->tscmIndexB].localQue.template AllocTensor<SrcT>();
            return bL1[bufferPos * MATMUL_MODULE(Context)->baseBlockSizeB];
        }
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<SrcT>& tensor = LocalTensor<SrcT>{})
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            if (MATMUL_MODULE(Context)->needFreeA) {
                tscmGlobal[MATMUL_MODULE(Context)->tscmIndexA].localQue.FreeTensor(
                    MATMUL_MODULE(Context)->CurrentTensorA);
            }
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            if (MATMUL_MODULE(Context)->needFreeB) {
                tscmGlobal[MATMUL_MODULE(Context)->tscmIndexB].localQue.FreeTensor(
                    MATMUL_MODULE(Context)->CurrentTensorB);
            }
        }
    }

    __aicore__ inline void EnQue(LocalTensor<SrcT>& tensor) {}

    __aicore__ inline void DeQue() {}

    __aicore__ inline void Reset() {}
};
}
}
}
#endif // FAG_COPY_IN_BUFFER_BASE_KERNEL_H_
