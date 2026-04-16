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
 * \file fag_copy_cube_in_base.h
 * \brief
 */

#ifndef FAG_COPY_CUBE_IN_BASE_KERNEL_H_
#define FAG_COPY_CUBE_IN_BASE_KERNEL_H_

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG> class FagCopyCubeIn {
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyUtils, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    using InputType = INPUT_TYPE;
    __aicore__ inline FagCopyCubeIn() = default;
    __aicore__ inline ~FagCopyCubeIn() = default;

    __aicore__ inline void Init()
    {
    }

    __aicore__ inline void SetInput(const LocalTensor<SrcT> &localMatrix, bool isTranspose)
    {
    }

    __aicore__ inline void SetInput(const GlobalTensor<SrcT> &globalMatrix, bool isTranspose)
    {
        // do Set Global Input here
        MATMUL_MODULE(MatmulTensorInfo)->template SetGlobalTensor<false>(globalMatrix, isTranspose);
    }

    __aicore__ inline LocalTensor<TransT> LoadData(const int32_t &curRow, const int32_t &curCol,
                                                   const int32_t &tileHeight, const int32_t &tileWidth)
    {
        this->InitLoadDataParams(curRow, curCol, tileHeight, tileWidth);

        // mm-dyv mm-qk A矩阵处理
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            tscmIndex = BUF_0;
            return this->LoadFullDataA(curRow, curCol, tileHeight, tileWidth, false);
        }

        // mm-dyv mm-qk B矩阵处理
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            tscmIndex = BUF_1;
            // A的Free是在B结束后完成的，只有在B阶段才能确定是否是最后一块
            MATMUL_MODULE(Context)->needFreeA =
                MATMUL_MODULE(Context)->isLastBaseBlockA && MATMUL_MODULE(Context)->isLastBaseBlockB;
            return this->LoadFullDataB(curRow, curCol, tileHeight, tileWidth, false);
        }

        return LocalTensor<TransT>{};
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT> &tensor = LocalTensor<TransT>{}, int32_t curRow = 0,
                                         int32_t curCol = 0)
    {
        MATMUL_MODULE(CubeInBuffer)->FreeTensor();
    }

    __aicore__ inline void Reset()
    {
    }

    __aicore__ inline void Destroy()
    {
        MATMUL_MODULE(CubeInBuffer)->Destroy();
    }

    __aicore__ inline void InitLoadDataParams(int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth)
    {
        // 获取用户设置的自定义数据
        bool isFirstBaseBlock = curRow == 0 && curCol == 0;
        if (isFirstBaseBlock) {
            uint64_t flagData = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
            mmIdx = flagData / NUM_10;
        }
        baseBlockSize = MATMUL_MODULE(CopyCubeInParams)->GetBufferSize();
        bool isLastBaseBlock = (curRow == MATMUL_MODULE(CopyCubeInParams)->GetTotalRow() - 1) &&
                               (curCol == MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() - 1);
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            MATMUL_MODULE(Context)->baseBlockSizeA = baseBlockSize;
            // 记录当前pos是本次计算的首块还是末块
            MATMUL_MODULE(Context)->isFirstBaseBlockA = isFirstBaseBlock;
            MATMUL_MODULE(Context)->isLastBaseBlockA = isLastBaseBlock;
        }
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            MATMUL_MODULE(Context)->baseBlockSizeB = baseBlockSize;
            MATMUL_MODULE(Context)->isFirstBaseBlockB = isFirstBaseBlock;
            MATMUL_MODULE(Context)->isLastBaseBlockB = isLastBaseBlock;
        }

        posL1 = MATMUL_MODULE(CubeInBuffer)->GetIterIndex(curRow, curCol);
        MATMUL_MODULE(Context)->mmIdx = static_cast<int32_t>(mmIdx);
    }

    __aicore__ inline LocalTensor<TransT> LoadFullDataA(const int32_t &curRow, const int32_t &curCol,
                                                        const int32_t &tileHeight, const int32_t &tileWidth,
                                                        const bool &reuse)
    {
        /*
         * A矩阵是一块可以容纳整个A的L1 buffer
         * A全局alloc一次，free一次
         * single块内复用
         */
        MATMUL_MODULE(Context)->tscmIndexA = this->tscmIndex;

        // 全载single块的大buffer，直到最后一个A-base块结束且最后一个B-base块结束，buffer生命周期结束
        // A矩阵free在B矩阵阶段进行
        MATMUL_MODULE(Context)->needAllocA =
            MATMUL_MODULE(Context)->isFirstBaseBlockA && MATMUL_MODULE(Context)->isFirstBaseBlockB;

        // single块内的base块复用，需要在搬运之前判断是否命中cache
        if (this->posL1 < tscmGlobal[this->tscmIndex].cacheSize) {
            MATMUL_MODULE(Context)->CurrentTensorA.SetAddr(tscmGlobal[this->tscmIndex].srcAddr);
            return MATMUL_MODULE(Context)->CurrentTensorA[this->posL1 * this->baseBlockSize];
        }

        if (MATMUL_MODULE(Context)->needAllocA) {
            MATMUL_MODULE(Context)->CurrentTensorA = MATMUL_MODULE(CubeInBuffer)->AllocTensor(this->posL1);
        } else {
            LocalTensor<TransT> aL1;
            aL1.SetAddr(tscmGlobal[this->tscmIndex].srcAddr);
            MATMUL_MODULE(Context)->CurrentTensorA = aL1[this->posL1 * this->baseBlockSize];
        }

        if (this->posL1 == 0) {
            tscmGlobal[this->tscmIndex].srcAddr = MATMUL_MODULE(Context)->CurrentTensorA.address_;
            tscmGlobal[this->tscmIndex].bufferSize = MATMUL_MODULE(Context)->CurrentTensorA.GetSize();
        }

        if (!reuse) {
            MATMUL_MODULE(DataCopyUtils)
                ->CopyTileToCube(MATMUL_MODULE(Context)->CurrentTensorA, curRow, curCol, tileHeight, tileWidth);
            tscmGlobal[this->tscmIndex].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorA);
            tscmGlobal[this->tscmIndex].localQue.template DeQue<TransT>();
        }

        tscmGlobal[this->tscmIndex].cacheSize += 1; // 每搬运一块，对应buffer缓存+1
        return MATMUL_MODULE(Context)->CurrentTensorA;
    }

    __aicore__ inline LocalTensor<TransT> LoadPartDataA(int32_t curRow, int32_t curCol, int32_t tileHeight,
                                                        int32_t tileWidth, uint8_t tscmIndex, int32_t posL1,
                                                        int32_t baseBlockSize)
    {
        /*
         * 不能容纳完整的single矩阵
         * 要分多次alloc + free
         * single块内复用
         */
        MATMUL_MODULE(Context)->tscmIndexA = tscmIndex;

        // 全载single块的大buffer，直到最后一个A-base块结束且最后一个B-base块结束，buffer生命周期结束
        MATMUL_MODULE(Context)->needAllocA =
            MATMUL_MODULE(Context)->isFirstBaseBlockA && MATMUL_MODULE(Context)->isFirstBaseBlockB;

        // A的free在B的阶段设置

        // buffer放不下整个矩阵A,需要重新alloc+Free
        int32_t baseBlockNum = tscmGlobal[tscmIndex].bufferSize / baseBlockSize;
        int32_t maxCacheSize = baseBlockNum / MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() *
                               MATMUL_MODULE(CopyCubeInParams)->GetTotalCol();
        MATMUL_MODULE(Context)->maxCacheSizePartA = maxCacheSize; // LoadPartDataA阶段设置，LoadDataB阶段使用，设置FreeA
        if (MATMUL_MODULE(Context)->isLastBaseBlockB && tscmGlobal[tscmIndex].cacheSize == maxCacheSize) {
            MATMUL_MODULE(Context)->needAllocA = true;
            tscmGlobal[tscmIndex].cacheSize = 0;
        }

        posL1 = posL1 % maxCacheSize; // buffer块重新计数
        // single块内的base块复用，需要在搬运之前判断是否命中cache
        if (posL1 < tscmGlobal[tscmIndex].cacheSize) {
            MATMUL_MODULE(Context)->CurrentTensorA.SetAddr(tscmGlobal[tscmIndex].srcAddr);
            return MATMUL_MODULE(Context)->CurrentTensorA[posL1 * baseBlockSize];
        }

        if (MATMUL_MODULE(Context)->needAllocA) {
            MATMUL_MODULE(Context)->CurrentTensorA = MATMUL_MODULE(CubeInBuffer)->AllocTensor(posL1);
        } else {
            LocalTensor<TransT> aL1;
            aL1.SetAddr(tscmGlobal[tscmIndex].srcAddr);
            MATMUL_MODULE(Context)->CurrentTensorA = aL1[posL1 * baseBlockSize];
        }

        MATMUL_MODULE(DataCopyUtils)
            ->CopyTileToCube(MATMUL_MODULE(Context)->CurrentTensorA, curRow, curCol, tileHeight, tileWidth);

        if (posL1 == 0) {
            tscmGlobal[tscmIndex].srcAddr = MATMUL_MODULE(Context)->CurrentTensorA.address_;
            tscmGlobal[tscmIndex].bufferSize = MATMUL_MODULE(Context)->CurrentTensorA.GetSize();
        }

        tscmGlobal[tscmIndex].cacheSize += 1; // 每搬运一块，对应的buffer缓存+1

        tscmGlobal[tscmIndex].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorA);
        tscmGlobal[tscmIndex].localQue.DeQue<TransT>();

        return MATMUL_MODULE(Context)->CurrentTensorA;
    }

    __aicore__ inline LocalTensor<TransT> LoadFullDataB(const int32_t &curRow, const int32_t &curCol,
                                                        const int32_t &tileHeight, const int32_t &tileWidth,
                                                        const bool &reuse)
    {
        /*
         * B矩阵是一块可以容纳整个B的L1 buffer
         * B全局alloc一次，free一次
         * single块内复用
         * 需要设置A矩阵的free标记
         */
        MATMUL_MODULE(Context)->tscmIndexB = this->tscmIndex;

        // B矩阵是一个大的single矩阵，B在A的M方向循环内一直复用，直到最后一个A块结束且最后一个B块结束，B矩阵生命周期结束
        MATMUL_MODULE(Context)->needAllocB =
            MATMUL_MODULE(Context)->isFirstBaseBlockA && MATMUL_MODULE(Context)->isFirstBaseBlockB;
        MATMUL_MODULE(Context)->needFreeB =
            MATMUL_MODULE(Context)->isLastBaseBlockA && MATMUL_MODULE(Context)->isLastBaseBlockB;

        // single块内的base块复用，需要在搬运之前判断是否命中cache
        if (this->posL1 < tscmGlobal[this->tscmIndex].cacheSize) {
            MATMUL_MODULE(Context)->CurrentTensorB.SetAddr(tscmGlobal[this->tscmIndex].srcAddr);
            return MATMUL_MODULE(Context)->CurrentTensorB[this->posL1 * this->baseBlockSize];
        }

        if (MATMUL_MODULE(Context)->needAllocB) {
            MATMUL_MODULE(Context)->CurrentTensorB = MATMUL_MODULE(CubeInBuffer)->AllocTensor(this->posL1);
        } else {
            LocalTensor<TransT> bL1;
            bL1.SetAddr(tscmGlobal[this->tscmIndex].srcAddr);
            MATMUL_MODULE(Context)->CurrentTensorB = bL1[this->posL1 * this->baseBlockSize];
        }

        if (this->posL1 == 0) {
            tscmGlobal[this->tscmIndex].srcAddr = MATMUL_MODULE(Context)->CurrentTensorB.address_;
            tscmGlobal[this->tscmIndex].bufferSize = MATMUL_MODULE(Context)->CurrentTensorB.GetSize();
        }

        if (!reuse) {
            MATMUL_MODULE(DataCopyUtils)
                ->CopyTileToCube(MATMUL_MODULE(Context)->CurrentTensorB, curRow, curCol, tileHeight, tileWidth);
            tscmGlobal[this->tscmIndex].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorB);
            tscmGlobal[this->tscmIndex].localQue.template DeQue<TransT>();
        }

        tscmGlobal[this->tscmIndex].cacheSize += 1; // 每搬运一块，对应buffer缓存+1
        return MATMUL_MODULE(Context)->CurrentTensorB;
    }

protected:
    int32_t mmIdx{0};
    int32_t baseBlockSize{0};
    int32_t posL1{0};
    int32_t tscmIndex{BUF_2};
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // FAG_COPY_CUBE_IN_BASE_KERNEL_H_
