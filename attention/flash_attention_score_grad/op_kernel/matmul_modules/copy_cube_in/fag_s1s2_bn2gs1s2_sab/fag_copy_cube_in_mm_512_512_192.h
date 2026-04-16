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
 * \file fag_copy_cube_in_mm_512_512_192.h
 * \brief
 */

#ifndef FAG_COPY_CUBE_IN_MM_512_512_192_KERNEL_H_
#define FAG_COPY_CUBE_IN_MM_512_512_192_KERNEL_H_

namespace AscendC {
namespace Impl {
namespace Detail {

static constexpr int64_t C0_ALIGN_SIZE = 16;

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FagCopyCubeInMM1HeadDim192 : public FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG> {
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE_ON(DataCopyUtils, INPUT_TYPE::TAG);

    using TransT = typename INPUT_TYPE::TRANS_T;

public:
    using BASE_MODULE = FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG>;

public:
    __aicore__ inline FagCopyCubeInMM1HeadDim192() = default;
    __aicore__ inline ~FagCopyCubeInMM1HeadDim192() = default;

    __aicore__ inline LocalTensor<TransT> LoadData(int32_t curRow, int32_t curCol,
                                                   int32_t tileHeight, int32_t tileWidth)
    {
        this->InitLoadDataParams(curRow, curCol, tileHeight, tileWidth);

        // A矩阵处理
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            return LoadTensorA(curRow, curCol, tileHeight, tileWidth, this->posL1, this->baseBlockSize);
        }

        // B矩阵处理
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            return LoadTensorB(curRow, curCol, tileHeight, tileWidth, this->posL1, this->baseBlockSize);
        }

        return LocalTensor<TransT>{};
    }

    __aicore__ inline LocalTensor<TransT> LoadTensorA(int32_t curRow, int32_t curCol,
        int32_t tileHeight, int32_t tileWidth, int32_t posL1, int32_t baseBlockSize)
    {
        /*
         *GM->L1 按singleK搬运，L1->L0A按baseK搬运
         */
        uint8_t tscmIndex = BUF_0;
        MATMUL_MODULE(Context)->tscmIndexA = tscmIndex;
        int32_t cacheBaseBlockNum = MATMUL_MODULE(CopyCubeInParams)->GetTotalCol();
        int32_t totalBlockNum = MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() * 
                                MATMUL_MODULE(CopyCubeInParams)->GetTotalRow();

        MATMUL_MODULE(Context)->needAllocA = MATMUL_MODULE(Context)->isFirstBaseBlockA &&
                                             MATMUL_MODULE(Context)->isFirstBaseBlockB;

        if (posL1 < totalBlockNum - 1) {
            this->offsetA[posL1 + 1] = 
                this->offsetA[posL1] + (tileHeight + C0_ALIGN_SIZE - 1) / C0_ALIGN_SIZE * C0_ALIGN_SIZE * tileWidth;
        }

        if (posL1 < tscmGlobal[tscmIndex].cacheSize || posL1 % cacheBaseBlockNum != 0) {
            MATMUL_MODULE(Context)->CurrentTensorA.SetAddr(tscmGlobal[tscmIndex].srcAddr);
            return MATMUL_MODULE(Context)->CurrentTensorA[this->offsetA[posL1]];
        }

        if (MATMUL_MODULE(Context)->needAllocA) {
            MATMUL_MODULE(Context)->CurrentTensorA = MATMUL_MODULE(CubeInBuffer)->AllocTensor(posL1);
        } else {
            LocalTensor<TransT> aL1;
            aL1.SetAddr(tscmGlobal[tscmIndex].srcAddr);
            MATMUL_MODULE(Context)->CurrentTensorA = aL1[this->offsetA[posL1]];
        }

        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(
            MATMUL_MODULE(Context)->CurrentTensorA,
            curRow,
            curCol,
            (tileHeight + C0_ALIGN_SIZE - 1) / C0_ALIGN_SIZE * C0_ALIGN_SIZE,
            MATMUL_MODULE(CopyCubeInParams)->GetSingleWidth()
        );

        if (posL1 == 0) {
            tscmGlobal[tscmIndex].srcAddr = MATMUL_MODULE(Context)->CurrentTensorA.address_;
            tscmGlobal[tscmIndex].bufferSize = MATMUL_MODULE(Context)->CurrentTensorA.GetSize();
        }

        tscmGlobal[tscmIndex].cacheSize += cacheBaseBlockNum;

        tscmGlobal[tscmIndex].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorA);
        tscmGlobal[tscmIndex].localQue.DeQue<TransT>();

        return MATMUL_MODULE(Context)->CurrentTensorA;
    }

    __aicore__ inline LocalTensor<TransT> LoadTensorB(int32_t curRow, int32_t curCol,
        int32_t tileHeight, int32_t tileWidth, int32_t posL1, int32_t baseBlockSize)
    {
        /*
         *GM->L1 按singleK搬运，L1->L0A按baseK搬运
         */
        uint8_t tscmIndex = BUF_1;
        // A的Free是在B结束后完成的，只有在B阶段才能确定是否是最后一块
        MATMUL_MODULE(Context)->needFreeA = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                                            MATMUL_MODULE(Context)->isLastBaseBlockB;
        MATMUL_MODULE(Context)->tscmIndexB = tscmIndex;

        // B矩阵是一个大的single矩阵，B在A的M方向循环内一直复用，直到最后一个A块结束且最后一个B块结束，B矩阵生命周期结束
        MATMUL_MODULE(Context)->needAllocB = MATMUL_MODULE(Context)->isFirstBaseBlockA &&
                                             MATMUL_MODULE(Context)->isFirstBaseBlockB;
        MATMUL_MODULE(Context)->needFreeB = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                                            MATMUL_MODULE(Context)->isLastBaseBlockB;

        int32_t cacheBaseBlockNum = MATMUL_MODULE(CopyCubeInParams)->GetTotalRow();
        int32_t totalBlockNum = MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() * 
                                MATMUL_MODULE(CopyCubeInParams)->GetTotalRow();
        if (posL1 < totalBlockNum - 1) {
            this->offsetB[posL1 + 1] = 
                this->offsetB[posL1] + tileHeight * ((tileWidth + C0_ALIGN_SIZE - 1) / C0_ALIGN_SIZE * C0_ALIGN_SIZE);
        }

        bool cacheHit = posL1 < tscmGlobal[tscmIndex].cacheSize || posL1 % cacheBaseBlockNum != 0;
        if (cacheHit) {
            MATMUL_MODULE(Context)->CurrentTensorB.SetAddr(tscmGlobal[tscmIndex].srcAddr);
            return MATMUL_MODULE(Context)->CurrentTensorB[this->offsetB[posL1]];
        }

        if (MATMUL_MODULE(Context)->needAllocB) {
            MATMUL_MODULE(Context)->CurrentTensorB = MATMUL_MODULE(CubeInBuffer)->AllocTensor(posL1);
        } else {
            LocalTensor<TransT> bL1;
            bL1.SetAddr(tscmGlobal[tscmIndex].srcAddr);
            MATMUL_MODULE(Context)->CurrentTensorB = bL1[this->offsetB[posL1]];
        }

        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(
            MATMUL_MODULE(Context)->CurrentTensorB,
            curRow,
            curCol,
            MATMUL_MODULE(CopyCubeInParams)->GetSingleHeight(),
            (tileWidth + C0_ALIGN_SIZE - 1) / C0_ALIGN_SIZE * C0_ALIGN_SIZE
        );
        
        if (posL1 == 0) {
            tscmGlobal[tscmIndex].srcAddr = MATMUL_MODULE(Context)->CurrentTensorB.address_;
            tscmGlobal[tscmIndex].bufferSize = MATMUL_MODULE(Context)->CurrentTensorB.GetSize();
        }

        tscmGlobal[tscmIndex].cacheSize += cacheBaseBlockNum;

        tscmGlobal[tscmIndex].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorB);
        tscmGlobal[tscmIndex].localQue.DeQue<TransT>();

        return MATMUL_MODULE(Context)->CurrentTensorB;
    }
private:
    int64_t offsetA[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t offsetB[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FagCopyCubeInMM2HeadDim192 : public FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG> {
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE_ON(DataCopyUtils, INPUT_TYPE::TAG);

    using TransT = typename INPUT_TYPE::TRANS_T;

public:
    using BASE_MODULE = FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG>;

public:
    __aicore__ inline FagCopyCubeInMM2HeadDim192() = default;
    __aicore__ inline ~FagCopyCubeInMM2HeadDim192() = default;

    __aicore__ inline LocalTensor<TransT> LoadData(int32_t curRow, int32_t curCol,
                                                   int32_t tileHeight, int32_t tileWidth)
    {
        this->InitLoadDataParams(curRow, curCol, tileHeight, tileWidth);

        // A矩阵处理
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            this->tscmIndex = BUF_0;
            return this->LoadPartDataA(curRow, curCol, tileHeight, tileWidth,
                                       this->tscmIndex, this->posL1, this->baseBlockSize);
        }

        // B矩阵处理
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            this->tscmIndex = BUF_1;

            MATMUL_MODULE(Context)->needFreeA = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                                                MATMUL_MODULE(Context)->isLastBaseBlockB;
            if (MATMUL_MODULE(Context)->isLastBaseBlockB &&
                tscmGlobal[MATMUL_MODULE(Context)->tscmIndexA].cacheSize == MATMUL_MODULE(Context)->maxCacheSizePartA) {
                MATMUL_MODULE(Context)->needFreeA = true;
            }

            return this->LoadFullDataB(curRow, curCol, tileHeight, tileWidth, false);
        }

        return LocalTensor<TransT>{};
    }
private:
    int64_t offsetA[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t offsetB[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};
}
}
}
#endif // FAG_COPY_CUBE_IN_MM_512_512_192_KERNEL_H_
