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
 * \file fag_copy_cube_in_mm_512_512_64.h
 * \brief
 */

#ifndef FAG_COPY_CUBE_IN_MM_512_512_64_KERNEL_H_
#define FAG_COPY_CUBE_IN_MM_512_512_64_KERNEL_H_

namespace AscendC {
namespace Impl {
namespace Detail {


template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FagCopyCubeInHeadDim64 : public FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG> {
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE_ON(DataCopyUtils, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);

    using TransT = typename INPUT_TYPE::TRANS_T;

public:
    using BASE_MODULE = FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG>;

public:
    __aicore__ inline FagCopyCubeInHeadDim64() = default;
    __aicore__ inline ~FagCopyCubeInHeadDim64() = default;

    __aicore__ inline void InitLoadDataParams(const int32_t curRow, const int32_t curCol, const int32_t tileHeight,
                                              const int32_t tileWidth)
    {
        bool isFirstBaseBlock = curRow == 0 && curCol == 0;
        bool isLastBaseBlock = (curRow == MATMUL_MODULE(CopyCubeInParams)->GetTotalRow() - 1) &&
                               (curCol == MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() - 1);
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            if (isFirstBaseBlock) {
                // 获取用户设置的自定义数据
                // 仅有A矩阵需要这个数据
                uint64_t flagData = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
                this->mmIdx = flagData / NUM_10;
                this->maxBuffer = tscmGlobal[BUF_0].bufferSize;
                this->bufferLimit = TSAB_MM345_BASE_A_NUM - this->maxBuffer;

                // 初始化参数
                MATMUL_MODULE(Context)->needFreeA = false;
                MATMUL_MODULE(Context)->needFreeB = false;
                MATMUL_MODULE(Context)->isLastBaseBlockA = false;
            }
            if (isLastBaseBlock) {
                MATMUL_MODULE(Context)->isLastBaseBlockA = true;
            }
            if (this->mmIdx == MM_DK0_INDEX) {
                this->posL1 = MATMUL_MODULE(CubeInBuffer)->GetTransIterIndex(curRow, curCol);
            } else {
                this->posL1 = MATMUL_MODULE(CubeInBuffer)->GetIterIndex(curRow, curCol);
            }
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            if (isLastBaseBlock && MATMUL_MODULE(Context)->isLastBaseBlockA) {
                // A和B都为最后一块的时候free
                MATMUL_MODULE(Context)->needFreeA = true;
                MATMUL_MODULE(Context)->needFreeB = true;
            }
            this->posL1 = MATMUL_MODULE(CubeInBuffer)->GetIterIndex(curRow, curCol);
        }
        this->baseBlockSize = MATMUL_MODULE(CopyCubeInParams)->GetBufferSize();
    }

    __aicore__ inline LocalTensor<TransT> LoadPartDataA(const int32_t curRow, const int32_t curCol,
                                                        const int32_t tileHeight, const int32_t tileWidth)
    {
        /*
         * 1.不能容纳完整的single矩阵
         * 2.计算dQ、dK、dV时调用
         * 3.maxBuffer 控制能够容纳最大的base块数量
         */
        LocalTensor<TransT> tmpLocalTensor;
        bool alloc = this->posL1 == 0;
        this->posL1 = this->posL1 % this->maxBuffer;

        if (alloc) {
            tmpLocalTensor = tscmGlobal[BUF_0].localQue.template AllocTensor<TransT>();
            tscmGlobal[BUF_0].srcAddr = tmpLocalTensor.address_;
        } else {
            tmpLocalTensor.SetAddr(tscmGlobal[BUF_0].srcAddr);
            tmpLocalTensor = tmpLocalTensor[this->posL1 * this->baseBlockSize];
        }

        if (this->mmIdx != MM_DK0_INDEX || (this->mmIdx == MM_DK0_INDEX && this->posL1 < this->bufferLimit)) {
            MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(tmpLocalTensor, curRow, curCol, tileHeight, tileWidth);
            tscmGlobal[BUF_0].localQue.EnQue(tmpLocalTensor);
            tscmGlobal[BUF_0].localQue.template DeQue<TransT>();
        }

        return tmpLocalTensor;
    }

    __aicore__ inline LocalTensor<TransT> LoadFullDataA(const int32_t curRow, const int32_t curCol,
                                                        const int32_t tileHeight, const int32_t tileWidth)
    {
        /*
         * 1.容纳完整的Single块
         * 2.计算QK、dyV时调用
         */
        LocalTensor<TransT> tmpLocalTensor;
        // single块内的base块复用，需要在搬运之前判断是否命中cache
        if (this->posL1 < tscmGlobal[BUF_0].cacheSize) {
            tmpLocalTensor.SetAddr(tscmGlobal[BUF_0].srcAddr);
            return tmpLocalTensor[this->posL1 * this->baseBlockSize];
        }

        if (this->posL1 == 0) {
            tmpLocalTensor = tscmGlobal[BUF_0].localQue.template AllocTensor<TransT>();
            tscmGlobal[BUF_0].srcAddr = tmpLocalTensor.address_;
        } else {
            tmpLocalTensor.SetAddr(tscmGlobal[BUF_0].srcAddr);
            tmpLocalTensor = tmpLocalTensor[this->posL1 * this->baseBlockSize];
        }

        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(tmpLocalTensor, curRow, curCol, tileHeight, tileWidth);
        tscmGlobal[BUF_0].localQue.EnQue(tmpLocalTensor);
        tscmGlobal[BUF_0].localQue.template DeQue<TransT>();
        tscmGlobal[BUF_0].cacheSize += 1;

        return tmpLocalTensor;
    }

    __aicore__ inline LocalTensor<TransT> LoadFullDataB(const int32_t curRow, const int32_t curCol,
                                                        const int32_t tileHeight, const int32_t tileWidth)
    {
        /*
         * 1.容纳完整的Single块
         */
        LocalTensor<TransT> tmpLocalTensor;
        // single块内的base块复用，需要在搬运之前判断是否命中cache
        if (this->posL1 < tscmGlobal[BUF_1].cacheSize) {
            tmpLocalTensor.SetAddr(tscmGlobal[BUF_1].srcAddr);
            return tmpLocalTensor[this->posL1 * this->baseBlockSize];
        }

        if (this->posL1 == 0) {
            tmpLocalTensor = tscmGlobal[BUF_1].localQue.template AllocTensor<TransT>();
            tscmGlobal[BUF_1].srcAddr = tmpLocalTensor.address_;
        } else {
            tmpLocalTensor.SetAddr(tscmGlobal[BUF_1].srcAddr);
            tmpLocalTensor = tmpLocalTensor[this->posL1 * this->baseBlockSize];
        }

        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(tmpLocalTensor, curRow, curCol, tileHeight, tileWidth);
        tscmGlobal[BUF_1].localQue.EnQue(tmpLocalTensor);
        tscmGlobal[BUF_1].localQue.template DeQue<TransT>();
        tscmGlobal[BUF_1].cacheSize += 1;

        return tmpLocalTensor;
    }

    __aicore__ inline LocalTensor<TransT> LoadData(const int32_t curRow, const int32_t curCol, const int32_t tileHeight,
                                                   const int32_t tileWidth)
    {
        this->InitLoadDataParams(curRow, curCol, tileHeight, tileWidth);

        // A矩阵处理
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            if (this->mmIdx == MM_QK_INDEX || this->mmIdx == MM_DYV_INDEX) {
                return this->LoadFullDataA(curRow, curCol, tileHeight, tileWidth);
            }
            return this->LoadPartDataA(curRow, curCol, tileHeight, tileWidth);
        }

        // B矩阵处理
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            return this->LoadFullDataB(curRow, curCol, tileHeight, tileWidth);
        }

        return LocalTensor<TransT>{};
    }

private:
    int32_t maxBuffer;
    int32_t bufferLimit;
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // FAG_COPY_CUBE_IN_MM_512_512_64_KERNEL_H_
