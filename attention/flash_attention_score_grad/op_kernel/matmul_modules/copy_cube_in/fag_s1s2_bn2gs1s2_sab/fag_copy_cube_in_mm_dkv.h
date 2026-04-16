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
 * \file fag_copy_cube_in_mm_dkv.h
 * \brief
 */

#ifndef FAG_COPY_CUBE_IN_MM_DKV_KERNEL_H_
#define FAG_COPY_CUBE_IN_MM_DKV_KERNEL_H_
namespace AscendC {
namespace Impl {
namespace Detail {
using namespace AscendC;

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FagCopyCubeInMMDkv : public FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG> {
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulShapeInfo, INPUT_TYPE::TAG);

    using TransT = typename INPUT_TYPE::TRANS_T;

public:
    using BASE_MODULE = FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG>;

public:
    __aicore__ inline FagCopyCubeInMMDkv() = default;
    __aicore__ inline ~FagCopyCubeInMMDkv() = default;

    __aicore__ inline LocalTensor<TransT> LoadData(int32_t curRow, int32_t curCol,
                                                   int32_t tileHeight, int32_t tileWidth)
    {
        this->InitLoadDataParams(curRow, curCol, tileHeight, tileWidth);

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            // reuse DQ0/DQ1 A matrix
            if (this->mmIdx == MM_DK0_INDEX || this->mmIdx == MM_DK1_INDEX) {
                this->tscmIndex = BUF_2;

                MATMUL_MODULE(Context)->tscmIndexA = this->tscmIndex;
                MATMUL_MODULE(Context)->needAllocA = MATMUL_MODULE(Context)->isFirstBaseBlockA &&
                                                    MATMUL_MODULE(Context)->isFirstBaseBlockB;

                int32_t posL1Trans = MATMUL_MODULE(CubeInBuffer)->GetTransIterIndex(curRow, curCol);

                if (MATMUL_MODULE(Context)->needAllocA) {
                    MATMUL_MODULE(Context)->CurrentTensorA = MATMUL_MODULE(CubeInBuffer)->AllocTensor(posL1Trans);
                } else {
                    LocalTensor<TransT> aL1;
                    aL1.SetAddr(tscmGlobal[this->tscmIndex].srcAddr);
                    MATMUL_MODULE(Context)->CurrentTensorA = aL1[posL1Trans * this->baseBlockSize];
                }

                return MATMUL_MODULE(Context)->CurrentTensorA;
            }

            if (this->mmIdx == MM_DV_INDEX) {
                this->tscmIndex = BUF_2;
                return this->LoadPartDataA(curRow, curCol, tileHeight, tileWidth,
                                           this->tscmIndex, this->posL1, this->baseBlockSize);
            }
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            if (this->mmIdx == MM_DK0_INDEX || this->mmIdx == MM_DV_INDEX) {
                this->tscmIndex = this->mmIdx == MM_DK0_INDEX ? BUF_0 : BUF_1;

                MATMUL_MODULE(Context)->needFreeA = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                                                    MATMUL_MODULE(Context)->isLastBaseBlockB;

                if (this->mmIdx == MM_DV_INDEX && MATMUL_MODULE(Context)->isLastBaseBlockB &&
                    tscmGlobal[MATMUL_MODULE(Context)->tscmIndexA].cacheSize == 
                        MATMUL_MODULE(Context)->maxCacheSizePartA) {
                    MATMUL_MODULE(Context)->needFreeA = true;
                }

                return this->LoadFullDataB(curRow, curCol, tileHeight, tileWidth, false);
            }
            
            // reuse DK0 B matrix
            if (this->mmIdx == MM_DK1_INDEX) {
                this->tscmIndex = BUF_0;

                MATMUL_MODULE(Context)->needFreeA = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                                                    MATMUL_MODULE(Context)->isLastBaseBlockB;

                MATMUL_MODULE(Context)->tscmIndexB = this->tscmIndex;
                MATMUL_MODULE(Context)->needAllocB = MATMUL_MODULE(Context)->isFirstBaseBlockA &&
                                                     MATMUL_MODULE(Context)->isFirstBaseBlockB;
                MATMUL_MODULE(Context)->needFreeB = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                                                    MATMUL_MODULE(Context)->isLastBaseBlockB;

                if (MATMUL_MODULE(Context)->needAllocB) {
                    MATMUL_MODULE(Context)->CurrentTensorB = MATMUL_MODULE(CubeInBuffer)->AllocTensor(this->posL1);
                    tscmGlobal[this->tscmIndex].srcAddr = MATMUL_MODULE(Context)->CurrentTensorB.address_;
                } else {
                    LocalTensor<TransT> bL1;
                    bL1.SetAddr(tscmGlobal[this->tscmIndex].srcAddr);
                    MATMUL_MODULE(Context)->CurrentTensorB = bL1[this->posL1 * this->baseBlockSize];
                }
                return MATMUL_MODULE(Context)->CurrentTensorB;
            }
        }

        return LocalTensor<TransT>{};
    }
};
}
}
}
#endif // FAG_COPY_CUBE_IN_MM_DKV_KERNEL_H_