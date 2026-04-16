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
 * \file fag_copy_cube_in_mm_256_256_128.h
 * \brief
 */

#ifndef FAG_COPY_CUBE_IN_MM_256_256_128_KERNEL_H_
#define FAG_COPY_CUBE_IN_MM_256_256_128_KERNEL_H_

namespace AscendC {
namespace Impl {
namespace Detail {

template<typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class T12FagCopyCubeIn : public FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG> {
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyUtils, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    using InputType = INPUT_TYPE;
    using BASE_MODULE = FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG>;
    inline __aicore__ T12FagCopyCubeIn() = default;
    inline __aicore__ ~T12FagCopyCubeIn() = default;

    __aicore__ inline void InitLoadDataParams2 (const int32_t& curRow, const int32_t& curCol, 
                                                const int32_t& tileHeight, const int32_t& tileWidth)
    {
        bool isFirstBaseBlock = curRow == 0 && curCol == 0;
        bool isLastBaseBlock = (curRow == MATMUL_MODULE(CopyCubeInParams)->GetTotalRow() - 1) &&
                               (curCol == MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() - 1);
        if (isFirstBaseBlock) {
            uint64_t flagData = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
            this->AReuse = false;
            this->BReuse = false;

            if(flagData >= 500){
                this->AReuse = true;
                flagData -= 500;
            }
            if(flagData >= 100){
                this->ApingPongIdx = 1;
                flagData -= 100;
            }else{
                this->ApingPongIdx = 0;
            }
            this->mmIdx = flagData / 10;
            flagData = flagData % 10;

            if(flagData >= 5){
                this->BReuse = true;
                flagData -= 5;
            }
            this->BpingPongIdx = flagData % 2;
            MATMUL_MODULE(Context)->mmIdx = static_cast<int32_t>(this->mmIdx);
        }

        this->posL1 = MATMUL_MODULE(CubeInBuffer)->GetIterIndex(curRow, curCol);
        this->baseBlockSize = MATMUL_MODULE(CopyCubeInParams)->GetBufferSize();
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            MATMUL_MODULE(Context)->baseBlockSizeA = this->baseBlockSize;
            MATMUL_MODULE(Context)->isFirstBaseBlockA = isFirstBaseBlock;
            MATMUL_MODULE(Context)->isLastBaseBlockA = isLastBaseBlock;
            this->tscmIndex = MM_BUFFER_TABLE_A[this->mmIdx][this->ApingPongIdx];
        }else if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            MATMUL_MODULE(Context)->baseBlockSizeB = this->baseBlockSize;
            MATMUL_MODULE(Context)->isFirstBaseBlockB = isFirstBaseBlock;
            MATMUL_MODULE(Context)->isLastBaseBlockB = isLastBaseBlock;
            this->tscmIndex = MM_BUFFER_TABLE_B[this->mmIdx][this->BpingPongIdx];
        }

    }
    
    __aicore__ inline LocalTensor<TransT> LoadBaseDataA(const int32_t& curRow, const int32_t&  curCol,
                                                        const int32_t& tileHeight, const int32_t& tileWidth)
    {
        /**
         * 只占用baseBlock大小的空间
         * 每次加载都需要alloc和free
         */
        MATMUL_MODULE(Context)->tscmIndexA = this->tscmIndex;
        MATMUL_MODULE(Context)->CurrentTensorA = MATMUL_MODULE(CubeInBuffer)->AllocTensor(0);
        
        if(this->posL1 < tscmGlobal[this->tscmIndex].cacheSize){
            return MATMUL_MODULE(Context)->CurrentTensorA;
        }

        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(MATMUL_MODULE(Context)->CurrentTensorA, 
                                                     curRow, curCol, tileHeight, tileWidth);
        tscmGlobal[this->tscmIndex].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorA);
        tscmGlobal[this->tscmIndex].localQue.template DeQue<TransT>();
        tscmGlobal[this->tscmIndex].cacheSize += 1;

        return MATMUL_MODULE(Context)->CurrentTensorA;
    }

    __aicore__ inline LocalTensor<TransT> LoadBaseDataB(const int32_t& curRow, const int32_t&  curCol,
                                                        const int32_t& tileHeight, const int32_t& tileWidth)
    {
        /**
         * 只占用baseBlock大小的空间
         * 每次加载都需要alloc和free
         */
        MATMUL_MODULE(Context)->tscmIndexB = this->tscmIndex;
        MATMUL_MODULE(Context)->needFreeB = true;
        MATMUL_MODULE(Context)->CurrentTensorB = MATMUL_MODULE(CubeInBuffer)->AllocTensor(0);

        if(this->posL1 < tscmGlobal[this->tscmIndex].cacheSize){
            return MATMUL_MODULE(Context)->CurrentTensorB;
        }

        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(MATMUL_MODULE(Context)->CurrentTensorB, 
                                                     curRow, curCol, tileHeight, tileWidth);
        tscmGlobal[this->tscmIndex].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorB);
        tscmGlobal[this->tscmIndex].localQue.template DeQue<TransT>();
        tscmGlobal[this->tscmIndex].cacheSize += 1;

        return MATMUL_MODULE(Context)->CurrentTensorB;
    }

    __aicore__ inline LocalTensor<TransT> LoadData(const int32_t& curRow, const int32_t& curCol,
                                                   const int32_t& tileHeight, const int32_t& tileWidth)
    {
        this->InitLoadDataParams2(curRow, curCol, tileHeight, tileWidth);
        
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            //mm1 mm2 A矩阵复用
            if(this->mmIdx == T12_MM_QK || this->mmIdx == T12_MM_DYV){
                return this->LoadFullDataA(curRow, curCol, tileHeight, tileWidth, this->AReuse);
            }else{
                return this->LoadBaseDataA(curRow, curCol, tileHeight, tileWidth);
            }
        }else if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            if(this->mmIdx == T12_MM_QK || this->mmIdx == T12_MM_DYV){
                MATMUL_MODULE(Context)->needFreeA = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                                                    MATMUL_MODULE(Context)->isLastBaseBlockB;
            }else{
                MATMUL_MODULE(Context)->needFreeA = true;
            }

            if(this->mmIdx == T12_MM_DYV){
                return this->LoadBaseDataB(curRow, curCol, tileHeight, tileWidth);
            }else{
                return this->LoadFullDataB(curRow, curCol, tileHeight, tileWidth, this->BReuse);
            }
        }

        return LocalTensor<TransT>{};
    }

private:
    bool AReuse{false};
    bool BReuse{false};
    uint8_t ApingPongIdx{0};
    uint8_t BpingPongIdx{0};
};
}
}
}
#endif // FAG_COPY_CUBE_IN_MM_256_256_128_KERNEL_H_
