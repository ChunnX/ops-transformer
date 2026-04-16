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
 * \file fag_copy_cube_in_mm_dq.h
 * \brief
 */

#ifndef FAG_COPY_CUBE_IN_MM_DQ_KERNEL_H_
#define FAG_COPY_CUBE_IN_MM_DQ_KERNEL_H_

namespace AscendC {
namespace Impl {
namespace Detail {
using namespace AscendC;

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FagCopyCubeInMMDq : public FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG> {
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulShapeInfo, INPUT_TYPE::TAG);

    using TransT = typename INPUT_TYPE::TRANS_T;

public:
    using BASE_MODULE = FagCopyCubeIn<IMPL, INPUT_TYPE, MM_CFG>;

public:
    __aicore__ inline FagCopyCubeInMMDq() = default;
    __aicore__ inline ~FagCopyCubeInMMDq() = default;

    __aicore__ inline LocalTensor<TransT> LoadData(int32_t curRow, int32_t curCol,
                                                   int32_t tileHeight, int32_t tileWidth)
    {
        this->InitLoadDataParams(curRow, curCol, tileHeight, tileWidth);

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            this->tscmIndex = BUF_2;
            return this->LoadFullDataA(curRow, curCol, tileHeight, tileWidth, false);
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            this->tscmIndex = BUF_1;
            MATMUL_MODULE(Context)->needFreeA = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                                                MATMUL_MODULE(Context)->isLastBaseBlockB;
            return this->LoadFullDataB(curRow, curCol, tileHeight, tileWidth, false);
        }

        return LocalTensor<TransT>{};
    }
};
}
}
}
#endif // FAG_COPY_CUBE_IN_MM_DQ_KERNEL_H_