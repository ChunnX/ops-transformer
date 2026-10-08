/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GROUPED_ROTATE_QUANT_TENSOR_API_H
#define GROUPED_ROTATE_QUANT_TENSOR_API_H

#include "grouped_rotate_quant_kernel_args.h"
#include "blaze/attention/block/block_mmad_rotate_quant.h"
#include "blaze/attention/kernel/kernel_grouped_rotate_quant.h"
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"
#include "adv_api/matmul/matmul.h"

namespace GroupedRotateQuant {
// Translate the operator ABI into library Params. No reinterpret_cast between
// tiling structures; the library does not depend on ops-transformer headers.
template <typename XType, typename YType>
struct TensorApiBackend {
    static constexpr bool AVAILABLE = true;
    struct RowMajor {};
    using A = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, XType, false>;
    using C = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, bfloat16_t>;
    using Scale = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AscendC::fp8_e8m0_t>;
    using Mmad = Blaze::Attention::Block::BlockMmad<Blaze::Attention::BlockRotateQuant<>,
        A, RowMajor, A, RowMajor, Scale, RowMajor, C, RowMajor>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<
        bfloat16_t, YType, AscendC::fp8_e8m0_t>;
    using Implementation = Blaze::Attention::Kernel::GroupedRotateQuantKernel<Mmad, Epilogue>;

    struct Kernel {
        __aicore__ inline void operator()(const KernelArguments& args)
        {
            const auto& t = args.tiling;
            typename Implementation::Params params;
            params.xList = args.xList;
            params.rotationList = args.rotationList;
            params.yList = args.yList;
            params.scaleList = args.scaleList;
            params.alpha = args.alpha;
            params.groupList = args.groupList;
            auto& groups = params.groups;
            groups.groupNum = t.groupNum;
            groups.xNum = t.xNum;
            groups.rotationNum = t.rotationNum;
            groups.yNum = t.yNum;
            groups.groupType = t.groupType;
            groups.groupListType = t.groupListType;
            groups.hasGroupList = t.hasGroupList != 0;
            groups.runtimeM = t.runtimeM != 0;
            groups.totalM = t.totalM;
            groups.totalElements = t.totalElements;
            groups.tileCount = t.tileCount;
            for (uint32_t i = 0; i < ROTATION_PROFILE_NUM; ++i) {
                const bool validProfile = t.tiles[i].rotationSize == 0 || t.tiles[i].rotationSize == (32U << i);
                ASSERT(validProfile);
                if (!validProfile) {
                    return;
                }
                groups.mTile[i] = t.tiles[i].mTile;
            }
            params.usedCoreNum = t.usedCoreNum;
            params.roundMode = t.roundMode;
            params.scaleAlg = t.scaleAlg;
            params.dstTypeMax = t.dstTypeMax;
            params.invDstTypeMax = t.invDstTypeMax;
            Implementation kernel;
            kernel(params);
        }
    };
};
} // namespace GroupedRotateQuant
#endif
