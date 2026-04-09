/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_FIXPIPE_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_FIXPIPE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"

namespace Catlass::Epilogue::Block {

template <
    class L0TileShape_,
    class ElementOut_,
    class ElementIn_
>
class BlockEpilogue<
    BlockEpilogueFixpipe,
    L0TileShape_,
    ElementOut_,
    ElementIn_
> {
public:
    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
    };

    using Params = Arguments;
    using DispatchPolicy = BlockEpilogueFixpipe;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOut = ElementOut_;
    using ElementIn = ElementIn_;

    static constexpr int64_t ML0_ = tla::get<0>(L0TileShape_{});
    static constexpr int64_t NL0_ = tla::get<1>(L0TileShape_{});

    // block shape
    using BlockShape = tla::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = tla::Shape<int64_t, int64_t, int64_t, int64_t>;




    // attribute
    ProblemShape problemShape_;

    CATLASS_DEVICE 
    BlockEpilogue(Params const &params, ProblemShape &problemShape)
    {


        problemShape_ = problemShape;
        ASCENDC_ASSERT(sizeof(ElementIn) >= sizeof(ElementOut),
            { KERNEL_LOG(KERNEL_EORROR, "Unsupport dtype size %zu, %zu!", sizeof(ElementIn), sizeof(ElementOut)); });
    }

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE 
    void operator()(TensorOut& tensorOut, TensorIn& tensorIn, bool splitM)
    {
        int64_t blockShapeM = tla::get<0>(tensorOut.shape());
        int64_t blockShapeN = tla::get<1>(tensorOut.shape());
        int64_t halfBlockShapeM = CeilDiv(blockShapeM, AscendC::GetTaskRation());
        if (splitM) {
            blockShapeM = (static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL
                            ? (halfBlockShapeM - AscendC::GetSubBlockIdx())
                            : halfBlockShapeM;
        }
        

        // real copy data size
        int64_t inputSize = blockShapeM * blockShapeN;



        if (inputSize <= 0) {
            return;
        }

        auto tensorTileOut = GetTile(
            tensorOut,
            tla::MakeCoord(halfBlockShapeM * (AscendC::GetSubBlockIdx() & 0x1), 0),
            tla::MakeShape(blockShapeM, blockShapeN)
        );

        using CopyUbToGm = Epilogue::Tile::CopyUb2GmTla<ArchTag, TensorIn, decltype(tensorTileOut)>;
        CopyUbToGm copyUbToGm;
        copyUbToGm(tensorTileOut, tensorIn);
    }
    
    CATLASS_HOST_DEVICE static Status CanImplement(Arguments const &args)
    {
        if (ML0_  * NL0_  * sizeof(ElementIn_) > ArchTag::UB_SIZE) {
            return Status::kInvalid;
        }
        return Status::kSuccess;
    }
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_FIXPIPE_HPP
