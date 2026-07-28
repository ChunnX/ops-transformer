/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP4
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP4

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class ElementO_,
    class ElementOTmp_,
    class ElementDm_,
    class TileCopy_,
    class OTmpSrcPos_,
    LseMode LSE_MODE_,
    LseFormat LSE_FORMAT_>
class BlockEpilogue<
    EpilogueAtlasA5BsaRescaleO<LSE_MODE_, LSE_FORMAT_, true>,
    ElementO_,
    ElementOTmp_,
    ElementS_,
    TileCopy_,
    OTmpSrcPos_>
{
public:
    using DispatchPolicy = EpilogueAtlasA5BsaRescaleO<LSE_MODE_, LSE_FORMAT_, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using ElementLse = ElementOTmp;
    using ElementDm = ElementDm_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;
    using LayoutOTmp = TileCopy::LayoutOTmp;
    using LayoutO = TileCopy::LayoutO;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;
    using CopyUbToGmLse = typename TileCopy::CopyUbToGmLse;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource) {
        // FIXPIPE<->V 区
        for (uint32_t i = 0; i < MXFP4::UB_S_BUF_CNT; i++) {
            sUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(
                MXFP4::UB_S_BUF_OFFSET + MXFP4::UB_S_BUF_SIZE * i);
        }
        oTmpUBTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(MXFP4::UB_OTMP_BUF_OFFSET);
        localRowSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP4::UB_LOCAL_ROW_SUM_OFFSET);
        globalRowSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP4::UB_GLOBAL_ROW_SUM_OFFSET);

        // 输出缓冲区
        for (uint32_t i = 0; i < MXFP4::UB_P_BUF_CNT; i++) {
            pUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDisguiseP>(
                MXFP4::UB_P_BUF_OFFSET + MXFP4::UB_P_BUF_SIZE * i);
        }
        oTransUBTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(MXFP4::UB_O_TRANS_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_P_SCALE_CNT; i++) {
            pscaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementPScale>(
                MXFP4::UB_P_SCALE_OFFSET + MXFP4::UB_P_SCALE_SIZE * i);
        }
        oUBTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(MXFP4::UB_O_OFFSET);

        // L1 <-> UB
        for (uint32_t i = 0; i < MXFP4::UB_PEER_GLOBAL_MAX_CNT; i++) {
            peerGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_PEER_GLOBAL_MAX_OFFSET + MXFP4::UB_PEER_GLOBAL_MAX_SIZE * i);
        }

        // 常驻 buffer
        softmaxMaxUBTensor = resource.ubBuf.template GetBufferByByte<ElementMax>(MXFP4::UB_SOFTMAX_MAX_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GROUP_MAX_OFFSET + MXFP4::UB_LOCAL_GROUP_MAX_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GLOBAL_MAX_OFFSET + MXFP4::UB_LOCAL_GLOBAL_MAX_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_UPDATE_SCALE_CNT; i++) {
            updateScaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                MXFP4::UB_UPDATE_SCALE_OFFSET + MXFP4::UB_UPDATE_SCALE_SIZE * i);
        }
        indexUBTensor = resource.ubBuf.template GetBufferByByte<ElementIndex>(MXFP4::UB_INDEX_OFFSET);
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorO, class TensorLse>
    __aicore__ inline
    void operator()(TensorO &gOTensor,
                    TensorLse &gLseTensor,
                    GemmCoord &actualOriShape,
                    TaskInfo &taskInfo,
                    TileInfo &tileInfo)
    { 
        

    }

private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementInput>  sUBTensor[MXFP4::UB_S_BUF_CNT];        // mm1Res(S)
    AscendC::LocalTensor<ElementOutput> oTmpUBTensor;                          // mm2Res(PV)
    AscendC::LocalTensor<float>        localRowSumUBTensor;                    // LocalRowSum
    AscendC::LocalTensor<float>        globalRowSumUBTensor;                  // GlobalRowSum
    AscendC::LocalTensor<ElementDisguiseP> pUBTensor[MXFP4::UB_P_BUF_CNT];      // vec1Res(P)
    AscendC::LocalTensor<ElementOutput> oTransUBTensor;                       // attnTrans(空间复用P)
    AscendC::LocalTensor<ElementPScale> pscaleUBTensor[MXFP4::UB_P_SCALE_CNT]; // pScale(复用P)
    AscendC::LocalTensor<ElementOutput> oUBTensor;                            // attentionOut
    AscendC::LocalTensor<ElementMax>   peerGlobalMaxUBTensor[MXFP4::UB_PEER_GLOBAL_MAX_CNT]; // peerGlobalMax
    AscendC::LocalTensor<ElementMax>   softmaxMaxUBTensor;                    // softmaxMax
    AscendC::LocalTensor<ElementMax>   localGroupMaxUBTensor[MXFP4::UB_LOCAL_GROUP_MAX_CNT]; // LocalGroupMax
    AscendC::LocalTensor<ElementMax>   localGlobalMaxUBTensor[MXFP4::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<float>        updateScaleUBTensor[MXFP4::UB_UPDATE_SCALE_CNT]; // updateScale
    AscendC::LocalTensor<ElementIndex> indexUBTensor;                        // Index

};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP4
