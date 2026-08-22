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
#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp4_vf/vf_rescale_o_dn_mxfp4.h"
#include "mxfp4_vf/vf_attn_out_transpose_mxfp4.h"

namespace NpuArch::Epilogue::Block {
using namespace MXFP4Kernel;
template <class ElementO_, class ElementOTmp_, class ElementDm_, class TileCopy_, class OTmpSrcPos_, class LseType_,
          LseMode LSE_MODE_, LseFormat LSE_FORMAT_>
class BlockEpilogue<EpilogueAtlasA5BsaRescaleOMX<LSE_MODE_, LSE_FORMAT_, true>, ElementO_, ElementOTmp_, ElementDm_,
                    TileCopy_, OTmpSrcPos_, LseType_> {
public:
    using DispatchPolicy = EpilogueAtlasA5BsaRescaleOMX<LSE_MODE_, LSE_FORMAT_, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using ElementLse = typename LseType_::Element;
    using ElementDm = ElementDm_;
    using ElementSum = ElementOTmp;
    using ElementGmax = ElementOTmp;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;
    using LayoutO = typename TileCopy::LayoutO;
    using LayoutLse = typename LseType_::Layout;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        oTmpUBTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(MXFP4::UB_OTMP_BUF_OFFSET);
        localRowSumUBTensor = resource.ubBuf.template GetBufferByByte<ElementSum>(MXFP4::UB_LOCAL_ROW_SUM_BUF_OFFSET);
        globalRowSumUBTensor = resource.ubBuf.template GetBufferByByte<ElementSum>(MXFP4::UB_GLOBAL_ROW_SUM_BUF_OFFSET);
        oTransUBTensor = resource.ubBuf.template GetBufferByByte<ElementO>(MXFP4::UB_O_TRANS_BUF_OFFSET);
        oUBTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(MXFP4::UB_O_BUF_OFFSET);
        lseUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementLse>(MXFP4::UB_LSE_BUF_OFFSET);
        softmaxMaxUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP4::UB_SOFTMAX_MAX_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_UPDATE_SCALE_CNT; i++) {
            dmUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDm>(MXFP4::UB_UPDATE_SCALE_BUF_OFFSET +
                                                                               MXFP4::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorLseGm, class TensorLseUb>
    __aicore__ inline void CopyUbToGmLse(TensorLseGm const &gLseTensorTlaTile, TensorLseUb const &ubLseTensorTla)
    {
        AscendC::DataCopyExtParams repeatParams;
        if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
            (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
            repeatParams.blockCount = tla::get<0>(ubLseTensorTla.shape());
            repeatParams.blockLen = sizeof(ElementLse);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = (tla::get<0>(gLseTensorTlaTile.stride()) - 1) * sizeof(ElementLse);
        } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = tla::get<0>(ubLseTensorTla.shape()) * sizeof(ElementLse);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
        }
        auto dstOffset = gLseTensorTlaTile.layout()(gLseTensorTlaTile.coord());
        auto srcOffset = ubLseTensorTla.layout()(ubLseTensorTla.coord());
        AscendC::DataCopyPad(gLseTensorTlaTile.data()[dstOffset], ubLseTensorTla.data()[srcOffset], repeatParams);
    }

    template <class TensorO>
    __aicore__ inline void operator()(TensorO &gOTensor, AscendC::GlobalTensor<ElementLse> &gLseGm, GemmCoord &actualOriShape,
                                      TaskInfo &taskInfo, TileInfo &tileInfo)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];

        if (tileInfo.curKvsTileLoopIdx / TILE_GROUP_N == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_ATTNOUT_BUF_FLAG);
            Mxfp4VF::ProcessUpdateOCallVF<false>(oUBTensor, oTmpUBTensor, dm, globalRowSumUBTensor,
                                                 localRowSumUBTensor);
        } else {
            Mxfp4VF::ProcessUpdateOCallVF<true>(oUBTensor, oTmpUBTensor, dm, globalRowSumUBTensor, localRowSumUBTensor);
        }

        if (tileInfo.isLastKvsTile) {
            const uint32_t actQsTile = actualOriShape.n();
            const uint32_t splitMSizePerCore = (MXFP4::QS_BASE_SIZE + 1) / 2;
            uint32_t actMSizeThisCore = actQsTile < splitMSizePerCore ? actQsTile : splitMSizePerCore;
            if (subBlockIdx != VEC0) {
                actMSizeThisCore = actQsTile < splitMSizePerCore ? 0 : (actQsTile - splitMSizePerCore);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (actMSizeThisCore != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF1_FLAG);
                AscendC::LocalTensor<ElementOTmp> outTensorFp32Ub =
                    oTransUBTensor.template ReinterpretCast<ElementOTmp>(); // 空间复用P, bf16 vf写出是强解释fp32
                Mxfp4VF::TransposeAttnOutCallVF<MXFP4::QS_BASE_SIZE, ElementO>(outTensorFp32Ub, oUBTensor,
                                                                               globalRowSumUBTensor);
                AscendC::PipeBarrier<PIPE_V>();
                // LSE 输出：从 globalRowSum 计算 LSE = log(globalRowSum) 并写回 GM
                if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
                    uint32_t colNumLseUb = 0;
                    uint32_t colStrideLseUb = 0;
                    if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                        (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
                        colNumLseUb = MXFP4::DATA_BLOCK_BYTE / sizeof(ElementLse);
                        colStrideLseUb = MXFP4::DATA_BLOCK_BYTE / sizeof(ElementLse);
                        // 计算LSE
                        Mxfp4VF::ComputeLseIncontinuousCallVF(softmaxMaxUBTensor, globalRowSumUBTensor, lseUbTensor32, actMSizeThisCore);

                    } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
                        colNumLseUb = 1;
                        colStrideLseUb = 1;
                        // 计算LSE
                        Mxfp4VF::ComputeLseContinuousCallVF(softmaxMaxUBTensor, globalRowSumUBTensor, lseUbTensor32, actMSizeThisCore);
                    }

                    AscendC::PipeBarrier<PIPE_V>();
                    // ub lse tla
                    auto ubLseLayoutTla = tla::MakeLayout(
                        tla::MakeShape(actMSizeThisCore, colNumLseUb),
                        tla::MakeStride(colStrideLseUb, tla::Int<1>{}));
                    auto ubLseTensorTla = tla::MakeTensor(lseUbTensor32, ubLseLayoutTla, Arch::PositionUB{});
                    // gm lse tla：基于 taskInfo.gmOffsetLse 和 lseShapeCol 在 epilogue 内部构建
                    auto gmLseLayoutTla = tla::MakeLayout<ElementLse, LayoutLse>(
                        tla::MakeShape(actQsTile, taskInfo.lseShapeCol));
                    auto gLseTensorTla = tla::MakeTensor(gLseGm[taskInfo.gmOffsetLse], gmLseLayoutTla,
                                                          Arch::PositionGM{});
                    auto gLseTensorTlaTile = tla::GetTile(gLseTensorTla,
                                                    tla::MakeCoord(rowOffsetCurSubCore, 0),
                                                    tla::MakeShape(actMSizeThisCore, 1));
                    
                    CopyUbToGmLse(gLseTensorTlaTile, ubLseTensorTla);
                }
                AscendC::LocalTensor<ElementO> oTransUBTensorB16 = oTransUBTensor.template ReinterpretCast<ElementO>();
                AscendC::LocalTensor<ElementO> oUBTensorB16 = oUBTensor.template ReinterpretCast<ElementO>();
                uint32_t embedColumnCnt = EMB_ALIGN128 + MXFP4::DATA_BLOCK_BYTE / sizeof(ElementO);
                // 搬回 oUBTensorB16
                AscendC::DataCopy(oUBTensorB16, oTransUBTensorB16, actMSizeThisCore * embedColumnCnt);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF1_FLAG);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_ATTNOUT_BUF_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_ATTNOUT_BUF_FLAG);

                // copyUBOToGm
                uint32_t embdingSize = actualOriShape.m();
                uint32_t rowOffsetCurSubCore = subBlockIdx * splitMSizePerCore;
                uint32_t colNumCurSubCore = tla::get<1>(gOTensor.shape());
                auto gOTensorTlaTile = GetTile(gOTensor, tla::MakeCoord(rowOffsetCurSubCore, 0),
                                               tla::MakeShape(actMSizeThisCore, colNumCurSubCore));
                auto ubOLayoutTla = tla::MakeLayout(tla::MakeShape(actMSizeThisCore, embdingSize),
                                                    tla::MakeStride(embedColumnCnt, tla::Int<1>{}));
                auto ubOTensorTla = tla::MakeTensor(oUBTensorB16, ubOLayoutTla, Arch::PositionUB{});
                copyUbToGmO(gOTensorTlaTile, ubOTensorTla);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_ATTNOUT_BUF_FLAG);
        }
    }

private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementOTmp> oTmpUBTensor;                         // mm2Res(PV)
    AscendC::LocalTensor<ElementSum> localRowSumUBTensor;                   // LocalRowSum
    AscendC::LocalTensor<ElementSum> globalRowSumUBTensor;                  // GlobalRowSum
    AscendC::LocalTensor<ElementO> oTransUBTensor;                          // attnTrans(空间复用P)
    AscendC::LocalTensor<ElementOTmp> oUBTensor;                            // attentionOut
    AscendC::LocalTensor<ElementDm> dmUBTensor[MXFP4::UB_UPDATE_SCALE_CNT]; // updateScale
    AscendC::LocalTensor<ElementLse> lseUbTensor32;                        // LSE 输出
    AscendC::LocalTensor<float> softmaxMaxUBTensor; ;                              // softmaxMax
    CopyUbToGmO copyUbToGmO;
    static constexpr uint32_t VEC0 = 0;
    static constexpr uint32_t EMB_ALIGN128 = 128;
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP4
