/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP8
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP8

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp8_vf/vf_updateScale_dn_mxfp8.h"
#include "mxfp8_vf/vf_attenOut_dn_mxfp8.h"

namespace NpuArch::Epilogue::Block {
using namespace MXFP8Kernel;
template <class ElementO_, class ElementOTmp_, class ElementDm_, class TileCopy_, class OTmpSrcPos_, LseMode LSE_MODE_,
          LseFormat LSE_FORMAT_>
class BlockEpilogue<EpilogueAtlasA5BsaRescaleOMxfp8<LSE_MODE_, LSE_FORMAT_, true>, ElementO_, ElementOTmp_, ElementDm_,
                    TileCopy_, OTmpSrcPos_> {
public:
    using DispatchPolicy = EpilogueAtlasA5BsaRescaleOMxfp8<LSE_MODE_, LSE_FORMAT_, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using ElementLse = ElementOTmp_;
    using ElementDm = ElementDm_;
    using ElementSum = ElementOTmp;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;
    using LayoutO = typename TileCopy::LayoutO;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        oTmpUBTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(MXFP8::UB_OTMP_BUF_OFFSET);
        localRowSumUBTensor = resource.ubBuf.template GetBufferByByte<ElementSum>(MXFP8::UB_LOCAL_ROW_SUM_BUF_OFFSET);
        globalRowSumUBTensor = resource.ubBuf.template GetBufferByByte<ElementSum>(MXFP8::UB_GLOBAL_ROW_SUM_BUF_OFFSET);
        oTransUBTensor = resource.ubBuf.template GetBufferByByte<ElementO>(MXFP8::UB_O_TRANS_BUF_OFFSET);
        oUBTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(MXFP8::UB_O_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP8::UB_UPDATE_SCALE_CNT; i++) {
            dmUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDm>(MXFP8::UB_UPDATE_SCALE_BUF_OFFSET +
                                                                               MXFP8::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorO>
    __aicore__ inline void operator()(TensorO &gOTensor, GemmCoord &actualOriShape, TaskInfo &taskInfo,
                                      TileInfo &tileInfo)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];
#ifndef BSA_MXFP8_DUMP
#define BSA_MXFP8_DUMP 0
#endif
#if BSA_MXFP8_DUMP
        if (tileInfo.isLastKvsTile) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::printf(
                "[MXFP8-DUMP] rescale pre blk=%u sub=%u last=%d lsum0=%f lsum50=%f lsum57=%f lsum63=%f "
                "gsum0=%f gsum50=%f gsum57=%f gsum63=%f otmp0=%f otmp50=%f otmp57=%f otmpRS50=%f otmpRS57=%f\n",
                AscendC::GetBlockIdx(), subBlockIdx, static_cast<int>(tileInfo.isLastKvsTile),
                static_cast<float>(localRowSumUBTensor.GetValue(0)),
                static_cast<float>(localRowSumUBTensor.GetValue(50)),
                static_cast<float>(localRowSumUBTensor.GetValue(57)),
                static_cast<float>(localRowSumUBTensor.GetValue(63)),
                static_cast<float>(globalRowSumUBTensor.GetValue(0)),
                static_cast<float>(globalRowSumUBTensor.GetValue(50)),
                static_cast<float>(globalRowSumUBTensor.GetValue(57)),
                static_cast<float>(globalRowSumUBTensor.GetValue(63)), static_cast<float>(oTmpUBTensor.GetValue(0)),
                static_cast<float>(oTmpUBTensor.GetValue(50)), static_cast<float>(oTmpUBTensor.GetValue(57)),
                static_cast<float>(oTmpUBTensor.GetValue(128 * 64 + 50)),
                static_cast<float>(oTmpUBTensor.GetValue(128 * 64 + 57)));
        }
#endif

        if (tileInfo.curKvsTileLoopIdx / MXFP8Kernel::TILE_GROUP_N == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_ATTN_BUF_FLAG);
            Mxfp8VF::processUpdate<false>(oUBTensor, oTmpUBTensor, dm, globalRowSumUBTensor, localRowSumUBTensor);
        } else {
            Mxfp8VF::processUpdate<true>(oUBTensor, oTmpUBTensor, dm, globalRowSumUBTensor, localRowSumUBTensor);
        }
#if BSA_MXFP8_DUMP
        if (tileInfo.isLastKvsTile) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DumpTensor(localRowSumUBTensor, 230, 64);
            AscendC::DumpTensor(globalRowSumUBTensor, 231, 64);
            AscendC::printf("[MXFP8-DUMP] rescale post blk=%u sub=%u lsum0=%f lsum50=%f lsum57=%f lsum63=%f "
                            "gsum0=%f gsum50=%f gsum57=%f gsum63=%f\n",
                            AscendC::GetBlockIdx(), subBlockIdx, static_cast<float>(localRowSumUBTensor.GetValue(0)),
                            static_cast<float>(localRowSumUBTensor.GetValue(50)),
                            static_cast<float>(localRowSumUBTensor.GetValue(57)),
                            static_cast<float>(localRowSumUBTensor.GetValue(63)),
                            static_cast<float>(globalRowSumUBTensor.GetValue(0)),
                            static_cast<float>(globalRowSumUBTensor.GetValue(50)),
                            static_cast<float>(globalRowSumUBTensor.GetValue(57)),
                            static_cast<float>(globalRowSumUBTensor.GetValue(63)));
        }
#endif

        if (tileInfo.isLastKvsTile) {
            const uint32_t actQsTile = actualOriShape.n();
            const uint32_t splitMSizePerCore = (MXFP8::QS_BASE_SIZE + 1) / 2;
            uint32_t actMSizeThisCore = actQsTile < splitMSizePerCore ? actQsTile : splitMSizePerCore;
            if (subBlockIdx != VEC0) {
                actMSizeThisCore = actQsTile < splitMSizePerCore ? 0 : (actQsTile - splitMSizePerCore);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (actMSizeThisCore != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);
                AscendC::LocalTensor<ElementOTmp> outTensorFp32Ub =
                    oTransUBTensor.template ReinterpretCast<ElementOTmp>(); // 空间复用P, bf16 vf写出是强解释fp32
                Mxfp8VF::processOut<MXFP8::QS_BASE_SIZE, ElementO>(outTensorFp32Ub, oUBTensor, globalRowSumUBTensor);
                AscendC::PipeBarrier<PIPE_V>();
#if BSA_MXFP8_DUMP
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::printf(
                    "[MXFP8-DUMP] O desc=300 blk=%u sub=%u o0=%f o50=%f o57=%f o63=%f gsum50=%f gsum57=%f\n",
                    AscendC::GetBlockIdx(), subBlockIdx, static_cast<float>(outTensorFp32Ub.GetValue(0)),
                    static_cast<float>(outTensorFp32Ub.GetValue(50)), static_cast<float>(outTensorFp32Ub.GetValue(57)),
                    static_cast<float>(outTensorFp32Ub.GetValue(63)),
                    static_cast<float>(globalRowSumUBTensor.GetValue(50)),
                    static_cast<float>(globalRowSumUBTensor.GetValue(57)));
#endif
                AscendC::LocalTensor<ElementO> oTransUBTensorB16 = oTransUBTensor.template ReinterpretCast<ElementO>();
                AscendC::LocalTensor<ElementO> oUBTensorB16 = oUBTensor.template ReinterpretCast<ElementO>();
                // embding size actualOriShape.m()
                uint32_t embedColumnCnt = EMB_ALIGN128 + MXFP8::DATA_BLOCK_BYTE / sizeof(ElementO);
                // 搬回 oUBTensorB16
                AscendC::DataCopy(oUBTensorB16, oTransUBTensorB16, actMSizeThisCore * embedColumnCnt);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_ATTN_BUF_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_ATTN_BUF_FLAG);

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
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_ATTN_BUF_FLAG);
        }
    }

private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementOTmp> oTmpUBTensor;                         // mm2Res(PV)
    AscendC::LocalTensor<ElementSum> localRowSumUBTensor;                   // LocalRowSum
    AscendC::LocalTensor<ElementSum> globalRowSumUBTensor;                  // GlobalRowSum
    AscendC::LocalTensor<ElementO> oTransUBTensor;                          // attnTrans(空间复用P)
    AscendC::LocalTensor<ElementOTmp> oUBTensor;                            // attentionOut
    AscendC::LocalTensor<ElementDm> dmUBTensor[MXFP8::UB_UPDATE_SCALE_CNT]; // updateScale
    CopyUbToGmO copyUbToGmO;
    static constexpr uint32_t VEC0 = 0;
    static constexpr uint32_t EMB_ALIGN128 = 128;
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP8
