/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP8
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP8

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp8_vf/vf_computeScale_dn_mxfp8.h"
#include "mxfp8_vf/vf_onlyUpdateScale_dn_mxfp8.h"
#include "mxfp8_vf/vf_computeScale_dn_mxfp8_qs64.h"

namespace NpuArch::Epilogue::Block {

using namespace MXFP8Kernel;

template <
    MXQuantMode MX_QUANT_MODE_,
    class ElementGroupMax_,
    class ElementDm_,
    class PScaleType_>
class BlockEpilogue<
    EpilogueComputePScaleBsaMxfp8<true, MX_QUANT_MODE_>,
    ElementGroupMax_,
    ElementDm_,
    PScaleType_>
{
public:
    using DispatchPolicy = EpilogueComputePScaleBsaMxfp8<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode MX_QUANT_MODE = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementGroupMax = ElementGroupMax_;
    using PScaleType = PScaleType_;
    using ElementPScale = typename PScaleType::Element;
    using LayoutPScale = typename PScaleType::Layout;
    using ElementDm = ElementDm_;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, ElementGroupMax LOG2_CX_CEIL) {
        for (uint32_t i = 0; i < MXFP8::UB_P_SCALE_CNT; i++) {
            pscaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementPScale>(
                MXFP8::UB_P_SCALE_BUF_OFFSET + MXFP8::UB_P_SCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::UB_PEER_GLOBAL_MAX_CNT; i++) {
            peerGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP8::UB_PEER_GLOBAL_MAX_BUF_OFFSET + MXFP8::UB_PEER_GLOBAL_MAX_BUF_SIZE * i);
        }
        softmaxMaxUBTensor = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(MXFP8::UB_SOFTMAX_MAX_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP8::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP8::UB_LOCAL_GROUP_MAX_BUF_OFFSET + MXFP8::UB_LOCAL_GROUP_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP8::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP8::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::UB_UPDATE_SCALE_CNT; i++) {
            dmUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDm>(
                MXFP8::UB_UPDATE_SCALE_BUF_OFFSET + MXFP8::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
        LOG2_CX_CEIL_ = LOG2_CX_CEIL;
    }
    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorL1Pscale>
    __aicore__ inline void operator()(TensorL1Pscale &l1PScaleTensor, TileInfo const &tileInfo, TaskInfo &taskInfo)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementGroupMax> localGlobalMax = localGlobalMaxUBTensor[tileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementGroupMax> peerGlobalMax = peerGlobalMaxUBTensor[tileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementGroupMax> softmaxMaxOld = softmaxMaxUBTensor;
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];

        // 如果当前s2只有一个softmax，就另外一个核不用做pscale操作
        if (tileInfo.kvsFirstTileStartVecCore != subBlockIdx && tileInfo.isTileGoupFirstTile) {
            if (tileInfo.curKvsTileLoopIdx / MXFP8Kernel::TILE_GROUP_N == 0) {
                Mxfp8VF::computeOnlyScale<true, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    localGlobalMax, peerGlobalMax, softmaxMaxOld, dm, static_cast<uint16_t>(subBlockIdx));
            } else {
                Mxfp8VF::computeOnlyScale<false, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    localGlobalMax, peerGlobalMax, softmaxMaxOld, dm, static_cast<uint16_t>(subBlockIdx));
            }
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        

        uint16_t firstLoop = 0;
        uint16_t secondLoop = 0;
        uint16_t firstLoopStart = 0;
        uint16_t secondLoopStart = 0;
        GetPScaleParams(tileInfo, subBlockIdx, firstLoopStart, firstLoop, secondLoopStart, secondLoop);
#ifndef BSA_MXFP8_DUMP
#define BSA_MXFP8_DUMP 0
#endif
        if (firstLoop == 0 && secondLoop == 0) {
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        AscendC::LocalTensor<ElementPScale> pscale1 = pscaleUBTensor[0];
        AscendC::LocalTensor<ElementPScale> pscale2 = pscaleUBTensor[firstLoop];
        AscendC::LocalTensor<ElementGroupMax> localGroupMax1 = localGroupMaxUBTensor[firstLoopStart];
        AscendC::LocalTensor<ElementGroupMax> localGroupMax2 = localGroupMaxUBTensor[secondLoopStart];

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);

#if BSA_MXFP8_DUMP
        if (tileInfo.isLastKvsTile && tileInfo.curKvsTileLoopIdx < 4) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::printf("[MXFP8-DUMP] pscale-pre localGmax=%f peerGmax=%f grpMax=%f sub=%u tileMax=%u\n",
                static_cast<float>(localGlobalMax.GetValue(0)),
                static_cast<float>(peerGlobalMax.GetValue(0)),
                static_cast<float>(localGroupMax1.GetValue(0)),
                subBlockIdx, tileInfo.tileMaxIdx);
        }
#endif

        if (tileInfo.curKvsTileLoopIdx / MXFP8Kernel::TILE_GROUP_N == 0) {
            if (taskInfo.qsActBaseTileAlign64 == MXFP8::QS_BASE_SIZE) {
                Mxfp8VF::computePscale<MX_QUANT_MODE, true, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    pscale1, pscale2, localGroupMax1, localGroupMax2,
                    localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                    firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx), LOG2_CX_CEIL_);
            } else {
                // AscendC::PRINTF("lwg first 64\n");
                Mxfp8VF::computePscaleQS64CallVF<MX_QUANT_MODE, true, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    pscale1, pscale2, localGroupMax1, localGroupMax2,
                    localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                    firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx), LOG2_CX_CEIL_);
            }
        } else {
            if (taskInfo.qsActBaseTileAlign64 == MXFP8::QS_BASE_SIZE) {
                Mxfp8VF::computePscale<MX_QUANT_MODE, false, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    pscale1, pscale2, localGroupMax1, localGroupMax2,
                    localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                    firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx), LOG2_CX_CEIL_);
            } else {
                // AscendC::PRINTF("lwg first222 64 \n");

                Mxfp8VF::computePscaleQS64CallVF<MX_QUANT_MODE, false, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    pscale1, pscale2, localGroupMax1, localGroupMax2,
                    localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                    firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx), LOG2_CX_CEIL_);
            }
        }
        // AscendC::printf("localGroupMax: \n");
        // for (int i =0; i < 8; i++) {
        //     AscendC::DumpTensor(localGroupMax1[128*i], 128, 128);
        // }
        //  AscendC::printf("localGroupMax end : \n \n");

        //          AscendC::printf("pscale1: \n");
        // for (int i =0; i < 8; i++) {
        //     AscendC::DumpTensor(pscale1[32*5*i], 128, 32*4);
        // }
        //  AscendC::printf("pscale1 end : \n \n");
        
        //  AscendC::printf("softmaxMaxOld: \n");
        // for (int i =0; i < 1; i++) {
        //     AscendC::DumpTensor(softmaxMaxOld[128*i], 128, 128);
        //         }



#ifndef BSA_MXFP8_DUMP
#define BSA_MXFP8_DUMP 0
#endif
#if BSA_MXFP8_DUMP
        if (tileInfo.isLastKvsTile && tileInfo.curKvsTileLoopIdx < 4) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DumpTensor(pscale1, 212, 32);
            AscendC::DumpTensor(localGlobalMax, 214, 32);
            AscendC::DumpTensor(peerGlobalMax, 215, 32);
            AscendC::printf("[MXFP8-DUMP] pscale desc=212 p0=%u firstLoop=%u firstStart=%u secondLoop=%u "
                "sub=%u kvsFirst=%u loop=%u curS2=%u localGmax=%f peerGmax=%f grpMax=%f\n",
                static_cast<uint32_t>(pscale1.GetValue(0)), firstLoop, firstLoopStart, secondLoop,
                subBlockIdx, tileInfo.kvsFirstTileStartVecCore,
                static_cast<uint32_t>(tileInfo.loop), tileInfo.curKvsTileLoopIdx,
                static_cast<float>(localGlobalMax.GetValue(0)),
                static_cast<float>(peerGlobalMax.GetValue(0)),
                static_cast<float>(localGroupMax1.GetValue(0)));
        }
#endif

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);

        
        // DataCopy(l1PScaleTensor.data()[subBlockIdx*(32*5*8)+firstLoopStart*(32*5*8)], pscale1, {static_cast<uint16_t>(firstLoop), 40, 0, 40});
        // if (secondLoop != 0) {
        //     DataCopy(l1PScaleTensor.data()[subBlockIdx*(32*5*8)+secondLoopStart*(32*5*8)], pscale1[firstLoop*(32*5*8)], {static_cast<uint16_t>(secondLoop), 40, 0, 40});
        // }

        // copy pscale1 ub to l1
        bool isQsAlign128 = taskInfo.qsActBaseTileAlign64 == MXFP8::QS_BASE_SIZE;
        CopyPScaleUbToL1(l1PScaleTensor, pscale1, pscale2, subBlockIdx, firstLoopStart, firstLoop, secondLoopStart, secondLoop, isQsAlign128);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);
    }

     template <class TensorDst>
    __aicore__ inline void CopyPScaleUbToL1(TensorDst &l1PScaleTensor, AscendC::LocalTensor<ElementPScale> &pscale1Src, AscendC::LocalTensor<ElementPScale> &pscale2Src, 
                                                uint32_t subBlockIdx, uint16_t firstLoopStart, uint16_t firstLoop, uint16_t secondLoopStart, uint16_t secondLoop, bool isQsAlign128) {
        // 第一部分: firstLoop 组, 从 firstLoopStart 开始
        uint32_t l1PScaleTensorNSize = tla::get<1>(l1PScaleTensor.shape());
        // uint32_t l1PScaleTensorMSize = tla::get<0>(l1PScaleTensor.shape());
        auto l1TlaTile1 = tla::GetTile(l1PScaleTensor,
            tla::MakeCoord(subBlockIdx + firstLoopStart, 0),
            tla::MakeShape(firstLoop, l1PScaleTensorNSize));
        uint32_t dstOffset = l1TlaTile1.layout()(l1TlaTile1.coord());
        uint16_t pscaleGroupLen = static_cast<uint16_t>(l1PScaleTensorNSize / MXFP8::DATA_BLOCK_BYTE);
        uint16_t pscaleGroupHalfLen = static_cast<uint16_t>(l1PScaleTensorNSize / 2 / MXFP8::DATA_BLOCK_BYTE);
        AscendC::DataCopyParams repeatParams1;
        repeatParams1.blockCount = firstLoop;
        repeatParams1.blockLen = isQsAlign128 ? pscaleGroupLen : pscaleGroupHalfLen;
        repeatParams1.srcStride = isQsAlign128 ? 0 : pscaleGroupHalfLen;
        repeatParams1.dstStride = isQsAlign128 ? pscaleGroupLen : (pscaleGroupHalfLen + pscaleGroupLen);


        // AscendC::printf("==== CopyPScaleUbToL1 debug ====\n");
        // AscendC::printf("subBlockIdx:%d, firstLoopStart:%d, firstLoop:%d, secondLoopStart:%d, secondLoop:%d\n",
        //     subBlockIdx, firstLoopStart, firstLoop, secondLoopStart, secondLoop);
        // AscendC::printf("l1PScaleTensor shape(M,N):(%d,%d), l1PScaleTensorNSize:%d\n", l1PScaleTensorMSize, l1PScaleTensorNSize, l1PScaleTensorNSize);
        // AscendC::printf("DATA_BLOCK_BYTE:%d, pscaleGroupLen:%d\n", MXFP8::DATA_BLOCK_BYTE, pscaleGroupLen);
        // AscendC::printf("l1TlaTile1 coord(M,N):(%d,%d), shape(M,N):(%d,%d)\n",
        //     tla::get<0>(l1TlaTile1.coord()), tla::get<1>(l1TlaTile1.coord()),
        //     tla::get<0>(l1TlaTile1.shape()), tla::get<1>(l1TlaTile1.shape()));
        // AscendC::printf("dstOffset:%d\n", dstOffset);
        // AscendC::printf("repeatParams1: blockCount:%d, blockLen:%d, srcStride:%d, dstStride:%d\n",
        //     repeatParams1.blockCount, repeatParams1.blockLen, repeatParams1.srcStride, repeatParams1.dstStride);

        AscendC::DataCopy(l1TlaTile1.data()[dstOffset], pscale1Src, repeatParams1);

        if (secondLoop != 0) {
            // 第二部分: secondLoop 组, 从 secondLoopStart 开始
            auto l1TlaTile2 = tla::GetTile(l1PScaleTensor,
                tla::MakeCoord(subBlockIdx + secondLoopStart, 0),
                tla::MakeShape(secondLoop, l1PScaleTensorNSize));
            uint32_t dstOffset2 = l1TlaTile2.layout()(l1TlaTile2.coord());
            AscendC::DataCopyParams repeatParams2;
            repeatParams2.blockCount = secondLoop;
            repeatParams2.blockLen = isQsAlign128 ? pscaleGroupLen : pscaleGroupHalfLen;
            repeatParams2.srcStride = isQsAlign128 ? 0 : pscaleGroupHalfLen;
            repeatParams2.dstStride = isQsAlign128 ? pscaleGroupLen : (pscaleGroupHalfLen + pscaleGroupLen);
            // repeatParams2.blockCount = secondLoop;
            // repeatParams2.blockLen = static_cast<uint16_t>(pscaleGroupLen);
            // repeatParams2.srcStride = 0;
            // repeatParams2.dstStride = static_cast<uint16_t>(pscaleGroupLen);

            // AscendC::printf("---- secondLoop branch ----\n");
            // AscendC::printf("l1TlaTile2 coord(M,N):(%d,%d), shape(M,N):(%d,%d)\n",
            //     tla::get<0>(l1TlaTile2.coord()), tla::get<1>(l1TlaTile2.coord()),
            //     tla::get<0>(l1TlaTile2.shape()), tla::get<1>(l1TlaTile2.shape()));
            // AscendC::printf("dstOffset2:%d\n", dstOffset2);
            // AscendC::printf("repeatParams2: blockCount:%d, blockLen:%d, srcStride:%d, dstStride:%d\n",
            //     repeatParams2.blockCount, repeatParams2.blockLen, repeatParams2.srcStride, repeatParams2.dstStride);
            // AscendC::printf("==== CopyPScaleUbToL1 debug end ====\n");

            AscendC::DataCopy(l1TlaTile2.data()[dstOffset2], pscale2Src, repeatParams2);
        }
    }

private:
    __aicore__ inline void GetPScaleParams(const TileInfo &tileInfo, uint32_t subBlockIdx,
                                           uint16_t &firstLoopStart, uint16_t &firstLoop, uint16_t &secondLoopStart,
                                           uint16_t &secondLoop)
    {
        uint16_t isLoopFirstTaskVecCore = (subBlockIdx == tileInfo.kvsFirstTileStartVecCore);
        uint32_t loop = tileInfo.loop;
        uint32_t curS2LoopIdx = tileInfo.curKvsTileLoopIdx;
        // 对照 mxfp4：单核目前为止第几个 softmax（跨 batch）
        uint32_t groupMaxEndLoop = (loop >> 1) - (subBlockIdx != 0 && loop % 2 == 0);
        // kvs128：1 基块 = 1 槽。mxfp4 是 *2+1（2 槽/基块的末槽）；这里末槽就是 groupMaxEndLoop。
        uint32_t groupMaxEndIdx = groupMaxEndLoop % GROUP_MAX_SPACE_LEN;

        // 对照 mxfp4：两个 AIV 交替 softmax，每核只处理自己做过的 softmax。
        // mxfp4 用 isLoopFirst + 偶对齐 + /2 把「2 槽单位」折成 softmax 数；
        // kvs128 一槽一 softmax，直接 ceil/floor(组内 tile 数 / 2)。
        uint16_t tilesInGroup = static_cast<uint16_t>(
            (curS2LoopIdx + 1) - (curS2LoopIdx / WHOLE_PROCESS_LOOP) * WHOLE_PROCESS_LOOP);
        if (tilesInGroup == 0) {
            tilesInGroup = WHOLE_PROCESS_LOOP;
        }
        uint16_t toProcessGroupMaxLoopLen = isLoopFirstTaskVecCore
            ? static_cast<uint16_t>((tilesInGroup + 1) / 2)
            : static_cast<uint16_t>(tilesInGroup / 2);
        if (toProcessGroupMaxLoopLen > WHOLE_PROCESS_LOOP) {
            toProcessGroupMaxLoopLen = WHOLE_PROCESS_LOOP;
        }
        firstLoop = toProcessGroupMaxLoopLen;
        secondLoop = 0;
        int32_t firstLoopStartNotU = static_cast<int32_t>(groupMaxEndIdx) - static_cast<int32_t>(toProcessGroupMaxLoopLen) + 1;
        if (firstLoopStartNotU < 0) {
            firstLoopStart = static_cast<uint16_t>(firstLoopStartNotU + GROUP_MAX_SPACE_LEN);
            firstLoop = static_cast<uint16_t>(GROUP_MAX_SPACE_LEN - firstLoopStart);
            secondLoop = static_cast<uint16_t>(toProcessGroupMaxLoopLen - firstLoop);
            secondLoopStart = 0;
        } else {
            firstLoopStart = static_cast<uint16_t>(firstLoopStartNotU);
            secondLoopStart = static_cast<uint16_t>(firstLoopStart + firstLoop);
        }
    }

private:
    static constexpr uint32_t GROUP_MAX_SPACE_LEN = 12; // UB_LOCAL_GROUP_MAX_CNT
    static constexpr uint32_t WHOLE_PROCESS_LOOP = 8;   // TILE_GROUP_N

    AscendC::LocalTensor<ElementPScale>   pscaleUBTensor[MXFP8::UB_P_SCALE_CNT]; // pScale(复用P)
    AscendC::LocalTensor<ElementGroupMax> peerGlobalMaxUBTensor[MXFP8::UB_PEER_GLOBAL_MAX_CNT]; // peerGlobalMax
    AscendC::LocalTensor<ElementGroupMax> softmaxMaxUBTensor;                    // softmaxMax
    AscendC::LocalTensor<ElementGroupMax> localGroupMaxUBTensor[MXFP8::UB_LOCAL_GROUP_MAX_CNT]; // LocalGroupMax
    AscendC::LocalTensor<ElementGroupMax> localGlobalMaxUBTensor[MXFP8::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<ElementDm>       dmUBTensor[MXFP8::UB_UPDATE_SCALE_CNT]; // updateScale
    ElementGroupMax                       LOG2_CX_CEIL_;
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP8
