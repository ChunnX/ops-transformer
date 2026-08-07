/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP4
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP4

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp4_vf/vf_computeScale_dn_mxfp4.h"
#include "mxfp4_vf/vf_onlyUpdateScale_dn_mxfp4.h"

namespace NpuArch::Epilogue::Block {

using namespace MXFP4Kernel;

template <
    MXQuantMode MX_QUANT_MODE_,
    class ElementGroupMax_,
    class ElementDm_,
    class PScaleType_>
class BlockEpilogue<
    EpilogueComputePScaleBsa<true, MX_QUANT_MODE_>,
    ElementGroupMax_,
    ElementDm_,
    PScaleType_>
{
public:
    using DispatchPolicy = EpilogueComputePScaleBsa<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode MX_QUANT_MODE = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementGroupMax = ElementGroupMax_;
    using PScaleType = PScaleType_;
    using ElementPScale = typename PScaleType::Element;
    using LayoutPScale = typename PScaleType::Layout;
    using ElementDm = ElementDm_;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, ElementGroupMax LOG2_CX_CEIL) {
        for (uint32_t i = 0; i < MXFP4::UB_P_SCALE_CNT; i++) {
            pscaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementPScale>(
                MXFP4::UB_P_SCALE_BUF_OFFSET + MXFP4::UB_P_SCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_PEER_GLOBAL_MAX_CNT; i++) {
            peerGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP4::UB_PEER_GLOBAL_MAX_BUF_OFFSET + MXFP4::UB_PEER_GLOBAL_MAX_BUF_SIZE * i);
        }
        softmaxMaxUBTensor = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(MXFP4::UB_SOFTMAX_MAX_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP4::UB_LOCAL_GROUP_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GROUP_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_UPDATE_SCALE_CNT; i++) {
            dmUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDm>(
                MXFP4::UB_UPDATE_SCALE_BUF_OFFSET + MXFP4::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
        LOG2_CX_CEIL_ = LOG2_CX_CEIL;
    }
    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorL1Pscale>
    __aicore__ inline void operator()(TensorL1Pscale &l1PScaleTensor, TileInfo const &tileInfo)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementGroupMax> localGlobalMax = localGlobalMaxUBTensor[tileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementGroupMax> peerGlobalMax = peerGlobalMaxUBTensor[tileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementGroupMax> softmaxMaxOld = softmaxMaxUBTensor;
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];

        // 如果当前s2只有一个softmax，就另外一个核不用做pscale操作
        if (tileInfo.kvsFirstTileStartVecCore != subBlockIdx && tileInfo.isTileGoupFirstTile) {
            if (tileInfo.curKvsTileLoopIdx / TILE_GROUP_N == 0) {
                Mxfp4VF::computeOnlyScale<true, ElementGroupMax, MXFP4::QS_BASE_SIZE>(
                    localGlobalMax, peerGlobalMax, softmaxMaxOld, dm, static_cast<uint16_t>(subBlockIdx));
            } else {
                Mxfp4VF::computeOnlyScale<false, ElementGroupMax, MXFP4::QS_BASE_SIZE>(
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
//         AscendC::PRINTF("firstLoopStart = %u, firstLoop = %u\n", firstLoopStart, firstLoop);
// AscendC::PRINTF("secondLoopStart = %u, secondLoop = %u\n", secondLoopStart, secondLoop);

        AscendC::LocalTensor<ElementPScale> pscale1 = pscaleUBTensor[0];
        AscendC::LocalTensor<ElementPScale> pscale2 = pscaleUBTensor[firstLoop];
        AscendC::LocalTensor<ElementGroupMax> localGroupMax1 = localGroupMaxUBTensor[firstLoopStart];
        AscendC::LocalTensor<ElementGroupMax> localGroupMax2 = localGroupMaxUBTensor[secondLoopStart];

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_VEC1_RES_BUF1_FLAG);

        if (tileInfo.curKvsTileLoopIdx / TILE_GROUP_N == 0) {
            Mxfp4VF::computePscale<MX_QUANT_MODE, true, ElementGroupMax, MXFP4::QS_BASE_SIZE>(
                pscale1, pscale2, localGroupMax1, localGroupMax2,
                localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx), LOG2_CX_CEIL_);
        } else {
            Mxfp4VF::computePscale<MX_QUANT_MODE, false, ElementGroupMax, MXFP4::QS_BASE_SIZE>(
                pscale1, pscale2, localGroupMax1, localGroupMax2,
                localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx), LOG2_CX_CEIL_);
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
        // }



        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_VEC1_RES_BUF1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_VEC1_RES_BUF1_FLAG);

        
        // DataCopy(l1PScaleTensor.data()[subBlockIdx*(32*5*8)+firstLoopStart*(32*5*8)], pscale1, {static_cast<uint16_t>(firstLoop), 40, 0, 40});
        // if (secondLoop != 0) {
        //     DataCopy(l1PScaleTensor.data()[subBlockIdx*(32*5*8)+secondLoopStart*(32*5*8)], pscale1[firstLoop*(32*5*8)], {static_cast<uint16_t>(secondLoop), 40, 0, 40});
        // }

        // copy pscale1 ub to l1
        CopyPScaleUbToL1(l1PScaleTensor, pscale1, pscale2, subBlockIdx, firstLoopStart, firstLoop, secondLoopStart, secondLoop);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_VEC1_RES_BUF1_FLAG);
    }

     template <class TensorDst>
    __aicore__ inline void CopyPScaleUbToL1(TensorDst &l1PScaleTensor, AscendC::LocalTensor<ElementPScale> &pscale1Src, AscendC::LocalTensor<ElementPScale> &pscale2Src, 
                                                uint32_t subBlockIdx, uint16_t firstLoopStart, uint16_t firstLoop, uint16_t secondLoopStart, uint16_t secondLoop) {
        // 第一部分: firstLoop 组, 从 firstLoopStart 开始
        uint32_t l1PScaleTensorNSize = tla::get<1>(l1PScaleTensor.shape());
        // uint32_t l1PScaleTensorMSize = tla::get<0>(l1PScaleTensor.shape());
        auto l1TlaTile1 = tla::GetTile(l1PScaleTensor,
            tla::MakeCoord(subBlockIdx + firstLoopStart, 0),
            tla::MakeShape(firstLoop, l1PScaleTensorNSize));
        uint32_t dstOffset = l1TlaTile1.layout()(l1TlaTile1.coord());
        uint32_t pscaleGroupLen = l1PScaleTensorNSize / MXFP4::DATA_BLOCK_BYTE;
        AscendC::DataCopyParams repeatParams1;
        repeatParams1.blockCount = firstLoop;
        repeatParams1.blockLen = static_cast<uint16_t>(pscaleGroupLen);
        repeatParams1.srcStride = 0;
        repeatParams1.dstStride = static_cast<uint16_t>(pscaleGroupLen);

        // AscendC::printf("==== CopyPScaleUbToL1 debug ====\n");
        // AscendC::printf("subBlockIdx:%d, firstLoopStart:%d, firstLoop:%d, secondLoopStart:%d, secondLoop:%d\n",
        //     subBlockIdx, firstLoopStart, firstLoop, secondLoopStart, secondLoop);
        // AscendC::printf("l1PScaleTensor shape(M,N):(%d,%d), l1PScaleTensorNSize:%d\n", l1PScaleTensorMSize, l1PScaleTensorNSize, l1PScaleTensorNSize);
        // AscendC::printf("DATA_BLOCK_BYTE:%d, pscaleGroupLen:%d\n", MXFP4::DATA_BLOCK_BYTE, pscaleGroupLen);
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
            repeatParams2.blockLen = static_cast<uint16_t>(pscaleGroupLen);
            repeatParams2.srcStride = 0;
            repeatParams2.dstStride = static_cast<uint16_t>(pscaleGroupLen);

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
        // 求单核目前为止，计算对应的第几个softmax(即第几个groupmax 8*128)（跨batch）
        // 等价groupMaxEndLoop = (loop % 2 == 0) ? ((subBlockIdx == 0) ? (loop / 2) : ((loop / 2) - 1)) : (loop / 2);
        uint32_t groupMaxEndLoop = (loop >> 1) - (subBlockIdx != 0 && loop % 2 == 0);
        // 这一轮pscale的任务loop在groupMax空间下发的groupMaxEndIdx，*2的原因是因为groupmax一片空间承载4*128
        uint32_t groupMaxEndIdx = groupMaxEndLoop * 2 % GROUP_MAX_SPACE_LEN + 1;

        // toProcessGroupMaxLoopLen 求当前FD切核对应的Tile(16个softmax是一个tile)在每个核上有占用几片groupmax(4*128)空间
        uint16_t toProcessGroupMaxLoopLen =
            (curS2LoopIdx + 1) - (curS2LoopIdx / WHOLE_PROCESS_LOOP) * WHOLE_PROCESS_LOOP + isLoopFirstTaskVecCore;
        toProcessGroupMaxLoopLen = toProcessGroupMaxLoopLen / 2 * 2;
        toProcessGroupMaxLoopLen = toProcessGroupMaxLoopLen < WHOLE_PROCESS_LOOP ? toProcessGroupMaxLoopLen :
                                                                                   WHOLE_PROCESS_LOOP;
        firstLoop = (toProcessGroupMaxLoopLen / 2 + 1) / 2 * 2;
        secondLoop = toProcessGroupMaxLoopLen - firstLoop;
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
        // 两片groupmax空间对应其中的一个softmax任务
        firstLoop /= 2;
        secondLoop /= 2;
    }

private:
    static constexpr uint32_t GROUP_MAX_SPACE_LEN = 20;
    static constexpr uint32_t WHOLE_PROCESS_LOOP = 16;

    AscendC::LocalTensor<ElementPScale>   pscaleUBTensor[MXFP4::UB_P_SCALE_CNT]; // pScale(复用P)
    AscendC::LocalTensor<ElementGroupMax> peerGlobalMaxUBTensor[MXFP4::UB_PEER_GLOBAL_MAX_CNT]; // peerGlobalMax
    AscendC::LocalTensor<ElementGroupMax> softmaxMaxUBTensor;                    // softmaxMax
    AscendC::LocalTensor<ElementGroupMax> localGroupMaxUBTensor[MXFP4::UB_LOCAL_GROUP_MAX_CNT]; // LocalGroupMax
    AscendC::LocalTensor<ElementGroupMax> localGlobalMaxUBTensor[MXFP4::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<ElementDm>       dmUBTensor[MXFP4::UB_UPDATE_SCALE_CNT]; // updateScale
    ElementGroupMax                       LOG2_CX_CEIL_;
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP4
