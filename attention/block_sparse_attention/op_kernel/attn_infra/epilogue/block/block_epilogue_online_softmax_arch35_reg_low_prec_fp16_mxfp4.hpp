/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp4_vf/vf_nd2nz_indexes_dn_mxfp4.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_qs128_kvs256.h"
#include "mxfp4_vf/vf_mm1_res_pre_padding_align_kvs32_multi.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_align_qs128_kvs32_multi.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_align_qs128_kvs32.h"


namespace NpuArch::Epilogue::Block {

using namespace MXFP4Kernel;

template <
    MXQuantMode MX_QUANT_MODE_,
    class OutputType_,
    class LayoutS_,
    class ScaleType_>
class BlockEpilogue<
    EpilogueOnlineSoftmaxBsa<true, MX_QUANT_MODE_>,
    OutputType_,
    Gemm::GemmType<half, LayoutS_>,
    ScaleType_>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode MX_QUANT_MODE = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element; // P
    using ElementInput = half; // S
    using ElementMax = ElementInput;
    using ElementDisguiseP = uint8_t;
    using ElementPScale = typename ScaleType_::Element;
    using ElementIndex = uint8_t;
    using LayoutPL1 = typename OutputType_::Layout;
    using LayoutPUB = layout::RowMajor;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue, ElementInput NEG_LOG2_CX)
    {
        // FIXPIPE<->V 区
        for (uint32_t i = 0; i < MXFP4::UB_S_BUF_CNT; i++) {
            sUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(
                MXFP4::UB_S_BUF_OFFSET + MXFP4::UB_S_INNER_BUF_OFFSET * i);
        }
        oTmpUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP4::UB_OTMP_BUF_OFFSET);
        localRowSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP4::UB_LOCAL_ROW_SUM_BUF_OFFSET);
        globalRowSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP4::UB_GLOBAL_ROW_SUM_BUF_OFFSET);

        // 输出缓冲区
        for (uint32_t i = 0; i < MXFP4::UB_P_BUF_CNT; i++) {
            pUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDisguiseP>(
                MXFP4::UB_P_BUF_OFFSET + MXFP4::UB_P_INNER_BUF_OFFSET * i);
        }
        oTransUBTensor = resource.ubBuf.template GetBufferByByte<bfloat16_t>(MXFP4::UB_O_TRANS_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_P_SCALE_CNT; i++) {
            pscaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementPScale>(
                MXFP4::UB_P_SCALE_BUF_OFFSET + MXFP4::UB_P_SCALE_BUF_SIZE * i);
        }
        oUBTensor = resource.ubBuf.template GetBufferByByte<bfloat16_t>(MXFP4::UB_O_BUF_OFFSET);

        // L1 <-> UB
        for (uint32_t i = 0; i < MXFP4::UB_PEER_GLOBAL_MAX_CNT; i++) {
            peerGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_PEER_GLOBAL_MAX_BUF_OFFSET + MXFP4::UB_PEER_GLOBAL_MAX_BUF_SIZE * i);
        }

        // 常驻 buffer
        softmaxMaxUBTensor = resource.ubBuf.template GetBufferByByte<ElementMax>(MXFP4::UB_SOFTMAX_MAX_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GROUP_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GROUP_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_UPDATE_SCALE_CNT; i++) {
            updateScaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                MXFP4::UB_UPDATE_SCALE_BUF_OFFSET + MXFP4::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
        indexUBTensor = resource.ubBuf.template GetBufferByByte<ElementIndex>(MXFP4::UB_INDEX_BUF_OFFSET);

        NEG_LOG2_CX_ = NEG_LOG2_CX;
        scaleValue_ = static_cast<ElementInput>(scaleValue);
        //  Init Tensor
        Mxfp4VF::InitIndexesAndDuplicateCallVF<ElementMax>(indexUBTensor, localGlobalMaxUBTensor[0]);
    }
    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorL1P>
    __aicore__ inline
    void operator()(TensorL1P &l1PTensorTla, GemmCoord &actualBlockShape, TileInfo const &tileInfo)
    {
        uint32_t spBuffIdx = GetSPBufferIdx(tileInfo.loop);
        // 跨 tile 同步: 每个 vec core 都需要一个跨 tile 的同步
        if (tileInfo.isKvsFirstTilePerCore) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                MXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);
    

        const uint32_t actKvsTile = actualBlockShape.m();
        const uint32_t actQsTile = actualBlockShape.n();
        const uint32_t groupMaxIdx = GetLocalGroupMaxBufIdx(tileInfo.loop);
        //     AscendC::PRINTF("mm1Res \n");
        // AscendC::DumpTensor(sUBTensor[spBuffIdx][0], 127, 128);
        // AscendC::DumpTensor(sUBTensor[spBuffIdx][256], 127, 128);
        //         AscendC::DumpTensor(sUBTensor[spBuffIdx][512], 127, 128);
        // AscendC::DumpTensor(sUBTensor[spBuffIdx][768], 127, 128);

        if (tileInfo.isFirstKvsTile) {
            if (actQsTile == MXFP4::QS_BASE_SIZE && actKvsTile == MXFP4::KVS_BASE_SIZE) {
                // qs=128, kvs=256 softmax
                Mxfp4VF::SoftmaxWithGroupMaxQs128Kvs256CallVF<MX_QUANT_MODE, true, ElementInput, ElementDisguiseP,
                    MXFP4::KVS_BASE_SIZE, MXFP4::QS_BASE_SIZE>(
                    pUBTensor[spBuffIdx], sUBTensor[spBuffIdx],
                    localGroupMaxUBTensor[groupMaxIdx], localGlobalMaxUBTensor[tileInfo.tileMaxIdx],
                    indexUBTensor, scaleValue_, NEG_LOG2_CX_);
            } else {
                // kvs padding 32 multi
                if (actKvsTile != tileInfo.kvsActBaseTileAlign32) {
                    Mxfp4VF::Mm1ResPrePaddingAlignKvs32MultiCallVF<ElementInput>(
                        sUBTensor[spBuffIdx], static_cast<uint16_t>(actKvsTile),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                // softmax
                if (tileInfo.kvsActBaseTileAlign32 == MXFP4::DATA_BLOCK_BYTE) {
                    // softmax padding 32
                    Mxfp4VF::SoftmaxWithGroupMaxAlignQs128Kvs32CallVF<MX_QUANT_MODE, true, ElementInput, ElementDisguiseP>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx],
                        localGroupMaxUBTensor[groupMaxIdx], localGlobalMaxUBTensor[tileInfo.tileMaxIdx],
                        indexUBTensor, scaleValue_, NEG_LOG2_CX_);
                } else {
                    // softmax padding 32 multi >= 64
                    Mxfp4VF::SoftmaxWithGroupMaxAlignQs128Kvs32MultiCallVF<MX_QUANT_MODE, true, ElementInput, ElementDisguiseP,
                        MXFP4::QS_BASE_SIZE>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx],
                        localGroupMaxUBTensor[groupMaxIdx], localGlobalMaxUBTensor[tileInfo.tileMaxIdx],
                        indexUBTensor, scaleValue_, NEG_LOG2_CX_,
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32), static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign64));
                }
            }
        } else {
            if (actQsTile == MXFP4::QS_BASE_SIZE && actKvsTile == MXFP4::KVS_BASE_SIZE) {
                // qs=128, kvs=256 softmax
                Mxfp4VF::SoftmaxWithGroupMaxQs128Kvs256CallVF<MX_QUANT_MODE, false, ElementInput, ElementDisguiseP,
                    MXFP4::KVS_BASE_SIZE, MXFP4::QS_BASE_SIZE>(
                    pUBTensor[spBuffIdx], sUBTensor[spBuffIdx],
                    localGroupMaxUBTensor[groupMaxIdx], localGlobalMaxUBTensor[tileInfo.tileMaxIdx],
                    indexUBTensor, scaleValue_, NEG_LOG2_CX_);
            } else {
                // kvs padding 32 multi
                if (actKvsTile != tileInfo.kvsActBaseTileAlign32) {
                    Mxfp4VF::Mm1ResPrePaddingAlignKvs32MultiCallVF<ElementInput>(
                        sUBTensor[spBuffIdx], static_cast<uint16_t>(actKvsTile),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                // softmax
                if (tileInfo.kvsActBaseTileAlign32 == MXFP4::DATA_BLOCK_BYTE) {
                    // softmax padding 32
                    Mxfp4VF::SoftmaxWithGroupMaxAlignQs128Kvs32CallVF<MX_QUANT_MODE, false, ElementInput, ElementDisguiseP>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx],
                        localGroupMaxUBTensor[groupMaxIdx], localGlobalMaxUBTensor[tileInfo.tileMaxIdx],
                        indexUBTensor, scaleValue_, NEG_LOG2_CX_);
                } else {
                    // softmax padding 32 multi >= 64
                    Mxfp4VF::SoftmaxWithGroupMaxAlignQs128Kvs32MultiCallVF<MX_QUANT_MODE, false, ElementInput, ElementDisguiseP,
                        MXFP4::QS_BASE_SIZE>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx],
                        localGroupMaxUBTensor[groupMaxIdx], localGlobalMaxUBTensor[tileInfo.tileMaxIdx],
                        indexUBTensor, scaleValue_, NEG_LOG2_CX_,
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32), static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign64));
                }
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);

        // DataCopy UB -> L1 搬运
        auto ubPLayoutTla = tla::MakeLayout<ElementDisguiseP, LayoutPUB>(MXFP4::KVS_BASE_SIZE / 4, MXFP4::QS_BASE_SIZE * 4);
        auto ubPTensorTla = tla::MakeTensor(pUBTensor[0], ubPLayoutTla, Arch::PositionUB{});

        CopyPUBToPL1(l1PTensorTla, ubPTensorTla, tileInfo, actQsTile, spBuffIdx);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline
    void CopyPUBToPL1(TensorDst const &l1PTensorTla, TensorSrc const &ubPTensorTla, TileInfo const &tileInfo, uint32_t actQsTile, uint32_t spBuffIdx)
    {
        uint32_t pUBTlaMSize = tla::get<0>(ubPTensorTla.shape());
        uint32_t pUBTlaNSize = tla::get<1>(ubPTensorTla.shape());

        if (actQsTile == MXFP4::QS_BASE_SIZE && tileInfo.kvsActBaseTileAlign32 == MXFP4::KVS_BASE_SIZE) {
            // qs=128, kvs=256 整块搬运
            AscendC::DataCopyParams repeatParams;
            repeatParams.blockCount = static_cast<uint16_t>(pUBTlaMSize); // UB的物理P一整行256B, 包含P的逻辑视图4行
            repeatParams.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams.dstStride = 0;

            auto pUBTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET), tla::MakeShape(pUBTlaMSize, pUBTlaNSize / 2));
            auto srcOffset = pUBTensorTlaTile.layout()(pUBTensorTlaTile.coord());

            AscendC::DataCopy(l1PTensorTla.data(), pUBTensorTlaTile.data()[srcOffset], repeatParams);
        } else {
            // 消去高位 低位中间的间隙搬运
            // 低位搬运
            auto pUBTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET), tla::MakeShape(pUBTlaMSize, pUBTlaNSize / 2));
            auto srcOffset = pUBTensorTlaTile.layout()(pUBTensorTlaTile.coord());

            // DataCopy(l1PTensorTla.data(), pUBTensorTlaTile.data()[srcOffset], {static_cast<uint16_t>(128/ 32 * 8 / 2), 8, 8, 0});
            // DataCopy(l1PTensorTla.data()[128 / 32* 8 / 2 * 256], pUBTensorTlaTile.data()[srcOffset + 16384],
            //     {static_cast<uint16_t>(128 / 32 * 8 / 2), 8, 8, 0});

            AscendC::DataCopyParams repeatParams1;
            repeatParams1.blockCount = static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32 / (pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE));// UB的物理P一整行256B, 包含P的逻辑视图低位(高位)地址8行
            repeatParams1.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams1.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams1.dstStride = 0;

            auto pUBTensorTlaTile1 = tla::GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET), tla::MakeShape(pUBTlaMSize / 2, pUBTlaNSize / 2));
            auto srcOffset1 = pUBTensorTlaTile1.layout()(pUBTensorTlaTile1.coord());
            AscendC::DataCopy(l1PTensorTla.data(), ubPTensorTla.data()[srcOffset1], repeatParams1);

            // 高位搬运
            AscendC::DataCopyParams repeatParams2;
            repeatParams2.blockCount = static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32 / (pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE));// UB的物理P一整行256B, 包含P的逻辑视图高位(低位)地址8行
            repeatParams2.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams2.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams2.dstStride = 0;

            uint32_t pL1NHalfSize = tla::get<1, 0>(l1PTensorTla.shape());
            uint32_t xL1P = tileInfo.kvsActBaseTileAlign64;
            uint32_t yL1P = 0;
            if (tileInfo.kvsActBaseTileAlign64 == MXFP4::KVS_BASE_SIZE) {
                xL1P = 0;
                yL1P = pL1NHalfSize;
            }
            auto pUBTensorTlaTile2 = tla::GetTile(pUBTensorTlaTile1, tla::MakeCoord(pUBTlaMSize / 2, 0u), tla::MakeShape(pUBTlaMSize / 2, pUBTlaNSize / 2)); // 遗留
            auto pl1TensorTlaTile = tla::GetTile(l1PTensorTla, tla::MakeCoord(xL1P, yL1P), tla::MakeShape(MXFP4::KVS_BASE_SIZE, pL1NHalfSize));

            auto srcOffset2 = pUBTensorTlaTile2.layout()(pUBTensorTlaTile2.coord());
            auto dstOffset2 = pl1TensorTlaTile.layout()(pl1TensorTlaTile.coord());

            AscendC::DataCopy(pl1TensorTlaTile.data()[dstOffset2], pUBTensorTlaTile2.data()[srcOffset2], repeatParams2);

            // AscendC::printf("==== CopyPUBToPL1 ELSE debug ====\n");
            // AscendC::printf("pUBTlaMSize:%d, pUBTlaNSize:%d, actQsTile:%d, spBuffIdx:%d\n", pUBTlaMSize, pUBTlaNSize, actQsTile, spBuffIdx);
            // AscendC::printf("tileInfo.kvsActBaseTileAlign32:%d, kvsActBaseTileAlign64:%d, KVS_BASE_SIZE:%d\n", tileInfo.kvsActBaseTileAlign32, tileInfo.kvsActBaseTileAlign64, MXFP4::KVS_BASE_SIZE);
            // AscendC::printf("UB_P_QS_MAJOR_BYTE:%d, DATA_BLOCK_BYTE:%d\n", MXFP4::UB_P_QS_MAJOR_BYTE, MXFP4::DATA_BLOCK_BYTE);
            // AscendC::printf("UB_P_INNER_BUF_ELEMENT_OFFSET:%d\n", MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET);
            // AscendC::printf("srcOffset:%d, srcOffset1:%d, srcOffset2:%d, dstOffset2:%d\n", srcOffset, srcOffset1, srcOffset2, dstOffset2);
            // AscendC::printf("repeatParams1: blockCount:%d, blockLen:%d, srcStride:%d, dstStride:%d\n", repeatParams1.blockCount, repeatParams1.blockLen, repeatParams1.srcStride, repeatParams1.dstStride);
            // AscendC::printf("repeatParams2: blockCount:%d, blockLen:%d, srcStride:%d, dstStride:%d\n", repeatParams2.blockCount, repeatParams2.blockLen, repeatParams2.srcStride, repeatParams2.dstStride);
            // AscendC::printf("pL1NHalfSize:%d, xL1P:%d, yL1P:%d\n", pL1NHalfSize, xL1P, yL1P);
            // AscendC::printf("pUBTensorTlaTile1 shape(M,N):(%d,%d)\n", tla::get<0>(pUBTensorTlaTile1.shape()), tla::get<1>(pUBTensorTlaTile1.shape()));
            // AscendC::printf("pUBTensorTlaTile2 shape(M,N):(%d,%d)\n", tla::get<0>(pUBTensorTlaTile2.shape()), tla::get<1>(pUBTensorTlaTile2.shape()));
            // AscendC::printf("pl1TensorTlaTile shape(M,N):(%d,%d)\n", tla::get<0>(pl1TensorTlaTile.shape()), tla::get<1>(pl1TensorTlaTile.shape()));
            // AscendC::printf("l1PTensorTla shape(M,N):(%d,%d)\n", tla::get<0>(l1PTensorTla.shape()), tla::get<1>(l1PTensorTla.shape()));
            // AscendC::printf("==== CopyPUBToPL1 ELSE debug end ====\n");
        }
    }


    // buffer id 获取
    __aicore__ inline uint32_t GetSPBufferIdx(const uint32_t loop)
    {
        return loop / 2 % 2;
    }

    __aicore__ inline uint32_t GetLocalGroupMaxBufIdx(uint32_t loop)
    {
        return loop / 2 * 2 % MXFP4::UB_LOCAL_GROUP_MAX_CNT;
    }

private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementInput>  sUBTensor[MXFP4::UB_S_BUF_CNT];        // mm1Res(S)
    AscendC::LocalTensor<float> oTmpUBTensor;                          // mm2Res(PV)
    AscendC::LocalTensor<float>        localRowSumUBTensor;                    // LocalRowSum
    AscendC::LocalTensor<float>        globalRowSumUBTensor;                  // GlobalRowSum
    AscendC::LocalTensor<ElementDisguiseP> pUBTensor[MXFP4::UB_P_BUF_CNT];      // vec1Res(P)
    AscendC::LocalTensor<bfloat16_t> oTransUBTensor;                       // attnTrans(空间复用P)
    AscendC::LocalTensor<ElementPScale> pscaleUBTensor[MXFP4::UB_P_SCALE_CNT]; // pScale(复用P)
    AscendC::LocalTensor<bfloat16_t> oUBTensor;                            // attentionOut
    AscendC::LocalTensor<ElementMax>   peerGlobalMaxUBTensor[MXFP4::UB_PEER_GLOBAL_MAX_CNT]; // peerGlobalMax
    AscendC::LocalTensor<ElementMax>   softmaxMaxUBTensor;                    // softmaxMax
    AscendC::LocalTensor<ElementMax>   localGroupMaxUBTensor[MXFP4::UB_LOCAL_GROUP_MAX_CNT]; // LocalGroupMax
    AscendC::LocalTensor<ElementMax>   localGlobalMaxUBTensor[MXFP4::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<float>        updateScaleUBTensor[MXFP4::UB_UPDATE_SCALE_CNT]; // updateScale
    AscendC::LocalTensor<ElementIndex> indexUBTensor;                        // Index
    ElementInput scaleValue_;
    ElementInput NEG_LOG2_CX_;
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP
