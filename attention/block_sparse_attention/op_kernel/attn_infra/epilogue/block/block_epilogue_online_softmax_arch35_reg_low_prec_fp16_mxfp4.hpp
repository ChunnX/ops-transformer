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

namespace NpuArch::Epilogue::Block {

using namespace MXFP4;

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
    static constexpr MXQuantMode mxQuantMode = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = half;
    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue)
    {
        // FIXPIPE<->V 区
        for (uint32_t i = 0; i < MXFP4::UB_S_BUF_CNT; i++) {
            SUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(
                MXFP4::UB_S_BUF_OFFSET + MXFP4::UB_S_BUF_SIZE * i);
        }
        oTmpUBTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(MXFP4::UB_OTMP_BUF_OFFSET);
        localRowSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP4::UB_LOCAL_ROW_SUM_OFFSET);
        GlobalRowSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP4::UB_GLOBAL_ROW_SUM_OFFSET);

        // 输出缓冲区
        for (uint32_t i = 0; i < MXFP4::UB_P_BUF_CNT; i++) {
            PUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                MXFP4::UB_P_BUF_OFFSET + MXFP4::UB_P_BUF_SIZE * i);
        }
        OTransUBTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(MXFP4::UB_O_TRANS_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_P_SCALE_CNT; i++) {
            PScaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<uint8_t>(
                MXFP4::UB_P_SCALE_OFFSET + MXFP4::UB_P_SCALE_SIZE * i);
        }
        OUBTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(MXFP4::UB_O_OFFSET);

        // L1 <-> UB
        for (uint32_t i = 0; i < MXFP4::UB_PEER_GLOBAL_MAX_CNT; i++) {
            peerGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<half>(
                MXFP4::UB_PEER_GLOBAL_MAX_OFFSET + MXFP4::UB_PEER_GLOBAL_MAX_SIZE * i);
        }

        // 常驻 buffer
        SoftmaxMaxUBTensor = resource.ubBuf.template GetBufferByByte<half>(MXFP4::UB_SOFTMAX_MAX_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GROUP_MAX_CNT; i++) {
            LocalGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<half>(
                MXFP4::UB_LOCAL_GROUP_MAX_OFFSET + MXFP4::UB_LOCAL_GROUP_MAX_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<half>(
                MXFP4::UB_LOCAL_GLOBAL_MAX_OFFSET + MXFP4::UB_LOCAL_GLOBAL_MAX_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_UPDATE_SCALE_CNT; i++) {
            updateScaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                MXFP4::UB_UPDATE_SCALE_OFFSET + MXFP4::UB_UPDATE_SCALE_SIZE * i);
        }
        indexUBTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(MXFP4::UB_INDEX_OFFSET);
    }
    __aicore__ inline ~BlockEpilogue() {}
private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementInput>  SUBTensor[MXFP4::UB_S_BUF_CNT];        // mm1Res(S)
    AscendC::LocalTensor<ElementOutput> oTmpUBTensor;                          // mm2Res(PV)
    AscendC::LocalTensor<float>        localRowSumUBTensor;                    // LocalRowSum
    AscendC::LocalTensor<float>        GlobalRowSumUBTensor;                  // GlobalRowSum
    AscendC::LocalTensor<ElementOutput> PUBTensor[MXFP4::UB_P_BUF_CNT];      // vec1Res(P)
    AscendC::LocalTensor<ElementOutput> OTransUBTensor;                       // attnTrans(空间复用P)
    AscendC::LocalTensor<uint8_t>      PScaleUBTensor[MXFP4::UB_P_SCALE_CNT]; // pScale(复用P)
    AscendC::LocalTensor<ElementOutput> OUBTensor;                            // attentionOut
    AscendC::LocalTensor<half>         peerGlobalMaxUBTensor[MXFP4::UB_PEER_GLOBAL_MAX_CNT]; // peerGlobalMax
    AscendC::LocalTensor<half>         SoftmaxMaxUBTensor;                    // softmaxMax
    AscendC::LocalTensor<half>         LocalGroupMaxUBTensor[MXFP4::UB_LOCAL_GROUP_MAX_CNT]; // LocalGroupMax
    AscendC::LocalTensor<half>         localGlobalMaxUBTensor[MXFP4::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<float>        updateScaleUBTensor[MXFP4::UB_UPDATE_SCALE_CNT]; // updateScale
    AscendC::LocalTensor<uint8_t>      indexUBTensor;                        // Index

};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP
