/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP4_HPP
#define GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP4_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Block {

template <
    class ElementLocalGlobalMax_>
struct BlockMmadTla<
    CopyGlobalMaxL1ToUBBsa,
    void,
    void,
    ElementLocalGlobalMax_,
    void,
    void,
    void,
    void,
    void>
{
public:
    using DispatchPolicy = CopyGlobalMaxL1ToUBBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementLocalGlobalMax = ElementLocalGlobalMax_;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;

    // L1 各区单份字节数（fp4 两元素占 1 字节；scale 为 E8M0，每 32 元素 1 字节）
    static constexpr uint32_t L1_Q_SIZE = 128 * 128 / 2;                                          // Q 数据：8K
    static constexpr uint32_t L1_Q_DESCALE_SIZE = 128 * (128 / 32) * sizeof(AscendC::fp8_e8m0_t); // Q-scale：0.5K
    static constexpr uint32_t L1_KV_SIZE = 512 * 128 / 2;                                         // KV/V 数据：32K
    static constexpr uint32_t L1_KV_DESCALE_SIZE = 512 * (128 / 32) * sizeof(AscendC::fp8_e8m0_t);// KV/V-scale：2K
    static constexpr uint32_t L1_P_SIZE = 128 * 256 / 2;                                          // P 数据：16K
    static constexpr uint32_t L1_P_DESCALE_SIZE = 128 * 2 * (256 / 64 + 1) * sizeof(AscendC::fp8_e8m0_t); // P-scale：1.25K
    static constexpr uint32_t L1_Q_BUFCNT = 2;    // Q 区份数
    static constexpr uint32_t L1_KV_BUFCNT = 2;   // KV/V 区份数（2+2 方案：PV 2 槽 ping-pong，QK 另 2 槽独立）
    static constexpr uint32_t L1_P_BUFCNT = 20;   // P 区份数

    static constexpr uint32_t L1_SINGLE_GLOBAL_MAX_SIZE = 128;
    static constexpr uint32_t L1_GLOBAL_MAX_SIZE = 256;
    static constexpr uint32_t L1_GLOBAL_MAX_BUFCNT = 4;

    static constexpr uint8_t  VEC0 = 0;
    static constexpr uint8_t  VEC1 = 1;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource) {
        // L1 区域顺序（与 QFA 一致）：P 数据 | P-scale | Q 数据 | Q-scale | KV(=V) 数据 | KV(=V)-scale
        uint32_t addrL1Start = 0;
        addrL1Start += L1_P_SIZE * L1_P_BUFCNT;
        addrL1Start += L1_P_DESCALE_SIZE * L1_P_BUFCNT;
        addrL1Start += L1_Q_SIZE * L1_Q_BUFCNT;
        addrL1Start += L1_Q_DESCALE_SIZE * L1_Q_BUFCNT;
        addrL1Start += L1_KV_SIZE * L1_KV_BUFCNT;
        addrL1Start += L1_KV_DESCALE_SIZE * L1_KV_BUFCNT;
        // localGlobalMaxL1 = 256*4
        for (uint32_t i = 0; i < L1_GLOBAL_MAX_BUFCNT; i++) {
            localGlobalMaxL1[i] = resource.l1Buf.template GetBufferByByte<half>(addrL1Start + L1_GLOBAL_MAX_SIZE * i);
        }
    }
    __aicore__ inline ~BlockMmadTla() {}

    template <class TileInfoT>
    __aicore__ inline
    void operator()(TileInfoT &delay3TileInfo, AscendC::LocalTensor<half> peerGlobalMaxUB)
    {
        AscendC::PRINTF("CopyGMaxL1ToUb \n");

        // [TODO] 此处也是假数据
        AscendC::InitConstValueParams<half> localGlobalMaxL1FillParams(1, static_cast<uint16_t>(L1_GLOBAL_MAX_SIZE * 2  / 32 ), 0, static_cast<half>(1));
        AscendC::Fill(localGlobalMaxL1[0], localGlobalMaxL1FillParams);          // localGlobalMaxL1 数据
        AscendC::PipeBarrier<PIPE_MTE2>();

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(7);

        AscendC::LocalTensor<half> localGlobalMax0 = localGlobalMaxL1[delay3TileInfo.tileMaxIdx];
        AscendC::LocalTensor<half> localGlobalMax1 = localGlobalMaxL1[delay3TileInfo.tileMaxIdx][L1_SINGLE_GLOBAL_MAX_SIZE];
        AscendC::LocalTensor<half> peerGlobalMax = peerGlobalMaxUB;

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1; // 连续数据块个数为1
        intriParams.blockLen = 8; // 连续数据块长度，单位为DataBlock，此处长度为128个half元素
        intriParams.srcGap = 0; // 源操作数做连续搬运
        intriParams.dstGap = 0; // 目的操作数连续排布
        AscendC::DataCopyL1ToUB<half,VEC0>(peerGlobalMax, localGlobalMax0, intriParams);
        AscendC::DataCopyL1ToUB<half,VEC1>(peerGlobalMax, localGlobalMax1, intriParams);
    }

protected:
    // MXFP4 专用：E8M0 scale 的 L1 buffer（V-scale 4 份，P-scale 20 份）+ MX 版 L1->L0 拷贝器
    AscendC::LocalTensor<half> localGlobalMaxL1[L1_KV_BUFCNT];

};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP4_HPP
