/*
 * Copyright (c) 2026 G_W_E. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "attention_block_cube.h"
// FP32 transport diagnostic uses the production SPLIT_M Fixpipe path.
extern "C" __global__ __aicore__ void sage_pv_fp32_debug(GM_ADDR pg, GM_ADDR vg, GM_ADDR og)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Catlass::Arch::Resource<Sage::Arch> r;
    if ASCEND_IS_AIC {
        using namespace Catlass;
        using namespace tla;
        using Copy = Gemm::Tile::PackedTileCopyTla<Sage::Arch, float8_e4m3_t, layout::RowMajor, float8_e4m3_t,
                                                   layout::RowMajor, float, layout::RowMajor>;
        GlobalTensor<float8_e4m3_t> p;
        p.SetGlobalBuffer((__gm__ float8_e4m3_t *)pg);
        auto src = MakeTensor(p, MakeLayout<float8_e4m3_t, layout::RowMajor>(128, 256), Arch::PositionGM{});
        auto dst = MakeTensor(r.l1Buf.GetBufferByByte<float8_e4m3_t>(Sage::P_L1),
                              MakeLayout<float8_e4m3_t, typename Copy::LayoutTagL1A>(128, 256), Arch::PositionL1{});
        typename Copy::template CopyGmToL1A<decltype(src)> copy;
        copy(dst, src);
        SetFlag<HardEvent::MTE1_MTE2>(3);
        SetFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::FIX_M>(3);
        Sage::Gemm<float8_e4m3_t, layout::RowMajor, true>(r, pg, vg, 256, 128, Sage::PV_UB);
        Sage::Notify(2);
        WaitFlag<HardEvent::MTE1_MTE2>(3);
        WaitFlag<HardEvent::M_MTE1>(3);
        WaitFlag<HardEvent::FIX_M>(3);
    } else {
        CrossCoreWaitFlag<4, PIPE_MTE3>(2);
        GlobalTensor<float> out;
        out.SetGlobalBuffer((__gm__ float *)og);
        DataCopy(out[GetSubBlockIdx() * 64 * 128], r.ubBuf.GetBufferByByte<float>(Sage::PV_UB), 64 * 128);
    }
}
extern "C" void sage_pv_fp32_debug_launch(void *stream, void **p)
{
    sage_pv_fp32_debug<<<1, nullptr, stream>>>((uint8_t *)p[0], (uint8_t *)p[1], (uint8_t *)p[2]);
}
// QK conversion diagnostic: exercise the actual Gemm DEQF16 branch and both AIVs.
extern "C" __global__ __aicore__ void sage_qk_half_debug(GM_ADDR qg, GM_ADDR kg, GM_ADDR og)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Catlass::Arch::Resource<Sage::Arch> r;
    if ASCEND_IS_AIC {
        SetFlag<HardEvent::MTE1_MTE2>(2);
        SetFlag<HardEvent::M_MTE1>(2);
        SetFlag<HardEvent::FIX_M>(2);
        Sage::Gemm<int8_t, Catlass::layout::ColumnMajor>(r, qg, kg, 128, 256, Sage::SCORE_UB);
        Sage::Notify(0);
        WaitFlag<HardEvent::MTE1_MTE2>(2);
        WaitFlag<HardEvent::M_MTE1>(2);
        WaitFlag<HardEvent::FIX_M>(2);
    } else {
        CrossCoreWaitFlag<4, PIPE_MTE3>(0);
        GlobalTensor<half> out;
        out.SetGlobalBuffer((__gm__ half *)og);
        DataCopy(out[GetSubBlockIdx() * 64 * 256], r.ubBuf.GetBufferByByte<half>(Sage::SCORE_UB), 64 * 256);
    }
}
extern "C" void sage_qk_half_debug_launch(void *stream, void **p)
{
    sage_qk_half_debug<<<1, nullptr, stream>>>((uint8_t *)p[0], (uint8_t *)p[1], (uint8_t *)p[2]);
}
