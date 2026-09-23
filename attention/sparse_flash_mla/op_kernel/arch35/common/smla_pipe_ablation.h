/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SMLA_PIPE_ABLATION_H
#define SMLA_PIPE_ABLATION_H
#define SMLA_PIPE_ABLATION_REVISION 1

#include "kernel_operator.h"
#include "../../../../common/op_kernel/matmul.h"
#include "../../../../common/op_kernel/CopyInL1.h"
#include "../../../../common/op_kernel/FixpipeOut.h"

namespace SMLAKernel {
using namespace AscendC;
using namespace fa_base_matmul;

// Diagnostic only. Mask bits [W2,R1,W3] ablate repeated transfers, NOT setup or
// MMAD consumption. Read storage and the two writers' storage never overlap.
// Timing mode adds scalar completion fences; its SYS_CNT ticks are reported
// separately, never converted using the nominal Cube clock or called bank PMU.
template <typename Q>
class SparseL1PipeAblation {
    static constexpr uint32_t KIB = 1024;
    static constexpr uint32_t M = 16;
    static constexpr uint32_t S = 128;
    static constexpr uint32_t CHECK_ELEMS = M * S;
    static constexpr uint32_t WRITE_ELEMS = S * 256;
    static constexpr uint32_t NZ_ELEMS = 16 * 17 * 16;
    static constexpr uint32_t TIME_OFFSET = 4 * 1024 * KIB;
    // Separate cache lines for Cube, AIV0 and AIV1 scalar GM stores.
    static constexpr uint32_t TIME_STRIDE = 48;

    template <pipe_t Pipe>
    __aicore__ static void SetBoth(uint32_t id)
    {
        CrossCoreSetFlag<4, Pipe>(id);
        CrossCoreSetFlag<4, Pipe>(id + 16);
    }

    __aicore__ static void WaitBoth(uint32_t id)
    {
        CrossCoreWaitFlag<4, PIPE_MTE1>(id);
        CrossCoreWaitFlag<4, PIPE_MTE1>(id + 16);
    }

    template <typename Out>
    __aicore__ static void Store(const GlobalTensor<Out> &dst, const LocalTensor<float> &src)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
        p.nSize = S;
        p.mSize = M;
        p.srcStride = M;
        p.dstStride = S;
        p.dualDstCtl = 0;
        p.params.ndNum = 1;
        p.params.srcNdStride = 0;
        p.params.dstNdStride = 0;
        if constexpr (std::is_same<Out, bfloat16_t>::value) {
            p.quantPre = QuantMode_t::F322BF16;
        } else if constexpr (std::is_same<Out, half>::value) {
            p.quantPre = QuantMode_t::F322F16;
        }
        Fixpipe<Out, float, PFA_CFG_ROW_MAJOR_GM>(dst, src, p);
    }

    __aicore__ static void Cube(GM_ADDR query, GM_ADDR kv, GM_ADDR writeSource, GM_ADDR output, GM_ADDR lse,
                                uint32_t mask, uint32_t n, uint32_t transpose, uint32_t layout,
                                uint32_t profile, uint32_t timing)
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t w2Base = (64 + ((layout & 1U) ? 256 : 0)) * KIB;
        const uint32_t w3Base = (128 + ((layout & 2U) ? 256 : 0)) * KIB;
        const uint32_t period = profile == 0 ? 1 : 20;
        GlobalTensor<Q> qGm, readGm, writeGm, checkGm;
        GlobalTensor<float> resultGm;
        GlobalTensor<uint64_t> times;
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(query));
        readGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(kv));
        writeGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(writeSource));
        checkGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(output));
        resultGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lse));
        times.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output + TIME_OFFSET));
        LocalTensor<Q> r(TPosition::B1, 0, 2 * S * S);
        LocalTensor<Q> selectors(TPosition::A1, 192 * KIB, S * S);
        LocalTensor<Q> w2(TPosition::B1, w2Base, WRITE_ELEMS);
        LocalTensor<Q> w3(TPosition::B1, w3Base, WRITE_ELEMS);
        LocalTensor<Q> a(TPosition::A2, 0, M * S);
        LocalTensor<Q> b0(TPosition::B2, 0, S * S);
        LocalTensor<Q> b1(TPosition::B2, 32 * KIB, S * S);
        LocalTensor<float> c(TPosition::CO1, 0, CHECK_ELEMS);

        CopyToL1Nd2Nz(r, readGm, S, S, 512);
        CopyToL1Nd2Nz(r[S * S], readGm[S], S, S, 512);
        for (uint32_t block = 0; block < 8; ++block) {
            CopyToL1Nd2Nz(selectors[block * CHECK_ELEMS], qGm[block * M * 512], M, S, 512);
        }
        // Rows 128..255 of Q are zero, so even never-written cells are defined.
        CopyToL1Nd2Nz(w2, qGm[S * 512], S, 256, 512);
        CopyToL1Nd2Nz(w3, qGm[S * 512], S, 256, 512);
        SetBoth<PIPE_MTE2>(3); // INIT: W3 belongs to the vector writers afterwards.
        SetFlag<HardEvent::MTE2_MTE1>(2);
        WaitFlag<HardEvent::MTE2_MTE1>(2);
        MMParam p = {M, S, S, false, transpose != 0};
        LoadDataToL0A(a, selectors, p, 0, S, M);
        LoadDataToL0B(b0, r, p, 0, S, S);
        LoadDataToL0B(b1, r[S * S], p, 0, S, S);
        SetFlag<HardEvent::MTE1_M>(2);
        WaitFlag<HardEvent::MTE1_M>(2);
        MmadParams mm;
        mm.m = M;
        mm.n = S;
        mm.k = S;
        mm.cmatrixInitVal = true;
        mm.cmatrixSource = false;
        Mmad(c, a, b0, mm);
        mm.cmatrixInitVal = false;
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
        WaitBoth(0); // Both AIVs have loaded immutable UB inputs and consumed INIT.
        uint64_t start = 0;
        uint64_t readDone = 0;
        uint64_t writeJoin = 0;
        uint64_t allJoin = 0;
        if (timing != 0) {
            SetFlag<HardEvent::MTE1_M>(3);
            WaitFlag<HardEvent::MTE1_M>(3);
            SetFlag<HardEvent::M_S>(3);
            WaitFlag<HardEvent::M_S>(3);
            start = GetSystemCycle();
        }
        SetBoth<PIPE_MTE1>(1); // START, independent of whether W3 is enabled.
        SetFlag<HardEvent::MTE1_MTE2>(3);
        WaitFlag<HardEvent::MTE1_MTE2>(3);
        uint32_t nextWrite = 0;
        uint32_t writeIndex = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t slot = i & 1U;
            LocalTensor<Q> b = slot == 0 ? b0 : b1;
            WaitFlag<HardEvent::M_MTE1>(slot);
            if ((mask & 2U) != 0) {
                LoadDataToL0B(b, r[slot * S * S], p, 0, S, S);
            }
            SetFlag<HardEvent::MTE1_M>(slot);
            WaitFlag<HardEvent::MTE1_M>(slot);
            Mmad(c, a, b, mm); // Same N MMADs, including when repeated R1 is off.
            SetFlag<HardEvent::M_MTE1>(slot);
            if (i == nextWrite) {
                if ((mask & 4U) != 0) {
                    const uint32_t half = writeIndex & 1U;
                    CopyToL1Nd2Nz(w2[half * S * S], writeGm[16384 + half * S], S, S, 512);
                }
                ++writeIndex;
                nextWrite += period;
            }
        }
        if (timing != 0) {
            SetFlag<HardEvent::M_S>(4);
            WaitFlag<HardEvent::M_S>(4);
            readDone = GetSystemCycle(); // Read + fixed MMAD consumer, not pure MTE1 service.
        }
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::MTE2_MTE1>(3);
        WaitFlag<HardEvent::MTE2_MTE1>(3);
        if (timing != 0) {
            SetFlag<HardEvent::MTE2_S>(3);
            WaitFlag<HardEvent::MTE2_S>(3);
            writeJoin = GetSystemCycle(); // Join after readDone; NOT MTE2's independent finish.
        }
        WaitBoth(2); // DONE from both AIVs, including disabled writers.
        if (timing != 0) {
            SetFlag<HardEvent::MTE1_M>(5);
            WaitFlag<HardEvent::MTE1_M>(5);
            SetFlag<HardEvent::M_S>(5);
            WaitFlag<HardEvent::M_S>(5);
            allJoin = GetSystemCycle();
            times.SetValue(core * TIME_STRIDE, start);
            times.SetValue(core * TIME_STRIDE + 1, readDone);
            times.SetValue(core * TIME_STRIDE + 2, writeJoin);
            times.SetValue(core * TIME_STRIDE + 3, allJoin);
        }
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        Store(resultGm[core * CHECK_ELEMS], c);
        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(0);

        // Check ALL rows/columns of both writers, including unwritten zeros.
        p.isRightTranspose = 0;
        mm.cmatrixInitVal = true;
        for (uint32_t writer = 0; writer < 2; ++writer) {
            LocalTensor<Q> w = writer == 0 ? w2 : w3;
            for (uint32_t rowBlock = 0; rowBlock < 8; ++rowBlock) {
                LoadDataToL0A(a, selectors[rowBlock * CHECK_ELEMS], p, 0, S, M);
                for (uint32_t half = 0; half < 2; ++half) {
                    LoadDataToL0B(b0, w[half * S * S], p, 0, S, S);
                    SetFlag<HardEvent::MTE1_M>(2);
                    WaitFlag<HardEvent::MTE1_M>(2);
                    Mmad(c, a, b0, mm);
                    SetFlag<HardEvent::M_MTE1>(2);
                    WaitFlag<HardEvent::M_MTE1>(2);
                    SetFlag<HardEvent::M_FIX>(0);
                    WaitFlag<HardEvent::M_FIX>(0);
                    Store(checkGm[((core * 2 + writer) * 16 + rowBlock * 2 + half) * CHECK_ELEMS], c);
                    SetFlag<HardEvent::FIX_M>(0);
                    WaitFlag<HardEvent::FIX_M>(0);
                }
            }
        }
    }

    __aicore__ static void Vector(GM_ADDR source, GM_ADDR output, uint32_t mask, uint32_t n,
                                  uint32_t layout, uint32_t profile, uint32_t timing)
    {
        const uint32_t sub = GetSubBlockIdx();
        const uint32_t core = GetBlockIdx() >> 1;
        const uint32_t base = (128 + ((layout & 2U) ? 256 : 0)) * KIB;
        const uint32_t multiplier = profile == 0 ? 2 : 1;
        GlobalTensor<Q> src;
        GlobalTensor<uint64_t> times;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(source));
        times.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output + TIME_OFFSET));
        LocalTensor<Q> ub(TPosition::VECIN, 0, NZ_ELEMS);
        LocalTensor<Q> dst(TPosition::B1, base, WRITE_ELEMS);
        DataCopy(ub, src[sub * NZ_ELEMS], NZ_ELEMS);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        CrossCoreWaitFlag<4, PIPE_MTE3>(3);
        CrossCoreSetFlag<4, PIPE_MTE3>(0);
        CrossCoreWaitFlag<4, PIPE_MTE3>(1);
        uint64_t start = 0;
        if (timing != 0) {
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
            start = GetSystemCycle();
        }
        DataCopyParams copy;
        copy.blockCount = 16;
        copy.blockLen = 16;
        copy.srcStride = 1;
        copy.dstStride = 112;
        if ((mask & 1U) != 0) {
            for (uint32_t i = 0; i < n * multiplier; ++i) {
                const uint32_t row = sub * 64 + (i & 3U) * 16;
                DataCopy(dst[row * 16], ub, copy);
            }
        }
        if (timing != 0) {
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
            const uint64_t done = GetSystemCycle();
            times.SetValue(core * TIME_STRIDE + 16 * (sub + 1), start);
            times.SetValue(core * TIME_STRIDE + 16 * (sub + 1) + 1, done);
        }
        CrossCoreSetFlag<4, PIPE_MTE3>(2);
    }

public:
    __aicore__ static bool Run(GM_ADDR query, GM_ADDR kv, GM_ADDR writeSource,
                               GM_ADDR metadata, GM_ADDR output, GM_ADDR lse)
    {
        if (metadata == nullptr) {
            return false;
        }
        GlobalTensor<int32_t> cfg;
        cfg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(metadata));
        if (cfg.GetValue(0) != 0x534D4C42 || cfg.GetValue(6) != 0x23456789) {
            return false;
        }
        const uint32_t mask = cfg.GetValue(1);
        const uint32_t n = cfg.GetValue(2);
        const uint32_t cores = cfg.GetValue(3);
        const uint32_t transpose = cfg.GetValue(4);
        const uint32_t layout = cfg.GetValue(5);
        const uint32_t profile = cfg.GetValue(7);
        const uint32_t timing = cfg.GetValue(8);
        if (mask > 7 || n > 4096 || cores == 0 || cores > 32 || transpose > 1 || layout > 3 ||
            profile > 1 || timing > 1 || lse == nullptr || output == nullptr) {
            return true;
        }
        if ASCEND_IS_AIC {
            if (GetBlockIdx() < cores) {
                Cube(query, kv, writeSource, output, lse, mask, n, transpose, layout, profile, timing);
            }
        } else {
            if ((GetBlockIdx() >> 1) < cores) {
                Vector(writeSource, output, mask, n, layout, profile, timing);
            }
        }
        return true;
    }
};
} // namespace SMLAKernel
#endif // SMLA_PIPE_ABLATION_H
