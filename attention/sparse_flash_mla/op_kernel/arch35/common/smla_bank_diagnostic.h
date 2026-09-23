/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SMLA_BANK_DIAGNOSTIC_H
#define SMLA_BANK_DIAGNOSTIC_H
#define SMLA_BANK_DIAGNOSTIC_REVISION 2

#include "kernel_operator.h"
#include "smla_experiment_config.h"
#include "../../../../common/op_kernel/matmul.h"
#include "../../../../common/op_kernel/CopyInL1.h"
#include "../../../../common/op_kernel/FixpipeOut.h"

namespace SMLAKernel {
using namespace AscendC;
using namespace fa_base_matmul;

// An isolated data-path diagnostic, never an attention result. Enabled only in
// an explicit diagnostic build AND by two metadata markers. All modes use the
// same binary and disjoint read/write storage. No global barrier or reserved ID.
template <typename Q>
class SparseL1BankDiagnostic {
    static constexpr uint32_t KIB = 1024;
    static constexpr uint32_t M = 16;
    static constexpr uint32_t S = 128;
    static constexpr uint32_t RESULT_ELEMENTS = M * S;
    static constexpr uint32_t NZ_SOURCE_ELEMENTS = 16 * 17 * 16;
    enum Mode : uint32_t {
        READ = 0, WRITE = 1, SAME = 2, CROSS = 3, SERIAL = 4, COMPUTE = 5,
        WRITE_CROSS = 6, SERIAL_CROSS = 7
    };

    __aicore__ static void SetBoth(uint32_t id)
    {
        CrossCoreSetFlag<4, PIPE_MTE1>(id);
        CrossCoreSetFlag<4, PIPE_MTE1>(id + 16);
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

    __aicore__ static void CopyMte2(const LocalTensor<Q> &dst, const GlobalTensor<Q> &src, uint32_t i)
    {
        const uint32_t half = i & 1U;
        CopyToL1Nd2Nz(dst[half * S * S], src[16384 + half * S], S, S, 512);
    }

    __aicore__ static void Cube(GM_ADDR query, GM_ADDR kv, GM_ADDR writeSource,
                                GM_ADDR output, GM_ADDR lse,
                                uint32_t core, uint32_t mode, uint32_t iterations,
                                uint32_t transpose, uint32_t writeBase, uint32_t readTiles,
                                uint32_t writerPipe)
    {
        GlobalTensor<Q> qGm, kvGm, writeGm, checkGm;
        GlobalTensor<float> resultGm;
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(query));
        kvGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(kv));
        writeGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(writeSource));
        checkGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(output));
        resultGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lse));
        const uint32_t qBase = readTiles == 2 ? 256 * KIB : 32 * KIB;
        LocalTensor<Q> readL1(TPosition::B1, 0, readTiles * S * S);
        LocalTensor<Q> qL1(TPosition::A1, qBase, M * S);
        LocalTensor<Q> qSecondL1(TPosition::A1, qBase + 4 * KIB, M * S);
        LocalTensor<Q> writeL1(TPosition::B1, writeBase, S * 256);
        LocalTensor<Q> a(TPosition::A2, 0, M * S);
        LocalTensor<Q> b0(TPosition::B2, 0, S * S);
        LocalTensor<Q> b1(TPosition::B2, 32 * KIB, S * S);
        LocalTensor<float> c(TPosition::CO1, 0, RESULT_ELEMENTS);

        // The diagnostic input has zero query rows 32..159. Initialize unwritten
        // NZ rows too: multiplying uninitialized NaN by a zero selector is unsafe.
        CopyToL1Nd2Nz(readL1, kvGm, S, S, 512);
        if (readTiles == 2) {
            CopyToL1Nd2Nz(readL1[S * S], kvGm[S], S, S, 512);
        }
        CopyToL1Nd2Nz(qL1, qGm, M, S, 512);
        CopyToL1Nd2Nz(qSecondL1, qGm[M * 512], M, S, 512);
        CopyToL1Nd2Nz(writeL1, qGm[32 * 512], S, 256, 512);
        CrossCoreSetFlag<4, PIPE_MTE2>(3);
        CrossCoreSetFlag<4, PIPE_MTE2>(19);
        SetFlag<HardEvent::MTE2_MTE1>(2);
        WaitFlag<HardEvent::MTE2_MTE1>(2);
        MMParam p = {M, S, S, false, transpose != 0};
        LoadDataToL0A(a, qL1, p, 0, S, M);
        LoadDataToL0B(b0, readL1, p, 0, S, S);
        LoadDataToL0B(b1, readL1[(readTiles - 1) * S * S], p, 0, S, S);
        SetFlag<HardEvent::MTE1_M>(2);
        WaitFlag<HardEvent::MTE1_M>(2);
        MmadParams mm;
        mm.m = M;
        mm.n = S;
        mm.k = S;
        mm.cmatrixInitVal = true;
        mm.cmatrixSource = false;
        Mmad(c, a, b0, mm); // Fixed initialization, present in every mode.
        mm.cmatrixInitVal = false;
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
        WaitBoth(0); // Both initial AIV writes have completed.
        if (writerPipe == 1) {
            // Transfer W ownership from the initial AIV writers to AIC MTE2.
            SetFlag<HardEvent::MTE1_MTE2>(3);
            WaitFlag<HardEvent::MTE1_MTE2>(3);
        }
        const bool serial = mode == SERIAL || mode == SERIAL_CROSS;
        const bool writeOnly = mode == WRITE || mode == WRITE_CROSS;
        const bool repeatedWrite = mode != READ && mode != COMPUTE;
        if (!serial) {
            SetBoth(1);
        }
        if (!writeOnly) {
            for (uint32_t i = 0; i < iterations; ++i) {
                const uint32_t slot = i & 1U;
                LocalTensor<Q> b = slot == 0 ? b0 : b1;
                WaitFlag<HardEvent::M_MTE1>(slot);
                if (mode != COMPUTE) {
                    LoadDataToL0B(b, readL1[(readTiles == 2 ? slot : 0) * S * S], p, 0, S, S);
                }
                SetFlag<HardEvent::MTE1_M>(slot);
                WaitFlag<HardEvent::MTE1_M>(slot);
                Mmad(c, a, b, mm); // Every measured read contributes to the checksum.
                SetFlag<HardEvent::M_MTE1>(slot);
                if (writerPipe == 1 && repeatedWrite && !serial) {
                    CopyMte2(writeL1, writeGm, i);
                }
            }
        }
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        if (writerPipe == 1 && repeatedWrite && (serial || writeOnly)) {
            SetFlag<HardEvent::MTE1_MTE2>(4);
            WaitFlag<HardEvent::MTE1_MTE2>(4);
            for (uint32_t i = 0; i < iterations; ++i) {
                CopyMte2(writeL1, writeGm, i);
            }
        }
        if (writerPipe == 1) {
            SetFlag<HardEvent::MTE2_MTE1>(3);
            WaitFlag<HardEvent::MTE2_MTE1>(3);
        }
        if (serial) {
            SetBoth(1); // Only now may the AIVs start their repeated writes.
        }
        WaitBoth(2);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        Store(resultGm[core * RESULT_ELEMENTS], c);
        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(0);

        // Validate all written rows and all 256 columns, on both AIVs. This fixed
        // postamble is outside the repeated loops but included in task duration.
        p.isRightTranspose = 0;
        mm.cmatrixInitVal = true;
        for (uint32_t v = 0; v < 2; ++v) {
            LocalTensor<Q> select = v == 0 ? qL1 : qSecondL1;
            LoadDataToL0A(a, select, p, 0, S, M);
            for (uint32_t half = 0; half < 2; ++half) {
                LocalTensor<Q> quarter = writeL1[half * S * S];
                LoadDataToL0B(b0, quarter, p, 0, S, S);
                SetFlag<HardEvent::MTE1_M>(2);
                WaitFlag<HardEvent::MTE1_M>(2);
                Mmad(c, a, b0, mm);
                SetFlag<HardEvent::M_MTE1>(2);
                WaitFlag<HardEvent::M_MTE1>(2);
                SetFlag<HardEvent::M_FIX>(0);
                WaitFlag<HardEvent::M_FIX>(0);
                Store(checkGm[(core * 4 + v * 2 + half) * RESULT_ELEMENTS], c);
                SetFlag<HardEvent::FIX_M>(0);
                WaitFlag<HardEvent::FIX_M>(0);
            }
        }
    }

    __aicore__ static void Vector(GM_ADDR source, uint32_t mode, uint32_t iterations,
                                  uint32_t writeBase, uint32_t writeMultiplier,
                                  uint32_t writerMask, uint32_t writerPipe)
    {
        const uint32_t sub = GetSubBlockIdx();
        GlobalTensor<Q> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(source));
        LocalTensor<Q> ub(TPosition::VECIN, 0, NZ_SOURCE_ELEMENTS);
        LocalTensor<Q> dst(TPosition::B1, writeBase, S * 256);
        DataCopy(ub, src[sub * NZ_SOURCE_ELEMENTS], NZ_SOURCE_ELEMENTS);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        CrossCoreWaitFlag<4, PIPE_MTE3>(3);
        DataCopyParams copy;
        copy.blockCount = 16;
        copy.blockLen = 16;
        copy.srcStride = 1;
        copy.dstStride = 112;
        const uint32_t dstRowOffset = sub * 64 * 16;
        DataCopy(dst[dstRowOffset], ub, copy);
        CrossCoreSetFlag<4, PIPE_MTE3>(0);
        CrossCoreWaitFlag<4, PIPE_MTE3>(1);
        if (writerPipe == 0 && (writerMask & (1U << sub)) != 0 && mode != READ && mode != COMPUTE) {
            for (uint32_t i = 0; i < iterations * writeMultiplier; ++i) {
                DataCopy(dst[dstRowOffset], ub, copy);
            }
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
        GlobalTensor<int32_t> config;
        config.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(metadata));
        if (config.GetValue(0) != 0x534D4C41 || config.GetValue(6) != 0x12345678) {
            return false;
        }
        const uint32_t mode = config.GetValue(1);
        const uint32_t iterations = config.GetValue(2);
        const uint32_t activeCores = config.GetValue(3);
        const uint32_t transpose = config.GetValue(4);
        const uint32_t offset = config.GetValue(5);
        const uint32_t writeMultiplier = config.GetValue(7);
        // Zero extension fields retain the original diagnostic ABI.
        const uint32_t readTiles = config.GetValue(8) == 0 ? 1 : config.GetValue(8);
        const uint32_t writerMask = config.GetValue(9) == 0 ? 3 : config.GetValue(9);
        const uint32_t writerPipe = config.GetValue(10); // 0 AIV MTE3, 1 AIC MTE2.
        if (mode > SERIAL_CROSS || iterations > 16384 || activeCores == 0 ||
            activeCores > 32 || transpose > 1 || offset > 2 || writeMultiplier == 0 || writeMultiplier > 4 ||
            readTiles > 2 || writerMask > 3 || writerPipe > 1 ||
            (writerPipe == 1 && (writeMultiplier != 2 || writerMask != 3)) ||
            lse == nullptr) {
            return true; // Invalid diagnostics must never fall through as attention metadata.
        }
        const bool highWrite = mode == CROSS || mode == WRITE_CROSS || mode == SERIAL_CROSS;
        const uint32_t writeBase = (64 + offset * 64 + (highWrite ? 256 : 0)) * KIB;
        if ASCEND_IS_AIC {
            if (GetBlockIdx() < activeCores) {
                Cube(query, kv, writeSource, output, lse, GetBlockIdx(), mode, iterations, transpose,
                     writeBase, readTiles, writerPipe);
            }
        } else {
            if ((GetBlockIdx() >> 1) < activeCores) {
                Vector(writeSource, mode, iterations, writeBase, writeMultiplier, writerMask, writerPipe);
            }
        }
        return true;
    }
};
} // namespace SMLAKernel
#endif // SMLA_BANK_DIAGNOSTIC_H
