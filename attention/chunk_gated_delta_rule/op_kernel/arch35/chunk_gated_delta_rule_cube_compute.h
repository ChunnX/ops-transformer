/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_CUBE_COMPUTE_H
#define CHUNK_GATED_DELTA_RULE_CUBE_COMPUTE_H

#include "chunk_gated_delta_rule_context.h"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"
#include "chunk_gated_delta_rule_sync.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// AIC 计算层：处理一个 head 的所有 chunk，组织前缀预取、矩阵乘和 Cube 搬运。
template <typename stateType, bool hasGamma>
class CubeCompute : private FusedLayout {
public:
    __aicore__ inline explicit CubeCompute(KernelContext<stateType> &context) : context_(context) {}

    // BasicBlock 已选择融合路径；本函数完成当前 head 的全部 chunk 和跨核握手收尾。
    // 缓冲区绑定及临时矩阵仅在本类内部使用，不向调度层传递。
    __aicore__ inline void Run(uint32_t valueHead)
    {
        if (context_.task.seqLength == 0) {
            return;
        }
        const uint32_t qkHead = context_.QueryKeyHead(valueHead);
        const uint32_t chunkNum = context_.ChunkNum();
        Reset();
        NotifyVector<PIPE_MTE2, CubeToVectorEvent::ChunkSlotFree>(0);
        if (chunkNum > 1) {
            NotifyVector<PIPE_MTE2, CubeToVectorEvent::ChunkSlotFree>(1);
        }
        CopyPrefixQkGmToL1(qkHead, 0);
        ComputePrefixScores(0);
        for (uint32_t chunk = 0; chunk < chunkNum; ++chunk) {
            ComputeCubeChunk(valueHead, qkHead, chunk);
        }
        // 收回最后一次发布的归还信号，避免计数 token 遗留到同一核组处理的下一个 head。
        WaitForVector<VectorToCubeEvent::StateDeltaBufferFree>();
        WaitForVector<VectorToCubeEvent::PrefixScoresBufferFree>(0);
        if (chunkNum > 1) {
            WaitForVector<VectorToCubeEvent::PrefixScoresBufferFree>(1);
        }
    }

private:
    // L1 完整槽图见 FusedLayout；本类私有区按实际容量衔接：
    // Q:[0,16)KiB -> K:[16,32)KiB -> 公共A:[32,40)KiB -> lowerTmp:[40,42)KiB。
    // lowerTmp 是紧凑 BF16[32,32]，只占2KiB；到 rawKcd 起点48KiB之间的6KiB当前未使用。
    static constexpr uint32_t L1_Q_OFFSET = 0;
    static constexpr uint32_t L1_K_OFFSET = L1_Q_OFFSET + L1_TILE_BYTES;
    static constexpr uint32_t L1_LOWER_TMP_OFFSET = L1_INVERSE_OFFSET + CHUNK * CHUNK * sizeof(bfloat16_t);

    // L0C FP32 布局（半开区间，KiB）：
    // [  0, 64) 临时结果，M -> FIX；求逆[32,32](4KiB)、前缀/投影(32KiB)、delta[D,D](64KiB)复用
    // [ 64, 96) U -> R:[C,D]，跨 MMAD 保留并原地累加，最终 FIX -> UB
    // [ 96,128) O_inter -> O:[C,D]，跨 MMAD 保留并原地累加，最终 FIX -> GM
    // [128,256) 当前未使用；三个活动区分开，避免临时结果覆盖仍存活的R/O。
    // 起点由前一区域最大shape推导；不是要求按64/96KiB对齐。
    static constexpr uint32_t L0C_RESIDUAL_OFFSET = DIM * DIM * sizeof(float);
    static constexpr uint32_t L0C_OUTPUT_OFFSET = L0C_RESIDUAL_OFFSET + CHUNK * DIM * sizeof(float);

    // L0A/L0B 各64KiB，均分 [0,32)/[32,64) 两个stage，MTE1装入 -> M消费。
    // 每stage最多 BF16[D,D]=32KiB；按MMAD步骤轮换，stage编号不等于chunk槽编号。
    // ProjectStatePair 的两次MMAD共用一份L0B状态，第二次完成才释放。
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0A_STAGE_BYTES = ArchTag::L0A_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0B_STAGE_BYTES = ArchTag::L0B_SIZE / L0AB_STAGES;
    static constexpr uint32_t AIC_L0AB_MUTEX_BASE = 0;
    static constexpr uint32_t AIC_L0C_TMP_MUTEX = 2;
    static constexpr uint32_t AIC_L0C_RESIDUAL_MUTEX = 3;
    static constexpr uint32_t AIC_L0C_OUTPUT_MUTEX = 4;
    static constexpr uint32_t AIC_L1_QK_MUTEX_BASE = 5;
    static constexpr uint32_t AIC_L1_FIX_MUTEX = 7;
    // Fixpipe 以 const 引用接收模板配置，配置对象必须有静态存储期。
    static constexpr FixpipeConfig CGDR_CFG_NZ_UB = {CO2Layout::NZ, true};

    static_assert(L1_Q_OFFSET + CHUNK * DIM * sizeof(bfloat16_t) <= L1_K_OFFSET,
                  "AIC Q scratch overlaps K scratch");
    static_assert(L1_K_OFFSET + CHUNK * DIM * sizeof(bfloat16_t) <= L1_INVERSE_OFFSET,
                  "AIC Q/K scratch overlaps the shared inverse matrix");
    static_assert(L1_INVERSE_OFFSET + CHUNK * CHUNK * sizeof(bfloat16_t) <= L1_LOWER_TMP_OFFSET,
                  "AIC inverse product overlaps the shared inverse matrix");
    static_assert(L1_LOWER_TMP_OFFSET + HALF_CHUNK * HALF_CHUNK * sizeof(bfloat16_t) <= L1_RAW_KCD_OFFSET,
                  "AIC inverse product overlaps shared rawKcd");
    static_assert(CHUNK * 2 * DIM * sizeof(bfloat16_t) <= L0A_STAGE_BYTES,
                  "arch35 exact L0A tile exceeds one ping-pong stage");
    static_assert(DIM * DIM * sizeof(bfloat16_t) <= L0B_STAGE_BYTES,
                  "arch35 exact L0B tile exceeds one ping-pong stage");
    static_assert(DIM * DIM * sizeof(float) <= L0C_RESIDUAL_OFFSET,
                  "AIC temporary accumulator overlaps residual");
    static_assert(L0C_RESIDUAL_OFFSET + CHUNK * DIM * sizeof(float) <= L0C_OUTPUT_OFFSET,
                  "AIC residual accumulator overlaps output");
    static_assert(L0C_OUTPUT_OFFSET + CHUNK * DIM * sizeof(float) <= ArchTag::L0C_SIZE,
                  "AIC accumulators exceed L0C capacity");
    static_assert(AIC_L1_FIX_MUTEX < MUTEX_COUNT,
                  "arch35 exact AIC mutex layout exceeds its allocation");

    // 一个 chunk 的内部工作集；shape 为补零后的计算范围，不分配缓冲区。
    struct ChunkMatrices {
        LocalTensor<bfloat16_t> inverse;            // A:[C,C]
        LocalTensor<bfloat16_t> inverseProduct;     // (-L10)@A00:[C/2,C/2]
        LocalTensor<bfloat16_t> rawStateProjection; // -beta*exp(G)*K:[C,DIM]
        LocalTensor<bfloat16_t> weightedValue;      // beta*V:[C,DIM]
        LocalTensor<bfloat16_t> stateProjections;   // [Kcd;Q']:[2*C,DIM]
        LocalTensor<bfloat16_t> state;              // S:[DIM,DIM]，逻辑方向 [Dv,Dk]
        LocalTensor<bfloat16_t> residualBf16;       // R:[C,DIM] 的 Cube 输入副本
        LocalTensor<bfloat16_t> decayedKey;         // Kg:[C,DIM]
        LocalTensor<bfloat16_t> causalScores;      // M:[C,C]
        LocalTensor<float> residual;              // U -> R:[C,DIM]，L0C
        LocalTensor<float> output;                // O_inter -> O:[C,DIM]，L0C
    };

    __aicore__ inline ChunkMatrices BindChunkMatrices(uint32_t chunk)
    {
        const uint32_t base = (chunk & 1) * L1_SLOT_BYTES;
        return {BufferViews::L1<bfloat16_t>(base + L1_INVERSE_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_LOWER_TMP_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_RAW_KCD_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_VBETA_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_STATE_PROJECTIONS_OFFSET),
                BufferViews::L1<bfloat16_t>(L1_STATE_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_RESIDUAL_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_KG_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_MASKED_QKT_OFFSET),
                BufferViews::L0C<float>(L0C_RESIDUAL_OFFSET),
                BufferViews::L0C<float>(L0C_OUTPUT_OFFSET)};
    }

    __aicore__ inline void ComputeCubeChunk(uint32_t valueHead, uint32_t qkHead, uint32_t chunk)
    {
        const auto x = BindChunkMatrices(chunk);
        const uint32_t slot = chunk & 1;
        if (chunk + 1 < context_.ChunkNum()) {
            // 优化方案：前缀只依赖 Q/K，预取下块并与 AIV 求逆、当前状态递推重叠。
            CopyPrefixQkGmToL1(qkHead, chunk + 1);
            ComputePrefixScores(chunk + 1);
        }

        WaitForVector<VectorToCubeEvent::InverseInputsReady>(slot);
        CompleteInverse(x);                                      // A:[C,C]
        ComputeStateProjection(x);                               // Kcd=A@rawKcd:[C,Dk]
        ComputeTransformedValue(x);                              // U=A@(beta*V):[C,Dv]
        WaitForVector<VectorToCubeEvent::StateReady>();
        ProjectStatePair(x, slot);                               // R=U+Kcd@S.T(); O_inter=Q'@S.T()
        WaitForVector<VectorToCubeEvent::ResidualReady>();
        ComputeStateDelta(x, chunk);                             // delta=R.T()@Kg:[Dv,Dk]
        WaitForVector<VectorToCubeEvent::CausalScoresReady>(slot);
        AccumulateOutput(x);                                     // O=O_inter+M@R:[C,Dv]

        if (chunk + 2 < context_.ChunkNum()) {
            // L1 操作数已进入 L0，chunk+2 可复用当前槽；独立 L0C/FIX 继续完成 O 的写回。
            NotifyVector<PIPE_MTE1, CubeToVectorEvent::ChunkSlotFree>(slot);
        }
        CopyOutputL0CToGm(valueHead, chunk, context_.ChunkValidRows(chunk), x.output);
    }

    __aicore__ inline void Reset() { l0abStage_ = 0; }

    struct CubeOperands {
        LocalTensor<bfloat16_t> lhs;
        LocalTensor<bfloat16_t> rhs;
        MutexID mutex;
    };
    __aicore__ inline uint32_t NextL0AbStage()
    {
        const uint32_t stage = l0abStage_;
        l0abStage_ ^= 1;
        return stage;
    }

    // MTE2：GM 行主序 [rows,cols] -> L1 NZ；两个行跨度以元素/逻辑行数传入。
    __aicore__ inline void CopyGmToL1(GlobalTensor<bfloat16_t> gm, LocalTensor<bfloat16_t> l1, uint32_t rows,
                                      uint32_t cols, uint32_t srcRowStride, uint32_t dstRows)
    {
        Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = rows;
        params.dValue = cols;
        params.srcDValue = srcRowStride;
        params.dstNzC0Stride = MmAlign(dstRows, MM_BLOCK_CUBE);
        params.dstNzNStride = 1;
        params.srcNdMatrixStride = 0;
        params.dstNzMatrixStride = 0;
        DataCopy(l1, gm, params);
    }

    // MTE1：从 L1 NZ 的 (rowStart,colStart) 装入 [rows,cols]，按模板选择 L0 装载转置。
    template <bool ifTranspose>
    __aicore__ inline void CopyL1ToL0(LocalTensor<bfloat16_t> l0, LocalTensor<bfloat16_t> l1, uint32_t rows,
                                      uint32_t cols, uint32_t fullRows, uint32_t rowStart = 0, uint32_t colStart = 0)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = rowStart / MM_BLOCK_CUBE;
        params.kStartPosition = colStart / MM_NZ_FRACTAL;
        params.mStep = MmAlign(rows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.kStep = MmAlign(cols, MM_NZ_FRACTAL) / MM_NZ_FRACTAL;
        params.srcStride = MmAlign(fullRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.dstStride = ifTranspose ? params.kStep : params.mStep;
        params.ifTranspose = ifTranspose;
        LoadData(l0, l1, params);
    }

    // MTE1：L1 的指定行片 -> L0A 指定行片，用于拼接 [K前半,Q前半,K后半,Q后半]。
    __aicore__ inline void CopyRowsL1ToL0A(LocalTensor<bfloat16_t> l0a, LocalTensor<bfloat16_t> l1, uint32_t rows,
                                           uint32_t cols, uint32_t srcFullRows, uint32_t srcRowStart,
                                           uint32_t dstFullRows, uint32_t dstRowStart)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = srcRowStart / MM_BLOCK_CUBE;
        params.kStartPosition = 0;
        params.mStep = MmAlign(rows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.kStep = MmAlign(cols, MM_NZ_FRACTAL) / MM_NZ_FRACTAL;
        params.srcStride = MmAlign(srcFullRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.dstStride = MmAlign(dstFullRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.ifTranspose = false;
        LoadData(l0a[dstRowStart * MM_NZ_FRACTAL], l1, params);
    }

    // M：C[m,n] = A[m,k] @ B[k,n]；init=false 时加到已有 FP32 C，不插入量化。
    __aicore__ inline void Mmad(LocalTensor<float> l0c, LocalTensor<bfloat16_t> l0a, LocalTensor<bfloat16_t> l0b,
                                uint32_t m, uint32_t n, uint32_t k, bool init = true)
    {
        MmadParams params;
        params.m = m < MM_BLOCK_CUBE ? MM_BLOCK_CUBE : m;
        params.n = n;
        params.k = k;
        params.cmatrixInitVal = init;
        params.cmatrixSource = false;
        AscendC::Mmad(l0c, l0a, l0b, params);
    }

    template <typename outputType>
    __aicore__ inline void CopyResultRowsL0CToUb(LocalTensor<outputType> dst, LocalTensor<float> src,
                                               uint32_t rows, uint32_t cols)
    {
        // [rows,cols] -> 两个 AIV 的 [rows/2,cols] row-major UB；保持两条 FIX 的先后顺序。
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> copy;
        copy.nSize = cols;
        copy.mSize = rows / 2;
        copy.srcStride = MmAlign(rows, MM_BLOCK_CUBE);
        copy.dstStride = cols;
        copy.quantPre = std::is_same_v<outputType, float> ? QuantMode_t::NoQuant : QuantMode_t::F322BF16;
        copy.dualDstCtl = 0;
        copy.params.ndNum = 1;
        copy.params.srcNdStride = 0;
        copy.params.dstNdStride = 0;
        copy.subBlockId = 0;
        Fixpipe<outputType, float, CFG_ROW_MAJOR_UB>(dst, src, copy);
        copy.subBlockId = 1;
        Fixpipe<outputType, float, CFG_ROW_MAJOR_UB>(dst, src[rows / 2 * MM_NZ_FRACTAL], copy);
    }

    __aicore__ inline void CopyStateDeltaL0CToUb(LocalTensor<float> result, uint32_t chunk)
    {
        const MutexID resultMutex = MutexId(AIC_L0C_TMP_MUTEX);
        // 释放 ComputeStateDelta 保留的 M 锁，然后由 FIX 读取结果。
        Mutex::Unlock<PIPE_M>(resultMutex);
        Mutex::Lock<PIPE_FIX>(resultMutex);
        if (chunk != 0) {
            WaitForVector<VectorToCubeEvent::StateDeltaBufferFree>();
        }
        // FP32 接口直接传 FP32；BF16 接口先量化 delta，再由 AIV 升为 FP32 与主状态累加。
        const uint32_t deltaOffset = STATE_FP32 ? PrefixTransformsUbOffset(chunk & 1) : UB_STATE_DELTA_BF16_OFFSET;
        auto delta = BufferViews::Ub<stateType>(deltaOffset);
        CopyResultRowsL0CToUb(delta, result, DIM, DIM);
        NotifyVector<PIPE_FIX, CubeToVectorEvent::StateDeltaReady>();
        Mutex::Unlock<PIPE_FIX>(resultMutex);
    }

    __aicore__ inline void CopyPrefixScoresL0CToUb(LocalTensor<float> result, uint32_t slot, bool waitReusable)
    {
        const MutexID resultMutex = MutexId(AIC_L0C_TMP_MUTEX);
        // RunQkPrefixMm 已持有 PIPE_M；此处只释放一次，再取得 FIX 所有权。
        Mutex::Unlock<PIPE_M>(resultMutex);
        Mutex::Lock<PIPE_FIX>(resultMutex);
        if (waitReusable) {
            WaitForVector<VectorToCubeEvent::PrefixScoresBufferFree>(slot);
        }
        // [K前半,Q前半,K后半,Q后半]@K.T()，每 AIV 收到自己的 KKT/QKT:[C/2,C]。
        auto scores = BufferViews::Ub<bfloat16_t>(PrefixDirectOffset(slot));
        CopyResultRowsL0CToUb(scores, result, CHUNK * 2, CHUNK);
        NotifyVector<PIPE_FIX, CubeToVectorEvent::PrefixScoresReady>(slot);
        Mutex::Unlock<PIPE_FIX>(resultMutex);
    }

    __aicore__ inline void CopyResultL0CToL1(LocalTensor<bfloat16_t> dst, LocalTensor<float> l0c, uint32_t m, uint32_t n,
                                     uint32_t fullRows, MutexID l0cMutex, uint32_t rowStart = 0, uint32_t colStart = 0)
    {
        FixpipeParamsC310<CO2Layout::NZ> params;
        params.nSize = n;
        params.mSize = m;
        params.srcStride = MmAlign(m, MM_BLOCK_CUBE);
        params.dstStride = MmAlign(fullRows, MM_BLOCK_CUBE) * MM_NZ_FRACTAL;
        params.quantPre = QuantMode_t::F322BF16;

        // 顺序不可交换：先占用 L1 目标，再结束 MMAD，最后允许 FIX 读取 L0C。
        Mutex::Lock<PIPE_FIX>(MutexId(AIC_L1_FIX_MUTEX));
        Mutex::Unlock<PIPE_M>(l0cMutex);
        Mutex::Lock<PIPE_FIX>(l0cMutex);
        Fixpipe<bfloat16_t, float, CFG_NZ>(dst[GetZnOffset(rowStart, colStart, fullRows)], l0c, params);
        Mutex::Unlock<PIPE_FIX>(l0cMutex);
        Mutex::Unlock<PIPE_FIX>(MutexId(AIC_L1_FIX_MUTEX));
    }

    __aicore__ inline void CopyOutputL0CToGm(uint32_t nvId, uint32_t chunk, uint32_t validRows, LocalTensor<float> l0c)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
        params.nSize = context_.params.dv;
        params.mSize = validRows;
        params.srcStride = MmAlign(CHUNK, MM_BLOCK_CUBE);
        params.dstStride = context_.params.nv * context_.params.dv;
        params.quantPre = QuantMode_t::F322BF16;
        params.dualDstCtl = 0;
        params.params.ndNum = 1;
        params.params.srcNdStride = 0;
        params.params.dstNdStride = 0;

        const MutexID outputMutex = MutexId(AIC_L0C_OUTPUT_MUTEX);
        Mutex::Unlock<PIPE_M>(outputMutex);
        Mutex::Lock<PIPE_FIX>(outputMutex);
        const uint64_t outputOffset =
            (context_.task.tokenStart + static_cast<uint64_t>(chunk) * CHUNK) * context_.params.nv * context_.params.dv +
            static_cast<uint64_t>(nvId) * context_.params.dv;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(context_.params.out[outputOffset], l0c, params);
        Mutex::Unlock<PIPE_FIX>(outputMutex);
    }

    // 只绑定下一组 L0A/L0B 双缓冲；搬运与计算的 Lock/Unlock 由各阶段显式安排。
    __aicore__ inline CubeOperands NextOperands()
    {
        const uint32_t stage = NextL0AbStage();
        return {BufferViews::L0A<bfloat16_t>(stage * L0A_STAGE_BYTES),
                BufferViews::L0B<bfloat16_t>(stage * L0B_STAGE_BYTES),
                MutexId(AIC_L0AB_MUTEX_BASE + stage)};
    }

    // Kcd = A @ (-beta*exp(G)*K)：[C,C] @ [C,DIM] -> [C,DIM]。
    // 结果以 BF16 写入 stateProjections 前 C 行，后 C 行的 Q' 由 Vector 准备。
    __aicore__ inline void ComputeStateProjection(const ChunkMatrices &x)
    {
        const auto operands = NextOperands();
        auto result = BufferViews::L0C<float>(0);
        const MutexID resultMutex = MutexId(AIC_L0C_TMP_MUTEX);
        Mutex::Lock<PIPE_MTE1>(operands.mutex);
        // CompleteInverse 的 FIX 刚写回 A；在读取该 L1 区域前等待它完成。
        Mutex::Lock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        CopyL1ToL0<false>(operands.lhs, x.inverse, CHUNK, CHUNK, CHUNK);
        CopyL1ToL0<true>(operands.rhs, x.rawStateProjection, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        Mutex::Unlock<PIPE_MTE1>(operands.mutex);

        Mutex::Lock<PIPE_M>(operands.mutex);
        Mutex::Lock<PIPE_M>(resultMutex);
        Mmad(result, operands.lhs, operands.rhs, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_M>(operands.mutex);
        // CopyResultL0CToL1 释放 result 的 M 锁，再交给 FIX；保留此次独立 BF16 舍入。
        CopyResultL0CToL1(x.stateProjections, result, CHUNK, DIM, DIM, resultMutex);
    }

    // U = A @ (beta*V)：[C,C] @ [C,DIM] -> FP32 L0C [C,DIM]。
    __aicore__ inline void ComputeTransformedValue(const ChunkMatrices &x)
    {
        const auto operands = NextOperands();
        // A 已在上一阶段读过；beta*V 的就绪由 ComputeCubeChunk 的 InverseInputsReady 保证。
        Mutex::Lock<PIPE_MTE1>(operands.mutex);
        CopyL1ToL0<false>(operands.lhs, x.inverse, CHUNK, CHUNK, CHUNK);
        CopyL1ToL0<true>(operands.rhs, x.weightedValue, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(operands.mutex);

        Mutex::Lock<PIPE_M>(operands.mutex);
        Mutex::Lock<PIPE_M>(MutexId(AIC_L0C_RESIDUAL_MUTEX));
        Mmad(x.residual, operands.lhs, operands.rhs, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_M>(operands.mutex);
        // 保留 residual 的 M 锁：ProjectStatePair 原地累加，CopyResidualL0CToUb 最终释放。
    }

    // O += M @ R：[C,C] @ [C,DIM]；沿用 ProjectStatePair 保留的 FP32 L0C 和 M 锁。
    __aicore__ inline void AccumulateOutput(const ChunkMatrices &x)
    {
        const auto operands = NextOperands();
        // ComputeCubeChunk 已分别等待 Vector 的 CausalScoresReady 与 ResidualReady。
        Mutex::Lock<PIPE_MTE1>(operands.mutex);
        CopyL1ToL0<false>(operands.lhs, x.causalScores, CHUNK, CHUNK, CHUNK);
        CopyL1ToL0<true>(operands.rhs, x.residualBf16, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(operands.mutex);

        Mutex::Lock<PIPE_M>(operands.mutex);
        Mmad(x.output, operands.lhs, operands.rhs, CHUNK, DIM, CHUNK, false);
        Mutex::Unlock<PIPE_M>(operands.mutex);
        // CopyOutputL0CToGm 负责释放 output 的 M 锁并取得 FIX 锁。
    }

    // delta = R.T() @ Kg：[DIM,C] @ [C,DIM] -> [DIM,DIM]，逻辑 shape 为 [Dv,Dk]。
    __aicore__ inline void ComputeStateDelta(const ChunkMatrices &x, uint32_t chunk)
    {
        const auto operands = NextOperands();
        auto result = BufferViews::L0C<float>(0);
        const MutexID resultMutex = MutexId(AIC_L0C_TMP_MUTEX);
        Mutex::Lock<PIPE_MTE1>(operands.mutex);
        // 沿用 L1/FIX 缓冲区交接协议；R 的跨核就绪由 ComputeCubeChunk 单独等待。
        Mutex::Lock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        // A 的 true 表示 R.T()；B 的 true 是 NZ -> L0B 的装载方式，Kg 数学上不转置。
        CopyL1ToL0<true>(operands.lhs, x.residualBf16, CHUNK, DIM, CHUNK);
        CopyL1ToL0<true>(operands.rhs, x.decayedKey, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        Mutex::Unlock<PIPE_MTE1>(operands.mutex);

        Mutex::Lock<PIPE_M>(operands.mutex);
        Mutex::Lock<PIPE_M>(resultMutex);
        Mmad(result, operands.lhs, operands.rhs, DIM, DIM, CHUNK);
        Mutex::Unlock<PIPE_M>(operands.mutex);
        CopyStateDeltaL0CToUb(result, chunk); // 释放 result 的 M 锁，并由 FIX 分发给两个 AIV。
    }

    __aicore__ inline void CopyResidualL0CToUb(LocalTensor<float> l0cResidual, uint32_t slot)
    {
        auto residual = BufferViews::Ub<float>(PrefixTransformsUbOffset(slot));
        const MutexID residualMutex = MutexId(AIC_L0C_RESIDUAL_MUTEX);
        // ComputeTransformedValue 初始化 U 后持续持有 PIPE_M；ProjectStatePair 累加 Kcd@S.T()，在此交给 FIX。
        Mutex::Unlock<PIPE_M>(residualMutex);
        Mutex::Lock<PIPE_FIX>(residualMutex);
        // R32 保留 16 列分块布局，一条 FIX 按 M 平分给两个 AIV。
        // 每个 AIV 收到 [DIM/16, HALF_CHUNK, 16]，随后仅转一个 BF16 副本送回 L1。
        FixpipeParamsC310<CO2Layout::NZ> fix;
        fix.nSize = DIM;
        fix.mSize = CHUNK;
        fix.srcStride = CHUNK;
        fix.dstStride = HALF_CHUNK * MM_NZ_FRACTAL;
        fix.quantPre = QuantMode_t::NoQuant;
        fix.dualDstCtl = 1;
        fix.subBlockId = 0;
        Fixpipe<float, float, CGDR_CFG_NZ_UB>(residual, l0cResidual, fix);
        NotifyVector<PIPE_FIX, CubeToVectorEvent::ResidualReady>();
        Mutex::Unlock<PIPE_FIX>(residualMutex);
    }

    __aicore__ inline void ProjectStatePair(const ChunkMatrices &matrices, uint32_t slot)
    {
        auto l0cResidual = matrices.residual;
        auto l0cOutput = matrices.output;
        auto l1StateProjections = matrices.stateProjections;
        auto l1State = matrices.state;
        // 优化方案：两次 MMAD 共用 L0B 中的 S:[Dv,Dk]，减少一次状态装载。
        // 输入 l0cResidual=U:[C,Dv]，原地累加为 R=U+Kcd*S^T；l0cOutput=Q'*S^T:[C,Dv]。
        // STATE_FREE 在 L1->L0B 完成后即可发出；L0B 的 Mutex 则保持到两次矩阵乘均结束。
        const uint32_t stateStage = NextL0AbStage();
        const uint32_t outputStage = NextL0AbStage();
        const MutexID stateMutex = MutexId(AIC_L0AB_MUTEX_BASE + stateStage);
        const MutexID outputMutex = MutexId(AIC_L0AB_MUTEX_BASE + outputStage);
        auto l0aState = BufferViews::L0A<bfloat16_t>(stateStage * L0A_STAGE_BYTES);
        auto l0aOutput = BufferViews::L0A<bfloat16_t>(outputStage * L0A_STAGE_BYTES);
        auto l0bState = BufferViews::L0B<bfloat16_t>(stateStage * L0B_STAGE_BYTES);

        Mutex::Lock<PIPE_MTE1>(stateMutex);
        Mutex::Lock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        CopyL1ToL0<false>(l0aState, l1StateProjections, CHUNK, DIM, DIM, 0, 0);
        CopyL1ToL0<false>(l0bState, l1State, DIM, DIM, DIM);
        Mutex::Unlock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        NotifyVector<PIPE_MTE1, CubeToVectorEvent::StateBufferFree>();
        Mutex::Unlock<PIPE_MTE1>(stateMutex);
        Mutex::Lock<PIPE_M>(stateMutex);
        Mmad(l0cResidual, l0aState, l0bState, CHUNK, DIM, DIM, false);
        CopyResidualL0CToUb(l0cResidual, slot);

        Mutex::Lock<PIPE_MTE1>(outputMutex);
        CopyL1ToL0<false>(l0aOutput, l1StateProjections, CHUNK, DIM, DIM, CHUNK, 0);
        Mutex::Unlock<PIPE_MTE1>(outputMutex);
        Mutex::Lock<PIPE_M>(outputMutex);
        Mutex::Lock<PIPE_M>(MutexId(AIC_L0C_OUTPUT_MUTEX));
        Mmad(l0cOutput, l0aOutput, l0bState, CHUNK, DIM, DIM, true);
        // 两次 MMAD 均完成后，先归还共享 S 所在的槽，再归还 output 操作数所在的槽。
        Mutex::Unlock<PIPE_M>(stateMutex);
        Mutex::Unlock<PIPE_M>(outputMutex);
        // output 的 M 锁继续保留，供 AccumulateOutput 累加，最后由 CopyOutputL0CToGm 释放。
    }

    __aicore__ inline void RunQkPrefixMm(LocalTensor<bfloat16_t> l1Q, LocalTensor<bfloat16_t> l1K, uint32_t prefixSlot,
                                         bool waitDirectFree)
    {
        // 优化方案：L0A 按 [K前32行,Q前32行,K后32行,Q后32行] 拼接，一次 [2*C,DIM]*K^T
        // 同时得到 KKT/QKT:[2*C,C]，两个分数矩阵各为 [C,C]。
        // 结果沿 M 平分后，每个 AIV 恰好收到自己 32 个 token 的 KKT 和 QKT，复用同一份 L0B 的 K。
        const uint32_t stage = NextL0AbStage();
        const MutexID l0abMutex = MutexId(AIC_L0AB_MUTEX_BASE + stage);
        const MutexID qkMutex = MutexId(AIC_L1_QK_MUTEX_BASE + prefixSlot);
        auto l0a = BufferViews::L0A<bfloat16_t>(stage * L0A_STAGE_BYTES);
        auto l0b = BufferViews::L0B<bfloat16_t>(stage * L0B_STAGE_BYTES);
        auto l0c = BufferViews::L0C<float>(0);

        Mutex::Lock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_MTE1>(qkMutex);
        CopyRowsL1ToL0A(l0a, l1K, HALF_CHUNK, DIM, CHUNK, 0, CHUNK * 2, 0);
        CopyRowsL1ToL0A(l0a, l1Q, HALF_CHUNK, DIM, CHUNK, 0, CHUNK * 2, HALF_CHUNK);
        CopyRowsL1ToL0A(l0a, l1K, HALF_CHUNK, DIM, CHUNK, HALF_CHUNK, CHUNK * 2, CHUNK);
        CopyRowsL1ToL0A(l0a, l1Q, HALF_CHUNK, DIM, CHUNK, HALF_CHUNK, CHUNK * 2, CHUNK + HALF_CHUNK);
        CopyL1ToL0<false>(l0b, l1K, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(qkMutex);
        Mutex::Unlock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_M>(l0abMutex);
        Mutex::Lock<PIPE_M>(MutexId(AIC_L0C_TMP_MUTEX));
        Mmad(l0c, l0a, l0b, CHUNK * 2, CHUNK, DIM);
        Mutex::Unlock<PIPE_M>(l0abMutex);
        CopyPrefixScoresL0CToUb(l0c, prefixSlot, waitDirectFree);
    }

    __aicore__ inline void CopyPrefixQkGmToL1(uint32_t qkHead, uint32_t chunk)
    {
        const uint32_t slot = chunk & 1;
        const uint32_t slotBase = slot * L1_SLOT_BYTES;
        const uint32_t validRows = context_.ChunkValidRows(chunk);
        const uint64_t tokenStart = context_.task.tokenStart + static_cast<uint64_t>(chunk) * CHUNK;
        const uint64_t qkOffset = tokenStart * context_.params.nk * context_.params.dk + qkHead * context_.params.dk;
        auto l1Q = BufferViews::L1<bfloat16_t>(slotBase + L1_Q_OFFSET);
        auto l1K = BufferViews::L1<bfloat16_t>(slotBase + L1_K_OFFSET);

        const MutexID qkMutex = MutexId(AIC_L1_QK_MUTEX_BASE + slot);
        Mutex::Lock<PIPE_MTE2>(qkMutex);
        if (validRows < CHUNK || context_.params.dk < DIM) {
            // 先清完整物理 Q/K，再覆盖有效矩形；尾部零值保证 padding token 不影响分数和状态。
            constexpr uint16_t L1_QK_BLOCKS = CHUNK * DIM * sizeof(bfloat16_t) / BLOCK_SIZE;
            InitConstValueParams<bfloat16_t> zeroParams{1, L1_QK_BLOCKS, 0, static_cast<bfloat16_t>(0)};
            Fill(l1Q, zeroParams);
            Fill(l1K, zeroParams);
        }
        CopyGmToL1(context_.params.query[qkOffset], l1Q, validRows, context_.params.dk, context_.params.nk * context_.params.dk, CHUNK);
        CopyGmToL1(context_.params.key[qkOffset], l1K, validRows, context_.params.dk, context_.params.nk * context_.params.dk, CHUNK);
        Mutex::Unlock<PIPE_MTE2>(qkMutex);
    }

    __aicore__ inline void ComputePrefixScores(uint32_t chunk)
    {
        const uint32_t slot = chunk & 1;
        const uint32_t slotBase = slot * L1_SLOT_BYTES;
        auto l1Q = BufferViews::L1<bfloat16_t>(slotBase + L1_Q_OFFSET);
        auto l1K = BufferViews::L1<bfloat16_t>(slotBase + L1_K_OFFSET);

        RunQkPrefixMm(l1Q, l1K, slot, chunk >= 2);
    }

    // A=(I+L)^(-1)：Vector 已算出两个 [32,32] 对角块 A00/A11，并在 A10 位置写入 -L10。
    // A10 = A11 @ ((-L10) @ A00)。两次矩乘分别写回 BF16，不能合并而改变舍入点。
    __aicore__ inline void CompleteInverse(const ChunkMatrices &x)
    {
        auto result = BufferViews::L0C<float>(0);
        const MutexID resultMutex = MutexId(AIC_L0C_TMP_MUTEX);
        {
            const auto operands = NextOperands();
            // 输入由 ComputeCubeChunk 的 InverseInputsReady 保证，此处没有前置 FIX 写回。
            Mutex::Lock<PIPE_MTE1>(operands.mutex);
            // 子块 shape 均为 [32,32]，但父矩阵 NZ 的物理行跨度仍为 C=64。
            CopyL1ToL0<false>(operands.lhs, x.inverse, HALF_CHUNK, HALF_CHUNK, CHUNK, HALF_CHUNK, 0);
            CopyL1ToL0<true>(operands.rhs, x.inverse, HALF_CHUNK, HALF_CHUNK, CHUNK, 0, 0);
            Mutex::Unlock<PIPE_MTE1>(operands.mutex);
            Mutex::Lock<PIPE_M>(operands.mutex);
            Mutex::Lock<PIPE_M>(resultMutex);
            Mmad(result, operands.lhs, operands.rhs, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK);
            Mutex::Unlock<PIPE_M>(operands.mutex);
            CopyResultL0CToL1(x.inverseProduct, result, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK, resultMutex);
        }
        {
            const auto operands = NextOperands();
            Mutex::Lock<PIPE_MTE1>(operands.mutex);
            // 等待上一步 FIX 写完 inverseProduct，再加载 A11 和这个紧凑 [32,32] 中间积。
            Mutex::Lock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
            CopyL1ToL0<false>(operands.lhs, x.inverse, HALF_CHUNK, HALF_CHUNK, CHUNK, HALF_CHUNK, HALF_CHUNK);
            CopyL1ToL0<true>(operands.rhs, x.inverseProduct, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK);
            Mutex::Unlock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
            Mutex::Unlock<PIPE_MTE1>(operands.mutex);
            Mutex::Lock<PIPE_M>(operands.mutex);
            Mutex::Lock<PIPE_M>(resultMutex);
            Mmad(result, operands.lhs, operands.rhs, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK);
            Mutex::Unlock<PIPE_M>(operands.mutex);
            // A10 覆盖原来的 -L10；目标仍保留父矩阵的 NZ 行跨度。
            CopyResultL0CToL1(x.inverse, result, HALF_CHUNK, HALF_CHUNK, CHUNK, resultMutex, HALF_CHUNK, 0);
        }
    }

    static constexpr bool STATE_FP32 = std::is_same_v<stateType, float>;
    KernelContext<stateType> &context_;
    uint32_t l0abStage_;
};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_CUBE_COMPUTE_H
