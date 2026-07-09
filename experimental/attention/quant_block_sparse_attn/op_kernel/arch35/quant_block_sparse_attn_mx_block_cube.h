/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_block_sparse_attn_mx_block_cube.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化 cube 路径。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_

#include "kernel_operator.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "common/CopyInL1.h"
#include "common/FixpipeOut.h"
#include "common/buffer_manager.h"
#include "common/buffers_policy.h"
#include "common/matmul.h"
#include "common/util_regbase_mx.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace fa_base_matmul;
using namespace regbasemx;

namespace BaseApi {
template <bool useDn = true>
class QuantBlockSparseAttnMxBlockCube {
public:
    using INPUT_T = fp8_e4m3fn_t;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = float;
    using OUTPUT_T = bfloat16_t;
    using MxBmm2Buf = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    static constexpr bool USE_DN = useDn;
    static constexpr uint32_t M_BASE = 128U;
    static constexpr uint32_t S2_BASE = 512U;
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t D_BASE = 128U;
    static constexpr uint32_t DV_BASE = 128U;
    // L1 中 e8m0 scale tensor 通过 bfloat16 reinterpret 方式搬运。
    // 当 D=128 且 shape 为 [D/64, 2] 时，单行有 4 个 e8m0 scale 元素；
    // 在当前紧凑 L1 寻址中，相当于每 32 个 fp8 数据元素对应 1 个 scale 元素。
    static constexpr uint32_t MX_SCALE_GROUP = 32U;
    // Value scale 为 per-channel-group：GM shape 是 [T/64, N, DV, 2]。
    // 一行 value scale 覆盖 64 个 value token。
    static constexpr uint32_t MX_TOKEN_GROUP = 64U;
    static constexpr uint32_t MX_SCALE_LAST_DIM = 2U;
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};

    __aicore__ inline void Init(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *qScale, __gm__ uint8_t *kScale, __gm__ uint8_t *vScale)
    {
        if ASCEND_IS_AIC {
            tPipe_ = pipe;
            l1BufferManager_ = l1BuffMgr;
            queryGm_.SetGlobalBuffer((__gm__ INPUT_T *)query);
            keyGm_.SetGlobalBuffer((__gm__ INPUT_T *)key);
            valueGm_.SetGlobalBuffer((__gm__ INPUT_T *)value);
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
            qScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)qScale);
            kScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)kScale);
            vScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)vScale);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIC {
            static_assert(USE_DN,
                          "MXFullQuantMode cube currently implements DN copy/load path; "
                          "use_dn is kept for future non-DN extension");
            // cube 侧 L1 buffer 的分配沿用 common MX full-quant 的思路：
            // 数据 tensor 与 e8m0 scale tensor 放在同一块 L1 内，随后 LoadDataToL0*Mx
            // 同时读取 data/scale，在搬到 L0A/L0B 时完成随路反量化。
            // 这样可以避免在 UB/L1 中显式展开 fp32 Q/K/V，节省带宽和临时空间。
            //
            // Q/K/V 的 fp8 数据和对应 e8m0 scale 在同一个 L1 buffer 中连续存放。
            // 每次 matmul 前根据对齐后的数据 tile 大小计算 scale 偏移：
            //   Q: [M, D] + QScale [M, D/64, 2]
            //      Q 与 sparse S2 无关，同一个 S1 tile 的多个 sparse block 可以复用；
            //   K: [S2/2, D] + KScale [S2/2, D/64, 2]
            //      K 每个 512 logical tile 拆成两个 256 subLoop；当 sparse block 为
            //      128/64 时，多个 sparse block 会先拼接到同一个 K L1 tile；
            //   V: 对于 256-token sub tile，为 [S2/2, DV] + VScale [S2/128, DV, 2]
            //      VScale 按 64 token 分组，所以 256-token sub tile 只需要 4 行 scale。
            constexpr uint32_t qSize = M_BASE * D_BASE * sizeof(INPUT_T);
            constexpr uint32_t qScaleSize = M_BASE * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
            constexpr uint32_t kvSize = S2_SPLIT * D_BASE * sizeof(INPUT_T);
            constexpr uint32_t kvScaleSize = S2_SPLIT * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
            constexpr uint32_t valueScaleSize = (S2_SPLIT / MX_TOKEN_GROUP) * DV_BASE * MX_SCALE_LAST_DIM *
                                                sizeof(SCALE_T);

            // Q 使用双 buffer：第一轮 S2 loop 拷贝 Q/QScale，后续 sparse block 复用上一块；
            // 双 buffer 还能和 MTE2/MTE1 事件配合，避免覆盖仍在被 matmul 读取的 Q。
            l1QBuffers_.Init(*l1BufferManager_, qSize + qScaleSize);
            // K 使用 4 buffer：BMM1 两个 256 subLoop 都要频繁从 PA BNBD 拷贝 K/KScale，
            // 4 buffer 用于覆盖 MTE2 拷贝、MTE1 搬运到 L0、matmul 计算之间的流水。
            // 这个策略与 BSA FP8/参考 common MX 中“KV 比 Q 更频繁切换”的背景一致。
            l1KVBuffers_.Init(*l1BufferManager_, kvSize + kvScaleSize);
            // V 使用双 buffer：BMM2 也按 256-token K 轴分块，但一个 C2 task 内顺序消费
            // V/VScale，双 buffer 足够覆盖 V 的 GM->L1 拷贝和当前 matmul 读取。
            l1VBuffers_.Init(*l1BufferManager_, kvSize + valueScaleSize);

            // L0A/L0B/L0C 采用独立 BufferManager，避免和 L1/UB 的跨核 buffer 生命周期混在一起。
            // 64KB 的 L0A/L0B manager 供双 buffer 轮换，单块 32KB 可以容纳当前 128/256/128
            // 模板下的一次 LoadData tile；L0C manager 预留 256KB，单块 64KB 用于 fp32 C 矩阵。
            l0aBufferManager_.Init(tPipe_, 64U * 1024U);
            l0bBufferManager_.Init(tPipe_, 64U * 1024U);
            l0cBufferManager_.Init(tPipe_, 256U * 1024U);
            mmL0ABuffers_.Init(l0aBufferManager_, 32U * 1024U);
            mmL0BBuffers_.Init(l0bBufferManager_, 32U * 1024U);
            mmL0CBuffers_.Init(l0cBufferManager_, 64U * 1024U);
        }
    }

    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                       MxRunInfo &runInfo, const MxConstInfo &constInfo, uint32_t subLoop)
    {
        // BMM1 计算一块 fp32 attention-score：
        //   K[subLoop: 256, D] x Q^T[D, 128] -> C1[256, 128]。
        // 当 sparse block size 为 128/64 时，当前 256 宽 K tile 会由 2/4 个
        // sparse block 按 sparseIndices 顺序拼接得到。
        // Q/K 为 fp8_e4m3fn；qScale/kScale 为 fp8_e8m0，由 LoadDataToL0*Mx 消费，
        // 因此反量化发生在 L1->L0 搬运过程中，而不是在 L1 或 UB 中显式展开 Q/K。
        const uint32_t s2CalcSize = GetSubLoopS2Size(runInfo, subLoop);
        if (unlikely(s2CalcSize == 0U)) {
            outputBuf.WaitCrossCore();
            outputBuf.SetCrossCore();
            return;
        }

        Buffer<BufferType::L1> qBuf;
        const uint32_t qScaleOffset = AlignUp32(runInfo.actMSize) * constInfo.dSize;
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0U)) {
            // Q 与 logical S2 tile 无关，因此同一个 S1 tile 的第一个 S2 loop 拷贝一次，
            // 后续 tile 复用前一次 L1 Q buffer。
            // Query GM 为 TND：数据用 ND2NZ；Q scale 用 DN2NZ，以匹配 MX matmul
            // 右操作数转置后的读取布局。
            qBuf = l1QBuffers_.Get();
            qBuf.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<INPUT_T> qTensor = qBuf.GetTensor<INPUT_T>();
            CopyToL1Nd2Nz<INPUT_T>(qTensor, queryGm_[runInfo.queryOffset], runInfo.actMSize, constInfo.dSize,
                                   constInfo.n2GD);

            LocalTensor<SCALE_T> qScaleTensor = qBuf.GetTensor<SCALE_T>(qScaleOffset);
            CopyScaleToL1Dn2Nz<SCALE_T>(qScaleTensor, qScaleGm_[runInfo.queryScaleOffset],
                                        constInfo.queryScaleDSize * constInfo.scaleLastDim, runInfo.actMSize,
                                        constInfo.qScaleN1D);
            qBuf.Set<HardEvent::MTE2_MTE1>();
        } else {
            qBuf = l1QBuffers_.GetPre();
            qBuf.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> kBuf = l1KVBuffers_.Get();
        kBuf.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<INPUT_T> kTensor = kBuf.GetTensor<INPUT_T>();
        // K 依赖 sparse/PA。sparseIndices 给出逻辑 KV block；
        // block_table 将每个逻辑 block 映射到物理 BNBD page。每个 BMM1 subLoop
        // 拷贝并拼接 256 行。
        CopyKeyToL1(kTensor, runInfo, constInfo, s2CalcSize, subLoop);

        const uint32_t kScaleOffset = AlignUp32(s2CalcSize) * constInfo.dSize;
        LocalTensor<SCALE_T> kScaleTensor = kBuf.GetTensor<SCALE_T>(kScaleOffset);
        // K scale 与 K 数据使用相同的 sparse/PA token 选择，但其 GM 单行宽度为 [D/64, 2]。
        // 这里 K 作为左操作数参与 MX matmul，因此 scale 使用 DN2NZ 搬运。
        CopyKeyScaleToL1(kScaleTensor, runInfo, constInfo, s2CalcSize, subLoop);
        kBuf.Set<HardEvent::MTE2_MTE1>();

        kBuf.Wait<HardEvent::MTE2_MTE1>();
        qBuf.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers_.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();
        // MatmulFullMX 将 fp8 数据 tensor 和 e8m0 scale tensor 一起传给 MX LoadData 重载。
        // L0 视图类型 mx_fp8_e4m3_t 用于选择硬件随路反量化路径。
        MMParam param = MakeMMParam(s2CalcSize, runInfo.actMSize, constInfo.dSize, false, true);
        MatmulFullMX<INPUT_T, INPUT_T, MM_T, S2_SPLIT, M_BASE, D_BASE, ABLayout::MK, ABLayout::KN,
                     BuffersPolicyDB<BufferType::L0A>, BuffersPolicyDB<BufferType::L0B>, SCALE_T, SCALE_T,
                     mx_fp8_e4m3_t, mx_fp8_e4m3_t>(kTensor, qBuf.GetTensor<INPUT_T>(), mmL0ABuffers_,
                                                   mmL0BBuffers_, mm1ResL0C.GetTensor<MM_T>(), param,
                                                   kScaleTensor, qBuf.GetTensor<SCALE_T>(qScaleOffset));

        if (unlikely(runInfo.isLastS2Loop && IsLastSubLoop(runInfo, subLoop))) {
            qBuf.Set<HardEvent::MTE1_MTE2>();
        }
        kBuf.Set<HardEvent::MTE1_MTE2>();
        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();
        // C1 结果以 V1 softmax 所需的 row-major 格式写入 UB。
        FixpipeMm1(outputBuf.GetTensor<MM_T>(), mm1ResL0C.GetTensor<MM_T>(), runInfo, s2CalcSize);
        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void IterateBmm2(MxBmm2Buf &outputBuf,
                                       BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,
                                       MxRunInfo &runInfo, const MxConstInfo &constInfo)
    {
        // BMM2 消费两个 V1 subLoop 生成的完整 P tile：
        //   P[128, 512] x V[512, DV] -> O[128, DV]。
        // P 已由 V1 量化为 fp8_e4m3fn，PScale(e8m0) 紧跟在同一个 L1 buffer 后半段。
        // V/VScale 按 256-token 分块从 PA BNBD page 中搬入；当 sparse block 为
        // 128/64 时，同样按 sparseIndices 顺序拼接多个 block。
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> pBuf = inputBuf.GetCube();
        pBuf.WaitCrossCore();

        constexpr uint32_t pScaleOffset = M_BASE * S2_BASE;
        if (runInfo.actSingleLoopS2Size > S2_SPLIT) {
            // logical S2 tile 超过 256 token 时，V1 会通过两个 subLoop 写入 PScale。
            // C2 读取前先把后半部分 padding 初始化为 e8m0(1.0)，避免 tail/padding lane
            // 改变 MX 反量化 scale。
            constexpr uint32_t int16Bytes = 2U;
            LocalTensor<int16_t> pScalePadTensor =
                pBuf.GetTensor<int16_t>(pScaleOffset / int16Bytes + M_BASE * S2_SPLIT / MX_SCALE_GROUP / int16Bytes);
            InitConstValue(pScalePadTensor,
                           {1, M_BASE * S2_SPLIT / MX_SCALE_GROUP * sizeof(SCALE_T) / 32U, 0, 0x7f7f});
        }
        LocalTensor<SCALE_T> pScaleTensor = pBuf.GetTensor<SCALE_T>(pScaleOffset);

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers_.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();
        constexpr uint32_t baseK = S2_SPLIT;
        constexpr uint64_t l1BaseKOffset = static_cast<uint64_t>(baseK) * M_BASE;
        constexpr uint64_t l1ScaleOffset = l1BaseKOffset / MX_SCALE_GROUP;
        const uint32_t kLoops = (runInfo.actSingleLoopS2Size + baseK - 1U) / baseK;
        for (uint32_t kIdx = 0U; kIdx < kLoops; ++kIdx) {
            const uint32_t realK = (kIdx == kLoops - 1U) ? runInfo.actSingleLoopS2Size - kIdx * baseK : baseK;
            Buffer<BufferType::L1> vBuf = l1VBuffers_.Get();
            vBuf.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<INPUT_T> vTensor = vBuf.GetTensor<INPUT_T>();
            // Value 行使用 ND2NZ 搬运。由于 value scale 按 token group 分组，且 MX LoadData
            // 需要对齐后的 K 轴，tail 行需要补零到 64-token 对齐。
            InitValuePadding(vTensor, realK, constInfo.dSizeV);
            CopyValueToL1(vTensor, runInfo, constInfo, realK, kIdx);

            const uint32_t vScaleOffset = AlignUp64(realK) * constInfo.dSizeV;
            LocalTensor<SCALE_T> vScaleTensor = vBuf.GetTensor<SCALE_T>(vScaleOffset);
            // V scale 为 per-channel-group，shape 是 [T/64, N, DV, 2]。
            // 因此其 PA 索引按 token group 前进，而不是按原始 token 行前进。
            CopyValueScaleToL1(vScaleTensor, runInfo, constInfo, realK, kIdx);
            vBuf.Set<HardEvent::MTE2_MTE1>();
            vBuf.Wait<HardEvent::MTE2_MTE1>();

            MMParam param = MakeMMParam(M_BASE, constInfo.dSizeV, AlignUp64(realK), true, false, kIdx == 0U,
                                        kIdx == 0U);
            // 左操作数 P 在 matmul 视图中转置；右操作数 V 保持 KN。
            // 两个操作数都携带 e8m0 scale，因此 C2 也走 MX LoadData 路径。
            MatmulFullMX<INPUT_T, INPUT_T, MM_T, M_BASE, DV_BASE, baseK, ABLayout::MK, ABLayout::KN,
                         BuffersPolicyDB<BufferType::L0A>, BuffersPolicyDB<BufferType::L0B>, SCALE_T, SCALE_T,
                         mx_fp8_e4m3_t, mx_fp8_e4m3_t>(pBuf.GetTensor<INPUT_T>()[kIdx * l1BaseKOffset], vTensor,
                                                       mmL0ABuffers_, mmL0BBuffers_, mm2ResL0C.GetTensor<MM_T>(),
                                                       param, pScaleTensor[kIdx * l1ScaleOffset], vScaleTensor);
            vBuf.Set<HardEvent::MTE1_MTE2>();
        }

        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();
        outputBuf.WaitCrossCore();
        FixpipeMm2(outputBuf.GetTensor<MM_T>(), mm2ResL0C.GetTensor<MM_T>(), runInfo, constInfo);
        mm2ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t value) const
    {
        return (value + 31U) >> 5 << 5;
    }

    __aicore__ inline uint32_t AlignUp64(uint32_t value) const
    {
        return (value + 63U) >> 6 << 6;
    }

    __aicore__ inline uint32_t GetSubLoopS2Size(const MxRunInfo &runInfo, uint32_t subLoop) const
    {
        if (runInfo.actSingleLoopS2Size <= S2_SPLIT) {
            return subLoop == 0U ? runInfo.actSingleLoopS2Size : 0U;
        }
        return subLoop == 0U ? S2_SPLIT : runInfo.actSingleLoopS2Size - S2_SPLIT;
    }

    __aicore__ inline bool IsLastSubLoop(const MxRunInfo &runInfo, uint32_t subLoop) const
    {
        return (runInfo.actSingleLoopS2Size <= S2_SPLIT) || (subLoop != 0U);
    }

    __aicore__ inline uint32_t GetBlockOverlapStart(uint32_t tileStart, uint32_t blockStart) const
    {
        return tileStart > blockStart ? tileStart : blockStart;
    }

    __aicore__ inline uint32_t GetBlockOverlapEnd(uint32_t tileEnd, uint32_t blockEnd) const
    {
        return tileEnd < blockEnd ? tileEnd : blockEnd;
    }

    __aicore__ inline void CopyKeyToL1(LocalTensor<INPUT_T> &dstTensor, const MxRunInfo &runInfo,
                                      const MxConstInfo &constInfo, uint32_t s2CalcSize, uint32_t subLoop)
    {
        // Key GM layout 为 PA BNBD：[physicalBlock, kvHead, blockSize, D]。
        // 当前 256-token subLoop 可能跨多个 sparse block。这里参考 BSA FP8 的拼接方式：
        // 每拷完一个 sparse block 片段，就按 L1 NZ-like 行偏移把下一个片段接在后面。
        const uint32_t tileStart = subLoop * S2_SPLIT;
        const uint32_t tileEnd = tileStart + s2CalcSize;
        constexpr uint32_t blockElementCnt = 32U / sizeof(INPUT_T);
        uint32_t sparseTileOffset = 0U;
        uint32_t copiedRowCount = 0U;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = sparseTileOffset;
            const uint32_t blockEnd = blockStart + blockRealSize;
            sparseTileOffset = blockEnd;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;

            Position startPos;
            startPos.bIdx = runInfo.bIdx;
            startPos.n2Idx = runInfo.n2Idx;
            startPos.s2Offset = static_cast<uint32_t>(runInfo.sparseBlockTokenOffset[i]) + copyStartInBlock;
            startPos.dIdx = 0U;

            PAShape shape;
            shape.blockSize = constInfo.kvSparseBlockSize;
            shape.headNum = constInfo.n2Size;
            shape.headDim = constInfo.dSize;
            shape.actHeadDim = constInfo.dSize;
            shape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
            shape.copyRowNum = copyRowNum;
            shape.copyRowNumAlign = AlignUp32(s2CalcSize);
            shape.pageStride = constInfo.paBlockStride;
            shape.rowStride = constInfo.dSize;
            LocalTensor<INPUT_T> keyDstTensor = dstTensor[copiedRowCount * blockElementCnt];
            GmCopyInToL1PA<INPUT_T>(keyDstTensor, keyGm_, blockTableGm_, KVLAYOUT::BNBD, shape, startPos);
            copiedRowCount += copyRowNum;
        }
    }

    __aicore__ inline void CopyKeyScaleToL1(LocalTensor<SCALE_T> &dstTensor, const MxRunInfo &runInfo,
                                           const MxConstInfo &constInfo, uint32_t s2CalcSize, uint32_t subLoop)
    {
        // Key scale 与 Key 数据使用相同的 sparse block 拼接顺序。
        // GM 逻辑 shape 为 [paBlockNumSum * blockSize, kvHead, D/64, 2]。
        const uint32_t tileStart = subLoop * S2_SPLIT;
        const uint32_t tileEnd = tileStart + s2CalcSize;
        constexpr uint32_t blockElementCnt = 32U / sizeof(SCALE_T);
        uint32_t sparseTileOffset = 0U;
        uint32_t copiedRowCount = 0U;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = sparseTileOffset;
            const uint32_t blockEnd = blockStart + blockRealSize;
            sparseTileOffset = blockEnd;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;

            Position startPos;
            startPos.bIdx = runInfo.bIdx;
            startPos.n2Idx = runInfo.n2Idx;
            startPos.s2Offset = static_cast<uint32_t>(runInfo.sparseBlockTokenOffset[i]) + copyStartInBlock;
            startPos.dIdx = 0U;

            PAShape shape;
            shape.blockSize = constInfo.kvSparseBlockSize;
            shape.headNum = constInfo.n2Size;
            shape.headDim = constInfo.keyScaleDSize * constInfo.scaleLastDim;
            shape.actHeadDim = shape.headDim;
            shape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
            shape.copyRowNum = copyRowNum;
            shape.copyRowNumAlign = AlignUp32(s2CalcSize);
            shape.pageStride = constInfo.kvSparseBlockSize * constInfo.kScaleN2D;
            shape.rowStride = constInfo.kScaleN2D;
            LocalTensor<SCALE_T> keyScaleDstTensor = dstTensor[copiedRowCount * blockElementCnt];
            GmFlatScaleCopyInToL1PAForDN<SCALE_T>(keyScaleDstTensor, kScaleGm_, blockTableGm_, shape, startPos);
            copiedRowCount += copyRowNum;
        }
    }

    __aicore__ inline void CopyValueToL1(LocalTensor<INPUT_T> &dstTensor, const MxRunInfo &runInfo,
                                        const MxConstInfo &constInfo, uint32_t realK, uint32_t kIdx)
    {
        // Value 数据与 Key 数据使用相同的 PA BNBD 寻址方式，但最后一维可以是 dSizeV。
        // kIdx 选择当前 512-token logical tile 的前/后 256-token 部分。
        const uint32_t tileStart = kIdx * S2_SPLIT;
        const uint32_t tileEnd = tileStart + realK;
        constexpr uint32_t blockElementCnt = 32U / sizeof(INPUT_T);
        uint32_t sparseTileOffset = 0U;
        uint32_t copiedRowCount = 0U;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = sparseTileOffset;
            const uint32_t blockEnd = blockStart + blockRealSize;
            sparseTileOffset = blockEnd;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;

            Position startPos;
            startPos.bIdx = runInfo.bIdx;
            startPos.n2Idx = runInfo.n2Idx;
            startPos.s2Offset = static_cast<uint32_t>(runInfo.sparseBlockTokenOffset[i]) + copyStartInBlock;
            startPos.dIdx = 0U;

            PAShape shape;
            shape.blockSize = constInfo.kvSparseBlockSize;
            shape.headNum = constInfo.n2Size;
            shape.headDim = constInfo.dSizeV;
            shape.actHeadDim = constInfo.dSizeV;
            shape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
            shape.copyRowNum = copyRowNum;
            shape.copyRowNumAlign = AlignUp64(realK);
            shape.pageStride = constInfo.paBlockStride;
            shape.rowStride = constInfo.dSizeV;
            LocalTensor<INPUT_T> valueDstTensor = dstTensor[copiedRowCount * blockElementCnt];
            GmCopyInToL1PA<INPUT_T>(valueDstTensor, valueGm_, blockTableGm_, KVLAYOUT::BNBD, shape, startPos);
            copiedRowCount += copyRowNum;
        }
    }

    __aicore__ inline void CopyValueScaleToL1(LocalTensor<SCALE_T> &dstTensor, const MxRunInfo &runInfo,
                                             const MxConstInfo &constInfo, uint32_t realK, uint32_t kIdx)
    {
        // Value scale 按 token 分组，因此一个逻辑 PA scale “行”对应 64 个 value token。
        // 多个 sparse block 拼接时，scale 也按相同 token 区间拼接，只是 offset 单位从
        // token 变成 token group。
        const uint32_t tileStart = kIdx * S2_SPLIT;
        const uint32_t tileEnd = tileStart + realK;
        const uint32_t scaleBlockSize = constInfo.kvSparseBlockSize / MX_TOKEN_GROUP;
        constexpr uint32_t blockElementCnt = 32U / sizeof(SCALE_T);
        uint32_t sparseTileOffset = 0U;
        uint32_t copiedGroupCount = 0U;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = sparseTileOffset;
            const uint32_t blockEnd = blockStart + blockRealSize;
            sparseTileOffset = blockEnd;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;
            const uint32_t groupStartInBlock = copyStartInBlock / MX_TOKEN_GROUP;
            const uint32_t groupEndInBlock = (copyStartInBlock + copyRowNum + MX_TOKEN_GROUP - 1U) / MX_TOKEN_GROUP;
            const uint32_t copyGroupNum = groupEndInBlock - groupStartInBlock;

            Position startPos;
            startPos.bIdx = runInfo.bIdx;
            startPos.n2Idx = runInfo.n2Idx;
            startPos.s2Offset =
                static_cast<uint32_t>(runInfo.sparseBlockIdx[i]) * scaleBlockSize + groupStartInBlock;
            startPos.dIdx = 0U;

            PAShape shape;
            shape.blockSize = scaleBlockSize;
            shape.headNum = constInfo.n2Size;
            shape.headDim = constInfo.valueScaleDSize;
            shape.actHeadDim = constInfo.valueScaleDSize;
            shape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
            shape.copyRowNum = copyGroupNum;
            shape.copyRowNumAlign = copyGroupNum;
            shape.pageStride = scaleBlockSize * constInfo.valueScaleN2D;
            shape.rowStride = constInfo.valueScaleN2D;
            LocalTensor<SCALE_T> valueScaleDstTensor = dstTensor[copiedGroupCount * blockElementCnt];
            GmFlatScaleCopyInToL1PAForND<SCALE_T>(valueScaleDstTensor, vScaleGm_, blockTableGm_, shape, startPos);
            copiedGroupCount += copyGroupNum;
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                      const MxRunInfo &runInfo, uint32_t s2RealSize)
    {
        // 将 C1 从 L0C layout 转成 VF 使用的 UB layout：
        // 每个 AIV sub-block 处理 128 行 query 的一半，S2 维保持连续。
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = AlignUp32(runInfo.actMSize);
        fixpipeParams.mSize = (s2RealSize + 7U) >> 3 << 3;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15U) / 16U) * 16U;
        fixpipeParams.dstStride = 64U;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void FixpipeMm2(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                      const MxRunInfo &, const MxConstInfo &constInfo)
    {
        // C2 输出以 fp32 写入 UB。后续 V2 会更新累计 O；只有当前 logical S2 tile
        // 是该 S1 tile 的最后一块时，才会 cast 并写回 GM。
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (constInfo.dSizeV + 7U) >> 3 << 3;
        fixpipeParams.mSize = M_BASE;
        fixpipeParams.srcStride = ((M_BASE + 15U) / 16U) * 16U;
        fixpipeParams.dstStride = (DV_BASE + 15U) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void InitValuePadding(const LocalTensor<INPUT_T> &valueL1, uint32_t realK, uint32_t curN)
    {
        // realK 按 64 token 对齐后，对超出 realK 的 value 行补零。
        // padding lane 搭配中性 scale padding，不能对 BMM2 结果产生贡献。
        const uint32_t curPadK = AlignUp64(realK);
        if (curPadK == realK) {
            return;
        }
        InitConstValueParams<half> initConstValueParams;
        initConstValueParams.repeatTimes = AlignUp64(curN) / 32U;
        initConstValueParams.blockNum = curPadK - realK;
        initConstValueParams.dstGap = realK;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>()[realK * 16U], initConstValueParams);
    }

    TPipe *tPipe_ = nullptr;
    BufferManager<BufferType::L1> *l1BufferManager_ = nullptr;
    GlobalTensor<INPUT_T> queryGm_;
    GlobalTensor<INPUT_T> keyGm_;
    GlobalTensor<INPUT_T> valueGm_;
    GlobalTensor<SCALE_T> qScaleGm_;
    GlobalTensor<SCALE_T> kScaleGm_;
    GlobalTensor<SCALE_T> vScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;

    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    BufferManager<BufferType::L0C> l0cBufferManager_;
    BuffersPolicyDB<BufferType::L1> l1QBuffers_;
    BuffersPolicy4buff<BufferType::L1> l1KVBuffers_;
    BuffersPolicyDB<BufferType::L1> l1VBuffers_;
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers_;
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers_;
    BuffersPolicy4buff<BufferType::L0C> mmL0CBuffers_;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_
