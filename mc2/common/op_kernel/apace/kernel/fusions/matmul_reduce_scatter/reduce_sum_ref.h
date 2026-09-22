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
 * \file reduce_sum_ref.h
 * \brief 使用 AIV 进行 reduce_sum — pingpong 双 buffer + 高精度中间类型累加
 *
 *   - LocalTensor (VECIN) + GlobalTensor + DataCopy/Cast/Add 实现
 *   - pingpong: 输入 slot 双份交替 (pingPongCount_ & (PINGPONG_NUM - 1))
 *   - 中间类型累加: IOType → Cast SumType → Add → Cast 回 IOType (精度优先);
 *     IOType == SumType (如 fp32) 时免 Cast 直接累加
 *   - IOType 支持 bfloat16_t / half / float, SumType 为模板参数控制累加精度 (默认 float)
 *   - self/received 分离读取; 外部传参统一归纳为 RsParams, Init 一次传入
 *
 * 注: 曾有 ExecuteReduceSumBatch 批量接口, 因多行批量 + 单次 flag 模式存在
 * 时序竞态(尾段数据错乱)已删除; 统一走逐行同步的 ExecuteReduceSum。
 *
 * v3 修复 (v2 存量 bug): slot 布局按 maxSegLen 固定 (不再按每段 segLen 重算)。
 * 旧实现在尾段收缩 (n 分段最后一段 segLen < maxSegLen) 时, 尾段 inIO slot 与
 * 前段 sum slot 在 UB 内重叠, 且 MTE2 (下段输入写) 与 MTE3 (上段输出读) 无排序,
 * 竞态覆写前段输出尾部元素 (n=16384: 坏列窗口 [7856, 8528))。
 */

#ifndef REDUCE_SUM_REF_H
#define REDUCE_SUM_REF_H

#include "kernel_operator.h"

#include "apace/utils/apace_common_utils.h"

namespace AiVReduceSumImpl {

using namespace AscendC;

constexpr static uint64_t UB_ALIGN_BYTES = 32UL;
constexpr int32_t FLAG_ID_NUM_ONE = 1;
constexpr uint32_t PINGPONG_NUM = 2U;

// UB 顶部栈保留区防护: AICore 栈保留 32KB + 安全余量 16KB = 48KB。reduceSum 的 6-slot pingpong
// 布局若全额使用 TOTAL_UB_SIZE, slot 数据会踩踏 UB 顶部栈区导致数据损坏;
// UB 预算 = TOTAL_UB_SIZE - 48KB (dav-3510: 248KB - 48KB = 200KB),
// maxSegLen = 预算 / (UB_SLOT_NUM x sizeof(SumType))。
// 注: AscendC::TOTAL_UB_SIZE 定义于 CANN kernel_utils_constants.h 的 __NPU_ARCH__ 设备编译分支内
// (3510 分支 = 248KB), host 编译 (main.cpp 经 impl.h 包含本文件) 不可见; host 分支回退为
// dav-3510 等值字面量 248KB, 若切换目标架构需与设备侧 AscendC::TOTAL_UB_SIZE 保持同步。
constexpr static uint64_t UB_RESERVED_BYTES = 48UL * 1024UL;
#if defined(__NPU_ARCH__)
constexpr static uint64_t SEG_UB_BUDGET_BYTES = AscendC::TOTAL_UB_SIZE - UB_RESERVED_BYTES;
#else
constexpr static uint64_t SEG_UB_BUDGET_BYTES = 248UL * 1024UL - UB_RESERVED_BYTES; // dav-3510 TOTAL_UB_SIZE
#endif

template <typename IOType, typename SumType = float>
class ReduceSumForAlltoAll {
public:
    __aicore__ inline ReduceSumForAlltoAll(){};

    // Init 外部传参: 归纳全部一次性输入, 生命周期内不变
    struct RsParams {
        GM_ADDR recvWindowAddr{0};  // 通信窗口: 远端 rank PUT 过来的各份部分和
        GM_ADDR selfPartialAddr{0}; // 本卡 workspace: 本 rank matmul 输出的部分和
        GM_ADDR outputAddr{0};      // 归约结果输出地址
        uint64_t rankDim{0};        // 参与归约的 rank 总数
        uint64_t rankId{0};         // 本 rank 编号
        uint64_t aivNum{0};         // 参与归约计算的 AIV 核数
        uint64_t rowStride{0};      // 相邻 rank 部分和的行间距 (元素数)
        uint64_t nSize{0};          // 每行元素数 (N 轴)
    };

    __aicore__ inline void Init(const RsParams &params)
    {
        rsParams_ = params;
    }

    // 逐行同步 reduce_sum: tileM 行均分到 aivNum 个核, 每行按 N 分段累加 rankDim 份输入
    __aicore__ inline void ExecuteReduceSum(uint64_t tileM)
    {
        // 行长按 32B 对齐后的元素数 (保证各分段 DataCopy/Cast 对齐)
        uint64_t alignedN = ApaceKernel::CeilDiv(rsParams_.nSize, SEG_ALIGN_ELEMS) * SEG_ALIGN_ELEMS;

        // N 分段: UB_SLOT_NUM 个 slot × SumType, 上限 = UB 预算 (TOTAL_UB_SIZE - 栈保留区 48KB,
        // 见文件头 SEG_UB_BUDGET_BYTES; 全额使用 TOTAL_UB_SIZE 会踩踏 AICore 栈区致数据损坏)
        uint64_t maxSegLen = SEG_UB_BUDGET_BYTES / (UB_SLOT_NUM * sizeof(SumType));
        if (maxSegLen > alignedN) {
            maxSegLen = alignedN;
        }
        maxSegLen = maxSegLen / SEG_ALIGN_ELEMS * SEG_ALIGN_ELEMS;
        if (maxSegLen == 0U) {
            maxSegLen = SEG_ALIGN_ELEMS;
        }

        uint32_t startRowIdx = 0;
        uint32_t rowNumPerCore = 0;
        SplitRowsToCores(static_cast<uint32_t>(tileM), static_cast<uint32_t>(rsParams_.aivNum),
                         static_cast<uint32_t>(GetBlockIdx()), startRowIdx, rowNumPerCore);

        LocalTensor<SumType> ubSum{TPosition::VECIN, 0,
                                   static_cast<uint32_t>(AscendC::TOTAL_UB_SIZE / sizeof(SumType))};
        LocalTensor<IOType> ubIO{TPosition::VECIN, 0, static_cast<uint32_t>(AscendC::TOTAL_UB_SIZE / sizeof(IOType))};

        // slot 布局按 maxSegLen 固定计算 (分段循环外), 全部段共用同一布局 —
        // 若按各段 segLen 重算, 收缩尾段的 inIO slot 会落入前段 sum slot 内, 且 MTE2/MTE3
        // 跨段无排序, 竞态覆写前段输出尾部 (根因详见文件头 v3 修复说明)。
        uint64_t slotIOElems = maxSegLen * sizeof(SumType) / sizeof(IOType);
        uint64_t sumElem = SLOT_SUM; // 累加器位于 slot#0
        uint64_t inIOElem[PINGPONG_NUM], inSumElem[PINGPONG_NUM];
        for (uint32_t pp = 0; pp < PINGPONG_NUM; ++pp) {
            inIOElem[pp] = SLOT_IN_IO[pp] * slotIOElems;
            inSumElem[pp] = SLOT_IN_SUM[pp] * maxSegLen;
        }
        uint64_t outElem = SLOT_OUT * slotIOElems;

        for (uint32_t rowIdx = 0; rowIdx < rowNumPerCore; ++rowIdx) {
            uint32_t tokenIndex = startRowIdx + rowIdx;

            // N 分段循环
            for (uint64_t nStart = 0; nStart < alignedN; nStart += maxSegLen) {
                uint64_t segLen = (nStart + maxSegLen <= alignedN) ? maxSegLen : (alignedN - nStart);

                // 1. 清零 sum (SumType 累加器)
                Duplicate<SumType>(ubSum[sumElem], static_cast<SumType>(0.0), segLen);
                PipeBarrier<PIPE_V>();

                // 2. 逐 rank 累加 (pingpong, SumType 中间类型转换)
                for (uint64_t curRank = 0; curRank < rsParams_.rankDim; ++curRank) {
                    uint32_t pingPongIdx = pingPongCount_ & (PINGPONG_NUM - 1U);
                    uint64_t curOffset =
                        curRank * rsParams_.rowStride + static_cast<uint64_t>(tokenIndex) * rsParams_.nSize + nStart;

                    GM_ADDR dataBaseAddr =
                        (curRank == rsParams_.rankId) ? rsParams_.selfPartialAddr : rsParams_.recvWindowAddr;
                    __gm__ IOType *gmPtr = reinterpret_cast<__gm__ IOType *>(dataBaseAddr + curOffset * sizeof(IOType));
                    GlobalTensor<IOType> gmData;
                    gmData.SetGlobalBuffer(gmPtr);

                    if constexpr (IsSameType<IOType, SumType>::value) {
                        // 同型: GM → UB 直入 SumType 输入 slot, 免 Cast
                        DataCopy(ubSum[inSumElem[pingPongIdx]], gmData, segLen);
                        SetFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                        WaitFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                    } else {
                        // 异型: GM → UB (IOType) → Cast SumType 输入 slot
                        DataCopy(ubIO[inIOElem[pingPongIdx]], gmData, segLen);
                        SetFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                        WaitFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                        Cast(ubSum[inSumElem[pingPongIdx]], ubIO[inIOElem[pingPongIdx]], RoundMode::CAST_NONE, segLen);
                        PipeBarrier<PIPE_V>();
                    }

                    // Add: sum += input (SumType 累加)
                    Add(ubSum[sumElem], ubSum[sumElem], ubSum[inSumElem[pingPongIdx]], segLen);
                    SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);

                    pingPongCount_++;
                }

                // 3. 输出: SumType → IOType → GM
                PipeBarrier<PIPE_V>();
                __gm__ IOType *outPtr = reinterpret_cast<__gm__ IOType *>(
                    rsParams_.outputAddr +
                    (static_cast<uint64_t>(tokenIndex) * rsParams_.nSize + nStart) * sizeof(IOType));
                GlobalTensor<IOType> gmOut;
                gmOut.SetGlobalBuffer(outPtr);
                if constexpr (IsSameType<IOType, SumType>::value) {
                    // 同型: 累加结果直接 UB → GM
                    SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                    DataCopy(gmOut, ubSum[sumElem], segLen);
                } else {
                    // 异型: Cast 回 IOType 后 UB → GM
                    Cast(ubIO[outElem], ubSum[sumElem], DOWN_CAST_MODE, segLen);
                    SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                    DataCopy(gmOut, ubIO[outElem], segLen);
                }
                SetFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
            }
        }
    }

private:
    // 行均分到各 AIV 核: 前 remainderNum 个核各多分 1 行
    __aicore__ inline void SplitRowsToCores(uint32_t totalRowNum, uint32_t totalAivNum, uint32_t coreId,
                                            uint32_t &startRowIdx, uint32_t &rowNumPerCore)
    {
        rowNumPerCore = totalRowNum / totalAivNum;
        uint32_t remainderNum = totalRowNum % totalAivNum;
        startRowIdx = rowNumPerCore * coreId;
        if (coreId < remainderNum) {
            rowNumPerCore += 1;
            startRowIdx += coreId;
        } else {
            startRowIdx += remainderNum;
        }
    }

    // UB slot 布局 (每 slot 容量 segLen 个 SumType 元素, 共 UB_SLOT_NUM 个):
    //   slot#0: sum 累加器 | slot#1/2: pingpong0 输入 (inIO, inSum) | slot#3/4: pingpong1 | slot#5: 输出
    constexpr static uint32_t SLOT_SUM = 0U;
    constexpr static uint32_t SLOT_IN_IO[PINGPONG_NUM] = {1U, 3U};
    constexpr static uint32_t SLOT_IN_SUM[PINGPONG_NUM] = {2U, 4U};
    constexpr static uint32_t SLOT_OUT = 5U;
    constexpr static uint32_t UB_SLOT_NUM = SLOT_OUT + 1U;

    // 分段对齐粒度 (元素数): 保证 segLen × sizeof(IOType/SumType) 均为 32B 整数倍
    constexpr static uint64_t SEG_ALIGN_ELEMS =
        UB_ALIGN_BYTES / ((sizeof(IOType) < sizeof(SumType)) ? sizeof(IOType) : sizeof(SumType));

    // 下转 (SumType → IOType) 舍入模式: bf16 用最近偶舍入保持精度, 其余 (fp16 等) 用默认舍入
    constexpr static RoundMode DOWN_CAST_MODE =
        IsSameType<IOType, bfloat16_t>::value ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;

    RsParams rsParams_{};
    uint32_t pingPongCount_{0};
};

} // namespace AiVReduceSumImpl
#endif // REDUCE_SUM_REF_H
