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
 * \brief 使用 AIV 进行 reduce_sum — 批量加载 + 高精度中间类型累加
 *
 *   - LocalTensor (VECIN) + GlobalTensor + DataCopy/Cast/Add 实现
 *   - 批量加载: rankDim 份输入背靠背 DataCopy (MTE2 流水交叠 GM 访问延迟),
 *     单次 MTE2_V 等待覆盖本批全部 load; 在途模式 (与 PUT 并发) 逐 rank 交替
 *     让出 GM/互连带宽, 尾声模式 (PUT 已 drain) 全量批量交叠全部加载延迟
 *   - slot 布局: 1(sum) + rankDim(输入, 每 rank 独立 slot) + 2(cast scratch pingpong) + 1(out),
 *     按 maxSegLen 固定于分段循环外 (各段共用, 收缩尾段 slot 不漂移)
 *   - 中间类型累加: IOType → Cast SumType → Add → Cast 回 IOType (精度优先);
 *     IOType == SumType (如 fp32) 时免 Cast 直接累加
 *   - IOType 支持 bfloat16_t / half / float, SumType 为模板参数控制累加精度 (默认 float)
 *   - self/received 分离读取; 外部传参统一归纳为 RsParams, Init 一次传入
 */

#ifndef REDUCE_SUM_REF_H
#define REDUCE_SUM_REF_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

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
        GM_ADDR recvWindowAddr{0};   // 通信窗口: 远端 rank PUT 过来的各份部分和
        GM_ADDR selfPartialAddr{0};  // 本卡 workspace: 本 rank matmul 输出的部分和
        GM_ADDR outputAddr{0};       // 归约结果输出地址
        uint64_t rankDim{0};         // 参与归约的 rank 总数 (前置条件: ≤ COMM_MAX_RS_RANK=64,
                                     //  超出时 ExecuteReduceSum 直接返回)
        uint64_t rankId{0};          // 本 rank 编号
        uint64_t aivNum{0};          // 参与归约计算的 AIV 核数
        uint64_t rowStride{0};       // 相邻 rank 部分和的行间距 (元素数)
        uint64_t nSize{0};           // 每行元素数 (N 轴)
        uint64_t ubReservedBytes{0}; // UB 低地址保留区 (调用方 commBuf/fence 轮询 buffer 等,
                                     // RS slot 布局整体上移避开, 防跨组件 MTE2/V 覆写竞态)
        uint64_t fullBatch{0};       // 1=全量批量加载 (rankDim 份一次发); 0=配对批量 (每批 2 份)。
                                     // 在途 RS (与 PUT 传输并发) 用配对 — 全量批量的突发 GM 读
                                     // 实测与 PUT 争抢带宽 (T≥4 通信并发场景回退 +2.6~5.7%);
                                     // 尾声 RS (PUT 已 drain, 无并发) 用全量捕获延迟交叠收益
    };

    __aicore__ inline void Init(const RsParams &params)
    {
        rsParams_ = params;
    }

    // 逐行同步 reduce_sum: tileM 行均分到 aivNum 个核, 每行按 N 分段累加 rankDim 份输入
    __aicore__ inline void ExecuteReduceSum(uint64_t tileM)
    {
        // rankDim 上界防护: inIOElem/inSumElem 为定长 [COMM_MAX_RS_RANK] 栈数组,
        // 超出时越界写, 防御性退出
        if (rsParams_.rankDim > COMM_MAX_RS_RANK) {
            return;
        }

        // 行长按 32B 对齐后的元素数 (保证各分段 DataCopy/Cast 对齐)
        uint64_t alignedN = ApaceKernel::CeilDiv(rsParams_.nSize, SEG_ALIGN_ELEMS) * SEG_ALIGN_ELEMS;

        // N 分段: slot 布局 = 1(sum) + rankDim(输入) + 2(cast scratch pingpong) + 1(out),
        // 每 slot 容量 maxSegLen × sizeof(SumType), 上限 = UB 预算 (TOTAL_UB_SIZE - 栈保留区 48KB,
        // 见文件头 SEG_UB_BUDGET_BYTES; 全额使用 TOTAL_UB_SIZE 会踩踏 AICore 栈区致数据损坏);
        // 布局整体从 ubReservedBytes 起 (避开调用方在 UB 低地址的 commBuf/fence 轮询 buffer,
        // 消除跨组件 MTE2 写 / V 读无排序的覆写竞态)
        uint64_t slotNum = rsParams_.rankDim + SCRATCH_SLOT_NUM + 2U;
        uint64_t ubBudgetBytes =
            (SEG_UB_BUDGET_BYTES > rsParams_.ubReservedBytes) ? (SEG_UB_BUDGET_BYTES - rsParams_.ubReservedBytes) : 0UL;
        uint64_t maxSegLen = ubBudgetBytes / (slotNum * sizeof(SumType));
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
        // 若按各段 segLen 重算, 收缩尾段的输入 slot 会落入其他 slot 区域。
        // 输入 slot 双视图: 混型路径按 IOType 元素偏移 (ubIO), 同型路径按 SumType 偏移 (ubSum);
        // 全部偏移加 ubReservedBytes 换算的基址 (保留区上移)。
        uint64_t slotIOElems = maxSegLen * sizeof(SumType) / sizeof(IOType);
        uint64_t sumBaseElem = rsParams_.ubReservedBytes / sizeof(SumType);
        uint64_t ioBaseElem = rsParams_.ubReservedBytes / sizeof(IOType);
        uint64_t sumElem = sumBaseElem; // 累加器位于 slot#0
        uint64_t inIOElem[COMM_MAX_RS_RANK];
        uint64_t inSumElem[COMM_MAX_RS_RANK];
        for (uint64_t r = 0; r < rsParams_.rankDim; ++r) {
            inIOElem[r] = ioBaseElem + (r + 1U) * slotIOElems;
            inSumElem[r] = sumBaseElem + (r + 1U) * maxSegLen;
        }
        uint64_t scratchElem[PINGPONG_NUM];
        for (uint32_t pp = 0; pp < PINGPONG_NUM; ++pp) {
            scratchElem[pp] = sumBaseElem + (rsParams_.rankDim + 1U + pp) * maxSegLen;
        }
        uint64_t outElem = ioBaseElem + (rsParams_.rankDim + 1U + PINGPONG_NUM) * slotIOElems;

        for (uint32_t rowIdx = 0; rowIdx < rowNumPerCore; ++rowIdx) {
            uint32_t tokenIndex = startRowIdx + rowIdx;

            // N 分段循环
            for (uint64_t nStart = 0; nStart < alignedN; nStart += maxSegLen) {
                uint64_t segLen = (nStart + maxSegLen <= alignedN) ? maxSegLen : (alignedN - nStart);
                // 尾段钳制: nSize 非 SEG_ALIGN_ELEMS 对齐时, alignedN 补齐部分不参与
                // GM 读写 (对齐读会越过本行行尾: 末行越过输出缓冲区末端 → GM 越界写; 跨核行边界
                // 溢出写与邻核正确写出无排序 → 数据竞争)。读写均按实际有效长度搬运 (DataCopyPad)
                uint64_t validLen = (nStart + segLen <= rsParams_.nSize) ? segLen : (rsParams_.nSize - nStart);
                bool isTailSeg = (validLen < segLen);

                // 1. 清零 sum (SumType 累加器)
                Duplicate<SumType>(ubSum[sumElem], static_cast<SumType>(0.0), segLen);
                PipeBarrier<PIPE_V>();
                // 2. 分批加载 + 逐批累加: 每批 batchRank 份输入背靠背 DataCopy (MTE2 流水
                //    交叠 GM 访问延迟, 单次 MTE2_V 等待覆盖本批 load), 批内 Cast/Add 串行;
                //    V→MTE2 排序: 上一行/上一段/上一次调用对输入 slot 的 V 读全部完成后,
                //    本段 MTE2 批量写才可覆写
                SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                // batchRank: 尾声 (fullBatch=1, 无并发 PUT) 全量批量交叠全部加载延迟;
                // 在途 (fullBatch=0) 逐 rank 串行, load→wait→compute 的交替为并发 PUT
                // 让出 GM/互连带宽 (连续突发读流实测与 PUT 争抢带宽)
                uint64_t batchRank = (rsParams_.fullBatch != 0U) ? rsParams_.rankDim : 1U;
                for (uint64_t baseRank = 0; baseRank < rsParams_.rankDim; baseRank += batchRank) {
                    uint64_t batchCnt = rsParams_.rankDim - baseRank;
                    if (batchCnt > batchRank) {
                        batchCnt = batchRank;
                    }
                    for (uint64_t i = 0; i < batchCnt; ++i) {
                        uint64_t curRank = baseRank + i;
                        uint64_t curOffset = curRank * rsParams_.rowStride +
                                             static_cast<uint64_t>(tokenIndex) * rsParams_.nSize + nStart;
                        GM_ADDR dataBaseAddr =
                            (curRank == rsParams_.rankId) ? rsParams_.selfPartialAddr : rsParams_.recvWindowAddr;
                        __gm__ IOType *gmPtr =
                            reinterpret_cast<__gm__ IOType *>(dataBaseAddr + curOffset * sizeof(IOType));
                        GlobalTensor<IOType> gmData;
                        gmData.SetGlobalBuffer(gmPtr);
                        // 尾段按实际字节数读入 (blockLen=Byte), 补齐区仅 UB 内参与计算、不落 GM
                        DataCopyExtParams copyInParams;
                        copyInParams.blockCount = 1;
                        copyInParams.blockLen = static_cast<uint32_t>(validLen * sizeof(IOType));
                        copyInParams.srcStride = 0;
                        copyInParams.dstStride = 0;
                        copyInParams.rsv = 0;
                        DataCopyPadExtParams<IOType> copyInPadParams; // 默认不填充, UB 尾部残留值不参与 GM 读写
                        if constexpr (IsSameType<IOType, SumType>::value) {
                            // 同型: GM → UB 直入 SumType 输入 slot, 免 Cast
                            if (isTailSeg) {
                                DataCopyPad(ubSum[inSumElem[curRank]], gmData, copyInParams, copyInPadParams);
                            } else {
                                DataCopy(ubSum[inSumElem[curRank]], gmData, segLen);
                            }
                        } else {
                            // 异型: GM → UB (IOType slot), 逐 rank Cast 后累加
                            if (isTailSeg) {
                                DataCopyPad(ubIO[inIOElem[curRank]], gmData, copyInParams, copyInPadParams);
                            } else {
                                DataCopy(ubIO[inIOElem[curRank]], gmData, segLen);
                            }
                        }
                    }
                    SetFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);

                    // 3. 批内逐 rank 累加 (SumType 中间类型转换)
                    //    scratch pingpong: Cast(r) 写 scratch[pp], Add 读; 下一次覆写间隔 2 个
                    //    rank, 中间隔 PipeBarrier (等待全部前序 V 完成), WAR 天然覆盖
                    if constexpr (IsSameType<IOType, SumType>::value) {
                        for (uint64_t i = 0; i < batchCnt; ++i) {
                            Add(ubSum[sumElem], ubSum[sumElem], ubSum[inSumElem[baseRank + i]], segLen);
                            if (batchRank == 1U) {
                                // 在途路径: 逐 rank V 排空后再发下一 load,
                                // 为并发 PUT 留出 GM/互连带宽间隙
                                SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                                WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                            } else {
                                PipeBarrier<PIPE_V>(); // 连续 Add 对累加器 RAW 的防御性排序
                            }
                        }
                    } else {
                        uint32_t pp = 0;
                        for (uint64_t i = 0; i < batchCnt; ++i) {
                            Cast(ubSum[scratchElem[pp]], ubIO[inIOElem[baseRank + i]], RoundMode::CAST_NONE, segLen);
                            PipeBarrier<PIPE_V>();
                            Add(ubSum[sumElem], ubSum[sumElem], ubSum[scratchElem[pp]], segLen);
                            if (batchRank == 1U) {
                                // 在途路径: 同上, 逐 rank V 排空
                                SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                                WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                            }
                            pp ^= 1U;
                        }
                    }
                }

                // 4. 输出: SumType → IOType → GM
                PipeBarrier<PIPE_V>();
                __gm__ IOType *outPtr = reinterpret_cast<__gm__ IOType *>(
                    rsParams_.outputAddr +
                    (static_cast<uint64_t>(tokenIndex) * rsParams_.nSize + nStart) * sizeof(IOType));
                GlobalTensor<IOType> gmOut;
                gmOut.SetGlobalBuffer(outPtr);
                DataCopyExtParams copyOutParams;
                copyOutParams.blockCount = 1;
                copyOutParams.blockLen = static_cast<uint32_t>(validLen * sizeof(IOType));
                copyOutParams.srcStride = 0;
                copyOutParams.dstStride = 0;
                copyOutParams.rsv = 0;
                if constexpr (IsSameType<IOType, SumType>::value) {
                    // 同型: 累加结果直接 UB → GM (尾段按实际字节数写出, 防对齐越界)
                    SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                    if (isTailSeg) {
                        DataCopyPad(gmOut, ubSum[sumElem], copyOutParams);
                    } else {
                        DataCopy(gmOut, ubSum[sumElem], segLen);
                    }
                } else {
                    // 异型: Cast 回 IOType 后 UB → GM
                    Cast(ubIO[outElem], ubSum[sumElem], DOWN_CAST_MODE, segLen);
                    SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                    if (isTailSeg) {
                        DataCopyPad(gmOut, ubIO[outElem], copyOutParams);
                    } else {
                        DataCopy(gmOut, ubIO[outElem], segLen);
                    }
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

    // slot 布局 (slot 容量按字节统一 = maxSegLen × sizeof(SumType), 每 rank 输入独立 slot):
    //   slot#0: sum 累加器 | slot#1..rankDim: 各 rank 输入 | slot#rankDim+1/2: cast scratch
    //   pingpong | slot#rankDim+3: 输出
    constexpr static uint32_t SCRATCH_SLOT_NUM = PINGPONG_NUM;

    // 输入 slot 数组上限 (rankDim 上界; 超出时 slot 布局仍正确, 仅 maxSegLen 相应收缩)
    constexpr static uint64_t COMM_MAX_RS_RANK = 64U;

    // 分段对齐粒度 (元素数): 保证 segLen × sizeof(IOType/SumType) 均为 32B 整数倍
    constexpr static uint64_t SEG_ALIGN_ELEMS =
        UB_ALIGN_BYTES / ((sizeof(IOType) < sizeof(SumType)) ? sizeof(IOType) : sizeof(SumType));

    // 下转 (SumType → IOType) 舍入模式: bf16 用最近偶舍入保持精度, 其余 (fp16 等) 用默认舍入
    constexpr static RoundMode DOWN_CAST_MODE =
        IsSameType<IOType, bfloat16_t>::value ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;

    RsParams rsParams_{};
};

} // namespace AiVReduceSumImpl
#endif // REDUCE_SUM_REF_H
