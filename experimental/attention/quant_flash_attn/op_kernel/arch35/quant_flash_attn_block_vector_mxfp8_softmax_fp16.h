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
 * \file quant_flash_attn_block_vector_mxfp8_softmax_fp16.h
 * \brief MxFP8 Softmax FP16 Vector Block 最小桩（阶段二占位）
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_

// ===== vf 头文件（编译冒烟：让编译器解析已实现的 vf 代码，提前暴露问题）=====
#include "vf/vf_common_def.h"
#include "vf/vf_calc_rescale_factor.h"
#include "vf/vf_calc_pscale.h"
#include "vf/vf_flashupdate_acc.h"
#include "vf/vf_div_cast_bf16.h"
#include "vf/vf_softmax_fp16_to_fp8.h"

namespace QFA_KERNEL {

template <typename QUANT_T, typename SCALE_T, typename OUT_T>
class QuantFlashAttnBlockVectorMxfp8SoftmaxFp16 {
public:
    // ===== 类型别名（对齐 cube 侧命名）=====
    using MM_OUT_T = float;       // mm2Res fp32
    using SOFTMAX_T = half;       // 状态 half（log2 整数域）

    // ===== 编译期常量（基础尺寸，对齐 common_def）=====
    static constexpr uint32_t s1BaseSize = S1_BASE_SIZE;        // 256（S1 总宽，V0/V1 各持一半）
    static constexpr uint32_t s2SubLoopSize = S2_SUB_LOOP_SIZE; // 128（每 subLoop 的 mm1Res/P 行数，即 S2 方向每次处理的行数）
    static constexpr uint32_t dVBaseSize = DV_BASE_SIZE;        // 128（V2 输出的 DV 方向行数）
    static constexpr half MIN_HALF_VALUE = static_cast<half>(-65504.0f);  // fp16 最小有限负值

private:
    // ===== UB 尺寸常量（vector 侧自有；mm1Res/mm2Res 直接用 common_def 跨核契约）=====
    static constexpr uint32_t UB_VEC1P_SIZE = 2 * s2SubLoopSize * (s1BaseSize / 2); // 2 slots × [128,128] fp8
    static constexpr uint32_t UB_STAGE2OUT_SIZE = dVBaseSize * (s1BaseSize / 2); // [128,128] fp32
    static constexpr uint32_t UB_ACCROWSUM_SIZE = s1BaseSize / 2;           // [128] fp32
    static constexpr uint32_t UB_SMAX_SLOT = s1BaseSize / 2;                // [128] half
    static constexpr uint32_t UB_SMAX_BUFCNT = 3;                           // mloop%3
    static constexpr uint32_t UB_SEXP_BUFCNT = 3;                           // loop%3
    static constexpr uint32_t UB_PSCALE_SLOT = s1BaseSize / 2;              // [128] e8m0
    static constexpr uint32_t UB_PSCALE_TASKS = 3;                          // loop%3
    static constexpr uint32_t UB_PSCALE_SUBLOOPS = 2;                       // subLoopIdx

    // ===== UB BASE 常量链（mm1Res/mm2Res 的 BASE 用 common_def 值，不自引用）=====
    static constexpr uint32_t UB_VEC1P_BASE = UB_MM2RES_BASE + UB_MM2RES_SIZE * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_STAGE2OUT_BASE = UB_VEC1P_BASE + UB_VEC1P_SIZE * sizeof(QUANT_T);
    static constexpr uint32_t UB_ACCROWSUM_BASE = UB_STAGE2OUT_BASE + UB_STAGE2OUT_SIZE * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_SMAX_BASE = UB_ACCROWSUM_BASE + UB_ACCROWSUM_SIZE * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_SEXP_BASE = UB_SMAX_BASE + UB_SMAX_SLOT * UB_SMAX_BUFCNT * sizeof(SOFTMAX_T);
    static constexpr uint32_t UB_BSMAX_BASE = UB_SEXP_BASE + UB_SMAX_SLOT * UB_SEXP_BUFCNT * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_SUBMAX_BASE = UB_BSMAX_BASE + UB_SMAX_SLOT * sizeof(SOFTMAX_T);
    static constexpr uint32_t UB_PSCALE_BASE = UB_SUBMAX_BASE + UB_SMAX_SLOT * UB_PSCALE_SUBLOOPS * sizeof(SOFTMAX_T);
    static constexpr uint32_t UB_TOTAL = UB_PSCALE_BASE + UB_PSCALE_SLOT * UB_PSCALE_TASKS * UB_PSCALE_SUBLOOPS * sizeof(SCALE_T);
    static_assert(UB_TOTAL <= 256 * 1024, "V-core UB budget (256KB) exceeded");

    // ===== UB Tensor 成员（对齐设计 §4 表）=====
    LocalTensor<SOFTMAX_T> mm1ResUB_;     // [2][128,128] half，C1 Fixpipe 目标（loop%2）
    LocalTensor<MM_OUT_T> mm2ResUB_;      // [129,128] fp32，C2 Fixpipe 目标（V2 输入 + divOut/outputT 别名）
    LocalTensor<QUANT_T> vec1PUb_;        // [2][128,128] fp8，V1 P 的 NZ 源布局暂存
    LocalTensor<MM_OUT_T> stage2OutUB_;   // [128,128] fp32，V2 跨块累积分子 [DV,S1]
    LocalTensor<MM_OUT_T> accRowsumUB_;   // [128] fp32，V2 跨块累积分母 per S1
    LocalTensor<SOFTMAX_T> softmaxMaxUB_; // [3][128] half，accMax log2 整数 k（mloop%3）
    LocalTensor<MM_OUT_T> softmaxExpUB_;  // [3][128] fp32，跨块因子 fp32 位模式（loop%3）
    LocalTensor<SOFTMAX_T> blockStartMaxUB_;   // [128] half，块首 K^{blkStart} 快照
    LocalTensor<SOFTMAX_T> subLoopUsedMaxUB_;  // [2][128] half，各 subLoop k_i（供末 subLoop 算 pscale）
    LocalTensor<SCALE_T> pscaleUB_;      // [3][2][128] e8m0，[loop%3][subLoopIdx]
    LocalTensor<uint8_t> pNzIdxUB_;      // P NZ 打包 Gather 索引表

    // ===== L1 共享区视图成员（AIC L1，V1 经 MTE3 写入 P/pscale 的目标地址）=====
    LocalTensor<QUANT_T> pL1V0Tensor_;         // V0 的 P 数据（fp8，AIC L1）
    LocalTensor<QUANT_T> pL1V1Tensor_;         // V1 的 P 数据（fp8，AIC L1）
    LocalTensor<SCALE_T> pScaleL1V0Tensor_;    // V0 的 pscale（e8m0，AIC L1）
    LocalTensor<SCALE_T> pScaleL1V1Tensor_;    // V1 的 pscale（e8m0，AIC L1）

    // ===== 依赖注入 =====
    const ConstInfo &constInfo_;          // 核信息（subBlockIdx、scaleValue 等）

    // ===== 外部 pScale（静态 headroom 因子，InitTensorsVec 时 cast half 存成员）=====
    half pScaleHalf_ = static_cast<half>(1.0f);

    // ===== 核内事件 ID（对齐设计 §4 事件表）=====
    static constexpr uint32_t EVT_P_COPY = 0;  // V↔MTE3：P/pscale MTE3 前后 + 顶部等待
    static constexpr uint32_t EVT_GM_OUT = 1;  // V→MTE3：GM 输出前

public:
    __aicore__ inline explicit QuantFlashAttnBlockVectorMxfp8SoftmaxFp16(ConstInfo &constInfo)
        : constInfo_(constInfo) {}

    __aicore__ inline void InitTensorsVec()
    {
        // ① UB 静态分配（编译期 BASE 常量链，对齐设计 §4 表）
        //    mm1Res/mm2Res 用 common_def 跨核契约；其余按 BASE 链推导
        //    addr = 字节地址，size = 元素数
        mm1ResUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_MM1RES_BASE,
                                           UB_MM1RES_SLOT * UB_MM1RES_BUFCNT);
        mm2ResUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_MM2RES_BASE, UB_MM2RES_SIZE);
        vec1PUb_ = LocalTensor<QUANT_T>(TPosition::VECCALC, UB_VEC1P_BASE, UB_VEC1P_SIZE);
        stage2OutUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_STAGE2OUT_BASE, UB_STAGE2OUT_SIZE);
        accRowsumUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_ACCROWSUM_BASE, UB_ACCROWSUM_SIZE);
        softmaxMaxUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_SMAX_BASE,
                                              UB_SMAX_SLOT * UB_SMAX_BUFCNT);
        softmaxExpUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_SEXP_BASE,
                                             UB_SMAX_SLOT * UB_SEXP_BUFCNT);
        blockStartMaxUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_BSMAX_BASE, UB_SMAX_SLOT);
        subLoopUsedMaxUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_SUBMAX_BASE,
                                                  UB_SMAX_SLOT * UB_PSCALE_SUBLOOPS);
        pscaleUB_ = LocalTensor<SCALE_T>(TPosition::VECCALC, UB_PSCALE_BASE,
                                        UB_PSCALE_SLOT * UB_PSCALE_TASKS * UB_PSCALE_SUBLOOPS);
        pNzIdxUB_ = LocalTensor<uint8_t>(TPosition::VECCALC, UB_PSCALE_BASE +
                                         UB_PSCALE_SLOT * UB_PSCALE_TASKS * UB_PSCALE_SUBLOOPS * sizeof(SCALE_T),
                                         256);  // 索引表 256B（实现期可调）

        // ② L1 共享区视图（按 common_def 常量构造，不分配、不初始化）
        pL1V0Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_P_V0_BASE,
                                            L1_P_BUFCNT * L1_P_V0_SINGLE_SLOT_SIZE);
        pL1V1Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_P_V1_BASE,
                                            L1_P_BUFCNT * L1_P_V1_SINGLE_SLOT_SIZE);
        pScaleL1V0Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V0_BASE,
                                                 L1_P_BUFCNT * L1_PSCALE_V0_SINGLE_SLOT_SIZE);
        pScaleL1V1Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V1_BASE,
                                                 L1_P_BUFCNT * L1_PSCALE_V1_SINGLE_SLOT_SIZE);

        // ③ Gather 索引表初始化（镜像 mxfp8 quant_flash_attn_block_vec_mxfp8.h:1050-1054）
        //    idx[i*64+j] = i + (j<<2)：4 路解交织，S1半=128 时 i∈[0,2), j∈[0,64)
        //    首选无 Gather 两段式（§9.1-③），此表作为回退方案初始化
        for (int32_t i = 0; i < 2; i++) {
            for (int32_t j = 0; j < 64; j++) {
                pNzIdxUB_.SetValue(i * 64 + j, static_cast<uint8_t>(i + (j << 2)));
            }
        }
    }

    __aicore__ inline void ComputeVec1(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subLoopIdx)
    {
        // S1 尾行（actMSize ≤ 128）时本核（V1）没有自己的 S1 列，直接返回：
        // S1 按 V0=前半 / V1=后半分给两个核，行尾块的 S1 总数不足 256 时 V1 分不到列。
        // 此时若继续执行，会在上一轮的残留数据（stale mm1Res）上空算一遍，
        // 产出的 P/pscale 虽无消费者但浪费流水。外层 ExecuteTask 的跨核 flag
        // 是无条件 Set 的，这里直接 return 不会引起死锁。
        if (runInfo.actVecS1Size == 0U) {
            return;
        }

        uint32_t c1v1Loop = AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, s2SubLoopSize);  // ≤2
        bool isLastSub = (subLoopIdx + 1U == c1v1Loop);
        uint32_t validRows = isLastSub
            ? (runInfo.actSingleLoopS2Size - subLoopIdx * s2SubLoopSize)   // 1..128
            : s2SubLoopSize;
        uint32_t mmSlot = runInfo.loop % 2U;                    // 对齐 cube FixpipeMm1 ring
        uint32_t stateSlot = runInfo.softmaxStateSlot;          // mloop % 3
        uint32_t taskSlot = runInfo.loop % 3U;                  // exp/pscale/pSlot 共用

        // 上轮任务的 P/pscale MTE3 排空后才允许 vf 覆写 vec1PUb（DN L279 同款）
        WaitFlag<HardEvent::MTE3_V>(EVT_P_COPY);

        // 行首复位 accMax：新 [bN2,gS1] 行的槽位含 3 行前旧值，必须复位为 MIN
        if (runInfo.isFirstS2Loop && subLoopIdx == 0U) {
            Duplicate(softmaxMaxUB_[stateSlot * UB_SMAX_SLOT], MIN_HALF_VALUE, UB_SMAX_SLOT);
        }

        // 块首快照 m_{k-1}：非首块的 subLoop 0 时拷贝 accMax → blockStartMaxUB
        if (subLoopIdx == 0U && !runInfo.isFirstS2Loop) {
            DataCopy(blockStartMaxUB_, softmaxMaxUB_[stateSlot * UB_SMAX_SLOT], UB_SMAX_SLOT);
        }

        // softmax 主体：max 归约 → 量化合并 → exp → cast fp8 → NZ 打包
        QfaVectorApi::VfSoftmaxFp16ToFp8(vec1PUb_, softmaxMaxUB_, subLoopUsedMaxUB_, pNzIdxUB_,
                                           mm1ResUB_, pScaleHalf_, validRows, subLoopIdx,
                                           mmSlot, stateSlot);

        // P → AIC L1（MTE3，NZ 块拷贝）
        SetFlag<HardEvent::V_MTE3>(EVT_P_COPY);
        WaitFlag<HardEvent::V_MTE3>(EVT_P_COPY);
        // TODO(批次3.3): NZ 三段链 MTE3 块参数确定后填充（当前线性占位）
        LocalTensor<QUANT_T> &pL1Target = (constInfo_.subBlockIdx == 0U) ? pL1V0Tensor_ : pL1V1Tensor_;
        DataCopy(pL1Target[runInfo.pSlot * L1_P_V0_SINGLE_SLOT_SIZE
                          + subLoopIdx * s2SubLoopSize * (s1BaseSize / 2)],
                 vec1PUb_[subLoopIdx * s2SubLoopSize * (s1BaseSize / 2)],
                 s2SubLoopSize * (s1BaseSize / 2));  // 占位：线性拷贝，NZ 布局待批次3.3

        // ===== 末 subLoop epilogue：跨块因子 + pscale 统一写 =====
        if (isLastSub) {
            // 跨块 rescale 因子（非首块才算）
            if (!runInfo.isFirstS2Loop) {
                QfaVectorApi::VfCalcRescaleFactor(softmaxExpUB_, blockStartMaxUB_, softmaxMaxUB_,
                                                   taskSlot, stateSlot);
            }
            // pscale 各 subLoop 统一写（末 subLoop 恒为 127，跳过计算）
            LocalTensor<SCALE_T> &pScaleL1Target =
                (constInfo_.subBlockIdx == 0U) ? pScaleL1V0Tensor_ : pScaleL1V1Tensor_;
            LocalTensor<uint8_t> pscaleU8 = pscaleUB_.template ReinterpretCast<uint8_t>();
            for (uint32_t i = 0; i < c1v1Loop; i++) {
                if (i < c1v1Loop - 1) {
                    QfaVectorApi::VfCalcPScale(pscaleU8, subLoopUsedMaxUB_, softmaxMaxUB_,
                                                taskSlot, i, stateSlot);
                } else {
                    // 末 subLoop pscale 恒为 2^0：e8m0 字节 127（uint8 视图填充，避免 e8m0 标量填充限制）
                    Duplicate(pscaleU8[(taskSlot * UB_PSCALE_SUBLOOPS + i) * UB_PSCALE_SLOT],
                              static_cast<uint8_t>(127), UB_PSCALE_SLOT);
                }
                SetFlag<HardEvent::V_MTE3>(EVT_P_COPY);
                WaitFlag<HardEvent::V_MTE3>(EVT_P_COPY);
                DataCopy(pScaleL1Target.template ReinterpretCast<uint8_t>()
                             [runInfo.pSlot * L1_PSCALE_V0_SINGLE_SLOT_SIZE + i * UB_PSCALE_SLOT],
                         pscaleU8[(taskSlot * UB_PSCALE_SUBLOOPS + i) * UB_PSCALE_SLOT],
                         UB_PSCALE_SLOT);
            }
            // 为下一任务的顶部等待补 Set（全部 P/pscale MTE3 排空后触发）
            SetFlag<HardEvent::MTE3_V>(EVT_P_COPY);
        }
    }

    __aicore__ inline void ComputeVec2(RunInfoMxfp8SoftmaxFp16 &runInfo)
    {
        // V1 半无有效列（actVecS1Size==0）：C2 由 cube 侧防御跳过，V2 同步早退
        //（外层 C2_V2 flag 无条件 Set，早退不破坏跨核配平）
        if (runInfo.actVecS1Size == 0U) {
            return;
        }

        // ===== (1) 跨 S2 分块累积（设计 §6.2：首块 = vf 纯拷贝起步，消费者统一 V 管道）=====
        uint32_t taskSlot = runInfo.loop % 3;
        if (runInfo.isFirstS2Loop) {
            QfaVectorApi::VfFlashUpdateAcc<false>(stage2OutUB_, accRowsumUB_, mm2ResUB_,
                                                   softmaxExpUB_, taskSlot);
        } else {
            QfaVectorApi::VfFlashUpdateAcc<true>(stage2OutUB_, accRowsumUB_, mm2ResUB_,
                                                  softmaxExpUB_, taskSlot);
        }

        // TODO(批次4.1d): 末块 PipeBarrier + outputT 别名 + VfDivAndCast（融合转置写）
    }

    __aicore__ inline void ReleaseTensorsVec()
    {
        // TODO(批次2.3): 跨核 flag 配平（kernel 层负责）
    }
};

} // namespace QFA_KERNEL

#endif // QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_
