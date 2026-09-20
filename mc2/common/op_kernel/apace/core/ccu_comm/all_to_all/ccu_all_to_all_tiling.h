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
 * \file ccu_all_to_all_tiling.h
 * \brief CCU AllToAll 通信 tiling——AIC kernel 的通信参数契约（kernel 侧代码）
 *
 * 本结构是 AIC 融合 kernel 的入参契约：host 侧按填充分工构造后经 <<<>>> 按值传入
 * （host 是填写者，契约归 kernel 侧——同 AscendC 工程 host 包含 kernel tiling 头的模式）。
 * 填充分工：形状/轮次字段由消费方 host 的 Validate 计算；资源字段（token/goSize/
 * mappedAddrs/偏移表）由消费方 host 的 Prepare 经 AllocAllToAllRegion（_host.h）烧入。
 *
 * 事件编号契约（AIC↔CCU 同步，消费方按此对接）：
 *   ev[0] = wake（AIC commit → CCU process，循环前唯一一次）
 *   ev[1] = scale 轮 done、ev[t+2] = data 轮 t done（CCU notify → AIC wait，
 *   等待节奏由消费方 kernel 自定——如经 CommPolicy 逐 tile 等待）
 * 槽位契约：每轮 COMM_SLOT_NUM(11) 槽；组 0 = scale 轮全量，组 1..dataCommTurn = data 轮
 */

#pragma once

#include "../ccu_comm.h"

// CCU AllToAll 通信侧 tiling（通信标量散参全部收纳于此；形状/轮次由消费方 host 填，
// 资源字段由 AllocAllToAllRegion/消费方 host 填）
#pragma pack(push, 8)
struct alignas(8) CcuCommTilingData {
    // ---- 形状/轮次（消费方 host 填；kernel 侧只读）----
    uint32_t rankId{0};
    uint32_t rankSize{0};
    uint32_t dataCommTurn{0};         // data 轮数（CCU 总轮数 = dataCommTurn+1，组 0 = scale 轮）
    uint32_t tileM{0};                // 每 data 轮传输行数
    uint64_t blockSize{0};            // data 每块字节 = blockRows*K
    uint64_t srcStride{0};            // 块间距字节 = blockSize（紧邻切分）
    uint64_t subBytes{0};             // 每 data 轮子段字节 = tileM*K
    uint64_t scaleSliceSize{0};       // scale 每块字节（全量 1 轮）
    uint64_t scaleSrcStride{0};       // = scaleSliceSize
    uint64_t scaleDstOffset{0};       // = rankId*scaleSliceSize（host 预乘烧入）
    uint64_t workspaceScaleOffset{0}; // workspace 内 recvScale 区偏移
    // ---- CCU 资源（host 填；AIC 写通信参数区时消费）----
    uint64_t token{0};                // x1（data）的 CCU 内存访问令牌
    uint64_t scaleToken{0};           // x1Scale 的 CCU 内存访问令牌
    A2aGoSize goSize{};               // data 本地拷贝 loop 参数
    A2aGoSize scaleGoSize{};          // scale 本地拷贝 loop 参数
    CcuMappedAddrs mappedAddrs{};     // 通信参数区 XN/CKE 寄存器槽位 MMIO 地址映射
    // TODO(后续重构)：三表均为现有标量闭式（src=t*subBytes, dst=rankId*blockSize+
    // t*subBytes, slice=subBytes），可删表改为 kernel 内闭式计算——落位约定归 core、
    // tiling 省 384B、与 scale 轮标量形态对称；性能无差（每轮 2 次乘加 vs 11 次 MMIO 写）
    // TODO(均分限制)：当前契约强制均分轮次（单 subBytes 标量 + Validate 整除校验），
    // 尾轮不等分不支持；若未来消费方需非均匀轮次，须保留/扩展 per-round 表（与上条闭式方案二选一）
    uint64_t srcOffsets[MAX_COMMTURN]{0};  // per-round data src 偏移表（host 填，AIC 读）
    uint64_t dstOffsets[MAX_COMMTURN]{0};  // per-round data dst 偏移表
    uint64_t sliceSizes[MAX_COMMTURN]{0};  // per-round data sliceSize 表
};
#pragma pack(pop)
