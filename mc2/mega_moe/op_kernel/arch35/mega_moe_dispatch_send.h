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
 * \file mega_moe_dispatch_send.h
 * \brief Dispatch Level-1 (跨server) URMA 分层通信的布局常量与辅助函数。
 *
 * L1 dispatch 通信模式：
 *   - 跨 server 同层通信：同一 localRank 的卡之间直连（server0 卡0 ↔ server1 卡0）
 *   - 计算核：量化 token → 写入 workspace 中转区 → 原子累加 count
 *   - 发送核：轮询中转区 → ReadGmByPassDCache 读 flag → WriteWithNotifyNbi 发到对端一级接收区
 *   - 对端一级接收区使用稀疏布局（按 tokenIdx 定位），metadata 最后 2 个 int 由 WriteWithNotifyNbi notify 写入
 */

#ifndef MEGA_MOE_DISPATCH_SEND_H
#define MEGA_MOE_DISPATCH_SEND_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"

namespace MegaMoeDispatchImpl {

// ============================================================
// 常量定义
// ============================================================
constexpr uint32_t MAX_L1_SEND_CORES = 8;        // 发送核最多 8 个
constexpr uint32_t L1_RELAY_HEADER_BYTES = 32;   // 中转区每 server 32B header：int[0]=count, rest padding
constexpr uint32_t L1_META_BYTES = 32;           // 每 slot 紧跟 data 的 metadata 32B：
                                                  //   本地中转区: int[0]=flag(1=ready), int[1]=token_id
                                                  // 远端接收区 notify 写入整个 slot 的最后 2 个 int (slotSize-8B),
                                                  //   二级通信读 slot 末尾 8B 判断 token 是否就绪

// ============================================================
// Server 拓扑辅助函数
// ============================================================

/// 本卡所属 server ID
__aicore__ inline uint32_t GetServerId(uint32_t rankId, uint32_t rankSizePerServer)
{
    return rankId / rankSizePerServer;
}

/// 本卡在 server 内的 local rank
__aicore__ inline uint32_t GetLocalRankInServer(uint32_t rankId, uint32_t rankSizePerServer)
{
    return rankId % rankSizePerServer;
}

/// 总 server 数
__aicore__ inline uint32_t GetTotalServers(uint32_t worldSize, uint32_t rankSizePerServer)
{
    return worldSize / rankSizePerServer;
}

/// 给定 target server, 计算同 localRank 的对端 global rankId
__aicore__ inline uint32_t GetL1TargetRankId(uint32_t targetServerId, uint32_t localRankInServer,
                                               uint32_t rankSizePerServer)
{
    return targetServerId * rankSizePerServer + localRankInServer;
}

/// 将 otherServerIdx (0..numOtherServers-1) 映射到实际 serverId (跳过 myServerId)
__aicore__ inline uint32_t OtherServerIdxToServerId(uint32_t otherServerIdx, uint32_t myServerId)
{
    return (otherServerIdx < myServerId) ? otherServerIdx : otherServerIdx + 1;
}

/// 将实际 serverId 映射回 otherServerIdx (0..numOtherServers-1)
__aicore__ inline uint32_t ServerIdToOtherServerIdx(uint32_t serverId, uint32_t myServerId)
{
    return (serverId < myServerId) ? serverId : serverId - 1;
}

// ============================================================
// 中转区 (Relay Area) 布局辅助 — workspace 本地侧
// ============================================================
//
// 每个 target server 的中转区布局：
//   [32B header: count=int[0]] [slot_0][slot_1]...[slot_m-1]
//   每个 slot 总大小 = slotSize (tokendata + scaledata + 32B, 已 512B 对齐)
//     slot 内: [data: dataBytes (32B 对齐)][metadata: 32B (紧跟 data)][padding 对齐至 512B]
//     metadata: int[0]=flag(1=ready), int[1]=token_id, int[2..7]=unused
//
// 发送核通过 AtomicAdd(countAddr, 1) 获取 slotIdx;
// 计算核写数据时：先写 data，再写 metadata(flag=1) 以保证发送核看到 flag=1 时 data 已就绪。
// 注意: slotStride (slotSize) 须 512B 对齐, 与对端接收区一致。

/// 计算单个 server 中转区大小 (含 header + m 个 slot, 512B 对齐)
__aicore__ inline uint64_t CalcRelayAreaPerServer(uint32_t m, uint32_t slotSize)
{
    uint64_t perServer = L1_RELAY_HEADER_BYTES + static_cast<uint64_t>(m) * slotSize;
    return Ops::Base::CeilAlign(perServer, static_cast<int64_t>(ALIGN_512));
}

/// 获取第 otherServerIdx 个中转区的基地址
__aicore__ inline GM_ADDR GetRelayAreaBase(GM_ADDR relayDataPtr, uint32_t otherServerIdx,
                                             uint64_t perServerSize)
{
    return relayDataPtr + static_cast<uint64_t>(otherServerIdx) * perServerSize;
}

/// 中转区 header 中 count 地址 (int[0])
__aicore__ inline __gm__ int32_t* GetRelayCountAddr(GM_ADDR relayAreaBase)
{
    return reinterpret_cast<__gm__ int32_t*>(relayAreaBase);
}

/// 中转区第 slotIdx 个 slot 的基地址 (步长 = slotSize, 须 512B 对齐)
__aicore__ inline GM_ADDR GetRelaySlotAddr(GM_ADDR relayAreaBase, uint32_t slotIdx, uint32_t slotSize)
{
    return relayAreaBase + L1_RELAY_HEADER_BYTES + static_cast<uint64_t>(slotIdx) * slotSize;
}

/// 中转区第 slotIdx 个 slot 的 data 部分地址 (slot 起始即 data 起始)
__aicore__ inline GM_ADDR GetRelaySlotDataAddr(GM_ADDR relayAreaBase, uint32_t slotIdx, uint32_t slotSize)
{
    return GetRelaySlotAddr(relayAreaBase, slotIdx, slotSize);
}

/// 中转区第 slotIdx 个 slot 的 metadata 部分地址 (data 之后紧跟 32B metadata)
__aicore__ inline GM_ADDR GetRelaySlotMetaAddr(GM_ADDR relayAreaBase, uint32_t slotIdx,
                                                  uint32_t slotSize, uint32_t dataBytes)
{
    return GetRelaySlotAddr(relayAreaBase, slotIdx, slotSize) + dataBytes;
}

// ============================================================
// 一级接收区 (L1 Receive Area) 布局辅助 — URMA 远端侧
// ============================================================
//
// 每个 source server 在对端 URMA 窗口的接收区布局：
//   稀疏布局, 按 tokenIdx 索引
//   [slot_0][slot_1]...[slot_m-1]  (步长 = slotSize, 512B 对齐)
//   每个 slot 总大小 = slotSize (tokendata + scaledata + 32B, 512B 对齐)
//     slot 内: [data: dataBytes][metadata/notify 区][padding 对齐至 512B]
//     整个 slot 的最后 2 个 int (slotSize - 8B) 为 WriteWithNotifyNbi notify(flag) 写入位置,
//     二级通信可直接读 slot 末尾 8B 获取 ready 标志。
//
// 注意: 接收区使用 totalServers 个 section (含本 server, 本 server section 不使用),
//       以简化偏移计算: sectionOffset = sourceServerId * perServerRecvSize

/// 计算单个 source server 接收区大小 (m 个 slot, 512B 对齐)
__aicore__ inline uint64_t CalcL1RecvPerServer(uint32_t m, uint32_t slotSize)
{
    int64_t perServer = static_cast<int64_t>(m) * slotSize;
    return static_cast<uint64_t>(Ops::Base::CeilAlign(perServer, static_cast<int64_t>(ALIGN_512)));
}

/// 对端接收区: source server section 基地址 (在 URMA 窗口内)
__aicore__ inline uint64_t GetL1RecvSectionOffset(uint32_t sourceServerId, uint64_t perServerRecvSize)
{
    return static_cast<uint64_t>(sourceServerId) * perServerRecvSize;
}

/// 对端接收区: tokenIdx 对应 slot 的偏移 (section 内, 步长 = slotSize)
__aicore__ inline uint64_t GetL1RecvTokenSlotOffset(uint32_t tokenIdx, uint32_t slotSize)
{
    return static_cast<uint64_t>(tokenIdx) * slotSize;
}

/// 对端接收区: tokenIdx 对应 slot 的 data 部分偏移 (slot 起始即 data 起始)
__aicore__ inline uint64_t GetL1RecvDataOffset(uint32_t tokenIdx, uint32_t slotSize)
{
    return GetL1RecvTokenSlotOffset(tokenIdx, slotSize);
}

/// 对端接收区: tokenIdx 对应 slot 的 notify 地址偏移 (整个 slot 的最后 2 个 int = 8B)
/// 二级通信读取 slot 末尾 8B 即可判断该 token 是否就绪
__aicore__ inline uint64_t GetL1RecvNotifyOffset(uint32_t tokenIdx, uint32_t slotSize)
{
    return GetL1RecvTokenSlotOffset(tokenIdx, slotSize) + slotSize - 2 * sizeof(int32_t);
}

// ============================================================
// 完成列表 (Complete List) 布局辅助
// ============================================================
//
// 每个 AIV 核在每个 expert round 完成后写 1 个 int32 标志位。
// Layout: [aivCore_0_round_0]..[aivCore_0_round_E-1]
//         [aivCore_1_round_0]..[aivCore_1_round_E-1] ...
// 发送核检查所有计算核的标志位, 确认当前 round 的数据全部就绪。

/// 计算核在 completeList 中的偏移
__aicore__ inline __gm__ int32_t* GetCompleteListAddr(GM_ADDR completeListPtr,
                                                        uint32_t aivCoreIdx, uint32_t expertPerRank)
{
    return reinterpret_cast<__gm__ int32_t*>(completeListPtr) + aivCoreIdx * expertPerRank;
}

// ============================================================
// 二级通信 (L2) 辅助函数
// ============================================================
//
// 二级通信: 本 server 内同层卡间读数据 (ReadNbi)
//   源卡 (一级通信接收方, 如 server1 卡0) 的 L1 接收区中存有跨 server 发来的数据
//   目标卡 (二级通信接收方, 如 server1 卡2) 用 ReadNbi 从源卡读数据到本卡 workspace
//
// L1 接收区 layout (在 URMA 窗口内, URMADispatchL1RecvPtr 偏移处):
//   按 sourceServer 分 section, 每 section 内按 tokenIdx 稀疏排列 slot
//   slot = [data: tokenScaleBytes][metadata: 32B]
//     metadata: int[6..7] = WriteWithNotifyNbi 写的 notify flag
//
// 二级通信读 flag 判断就绪, 然后读 data 到 dispatchRevDataPtr/dispatchRevScalePtr

/// L2 源地址: 远端卡 L1 接收区中指定 sourceServer section + tokenIdx slot 的 data 偏移
__aicore__ inline uint64_t GetL2SrcDataOffset(uint32_t sourceServerId, uint32_t tokenIdx,
                                                 uint64_t perServerRecvSize, uint32_t slotDataBytes)
{
    uint64_t sectionOffset = GetL1RecvSectionOffset(sourceServerId, perServerRecvSize);
    uint64_t tokenSlotOffset = static_cast<uint64_t>(tokenIdx) *
        (static_cast<uint64_t>(slotDataBytes) + L1_META_BYTES);
    return sectionOffset + tokenSlotOffset;
}

/// L2 源地址: 远端卡 L1 接收区中指定 tokenIdx slot 的 notify flag 偏移 (metadata int[6])
__aicore__ inline uint64_t GetL2SrcNotifyOffset(uint32_t sourceServerId, uint32_t tokenIdx,
                                                   uint64_t perServerRecvSize, uint32_t slotDataBytes)
{
    return GetL2SrcDataOffset(sourceServerId, tokenIdx, perServerRecvSize, slotDataBytes) +
        slotDataBytes + 6 * sizeof(int32_t);
}

/// 本卡 L1 接收区中, 本 server section + tokenIdx slot 的 notify flag 地址 (用于本卡读自己的 flag)
__aicore__ inline __gm__ int32_t* GetL2LocalNotifyAddr(GM_ADDR urmaBase, uint32_t myServerId,
                                                          uint32_t tokenIdx, uint64_t perServerRecvSize,
                                                          uint32_t slotDataBytes)
{
    uint64_t offset = GetL2SrcNotifyOffset(myServerId, tokenIdx, perServerRecvSize, slotDataBytes);
    return reinterpret_cast<__gm__ int32_t*>(urmaBase + offset);
}

} // namespace MegaMoeDispatchImpl

#endif // MEGA_MOE_DISPATCH_SEND_H
