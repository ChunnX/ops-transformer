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
 * \file mega_moe_combine_send.h
 * \brief
 */

 #ifndef MEGA_MOE_COMBINE_SEND_H
 #define MEGA_MOE_COMBINE_SEND_H
 
 #include "kernel_operator.h"
 #include "mega_moe_base.h"
 #if __has_include("../../common/mc2_kernel_utils.h")
 #include "../../common/mc2_kernel_utils.h"
 #include "../../moe_distribute_dispatch_v2/quantize_functions.h"
 #else
 #include "../../../common/op_kernel/mc2_kernel_utils.h"
 #include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
 #endif
 
 #include "mega_moe_impl_base.h"
 using namespace AscendC;
 
 namespace MegaMoeCombineImpl {
 constexpr uint32_t COMBINE_SEND_ADDR = 140 * 1024U; // metaInfo tensor 在 UB 中的起始地址
 constexpr uint32_t rankIdIndex = 0;
 constexpr uint32_t tokenIdxIndex = 1;
 constexpr uint32_t topkIdxIndex = 2;
 constexpr uint32_t blockLenIndex = 3;
 constexpr uint32_t tokenLenIndex = 4;
 constexpr uint32_t tokenActualLenIndex = 5;
 constexpr uint32_t flagIndex = 7;
 
 // 每个 N 方向 tile 块在 GM 上的字节数：数据区按 COMBINE_BLOCK_N 定长（尾块不足处为 padding），块尾附 flag。
 // 放在本文件而非 mega_moe_impl_base.h —— 后者会被 host tiling 编译，那里没有 __aicore__。
 template <typename T>
 __aicore__ inline constexpr uint32_t CombineBlockBytes()
 {
     return COMBINE_BLOCK_N * sizeof(T) + COMBINE_FLAG_BYTES;
 }
 
 // =============================================
 // InitCombineSendFlagBuf: 把 AIV 侧 [数据|flag] 拼接区整块铺成 COMBINE_FLAG_ARRIVED。
 //   之后每轮只覆盖各行的数据区（≤ L1_TILE_N*sizeof(T) 字节），块尾 32B flag 区永不被覆盖，
 //   故 flag 恒有效，无需每个 tile 重铺。调用点在 AivGmm2PostGeneric 的 tile 循环之前。
 // =============================================
 template <typename ElementMMadOut2>
 __aicore__ inline void InitCombineSendFlagBuf(uint32_t maxTileM)
 {
     constexpr uint32_t blockBytes = CombineBlockBytes<ElementMMadOut2>();
     LocalTensor<float> sendBufF(TPosition::VECCALC, MegaMoeImpl::COMBINE_SEND_BUF_ADDR,
                                 maxTileM * blockBytes / sizeof(float));
     AscendC::Duplicate<float>(sendBufF, COMBINE_FLAG_ARRIVED, maxTileM * blockBytes / sizeof(float));
 }
 
 // =============================================
 // CombineTokens: 把 GMM2 的一个 N 方向 tile 按 token 发回各自源卡。
 //   每个 tile 块在远端 slot 内的布局 = [数据区 L1_TILE_N*sizeof(T) | flag 区 32B]，
 //   数据与 flag 在同一次 DataCopyPad 中发出——flag 位于块尾，接收侧看到 flag 时数据必然已可见，
 //   接收侧(Unpermute)据此判定到达，无需 CrossRankSyncInWorldSize 全卡同步。
 //   发送长度固定为整块 blockBytes（尾块数据不满也发满），保证 flag 始终落在块尾固定偏移。
 // =============================================
 template <typename ElementMMadOut2, typename BlockShape>
 __aicore__ inline void CombineTokens(uint32_t mLoc, uint32_t nLoc, uint32_t n, LocalTensor<int32_t> &metaInfoTensor,
                                      LocalTensor<ElementMMadOut2> &l0cOutUbGMM2, BlockShape &actualBlockShape,
                                      const Params &params)
 {
     uint32_t lenTile = static_cast<uint32_t>(Get<M_VALUE>(actualBlockShape));
     uint32_t nValue = Get<N_VALUE>(actualBlockShape); // N_VALUE是当前tile块的n长度
     constexpr uint32_t blockBytes = CombineBlockBytes<ElementMMadOut2>();
     constexpr uint32_t blockElems = blockBytes / sizeof(ElementMMadOut2);
     constexpr uint32_t maxRows = MegaMoeImpl::COMBINE_SEND_MAX_ROWS;
     // 拼接区容量与 Copy repeat 上限的编译期兜底：lenTile 上界随 GMM2 的 tileM 走
     // （A8W4/A4W4 路径为 256，Generic 路径为 128），一律按 maxRows 分批消化，二者都不能越界。
     static_assert(maxRows * blockBytes <= MegaMoeImpl::COMBINE_SEND_BUF_BYTES,
                   "combine send buf overflow: COMBINE_SEND_MAX_ROWS * blockBytes exceeds "
                   "[COMBINE_SEND_BUF_ADDR, META_INFO_TENSOR_ADDR)");
     static_assert(maxRows <= 255U, "Copy repeatTimes is uint8_t, batch rows must fit");
 
     uint32_t dataBytes = nValue * sizeof(ElementMMadOut2);
     uint32_t blockIdxInToken = nLoc / MegaMoeImpl::L1_TILE_N; // 该 tile 在 token 内的块序号
     uint32_t blockCntPerToken =
         Ops::Base::CeilDiv(n, MegaMoeImpl::L1_TILE_N); // 一个 token 被切成多少个 N 方向块
 
     // 拼接区按 maxRows 行定长开，跨批复用；块尾 flag 区由 InitCombineSendFlagBuf 铺一次后永不被数据覆盖。
     LocalTensor<ElementMMadOut2> sendBuf(TPosition::VECCALC, MegaMoeImpl::COMBINE_SEND_BUF_ADDR,
                                          maxRows * blockElems);
     LocalTensor<float> sendBufF = sendBuf.template ReinterpretCast<float>();
     LocalTensor<float> srcF = l0cOutUbGMM2.template ReinterpretCast<float>();
     constexpr uint32_t floatPerRepeat = 64U; // 一次 Copy repeat 处理 256B
     uint32_t rowFloats = dataBytes / sizeof(float);
     uint32_t dstRowFloats = blockBytes / sizeof(float);
 
     AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
     uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
     AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
     ub2GmParams.blockCount = 1;
     ub2GmParams.blockLen = blockBytes; // 数据 + flag 一次发出
     // metaInfoTensor 由调用方在本函数前从 GM 载入，等一次 MTE2 即可，无需每批重等。
     AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
     AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
 
     for (uint32_t base = 0; base < lenTile; base += maxRows) {
         uint32_t rows = (lenTile - base) < maxRows ? (lenTile - base) : maxRows;
         // 第二批起：上一批的 DataCopyPad 仍在读 sendBuf，必须等 MTE3 读完才能用 V 覆盖同一块拼接区。
         if (base != 0U) {
             AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
             AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
         }
         // 数据 UB→UB 摆进拼接区：每行写入 [行首, 行首+dataBytes)，块尾 flag 区保持铺好的值。
         if (dataBytes % ALIGN_32 == 0) {
             // 快路径：源行距 32B 对齐，一次 Copy 覆盖本批 rows 行
             uint16_t dstRepStride = static_cast<uint16_t>(blockBytes / ALIGN_32);
             uint16_t srcRepStride = static_cast<uint16_t>(dataBytes / ALIGN_32);
             for (uint32_t off = 0; off < rowFloats; off += floatPerRepeat) {
                 uint64_t maskCnt = (rowFloats - off) < floatPerRepeat ? (rowFloats - off) : floatPerRepeat;
                 AscendC::Copy(sendBufF[off], srcF[base * rowFloats + off], maskCnt, static_cast<uint8_t>(rows),
                               {1, 1, dstRepStride, srcRepStride});
             }
         } else {
             // 兜底：源行距非 32B 对齐无法用 repeat stride 表达，逐行搬（repeat=1 时 repStride 不生效）
             for (uint32_t r = 0; r < rows; ++r) {
                 for (uint32_t off = 0; off < rowFloats; off += floatPerRepeat) {
                     uint64_t maskCnt = (rowFloats - off) < floatPerRepeat ? (rowFloats - off) : floatPerRepeat;
                     AscendC::Copy(sendBufF[r * dstRowFloats + off], srcF[(base + r) * rowFloats + off], maskCnt, 1,
                                   {1, 1, 8, 8});
                 }
             }
         }
         AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
 
         for (uint32_t r = 0; r < rows; ++r) {
             int32_t metaIdx = static_cast<int32_t>((base + r) * META_INFO_SIZE);
             uint32_t toRankId = metaInfoTensor.GetValue(metaIdx);
             uint32_t tokenIdx = metaInfoTensor.GetValue(metaIdx + 1);
             uint32_t topkIdx = metaInfoTensor.GetValue(metaIdx + 2);
             gmRemoteD.SetGlobalBuffer(
                 reinterpret_cast<__gm__ ElementMMadOut2 *>(GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
             uint64_t gmDstOffset =
                 ((static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * blockCntPerToken +
                  blockIdxInToken) *
                 blockElems;
             AscendC::DataCopyPad(gmRemoteD[gmDstOffset], sendBuf[r * blockElems], ub2GmParams);
         }
     }
 }
 
 template <typename ElementMMadOut2, typename BlockShape>
 __aicore__ inline void
 CombineTokensLayered(uint32_t mLoc, uint32_t nLoc, uint32_t n, LocalTensor<int32_t> &metaInfoTensor,
                      LocalTensor<ElementMMadOut2> &l0cOutUbGMM2, BlockShape &actualBlockShape, const Params &params)
 {
     int32_t lenTile = Get<M_VALUE>(actualBlockShape);
     uint32_t actualDataLength = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
     uint32_t maxDataLengthPerBlock =
         Ops::Base::CeilAlign(static_cast<int64_t>(MegaMoeImpl::L1_TILE_N * sizeof(ElementMMadOut2)), (int64_t)ALIGN_32);
     // 每个 token 含数据+三元组(32B)
     uint32_t maxDataSizePerToken =
         Ops::Base::CeilDiv(static_cast<int64_t>(params.tilingData->h), (int64_t)MegaMoeImpl::L1_TILE_N) *
         (ALIGN_32 + maxDataLengthPerBlock);
     LocalTensor<int32_t> metaInfoTempTensor =
         LocalTensor<int32_t>(TPosition::VECCALC, COMBINE_SEND_ADDR, ALIGN_32 / sizeof(int32_t));
     AscendC::GlobalTensor<ElementMMadOut2> gmLocalD;
     AscendC::GlobalTensor<int32_t> gmLocalDMetaInfo;
     AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
     ub2GmParams.blockCount = 1;
     ub2GmParams.blockLen = actualDataLength;
 
     AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
     AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
     for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
         uint32_t toRankId = metaInfoTensor.GetValue(tileIdx * 8);
         uint32_t tokenIdx = metaInfoTensor.GetValue(tileIdx * 8 + 1);
         uint32_t topkIdx = metaInfoTensor.GetValue(tileIdx * 8 + 2);
         if (toRankId == params.combineCommParams.rankId) {
             uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
             GM_ADDR localAddr = (GM_ADDR)(params.peermemInfo.combineSendPtr + gmDstOffset * sizeof(ElementMMadOut2));
             gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2 *>(localAddr));
             AscendC::DataCopyPad(gmLocalD, l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
             continue;
         }
         metaInfoTempTensor.SetValue(rankIdIndex, toRankId);
         metaInfoTempTensor.SetValue(tokenIdxIndex, tokenIdx);
         metaInfoTempTensor.SetValue(topkIdxIndex, topkIdx);
         metaInfoTempTensor.SetValue(blockLenIndex, nLoc);
         metaInfoTempTensor.SetValue(tokenLenIndex, n);
         metaInfoTempTensor.SetValue(tokenActualLenIndex, actualDataLength);
         metaInfoTempTensor.SetValue(flagIndex, 1); // 标记为已处理
         __gm__ int32_t *notifyAddr =
             (__gm__ int32_t *)(params.workspaceInfo.combineCommNotifyPtr + toRankId * sizeof(int32_t));
         GM_ADDR dataAddr = params.workspaceInfo.combineCommDataPtr + static_cast<uint64_t>(toRankId) *
                                                                          maxDataSizePerToken * params.tilingData->bs *
                                                                          params.tilingData->expertPerRank;
         int32_t cnt = AscendC::AtomicAdd(notifyAddr, 1);
         AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
         AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
         // 数据区: 每个 rank 独立空间, 每个槽位 = maxDataLength(数据) + 32B(三元组)
         gmLocalD.SetGlobalBuffer(
             reinterpret_cast<__gm__ ElementMMadOut2 *>(dataAddr + cnt * (maxDataLengthPerBlock + ALIGN_32)));
         AscendC::DataCopyPad(gmLocalD, l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
         AscendC::PipeBarrier<PIPE_MTE3>();
         // 三元组区: 紧跟数据之后
         gmLocalDMetaInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
             dataAddr + cnt * (maxDataLengthPerBlock + ALIGN_32) + maxDataLengthPerBlock));
         AscendC::DataCopy(gmLocalDMetaInfo, metaInfoTempTensor, ALIGN_32 / sizeof(int32_t));
         AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
     }
 }
 
 // =============================================
 // ComputeRecvTokenCounts: 从 mask 统计每个 rank 的应收 token 总数
 // 返回已完成 rank 数 (本卡 + 0-token)
 // =============================================
 template <typename ElementMMadOut2>
 __aicore__ inline uint32_t
 ComputeRecvTokenCounts(uint32_t startRankId, uint32_t endRankId, uint32_t rankId, uint32_t expertPerRank,
                        uint32_t worldSize, int64_t maskSlotSize, uint32_t slotCopyStride, uint32_t countWordIdx,
                        LocalTensor<uint32_t> &slotReadTensor, LocalTensor<uint8_t> &slotCopyTensor,
                        AscendC::GlobalTensor<uint8_t> &maskSrcGlobal, LocalTensor<int32_t> &totalTokenCnt,
                        const Params &params)
 {
     uint32_t completedRank = 0;
     for (uint32_t index = startRankId; index < endRankId; index++) {
         if (index == rankId) {
             completedRank++;
             continue;
         }
         uint32_t cnt = 0;
         for (uint32_t expertIdx = 0; expertIdx < expertPerRank; expertIdx++) {
             maskSrcGlobal.SetGlobalBuffer(
                 (__gm__ uint8_t *)(params.peermemInfo.maskRecvPtr +
                                    static_cast<uint64_t>(expertIdx) * worldSize * maskSlotSize +
                                    static_cast<uint64_t>(index) * maskSlotSize));
             AscendC::DataCopy(slotCopyTensor, maskSrcGlobal, slotCopyStride);
             SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
             cnt += slotReadTensor.GetValue(countWordIdx);
             SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID2>();
         }
         uint32_t totalDataBlockCnt =
             cnt * static_cast<uint32_t>(Ops::Base::CeilDiv(static_cast<int64_t>(params.tilingData->h),
                                                            static_cast<int64_t>(MegaMoeImpl::L1_TILE_N)));
         totalTokenCnt.SetValue(index - startRankId, totalDataBlockCnt);
         if (totalDataBlockCnt == 0) {
             completedRank++;
         }
     }
     return completedRank;
 }
 
 // =============================================
 // PollNotifyAndSendData: 轮询 notify 计数, 读取 metaInfo 并 URMA 发送数据
 // initialCompletedRank: Phase 1 已完成的 rank 数 (本卡 + 0-token)
 // =============================================
 template <typename ElementMMadOut2>
 __aicore__ inline void
 PollNotifyAndSendData(uint32_t startRankId, uint32_t initialCompletedRank, uint32_t processRankNum, uint32_t rankId,
                       uint32_t maxDataLengthPerBlock, uint32_t maxDataSizePerToken, LocalTensor<int32_t> &sendTokenCnt,
                       LocalTensor<int32_t> &totalTokenCnt, LocalTensor<int32_t> &sendTensor, const Params &params)
 {
     uint32_t completedRank = initialCompletedRank;
     uint32_t rankIndex = 0;
     AscendC::GlobalTensor<int32_t> gmLocalDMetaInfo;
     uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
 
     while (completedRank < processRankNum) {
         // 跳过本卡 rank 或已完成的 rank
         while ((rankIndex + startRankId == rankId) ||
                sendTokenCnt.GetValue(rankIndex) == totalTokenCnt.GetValue(rankIndex)) {
             rankIndex = (rankIndex + 1) % processRankNum;
         }
         // 读 notify count
         __gm__ int32_t *notifyAddr =
             (__gm__ int32_t *)(params.workspaceInfo.combineCommNotifyPtr + (rankIndex + startRankId) * sizeof(int32_t));
         uint32_t curCnt = ReadGmByPassDCache(notifyAddr);
         uint32_t completeCnt = sendTokenCnt.GetValue(rankIndex);
 
         if (curCnt > completeCnt) {
             uint32_t needCnt = curCnt - completeCnt;
             uint32_t slotIdx = completeCnt;
             while (needCnt > 0) {
                 GM_ADDR dataAddr = params.workspaceInfo.combineCommDataPtr +
                                    static_cast<uint64_t>(rankIndex + startRankId) * maxDataSizePerToken *
                                        params.tilingData->bs * params.tilingData->expertPerRank +
                                    slotIdx * (maxDataLengthPerBlock + ALIGN_32);
                 gmLocalDMetaInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dataAddr + maxDataLengthPerBlock));
                 AscendC::DataCopy(sendTensor, gmLocalDMetaInfo, 8);
                 SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
                 if (sendTensor.GetValue(flagIndex) == 1) {
                     uint32_t toRankId = sendTensor.GetValue(rankIdIndex);
                     uint32_t tokenIdx = sendTensor.GetValue(tokenIdxIndex);
                     uint32_t topkIdx = sendTensor.GetValue(topkIdxIndex);
                     uint32_t nLoc = sendTensor.GetValue(blockLenIndex);
                     uint32_t n = sendTensor.GetValue(tokenLenIndex);
                     uint32_t dataLen = sendTensor.GetValue(tokenActualLenIndex);
                     uint64_t channelHandle = GetUrmaCommHandle(params.combineCommParams.mc2Context, toRankId, rankId);
                     uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
                     GM_ADDR remoteAddr =
                         GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset) + gmDstOffset * sizeof(ElementMMadOut2);
                     params.combineCommParams.hcomm->WriteNbi(channelHandle, remoteAddr, dataAddr, dataLen);
                     // 标记已处理, 避免重复发送
                     sendTensor.SetValue(flagIndex, 0);
                     SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
                     AscendC::DataCopy(gmLocalDMetaInfo, sendTensor, 8);
                     SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
                     needCnt--;
                 }
                 slotIdx = (slotIdx + 1 < curCnt) ? slotIdx + 1 : completeCnt;
             }
             sendTokenCnt.SetValue(rankIndex, curCnt);
             if (curCnt == totalTokenCnt.GetValue(rankIndex)) {
                 completedRank++;
             }
         }
         rankIndex = (rankIndex + 1) % processRankNum;
     }
 }
 
 // =============================================
 // DrainUrmaConnections: drain 本核负责的 rank 的 urma 发送
 // =============================================
 __aicore__ inline void DrainUrmaConnections(uint32_t startRankId, uint32_t endRankId, uint32_t rankId,
                                             const Params &params)
 {
     for (uint32_t dr = startRankId; dr < endRankId; ++dr) {
         if (dr == rankId) {
             continue;
         }
         params.combineCommParams.hcomm->Drain(GetUrmaCommHandle(params.combineCommParams.mc2Context, dr, rankId));
     }
 }
 
 // =============================================
 // CombineSendTokenToRemote: 编排器 — 分核、VECCALC 分配、调用三阶段子函数
 // =============================================
 template <typename ElementMMadOut2>
 __aicore__ inline void CombineSendTokenToRemote(const Params &params)
 {
     uint32_t maxDataLengthPerBlock =
         Ops::Base::CeilAlign(static_cast<int64_t>(MegaMoeImpl::L1_TILE_N * sizeof(ElementMMadOut2)), (int64_t)ALIGN_32);
     uint32_t maxDataSizePerToken =
         Ops::Base::CeilDiv(static_cast<int64_t>(params.tilingData->h), (int64_t)MegaMoeImpl::L1_TILE_N) *
         (ALIGN_32 + maxDataLengthPerBlock);
     LocalTensor<int32_t> sendTensor = LocalTensor<int32_t>(TPosition::VECCALC, ALIGN_512, ALIGN_32 / sizeof(int32_t));
 
     // 分核: 仅 subBlockIdx_==1 的核参与, 核数 = GetBlockNum() (每个 block 的核1)
     uint32_t aivIdx = GetBlockIdx() / GetTaskRation();
     uint32_t aivNum = GetBlockNum();
     uint32_t rankId = params.combineCommParams.rankId;
     uint32_t worldSize = params.tilingData->epWorldSize;
     uint32_t rankPerCore = worldSize / aivNum;
     uint32_t remainder = worldSize % aivNum;
     uint32_t startRankId = rankPerCore * aivIdx + (aivIdx < remainder ? aivIdx : remainder);
     uint32_t processRankNum = rankPerCore + (aivIdx < remainder ? 1 : 0);
     uint32_t endRankId = startRankId + processRankNum;
     uint32_t cntTensorLength =
         Ops::Base::CeilAlign(static_cast<uint64_t>(processRankNum) * sizeof(uint32_t), (uint64_t)ALIGN_32);
     if (startRankId >= endRankId || processRankNum == 0) {
         return;
     }
 
     LocalTensor<int32_t> sendTokenCnt =
         LocalTensor<int32_t>(TPosition::VECCALC, ALIGN_32 + ALIGN_512, cntTensorLength / sizeof(int32_t));
     LocalTensor<int32_t> totalTokenCnt = LocalTensor<int32_t>(
         TPosition::VECCALC, ALIGN_32 + ALIGN_512 + cntTensorLength, cntTensorLength / sizeof(int32_t));
     AscendC::Duplicate<int32_t>(sendTokenCnt, 0, processRankNum);
     AscendC::Duplicate<int32_t>(totalTokenCnt, 0, processRankNum);
     SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
     SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID3>();
 
     // mask slot 参数
     int64_t sendTotalNum = static_cast<int64_t>(params.tilingData->bs) * params.tilingData->topK;
     int64_t compareCount =
         Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum) * (int64_t)sizeof(int32_t), (int64_t)ALIGN_256) /
         (int64_t)sizeof(int32_t);
     int64_t maskAlignSize = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount) / 8, (int64_t)ALIGN_32);
     int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32;
     uint32_t slotWordStride = static_cast<uint32_t>(maskSlotSize / sizeof(int32_t));
     uint32_t slotCopyStride = static_cast<uint32_t>(maskSlotSize / sizeof(uint8_t));
     uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize / sizeof(int32_t));
     uint32_t expertPerRank = params.tilingData->expertPerRank;
     LocalTensor<uint32_t> slotReadTensor =
         LocalTensor<uint32_t>(TPosition::VECCALC, ALIGN_32 + ALIGN_512 + cntTensorLength * 2, slotWordStride);
     LocalTensor<uint8_t> slotCopyTensor =
         LocalTensor<uint8_t>(TPosition::VECCALC, ALIGN_32 + ALIGN_512 + cntTensorLength * 2, slotCopyStride);
     AscendC::GlobalTensor<uint8_t> maskSrcGlobal;
 
     // Phase 1: 从 mask 统计每个 rank 的应收 token 总数
     uint32_t completedRank = ComputeRecvTokenCounts<ElementMMadOut2>(
         startRankId, endRankId, rankId, expertPerRank, worldSize, maskSlotSize, slotCopyStride, countWordIdx,
         slotReadTensor, slotCopyTensor, maskSrcGlobal, totalTokenCnt, params);
     if (completedRank == processRankNum) {
         return;
     }
 
     // Phase 2: 轮询 notify 计数并 URMA 发送数据
     PollNotifyAndSendData<ElementMMadOut2>(startRankId, completedRank, processRankNum, rankId, maxDataLengthPerBlock,
                                            maxDataSizePerToken, sendTokenCnt, totalTokenCnt, sendTensor, params);
 
     // Phase 3: drain 本核负责的 rank 的 urma 发送
     DrainUrmaConnections(startRankId, endRankId, rankId, params);
 }
 
 
 // =============================================
 // QuantMxFp8：将 bf16 数据量化为 MXFP8 格式
 // =============================================
 template <uint8_t QuantMode, typename ExpandXType>
 __aicore__ inline void QuantMxFp8(LocalTensor<ExpandXType> &outLocal, LocalTensor<ExpandXType> &inLocal,
                                   LocalTensor<float> &floatTemp, int32_t processLen)
 {
     PipeBarrier<PIPE_V>();
     uint32_t mxScaleNum = Align2(Ceil32(processLen));
     using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
     LocalTensor<Fp8Type> castFp8LocalTensor = outLocal.template ReinterpretCast<Fp8Type>();
     __ubuf__ ExpandXType *srcAddr = (__ubuf__ ExpandXType *)inLocal.GetPhyAddr();
     __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)floatTemp.GetPhyAddr();
     __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)floatTemp[Align32(mxScaleNum)].GetPhyAddr();
     __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)castFp8LocalTensor.GetPhyAddr();
     __ubuf__ uint16_t *mxScaleLocalAddr = (__ubuf__ uint16_t *)castFp8LocalTensor[processLen].GetPhyAddr();
     Quant::ComputeMaxExp(srcAddr, maxExpAddr, processLen); // 计算最大Exp
     // 计算scales并填充
     Quant::ComputeScale<Fp8Type>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
     Quant::ComputeFp8Data<ExpandXType, Fp8Type, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
         srcAddr, halfScaleLocalAddr, outLocalAddr, processLen); // 计算量化后的expandx并填充
 }
 
 // =============================================
 // DeQuantMxFp8：FP8 反量化，将 FP8 数据转换回 BF16/FP32
 // =============================================
 template <typename T, typename XType>
 __aicore__ inline void DeQuantMxFp8(LocalTensor<XType> &inLocal, LocalTensor<float> &sumTensor,
                                     LocalTensor<bfloat16_t> &scaleBf16Tensor, LocalTensor<float> &scaleFP32Tensor,
                                     uint32_t scaleLen, uint32_t tokenLen)
 {
     LocalTensor<T> castFp8LocalTensor_ = inLocal.template ReinterpretCast<T>();
     LocalTensor<fp8_e8m0_t> scaleDivFp8Tensor_ =
         inLocal[Align256<uint32_t>(tokenLen) / 2].template ReinterpretCast<fp8_e8m0_t>();
     __ubuf__ bfloat16_t *dyScaleBf16Ptr = (__ubuf__ bfloat16_t *)scaleBf16Tensor.GetPhyAddr();
     __ubuf__ float *dyScaleFp32Ptr = (__ubuf__ float *)scaleFP32Tensor.GetPhyAddr();
     __ubuf__ fp8_e8m0_t *srcPtr0 = (__ubuf__ fp8_e8m0_t *)scaleDivFp8Tensor_.GetPhyAddr();
     __ubuf__ T *tokenPtr0 = (__ubuf__ T *)castFp8LocalTensor_.GetPhyAddr();
     __ubuf__ float *sumDstPtr = (__ubuf__ float *)sumTensor.GetPhyAddr();
     uint32_t bf16RepeatSize = Quant::GetVRegSizeDispatch() / sizeof(bfloat16_t);
     uint32_t fp32RepeatSize = Quant::GetVRegSizeDispatch() / sizeof(float);
     uint16_t repeatTimes = Ceil(scaleLen, bf16RepeatSize);
     uint16_t fp32RepeatTimes = Ceil(tokenLen, fp32RepeatSize);
     uint16_t repeatTimes2 = Ceil(scaleLen * 2, fp32RepeatSize);
     uint32_t quantCount2 = scaleLen * 2;
     __VEC_SCOPE__
     {
         AscendC::MicroAPI::RegTensor<fp8_e8m0_t> vSrcReg;
         AscendC::MicroAPI::RegTensor<T> tokenSrcReg;
         AscendC::MicroAPI::RegTensor<float> tokenFp32SrcReg;
         AscendC::MicroAPI::RegTensor<bfloat16_t> vDstReg;
         AscendC::MicroAPI::RegTensor<bfloat16_t> dyScaleBf16Reg;
         AscendC::MicroAPI::RegTensor<float> dyScaleFp32Reg;
         AscendC::MicroAPI::RegTensor<float> sumDstReg;
         AscendC::MicroAPI::RegTensor<float> sumLocalDstReg;
         static constexpr AscendC::MicroAPI::CastTrait FP82BF16CastTraitZero = {
             AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
             AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
         static constexpr AscendC::MicroAPI::CastTrait FP162FP32CastTraitZero = {
             AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
             AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
         AscendC::MicroAPI::MaskReg maskReg;
         AscendC::MicroAPI::MaskReg maskReg1;
         AscendC::MicroAPI::MaskReg maskReg2;
         for (uint16_t i = 0; i < repeatTimes; i++) {
             maskReg = AscendC::MicroAPI::UpdateMask<bfloat16_t>(scaleLen);
             MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(vSrcReg, srcPtr0 + i * bf16RepeatSize);
             MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, FP82BF16CastTraitZero>(vDstReg, vSrcReg, maskReg);
             MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(dyScaleBf16Ptr + i * bf16RepeatSize * 2,
                                                                                 vDstReg, vDstReg, maskReg);
         }
         MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
         for (uint16_t i = 0; i < repeatTimes2; i++) {
             maskReg1 = AscendC::MicroAPI::UpdateMask<float>(quantCount2);
             MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(dyScaleBf16Reg,
                                                                                 dyScaleBf16Ptr + i * fp32RepeatSize);
             MicroAPI::Cast<float, bfloat16_t, FP162FP32CastTraitZero>(dyScaleFp32Reg, dyScaleBf16Reg, maskReg1);
             MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(dyScaleFp32Ptr + i * fp32RepeatSize * 2,
                                                                            dyScaleFp32Reg, dyScaleFp32Reg, maskReg1);
         }
         MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
         for (uint16_t i = 0; i < fp32RepeatTimes; i++) {
             maskReg2 = AscendC::MicroAPI::UpdateMask<float>(tokenLen);
             MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(dyScaleFp32Reg, dyScaleFp32Ptr + i * 8);
             MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(tokenSrcReg, tokenPtr0 + i * fp32RepeatSize);
             MicroAPI::Cast<float, T, FP82BF16CastTraitZero>(tokenFp32SrcReg, tokenSrcReg, maskReg2);
             MicroAPI::Mul(sumLocalDstReg, dyScaleFp32Reg, tokenFp32SrcReg, maskReg2);
             MicroAPI::DataCopy(sumDstPtr + i * fp32RepeatSize, sumLocalDstReg, maskReg2);
         }
     }
 }
 
 // =============================================
 // CombineQuantizedTokens：将量化后的 token 发送到目标 rank
 // =============================================
 // 量化 combine 路径下一个 token 整体一次发出，故每个 slot 只需一份 flag：
 //   slot 布局 = [量化数据区 quantSlotBytes-32 | flag 区 32B]，数据与 flag 同一次 DataCopyPad 发出。
 template <typename QuantOutType>
 __aicore__ inline void CombineQuantizedTokens(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                               uint32_t groupIdx, uint32_t rankId, LocalTensor<int32_t> &metaInfoTensor,
                                               LocalTensor<QuantOutType> &ubQuant, const Params &params,
                                               uint32_t quantSlotBytes)
 {
     uint32_t toRankId = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + RANK_ID);
     uint32_t tokenIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOKEN_ID);
     uint32_t topkIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOPK_INDEX);
 
     AscendC::GlobalTensor<QuantOutType> gmRemoteD;
     uint64_t gmRemoteOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
     __gm__ void *dstPeermemPtr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
     gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(dstPeermemPtr));
 
     uint64_t dstBaseOffset = (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * quantSlotBytes;
 
     AscendC::DataCopyExtParams singleCopyParams{1, quantSlotBytes, 0, 0, 0};
     AscendC::DataCopyPad(gmRemoteD[dstBaseOffset], ubQuant, singleCopyParams);
 }
 
 // =============================================
 // CombineTokenGroup：处理一个 token group 的 Combine 操作，从 GMM2 输出读取数据，量化后发送到目标 rank
 //   DataAsFlag=true（非 layered 路径）：slot 尾部带 32B flag，与数据一次发出，接收侧靠 flag 判到达；
 //     此时 quantTokenSizeBytes 已含 flag 区（见 InitCombineBuffers）。
 //   DataAsFlag=false（layered 路径）：保持原布局与原发送长度 n+nScale，行为不变。
 // =============================================
 template <uint8_t QuantMode, typename T, bool DataAsFlag = false>
 __aicore__ inline void CombineTokenGroup(uint32_t tokenStart, uint32_t tokenCount, uint32_t n, uint32_t groupIdx,
                                          uint32_t rankId, GM_ADDR gmm2OutAddr, const Params &params,
                                          LocalTensor<int32_t> &metaInfoTensor, int64_t ubTensorSize, int64_t offset,
                                          uint32_t quantTokenSizeBytes)
 {
     LocalTensor<T> combineUbTensor(TPosition::VECIN, offset, ubTensorSize);
     offset += ubTensorSize * sizeof(T);
 
     uint32_t nScale = Ops::Base::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
     uint32_t nAlign32 = Ops::Base::CeilAlign(n, static_cast<uint32_t>(ALIGN_32));
     uint32_t floatTempSize = nScale + nScale / 2;
     LocalTensor<float> floatTemp = LocalTensor<float>(TPosition::VECIN, offset, floatTempSize);
 
     GlobalTensor<T> gmm2OutGm;
     gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gmm2OutAddr));
 
     using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
 
     uint32_t singleTokenElems = (nAlign32 * sizeof(T) + quantTokenSizeBytes) / sizeof(T);
     DataCopyPadExtParams<T> copyPadParams{false, 0U, 0U, 0U};
     AscendC::DataCopyExtParams gm2UbParams{static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
 
     // 发送长度：data-as-flag 路径按含 flag 的 slot 定长发；layered 路径保持原来的 n+nScale。
     uint32_t slotBytes = DataAsFlag ? quantTokenSizeBytes : static_cast<uint32_t>(n + nScale);
     if constexpr (DataAsFlag) {
         // quantTokenSizeBytes 已含块尾 32B flag 区（见 InitCombineBuffers）。
         // 量化输出 QuantMxFp8 只写 [0, n+nScale)，flag 区不会被覆盖，故两个 pingpong 槽各铺一次即可。
         uint32_t flagFloatOffset = (quantTokenSizeBytes - COMBINE_FLAG_BYTES) / sizeof(float);
         for (uint32_t pp = 0; pp < 2; pp++) {
             LocalTensor<float> slotF =
                 combineUbTensor[pp * singleTokenElems + nAlign32].template ReinterpretCast<float>();
             AscendC::Duplicate<float>(slotF[flagFloatOffset], COMBINE_FLAG_ARRIVED,
                                       COMBINE_FLAG_FLOAT_NUM);
         }
         SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
     }
 
     for (uint32_t i = 0; i < tokenCount; i++) {
         uint32_t pingPong = i % 2;
         LocalTensor<T> ubBf16 = combineUbTensor[pingPong * singleTokenElems];
         LocalTensor<T> ubQuantData = ubBf16[nAlign32];
 
         // MTE2: read from GM
         SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
         AscendC::DataCopyPad(ubBf16, gmm2OutGm[(tokenStart + i) * n], gm2UbParams, copyPadParams);
         SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID4>();
 
         // V: quantize
         QuantMxFp8<QuantMode, T>(ubQuantData, ubBf16, floatTemp, n);
         SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID5>();
 
         // MTE3: send to GM
         LocalTensor<Fp8Type> ubQuantDataFp8 = ubQuantData.template ReinterpretCast<Fp8Type>();
         CombineQuantizedTokens<Fp8Type>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor, ubQuantDataFp8, params,
                                         slotBytes);
     }
 
     // Wait for all MTE3 operations to complete
     SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
 }
 
 } // namespace MegaMoeCombineImpl
 
 #endif // MEGA_MOE_COMBINE_SEND_H