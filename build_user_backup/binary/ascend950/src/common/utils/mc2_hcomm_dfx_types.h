// /**
//  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /*!
//  * \file mc2_hcomm_dfx_types.h
//  * \brief HCOMM channel DFX types for SQE/CQE dump.
//  *
//  * Struct layouts are vendored from asc-comm hcomm_inner_def.h to enable host-side
//  * D2H reading of ChannelEntity / SqContext / CqContext during exception callbacks.
//  * Only the types needed for SQ/CQ context dump are included; batch handle types
//  * that depend on device-only LocalTensor are omitted.
//  */
// #ifndef MC2_HCOMM_DFX_TYPES_H
// #define MC2_HCOMM_DFX_TYPES_H

// #include <cstdint>

// namespace AscendC {

// // ---- Enums (from hcomm_common.h + hcomm_inner_def.h) ----

// using ChannelHandle = uint64_t;

// typedef enum {
//     COMM_PROTOCOL_RESERVED = -1,
//     COMM_PROTOCOL_HCCS = 0,
//     COMM_PROTOCOL_ROCE = 1,
//     COMM_PROTOCOL_PCIE = 2,
//     COMM_PROTOCOL_SIO = 3,
//     COMM_PROTOCOL_UBC_CTP = 4,
//     COMM_PROTOCOL_UBC_TP = 5,
//     COMM_PROTOCOL_UB_MEM = 6,
//     COMM_PROTOCOL_UBOE = 7,
//     COMM_PROTOCOL_HCCS_ONLY = 8,
// } CommProtocol;

// typedef enum {
//     COMM_ENGINE_RESERVED = -1,
//     COMM_ENGINE_CPU = 0,
//     COMM_ENGINE_CPU_TS = 1,
//     COMM_ENGINE_AICPU = 2,
//     COMM_ENGINE_AICPU_TS = 3,
//     COMM_ENGINE_AIV = 4,
//     COMM_ENGINE_CCU = 5,
// } CommEngine;

// typedef enum {
//     PROTECTION_TYPE_INVALID = -1,
//     PROTECTION_TYPE_ROCE = 0,
//     PROTECTION_TYPE_UB = 1,
// } ProtectionType;

// typedef enum {
//     SQ_CONTEXT_TYPE_INVALID = -1,
//     SQ_CONTEXT_TYPE_UB_JFS = 0,
//     SQ_CONTEXT_TYPE_ROCE = 1,
// } SqContextType;

// typedef enum {
//     CQ_CONTEXT_TYPE_INVALID = -1,
//     CQ_CONTEXT_TYPE_UB_JFC = 0,
//     CQ_CONTEXT_TYPE_ROCE = 1,
// } CqContextType;

// typedef enum {
//     REGED_NOTIFY_INVALID = -1,
//     REGED_NOTIFY_IPC_RT = 0,
//     REGED_NOTIFY_IPC_MEM = 1,
//     REGED_NOTIFY_RMA_RT = 2,
//     REGED_NOTIFY_RMA_MEM = 3,
// } RegedNotifyType;

// typedef enum {
//     REGED_BUFFER_INVALID = -1,
//     REGED_BUFFER_IPC = 0,
//     REGED_BUFFER_RMA = 1,
// } RegedBufferType;

// // ---- ABI header ----

// typedef struct {
//     uint32_t version;
//     uint32_t magicWord;
//     uint32_t size;
//     uint32_t reserved;
// } CommAbiHeader;

// // ---- Protection / Buffer / Notify entities ----

// typedef struct {
//     ProtectionType type;
//     union {
//         struct {
//             uint32_t lkey;
//             uint32_t rkey;
//         } roce;
//         struct {
//             uint32_t tokenId;
//             uint32_t tokenValue;
//         } ub;
//         uint8_t raws[24];
//     } memInfo;
// } ProtectionInfo; // 32B

// typedef struct {
//     RegedBufferType type;
//     union {
//         struct {
//             uint64_t addr;
//             uint64_t size;
//         } ipc;
//         struct {
//             uint64_t addr;
//             uint64_t size;
//             ProtectionInfo protectionInfo;
//         } rma;
//         uint8_t raws[56];
//     } bufferInfo;
// } RegedBufferEntity; // 64B

// typedef struct {
//     RegedNotifyType type;
//     union {
//         struct {
//             uint64_t addr;
//             uint32_t size;
//             int32_t notifyId;
//         } ipcRt;
//         struct {
//             uint64_t addr;
//             uint32_t size;
//         } ipcMem;
//         struct {
//             uint64_t addr;
//             uint32_t size;
//             int32_t notifyId;
//             ProtectionInfo protectionInfo;
//         } rmaRt;
//         struct {
//             uint64_t addr;
//             uint32_t size;
//             ProtectionInfo protectionInfo;
//         } rmaMem;
//         uint8_t raws[56];
//     } notifyInfo;
// } RegedNotifyEntity; // 64B

// // ---- SQ / CQ context ----

// typedef struct {
//     SqContextType type;
//     union {
//         struct {
//             uint64_t sqVa;
//             uint64_t headAddr;
//             uint64_t tailAddr;
//             uint64_t dbVa;
//             uint32_t jfsID;
//             uint32_t wqeSize;
//             uint32_t sqDepth;
//             uint32_t tpID;
//             uint8_t remoteEID[16];
//         } ubJfs;
//         struct {
//             uint64_t sqVa;
//             uint64_t headAddr;
//             uint64_t tailAddr;
//             uint64_t dbHwVa;
//             uint64_t dbSwVa;
//             uint32_t qpn;
//             uint32_t wqeSize;
//             uint32_t depth;
//             uint8_t sl;
//             uint8_t mtuShift;
//         } roceSq;
//         uint8_t raws[120];
//     } contextInfo;
// } SqContext;

// typedef struct {
//     CqContextType type;
//     union {
//         struct {
//             uint64_t scqVa;
//             uint64_t headAddr;
//             uint64_t tailAddr;
//             uint64_t dbVa;
//             uint32_t jfcID;
//             uint32_t cqeSize;
//             uint32_t cqDepth;
//         } ubJfc;
//         struct {
//             uint64_t cqVa;
//             uint64_t headAddr;
//             uint64_t tailAddr;
//             uint64_t dbHwVa;
//             uint64_t dbSwVa;
//             uint32_t cqn;
//             uint32_t cqeSize;
//             uint32_t cqDepth;
//         } roceCq;
//         uint8_t raws[120];
//     } contextInfo;
// } CqContext;

// // ---- Channel entity (the struct ChannelHandle points to) ----

// typedef struct {
//     CommAbiHeader abiHeader;
//     CommEngine engine;
//     CommProtocol protocol;
//     uint32_t localNotifyNum;
//     uint32_t remoteNotifyNum;
//     uint32_t localBufferNum;
//     uint32_t remoteBufferNum;
//     uint32_t sqNum;
//     uint32_t cqNum;
//     RegedNotifyEntity* localNotifyAddr;
//     RegedNotifyEntity* remoteNotifyAddr;
//     RegedBufferEntity* localBufferAddr;
//     RegedBufferEntity* remoteBufferAddr;
//     SqContext* sqContextAddr;
//     CqContext* cqContextAddr;
//     uint32_t sqHead;
//     uint32_t sqTail;
//     uint32_t cqHead;
//     uint32_t cqTail;
//     uint8_t reserve[144];
// } ChannelEntity;

// static_assert(sizeof(ChannelEntity) == 256, "ChannelEntity size must keep aligned with hcomm");

// // ---- URMA WQE (SQE) / CQE decoded structs ----

// typedef struct {
//     uint32_t sqeBbIdx : 16;
//     uint32_t flag : 8;
//     uint32_t rsv0 : 3;
//     uint32_t nf : 1;
//     uint32_t tokenEn : 1;
//     uint32_t rmtJettyType : 2;
//     uint32_t owner : 1;
//     uint32_t targetHint : 8;
//     uint32_t opcode : 8;
//     uint32_t rsv1 : 6;
//     uint32_t inlineMsgLen : 10;
//     uint32_t tpId : 24;
//     uint32_t sgeNum : 8;
//     uint32_t rmtJettyOrSegId : 20;
//     uint32_t rsv2 : 12;
//     uint64_t rmtEidL;
//     uint64_t rmtEidH;
//     uint32_t rmtTokenValue;
//     uint32_t udfType : 8;
//     uint32_t reduceDataType : 4;
//     uint32_t reduceOpcode : 4;
//     uint32_t rsv3 : 16;
//     uint32_t rmtAddrLOrTokenId;
//     uint32_t rmtAddrHOrTokenValue;
// } HcommUrmaSqeCtx;

// typedef struct {
//     uint32_t sR : 1;
//     uint32_t isJetty : 1;
//     uint32_t owner : 1;
//     uint32_t inlineEn : 1;
//     uint32_t opcode : 3;
//     uint32_t fd : 1;
//     uint32_t rsv : 8;
//     uint32_t substatus : 8;
//     uint32_t status : 8;
//     uint32_t entryIdx : 16;
//     uint32_t localNumL : 16;
//     uint32_t localNumH : 4;
//     uint32_t rmtIdx : 20;
//     uint32_t rsv1 : 8;
//     uint32_t tpn : 24;
//     uint32_t rsv2 : 8;
//     uint32_t byteCnt;
//     uint32_t userDataL;
//     uint32_t userDataH;
//     uint32_t rmtEid[4];
//     uint32_t dataL;
//     uint32_t dataH;
//     uint32_t inlineData[3];
// } HcommUrmaJfcCqeCtx;

// } // namespace AscendC

// #endif // MC2_HCOMM_DFX_TYPES_H
