/**
 * aiv_urma_sqe_mc2.h — AIV-side URMA WriteWithNotify SQE primitives.
 *
 * Based on mc2/moe_distribute_base.h's WriteWithNotify SQE layout:
 *   96-byte SQE, opcode=0x5, includes notifyAddr/notifyData fields.
 *
 * Channel fields (sqVa, dbVa, eid, token, etc.) are sourced from the
 * same DecodedChannel struct used by aiv_urma_sqe.h (the plain Write
 * variant), so both modes share the same channel decode path.
 */

#ifndef AIV_URMA_SQE_MC2_H
#define AIV_URMA_SQE_MC2_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#endif
#include "kernel_operator.h"
#include "../sqe_write/aiv_urma_sqe.h"
#include "../common/aiv_urma_channel_decode.h"

namespace CommWithUrmaImpl {

// ---- WriteWithNotify SQE size (96 bytes, occupies 2 WQEBBs of 64B each) ----
constexpr uint32_t WRITE_WITH_NOTIFY_SQE_SIZE = 96U;

// ---- SQE byte-stream byte-offset (uint8 indexed, shared with plain Write) ----
constexpr uint32_t WDN_UINT8_OFFSET_2       = 2U;   // udf_flg:1 inline_en:1 cqe:1 se:1 fence:1 odr:3
constexpr uint32_t WDN_UINT8_OFFSET_3       = 3U;   // owner:1 rmt_jetty_type:2 token_en:1 nf:1 rsv:3
constexpr uint32_t WDN_OPCODE_OFFSET         = 5U;
constexpr uint32_t WDN_SGE_NUM_OFFSET        = 11U;

// ---- SQE u32-indexed offsets (shared fields) ----
constexpr uint32_t WDN_U32_SGE_TP_ID         = 2U;   // sge_num:8 tp_id:24
constexpr uint32_t WDN_U32_RMT_JETTY_ID      = 3U;   // rmt_jetty_or_seg_id
constexpr uint32_t WDN_U32_RMT_EID_0         = 4U;   // rmt_eid[31:0]
constexpr uint32_t WDN_U32_RMT_EID_1         = 5U;   // rmt_eid[63:32]
constexpr uint32_t WDN_U32_RMT_EID_2         = 6U;   // rmt_eid[95:64]
constexpr uint32_t WDN_U32_RMT_EID_3         = 7U;   // rmt_eid[127:96]
constexpr uint32_t WDN_U32_RMT_TOKEN_VALUE   = 8U;   // rmt_token_value

// ---- WriteWithNotify-specific u32 offsets ----
constexpr uint32_t WDN_U32_NOTIFY_TOKEN_ID    = 12U;  // rsv:12 notify_token_id:20
constexpr uint32_t WDN_U32_NOTIFY_TOKEN_VALUE = 13U;  // notify_token_value
constexpr uint32_t WDN_U32_LENGTH             = 20U;  // length
constexpr uint32_t WDN_U32_TOKEN_ID           = 21U;  // rsv:12 token_id:20 (rmt_token_id)

// ---- WriteWithNotify-specific u64 offsets ----
constexpr uint32_t WDN_U64_RMT_ADDR     = 5U;   // remoteAddr (byte 40)
constexpr uint32_t WDN_U64_NOTIFY_ADDR  = 7U;   // notifyAddr  (byte 56)
constexpr uint32_t WDN_U64_NOTIFY_DATA  = 8U;   // notifyData  (byte 64)
constexpr uint32_t WDN_U64_DATA_ADDR    = 11U;  // localAddr   (byte 88)

// ---- Default notify flag (1.0f as uint32 = 0x3F800000) ----
constexpr uint64_t WDN_DEFAULT_NOTIFY_DATA = 0x3F800000UL;

/**
 * Initialize a 96-byte WriteWithNotify SQE template:
 *   odr=0b001, owner=1, rmt_jetty_type=0b01, token_en=1, opcode=0x5, sge_num=1
 */
__aicore__ inline void GenerateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor)
{
    AscendC::Duplicate<uint8_t>(sqeTensor, 0, WRITE_WITH_NOTIFY_SQE_SIZE);
    SyncFunc<AscendC::HardEvent::V_S>();
    sqeTensor(WDN_UINT8_OFFSET_2) = 0b001;       // odr=0b001
    sqeTensor(WDN_UINT8_OFFSET_3) = 0b10110000;  // owner=1, rmt_jetty_type=0b01, token_en=1
    sqeTensor(WDN_OPCODE_OFFSET)  = 0x5;         // opcode=0x5 (WriteWithNotify)
    sqeTensor(WDN_SGE_NUM_OFFSET) = 1;           // sge_num=1
}

/**
 * Fill per-peer channel-driven fields from DecodedChannel:
 *   tp_id, rmt_jetty_id, rmt_eid[128], rmt_token_value,
 *   rmt_token_id, notify_token_value, notify_token_id
 */
__aicore__ inline void FillCommWriteWithNotifySQEFromChannel(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                              const DecodedChannel &ch)
{
    AscendC::LocalTensor<uint32_t> sqeU32 = sqeTensor.ReinterpretCast<uint32_t>();

    // tp_id (preserve sge_num high 8 bits)
    sqeU32(WDN_U32_SGE_TP_ID) &= 0xff000000;
    sqeU32(WDN_U32_SGE_TP_ID) |= (ch.tpID & 0x00ffffff);

    // rmt_jetty_id = remoteTokenId
    sqeU32(WDN_U32_RMT_JETTY_ID) = ch.remoteTokenId;

    // rmt_eid[128]
    sqeU32(WDN_U32_RMT_EID_0) = ch.remoteEID0;
    sqeU32(WDN_U32_RMT_EID_1) = ch.remoteEID1;
    sqeU32(WDN_U32_RMT_EID_2) = ch.remoteEID2;
    sqeU32(WDN_U32_RMT_EID_3) = ch.remoteEID3;

    // rmt_token_value
    sqeU32(WDN_U32_RMT_TOKEN_VALUE) = ch.remoteTokenValue;

    // rmt_token_id (same as rmt_jetty_id per mc2 convention)
    sqeU32(WDN_U32_TOKEN_ID) = ch.remoteTokenId;

    // notify_token_id and notify_token_value (same as rmt per mc2 convention)
    sqeU32(WDN_U32_NOTIFY_TOKEN_ID)    = ch.remoteTokenId;
    sqeU32(WDN_U32_NOTIFY_TOKEN_VALUE) = ch.remoteTokenValue;
}

/**
 * Fill per-transfer fields: data source, remote dest, length, notify addr/data, cqe flag.
 */
__aicore__ inline void SetCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                  uint64_t dataAddr, uint64_t rmtAddr,
                                                  uint32_t length, uint64_t notifyAddr,
                                                  uint64_t notifyData, uint8_t cqe)
{
    AscendC::LocalTensor<uint32_t> sqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqeU64 = sqeTensor.ReinterpretCast<uint64_t>();

    sqeU64(WDN_U64_RMT_ADDR)    = rmtAddr;
    sqeU64(WDN_U64_DATA_ADDR)   = dataAddr;
    sqeU32(WDN_U32_LENGTH)      = length;
    sqeU64(WDN_U64_NOTIFY_ADDR) = notifyAddr;
    sqeU64(WDN_U64_NOTIFY_DATA) = notifyData;

    if (cqe > 0U) {
        sqeTensor(WDN_UINT8_OFFSET_2) |= 0b00100010;
        sqeTensor(WDN_UINT8_OFFSET_2) &= 0b11111010;
    } else {
        sqeTensor(WDN_UINT8_OFFSET_2) &= 0b11011001;
        sqeTensor(WDN_UINT8_OFFSET_2) |= 0b00000001;
    }
}

/**
 * Copy a single 96-byte WriteWithNotify SQE from UB into the HBM SQ ring.
 * Each 96-byte SQE occupies 2 WQEBBs (128 bytes) in the ring, so sqPiLinear
 * advances by 2 and sqPi wraps mod sqDepth.
 */
__aicore__ inline void PutWriteWithNotifySQEToRing(uint64_t sqVa,
                                                    uint32_t wqeSize,
                                                    uint32_t sqDepth,
                                                    const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                    uint32_t &sqPi,
                                                    uint32_t &sqPiLinear)
{
    AscendC::GlobalTensor<uint8_t> sqGm;
    sqGm.SetGlobalBuffer((__gm__ uint8_t*)sqVa);
    // 96 bytes = 3 × 32B blocks, fits in one DataCopy
    AscendC::DataCopyParams params{1U, static_cast<uint16_t>(WRITE_WITH_NOTIFY_SQE_SIZE / 32U), 0U, 0U};
    AscendC::DataCopy(sqGm[wqeSize * sqPi], sqeTensor, params);
    sqPiLinear += 2U;  // each WriteWithNotify SQE occupies 2 WQEBBs
    sqPi = sqPiLinear % sqDepth;
}

}  // namespace CommWithUrmaImpl

#endif  // AIV_URMA_SQE_MC2_H
