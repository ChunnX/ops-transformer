/**
 * aiv_urma_sqe.h — AIV-side URMA Write SQE byte-layout primitives.
 *
 * Layout & opcode bits vendored from CANN ops-transformer:
 *   mc2/moe_distribute_dispatch_setup/op_kernel/moe_distribute_base.h
 *
 * Subset kept here: enough to issue a plain URMA Write (no notify) and poll
 * its CQE. The channel descriptors (sqVa, dbVa, jfsID, remoteEID, tokenId,
 * tokenValue) are sourced from hcomm PR 2032's ChannelEntity at runtime —
 * see aiv_urma_channel_decode.h for the decoder.
 */

#ifndef AIV_URMA_SQE_H
#define AIV_URMA_SQE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#endif
#include "kernel_operator.h"

namespace CommWithUrmaImpl {

// Inline replacement for the vendor extension `SyncFunc<>` used by
// ops-transformer's mc2_kernel_utils.h (not present in stock CANN headers).
// Identical semantics: fetch an event id from TPipe and SetFlag/WaitFlag.
template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// 64-byte URMA Write SQE (no notify variant).
constexpr uint32_t WRITE_SQE_SIZE = 64U;
// 64-byte CQE.
constexpr uint32_t WRITE_CQE_SIZE = 64U;

// SQE byte-stream byte-offset (uint8 indexed).
constexpr uint32_t SQE_COMMON_UINT8_OFFSET_2 = 2U;   // udf_flg:1 inline_en:1 cqe:1 se:1 fence:1 odr:3
constexpr uint32_t SQE_COMMON_UINT8_OFFSET_3 = 3U;   // owner:1 rmt_jetty_type:2 token_en:1 nf:1 rsv:3
constexpr uint32_t SQE_COMMON_OPCODE_OFFSET  = 5U;
constexpr uint32_t SQE_COMMON_SGE_NUM_OFFSET = 11U;

// SQE u32-indexed offsets.
constexpr uint32_t SQE_COMMON_UINT32_OFFSET_2          = 2U;   // sge_num:8 tp_id:24
constexpr uint32_t SQE_COMMON_RMT_JETTY_OR_SEG_ID_OFFSET = 3U;
constexpr uint32_t SQE_COMMON_RMT_EID_31_0_OFFSET      = 4U;
constexpr uint32_t SQE_COMMON_RMT_EID_63_32_OFFSET     = 5U;
constexpr uint32_t SQE_COMMON_RMT_EID_95_64_OFFSET     = 6U;
constexpr uint32_t SQE_COMMON_RMT_EID_127_96_OFFSET    = 7U;
constexpr uint32_t SQE_COMMON_RMT_TOKEN_VALUE_OFFSET   = 8U;
constexpr uint32_t SQE_LENGTH_OFFSET                   = 12U;
constexpr uint32_t SQE_TOKEN_ID_OFFSET                 = 13U;  // rsv:12 token_id:20

// SQE u64-indexed offsets.
constexpr uint32_t SQE_RMT_ADDR_OFFSET  = 5U;
constexpr uint32_t SQE_DATA_ADDR_OFFSET = 7U;

// CQE byte-offset.
constexpr uint32_t CQE_SUBSTATUS_OFFSET = 2U;
constexpr uint32_t CQE_STATUS_OFFSET    = 3U;

/**
 * Initialize a 64-byte Write SQE template with static fields:
 *   odr=0b001, owner=1, rmt_jetty_type=0b01 (token_id), token_en=1, opcode=0x3 (write), sge_num=1
 *
 * After this the SQE still needs dynamic fields filled in via Set* / Fill*
 * helpers below.
 */
__aicore__ inline void GenerateCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor)
{
    AscendC::Duplicate<uint8_t>(sqeTensor, 0, WRITE_SQE_SIZE);
    SyncFunc<AscendC::HardEvent::V_S>();
    sqeTensor(SQE_COMMON_UINT8_OFFSET_2) = 0b001;       // odr=0b001
    sqeTensor(SQE_COMMON_UINT8_OFFSET_3) = 0b10110000;  // owner=1, rmt_jetty_type=0b01, token_en=1, nf=0
    sqeTensor(SQE_COMMON_OPCODE_OFFSET)  = 0x3;         // opcode=0x3 (URMA Write)
    sqeTensor(SQE_COMMON_SGE_NUM_OFFSET) = 1;           // sge_num=1
}

/**
 * Fill per-peer channel-driven fields: tp_id, rmt_jetty_or_seg_id (=remote
 * tokenId), rmt_eid[128], rmt_token_value. The SGE.token_id at byte 52 is
 * SOURCE-side and is set later via SetCommWriteSQE's srcTokenId argument.
 */
__aicore__ inline void FillCommWriteSQEFromChannel(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                   uint32_t tpID,
                                                   uint32_t remoteEID0,
                                                   uint32_t remoteEID1,
                                                   uint32_t remoteEID2,
                                                   uint32_t remoteEID3,
                                                   uint32_t remoteTokenId,
                                                   uint32_t remoteTokenValue)
{
    AscendC::LocalTensor<uint32_t> sqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    // sge_num (high 8 bits) is preserved by GenerateCommWriteSQE; mask in tp_id (low 24 bits).
    sqeU32(SQE_COMMON_UINT32_OFFSET_2)              &= 0xff000000;
    sqeU32(SQE_COMMON_UINT32_OFFSET_2)              |= (tpID & 0x00ffffff);
    sqeU32(SQE_COMMON_RMT_JETTY_OR_SEG_ID_OFFSET)    = remoteTokenId;
    sqeU32(SQE_COMMON_RMT_EID_31_0_OFFSET)           = remoteEID0;
    sqeU32(SQE_COMMON_RMT_EID_63_32_OFFSET)          = remoteEID1;
    sqeU32(SQE_COMMON_RMT_EID_95_64_OFFSET)          = remoteEID2;
    sqeU32(SQE_COMMON_RMT_EID_127_96_OFFSET)         = remoteEID3;
    sqeU32(SQE_COMMON_RMT_TOKEN_VALUE_OFFSET)        = remoteTokenValue;
}

/**
 * Fill the per-transfer fields: data source, remote destination, length, cqe flag.
 * Mirrors ops-transformer SetCommWriteSQE.
 */
__aicore__ inline void SetCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                       uint64_t dataAddr,
                                       uint64_t rmtAddr,
                                       uint32_t length,
                                       uint32_t srcTokenId,
                                       uint8_t cqe)
{
    AscendC::LocalTensor<uint32_t> sqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqeU64 = sqeTensor.ReinterpretCast<uint64_t>();

    sqeU64(SQE_RMT_ADDR_OFFSET)  = rmtAddr;
    sqeU64(SQE_DATA_ADDR_OFFSET) = dataAddr;
    sqeU32(SQE_LENGTH_OFFSET)    = length;
    sqeU32(SQE_TOKEN_ID_OFFSET)  = srcTokenId;  // SGE.token_id = SOURCE segment token
    if (cqe > 0U) {
        // cqe=1, odr=0b010 — request completion event
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00100010;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11111010;
    } else {
        // cqe=0, odr=0b001
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11011001;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00000001;
    }
}

/**
 * Copy a single 64B SQE from UB into a slot in the HBM-resident SQ ring,
 * then update sqPiLinear / sqPi locals (caller passes by-reference).
 */
__aicore__ inline void PutSQEToRing(uint64_t sqVa,
                                    uint32_t wqeSize,
                                    uint32_t sqDepth,
                                    const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                    uint32_t &sqPi,
                                    uint32_t &sqPiLinear)
{
    AscendC::GlobalTensor<uint8_t> sqGm;
    sqGm.SetGlobalBuffer((__gm__ uint8_t*)sqVa);
    AscendC::DataCopy(sqGm[wqeSize * sqPi], sqeTensor, WRITE_SQE_SIZE);
    ++sqPiLinear;
    sqPi = sqPiLinear % sqDepth;
}

/**
 * Copy `sqeCount` CONTIGUOUS 64B SQEs from a UB batch buffer into the SQ ring
 * starting at slot `sqPi`, in a SINGLE DataCopy when the batch fits before the
 * ring wrap, or two DataCopy's split at the wrap boundary. Advances
 * sqPi/sqPiLinear by sqeCount. Mirrors ops-transformer's PutCommSQE ring-copy.
 *
 * Caller must guarantee sqeCount <= sqDepth (leave >=1 slot free per URMA
 * convention). WRITE_SQE_SIZE (64) is 32B-aligned so every byte count here is a
 * legal DataCopy length. Assumes wqeSize == WRITE_SQE_SIZE (true for UB JFS on
 * this stack); the batch is laid out with a 64B stride to match the ring.
 */
__aicore__ inline void PutSQEBatchToRing(uint64_t sqVa,
                                         uint32_t wqeSize,
                                         uint32_t sqDepth,
                                         const AscendC::LocalTensor<uint8_t> &batchTensor,
                                         uint32_t sqeCount,
                                         uint32_t &sqPi,
                                         uint32_t &sqPiLinear)
{
    AscendC::GlobalTensor<uint8_t> sqGm;
    sqGm.SetGlobalBuffer((__gm__ uint8_t*)sqVa);
    if (sqDepth - sqPi >= sqeCount) {
        AscendC::DataCopy(sqGm[wqeSize * sqPi], batchTensor, WRITE_SQE_SIZE * sqeCount);
    } else {
        uint32_t firstPart  = sqDepth - sqPi;                 // slots to ring end
        uint32_t secondPart = sqeCount - firstPart;           // slots wrapped to ring start
        AscendC::DataCopy(sqGm[wqeSize * sqPi], batchTensor, WRITE_SQE_SIZE * firstPart);
        AscendC::DataCopy(sqGm, batchTensor[WRITE_SQE_SIZE * firstPart], WRITE_SQE_SIZE * secondPart);
    }
    sqPiLinear += sqeCount;
    sqPi = sqPiLinear % sqDepth;
}

/**
 * Ring the JFS doorbell to signal the new sqPi to hardware. `dbVa` is the
 * per-channel doorbell VA from SqContext::ubJfs.dbVa.
 */
__aicore__ inline void RingJFSDoorbell(uint64_t dbVa, uint32_t sqPi)
{
    st_dev(sqPi, (__gm__ uint32_t*)dbVa, 0);
}

/**
 * Poll one CQE slot at `cqVa + cqeSize*cqCi`. Returns true if the slot is
 * still invalid (status==0xff); false if a completion landed (caller should
 * advance cqCi). On real error status the kernel aborts via ascendc_assert.
 */
__aicore__ inline bool CqeSlotEmpty(uint64_t cqVa,
                                    uint32_t cqeSize,
                                    uint32_t cqCi,
                                    const AscendC::LocalTensor<uint8_t> &cqeTensor)
{
    AscendC::GlobalTensor<uint8_t> cqGm;
    cqGm.SetGlobalBuffer((__gm__ uint8_t*)cqVa);
    AscendC::DataCopyExtParams cp{1U, 8U, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(cqeTensor, cqGm[cqeSize * cqCi], cp, pad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint8_t status = cqeTensor(CQE_STATUS_OFFSET);
    if (status == 0xff) {
        return true;
    }
    ascendc_assert(status == 0,
                   "CQE status abnormal: status=%u substatus=%u",
                   status, cqeTensor(CQE_SUBSTATUS_OFFSET));
    // Invalidate slot for future reuse, write back.
    cqeTensor(CQE_STATUS_OFFSET) = 0xff;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(cqGm[cqeSize * cqCi], cqeTensor, cp);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    return false;
}

/**
 * Busy-wait `n` iterations of Nop<5>() (each = a PIPE_ALL fence + 5 asm nops +
 * a PIPE_ALL fence). Used to inject a configurable per-SQE delay in benchmarks.
 * Guarded like AscendC::Nop itself, which only exists for arch 3510/5102 — the
 * host pass of the unified compile doesn't set __NPU_ARCH__, so it must be a
 * no-op there.
 */
__aicore__ inline void SleepNop(uint32_t n)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102)
    for (uint32_t j = 0; j < n; ++j) {
        AscendC::Nop<5>();
    }
#else
    (void)n;
#endif
}

/**
 * One-time initialize a channel's CQ ring: set every CQE slot to the 0xff
 * "empty/invalid" sentinel that CqeSlotEmpty polls against.
 *
 * The CQ memory is not guaranteed to come up as 0xff (HCCL/the driver may leave
 * it zeroed or garbage), so the very first CqeSlotEmpty on a slot can misfire —
 * a zeroed slot reads status==0 and is mistaken for a landed completion, so the
 * poll returns without waiting. Blast the whole ring to 0xff once, on first use,
 * and crucially BEFORE the first doorbell so a real completion can never be
 * overwritten. CqeSlotEmpty re-invalidates each slot as it consumes it, so this
 * only needs to run once per channel per process. `scratch` holds `scratchSlots`
 * CQEs; the ring is filled in contiguous scratchSlots-sized chunks.
 */
__aicore__ inline void InvalidateCqRing(uint64_t cqVa,
                                        uint32_t cqeSize,
                                        uint32_t cqDepth,
                                        const AscendC::LocalTensor<uint8_t> &scratch,
                                        uint32_t scratchSlots)
{
    if (scratchSlots < 1U) scratchSlots = 1U;
    AscendC::Duplicate<uint8_t>(scratch, static_cast<uint8_t>(0xff), scratchSlots * cqeSize);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    AscendC::GlobalTensor<uint8_t> cqGm;
    cqGm.SetGlobalBuffer((__gm__ uint8_t*)cqVa);
    for (uint32_t done = 0; done < cqDepth; ) {
        uint32_t n = (cqDepth - done < scratchSlots) ? (cqDepth - done) : scratchSlots;
        AscendC::DataCopy(cqGm[cqeSize * done], scratch, cqeSize * n);
        done += n;
    }
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
}

}  // namespace CommWithUrmaImpl

#endif  // AIV_URMA_SQE_H
