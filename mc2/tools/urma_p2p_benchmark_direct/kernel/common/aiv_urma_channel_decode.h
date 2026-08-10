/**
 * aiv_urma_channel_decode.h — kernel-side decode of hcomm PR 2032 ChannelEntity.
 *
 * Reads the device-side serialized ChannelEntity (from HcommChannelGetPtrByHandle)
 * and extracts the fields needed to write/issue a URMA Write SQE.
 *
 * Layout source: hcomm PR 2032, pkg_inc/hcomm/hcomm_res_entity_defs.h
 * Packing source: aiv_urma_channel.cc PackChannelEntityHostBuf (pointers in
 * ChannelEntity are rewritten to point at device offsets inside the same
 * packed buffer, so a __gm__ pointer dereference walks them naturally.)
 *
 * We use raw byte offsets (computed once below) rather than vendoring the
 * union-heavy host struct definitions — keeps the kernel header tiny and
 * decouples us from CommAbiHeader / CommEngine / CommProtocol typedefs that
 * differ between stock CANN 9.0.0 and PR 2032's snapshot.
 */

#ifndef AIV_URMA_CHANNEL_DECODE_H
#define AIV_URMA_CHANNEL_DECODE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#endif
#include "kernel_operator.h"

namespace CommWithUrmaImpl {

// ---- ChannelEntity (256 B) ----
// 0..15  : CommAbiHeader (4×u32: version/magicWord/size/reserved)
// 16     : CommEngine (u32 enum)
// 20     : CommProtocol (u32 enum)
// 24..47 : 6 × u32 counts (localNotifyNum/remoteNotifyNum/localBufferNum/remoteBufferNum/sqNum/cqNum)
// 48..95 : 6 × u64 pointers (localNotifyAddr/remoteNotifyAddr/localBufferAddr/remoteBufferAddr/sqContextAddr/cqContextAddr)
// 96..255: reserve[160]
constexpr uint32_t CH_OFF_LOCAL_BUFFER_NUM  = 32U;  // localBufferNum   (u32)
constexpr uint32_t CH_OFF_REMOTE_BUFFER_NUM = 36U;  // remoteBufferNum  (u32)
constexpr uint32_t CH_OFF_LOCAL_BUFFER_PTR  = 64U;  // localBufferAddr  (u64)
constexpr uint32_t CH_OFF_REMOTE_BUFFER_PTR = 72U;  // remoteBufferAddr (u64)
constexpr uint32_t CH_OFF_SQ_CONTEXT_PTR    = 80U;  // sqContextAddr    (u64)
constexpr uint32_t CH_OFF_CQ_CONTEXT_PTR    = 88U;  // cqContextAddr    (u64)

// ---- SqContext (128 B) — only ubJfs variant used for AIV URMA ----
// 0..3   : SqContextType (u32 enum, expected SQ_CONTEXT_TYPE_UB_JFS = 0)
// 4..7   : pad
// 8..15  : sqVa (u64)
// 16..23 : headAddr (u64)
// 24..31 : tailAddr (u64)
// 32..39 : dbVa (u64)
// 40..43 : jfsID (u32)
// 44..47 : wqeSize (u32)
// 48..51 : sqDepth (u32)
// 52..55 : tpID (u32)
// 56..71 : remoteEID[16] (u8×16 = 4×u32)
constexpr uint32_t SQ_OFF_SQVA       = 8U;
constexpr uint32_t SQ_OFF_DBVA       = 32U;
constexpr uint32_t SQ_OFF_WQESIZE    = 44U;
constexpr uint32_t SQ_OFF_SQDEPTH    = 48U;
constexpr uint32_t SQ_OFF_TPID       = 52U;
constexpr uint32_t SQ_OFF_REMOTE_EID = 56U;  // 16 bytes

// ---- CqContext (128 B) — only ubJfc variant used for AIV URMA ----
// Same layout shape; offsets:
constexpr uint32_t CQ_OFF_CQVA    = 8U;
constexpr uint32_t CQ_OFF_DBVA    = 32U;
constexpr uint32_t CQ_OFF_CQESIZE = 44U;
constexpr uint32_t CQ_OFF_CQDEPTH = 48U;

// ---- RegedBufferEntity (64 B) — rma variant used for AIV URMA ----
// 0..3   : RegedBufferType (u32 enum, expected REGED_BUFFER_RMA = 1)
// 4..7   : pad
// 8..15  : addr (u64)
// 16..23 : size (u64)
// 24..27 : ProtectionType (u32 enum, expected PROTECTION_TYPE_UB = 1)
// 28..31 : ub.tokenId (u32)
// 32..35 : ub.tokenValue (u32)
constexpr uint32_t RB_OFF_ADDR        = 8U;
constexpr uint32_t RB_OFF_SIZE        = 16U;
constexpr uint32_t RB_OFF_UB_TOKEN_ID = 28U;
constexpr uint32_t RB_OFF_UB_TOKEN_VAL = 32U;

constexpr uint32_t CHANNEL_ENTITY_SIZE = 256U;
constexpr uint32_t SQ_CONTEXT_SIZE     = 128U;
constexpr uint32_t CQ_CONTEXT_SIZE     = 128U;
constexpr uint32_t REGED_BUFFER_SIZE   = 64U;

/**
 * Resolve one of the ChannelEntity's four container pointers (localBuffer /
 * remoteBuffer / sqContext / cqContext), which are packed one of two ways
 * depending on the hcomm build:
 *
 *   * old ABI (relative): stored as a small byte offset from the channel base.
 *     Its real device VA is `base + stored`. Seen on the PR2032-era sidecar
 *     (PackChannelEntityHostBuf) and CANN 9.1.0-beta's e533151 hcomm.
 *   * new ABI (absolute, CANN 9.1.0 master): stored as the absolute device VA
 *     `slabPtr + section.offset` (CopyChannelEntityToSlab / GetSlabPtr). Since
 *     the channel base handed to the kernel IS slabPtr, this is `base + offset`
 *     already — use it as-is.
 *
 * A relative offset is always tiny (< the slab size, a few KB) and thus far
 * below the multi-GB device VA of the channel base; an absolute VA is always
 * > base. So the magnitude alone disambiguates the two without a compile-time
 * or plumbed-through flag, keeping one binary correct on both hcomm ABIs.
 */
__aicore__ inline uint64_t ResolveChannelPtr(uint64_t base, uint64_t stored)
{
    return (stored < base) ? (base + stored) : stored;
}

/**
 * Per-peer decoded view of a ChannelEntity, ready to feed into the SQE
 * builders in aiv_urma_sqe.h. All values are extracted by Decode() below.
 */
struct DecodedChannel {
    // SQ
    uint64_t sqVa;
    uint64_t sqDbVa;
    uint32_t wqeSize;
    uint32_t sqDepth;
    uint32_t tpID;
    uint32_t remoteEID0;
    uint32_t remoteEID1;
    uint32_t remoteEID2;
    uint32_t remoteEID3;
    // CQ
    uint64_t cqVa;
    uint32_t cqeSize;
    uint32_t cqDepth;
    // local buffer = my own registered region (where peers write into via URMA)
    uint64_t localBufAddr;
    uint64_t localBufSize;
    // remote buffer = peer's registered region (where my URMA writes go) + URMA token
    uint64_t remoteBufAddr;
    uint64_t remoteBufSize;
    uint32_t remoteTokenId;
    uint32_t remoteTokenValue;
    // local registered segment token — needed for SGE's source token_id
    // when issuing URMA Writes with source IN this rank's localBuf.
    uint32_t localTokenId;
    uint32_t localTokenValue;
};

/**
 * DataCopy three sub-structs (256+128+128+64=576B) from GM into a single UB
 * scratch tensor, then extract fields. Scratch must hold at least 576 B.
 *
 * `channelEntityVA` is a `ChannelPtr` from HcommChannelGetPtrByHandle —
 * device VA of the packed ChannelEntity blob for this peer.
 */
__aicore__ inline void DecodePeerChannel(uint64_t channelEntityVA,
                                         const AscendC::LocalTensor<uint8_t> &scratch,
                                         DecodedChannel &out)
{
    AscendC::GlobalTensor<uint8_t> chGm;
    chGm.SetGlobalBuffer((__gm__ uint8_t*)channelEntityVA);
    AscendC::DataCopyExtParams chCp{1U, CHANNEL_ENTITY_SIZE, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> chPad{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(scratch, chGm, chCp, chPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    AscendC::LocalTensor<uint64_t> chU64 = scratch.ReinterpretCast<uint64_t>();
    AscendC::LocalTensor<uint32_t> chU32 = scratch.ReinterpretCast<uint32_t>();
    // The four container pointers are either RELATIVE offsets from the channel
    // base (old hcomm ABI) or ABSOLUTE device VAs (CANN 9.1.0). ResolveChannelPtr
    // picks the right interpretation by magnitude — correct on both ABIs.
    //
    // The localBuffer/remoteBuffer arrays hold MULTIPLE RegedBufferEntity: HCCL's
    // built-in ccl_buffer occupies the earlier slot(s) and OUR HcclCommMemReg
    // workspace is the LAST-registered one. On CANN 9.1.0 localBufferNum ==
    // remoteBufferNum == 2 (built-in + ours); on the 9.0.0 sidecar they were 1.
    // Selecting index (num-1) picks our workspace's addr/token on both — reading
    // index 0 grabs the built-in buffer (wrong addr+token → URMA write faults).
    uint32_t localBufNum  = chU32(CH_OFF_LOCAL_BUFFER_NUM  / sizeof(uint32_t));
    uint32_t remoteBufNum = chU32(CH_OFF_REMOTE_BUFFER_NUM / sizeof(uint32_t));
    uint32_t lcIdx = (localBufNum  > 0U) ? (localBufNum  - 1U) : 0U;
    uint32_t rmIdx = (remoteBufNum > 0U) ? (remoteBufNum - 1U) : 0U;
    uint64_t lcBufVA = ResolveChannelPtr(channelEntityVA, chU64(CH_OFF_LOCAL_BUFFER_PTR  / sizeof(uint64_t)))
                       + static_cast<uint64_t>(lcIdx) * REGED_BUFFER_SIZE;
    uint64_t rmBufVA = ResolveChannelPtr(channelEntityVA, chU64(CH_OFF_REMOTE_BUFFER_PTR / sizeof(uint64_t)))
                       + static_cast<uint64_t>(rmIdx) * REGED_BUFFER_SIZE;
    uint64_t sqCtxVA = ResolveChannelPtr(channelEntityVA, chU64(CH_OFF_SQ_CONTEXT_PTR    / sizeof(uint64_t)));
    uint64_t cqCtxVA = ResolveChannelPtr(channelEntityVA, chU64(CH_OFF_CQ_CONTEXT_PTR    / sizeof(uint64_t)));

    // ---- SqContext into scratch[256..384) ----
    AscendC::GlobalTensor<uint8_t> sqGm;
    sqGm.SetGlobalBuffer((__gm__ uint8_t*)sqCtxVA);
    AscendC::DataCopyExtParams sqCp{1U, SQ_CONTEXT_SIZE, 0U, 0U, 0U};
    AscendC::DataCopyPad(scratch[CHANNEL_ENTITY_SIZE], sqGm, sqCp, chPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // ---- CqContext into scratch[384..512) ----
    AscendC::GlobalTensor<uint8_t> cqGm;
    cqGm.SetGlobalBuffer((__gm__ uint8_t*)cqCtxVA);
    AscendC::DataCopyExtParams cqCp{1U, CQ_CONTEXT_SIZE, 0U, 0U, 0U};
    AscendC::DataCopyPad(scratch[CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE], cqGm, cqCp, chPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // ---- remote RegedBufferEntity into scratch[512..576) ----
    AscendC::GlobalTensor<uint8_t> rbGm;
    rbGm.SetGlobalBuffer((__gm__ uint8_t*)rmBufVA);
    AscendC::DataCopyExtParams rbCp{1U, REGED_BUFFER_SIZE, 0U, 0U, 0U};
    AscendC::DataCopyPad(scratch[CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE + CQ_CONTEXT_SIZE], rbGm, rbCp, chPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // ---- local RegedBufferEntity into scratch[576..640) ----
    AscendC::GlobalTensor<uint8_t> lcbGm;
    lcbGm.SetGlobalBuffer((__gm__ uint8_t*)lcBufVA);
    AscendC::DataCopyPad(scratch[CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE + CQ_CONTEXT_SIZE + REGED_BUFFER_SIZE],
                         lcbGm, rbCp, chPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // ---- field extraction ----
    AscendC::LocalTensor<uint64_t> sqU64 = scratch[CHANNEL_ENTITY_SIZE].ReinterpretCast<uint64_t>();
    AscendC::LocalTensor<uint32_t> sqU32 = scratch[CHANNEL_ENTITY_SIZE].ReinterpretCast<uint32_t>();
    out.sqVa       = sqU64(SQ_OFF_SQVA       / sizeof(uint64_t));
    out.sqDbVa     = sqU64(SQ_OFF_DBVA       / sizeof(uint64_t));
    out.wqeSize    = sqU32(SQ_OFF_WQESIZE    / sizeof(uint32_t));
    out.sqDepth    = sqU32(SQ_OFF_SQDEPTH    / sizeof(uint32_t));
    out.tpID       = sqU32(SQ_OFF_TPID       / sizeof(uint32_t));
    out.remoteEID0 = sqU32(SQ_OFF_REMOTE_EID / sizeof(uint32_t));
    out.remoteEID1 = sqU32(SQ_OFF_REMOTE_EID / sizeof(uint32_t) + 1U);
    out.remoteEID2 = sqU32(SQ_OFF_REMOTE_EID / sizeof(uint32_t) + 2U);
    out.remoteEID3 = sqU32(SQ_OFF_REMOTE_EID / sizeof(uint32_t) + 3U);

    AscendC::LocalTensor<uint64_t> cqU64 = scratch[CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE].ReinterpretCast<uint64_t>();
    AscendC::LocalTensor<uint32_t> cqU32 = scratch[CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE].ReinterpretCast<uint32_t>();
    out.cqVa    = cqU64(CQ_OFF_CQVA    / sizeof(uint64_t));
    out.cqeSize = cqU32(CQ_OFF_CQESIZE / sizeof(uint32_t));
    out.cqDepth = cqU32(CQ_OFF_CQDEPTH / sizeof(uint32_t));

    AscendC::LocalTensor<uint64_t> rbU64 =
        scratch[CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE + CQ_CONTEXT_SIZE].ReinterpretCast<uint64_t>();
    AscendC::LocalTensor<uint32_t> rbU32 =
        scratch[CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE + CQ_CONTEXT_SIZE].ReinterpretCast<uint32_t>();
    out.remoteBufAddr     = rbU64(RB_OFF_ADDR         / sizeof(uint64_t));
    out.remoteBufSize     = rbU64(RB_OFF_SIZE         / sizeof(uint64_t));
    out.remoteTokenId     = rbU32(RB_OFF_UB_TOKEN_ID  / sizeof(uint32_t));
    out.remoteTokenValue  = rbU32(RB_OFF_UB_TOKEN_VAL / sizeof(uint32_t));

    AscendC::LocalTensor<uint64_t> lcbU64 =
        scratch[CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE + CQ_CONTEXT_SIZE + REGED_BUFFER_SIZE].ReinterpretCast<uint64_t>();
    AscendC::LocalTensor<uint32_t> lcbU32 =
        scratch[CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE + CQ_CONTEXT_SIZE + REGED_BUFFER_SIZE].ReinterpretCast<uint32_t>();
    out.localBufAddr      = lcbU64(RB_OFF_ADDR / sizeof(uint64_t));
    out.localBufSize      = lcbU64(RB_OFF_SIZE / sizeof(uint64_t));
    out.localTokenId      = lcbU32(RB_OFF_UB_TOKEN_ID  / sizeof(uint32_t));
    out.localTokenValue   = lcbU32(RB_OFF_UB_TOKEN_VAL / sizeof(uint32_t));
}

constexpr uint32_t DECODE_SCRATCH_SIZE =
    CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE + CQ_CONTEXT_SIZE + 2U * REGED_BUFFER_SIZE;  // 640

}  // namespace CommWithUrmaImpl

#endif  // AIV_URMA_CHANNEL_DECODE_H
