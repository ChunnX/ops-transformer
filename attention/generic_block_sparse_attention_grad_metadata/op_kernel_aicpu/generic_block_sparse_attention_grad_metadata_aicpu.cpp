/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "generic_block_sparse_attention_grad_metadata_aicpu.h"

#include <cstring>
#include <climits>
#include <algorithm>

#include "status.h"
#include "log.h"
#include "../../common/op_kernel/aicpu_common.h"
#include "generic_block_sparse_attention_grad_metadata_layout.h"

namespace aicpu {

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    if (!Prepare(ctx)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (!BalanceSchedule()) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (!GenMetadata()) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernel::GetOutputCapacityWords() const
{
    if (metadata_ == nullptr || metadata_->GetDataSize() == 0) {
        return 0;
    }
    return static_cast<uint32_t>(metadata_->GetDataSize() / sizeof(int32_t));
}

bool GenericBlockSparseAttentionGradMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    using PI = ParamId;
    sparseBlockIdx_ = ctx.Input(static_cast<uint32_t>(PI::sparseBlockIdx));
    cuSeqQ_ = ctx.Input(static_cast<uint32_t>(PI::cuSeqQ));
    cuSeqKv_ = ctx.Input(static_cast<uint32_t>(PI::cuSeqKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(PI::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(PI::sequsedKv));
    blockShape_ = ctx.Input(static_cast<uint32_t>(PI::blockShape));
    metadata_ = ctx.Output(0);

    // ---- required attrs ----
    bool ok = GetAttrValue(ctx, "max_q_seqlen", maxQSeqlenAttr_) &&
              GetAttrValue(ctx, "max_kv_seqlen", maxKvSeqlenAttr_) &&
              GetAttrValue(ctx, "num_q_heads", numQHeads_) &&
              GetAttrValue(ctx, "num_kv_heads", numKvHeads_) &&
              GetAttrValue(ctx, "head_dim", headDim_);
    if (!ok) {
        return false;
    }
    // ---- optional attrs ----
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);

    if (metadata_ == nullptr || metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata output is null");
        return false;
    }
    if (sparseBlockIdx_ == nullptr || sparseBlockIdx_->GetData() == nullptr ||
        sparseBlockIdx_->GetTensorShape() == nullptr) {
        KERNEL_LOG_ERROR("sparseBlockIdx is null");
        return false;
    }

    // Host guarantees a supported layout; only decode the ABI value here.
    inputLayout_ = layoutQ_ == "TND"  ? gbsag_meta::LAYOUT_TND :
                   layoutQ_ == "BNSD" ? gbsag_meta::LAYOUT_BNSD :
                                        gbsag_meta::LAYOUT_BSND;

    // 契约：TND 场景 cuSeq 必传（int64 前缀和数组，长度 b+1）；
    // seqused 仅供 packed 布局（BNSD/BSND）作为可选长度来源。
    if (inputLayout_ == gbsag_meta::LAYOUT_TND &&
        ((cuSeqQ_ == nullptr || cuSeqQ_->GetData() == nullptr) ||
         (cuSeqKv_ == nullptr || cuSeqKv_->GetData() == nullptr))) {
        KERNEL_LOG_ERROR("TND layout requires both cuSeqQ and cuSeqKv inputs");
        return false;
    }

    // batch_ is used to size the derived arrays below, so the first dimension must exist.
    const auto sparseBlockIdxShape = sparseBlockIdx_->GetTensorShape();
    if (sparseBlockIdxShape == nullptr || sparseBlockIdxShape->GetDims() < 1) {
        KERNEL_LOG_ERROR("sparseBlockIdx shape has no batch dimension");
        return false;
    }
    batch_ = static_cast<uint32_t>(sparseBlockIdxShape->GetDimSize(0));

    // blockShape is read below; retain only the checks required for safe access and division.
    if (blockShape_ == nullptr || blockShape_->GetData() == nullptr ||
        blockShape_->GetTensorShape() == nullptr ||
        blockShape_->GetTensorShape()->GetDimSize(0) < 2) {
        KERNEL_LOG_ERROR("blockShape input is null or malformed");
        return false;
    }
    const int64_t *blockShape = static_cast<const int64_t *>(blockShape_->GetData());
    blockShapeY_ = static_cast<uint32_t>(blockShape[1]);
    if (blockShapeY_ == 0) {
        KERNEL_LOG_ERROR("blockShapeY must be positive");
        return false;
    }

    // Build per-batch lengths, prefixes and maximum lengths.
    qSeqlen_.assign(batch_, 0);
    kvSeqlen_.assign(batch_, 0);
    qPrefix_.assign(batch_ + 1, 0);
    kvPrefix_.assign(batch_ + 1, 0);

    // Preserve physical-stride semantics for packed layouts:
    //   TND      : 由 per-batch 值推导（max over b）。
    //   BNSD/BSND: 保持物理 shape（attr max_q_seqlen / max_kv_seqlen），
    //              用于 batch 物理 stride 与 maskQBlockNum/maskKvBlockNum，
    //              不能取 seqused 的实际最大值（否则 < S 维时 stride 会错位）。
    maxQSeqlen_ = static_cast<uint32_t>(maxQSeqlenAttr_);
    maxKvSeqlen_ = static_cast<uint32_t>(maxKvSeqlenAttr_);
    for (uint32_t b = 0; b < batch_; ++b) {
        const uint32_t qLen = QSeqLen(b);
        const uint32_t kvLen = KvSeqLen(b);
        qSeqlen_[b] = qLen;
        kvSeqlen_[b] = kvLen;
        if (inputLayout_ == gbsag_meta::LAYOUT_TND) {
            maxQSeqlen_ = std::max(maxQSeqlen_, qLen);
            maxKvSeqlen_ = std::max(maxKvSeqlen_, kvLen);
        }
        qPrefix_[b + 1] = qPrefix_[b] + qLen;
        kvPrefix_[b + 1] = kvPrefix_[b] + kvLen;
    }

    return true;
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernel::QSeqLen(uint32_t b) const
{
    // TND 契约：cuSeq 必传（Prepare 已校验），长度取前缀和差分；
    // packed 布局（BNSD/BSND）seqused 可选，缺省回退物理 max。
    if (inputLayout_ == gbsag_meta::LAYOUT_TND) {
        const int64_t *cu = static_cast<const int64_t *>(cuSeqQ_->GetData());
        return static_cast<uint32_t>(cu[b + 1] - cu[b]);
    }
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *used = static_cast<const int32_t *>(sequsedQ_->GetData());
        return static_cast<uint32_t>(used[b]);
    }
    return static_cast<uint32_t>(maxQSeqlenAttr_);
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernel::KvSeqLen(uint32_t b) const
{
    if (inputLayout_ == gbsag_meta::LAYOUT_TND) {
        const int64_t *cu = static_cast<const int64_t *>(cuSeqKv_->GetData());
        return static_cast<uint32_t>(cu[b + 1] - cu[b]);
    }
    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        const int32_t *used = static_cast<const int32_t *>(sequsedKv_->GetData());
        return static_cast<uint32_t>(used[b]);
    }
    return static_cast<uint32_t>(maxKvSeqlenAttr_);
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernel::GetBasicBlocks(uint32_t seqLen,
                                                                          uint32_t sparseBlockSize)
{
    if (sparseBlockSize == 0) {
        return 0;
    }
    const uint32_t blocksPerSparse = (sparseBlockSize + gbsag_meta::BASIC_BLOCK_SIZE - 1) /
                                     gbsag_meta::BASIC_BLOCK_SIZE;
    const uint32_t fullSparseBlocks = seqLen / sparseBlockSize;
    const uint32_t tail = seqLen % sparseBlockSize;
    return fullSparseBlocks * blocksPerSparse +
           (tail + gbsag_meta::BASIC_BLOCK_SIZE - 1) / gbsag_meta::BASIC_BLOCK_SIZE;
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernel::DecodeBasicBlockStart(uint32_t basicBlockIdx,
                                                                                 uint32_t sparseBlockSize)
{
    const uint32_t blocksPerSparse = (sparseBlockSize + gbsag_meta::BASIC_BLOCK_SIZE - 1) /
                                     gbsag_meta::BASIC_BLOCK_SIZE;
    if (blocksPerSparse == 0) {
        return 0;
    }
    return (basicBlockIdx / blocksPerSparse) * sparseBlockSize +
           (basicBlockIdx % blocksPerSparse) * gbsag_meta::BASIC_BLOCK_SIZE;
}

bool GenericBlockSparseAttentionGradMetadataCpuKernel::BalanceSchedule()
{
    const uint32_t kvHeads = static_cast<uint32_t>(numKvHeads_);
    if (kvHeads == 0) {
        KERNEL_LOG_ERROR("numKvHeads must be positive");
        return false;
    }

    totalTaskNum_ = 0;
    tasksInBatch_.assign(batch_, 0);
    for (uint32_t b = 0; b < batch_; ++b) {
        const uint32_t kBlocks = GetBasicBlocks(kvSeqlen_[b], blockShapeY_);
        tasksInBatch_[b] = kBlocks * kvHeads;
        totalTaskNum_ += tasksInBatch_[b];
    }
    if (totalTaskNum_ == 0) {
        KERNEL_LOG_ERROR("totalTaskNum is 0");
        return false;
    }

    const uint32_t coreNum = std::min<uint32_t>(
        std::min<uint32_t>(std::max<uint32_t>(aicCoreNum_, 1), gbsag_meta::MAX_AIC_CORE_NUM),
        totalTaskNum_);
    taskNumPerCore_ = totalTaskNum_ / coreNum;
    tailTaskNum_ = totalTaskNum_ % coreNum;

    // 输出容量防护：保证实际 totalLen ≤ 分配上界。
    const uint32_t actualLen = gbsag_meta::MetadataWords(coreNum);
    if (GetOutputCapacityWords() < actualLen) {
        KERNEL_LOG_ERROR("metadata capacity %u words < required %u words", GetOutputCapacityWords(), actualLen);
        return false;
    }
    return true;
}

bool GenericBlockSparseAttentionGradMetadataCpuKernel::FillKTask(uint32_t coreIdx,
                                                                 uint32_t startId,
                                                                 uint32_t *out) const
{
    const uint32_t count = taskNumPerCore_ + (coreIdx < tailTaskNum_ ? 1U : 0U);
    const uint32_t taskId = startId;
    if (count == 0) {
        return true;
    }
    uint32_t remaining = taskId;
    uint32_t b = 0;
    while (b + 1 < batch_ && remaining >= tasksInBatch_[b]) {
        remaining -= tasksInBatch_[b++];
    }
    uint32_t h = 0;
    uint32_t kBlk = 0;
    if (b < batch_) {
        if (inputLayout_ == gbsag_meta::LAYOUT_TND) {
            kBlk = remaining / static_cast<uint32_t>(numKvHeads_);
            h = remaining % static_cast<uint32_t>(numKvHeads_);
        } else {
            const uint32_t kBlocks = GetBasicBlocks(kvSeqlen_[b], blockShapeY_);
            if (kBlocks != 0) {
                h = remaining / kBlocks;
                kBlk = remaining % kBlocks;
            }
        }
    }
    auto *task = reinterpret_cast<gbsag_meta::KTask *>(out + gbsag_meta::HEADER_WORDS) + coreIdx;
    task->beginBatch = b;
    task->beginKvHead = h;
    task->beginKvSeqOffset = DecodeBasicBlockStart(kBlk, blockShapeY_);
    task->reserved = count;
    const uint64_t preQ = inputLayout_ == gbsag_meta::LAYOUT_TND ? qPrefix_[b] : static_cast<uint64_t>(b) * static_cast<uint64_t>(numQHeads_) * maxQSeqlen_;
    const uint64_t preKv = inputLayout_ == gbsag_meta::LAYOUT_TND ? kvPrefix_[b] : static_cast<uint64_t>(b) * static_cast<uint64_t>(numKvHeads_) * maxKvSeqlen_;
    task->preQLo = static_cast<uint32_t>(preQ);
    task->preQHi = static_cast<uint32_t>(preQ >> 32);
    task->preKvLo = static_cast<uint32_t>(preKv);
    task->preKvHi = static_cast<uint32_t>(preKv >> 32);
    return true;
}

bool GenericBlockSparseAttentionGradMetadataCpuKernel::GenMetadata()
{
    uint32_t *out = reinterpret_cast<uint32_t *>(metadata_->GetData());
    if (GetOutputCapacityWords() < gbsag_meta::MetadataWords(std::max<uint32_t>(aicCoreNum_, 1))) {
        return false;
    }
    std::memset(out, 0, metadata_->GetDataSize());
    out[gbsag_meta::OFF_MAGIC] = gbsag_meta::MAGIC;
    const uint32_t coreNum = std::min<uint32_t>(
        std::min<uint32_t>(std::max<uint32_t>(aicCoreNum_, 1), gbsag_meta::MAX_AIC_CORE_NUM),
        totalTaskNum_);
    out[gbsag_meta::OFF_TOTAL_LEN] = gbsag_meta::MetadataWords(coreNum);
    out[gbsag_meta::OFF_BATCH] = batch_;
    out[gbsag_meta::OFF_NUM_Q_HEADS] = static_cast<uint32_t>(numQHeads_);
    out[gbsag_meta::OFF_NUM_KV_HEADS] = static_cast<uint32_t>(numKvHeads_);
    out[gbsag_meta::OFF_HEAD_DIM] = static_cast<uint32_t>(headDim_);
    out[gbsag_meta::OFF_INPUT_LAYOUT] = inputLayout_;
    out[gbsag_meta::OFF_MAX_Q_SEQLEN] = maxQSeqlen_;
    out[gbsag_meta::OFF_MAX_KV_SEQLEN] = maxKvSeqlen_;
    out[gbsag_meta::OFF_TOTAL_TASK_NUM] = totalTaskNum_;
    out[gbsag_meta::OFF_TASK_NUM_PER_CORE] = taskNumPerCore_;
    out[gbsag_meta::OFF_TAIL_TASK_NUM] = tailTaskNum_;
    out[gbsag_meta::OFF_SEQ_LEN_OFFSET] = gbsag_meta::HEADER_WORDS;
    out[gbsag_meta::OFF_TASK_TABLE_OFFSET] = gbsag_meta::HEADER_WORDS;
    out[gbsag_meta::OFF_CORE_NUM] = coreNum;
    out[gbsag_meta::OFF_TASK_WORDS] = gbsag_meta::KTASK_WORDS;
    uint32_t startId = 0;
    for (uint32_t core = 0; core < coreNum; ++core) {
        if (!FillKTask(core, startId, out)) {
            return false;
        }
        startId += taskNumPerCore_ + (core < tailTaskNum_ ? 1U : 0U);
    }
    return startId == totalTaskNum_;
}

namespace {
static const char *kernelType = "GenericBlockSparseAttentionGradMetadata";
REGISTER_CPU_KERNEL(kernelType, GenericBlockSparseAttentionGradMetadataCpuKernel);
} // namespace

} // namespace aicpu
