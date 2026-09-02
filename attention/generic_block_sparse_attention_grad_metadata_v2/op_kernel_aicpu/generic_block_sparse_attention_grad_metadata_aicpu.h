#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_AICPU_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_AICPU_H

#include <cstdint>
#include <string>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {

/*
 * GenericBlockSparseAttentionGradMetadata 的 AICPU 实现：
 *   Compute            -> Prepare -> (BalanceSchedule -> GenMetadata)
 *   Prepare           读取 Host 已校验的输入 ABI，准备 per-batch 长度、前缀和 max 长度。
 *   BalanceSchedule   按 basic-block 粒度计算 K_OUT task 总数与分核。
 *   GenMetadata       写入 metadata header 和每个 AIC core 的 KTask。
 *
 * Metadata 输出 ABI 见 generic_block_sparse_attention_grad_metadata_layout.h。
 */
class GenericBlockSparseAttentionGradMetadataCpuKernel : public CpuKernel {
public:
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool BalanceSchedule();
    bool GenMetadata();

    // Read sequence values from the Host-validated tensors.
    uint32_t QSeqLen(uint32_t batchIdx) const;
    uint32_t KvSeqLen(uint32_t batchIdx) const;

    // Task partitioning uses the shared basic-block contract.
    static uint32_t GetBasicBlocks(uint32_t seqLen, uint32_t sparseBlockSize);
    static uint32_t DecodeBasicBlockStart(uint32_t basicBlockIdx, uint32_t sparseBlockSize);

    uint32_t GetOutputCapacityWords() const;
    bool FillKTask(uint32_t coreIdx, uint32_t startId, int64_t *out) const;

    // ---- input（device tensor，AICPU 侧 GetData 可直接读） ----
    Tensor *sparseBlockIdx_ = nullptr;
    Tensor *cuSeqQ_ = nullptr;
    Tensor *cuSeqKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedKv_ = nullptr;
    Tensor *blockShape_ = nullptr;
    // ---- output ----
    Tensor *metadata_ = nullptr;

    // ---- attr ----
    int64_t maxQSeqlenAttr_ = 0;
    int64_t maxKvSeqlenAttr_ = 0;
    int64_t numQHeads_ = 0;
    int64_t numKvHeads_ = 0;
    int64_t headDim_ = 0;
    std::string layoutQ_ = "TND";
    uint32_t aicCoreNum_ = 1;

    // ---- 派生 ----
    uint32_t batch_ = 0;
    uint32_t blockShapeY_ = 128;
    uint32_t inputLayout_ = 0; // 0=TND 1=BNSD 2=BSND
    uint32_t maxQSeqlen_ = 0;
    uint32_t maxKvSeqlen_ = 0;
    std::vector<uint32_t> qSeqlen_;
    std::vector<uint32_t> kvSeqlen_;
    std::vector<uint32_t> qPrefix_;  // 长度 batch+1，qPrefix_[b] = Σ_{i<b} qSeqlen_[i]
    std::vector<uint32_t> kvPrefix_; // 长度 batch+1

    std::vector<uint32_t> tasksInBatch_;
    uint32_t totalTaskNum_ = 0;
    uint32_t taskNumPerCore_ = 0;
    uint32_t tailTaskNum_ = 0;

    enum class ParamId : uint32_t {
        sparseBlockIdx = 0,
        sparseBlockCount = 1,
        cuSeqQ = 2,
        cuSeqKv = 3,
        sequsedQ = 4,
        sequsedKv = 5,
        blockShape = 6,
    };
};

} // namespace aicpu

#endif
