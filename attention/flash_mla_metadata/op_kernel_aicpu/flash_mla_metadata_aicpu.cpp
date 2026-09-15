/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "flash_mla_metadata_aicpu_sequence.h"
#include "../op_host/flash_mla_metadata_config.h"
#include "../../common/op_kernel/aicpu_common.h"
#include "../../flash_mla/op_host/flash_mla_value_tiling.h"
#include <limits>

namespace aicpu {
namespace {
using namespace FlashMla;
using flash_mla_metadata::PrepareSeq;
constexpr uint32_t kStatusOk = 0;
constexpr uint32_t kStatusInvalid = 1;

template <typename T>
bool ReadUnsignedAttr(CpuKernelContext &ctx, const char *name, T &value)
{
    int64_t input;
    if (!GetAttrValue(ctx, name, input) || input < 0 ||
        static_cast<uint64_t>(input) > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
        return false;
    }
    value = static_cast<T>(input);
    return true;
}

bool ResolveSeqSize(int64_t requested, bool cumulative, uint32_t batchSize, MlaSeqView seq, uint64_t &size)
{
    if (requested == kMlaInferDimension) {
        if (!cumulative || batchSize == 0 || batchSize > 65536U || seq.count != batchSize || seq.data == nullptr) {
            return false;
        }
        // Validate before accepting a zero total: empty schedules bypass the
        // normal value tiler's checks, so a decreasing [positive, 0] must fail here.
        int64_t previous = 0;
        for (uint32_t i = 0; i < batchSize; ++i) {
            const int64_t value = seq.data[i];
            if (value < previous) {
                return false;
            }
            previous = value;
        }
        requested = previous;
    }
    if (requested < 0) {
        return false;
    }
    size = static_cast<uint64_t>(requested);
    return true;
}

bool Prepare(CpuKernelContext &ctx, MlaScheduleConfig &c, MlaPlatformInfo &p, MlaSeqView &q, MlaSeqView &kv)
{
    std::string inputLayer;
    int64_t quantMode;
    int64_t qSeqSize, kvSeqSize;
    bool mask;
    if (!ReadUnsignedAttr(ctx, "num_heads", c.numHeads) || !ReadUnsignedAttr(ctx, "batch_size", c.batchSize) ||
        !GetAttrValue(ctx, "q_seq_size", qSeqSize) || !GetAttrValue(ctx, "kv_seq_size", kvSeqSize) ||
        !GetAttrValue(ctx, "input_layer", inputLayer) || !ReadUnsignedAttr(ctx, "block_size", c.blockSize) ||
        !ReadUnsignedAttr(ctx, "sparse_mode", c.sparseMode) || !GetAttrValue(ctx, "pre_tokens", c.preTokens) ||
        !GetAttrValue(ctx, "next_tokens", c.nextTokens) || !GetAttrValue(ctx, "has_atten_mask", mask) ||
        !GetAttrValue(ctx, "quant_mode", quantMode) || !ReadUnsignedAttr(ctx, "aic_core_num", p.aicNum) ||
        !ReadUnsignedAttr(ctx, "aiv_core_num", p.aivNum) || !ConfigureMlaMetadata(inputLayer.c_str(), quantMode, c)) {
        return false;
    }
    c.hasAttenMask = mask;
    p.architecture = kMlaArch22;
    return PrepareSeq(ctx.Input(0), q, c.seqQPresent, c.seqQCount) &&
           PrepareSeq(ctx.Input(1), kv, c.seqKvPresent, c.seqKvCount) &&
           ResolveSeqSize(qSeqSize, c.layoutQ == MlaLayout::TND, c.batchSize, q, c.qSeqSize) &&
           ResolveSeqSize(kvSeqSize, c.layoutKv == MlaLayout::TND, c.batchSize, kv, c.kvSeqSize) &&
           NormalizeMlaScheduleConfig(c) == MlaStatus::SUCCESS;
}

bool CheckMetadataCapacity(const Tensor *tensor)
{
    return tensor != nullptr && tensor->GetData() != nullptr && tensor->GetDataType() == DT_INT32 &&
           tensor->GetTensorShape() != nullptr && tensor->GetTensorShape()->GetDims() == 1 &&
           tensor->GetTensorShape()->GetDimSize(0) == kMlaMetadataElements &&
           (reinterpret_cast<uintptr_t>(tensor->GetData()) & 63U) == 0;
}
} // namespace

class FlashMlaMetadataAicpu final : public CpuKernel {
public:
    uint32_t Compute(CpuKernelContext &ctx) override
    {
        auto *output = ctx.Output(0);
        if (!CheckMetadataCapacity(output)) {
            KERNEL_LOG_ERROR("FlashMlaMetadata requires aligned INT32[480] output.");
            return kStatusInvalid;
        }
        InvalidateMlaMetadata(output->GetData());
        MlaScheduleConfig config;
        MlaPlatformInfo platform;
        MlaSeqView q, kv;
        if (!Prepare(ctx, config, platform, q, kv)) {
            KERNEL_LOG_ERROR("FlashMlaMetadata scheduling attributes or sequence descriptors are invalid.");
            return kStatusInvalid;
        }
        MlaValuePatch patch;
        const auto status = ComputeMlaValueTiling(config, platform, q, kv, patch);
        if (WriteMlaMetadata(output->GetData(), kMlaMetadataBytes, patch) != MlaStatus::SUCCESS) {
            KERNEL_LOG_ERROR("FlashMlaMetadata value tiling failed, status=%u, detail=%u.",
                             static_cast<uint32_t>(status), patch.header.errorDetail);
            return kStatusInvalid;
        }
        return kStatusOk;
    }
};
const char *kFlashMlaMetadata = "FlashMlaMetadata";
REGISTER_CPU_KERNEL(kFlashMlaMetadata, FlashMlaMetadataAicpu);
} // namespace aicpu
