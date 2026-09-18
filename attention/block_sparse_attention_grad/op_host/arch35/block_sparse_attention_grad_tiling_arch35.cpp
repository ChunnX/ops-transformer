/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "err/ops_err.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../block_sparse_attention_grad_tiling.h"
#include <algorithm>
#include "non_zero_tiling_arch35.h"

namespace optiling {
namespace BSA_ARC35 {

int64_t AlignTo(int64_t x, int64_t align)
{
    return (x + align - 1) / align * align;
}

int64_t CeilDiv(int64_t x, int64_t align)
{
    return AlignTo(x, align) / align;
}

template <typename... Args>
bool IsSameShape(Args... shapes)
{
    std::array<int32_t, sizeof...(Args)> arr = {shapes...};
    if (arr.size() <= 1)
        return true;
    int32_t first = arr[0];
    if (first <= 0) {
        return false;
    }

    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] != first)
            return false;
    }
    return true;
}

ge::graphStatus InputCheck(const std::vector<int32_t> &q_shape, const std::vector<int32_t> &k_shape,
                           const std::vector<int32_t> &v_shape, const int32_t q_group, gert::TilingContext *context)
{
    /* function: 根据固定格式传入shape检查输入是否符合要求
     * shape：[b, n, s, d]
     */
    if (!IsSameShape(q_shape[0], k_shape[0], v_shape[0])) {
        OP_LOGE(context->GetNodeName(), "q_batch must be same as k_batch and v_batch.");
        return ge::GRAPH_FAILED;
    } else if (!IsSameShape(q_shape[1], k_shape[1] * q_group, v_shape[1] * q_group)) {
        OP_LOGE(context->GetNodeName(), "q_group * kv_head_num must same as q_head_num.");
        return ge::GRAPH_FAILED;
    } else if (!IsSameShape(q_shape[2], k_shape[2], v_shape[2])) {
        OP_LOGE(context->GetNodeName(), "q_seq_len must be same as k_seq_len and v_seq_len.");
        return ge::GRAPH_FAILED;
    } else if (!IsSameShape(q_shape[3], k_shape[3], v_shape[3])) {
        OP_LOGE(context->GetNodeName(), "q_head_dim must be same as k_head_dim and v_head_dim.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

class BlockSparseAttentionGradArch35Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockSparseAttentionGradArch35Tiling(gert::TilingContext *context_)
        : TilingBaseClass(context_)
    {}

protected:
    bool IsCapable() override
    {
        if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16) {
            OP_LOGE(context_->GetNodeName(), "BlockSparseAttentionGrad only support DT_FLOAT16 and DT_BF16.");
            return false;
        }

        if (head_dim_ != 128) {
            OP_LOGE(context_->GetNodeName(), "BlockSparseAttentionGrad only support head_dim 128.");
            return false;
        }

        if (blockShapeX_ == 0 || blockShapeY_ == 0 || blockShapeX_ % 16 != 0 || blockShapeY_ % 16 != 0) {
            OP_LOGE(context_->GetNodeName(),
                    "BlockSparseAttentionGrad only supports blockShapeX and blockShapeY that are multiples of 16.");
            return false;
        }

        return true;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        deterministic_ = (context_->GetDeterministic() == 1);
        int32_t q_batch_num, q_group, q_head_num, q_seq_len, q_head_dim;
        int32_t k_batch_num, k_head_num, k_seq_len, k_head_dim;
        int32_t v_batch_num, v_head_num, v_seq_len, v_head_dim;
        auto qInputDesc = context_->GetInputDesc(1);
        const gert::StorageShape *queryShape = context_->GetInputShape(1);
        const gert::StorageShape *keyShape = context_->GetInputShape(2);
        const gert::StorageShape *valueShape = context_->GetInputShape(3);
        const char *qInputLayout = context_->GetAttrs()->GetAttrPointer<char>(0);
        const char *kvInputLayout = context_->GetAttrs()->GetAttrPointer<char>(1);
        const float softmaxScale = *context_->GetAttrs()->GetAttrPointer<float>(4);
        auto blockShapeOptional = context_->GetInputTensor(8);
        if (blockShapeOptional == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Block shape tensor is null");
            return ge::GRAPH_FAILED;
        }
        const int64_t *blockShapeList = blockShapeOptional->GetData<int64_t>();
        if (blockShapeList == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Block shape tensor is null");
            return ge::GRAPH_FAILED;
        }

        if (qInputDesc == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Query inputDesc is null");
            return ge::GRAPH_FAILED;
        } else {
            dataType_ = qInputDesc->GetDataType();
        }

        if (strcmp(qInputLayout, kvInputLayout) != 0) {
            OP_LOGE(context_->GetNodeName(), "kvInputLayout must same as qInputLayout.");
            return ge::GRAPH_FAILED;
        }

        if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
            OP_LOGE(context_->GetNodeName(), "queryShape or keyShape or valueShape is nullptr.");
            return ge::GRAPH_FAILED;
        }

        if (strcmp(qInputLayout, BSND_STR) == 0) {
            // BSND layout
            if (queryShape->GetOriginShape().GetDimNum() != 4) {
                OP_LOGE(context_->GetNodeName(), "The shape of query must be 4 dimension.");
                return ge::GRAPH_FAILED;
            }
            // query
            q_batch_num = queryShape->GetStorageShape().GetDim(0);
            q_seq_len = queryShape->GetStorageShape().GetDim(1);
            q_head_num = queryShape->GetStorageShape().GetDim(2);
            q_head_dim = queryShape->GetStorageShape().GetDim(3);
            // key
            k_batch_num = keyShape->GetStorageShape().GetDim(0);
            k_seq_len = keyShape->GetStorageShape().GetDim(1);
            k_head_num = keyShape->GetStorageShape().GetDim(2);
            k_head_dim = keyShape->GetStorageShape().GetDim(3);
            // value
            v_batch_num = valueShape->GetStorageShape().GetDim(0);
            v_seq_len = valueShape->GetStorageShape().GetDim(1);
            v_head_num = valueShape->GetStorageShape().GetDim(2);
            v_head_dim = valueShape->GetStorageShape().GetDim(3);
            q_group = q_head_num / k_head_num;
        } else if (strcmp(qInputLayout, BNSD_STR) == 0) {
            // BNSD layout
            if (queryShape->GetOriginShape().GetDimNum() != 4) {
                OP_LOGE(context_->GetNodeName(), "The shape of query must be 4 dimension.");
                return ge::GRAPH_FAILED;
            }
            // query
            q_batch_num = queryShape->GetStorageShape().GetDim(0);
            q_head_num = queryShape->GetStorageShape().GetDim(1);
            q_seq_len = queryShape->GetStorageShape().GetDim(2);
            q_head_dim = queryShape->GetStorageShape().GetDim(3);
            // key
            k_batch_num = keyShape->GetStorageShape().GetDim(0);
            k_head_num = keyShape->GetStorageShape().GetDim(1);
            k_seq_len = keyShape->GetStorageShape().GetDim(2);
            k_head_dim = keyShape->GetStorageShape().GetDim(3);
            // value
            v_batch_num = valueShape->GetStorageShape().GetDim(0);
            v_head_num = valueShape->GetStorageShape().GetDim(1);
            v_seq_len = valueShape->GetStorageShape().GetDim(2);
            v_head_dim = valueShape->GetStorageShape().GetDim(3);
            q_group = q_head_num / k_head_num;
        } else if (strcmp(qInputLayout, TND_STR) == 0) {
            // TND layout
            if (queryShape->GetOriginShape().GetDimNum() != 3) {
                OP_LOGE(context_->GetNodeName(), "The shape of query must be 3 dimension.");
                return ge::GRAPH_FAILED;
            }
            auto actualSeqLengths = context_->GetOptionalInputTensor(9);
            auto actualKVSeqLengths = context_->GetOptionalInputTensor(10);
            if (actualSeqLengths == nullptr || actualKVSeqLengths == nullptr) {
                OP_LOGE(context_->GetNodeName(),
                        "The layout is TND, actualSeqLengths or actualKVSeqLengths must not be nullptr.");
                return ge::GRAPH_FAILED;
            }
            q_batch_num = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
            k_batch_num = static_cast<uint32_t>(actualKVSeqLengths->GetShapeSize());
            v_batch_num = k_batch_num;
            // query
            q_seq_len = queryShape->GetStorageShape().GetDim(0);
            q_head_num = queryShape->GetStorageShape().GetDim(1);
            q_head_dim = queryShape->GetStorageShape().GetDim(2);
            // key
            k_seq_len = keyShape->GetStorageShape().GetDim(0);
            k_head_num = keyShape->GetStorageShape().GetDim(1);
            k_head_dim = keyShape->GetStorageShape().GetDim(2);
            // value
            v_seq_len = valueShape->GetStorageShape().GetDim(0);
            v_head_num = valueShape->GetStorageShape().GetDim(1);
            v_head_dim = valueShape->GetStorageShape().GetDim(2);
            q_group = q_head_num / k_head_num;
        } else {
            OP_LOGE(context_->GetNodeName(), "Invaild Input Layout: InputLayout must be BSND or BNSD or TND.");
            return ge::GRAPH_FAILED;
        }

        tilingData_.set_batchNum(q_batch_num);
        tilingData_.set_qSeqLen(q_seq_len);  // TND时含义为total_seq_len
        tilingData_.set_kvSeqLen(k_seq_len); // TND时含义为total_seq_len
        tilingData_.set_qGroup(q_group);
        tilingData_.set_qHeadNum(q_head_num);
        tilingData_.set_kvHeadNum(k_head_num);
        tilingData_.set_headDim(q_head_dim);
        tilingData_.set_softmaxScale(softmaxScale);
        batch_num_ = q_batch_num;
        q_seq_len_ = q_seq_len;
        kv_seq_len_ = k_seq_len;
        q_head_num_ = q_head_num;
        q_group_ = q_group;
        kv_head_num_ = k_head_num;
        head_dim_ = q_head_dim;
        layout_ = qInputLayout;
        blockShapeX_ = blockShapeList[0];
        blockShapeY_ = blockShapeList[1];

        return InputCheck({q_batch_num, q_head_num, q_seq_len, q_head_dim},
                          {k_batch_num, k_head_num, k_seq_len, k_head_dim},
                          {v_batch_num, v_head_num, v_seq_len, v_head_dim}, q_group, context_);
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        cubeCoreNum_ = ascendcPlatform.GetCoreNumAic();

        baseM_ = blockShapeX_ <= 128 ? blockShapeX_ : 128;
        baseN_ = blockShapeY_ <= 128 ? blockShapeY_ : 128;
        x_block_num_ = CeilDiv(q_seq_len_, blockShapeX_);
        y_block_num_ = CeilDiv(kv_seq_len_, blockShapeY_);

        // 确定性与非确定性路径统一使用 host 预生成的 block index 遍历（NonZero 路径），
        // 因此这里总是需要 NonZero 预 kernel 的 tiling 与 workspace。
        uint64_t vecCoreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t vecRegLen = ascendcPlatform.GetVecRegLen();
        uint64_t ubSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        NonZeroAscendCTilingImpl tilingImpl = NonZeroAscendCTilingImpl(context_);

        if (tilingImpl.DoTiling(vecCoreNum, ubSize, vecRegLen, tilingData_.mNonZeroTilingData) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "Tiling4NonZero do tiling failed, unsupported shape.");
            return ge::GRAPH_FAILED;
        }
        needNonZeroWorkspace_ = true;

        tilingData_.set_BlockX(blockShapeX_);
        tilingData_.set_BlockY(blockShapeY_);
        tilingData_.set_cubeCoreNum(cubeCoreNum_);
        tilingData_.set_baseM(baseM_);
        tilingData_.set_baseN(baseN_);
        context_->SetBlockDim(cubeCoreNum_);
        context_->SetScheduleMode(1);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        // softmaxGrad tiling，一次最多处理16K个元素个数
        constexpr uint32_t element_num = 8 * 1024;
        uint32_t process_s1_size = element_num / head_dim_;
        auto input_shape = ge::Shape({process_s1_size, head_dim_});
        uint32_t tmp_space_size = AscendC::GetSoftMaxGradMaxTmpSize(input_shape, sizeof(float), true, true);
        AscendC::SoftMaxGradTilingFunc(input_shape, sizeof(float), tmp_space_size,
                                       tilingData_.softmaxGradFrontTilingData, true);
        tilingData_.set_sftgTmpSpaceSize(tmp_space_size); // 元素个数=8K时需要50K的临时空间
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        uint64_t dq_size, dk_size, dv_size, sftg_size;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t sys_workspace_size = ascendcPlatform.GetLibApiWorkSpaceSize();
        uint64_t usr_workspace_offset = 0;
        if (strcmp(layout_, BSND_STR) == 0 || strcmp(layout_, BNSD_STR) == 0) {
            dq_size = batch_num_ * q_seq_len_ * q_head_num_ * head_dim_;
            dk_size = batch_num_ * kv_seq_len_ * kv_head_num_ * head_dim_;
            dv_size = dk_size;
            sftg_size = batch_num_ * q_head_num_ * q_seq_len_ * 8;
        } else if (strcmp(layout_, TND_STR) == 0) {
            dq_size = q_seq_len_ * q_head_num_ * head_dim_;
            dk_size = kv_seq_len_ * kv_head_num_ * head_dim_;
            dv_size = dk_size;
            sftg_size = q_seq_len_ * q_head_num_ * 8; // 存放逻辑按[B,N,S,8]
        } else {
            OP_LOGE(context_->GetNodeName(), "Invaild Input Layout: InputLayout must be BSDN or BNSD or TND.");
            return ge::GRAPH_FAILED;
        }

        tilingData_.set_dqSize(dq_size);
        tilingData_.set_dkSize(dk_size);
        // 确定性路径核间分工（必须与 kernel 侧 bsa_addr_compute.h 的 unit/子段划分保持一致）：
        //   G = batchNum * kvHeadNum；组内 kv_block 是一个连续段。
        //   k = dqSplitNum = max(1, ceil(cubeCoreNum / G))：把每组的 kv_block 均分为 k 个子段，
        //   子段 p 覆盖 kv_block ∈ [p*KV/k, (p+1)*KV/k)。cube 核 c 负责所有 unit = (组 g, 子段 p)
        //   中满足 (g*k + p) % cubeCoreNum == c 的单位。
        //   同一 dq 槽位 (b, n1, s1) 的各 kv_block 落在不同子段上、每段只有一个拥有者，
        //   因此每段 buffer 单写者、累加顺序 = 程序顺序，vector 端按段号固定升序归约 => bitwise 可复现；
        //   k == 1（头数足够多）时所有核写同一份 dq buffer 的不重叠槽位，无需分段归约，性能与
        //   非确定性路径一致。
        //   s1PhaseNum：再把 q_block 空间均分为若干相位，外层按相位推进遍历，使同一核重复访问同一段
        //   dq buffer 的距离降到 1/s1PhaseNum。否则 k 份 dq 分段（k × dqSize）整体远超 L2，
        //   fp32 原子加退化为 HBM 读改写，实测会把 cube MTE2 从 416us 拖到 1015us。
        uint32_t dq_split_num = 1;
        uint32_t s1_phase_num = 1;
        if (deterministic_) {
            uint64_t group_num =
                static_cast<uint64_t>(batch_num_) * static_cast<uint64_t>(std::max<int32_t>(kv_head_num_, 1));
            if (group_num == 0) {
                group_num = 1;
            }
            uint64_t split_num =
                static_cast<uint64_t>(CeilDiv(static_cast<int64_t>(cubeCoreNum_), static_cast<int64_t>(group_num)));
            if (dq_size != 0) {
                // 多占用 (k-1) 份 dq buffer，按 workspace 预算设上限（至少 1 段）
                uint64_t max_split = 1 + DQ_SPLIT_EXTRA_WS_LIMIT / (dq_size * sizeof(float));
                split_num = std::min(split_num, max_split);
            }
            dq_split_num = static_cast<uint32_t>(std::max<uint64_t>(split_num, 1));
            if (dq_split_num > 1 && dq_size != 0) {
                uint64_t hot = static_cast<uint64_t>(dq_split_num) * dq_size * sizeof(float);
                uint64_t phase = CeilDiv(static_cast<int64_t>(hot), static_cast<int64_t>(DQ_HOT_L2_BUDGET));
                uint64_t q_block_upper = static_cast<uint64_t>(
                    CeilDiv(static_cast<int64_t>(q_seq_len_), static_cast<int64_t>(blockShapeX_)));
                if (q_block_upper > 0 && phase > q_block_upper) {
                    phase = q_block_upper;
                }
                phase = std::min(phase, static_cast<uint64_t>(MAX_S1_PHASE_NUM));
                s1_phase_num = static_cast<uint32_t>(std::max<uint64_t>(phase, 1));
            }
        }
        tilingData_.set_dqSplitNum(dq_split_num);
        tilingData_.set_s1PhaseNum(s1_phase_num);
        if (needNonZeroWorkspace_) {
            // index workspace：存放 NonZero 预 kernel 输出的有效 block 下标（数量上界 = mask 元素个数）
            uint64_t block_sparse_mask_size =
                static_cast<uint64_t>(batch_num_) * q_head_num_ * x_block_num_ * y_block_num_;
            auto maskShape = context_->GetInputShape(6);
            if (maskShape != nullptr) {
                const gert::Shape &shape = maskShape->GetStorageShape();
                uint64_t maskElemNum = 1;
                for (size_t i = 0; i < shape.GetDimNum(); i++) {
                    maskElemNum *= static_cast<uint64_t>(shape.GetDim(i));
                }
                block_sparse_mask_size = maskElemNum;
            }
            tilingData_.set_indexWorkspaceOffset(usr_workspace_offset);
            // 按 512B 对齐：index shape workspace 会被主 kernel 以 uint64 标量读取（必须 8B 对齐），
            // 而 mask 元素个数很少时（例如整个 mask 只有 1 个 block）本段只有 4B，不补齐会触发标量非对齐异常。
            usr_workspace_offset += AlignTo(static_cast<int64_t>(block_sparse_mask_size) * sizeof(int32_t), 512);
            // index shape workspace：NonZero 输出的 [dim, dims, count]
            tilingData_.set_indexShapeWorkspaceOffset(usr_workspace_offset);
            usr_workspace_offset += NON_ZERO_SHAPE_WORKSPACE_SIZE;
            // NonZero 预 kernel 的分核计数 workspace：每个 AIV 核 128B
            uint64_t vecCoreNum = ascendcPlatform.GetCoreNumAiv();
            tilingData_.set_nonZeroWorkspaceOffset(usr_workspace_offset);
            usr_workspace_offset += AlignTo(static_cast<int64_t>(vecCoreNum) * NON_ZERO_COUNT_BYTES_PER_CORE, 512);
        }
        tilingData_.set_sftgWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += sftg_size * sizeof(float);
        tilingData_.set_dqWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += dq_size * dq_split_num * sizeof(float);
        tilingData_.set_dkWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += dk_size * sizeof(float);
        tilingData_.set_dvWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += dv_size * sizeof(float);

        context_->GetWorkspaceSizes(1)[0] = sys_workspace_size + usr_workspace_offset;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        uint64_t default_key = 1000;
        /*
         *  0000: BF16 BSND
         *  0001: FP16 BSND
         *  0010: BF16 BNSD
         *  0011: FP16 BNSD
         *  0100: BF16 TND
         *  0101: FP16 TND
         *  1000: BF16 BSND DET
         *  1001: FP16 BSND DET
         *  1010: BF16 BNSD DET
         *  1011: FP16 BNSD DET
         *  1100: BF16 TND  DET
         *  1101: FP16 TND  DET
         *
         *  说明：非确定性 key（0000~0101）与确定性 key（1000~1101）都在 host 侧预生成
         *  block index、kernel 侧按 index 遍历有效块（NonZero 路径），不再读 blockSparseMask 判块。
         */
        default_key = (dataType_ == ge::DT_BF16) ? default_key : default_key + 0b001;

        if (strcmp(layout_, BSND_STR) == 0) {
            default_key += 0b000;
        } else if (strcmp(layout_, BNSD_STR) == 0) {
            default_key += 0b010;
        } else if (strcmp(layout_, TND_STR) == 0) {
            default_key += 0b100;
        }

        if (deterministic_) {
            default_key += 0b1000;
        }

        return default_key;
    }

private:
    // 是否需要 NonZero 相关的 workspace（index / index shape / 分核计数）
    bool needNonZeroWorkspace_ = false;

    static constexpr uint64_t NON_ZERO_SHAPE_WORKSPACE_SIZE = 512; // NonZero 输出 shape 占用 24B，留 512B
    static constexpr uint64_t NON_ZERO_COUNT_BYTES_PER_CORE = 128; // 每个 AIV 在计数 workspace 占 128B
    // 确定性路径 dq 分段归约允许的额外 workspace 上限（仅 (dqSplitNum-1) 份 dq buffer 计入）
    static constexpr uint64_t DQ_SPLIT_EXTRA_WS_LIMIT = 1UL << 30; // 1GB
    // 确定性路径 dq 分段热工作集预算：超出该预算即用 s1 相位切分降低重复访问距离（见 GetWorkspaceSize）
    static constexpr uint64_t DQ_HOT_L2_BUDGET = 32UL * 1024 * 1024; // 32MB
    // s1 相位上限：相位过多会让 dk/dv 的 L0C->GM 落盘次数线性增长，实测 4 为最优
    static constexpr uint64_t MAX_S1_PHASE_NUM = 4;

    const char *layout_;
    ge::DataType dataType_;
    uint64_t tilingKey_ = 1000; // arc35 默认tilingkey从1000开始
    BlockSparseAttentionGradTilingDataArch35 tilingData_;
    static constexpr const char *BSND_STR = "BSND";
    static constexpr const char *BNSD_STR = "BNSD";
    static constexpr const char *TND_STR = "TND";
    int32_t batch_num_ = 0;
    int32_t q_seq_len_ = 0;
    int32_t kv_seq_len_ = 0;
    int32_t q_head_num_ = 0;
    int32_t q_group_ = 0;
    int32_t kv_head_num_ = 0;
    int32_t head_dim_ = 0;
    int32_t blockShapeX_ = 0;
    int32_t blockShapeY_ = 0;
    int32_t x_block_num_ = 0;
    int32_t y_block_num_ = 0;
    int32_t max_q_seq_len_ = 0;
    int32_t max_kv_seq_len_ = 0;
    bool deterministic_ = false;
    uint32_t cubeCoreNum_ = 0;
    uint32_t baseM_ = 0;
    uint32_t baseN_ = 0;
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockSparseAttentionGrad, BlockSparseAttentionGradArch35Tiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace BSA_ARC35
} // namespace optiling
