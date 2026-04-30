/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 */

#include "../block_sparse_attention_grad_tiling.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
class BlockSparseAttentionGradArch35Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockSparseAttentionGradArch35Tiling(gert::TilingContext *context_) : TilingBaseClass(context_)
    {
    }

private:
    uint64_t tilingKey_ = 1000; // arc35 默认tilingkey从1000开始
    BlockSparseAttentionGradTilingDataArch35 tilingData_;
    static constexpr const char *BSH_STR = "BSH";
    static constexpr const char *SBH_STR = "SBH";
    static constexpr const char *BNSD_STR = "BNSD";
    static constexpr const char *BSND_STR = "BSND";
    static constexpr const char *TND_STR = "TND";

protected:
    bool IsCapable() override
    {
        std::cout << "arc 35 is capable" << std::endl;
        return true;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        int32_t batch_num, q_group, q_head_num, kv_head_num, q_seq_len, kv_seq_len, head_dim;
        const gert::StorageShape *queryShape = context_->GetInputShape(1);
        const gert::StorageShape *keyShape = context_->GetInputShape(2);
        const gert::StorageShape *valueShape = context_->GetInputShape(3);
        const char *qInputLayout = context_->GetAttrs()->GetAttrPointer<char>(0);
        const char *kvInputLayout = context_->GetAttrs()->GetAttrPointer<char>(1);
        kv_head_num = *context_->GetAttrs()->GetAttrPointer<int64_t>(2);
        const float scaleValue = *context_->GetAttrs()->GetAttrPointer<float>(4);

        if (strcmp(qInputLayout, kvInputLayout) != 0) {
            OP_LOGE(context_->GetNodeName(), "kvInputLayout must same as qInputLayout.");
            return ge::GRAPH_FAILED;
        }

        if (strcmp(qInputLayout, BSH_STR) == 0) {
            if (queryShape->GetOriginShape().GetDimNum() != 3) {
                OP_LOGE(context_->GetNodeName(), "The shape of query must be 3 dimension.");
                return ge::GRAPH_FAILED;
            }
            batch_num = queryShape->GetStorageShape().GetDim(0);
            q_seq_len = queryShape->GetStorageShape().GetDim(1);
            int32_t q_hidden_state = queryShape->GetStorageShape().GetDim(2);
            int32_t kv_hidden_state = keyShape->GetStorageShape().GetDim(2);
            head_dim = kv_hidden_state / kv_head_num;
            q_head_num = q_hidden_state / head_dim;
            q_group = q_head_num / kv_head_num;

            if (q_group * kv_head_num != q_head_num) {
                OP_LOGE(context_->GetNodeName(), "q_group * kv_head_num must same as q_head_num.");
                return ge::GRAPH_FAILED;
            }
        } else if (strcmp(qInputLayout, BNSD_STR) == 0) {
            if (queryShape->GetOriginShape().GetDimNum() != 4) {
                OP_LOGE(context_->GetNodeName(), "The shape of query must be 4 dimension.");
                return ge::GRAPH_FAILED;
            }
            batch_num = queryShape->GetStorageShape().GetDim(0);
            q_head_num = queryShape->GetStorageShape().GetDim(1);
            q_seq_len = queryShape->GetStorageShape().GetDim(2);
            head_dim = queryShape->GetStorageShape().GetDim(3);
            kv_head_num = keyShape->GetStorageShape().GetDim(1);
            kv_seq_len = keyShape->GetStorageShape().GetDim(2);
            q_group = q_head_num / kv_head_num;
        } else if (strcmp(qInputLayout, TND_STR) == 0) {
            if (queryShape->GetOriginShape().GetDimNum() != 3) {
                OP_LOGE(context_->GetNodeName(), "The shape of query must be 3 dimension.");
                return ge::GRAPH_FAILED;
            }
            auto actualSeqLengths = context_->GetOptionalInputTensor(9);
            auto actualKVSeqLengths = context_->GetOptionalInputTensor(10);
            if (actualSeqLengths == nullptr || actualKVSeqLengths == nullptr) {
                OP_LOGE(context_->GetNodeName(), "TND format must have is actualSeqLengthsOptional");
                return ge::GRAPH_FAILED;
            }
            if (actualSeqLengths->GetShapeSize() != actualKVSeqLengths->GetShapeSize()) {
                OP_LOGE(context_->GetNodeName(), "actualSeqLengths and actualKVSeqLengths must have same shape.");
                return ge::GRAPH_FAILED;
            }

            batch_num = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
            q_seq_len = queryShape->GetStorageShape().GetDim(0);
            q_head_num = queryShape->GetStorageShape().GetDim(1);
            head_dim = queryShape->GetStorageShape().GetDim(2);
            kv_seq_len = keyShape->GetStorageShape().GetDim(0);
            kv_head_num = keyShape->GetStorageShape().GetDim(1);
            q_group = q_head_num / kv_head_num;
        } else {
            OP_LOGE(context_->GetNodeName(), "Invaild Input Layout: InputLayout must be BSH or BNSD or TND.");
            return ge::GRAPH_FAILED;
        }

        tilingData_.set_batchNum(batch_num);
        tilingData_.set_qSeqLen(q_seq_len);
        tilingData_.set_kvSeqLen(kv_seq_len);
        tilingData_.set_qGroup(q_group);
        tilingData_.set_qHeadNum(q_head_num);
        tilingData_.set_kvHeadNum(kv_head_num);
        tilingData_.set_headDim(head_dim);
        tilingData_.set_scaleValue(scaleValue);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        context_->SetBlockDim(1);
        context_->SetScheduleMode(1);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        context_->GetWorkspaceSizes(1)[0] = 16 * 1024 * 1024;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        printf("tilingKey_ is %lu\n", tilingKey_);
        return tilingKey_;
    }
};


REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockSparseAttentionGrad, BlockSparseAttentionGradArch35Tiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

} // namespace optiling
