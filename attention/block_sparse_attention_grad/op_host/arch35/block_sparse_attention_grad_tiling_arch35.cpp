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
    explicit BlockSparseAttentionGradArch35Tiling(gert::TilingContext *context) : TilingBaseClass(context) {}

protected:
    bool IsCapable() override
    {
        std::cout<<"arc 35 is capable"<<std::endl;
        return true;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        OP_CHECK_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR("BlockSparseAttentionGrad",
            "Context is nullptr."), return ge::GRAPH_FAILED);
        if (tiling_.GetBSAGradTiling(context_, tilingData_) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "GetBSAGradTiling failed");
            return ge::GRAPH_FAILED;
        }
        tilingKey_ = tilingData_.get_tilingKey();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        return tiling_.SetTilingData(context_, tilingData_);
    }

    uint64_t GetTilingKey() const override
    {
        printf("tilingKey_ is %lu\n", tilingKey_);
        return tilingKey_;
    }

private:
    BSAGradTiling tiling_;
    BlockSparseAttentionGradTilingData tilingData_;
    uint64_t tilingKey_ = 0;
};


REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockSparseAttentionGrad,
    BlockSparseAttentionGradArch35Tiling,
    static_cast<int32_t>(NpuArch::DAV_3510),
    1);

}  // namespace optiling
