/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "err/ops_err.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../block_sparse_attention_grad_tiling.h"
#include <cmath>

namespace optiling {
namespace BSA_ARC35 {

constexpr int32_t MAX_BATCH_SIZE = 20000;
constexpr int32_t MAX_TOKEN_COUNT = 1024 * 1024;
constexpr int32_t MAX_HEAD_NUM = 256;
constexpr int32_t SUPPORTED_HEAD_DIM = 128;
constexpr int32_t MIN_BLOCK_SHAPE = 16;

constexpr int DOUT_INDEX = 0;
constexpr int KEY_INDEX = 2;
constexpr int VALUE_INDEX = 3;
constexpr int OUT_INDEX = 4;
constexpr int SOFTMAX_LSE_INDEX = 5;
constexpr int BLOCK_SPARSE_MASK_INDEX = 6;

constexpr int Q_INPUT_LAYOUT_INDEX = 0;
constexpr int KV_INPUT_LAYOUT_INDEX = 1;
constexpr int NUM_KEY_VALUE_HEADS_INDEX = 2;
constexpr int MASK_TYPE_INDEX = 3;
constexpr int SCALE_VALUE_INDEX = 4;
constexpr int PRE_TOKENS_INDEX = 5;
constexpr int NEXT_TOKENS_INDEX = 6;

constexpr int TND_LSE_DIM_T = 0;
constexpr int TND_LSE_DIM_N = 1;
constexpr int TND_LSE_DIM_LAST = 2;
constexpr int LSE_DIM_B = 0;
constexpr int LSE_DIM_N = 1;
constexpr int LSE_DIM_S = 2;
constexpr int LSE_DIM_LAST = 3;

constexpr int DENSE_DIM_NUM = 4;
constexpr int TND_DIM_NUM = 3;
constexpr int BLOCK_SPARSE_MASK_DIM_NUM = 4;
constexpr int BLOCK_SHAPE_ELEMENT_NUM = 2;
constexpr int BLOCK_SHAPE_X_INDEX = 0;
constexpr int BLOCK_SHAPE_Y_INDEX = 1;
constexpr int LSE_LAST_DIM_SIZE = 1;
constexpr int64_t SUPPORTED_MASK_TYPE = 0;
constexpr int64_t FULL_ATTENTION_TOKEN_VALUE = 2147483647;

int64_t AlignTo(int64_t x, int64_t align)
{
    return (x + align - 1) / align * align;
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
                           const std::vector<int32_t> &v_shape, const bool isTnd, gert::TilingContext *context)
{
    /* function: 根据固定格式传入shape检查输入是否符合要求
     * shape：[b, n, s, d]
     */
    if (q_shape[0] <= 0 || q_shape[0] > MAX_BATCH_SIZE) {
        OP_LOGE(context->GetNodeName(), "batch size must be in [1, 20000], but got %d.", q_shape[0]);
        return ge::GRAPH_FAILED;
    } else if (!IsSameShape(q_shape[0], k_shape[0], v_shape[0])) {
        OP_LOGE(context->GetNodeName(), "q_batch must be same as k_batch and v_batch.");
        return ge::GRAPH_FAILED;
    } else if (q_shape[1] <= 0 || q_shape[1] > MAX_HEAD_NUM || k_shape[1] <= 0 ||
               q_shape[1] % k_shape[1] != 0 || k_shape[1] != v_shape[1]) {
        OP_LOGE(context->GetNodeName(),
                "NQ must be in [1, 256], NKV must be positive, key/value heads must match, and NQ must be divisible by NKV.");
        return ge::GRAPH_FAILED;
    } else if (q_shape[2] <= 0 || k_shape[2] <= 0 || k_shape[2] != v_shape[2] ||
               (isTnd && (q_shape[2] > MAX_TOKEN_COUNT || k_shape[2] > MAX_TOKEN_COUNT)) ||
               (!isTnd && (q_shape[2] > MAX_TOKEN_COUNT / q_shape[0] ||
                           k_shape[2] > MAX_TOKEN_COUNT / q_shape[0]))) {
        OP_LOGE(context->GetNodeName(),
                "Q/KV sequence lengths must be positive, K/V lengths must match, and total Q/KV tokens must not exceed 512.");
        return ge::GRAPH_FAILED;
    } else if (!IsSameShape(q_shape[3], k_shape[3], v_shape[3], SUPPORTED_HEAD_DIM)) {
        OP_LOGE(context->GetNodeName(), "q/k/v head dimension must all be 128.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

class BlockSparseAttentionGradArch35Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockSparseAttentionGradArch35Tiling(gert::TilingContext *context_) : TilingBaseClass(context_)
    {
    }


protected:
    bool IsCapable() override
    {
        if (context_->GetDeterministic() == 1) {
            OP_LOGE(context_->GetNodeName(), "BlockSparseAttentionGrad not support Deterministic.");
            return false;
        }

        if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16) {
            OP_LOGE(context_->GetNodeName(), "BlockSparseAttentionGrad only support DT_FLOAT16 and DT_BF16.");
            return false;
        }

        if (head_dim_ != SUPPORTED_HEAD_DIM) {
            OP_LOGE(context_->GetNodeName(), "BlockSparseAttentionGrad only support head_dim 128.");
            return false;
        }

        if (blockShapeX_ < MIN_BLOCK_SHAPE || blockShapeY_ < MIN_BLOCK_SHAPE ||
            blockShapeX_ > MAX_TOKEN_COUNT || blockShapeY_ > MAX_TOKEN_COUNT ||
            blockShapeX_ % MIN_BLOCK_SHAPE != 0 || blockShapeY_ % MIN_BLOCK_SHAPE != 0) {
            OP_LOGE(context_->GetNodeName(),
                    "BlockSparseAttentionGrad requires blockShapeX/Y in [16, 512] and multiples of 16.");
            return false;
        }

        return true;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        int32_t q_batch_num, q_group, q_head_num, q_seq_len, q_head_dim;
        int32_t k_batch_num, k_head_num, k_seq_len, k_head_dim;
        int32_t v_batch_num, v_head_num, v_seq_len, v_head_dim;
        auto attrs = context_->GetAttrs();
        if (attrs == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Attrs is nullptr.");
            return ge::GRAPH_FAILED;
        }
        auto qInputDesc = context_->GetInputDesc(1);
        auto kInputDesc = context_->GetInputDesc(KEY_INDEX);
        auto vInputDesc = context_->GetInputDesc(VALUE_INDEX);
        auto doutInputDesc = context_->GetInputDesc(DOUT_INDEX);
        auto outInputDesc = context_->GetInputDesc(OUT_INDEX);
        auto lseInputDesc = context_->GetInputDesc(SOFTMAX_LSE_INDEX);
        const gert::StorageShape *queryShape = context_->GetInputShape(1);
        const gert::StorageShape *keyShape = context_->GetInputShape(2);
        const gert::StorageShape *valueShape = context_->GetInputShape(3);
        const gert::StorageShape *doutShape = context_->GetInputShape(DOUT_INDEX);
        const gert::StorageShape *attentionOutShape = context_->GetInputShape(OUT_INDEX);
        const gert::StorageShape *softmaxLseShape = context_->GetInputShape(SOFTMAX_LSE_INDEX);
        const gert::StorageShape *blockSparseMaskShape = context_->GetInputShape(BLOCK_SPARSE_MASK_INDEX);
        const char *qInputLayout = attrs->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX);
        const char *kvInputLayout = attrs->GetAttrPointer<char>(KV_INPUT_LAYOUT_INDEX);
        const int64_t *numKeyValueHeads = attrs->GetAttrPointer<int64_t>(NUM_KEY_VALUE_HEADS_INDEX);
        const int64_t *maskType = attrs->GetAttrPointer<int64_t>(MASK_TYPE_INDEX);
        const float *softmaxScalePtr = attrs->GetAttrPointer<float>(SCALE_VALUE_INDEX);
        const int64_t *preTokens = attrs->GetAttrPointer<int64_t>(PRE_TOKENS_INDEX);
        const int64_t *nextTokens = attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_INDEX);
        if (qInputLayout == nullptr || kvInputLayout == nullptr || numKeyValueHeads == nullptr || maskType == nullptr ||
            softmaxScalePtr == nullptr || preTokens == nullptr || nextTokens == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Required attrs must not be nullptr.");
            return ge::GRAPH_FAILED;
        }
        if (*maskType != SUPPORTED_MASK_TYPE || *preTokens != FULL_ATTENTION_TOKEN_VALUE ||
            *nextTokens != FULL_ATTENTION_TOKEN_VALUE ||
            !std::isfinite(*softmaxScalePtr)) {
            OP_LOGE(context_->GetNodeName(),
                    "maskType must be 0, preTokens/nextTokens must be 2147483647, and scaleValue must be finite.");
            return ge::GRAPH_FAILED;
        }
        const float softmaxScale = *softmaxScalePtr;
        auto blockShapeOptional = context_->GetInputTensor(8);
        if (blockShapeOptional == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Block shape tensor is null");
            return ge::GRAPH_FAILED;
        }
        if (blockShapeOptional->GetShapeSize() != BLOCK_SHAPE_ELEMENT_NUM) {
            OP_LOGE(context_->GetNodeName(), "Block shape must contain exactly two elements.");
            return ge::GRAPH_FAILED;
        }
        const int64_t *blockShapeList = blockShapeOptional->GetData<int64_t>();
        if (blockShapeList == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Block shape tensor is null");
            return ge::GRAPH_FAILED;
        }
        if (blockShapeList[BLOCK_SHAPE_X_INDEX] < MIN_BLOCK_SHAPE ||
            blockShapeList[BLOCK_SHAPE_Y_INDEX] < MIN_BLOCK_SHAPE ||
            blockShapeList[BLOCK_SHAPE_X_INDEX] > MAX_TOKEN_COUNT ||
            blockShapeList[BLOCK_SHAPE_Y_INDEX] > MAX_TOKEN_COUNT ||
            blockShapeList[BLOCK_SHAPE_X_INDEX] % MIN_BLOCK_SHAPE != 0 ||
            blockShapeList[BLOCK_SHAPE_Y_INDEX] % MIN_BLOCK_SHAPE != 0) {
            OP_LOGE(context_->GetNodeName(), "Block shape values must be in [16, 512] and multiples of 16.");
            return ge::GRAPH_FAILED;
        }

        if (qInputDesc == nullptr || kInputDesc == nullptr || vInputDesc == nullptr || doutInputDesc == nullptr ||
            outInputDesc == nullptr || lseInputDesc == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Required inputDesc is null");
            return ge::GRAPH_FAILED;
        } else {
            dataType_ = qInputDesc->GetDataType();
        }
        if ((dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16) ||
            kInputDesc->GetDataType() != dataType_ || vInputDesc->GetDataType() != dataType_ ||
            doutInputDesc->GetDataType() != dataType_ || outInputDesc->GetDataType() != dataType_ ||
            lseInputDesc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE(context_->GetNodeName(), "Q/K/V/dout/attentionOut dtypes must match and softmaxLse must be FLOAT32.");
            return ge::GRAPH_FAILED;
        }

        if (strcmp(qInputLayout, kvInputLayout) != 0) {
            OP_LOGE(context_->GetNodeName(), "kvInputLayout must same as qInputLayout.");
            return ge::GRAPH_FAILED;
        }

        if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr || doutShape == nullptr ||
            attentionOutShape == nullptr || softmaxLseShape == nullptr || blockSparseMaskShape == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Required input shape is nullptr.");
            return ge::GRAPH_FAILED;
        }
        if (blockSparseMaskShape->GetOriginShape().GetDimNum() != BLOCK_SPARSE_MASK_DIM_NUM) {
            OP_LOGE(context_->GetNodeName(), "blockSparseMask must be 4 dimensional.");
            return ge::GRAPH_FAILED;
        }

        if (strcmp(qInputLayout, BSND_STR) == 0) {
            // BSND layout
            if (queryShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                keyShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                valueShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                doutShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                attentionOutShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                softmaxLseShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM) {
                OP_LOGE(context_->GetNodeName(), "BSND tensors must be 4 dimensional.");
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
        } else if (strcmp(qInputLayout, BNSD_STR) == 0) {
            // BNSD layout
            if (queryShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                keyShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                valueShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                doutShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                attentionOutShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM ||
                softmaxLseShape->GetOriginShape().GetDimNum() != DENSE_DIM_NUM) {
                OP_LOGE(context_->GetNodeName(), "BNSD tensors must be 4 dimensional.");
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
        } else if (strcmp(qInputLayout, TND_STR) == 0) {
            // TND layout
            if (queryShape->GetOriginShape().GetDimNum() != TND_DIM_NUM ||
                keyShape->GetOriginShape().GetDimNum() != TND_DIM_NUM ||
                valueShape->GetOriginShape().GetDimNum() != TND_DIM_NUM ||
                doutShape->GetOriginShape().GetDimNum() != TND_DIM_NUM ||
                attentionOutShape->GetOriginShape().GetDimNum() != TND_DIM_NUM ||
                softmaxLseShape->GetOriginShape().GetDimNum() != TND_DIM_NUM) {
                OP_LOGE(context_->GetNodeName(), "TND tensors must be 3 dimensional.");
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
            if (q_batch_num != k_batch_num || q_batch_num <= 0 || q_batch_num > MAX_BATCH_SIZE) {
                OP_LOGE(context_->GetNodeName(), "TND actualSeqLengths Q/KV sizes must match and be in [1, 20000].");
                return ge::GRAPH_FAILED;
            }
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

        if (k_head_num <= 0 || q_head_num <= 0 || q_head_num % k_head_num != 0) {
            OP_LOGE(context_->GetNodeName(), "NKV must be positive and NQ must be divisible by NKV.");
            return ge::GRAPH_FAILED;
        }
        q_group = q_head_num / k_head_num;
        if (*numKeyValueHeads != k_head_num) {
            OP_LOGE(context_->GetNodeName(), "numKeyValueHeads must equal the key/value N dimension.");
            return ge::GRAPH_FAILED;
        }

        const auto &qStorage = queryShape->GetStorageShape();
        const auto &doutStorage = doutShape->GetStorageShape();
        const auto &outStorage = attentionOutShape->GetStorageShape();
        for (size_t i = 0; i < qStorage.GetDimNum(); ++i) {
            if (qStorage.GetDim(i) != doutStorage.GetDim(i) || qStorage.GetDim(i) != outStorage.GetDim(i)) {
                OP_LOGE(context_->GetNodeName(), "dout and attentionOut shapes must equal query shape.");
                return ge::GRAPH_FAILED;
            }
        }
        const auto &lseStorage = softmaxLseShape->GetStorageShape();
        if ((strcmp(qInputLayout, TND_STR) == 0 &&
             (lseStorage.GetDim(TND_LSE_DIM_T) != q_seq_len || lseStorage.GetDim(TND_LSE_DIM_N) != q_head_num ||
              lseStorage.GetDim(TND_LSE_DIM_LAST) != LSE_LAST_DIM_SIZE)) ||
            (strcmp(qInputLayout, TND_STR) != 0 &&
             (lseStorage.GetDim(LSE_DIM_B) != q_batch_num || lseStorage.GetDim(LSE_DIM_N) != q_head_num ||
              lseStorage.GetDim(LSE_DIM_S) != q_seq_len ||
              lseStorage.GetDim(LSE_DIM_LAST) != LSE_LAST_DIM_SIZE))) {
            OP_LOGE(context_->GetNodeName(), "softmaxLse must be [TQ,NQ,1] for TND or [B,NQ,SQ,1] otherwise.");
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
                          {v_batch_num, v_head_num, v_seq_len, v_head_dim}, strcmp(qInputLayout, TND_STR) == 0,
                          context_);
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        auto cubeCoreNum = ascendcPlatform.GetCoreNumAic();

        uint32_t baseM = blockShapeX_ <= 128 ? blockShapeX_ : 128;
        uint32_t baseN = blockShapeY_ <= 128 ? blockShapeY_ : 128;
        uint32_t singleM = AlignTo(2048, baseM);
        // Keep all Q tiles for the current KV block/head on one cube core when
        // possible so DK/DV can accumulate in L0 and K/V stays cached in L1.

        tilingData_.set_BlockX(blockShapeX_);
        tilingData_.set_BlockY(blockShapeY_);
        tilingData_.set_cubeCoreNum(cubeCoreNum);
        tilingData_.set_baseM(baseM);
        tilingData_.set_baseN(baseN);
        tilingData_.set_singleM(singleM);
        context_->SetBlockDim(cubeCoreNum);
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
            dq_size = static_cast<uint64_t>(batch_num_) * q_seq_len_ * q_head_num_ * head_dim_;
            dk_size = static_cast<uint64_t>(batch_num_) * kv_seq_len_ * kv_head_num_ * head_dim_;
            dv_size = dk_size;
            sftg_size = static_cast<uint64_t>(batch_num_) * q_head_num_ * q_seq_len_ * 8;
        } else if (strcmp(layout_, TND_STR) == 0) {
            dq_size = static_cast<uint64_t>(q_seq_len_) * q_head_num_ * head_dim_;
            dk_size = static_cast<uint64_t>(kv_seq_len_) * kv_head_num_ * head_dim_;
            dv_size = dk_size;
            sftg_size = static_cast<uint64_t>(q_seq_len_) * q_head_num_ * 8; // 存放逻辑按[B,N,S,8]
        } else {
            OP_LOGE(context_->GetNodeName(), "Invaild Input Layout: InputLayout must be BSDN or BNSD or TND.");
            return ge::GRAPH_FAILED;
        }

        tilingData_.set_dqSize(dq_size);
        tilingData_.set_dkSize(dk_size);
        tilingData_.set_sftgWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += sftg_size * sizeof(float);
        tilingData_.set_dqWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += dq_size * sizeof(float);
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
         *  000: BF16 BSND
         *  001: FP16 BSND
         *  010: BF16 BNSD
         *  011: FP16 BNSD
         *  100: BF16 TND
         *  101: FP16 TND
         */
        default_key = (dataType_ == ge::DT_BF16) ? default_key : default_key + 0b001;

        if (strcmp(layout_, BSND_STR) == 0) {
            default_key += 0b000;
        } else if (strcmp(layout_, BNSD_STR) == 0) {
            default_key += 0b010;
        } else if (strcmp(layout_, TND_STR) == 0) {
            default_key += 0b100;
        }

        return default_key;
    }

private:
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
    int32_t max_q_seq_len_ = 0;
    int32_t max_kv_seq_len_ = 0;
};


REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockSparseAttentionGrad, BlockSparseAttentionGradArch35Tiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace BSA_ARC35
} // namespace optiling
