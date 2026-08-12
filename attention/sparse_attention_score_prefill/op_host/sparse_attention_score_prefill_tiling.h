/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef SPARSE_ATTENTION_SCORE_PREFILL_TILING_H
#define SPARSE_ATTENTION_SCORE_PREFILL_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

// Tiling Key 常量 (与 op_kernel/sparse_attention_score_prefill_tilingkey.h 保持同步)
// 用于编译期选择不同的 kernel 代码路径
constexpr uint64_t SASA_PREFILL_BASE_TILING = 20000;               // base key, 所有合法 key >= 此值
constexpr uint64_t SASA_PREFILL_BF16_D128_TILING = 20001;          // bf16 输入, D=128, fp32 O_partial 路径
// innerPrecise==1: O_partial 以 bf16 存储 (PV fixpipe F322BF16 + Phase2 regbase cast bf16→fp32)
// 否则 (如 innerPrecise==4) 使用 fp32 O_partial 路径 (SASA_PREFILL_BF16_D128_TILING)
constexpr uint64_t SASA_PREFILL_BF16_D128_INNER_LOW_TILING = 20002; // bf16 输入, D=128, bf16 O_partial 路径

BEGIN_TILING_DATA_DEF(SparseAttentionScorePrefillTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);                  // batch 数
TILING_DATA_FIELD_DEF(uint32_t, numHeads);               // Q head 数
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);                // KV head 数
TILING_DATA_FIELD_DEF(uint32_t, groupSize);              // numHeads/kvHeads, GQA 分组大小
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);          // head 维度 D
TILING_DATA_FIELD_DEF(uint32_t, blockSize);              // KV block token 数 (通常 128)
TILING_DATA_FIELD_DEF(uint32_t, topK);                   // 每个 Q token 选择的 KV block 数
TILING_DATA_FIELD_DEF(uint32_t, totalQTokens);           // 所有 batch 的 Q token 总数
TILING_DATA_FIELD_DEF(uint32_t, numKvBlocks);            // totalPackedRows = Σ ceil(kv_seqlen/blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerBatch);      // block_table 行 stride
TILING_DATA_FIELD_DEF(uint32_t, k2qNnzUpperBound);       // CSR 数据区每 head 上界
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNumP1);         // Phase1 任务数 = numKvBlocks * kvHeads
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNumP2);         // Phase2 任务数 = totalQTokens * kvHeads
TILING_DATA_FIELD_DEF(float, scaleValue);                // softmax 缩放因子
TILING_DATA_FIELD_DEF(uint32_t, innerPrecise);           // 精度模式: 1=bf16 O_partial, 4=fp32 O_partial
TILING_DATA_FIELD_DEF(uint64_t, accumOutSize);           // workspace O_partial 元素数
TILING_DATA_FIELD_DEF(uint64_t, lseStatSize);            // workspace max/sum 各自元素数
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);          // workspace 总字节数
TILING_DATA_FIELD_DEF(uint64_t, gmSWorkspaceSize);       // A2 GM S 中转 workspace per-core 字节数 (A5=0)
TILING_DATA_FIELD_DEF(uint64_t, gmPWorkspaceSize);       // A2 GM P 中转 workspace per-core 字节数 (A5=0)
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);              // tiling key 用于 kernel 派发
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseAttentionScorePrefill, SparseAttentionScorePrefillTilingData)

struct SparseAttentionScorePrefillCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
};

// =========================================================================================
// Tiling 类层次: 基类 (公共逻辑) + A5/A2 子类 (架构差异化逻辑)
//
// SASAPrefillTilingBase  —— 公共: 解析属性/shape/任务切分, GetTiling 编排, SetTilingData
//   ├── SASAPrefillTilingA5  —— A5(ascend950): L0C→UB 直接 Fixpipe, 无需 GM S 中转
//   └── SASAPrefillTilingA2  —— A2(ascend910b): L0C→GM→UB 中转, 需额外 GM S staging workspace
//
// 架构差异点 (virtual):
//   CalculateWorkSpace — A2 额外计算 blockDim×64KB GM S workspace
//   FillTilingData     — 各自独立填充 tiling 字段 (A5 gmSWorkspaceSize=0, A2 非零)
// =========================================================================================
class SASAPrefillTilingBase {
public:
    SASAPrefillTilingBase() = default;
    virtual ~SASAPrefillTilingBase() = default;

    ge::graphStatus GetTiling(gert::TilingContext *context,
                              SparseAttentionScorePrefillTilingData &tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext *context,
                                  SparseAttentionScorePrefillTilingData &tilingData);

protected:
    ge::graphStatus GetNpuInfo(gert::TilingContext *context);
    ge::graphStatus ParseAttrs(gert::TilingContext *context);
    ge::graphStatus ParseInputTensors(gert::TilingContext *context);
    ge::graphStatus ParseSeqlens(gert::TilingContext *context);
    ge::graphStatus CalculateReverseIndexMeta(gert::TilingContext *context);
    ge::graphStatus CalculateTaskSplit(gert::TilingContext *context);

    // 架构差异化: workspace 计算 + tiling 填充
    virtual ge::graphStatus CalculateWorkSpace(gert::TilingContext *context) = 0;
    virtual ge::graphStatus FillTilingData(gert::TilingContext *context) = 0;

protected:
    uint32_t batch_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t embeddingSize_ = 0;
    uint32_t blockSize_ = 128;
    uint32_t topK_ = 8;
    uint32_t totalQTokens_ = 0;
    uint32_t numKvBlocks_ = 0;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t k2qNnzUpperBound_ = 0;
    float scaleValue_ = 0.0f;
    uint32_t innerPrecise_ = 0;

    const int32_t *qSeqLenList_ = nullptr;
    const int32_t *kvSeqLenList_ = nullptr;

    uint64_t workSpaceSize_ = 0;
    uint64_t accumOutSize_ = 0;
    uint64_t lseStatSize_ = 0;
    uint64_t gmSWorkspaceSize_ = 0;   // A2 GM S 中转 workspace per-core 字节数 (A5=0)
    uint64_t gmPWorkspaceSize_ = 0;   // A2 GM P 中转 workspace per-core 字节数 (A5=0)
    bool isA2_ = false;               // 架构标志: true=A2(ascend910b), false=A5(ascend950)

    uint32_t blockDim_ = 20;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t libapiSize_ = 0;

    ge::DataType dataType_ = ge::DT_BF16;

    SparseAttentionScorePrefillTilingData *tilingData_ = nullptr;
};

// A5 (ascend950): L0C→UB 直接 Fixpipe, 无需 GM S 中转 workspace
// 实现位于 sparse_attention_score_prefill_tiling.cpp
class SASAPrefillTilingA5 : public SASAPrefillTilingBase {
protected:
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context) override;
    ge::graphStatus FillTilingData(gert::TilingContext *context) override;
};

// A2 (ascend910b): L0C→GM→UB 中转, 需额外 per-core GM S staging workspace
// 实现位于 sparse_attention_score_prefill_tiling_a2.cpp
class SASAPrefillTilingA2 : public SASAPrefillTilingBase {
protected:
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context) override;
    ge::graphStatus FillTilingData(gert::TilingContext *context) override;
};

}  // namespace optiling

#endif  // SPARSE_ATTENTION_SCORE_PREFILL_TILING_H
