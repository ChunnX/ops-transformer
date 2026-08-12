/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

/*
 * 本文件定义 SparseAttentionScorePrefill 算子的输入、输出与属性，供 GE 图使用。
 * 通过继承 OpDef 描述算子在图中的接口契约：输入/输出的名称、参数类型、数据类型与格式，以及可选属性。
 */

#include "register/op_def_registry.h"

namespace ops {

// 算子定义类：描述算子在 GE 图中的输入、输出与属性规格。
class SparseAttentionScorePrefill : public OpDef {
public:
    explicit SparseAttentionScorePrefill(const char* name) : OpDef(name)
    {
        // query: [total_q_tokens, num_q_heads, D], bf16, TND 布局的 Q 矩阵
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        // key: [num_physical_blocks, blockSize, kvHeads, D], bf16, Paged K Cache（分页存储的 K 缓存）
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        // value: [num_physical_blocks, blockSize, kvHeads, D], bf16, Paged V Cache（分页存储的 V 缓存）
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        // blockTable: [batch, maxBlocksPerBatch], int32, 逻辑块到物理块的映射表
        this->Input("blockTable")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        // k2qRowPtr: per-head CSR row pointers [kvHeads, totalKvRows+1], int32. A plain runtime
        // tensor Input (no const-fold) — the kernel reads csrStart/csrEnd from this GM tensor
        // (params.k2qRowPtr) via GetValue; host tiling derives totalKvRows from the Input SHAPE
        // only (no host value-read, so no .tolist / aclIntArray / ConvertToTensor front-end).
        this->Input("k2qRowPtr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        // k2qQIndices: CSR 格式的列数据，全局 Q token id 列表
        this->Input("k2qQIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        // k2qSlotIndices: workspace 中的槽位索引，范围 [0, topK)
        this->Input("k2qSlotIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        // actualSeqLengths: 每个 batch 的实际 Q 序列长度
        this->Input("actualSeqLengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        // actualSeqLengthsKv: 每个 batch 的实际 KV 序列长度
        this->Input("actualSeqLengthsKv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        // attentionOut: [total_q_tokens, num_q_heads, D], bf16, 注意力计算输出
        this->Output("attentionOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        // numKeyValueHeads: KV 头数，默认值 1
        this->Attr("numKeyValueHeads").AttrType(OPTIONAL).Int(1);
        // scaleValue: 缩放因子，默认值 0.0（表示使用 1/sqrt(D)）
        this->Attr("scaleValue").AttrType(OPTIONAL).Float(0.0);
        // blockSize: 分页块大小，默认值 128
        this->Attr("blockSize").AttrType(OPTIONAL).Int(128);
        // topK: 稀疏注意力选取的 top-K 数量，默认值 8
        this->Attr("topK").AttrType(OPTIONAL).Int(8);
        // innerPrecise: 内部精度模式，默认值 4
        this->Attr("innerPrecise").AttrType(OPTIONAL).Int(4);

        // 声明算子支持的 AICore 型号配置。
        // ascend950: Atlas A5 (arch35, __CCE_AICORE__==310)
        // ascend910b: Atlas A2 (DAV_2201, __CCE_AICORE__==220)
        this->AICore().AddConfig("ascend950");
        this->AICore().AddConfig("ascend910b");
    }
};

// 将算子定义注册到算子注册表，使其可被 GE 识别与调用。
OP_ADD(SparseAttentionScorePrefill);

}  // namespace ops
