# kv_quant_sparse_attn_sharedkv 算子代码修改点识别

## 修改概览

| 层级 | 文件数量 | 修改点数量 | 优先级 |
|------|---------|-----------|--------|
| Host端（op_host） | 7个文件 | 23个修改点 | P0（必须） |
| Kernel端（op_kernel） | 5个文件 | 18个修改点 | P0（必须） |
| Kernel入口 | 1个文件 | 3个修改点 | P0（必须） |
| 测试端（tests） | 6个文件 | 12个修改点 | P1（推荐） |
| 文档 | 1个文件 | 1个修改点 | P2（可选） |

---

## 一、Host端修改（op_host）

### 1.1 参数定义层（op_host/kv_quant_sparse_attn_sharedkv_def.cpp）

**文件位置**：`op_host/kv_quant_sparse_attn_sharedkv_def.cpp`

**修改点1：新增可选输入参数**
```cpp
// 行号：78-82（在seqused_kv之后新增）

当前代码：
this->Input("seqused_kv")
    .ParamType(OPTIONAL)
    .DataType({ge::DT_INT32})
    .Format({ge::FORMAT_ND})
    .AutoContiguous();

修改后：
this->Input("seqused_kv")  // 保留原参数
    .ParamType(OPTIONAL)
    .DataType({ge::DT_INT32})
    .Format({ge::FORMAT_ND})
    .AutoContiguous();
    
// 新增参数
this->Input("seqused_ori_kv")  // 新增：ori_kv的真实使用长度
    .ParamType(OPTIONAL)
    .DataType({ge::DT_INT32})
    .Format({ge::FORMAT_ND})
    .AutoContiguous();

this->Input("seqused_cmp_kv")  // 新增：cmp_kv的真实使用长度
    .ParamType(OPTIONAL)
    .DataType({ge::DT_INT32})
    .Format({ge::FORMAT_ND})
    .AutoContiguous();

this->Input("cmp_residual_kv")  // 新增：cmp的余数
    .ParamType(OPTIONAL)
    .DataType({ge::DT_INT32})
    .Format({ge::FORMAT_ND})
    .AutoContiguous();
```

**影响范围**：参数索引需要重新编排

---

### 1.2 参数索引定义（op_host/kv_quant_sparse_attn_sharedkv_check.h）

**文件位置**：`op_host/kv_quant_sparse_attn_sharedkv_check.h`

**修改点2：新增参数索引常量**
```cpp
// 行号：99-114（Inputs Index定义）

当前代码：
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t ORI_KV_INDEX = 1;
constexpr uint32_t CMP_KV_INDEX = 2;
constexpr uint32_t ORI_SPARSE_INDICES_INDEX = 3;
constexpr uint32_t CMP_SPARSE_INDICES_INDEX = 4;
constexpr uint32_t ORI_BLOCK_TABLE_INDEX = 5;
constexpr uint32_t CMP_BLOCK_TABLE_INDEX = 6;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 7;
constexpr uint32_t CU_SEQLENS_ORI_KV_INDEX = 8;
constexpr uint32_t CU_SEQLENS_CMP_KV_INDEX = 9;
constexpr uint32_t SEQUSED_Q_INDEX = 10;
constexpr uint32_t SEQUSED_KV_INDEX = 11;
constexpr uint32_t SINKS_INDEX = 12;
constexpr uint32_t METADATA_INDEX = 13;

修改后：
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t ORI_KV_INDEX = 1;
constexpr uint32_t CMP_KV_INDEX = 2;
constexpr uint32_t ORI_SPARSE_INDICES_INDEX = 3;
constexpr uint32_t CMP_SPARSE_INDICES_INDEX = 4;
constexpr uint32_t ORI_BLOCK_TABLE_INDEX = 5;
constexpr uint32_t CMP_BLOCK_TABLE_INDEX = 6;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 7;
constexpr uint32_t CU_SEQLENS_ORI_KV_INDEX = 8;
constexpr uint32_t CU_SEQLENS_CMP_KV_INDEX = 9;
constexpr uint32_t SEQUSED_Q_INDEX = 10;
constexpr uint32_t SEQUSED_KV_INDEX = 11;  // 保留
constexpr uint32_t SEQUSED_ORI_KV_INDEX = 12;  // 新增
constexpr uint32_t SEQUSED_CMP_KV_INDEX = 13;  // 新增
constexpr uint32_t CMP_RESIDUAL_KV_INDEX = 14;  // 新增
constexpr uint32_t SINKS_INDEX = 15;  // 索引调整：12→15
constexpr uint32_t METADATA_INDEX = 16;  // 索引调整：13→16
```

**修改点3：新增参数名称常量**
```cpp
// 行号：46（在SEQUSED_KV_NAME之后新增）

当前代码：
const std::string SEQUSED_KV_NAME = "seqused_kv";

修改后：
const std::string SEQUSED_KV_NAME = "seqused_kv";
const std::string SEQUSED_ORI_KV_NAME = "seqused_ori_kv";  // 新增
const std::string SEQUSED_CMP_KV_NAME = "seqused_cmp_kv";  // 新增
const std::string CMP_RESIDUAL_KV_NAME = "cmp_residual_kv";  // 新增
```

**修改点4：参数信息结构体新增字段**
```cpp
// 行号：220-224（KvQuantSASParaInfo结构体）

当前代码：
SASTilingOptionalParaInfo seqUsedQ = {nullptr, nullptr};
SASTilingOptionalParaInfo sequsedKv = {nullptr, nullptr};
SASTilingOptionalParaInfo sinks = {nullptr, nullptr};
SASTilingOptionalParaInfo metadata = {nullptr, nullptr};

修改后：
SASTilingOptionalParaInfo seqUsedQ = {nullptr, nullptr};
SASTilingOptionalParaInfo sequsedKv = {nullptr, nullptr};  // 保留
SASTilingOptionalParaInfo sequsedOriKv = {nullptr, nullptr};  // 新增
SASTilingOptionalParaInfo sequsedCmpKv = {nullptr, nullptr};  // 新增
SASTilingOptionalParaInfo cmpResidualKv = {nullptr, nullptr};  // 新增
SASTilingOptionalParaInfo sinks = {nullptr, nullptr};  // 索引调整
SASTilingOptionalParaInfo metadata = {nullptr, nullptr};  // 索引调整
```

---

### 1.3 参数获取层（op_host/kv_quant_sparse_attn_sharedkv_tiling.cpp）

**文件位置**：`op_host/kv_quant_sparse_attn_sharedkv_tiling.cpp`

**修改点5：获取新增参数**
```cpp
// 行号：108-109（GetOpParamInfo函数）

当前代码：
opParamInfo_.sequsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
opParamInfo_.sequsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);

修改后：
opParamInfo_.sequsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
opParamInfo_.sequsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);

// 新增参数获取
opParamInfo_.sequsedOriKv.tensor = context_->GetOptionalInputTensor(SEQUSED_ORI_KV_INDEX);
opParamInfo_.sequsedOriKv.desc = context_->GetOptionalInputDesc(SEQUSED_ORI_KV_INDEX);

opParamInfo_.sequsedCmpKv.tensor = context_->GetOptionalInputTensor(SEQUSED_CMP_KV_INDEX);
opParamInfo_.sequsedCmpKv.desc = context_->GetOptionalInputDesc(SEQUSED_CMP_KV_INDEX);

opParamInfo_.cmpResidualKv.tensor = context_->GetOptionalInputTensor(CMP_RESIDUAL_KV_INDEX);
opParamInfo_.cmpResidualKv.desc = context_->GetOptionalInputDesc(CMP_RESIDUAL_KV_INDEX);

// 索引调整
opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);  // 索引调整
opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);

opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);  // 索引调整
opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
```

**修改点6：参数校验与默认值处理**
```cpp
// 新增函数：在GetOpParamInfo之后

ge::graphStatus KvQuantSASInfoParser::ProcessNewActualSeqParams()
{
    // 处理seqused_ori_kv
    if (opParamInfo_.sequsedOriKv.tensor == nullptr) {
        // 默认值：使用seqused_kv（兼容旧逻辑）
        opParamInfo_.sequsedOriKv.tensor = opParamInfo_.sequsedKv.tensor;
        actualLenDimsOriKV_ = actualLenDimsKV_;
    } else {
        actualLenDimsOriKV_ = opParamInfo_.sequsedOriKv.tensor->GetShapeSize();
    }
    
    // 处理seqused_cmp_kv
    if (opParamInfo_.sequsedCmpKv.tensor == nullptr) {
        // 默认值：floor(seqused_ori_kv / cmp_ratio)
        // 注意：此时需要在kernel端计算，这里只标记为nullptr
        actualLenDimsCmpKV_ = 0;  // 标记为未传入
    } else {
        actualLenDimsCmpKV_ = opParamInfo_.sequsedCmpKv.tensor->GetShapeSize();
    }
    
    // 处理cmp_residual_kv
    if (opParamInfo_.cmpResidualKv.tensor == nullptr) {
        // cmp_mask_mode=3且cmp_ratio≠1时需要计算
        if (*opParamInfo_.cmpMaskMode == 3 && *opParamInfo_.cmpRatio != 1) {
            // 在kernel端计算：residual = seqused_ori_kv % cmp_ratio
            cmpResidualKVSize_ = 0;  // 标记为未传入
        }
    } else {
        cmpResidualKVSize_ = opParamInfo_.cmpResidualKv.tensor->GetShapeSize();
    }
    
    return ge::GRAPH_SUCCESS;
}
```

**修改点7：TilingData传递**
```cpp
// 新增TilingData字段（kv_quant_sparse_attn_sharedkv_tiling.h）

BEGIN_TILING_DATA_DEF(KvQuantSparseAttnSharedkvBaseParams)
// ... 原有字段
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLenSizeOriKV)  // 新增
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLenSizeCmpKV)  // 新增
TILING_DATA_FIELD_DEF(uint32_t, cmpResidualKVSize)      // 新增
END_TILING_DATA_DEF
```

**修改点8：设置TilingData**
```cpp
// kv_quant_sparse_attn_sharedkv_tiling.cpp中的DoOpTiling函数

// 新增
tilingData_.baseParams.set_actualSeqLenSizeOriKV(tilingInfo->actualLenDimsOriKV);
tilingData_.baseParams.set_actualSeqLenSizeCmpKV(tilingInfo->actualLenDimsCmpKV);
tilingData_.baseParams.set_cmpResidualKVSize(tilingInfo->cmpResidualKVSize);
```

---

### 1.4 参数校验层（op_host/kv_quant_sparse_attn_sharedkv_check.cpp等）

**文件位置**：`op_host/kv_quant_sparse_attn_sharedkv_check.cpp`

**修改点9：新增校验逻辑**
```cpp
// 在现有校验函数中新增

ge::graphStatus CheckNewActualSeqParams(KvQuantSASTilingInfo *sasInfo)
{
    // 校验seqused_ori_kv
    if (sasInfo->opParamInfo.sequsedOriKv.tensor != nullptr) {
        if (sasInfo->opParamInfo.sequsedOriKv.tensor->GetShapeSize() != sasInfo->bSize) {
            OP_LOGE(sasInfo->opName, "seqused_ori_kv size should be equal to batch size %u, but got %ld",
                sasInfo->bSize, sasInfo->opParamInfo.sequsedOriKv.tensor->GetShapeSize());
            return ge::GRAPH_FAILED;
        }
        
        // 校验：seqused_ori_kv <= storage_ori_kv
        // 需要读取ori_kv的storage shape
    }
    
    // 校验seqused_cmp_kv
    if (sasInfo->opParamInfo.sequsedCmpKv.tensor != nullptr) {
        if (sasInfo->opParamInfo.sequsedCmpKv.tensor->GetShapeSize() != sasInfo->bSize) {
            OP_LOGE(sasInfo->opName, "seqused_cmp_kv size should be equal to batch size %u, but got %ld",
                sasInfo->bSize, sasInfo->opParamInfo.sequsedCmpKv.tensor->GetShapeSize());
            return ge::GRAPH_FAILED;
        }
        
        // 校验：seqused_cmp_kv <= storage_cmp_kv
    }
    
    // 校验cmp_residual_kv
    if (sasInfo->opParamInfo.cmpResidualKv.tensor != nullptr) {
        if (*sasInfo->opParamInfo.cmpMaskMode != 3 || *sasInfo->opParamInfo.cmpRatio == 1) {
            OP_LOGE(sasInfo->opName, "cmp_residual_kv should only be provided when cmp_mask_mode=3 and cmp_ratio!=1");
            return ge::GRAPH_FAILED;
        }
        
        if (sasInfo->opParamInfo.cmpResidualKv.tensor->GetShapeSize() != sasInfo->bSize) {
            OP_LOGE(sasInfo->opName, "cmp_residual_kv size should be equal to batch size %u, but got %ld",
                sasInfo->bSize, sasInfo->opParamInfo.cmpResidualKv.tensor->GetShapeSize());
            return ge::GRAPH_FAILED;
        }
        
        // 校验：residual < cmp_ratio
        // 需要读取每个batch的residual值并校验
    }
    
    return ge::GRAPH_SUCCESS;
}
```

**文件位置**：`op_host/kv_quant_sparse_attn_sharedkv_check_consistancy.cpp`

**修改点10：一致性校验**
```cpp
// 新增一致性校验逻辑

ge::graphStatus CheckActualSeqConsistancy(KvQuantSASTilingInfo *sasInfo)
{
    // 如果传入seqused_cmp_kv和cmp_residual_kv，校验关系
    if (sasInfo->opParamInfo.sequsedOriKv.tensor != nullptr &&
        sasInfo->opParamInfo.sequsedCmpKv.tensor != nullptr &&
        sasInfo->opParamInfo.cmpResidualKv.tensor != nullptr) {
        
        // 校验公式：seqused_ori_kv = seqused_cmp_kv * cmp_ratio + cmp_residual_kv
        for (uint32_t bIdx = 0; bIdx < sasInfo->bSize; bIdx++) {
            int32_t ori_len = sasInfo->opParamInfo.sequsedOriKv.tensor->GetData<int32_t>(bIdx);
            int32_t cmp_len = sasInfo->opParamInfo.sequsedCmpKv.tensor->GetData<int32_t>(bIdx);
            int32_t residual = sasInfo->opParamInfo.cmpResidualKv.tensor->GetData<int32_t>(bIdx);
            
            if (ori_len != cmp_len * (*sasInfo->opParamInfo.cmpRatio) + residual) {
                OP_LOGE(sasInfo->opName, "batch %u: seqused_ori_kv=%d should equal to "
                    "seqused_cmp_kv=%d * cmp_ratio=%d + residual=%d",
                    bIdx, ori_len, cmp_len, *sasInfo->opParamInfo.cmpRatio, residual);
                return ge::GRAPH_FAILED;
            }
        }
    }
    
    return ge::GRAPH_SUCCESS;
}
```

---

## 二、Kernel端修改（op_kernel）

### 2.1 数据结构定义（op_kernel/arch35/util_regbase.h）

**文件位置**：`op_kernel/arch35/util_regbase.h`

**修改点11：ConstInfo新增字段**
```cpp
// 行号：190-211（COMMON_CONST_INFO宏定义）

当前代码：
#define COMMON_CONST_INFO \
    /* 推理 */ \
    bool isActualLenDimsNull; /* 判断是否有actualseq */ \
    bool isActualLenDimsKVNull; /* 判断是否有actualseq_kv */ \
    bool isSoftmaxLseEnable; \
    bool rsvd1; \
    uint32_t sparseBlockCount; \
    uint32_t actualSeqLenSize; /* 用户输入的actualseq的长度 */ \
    uint32_t actualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */

修改后：
#define COMMON_CONST_INFO \
    /* 推理 */ \
    bool isActualLenDimsNull; /* 判断是否有actualseq */ \
    bool isActualLenDimsKVNull; /* 判断是否有actualseq_kv */ \
    bool isActualLenDimsOriKVNull; /* 判断是否有seqused_ori_kv */ /* 新增 */ \
    bool isActualLenDimsCmpKVNull; /* 判断是否有seqused_cmp_kv */ /* 新增 */ \
    bool isCmpResidualKVNull; /* 判断是否有cmp_residual_kv */ /* 新增 */ \
    bool isSoftmaxLseEnable; \
    bool rsvd1; \
    uint32_t sparseBlockCount; \
    uint32_t actualSeqLenSize; /* 用户输入的actualseq的长度 */ \
    uint32_t actualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */ \
    uint32_t actualSeqLenSizeOriKV; /* seqused_ori_kv的长度 */ /* 新增 */ \
    uint32_t actualSeqLenSizeCmpKV; /* seqused_cmp_kv的长度 */ /* 新增 */ \
    uint32_t cmpResidualKVSize; /* cmp_residual_kv的长度 */ /* 新增 */
```

**修改点12：CVSharedParams新增字段**
```cpp
// 行号：212-243（CV_SHARED_PARAMS宏定义）

当前代码：
#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    uint32_t bSize;  \
    uint32_t n2Size;  \
    uint32_t gSize;  \
    uint32_t s1Size;  \
    uint32_t s2Size;  \
    uint32_t dSize : 10;  \
    uint32_t dSizeVInput : 12;  \
    uint32_t needInit : 4;  \
    uint32_t layoutType : 4;  \
    uint32_t isActualSeqLengthsNull : 1; \
    uint32_t isActualSeqLengthsKVNull : 1; \
    uint32_t sparseBlockCount; \
    ...

修改后：
#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    uint32_t bSize;  \
    uint32_t n2Size;  \
    uint32_t gSize;  \
    uint32_t s1Size;  \
    uint32_t s2Size;  \
    uint32_t dSize : 10;  \
    uint32_t dSizeVInput : 12;  \
    uint32_t needInit : 4;  \
    uint32_t layoutType : 4;  \
    uint32_t isActualSeqLengthsNull : 1; \
    uint32_t isActualSeqLengthsKVNull : 1; \
    uint32_t isActualSeqLengthsOriKVNull : 1; /* 新增 */ \
    uint32_t isActualSeqLengthsCmpKVNull : 1; /* 新增 */ \
    uint32_t isCmpResidualKVNull : 1; /* 新增 */ \
    uint32_t sparseBlockCount; \
    ...
```

---

### 2.2 Kernel入口（op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h）

**文件位置**：`op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h`

**修改点13：Init函数新增参数**
```cpp
// 行号：46-53（Init函数签名）

当前代码：
__aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
    __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *user, __gm__ uint8_t *tiling);

修改后：
__aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
    __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv, /* 新增3个参数 */
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *user, __gm__ uint8_t *tiling);
```

**修改点14：InitGlobalBuffer新增参数**
```cpp
// 行号：160-167（InitGlobalBuffer函数）

当前代码：
__aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
    __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sinks, __gm__ uint8_t *workspace);

修改后：
__aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
    __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv, /* 新增 */
    __gm__ uint8_t *sinks, __gm__ uint8_t *workspace);
```

**修改点15：地址初始化**
```cpp
// 在InitGlobalBuffer函数内部新增

if (sequsedOriKv != nullptr) {
    actualSeqOriKvlenAddr = (__gm__ int32_t *)sequsedOriKv;
}

if (sequsedCmpKv != nullptr) {
    actualSeqCmpKvlenAddr = (__gm__ int32_t *)sequsedCmpKv;
}

if (cmpResidualKv != nullptr) {
    cmpResidualKvAddr = (__gm__ int32_t *)cmpResidualKv;
}
```

**修改点16：InitUniqueConstInfo新增**
```cpp
// 行号：282-287（InitUniqueConstInfo函数）

当前代码：
__aicore__ inline void InitUniqueConstInfo()
{
    this->constInfo.actualSeqLenSize = this->sharedParams.bSize + 1;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.bSize;
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
}

修改后：
__aicore__ inline void InitUniqueConstInfo()
{
    this->constInfo.actualSeqLenSize = this->sharedParams.bSize + 1;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.bSize;
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
    
    // 新增
    this->constInfo.actualSeqLenSizeOriKV = this->sharedParams.bSize;
    this->constInfo.actualSeqLenSizeCmpKV = this->sharedParams.bSize;
    this->constInfo.cmpResidualKVSize = this->sharedParams.bSize;
    
    this->constInfo.isActualLenDimsOriKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsOriKVNull);
    this->constInfo.isActualLenDimsCmpKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsCmpKVNull);
    this->constInfo.isCmpResidualKVNull = static_cast<bool>(this->sharedParams.isCmpResidualKVNull);
}
```

**修改点17：新增成员变量**
```cpp
// 在类定义中新增成员变量

__gm__ int32_t *actualSeqOriKvlenAddr = nullptr;  // 新增
__gm__ int32_t *actualSeqCmpKvlenAddr = nullptr;  // 新增
__gm__ int32_t *cmpResidualKvAddr = nullptr;      // 新增
```

---

### 2.3 KV长度计算（op_kernel/arch35/kv_quant_sparse_attn_sharedkv_kvcache.h）

**文件位置**：`op_kernel/arch35/kv_quant_sparse_attn_sharedkv_kvcache.h`

**修改点18：GetSingleCoreParam使用新参数**
```cpp
// 行号：29-69（GetSingleCoreParam函数）

当前代码（核心部分）：
if (constInfo.isActualLenDimsKVNull) {
    actualS2Size = constInfo.s2Size;
} else {
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        actualS2Size = actualSeqKvlenAddr[sIdx];
        if ((sIdx > 0) && (!isPa)) {
            actualS2Size -= actualSeqKvlenAddr[sIdx - 1];
        }
    } else {
        actualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ?
            actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
    }
}

修改后：
// ori部分真实使用长度
if (!constInfo.isActualLenDimsOriKVNull) {
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        actualOriS2Size = actualSeqOriKvlenAddr[sIdx];
    } else {
        actualOriS2Size = actualSeqOriKvlenAddr[sIdx];
    }
} else {
    // 兼容旧逻辑：使用seqused_kv或计算
    if (!constInfo.isActualLenDimsKVNull) {
        if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
            actualOriS2Size = actualSeqKvlenAddr[sIdx];
        } else {
            actualOriS2Size = actualSeqKvlenAddr[sIdx];
        }
    } else {
        actualOriS2Size = constInfo.s2Size;
    }
}

// cmp部分真实使用长度
if (!constInfo.isActualLenDimsCmpKVNull) {
    actualCmpS2Size = actualSeqCmpKvlenAddr[sIdx];
} else {
    // 兼容旧逻辑：floor(actualOriS2Size / cmpRatio)
    actualCmpS2Size = actualOriS2Size / constInfo.cmpRatio;
}

// 总的KV长度（ori + cmp）
actualS2Size = actualOriS2Size;  // 对于当前代码，actualS2Size代表ori部分

runParam.actualS1Size = actualS1Size;
runParam.actualS2Size = actualS2Size;
runParam.actualOriS2Size = actualOriS2Size;  // 新增字段
runParam.actualCmpS2Size = actualCmpS2Size;  // 新增字段
```

**修改点19：RunParamStr新增字段**
```cpp
// 在util_regbase.h的RunParamStr结构体中新增

int64_t actualOriS2Size;  /* ori部分的真实使用长度 */ /* 新增 */
int64_t actualCmpS2Size;  /* cmp部分的真实使用长度 */ /* 新增 */
int64_t cmpResidual;      /* cmp的余数 */ /* 新增 */
```

---

### 2.4 Mask计算修正（op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h）

**文件位置**：`op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h`

**修改点20：CopyInKvSparse mask计算**
```cpp
// 行号：237-240（CopyInKvSparse函数）

当前代码：
int64_t s2IdLimit = runInfo.s2RealSize;
s2IdLimit = (runInfo.s2RealSize - runInfo.actualS1Size + runInfo.s1oIdx + 1) / constInfo.cmpRatio;

修改后：
// 新的mask计算逻辑
int64_t accessible_kv = runInfo.actualS2Size - runInfo.actualS1Size + runInfo.s1oIdx + 1;

int64_t s2IdLimit = 0;
if (constInfo.cmpMaskMode == 3 && constInfo.cmpRatio != 1) {
    // 获取余数
    int64_t residual = 0;
    if (!constInfo.isCmpResidualKVNull) {
        residual = cmpResidualKvAddr[runInfo.boIdx];
    } else {
        // 兼容计算：residual = actualS2Size % cmpRatio
        residual = runInfo.actualS2Size % constInfo.cmpRatio;
    }
    
    // 使用新公式
    if (accessible_kv <= residual) {
        s2IdLimit = 0;  // 全在ori部分
    } else {
        s2IdLimit = (accessible_kv - residual) / constInfo.cmpRatio;
    }
    
    // 确保不越界
    int64_t cmpActLen = 0;
    if (!constInfo.isActualSeqLengthsCmpKVNull) {
        cmpActLen = actualSeqCmpKvlenAddr[runInfo.boIdx];
    } else {
        cmpActLen = runInfo.actualS2Size / constInfo.cmpRatio;
    }
    s2IdLimit = Min(s2IdLimit, cmpActLen);
} else {
    // 原逻辑：cmp_ratio=1或cmp_mask_mode!=3
    s2IdLimit = accessible_kv / constInfo.cmpRatio;
}

// 确保不超过实际cmp长度
s2IdLimit = Min(s2IdLimit, runInfo.actualCmpS2Size);
```

**修改点21：InitCubeVecSharedParams参数传递**
```cpp
// 行号：57-68（InitCubeVecSharedParams函数签名和调用）

当前代码：
vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx, 
    cuSeqlensQ, sequsedKv);

修改后：
vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx, 
    cuSeqlensQ, sequsedKv, sequsedOriKv, sequsedCmpKv, cmpResidualKv);  // 新增参数
```

**修改点22：InitVecBlock新增参数**
```cpp
// 在SCFABlockVec类中修改InitVecBlock函数签名

当前代码：
__aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseAttnSharedkvTilingData *__restrict tiling,
    CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, 
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv);

修改后：
__aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseAttnSharedkvTilingData *__restrict tiling,
    CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, 
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv);  // 新增
```

---

### 2.5 Kernel主入口（op_kernel/kv_quant_sparse_attn_sharedkv.cpp）

**文件位置**：`op_kernel/kv_quant_sparse_attn_sharedkv.cpp`

**修改点23：Kernel函数签名**
```cpp
// 行号：52-59（kernel函数签名）

当前代码：
template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int SPLIT_G>
__global__ __aicore__ void
kv_quant_sparse_attn_sharedkv(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
    __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
    __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKV,
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmax_lse, 
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)

修改后：
template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int SPLIT_G>
__global__ __aicore__ void
kv_quant_sparse_attn_sharedkv(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
    __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
    __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKV,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv, /* 新增 */
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmax_lse, 
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
```

**修改点24：Init调用**
```cpp
// 行号：32-34和46-48（SAS_OP_IMPL宏）

当前代码：
op.Init(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ,
    seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, user, nullptr, &tPipe);

修改后：
op.Init(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ,
    seqUsedQ, seqUsedKV, sequsedOriKv, sequsedCmpKv, cmpResidualKv, /* 新增 */
    sinks, metadata, attentionOut, user, nullptr, &tPipe);
```

---

## 三、测试端修改（tests）

### 3.1 测试参数生成（tests/pytest/kv_quant_sparse_attn_sharedkv_golden.py）

**文件位置**：`tests/pytest/kv_quant_sparse_attn_sharedkv_golden.py`

**修改点25：新增测试参数**
```python
# 行号：506-510（gen_golden_data函数）

当前代码：
seqused_kv = torch.tensor(seqused_kv).to(torch.int32)

修改后：
seqused_kv = torch.tensor(seqused_kv).to(torch.int32)  # 保留（存储长度）

# 新增参数生成
seqused_ori_kv = seqused_kv.clone()  # 真实使用长度（默认等于存储长度）
seqused_cmp_kv = torch.tensor([math.floor(s / cmp_ratio) for s in seqused_kv]).to(torch.int32)
cmp_residual_kv = torch.tensor([s % cmp_ratio for s in seqused_kv]).to(torch.int32)
```

**修改点26：返回数据结构**
```python
# 行号：617-641（返回字典）

当前代码：
'seqused_kv': seqused_kv,

修改后：
'seqused_kv': seqused_kv,  # 保留（存储长度）
'seqused_ori_kv': seqused_ori_kv,  # 新增
'seqused_cmp_kv': seqused_cmp_kv,  # 新增
'cmp_residual_kv': cmp_residual_kv,  # 新增
```

**修改点27：mask计算使用新公式**
```python
# 行号：127（gather_cmp_kv函数）

当前代码：
threshold = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / self.cmp_ratio)

修改后：
# 使用新公式
residual = cur_act_kv % self.cmp_ratio
accessible_kv = cur_act_kv - cur_act_q + i_S1 + 1
if accessible_kv <= residual:
    threshold = 0
else:
    threshold = math.floor((accessible_kv - residual) / self.cmp_ratio)
threshold = min(threshold, self.seqused_cmp_kv[i_B])  # 确保不越界
```

---

### 3.2 测试参数集合（tests/pytest/kv_quant_sparse_attn_sharedkv_paramset.py）

**文件位置**：`tests/pytest/kv_quant_sparse_attn_sharedkv_paramset.py`

**修改点28：新增参数组合**
```python
# 行号：36-40和70-74

当前代码：
"seqused_kv": [None],
"cmp_ratio": [1],

修改后：
"seqused_kv": [None],  # 保留
"seqused_ori_kv": [None],  # 新增
"seqused_cmp_kv": [None],  # 新增
"cmp_residual_kv": [None],  # 新增
"cmp_ratio": [1],  # 保留（但可选参数传入时，cmp长度由参数决定）
```

---

### 3.3 测试工具函数（tests/pytest/utils.py）

**文件位置**：`tests/pytest/utils.py`

**修改点29：参数列表新增**
```python
# 行号：48（参数列表）

当前代码：
"block_size1", "block_size2", "softmax_scale", "cmp_ratio", "ori_mask_mode", "cmp_mask_mode", ...

修改后：
"block_size1", "block_size2", "softmax_scale", "cmp_ratio", "ori_mask_mode", "cmp_mask_mode",
"seqused_ori_kv", "seqused_cmp_kv", "cmp_residual_kv",  # 新增
...
```

**修改点30：参数字典新增**
```python
# 行号：92-95

修改后：
"seqused_kv": params[18],  # 保留
"seqused_ori_kv": params[NEW_INDEX_1],  # 新增
"seqused_cmp_kv": params[NEW_INDEX_2],  # 新增
"cmp_residual_kv": params[NEW_INDEX_3],  # 新增
```

---

### 3.4 Batch测试（tests/pytest/batch/）

**修改点31-34：batch相关文件修改**
- `test_kv_quant_sparse_attn_sharedkv_pt_save.py`：参数传递新增
- `kv_quant_sparse_attn_sharedkv_process.py`：函数签名新增
- 其他batch测试文件适配

---

### 3.5 单算子测试（tests/pytest/test_kv_quant_sparse_attn_sharedkv_single.py）

**修改点35：测试参数列表新增**
```python
# 行号：44-45

修改后：
"block_num1", "block_num2", "block_size1", "block_size2", "cu_seqlens_q", "seqused_kv", 
"seqused_ori_kv", "seqused_cmp_kv", "cmp_residual_kv",  # 新增
"softmax_scale", "cmp_ratio", ...
```

---

## 四、修改优先级与依赖关系

### P0级别（核心功能，必须修改）

**阶段1：参数定义与传递**
1. 修改点1-4：op_host参数定义层
2. 修改点5-8：op_host参数获取与TilingData
3. 修改点11-12：Kernel数据结构定义
4. 修改点23-24：Kernel入口函数签名

**阶段2：参数使用与计算**
5. 修改点13-17：Kernel初始化
6. 修改点18-19：KV长度计算
7. 修改点20：Mask计算核心逻辑

### P1级别（测试验证，推荐修改）

**阶段3：测试适配**
8. 修改点25-27：Golden计算修正
9. 修改点28-35：测试参数与工具函数

### P2级别（文档与示例，可选修改）

**阶段4：文档更新**
10. README.md：参数说明更新
11. 设计文档：使用示例补充

---

## 五、关键修改点详解

### 5.1 最核心的修改点：Mask计算（修改点20）

**位置**：`op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h:237-240`

**重要性**：★★★★★（最高）

**原因**：
- 这是解决cmp_mask_mode=3余数问题的关键
- 直接影响算子正确性
- 需要精确的数学推导和验证

**修改复杂度**：高
- 需要理解当前mask逻辑
- 需要理解余数问题的数学本质
- 需要处理兼容性（新旧公式共存）

---

### 5.2 参数索引调整（修改点2）

**位置**：`op_host/kv_quant_sparse_attn_sharedkv_check.h:99-114`

**重要性**：★★★★☆

**影响范围**：
- 所有使用参数索引的代码都需要调整
- SINKS_INDEX和METADATA_INDEX索引变化
- 可能导致现有调用错误

**修改策略**：
- 严格按照顺序插入新参数
- 更新所有引用这些索引的代码
- 编译验证索引正确性

---

### 5.3 默认值处理（修改点6）

**位置**：`op_host/kv_quant_sparse_attn_sharedkv_tiling.cpp`

**重要性**：★★★★☆

**关键逻辑**：
- `seqused_ori_kv`默认值：使用`seqused_kv`
- `seqused_cmp_kv`默认值：计算`floor(seqused_ori_kv / cmp_ratio)`
- `cmp_residual_kv`默认值：计算`seqused_ori_kv % cmp_ratio`

**修改策略**：
- 确保默认值计算正确
- 处理cmp_ratio=1的特殊情况
- 标记未传入参数，便于Kernel端判断

---

## 六、验证检查清单

### 编译检查
- [ ] 所有文件编译通过
- [ ] 无新增编译警告
- [ ] 参数索引一致性检查

### 功能检查
- [ ] 参数传入正确性
- [ ] 默认值计算正确性
- [ ] Mask边界计算正确性
- [ ] 兼容性测试通过

### 性能检查
- [ ] 无性能下降（参数传输开销）
- [ ] Kernel运行时间正常

---

## 七、实施建议

### 7.1 分阶段实施

**第1周**：参数定义与传递（修改点1-12, 23-24）
- 建立参数传递通路
- 验证参数获取正确性

**第2周**：参数使用与核心逻辑（修改点13-20）
- 实现参数读取
- 实现Mask计算修正
- 单元测试验证

**第3周**：测试与文档（修改点25-35）
- 测试用例适配
- 兼容性验证
- 文档更新

### 7.2 风险规避

**风险1：参数索引冲突**
- 解决：严格按照设计文档顺序插入
- 验证：编译时检查所有索引引用

**风险2：默认值计算错误**
- 解决：单元测试覆盖所有默认值场景
- 验证：与旧逻辑对比测试

**风险3：Mask公式复杂**
- 解决：数学推导+充分测试
- 验证：多个典型场景验证

---

**文档版本**：v1.0
**生成日期**：2026-05-11
**作者**：CANNBot