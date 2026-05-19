# kv_quant_sparse_attn_sharedkv 算子 actual_seq 参数增强设计文档

## 1. 需求概述

### 1.1 背景
当前算子存在以下限制：
- `seqused_kv`仅表示ori_kv的存储长度，无法区分存储长度与真实使用长度
- cmp部分长度由`cmp_ratio`和ori部分长度计算得出，不够灵活
- `cmp_mask_mode=3`且`cmp_ratio≠1`场景下，无法精确计算mask边界（存在余数问题）

### 1.2 新增需求
1. 新增4个可选输入参数，用于表示真实使用长度：
   - `seqused_q`：query每个batch的真实使用长度
   - `seqused_ori_kv`：ori_kv每个batch的真实使用长度
   - `seqused_cmp_kv`：cmp_kv每个batch的真实使用长度
   - `cmp_residual_kv`：cmp_mask_mode=3且cmp_ratio≠1时，每个batch的余数

2. cmp部分长度改为参数传入，不再根据cmp_ratio计算

3. cmp_residual_kv用于精确计算mask边界

---

## 2. 当前代码分析

### 2.1 存储长度 vs 真实使用长度
```
存储长度（Storage Length）：
- 指tensor实际分配的内存大小
- 用于数据搬运、内存访问
- 当前seqused_kv表示ori_kv的存储长度

真实使用长度（Actual Usage Length）：
- 指参与attention计算的有效token数
- 用于mask计算、循环边界判断
- 当前没有区分，与存储长度混用
```

**典型场景**：
- Prefill阶段：存储长度=真实使用长度
- Decode阶段：存储长度>真实使用长度（KV cache累积）

### 2.2 当前mask使用逻辑

#### 2.2.1 ori_mask_mode=4（band模式）
```cpp
// op_kernel/arch35/kv_quant_sparse_attn_sharedkv_kvcache.h:63-68
runParam.preTokensPerBatch = constInfo.oriWinLeft == -1 ? 
    runParam.actualS1Size : -(runParam.actualS2Size - runParam.actualS1Size - constInfo.oriWinLeft);
runParam.nextTokensPerBatch = runParam.actualS2Size - runParam.actualS1Size;
```
- 基于滑动窗口，由`oriWinLeft`和`oriWinRight`控制

#### 2.2.2 cmp_mask_mode=3（rightDownCausal模式）
```python
# tests/pytest/kv_quant_sparse_attn_sharedkv_golden.py:127
threshold = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
```

```cpp
// op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h:240
s2IdLimit = (runInfo.s2RealSize - runInfo.actualS1Size + runInfo.s1oIdx + 1) / constInfo.cmpRatio;
```

**mask计算公式**：
```
threshold = floor((actualS2Size - actualS1Size + s1Idx + 1) / cmpRatio)
```
- `actualS2Size`：KV的总长度（ori部分+cmp部分）
- `actualS1Size`：Q的长度
- `s1Idx`：当前Q的行索引（0~actualS1Size-1）
- `threshold`：cmp部分中有效的token数量

**几何含义**（下三角，右顶点划分）：
```
ori部分：实际KV长度 = actualS2Size
cmp部分：压缩后长度 = floor(actualS2Size / cmpRatio)

对于第s1Idx行的Q：
- 可以访问的KV token数：actualS2Size - actualS1Size + s1Idx + 1
- 在cmp部分可以访问的token数：threshold
```

#### 2.2.3 cmp_mask_mode=0
- 不进行mask，访问全部KV

### 2.3 当前cmp_ratio使用逻辑

#### 2.3.1 cmp长度计算
```python
# tests/pytest/kv_quant_sparse_attn_sharedkv_golden.py:124
cur_cmp_act_kv = math.floor(cur_act_kv / self.cmp_ratio)
```

```cpp
// op_kernel/arch35/kv_quant_sparse_attn_sharedkv_kvcache.h:229-232
runParam.s2CmpLineEndIdx = runParam.s2LineEndIdx / constInfo.cmpRatio;
runParam.cmpKvLoopEndIdx = (runParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
```

#### 2.3.2 cmp_ratio的余数问题
**问题描述**：
```
ori_act_kv = 10, cmp_ratio = 4
cmp_act_kv = floor(10 / 4) = 2

余数 = 10 - 2*4 = 2

mask_threshold计算：
第s1Idx=0行：
  threshold = floor((10 - s1 + 0 + 1) / 4) = floor(11 / 4) = 2
  
第s1Idx=1行：
  threshold = floor((10 - s1 + 1 + 1) / 4) = floor(12 / 4) = 3 ❌错误
  
问题：threshold=3超过了cmp_act_kv=2，导致访问越界
```

**根本原因**：
- cmp_ratio的除法存在余数，导致不同行的threshold不一致
- 当前公式`threshold = floor((actualS2Size - actualS1Size + s1Idx + 1) / cmpRatio)`在某些行会超过`cmp_act_kv`

**解决方案**：
- 需要传入`cmp_residual_kv`参数，表示每个batch的余数
- 用于精确计算mask边界

---

## 3. 设计方案

### 3.1 参数定义

#### 3.1.1 新增参数列表

| 参数名 | 类型 | 维度 | 含义 | 是否可选 |
|--------|------|------|------|---------|
| `seqused_q` | int32 | [B] | query每个batch的真实使用长度 | 可选 |
| `seqused_ori_kv` | int32 | [B] | ori_kv每个batch的真实使用长度 | 可选 |
| `seqused_cmp_kv` | int32 | [B] | cmp_kv每个batch的真实使用长度 | 可选 |
| `cmp_residual_kv` | int32 | [B] | cmp_mask_mode=3且cmp_ratio≠1时，每个batch的余数 | 可选 |

#### 3.1.2 参数语义说明

**存储长度 vs 真实使用长度**：
```
原始参数（存储长度）：
- query tensor shape：[B, N, S1, D] 或 [T, N, D]
- ori_kv tensor shape：[B, N, S2_ori, D] 或其他layout
- cmp_kv tensor shape：[B, N, S2_cmp, D] 或其他layout

新增参数（真实使用长度）：
- seqused_q[i]：batch i的query中实际参与计算的token数
- seqused_ori_kv[i]：batch i的ori_kv中实际参与计算的token数
- seqused_cmp_kv[i]：batch i的cmp_kv中实际参与计算的token数

关系：
- seqused_q[i] ≤ S1
- seqused_ori_kv[i] ≤ S2_ori
- seqused_cmp_kv[i] ≤ S2_cmp
```

**cmp_residual_kv的定义**：
```
当cmp_mask_mode=3且cmp_ratio≠1时：
cmp_residual_kv[i] = seqused_ori_kv[i] % cmp_ratio

示例：
ori_act_kv = 10, cmp_ratio = 4
cmp_residual_kv = 10 % 4 = 2

cmp_act_kv = floor(10 / 4) = 2
```

### 3.2 参数兼容性设计

#### 3.2.1 默认值规则

```python
# 未传入新参数时的默认行为
if seqused_q is None:
    seqused_q = seqused_kv  # 兼容旧逻辑，假设Q长度=KV长度

if seqused_ori_kv is None:
    seqused_ori_kv = seqused_kv  # 兼容旧逻辑

if seqused_cmp_kv is None:
    # 根据cmp_ratio计算
    seqused_cmp_kv = floor(seqused_ori_kv / cmp_ratio)

if cmp_residual_kv is None:
    # 根据公式计算
    cmp_residual_kv = seqused_ori_kv % cmp_ratio
```

#### 3.2.2 参数校验规则

```cpp
// op_host/kv_quant_sparse_attn_sharedkv_check.cpp
CHECK(seqused_q.size() == B);
CHECK(seqused_ori_kv.size() == B);
CHECK(seqused_cmp_kv.size() == B);
CHECK(cmp_residual_kv.size() == B);

CHECK(seqused_q[i] <= S1);
CHECK(seqused_ori_kv[i] <= S2_ori_storage);
CHECK(seqused_cmp_kv[i] <= S2_cmp_storage);

if (cmp_mask_mode == 3 && cmp_ratio != 1) {
    CHECK(cmp_residual_kv[i] < cmp_ratio);
    CHECK(seqused_ori_kv[i] == seqused_cmp_kv[i] * cmp_ratio + cmp_residual_kv[i]);
}
```

### 3.3 Mask计算改进

#### 3.3.1 cmp_mask_mode=3的修正公式

**原公式（存在余数问题）**：
```cpp
threshold = floor((actualS2Size - actualS1Size + s1Idx + 1) / cmpRatio)
```

**新公式（精确计算）**：
```cpp
// step1: 计算当前行可访问的总KV token数
int64_t accessible_kv_tokens = actualS2Size - actualS1Size + s1Idx + 1;

// step2: 计算在cmp部分可访问的token数（考虑余数）
int64_t cmp_accessible_tokens;
if (accessible_kv_tokens <= cmp_residual_kv) {
    cmp_accessible_tokens = 0;  // 全在ori部分
} else {
    cmp_accessible_tokens = (accessible_kv_tokens - cmp_residual_kv) / cmpRatio;
}

// step3: 确保不超过cmp_act_kv
threshold = min(cmp_accessible_tokens, seqused_cmp_kv);
```

**公式推导**：
```
ori部分 + cmp部分 = actualS2Size
ori部分 = cmp_act_kv * cmp_ratio + residual

第s1Idx行可访问的KV token总数：
accessible_kv = actualS2Size - actualS1Size + s1Idx + 1

判断是否跨越ori/cmp边界：
if accessible_kv <= residual:
    全在ori部分，cmp部分threshold=0
else:
    cmp部分threshold = (accessible_kv - residual) / cmp_ratio
```

#### 3.3.2 示例验证

**示例1：ori=10, cmp_ratio=4, residual=2**
```
cmp_act_kv = 2, residual = 2

第s1Idx=0行（Q第0行）：
  accessible_kv = 10 - s1 + 0 + 1 = 11
  threshold = (11 - 2) / 4 = 2.25 → floor=2 ✓

第s1Idx=1行（Q第1行）：
  accessible_kv = 10 - s1 + 1 + 1 = 12
  threshold = (12 - 2) / 4 = 2.5 → floor=2 ✓
  
原公式错误：
  threshold_old = 12 / 4 = 3 ✗ (越界)
```

**示例2：ori=8, cmp_ratio=4, residual=0**
```
cmp_act_kv = 2, residual = 0

第s1Idx=0行：
  accessible_kv = 8 - s1 + 0 + 1 = 9
  threshold = (9 - 0) / 4 = 2.25 → floor=2 ✓

第s1Idx=1行：
  accessible_kv = 8 - s1 + 1 + 1 = 10
  threshold = (10 - 0) / 4 = 2.5 → floor=2 ✓
```

### 3.4 代码修改点

#### 3.4.1 Host端（op_host）

**文件1：kv_quant_sparse_attn_sharedkv_def.cpp**
```cpp
// 新增可选输入
this->Input("seqused_q")  // 新增
this->Input("seqused_ori_kv")  // 新增
this->Input("seqused_cmp_kv")  // 新增
this->Input("cmp_residual_kv")  // 新增

// 保留原参数（存储长度）
this->Input("seqused_kv")  // 保留，改名建议：storage_len_kv
```

**文件2：kv_quant_sparse_attn_sharedkv_check.h**
```cpp
// 参数信息结构体新增字段
SASTilingOptionalParaInfo sequsedQ = {nullptr, nullptr};
SASTilingOptionalParaInfo sequsedOriKv = {nullptr, nullptr};
SASTilingOptionalParaInfo sequsedCmpKv = {nullptr, nullptr};
SASTilingOptionalParaInfo cmpResidualKv = {nullptr, nullptr};
```

**文件3：kv_quant_sparse_attn_sharedkv_tiling.cpp**
```cpp
// 获取新增参数
opParamInfo_.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
opParamInfo_.sequsedOriKv.tensor = context_->GetOptionalInputTensor(SEQUSED_ORI_KV_INDEX);
opParamInfo_.sequsedCmpKv.tensor = context_->GetOptionalInputTensor(SEQUSED_CMP_KV_INDEX);
opParamInfo_.cmpResidualKv.tensor = context_->GetOptionalInputTensor(CMP_RESIDUAL_KV_INDEX);

// 参数校验
if (opParamInfo_.sequsedCmpKv.tensor != nullptr && opParamInfo_.cmpRatio != nullptr) {
    // cmp长度由参数传入，不再由cmp_ratio计算
    // 校验：seqused_cmp_kv <= storage_cmp_kv
}

if (*opParamInfo_.cmpMaskMode == 3 && *opParamInfo_.cmpRatio != 1) {
    // 必须传入cmp_residual_kv或可以计算
    if (opParamInfo_.cmpResidualKv.tensor == nullptr) {
        // 计算余数：residual = seqused_ori_kv % cmp_ratio
    }
}
```

#### 3.4.2 Kernel端（op_kernel）

**文件1：util_regbase.h**
```cpp
struct ConstInfo {
    // 新增字段
    uint32_t actualSeqLenSizeQ;      // seqused_q的长度
    uint32_t actualSeqLenSizeOriKV;  // seqused_ori_kv的长度
    uint32_t actualSeqLenSizeCmpKV;  // seqused_cmp_kv的长度
    uint32_t cmpResidualKVSize;      // cmp_residual_kv的长度
    
    bool isActualSeqLenQNull;        // 是否传入seqused_q
    bool isActualSeqLenOriKVNull;    // 是否传入seqused_ori_kv
    bool isActualSeqLenCmpKVNull;    // 是否传入seqused_cmp_kv
    bool isCmpResidualKVNull;        // 是否传入cmp_residual_kv
};
```

**文件2：kv_quant_sparse_attn_sharedkv_scfa_kernel.h**
```cpp
// InitGlobalBuffer新增参数
__gm__ uint8_t *sequsedQ, 
__gm__ uint8_t *sequsedOriKv,
__gm__ uint8_t *sequsedCmpKv,
__gm__ uint8_t *cmpResidualKv

// 初始化地址
if (sequsedQ != nullptr) {
    actualSeqQlenAddr = (__gm__ int32_t *)sequsedQ;
}
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

**文件3：kv_quant_sparse_attn_sharedkv_kvcache.h**
```cpp
// GetSingleCoreParam使用新参数
if (!constInfo.isActualSeqLenQNull) {
    actualS1Size = actualSeqQlenAddr[sIdx];
} else {
    // 兼容旧逻辑
}

if (!constInfo.isActualSeqLenOriKVNull) {
    actualOriS2Size = actualSeqOriKvlenAddr[sIdx];
} else {
    // 兼容旧逻辑
}

if (!constInfo.isActualSeqLenCmpKVNull) {
    actualCmpS2Size = actualSeqCmpKvlenAddr[sIdx];
} else {
    // 兼容旧逻辑：floor(actualOriS2Size / cmpRatio)
}
```

**文件4：kv_quant_sparse_attn_sharedkv_scfa_block_vector.h**
```cpp
// CopyInKvSparse修改mask计算
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t SCFABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx, int64_t token1Idx,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    // 新的mask计算逻辑
    int64_t accessible_kv = runInfo.actualS2Size - runInfo.actualS1Size + runInfo.s1oIdx + 1;
    
    int64_t s2IdLimit = 0;
    if (constInfo.cmpMaskMode == 3 && constInfo.cmpRatio != 1) {
        // 使用新公式
        int64_t residual = constInfo.isCmpResidualKVNull ? 
            (runInfo.actualS2Size % constInfo.cmpRatio) : 
            cmpResidualKvAddr[runInfo.boIdx];
        
        if (accessible_kv <= residual) {
            s2IdLimit = 0;
        } else {
            s2IdLimit = (accessible_kv - residual) / constInfo.cmpRatio;
        }
        
        // 确保不越界
        int64_t cmpActLen = constInfo.isActualSeqLenCmpKVNull ?
            (runInfo.actualS2Size / constInfo.cmpRatio) :
            actualSeqCmpKvlenAddr[runInfo.boIdx];
        s2IdLimit = min(s2IdLimit, cmpActLen);
    } else {
        // 原逻辑：cmp_ratio=1或cmp_mask_mode!=3
        s2IdLimit = accessible_kv / constInfo.cmpRatio;
    }
    
    // 后续逻辑保持不变...
}
```

#### 3.4.3 Kernel入口（kv_quant_sparse_attn_sharedkv.cpp）

```cpp
// 新增参数传递
template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int SPLIT_G>
__global__ __aicore__ void
kv_quant_sparse_attn_sharedkv(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKV,  // 保留原参数（存储长度）
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv,  // 新增（真实使用长度）
    __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,  // 新增
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmax_lse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    // 传递给Init
    op.Init(query, oriKV, cmpKV, ..., sequsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv, ...);
}
```

---

## 4. 测试用例设计

### 4.1 基础功能测试

**测试1：参数传入验证**
```python
# 新参数全部传入
seqused_q = [s1_1, s1_2, ...]
seqused_ori_kv = [s2_ori_1, s2_ori_2, ...]
seqused_cmp_kv = [s2_cmp_1, s2_cmp_2, ...]
cmp_residual_kv = [residual_1, residual_2, ...]

# 验证：
# 1. 参数维度正确
# 2. 参数值不越界
# 3. cmp_residual_kv正确（residual = ori % cmp_ratio）
```

**测试2：默认值兼容**
```python
# 不传入新参数，使用默认值
# 验证：兼容旧逻辑，结果一致
```

**测试3：部分参数传入**
```python
# 只传入seqused_cmp_kv，其他使用默认值
# 验证：cmp长度由参数决定，ori长度由计算得出
```

### 4.2 Mask边界测试

**测试4：cmp_mask_mode=3 + cmp_ratio≠1**
```python
# 场景：ori=10, cmp_ratio=4, residual=2
# 测试不同s1Idx行的mask边界

for s1Idx in range(actualS1Size):
    accessible_kv = actualS2Size - actualS1Size + s1Idx + 1
    if accessible_kv <= residual:
        threshold = 0
    else:
        threshold = min((accessible_kv - residual) / cmp_ratio, cmp_act_kv)
    
    # 验证：threshold不越界，不超过cmp_act_kv
```

**测试5：不同cmp_ratio**
```python
# 测试cmp_ratio=1, 4, 128的场景
# 验证：mask边界正确
```

### 4.3 性能测试

**测试6：大规模场景**
```python
# B=128, S1=4096, S2=8192, cmp_ratio=128
# 验证：
# 1. 计算正确性
# 2. 性能无明显下降（新参数传输开销）
```

---

## 5. 向后兼容性

### 5.1 兼容策略

```cpp
// 兼容规则：
// 1. 新参数可选，不传入时使用默认值
// 2. 原seqused_kv参数保留（建议改名：storage_len_kv）
// 3. cmp_ratio参数保留（但cmp长度可由参数覆盖）

// 默认值计算：
if (seqused_q == nullptr) {
    // 使用cu_seqlens_q或seqused_kv
}

if (seqused_cmp_kv == nullptr) {
    // 使用floor(seqused_ori_kv / cmp_ratio)
}

if (cmp_residual_kv == nullptr && cmp_mask_mode == 3 && cmp_ratio != 1) {
    // 计算：residual = seqused_ori_kv % cmp_ratio
}
```

### 5.2 推荐使用方式

**新用户**：
- 传入全部新参数，明确区分存储长度与真实使用长度
- cmp_mask_mode=3且cmp_ratio≠1时，传入cmp_residual_kv

**旧用户**：
- 不传入新参数，自动使用默认值
- 保持原有逻辑不变

---

## 6. 实施计划

### 6.1 阶段划分

**阶段1：参数定义与校验（2天）**
- op_host层新增参数定义
- 参数校验逻辑
- 兼容性处理

**阶段2：Kernel实现（3天）**
- ConstInfo新增字段
- InitGlobalBuffer参数传递
- GetSingleCoreParam参数读取
- CopyInKvSparse mask计算改进

**阶段3：测试验证（2天）**
- 基础功能测试
- Mask边界测试
- 性能测试
- 兼容性测试

**阶段4：文档与示例（1天）**
- README更新
- 测试用例示例
- API使用说明

### 6.2 风险点

**风险1：参数传输开销**
- 新增4个参数，kernel入口参数增加
- 解决：使用workspace传递，减少参数数量

**风险2：默认值计算准确性**
- 兼容逻辑需要保证计算正确
- 解决：充分测试，单元测试覆盖

**风险3：mask公式复杂性**
- 新公式逻辑复杂，容易出错
- 解决：充分验证，数学推导+代码测试

---

## 7. 附录

### 7.1 参数命名规范建议

```
存储长度（Storage）：
- storage_len_q（建议改名）
- storage_len_ori_kv（建议改名）
- storage_len_cmp_kv（建议改名）

真实使用长度（Actual Usage）：
- seqused_q
- seqused_ori_kv
- seqused_cmp_kv

其他：
- cmp_residual_kv
```

### 7.2 公式推导详解

**原公式问题分析**：
```
设：
- actualS2Size = ori_act_kv
- cmp_act_kv = floor(ori_act_kv / cmp_ratio)
- residual = ori_act_kv % cmp_ratio

第s1Idx行的mask threshold：
原公式：threshold = floor((actualS2Size - actualS1Size + s1Idx + 1) / cmp_ratio)

问题：
当 actualS2Size - actualS1Size + s1Idx + 1 接近 ori_act_kv 时：
threshold ≈ floor(ori_act_kv / cmp_ratio) = cmp_act_kv

但实际：
threshold可能超过cmp_act_kv，因为：
  floor((ori_act_kv) / cmp_ratio) = cmp_act_kv
  但 floor((ori_act_kv + 1) / cmp_ratio) 可能 = cmp_act_kv + 1 ❌

根本原因：
除法的余数在不同行会累积，导致threshold不一致
```

**新公式推导**：
```
几何视角：
ori部分：长度 = cmp_act_kv * cmp_ratio + residual
cmp部分：长度 = cmp_act_kv

对于第s1Idx行的Q：
可访问的KV token总数：accessible_kv = actualS2Size - actualS1Size + s1Idx + 1

判断访问范围：
1. 如果 accessible_kv <= residual：
   全在ori部分，cmp部分threshold = 0
   
2. 如果 accessible_kv > residual：
   部分在ori，部分在cmp
   cmp部分可访问的token数：
   threshold = floor((accessible_kv - residual) / cmp_ratio)
   
3. 确保不越界：
   threshold = min(threshold, cmp_act_kv)

验证：
示例1：ori=10, cmp_ratio=4, residual=2, cmp_act=2
第s1Idx=0行：accessible=11
  threshold = floor((11-2)/4) = floor(2.25) = 2 ✓

示例2：ori=10, cmp_ratio=4, residual=2, cmp_act=2
第s1Idx=1行：accessible=12
  threshold = floor((12-2)/4) = floor(2.5) = 2 ✓
  
原公式：threshold = floor(12/4) = 3 ✗（超过cmp_act=2）
```

### 7.3 代码示例

**Python调用示例**：
```python
import torch

# 场景：B=2, S1=8, S2_ori=10, S2_cmp=2, cmp_ratio=4
query = torch.randn(2, 1, 8, 128)
ori_kv = torch.randn(2, 1, 10, 128)
cmp_kv = torch.randn(2, 1, 2, 128)

# 新参数
seqused_q = torch.tensor([8, 8], dtype=torch.int32)
seqused_ori_kv = torch.tensor([10, 10], dtype=torch.int32)
seqused_cmp_kv = torch.tensor([2, 2], dtype=torch.int32)
cmp_residual_kv = torch.tensor([2, 2], dtype=torch.int32)  # 10 % 4 = 2

# 调用算子
output = torch_npu.npu_kv_quant_sparse_attn_sharedkv(
    query, ori_kv, cmp_kv, ...,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    cmp_mask_mode=3,
    cmp_ratio=4
)
```

---

**文档版本**：v1.0
**创建日期**：2026-05-11
**作者**：CANNBot