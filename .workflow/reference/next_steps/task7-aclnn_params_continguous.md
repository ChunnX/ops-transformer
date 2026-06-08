# op_api Aclnn代码实现检查和修复

> 你应当在必要时候使用subagent来完成任务，提供一个独立、聚焦的上下文来完成具体的任务，并且为其提供必要的知识或信息。

当前qkv_rms_norm_rope_cache_with_kscale算子的aclnn接口描述中对参数k_cache, v_cache, k_scale_cache的连续性没有清晰定义，需要根据任务说明进行详细检查。

## 任务说明

当前算子设计文档中对k_cache, v_cache, k_scale_cache三个参数的没有明确说明连续性问题，然而实际上这三个参数不是连续。在设计文档中写到：

| 输入 | dtype | shape | 说明 |
|------|-------|-------|------|
| `k_cache` | `float8_e4m3fn` | `[BlockNum, Nk, BlockSize, D]` | K cache，输入输出复用空间 |
| `v_cache` | `float8_e4m3fn` | `[BlockNum, Nv, BlockSize, D]` | V cache，输入输出复用空间 |
| `k_scale_cache` | `float32` | `[BlockNum, Nk, BlockSize, 1]` | K scale cache，输入输出复用空间 |

然而，这三个参数都是非连续的tensor，除了尾轴D或者1是连续的，其他轴都可能不是连续的，因此他们的stride可能不是连续的ND格式的stride。

现在的问题是针对这种非连续的tensor，需要在aclnn接口中保持他们的非连续性，保证原始的stride不要被误修改，下层的tiling（本任务中不关注该实现）部分会获取相关的stride信息来保证写出位置的正确性，否则可能会导致写出位置错误。请你帮我调研已有代码，检查这种本身就需要stride的tensor应当如何处理，看看是否会被误修改tensor的连续性导致后续调用出错。

## 任务计划

- [x] 调研scatter_pa_kv_cache/gather_pa_kv_cache的非连续tensor处理模式
- [x] 分析AutoContiguous()对非连续cache tensor的影响
- [x] 移除def.cpp中cache参数的AutoContiguous()
- [x] 确认L2中不对cache参数调用Contiguous()
- [x] 编译验证 + 57/57测试通过

## 任务完成情况

### 问题分析

k_cache/v_cache/k_scale_cache是非连续tensor（仅尾轴D或1连续），其stride不是标准ND格式。如果对这些参数调用AutoContiguous()或Contiguous()，会：
1. 创建连续副本，丢失原始stride信息
2. 下层tiling无法获取正确stride，导致写出位置错误
3. 额外内存分配和拷贝开销

### 变更内容

**def.cpp**: 移除k_cache/v_cache/k_scale_cache的`.AutoContiguous()`
```cpp
// Before
this->Input("k_cache")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT8_E4M3FN})
    .FormatList({ge::FORMAT_ND})
    .AutoContiguous();  // ← 移除此行

// After
this->Input("k_cache")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT8_E4M3FN})
    .FormatList({ge::FORMAT_ND});
```

**L2 (aclnn_*.cpp)**: 已正确不对cache参数调用Contiguous()
- `Contiguous()`函数仅处理9个非cache输入：qkv, qGamma, kGamma, cosSin, slotMapping, cuSeqLen, optionalQRotation, optionalKRotation, optionalVScale
- kCache, vCache, kScaleCache直接传给L0，保留原始stride

### 参考模式

scatter_pa_kv_cache提供更复杂的两路处理：
- `ProcessNonContiguous()`: 使用`CreateView`保留stride，kernel直接写入非连续内存
- `ProcessContiguous()`: 使用`Contiguous()`+`ViewCopy`回写

当前算子采用简化模式（不Contiguous也不CreateView），因为L2已有AutoContiguous机制保护非cache输入，cache参数直接透传。

### 二次验证结果（RECHECK）：确认无需修改

**对比算子**：scatter_pa_kv_cache、gather_pa_kv_cache、quant_grouped_matmul_inplace_add

**def.cpp对比**：
- scatter_pa_kv_cache：所有输入/输出均**无**AutoContiguous()
- gather_pa_kv_cache：所有输入/输出均**无**AutoContiguous()
- quant_grouped_matmul_inplace_add：所有输入/输出均**无**AutoContiguous()
- 本算子：cache参数**无**AutoContiguous()，非cache输入**有**AutoContiguous() → cache处理一致，非cache的AutoContiguous()提供额外安全保护（与L2 Contiguous()双保险）

**L2对比**：
- scatter/gather：cache使用CreateView（ProcessNonContiguous路径）或Contiguous+ViewCopy（ProcessContiguous路径）
- quant_grouped_matmul_inplace_add：yRef不调用Contiguous，L0直接写入，之后ViewCopy回写
- 本算子：cache参数不调用Contiguous，直接传给L0 → kernel通过原始stride直接写入正确位置 → 无需ViewCopy

**结论**：当前实现正确。cache参数的非连续stride通过def.cpp（无AutoContiguous）和L2（无Contiguous）双重保证，完整传递给L0 kernel。