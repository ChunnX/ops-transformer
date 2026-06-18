# kv_quant_sparse_attn_sharedkv 参数扩展工作进展

**日期**: 2026-05-15
**最新提交ID**: 7ed0c046
**状态**: 已完成核心适配 + 多轮优化迭代

---

## 需求背景

### 问题描述
当前算子存在以下限制：
1. `seqused_kv`仅表示ori_kv的存储长度，无法区分存储长度与真实使用长度
2. cmp部分长度由`cmp_ratio`和ori部分长度计算得出，不够灵活
3. `cmp_mask_mode=3`且`cmp_ratio≠1`场景下，无法精确计算mask边界（存在余数问题）

### 解决方案
新增5个可选输入参数：
- `seqused_ori_kv`：ori_kv每个batch的真实使用长度
- `seqused_cmp_kv`：cmp_kv每个batch的真实使用长度
- `cmp_residual_kv`：cmp_mask_mode=3且cmp_ratio≠1时，每个batch的余数
- `cu_seqlens_ori_kv`：ori_kv的TND layout累加和
- `cu_seqlens_cmp_kv`：cmp_kv的TND layout累加和

---

## 完成工作

### ✅ 阶段1：Host端适配（已完成）

**修改文件**：
| 文件 | 修改内容 |
|------|---------|
| `kv_quant_sparse_attn_sharedkv_def.cpp` | 新增5个Optional Input定义 |
| `kv_quant_sparse_attn_sharedkv_check.h` | 新增参数名称常量、结构体字段、TilingInfo字段 |
| `kv_quant_sparse_attn_sharedkv_tiling.cpp` | 获取新参数tensor、解析shapeSize、传递给TilingInfo |
| `kv_quant_sparse_attn_sharedkv_check_single_para.cpp` | 新增CheckSingleParaCmpResidualKv校验函数 |

**关键修改点**：
```cpp
// def.cpp 新增参数定义
this->Input("cu_seqlens_ori_kv")    // Line 63
this->Input("cu_seqlens_cmp_kv")    // Line 68
this->Input("seqused_ori_kv")       // Line 83
this->Input("seqused_cmp_kv")       // Line 88
this->Input("cmp_residual_kv")      // Line 93

// check_single_para.cpp 校验逻辑
ge::graphStatus KvQuantSASTilingCheck::CheckSingleParaCmpResidualKv() const {
    if (*opParamInfo_.cmpMaskMode == 3 && *opParamInfo_.cmpRatio != 1) {
        OP_CHECK_IF(opParamInfo_.cmpResidualKv.tensor == nullptr,
            OP_LOGE(opName_, "cmp_residual_kv is required when cmp_mask_mode=3 and cmp_ratio!=1"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
```

### ✅ 阶段2：Kernel端适配（已完成）

**修改文件**：
| 文件 | 修改内容 |
|------|---------|
| `util_regbase.h` | ConstInfo新增actualLenDimsOriKV/CmpKV/cmpResidualKVSize字段 |
| `kv_quant_sparse_attn_sharedkv.cpp` | Kernel入口添加5个参数 |
| `kv_quant_sparse_attn_sharedkv_scfa_kernel.h` | Init/InitGlobalBuffer添加参数、新增5个GM地址成员 |
| `kv_quant_sparse_attn_sharedkv_scfa_block_vector.h` | InitVecBlock/InitGlobalBuffer添加参数 |
| `kv_quant_sparse_attn_sharedkv_kvcache.h` | GetSingleCoreParam新增参数获取逻辑、ComputeS2LoopInfo改进 |

**关键修改点**：

#### 1. 数据结构扩展
```cpp
// util_regbase.h ConstInfo新增字段
uint32_t actualLenDimsOriKV;   /* seqused_ori_kv的维度 */
uint32_t actualLenDimsCmpKV;   /* seqused_cmp_kv的维度 */
uint32_t cmpResidualKVSize;    /* cmp_residual_kv的长度 */

// kernel.cpp 参数传递
cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv
```

#### 2. 参数获取逻辑（GetSingleCoreParam）
```cpp
// kvcache.h 使用指针判空（已删除isActualLenDimsOriKVNull标记位）
actualS2OriSize = (actualSeqOriKvlenAddr == nullptr) ? constInfo.s2Size :
    (constInfo.actualLenDimsOriKV == actualSeqKVMin) ? actualSeqOriKvlenAddr[0] : actualSeqOriKvlenAddr[bIdx];

actualS2CmpSize = (actualSeqCmpKvlenAddr == nullptr) ? (actualS2OriSize / constInfo.cmpRatio) :
    (constInfo.actualLenDimsCmpKV == actualSeqKVMin) ? actualSeqCmpKvlenAddr[0] : actualSeqCmpKvlenAddr[bIdx];

runParam.cmpResidual = cmpResidualKvAddr[bIdx];
```

#### 3. Mask计算修正（ComputeS2LoopInfo）
```cpp
// kvcache.h:252-254 新公式（考虑余数）
runParam.s2LineCmpEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
    (runParam.cubeSOuterOffset + runParam.s1RealSize + runParam.cmpResidual) / constInfo.cmpRatio +
    runParam.nextTokensPerBatchCmp, 0, runParam.actualS2CmpSize);

// 限制不超过真实使用长度
runParam.s2CmpLineEndIdx = Min(runParam.s2LineCmpEndIdx, runParam.actualS2CmpSize);
```

### ✅ 阶段3：Git提交历史（已完成）

**关键提交记录**：
| 提交ID | 描述 |
|--------|------|
| 7ed0c046 | tnd layout支持 |
| da1b09f3 | tnd layout支持 |
| 105957e6 | 删除冗余变量isCmpResidualKVNull |
| b0298606 | 删除host侧isActualLenDimsOriKVNull/CmpKVNull标记位 |
| 31fefa92 | 删除isActualLenDimsOriKVNull/CmpKVNull标记位，使用指针判空 |
| ed73cb2c | 支持TND layout使用cuSeqlensOriKv和cuSeqlensCmpKv |
| 00796840 | 参考cuSeqlensQ用法适配cuSeqlensOriKv和cuSeqlensCmpKv |
| 26d699c0 | 增加cuSeqlensOriKvAddr和cuSeqlensCmpKvAddr参数 |

### ✅ 阶段4：优化迭代（已完成）

**优化内容**：
1. **删除冗余标记位**：用指针判空替代 `isActualLenDimsOriKVNull`/`isActualLenDimsCmpKVNull`/`isCmpResidualKVNull`
2. **新增cuSeqlens支持**：支持TND layout下ori_kv和cmp_kv的累加和参数
3. **TND layout适配**：完成TND layout的全链路支持

---

## 待完成工作

### ⏳ 阶段5：测试用例更新（待实现）

**当前状态**：pytest测试用例尚未更新新参数

需要在 `tests/pytest/kv_quant_sparse_attn_sharedkv_paramset.py` 中添加：
```python
# 新增参数配置
"seqused_ori_kv": [None],
"seqused_cmp_kv": [None],
"cmp_residual_kv": [None],
"cu_seqlens_ori_kv": [None],
"cu_seqlens_cmp_kv": [None],
```

**测试场景**：
1. **基础功能测试**：参数传入验证、默认值兼容、部分参数传入
2. **Mask边界测试**：cmp_mask_mode=3 + cmp_ratio≠1（验证余数公式）
3. **TND layout测试**：cu_seqlens_ori_kv/cu_seqlens_cmp_kv功能验证
4. **性能测试**：大规模场景（B=128, S1=4096, S2=8192）

### ⏳ 阶段6：参数校验增强（可选）

**当前状态**：仅有基础校验（cmp_residual_kv必传检查）

可增强校验规则：
```cpp
// 更完整的校验（可选实现）
CHECK(seqused_ori_kv[i] <= storage_ori_kv[i]);
CHECK(seqused_cmp_kv[i] <= storage_cmp_kv[i]);
CHECK(seqused_ori_kv[i] == seqused_cmp_kv[i] * cmp_ratio + cmp_residual_kv[i]);
```

---

## 向后兼容性

### 兼容策略
- 新参数可选，不传入时使用默认值
- 原seqused_kv/cu_seqlens_q参数保留
- cmp_ratio参数保留（但cmp长度可由参数覆盖）

### 默认值计算
```cpp
// 未传入seqused_ori_kv时
actualS2OriSize = constInfo.s2Size;  // 使用总长度

// 未传入seqused_cmp_kv时
actualS2CmpSize = actualS2OriSize / constInfo.cmpRatio;

// 未传入cmp_residual_kv时（当cmp_mask_mode=3且cmp_ratio≠1，需强制传入）
```

---

## 技术难点与解决

### 1. Mask余数问题

**问题描述**：
```
ori_act_kv = 10, cmp_ratio = 4
cmp_act_kv = floor(10 / 4) = 2
余数 = 2

原公式计算第s1Idx=1行：
threshold = floor(12 / 4) = 3  ❌ 超过cmp_act_kv=2，导致越界
```

**解决方案**：
在 `ComputeS2LoopInfo` 中使用新公式：
```cpp
(runParam.cubeSOuterOffset + runParam.s1RealSize + runParam.cmpResidual) / constInfo.cmpRatio
```
考虑余数，并限制不超过 `actualS2CmpSize`

### 2. 参数传递链路复杂性

**挑战**：5层传递链路（Host → Tiling → Kernel → Block → Kvcache）

**解决**：
- 逐层适配，确保每层正确传递和存储
- 使用指针判空简化逻辑，删除冗余标记位

### 3. TND layout适配

**挑战**：需要支持累加和形式（cuSeqlens）

**解决**：
- 新增 `cu_seqlens_ori_kv` 和 `cu_seqlens_cmp_kv` 参数
- 参考 `cu_seqlens_q` 的实现方式

---

## 参考文档

- 设计文档：`docs/design_actual_seq_enhancement.md`
- 代码修改点：`docs/code_modification_points.md`
- Python调用示例：设计文档第7.3节

---

## 下一步计划

1. ✅ 已完成核心适配
2. ✅ 已完成优化迭代（删除冗余变量、新增cuSeqlens、TND支持）
3. ⏳ 待完成：更新测试用例（pytest参数配置）
4. ⏳ 可选：增强参数校验逻辑
5. ⏳ 待完成：更新算子README文档

---

**备注**: 核心适配已完成并通过多次迭代优化，测试用例更新为后续工作。