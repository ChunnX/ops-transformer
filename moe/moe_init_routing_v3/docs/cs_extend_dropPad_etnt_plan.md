# A5 CountingSort 扩展支持 drop_pad_mode=1 / expert_tokens_num_type=0/2 实现方案

> **目标**：参考 A3 对 `drop_pad_mode=1` 和 `expert_tokens_num_type`（COUNT/CUMSUM/KEY_VALUE）的支持，在 A5 CountingSort 模板中放开 tiling 约束并补齐缺失的 kernel 分支，实现同等覆盖。

---

## 1. A3 支持情况调研结论

### 1.1 A3 tiling 侧约束（`ShouldEnableCountingSortByShape`）

位置：`moeInitrouteV3/.../op_host/moe_init_routing_v3_tiling.cpp` L809-819

```cpp
bool ShouldEnableCountingSortByShape() const
{
    int64_t actualExpertNum = expertEnd_ - expertStart_;
    bool expertNumValid = expertNum_ <= 1024;
    bool dropPadValid =
        (dropPadMode_ == DROP_PAD) || (dropPadMode_ == DROP_LESS && actualExpertNum * NUM_TWO <= expertNum_);
    bool quantValid = quantMode_ != DYNAMIC_QUANT;
    bool excludeNonDropPadSmallN = !(n_ <= 2560 && dropPadMode_ == DROP_LESS && quantMode_ == STATIC_QUANT);
    bool excludeDropPadTinyN = !(dropPadMode_ == DROP_PAD && n_ < 512);
    return expertNumValid && dropPadValid && quantValid && excludeNonDropPadSmallN && excludeDropPadTinyN;
}
```

**A3 支持矩阵**：

| 参数 | A3 支持 | 约束 |
|------|---------|------|
| **drop_pad_mode** | **0/1 均支持** | dropPad(=1) 时 n≥512 |
| **expert_tokens_num_type** | **0/1/2 均支持** | tiling 无显式拦截（kernel 三分支完整） |
| **quant_mode** | -1/0（排除 1） | 动态量化全局关闭 |

### 1.2 A3 kernel 侧实现

**CutOrigin（`moe_v3_cut_origin_t.h`）**：

- **Phase B `WriteExpertTokens`**（L667-713）：完整支持 COUNT/CUMSUM/KEY_VALUE 三模式
  - COUNT: `Cast(int32→int64) + DataCopyPad`
  - CUMSUM: 标量循环累加 + DataCopyPad
  - KEY_VALUE: 构造 (expert_idx, count) 紧凑对 + DataCopyPad
- **Phase C `GatherAndWriteDroppad`**（L1195-1532）：完整 dropPad 实现
  - slot 溢出判定：`newPos < expertCapEnd` 决定保留/丢弃
  - 丢弃行 rowIdx 写 -1
  - 双缓冲 xBuf ping-pong（unquant + static quant 两条路径）
  - dropPad 专属 Phase C 末尾零填充（L1475-1531 `GatherAndWriteDroppad` 后段）

**FullLoad（`moe_v3_full_load_cut_origin_t.h`）**：

- **Phase B `WriteExpertTokens`**（L469-478）：仅直接写出 `actualExpertNum` 个 int64（依赖上层 Phase B 已按模式填好）
- **FullLoad 限定**：A3 `ComputeCountingSortTiling` L847-852 FullLoad 判定要求 `dropPadMode_ == DROP_LESS`，FullLoad 不支持 dropPad

### 1.3 A3 关键约束总结

| 模板 | drop_pad_mode | expert_tokens_num_type | quant_mode |
|------|---------------|------------------------|------------|
| **A3 FullLoad** | 仅 0（dropless） | 仅 1（COUNT） | -1/0 |
| **A3 CutOrigin** | **0/1 均支持** | **0/1/2 均支持** | -1/0 |

---

## 2. A5 现状调研结论

### 2.1 A5 tiling 侧约束

位置：`op_host/moe_init_routing_v3_tiling_arch35.cpp` L2269-2272

```cpp
if (dropPadMode_ != DROP_PAD_MODE_DROPLESS) {
    return false;
}
if (expertTokensNumType_ != EXPERT_TOKENS_TYPE_COUNT) {
    return false;
}
```

### 2.2 A5 kernel 侧实现现状（关键发现）

#### 2.2.1 expert_tokens_num_type：Phase B **已完整支持三模式**

位置：`op_kernel/arch35/moe_v3_counting_sort_cut_origin_base.h` L648-694

```cpp
// ComputeGlobalOffset L648-679：expertTokens 由 core 0 统一写出，支持 COUNT / CUMSUM / KEY_VALUE 三种模式
if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
    if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        // KEY_VALUE: 紧凑 (expert_idx, count) 对，跳过 count==0 的专家
        if (totalForExpert != 0) {
            expertTokensLocal.SetValue(keyValueOffset * EXPERT_ID_VALUE_NUM, e + expertStart_);
            expertTokensLocal.SetValue(keyValueOffset * EXPERT_ID_VALUE_NUM + 1, totalForExpert);
            keyValueOffset++;
        }
    } else if (expertTokensNumType_ == EXERPT_TOKENS_COUNT) {
        expertTokensLocal.SetValue(e, totalForExpert);  // COUNT: 直接写
    } else {
        expertTokensLocal.SetValue(e, cumulativeSum + totalForExpert);  // CUMSUM: 累加
    }
}

// WriteExpertTokens L683-694：动态 expertCountElements_ 适配 KEY_VALUE 变长输出
DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
```

**结论**：A5 kernel Phase B 完整支持三种模式，**只需放开 tiling 约束**。

#### 2.2.2 drop_pad_mode=1：**非量化已完整支持，静态量化未实现**

**非量化 CutOrigin（已完整支持）**：

| 函数 | 位置 | dropPad 实现 |
|------|------|-------------|
| `Init` | unquantized.h L61-73 | `ZeroOutExpandedDropPad(expandedXGm_)` + scale 透传空槽清零 |
| `GatherOneRow` | unquantized.h L98-100 | `if (newPos >= outputRows_) return;` 越界保护 |
| `GatherAndWrite` | unquantized.h L141-155 | dropPad slot 溢出判定 + expertCapacity newPos 计算 |
| `GatherAndWriteDoubleBuffer` | unquantized.h L209-216 | dropPad slot 溢出 + skipArr 标记 + 批量写 -1 |
| `ComputeGlobalOffset` | base.h L669-673 | dropPad seed = per-expert 跨核前缀（不含 cumulativeSum） |

**静态量化 CutOrigin（未实现 dropPad）**：

| 函数 | 位置 | 现状 |
|------|------|------|
| `Init` | static_quant.h | ❌ 无 `ZeroOutExpandedDropPad` 调用 |
| `GatherAndWrite` | static_quant.h L181-182 | ❌ 注释明确 `// 静态量化非 dropPad：newPos 即全局 slot` |
| `GatherAndWriteDoubleBuffer` | static_quant.h L262 | ❌ 注释明确 `newPosArr[i] = slot; // 静态量化非 dropPad` |

**结论**：
- 非量化 CutOrigin：**只需放开 tiling 约束**
- 静态量化 CutOrigin：**需补 dropPad 分支**（参照非量化实现）

---

## 3. A5 扩展实现方案

### 3.1 方案 A：expert_tokens_num_type 扩展（低风险，P0）

**改动量**：tiling 1 处删除 + 验证

**步骤**：

1. **tiling 侧**：删除 `IsCountingSortApplicable` L2270-2272 的 expertTokensNumType 拦截
   ```cpp
   // 删除：
   if (expertTokensNumType_ != EXPERT_TOKENS_TYPE_COUNT) {
       return false;
   }
   ```

2. **kernel 侧**：无改动（Phase B 已完整支持三模式）

3. **UB 预算校验**：`EstimateArch35CountingSortFullLoadUB` / `ComputeArch35CountingSortCutOriginTiling` 需确认 expertTokensLocal 的 UB 预算是否覆盖 KEY_VALUE 模式（上界 `actualExpertNum * 2` 个 int64）。
   - 现状（base.h L202）：`expertTokensBufElems_ = (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) ? actualExpertNum_ * EXPERT_ID_VALUE_NUM : actualExpertNum_;`
   - tiling 侧 L2414 已按上界 2 倍恒计：`Ops::Base::CeilAlign((expertEnd_ - expertStart_) * NUM_TWO * sizeof(int64_t), UB_BLOCK_SIZE)`
   - **结论：UB 预算已覆盖，无需改动**

4. **验证**：补 cumsum / KEY_VALUE 测试用例，跑 39+2 回归

### 3.2 方案 B：drop_pad_mode=1 扩展（中等风险，P1）

#### 3.2.1 非量化 CutOrigin（低风险）

**改动量**：tiling 1 处删除

1. **tiling 侧**：删除 `IsCountingSortApplicable` L2269-2270 的 dropPadMode 拦截，改为 A3 的约束逻辑
   ```cpp
   // 替换为（参照 A3 L813-814 + L817）：
   if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
       if (n_ < 512LL) { return false; }  // A3 excludeDropPadTinyN
   }
   // dropless 保留原有 actualExpertNum*2 <= expertNum 约束
   ```

2. **kernel 侧**：无改动（非量化已完整支持 dropPad）

3. **验证**：补 dropPad 测试用例

#### 3.2.2 静态量化 CutOrigin（需补 kernel 分支）

**改动量**：static_quant.h 3 处补 dropPad 分支

1. **Init 补 `ZeroOutExpandedDropPad`**（参照 unquantized.h L59-73）：
   ```cpp
   // static_quant.h Init() 中 PrefillAndSync() 之后追加：
   if (this->rowIdxType_ == GATHER && this->dropPadMode_ == DROP_PAD_MODE) {
       this->ZeroOutExpandedDropPad(expandedXGm_);  // 注意：static quant expandedX 是 int8
   }
   ```
   **注意**：`ZeroOutExpandedDropPad` 需确认是否支持 int8 类型（static quant 的 expandedXGm_ 是 `GlobalTensor<int8_t>`）。如不支持，需补 int8 特化版本。

2. **GatherAndWrite 补 dropPad 分支**（参照 unquantized.h L141-155）：
   ```cpp
   // static_quant.h GatherAndWrite() 中替换 L181-183：
   int64_t newPos;
   if (this->dropPadMode_ == DROP_PAD_MODE) {
       if (slot >= this->expertCapacity_) {
           // dropPad 溢出：写 -1 rowIdx，跳过 expandedX
           idxBuf.SetValue(0, -1);
           DataCopyPad(this->expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
           continue;
       }
       newPos = (expertOffset + this->expertStart_) * this->expertCapacity_ + slot;
   } else {
       newPos = slot;
   }
   GatherOneRow(newPos, tokenRow, origFlatIdx);
   ```

3. **GatherAndWriteDoubleBuffer 补 dropPad 分支**（参照 unquantized.h L209-221）：
   ```cpp
   // static_quant.h GatherAndWriteDoubleBuffer() L262 替换：
   if (this->dropPadMode_ == DROP_PAD_MODE) {
       if (slot >= this->expertCapacity_) {
           skipArr[i] = true;
           newPosArr[i] = -1;
           continue;
       }
       newPosArr[i] = (expertOffset + this->expertStart_) * this->expertCapacity_ + slot;
   } else {
       newPosArr[i] = slot;
   }
   skipArr[i] = false;
   ```

4. **tiling UB 预算校验**：`ComputeArch35CountingSortCutOriginTiling` 需确认 dropPad 场景的 `outputRows_` 语义变化（dropPad = `expertNum * expertCapacity`，dropless = `activeNum`），可能影响 `expertTokensLocal` UB 预算。当前 L2414 已按 `actualExpertNum * 2` 上界恒计，应已覆盖。

---

## 4. 实施优先级与风险

| 方案 | 优先级 | 改动量 | 风险 | 验证复杂度 |
|------|--------|--------|------|-----------|
| **A: expert_tokens_num_type** | P0 | tiling 1 处删除 | 低（kernel 已支持） | 中（补 2 条测试用例） |
| **B1: 非量化 dropPad** | P1 | tiling 1 处删除 | 低（kernel 已支持） | 中（补 1 条测试用例） |
| **B2: 静态量化 dropPad** | P2 | kernel 3 处补分支 | 中（需精度验证） | 高（ZeroOutExpandedDropPad int8 适配 + 精度回归） |

**建议实施顺序**：A → B1 → B2

---

## 5. 测试用例需求

### 5.1 expert_tokens_num_type 扩展用例

| 用例 ID | quant_mode | drop_pad_mode | expert_tokens_num_type | row_idx_type | N | cols | actualExpertNum |
|---------|-----------|---------------|------------------------|-------------|---|------|-----------------|
| A5_etnt_cumsum | -1 | 0 | **0 (cumsum)** | 0 | 9000 | 7168 | 256 |
| A5_etnt_key_value | -1 | 0 | **2 (KEY_VALUE)** | 0 | 9000 | 7168 | 256 |

### 5.2 drop_pad_mode 扩展用例

| 用例 ID | quant_mode | drop_pad_mode | expert_tokens_num_type | row_idx_type | N | cols | expertNum | expertCapacity |
|---------|-----------|---------------|------------------------|-------------|---|------|-----------|----------------|
| A5_droppad_unquant | -1 | **1 (dropPad)** | 1 (COUNT) | 0 | 9000 | 7168 | 256 | 64 |
| A5_droppad_static | 0 | **1 (dropPad)** | 1 (COUNT) | 0 | 9000 | 7168 | 256 | 64 |

**注意**：dropPad 用例的 `expert_capacity` 必须显式指定（决定 outputRows = expertNum × expertCapacity）。

---

## 6. A3 vs A5 FullLoad 限制对比

| 维度 | A3 FullLoad | A5 FullLoad |
|------|-------------|-------------|
| drop_pad_mode | 仅 0 | 仅 0（本次不改） |
| expert_tokens_num_type | 仅 1 | 仅 1（本次不改） |
| **原因** | FullLoad UB 全载模型与 dropPad 的 expertCapacity 固定槽位语义不匹配；CUMSUM/KEY_VALUE 需额外标量循环，FullLoad 的 Phase B 已被精简 | 同 A3 |

**本次扩展范围**：仅 CutOrigin 模板（FullLoad 保持现有约束）。

---

## 7. 附录：关键代码位置速查

### A5 kernel（`op_kernel/arch35/`）

| 文件 | 函数 | 行号 | 说明 |
|------|------|------|------|
| base.h | `ComputeGlobalOffset` | 607-680 | Phase B，支持三模式 expertTokens + dropPad seed |
| base.h | `WriteExpertTokens` | 683-694 | 动态 expertCountElements_ 适配 KEY_VALUE 变长 |
| unquantized.h | `Init` | 59-80 | dropPad zeroOut + scale 清零 |
| unquantized.h | `GatherAndWrite` | 123-163 | dropPad 分支已实现 |
| unquantized.h | `GatherAndWriteDoubleBuffer` | 167-335 | dropPad skipArr + 批量 -1 写出 |
| static_quant.h | `GatherAndWrite` | 164-186 | ❌ 缺 dropPad 分支 |
| static_quant.h | `GatherAndWriteDoubleBuffer` | 223-365 | ❌ 缺 dropPad 分支 |

### A5 tiling（`op_host/moe_init_routing_v3_tiling_arch35.cpp`）

| 函数 | 行号 | 说明 |
|------|------|------|
| `IsCountingSortApplicable` | 2264-2304 | dropPadMode + expertTokensNumType 拦截（待放开） |
| `ComputeArch35CountingSortCutOriginTiling` | 2388-2454 | UB 预算（expertTokens 上界已恒计） |

---

**文档版本**：v1.0
**生成日期**：2026-08-07
**关联文档**：`docs/cs_template_support_matrix.md` §2.2.1/2.2.2
