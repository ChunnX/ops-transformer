# MoeInitRoutingV3 CountingSort 模板支持矩阵（A3/A5 对比）

> **文档目标**：完整梳理 A3 和 A5 两个 CountingSort 性能模板（FullLoad / CutOrigin）支持的数据类型和参数组合，为测试用例设计和跨版本迁移提供依据。

---

## 1. 模板概述

### 1.1 两个模板

| 模板 | 触发条件（A5） | 主要特征 |
|------|---------------|----------|
| **FullLoad** | `isFullload_=true` 或 EstimateArch35CountingSortFullLoadUB ≤ availUbSize | X 全载入 UB，Phase C 仅 UB→GM 写出 |
| **CutOrigin** | FullLoad UB 预算不足 | X 按需流式载入，Phase C GM→UB→GM（已引入双缓冲优化） |

### 1.2 三相流程（两模板共用）

1. **Phase A**（核间并行）：Filter + Count（向量过滤 + 计数）
2. **Phase B**（核间归约）：全局偏移计算
3. **Phase C**（并行搬运）：Gather + 写 expandedX / expandedRowIdx

---

## 2. A5 支持矩阵（arch35，当前算子仓）

### 2.1 数据类型（kernel 侧 `constexpr` 守卫）

位置：`op_kernel/moe_init_routing_v3_apt.cpp` L348-425

| 量化模式 | X 支持类型 | expandedX 类型 | 备注 |
|----------|-----------|---------------|------|
| 非量化（quant_mode=-1） | bf16, fp16, fp32 | 与 X 相同 | 纯浮点透传 |
| 静态量化（quant_mode=0） | bf16, fp16, fp32 | int8 | scale/offset 必输入 |
| **动态量化（quant_mode=1）** | bf16, fp16, fp32 | int8 | **tiling 全局关闭**（见 §2.2） |

### 2.2 参数组合约束（tiling 侧 `IsCountingSortApplicable`）

位置：`op_host/moe_init_routing_v3_tiling_arch35.cpp` L2264-2293

| 参数 | 约束 | 常量 / 说明 |
|------|------|------------|
| **N（token 数）** | [256, 1M] | `CS_MIN_N=256`, `CS_MAX_N=1024*1024` |
| **actualExpertNum** | (0, 256] | `CS_MAX_ACTUAL_EXPERT_NUM=256`（expertEnd - expertStart） |
| **drop_pad_mode** | 0（dropless） | 必须 dropless，dropPad(=1) **tiling 主动禁用**（见 §2.2.1） |
| **expert_tokens_num_type** | 1（COUNT） | 必须 COUNT，cumsum(=0)/none(=2) **tiling 主动禁用**（见 §2.2.2） |
| **quant_mode** | -1/0 **（排除 1）** | **动态量化全局关闭**（见 §2.2.3） |
| **row_idx_type** | 0/1（均支持） | 0=GATHER, 1=SCATTER（tiling 无显式拦截，kernel 实现支持） |
| **expertNum / LE** | ≤1024 | LE 收益门槛：actualExpertNum!=expertNum 且 actualExpertNum*2>expertNum |

#### 2.2.1 drop_pad_mode 约束根因

**A5 限定**：`drop_pad_mode` 必须为 0（dropless），tiling 拦截 1（dropPad）。

位置：`op_host/moe_init_routing_v3_tiling_arch35.cpp` L2269-2272

```cpp
if (dropPadMode_ != DROP_PAD_MODE_DROPLESS) {
    return false;
}
```

**技术根因**：
- **kernel 理论支持**：base.h L183/L669、unquantized.h L61 等有 `DROP_PAD_MODE` 分支逻辑，kernel 侧具备 dropPad 处理能力。
- **tiling 主动禁用**：A5 CountingSort 模板设计聚焦于 **dropless + LE（actualExpertNum*2 > expertNum）场景的性能优化**，dropPad 场景的 Phase C 搬运策略（预填 -1 rowIdx、expertCapacity 固定槽位）与 A5 双缓冲 ping-pong 优化的批量预计算逻辑（newPos/srcRow/origFlatIdx 数组）不匹配——dropPad 下 slot 溢出丢弃需逐 pair 判定、写 -1 标记，打断批量流水。
- **A3 支持 dropPad**：A3 `ShouldEnableCountingSortByShape` L813-814 包含 `dropPadMode_ == DROP_PAD` 分支，说明 A3 CS 模板覆盖了 dropPad 场景。A5 迁移时聚焦 dropless 优化未同步 dropPad 分支。

**适用性影响**：现有用例（ti=1/19/20/27/29）全部 `drop_pad_mode=0`，dropPad 场景回退旧分阶段路径（TilingKey 10xxxxxx/11xxxxxx）。

#### 2.2.2 expert_tokens_num_type 约束根因

**A5 限定**：`expert_tokens_num_type` 必须为 1（COUNT），tiling 拦截 0（cumsum）和 2（none）。

位置：`op_host/moe_init_routing_v3_tiling_arch35.cpp` L2270-2272

```cpp
if (expertTokensNumType_ != EXPERT_TOKENS_TYPE_COUNT) {
    return false;
}
```

**技术根因**：
- **kernel 理论支持**：base.h L662-669 `WriteExpertTokens()` 有 KEY_VALUE(=0) / COUNT(=1) / CUMSUM(=2) 三分支逻辑，kernel 侧具备三种模式处理能力。
- **tiling 主动禁用**：A5 CountingSort 模板的 Phase B 全局偏移计算（`ComputeGlobalOffset`）与后续 Phase C 搬运紧密耦合 **COUNT 模式**的 `expertCountLocal` 语义（每专家当前累计 token 数 + slot 偏移）。cumsum（前缀和）和 none（不输出 expertTokens）需要不同的 offset 初始化和更新逻辑，A5 模板未实现这两个分支的适配代码。
- **A3 可能支持多模式**：需进一步确认 A3 `ComputeCountingSortTiling` 是否对 expertTokensNumType 有显式拦截。从 kernel `WriteExpertTokens()` 三分支看，A3 至少 kernel 侧支持三种。

**适用性影响**：现有用例全部 `expert_tokens_num_type=1`，cumsum/none 场景回退旧路径。

#### 2.2.3 动态量化（quant_mode=1）关闭根因

**A5 全局关闭**：动态量化（含 int8/int4）完全禁用 CountingSort。

位置：`op_host/moe_init_routing_v3_tiling_arch35.cpp` L2274-2282

```cpp
// 动态量化场景不支持 CountingSort（含 int8/int4）：
// 1) 与 A3 ShouldEnableCountingSortByShape 对齐（quantValid = quantMode_ != DYNAMIC_QUANT，动态量化走
//    sort + per-token absmax 分段路径）；
// 2) A5 实测 CutOrigin 动态量化（ti=20 9000×7168，smooth 输入）即使重建双缓冲仍 700µs vs 原路径 178µs
//    (+292%)——瓶颈在 per-token absmax 两趟 VF 的串行链，DMA 重叠只是副项，双缓冲无法追平。
//    关闭后 ti=20/37 回落到原代码路径（~178µs）。
if (IsAnyDynamicQuantCase()) {
    return false;
}
```

**技术根因**（已在 `docs/perf_baseline_vs_new_template.md` §7.2 详细记录）：

1. **A3 同样关闭**：A3 `ShouldEnableCountingSortByShape` L815 `quantValid = quantMode_ != DYNAMIC_QUANT`，动态量化在 A3 上从不走 CS（走旧 sort + per-token absmax 分段路径）。
2. **A5 双缓冲无效**：A5 实测（ti=20 cutorigin_dynamic_quant）即使重建 Phase C 双缓冲 ping-pong 仍 700µs vs 原路径 178µs（+292%）。根因：动态量化 Phase C `QuantDynamicOneRow` 需 per-token **absmax 两趟 VF**（smooth 预乘写回 + absmax、Div+Cast f32→f16→int8），是**纯计算串行链**（Vector 流水瓶颈），DMA 重叠只是副项，双缓冲 ping-pong 无法追平。
3. **关闭效果**：关闭后 ti=20/37 等 8 个动态量化用例全部回落 TilingKey=11020000（原分阶段路径），耗时 ~175-178µs，追平原代码（-0.3%）。

**适用性影响**：现有 `ti=20`（cutorigin_dynamic_quant）已回落原路径，不走 CS 模板；`ti=18`（fullload_dynamic_quant）同理回落 11020000。动态量化用例不生成 CS 测试覆盖。

### 2.3 FullLoad vs CutOrigin 分派逻辑

位置：`op_host/moe_init_routing_v3_tiling_arch35.cpp` L556-577

```
DoOpTiling():
  isFullload_ = IsFullLoad()  // dropPadMode=0 + totalLength ≤ sortLoopMaxElement + UB 预算

  if !isFullload_ && IsCountingSortApplicable():
      perCoreTokensEst = Ceil(n_, aivCoreNum_)
      estUb = EstimateArch35CountingSortFullLoadUB(perCoreTokensEst)
      if estUb > 0 && estUb <= availUbSize_ - 1024:
          countingSortMode_ = 1  // FullLoad
          ComputeArch35CountingSortFullLoadTiling()
      else:
          countingSortMode_ = 2  // CutOrigin
          ComputeArch35CountingSortCutOriginTiling()
```

CutOrigin 内部还有**双缓冲降级 fallback**（L2446-2453）：
- 若 `serialBytes + dbExtraBytes + 1024 > availUbSize_`：`csDbEnable=0`（降级串行 CutOrigin，仍优于回退旧路径）
- 若串行也装不下：`countingSortMode=0`（完全回退旧分阶段路径）

### 2.4 TilingKey 编码

位置：`op_host/moe_init_routing_v3_tiling_arch35.cpp` L615-629

```
base = {
  FullLoad:  400000(unquant) / 410000(static) / 420000(dynamic)
  CutOrigin: 500000(unquant) / 510000(static) / 520000(dynamic)
}
TilingKey = base + rowIdxType×1 + dropPadMode×10
```

实际值示例（dropPadMode=0 场景）：
- 400000（FullLoad unquant GATHER）、400001（FullLoad unquant SCATTER）
- 500000（CutOrigin unquant GATHER）、500001（CutOrigin unquant SCATTER）
- 510000（CutOrigin static GATHER）、510001（CutOrigin static SCATTER）

---

## 3. A3 支持矩阵（参照，迁移源）

### 3.1 数据类型（kernel 侧模板实例化）

位置：`moeInitrouteV3/.../op_kernel/moe_init_routing_v3.cpp`（A3 单文件，含 FullLoad/CutOrigin）

| 量化模式 | X 支持类型 | expandedX 类型 | 备注 |
|----------|-----------|---------------|------|
| 非量化（UN_QUANT=-1） | bf16, fp16, fp32 | 与 X 相同 | 同 A5 |
| 静态量化（STATIC_QUANT=0） | bf16, fp16, fp32 | int8 | 同 A5 |
| **动态量化（DYNAMIC_QUANT=1）** | bf16, fp16, fp32 | int8 | **tiling 全局关闭**（见 §3.2） |

### 3.2 参数组合约束（tiling 侧 `ShouldEnableCountingSortByShape`）

位置：`moeInitrouteV3/.../op_host/moe_init_routing_v3_tiling.cpp` L809-819

```cpp
bool ShouldEnableCountingSortByShape() const
{
    int64_t actualExpertNum = expertEnd_ - expertStart_;
    bool expertNumValid = expertNum_ <= 1024;
    bool dropPadValid =
        (dropPadMode_ == DROP_PAD) || (dropPadMode_ == DROP_LESS && actualExpertNum * 2 <= expertNum_);
    bool quantValid = quantMode_ != DYNAMIC_QUANT;  // ← 动态量化全局关闭
    bool excludeNonDropPadSmallN = !(n_ <= 2560 && dropPadMode_ == DROP_LESS && quantMode_ == STATIC_QUANT);
    bool excludeDropPadTinyN = !(dropPadMode_ == DROP_PAD && n_ < 512);
    return expertNumValid && dropPadValid && quantValid && excludeNonDropPadSmallN && excludeDropPadTinyN;
}
```

| 参数 | 约束 | 说明 |
|------|------|------|
| **N** | [COUNTING_SORT_MIN_N, COUNTING_SORT_MAX_N] | 需查 A3 常量定义（A5 为 [256, 1M]） |
| **expertNum** | ≤1024 | 与 A5 一致 |
| **actualExpertNum** | 无显式上限 | 由 dropPadValid 间接约束（A5 显式 ≤256） |
| **drop_pad_mode** | 0（DROP_LESS）或 1（DROP_PAD） | **A5 仅支持 0** |
| **quant_mode** | -1/0 **（排除 1）** | **动态量化全局关闭** |
| **row_idx_type** | 0/1（均支持） | 与 A5 一致 |
| **expert_tokens_num_type** | 需查 A3 判定逻辑 | A5 限定 COUNT(=1) |

**关键差异总结**：

| 维度 | A3 | A5 |
|------|----|----|
| **动态量化** | 全局关闭（L815 `quantMode_ != DYNAMIC_QUANT`） | 全局关闭（L2274-2282 `IsAnyDynamicQuantCase()`） |
| **actualExpertNum 上限** | 无显式上限（expertNum≤1024 + dropPadValid） | **显式 ≤256**（`CS_MAX_ACTUAL_EXPERT_NUM`） |
| **drop_pad_mode** | **0/1 都支持**（L813-814 分支逻辑） | **仅支持 0**（L2269-2272 拦截 1） |
| **expert_tokens_num_type** | 需查 A3（可能无显式拦截） | **仅支持 1（COUNT）**（L2271 拦截 0/2） |

---

## 4. 完整参数组合列表（A5 测试覆盖目标）

### 4.1 非量化（quant_mode=-1）

**基线组合**（必测，覆盖 row_idx_type×drop_pad_mode 二维）：

| 组合 ID | row_idx_type | drop_pad_mode | expert_tokens_num_type | N | cols | actualExpertNum | 备注 |
|---------|-------------|---------------|------------------------|---|------|-----------------|------|
| A5_unquant_01 | 0 (GATHER) | 0 (dropless) | 1 (COUNT) | 9000 | 7168 | 256 | CutOrigin 典型（已有 ti=1） |
| A5_unquant_02 | 1 (SCATTER) | 0 (dropless) | 1 (COUNT) | 9000 | 7168 | 256 | row_idx 变体 |
| ~~A5_unquant_03~~ | 0 | **1 (dropPad)** | 1 | 9000 | 7168 | 256 | ❌ A5 不支持 dropPad |
| ~~A5_unquant_04~~ | 0 | 0 | **0 (cumsum)** | 9000 | 7168 | 256 | ❌ A5 不支持 cumsum |

**边界组合**（压力测试）：

| 组合 ID | 场景 | N | cols | actualExpertNum | 备注 |
|---------|------|---|------|-----------------|------|
| A5_unquant_boundary_N_min | N 下界 | **256** | 128 | 128 | CS_MIN_N |
| A5_unquant_boundary_N_max | N 上界 | **1000000** | 128 | 256 | CS_MAX_N（1M） |
| A5_unquant_boundary_LE_min | actualExpertNum 下界 | 9000 | 7168 | **1** | 极窄 range |
| A5_unquant_boundary_LE_max | actualExpertNum 上界 | 9000 | 7168 | **256** | CS_MAX_ACTUAL_EXPERT_NUM |
| A5_unquant_boundary_cols_large | 大列（双缓冲降级） | 9000 | **32768** | 256 | csDbEnable=0 fallback（已有 ti=27） |

### 4.2 静态量化（quant_mode=0）

**基线组合**：

| 组合 ID | row_idx_type | drop_pad_mode | expert_tokens_num_type | N | cols | actualExpertNum | 备注 |
|---------|-------------|---------------|------------------------|---|------|-----------------|------|
| A5_static_01 | 0 (GATHER) | 0 (dropless) | 1 (COUNT) | 9000 | 7168 | 256 | CutOrigin 典型（已有 ti=19） |
| A5_static_02 | 1 (SCATTER) | 0 (dropless) | 1 (COUNT) | 9000 | 7168 | 256 | row_idx 变体 |

**边界组合**：

| 组合 ID | 场景 | N | cols | actualExpertNum | 备注 |
|---------|------|---|------|-----------------|------|
| A5_static_boundary_N_min | N 下界 | **256** | 128 | 128 | |
| A5_static_boundary_LE_max | actualExpertNum 上界 | 9000 | 7168 | **256** | |
| A5_static_boundary_cols_large | 大列 | 9000 | **32768** | 256 | 双缓冲降级 |

### 4.3 动态量化（quant_mode=1）

**全部场景**：❌ **A5 全局关闭**，不生成测试用例。

现有 `ti=20`（cutorigin_dynamic_quant）已回落 TilingKey=11020000（原路径），不走 CS 模板。

---

## 5. 与现有测试用例的对比

### 5.1 现有 CSV 覆盖情况（`transpose/moe_v3_test.csv`）

已覆盖（共 41 行，含 header）：

| ti | testcase_name | 模板 | quant_mode | row_idx_type | N | cols | actualExpertNum | 备注 |
|----|--------------|------|-----------|-------------|---|------|-----------------|------|
| 0 | moe_v3_cs_fullload_256 | FullLoad | -1 | 0 | 10000 | 128 | 256 | ✓ 基线 |
| 1 | moe_v3_cs_cutorigin_256 | CutOrigin | -1 | 0 | 9000 | 7168 | 256 | ✓ 基线 |
| 17 | moe_v3_cs_fullload_static_quant | FullLoad | 0 | 0 | 10000 | 128 | 256 | ✓ 基线 |
| 18 | ~~moe_v3_cs_fullload_dynamic_quant~~ | FullLoad | **1** | 0 | 10000 | 128 | 256 | ❌ 已关闭（回落 11020000） |
| 19 | moe_v3_cs_cutorigin_static_quant | CutOrigin | 0 | 0 | 9000 | 7168 | 256 | ✓ 基线 |
| 20 | ~~moe_v3_cs_cutorigin_dynamic_quant~~ | CutOrigin | **1** | 0 | 9000 | 7168 | 256 | ❌ 已关闭（回落 11020000） |
| 27 | moe_v3_cs_cutorigin_cols32768 | CutOrigin | -1 | 0 | 9000 | **32768** | 256 | ✓ 边界（大列） |
| 29 | moe_v3_cs_switch_cols1024 | CutOrigin | -1 | 0 | 9000 | **1024** | 256 | ✓ 边界（中列） |

**缺失覆盖**（§4 需补充）：

1. **row_idx_type=1（SCATTER）**：现有全部为 0（GATHER），需补 unquant/static 各 1 条。
2. **N 边界**：无 N=256（下界）和 N=1M（上界）用例。
3. **actualExpertNum 边界**：无 actualExpertNum=1（下界）用例。

---

## 6. 测试用例生成建议

### 6.1 增补优先级

**P0（必补）**：
- `A5_unquant_scatter`：row_idx_type=1, drop_pad_mode=0, N=9000, cols=7168
- `A5_static_scatter`：row_idx_type=1, drop_pad_mode=0, N=9000, cols=7168

**P1（边界覆盖）**：
- `A5_unquant_N_min`：N=256, cols=128, actualExpertNum=128
- `A5_unquant_N_max`：N=1000000, cols=128, actualExpertNum=256（注意 UB 预算，可能回退旧路径）
- `A5_unquant_LE_min`：N=9000, cols=7168, actualExpertNum=1

**P2（压力测试）**：
- `A5_static_N_min`：N=256, cols=128
- `A5_static_cols_large`：N=9000, cols=32768（静态量化大列双缓冲降级）

### 6.2 CSV 格式模板

参照 `ti=1`（cutorigin_256）格式：

```csv
testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries

A5_unquant_scatter,,moe_init_routing_v3,"((9000, 7168), (9000, 1), (9000,), None)","('float16', 'int32', 'float32', 'float32')","('ND',)","((9000, 7168), (9000,), (256,), (9000,))","('float16', 'int32', 'int64', 'float32')","('ND',)","((9000, 7168), (9000, 1), (9000,), None)","('ND',)","((9000, 7168), (9000,), (256,), (9000,))","('ND',)","{'active_num': 9000, 'expert_capacity': 1, 'expert_num': 256, 'drop_pad_mode': 0, 'expert_tokens_num_type': 1, 'expert_tokens_num_flag': True, 'quant_mode': -1, 'active_expert_range': [0, 256], 'row_idx_type': 1}","((-1, 1), (0, 255), (1, 1), (0, 0))","((0.001, 0.001),)",1e-5,(),(),True,,,0,,(),()
```

**关键字段差异点**（相对 ti=1）：
- `row_idx_type`: 0 → **1**（SCATTER）
- `quant_mode`: -1（unquant）/ 0（static）
- expandedX dtype（static）：'float16' → **'int8'**
- input_shapes（static 需增加 scale/offset）

---

## 7. 附录：常量定义速查

### A5 常量（`op_host/moe_init_routing_v3_tiling_arch35.cpp`）

```cpp
const static int64_t CS_MIN_N = 256LL;
const static int64_t CS_MAX_N = 1024LL * 1024LL;
const static int64_t CS_MAX_ACTUAL_EXPERT_NUM = 256LL;
const static int64_t CS_ONE_BLOCK_ELEMENT = 8LL;
const static int64_t CS_DB_BATCH_CAP = 512LL;
```

### 枚举值（A3/A5 一致）

```cpp
enum RowIdxType { ROW_IDX_GATHER = 0, ROW_IDX_SCATTER = 1 };
enum QuantMode { QUANT_MODE_UNQUANT = -1, QUANT_MODE_STATIC = 0, QUANT_MODE_DYNAMIC = 1 };
enum DropPadMode { DROP_PAD_MODE_DROPLESS = 0, DROP_PAD_MODE_DROPPAD = 1 };
enum ExpertTokensNumType { EXPERT_TOKENS_TYPE_CUMSUM = 0, EXPERT_TOKENS_TYPE_COUNT = 1, EXPERT_TOKENS_TYPE_NONE = 2 };
```

---

**文档版本**：v1.0
**生成日期**：2026-08-07
**维护者**：ops-transformer0716 项目组
**关联文档**：`docs/perf_baseline_vs_new_template.md`（§6-§7 迁移劣化根因 + 双缓冲修复）
