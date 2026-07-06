# MegaMoe ascend950 编译优化

## 问题

`mega_moe` 算子的 `ascend950` AICore 配置继承了基础 `OpDef` 的输入定义，多个输入使用 `DataTypeList`（组合式）声明数据类型。这导致算子编译 / 预编译阶段发生 **dtype-format 组合数爆炸**，编译时间过长，二进制体积过大。

### 根因分析

`DataTypeList({A, B, C})` 声明某输入可独立取列表中**任意一个**类型。当多个输入都带有多元素 `DataTypeList` 时，编译器会枚举所有列表的**笛卡尔积**，再乘以位置型的 `DataType` 列表。

原 `ascend950` 配置（继承自基础 `OpDef`）情况如下：

| 输入             | API          | 类型 / 条目                                                  | 数量  |
|-----------------|-------------|--------------------------------------------------------------|-------|
| x                | DataTypeList | BF16, FP16, E5M2, E4M3FN, HIFLOAT8, E2M1, E1M2              | 7     |
| topk_ids         | DataTypeList | INT32                                                        | 1     |
| topk_weights     | DataTypeList | FLOAT, BF16                                                  | 2     |
| weight1          | DataType     | 12 条目（位置型）                                            | 12    |
| weight2          | DataType     | 12 条目（位置型）                                            | 12    |
| weight_scales1   | DataTypeList | FLOAT, E8M0                                                  | 2     |
| weight_scales2   | DataTypeList | FLOAT, E8M0                                                  | 2     |
| scales           | DataTypeList | FLOAT, E8M0                                                  | 2     |
| y                | DataTypeList | BF16, FP16                                                   | 2     |

笛卡尔积 = 7 × 2 × 12 × 2 × 2 × 2 × 2 = **2688** 种组合
（理论上限；实际可能被过滤，但仍然非常大）。

### 与 ascend910b / ascend910_93 的对比

`aicore_config_arch22` 配置（`ascend910b` 和 `ascend910_93` 使用）已经对**每个**输入都做了显式覆盖，多值输入使用等长的位置型 `.DataType()` 列表（18 条目），单值输入使用 `DataTypeList({T})`。这样组合数小且固定。

---

## 解决方案

在 `ascend950` 的 `aicore_config` 上添加显式的逐输入覆盖，**按输入间是否存在配对约束区分使用 `DataType` 与 `DataTypeList`**：

1. **有配对约束的输入**（weight1、weight2）使用 `.DataType({...})` 位置型列表（6 条目），按下标一一对应。只有这两个输入用位置型，因此列表只需覆盖 weight 类型组合，无需展开 topk。

2. **无配对约束的多值输入**（topk_weights）使用 `DataTypeList({A, B})`，独立参与笛卡尔扩展。它与 weight 类型无关，不需要塞进位置型列表。

3. **单值输入**（x、topk_ids、weight_scales1、weight_scales2、scales、y、context、bias1、bias2、x_active_mask、expert_token_nums）使用 `DataTypeList({T})` 单元素列表 —— 不产生笛卡尔乘法。

### ascend950 实际支持的组合

仅声明生产环境实际使用的 dtype-format 组合：

| 输入             | API          | 允许取值                                                          |
|-----------------|-------------|-------------------------------------------------------------------|
| x                | DataTypeList | BF16                                                              |
| topk_ids        | DataTypeList | INT32                                                             |
| topk_weights    | DataTypeList | BF16、FP32                                                        |
| weight1         | DataType     | E5M2(ND)、E4M3FN(ND)、E4M3FN(NZ)、E2M1(ND)、E2M1(NZ)、E2M1(NZ_C0_32) |
| weight2         | DataType     | E5M2(ND)、E4M3FN(ND)、E4M3FN(NZ)、E2M1(ND)、E2M1(NZ_C0_32)、E2M1(NZ_C0_32) |

> weight1 与 weight2 按下标**一一对应**。注意 weight1 的 E2M1 有 ND/NZ/NZ_C0_32 三种格式，
> 而 weight2 的 E2M1 仅有 ND/NZ_C0_32 两种格式 —— weight1 的 E2M1 NZ 对应 weight2 的 NZ_C0_32。

| weight_scales1  | DataTypeList | E8M0                                                              |
| weight_scales2  | DataTypeList | E8M0                                                              |
| scales          | DataTypeList | E8M0                                                              |
| y               | DataTypeList | BF16                                                              |

### 组合数计算

- weight1 / weight2 位置型配对数：**6**
- topk_ids（DataTypeList）：**1**（INT32）
- topk_weights（DataTypeList）：**2**
- 其余输入：单值（因子 = 1）

**总计 = 6 × 2 = 12 种组合**（启用 `ENABLE_FORMAT_NZ_C0_32` 时）。

未启用 `ENABLE_FORMAT_NZ_C0_32`（旧 devkit）时：weight1/weight2 去掉 E2M1 的 NZ 和 NZ_C0_32 两条，配对数降为 4，总计 **4 × 2 = 8 种组合**。

### weight1 / weight2 位置型配对表

weight1 与 weight2 各 6 条目，按下标一一对应（topk_weights 的笛卡尔扩展在此表之外）：

| 下标 | 类型    | weight1 格式 | weight2 格式 |
|------|---------|--------------|--------------|
| 0    | E5M2    | ND           | ND           |
| 1    | E4M3FN  | ND           | ND           |
| 2    | E4M3FN  | NZ           | NZ           |
| 3    | E2M1    | ND           | ND           |
| 4    | E2M1    | NZ           | NZ_C0_32     |
| 5    | E2M1    | NZ_C0_32     | NZ_C0_32     |

下标 4–5 受 `#ifdef ENABLE_FORMAT_NZ_C0_32` 守卫。每个下标再与 topk_weights（2）做笛卡尔积，得到最终 12 组合。

---

## 优化前后对比

| 指标                | 优化前（继承 base）    | 优化后（显式覆盖）          |
|---------------------|------------------------|----------------------------|
| 组合数              | ~2688（理论值）        | 12（无 NZ_C0_32 时为 8）   |
| x 的 dtype 数       | 7                      | 1（BF16）                  |
| weight_scales dtype | 各 2（FLOAT、E8M0）    | 各 1（E8M0）               |
| scales dtype        | 2（FLOAT、E8M0）       | 1（E8M0）                  |
| y dtype             | 2（BF16、FP16）        | 1（BF16）                  |
| topk_ids            | 1（INT32，DataTypeList）| 1（INT32，DataTypeList）   |
| topk_weights        | 2（FLOAT、BF16，DataTypeList）| 2（BF16、FP32，DataTypeList）|
| weight1/2 配对关系  | 无（笛卡尔积）         | 位置型一一对应（6 条目）   |
| 编译耗时            | 非常长                 | 显著降低                   |

---

## 关键设计决策

1. **按配对约束选择 `DataType` 或 `DataTypeList`** —— weight1/weight2 必须一一对应，用位置型 `DataType`（6 条目）锁定配对；topk_weights 与 weight 无关且独立，用 `DataTypeList` 让编译器自动笛卡尔扩展，无需手工展开。这样位置型列表只保留必要的配对维度，代码最简。

2. **固定类型输入用单元素 `DataTypeList`** —— 这些不增加组合数（因子 = 1），同时保持代码简洁。

3. **`#ifdef ENABLE_FORMAT_NZ_C0_32` 守卫** —— weight1/weight2 位置型列表的最后 2 条（E2M1 NZ / NZ_C0_32 变体）条件编译，与基础 `OpDef` 和 `aicore_config_arch22` 的行为保持一致。两个列表（weight1、weight2）都带同一守卫，保证长度始终一致。

4. **未列出的输入保持原样** —— `context`、`bias1`、`bias2`、`x_active_mask`、`expert_token_nums` 保留原有的单类型 `DataTypeList` 定义。
