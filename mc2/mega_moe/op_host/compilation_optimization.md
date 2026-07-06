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

在 `ascend950` 的 `aicore_config` 上添加显式的逐输入覆盖，沿用 `aicore_config_arch22` 的同一模式：

1. **多值输入**（topk_ids、topk_weights、weight1、weight2）使用 `.DataType({...})` 位置型列表，等长。每个下标构成一组有效组合。

2. **单值输入**（x、weight_scales1、weight_scales2、scales、y、context、bias1、bias2、x_active_mask、expert_token_nums）使用 `DataTypeList({T})` 单元素列表 —— 不产生笛卡尔乘法。

### ascend950 实际支持的组合

仅声明生产环境实际使用的 dtype-format 组合：

| 输入             | 允许取值                                                          |
|-----------------|-------------------------------------------------------------------|
| x                | BF16                                                              |
| topk_ids        | BF16、FP32                                                        |
| topk_weights    | BF16、FP32                                                        |
| weight1         | E5M2(ND)、E4M3FN(ND)、E4M3FN(NZ)、E2M1(ND)、E2M1(NZ)、E2M1(NZ_C0_32) |
| weight2         | 同 weight1，且与 weight1 **一一对应**                            |
| weight_scales1  | E8M0                                                              |
| weight_scales2  | E8M0                                                              |
| scales          | E8M0                                                              |
| y               | BF16                                                              |

### 组合数计算

- weight1 / weight2 配对类型数：**6**
- topk_ids × topk_weights（独立）：2 × 2 = **4**
- 其余输入：单值（因子 = 1）

**总计 = 6 × 4 = 24 种组合**（启用 `ENABLE_FORMAT_NZ_C0_32` 时）。

未启用 `ENABLE_FORMAT_NZ_C0_32`（旧 devkit）时：去掉 E2M1 的两个 NZ 变体，为 **4 × 4 = 16 种组合**。

### 索引布局（位置型 DataType 列表）

位置型列表中每个下标编码一组完整组合。24 条目按 **weight 类型（外层）× topk 组合（内层）** 排序。

| 下标 | weight1/2 类型 | weight1/2 格式 | topk_ids | topk_weights |
|------|---------------|-----------------|----------|-------------|
| 0    | E5M2          | ND              | BF16     | BF16        |
| 1    | E5M2          | ND              | BF16     | FP32        |
| 2    | E5M2          | ND              | FP32     | BF16        |
| 3    | E5M2          | ND              | FP32     | FP32        |
| 4    | E4M3FN        | ND              | BF16     | BF16        |
| 5    | E4M3FN        | ND              | BF16     | FP32        |
| 6    | E4M3FN        | ND              | FP32     | BF16        |
| 7    | E4M3FN        | ND              | FP32     | FP32        |
| 8    | E4M3FN        | NZ              | BF16     | BF16        |
| 9    | E4M3FN        | NZ              | BF16     | FP32        |
| 10   | E4M3FN        | NZ              | FP32     | BF16        |
| 11   | E4M3FN        | NZ              | FP32     | FP32        |
| 12   | E2M1          | ND              | BF16     | BF16        |
| 13   | E2M1          | ND              | BF16     | FP32        |
| 14   | E2M1          | ND              | FP32     | BF16        |
| 15   | E2M1          | ND              | FP32     | FP32        |
| 16   | E2M1          | NZ              | BF16     | BF16        |
| 17   | E2M1          | NZ              | BF16     | FP32        |
| 18   | E2M1          | NZ              | FP32     | BF16        |
| 19   | E2M1          | NZ              | FP32     | FP32        |
| 20   | E2M1          | NZ_C0_32        | BF16     | BF16        |
| 21   | E2M1          | NZ_C0_32        | BF16     | FP32        |
| 22   | E2M1          | NZ_C0_32        | FP32     | BF16        |
| 23   | E2M1          | NZ_C0_32        | FP32     | FP32        |

下标 16–23 受 `#ifdef ENABLE_FORMAT_NZ_C0_32` 守卫。

---

## 优化前后对比

| 指标                | 优化前（继承 base）    | 优化后（显式覆盖）        |
|---------------------|------------------------|--------------------------|
| 组合数              | ~2688（理论值）        | 24（无 NZ_C0_32 时为 16）|
| x 的 dtype 数       | 7                      | 1（BF16）                |
| weight_scales dtype | 各 2（FLOAT、E8M0）    | 各 1（E8M0）             |
| scales dtype        | 2（FLOAT、E8M0）       | 1（E8M0）                |
| y dtype             | 2（BF16、FP16）        | 1（BF16）                |
| topk_ids            | 1（INT32，DataTypeList）| 2（BF16、FP32，位置型）  |
| weight1/2 配对关系  | 无（笛卡尔积）         | 显式一一对应             |
| 编译耗时            | 非常长                 | 显著降低                 |

---

## 关键设计决策

1. **多值输入用位置型 `DataType` 而非 `DataTypeList`** —— 避免独立笛卡尔乘法，并强制 weight1/weight2 一一对应配对。

2. **固定类型输入用单元素 `DataTypeList`** —— 这些不增加组合数（因子 = 1），同时保持代码简洁。

3. **`#ifdef ENABLE_FORMAT_NZ_C0_32` 守卫** —— 最后 8 个条目（E2M1 NZ 和 E2M1 NZ_C0_32 的 weight 变体）条件编译，与基础 `OpDef` 和 `aicore_config_arch22` 的行为保持一致。四个位置型列表（topk_ids、topk_weights、weight1、weight2）都带同一守卫，保证长度始终一致。

4. **未列出的输入保持原样** —— `context`、`bias1`、`bias2`、`x_active_mask`、`expert_token_nums` 保留原有的单类型 `DataTypeList` 定义。
