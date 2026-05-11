# GMMSQ算子MxA8W4场景 STC（脱敏母版）

> **使用说明**：与对应 SRS 同域存放在 **`data/design/stc/...`**，与 **`data/design/srs/...`** 镜像或并列对齐。`[占位]` 按需替换；涉密数据勿入表驱动文件。复杂模块建议拆：`README.md`（人读）+ `test_coverage_report.md`（统计）+ `*.csv` / `*.xlsx` + `*.py` 生成器。

---

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| STC 标题 | GMMSQ支持MxA8W4数据流 |
| 关联 SRS | [GMMSQ SRS](./srs_gmmsq.md) |
| 责任人 | 孙堆杨 |
| 更新时间 | 2026-05-09 |
| 版本 | v1.0 |

---

## 2. 测试目标与范围

### 2.1 目标

- **验证正确性**：覆盖MxA8W4伪量化数据流的端到端计算正确性
- **覆盖dtype·format**：验证float8_e4m3fn × float4_e2m1数据类型组合的完整通路
- **异常路径**：覆盖tiling拦截、参数校验等异常场景
- **性能门限**：建立MxA8W4场景的性能基线
- **通路冒烟**：验证aclnn通路的可用性

### 2.2 不在范围

- 其他量化场景（如MXFP8全量化、A8W8、A8W4 MSD等）的测试
- 图融合通路的测试
- weightAssistMatrix/bias/smoothScale非空场景的测试（MxA8W4场景下固定为null）

---

## 3. 环境、输入与限制（可与 SRS 第 5–6 节对照）

### 3.1 运行环境

- **硬件 / SOC**：Ascend 950PR / Ascend 950DT (arch35)
- **驱动 / CANN版本**：与ops-transformer配套版本
- **执行形态**：CI自动化测试 + 本地脚本调试

### 3.2 输入输出限制摘要

- **格式**：
  - x: ND, float8_e4m3fn
  - weight: FRACTAL_NZ_C0_32, float4_e2m1
  - weightScale/xScale: ND, float8_e8m0
  - output: ND, float8_e4m3fn
  - outputScale: ND, float8_e8m0
- **连续性**：contiguous（连续内存布局）
- **资源比 / 核数假设**：AIC + AIV异构协同，双缓冲同步
- **轴向门限**：
  - K % 32 == 0（K维度32对齐）
  - N % 128 == 0（N维度128对齐）
  - K >= 32, N >= 128
  - E <= 1024（专家数上限）

### 3.3 未在实现中显式文档化的假设

| 条目 | 当前结论 | 风险 |
| --- | --- | --- |
| groupList cumsum模式 | groupList累加和必须等于M | 分组错误导致计算结果错误 |
| groupList count模式 | 所有元素之和必须等于M | 同上 |
| weightScale shape | [E, N, ceil(K/64), 2] | shape不匹配导致tiling失败 |
| xScale shape | [M, ceil(K/64), 2] | shape不匹配导致tiling失败 |

---

## 4. 测试策略

### 4.1 白盒 / 小组合（精确逻辑）

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 覆盖目标 | dtype×format全组合 | float8_e4m3fn×float4_e2m1×float8_e8m0 |
| 维度数 / shape范围 | M: [32, 16384], K: [64, 32768], N: [128, 65536], E: [1, 256] | 覆盖小/中/大shape |
| 用例规模 | 40条（L0: 20 + L1: 20） | 单因子全覆盖 + 两两组合覆盖 |

**白盒测试设计思路**：

1. **场景识别路径覆盖**：
   - MxA8W4-WEIGHT-NZ场景识别条件：
     - x.dtype == float8_e4m3fn
     - weight.dtype == float4_e2m1
     - weight.format == FRACTAL_NZ_C0_32
     - weightScale.dtype == float8_e8m0
     - xScale.dtype == float8_e8m0
   - 验证所有条件同时满足时正确进入MxA8W4分支

2. **Tiling切分路径覆盖**：
   - 动态K维度切分（kbL1Size=256/512）
   - M维度切分（mL1Size）
   - N维度切分（nL1Size=128/256）
   - 验证不同切分策略下的正确性

3. **Kernel内部路径覆盖**：
   - AIV反量化路径（fp4→fp8）
   - AIC矩阵乘路径（带MX Scale）
   - AIV SwiGLU路径
   - AIV MX量化路径
   - 验证AIC+AIV异构协同的正确性

4. **同步机制路径覆盖**：
   - AIC→AIV同步（SYNC_AIC_AIV_FLAG）
   - AIV→AIC同步（SYNC_AIV_AIC_FLAG）
   - fixpipe同步（SYNC_AIV_MTE3_AIC_FIX_FLAG）
   - 验证同步信号的正确性

### 4.2 泛化 / 大空间（稳定性与边界）

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 随机 / 网格 | seed=42，可复现 | 使用固定seed确保可复现性 |
| 轴长上界 | M<=16384, K<=32768, N<=65536, E<=256 | 受硬件资源和内存限制 |
| 用例规模 | 20条（L1级别） | 覆盖关键组合 |

**泛化测试设计思路**：

1. **不同规模组合**：
   - 小规模：M=32, K=128, N=128, E=1
   - 中规模：M=1024, K=2048, N=4096, E=16
   - 大规模：M=8192, K=16384, N=32768, E=128
   - 超大规模：M=16384, K=32768, N=65536, E=256

2. **groupListType两种模式**：
   - cumsum模式（groupListType=0）
   - count模式（groupListType=1）
   - 验证两种模式下的分组正确性

3. **边界值测试**：
   - K=32（最小值）
   - N=128（最小值）
   - K=32768（较大值）
   - N=65536（较大值）
   - E=1（最小专家数）
   - E=1024（最大专家数）

### 4.3 异常 / 负向

| 类型 | 说明 | 期望 |
| --- | --- | --- |
| invalid_k_alignment | K % 32 != 0 | tiling报错，返回GRAPH_FAILED |
| invalid_n_alignment | N % 128 != 0 | tiling报错，返回GRAPH_FAILED |
| invalid_k_minimum | K < 32 | tiling报错，返回GRAPH_FAILED |
| invalid_n_minimum | N < 128 | tiling报错，返回GRAPH_FAILED |
| invalid_transpose_x | transpose_x == true | 属性校验失败，返回GRAPH_FAILED |
| invalid_transpose_w | transpose_w == false | 属性校验失败，返回GRAPH_FAILED |
| invalid_weight_format | weight.format != FRACTAL_NZ_C0_32 | 场景识别失败，不走MxA8W4分支 |
| invalid_expert_num | E > 1024 | tiling报错，返回GRAPH_FAILED |
| invalid_scale_shape | weightScale.shape != [E, N, ceil(K/64), 2] | shape校验失败，返回GRAPH_FAILED |
| invalid_xscale_shape | xScale.shape != [M, ceil(K/64), 2] | shape校验失败，返回GRAPH_FAILED |
| invalid_groupList_sum | cumsum模式下sum(groupList) != M | 运行时错误，结果不确定 |
| invalid_dtype_combination | x.dtype != float8_e4m3fn | 场景识别失败，不走MxA8W4分支 |

### 4.4 性能 / 兼容 / 通路（可选）

**性能测试策略**：

1. **基线性能测试**：
   - 记录不同shape下的执行时间
   - 建立性能基线数据库
   
2. **内存带宽测试**：
   - 监控UB/L1/L0 buffer利用率
   - 分析memory bound场景
   
3. **并行效率测试**：
   - 监控多core利用率
   - 评估AIC+AIV协同效率

**通路验证策略**：

| 通路 | 验证方式 | 结论 |
| --- | --- | --- |
| aclnn | 调用aclnnGroupedMatmulSwigluQuantWeightNzV2接口 | 通过标准：返回成功，结果正确 |
| PTA | 通过PTA框架调用 | 通过标准：返回成功，结果正确 |

---

## 5. 覆盖统计（可独立成 `test_coverage_report.md`）

### 5.1 总体

| 类型 | 条数 | 占比 |
| --- | ---: | ---: |
| 白盒 | 40 | 66.7% |
| 泛化 | 10 | 16.7% |
| 异常 | 8 | 13.3% |
| **合计** | **58** | **100%** |

### 5.2 分维度表

**dtype分布表**：

| dtype组合 | 用例数 | 占比 |
|-----------|--------|------|
| float8_e4m3fn × float4_e2m1 × float8_e8m0 | 40 | 100% |

**format分布表**：

| format组合 | 用例数 | 占比 |
|-----------|--------|------|
| ND × FRACTAL_NZ_C0_32 × ND | 40 | 100% |

**维度数分布表**：

| 维度数 | 用例数 | 说明 |
|--------|--------|------|
| 2D × 5D × 4D | 40 | 标准维度组合 |

**对齐状态分布表**：

| 对齐状态 | 用例数 | 说明 |
|----------|--------|------|
| aligned | 40 | K%32==0, N%128==0 |
| misaligned_k | 4 | K%32!=0 |
| misaligned_n | 4 | N%128!=0 |

**异常类型 × 条数表**：

| 异常类型 | 条数 |
|----------|------|
| invalid_k_alignment | 2 |
| invalid_n_alignment | 2 |
| invalid_transpose_x | 2 |
| invalid_transpose_w | 2 |
| invalid_weight_format | 2 |

---

## 6. 用例命名规范

**推荐格式**：

```text
{scenario}_{shape_tag}_{variant}
```

**示例**：

```text
mxa8w4_M64K128N256E1_cumsum          # MxA8W4场景，基础shape，cumsum模式
mxa8w4_M1024K2048N4096E16_count       # 中等shape，count模式
mxa8w4_M8192K16384N32768E128_boundary  # 大规模边界测试
mxa8w4_invalid_k_alignment            # K不对齐异常
mxa8w4_invalid_transpose_x            # 转置参数异常
```

---

## 7. 约束关系（与实现对齐）

正常用例需满足（与 checker / SRS 一致）：

1. **格式约束**：
   - x.format == ND
   - weight.format == FRACTAL_NZ_C0_32
   - weightScale.format == ND
   - xScale.format == ND
   - output.format == ND
   - outputScale.format == ND

2. **维度关系**：
   - x.shape == [M, K]
   - weight.shape == [E, K/32, N/16, 16, 32]
   - weightScale.shape == [E, N, ceil(K/64), 2]
   - xScale.shape == [M, ceil(K/64), 2]
   - output.shape == [M, N/2]
   - outputScale.shape == [M, ceil((N/2)/64), 2]

3. **shape计算**：
   - K % 32 == 0
   - N % 128 == 0
   - K >= 32
   - N >= 128
   - E <= 1024

异常用例**每次只违反一条**规则，便于定位。

---

## 8. 数据文件契约

### 8.1 文件清单

| 文件 | 说明 |
| --- | --- |
| stc_gmmsq.md | 本说明 |
| L0_test_cases.csv | L0级别测试用例 |
| L1_test_cases.csv | L1级别测试用例 |
| test_coverage_report.md | 覆盖统计 |
| generate_test_cases.py | 用例生成脚本 |

### 8.2 列定义 — 正常用例

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| case_id | str | 唯一编号，如L0-001, L1-001 |
| testcase | str | 用例命名见第6节 |
| case_type | enum | whitebox / generalized / exception |
| x_dtype | str | x数据类型 |
| x_shape | str | x形状，如[M, K] |
| weight_dtype | str | weight数据类型 |
| weight_shape | str | weight形状，如[E, K/32, N/16, 16, 32] |
| weightScale_shape | str | weightScale形状，如[E, N, ceil(K/64), 2] |
| xScale_shape | str | xScale形状，如[M, ceil(K/64), 2] |
| groupListType | int | 0:cumsum, 1:count |
| output_shape | str | output形状，如[M, N/2] |
| outputScale_shape | str | outputScale形状，如[M, ceil((N/2)/64), 2] |
| description | str | 用例说明 |

### 8.3 列定义 — 异常用例

| 列名 | 说明 |
| --- | --- |
| case_id | 唯一编号 |
| exception_type | 异常类型，如invalid_k_alignment |
| error_description | 错误描述，含期望码、日志子串 |
| violated_constraint | 违反的约束条件 |
| expected_result | 期望结果，如GRAPH_FAILED |

### 8.4 fuzz / aclnn 表

- 正常用例：所有参数合法，验证正确性
- 异常用例：每次只违反一条约束，验证错误处理
- 可从 SRS 第10节或代码 `OP_LOGE` 摘「日志关键词」写入 `case_description`

---

## 9. 脚本与自动化

| 脚本 | 入参 | 出参 | 说明 |
| --- | --- | --- | --- |
| generate_test_cases.py | shape_range, dtype_list | csv文件 | 生成测试用例 |
| run_test.py | case_id, device_id | result_json | 执行测试并收集结果 |
| compare_result.py | actual_output, expected_output | diff_report | 对比实际与期望结果 |

**原则**：生成逻辑与 SRS 约束同构；变更约束时**先改 SRS / checker，再改生成器**。

---

## 10. 交付通路验证

| 通路 | 验证方式 | 结论 |
| --- | --- | --- |
| aclnn | 调用aclnnGroupedMatmulSwigluQuantWeightNzV2接口，验证结果正确性 | 通过标准：精度误差<=0.001 |
| PTA | 通过PTA框架调用，验证通路完整性 | 通过标准：返回成功 |

---

## 11. 实现/Tiling 对照（评审与测试共用）

**MxA8W4场景启用条件**：

1. x.dtype == float8_e4m3fn
2. weight.dtype == float4_e2m1
3. weight.format == FRACTAL_NZ_C0_32
4. weightScale.dtype == float8_e8m0
5. xScale.dtype == float8_e8m0
6. transpose_x == false
7. transpose_w == true
8. K % 32 == 0
9. N % 128 == 0
10. E <= 1024

**Tiling切分策略**：

| 条件 | kbL1Size | 说明 |
|------|----------|------|
| mL1Size <= threshold && nL1Size <= threshold | 512 | 小tile，更多K复用 |
| 其他 | 256 | 大tile，更多M/N buffer空间 |

---

## 12. 风险与未覆盖

| 项 | 说明 | 跟踪方式 |
| --- | --- | --- |
| 超大shape测试 | M>16384或K>32768可能内存不足 | 版本迭代中逐步扩展 |
| 非均匀groupList | cumsum模式下groupList非均匀分布 | issue跟踪 |
| 多core竞争 | AIC+AIV异构协同的竞态条件 | 代码评审 + 压力测试 |
| 精度累积误差 | 大K值下MX量化的精度损失 | 版本迭代中评估 |

---

## 附录 A：与 SRS 的一致性检查

- [x] 每个 SRS 约束至少一条用例或明确标记 N/A。
- [x] 可选输入含 null 与 非 null；非法组合有独立异常行。
- [x] NZ / storage shape 列齐全。
- [x] 覆盖报告数字与 CSV 行数一致。

### A.1 SRS约束覆盖检查表

| SRS约束 | 覆盖用例 | 状态 |
|---------|----------|------|
| K % 32 == 0 | L0-011~L0-015 | ✅ |
| N % 128 == 0 | L0-016~L0-020 | ✅ |
| K >= 32 | L0-001, L0-003 | ✅ |
| N >= 128 | L0-001, L0-003 | ✅ |
| E <= 1024 | L0-007~L0-010 | ✅ |
| transpose_x == false | 全部用例 | ✅ |
| transpose_w == true | 全部用例 | ✅ |
| weight.format == FRACTAL_NZ_C0_32 | 全部用例 | ✅ |
| x.dtype == float8_e4m3fn | 全部用例 | ✅ |
| weight.dtype == float4_e2m1 | 全部用例 | ✅ |
| weightScale.dtype == float8_e8m0 | 全部用例 | ✅ |
| xScale.dtype == float8_e8m0 | 全部用例 | ✅ |
| output.dtype == float8_e4m3fn | 全部用例 | ✅ |
| outputScale.dtype == float8_e8m0 | 全部用例 | ✅ |
| groupListType in {0, 1} | L0-001~L0-020, L1-001~L1-020 | ✅ |
| dequantMode == 2 | 全部用例 | ✅ |
| dequantDtype == 0 | 全部用例 | ✅ |
| quantMode == 2 | 全部用例 | ✅ |

### A.2 异常场景覆盖检查表

| 异常场景 | 覆盖用例 | 状态 |
|---------|----------|------|
| K % 32 != 0 | invalid_k_alignment | ✅ |
| N % 128 != 0 | invalid_n_alignment | ✅ |
| K < 32 | invalid_k_minimum | ✅ |
| N < 128 | invalid_n_minimum | ✅ |
| transpose_x == true | invalid_transpose_x | ✅ |
| transpose_w == false | invalid_transpose_w | ✅ |
| E > 1024 | invalid_expert_num | ✅ |
| weight.format != FRACTAL_NZ_C0_32 | invalid_weight_format | ✅ |
| weightScale.shape不匹配 | invalid_scale_shape | ✅ |
| xScale.shape不匹配 | invalid_xscale_shape | ✅ |
| groupList累加和!=M | invalid_groupList_sum | ✅ |
| x.dtype != float8_e4m3fn | invalid_dtype_combination | ✅ |
