# GMM A16W4 PerGroup ND — STC 用例设计方案

> 本文档记录 A16W4 pergroup ND 数据流的用例设计**方案**（方法论）和数量统计，不列出具体用例。
> 具体用例详见 [05 测试用例设计](../design/05_test_case_design.md)。

---

## 1. 测试策略概述

### 1.1 测试层次

| 层次 | 说明 | 用例数 |
|------|------|:------:|
| **UT** | Tiling 异常拦截 + 白盒验证（tiling 端到端 + kernel 打桩 + kernel 端到端） | 108 |
| **ST** | 端到端功能/性能验证（精度 + 功能 + 性能 + 泛化） | 54 |
| **总计** | | **162** |

### 1.2 测试方法

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| **黑盒** | 基于算子规格文档的约束条件，不依赖代码实现 | Tiling 异常拦截、精度/功能/性能/泛化验证 |
| **白盒（插桩级）** | 基于 kernel 数据流，在关键节点打桩验证中间结果 | Kernel 打桩验证（MTE/VF/地址偏移/matmul） |
| **白盒（端到端级）** | 基于代码路径的分支覆盖，通过端到端执行间接验证 | Tiling 白盒、Kernel 白盒 |

### 1.3 白盒用例的两个级别

- **插桩级**：在 kernel 内部关键节点（MTE 搬运后、VF 计算后、地址偏移后等）插入检查点，验证中间数据正确性。适用于数据搬运和计算逻辑的单元验证。
- **端到端级**：不插入检查点，通过构造特定输入使特定代码路径被执行，检查最终输出正确性来间接验证路径覆盖。适用于分支逻辑和集成验证。

---

## 2. 黑盒用例设计方案

### 2.1 Tiling 异常拦截（UT）

- **设计方法**：基于算子规格文档约束条件的等价类划分 + 边界值分析
- **覆盖维度**：
  - dtype 校验（x/weight/bias/scale/offset/groupList 的 dtype 合法性）
  - shape 校验（维度、数值约束、shape 匹配三个子维度）
  - format 校验（所有 tensor 的 ND 格式约束）
  - 属性校验（groupType/groupListType/actType/splitItem 等属性合法性）
  - 场景校验（组合参数的逻辑一致性，如 groupType 与 groupList 的匹配）
- **用例数量**：44 个（T-01 ~ T-44）
- **优先级分布**：P0（核心）28 个，P1（重要）14 个，P2（边界）2 个

### 2.2 精度验证（ST）

- **设计方法**：基于 dtype 组合的全覆盖
- **覆盖维度**：x dtype × bias dtype × offset 有无
  - x=FLOAT16：bias(FP16/null) × offset(FP16/null) = 4 组合
  - x=BFLOAT16：bias(BF16/FP32/null) × offset(BF16/null) = 5 组合（含 FP32 bias 特殊场景）
- **验证方法**：与 CPU 参考实现对比，FP16 容差 rtol=1e-3/atol=1e-3，BF16 容差 rtol=1e-2/atol=1e-2
- **用例数量**：9 个（P-01 ~ P-09）

### 2.3 功能验证（ST）

- **设计方法**：基于参数组合的正交试验
- **覆盖维度**：
  - groupType（单单单/多多多）
  - 转置/非转置
  - group_size（32/64/128/256）
  - groupListType（cumsum/count）
  - shape 边界（极小 M/N/K、单/最大 expert、不等 m_i）
- **用例数量**：27 个（F-01 ~ F-27）
  - 单单单 groupType=0：12 个（F-01 ~ F-12）
  - 多多多 groupType=-1：15 个（F-13 ~ F-27）

### 2.4 性能验证（ST）

- **设计方法**：基于典型应用场景的基准测试
- **覆盖维度**：
  - MoE expert 数量（8/64 experts）
  - shape 大小（大 shape 1024×8192×8192、小 shape 1×128×128）
  - 转置/非转置
  - 单单单大 E 场景
- **用例数量**：6 个（PERF-01 ~ PERF-06）
- **性能指标**：PERF-01/02/05 作为 MoE 基线，PERF-03/04/06 验证极端 shape 下的效率

### 2.5 泛化验证（ST）

- **设计方法**：（待补充 — 当前覆盖度不足，后续补充设计方案）
- **当前覆盖维度**：
  - group_size 边界（最小 32、最大 256、单 group、多 group）
  - shape 边界（极小 M/N、单 expert、最大 expert）
  - 特殊组合（各 expert m_i 极端不同、k_i 不同、非均匀分组）
- **当前用例数量**：12 个（G-01 ~ G-12）
- ⚠️ **预留章节**，后续补充泛化用例设计方案，重点考虑：
  - 更多 group_size × shape 交叉组合
  - 大 K 场景（K > 4096）下的多 group 行为
  - 边界对齐与非对齐场景
  - weight不转置时，n为偶数，当前kernel实现可能不够泛化，需要重点泛化验证

---

## 3. 白盒用例设计方案

### 3.1 Kernel 打桩验证（UT，插桩级）

- **设计方法**：基于 kernel 数据流的关键节点验证
- **覆盖阶段**：
  1. **MTE 搬运**（7 个）：MTE2（weight/scale/offset GM→UB）、MTE3（反量化后 weight UB→L1）、MTE1（A/Bias GM→L1）、输出写回（C L1→GM）
  2. **VF 计算**（7 个）：INT4 unpack、INT4→FP16 cast、pergroup scale/offset 加载（NK 广播 + KN 直接）、反量化计算（有/无 offset）
  3. **地址偏移**（3 个）：pergroup scale 偏移、bias 偏移分离、INT4 weight >>1 偏移
  4. **Tiling 参数传递**（3 个）：groupSize 传递、mList/kList/nList 填充、baseM/baseN 动态调整
  5. **Matmul 计算**（2 个）：基础 matmul、matmul + bias
- **用例数量**：22 个（KS-01 ~ KS-22）

### 3.2 Tiling 白盒验证（UT，端到端级）

- **设计方法**：基于 tiling 代码路径的分支覆盖
- **覆盖路径**：
  1. **IsA16W4NDPergroup 判断**（3 个）：pergroup 正确识别、perchannel 不误判、多多多 2 维 scale 识别
  2. **SetAntiquantGroupSize**（4 个）：3 维 scale 推导、gs=32/64/256 各分支
  3. **SetTilingKey**（4 个）：PER_GROUP/PER_CHANNEL 设置、转置/非转置 key
  4. **SetShapeList**（3 个）：单单单/多多多 shape 填充、INT4 nSize 扩展
  5. **baseM/baseN/baseK**（2 个）：默认值、有 bias 调整
  6. **CalcResplitTiling**（2 个）：多多多不使能、单单单 ND 非转置不使能
  7. **字段完整性**（2 个）：单单单/多多多完整参数传递
- **验证方式**：检查 tiling 输出结构体 `GMMWeightQuantParam` 的字段值
- **用例数量**：20 个（TW-01 ~ TW-20）

### 3.3 Kernel 白盒验证（UT，端到端级）

- **设计方法**：基于 kernel 代码路径的分支覆盖
- **覆盖路径**：
  1. **if constexpr 分支**（6 个）：8 种 tiling key 组合中选取 6 种代表性组合（无/有 offset × 无/有 bias × 非转置/转置）
  2. **ComputeBasicBlock 分发**（2 个）：ND 非转置→NdKn 路径、ND 转置→NdNk 路径
  3. **GetBufferConfig 分支**（2 个）：pergroup ND buffer 配置、perchannel ND buffer 配置（对比验证不影响已有功能）
  4. **VF groupSize 运行时分支**（4 个）：gs=32/64/128/256 各分支
  5. **CopyAntiQuantParamsGmToUb PER_GROUP**（2 个）：多多多/单单单 2D copy
  6. **SetGmAddr pergroup 偏移**（2 个）：多 group 偏移序列、bias 偏移分离
  7. **AIC/AIV 协作**（4 个）：多 BasicBlock 迭代（M/N 方向）、K 方向多 kBlock、单 BasicBlock
- **验证方式**：端到端执行并检查最终输出与 CPU 参考实现的容差对比
- **用例数量**：22 个（KW-01 ~ KW-22）

---

## 4. 用例统计汇总

| 类型 | 数量 | 测试方法 | 测试层次 | 编号范围 |
|------|:----:|----------|----------|----------|
| Tiling 异常拦截 | 44 | 黑盒 | UT | T-01 ~ T-44 |
| 精度验证 | 9 | 黑盒 | ST | P-01 ~ P-09 |
| 功能验证 | 27 | 黑盒 | ST | F-01 ~ F-27 |
| 性能验证 | 6 | 黑盒 | ST | PERF-01 ~ PERF-06 |
| 泛化验证 | 12 | 黑盒 | ST | G-01 ~ G-12 |
| Kernel 打桩验证 | 22 | 白盒（插桩级） | UT | KS-01 ~ KS-22 |
| Tiling 白盒 | 20 | 白盒（端到端级） | UT | TW-01 ~ TW-20 |
| Kernel 白盒 | 22 | 白盒（端到端级） | UT | KW-01 ~ KW-22 |
| **总计** | **162** | | | |

**按测试层次统计**：
- UT：44 + 22 + 20 + 22 = **108 个**
- ST：9 + 27 + 6 + 12 = **54 个**

**按测试方法统计**：
- 黑盒：44 + 9 + 27 + 6 + 12 = **98 个**
- 白盒：22 + 20 + 22 = **64 个**

---

## 5. 待补充项

| 待补充项 | 当前状态 | 后续计划 |
|----------|----------|----------|
| 泛化用例设计方案 | 覆盖度不足，仅 12 个用例 | 补充 group_size × shape 交叉组合、大 K 场景等 |
| 兼容性用例 | 暂不考虑 | 由 A2/A3 和 Ascend 950 已有回归测试覆盖 |
