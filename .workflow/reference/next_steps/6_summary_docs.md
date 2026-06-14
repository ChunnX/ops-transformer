# 任务 6：总结归纳设计文档

## 背景

阶段一（规划与设计）已完成 5 个任务，产出了大量文档：算子规格、代码走读、SRS 实现方案（4 个设计文档）、测试用例设计（162 个用例）。这些文档内容丰富但分散，需要总结归纳为便于留档查看的设计文档。

本任务的目标是在**最大程度保留核心信息**的情况下，减少不必要的设计细节，形成 3 个总结性文档。

## 任务目标

产出 3 个文档，均放在 `.workflow/raw/development/` 目录下：

1. **更新脑图**：`gmm_a16w4_pergroup_nd_mindmap.md`
2. **STC 用例设计方案文档**：`gmm_a16w4_pergroup_nd_stc.md`
3. **算子功能实现方案说明文档**：`gmm_a16w4_pergroup_nd_design_summary.md`

## 必须读取的输入文档

### 必读文档

1. **当前脑图**：`.workflow/raw/development/gmm_a16w4_pergroup_nd_mindmap.md`
   - 了解当前脑图结构，在此基础上更新

2. **算子规格文档**：`.workflow/raw/spec/gmm_a16w4nd_op.md`
   - 输入输出规格、约束条件

3. **需求规格**：`.workflow/raw/spec/gmm_a16w4_950_requirement.md`
   - 原始需求和实现方案要点

4. **01 交付件清单**：`.workflow/raw/design/01_deliverables.md`
   - 完整交付件范围

5. **02 与硬件无关方案**：`.workflow/raw/design/02_platform_independent_design.md`
   - 地址偏移公式、兼容性策略、测试方案

6. **03 Tiling 方案**：`.workflow/raw/design/03_tiling_design.md`
   - Tiling 拦截校验、切分策略、tiling key 注册

7. **04 Kernel 方案**：`.workflow/raw/design/04_kernel_design.md`
   - Controller 适配、BasicBlock 适配、VF 函数、L1/UB 空间划分

8. **05 测试用例设计**：`.workflow/raw/design/05_test_case_design.md`
   - 162 个用例的完整设计

9. **Kernel 走读文档**（总体视角章节）：`.workflow/raw/docs/gmm_a16w8_kernel_walkthrough.md`
   - 第 0 章总体视角中的 SVG 图和数据流描述，作为实现方案文档的参考

### 选读文档

- `.workflow/GOAL.md` — 项目总体目标
- `.workflow/raw/docs/gmm_a16w8_tiling_walkthrough.md` — Tiling 走读细节
- `.workflow/raw/docs/wqbmmv2_nd_antiquant_walkthrough.md` — wqbmmv2 反量化走读

---

## 产出 1：更新脑图（gmm_a16w4_pergroup_nd_mindmap.md）

### 目标

对照用例设计文档（05_test_case_design.md）更新脑图内容，使脑图作为用例设计文档和用例表格的**总览图**。

### 更新要求

#### UT 部分更新

当前脑图的 UT 部分只有简单的分类描述，需要更新为：

```
## UT（Tiling 异常拦截）

### 黑盒用例（44 个）
- dtype 校验（7 个）：T-01 ~ T-07
- shape 校验（19 个）：T-08 ~ T-26
  - 维度校验（8 个）
  - 数值约束（6 个）
  - shape 匹配（5 个）
- format 校验（5 个）：T-27 ~ T-31
- 属性校验（6 个）：T-32 ~ T-37
- 场景校验（7 个）：T-38 ~ T-44

### 白盒用例 — Tiling 端到端（20 个）
- IsA16W4NDPergroup 判断路径（3 个）：TW-01 ~ TW-03
- SetAntiquantGroupSize 路径（4 个）：TW-04 ~ TW-07
- SetTilingKey 路径（4 个）：TW-08 ~ TW-11
- SetShapeList 路径（3 个）：TW-12 ~ TW-14
- baseM/baseN/baseK 计算路径（2 个）：TW-15 ~ TW-16
- CalcResplitTiling 路径（2 个）：TW-17 ~ TW-18
- GMMWeightQuantParam 字段完整性（2 个）：TW-19 ~ TW-20

### 白盒用例 — Kernel 打桩验证（22 个）
- MTE 搬运验证（7 个）：KS-01 ~ KS-07
- VF 计算验证（7 个）：KS-08 ~ KS-14
- 地址偏移验证（3 个）：KS-15 ~ KS-17
- Tiling 参数传递验证（3 个）：KS-18 ~ KS-20
- Matmul 计算验证（2 个）：KS-21 ~ KS-22

### 白盒用例 — Kernel 端到端（22 个）
- if constexpr 分支覆盖（6 个）：KW-01 ~ KW-06
- ComputeBasicBlock 分发路径（2 个）：KW-07 ~ KW-08
- GetBufferConfig 分支覆盖（2 个）：KW-09 ~ KW-10
- VF groupSize 运行时分支（4 个）：KW-11 ~ KW-14
- CopyAntiQuantParamsGmToUb PER_GROUP（2 个）：KW-15 ~ KW-16
- SetGmAddr pergroup 偏移（2 个）：KW-17 ~ KW-18
- AIC/AIV 协作路径（4 个）：KW-19 ~ KW-22
```

#### ST 部分更新

```
## ST

### 精度（黑盒，9 个）
- x=FLOAT16（4 个）：P-01 ~ P-04
- x=BFLOAT16（5 个）：P-05 ~ P-09

### 功能（黑盒，27 个）
- 单单单 groupType=0（12 个）：F-01 ~ F-12
- 多多多 groupType=-1（15 个）：F-13 ~ F-27

### 性能（黑盒，6 个）
- 典型 MoE 场景（3 个）：PERF-01 ~ PERF-03
- 大/小 shape 场景（3 个）：PERF-04 ~ PERF-06

### 泛化（黑盒，12 个）
- group_size 边界（4 个）：G-01 ~ G-04
- shape 边界（4 个）：G-05 ~ G-08
- 特殊组合（4 个）：G-09 ~ G-12
- （待补充：泛化用例覆盖度不足，后续补充）

### 兼容性（待补充）
```

### 注意事项

- **不需要全部展开到脑图中**，脑图作为总览图，说明用例分类和条数即可
- **区分黑盒/白盒**：每个大类标注测试方法
- **区分 UT/ST**：UT 包含 tiling 异常拦截 + 白盒验证，ST 包含精度/功能/性能/泛化/兼容性
- 保留脑图原有的输入/输出部分不变
- 用例编号范围要与 05_test_case_design.md 一致

---

## 产出 2：STC 用例设计方案文档（gmm_a16w4_pergroup_nd_stc.md）

### 目标

记录用例设计**方案**（方法论），不关注具体用例是什么。记录最终各类用例的数量统计。

### 文档结构

```markdown
# GMM A16W4 PerGroup ND — STC 用例设计方案

## 1. 测试策略概述
- 测试层次：UT（tiling 异常拦截 + 白盒验证）+ ST（端到端功能/性能验证）
- 测试方法：黑盒（基于规格约束）+ 白盒（基于代码路径）
- 白盒用例的两个级别：插桩级（kernel 打桩）和端到端级

## 2. 黑盒用例设计方案
### 2.1 Tiling 异常拦截（UT）
- 设计方法：基于算子规格文档约束条件的等价类划分 + 边界值分析
- 覆盖维度：dtype / shape / format / 属性 / 场景组合
- 用例数量：44 个（T-01 ~ T-44）

### 2.2 精度验证（ST）
- 设计方法：基于 dtype 组合的全覆盖
- 覆盖维度：x dtype × bias dtype × offset 有无
- 用例数量：9 个（P-01 ~ P-09）

### 2.3 功能验证（ST）
- 设计方法：基于参数组合的正交试验
- 覆盖维度：groupType × 转置/非转置 × group_size × groupListType × shape 边界
- 用例数量：27 个（F-01 ~ F-27）

### 2.4 性能验证（ST）
- 设计方法：基于典型应用场景的基准测试
- 覆盖维度：MoE expert 数量 × shape 大小
- 用例数量：6 个（PERF-01 ~ PERF-06）

### 2.5 泛化验证（ST）
- 设计方法：（待补充 — 当前覆盖度不足，后续补充设计方案）
- 当前用例数量：12 个（G-01 ~ G-12）
- ⚠️ 预留章节，后续补充泛化用例设计方案

## 3. 白盒用例设计方案
### 3.1 Kernel 打桩验证（UT，插桩级）
- 设计方法：基于 kernel 数据流的关键节点验证
- 覆盖阶段：MTE 搬运 → VF 计算 → 地址偏移 → Tiling 参数传递 → Matmul 计算
- 用例数量：22 个（KS-01 ~ KS-22）

### 3.2 Tiling 白盒验证（UT，端到端级）
- 设计方法：基于 tiling 代码路径的分支覆盖
- 覆盖路径：IsA16W4NDPergroup / SetAntiquantGroupSize / SetTilingKey / SetShapeList / baseM/baseN/baseK / CalcResplitTiling / 字段完整性
- 用例数量：20 个（TW-01 ~ TW-20）

### 3.3 Kernel 白盒验证（UT，端到端级）
- 设计方法：基于 kernel 代码路径的分支覆盖
- 覆盖路径：if constexpr / ComputeBasicBlock / GetBufferConfig / VF groupSize / CopyAntiQuantParamsGmToUb / SetGmAddr / AIC/AIV 协作
- 用例数量：22 个（KW-01 ~ KW-22）

## 4. 用例统计汇总

| 类型 | 数量 | 测试方法 | 测试层次 | 编号范围 |
|------|------|----------|----------|----------|
| Tiling 异常拦截 | 44 | 黑盒 | UT | T-01 ~ T-44 |
| 精度验证 | 9 | 黑盒 | ST | P-01 ~ P-09 |
| 功能验证 | 27 | 黑盒 | ST | F-01 ~ F-27 |
| 性能验证 | 6 | 黑盒 | ST | PERF-01 ~ PERF-06 |
| 泛化验证 | 12 | 黑盒 | ST | G-01 ~ G-12 |
| Kernel 打桩验证 | 22 | 白盒（插桩级） | UT | KS-01 ~ KS-22 |
| Tiling 白盒 | 20 | 白盒（端到端级） | UT | TW-01 ~ TW-20 |
| Kernel 白盒 | 22 | 白盒（端到端级） | UT | KW-01 ~ KW-22 |
| **总计** | **162** | | | |

## 5. 待补充项
- 泛化用例设计方案（当前覆盖度不足）
- 兼容性用例（暂不考虑，由回归测试覆盖）
```

### 注意事项

- **不列出具体用例**，只记录设计方案和数量统计
- 泛化用例部分**预留章节**，标注"待补充"
- 用例数量必须与 05_test_case_design.md 一致

---

## 产出 3：SRS 算子实现方案文档（gmm_a16w4_pergroup_nd_srs.md）

### 目标

从设计和逻辑链路上说明 GMM A16W4 weight ND pergroup 数据流的实现方案。便于他人理解该算子的需求和实现方案。使用表格、SVG 图、ASCII 图、简单伪代码说明关键细节，**省略细枝末节的内容**。

### 文档结构

```markdown
# GMM A16W4 PerGroup ND — SRS 算子实现方案

## 1. 需求概述
- 数据流说明（A16W4 INT4 weight ND pergroup 伪量化）
- 计算公式
- 与 A2/A3 已有场景的关系（兼容性扩展）
- 关键约束（group_size ∈ {32,64,128,256}、K % gs == 0、ND 格式）

## 2. Tiling 切分策略
- 绘制 SVG 图说明 tiling 和分核算法：
  - M/N/K 三维切分示意图
  - baseM=256, baseN=256, baseK=64 的切分方式
  - 单单单 vs 多多多的切分差异
  - 多 AI Core 分配策略
- 说明 tiling key 的选择逻辑
- 说明 pergroup 场景下 groupSize 的传递方式

## 3. Buffer 空间分配
- 绘制 SVG 图展示 L1（512KB）和 UB（256KB/248KB）的空间划分：
  - L1 分配：WeightL1_P0(128KB) | AL1_P0(128KB) | AL1_P1(128KB) | WeightL1_P1(128KB)
  - 有 bias 时的 L1 分配
  - UB 分配：weightInputLowBit(174KB) | highBitData(62KB) | antiQuantScale(8KB) | antiQuantOffset(8KB)
  - 双缓冲机制
- 说明 pergroup 场景下 scale UB 从 4KB 增大到 8KB 的原因

## 4. 计算流程
- 使用伪代码说明完整的计算流程：
  - Controller.Process() 主循环
  - AIV 核：MTE2 搬运 → VF 反量化 → MTE3 搬运到 L1
  - AIC 核：MTE1 搬运 A/Bias → Matmul → GetTensorC 写回
  - AIC/AIV 同步机制
- 说明 pergroup 反量化的核心计算：
  - NK 排布：按 groupSize 分支，scale 广播加载
  - KN 排布：统一处理，scale 直接加载

## 5. 软件算法到硬件映射
- 绘制 SVG 图展示 GMM A16W4 weight ND 数据流在 Ascend 950 硬件上的执行过程：
  - 核心目标：将软件算法功能与硬件架构做对应
  - 展示内容：
    - 32 个 AI Core 的整体布局（每个 Core 含 1 Cube + 2 Vector）
    - 数据从 GM 到各层 local memory 的搬运路径（MTE1/MTE2/MTE3）
    - 软件计算步骤（反量化、矩阵乘、bias 加）映射到具体硬件单元（Vector Core 做反量化，Cube Core 做矩阵乘）
    - 数据在 L1/UB/L0A/L0B/L0C/BiasTable 之间的流动
    - Fixpipe 从 L0C 到 UB 的回传路径
    - 多 Core 并行执行时的数据分片和同步
  - 该 SVG 应结合伪代码中的计算步骤，标注每个步骤对应的硬件单元和数据搬运操作
- 表格说明软件操作与硬件单元的映射关系：
  | 软件操作 | 硬件单元 | Local Memory | 数据搬运 |
  |----------|----------|--------------|----------|
  | MTE2: weight/scale/offset GM→UB | Vector Core | UB | MTE2 |
  | VF: INT4 unpack + 反量化 | Vector Core | UB | VF 指令 |
  | MTE3: 反量化后 weight UB→L1 | Vector Core | L1 | MTE3 |
  | MTE1: A/Bias GM→L1 | Cube Core | L1/BiasTable | MTE1 |
  | Matmul: A × W_dequant | Cube Core | L0A×L0B→L0C | Cube 指令 |
  | BiasAdd: C + bias | Cube Core | L0C + BiasTable | Fixpipe |
  | GetTensorC: C L1→GM | Cube Core | L1→GM | MTE1 |

## 6. 地址偏移方案
- 表格对比 perchannel vs pergroup 的地址偏移公式
- 重点说明 pergroup 场景下 antiquantScale/Offset 偏移公式的变化
- 说明 bias 偏移分离的设计（biasBaseOffset 独立于 antiquantParamsBaseOffset）
- INT4 weight 的 >>1 字节偏移处理

## 7. A2/A3 → Ascend 950 兼容性关键修改点
- 表格列出所有关键修改点：
  | 修改点 | A2/A3 实现 | Ascend 950 适配 | 修改文件 |
  |--------|-----------|----------------|----------|
  | ... | ... | ... | ... |
- 说明代码分支隔离策略（编译期 if constexpr）
- 说明不影响已有功能的关键保护措施

## 8. 交付件概览
- 简要列出需要修改/新增的文件（参照 01_deliverables.md 的汇总表）
- 不需要展开每个文件的具体修改内容
```

### SVG 图要求

#### Tiling 切分策略图

需要展示：
- M/N/K 三维矩阵的切分方式
- baseM/baseN/baseK 的切分块
- 多 AI Core 的分配方式
- 单单单 vs 多多多的差异

#### Buffer 空间分配图

需要展示：
- L1（512KB）的空间划分（无 bias / 有 bias 两种）
- UB（248KB 可用）的空间划分
- 双缓冲机制（P0/P1 交替）
- pergroup 场景下 scale/offset UB 的变化

#### 软件算法到硬件映射图（核心 SVG）

**这是本文档最重要的 SVG 图**，需要展示：
- Ascend 950 的硬件架构（32 AI Core，每个含 1 Cube + 2 Vector）
- GMM A16W4 weight ND 数据流的完整执行过程在硬件上的映射
- 数据流动路径：
  - GM → UB（MTE2）：weight INT4 + scale + offset
  - UB 上 Vector Core 执行反量化（INT4 unpack → FP16，×scale，+offset）
  - UB → L1（MTE3）：反量化后的 FP16 weight
  - GM → L1（MTE1）：A 矩阵 + bias
  - L1 → L0A/L0B：A 和 W_dequant 加载到 Cube Core
  - Cube Core 执行 Matmul（L0A × L0B → L0C）
  - L0C → UB（Fixpipe）：bias 加和结果回传
  - UB → GM（MTE2）：输出 C 写回
- 标注每个步骤对应的伪代码行号或函数名
- 展示多 Core 并行时的数据分片（不同 Core 处理不同的 M/N 块）

### 伪代码要求

计算流程伪代码应包含：
```
Controller.Process():
  for groupIdx in 0..groupNum:
    SetMKN(groupIdx)
    SetGmAddr(groupIdx)
    for each AI Core:
      AIV: CopyWeight + Scale/Offset GM→UB
      AIV: VF 反量化 (INT4→FP16, ×scale, +offset)
      AIV: MTE3 反量化后 weight UB→L1
      AIV: MTE1 A/Bias GM→L1
      AIC: Wait AIV
      AIC: Matmul (A × W_dequant + bias)
      AIC: GetTensorC → GM
```

### 注意事项

- **面向读者**：文档目标是让他人理解算子需求和实现方案，不是给编码人员看的详细设计
- **省略细节**：不需要列出具体的 VF 函数签名、WqmmConfig 字段值等
- **图表优先**：能用 SVG/表格/ASCII 图说明的，不用大段文字
- **关键修改点**：A2/A3 → Ascend 950 的兼容性修改是重点，需要明确列出
- **软件到硬件映射 SVG 是核心**：该图应放在计算流程伪代码之后，用图形化方式展示伪代码中每个步骤在硬件上的执行过程

---

## 完成标准

- [ ] 脑图更新完成，UT/ST 部分对齐用例设计文档，标注黑盒/白盒和用例条数
- [ ] STC 文档完成，记录用例设计方案和数量统计，泛化用例部分预留
- [ ] SRS 实现方案文档完成，包含 tiling 切分 SVG、buffer 分配 SVG、软件算法到硬件映射 SVG、计算流程伪代码、关键修改点表格
- [ ] 软件算法到硬件映射 SVG 放在计算流程伪代码之后，展示数据流在 Ascend 950 硬件上的完整执行过程
- [ ] 3 个文档均放在 `.workflow/raw/development/` 目录下
- [ ] 文档间交叉引用正确
- [ ] SRS 文档的 SVG 图可正确渲染
- [ ] 用例数量统计与 05_test_case_design.md 一致（162 个）

## 执行步骤

1. **读取所有必读文档**，理解现有内容
2. **更新脑图**：对照 05_test_case_design.md 更新 UT/ST 部分
3. **编写 STC 文档**：记录用例设计方案和数量统计
4. **编写 SRS 实现方案文档**：
   - 先写需求概述
   - 再写 tiling 切分策略（含 SVG）
   - 再写 buffer 空间分配（含 SVG）
   - 再写计算流程（伪代码）
   - 再写软件算法到硬件映射（核心 SVG，放在伪代码之后，展示计算流程在硬件上的执行过程）
   - 再写地址偏移方案
   - 最后写 A2/A3 → Ascend 950 关键修改点
5. **检查文档间一致性**

## 注意事项

- **不要复制粘贴设计文档的详细内容**，要提炼核心信息
- **SVG 图要简洁清晰**，不要过于复杂
- **伪代码要简明**，展示核心逻辑即可
- **表格要精炼**，只列出关键信息
