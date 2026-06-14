# 任务 5：设计 GMM A16W4 PerGroup ND 测试用例

## 背景

基于已完成的算子规格文档（任务1）、代码走读（任务2/3）和 SRS 实现方案（任务4），现在需要设计具体的测试用例，为后续编码实现和验证提供明确的测试目标。

脑图文档（`.workflow/raw/development/gmm_a16w4_pergroup_nd_mindmap.md`）已给出 UT 和 ST 的框架描述，但需要进一步细化为具体的用例，包含完整的参数设置。

## 任务目标

设计 GMM A16W4 PerGroup ND 数据流的具体测试用例，给出每个用例的完整参数设置，包括：
- **Tiling 异常拦截用例**：验证 tiling 阶段的参数校验和错误拦截（黑盒，基于规格约束设计）
- **精度用例**：验证计算结果的正确性（与 CPU 参考实现对比）（黑盒，基于 dtype 组合设计）
- **功能用例**：验证各种参数组合下的功能正确性（黑盒，基于参数组合设计）
- **性能用例**：验证关键场景下的性能表现（黑盒）
- **泛化用例**：验证边界值和特殊场景（黑盒，基于规格边界值设计）
- **Kernel 打桩验证用例**：通过打桩调试逐步确认 kernel 中每个搬运和计算逻辑的正确性（白盒，插桩级）
- **Tiling 白盒用例**：基于 tiling 代码路径设计，端到端验证 tiling 计算中间结果的正确性（白盒，端到端级）
- **Kernel 白盒用例**：基于 kernel 代码路径设计，端到端验证各分支/分发逻辑的正确性（白盒，端到端级）

**注意**：兼容性用例暂不考虑（由后续回归测试覆盖）。

**白盒用例设计原则**：
- 白盒用例的设计依据是**代码内部路径**，而非规格文档的约束条件
- 每个白盒用例必须明确标注**覆盖的代码路径**（函数名 + 分支条件）
- 白盒用例可以是端到端执行（不需要插桩），但设计时必须明确"这个用例是为了覆盖哪条代码路径"
- 白盒用例与黑盒用例可能有重叠（同一个输入既覆盖规格组合又覆盖代码路径），但设计出发点不同

## 必须读取的输入文档

在开始设计之前，**必须按顺序读取以下文档**：

### 必读文档

1. **算子规格文档**：`.workflow/raw/spec/gmm_a16w4nd_op.md`
   - A16W4 weight ND pergroup 的完整输入输出规格、约束条件
   - 这是用例设计的根本依据

2. **脑图文档**：`.workflow/raw/development/gmm_a16w4_pergroup_nd_mindmap.md`
   - UT 和 ST 的框架描述
   - 已列出的测试场景分类

3. **01 交付件清单**：`.workflow/raw/design/01_deliverables.md`
   - 了解需要验证的交付件范围

4. **02 与硬件无关方案**：`.workflow/raw/design/02_platform_independent_design.md`
   - 地址偏移公式推导
   - 测试方案设计（§4）

5. **03 Tiling 方案**：`.workflow/raw/design/03_tiling_design.md`
   - Tiling 拦截校验逻辑（§2）
   - Tiling 计算方案（§3）
   - Tiling key 注册（§5）

6. **04 Kernel 方案**：`.workflow/raw/design/04_kernel_design.md`
   - Controller 层适配（§1）
   - VF 函数实现（§4）
   - 核心代码适配点汇总（§6）

## 用例类型和设计要求

### 1. Tiling 异常拦截用例

**目标**：验证 tiling 阶段的参数校验和错误拦截机制

**设计依据**：
- 03_tiling_design.md §2（Tiling 拦截校验）
- gmm_a16w4nd_op.md 约束条件表
- 脑图文档 UT 章节

**用例分类**：

#### 1.1 dtype 校验
- x dtype 非法（非 FLOAT16/BFLOAT16）
- weight dtype 非法（非 INT4）
- bias dtype 与 x 不匹配
- antiquantScale dtype 与 x 不一致
- antiquantOffset dtype 与 x 不一致
- groupList dtype 非法（非 INT64）

#### 1.2 shape 校验
- 维度校验（x/weight/bias/antiquantScale/groupList 维度错误）
- 数值约束（K % group_size != 0、group_size 非法、weight 最后一维非偶数等）
- shape 匹配（x.k 与 weight.k 不匹配、antiquantScale shape 不匹配等）

#### 1.3 format 校验
- x/weight/bias/antiquantScale/antiquantOffset format 非 ND

#### 1.4 属性校验
- groupType/groupListType/actType/splitItem 非法
- tuningConfigOptional 传入非空

#### 1.5 场景校验（组合参数）
- groupType 与 groupList 不匹配
- groupListType 与 groupList 内容不匹配
- splitItem 与 groupType 不匹配
- 多多多时 weight tensorList 转置状态不统一

**参数设置要求**：
- 每个用例需给出完整的输入参数（x/weight/bias/antiquantScale/antiquantOffset/groupList 的 dtype/format/shape）
- 明确标注预期错误类型（如 `ERR_INVALID_DTYPE`、`ERR_K_NOT_ALIGNED` 等）
- 给出用例编号和简要描述

### 2. 精度用例

**目标**：验证计算结果的正确性（与 CPU 参考实现对比）

**设计依据**：
- 02_platform_independent_design.md §1（数据流总体设计）
- gmm_a16w4nd_op.md 计算公式

**用例分类**：

#### 2.1 x=FLOAT16, weight=INT4
- 有 bias（FLOAT16）+ 有 antiquantOffset（FLOAT16）
- 有 bias（FLOAT16）+ 无 antiquantOffset（对称量化）
- 无 bias + 有 antiquantOffset
- 无 bias + 无 antiquantOffset

#### 2.2 x=BFLOAT16, weight=INT4
- 有 bias（BFLOAT16）+ 有 antiquantOffset（BFLOAT16）
- 有 bias（FLOAT32）+ 无 antiquantOffset
- 无 bias + 有 antiquantOffset
- 无 bias + 无 antiquantOffset

**参数设置要求**：
- 给出完整的 shape 设置（M/K/N/E/group_size）
- 给出 groupType（单单单/多多多）
- 给出转置状态
- 给出精度容差（如 rtol=1e-3, atol=1e-3）
- 给出用例编号和简要描述

### 3. 功能用例

**目标**：验证各种参数组合下的功能正确性

**设计依据**：
- gmm_a16w4nd_op.md 输入规格
- 脑图文档 ST 功能章节

**用例分类**：

#### 3.1 单单单（groupType=0）
- 非转置 + 各 group_size（32/64/128/256）
- 转置 + 各 group_size
- groupListType=0（cumsum）
- groupListType=1（count）
- shape 边界（M=1、K=group_size、N=1、E=1、E=1024）

#### 3.2 多多多（groupType=-1）
- 非转置 + 各 group_size
- 转置 + 各 group_size
- groupListType=0/1
- shape 边界（m_i=1、k_i=group_size、n_i=1、多 expert 数量 2/8/64/128、各 expert 的 m_i 不同）

**参数设置要求**：
- 给出完整的 shape 设置
- 给出 groupList 内容（单单单场景）
- 给出 bias/antiquantOffset 的有无设置
- 给出用例编号和简要描述

### 4. 性能用例

**目标**：验证关键场景下的性能表现

**设计依据**：
- 04_kernel_design.md §3（L1/UB 空间划分方案）
- 实际应用场景的典型 shape

**用例分类**：

#### 4.1 典型 MoE 场景
- 多多多 + 8 experts + 典型 shape（如 M=256, K=4096, N=4096）
- 多多多 + 64 experts + 典型 shape

#### 4.2 大 shape 场景
- M/K/N 较大值（如 1024/8192/8192）
- 验证 tiling 切分和 kernel 执行效率

#### 4.3 小 shape 场景
- M/K/N 较小值（如 1/32/32）
- 验证尾块处理和资源利用率

**参数设置要求**：
- 给出完整的 shape 设置
- 给出 groupType/group_size/转置状态
- 给出性能指标预期（如执行时间、GFLOPS）
- 给出用例编号和简要描述

### 5. 泛化用例

**目标**：验证边界值和特殊场景

**设计依据**：
- gmm_a16w4nd_op.md 约束条件
- 脑图文档 ST 功能章节 shape 边界

**用例分类**：

#### 5.1 group_size 边界
- group_size=32（最小）
- group_size=256（最大）
- K=group_size（单 group）
- K=group_size*10（多 group）

#### 5.2 shape 边界
- M=1（极小 M）
- N=1（极小 N）
- K=group_size（最小 K）
- E=1（单 expert）
- E=1024（最大 expert）

#### 5.3 特殊组合
- 多多多 + 各 expert 的 m_i 不同
- 多多多 + 各 expert 的 k_i 不同（但 k_i/group_size 相同）
- 单单单 + groupListType=0 + 非均匀分组
- 单单单 + groupListType=1 + 非均匀分组

**参数设置要求**：
- 给出完整的 shape 设置
- 给出边界值的具体数值
- 给出用例编号和简要描述

### 6. Kernel 打桩验证用例

**目标**：通过打桩（stub/mock）调试方式，逐步验证 kernel 中每个搬运和计算逻辑的正确性。这些用例用于开发阶段的调试，帮助定位数据搬运或计算错误的具体位置。

**设计依据**：
- 04_kernel_design.md §1（Controller 层适配）
- 04_kernel_design.md §3（L1/UB 空间划分方案）
- 04_kernel_design.md §4（Vec 反量化函数实现）
- gmm_a16w8_kernel_walkthrough.md（AIC/AIV 协作流程）

**打桩验证策略**：
- 在 kernel 代码的关键位置插入打桩点（输出中间结果到 GM）
- 使用简单的、可预测的输入数据（如全 1、等差数列等）
- 逐步验证每个阶段的输出是否符合预期

**用例分类**：

#### 6.1 数据搬运验证（MTE 阶段）

**MTE2 搬运验证**：
- **KS-01**: Weight INT4 GM→UB 搬运验证
  - 输入：weight 使用等差数列（0,1,2,...,15 重复）
  - 打桩点：CopyWeightGmToUb 后输出 UB 内容到 GM
  - 验证：UB 中 INT4 数据与 GM 中原始数据一致
  
- **KS-02**: Scale GM→UB 搬运验证（pergroup 2D copy）
  - 输入：scale 使用可预测值（如 scale[g,n] = g*100 + n）
  - 打桩点：CopyAntiQuantParamsGmToUb 后输出 UB 内容到 GM
  - 验证：UB 中 scale 布局正确（shape = (kL1/gs, nL1)）
  - 参数：group_size=128, kL1=256, nL1=256 → scale shape = (2, 256)
  
- **KS-03**: Offset GM→UB 搬运验证（pergroup 2D copy）
  - 输入：offset 使用可预测值（如 offset[g,n] = g*10 + n）
  - 打桩点：CopyAntiQuantParamsGmToUb 后输出 UB 内容到 GM
  - 验证：UB 中 offset 布局正确

**MTE3 搬运验证**：
- **KS-04**: 反量化后 weight UB→L1 搬运验证
  - 输入：已知 UB 中的 FP16 weight 数据
  - 打桩点：WeightHighBitUbToL1 后输出 L1 内容到 GM
  - 验证：L1 中数据与 UB 中数据一致（考虑 NZ 排布转换）

**MTE1 搬运验证**：
- **KS-05**: A 矩阵 GM→L1 搬运验证
  - 输入：A 使用等差数列
  - 打桩点：CopyAAndBiasGmToL1 后输出 L1 内容到 GM
  - 验证：AL1 中数据与 GM 中原始数据一致
  
- **KS-06**: Bias GM→L1 搬运验证
  - 输入：bias 使用可预测值（如 bias[n] = n）
  - 打桩点：CopyAAndBiasGmToL1 后输出 BiasL1 内容到 GM
  - 验证：BiasL1 中数据与 GM 中原始数据一致

**输出写回验证**：
- **KS-07**: C 矩阵 L1→GM 写回验证
  - 输入：已知 L1 中的 C 数据
  - 打桩点：GetTensorC 后检查 GM 中输出
  - 验证：GM 中输出数据与 L1 中数据一致

#### 6.2 计算逻辑验证（VF 阶段）

**INT4 unpack 验证**：
- **KS-08**: INT4 unpack 正确性验证
  - 输入：weight INT4 使用已知值（如 0,1,2,...,7,-8,-7,...,-1）
  - 打桩点：DataCopy UNPACK4 后输出 unpack 结果到 GM
  - 验证：unpack 后的 INT8 值与预期一致（每字节拆分为 2 个 INT4）

**INT4→FP16 cast 验证**：
- **KS-09**: INT4→FP16 cast 正确性验证
  - 输入：unpack 后的 INT8 数据
  - 打桩点：Cast 后输出 FP16 结果到 GM
  - 验证：FP16 值与 INT8 值数值相等

**PerGroup scale 加载验证**：
- **KS-10**: NK 排布 scale 广播加载验证
  - 输入：scale[g,n] = g*100 + n
  - 打桩点：NdNkLoadScaleOffset 后输出 scale VREG 到 GM
  - 验证：每个 weight 元素对应的 scale 值正确（broadcast 模式）
  - 参数：group_size=128, nOffset=0, kOffset=0
  
- **KS-11**: KN 排布 scale 直接加载验证
  - 输入：scale[g,n] = g*100 + n
  - 打桩点：NdKnLoadScaleOffset 后输出 scale VREG 到 GM
  - 验证：每个 weight 元素对应的 scale 值正确（direct 模式）

**PerGroup offset 加载验证**：
- **KS-12**: NK 排布 offset 广播加载验证
  - 输入：offset[g,n] = g*10 + n
  - 打桩点：NdNkLoadScaleOffset 后输出 offset VREG 到 GM
  - 验证：每个 weight 元素对应的 offset 值正确

**反量化计算验证**：
- **KS-13**: 反量化公式验证（有 offset）
  - 输入：weight=2, scale=1.5, offset=1 → 预期 (2+1)*1.5 = 4.5
  - 打桩点：Add + Mul 后输出结果到 GM
  - 验证：结果与预期一致（容差 1e-3）
  
- **KS-14**: 反量化公式验证（无 offset，对称量化）
  - 输入：weight=2, scale=1.5 → 预期 2*1.5 = 3.0
  - 打桩点：Mul 后输出结果到 GM
  - 验证：结果与预期一致

#### 6.3 地址偏移验证

**SetGmAddr 偏移验证**：
- **KS-15**: PerGroup scale 偏移公式验证（单单单）
  - 输入：E=4, K=512, N=256, group_size=128
  - 打桩点：SetGmAddr 后输出 antiquantParamsBaseOffset 到 GM
  - 验证：每个 group 的偏移 = (K/gs)*N = 4*256 = 1024 元素
  - 预期：group0=0, group1=1024, group2=2048, group3=3072
  
- **KS-16**: Bias 偏移分离验证（单单单）
  - 输入：E=4, N=256
  - 打桩点：SetGmAddr 后输出 biasBaseOffset 到 GM
  - 验证：每个 group 的 bias 偏移 = N = 256 元素
  - 预期：group0=0, group1=256, group2=512, group3=768
  
- **KS-17**: INT4 weight 偏移 >>1 验证（单单单）
  - 输入：E=4, K=512, N=256
  - 打桩点：SetGmAddr 后输出 weightBaseOffset 到 GM
  - 验证：每个 group 的 weight 偏移 = (K*N)>>1 = 65536 字节
  - 预期：group0=0, group1=65536, group2=131072, group3=196608

#### 6.4 Tiling 参数传递验证

**groupSize 传递验证**：
- **KS-18**: groupSize 从 tiling 到 kernel 传递验证
  - 输入：group_size=128
  - 打桩点：Controller Init 后输出 gmmBaseTiling_->groupSize 到 GM
  - 验证：kernel 中读取的 groupSize 与 tiling 中设置的值一致

**mList/kList/nList 验证**：
- **KS-19**: mList/kList/nList 值验证（多多多）
  - 输入：E=4, 各 expert 的 m/k/n 不同
  - 打桩点：Process 循环中输出 mListGm_/kListGm_/nListGm_ 到 GM
  - 验证：kernel 中读取的 list 值与 tiling 中设置的值一致

**baseM/baseN/baseK 验证**：
- **KS-20**: baseM/baseN/baseK 动态调整验证
  - 输入：M=100, N=200（小于默认 baseM=256, baseN=256）
  - 打桩点：Process 中动态调整后输出 baseM/baseN 到 GM
  - 验证：调整后的值符合预期（M=100 < 256 → baseM=50）

#### 6.5 Matmul 计算验证

**Cube 计算验证**：
- **KS-21**: Matmul 基础计算验证
  - 输入：A 全 1, W_dequant 全 1, bias 全 0
  - 打桩点：LaunchMatmul 后输出 C 到 GM
  - 验证：C[m,n] = K（累加 K 次 1*1）
  - 参数：M=16, K=64, N=16
  
- **KS-22**: Matmul + bias 验证
  - 输入：A 全 1, W_dequant 全 1, bias[n]=n
  - 打桩点：LaunchMatmul 后输出 C 到 GM
  - 验证：C[m,n] = K + n

**参数设置要求**：
- 给出完整的 shape 设置（M/K/N/E/group_size）
- 给出打桩点位置（函数名 + 代码行号或逻辑位置）
- 给出输入数据的具体值（便于预测输出）
- 给出预期输出值或验证方法
- 给出用例编号（KS-XX）和简要描述
- 标注验证阶段（MTE/VF/Address/Tiling/Matmul）

### 7. Tiling 白盒用例

**目标**：基于 tiling 代码路径设计端到端用例，验证 tiling 计算过程中各分支/函数的中间结果正确性。不需要插桩，通过检查 tiling 输出结构体（`GMMWeightQuantParam`）的字段值来验证。

**设计依据**：
- 03_tiling_design.md §1（Tiling Data 设计）
- 03_tiling_design.md §2（Tiling 拦截校验）
- 03_tiling_design.md §3（Tiling 计算方案）
- 03_tiling_design.md §4（Shape List 切分）
- gmm_a16w8_tiling_walkthrough.md（SetTiling 完整流程）

**用例分类**：

#### 7.1 IsA16W4NDPergroup 判断路径

- **TW-01**: pergroup 场景正确识别
  - 输入：antiquantScale shape = (E, G, N) = (4, 4, 256)，3 维
  - 覆盖路径：`IsA16W4NDPergroup()` → `groupSize_ > 0` 为 true
  - 验证：tiling 输出中 `groupSize = 128`（K=512, G=4 → gs=128）
  
- **TW-02**: perchannel 场景不误判为 pergroup
  - 输入：antiquantScale shape = (E, N) = (4, 256)，2 维
  - 覆盖路径：`IsA16W4NDPergroup()` → `groupSize_ == 0` 为 true
  - 验证：tiling 输出中 `groupSize = 0`

- **TW-03**: 多多多 pergroup 2 维 scale 正确识别
  - 输入：多多多，antiquantScale shape = (G_i, N_i) = (4, 256)，2 维，K=512
  - 覆盖路径：`SetAntiquantGroupSize()` 中 `antiquantScaleDimNum == 2 && IsA16W4ND()` 分支
  - 验证：`groupSize = 128`（K/G = 512/4 = 128）

#### 7.2 SetAntiquantGroupSize 路径

- **TW-04**: 单单单 3 维 scale groupSize 推导
  - 输入：单单单，scale shape = (4, 8, 256)，K=1024
  - 覆盖路径：`SetAntiquantGroupSize()` 中 `antiquantScaleDimNum == 3` 分支
  - 验证：`groupSize = 128`（K/G = 1024/8 = 128）

- **TW-05**: group_size=32 场景
  - 输入：K=256，scale shape = (4, 8, 256)（G=8 → gs=32）
  - 覆盖路径：`CheckGroupSize()` 中 `groupSize_ == 32` 分支
  - 验证：`groupSize = 32`，校验通过

- **TW-06**: group_size=64 场景
  - 输入：K=256，scale shape = (4, 4, 256)（G=4 → gs=64）
  - 覆盖路径：`CheckGroupSize()` 中 `groupSize_ == 64` 分支
  - 验证：`groupSize = 64`

- **TW-07**: group_size=256 场景
  - 输入：K=256，scale shape = (4, 1, 256)（G=1 → gs=256）
  - 覆盖路径：`CheckGroupSize()` 中 `groupSize_ == 256` 分支
  - 验证：`groupSize = 256`

#### 7.3 SetTilingKey 路径

- **TW-08**: pergroup antiquantType 设置
  - 输入：pergroup 场景（groupSize > 0）
  - 覆盖路径：`SetTilingKey()` 中 `IsA16W4NDPergroup()` → `antiquantType = PER_GROUP`
  - 验证：tiling key 中 `W_QUANT_TYPE = WQGMM_PER_GROUP`

- **TW-09**: perchannel antiquantType 设置（对比）
  - 输入：perchannel 场景（groupSize == 0）
  - 覆盖路径：`SetTilingKey()` 中 else 分支 → `antiquantType = PER_CHANNEL`
  - 验证：tiling key 中 `W_QUANT_TYPE = WQGMM_PER_CHANNEL`

- **TW-10**: 转置场景 tiling key
  - 输入：pergroup + 转置
  - 覆盖路径：`SetTilingKey()` 中 `transB_ == true` → `WQ_B_TRANS = WQGMM_TRANS`
  - 验证：tiling key 中 `WQ_B_TRANS = 1`

- **TW-11**: 非转置场景 tiling key
  - 输入：pergroup + 非转置
  - 覆盖路径：`SetTilingKey()` 中 `transB_ == false` → `WQ_B_TRANS = WQGMM_NO_TRANS`
  - 验证：tiling key 中 `WQ_B_TRANS = 0`

#### 7.4 SetShapeList 路径

- **TW-12**: 单单单 shape list 填充
  - 输入：单单单，x(256,512), weight(4,256,512), E=4
  - 覆盖路径：`SetShapeListSplitMSingleXSingleWeightSingleY()`
  - 验证：`groupNum=4, mSize=256, kSize=512, nSize=256, kList[0]=512, nList[0]=256, mList[0]=-1`

- **TW-13**: 多多多 shape list 填充
  - 输入：多多多，E=4，各 expert 的 m/k/n 不同
  - 覆盖路径：`SetShapeListMultiXMultiWeightMultiY()`
  - 验证：`mList[i]`、`kList[i]`、`nList[i]` 逐 tensor 正确填充

- **TW-14**: INT4 weight nSize 扩展
  - 输入：单单单，非转置，weight shape = (4, 32, 512)（INT4 存储，N 实际 = 32*8 = 256）
  - 覆盖路径：`SetShapeListSplitMSingleXSingleWeightSingleY()` 中 `weightDtype_ == DT_INT4 && !transB_` → `nSize_ = 8 * nSize_`
  - 验证：`nSize = 256`

#### 7.5 baseM/baseN/baseK 计算路径

- **TW-15**: 默认 base 值（无 bias）
  - 输入：pergroup，无 bias，M=512, N=512
  - 覆盖路径：`SetBaseTiling()` 中默认值设置
  - 验证：`baseM=256, baseN=256, baseK=64`

- **TW-16**: 有 bias 时 baseM 调整
  - 输入：pergroup，有 bias，M=512, N=512
  - 覆盖路径：`SetBaseTiling()` 中 `hasBias` 分支 → `baseM=240`
  - 验证：`baseM=240`

#### 7.6 CalcResplitTiling 路径

- **TW-17**: 多多多 ND 不使能尾块重切分
  - 输入：多多多 + ND
  - 覆盖路径：`EnableTailResplit()` → 多多多 → 不使能
  - 验证：`mainBlockSize = 0`（不使能标志）

- **TW-18**: 单单单 ND 非转置不使能尾块重切分
  - 输入：单单单 + ND + 非转置
  - 覆盖路径：`EnableTailResplit()` → 单单单 + ND + B非转置 → 不使能
  - 验证：`mainBlockSize = 0`

#### 7.7 GMMWeightQuantParam 字段完整性

- **TW-19**: 完整参数传递验证（单单单）
  - 输入：单单单，E=4, M=256, K=512, N=256, gs=128, 有 bias, 非转置
  - 覆盖路径：`SetTiling()` 完整流程
  - 验证：`groupNum=4, coreNum>0, kSize=512, nSize=256, singleX=1, singleWeight=1, singleY=1, groupType=0, groupListType=0, hasBias=1, groupSize=128`

- **TW-20**: 完整参数传递验证（多多多）
  - 输入：多多多，E=4, 各 expert m_i/k_i/n_i 不同, gs=128, 无 bias, 非转置
  - 覆盖路径：`SetTiling()` 完整流程
  - 验证：`singleX=0, singleWeight=0, singleY=0, groupType=-1, hasBias=0, groupSize=128`

**参数设置要求**：
- 给出完整的 shape 设置（M/K/N/E/group_size）
- 给出 groupType（单单单/多多多）和转置状态
- 明确标注**覆盖的代码路径**（函数名 + 分支条件）
- 给出 tiling 输出结构体中的预期字段值
- 给出用例编号（TW-XX）和简要描述

### 8. Kernel 白盒用例

**目标**：基于 kernel 代码路径设计端到端用例，验证各分支/分发逻辑的正确性。不需要插桩，通过端到端执行并检查最终输出来间接验证特定代码路径被正确执行。

**设计依据**：
- 04_kernel_design.md §1（Controller 层适配）
- 04_kernel_design.md §2（BasicBlock 层适配）
- 04_kernel_design.md §3（L1/UB 空间划分方案）
- 04_kernel_design.md §4（Vec 反量化函数实现）
- 01_deliverables.md §4（grouped_matmul_apt.cpp kernel 入口适配）
- gmm_a16w8_kernel_walkthrough.md（Controller 完整流程）

**用例分类**：

#### 8.1 grouped_matmul_apt.cpp if constexpr 分支覆盖

- **KW-01**: pergroup 无offset无bias 非转置分支
  - 输入：pergroup, 无 offset, 无 bias, 非转置
  - 覆盖路径：`V310_GMM_ANTI_QUANT` → `W_TYPE==WQGMM_ND && OFFSET_OR_BIAS_EXIT==NOT_EXIST_BIAS_NOT_EXIST && W_QUANT_TYPE==WQGMM_PER_GROUP && WQ_B_TRANS==WQGMM_NO_TRANS`
  - 验证：端到端计算正确，确认使用了 `A16W4_NDKN_PG` 配置
  
- **KW-02**: pergroup 无offset有bias 非转置分支
  - 输入：pergroup, 无 offset, 有 bias, 非转置
  - 覆盖路径：`OFFSET_OR_BIAS_EXIT==NOT_EXIST_BIAS_EXIST`
  - 验证：端到端计算正确

- **KW-03**: pergroup 有offset无bias 非转置分支
  - 输入：pergroup, 有 offset, 无 bias, 非转置
  - 覆盖路径：`OFFSET_OR_BIAS_EXIT==EXIST_BIAS_NOT_EXIST`
  - 验证：端到端计算正确

- **KW-04**: pergroup 有offset有bias 非转置分支
  - 输入：pergroup, 有 offset, 有 bias, 非转置
  - 覆盖路径：`OFFSET_OR_BIAS_EXIT==EXIST_BIAS_EXIST`
  - 验证：端到端计算正确

- **KW-05**: pergroup 无offset无bias 转置分支
  - 输入：pergroup, 无 offset, 无 bias, 转置
  - 覆盖路径：`WQ_B_TRANS==WQGMM_TRANS` → 使用 `A16W4_NDNK_PG` 配置
  - 验证：端到端计算正确

- **KW-06**: pergroup 有offset有bias 转置分支
  - 输入：pergroup, 有 offset, 有 bias, 转置
  - 覆盖路径：`WQ_B_TRANS==WQGMM_TRANS && OFFSET_OR_BIAS_EXIT==EXIST_BIAS_EXIST`
  - 验证：端到端计算正确

#### 8.2 ComputeBasicBlock 分发路径

- **KW-07**: ND 非转置 → NdKn 路径
  - 输入：pergroup, ND, 非转置
  - 覆盖路径：`ComputeBasicBlock()` → `(ND && !bTrans)` → `ComputeBasicBlockAivNdKnNzNk()`
  - 验证：端到端计算正确（K 为内轴路径）

- **KW-08**: ND 转置 → NdNk 路径
  - 输入：pergroup, ND, 转置
  - 覆盖路径：`ComputeBasicBlock()` → else → `ComputeBasicBlockAivNdNkNzKn()`
  - 验证：端到端计算正确（N 为内轴路径）

#### 8.3 GetBufferConfig 分支覆盖

- **KW-09**: pergroup ND buffer 配置
  - 输入：pergroup, ND
  - 覆盖路径：`GetBufferConfig()` → `weightFormat==ND && antiQuantType==PER_GROUP` → `GetNdPerGroupBufferInfo()`
  - 验证：端到端计算正确（scale UB = 8KB 配置被使用）

- **KW-10**: perchannel ND buffer 配置（对比，确保不影响）
  - 输入：perchannel, ND
  - 覆盖路径：`GetBufferConfig()` → `weightFormat==ND && antiQuantType!=PER_GROUP` → `GetNdBufferInfo()`
  - 验证：端到端计算正确（原有 perchannel 路径不受影响）

#### 8.4 VF 函数 groupSize 运行时分支覆盖

- **KW-11**: groupSize=32 VF 分支
  - 输入：pergroup, gs=32, K=256
  - 覆盖路径：`AntiQuantInt4NdNkPerGroup()` → `groupSize == 32` 分支
  - 验证：端到端计算正确

- **KW-12**: groupSize=64 VF 分支
  - 输入：pergroup, gs=64, K=256
  - 覆盖路径：`AntiQuantInt4NdNkPerGroup()` → `groupSize == 64` 分支
  - 验证：端到端计算正确

- **KW-13**: groupSize=128 VF 分支
  - 输入：pergroup, gs=128, K=512
  - 覆盖路径：`AntiQuantInt4NdNkPerGroup()` → `groupSize == 128` 分支
  - 验证：端到端计算正确

- **KW-14**: groupSize=256 VF 分支
  - 输入：pergroup, gs=256, K=512
  - 覆盖路径：`AntiQuantInt4NdNkPerGroup()` → `groupSize == 256` 分支
  - 验证：端到端计算正确

#### 8.5 CopyAntiQuantParamsGmToUb PER_GROUP 分支

- **KW-15**: pergroup scale 2D copy（多多多）
  - 输入：多多多, pergroup, scale shape = (G_i, N_i)
  - 覆盖路径：`CopyAntiQuantParamsGmToUb()` → `PER_GROUP` 分支 → 2D copy
  - 验证：端到端计算正确（scale 正确加载到 UB）

- **KW-16**: pergroup scale 2D copy（单单单）
  - 输入：单单单, pergroup, scale shape = (E, G, N)
  - 覆盖路径：`CopyAntiQuantParamsGmToUb()` → `PER_GROUP` 分支 → 2D copy（单单单偏移）
  - 验证：端到端计算正确

#### 8.6 SetGmAddr pergroup 偏移路径

- **KW-17**: pergroup 单单单多 group 偏移序列
  - 输入：单单单, E=8, K=512, N=256, gs=128
  - 覆盖路径：`SetGmAddr()` → `if constexpr (PER_GROUP)` → `antiquantParamsBaseOffset += (kSize/gs) * nSize`
  - 验证：端到端计算正确（8 个 group 的 scale 偏移序列正确）

- **KW-18**: pergroup bias 偏移分离
  - 输入：单单单, E=4, K=512, N=256, gs=128, 有 bias
  - 覆盖路径：`SetGmAddr()` → `biasBaseOffset += nSize`（独立于 antiquantParamsBaseOffset）
  - 验证：端到端计算正确（bias 和 scale 偏移独立）

#### 8.7 AIC/AIV 协作路径

- **KW-19**: 多 BasicBlock 迭代（M > baseM）
  - 输入：多多多, M=1024, N=256, K=512, gs=128
  - 覆盖路径：`Process()` 中 baseM 动态调整 → 多个 BasicBlock 迭代
  - 验证：端到端计算正确（多块拼接正确）

- **KW-20**: 多 BasicBlock 迭代（N > baseN）
  - 输入：多多多, M=256, N=1024, K=512, gs=128
  - 覆盖路径：`Process()` 中 baseN 动态调整 → N 方向多块
  - 验证：端到端计算正确

- **KW-21**: K 方向多 kBlock 迭代
  - 输入：多多多, M=256, N=256, K=2048, gs=128
  - 覆盖路径：`WeightAntiQuantCompute()` 中 K 方向多 kBlock 循环
  - 验证：端到端计算正确（K 方向累加正确）

- **KW-22**: 单 BasicBlock（M ≤ baseM, N ≤ baseN）
  - 输入：多多多, M=64, N=64, K=128, gs=128
  - 覆盖路径：`Process()` 中单块直接计算，无迭代
  - 验证：端到端计算正确

**参数设置要求**：
- 给出完整的 shape 设置（M/K/N/E/group_size）
- 给出 groupType（单单单/多多多）和转置状态
- 明确标注**覆盖的代码路径**（函数名 + 分支条件）
- 给出预期结果（端到端计算正确，与 CPU 参考实现对比）
- 给出用例编号（KW-XX）和简要描述
- 标注精度容差（rtol/atol）

## 产出要求

### 1. 用例设计文档

**文件路径**：`.workflow/raw/design/05_test_case_design.md`

**内容结构**：
- 用例设计总体说明
- 各类型用例的详细列表（编号、描述、参数设置、预期结果）
- 用例覆盖度分析（对照算子规格文档的约束条件）
- 用例优先级标注（P0/P1/P2）

### 2. Excel 表格

**文件路径**：`.workflow/raw/design/test_cases_gmm_a16w4_pergroup_nd.xlsx`

**Sheet 结构**：
- Sheet 1: `tiling_exception` — Tiling 异常拦截用例（黑盒）
- Sheet 2: `precision` — 精度用例（黑盒）
- Sheet 3: `functional` — 功能用例（黑盒）
- Sheet 4: `performance` — 性能用例（黑盒）
- Sheet 5: `generalization` — 泛化用例（黑盒）
- Sheet 6: `kernel_stub` — Kernel 打桩验证用例（白盒，插桩级）
- Sheet 7: `tiling_whitebox` — Tiling 白盒用例（白盒，端到端级）
- Sheet 8: `kernel_whitebox` — Kernel 白盒用例（白盒，端到端级）

**每个 Sheet 的列定义**：

**Sheet 1-5 通用列**：
- `case_id`: 用例编号（如 T-01、P-01、F-01 等）
- `description`: 用例描述
- `group_type`: groupType（-1/0）
- `x_dtype`: x 的 dtype
- `weight_dtype`: weight 的 dtype
- `bias_dtype`: bias 的 dtype（可为 null）
- `antiquant_scale_dtype`: antiquantScale 的 dtype
- `antiquant_offset_dtype`: antiquantOffset 的 dtype（可为 null）
- `group_list_dtype`: groupList 的 dtype
- `x_shape`: x 的 shape（如 "256,512"）
- `weight_shape`: weight 的 shape
- `bias_shape`: bias 的 shape（可为空）
- `antiquant_scale_shape`: antiquantScale 的 shape
- `antiquant_offset_shape`: antiquantOffset 的 shape（可为空）
- `group_list_shape`: groupList 的 shape
- `group_size`: group_size 值
- `transpose_weight`: 是否转置（0/1）
- `group_list_type`: groupListType（0/1）
- `group_list_content`: groupList 内容（单单单场景，如 "128,256,384"）
- `expected_result`: 预期结果（如 "PASS"、"ERR_INVALID_DTYPE" 等）
- `priority`: 优先级（P0/P1/P2）
- `remarks`: 备注

**Sheet 6 (kernel_stub) 专用列**：
- `case_id`: 用例编号（KS-XX）
- `description`: 用例描述
- `verification_stage`: 验证阶段（MTE2/MTE3/MTE1/VF/Address/Tiling/Matmul）
- `stub_location`: 打桩点位置（函数名 + 逻辑位置）
- `input_data_pattern`: 输入数据模式（如 "等差数列"、"全1"、"g*100+n"）
- `x_shape`: x 的 shape
- `weight_shape`: weight 的 shape
- `antiquant_scale_shape`: antiquantScale 的 shape
- `antiquant_offset_shape`: antiquantOffset 的 shape（可为空）
- `group_size`: group_size 值
- `transpose_weight`: 是否转置（0/1）
- `group_type`: groupType（-1/0）
- `expected_output`: 预期输出值或验证方法
- `priority`: 优先级（P0/P1/P2）
- `remarks`: 备注

**Sheet 7 (tiling_whitebox) 专用列**：
- `case_id`: 用例编号（TW-XX）
- `description`: 用例描述
- `code_path`: 覆盖的代码路径（函数名 + 分支条件）
- `group_type`: groupType（-1/0）
- `x_dtype`: x 的 dtype
- `weight_dtype`: weight 的 dtype
- `bias_dtype`: bias 的 dtype（可为 null）
- `antiquant_scale_dtype`: antiquantScale 的 dtype
- `antiquant_offset_dtype`: antiquantOffset 的 dtype（可为 null）
- `x_shape`: x 的 shape
- `weight_shape`: weight 的 shape
- `antiquant_scale_shape`: antiquantScale 的 shape
- `antiquant_offset_shape`: antiquantOffset 的 shape（可为空）
- `group_size`: group_size 值
- `transpose_weight`: 是否转置（0/1）
- `group_list_type`: groupListType（0/1）
- `expected_tiling_fields`: 预期 tiling 输出字段值（如 "groupSize=128, groupNum=4"）
- `priority`: 优先级（P0/P1/P2）
- `remarks`: 备注

**Sheet 8 (kernel_whitebox) 专用列**：
- `case_id`: 用例编号（KW-XX）
- `description`: 用例描述
- `code_path`: 覆盖的代码路径（函数名 + 分支条件）
- `group_type`: groupType（-1/0）
- `x_dtype`: x 的 dtype
- `weight_dtype`: weight 的 dtype
- `bias_dtype`: bias 的 dtype（可为 null）
- `antiquant_scale_dtype`: antiquantScale 的 dtype
- `antiquant_offset_dtype`: antiquantOffset 的 dtype（可为 null）
- `x_shape`: x 的 shape
- `weight_shape`: weight 的 shape
- `antiquant_scale_shape`: antiquantScale 的 shape
- `antiquant_offset_shape`: antiquantOffset 的 shape（可为空）
- `group_size`: group_size 值
- `transpose_weight`: 是否转置（0/1）
- `expected_result`: 预期结果（如 "PASS"）
- `rtol`: 相对精度容差
- `atol`: 绝对精度容差
- `priority`: 优先级（P0/P1/P2）
- `remarks`: 备注

### 3. 用例生成脚本

**文件路径**：`.workflow/raw/design/generate_test_cases.py`

**功能要求**：
- 读取用例设计文档或硬编码的用例列表
- 使用 `openpyxl` 或 `pandas` 生成 Excel 文件
- 每个用例类型对应一个 sheet
- 支持命令行参数指定输出路径

**脚本结构**：
```python
#!/usr/bin/env python3
"""
GMM A16W4 PerGroup ND 测试用例生成脚本
"""

import pandas as pd
# 或 import openpyxl

def generate_tiling_exception_cases():
    """生成 tiling 异常拦截用例"""
    cases = []
    # ... 添加用例
    return cases

def generate_precision_cases():
    """生成精度用例"""
    cases = []
    # ... 添加用例
    return cases

def generate_functional_cases():
    """生成功能用例"""
    cases = []
    # ... 添加用例
    return cases

def generate_performance_cases():
    """生成性能用例"""
    cases = []
    # ... 添加用例
    return cases

def generate_generalization_cases():
    """生成泛化用例"""
    cases = []
    # ... 添加用例
    return cases

def generate_kernel_stub_cases():
    """生成 kernel 打桩验证用例"""
    cases = []
    # ... 添加用例（KS-01 到 KS-22）
    return cases

def generate_tiling_whitebox_cases():
    """生成 tiling 白盒用例"""
    cases = []
    # ... 添加用例（TW-01 到 TW-20）
    return cases

def generate_kernel_whitebox_cases():
    """生成 kernel 白盒用例"""
    cases = []
    # ... 添加用例（KW-01 到 KW-22）
    return cases

def main():
    output_path = "test_cases_gmm_a16w4_pergroup_nd.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        tiling_cases = generate_tiling_exception_cases()
        pd.DataFrame(tiling_cases).to_excel(writer, sheet_name='tiling_exception', index=False)
        
        precision_cases = generate_precision_cases()
        pd.DataFrame(precision_cases).to_excel(writer, sheet_name='precision', index=False)
        
        functional_cases = generate_functional_cases()
        pd.DataFrame(functional_cases).to_excel(writer, sheet_name='functional', index=False)
        
        performance_cases = generate_performance_cases()
        pd.DataFrame(performance_cases).to_excel(writer, sheet_name='performance', index=False)
        
        generalization_cases = generate_generalization_cases()
        pd.DataFrame(generalization_cases).to_excel(writer, sheet_name='generalization', index=False)
        
        kernel_stub_cases = generate_kernel_stub_cases()
        pd.DataFrame(kernel_stub_cases).to_excel(writer, sheet_name='kernel_stub', index=False)
        
        tiling_whitebox_cases = generate_tiling_whitebox_cases()
        pd.DataFrame(tiling_whitebox_cases).to_excel(writer, sheet_name='tiling_whitebox', index=False)
        
        kernel_whitebox_cases = generate_kernel_whitebox_cases()
        pd.DataFrame(kernel_whitebox_cases).to_excel(writer, sheet_name='kernel_whitebox', index=False)
    
    print(f"Generated {output_path}")

if __name__ == "__main__":
    main()
```

## 完成标准

- [ ] 用例设计文档覆盖所有 8 种用例类型（5 黑盒 + 1 插桩白盒 + 2 端到端白盒）
- [ ] 每个用例有完整的参数设置（dtype/format/shape/group_size 等）
- [ ] Tiling 异常拦截用例覆盖算子规格文档中的所有约束条件
- [ ] 精度用例覆盖所有 dtype 组合（FLOAT16/BFLOAT16 × bias/offset 有无）
- [ ] 功能用例覆盖单单单/多多多 × 转置/非转置 × 各 group_size
- [ ] 性能用例覆盖典型 MoE 场景和大/小 shape 场景
- [ ] 泛化用例覆盖边界值和特殊组合
- [ ] Kernel 打桩验证用例覆盖 MTE 搬运、VF 计算、地址偏移、tiling 参数传递、matmul 计算等关键阶段
- [ ] Tiling 白盒用例覆盖 IsA16W4NDPergroup、SetAntiquantGroupSize、SetTilingKey、SetShapeList、baseM/baseN/baseK、CalcResplitTiling 等代码路径
- [ ] Kernel 白盒用例覆盖 if constexpr 分支、ComputeBasicBlock 分发、GetBufferConfig 分支、VF groupSize 分支、CopyAntiQuantParamsGmToUb PER_GROUP 分支、SetGmAddr pergroup 偏移、AIC/AIV 协作等代码路径
- [ ] 每个白盒用例明确标注覆盖的代码路径（函数名 + 分支条件）
- [ ] Excel 文件包含 8 个 sheet，每个 sheet 列定义完整
- [ ] Python 脚本可执行，生成的 Excel 文件与文档一致
- [ ] 用例编号规范，描述清晰，优先级标注合理

## 执行步骤

1. **读取所有必读文档**，理解算子规格、约束条件和实现方案
2. **设计 Tiling 异常拦截用例**，覆盖所有约束条件
3. **设计精度用例**，覆盖所有 dtype 组合
4. **设计功能用例**，覆盖所有参数组合
5. **设计性能用例**，覆盖典型场景
6. **设计泛化用例**，覆盖边界值
7. **设计 Kernel 打桩验证用例**，覆盖 MTE 搬运、VF 计算、地址偏移、tiling 参数传递、matmul 计算等关键阶段
8. **设计 Tiling 白盒用例**，基于 tiling 代码路径设计端到端用例（IsA16W4NDPergroup、SetAntiquantGroupSize、SetTilingKey、SetShapeList、baseM/baseN/baseK、CalcResplitTiling）
9. **设计 Kernel 白盒用例**，基于 kernel 代码路径设计端到端用例（if constexpr 分支、ComputeBasicBlock 分发、GetBufferConfig 分支、VF groupSize 分支、CopyAntiQuantParamsGmToUb PER_GROUP 分支、SetGmAddr pergroup 偏移、AIC/AIV 协作）
10. **编写用例设计文档**（05_test_case_design.md）
11. **编写 Python 脚本**（generate_test_cases.py）
12. **执行脚本生成 Excel 文件**（test_cases_gmm_a16w4_pergroup_nd.xlsx）
13. **检查文档和 Excel 的一致性**

## 注意事项

- **严格按照算子规格文档的约束条件设计用例**，不要凭空猜测
- **参数设置要具体**，不要使用模糊描述（如"较大值"），要给出具体数值
- **用例编号要规范**，便于后续引用和追踪
- **优先级标注要合理**：
  - P0：核心功能，必须通过
  - P1：重要功能，应该通过
  - P2：边界场景，最好通过
- **Excel 文件要便于后续使用**，列名清晰，数据格式统一
