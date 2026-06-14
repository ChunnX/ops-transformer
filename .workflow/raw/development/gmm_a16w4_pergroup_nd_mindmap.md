# GMM A16W4 PerGroup ND 脑图

## 输入

### Tensor 参数

- x（必选, ParamType=DYNAMIC）
  - dtype
    - FLOAT16
    - BFLOAT16
  - format: ND
  - shape
    - 多多多（groupType=-1）: 2维 (m_i, k_i)
    - 单单单（groupType=0）: 2维 (M, K)
  - 约束: 固定不转置
- weight（必选, ParamType=DYNAMIC）
  - dtype: INT4
  - format: ND
  - shape
    - 多多多（groupType=-1）
      - 非转置: 2维 (n_i, k_i)
      - 转置: 2维 (k_i, n_i)
    - 单单单（groupType=0）
      - 非转置: 3维 (E, N, K)
      - 转置: 3维 (E, K, N)
  - 约束
    - 最后一维为偶数（不转置时N轴，转置时K轴）
    - 多多多时所有tensor转置状态须统一
- biasOptional（可选, ParamType=DYNAMIC）
  - dtype
    - x=FLOAT16 时: FLOAT16 / null
    - x=BFLOAT16 时: BFLOAT16 / FLOAT32 / null
  - format: ND
  - shape
    - 多多多（groupType=-1）: 1维 (n_i)
    - 单单单（groupType=0）: 2维 (E, N)
- antiquantScaleOptional（必选, ParamType=DYNAMIC）
  - dtype: FLOAT16 / BFLOAT16（与x一致）
  - format: ND
  - shape（pergroup）
    - 多多多（groupType=-1）: (G_i, N_i)，其中 G_i = k_i / gs
    - 单单单（groupType=0）: (E, G, N)，其中 G = K / gs
- antiquantOffsetOptional（可选, ParamType=DYNAMIC）
  - dtype: FLOAT16 / BFLOAT16 / null（与x一致，对称量化时为null）
  - format: ND
  - shape: 同 antiquantScaleOptional
  - 约束: 对称量化时为 null（本场景仅支持对称量化 pergroup）
- groupListOptional（条件必选, ParamType=OPTIONAL）
  - dtype: INT64
  - format: ND
  - shape: 1维 (E,)
  - 约束
    - groupType=0 时必传，第1维最大1024（E最大1024）
    - groupType=-1 时必须传空
    - groupListType=0: 非负单调非递减，最后一个值 ≤ x第一维
    - groupListType=1: 非负，数值总和 ≤ x第一维
- scaleOptional: null（不使用, ParamType=DYNAMIC）
- offsetOptional: null（不使用, ParamType=DYNAMIC）
- perTokenScaleOptional: null（不使用, ParamType=OPTIONAL）
- activationInputOptional: null（不使用）
- activationQuantScaleOptional: null（不使用）
- activationQuantOffsetOptional: null（不使用）

### Scalar / 属性参数

- splitItem（int64）
  - 多多多: {0, 1}
  - 单单单: {2, 3}
- groupType（int64）: {-1, 0}
  - -1: 不分组（多多多）
  - 0: m轴分组（单单单）
- groupListType（int64）: {0, 1}
  - 0: cumsum 模式
  - 1: count 模式
- actType（int64）: {0}（固定，GMM_ACT_TYPE_NONE）
- transpose_weight（bool）: {false}（保留属性，未启用）
- transpose_x（bool）: {false}（保留属性，未启用）
- tuningConfigOptional: 不支持（Ascend 950）

### 关键约束

- group_size ∈ {32, 64, 128, 256}
- K % group_size == 0
- pergroup 数 G_i 整除 k_i
- 多多多时所有 pergroup 长度 s_i = k_i/G_i 必须相等
- pergroup 转置时 s_i 为偶数

## 输出

### Tensor 参数

- out（ParamType=DYNAMIC）
  - dtype: FLOAT16 / BFLOAT16（与x一致）
  - format: ND
  - shape
    - 多多多（groupType=-1）: 2维 (m_i, n_i)
    - 单单单（groupType=0）: 2维 (M, N)
- activationFeatureOutOptional: null
- dynQuantScaleOutOptional: null

## UT（Tiling 异常拦截 + 白盒验证）

### 黑盒用例（44 个）
- dtype 校验（7 个）：T-01 ~ T-07
- shape 校验（19 个）：T-08 ~ T-26
  - 维度校验（8 个）：T-08 ~ T-15
  - 数值约束（6 个）：T-16 ~ T-21
  - shape 匹配（5 个）：T-22 ~ T-26
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

## ST

### 精度（黑盒，9 个）
- x=FLOAT16（4 个）：P-01 ~ P-04
- x=BFLOAT16（5 个）：P-05 ~ P-09

### 功能（黑盒，27 个）
- 单单单 groupType=0（12 个）：F-01 ~ F-12
- 多多多 groupType=-1（15 个）：F-13 ~ F-27

### 性能（黑盒，6 个）
- 典型场景（3 个）：PERF-01 ~ PERF-03
- 大/小 shape 场景（3 个）：PERF-04 ~ PERF-06

### 泛化（黑盒，12 个）
- group_size 边界（4 个）：G-01 ~ G-04
- shape 边界（4 个）：G-05 ~ G-08
- 特殊组合（4 个）：G-09 ~ G-12
- （待补充：泛化用例覆盖度不足，后续补充）

### 兼容性（待补充）
