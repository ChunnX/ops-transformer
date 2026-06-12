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

## UT（Tiling 异常拦截）

### dtype 校验

- x dtype 非法（非 FLOAT16/BFLOAT16）
- weight dtype 非法（非 INT4）
- bias dtype 与 x 不匹配
  - x=FLOAT16 时 bias 传入 BFLOAT16/FLOAT32
  - x=BFLOAT16 时 bias 传入 FLOAT16
- antiquantScale dtype 与 x 不一致
- antiquantOffset dtype 与 x 不一致
- groupList dtype 非法（非 INT64）

### shape 校验

- 维度校验
  - x 维度不为 2
  - weight 维度错误
    - 多多多时不为 2 维
    - 单单单时不为 3 维
  - bias 维度错误
    - 多多多时不为 1 维
    - 单单单时不为 2 维
  - antiquantScale 维度错误
    - 多多多时不为 2 维 (G_i, N_i)
    - 单单单时不为 3 维 (E, G, N)
  - groupList 维度不为 1
- 数值约束
  - K % group_size != 0
  - group_size 不在 {32, 64, 128, 256}
  - weight 最后一维非偶数
  - pergroup 数 G_i 不整除 k_i
  - 多多多时 pergroup 长度 s_i 不一致
  - pergroup 转置时 s_i 非偶数
- shape 匹配
  - x.k 与 weight.k 不匹配（非转置）
  - x.k 与 weight.n 不匹配（转置时 weight shape 为 (k,n)）
  - antiquantScale 的 N 与 weight 的 N 不匹配
  - antiquantScale 的 G 与 K/gs 不匹配
  - bias 的 N 与 weight 的 N 不匹配
  - groupList 长度与 E 不匹配（单单单）
  - groupList 值超出 x 第一维

### format 校验

- x format 非 ND
- weight format 非 ND
- bias format 非 ND
- antiquantScale format 非 ND
- antiquantOffset format 非 ND

### 属性校验

- groupType 非法（非 -1/0）
- groupListType 非法（非 0/1）
- actType 非法（非 0）
- splitItem 非法
  - 多多多时不在 {0, 1}
  - 单单单时不在 {2, 3}
- tuningConfigOptional 传入非空（950 不支持）

### 场景校验（组合参数）

- groupType=0 但 groupList 为空
- groupType=-1 但 groupList 非空
- groupType=0 且 groupListType=0 但数列非单调非递减
- groupType=0 且 groupListType=1 但数值为负
- groupType=0 且 groupListType=0 但最后一个值 > x 第一维
- groupType=0 且 groupListType=1 但数值总和 > x 第一维
- splitItem 与 groupType 不匹配
  - groupType=-1 但 splitItem ∈ {2, 3}
  - groupType=0 但 splitItem ∈ {0, 1}
- 多多多时 weight tensorList 转置状态不统一
- 非对称量化（antiquantOffset 非 null）+ pergroup（不支持）

## ST

### 精度

- x=FLOAT16, weight=INT4
  - 有 bias（FLOAT16）
  - 无 bias（null）
  - 有 antiquantOffset（FLOAT16）
  - 无 antiquantOffset（null，对称量化）
- x=BFLOAT16, weight=INT4
  - 有 bias（BFLOAT16）
  - 有 bias（FLOAT32）
  - 无 bias（null）
  - 有 antiquantOffset（BFLOAT16）
  - 无 antiquantOffset（null，对称量化）

### 功能

- 单单单（groupType=0）
  - 非转置: x(M,K), weight(E,N,K), out(M,N)
  - 转置: x(M,K), weight(E,K,N), out(M,N)
  - group_size = 32
  - group_size = 64
  - group_size = 128
  - group_size = 256
  - groupListType = 0（cumsum）
  - groupListType = 1（count）
  - shape 边界
    - M = 1（极小 M）
    - K = group_size（最小 K）
    - N = 1（极小 N）
    - E = 1（单 expert）
    - E = 1024（最大 expert）
- 多多多（groupType=-1）
  - 非转置: x(m_i,k_i), weight(n_i,k_i), out(m_i,n_i)
  - 转置: x(m_i,k_i), weight(k_i,n_i), out(m_i,n_i)
  - group_size = 32
  - group_size = 64
  - group_size = 128
  - group_size = 256
  - groupListType = 0（cumsum）
  - groupListType = 1（count）
  - shape 边界
    - m_i = 1（极小 M）
    - k_i = group_size（最小 K）
    - n_i = 1（极小 N）
    - 多 expert 数量（2/8/64/128）
    - 各 expert 的 m_i 不同

### 性能

- （待补充）

### 泛化

- （待补充）

### 兼容性

- （待补充）
