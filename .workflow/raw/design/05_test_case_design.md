# 05 测试用例设计 — GMM A16W4 PerGroup ND

> 本文档描述 A16W4 pergroup ND 数据流的完整测试用例设计，覆盖 tiling 异常拦截、精度、功能、性能、泛化、kernel 打桩验证、tiling 白盒和 kernel 白盒 8 种用例类型。
> 相关文档：[算子规格](../spec/gmm_a16w4nd_op.md)、[02 与硬件无关方案](./02_platform_independent_design.md)、[03 Tiling 方案](./03_tiling_design.md)、[04 Kernel 方案](./04_kernel_design.md)

---

## 1. 用例设计总体说明

### 1.1 测试对象

- **算子**：aclnnGroupedMatmulV5 — A16W4 Weight ND PerGroup
- **平台**：Ascend 950PR/Ascend 950DT (arch35)
- **数据流**：x(FLOAT16/BFLOAT16) × weight(INT4, ND) → 反量化(pergroup) → matmul → out

### 1.2 计算公式

$$
y_i = x_i \times (weight_i + antiquant\_offset_i) \times antiquant\_scale_i + bias_i
$$

对称量化（antiquantOffset 为 null）时：

$$
y_i = x_i \times weight_i \times antiquant\_scale_i + bias_i
$$

### 1.3 关键约束

| 约束项 | 条件 |
|--------|------|
| group_size | ∈ {32, 64, 128, 256} |
| K % group_size | == 0 |
| weight 最后一维 | 偶数 |
| pergroup 长度 s_i | 多多多时所有 s_i 相等 |
| pergroup 转置 | s_i 为偶数 |
| x format | ND（固定不转置） |
| weight format | ND |
| groupType | -1（多多多）或 0（单单单） |
| splitItem | 多多多 {0,1}，单单单 {2,3} |

### 1.4 优先级定义

| 优先级 | 含义 | 要求 |
|--------|------|------|
| P0 | 核心功能 | 必须通过 |
| P1 | 重要功能 | 应该通过 |
| P2 | 边界场景 | 最好通过 |

### 1.5 用例编号规范

| 前缀 | 用例类型 | 数量 |
|------|----------|------|
| T-XX | Tiling 异常拦截 | 44 |
| P-XX | 精度验证 | 9 |
| F-XX | 功能验证 | 27 |
| PERF-XX | 性能验证 | 6 |
| G-XX | 泛化验证 | 12 |
| KS-XX | Kernel 打桩验证 | 22 |
| TW-XX | Tiling 白盒验证 | 20 |
| KW-XX | Kernel 白盒验证 | 22 |

**总计：162 个用例**

---

## 2. Tiling 异常拦截用例（T-01 ~ T-44）

### 2.1 dtype 校验（T-01 ~ T-07）

| 编号 | 描述 | 异常参数 | 预期错误 | 优先级 |
|------|------|----------|----------|--------|
| T-01 | x dtype 非法(FLOAT32) | x_dtype=FLOAT32 | ERR_INVALID_DTYPE | P0 |
| T-02 | weight dtype 非法(INT8) | weight_dtype=INT8 | ERR_INVALID_DTYPE | P0 |
| T-03 | bias dtype 与 x 不匹配: x=FP16, bias=BF16 | bias_dtype=BFLOAT16 | ERR_INVALID_DTYPE | P0 |
| T-04 | bias dtype 与 x 不匹配: x=BF16, bias=FP16 | bias_dtype=FLOAT16 | ERR_INVALID_DTYPE | P0 |
| T-05 | antiquantScale dtype 与 x 不一致 | x=FP16, scale=BF16 | ERR_INVALID_DTYPE | P0 |
| T-06 | antiquantOffset dtype 与 x 不一致 | x=FP16, offset=BF16 | ERR_INVALID_DTYPE | P0 |
| T-07 | groupList dtype 非法(INT32) | groupList_dtype=INT32 | ERR_INVALID_DTYPE | P0 |

### 2.2 shape 校验 — 维度校验（T-08 ~ T-15）

| 编号 | 描述 | 异常参数 | 预期错误 | 优先级 |
|------|------|----------|----------|--------|
| T-08 | x 维度不为 2 (3D) | x_shape=(4,64,512) | ERR_INVALID_DIM | P0 |
| T-09 | 多多多 weight 维度不为 2 (3D) | weight_shape=(4,256,512) | ERR_INVALID_DIM | P0 |
| T-10 | 单单单 weight 维度不为 3 (2D) | weight_shape=(256,512) | ERR_INVALID_DIM | P0 |
| T-11 | 多多多 bias 维度不为 1 (2D) | bias_shape=(4,256) | ERR_INVALID_DIM | P1 |
| T-12 | 单单单 bias 维度不为 2 (1D) | bias_shape=(256) | ERR_INVALID_DIM | P1 |
| T-13 | 多多多 antiquantScale 维度错误 (3D) | scale_shape=(1,4,256) | ERR_INVALID_DIM | P0 |
| T-14 | 单单单 pergroup antiquantScale 维度错误 (2D) | scale_shape=(4,256) | ERR_INVALID_DIM | P0 |
| T-15 | groupList 维度不为 1 (2D) | groupList_shape=(2,2) | ERR_INVALID_DIM | P1 |

### 2.3 shape 校验 — 数值约束（T-16 ~ T-21）

| 编号 | 描述 | 异常参数 | 预期错误 | 优先级 |
|------|------|----------|----------|--------|
| T-16 | K 不是 group_size 的整数倍 | K=100, gs=32 | ERR_K_NOT_ALIGNED | P0 |
| T-17 | group_size 不在合法集合 | gs=48 | ERR_INVALID_GROUP_SIZE | P0 |
| T-18 | weight 最后一维(N)非偶数 | N=3, 非转置 | ERR_WEIGHT_LAST_DIM_ODD | P0 |
| T-19 | pergroup 数 G_i 不整除 k_i | k=512, G=3 | ERR_PERGROUP_NOT_DIVISIBLE | P0 |
| T-20 | 多多多 pergroup 长度 s_i 不一致 | s_1=128, s_2=64 | ERR_PERGROUP_LENGTH_INCONSISTENT | P0 |
| T-21 | pergroup 转置时 s_i 非偶数 | gs=33, 转置 | ERR_PERGROUP_TRANSPOSE_ODD | P1 |

### 2.4 shape 校验 — shape 匹配（T-22 ~ T-26）

| 编号 | 描述 | 异常参数 | 预期错误 | 优先级 |
|------|------|----------|----------|--------|
| T-22 | x.k 与 weight.k 不匹配 | x K=512, weight K=256 | ERR_SHAPE_MISMATCH | P0 |
| T-23 | antiquantScale N 与 weight N 不匹配 | scale N=128, weight N=256 | ERR_SCALE_SHAPE_MISMATCH | P0 |
| T-24 | antiquantScale G 与 K/gs 不匹配 | G=5, K/gs=4 | ERR_SCALE_SHAPE_MISMATCH | P0 |
| T-25 | bias N 与 weight N 不匹配 | bias N=128, weight N=256 | ERR_SHAPE_MISMATCH | P1 |
| T-26 | groupList 长度与 E 不匹配 | len=3, E=4 | ERR_SHAPE_MISMATCH | P1 |

### 2.5 format 校验（T-27 ~ T-31）

| 编号 | 描述 | 异常参数 | 预期错误 | 优先级 |
|------|------|----------|----------|--------|
| T-27 | x format 非 ND (NZ) | x_format=NZ | ERR_INVALID_FORMAT | P0 |
| T-28 | weight format 非 ND (NZ) | weight_format=NZ | ERR_INVALID_FORMAT | P0 |
| T-29 | bias format 非 ND | bias_format=NZ | ERR_INVALID_FORMAT | P1 |
| T-30 | antiquantScale format 非 ND | scale_format=NZ | ERR_INVALID_FORMAT | P1 |
| T-31 | antiquantOffset format 非 ND | offset_format=NZ | ERR_INVALID_FORMAT | P1 |

### 2.6 属性校验（T-32 ~ T-37）

| 编号 | 描述 | 异常参数 | 预期错误 | 优先级 |
|------|------|----------|----------|--------|
| T-32 | groupType 非法 | groupType=1 | ERR_INVALID_ATTR | P0 |
| T-33 | groupListType 非法 | groupListType=2 | ERR_INVALID_ATTR | P0 |
| T-34 | actType 非法 | actType=1 | ERR_INVALID_ATTR | P1 |
| T-35 | 多多多 splitItem 非法 | splitItem=2 | ERR_INVALID_ATTR | P0 |
| T-36 | 单单单 splitItem 非法 | splitItem=0 | ERR_INVALID_ATTR | P0 |
| T-37 | tuningConfigOptional 非空 | 传入非空数组 | ERR_INVALID_ATTR | P1 |

### 2.7 场景校验 — 组合参数（T-38 ~ T-44）

| 编号 | 描述 | 异常参数 | 预期错误 | 优先级 |
|------|------|----------|----------|--------|
| T-38 | groupType=0 但 groupList 为空 | groupList=null | ERR_GROUPLIST_REQUIRED | P0 |
| T-39 | groupType=-1 但 groupList 非空 | groupList 非空 | ERR_GROUPLIST_NOT_ALLOWED | P0 |
| T-40 | groupListType=0 数列非单调非递减 | [64,32,192,256] | ERR_GROUPLIST_INVALID | P1 |
| T-41 | groupListType=1 数值为负 | [64,-32,128,96] | ERR_GROUPLIST_INVALID | P1 |
| T-42 | groupListType=0 最后值 > x 第一维 | last=300 > M=256 | ERR_GROUPLIST_INVALID | P1 |
| T-43 | groupListType=1 总和 > x 第一维 | sum=300 > M=256 | ERR_GROUPLIST_INVALID | P1 |
| T-44 | 多多多 weight 转置状态不统一 | tensor0 非转置, tensor1 转置 | ERR_TRANSPOSE_INCONSISTENT | P0 |

---

## 3. 精度用例（P-01 ~ P-09）

### 3.1 x=FLOAT16 精度用例

| 编号 | 描述 | bias | offset | shape | 容差 | 优先级 |
|------|------|------|--------|-------|------|--------|
| P-01 | FP16 有bias有offset | FP16 | FP16 | M=256,K=512,N=256 | rtol=1e-3,atol=1e-3 | P0 |
| P-02 | FP16 有bias无offset(对称) | FP16 | null | M=256,K=512,N=256 | rtol=1e-3,atol=1e-3 | P0 |
| P-03 | FP16 无bias有offset | null | FP16 | M=256,K=512,N=256 | rtol=1e-3,atol=1e-3 | P0 |
| P-04 | FP16 无bias无offset(对称) | null | null | M=256,K=512,N=256 | rtol=1e-3,atol=1e-3 | P0 |

### 3.2 x=BFLOAT16 精度用例

| 编号 | 描述 | bias | offset | shape | 容差 | 优先级 |
|------|------|------|--------|-------|------|--------|
| P-05 | BF16 有bias(BF16)有offset(BF16) | BF16 | BF16 | M=256,K=512,N=256 | rtol=1e-2,atol=1e-2 | P0 |
| P-06 | BF16 有bias(BF16)无offset(对称) | BF16 | null | M=256,K=512,N=256 | rtol=1e-2,atol=1e-2 | P0 |
| P-07 | BF16 有bias(FP32)无offset | FP32 | null | M=256,K=512,N=256 | rtol=1e-2,atol=1e-2 | P1 |
| P-08 | BF16 无bias有offset(BF16) | null | BF16 | M=256,K=512,N=256 | rtol=1e-2,atol=1e-2 | P0 |
| P-09 | BF16 无bias无offset(对称) | null | null | M=256,K=512,N=256 | rtol=1e-2,atol=1e-2 | P0 |

**公共参数**：groupType=-1（多多多），group_size=128，transpose_weight=0，E=1

---

## 4. 功能用例（F-01 ~ F-27）

### 4.1 单单单（groupType=0）

| 编号 | 描述 | group_size | 转置 | groupListType | shape | 优先级 |
|------|------|:----------:|:----:|:-------------:|-------|--------|
| F-01 | 非转置, gs=32 | 32 | 否 | 0(cumsum) | M=256,K=512,N=256,E=4 | P0 |
| F-02 | 非转置, gs=64 | 64 | 否 | 0(cumsum) | M=256,K=512,N=256,E=4 | P0 |
| F-03 | 非转置, gs=128 | 128 | 否 | 0(cumsum) | M=256,K=512,N=256,E=4 | P0 |
| F-04 | 非转置, gs=256 | 256 | 否 | 0(cumsum) | M=256,K=512,N=256,E=4 | P0 |
| F-05 | 转置, gs=128 | 128 | 是 | 0(cumsum) | M=256,K=512,N=256,E=4 | P0 |
| F-06 | groupListType=0(cumsum) | 128 | 否 | 0 | M=256,K=512,N=256,E=4 | P0 |
| F-07 | groupListType=1(count) | 128 | 否 | 1 | M=256,K=512,N=256,E=4 | P0 |
| F-08 | M=1(极小M) | 128 | 否 | 1(count) | M=1,K=512,N=256,E=4 | P1 |
| F-09 | K=group_size(最小K) | 128 | 否 | 0(cumsum) | M=256,K=128,N=256,E=4 | P1 |
| F-10 | N=2(极小N,偶数约束) | 128 | 否 | 0(cumsum) | M=256,K=512,N=2,E=4 | P1 |
| F-11 | E=1(单expert) | 128 | 否 | 0(cumsum) | M=256,K=512,N=256,E=1 | P1 |
| F-12 | E=1024(最大expert) | 128 | 否 | 1(count) | M=1024,K=512,N=256,E=1024 | P2 |

**单单单 groupList 内容**：
- F-01~F-06, F-09~F-11: cumsum `[64,128,192,256]`（均匀分组，每组 m=64）
- F-07: count `[64,64,64,64]`
- F-08: count `[1,0,0,0]`（仅第1个group有数据）
- F-12: count `[1,1,...,1]`（1024个1）

### 4.2 多多多（groupType=-1）

| 编号 | 描述 | group_size | 转置 | E | shape (per tensor) | 优先级 |
|------|------|:----------:|:----:|:-:|---------------------|--------|
| F-13 | 非转置, gs=32 | 32 | 否 | 4 | m=64,k=512,n=256 | P0 |
| F-14 | 非转置, gs=64 | 64 | 否 | 4 | m=64,k=512,n=256 | P0 |
| F-15 | 非转置, gs=128 | 128 | 否 | 4 | m=64,k=512,n=256 | P0 |
| F-16 | 非转置, gs=256 | 256 | 否 | 4 | m=64,k=512,n=256 | P0 |
| F-17 | 转置, gs=128 | 128 | 是 | 4 | m=64,k=512,n=256 | P0 |
| F-18 | m_i=1(极小M) | 128 | 否 | 4 | m=1,k=512,n=256 | P1 |
| F-19 | k_i=group_size(最小K) | 128 | 否 | 4 | m=64,k=128,n=256 | P1 |
| F-20 | n_i=2(极小N) | 128 | 否 | 4 | m=64,k=512,n=2 | P1 |
| F-21 | E=2 | 128 | 否 | 2 | m=128,k=512,n=256 | P1 |
| F-22 | E=8 | 128 | 否 | 8 | m=32,k=512,n=256 | P0 |
| F-23 | E=64 | 128 | 否 | 64 | m=4,k=512,n=256 | P1 |
| F-24 | E=128 | 128 | 否 | 128 | m=2,k=512,n=256 | P2 |
| F-25 | 各expert m_i不同 | 128 | 否 | 4 | m=[1,32,64,128],k=512,n=256 | P1 |

### 4.3 完整参数组合

| 编号 | 描述 | groupType | bias | offset | 转置 | gs | 优先级 |
|------|------|:---------:|:----:|:------:|:----:|:--:|--------|
| F-26 | 单单单+bias+offset | 0 | FP16 | FP16 | 否 | 128 | P0 |
| F-27 | 多多多+bias+offset+转置 | -1 | FP16 | FP16 | 是 | 128 | P0 |

---

## 5. 性能用例（PERF-01 ~ PERF-06）

| 编号 | 描述 | groupType | E | M | K | N | gs | 转置 | 优先级 |
|------|------|:---------:|:-:|:---:|:---:|:---:|:--:|:----:|--------|
| PERF-01 | 典型MoE 8experts | -1 | 8 | 256 | 4096 | 4096 | 128 | 否 | P0 |
| PERF-02 | 典型MoE 64experts | -1 | 64 | 32 | 4096 | 4096 | 128 | 否 | P0 |
| PERF-03 | 大shape | -1 | 1 | 1024 | 8192 | 8192 | 128 | 否 | P1 |
| PERF-04 | 小shape | -1 | 1 | 1 | 128 | 128 | 128 | 否 | P1 |
| PERF-05 | 典型MoE+转置 | -1 | 8 | 256 | 4096 | 4096 | 128 | 是 | P1 |
| PERF-06 | 单单单大E | 0 | 256 | 256 | 4096 | 1024 | 128 | 否 | P1 |

**性能指标预期**：
- PERF-01/02/05：作为 MoE 典型场景的性能基线
- PERF-03：验证大 shape 下 tiling 切分效率
- PERF-04：验证小 shape 下尾块处理和资源利用率
- PERF-06：验证单单单大 E 场景的 group 遍历效率

---

## 6. 泛化用例（G-01 ~ G-12）

### 6.1 group_size 边界

| 编号 | 描述 | group_size | K | G=K/gs | groupType | 优先级 |
|------|------|:----------:|:--:|:------:|:---------:|--------|
| G-01 | gs=32(最小) | 32 | 256 | 8 | -1 | P1 |
| G-02 | gs=256(最大) | 256 | 1024 | 4 | -1 | P1 |
| G-03 | K=gs(单group) | 128 | 128 | 1 | -1 | P1 |
| G-04 | K=gs*10(多group) | 128 | 1280 | 10 | -1 | P1 |

### 6.2 shape 边界

| 编号 | 描述 | shape | groupType | 优先级 |
|------|------|-------|:---------:|--------|
| G-05 | M=1(极小M) | m=1,k=512,n=256 | -1 | P1 |
| G-06 | N=2(极小N,偶数约束) | m=64,k=512,n=2 | -1 | P1 |
| G-07 | E=1(单expert,多多多) | m=256,k=512,n=256 | -1 | P1 |
| G-08 | E=1024(最大expert,单单单) | M=1024,K=512,N=256 | 0 | P2 |

### 6.3 特殊组合

| 编号 | 描述 | 参数 | groupType | 优先级 |
|------|------|------|:---------:|--------|
| G-09 | 各expert m_i极端不同 | m=[1,16,64,256] | -1 | P1 |
| G-10 | 各expert k_i不同(gs相同) | k=[256,512], gs=128 | -1 | P1 |
| G-11 | 单单单 cumsum 非均匀分组 | [1,33,97,256] → m=[1,32,64,159] | 0 | P1 |
| G-12 | 单单单 count 非均匀分组 | [1,32,64,159] → sum=256 | 0 | P1 |

---

## 7. Kernel 打桩验证用例（KS-01 ~ KS-22）

### 7.1 数据搬运验证 — MTE 阶段（KS-01 ~ KS-07）

#### MTE2 搬运验证

| 编号 | 描述 | 打桩点 | 输入数据模式 | 预期输出 | 优先级 |
|------|------|--------|-------------|----------|--------|
| KS-01 | Weight INT4 GM→UB | CopyWeightGmToUb 后 | 等差数列(0~15重复) | UB中INT4与GM一致 | P0 |
| KS-02 | Scale GM→UB (pergroup 2D) | CopyAntiQuantParamsGmToUb 后 | scale[g,n]=g*100+n | UB中scale布局正确 | P0 |
| KS-03 | Offset GM→UB (pergroup 2D) | CopyAntiQuantParamsGmToUb 后 | offset[g,n]=g*10+n | UB中offset布局正确 | P0 |

**KS-02 参数**：group_size=128, kL1=256, nL1=256 → scale UB shape=(2, 256)

#### MTE3 搬运验证

| 编号 | 描述 | 打桩点 | 输入数据模式 | 预期输出 | 优先级 |
|------|------|--------|-------------|----------|--------|
| KS-04 | 反量化后weight UB→L1 | WeightHighBitUbToL1 后 | 全1.0 FP16 | L1与UB一致(含NZ转换) | P0 |

#### MTE1 搬运验证

| 编号 | 描述 | 打桩点 | 输入数据模式 | 预期输出 | 优先级 |
|------|------|--------|-------------|----------|--------|
| KS-05 | A矩阵 GM→L1 | CopyAAndBiasGmToL1 后 | A[m,k]=m*K+k | AL1与GM一致 | P0 |
| KS-06 | Bias GM→L1 | CopyAAndBiasGmToL1 后 | bias[n]=n | BiasL1与GM一致 | P1 |

#### 输出写回验证

| 编号 | 描述 | 打桩点 | 输入数据模式 | 预期输出 | 优先级 |
|------|------|--------|-------------|----------|--------|
| KS-07 | C矩阵 L1→GM | GetTensorC 后 | L1中C全2.0 | GM输出与L1一致 | P0 |

### 7.2 计算逻辑验证 — VF 阶段（KS-08 ~ KS-14）

#### INT4 unpack 和 cast

| 编号 | 描述 | 打桩点 | 输入数据模式 | 预期输出 | 优先级 |
|------|------|--------|-------------|----------|--------|
| KS-08 | INT4 unpack (UNPACK4) | DataCopy UNPACK4 后 | 0,1,...,7,-8,...,-1 | unpack后INT8值正确 | P0 |
| KS-09 | INT4→FP16 cast | Cast 后 | unpack后INT8 | FP16值与INT8数值相等 | P0 |

#### PerGroup scale/offset 加载

| 编号 | 描述 | 打桩点 | 输入数据模式 | 预期输出 | 优先级 |
|------|------|--------|-------------|----------|--------|
| KS-10 | NK排布scale广播加载 | NdNkLoadScaleOffset 后 | scale[g,n]=g*100+n | broadcast值正确 | P0 |
| KS-11 | KN排布scale直接加载 | NdKnLoadScaleOffset 后 | scale[g,n]=g*100+n | direct值正确 | P0 |
| KS-12 | NK排布offset广播加载 | NdNkLoadScaleOffset 后 | offset[g,n]=g*10+n | broadcast值正确 | P1 |

**KS-10/12 参数**：group_size=128, nOffset=0, kOffset=0

#### 反量化计算

| 编号 | 描述 | 打桩点 | 输入 | 预期输出 | 优先级 |
|------|------|--------|------|----------|--------|
| KS-13 | 反量化(有offset) | Add+Mul 后 | w=2,scale=1.5,offset=1 | (2+1)*1.5=4.5 | P0 |
| KS-14 | 反量化(无offset,对称) | Mul 后 | w=2,scale=1.5 | 2*1.5=3.0 | P0 |

### 7.3 地址偏移验证（KS-15 ~ KS-17）

| 编号 | 描述 | 打桩点 | 参数 | 预期输出 | 优先级 |
|------|------|--------|------|----------|--------|
| KS-15 | PerGroup scale偏移(单单单) | SetGmAddr 后 | E=4,K=512,N=256,gs=128 | group偏移: 0,1024,2048,3072 | P0 |
| KS-16 | Bias偏移分离(单单单) | SetGmAddr 后 | E=4,N=256 | bias偏移: 0,256,512,768 | P0 |
| KS-17 | INT4 weight偏移>>1(单单单) | SetGmAddr 后 | E=4,K=512,N=256 | weight偏移: 0,65536,131072,196608 | P0 |

**KS-15 推导**：每group scale元素数 = (K/gs)*N = (512/128)*256 = 4*256 = 1024
**KS-16 推导**：每group bias元素数 = N = 256
**KS-17 推导**：每group weight字节数 = (N*K)>>1 = (256*512)>>1 = 65536

### 7.4 Tiling 参数传递验证（KS-18 ~ KS-20）

| 编号 | 描述 | 打桩点 | 参数 | 预期输出 | 优先级 |
|------|------|--------|------|----------|--------|
| KS-18 | groupSize传递 | Controller Init 后 | gs=128 | kernel中groupSize=128 | P0 |
| KS-19 | mList/kList/nList(多多多) | Process循环中 | E=4,各expert不同 | list值与tiling一致 | P0 |
| KS-20 | baseM/baseN动态调整 | Process中 | M=100,N=200 | baseM=50,baseN=128 | P1 |

**KS-19 参数**：m=[1,32,64,128], k=[512,512,512,512], n=[256,256,256,256]
**KS-20 推导**：M=100 < baseM(256) → baseM = CeilDiv(100,2) = 50; N=200 < baseN(256)*coreNum → baseN = 256>>1 = 128

### 7.5 Matmul 计算验证（KS-21 ~ KS-22）

| 编号 | 描述 | 打桩点 | 输入 | 预期输出 | 优先级 |
|------|------|--------|------|----------|--------|
| KS-21 | Matmul基础计算 | LaunchMatmul 后 | A全1,W_dequant全1,bias全0 | C[m,n]=K=64 | P0 |
| KS-22 | Matmul+bias | LaunchMatmul 后 | A全1,W_dequant全1,bias[n]=n | C[m,n]=K+n=64+n | P0 |

**KS-21/22 参数**：M=16, K=64, N=16, group_size=64, groupType=-1

---

## 8. Tiling 白盒用例（TW-01 ~ TW-20）

> 基于 tiling 代码路径设计端到端用例，验证 tiling 计算过程中各分支/函数的中间结果正确性。
> 通过检查 tiling 输出结构体（`GMMWeightQuantParam`）的字段值来验证。

### 8.1 IsA16W4NDPergroup 判断路径（TW-01 ~ TW-03）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期 tiling 输出 | 优先级 |
|------|------|-------------|----------|-----------------|--------|
| TW-01 | pergroup场景正确识别 | `IsA16W4NDPergroup()` → `groupSize_ > 0` 为 true | 单单单, E=4, M=256, K=512, N=256, scale_shape=(4,4,256), gs=128, 非转置 | `groupSize=128` (K/G=512/4=128) | P0 |
| TW-02 | perchannel场景不误判为pergroup | `IsA16W4NDPergroup()` → `groupSize_ == 0` 为 true | 单单单, E=4, M=256, K=512, N=256, scale_shape=(4,256), 非转置 | `groupSize=0` | P0 |
| TW-03 | 多多多pergroup 2维scale正确识别 | `SetAntiquantGroupSize()` 中 `antiquantScaleDimNum==2 && IsA16W4ND()` 分支 | 多多多, E=4, m=64, k=512, n=256, scale_shape=(4,256), gs=128, 非转置 | `groupSize=128` (K/G=512/4=128) | P0 |

### 8.2 SetAntiquantGroupSize 路径（TW-04 ~ TW-07）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期 tiling 输出 | 优先级 |
|------|------|-------------|----------|-----------------|--------|
| TW-04 | 单单单3维scale groupSize推导 | `SetAntiquantGroupSize()` 中 `antiquantScaleDimNum==3` 分支 | 单单单, E=4, M=256, K=1024, N=256, scale_shape=(4,8,256), gs=128, 非转置 | `groupSize=128` (K/G=1024/8=128) | P0 |
| TW-05 | group_size=32场景 | `CheckGroupSize()` 中 `groupSize_==32` 分支 | 单单单, E=4, M=256, K=256, N=256, scale_shape=(4,8,256), gs=32, 非转置 | `groupSize=32`, 校验通过 | P0 |
| TW-06 | group_size=64场景 | `CheckGroupSize()` 中 `groupSize_==64` 分支 | 单单单, E=4, M=256, K=256, N=256, scale_shape=(4,4,256), gs=64, 非转置 | `groupSize=64` | P0 |
| TW-07 | group_size=256场景 | `CheckGroupSize()` 中 `groupSize_==256` 分支 | 单单单, E=4, M=256, K=256, N=256, scale_shape=(4,1,256), gs=256, 非转置 | `groupSize=256` | P1 |

### 8.3 SetTilingKey 路径（TW-08 ~ TW-11）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期 tiling 输出 | 优先级 |
|------|------|-------------|----------|-----------------|--------|
| TW-08 | pergroup antiquantType设置 | `SetTilingKey()` 中 `IsA16W4NDPergroup()` → `antiquantType=PER_GROUP` | 单单单, pergroup, gs=128, 非转置 | `W_QUANT_TYPE=WQGMM_PER_GROUP` | P0 |
| TW-09 | perchannel antiquantType设置(对比) | `SetTilingKey()` 中 else 分支 → `antiquantType=PER_CHANNEL` | 单单单, perchannel, 非转置 | `W_QUANT_TYPE=WQGMM_PER_CHANNEL` | P1 |
| TW-10 | 转置场景tiling key | `SetTilingKey()` 中 `transB_==true` → `WQ_B_TRANS=WQGMM_TRANS` | 单单单, pergroup, gs=128, 转置 | `WQ_B_TRANS=1` | P0 |
| TW-11 | 非转置场景tiling key | `SetTilingKey()` 中 `transB_==false` → `WQ_B_TRANS=WQGMM_NO_TRANS` | 单单单, pergroup, gs=128, 非转置 | `WQ_B_TRANS=0` | P0 |

### 8.4 SetShapeList 路径（TW-12 ~ TW-14）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期 tiling 输出 | 优先级 |
|------|------|-------------|----------|-----------------|--------|
| TW-12 | 单单单shape list填充 | `SetShapeListSplitMSingleXSingleWeightSingleY()` | 单单单, E=4, M=256, K=512, N=256, 非转置 | `groupNum=4, mSize=256, kSize=512, nSize=256, kList[0]=512, nList[0]=256, mList[0]=-1` | P0 |
| TW-13 | 多多多shape list填充 | `SetShapeListMultiXMultiWeightMultiY()` | 多多多, E=4, m=[1,32,64,128], k=512, n=256, 非转置 | `mList[i]`/`kList[i]`/`nList[i]` 逐tensor正确填充 | P0 |
| TW-14 | INT4 weight nSize扩展 | `SetShapeListSplitMSingleXSingleWeightSingleY()` 中 `weightDtype_==DT_INT4 && !transB_` → `nSize_=8*nSize_` | 单单单, E=4, M=256, K=512, weight_shape=(4,32,512), 非转置 | `nSize=256` (32*8=256) | P0 |

### 8.5 baseM/baseN/baseK 计算路径（TW-15 ~ TW-16）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期 tiling 输出 | 优先级 |
|------|------|-------------|----------|-----------------|--------|
| TW-15 | 默认base值(无bias) | `SetBaseTiling()` 中默认值设置 | 单单单, pergroup, M=512, N=512, 无bias, 非转置 | `baseM=256, baseN=256, baseK=64` | P0 |
| TW-16 | 有bias时baseM调整 | `SetBaseTiling()` 中 `hasBias` 分支 → `baseM=240` | 单单单, pergroup, M=512, N=512, 有bias, 非转置 | `baseM=240` | P0 |

### 8.6 CalcResplitTiling 路径（TW-17 ~ TW-18）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期 tiling 输出 | 优先级 |
|------|------|-------------|----------|-----------------|--------|
| TW-17 | 多多多ND不使能尾块重切分 | `EnableTailResplit()` → 多多多 → 不使能 | 多多多, pergroup, ND, 非转置 | `mainBlockSize=0` (不使能标志) | P1 |
| TW-18 | 单单单ND非转置不使能尾块重切分 | `EnableTailResplit()` → 单单单+ND+B非转置 → 不使能 | 单单单, pergroup, ND, 非转置 | `mainBlockSize=0` | P1 |

### 8.7 GMMWeightQuantParam 字段完整性（TW-19 ~ TW-20）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期 tiling 输出 | 优先级 |
|------|------|-------------|----------|-----------------|--------|
| TW-19 | 完整参数传递验证(单单单) | `SetTiling()` 完整流程 | 单单单, E=4, M=256, K=512, N=256, gs=128, 有bias, 非转置, groupListType=0 | `groupNum=4, coreNum>0, kSize=512, nSize=256, singleX=1, singleWeight=1, singleY=1, groupType=0, groupListType=0, hasBias=1, groupSize=128` | P0 |
| TW-20 | 完整参数传递验证(多多多) | `SetTiling()` 完整流程 | 多多多, E=4, m=[1,32,64,128], k=512, n=256, gs=128, 无bias, 非转置 | `singleX=0, singleWeight=0, singleY=0, groupType=-1, hasBias=0, groupSize=128` | P0 |

---

## 9. Kernel 白盒用例（KW-01 ~ KW-22）

> 基于 kernel 代码路径设计端到端用例，验证各分支/分发逻辑的正确性。
> 通过端到端执行并检查最终输出来间接验证特定代码路径被正确执行。

### 9.1 grouped_matmul_apt.cpp if constexpr 分支覆盖（KW-01 ~ KW-06）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期结果 | 容差 | 优先级 |
|------|------|-------------|----------|----------|------|--------|
| KW-01 | pergroup无offset无bias非转置分支 | `V310_GMM_ANTI_QUANT` → `W_TYPE==WQGMM_ND && OFFSET_OR_BIAS_EXIT==NOT_EXIST_BIAS_NOT_EXIST && W_QUANT_TYPE==WQGMM_PER_GROUP && WQ_B_TRANS==WQGMM_NO_TRANS` | 多多多, pergroup, 无offset, 无bias, 非转置, M=256, K=512, N=256, gs=128 | PASS (使用A16W4_NDKN_PG配置) | rtol=1e-3,atol=1e-3 | P0 |
| KW-02 | pergroup无offset有bias非转置分支 | `OFFSET_OR_BIAS_EXIT==NOT_EXIST_BIAS_EXIST` | 多多多, pergroup, 无offset, 有bias(FP16), 非转置, M=256, K=512, N=256, gs=128 | PASS | rtol=1e-3,atol=1e-3 | P0 |
| KW-03 | pergroup有offset无bias非转置分支 | `OFFSET_OR_BIAS_EXIT==EXIST_BIAS_NOT_EXIST` | 多多多, pergroup, 有offset(FP16), 无bias, 非转置, M=256, K=512, N=256, gs=128 | PASS | rtol=1e-3,atol=1e-3 | P0 |
| KW-04 | pergroup有offset有bias非转置分支 | `OFFSET_OR_BIAS_EXIT==EXIST_BIAS_EXIST` | 多多多, pergroup, 有offset(FP16), 有bias(FP16), 非转置, M=256, K=512, N=256, gs=128 | PASS | rtol=1e-3,atol=1e-3 | P0 |
| KW-05 | pergroup无offset无bias转置分支 | `WQ_B_TRANS==WQGMM_TRANS` → 使用`A16W4_NDNK_PG`配置 | 多多多, pergroup, 无offset, 无bias, 转置, M=256, K=512, N=256, gs=128 | PASS | rtol=1e-3,atol=1e-3 | P0 |
| KW-06 | pergroup有offset有bias转置分支 | `WQ_B_TRANS==WQGMM_TRANS && OFFSET_OR_BIAS_EXIT==EXIST_BIAS_EXIST` | 多多多, pergroup, 有offset(FP16), 有bias(FP16), 转置, M=256, K=512, N=256, gs=128 | PASS | rtol=1e-3,atol=1e-3 | P0 |

### 9.2 ComputeBasicBlock 分发路径（KW-07 ~ KW-08）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期结果 | 容差 | 优先级 |
|------|------|-------------|----------|----------|------|--------|
| KW-07 | ND非转置→NdKn路径 | `ComputeBasicBlock()` → `(ND && !bTrans)` → `ComputeBasicBlockAivNdKnNzNk()` | 多多多, pergroup, ND, 非转置, M=256, K=512, N=256, gs=128 | PASS (K为内轴路径) | rtol=1e-3,atol=1e-3 | P0 |
| KW-08 | ND转置→NdNk路径 | `ComputeBasicBlock()` → else → `ComputeBasicBlockAivNdNkNzKn()` | 多多多, pergroup, ND, 转置, M=256, K=512, N=256, gs=128 | PASS (N为内轴路径) | rtol=1e-3,atol=1e-3 | P0 |

### 9.3 GetBufferConfig 分支覆盖（KW-09 ~ KW-10）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期结果 | 容差 | 优先级 |
|------|------|-------------|----------|----------|------|--------|
| KW-09 | pergroup ND buffer配置 | `GetBufferConfig()` → `weightFormat==ND && antiQuantType==PER_GROUP` → `GetNdPerGroupBufferInfo()` | 多多多, pergroup, ND, M=256, K=512, N=256, gs=128 | PASS (scale UB=8KB配置被使用) | rtol=1e-3,atol=1e-3 | P0 |
| KW-10 | perchannel ND buffer配置(对比) | `GetBufferConfig()` → `weightFormat==ND && antiQuantType!=PER_GROUP` → `GetNdBufferInfo()` | 多多多, perchannel, ND, M=256, K=512, N=256 | PASS (原有perchannel路径不受影响) | rtol=1e-3,atol=1e-3 | P1 |

### 9.4 VF 函数 groupSize 运行时分支覆盖（KW-11 ~ KW-14）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期结果 | 容差 | 优先级 |
|------|------|-------------|----------|----------|------|--------|
| KW-11 | groupSize=32 VF分支 | `AntiQuantInt4NdNkPerGroup()` → `groupSize==32` 分支 | 多多多, pergroup, gs=32, K=256, M=64, N=256 | PASS | rtol=1e-3,atol=1e-3 | P0 |
| KW-12 | groupSize=64 VF分支 | `AntiQuantInt4NdNkPerGroup()` → `groupSize==64` 分支 | 多多多, pergroup, gs=64, K=256, M=64, N=256 | PASS | rtol=1e-3,atol=1e-3 | P0 |
| KW-13 | groupSize=128 VF分支 | `AntiQuantInt4NdNkPerGroup()` → `groupSize==128` 分支 | 多多多, pergroup, gs=128, K=512, M=64, N=256 | PASS | rtol=1e-3,atol=1e-3 | P0 |
| KW-14 | groupSize=256 VF分支 | `AntiQuantInt4NdNkPerGroup()` → `groupSize==256` 分支 | 多多多, pergroup, gs=256, K=512, M=64, N=256 | PASS | rtol=1e-3,atol=1e-3 | P0 |

### 9.5 CopyAntiQuantParamsGmToUb PER_GROUP 分支（KW-15 ~ KW-16）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期结果 | 容差 | 优先级 |
|------|------|-------------|----------|----------|------|--------|
| KW-15 | pergroup scale 2D copy(多多多) | `CopyAntiQuantParamsGmToUb()` → `PER_GROUP` 分支 → 2D copy | 多多多, pergroup, scale_shape=(G_i,N_i)=(4,256), M=64, K=512, N=256, gs=128 | PASS (scale正确加载到UB) | rtol=1e-3,atol=1e-3 | P0 |
| KW-16 | pergroup scale 2D copy(单单单) | `CopyAntiQuantParamsGmToUb()` → `PER_GROUP` 分支 → 2D copy(单单单偏移) | 单单单, pergroup, scale_shape=(E,G,N)=(4,4,256), M=256, K=512, N=256, gs=128 | PASS | rtol=1e-3,atol=1e-3 | P0 |

### 9.6 SetGmAddr pergroup 偏移路径（KW-17 ~ KW-18）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期结果 | 容差 | 优先级 |
|------|------|-------------|----------|----------|------|--------|
| KW-17 | pergroup单单单多group偏移序列 | `SetGmAddr()` → `if constexpr (PER_GROUP)` → `antiquantParamsBaseOffset += (kSize/gs)*nSize` | 单单单, E=8, K=512, N=256, gs=128, M=256 | PASS (8个group的scale偏移序列正确) | rtol=1e-3,atol=1e-3 | P0 |
| KW-18 | pergroup bias偏移分离 | `SetGmAddr()` → `biasBaseOffset += nSize`(独立于antiquantParamsBaseOffset) | 单单单, E=4, K=512, N=256, gs=128, 有bias, M=256 | PASS (bias和scale偏移独立) | rtol=1e-3,atol=1e-3 | P0 |

### 9.7 AIC/AIV 协作路径（KW-19 ~ KW-22）

| 编号 | 描述 | 覆盖代码路径 | 输入参数 | 预期结果 | 容差 | 优先级 |
|------|------|-------------|----------|----------|------|--------|
| KW-19 | 多BasicBlock迭代(M>baseM) | `Process()` 中 baseM 动态调整 → 多个 BasicBlock 迭代 | 多多多, pergroup, M=1024, N=256, K=512, gs=128 | PASS (多块拼接正确) | rtol=1e-3,atol=1e-3 | P0 |
| KW-20 | 多BasicBlock迭代(N>baseN) | `Process()` 中 baseN 动态调整 → N方向多块 | 多多多, pergroup, M=256, N=1024, K=512, gs=128 | PASS | rtol=1e-3,atol=1e-3 | P0 |
| KW-21 | K方向多kBlock迭代 | `WeightAntiQuantCompute()` 中 K方向多kBlock循环 | 多多多, pergroup, M=256, N=256, K=2048, gs=128 | PASS (K方向累加正确) | rtol=1e-3,atol=1e-3 | P0 |
| KW-22 | 单BasicBlock(M≤baseM,N≤baseN) | `Process()` 中单块直接计算,无迭代 | 多多多, pergroup, M=64, N=64, K=128, gs=128 | PASS | rtol=1e-3,atol=1e-3 | P1 |

---

## 10. 用例覆盖度分析

### 10.1 约束条件覆盖

| 约束条件 | 覆盖用例 | 覆盖状态 |
|----------|----------|:--------:|
| x dtype ∈ {FP16, BF16} | T-01, P-01~P-09 | ✅ |
| weight dtype = INT4 | T-02 | ✅ |
| bias dtype 匹配 | T-03, T-04, P-01~P-09 | ✅ |
| antiquantScale dtype 与 x 一致 | T-05 | ✅ |
| antiquantOffset dtype 与 x 一致 | T-06 | ✅ |
| groupList dtype = INT64 | T-07 | ✅ |
| x 维度 = 2 | T-08 | ✅ |
| weight 维度 (多多多=2, 单单单=3) | T-09, T-10 | ✅ |
| bias 维度 (多多多=1, 单单单=2) | T-11, T-12 | ✅ |
| antiquantScale 维度 | T-13, T-14 | ✅ |
| groupList 维度 = 1 | T-15 | ✅ |
| K % group_size == 0 | T-16 | ✅ |
| group_size ∈ {32,64,128,256} | T-17, F-01~F-04, F-13~F-16, G-01~G-02 | ✅ |
| weight 最后一维偶数 | T-18 | ✅ |
| pergroup G_i 整除 k_i | T-19 | ✅ |
| 多多多 pergroup s_i 一致 | T-20 | ✅ |
| pergroup 转置 s_i 偶数 | T-21 | ✅ |
| x/weight shape 匹配 | T-22 | ✅ |
| scale shape 匹配 | T-23, T-24 | ✅ |
| bias shape 匹配 | T-25 | ✅ |
| groupList 长度匹配 | T-26 | ✅ |
| ND format | T-27~T-31 | ✅ |
| groupType ∈ {-1, 0} | T-32 | ✅ |
| groupListType ∈ {0, 1} | T-33 | ✅ |
| actType = 0 | T-34 | ✅ |
| splitItem 匹配 | T-35, T-36 | ✅ |
| tuningConfigOptional 不支持 | T-37 | ✅ |
| groupList 条件必选 | T-38, T-39 | ✅ |
| groupList 内容约束 | T-40~T-43 | ✅ |
| weight 转置统一 | T-44 | ✅ |

### 10.2 dtype 组合覆盖

| x dtype | bias | offset | 覆盖用例 |
|---------|------|--------|----------|
| FP16 | FP16 | FP16 | P-01 |
| FP16 | FP16 | null | P-02 |
| FP16 | null | FP16 | P-03 |
| FP16 | null | null | P-04 |
| BF16 | BF16 | BF16 | P-05 |
| BF16 | BF16 | null | P-06 |
| BF16 | FP32 | null | P-07 |
| BF16 | null | BF16 | P-08 |
| BF16 | null | null | P-09 |

### 10.3 groupType × 转置 × group_size 覆盖

| groupType | 转置 | gs=32 | gs=64 | gs=128 | gs=256 |
|:---------:|:----:|:-----:|:-----:|:------:|:------:|
| 单单单(0) | 否 | F-01 | F-02 | F-03 | F-04 |
| 单单单(0) | 是 | — | — | F-05 | — |
| 多多多(-1) | 否 | F-13 | F-14 | F-15 | F-16 |
| 多多多(-1) | 是 | — | — | F-17 | — |

### 10.4 Kernel 关键阶段覆盖

| 阶段 | 覆盖用例 | 覆盖状态 |
|------|----------|:--------:|
| MTE2 搬运 | KS-01~KS-03 | ✅ |
| MTE3 搬运 | KS-04 | ✅ |
| MTE1 搬运 | KS-05~KS-06 | ✅ |
| 输出写回 | KS-07 | ✅ |
| INT4 unpack | KS-08 | ✅ |
| INT4 cast | KS-09 | ✅ |
| scale 加载 (NK/KN) | KS-10, KS-11 | ✅ |
| offset 加载 | KS-12 | ✅ |
| 反量化计算 | KS-13, KS-14 | ✅ |
| 地址偏移 | KS-15~KS-17 | ✅ |
| tiling 参数传递 | KS-18~KS-20 | ✅ |
| matmul 计算 | KS-21~KS-22 | ✅ |

### 10.5 Tiling 白盒代码路径覆盖

| 代码路径 | 覆盖用例 | 覆盖状态 |
|----------|----------|:--------:|
| IsA16W4NDPergroup() 判断 | TW-01, TW-02 | ✅ |
| SetAntiquantGroupSize() 3维scale | TW-04 | ✅ |
| SetAntiquantGroupSize() 2维scale(多多多) | TW-03 | ✅ |
| CheckGroupSize() gs=32 | TW-05 | ✅ |
| CheckGroupSize() gs=64 | TW-06 | ✅ |
| CheckGroupSize() gs=128 | TW-01, TW-04 | ✅ |
| CheckGroupSize() gs=256 | TW-07 | ✅ |
| SetTilingKey() PER_GROUP | TW-08 | ✅ |
| SetTilingKey() PER_CHANNEL | TW-09 | ✅ |
| SetTilingKey() WQ_B_TRANS=TRANS | TW-10 | ✅ |
| SetTilingKey() WQ_B_TRANS=NO_TRANS | TW-11 | ✅ |
| SetShapeList 单单单 | TW-12 | ✅ |
| SetShapeList 多多多 | TW-13 | ✅ |
| SetShapeList INT4 nSize扩展 | TW-14 | ✅ |
| SetBaseTiling 默认值 | TW-15 | ✅ |
| SetBaseTiling hasBias调整 | TW-16 | ✅ |
| EnableTailResplit 多多多不使能 | TW-17 | ✅ |
| EnableTailResplit 单单单ND非转置不使能 | TW-18 | ✅ |
| SetTiling 完整流程(单单单) | TW-19 | ✅ |
| SetTiling 完整流程(多多多) | TW-20 | ✅ |

### 10.6 Kernel 白盒代码路径覆盖

| 代码路径 | 覆盖用例 | 覆盖状态 |
|----------|----------|:--------:|
| if constexpr 无offset无bias非转置 | KW-01 | ✅ |
| if constexpr 无offset有bias非转置 | KW-02 | ✅ |
| if constexpr 有offset无bias非转置 | KW-03 | ✅ |
| if constexpr 有offset有bias非转置 | KW-04 | ✅ |
| if constexpr 无offset无bias转置 | KW-05 | ✅ |
| if constexpr 有offset有bias转置 | KW-06 | ✅ |
| ComputeBasicBlock NdKn(非转置) | KW-07 | ✅ |
| ComputeBasicBlock NdNk(转置) | KW-08 | ✅ |
| GetBufferConfig PER_GROUP | KW-09 | ✅ |
| GetBufferConfig PER_CHANNEL | KW-10 | ✅ |
| VF groupSize=32 分支 | KW-11 | ✅ |
| VF groupSize=64 分支 | KW-12 | ✅ |
| VF groupSize=128 分支 | KW-13 | ✅ |
| VF groupSize=256 分支 | KW-14 | ✅ |
| CopyAntiQuantParamsGmToUb PER_GROUP(多多多) | KW-15 | ✅ |
| CopyAntiQuantParamsGmToUb PER_GROUP(单单单) | KW-16 | ✅ |
| SetGmAddr pergroup scale偏移 | KW-17 | ✅ |
| SetGmAddr bias偏移分离 | KW-18 | ✅ |
| Process 多BasicBlock(M>baseM) | KW-19 | ✅ |
| Process 多BasicBlock(N>baseN) | KW-20 | ✅ |
| WeightAntiQuantCompute K方向多kBlock | KW-21 | ✅ |
| Process 单BasicBlock | KW-22 | ✅ |

---

## 11. 附录

### 11.1 基础配置参数

**多多多基础配置**（groupType=-1）：
- x: FLOAT16, ND, shape (256, 512)
- weight: INT4, ND, shape (256, 512) [非转置: (N, K)]
- antiquantScale: FLOAT16, ND, shape (4, 256) [G=K/gs=512/128=4]
- group_size: 128
- splitItem: 0

**单单单基础配置**（groupType=0）：
- x: FLOAT16, ND, shape (256, 512)
- weight: INT4, ND, shape (4, 256, 512) [非转置: (E, N, K)]
- antiquantScale: FLOAT16, ND, shape (4, 4, 256) [E=4, G=4, N=256]
- groupList: INT64, ND, shape (4,), cumsum [64, 128, 192, 256]
- group_size: 128
- splitItem: 2

### 11.2 Excel 文件说明

配套 Excel 文件 `test_cases_gmm_a16w4_pergroup_nd.xlsx` 包含 8 个 sheet：
- `tiling_exception`: 44 条 tiling 异常拦截用例
- `precision`: 9 条精度用例
- `functional`: 27 条功能用例
- `performance`: 6 条性能用例
- `generalization`: 12 条泛化用例
- `kernel_stub`: 22 条 kernel 打桩验证用例
- `tiling_whitebox`: 20 条 tiling 白盒用例
- `kernel_whitebox`: 22 条 kernel 白盒用例

生成脚本：`generate_test_cases.py`
