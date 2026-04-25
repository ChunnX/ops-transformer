# aclnnGroupedMatmulV4 与 DeepGEMM 参数映射关系

## 1. 参数对照表

### 1.1 输入输出参数

| aclnnGroupedMatmulV4 | DeepGEMM | 说明 |
|---------------------|----------|------|
| `x` | `a.first` | 输入矩阵 A (FP8/FP4/BF16) |
| `weight` | `b.first` | 输入矩阵 B (FP8/FP4/BF16) |
| `out` | `d` | 输出矩阵 |
| `biasOptional` | `c` | 累积/偏置 (可选) |

### 1.2 量化参数

| aclnnGroupedMatmulV4 | DeepGEMM | 说明 |
|---------------------|----------|------|
| `scaleOptional` | `a.second` / `b.second` | Scaling Factor (FP8/FP4 的缩放因子) |
| `offsetOptional` | - | 量化偏移量 (DeepGEMM 不支持) |
| `antiquantScaleOptional` | - | 反量化 scale (伪量化, DeepGEMM 不支持) |
| `antiquantOffsetOptional` | - | 反量化 offset (伪量化, DeepGEMM 不支持) |
| `perTokenScaleOptional` | - | Per-token scale (DeepGEMM 内置在 a.second 中) |

### 1.3 分组参数

| aclnnGroupedMatmulV4 | DeepGEMM | 说明 |
|---------------------|----------|------|
| `groupListOptional` | `grouped_layout` / `masked_m` | 分组布局 |
| `groupType` | - | 分组类型 (映射到不同接口) |
| `groupListType` | - | 分组列表类型 (DeepGEMM 隐式处理) |
| `splitItem` | - | 输出切分 (DeepGEMM 固定) |

### 1.4 配置参数

| aclnnGroupedMatmulV4 | DeepGEMM | 说明 |
|---------------------|----------|------|
| - | `recipe` | 整体量化 recipe |
| - | `recipe_a` | A 矩阵量化 recipe (gran_m, gran_k) |
| - | `recipe_b` | B 矩阵量化 recipe (gran_m, gran_k) |
| - | `compiled_dims` | 编译维度 ("nk", "mn", 等) |
| - | `disable_ue8m0_cast` | 禁用 UE8M0 格式 |
| - | `use_psum_layout` | 使用 PSUM 布局 (SM100) |
| - | `expected_m_for_psum_layout` | PSUM 布局预期 M 值 |
| `actType` | - | 激活函数类型 (DeepGEMM 不支持) |

## 2. groupType 到 DeepGEMM 接口的映射

| groupType | 分组方式 | DeepGEMM 接口 |
|-----------|---------|---------------|
| `-1` | 不分组 | `fp8_fp4_gemm_nt/nn/tn/tt`, `bf16_gemm_*` |
| `0` | M轴分组 | `m_grouped_fp8_fp4_gemm_nt_contiguous`, `m_grouped_bf16_gemm_nt_contiguous` |
| `2` | K轴分组 | `k_grouped_fp8_gemm_tn_contiguous`, `k_grouped_fp8_gemm_nt_contiguous` |

## 3. 数据类型映射

### 3.1 输入数据类型

| aclnnGroupedMatmulV4 | DeepGEMM | 备注 |
|---------------------|----------|------|
| `FLOAT16` | `torch.float16` | ✅ |
| `BFLOAT16` | `torch.bfloat16` | ✅ |
| `FLOAT32` | `torch.float32` | 仅 BF16 GEMM |
| `INT8` | `torch.float8_e4m3fn` | FP8 格式 |
| `INT4` | `uint8` (packed) | FP4 格式 |
| `FLOAT8_E4M3FN` | `torch.float8_e4m3fn` | ✅ |
| `FLOAT8_E5M2` | `torch.float8_e5m2` | ✅ |
| `HIFLOAT8` | 自定义 | DeepGEMM 不支持 |

### 3.2 量化模式映射

| aclnnGroupedMatmulV4 量化模式 | DeepGEMM 量化配置 | 说明 |
|------------------------------|------------------|------|
| 非量化 | 无 recipe | BF16/FP16/FP32 |
| T-T (per-tensor) | `recipe=(1,1,128)` | Per-tensor 量化 |
| T-C (per-channel) | `recipe_a=(1,128)`, `recipe_b=(1,N)` | Per-channel 量化 |
| K-C (per-group K) | `recipe_a=(1,128)` | K 方向 per-group |
| K-T | `recipe_a=(1,128)` | K 方向 per-tensor |
| MX 量化 | QuantConfig | DeepGEMM 特殊处理 |
| G-B 量化 | QuantConfig | DeepGEMM 特殊处理 |

## 4. 使用示例对比

### 4.1 aclnnGroupedMatmulV4 (M轴分组)

```python
# aclnnGroupedMatmulV4
aclnnGroupedMatmulV4(
    x=x,                          # [M, K]
    weight=weight,                # [num_experts, N, K]
    biasOptional=bias,
    scaleOptional=scale,          # [num_experts, N]
    groupListOptional=group_list,  # [M]
    groupType=0,                  # M轴分组
    groupListType=0,
    actType=0,
    out=out
)
```

### 4.2 DeepGEMM (m_grouped_fp8_fp4_gemm_nt_contiguous)

```python
# DeepGEMM
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a=(a_fp8, sfa),               # FP8 A + Scaling Factor
    b=(b_fp8, sfb),               # FP8 B + Scaling Factor
    d=output,                     # 输出
    grouped_layout=group_layout,  # 分组布局
    recipe_a=(1, 128),            # 量化配置
    recipe_b=(1, 128),
    compiled_dims="nk",           # 编译维度
    use_psum_layout=False,
    expected_m_for_psum_layout=None
)
```

## 5. 参数详细说明

### 5.1 DeepGEMM 独有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `recipe` | `Tuple[int,int,int]` | `None` | 整体量化 recipe: `(gran_m, gran_n, gran_k)` |
| `recipe_a` | `Tuple[int,int]` | `None` | A 矩阵量化 recipe: `(gran_m, gran_k)` |
| `recipe_b` | `Tuple[int,int]` | `None` | B 矩阵量化 recipe: `(gran_m, gran_k)` |
| `compiled_dims` | `str` | `"nk"` | 编译维度: `"nk"`, `"mn"`, `"mnk"`, `""` |
| `disable_ue8m0_cast` | `bool` | `False` | 是否禁用 UE8M0 格式 |
| `use_psum_layout` | `bool` | `False` | 是否使用 PSUM 布局 (仅 SM100) |
| `expected_m_for_psum_layout` | `int` | `None` | PSUM 布局预期 M 值 |

### 5.2 aclnnGroupedMatmulV4 独有参数

| 参数 | 说明 |
|------|------|
| `splitItem` | 输出是否切分 (0/1: 多tensor, 2/3: 单tensor) |
| `groupType` | 分组轴: -1: 不分组, 0: M轴, 2: K轴 |
| `groupListType` | 分组列表类型: 0: cumsum, 1: 每组大小, 2: 索引+大小 |
| `actType` | 激活函数: 0: none, 1: relu, 2: gelu_tanh, 3: gelu_err, 4: fast_gelu, 5: silu |
| `offsetOptional` | 量化偏移量 |
| `antiquantScaleOptional` | 反量化 scale (伪量化) |
| `antiquantOffsetOptional` | 反量化 offset (伪量化) |
| `activationInputOptional` | 激活输入 (当前仅支持 nullptr) |
| `activationQuantScaleOptional` | 激活量化 scale |
| `activationQuantOffsetOptional` | 激活量化 offset |
| `activationFeatureOutOptional` | 激活特征输出 |
| `dynQuantScaleOutOptional` | 动态量化 scale 输出 |

## 8. 补充参数详解

### 8.1 compiled_dims - 编译维度

**作用**: 指定哪些维度在 CUDA JIT 编译时编译为常量，让编译器生成更优代码。

**取值**:

| compiled_dims | SHAPE_M | SHAPE_N | SHAPE_K |
|--------------|---------|---------|---------|
| `"nk"` (默认) | 动态 | 常量 | 常量 |
| `"mn"` | 常量 | 常量 | 动态 |
| `"kn"` | 动态 | 常量 | 常量 |
| `"mnk"` | 常量 | 常量 | 常量 |
| `""` | 动态 | 动态 | 动态 |

**举例**:

```python
# 矩阵形状: m=4096, n=7168, k=4096

# 默认: N和K编译为常量
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a, b, d, grouped_layout,
    compiled_dims="nk"  # SHAPE_N=7168, SHAPE_K=4096, SHAPE_M=0(动态)
)

# M和N固定，K动态变化
compiled_dims="mn"  # SHAPE_M=4096, SHAPE_N=7168, SHAPE_K=0(动态)

# 形状完全固定
compiled_dims="mnk"  # 全部常量
```

### 8.2 disable_ue8m0_cast - 禁用 UE8M0 格式

**作用**: 控制是否使用 UE8M0 (Unsigned E8M0) 格式存储 scaling factor。

**UE8M0 格式**: SM100 特有的紧凑格式，将 FP32 scaling factor 压缩为 INT8 存储。

| disable_ue8m0_cast | 说明 |
|-------------------|------|
| `False` (默认) | SM100 上自动使用 UE8M0 格式，节省显存 |
| `True` | 禁用 UE8M0，使用原始 FP32 格式 |

**代码逻辑** (layout.hpp:35-49):

```cpp
// FP32 on SM90: always transform to TMA-aligned
if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
    return get_mn_major_tma_aligned_tensor(sf);

// FP32 on SM100: transform to UE8M0 format (if not disabled)
if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and arch_major == 10) {
    DG_HOST_ASSERT(not disable_ue8m0_cast);  // 禁用时不能使用 UE8M0
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);
}
```

**举例**:

```python
# 正常使用 (SM100)
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a, b, d, grouped_layout,
    disable_ue8m0_cast=False  # 自动使用 UE8M0 格式
)

# 禁用 UE8M0 (可能影响性能，但节省调试时间)
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a, b, d, grouped_layout,
    disable_ue8m0_cast=True   # 使用原始 FP32 格式
)
```

### 8.3 use_psum_layout - PSUM 布局 vs groupListType

**作用**: 控制分组布局的存储方式，与 aclnn 的 groupListType 功能类似但实现不同。

#### groupListType (aclnnGroupedMatmulV4)

| groupListType | groupListOptional 内容 | 例子 |
|---------------|----------------------|------|
| `0` | cumsum (累积和) | `[123, 456, 789]` |
| `1` | 每组大小 | `[123, 333, 333]` |
| `2` | 索引+大小 | `[[0,123], [2,333], [5,333]]` |

#### use_psum_layout (DeepGEMM)

| use_psum_layout | grouped_layout 内容 | shape | 架构限制 |
|-----------------|---------------------|-------|----------|
| `False` (默认) | 每行分组ID (-1填充) | `[M]` | SM90/SM100 |
| `True` | 累积M值 | `[G]` (num_groups) | 仅 SM100 |

**对比举例**:

```python
# 假设: num_groups=4, m_per_group=4096, 总m=16384

# aclnn groupListType=0 (cumsum)
groupListOptional = [4096, 8192, 12288, 16384]

# DeepGEMM use_psum_layout=True
grouped_layout = torch.tensor([4096, 8192, 12288, 16384], device='cuda', dtype=torch.int32)

# aclnn groupListType=1 (每组大小)
groupListOptional = [4096, 4096, 4096, 4096]

# DeepGEMM use_psum_layout=False
grouped_layout = torch.zeros(16384, dtype=torch.int32, device='cuda')
for i in range(4):
    grouped_layout[i*4096:(i+1)*4096] = i
# 填充区域: grouped_layout[16384:] = -1
```

**关键区别**:

| 特性 | groupListType (aclnn) | use_psum_layout (DeepGEMM) |
|------|----------------------|---------------------------|
| 架构 | Ascend NPU | NVIDIA GPU (仅SM100) |
| 布局 | 多种格式 (0/1/2) | 两种格式 |
| 性能 | 标准优化 | 更高性能 (SM100) |
| shape | 灵活 | `[M]` 或 `[G]` |

### 8.4 expected_m_for_psum_layout - PSUM 布局预期 M 值

**作用**: 当 `use_psum_layout=True` 时，指定预期的每组 M 值，帮助 JIT 编译器选择最优配置。

**取值**: 通常设为预期的每组平均 M 值

**使用场景**:

```python
# 参数
num_groups = 4
m_per_group = 4096  # 预期每组 M 值

# 调用
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a, b, d, grouped_layout,  # shape [4] = [4096, 8192, 12288, 16384]
    use_psum_layout=True,
    expected_m_for_psum_layout=m_per_group  # 4096
)
```

**动态 M 场景**:

```python
# 实际 M 动态变化: [3500, 4200, 3800, 4100]
# 使用最大值作为 expected_m

expected_m = max(3500, 4200, 3800, 4100)  # 4200

deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a, b, d, grouped_layout,
    use_psum_layout=True,
    expected_m_for_psum_layout=expected_m
)
```

### 8.5 recipe_a / recipe_b - 量化粒度

**作用**: 指定 A/B 矩阵的量化粒度，映射到 aclnnGroupedMatmulV4 的量化模式。

#### recipe_a 格式

```python
recipe_a = (gran_m, gran_k)
```

| gran_m | gran_k | aclnn 量化模式 | 说明 |
|--------|--------|---------------|------|
| `1` | `128` | per-tensor (T-T) | 每行一个 scale |
| `1` | `32` | per-tensor (T-T) | 每行一个 scale, K=32 |
| `128` | `128` | per-channel (T-C) | 每 128 行一个 scale |

#### recipe_b 格式

```python
recipe_b = (gran_n, gran_k)
```

| gran_n | gran_k | aclnn 量化模式 | 说明 |
|--------|--------|---------------|------|
| `1` | `128` | per-tensor (T-T) | 每列一个 scale |
| `N` | `128` | per-channel (T-C) | 每列一个 scale (per-channel) |
| `1` | `32` | per-tensor (T-T) | K=32 粒度 |

#### 量化模式映射表

| DeepGEMM recipe | aclnn 量化模式 | scale shape (A) | scale shape (B) |
|-----------------|---------------|-----------------|------------------|
| `recipe_a=(1,128)` | T-T (per-tensor) | `[M/128, 1]` | `[N/128, 1]` |
| `recipe_a=(128,128)` | T-C (per-channel) | `[M, 1]` | `[N, 1]` |
| `recipe_b=(1,128)` | T-T (per-tensor) | - | `[1, N/128]` |
| `recipe_b=(N,128)` | T-C (per-channel) | - | `[1, N]` |

**举例**:

```python
# DeepGEMM: FP8 per-tensor 量化
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a=(a_fp8, sfa),  # sfa shape: [M/128, 1]
    b=(b_fp8, sfb),  # sfb shape: [1, N/128]
    d=output,
    grouped_layout=grouped_layout,
    recipe_a=(1, 128),  # per-tensor
    recipe_b=(1, 128),  # per-tensor
    compiled_dims="nk"
)

# DeepGEMM: FP8 per-channel 量化 (B矩阵)
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a=(a_fp8, sfa),
    b=(b_fp8, sfb),  # sfb shape: [1, N]
    d=output,
    grouped_layout=grouped_layout,
    recipe_a=(1, 128),
    recipe_b=(N, 128),  # per-channel: gran_n = N
    compiled_dims="nk"
)
```

#### 实际代码逻辑 (generators.py:54-61)

```python
def get_recipes(self, is_wgrad: bool = False):
    recipe, recipe_a, recipe_b = None, None, None
    if self.is_legacy():
        recipe = (1, 1, 128) if is_wgrad else None
    else:
        recipe_a = (1, self.gran_k_a)      # gran_m=1 (per-token/row)
        # FP4 或 wgrad: gran_n=1 (per-tensor)
        # FP8: gran_n=gran_k_b (per-channel)
        recipe_b = (1, self.gran_k_b) if self.is_fp4_b or is_wgrad else (self.gran_k_b, self.gran_k_b)
    return recipe, recipe_a, recipe_b
```

## 9. 参数速查表

### DeepGEMM 完整参数列表

| 参数 | 类型 | 默认值 | 对应 aclnn |
|------|------|--------|------------|
| `a` | `Tuple[Tensor, Tensor]` | 必选 | `x` + `scaleOptional` |
| `b` | `Tuple[Tensor, Tensor]` | 必选 | `weight` + `scaleOptional` |
| `d` | `Tensor` | 必选 | `out` |
| `c` | `Tensor` | 可选 | `biasOptional` |
| `grouped_layout` | `Tensor` | 必选 | `groupListOptional` |
| `recipe` | `Tuple[int,int,int]` | `None` | 量化配置 |
| `recipe_a` | `Tuple[int,int]` | `None` | A 矩阵量化粒度 |
| `recipe_b` | `Tuple[int,int]` | `None` | B 矩阵量化粒度 |
| `compiled_dims` | `str` | `"nk"` | - |
| `disable_ue8m0_cast` | `bool` | `False` | - |
| `use_psum_layout` | `bool` | `False` | `groupListType` (类似) |
| `expected_m_for_psum_layout` | `int` | `None` | - |

## 6. 功能对比总结

| 功能 | aclnnGroupedMatmulV4 | DeepGEMM |
|------|---------------------|----------|
| M轴分组 | ✅ | ✅ |
| K轴分组 | ✅ | ✅ |
| 不分组 | ✅ | ✅ |
| FP8 量化 | ✅ | ✅ |
| FP4 量化 | ✅ | ✅ |
| BF16 | ✅ | ✅ |
| 激活函数 | ✅ (actType) | ❌ |
| 伪量化 | ✅ | ❌ |
| UE8M0 | ✅ | ✅ |
| PSUM 布局 | ❌ | ✅ (SM100) |
| CUDA Graph | ✅ | ✅ |

## 7. 架构差异

| 特性 | aclnnGroupedMatmulV4 | DeepGEMM |
|------|---------------------|----------|
| 目标硬件 | Ascend NPU | NVIDIA GPU (Hopper) |
| 架构 | CANN | CUDA |
| 接口层级 | ACLNN C++ API | Python + JIT 编译 |
| 量化灵活性 | 高 (多种量化模式) | 中 (主要 FP8/FP4) |
| 性能优化 | 静态图优化 | JIT 动态编译 |
