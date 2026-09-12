# GroupedMatmulSwigluQuantV2 MXFP8 Tiling 与 Kernel 实现分析

## 1. 文档目的与结论先行

本文分析 GroupedMatmulSwigluQuantV2 在 Ascend 950（代码目录中的 arch35）上的 MXFP8 实现，重点覆盖：

- host 侧如何识别 MXFP8，并选择 tiling 模板；
- WeightNZ 路径如何计算 shape、量化参数、L1/L0 tile 和 tiling key；
- device 侧如何按照 groupList 逐组执行 grouped matmul；
- 两个矩阵乘结果如何拼成 SwiGLU 的两个输入；
- AIC 上的矩阵乘、AIV 上的 SwiGLU 和 MX 量化如何通过跨核同步串起来；
- Blaze Tensor API 路径如何组织 BlockMmad、BlockScheduler、BlockEpilogue 和 GemmUniversal。

最重要的结论是：

1. **V2 的 MXFP8 不是一条唯一 kernel 路径。**
   - weight 为 FRACTAL_NZ 时，当前实现走原有 Cgmct/Cube-Vector 融合 kernel：GmmSwigluAswt<RowMajor, Nz/Zn>。
   - weight 为 ND/NCL 且满足更严格的尺寸和平台条件时，host 会优先选择 Tensor API tiling，device 侧再走 Blaze kernel：GmmTensorApiSwigluQuantMxFp8Kernel。

2. **V2 的 WeightNZ MXFP8 目前并没有直接使用 Blaze 的 kernel_qgmm_swiglu_mx.h。**
   Blaze 的 MXFP8 SwiGLU kernel 当前通过静态断言限制 LayoutB 为 ND/DN，不能直接承载 NZ/ZN 权重。因此，V3 如果仍然只支持 WeightNZ MXFP8，首先应复用和改造 V2 的 Cgmct NZ 路径；不能只复制 Blaze 的 ND 路径并修改一个 layout 名称。

3. **clamp 属于 SwiGLU 后处理阶段，不属于 matmul tile。**
   对 V3 而言，matmul 的 M/N/K 分块、NZ 权重搬运、MXFP8 输入反量化仍可复用 V2；主要变化应落在 AIV epilogue 的 SwiGLU 计算阶段，并且必须把 V3 的 SwiGLU 参数通过 tiling data 传到 kernel。

本文中的“MXFP8”指输入 x 和 weight 使用 FP8 元素，xScale、weightScale 和输出 yScale 使用 E8M0 scale，并且每 64 个元素共享 1 个 MX scale block；每个 scale block 在内存中占 2 个字节。

---

## 2. 代码范围和入口文件

### 2.1 host 侧

主要文件位于：

~~~text
gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/
~~~

| 文件 | 作用 |
| --- | --- |
| grouped_matmul_swiglu_quant_v2_tiling.cpp | 总 tiling dispatcher、950/非 950 模板选择、compile info 准备 |
| arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.cpp | 950 原始 Cgmct 路径的 dtype、shape、量化参数、L1/L0 tiling |
| arch35/grouped_matmul_swiglu_quant_v2_basic_api_tiling.cpp | 950 Tensor API/Blaze 路径的 capability 判断和 Tensor API tiling |
| grouped_matmul_swiglu_quant_v2_base_tiling.cpp | 通用 grouped matmul tiling 基类能力，例如基础 block、L1 空间计算 |
| arch35/grouped_quant_matmul_tiling.cpp | MX 量化 scale shape 检查等通用逻辑 |
| grouped_matmul_swiglu_quant_v2_host_utils.h | 输入、输出和属性索引，以及 MX shape 常量 |

### 2.2 device 侧

主要文件位于：

~~~text
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/
~~~

| 文件 | 作用 |
| --- | --- |
| grouped_matmul_swiglu_quant_v2_apt.cpp | 根据编译期 tiling key、权重 format 和量化类型选择具体 kernel 模板 |
| arch35/grouped_matmul_swiglu_quant_v2_mxquant.h | 原始 Cgmct MXFP8 kernel wrapper，组装 MMAD、scheduler、epilogue |
| arch35/grouped_matmul_tensor_api_swiglu_quant_v2_mxfp8_kernel.h | Blaze Tensor API MXFP8 kernel wrapper |
| arch35/grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h | Blaze 路径使用的 tiling data 结构 |
| grouped_matmul_swiglu_quant_v2_tiling_key.h | transA、transB、kernel 类型三个维度组成的 tiling key |

### 2.3 Cgmct 和 Blaze 公共模板

WeightNZ 路径依赖 Cgmct：

~~~text
gmm/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h
gmm/common/cgmct/block/...
gmm/common/cgmct/epilogue/...
~~~

Tensor API ND 路径依赖用户指定的 Blaze 目录：

~~~text
C:\hw-workspace\ops-tensor_\include\blaze
~~~

其中与本算子直接相关的文件是：

~~~text
blaze/gemm/block/block_mmad_qgmm_mx.h
blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h
blaze/gemm/kernel/kernel_qgmm_swiglu_mx.h
blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h
blaze/gemm/policy/dispatch_policy.h
~~~

---

## 3. MXFP8 的整体数据流

设：

- M：总 token 数；
- K：matmul 的归约维；
- N：matmul 的完整输出维，也就是 SwiGLU 拆分前的列数；
- E：专家数，也就是 group 数；
- N/2：SwiGLU 输出维。

逻辑数据流如下：

~~~text
x[M, K]                     FP8
       │
       │ xScale[M, ceil(K/64), 2]       E8M0
       │
       ├───────────────┐
       │               │
weight[e, K, N]        │ weightScale[e, ceil(K/64), N, 2]  E8M0
FP8, 逻辑布局          │
实际 WeightNZ 存储      │
       │               │
       └─────── MXFP8 grouped matmul ───────┐
                                             │
                         C[e, token, N]      │
                         中间结果，通常在   │
                         L0C/UB 中以 float 处理
                                             │
                         split: [act | gate]
                                             │
                         SwiGLU
                         swish(act) * gate
                                             │
                         MXFP8 quant
                                             │
                         y[M, N/2]            FP8
                         yScale[M, ceil((N/2)/64), 2] E8M0
~~~

MXFP8 的逻辑计算公式可以写成：

$$
C_i = (X_i \times W_i) \odot xScale_i \odot weightScale_i
$$

然后沿最后一维拆成两个长度为 N/2 的矩阵：

$$
C_i = [A_i \mid G_i]
$$

V2 的标准 SwiGLU 为：

$$
S_i = Swish(A_i) \odot G_i
$$

其中：

$$
Swish(x)=\frac{x}{1+e^{-x}}
$$

最后对 S_i 按输出方向每 64 个元素做 MXFP8 量化。每个 64 元素 block 的共享指数由 block 内最大绝对值决定：

$$
shared\_exp = \left\lfloor \log_2(\max_j |S_{i,j}|) \right\rceil - emax
$$

$$
QScale = 2^{shared\_exp}
$$

$$
Q_{i,j}=quantize\_to\_element\_format(S_{i,j}/QScale)
$$

其中 emax 由目标输出类型决定。V2 的 MXFP8 WeightNZ 重点场景中，目标通常是 FP8；yScale 保存每个输出 MX block 的 E8M0 scale。

---

## 4. 输入、输出和内存布局

### 4.1 逻辑 shape 和 dtype

下面以单 Tensor WeightNZ、非转置权重为主说明；转置权重只改变权重和 scale 的二维逻辑顺序。

| 数据 | 逻辑 shape | dtype | 说明 |
| --- | --- | --- | --- |
| x | [M, K] | FP8 | 激活矩阵，V2 NZ FP8 检查要求 x/weight 的 FP8 类型满足对应组合；当前 NZ FP8 分支主要按 E4M3FN 组合验证 |
| weight | [E, K, N] | FP8 | 逻辑上是 E 个 [K,N] 权重矩阵；单 Tensor 表示多个 group 的权重 |
| xScale | [M, ceil(K/64), 2] | FLOAT8_E8M0 | 每个 token 在 K 方向每 64 个元素共享一个 scale |
| weightScale | [E, ceil(K/64), N, 2] | FLOAT8_E8M0 | 单 Tensor 权重的每个专家、K block、N 列各有 2 字节 scale |
| groupList | [E] | 通常为 INT64 | groupListType=0 表示 cumsum，1 表示每组 token count |
| y | [M, N/2] | FP8 | SwiGLU 后的 MXFP8 输出 |
| yScale | [M, ceil((N/2)/64), 2] | FLOAT8_E8M0 | 输出按最后一维每 64 个元素的 MX scale |

V2 host 侧使用 MXFP_BASEK_FACTOR=64、MXFP_MULTI_BASE_SIZE=2，因此 K 方向的 scale span 是：

~~~text
scaleK = ceil(K / 64) * 2
~~~

输出方向同理：

~~~text
outputScaleK = ceil((N / 2) / 64) * 2
~~~

### 4.2 WeightNZ 的物理含义

weight 的逻辑 shape 和物理 storage shape 不是一回事。host 侧通过 descriptor 的 primary format 识别 FRACTAL_NZ，device 侧以 NZ/ZN layout 访问权重。

对于 FP8 的非转置 NZ 权重，V2 kernel 中每个 group 的权重地址步长按下式计算：

~~~text
perGroupBOffset = Align32(N) * Align16(K)
~~~

对于转置权重 Zn，地址步长为：

~~~text
perGroupBOffset = Align16(N) * Align32(K)
~~~

这里的 Align32 对应 FP8 NZ 的 C0/列块粒度，Align16 对应 K 方向的 cube 对齐。Cgmct kernel 中还会根据 FP4/FP8 选择不同的对齐值，但 MXFP8 走的是上面两条 FP8 公式。

这也是为什么不能把 ND kernel 简单地改一个模板参数就当成 NZ kernel：NZ 不只是 shape 名字不同，GM 地址步长、L1 搬运布局、scale 搬运布局和 tail 对齐都不同。

### 4.3 groupList 的两种语义

device 侧在两个实现中都使用同样的分组语义：

~~~text
groupListType == 0: groupList 存累计结束位置
    currentM = groupList[e] - previousEnd
    previousEnd = groupList[e]

groupListType == 1: groupList 存每组 token 数
    currentM = groupList[e]
    previousEnd += currentM
~~~

每组的 token 在 x、xScale 和输出 y/yScale 中都是连续切片。一个 group 的 matmul 结果只写该 group 对应的输出行区间。

---

## 5. host 侧 tiling 模板选择

### 5.1 compile info 准备

TilingPrepareForGMMSwigluQuantV2 读取平台信息并保存到 GMMSwigluV2CompileInfo：

- aicNum：Cube 核数量；
- aivNum：Vector 核数量；
- ubSize：UB 大小；
- supportL12BtBf16：平台是否支持所需的 L1/L2 BF16 数据搬运能力。

950 分支由 supportL12BtBf16 进入。代码并不是通过一个写死的字符串比较来选择 arch35，而是通过平台能力决定是否注册 950 tiling 模板。

### 5.2 dispatcher 的选择逻辑

GroupedMatmulSwigluQuantV2TilingFunc 的核心逻辑可以概括为：

~~~text
if platform supports 950 capability:
    if x is FP8_E4M3FN and weight is FP4_E2M1:
        register weight-quant template
    else if xScale is FLOAT8_E8M0 and weight format is ND/NCL:
        register Tensor API template first
        register original 950 template as fallback
    else:
        register original 950 template
else:
    register int8 fusion/base templates
~~~

因此：

- **ND/NCL + MXFP8**：先尝试 Blaze Tensor API；若不满足 Tensor API 的 capability 条件，回退到原始 Cgmct 950 tiling。
- **FRACTAL_NZ + MXFP8**：不会进入 Tensor API 优先分支，直接使用原始 950 tiling，也就是后续的 Cgmct GmmSwigluAswt。

这个选择逻辑位于：

~~~text
gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/
grouped_matmul_swiglu_quant_v2_tiling.cpp
~~~

对应的关键分支是 GroupedMatmulSwigluQuantV2TilingFunc 中对 xScaleDtype 和 weightFormat 的判断。

### 5.3 两套 tiling data

原始 Cgmct 路径保存的是：

~~~text
GMMSwigluQuantTilingDataParams
├── gmmSwigluQuantParams
└── mmTilingData
~~~

其中 gmmSwigluQuantParams 至少包含：

~~~text
groupNum
groupListType
isMxWeightNzMultiTensor
quantDtype
~~~

mmTilingData 保存 Cube matmul 使用的 tile 和 L1/L0 参数。

Blaze Tensor API 路径保存的是：

~~~text
GMMSwigluQuantV2TensorApiTilingData
├── gmmQuantParams
└── mmTilingData
~~~

gmmQuantParams 主要包含：

~~~text
groupNum
activeType
aQuantMode
bQuantMode
singleX
singleW
singleY
groupType
groupListType
hasBias
~~~

mmTilingData 主要包含：

~~~text
m, n, k
baseM, baseN, baseK
kAL1, kBL1
scaleKAL1, scaleKBL1
dbL0C
~~~

两套结构不同，说明 V2 并不是同一个通用 kernel 仅仅更换宏，而是存在两套不同的 kernel 组织方式。

---

## 6. 原始 950 tiling：WeightNZ MXFP8 路径

本节是 V3 WeightNZ MXFP8 最应该直接参考的部分。

### 6.1 属性校验

GroupedMatmulSwigluQuantV2Tiling950::ValidateAttrsCommon 和相关函数会检查：

- dequant_mode == 2，表示 MX 反量化模式；
- quant_mode == 2，表示 MX 量化模式；
- dequant_dtype == DT_FLOAT；
- transpose_weight 决定 transB；
- group_list_type 只能为 0 或 1；
- group 类型固定为 SPLIT_M。

其中 SPLIT_M 意味着不同 group 沿 M/token 方向切分，每组使用自己的权重矩阵。

### 6.2 dtype 识别

host 侧将 MXFP8 识别为：

~~~text
IsMXFp8Input() ==
    IsFp8(x.dtype) && IsFp8(weight.dtype) && weightScale.dtype == FLOAT8_E8M0
~~~

FP8 类型判断函数允许 FLOAT8_E4M3FN 和 FLOAT8_E5M2。但对于 FRACTAL_NZ，CheckWeightNzDtype 又对 x/weight 的组合做了额外限制；当前代码中的 FP8 NZ 分支要求 x 和 weight 都为 DT_FLOAT8_E4M3FN。

随后 CheckDtype 还会检查：

- x 和 weight 不能一个 FP8、一个 FP4；
- x 和 weight 必须同为 FP8 或同为 FP4；
- weightScale 和 xScale 必须是 DT_FLOAT8_E8M0；
- 输出 y 必须是 FP8 或 FP4；
- yScale 必须是 DT_FLOAT8_E8M0；
- FP8 输入不能生成 FP4 输出。

对于 V3 只做 WeightNZ MXFP8 时，建议把支持范围进一步收窄成明确的 FP8 输入、FP8 输出和 E8M0 scale，避免复制 V2 中 FP4、per-token 等无关分支。

### 6.3 quant mode 设置

当 IsMicroScaling() 成立时：

~~~text
aQuantMode = MX_PERGROUP_MODE
bQuantMode = MX_PERGROUP_MODE
~~~

这意味着 x 和 weight 都按照 K 方向的 64 元素 block 使用 MX scale，而不是 per-token 或 per-channel scale。

### 6.4 shape 校验

原始 tiling 对 MXFP8 主要检查：

1. xScale 必须是 3 维。
2. 单 Tensor WeightNZ 的 weightScale 必须是 4 维。
3. K 方向 block 数为：

   ~~~text
   kBlocks = ceil(K / 64)
   ~~~

4. 单 Tensor WeightNZ 的 scale 逻辑 shape 为：

   ~~~text
   xScale       = [M, kBlocks, 2]
   weightScale  = [E, kBlocks, N, 2]      // 非转置
   weightScale  = [E, N, kBlocks, 2]      // 转置
   ~~~

5. xScale 最后一维固定为 2，代表 E8M0 scale 的存储方式。
6. MXFP8 + WeightNZ 时，N 必须是 64 的整数倍。
7. group 数不能超过通用 GMM 的最大 group list 长度。
8. AIC/AIV 数量必须满足：

   ~~~text
   aivNum == 2 * aicNum
   ~~~

CheckMxPerGroupShape 的通用实现还兼容 scale 的 NK/KN 逻辑排列以及多 Tensor NZ 场景；V3 当前只支持单 Tensor 时，可以只保留单 Tensor 相关检查。

### 6.5 M、N、K 的确定

V2 host 先从 x 和第一个 weight descriptor 中得到逻辑 M、N、K：

~~~text
M = x 的倒数第二维
K = x 的最后一维

transpose_weight == false:
    weightK = weight 的倒数第二维
    N       = weight 的最后一维

transpose_weight == true:
    weightK = weight 的最后一维
    N       = weight 的倒数第二维
~~~

要求 weightK == K。

注意：这里的 N 是 matmul 的完整输出列数，不是最终 y 的列数。SwiGLU 在 kernel 中将 N 拆成两个 N/2。

### 6.6 基础 tile 和 L1/L0 参数

原始 950 tiling 的主要步骤如下：

~~~text
CalBasicBlock()
baseM = min(baseM, 128)
baseM = align_up(baseM, CUBE_BLOCK)

if MXFP4 + WeightNZ:
    CalWeightNzL1Tiling()
else:
    CalL1Tiling()
~~~

因此，**MXFP8 + WeightNZ 使用普通的 CalL1Tiling()，不是专门的 MXFP4 WeightNZ L1 策略。**

最终写入 mmTilingData 的字段包括：

| 字段 | 含义 |
| --- | --- |
| M/N/Ka/Kb | matmul 的逻辑问题规模；对于本场景 Ka=Kb=K |
| usedCoreNum | 使用的 AIC 数量 |
| baseM/baseN/baseK | L0/Cube 的基础 tile |
| singleCoreM/singleCoreN/singleCoreK | 单核分配规模 |
| depthA1/depthB1 | A、B 在 L1 的 double buffer 深度 |
| stepM/stepN | M/N 方向的搬运步长 |
| stepKa/stepKb | A/B 在 K 方向的 L1 步长 |
| isBias | 是否有 bias；MXFP8 重点路径通常不使用 |
| iterateOrder | K/M/N 的迭代顺序 |
| dbL0A/dbL0B | 都被设置为 2，打开 L0 A/B double buffer |
| dbL0C | L0C double buffer 配置 |
| mxTypePara | MX 类型相关的 scale factor 编码 |

mxTypePara 会将 scaleFactorA、scaleFactorB 等信息按 bit field 打包。当计算出的 scale factor 不在允许范围内时，代码会回退到默认编码。

### 6.7 L1 深度和 scale factor 的含义

MXFP8 的 L1 空间不只需要保存 A/B，还需要保存对应的 x/weight scale：

~~~text
baseASize      = size(baseM * baseK, dtypeA)
baseBSize      = size(baseN * baseK, dtypeB)
baseScaleASize = size(ceil(baseK / 64) * baseM, scaleA)
baseScaleBSize = size(ceil(baseK / 64) * baseN, scaleB)
~~~

实际实现还会结合 L1 剩余空间、A/B depth、双缓冲和 MTE2 最小搬运量调整 depth。核心目标是：

~~~text
depthA1 * baseASize
+ depthB1 * baseBSize
+ max(depthA1, depthB1) * (baseScaleASize + baseScaleBSize)
<= 可用 L1 空间
~~~

scaleFactorA/B 的作用是决定一个 L1 tile 对应多少个 K 方向 scale block，并受到：

- K 方向 tile 步长；
- scale 的单次搬运上限；
- 可用 L1 空间；
- A/B 两侧 depth 平衡；

共同约束。

### 6.8 tiling key

原始 950 路径的 key 是：

~~~cpp
GET_TPL_TILING_KEY(
    static_cast<uint64_t>(transB),
    static_cast<uint64_t>(transA),
    GMM_SWIGLU_QUANT_ORIGINAL_KERNEL_TYPE)
~~~

这里本算子固定 transA=false；transB 由 transpose_weight 决定。第三个字段为 original kernel type，device 侧据此实例化 Cgmct 路径。

tiling 结束时：

- blockDim = aicNum；
- schedule mode 设置为 1；
- 序列化 GMMSwigluQuantTilingDataParams。

---

## 7. 原始 Cgmct MXFP8 WeightNZ kernel

### 7.1 kernel 入口选择

grouped_matmul_swiglu_quant_v2_apt.cpp 先读取：

- ORIG_DTYPE_X_SCALE，判断是否为 MX（E8M0）；
- weightFormat，判断是否为 NZ；
- QUANT_A_TRANS、QUANT_B_TRANS；
- KERNEL_TYPE，判断 original 或 tensor-level。

MXFP8 + WeightNZ 时：

~~~text
transA = false, transB = false:
    GmmSwigluAswt<RowMajor, Nz>(...)

transA = false, transB = true:
    GmmSwigluAswt<RowMajor, Zn>(...)
~~~

GmmSwigluAswt 位于：

~~~text
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/
grouped_matmul_swiglu_quant_v2_mxquant.h
~~~

它不是一个具体的手写 matmul，而是把 Cgmct 的多个 block 组件组装成：

~~~text
BlockScheduler
    = GroupedMatmulAswtWithTailSplitScheduler

BlockMmad
    = BlockMxMmAicToAivBuilder<...
          QuantMatmulWithTileMultiBlock<>,
          TileCopy<..., CopyInAndCopyOutSplitMWithParams>>

BlockEpilogue
    = BlockEpilogueSwigluQuant<...>

Kernel
    = KernelGmmSwiGluMixOnlineDynamic<...>
~~~

### 7.2 kernel 中的数据类型

原始 Cgmct wrapper 的类型关系是：

| 阶段 | 类型 |
| --- | --- |
| A/GMem x | DTYPE_X，MXFP8 |
| B/GMem weight | DTYPE_WEIGHT，MXFP8 |
| xScale/weightScale | fp8_e8m0_t |
| Cube 累加结果 | C1Type=float |
| SwiGLU 输入 | 两个 float tile，分别对应 act 和 gate |
| SwiGLU 中间结果 | 由 epilogue 内部转换和保存 |
| y | DTYPE_Y，MXFP8 输出类型 |
| yScale | fp8_e8m0_t |

AIC 负责 Cube MMAD，AIV 负责 SwiGLU 和输出量化；两类核共享同一个 tiling data。

### 7.3 GMM tiling 参数传入

wrapper 从 host tiling data 中提取：

~~~text
groupNum
groupListType
baseM
baseN
baseK
isMxWeightNzMultiTensor
TCubeTiling* matmulTiling
~~~

device kernel 用 TCubeTiling 得到真实 N 和 K，并用 baseM/baseN/baseK 构造每个 group 的 block scheduler。

### 7.4 group 级别的地址偏移

KernelGmmSwiGluMixOnlineDynamic 的执行单位是一个 group。以非转置 NZ FP8、单 Tensor weight 为例：

设当前 group 的 token 行区间为 [groupStart, groupEnd)，本组 M 为：

~~~text
groupM = groupEnd - groupStart
~~~

则主要 base offset 为：

~~~text
x offset       = groupStart * K
xScale offset  = groupStart * ceil(K/64) * 2

weight offset  = groupIdx * (Align32(N) * Align16(K))
weightScale offset
               = groupIdx * N * ceil(K/64) * 2

y offset       = groupStart * (N/2)
yScale offset  = groupStart * ceil((N/2)/64) * 2
~~~

权重右半部分的偏移为：

~~~text
right weight offset = (N/2) * Align16(K)       // 非转置 NZ FP8
right scale offset  = (N/2) * 2                // 非转置 scale 的 N-major 访问
~~~

转置 NZ 使用另一套 weight/scale 步长：

~~~text
right weight offset = (N/2) * C0_SIZE_B8
right scale offset  = (N/2) * ceil(K/64) * 2
~~~

这些偏移正是 Cgmct kernel 与普通 ND kernel 最大的差异之一。

### 7.5 单个 group 的 tile 循环

ProcessSingleGroup 的核心流程如下：

~~~text
创建当前 group 的 Coord(m, n, k, baseM, baseN, baseK)

while scheduler 还能产生 tile:
    singleShape = scheduler.GetBlockShape(tileIdx)
    如果是本组第一个 tile:
        计算 group base offset
        更新 A/B/xScale/weightScale/y/yScale 的 GlobalTensor
        设置必要的 L2 cache hint

    blockOffset = coord.GetQuantIOOffset<MX_PERGROUP_MODE>(...)

    AIC:
        等待上一个 tile 的 Vector 处理结束
        MMAD left half:  A × weight[:, 0:N/2]
        MMAD right half: A × weight[:, N/2:N]
        将两个 L0C 结果通知 AIV

    AIV:
        等待 AIC 的两个 MMAD 完成
        读取 left/right 结果
        执行 SwiGLU
        执行 MXFP8 quant
        写 y 和 yScale
        通知 AIC 可以复用 buffer
~~~

V2 Cgmct 路径是**两次逻辑 matmul 调用**：一次 left，一次 right，结果分别放在 l0cOutUbFirst_ 和 l0cOutUbSecond_。这与 Blaze 路径的“拼接 B 后一次扩大 N 的 MMAD”不同。

### 7.6 AIC/AIV 跨核同步

Cgmct 路径使用两组方向相反的同步：

~~~text
AIC -> AIV:
    AIC 完成两个 MMAD
    NotifyVector()

AIV -> AIC:
    AIV 完成 SwiGLU + quant + GM 写回
    NotifyCube()
~~~

AIC 处理下一个 tile 前需要等待 AIV，避免覆盖仍在使用的 L0C/UB buffer；AIV 处理当前 tile 前需要等待 AIC，确保两个矩阵乘结果已经可见。

最后一个 tile 结束后，AIC 还要等待最后一轮 Vector 完成，避免 kernel 提前退出。

### 7.7 空 group

当 groupList 产生的 groupM==0 时，kernel 跳过这个 group，不执行 matmul，也不初始化对应输出区间。这样和 V2 文档中“没有被 groupList 覆盖的输出区域不保证被更新”的语义一致。

---

## 8. Blaze Tensor API MXFP8 路径

这一节解释用户要求查看的 C:\hw-workspace\ops-tensor_\include\blaze 模板，以及它在 V2 中是怎样被接起来的。

### 8.1 Blaze 路径的入口 wrapper

入口文件：

~~~text
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/
grouped_matmul_tensor_api_swiglu_quant_v2_mxfp8_kernel.h
~~~

wrapper 做四件事：

1. 读取 GMMSwigluQuantV2TensorApiTilingData；
2. 用 Blaze 类型组装 BlockMmad、BlockEpilogue、BlockScheduler；
3. 将 GM 地址、group list 和 tiling 参数组成 Blaze 的 Params；
4. 调用 Blaze::Gemm::Kernel::GemmUniversal。

核心模板关系如下：

~~~cpp
using BlockMmadPolicy =
    Blaze::Gemm::GroupedMatmulWithScaleMx<
        0, false, Blaze::Gemm::KernelGmmSwiGluMixMx>;

using SwigluBlockMmad =
    Blaze::Gemm::Block::BlockMmad<
        BlockMmadPolicy, AType, LayoutA, BType, LayoutB,
        C1Type, LayoutC1, BiasType, LayoutBias>;

using BlockEpilogue =
    Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<
        CType, C1Type, weightscaleType>;

using BlockScheduler =
    Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;

using SwigluKernel =
    Blaze::Gemm::Kernel::GemmUniversal<
        ProblemShape, SwigluBlockMmad,
        BlockEpilogue, BlockScheduler>;
~~~

KernelGmmSwiGluMixMx 是 Blaze dispatch policy 的标记类型，表示该 kernel 是 GMM + SwiGLU + MX quant 的 AIC/AIV 混合调度。

### 8.2 Blaze Kernel 的支持范围

blaze/gemm/kernel/kernel_qgmm_swiglu_mx.h 中有重要的编译期限制：

~~~text
AType/BType 必须是 FP8
CType/BiasType 必须是 float
LayoutA 必须是 NDExtLayoutPtn
LayoutB 只能是 NDExtLayoutPtn 或 DNExtLayoutPtn
LayoutC/LayoutBias 不能是 NZ/ZN
~~~

所以 Blaze 当前这条 V2 路径是 ND/DN Tensor API 路径，不是 WeightNZ 路径。V2 host 也只会在 weight format 为 ND/NCL 时把 Tensor API 模板放到优先注册列表中。

### 8.3 Blaze host tiling 的 capability 条件

GroupedMatmulSwigluQuantV2BasicApiTiling950::IsCapable() 会在“可回退”的语义下判断 Blaze 是否适用。主要条件包括：

- 能够获取 L1/L0/UB 平台内存信息；
- x、weight、y 为 FP8；
- xScale、weightScale、yScale 为 E8M0；
- quant dtype 与输出 dtype 匹配；
- x、weight、scale、y 的 format 属于 ND/NCL/NCHW 支持集合；
- AIV 数量为 AIC 的 2 倍；
- 只支持单 x、单 y、单 weight list 结构；
- xScale 和 weightScale shape 满足 Tensor API 的 [M, ceil(K/64), 2]、[E, ..., ..., 2] 约束；
- N 为 128 的整数倍；
- K 大于 64 且 K 为 128 的整数倍；
- 非转置 weight 时 M 大于 512；
- 每组平均 M 满足阈值：非转置大于 512，转置大于 128。

不满足这些条件时，host 返回 false，TilingRegistry 会尝试下一个注册的原始 950 模板。因此 Tensor API 的 capability 检查本身也是一种“选择高性能路径，否则安全回退”的机制。

### 8.4 Blaze tiling data

Blaze 路径的 DoLibApiTiling 大致流程是：

~~~text
GroupedQmmTiling::CalBasicBlock()
GroupedQmmBasicApiTiling::CalL1Tiling()

if M >= 256:
    baseM = 256

if N > 256 and K >= 128:
    baseN = 128

kAL1       = stepKa * baseK
kBL1       = stepKb * baseK
scaleKL1   = min(max(scaleFactorA * stepKa,
                     scaleFactorB * stepKb) * baseK, K)
scaleKAL1  = scaleKL1
scaleKBL1  = scaleKL1
~~~

最终 GMMTiling 不再传 TCubeTiling*，而是直接传一组 Blaze 需要的字段：

~~~text
m, n, k
baseM, baseN, baseK
kAL1, kBL1
scaleKAL1, scaleKBL1
dbL0C
groupType, groupListType
singleW
~~~

### 8.5 Blaze GemmUniversal 的 group 循环

Blaze kernel 的 Run 逻辑如下：

~~~text
Init(params)
创建 groupList Tensor
创建 BlockScheduler
设置 mTailAlign=1
设置 nTailAlign:
    transB ? BLOCK_CUBE : C0_SIZE_B8

for groupIdx in [0, groupNum):
    从 groupList 得到当前 group 的 M
    空 group 直接跳过
    如果在 AIC：更新 BlockMmad 的问题规模
    scheduler.UpdateNextProblem(M, N/2, K, 0)
    ProcessSingleGroup(scheduler, groupIdx)

等待最后一轮 AIV
~~~

这里有一个容易混淆的点：

- GMMTiling.n 保存的是 matmul 完整 N；
- scheduler 的 UpdateNextProblem 接收的是 N/2；
- 因为 scheduler 负责的是 SwiGLU 输出 tile；
- 但 B 矩阵仍然需要从完整 N 中取 left/right 两个半区。

### 8.6 Blaze scheduler：SWAT 和 tail split

BlockSchedulerGmmSwatWithTailSplit 负责两类工作：

1. 设计 M/N tile 的物理核分配顺序；
2. 当最后一个 M/N tail 不够填满所有空闲核时，把 tail tile 再拆分给多个核。

它的主要过程是：

~~~text
根据 M/baseM 得到 mBlockNums
根据 N/2/baseN 得到 nBlockNums
按 SWAT 顺序为物理 core 分配 tile
对最后一个 tail tile 计算 mTailCnt/nTailCnt
按 mTailAlign/nTailAlign 继续切分
~~~

对于每个 SwiGLU tile，scheduler 返回：

~~~text
blockM/blockN
aMOffset
bLeftNOffset  = nOffset
bRightNOffset = nOffset + N/2

outputOffset      = mOffset * (N/2) + nOffset
outputScaleOffset = mOffset * ceil((N/2)/64) * 2
                    + nOffset / 32
~~~

nOffset/32 是因为输出 scale 每 64 个输出元素对应 2 个字节，所以每个输出元素方向的 scale byte offset 是 2/64 = 1/32。

### 8.7 Blaze BlockMmad：MX scale 和双半区计算

blaze/gemm/block/block_mmad_qgmm_mx.h 的职责是完成：

~~~text
GM -> L1
L1 -> L0
MX scale -> L1/L0 对应 scale buffer
L0 MMAD
L0C -> UB
~~~

关键点：

- scale K span 使用 ceil(kSpan/64) * 2；
- L1 同时管理 A、B 和 A/B scale buffer；
- 根据 kAL1 和 kBL1 选择 A-major 或 B-major 的 L1 复用策略；
- K 方向循环中用 double buffer 交替搬运；
- 最后一个 K tile 使用 unit flag 控制累加初始化和最终写回；
- 对 SwiGLU 的 left/right 两个 B 子矩阵，Blaze 的 overload 会将两个 B 半区组织到 concat buffer 中，把逻辑上的两个半区计算合并为一个更宽的 N 计算，再交给 epilogue 使用。

因此 Blaze 与 Cgmct 的差异可以概括为：

~~~text
Cgmct WeightNZ:
    left MMAD + right MMAD
    两个 L0C/UB 结果

Blaze ND:
    B left/right 组织到 concat N
    一个扩展 N 的 MMAD
    epilogue 再按前半区/后半区解释
~~~

两者的逻辑结果相同，但对 layout、L1 搬运和尾块处理的要求不同。

### 8.8 Blaze epilogue：SwiGLU + MXFP8 quant

文件：

~~~text
blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h
~~~

#### UB buffer 组织

Epilogue 在 UB 中依次安排：

~~~text
输入 concat：       [left act | right gate]
量化输出 buffer：    FP8 output
scale 输出 buffer：  临时 scale layout
SwiGLU 结果：        BF16
max exponent：       每个 64 元素 block 一个
half scale：         每个量化 block 一个
scale block output： 用于转置后写 GM
~~~

GetConcatL0c2UbTensor(rows, cols) 返回一个列数为 2 * cols 的 UB tensor，正好容纳 left/right 两部分。

#### SwiGLU 计算

ComputeSwiglu 对每一行执行：

~~~text
从 concat UB 读取左半区 act
swish = act / (1 + exp(-act))
从 concat UB 读取右半区 gate
glu = swish * gate
float -> BF16
写入 gluRes UB
~~~

当前 V2 Blaze epilogue 是标准 SwiGLU，没有 clamp、alpha、bias 等 V3 参数。

#### MX scale 计算

ComputeMxQuant 分三步：

~~~text
1. ComputeMaxExp
   对每 64 个 BF16 结果提取 exponent 并求最大值

2. ComputeScale
   用目标 FP8 的 emax 计算 shared exponent
   生成 E8M0 scale 和量化所需 reciprocal/half scale

3. ComputeDataForQuantTargetFp8
   将 BF16 结果乘以 reciprocal scale
   转 float
   以 rint + saturation 转成目标 FP8
~~~

代码中：

~~~text
totalScaleInUb = totalDataInUb / 32
~~~

原因是每 64 个数据生成 2 个 scale byte。

#### scale layout 转换

向量计算期间 scale 更适合按连续 block 组织；写回 GM 前，TransMxScaleLayout 将 scale 从临时 UB layout 转成：

~~~text
[M, ceil(outputN / 64), 2]
~~~

同时输出 y 按 N/2 的行 pitch 写回。

---

## 9. 原始 Cgmct 与 Blaze 的对比

| 对比项 | 原始 Cgmct MXFP8 | Blaze Tensor API MXFP8 |
| --- | --- | --- |
| 主要场景 | WeightNZ，也可作为 ND fallback | 满足条件的 ND/NCL |
| host 模板 | GroupedMatmulSwigluQuantV2Tiling950 | GroupedMatmulSwigluQuantV2BasicApiTiling950 |
| tiling key | GMM_SWIGLU_QUANT_ORIGINAL_KERNEL_TYPE | GMM_SWIGLU_QUANT_TENSOR_LEVEL_KERNEL_TYPE |
| B layout | Nz / Zn / ND fallback | NDExtLayoutPtn / DNExtLayoutPtn |
| matmul 组织 | left MMAD + right MMAD | concat B 后的宽 N MMAD |
| group scheduler | GroupedMatmulAswtWithTailSplitScheduler | BlockSchedulerGmmSwatWithTailSplit |
| output epilogue | Cgmct BlockEpilogueSwigluQuant | Blaze BlockEpilogueSwigluMxQuant |
| MX scale | E8M0，K 方向 64 元素/block | E8M0，K 方向 64 元素/block |
| AIC/AIV | AIC matmul，AIV epilogue | AIC matmul，AIV epilogue |
| NZ 适配 | 已有 NZ 地址和搬运逻辑 | 当前 static assert 不支持 NZ/ZN |
| tail 处理 | Cgmct scheduler + layout 专用对齐 | SWAT + tail split |
| V3 WeightNZ 复用价值 | **最高** | 不能直接复用为 NZ kernel |

---

## 10. 对 V3 tiling/kernel 设计的直接启示

### 10.1 V3 只做单 Tensor WeightNZ MXFP8 时建议复用的部分

可以直接参考或复用：

1. GroupedMatmulSwigluQuantV2Tiling950 的：
   - M/N/K 推导；
   - transpose_weight 处理；
   - MXFP8 dtype 和 scale dtype 检查；
   - xScale、weightScale shape 检查；
   - N 为 64 对齐的 NZ 约束；
   - 基础 block、L1/L0 tiling；
   - TCubeTiling 和原始 tiling data 的组织方式。

2. GmmSwigluAswt<RowMajor, Nz/Zn> 的：
   - NZ weight 地址步长；
   - weight 左右半区 offset；
   - x/xScale/output/yScale 的 group offset；
   - Cgmct scheduler 和 tail split；
   - AIC 两次 matmul + AIV epilogue 的流水；
   - AIC/AIV 跨核同步。

### 10.2 V3 必须新增或修改的部分

V3 的 swiglu_mode、clamp_limit、glu_alpha、glu_bias 只改变 SwiGLU 部分，因此不应改变：

- matmul 的 M/N/K 逻辑；
- MXFP8 x/weight 反量化方式；
- WeightNZ 的物理布局；
- groupList 到 token 区间的映射；
- 输出 y/yScale 的 MX block 划分。

需要改变的是 AIV epilogue：

~~~text
读取 left/right matmul 结果
根据 swigluMode 选择拆分方式
在指定位置执行 clamp
根据模式使用 alpha/bias
得到 SwiGLU 结果
再执行原有 MXFP8 quant
~~~

由于当前 Cgmct 和 Blaze 的 V2 epilogue 都没有 V3 参数，V3 至少需要：

1. 在 V3 tiling data 中增加 swigluMode、clampLimit、gluAlpha、gluBias 等字段；
2. host 侧将 aclnn 入参写入 tiling data；
3. 为 Cgmct epilogue 增加 V3 分支，或者新增 V3 epilogue 类；
4. 在 ComputeSwiglu 之前或内部按 mode 做 clamp/alpha/bias；
5. 保证量化使用的是经过 SwiGLU 处理后的结果，而不是 matmul 原始结果。

### 10.3 为什么不能直接把 Blaze kernel 改成 V3 WeightNZ

当前 Blaze kernel_qgmm_swiglu_mx.h 明确要求：

~~~text
LayoutA = ND
LayoutB = ND 或 DN
LayoutC/LayoutBias 不能是 NZ/ZN
~~~

如果 V3 目标是 WeightNZ MXFP8，直接套用 Blaze 会遇到至少四类问题：

1. FrameLayoutFormat 不能按 NZ 物理地址解释 B；
2. block_mmad_qgmm_mx.h 的 GM->L1 copy layout 需要增加 NZ/ZN 特化；
3. weight scale 的 N/K 排列和 B 的 NZ 地址步长不一致；
4. tail 对齐必须跟 NZ 的 C0、cube block 规则匹配。

因此更稳妥的 V3 路线是：

~~~text
复用 V2 原始 Cgmct WeightNZ matmul/scheduler
只替换或扩展 Cgmct 的 SwiGLU epilogue
把 V3 参数添加到新的 V3 tiling data
~~~

如果后续确实需要 Blaze 统一框架，再单独设计 Blaze NZ layout 和 NZ BlockMmad，不建议把这件事和 V3 的 clamp 迭代混在一起。

---

## 11. 建议的 V3 最小实现分层

在不实现具体 kernel 细节的前提下，V3 可以按下面的层次设计：

~~~text
V3 host tiling
├── 只接受 950
├── 只接受单 Tensor WeightNZ
├── 只接受 MXFP8
├── 复用 V2 的 M/N/K、scale shape、NZ 对齐校验
├── 复用 V2 的 base block/L1/L0 计算
└── 新增 V3 SwiGLU 参数写入 tiling data

V3 device entry
├── 只保留 RowMajor + Nz/Zn 两个实例
├── 复用 Cgmct MXFP8 NZ matmul
├── 复用 groupList 和 tail scheduler
└── 切换到 V3 epilogue

V3 epilogue
├── left/right 结果取出
├── swigluMode 处理
├── clamp/alpha/bias 处理
├── 输出仍为 MXFP8
└── yScale 仍按输出 N/2 的 64 元素 block 生成
~~~

这样的分层可以让 V3 的变化严格限制在用户本轮关心的 SwiGLU 计算部分，不会无意中改变 V2 已经验证过的 WeightNZ 搬运和 matmul 行为。

---

## 12. 代码阅读索引

以下是本次分析涉及的关键代码位置：

### V2 dispatcher 和原始 tiling

~~~text
C:\hw-workspace\ops-transformer_\gmm\grouped_matmul_swiglu_quant_v2\op_host\op_tiling\grouped_matmul_swiglu_quant_v2_tiling.cpp
C:\hw-workspace\ops-transformer_\gmm\grouped_matmul_swiglu_quant_v2\op_host\op_tiling\arch35\grouped_matmul_swiglu_quant_v2_basic_tiling.cpp
~~~

重点函数：

~~~text
GroupedMatmulSwigluQuantV2TilingFunc
TilingPrepareForGMMSwigluQuantV2
ValidateAttrsCommon
AnalyzeDtype
CheckDtype
CheckDims
CheckMxPerGroupShape
AnalyzeInputs
DoLibApiTiling
PostTiling
GetTilingKey
~~~

### V2 原始 Cgmct kernel

~~~text
C:\hw-workspace\ops-transformer_\gmm\grouped_matmul_swiglu_quant_v2\op_kernel\grouped_matmul_swiglu_quant_v2_apt.cpp
C:\hw-workspace\ops-transformer_\gmm\grouped_matmul_swiglu_quant_v2\op_kernel\arch35\grouped_matmul_swiglu_quant_v2_mxquant.h
C:\hw-workspace\ops-transformer_\gmm\common\cgmct\kernel\kernel_gmm_swiglu_mxquant.h
~~~

重点函数：

~~~text
GmmSwigluAswt
KernelGmmSwiGluMixOnlineDynamic::GetSplitValueFromGroupList
KernelGmmSwiGluMixOnlineDynamic::UpdateOffset
KernelGmmSwiGluMixOnlineDynamic::InitParamsAndTensor
KernelGmmSwiGluMixOnlineDynamic::ProcessSingleGroup
~~~

### Blaze Tensor API kernel

~~~text
C:\hw-workspace\ops-transformer_\gmm\grouped_matmul_swiglu_quant_v2\op_kernel\arch35\grouped_matmul_tensor_api_swiglu_quant_v2_mxfp8_kernel.h
C:\hw-workspace\ops-tensor_\include\blaze\gemm\kernel\kernel_qgmm_swiglu_mx.h
C:\hw-workspace\ops-tensor_\include\blaze\gemm\block\block_mmad_qgmm_mx.h
C:\hw-workspace\ops-tensor_\include\blaze\gemm\block\block_scheduler_gmm_swat_with_tail_split.h
C:\hw-workspace\ops-tensor_\include\blaze\epilogue\block\block_epilogue_swiglu_mx_quant.h
~~~

重点函数：

~~~text
GmmTensorApiSwigluQuantMxFp8Kernel
GemmUniversal::Run
GemmUniversal::Init
GemmUniversal::ProcessSingleGroup
BlockSchedulerGmmSwatWithTailSplit::UpdateSwigluGroup
BlockSchedulerGmmSwatWithTailSplit::GetNextSwigluBlock
BlockEpilogueSwigluMxQuant::ComputeSwiglu
BlockEpilogueSwigluMxQuant::ComputeMxQuant
~~~

---

## 13. 一句话总结

V2 MXFP8 的核心模式是：host 根据 format、dtype、shape 和平台能力选择 tiling；tiling 计算 M/N/K、基础 tile、L1 深度和 MX scale 搬运参数；device 按 group 将连续 token 映射到对应 expert，AIC 完成 MXFP8 grouped matmul，AIV 将完整 N 的结果拆成 act/gate 做 SwiGLU，再按输出 N/2 的 64 元素 block 生成 E8M0 scale 并写回 MXFP8 输出。

对当前 V3 WeightNZ 目标，最直接的实现基础是 V2 的 **Cgmct NZ 路径**；Blaze 代码主要帮助理解 ND Tensor API 的模板组织方式、MX scale 处理和 SwiGLU epilogue，但不能直接作为 NZ kernel 使用。

