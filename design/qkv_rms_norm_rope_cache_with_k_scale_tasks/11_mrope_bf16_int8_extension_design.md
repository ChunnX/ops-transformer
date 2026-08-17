# M-RoPE、BF16 Q 与 INT8 K 增量设计

## 1. 文档范围

本文定义 `QkvRmsNormRopeCacheWithKScale` 的 M-RoPE 组合模式最终功能合同。

本次新增能力固定为 `MROPE_BF16_Q_INT8_K` 模式，正文简称“M-RoPE 组合模式”；它不拆成可任意
组合的独立开关：

```text
ROPE_FP8_QK mode:
  普通 RoPE + FP8 E4M3FN Q + FP8 E4M3FN K cache

MROPE_BF16_Q_INT8_K mode:
  3-axis interleaved M-RoPE + BF16 Q + INT8 K cache
```

以下行为保持或按本次目标合同调整：

1. `qkv`、gamma、`cos_sin`、`rotation` 的既有 dtype/shape 不变；`v_scale` 仍为必传 FP32，但 M-RoPE
   shape 从 `[Nv]` 改为 `[Nv,D]`。RoPE 继续使用 `[Nv]`。
2. V 分支仍在缩放后转 FP8 E4M3FN，`v_cache` 仍为 FP8 E4M3FN；M-RoPE 计算改为
   `V[t,n,d] * v_scale[n,d]`，只对 token 维广播。
3. OpDef 的 `q_out/q_scale/k_cache_out/v_cache_out/k_scale_cache_out` 均为 REQUIRED 输出；Torch 的
   `q_scale` 类型为 `Tensor`，RoPE 运行时返回 FP32 scale，M-RoPE 运行时返回 `None`；ACLNN 继续使用
   `qScaleOptional`，M-RoPE 传 `nullptr`。
4. RoPE 场景仍支持 `NTD->NTD`、`TND->TND`、`TND->NTD`；`MROPE_BF16_Q_INT8_K` 只支持
   `TND->TND`。
5. D128、`0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`、arch35、每 AIV token cap 为 4、V-first
   和现有同步协议保持不变；这些 head 约束与 RoPE 场景一致。

## 2. 接口增量

`v_scale` 的 input index、OPTIONAL ParamType、FP32 dtype 和“当前必须存在”约束不变，但 shape 是场景相关合同：

| 场景 | `v_scale` shape | V 缩放语义 |
|------|-----------------|------------|
| `ROPE_FP8_QK` | `[Nv]` | `V[t,n,d] * v_scale[n]` |
| `MROPE_BF16_Q_INT8_K` | `[Nv,D]` | `V[t,n,d] * v_scale[n,d]` |

host 必须先用 optional presence 解析候选场景，再校验 `v_scale` 精确 rank/shape。禁止 M-RoPE 接受
`[Nv]` 后隐式广播，也禁止 RoPE 接受 `[Nv,D]`。

### 2.1 既有位置输入改为接口级可选

`query_start_loc` 和 `seq_lens` 的输入 anchor 与 dtype 不变，但 OpDef `ParamType` 均从 `REQUIRED`
改为 `OPTIONAL`：

| 名称 | input index | 原型类型 | dtype | 非空时 shape | 说明 |
|------|------------:|----------|-------|---------------|------|
| `query_start_loc` | 8 | optional tensor | `int32` | `[Batch + 1]` | RoPE 场景的当前调用 local 前缀和 |
| `seq_lens` | 9 | optional tensor | `int32` | `[Batch]` | RoPE 场景的实际序列长度 |

Optional 是算子接口本身的声明，不是调用方先指定 scene 后再切换参数必选性。ACLNN checker 与 graph
direct base tiling 必须先读取各 optional 参数是否实际存在，再由完整 presence 组合识别候选场景；识别前
不得解引用空 tensor。`query_start_loc` 与 `seq_lens` 是一个完整位置源，只有同时存在才合法，单独传入
任一项不能形成第三种场景。

### 2.2 新增可选输入 `mrope_position`

| 名称 | 原型类型 | dtype | shape | 说明 |
|------|----------|-------|-------|------|
| `mrope_position` | optional tensor | `int32` | `[T,3]` | 每个 token 占一行，列 0/1/2 为 T/H/W 三路 `cos_sin` 行下标 |

公开接口名固定为 `mrope_position`；早期草案中的 `position` 不保留为别名。

`mrope_position[u,0]`、`mrope_position[u,1]`、`mrope_position[u,2]` 分别对应 token $u$ 的 T、H、W 三路位置。
即第 0 维按 token 递增，第 1 维固定为三路轴列。记
$P=\mathrm{mrope\_position}$，合法输入满足

$$
0 \le P_{u,a} < \mathrm{MaxSeqLen},
\qquad u\in\{0,\ldots,T-1\},\quad a\in\{0,1,2\}.
$$

`mrope_position` 的值位于 tensor 数据中，常规 host checker 不读取 device tensor 内容；数值范围属于调用方合法输入
合同，由 DiscussionForum golden 在运行前校验。新模式不允许假设三路下标相同、单调或连续。

为保持既有 input logical index 和相对顺序，`query_start_loc/seq_lens` 仍占用输入 8/9，`mrope_position` 追加为
输入 12，放在现有 `v_scale` 之后：

```text
0..7:  qkv ... k_scale_cache // 顺序不变
8:     query_start_loc        // 既有 anchor，改为 optional
9:     seq_lens               // 既有 anchor，改为 optional
10..11: rotation, v_scale     // 顺序不变
12:    mrope_position             // 新增 optional input
```

### 2.3 新增可选属性 `mrope_section`

| 名称 | OpDef 类型 | ACLNN 类型 | 默认值 | 语义 |
|------|------------|------------|--------|------|
| `mrope_section` | optional `ListInt` | `const aclIntArray *` | absent / `nullptr` | `[t,h,w]`，其中仅 `h/w` 控制 H/W 交织覆盖 |

公开属性名固定为 `mrope_section`；早期草案中的 `msection` 不保留为别名。

属性没有 tensor dtype 和 tensor shape。需求中的“`int32`、shape `[3]`”在工程原型中落为：

1. ListInt 长度必须为 3。
2. 三项必须非负；H/W 分别满足下述 lane 容量上限，第 0 项不做独立 lane 上限校验。
3. 三项之和必须满足 $s_{\mathrm T}+s_{\mathrm H}+s_{\mathrm W}\le D_{\mathrm{half}}$。
4. ACLNN/Runtime 的 `ListInt` 物理承载元素仍为 `int64_t`，不能把它误写成 `int32` tensor。

新模式固定 D128。令 $D_{\mathrm{half}}=D/2$，并将
`mrope_section=[t,h,w]` 记为 $\boldsymbol{s}=(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})$。cos 和 sin
分别在各自长度为 $D_{\mathrm{half}}$ 的 half 内应用该属性，H/W 容量上限定义为

$$
\begin{aligned}
C_{\mathrm H} &= \left\lfloor\frac{D_{\mathrm{half}}+1}{3}\right\rfloor, \\
C_{\mathrm W} &= \left\lfloor\frac{D_{\mathrm{half}}}{3}\right\rfloor,
\end{aligned}
\qquad
s_{\mathrm T}\ge 0,
\qquad 0\le s_{\mathrm H}\le C_{\mathrm H},
\qquad 0\le s_{\mathrm W}\le C_{\mathrm W},
\qquad s_{\mathrm T}+s_{\mathrm H}+s_{\mathrm W}\le D_{\mathrm{half}},
\qquad |\boldsymbol{s}|=3.
$$

D128 时 $D_{\mathrm{half}}=64$，H/W 上限为 $(C_{\mathrm H},C_{\mathrm W})=(21,21)$。
`mrope_section[0]` 不参与公式、模式判断或 T 路搬运范围，也不做独立 lane 上限校验；它仍须非负并
参与三项总和校验，三项之和不得超过 64。T/H/W 三行始终整行采集。
本设计不继承其他 RoPE 算子的固定取值白名单。

### 2.4 新增可选属性 `q_quant_mode`

| 名称 | OpDef 类型 | ACLNN/L0 类型 | 默认值 | 当前支持值 |
|------|------------|--------------|--------|------------|
| `q_quant_mode` | optional `String` | `const char *` | `PerTokenPerHead` | `PerTokenPerHead`、`NoQuant` |

`q_quant_mode` 显式决定 Q 分支是否量化：`PerTokenPerHead` 执行现有逐 token、逐 head 的 dynamic
quant 并产生 q_scale；`NoQuant` 不做 quant。匹配大小写，其他字符串
均返回不支持。ACLNN 的空指针或空串按默认值 `PerTokenPerHead` 处理。

### 2.5 新增可选属性 `q_out_dtype`

| 名称 | OpDef/L0 类型 | Torch public/schema 类型 | ACLNN 来源 | 默认值 | 当前支持值 |
|------|---------------|--------------------------|------------|--------|------------|
| `q_out_dtype` | optional `Int`/`int64_t` | `torch.dtype`/`ScalarType?` | `qOut->GetDataType()` | `ge::DT_FLOAT8_E4M3FN`/`torch.float8_e4m3fn` | FP8 E4M3FN、BF16 |

OpDef/L0 内部的枚举值通常为 `ge::DT_BF16=27`、`ge::DT_FLOAT8_E4M3FN=36`，实现和文档应使用符号
常量，不得把魔法数字作为主要判断口径。Torch 调用方只传 `torch.dtype`，ACLNN C ABI 不暴露独立
dtype 标量。FP8 E5M2、HIFLOAT8、FP16、FP32 和其他 dtype 不支持。

属性顺序采用 append-only：

```text
0: head_nums
1: layout_qkv
2: layout_q_out
3: epsilon
4: mrope_section
5: q_quant_mode
6: q_out_dtype
```

该顺序以本次接口合同为准：三个扩展属性相对前 4 个属性整体 append。早期设计稿中的
`dtype_q_out` 不作为公开名称，也不保留其草案序号兼容性；最终公开名称为属性 6 的
`q_out_dtype`。这里的公开名称指 OpDef 属性和 Torch keyword，不表示 ACLNN C ABI 增加同名标量参数。

### 2.6 ACLNN、L0 与 Torch 参数顺序

OpDef/L0 在已有 input 参数组尾部追加 `mrope_position`，在已有属性组尾部追加
`mrope_section/q_quant_mode/q_out_dtype`。ACLNN C 签名只追加 `mropePositionOptional`、
`mropeSectionOptional/qQuantMode`；既有参数保持相对顺序，且不增加 `qOutDtype` 参数：

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qGamma,
    const aclTensor *kGamma,
    const aclTensor *cosSin,
    const aclTensor *slotMapping,
    aclTensor *kCacheRef,
    aclTensor *vCacheRef,
    aclTensor *kScaleCacheRef,
    const aclTensor *queryStartLocOptional,
    const aclTensor *seqLensOptional,
    const aclTensor *rotationOptional,
    const aclTensor *vScaleOptional,
    const aclTensor *mropePositionOptional,
    const aclIntArray *headNums,
    const char *layoutQkv,
    const char *layoutQOut,
    float epsilon,
    const aclIntArray *mropeSectionOptional,
    const char *qQuantMode,
    aclTensor *qOut,
    aclTensor *qScaleOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

执行接口不承载算子参数，本次保持不变。ACLNN 读取 `qOut->GetDataType()` 并据此校验场景，再将
qOut dtype 映射为属性 6 的标准 GE enum。

Torch 的 inplace 与 functional schema 都保留既有参数相对顺序。前两个参数后面仍有必选
`head_nums`，因此使用无默认值的 `Tensor?`，M-RoPE 在固定参数位显式传 `None`；新增参数追加到
keyword-only 尾部：

```text
qkv_rms_norm_rope_cache_with_k_scale_(
    Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping,
    Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor(c!) k_scale_cache,
    Tensor? query_start_loc, Tensor? seq_lens, int[] head_nums, *,
    Tensor? rotation=None, Tensor? v_scale=None,
    str? layout_qkv="TND", str? layout_q_out="NTD", float epsilon=0.000001,
    Tensor? mrope_position=None, int[]? mrope_section=None,
    str q_quant_mode="PerTokenPerHead", ScalarType? q_out_dtype=None
) -> (Tensor, Tensor)

qkv_rms_norm_rope_cache_with_k_scale(
    Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping,
    Tensor k_cache, Tensor v_cache, Tensor k_scale_cache,
    Tensor? query_start_loc, Tensor? seq_lens, int[] head_nums, *,
    Tensor? rotation=None, Tensor? v_scale=None,
    str? layout_qkv="TND", str? layout_q_out="NTD", float epsilon=0.000001,
    Tensor? mrope_position=None, int[]? mrope_section=None,
    str q_quant_mode="PerTokenPerHead", ScalarType? q_out_dtype=None
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

`q_quant_mode` 的 schema 默认值为 `PerTokenPerHead`。Python public API 接收 `torch.dtype`，默认
`torch.float8_e4m3fn`；schema 使用 `ScalarType? q_out_dtype=None`，wrapper 将 `None` 归一化为该默认
dtype。Python 函数将两个既有位置形参标注为无默认值的 `Optional[torch.Tensor]`。C++ bridge 使用
`const c10::optional<at::Tensor> &`，在调用 `.value()`、创建 aclTensor 或发起 contiguous 之前先解析
presence；显式 `None` 转为 ACLNN `nullptr`，有值但 undefined 的 tensor 返回参数错误。

新增参数位都在 Torch keyword-only 尾部，因此既有 Torch 调用表达式保持不变。ACLNN 是固定 C
签名，既有调用方仍需补 `mropePositionOptional=nullptr`、`mropeSectionOptional=nullptr`、
`qQuantMode=PerTokenPerHead` 三个新增参数位，并传入 FP8 E4M3FN 的 `qOut`。

Torch schema 和 Python 返回注解中的 `q_scale` 均为 `Tensor`。RoPE 运行时返回 FP32 q_scale，M-RoPE
运行时返回 `None`；显式空 `mrope_section=[]` 不触发 M-RoPE 输出结构。Torch 不校验量化模式、dtype、
shape、presence 或 scene 的完整合同，这些门禁由 ACLNN checker 负责。

Torch private dtype 的内部偏移编码不属于本算子原型合同。wrapper 只用归一化后的 `torch.dtype` 创建
`qOut`；ACLNN 从该 tensor 读取 dtype，不接收 PyTorch 或 GE dtype 整数参数。

OpDef 的五个输出均为 REQUIRED，输出索引保持不变。ACLNN 保留 `aclTensor *qScaleOptional`：RoPE 传
FP32 tensor，M-RoPE 传 `nullptr`。Torch 返回 tuple 长度不变，`q_scale` 声明为 `Tensor`；RoPE 运行时
返回 FP32 scale，M-RoPE 运行时返回 `None`。

## 3. 两种模式与组合门禁

### 3.1 实参 presence 先识别候选场景

接口不提供独立 `scene` 属性。令 `Q/S/P` 分别表示
`query_start_loc/seq_lens/mrope_position` tensor 是否实际存在；`M` 表示 `mrope_section` 是否非空，即
`mrope_section != nullptr && size != 0`。host 先按下面的 effective signature 识别候选场景：

| `Q` | `S` | `P` | `M` | 判定 |
|----:|----:|----:|----:|------|
| 1 | 1 | 0 | 0 | 候选 `ROPE_FP8_QK` |
| 0 | 0 | 1 | 1 | 候选 `MROPE_BF16_Q_INT8_K` |
| 其他 |  |  |  | 拒绝 |

这里的 host 指 ACLNN checker 或 graph direct host。Torch wrapper 不实现上述 effective signature 校验；为构造
输出，只要 `P=1` 或 `M=1` 就按 M-RoPE 候选结构处理，随后由 ACLNN 按上表拒绝不完整或冲突的组合。

tensor 的“不存在”必须是 ACLNN `nullptr`、graph optional anchor 未提供或 Torch `None`。非空但零元素的
tensor、graph desc 存在但 shape 缺失和 Torch undefined tensor 都按 present 处理并在后续合同校验中失败。
`mrope_section` 是例外：框架可能把未提供的 optional `ListInt` 物化为空列表，显式 `[]` 也按 `M=0`
处理；非空但长度不为 3 的列表按 `M=1` 进入 M-RoPE 候选后由属性校验拒绝。

因此 `Q xor S`、`P xor M`、两组位置源同时存在和两组位置源同时缺失都非法。`mrope_section[0]` 的数值
不参与候选场景识别；presence 匹配后才校验其值域。

### 3.2 完整场景合同

候选场景还必须满足下面的完整相关合同，才能生成 tiling key：

| 条件 | `ROPE_FP8_QK` | `MROPE_BF16_Q_INT8_K` |
|------|----------------------|---------------------------|
| `query_start_loc` | 存在，`int32[Batch+1]` | 不存在，ACLNN 传 `nullptr` |
| `seq_lens` | 存在，`int32[Batch]` | 不存在，ACLNN 传 `nullptr` |
| `mrope_position` | 不存在 | 存在，`int32[T,3]`；每行按 T/H/W 列顺序存储 |
| `mrope_section` | 不存在或空列表 | 非空且长度为 3、三项非负，H/W 上限为 `[H_CAP,W_CAP]`，总和不超过 $D_{\mathrm{half}}$；第 0 项无独立 lane 上限且不参与计算 |
| `q_quant_mode` | `PerTokenPerHead` | `NoQuant` |
| OpDef/L0 `q_out_dtype` | `ge::DT_FLOAT8_E4M3FN` | `ge::DT_BF16` |
| `q_out` dtype（Torch/ACLNN dtype 来源） | FP8 E4M3FN | BF16 |
| `q_scale`（Torch / ACLNN） | FP32 tensor / FP32 tensor | `None` / `nullptr` |
| `k_cache` 输入/输出 dtype | FP8 E4M3FN | INT8 |
| `v_cache` 输入/输出 dtype | FP8 E4M3FN | FP8 E4M3FN |
| `v_scale` shape | `[Nv]` | `[Nv,D]` |
| RoPE 位置来源 | `query_start_loc + seq_lens` | `mrope_position` |
| `layout_qkv->layout_q_out` | NTD->NTD、TND->TND、TND->NTD | 仅 TND->TND |

presence 合法但量化模式、dtype、output、cache 或 layout 不匹配仍要拒绝。例如只传
`mrope_position/mrope_section` 但 `q_quant_mode=PerTokenPerHead` 或 `q_out` 仍为 FP8，或
`k_cache=INT8` 但仍传入 RoPE 位置源，都不能形成支持场景。M-RoPE 查表只
使用 `mrope_position`；`query_start_loc/seq_lens` 在该 signature 中必须为空，不能被静默忽略，也不能把其推导
的 offset 叠加到 `mrope_position` 上。

ACLNN 路径不接收独立的 `q_out_dtype` 标量，内部属性值由 `qOut` dtype 派生，因此不存在两份 dtype
互相不一致的输入。graph direct 路径仍保留属性 6，必须校验该属性与 q_out descriptor dtype 一致。

host checker/base tiling 根据 optional effective presence 匹配合同，并分别选择
`ROPE_MODE/K_QUANT_MODE/Q_QUANT_MODE` 三个枚举 tiling key 字段；host 只放行两组受支持的相关组合，
不在 kernel 暴露 runtime `isMrope/isQBf16/isKInt8` 开关。OpDef 的输入/输出 dtype 需要使用**等长两项相关
列表**表达两个组合，所有 dtype 不变的 tensor 也要重复两次；optional 的 `query_start_loc/seq_lens`
在两项中均声明 int32，第二项只是 tensor 缺席，不能因此缩短 dtype 列表。

### 3.3 OpDef 相关 type/format 向量

OpDef 的 `.DataType()`、`.Format()` 和 `.UnknownShapeFormat()` 必须对每个 tensor anchor 都提供等长
两项，不能只扩展发生 dtype 变化的 tensor。dtype 相关表为：

| input/output anchor | `ROPE_FP8_QK` | `MROPE_BF16_Q_INT8_K` |
|---------------------|----------------------|---------------------------|
| `qkv/rotation` | BF16 | BF16 |
| `q_gamma/k_gamma/cos_sin/v_scale` | FP32 | FP32 |
| `slot_mapping/query_start_loc/seq_lens/mrope_position` | INT32 | INT32 |
| `k_cache` input/output | FP8 E4M3FN | INT8 |
| `v_cache` input/output | FP8 E4M3FN | FP8 E4M3FN |
| `k_scale_cache` input/output | FP32 | FP32 |
| `q_out` | FP8 E4M3FN | BF16 |
| `q_scale` | FP32 | FP32 |

每个 anchor 的 format 和 unknown-shape format 都是 `[ND, ND]`。optional input 在对应场景缺席不代表
可以省略相关向量中的占位；五个 output anchor 均为 REQUIRED，并保持两项相关向量等长。

## 4. M-RoPE 合并公式

### 4.1 三路原始矩阵

令 $C=\mathrm{cos\_sin}$、$P=\mathrm{mrope\_position}$、$D_{\mathrm{half}}=D/2$，并定义轴到
`mrope_position` 列号的映射

$$
\iota(\mathrm T)=0,\qquad \iota(\mathrm H)=1,\qquad \iota(\mathrm W)=2.
$$

对 token $u\in\{0,\ldots,T-1\}$、轴 $a\in\{\mathrm T,\mathrm H,\mathrm W\}$ 和列
$d\in\{0,\ldots,D-1\}$，三路完整原始矩阵统一定义为

$$
R_{a,u,d}=C_{P_{u,\iota(a)},d}.
$$

因此 UB 中每个 token 的原始向量是三行沿列方向的串接：

$$
\operatorname{raw}_{u}
=R_{\mathrm T,u,:}\mathbin\Vert R_{\mathrm H,u,:}\mathbin\Vert R_{\mathrm W,u,:}
\in\mathbb{R}^{3D},
$$

其中 $\Vert$ 表示向量串接。该定义要求三路均读取完整的 $D$ 个元素。

### 4.2 cos/sin 分别交织

`mrope_section=[t,h,w]` 记为 $\boldsymbol{s}=(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})$，其中
$s_{\mathrm T}$ 完全不参与选源。cos 和 sin 均以 T 轴对应 half 为基底，再由
$s_{\mathrm H}$ 和 $s_{\mathrm W}$ 控制 H/W 覆盖。对局部 half lane
$\ell\in\{0,\ldots,D_{\mathrm{half}}-1\}$，定义

$$
r(\ell)=\left\lfloor\frac{\ell}{3}\right\rfloor,
\qquad \rho(\ell)=\ell\bmod 3,
$$

以及分段选源函数

$$
\sigma(\ell)=
\begin{cases}
\mathrm H, & \rho(\ell)=1\ \land\ r(\ell)<s_{\mathrm H},\\
\mathrm W, & \rho(\ell)=2\ \land\ r(\ell)<s_{\mathrm W},\\
\mathrm T, & \text{其他情况}.
\end{cases}
$$

在第 $r$ 个三元组内，被 H/W 覆盖的三个候选元素分别为

$$
R_{\mathrm T,u,3r},\qquad R_{\mathrm H,u,3r+1},\qquad R_{\mathrm W,u,3r+2}.
$$

则 H/W 覆盖范围内的结果正是需求中的 `[t0,h0,w0,t1,h1,w1,...]`。当 H 或 W 超出各自
`mrope_section` 覆盖范围后，相应 lane 回退到同一 lane 的 T 轴值。该公式不会把输出扩成 `3D`，也不是
按 mrope_section 做连续分段拼接。

例如 $D=16$、$D_{\mathrm{half}}=8$、$(s_{\mathrm H},s_{\mathrm W})=(1,2)$ 时，cos/sin 的
物理源列分别为

$$
\widehat{\boldsymbol c}_u
=(R_{\mathrm T,u,0},R_{\mathrm H,u,1},R_{\mathrm W,u,2},R_{\mathrm T,u,3},
  R_{\mathrm T,u,4},R_{\mathrm W,u,5},R_{\mathrm T,u,6},R_{\mathrm T,u,7}),
$$

$$
\widehat{\boldsymbol s}_u
=(R_{\mathrm T,u,8},R_{\mathrm H,u,9},R_{\mathrm W,u,10},R_{\mathrm T,u,11},
  R_{\mathrm T,u,12},R_{\mathrm W,u,13},R_{\mathrm T,u,14},R_{\mathrm T,u,15}).
$$

一般情况下，最终有效 cos/sin 定义为

$$
\widehat c_{u,\ell}=R_{\sigma(\ell),u,\ell},
\qquad
\widehat s_{u,\ell}=R_{\sigma(\ell),u,D_{\mathrm{half}}+\ell},
\qquad 0\le\ell<D_{\mathrm{half}}.
$$

之后 Q/K 的 half-split RoPE 公式完全不变：

$$
\begin{aligned}
z^{\mathrm{low}}_{u,\ell}
&=y^{\mathrm{low}}_{u,\ell}\widehat c_{u,\ell}
  -y^{\mathrm{high}}_{u,\ell}\widehat s_{u,\ell},\\
z^{\mathrm{high}}_{u,\ell}
&=y^{\mathrm{high}}_{u,\ell}\widehat c_{u,\ell}
  +y^{\mathrm{low}}_{u,\ell}\widehat s_{u,\ell}.
\end{aligned}
$$

### 4.3 Gather 索引

`mrope_section` 在一次调用内不变，因此每个 AIV 初始化时预建一个 $D_{\mathrm{half}}$ 项 `uint32`
元素索引向量。沿用 4.2 节的 $\sigma$ 与 $\iota$，第 $\ell$ 项为

$$
g_{\ell}=\iota\!\left(\sigma(\ell)\right)D+\ell,
\qquad 0\le\ell<D_{\mathrm{half}}.
$$

cos/sin 使用同一索引向量 $\boldsymbol g$；若以 `raw_token_base` 为零点，则两次 Gather 的源元素偏移集合分别为

$$
\mathcal I_{\cos}=\{g_{\ell}\mid 0\le\ell<D_{\mathrm{half}}\},
\qquad
\mathcal I_{\sin}=\{D_{\mathrm{half}}+g_{\ell}\mid 0\le\ell<D_{\mathrm{half}}\}.
$$

RegBase `MicroAPI::Gather` 的 index 单位是源类型元素，不是 byte。FP32 源对应
`RegTensor<uint32_t>` 索引；`gather_idx` 不乘 `sizeof(float)`。MemBase `Gather(LocalTensor,...)` 的 byte
offset 语义不适用于本 VF。

## 5. Q/K 后处理公式

令 $C^Q$、$C^K$ 分别为 RoPE 后右乘共享 `rotation` 得到的 FP32 Cube 输出。

### 5.1 统一动态量化

令目标量化类型为 $\tau_A\in\{\mathrm{FP8\ E4M3FN},\mathrm{INT8}\}$，其正向量化上限统一记为

$$
M_{\tau_A}=
\begin{cases}
448, & \tau_A=\mathrm{FP8\ E4M3FN},\\
127, & \tau_A=\mathrm{INT8}.
\end{cases}
$$

对 $A\in\{Q,K\}$、token $u$ 和 head $n$，两种目标类型使用同一套逐 token、逐 head 公式：

$$
\begin{aligned}
m^A_{u,n}&=\max_{0\le d<D}\left|C^A_{u,n,d}\right|,\\
\alpha^A_{u,n}&=\frac{m^A_{u,n}}{M_{\tau_A}},\\
\widehat C^A_{u,n,d}
&=\operatorname{cast}_{\tau_A}
  \!\left(\frac{C^A_{u,n,d}}{\alpha^A_{u,n}}\right).
\end{aligned}
$$

`cast` 按目标类型执行舍入和饱和。RoPE 的 Q/K 取 $\tau_Q=\tau_K=\mathrm{FP8\ E4M3FN}$；M-RoPE
的 K 取 $\tau_K=\mathrm{INT8}$。量化公式本身不描述位置编码差异，两组受支持组合仍由第 3 节约束。

### 5.2 M-RoPE 组合模式 `NoQuant` BF16 Q

Q 不执行 abs-max、reduce、scale、div 和 FP8 cast，其输出为

$$
Q^{\mathrm{out}}_{u,n,d}=\operatorname{RNE}_{\mathrm{BF16}}
\!\left(C^{Q}_{u,n,d}\right).
$$

M-RoPE 模式下 Torch 返回 `q_scale=None`，ACLNN 的 `qScaleOptional` 传 `nullptr`；ACLNN 调用方传入
非空 `qScaleOptional` 属于非法组合。

### 5.3 M-RoPE 组合模式 INT8 K

K 使用第 5.1 节统一公式，并取 $\tau_K=\mathrm{INT8}$、$M_{\tau_K}=127$。记量化结果为
$K^{\mathrm{int8}}=\widehat C^K$；INT8 转换采用 RINT，并饱和到 $[-127,127]$。

若 $b_u$ 和 $o_u$ 分别表示 `slot_mapping[u]` 对应的 cache block 与 block 内偏移，则写回语义为

$$
\begin{aligned}
K^{\mathrm{cache}}_{b_u,n,o_u,d}&=K^{\mathrm{int8}}_{u,n,d},\\
K^{\mathrm{scale}}_{b_u,n,o_u,0}&=\alpha^K_{u,n}.
\end{aligned}
$$

INT8 对称目标范围为 `[-127,127]`。

### 5.4 M-RoPE V 逐通道 scale

令 $S^V=\mathrm{v\_scale}\in\mathbb{R}^{N_v\times D}$。M-RoPE V 输出为

$$
V^{\mathrm{out}}_{u,n,d}
=\operatorname{cast}_{\mathrm{FP8\ E4M3FN}}^{\mathrm{RINT,SAT}}
\!\left(\operatorname{float32}(V_{u,n,d})\,S^V_{n,d}\right).
$$

$S^V_{n,d}$ 仅沿 token 维广播；不同 head、不同通道可以取不同值，0 合法。cache scatter 地址不变。

## 6. Host、InferShape 与 InferDtype

### 6.1 Checker/base tiling

ACLNN checker 和 graph direct base tiling 必须按同一顺序执行完整门禁：

1. 只读取 `query_start_loc/seq_lens/mrope_position` 的 tensor presence 和 `mrope_section` 的列表长度，不解引用
   optional tensor 数据；严格按 3.1 节的 effective signature 识别候选场景。
2. 拒绝任一位置源 pair 只出现一项、两组位置源同时存在或同时缺失。
3. 对 RoPE 与 M-RoPE 统一校验 `head_nums` 三项为正整数、`Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`。
4. RoPE 才校验 `query_start_loc: int32[Batch+1]`、`seq_lens: int32[Batch]` 及二者的既有值语义；
   M-RoPE 不读取这两个空 anchor。
5. M-RoPE 才校验 `mrope_position` 的 dtype、rank 和 `[T,3]` shape，其中第 0 维等于 qkv token 数；RoPE 不读取空 `mrope_position` anchor。
6. `mrope_section` 长度为 3 且三项非负，H/W 上限为第 2.3 节定义的
   $(C_{\mathrm H},C_{\mathrm W})$，三项之和不超过 $D_{\mathrm{half}}$；第 0 项不做独立 lane 上限校验，
   通过非负和总和校验后不参与模式判断或计算。
7. `q_quant_mode` 只允许 `PerTokenPerHead`/`NoQuant`，并必须与候选场景一致：RoPE 只能是
   `PerTokenPerHead`，M-RoPE 只能是 `NoQuant`。
8. ACLNN 从 `qOut` 读取 dtype，只允许 FP8 E4M3FN/BF16，并校验它与 `q_quant_mode` 一致；graph direct
   路径还要校验属性 6 的 `q_out_dtype` 与 q_out descriptor dtype 一致。
9. `k_cache` 输入/输出 alias tensor dtype 与候选场景一致。
10. `v_cache` 始终为 FP8 E4M3FN，`k_scale_cache` 始终为 FP32；ACLNN 的 `qScaleOptional` 在 RoPE
    场景必须为 FP32 tensor，在 M-RoPE 场景必须为 `nullptr`。
11. 按候选场景校验 `v_scale` 精确 shape：RoPE `[Nv]`，M-RoPE `[Nv,D]`；M-RoPE 第二维必须等于
    `headDim=128`。
12. presence 合法但量化模式、dtype/output/cache/layout 任一项不匹配时拒绝，不生成 tiling key。

graph tiling 的 input tensor contract 需要单独保存 `present` 与 `shapePresent`，不能以 shape 是否可读代替
optional presence。`GetOptionalInputDesc(index) != nullptr` 只确定 input present，shape 缺失仍在后续
校验中失败。`mrope_section == nullptr` 与长度为 0 在场景判定中都映射为 `M=0`；只有非空列表才映射为
`M=1` 并进入 M-RoPE 属性校验。ACLNN checker 按候选场景校验 `qScaleOptional`。

scene resolver 必须先于任何会把这四个 optional anchor 当作 required 的 pointer、empty、dtype、format
或 shape 校验，以及 graph tiling 的 `FillShapeDerivedFields()` 和 `ValidateScalarInputs()` 执行。ACLNN 的
`ContiguousInputs()` 只处理当前候选场景实际存在的 optional tensor。

### 6.2 InferShape 最小集

InferShape 不读取 `query_start_loc`、`seq_lens`、`mrope_position`、`mrope_section`、`q_quant_mode`、
`q_out_dtype`，也不根据 optional presence 判定场景，不复制 checker/base tiling 的完整校验。五个输出均
为 REQUIRED，InferShape 按固定逻辑索引取得并设置输出 shape。InferShape 继续保留：

1. `NTD->TND` 拒绝。
2. 逻辑 `qkvN >= Nq`。
3. `q_out` 和 `q_scale` shape 均按 `layout_q_out` 推导。
4. 三个 cache 输出 shape 继承对应 cache 输入。

### 6.3 InferDtype

InferDtype 读取属性 6 的 `q_out_dtype`：缺失时按 FP8 E4M3FN，BF16 时将 `q_out` 设为 BF16；其他值
返回失败。output index 1 的 `q_scale` 始终设置为 FP32。三个 cache 输出 dtype 继续继承输入，因此
`k_cache` 的 INT8 输出不需要在 InferShape 中增加 dtype 分支。

## 7. Tiling 与模板分发

### 7.1 新增 `ROPE_MODE/K_QUANT_MODE/Q_QUANT_MODE` 模板

tiling key 在现有 D128、`QKV_LAYOUT`、`Q_OUT_LAYOUT` 后追加三个独立枚举字段：

```cpp
ASCENDC_TPL_UINT_DECL(ROPE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
    QKV_K_SCALE_ROPE_MODE_ROPE, QKV_K_SCALE_ROPE_MODE_MROPE)
ASCENDC_TPL_UINT_DECL(K_QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
    QKV_K_SCALE_K_QUANT_MODE_FP8, QKV_K_SCALE_K_QUANT_MODE_INT8)
ASCENDC_TPL_UINT_DECL(Q_QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
    QKV_K_SCALE_Q_QUANT_MODE_PER_TOKEN_PER_HEAD, QKV_K_SCALE_Q_QUANT_MODE_NO_QUANT)
```

host checker/base tiling 先由 optional effective signature 识别候选场景，完成相关 dtype、output、cache 与
layout 合同校验后，分别设置三个 mode。当前只放行以下两组相关组合：

```text
ROPE_FP8_QK             = (ROPE,  FP8,  PER_TOKEN_PER_HEAD)
MROPE_BF16_Q_INT8_K     = (MROPE, INT8, NO_QUANT)
```

三个 mode 均独立于 layout，且都不是 tiling data 字段。kernel entry、Controller、BasicBlock 和 Vec/VF
按如下方向传播：

```cpp
template <uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT, uint32_t ROPE_MODE,
          uint32_t K_QUANT_MODE, uint32_t Q_QUANT_MODE>
class QkvRmsNormRopeCacheWithKScaleController;

template <uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT, uint32_t ROPE_MODE,
          uint32_t K_QUANT_MODE, uint32_t Q_QUANT_MODE>
class QkvRmsNormRopeCacheWithKScaleBasicBlock;

template <uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT, uint32_t ROPE_MODE,
          uint32_t K_QUANT_MODE, uint32_t Q_QUANT_MODE>
class QkvRmsNormRopeCacheWithKScaleVec;
```

kernel 分别用三个编译期常量隔离：

1. `ROPE_MODE`：普通位置推导或 `mrope_position` 三路查表，以及连续 cos/sin load 或 RegBase Gather。
2. `Q_QUANT_MODE`：Q `PerTokenPerHead` FP8 dynamic quant 或 `NoQuant` BF16 输出；
   当前 M-RoPE `NoQuant` 实例由 Cube Fixpipe F32->BF16 直写 GM，不进入 Vec QCast。
3. `K_QUANT_MODE`：K FP8 dynamic quant 或 INT8 dynamic quant。

host 必须读取 optional effective presence 来选择 tiling key；该行为不属于 kernel runtime 分支。进入 kernel 后，
场景差异只能使用 `if constexpr` 或两个场景专用的模板入口；禁止根据
`query_start_loc/seq_lens/mrope_position` 空地址、`q_quant_mode`、`q_out_dtype`、cache dtype、`mrope_section`
值或 tiling data flag
再次判断场景。Cube 数学、A_READY/A_CONSUMED 和核内 event 可以复用；RoPE/M-RoPE 共享固定语义
`FIX_OUTPUT_READY` 正向边。RoPE 的反向复用使用 `FIX_OUTPUT_CONSUMED`，M-RoPE direct-Q 使用
`MROPE_K_CONSUMED`，不能把两组 consumed 计数混写。

三个 mode 不与 layout 做完整笛卡尔积；M-RoPE 只实例化一个 compact direct-Q 模板。当前共生成
4 个注册选择：

```text
ROPE_FP8_QK:
  NTD->NTD
  TND->TND
  TND->NTD

MROPE_BF16_Q_INT8_K:
  TND->TND (compact direct-Q)
```

M-RoPE 场景的 `NTD->NTD`、`TND->NTD` 和所有 `NTD->TND` 组合不生成 key，由 checker/base
tiling 拒绝。

kernel GM 形参保留 input index 8/9，并在 input 参数组尾部增加 index 12；任一场景都不删除或压缩
logical anchor。新增 input 位于 outputs 之前，因此 flat kernel signature 的 `q_out` 及后续形参序号
顺延 1，入口与 GlobalTensors 初始化必须同步更新。RoPE launch 的 `mrope_position` 地址为空；M-RoPE launch
的 `query_start_loc/seq_lens` 地址为空。GlobalTensors 对 optional input 的绑定和读取必须放在相应
`ROPE_MODE` 的编译期分支内，不允许先无条件绑定 optional 地址再在热路径判空。

### 7.2 Tiling data 增量

现有字段尾部只追加实际参与 kernel 计算的两项：

```cpp
uint64_t mropeSectionH;
uint64_t mropeSectionW;
```

`mrope_section[0]` 不写入 tiling data，从结构上避免误用；host 不校验它的独立 lane 上限，只校验
其非负及三项总和不超过 $D_{\mathrm{half}}$，随后将其丢弃。
三个 mode 已进入 tiling key，不重复写入 tiling data。连续化后的 `mrope_position[T,3]` 使用
token-major 行布局：末维列 stride 为 1，token stride 为 3，GM 元素偏移为 $3u+\iota(a)$，
不增加独立 stride 字段。既有 `batch` 字段只对 RoPE 有效，由
`query_start_loc.shape[0]-1` 推导；M-RoPE 写 0 sentinel，M-RoPE 编译实例不得读取 `batch` 或两个
RoPE 位置输入。

初始化链路统一直接透传只读 tiling data 指针：`BasicBlock::InitAic/InitAiv` 分别调用
`Cube::Init(tensors, tilingData)` 与 `Vec::Init(tensors, tilingData)`。Cube/Vec 在 Init 内复制本层需要的
标量，不保存指针。Vec 通过编译期 `IS_MROPE` 条件令 `batch_ = 0` 或复制 `tilingData->batch`，并只在
M-RoPE 实例复制 `mropeSectionH/W`；后续热循环不再读取 tiling data，也不根据 sentinel 重新选场景。

### 7.3 cap=4 与 UB 预算

普通 RoPE 共享布局的 raw T/H/W 静态 buffer 绑定 host 的 `TOKEN_TILE_PER_AIV_CAP=4`；M-RoPE 实际只使用 compact，`mrope_position` slot 则按 kernel 通用
`MAX_TOKEN_TILE_PER_AIV=8` 分配，当前 tile 只使用其中最多 48 B。每 AIV 的新资源为：

```text
raw T/H/W cos_sin one buffer = 3 * 4 * 128 * 4 B = 6144 B = 6 KiB
raw cos_sin DB                                         = 12 KiB
mrope_position one buffer = 128 B (generic eight-token slot; current cap=4 uses 48 B)
mrope_position DB                                                  = 256 B
gather_idx = 64 * sizeof(uint32_t)                     = 256 B
```

这里的 128 B 是 UB 单 slot 容量，不是 GM 搬运长度。当前 cap=4 时，`tileT=1/2/3/4` 的有效数据分别为 12/24/36/48 B；
必须使用 `DataCopyPad` 或等价的长度安全尾搬运只读取 `12 * tileT` 字节，再在 UB 侧补齐。禁止普通
DataCopy 按 128 B 从 GM 扩读，否则最后一个 tile 会越过 `mrope_position` 的 tensor 边界。

普通 RoPE 其他固定 buffers 的静态分配容量合计为：

```text
gamma 1 KiB + cos_sin DB 8 KiB + slot DB 1 KiB
+ v_scale 0.5 KiB + QK scatter index 0.5 KiB + V out DB 20 KiB
= 31 KiB
```

替换为 M-RoPE raw buffer 并加入 mrope_position/index 后：

```text
31 KiB + (12 - 8) KiB + 256 B + 256 B = 35.5 KiB
```

普通 RoPE 共享布局的这些 buffers 静态上限为 40 KiB，其中 35.5 KiB 已分配、4.5 KiB 未使用；总 UB 为
`80 + 128 + 40 = 248 KiB`。该布局与 RoPE `[Nv]` scale 保持不变。
host tiling 仍必须保证 `ceil(tokenTile/2) <= 4`。

目标 M-RoPE 只使用 compact 布局：80 KiB input pool 划分为四个 20 KiB slot，另分配 80 KiB
`outputDbPoolUb_`（两个 Q/K 前处理 slot）、14 KiB K rotation DB、40 KiB compact reserve，并增加
12 KiB 的 1024-token position window，总 UB 为 `80+80+14+40+12=226 KiB`。compact reserve 中为
`[Nv,D]` scale 分配 4 KiB，按当前 `Nv<=8,D=128` 覆盖最大 `8*128*sizeof(float)`。host 在当前 tile 不满足 `kRows<=14`、
Q/K preprocess 不超过 40 KiB、输入行数不超过 80 时降低 token tile；最小 tile 仍不满足则拒绝，不切换
第二条 M-RoPE 路径。

q_out Fixpipe target 使用默认 GM cache 行为。平台 compile info 不读取 L2 大小，host 不按
Q/K/V/K-scale 输出总字节生成第二个模板。

## 8. Vec 流水与同步

M-RoPE 当前公共前处理流水：

```text
stage qkv + slot_mapping + token-major mrope_position[T,3]
  -> 根据 mrope_position 发起 T/H/W 三行 raw cos_sin 搬运
  -> QuantVToFp8
  -> 等待 raw cos_sin 可读
  -> VF Gather + Q/K RMSNorm/RoPE
  -> ScatterVCache
  -> CopyRopeBf16NzUbToL1Nz
  -> A_READY
```

V-first 的定义不变：V vector quant 先于 Q/K preprocess 发起；它不表示 `v_cache` 写回先于 Q/K
preprocess。M-RoPE 由 Fixpipe 将 Q FP32 转为 BF16 后直接写 `q_out` GM；`FIX_OUTPUT_READY`（flag 1）
发布 K rotation，并由 `MROPE_K_CONSUMED`（flag 4）管理反向复用，避免让 Q 占用 K 的 UB credit。
旧的独立 M-RoPE ready flag 和不可达的 Vec `QCastBf16D128TndVfImpl` 路径已删除。

compact 的 K rotation 进入独立 14 KiB DB，AIV 将其量化到独立 K quant
staging。控制迭代处理上一拍 `lastTile` 时，调用顺序为
`StageMropeK(lastTile) -> Set MROPE_K_CONSUMED -> ScatterMropeK(lastTile)`；Stage 与 Scatter 在同一控制
迭代，保留的是 current/last AIC/AIV 一拍 rolling。K quant 仍为双缓冲，其 slot 由 `MTE3 -> V`
reusable event 保护，K cache/K scale 两路写回完成后才返还。compact 的 input 与 raw cos/sin 分别由独立
use counter 管理，槽数由各自 pool/slot 容量在 Vec 内局部推导为 4；metadata 不再借用 input buffer id。
empty slice 不占 K quant slot，因此不推进 `kQuantBufferUseId_`，但仍参与 ready/consumed CrossCore 握手。

## 9. Golden 与验证矩阵

数值与 kernel 验证继续只使用 DiscussionForum NPU golden/TTK，不新增 `ops-transformer/tests/ut/op_kernel`
或 ST target。Host/Tiling/API UT 仍保留在 `ops-transformer`。

最低覆盖集：

| Case | 关键覆盖 |
|------|----------|
| head-contract | RoPE/M-RoPE 都覆盖 `(24,3,3)` 非 2 的幂合法 tuple 和 `(64,8,8)` 上边界；非法关系只进入 checker/base tiling 拒绝路径 |
| rope-regression | `query_start_loc/seq_lens` 非空、`mrope_position/mrope_section` 缺失、`PerTokenPerHead + FP8` Q/K，证明 RoPE 路径不变 |
| mrope-smoke | `query_start_loc/seq_lens=None`、`T=1`、唯一 token 行的三列 mrope_position 相同，验证新 signature 基本链路 |
| mrope-default-random | 所有 M-RoPE case 默认按各自 seed 在 `[0,MaxSeqLen)` 独立均匀随机生成 `mrope_position[T,3]`；包含 `T=5` 跨 cap=4，检测错误的列索引或 token/axis 维颠倒，不能只用产生 `[3,3]` 方阵的 `T=3` |
| mrope-t-base | 保持 `mrope_section[1:3]=(20,20)`，将第 0 项遍历 `{0,1,11,24}`；24 超过旧上限 22 但总和为 64，输出必须完全一致 |
| mrope-asymmetric | `mrope_section[1] != mrope_section[2]`，分别验证 H/W 覆盖边界，避免两项误共用 |
| mrope-upper-bound | D128 下分别验证 H/W=`[21,21]` 的最后一个 lane |
| mrope-sum-bound | D128 下覆盖 `[22,21,21]` 总和 64 的合法边界和 `[23,21,21]` 总和 65 的拒绝路径 |
| random-mrope_position | 长序列默认随机 position 自然覆盖非单调和重复值，证明只依赖 `[u,axis]` 值而不依赖位置连续性；随机 generator 与其他输入解耦，固定 case seed 时可复现 |
| layout | M-RoPE 只覆盖 TND->TND；其他 layout 作为拒绝用例，RoPE 保留三条回归 |
| bf16-q | random dense rotation，`NoQuant + BF16`，Torch 返回 `q_scale=None`，ACLNN 使用 `qScaleOptional=nullptr` |
| int8-k | K scale `/127`、RINT/SAT、cache scatter 和跨 block；逐元素有符号差值绝对值不超过 1，且 100% 元素通过 |
| mrope-v-scale | `[Nv,D]` 每个 head/channel 使用不同 scale；验证逐通道乘法、FP8 cast 和 V cache scatter，不允许整行广播；零值边界由独立用例覆盖 |
| mrope-compact | 明确命中 compact key，覆盖 input/raw 各四槽、Fixpipe direct-Q、独立 K rotation/K quant、flag 1/4、同迭代 Stage/Scatter 和 K quant reusable event |
| mrope-large-output | 单一 compact 模板覆盖 `T=45000/65536/131072`，验证长序列、多 tile 与四路输出；性能矩阵中 `T=65536/131072` 与同 shape 的 RoPE TND->TND case 配对 |
| metadata-window | 单核 token span 跨过 1024 边界，覆盖 metadata refill、非单调 position 和跨窗口 cache slot |
| tail/multi-tile | `T in {1,4,5,17}`，tail、pingpong、drain、无死锁 |

接口拒绝用例至少覆盖：

1. effective presence mask 全矩阵：`M` 表示 `mrope_section` 非空；`query_start_loc xor seq_lens`、
   `mrope_position xor M`、两组位置源同时存在、两组位置源同时缺失均拒绝，只接受 `1100` 和 `0011`。
   另覆盖显式 `mrope_section=[]` 映射为 `M=0`，配合合法 RoPE 位置输入正常执行。
2. RoPE 位置输入非空时的 dtype/rank/shape 错误，以及 `mrope_position` 非空时的 dtype/rank/shape 错误；后者明确覆盖 token/axis 维颠倒、末维不为 3 和 token 维不匹配。
3. `mrope_section` 长度错误、任一项小于 0、H/W 分别超过 `H_CAP/W_CAP`，或三项之和超过
   $D_{\mathrm{half}}$；第 0 项不做独立 lane 上限校验，满足非负和总和约束时其值不得改变结果。
4. `q_quant_mode` 非 `PerTokenPerHead`/`NoQuant`，以及 `PerTokenPerHead + BF16`、`NoQuant + FP8`。
5. 通过 Torch 传入不属于支持合同的 `torch.dtype`，确认 wrapper 构造 qOut 后由 ACLNN 根据其 dtype 拒绝；
   同时覆盖直接 ACLNN 调用的 `qOut` dtype 非 FP8 E4M3FN/BF16，以及 graph direct 的 `q_out_dtype` 属性
   不在这两个 GE enum 中。
6. graph direct 的 `q_out` descriptor dtype 与 `q_out_dtype` 属性不一致，或任一路径的
   cache/output/layout 与候选场景不匹配；ACLNN 的 `PerTokenPerHead + FP8` 缺少 q_scale 或
   `NoQuant + BF16` 传入非空 q_scale。ACLNN 不存在独立 dtype 标量与 `qOut` 不一致的用例。
7. `MROPE_BF16_Q_INT8_K` 使用非 TND->TND layout。
8. RoPE 与 M-RoPE 分别覆盖 `Nq > 64`（如 `(72,9,9)`）、`Nq != 8 * Nk`（如 `(16,4,4)`）和
   `Nk != Nv`（如 `(16,2,1)`），确认三类非法 head tuple 均在 checker/base tiling 阶段拒绝。
9. M-RoPE `v_scale` 为 `[Nv]`、`[Nv,D-1]`、`[Nv+1,D]`，以及 RoPE `v_scale` 为 `[Nv,D]`；均应在
   checker/base tiling 阶段拒绝，不进入 kernel。

## 10. 设计边界

OpDef、ACLNN/L0、checker、InferShape、InferDtype、tiling、kernel、Torch schema、API 文档和
DiscussionForum 可执行资产均应遵循本文合同。`09/10` 中的历史性能记录仅描述对应版本的性能数据，
不改变本文定义的功能语义。

截至 2026-08-03，DiscussionForum CPU Golden 与 `ops-transformer 6ca8b3400` 均已按本节合同更新。
Host/Tiling UT `120/120`、ACLNN UT `128/128`、NPU 全量 `185/185` 和 section-0 invariant `3/3`
均通过；M-RoPE `v_cache` 对逐通道 Golden 为逐元素一致。
