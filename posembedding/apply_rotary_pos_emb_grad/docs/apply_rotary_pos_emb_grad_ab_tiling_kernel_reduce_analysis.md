# ApplyRotaryPosEmbGrad AB 模板 tiling/kernel/reduce 梳理

本文只梳理当前目录中的 AB 模板实现，重点分析四输出场景下 `grad_cos`、`grad_sin`
精度 fail 的可能原因。代码路径以 Ascend950/regbase 为前提。

## 1. 结论先行

AB 模板的两输出路径只计算 `grad_query`、`grad_key`，不进入 `grad_cos/grad_sin`
部分积与 reduce 阶段，因此当前“两输出 pass、四输出后两个输出 fail”的现象，问题范围可以基本收敛到：

1. phase1 写入 `grad_cos/grad_sin` 部分积的 workspace 布局；
2. phase2 `ReduceSch` 使用的输入 shape/axes 与 workspace 逻辑布局不一致；
3. 修改前部分积以 `T` 类型写入 workspace，后续 reduce 再 cast 到 fp32，导致 fp16/bf16
   场景下先截断再规约。

当前代码中最明确的结构性风险是 reduce tiling 固定用 `grad_query_embed` 的 shape 推导输入形状：
`TilingReduce()` 调 `GetInputParam(opInput, GQ_INDEX, COS_INDEX)`，即 reduce 输入 N 维取 `nQ`。
但 AB phase1 的部分积 workspace 按 `max(nQ, nK)` 写入。如果 `nK > nQ`，K 路多出来的 head
部分积会被写进 workspace，却不会被 reduce 模板覆盖，最终 `grad_cos/grad_sin` 少加 K 路尾部 head。

## 2. AB 模板适用范围

host 侧 AB tiling 在 `op_host/apply_rotary_pos_emb_grad_tiling_ab.cpp` 中注册，优先级 30000，
适用条件是 regbase SoC 且内部 `layout_ == SBND`：

- `layout=BSND` 且 `cosB > 1`：输入实际是 `(B,S,N,D)`，`cos/sin` 是 `(B,S,1,D)`，
  内部归入 AB。
- `layout=SBND`：输入实际是 `(S,B,N,D)`，`cos/sin` 通常是 `(S,B|1,1,D)`，
  内部也走 AB。

AB 的核心分核方式是二维分核：

- BS 合并轴：`blockDimBS_ = blockIdx / blockNumN`
- N 轴：`blockDimN_ = blockIdx % blockNumN`

也就是说，kernel 中一个 core 负责一段连续的 BS 行，以及一段连续的 N head。

## 3. host tiling 逻辑

### 3.1 BS/N 归一化

`ApplyRotaryPosEmbGradRegbaseTilingClassAB::DoOpTiling()` 会先把输入规整为 `bs_` 和 `totalN*`：

- `cosb_ == 1`：认为 cos 沿 B 广播，因此 `bs_ = s_`，并把 B 合并进 N：
  `totalNQ_ = b_ * nQ_`，`totalNK_ = b_ * nK_`。
- `cosb_ > 1`：认为每个 `(B,S)` 都有独立 cos/sin 行，因此 `bs_ = b_ * s_`，
  `totalNQ_ = nQ_`，`totalNK_ = nK_`。
- `bn_ = max(totalNQ_, totalNK_)`。

`PostTiling()` 中：

- `ropeGradABParams.nQ/nK` 写入归一化后的 `totalNQ_/totalNK_`，供 AB kernel 使用。
- `ropeGradParams.nQ/nK` 仍写原始 `nQ_/nK_`，供 `apt.cpp` 计算 `partSize`：
  `b * s * max(nQ, nK) * d`。

两者在元素总量上是一致的：

- `cosb > 1`：`bs * maxN = b*s*max(nQ,nK)`。
- `cosb == 1`：`bs * maxN = s * b*max(nQ,nK)`。

### 3.2 core split

`SplitCore()` 先切 BS：

- `blockFactorBS = ceil(bs / coreNum)`
- `blockNumBS = ceil(bs / blockFactorBS)`

如果 BS 数较少，即 `bs <= coreNum / 2`，再把剩余 core 分给 N：

- `blockNumN = coreNum / blockNumBS`
- `blockFactorN = ceil(bn / blockNumN)`
- `blockNumN = ceil(bn / blockFactorN)`

否则 N 轴不分核：

- `blockNumN = 1`
- `blockFactorN = bn`

### 3.3 UB split

`SplitUb()` 按 half rotary 模式计算 `dAlign`：

```text
dAlign = ceil_align(D/2, 32B/typeSize) * 2
```

`dAlign == d` 表示 `D/2` 已经 32B 对齐，kernel 可以走整行搬运；否则按前后半行拆开搬运。

UB 预算用 divisor 粗分。当前修复后，AB 四输出下 `gradCosPartial/gradSinPartial`
按 float 申请，因此非 float 输入的四输出 divisor 从 16 调整为 20：

- 两输出：`AB_UB_DIVISOR_WITHOUT_DCOS = 8`
- 四输出且输入为 float：`AB_UB_DIVISOR_WITH_DCOS = 16`
- 四输出且输入为 fp16/bf16：`AB_UB_DIVISOR_WITH_DCOS_FLOAT_PARTIAL = 20`

四输出多了 `query/key/gradCosPartial/gradSinPartial` 四类 tensor，因此 UB tile 会比两输出更小。

## 4. kernel phase1: dx 与部分积

AB kernel 在 `op_kernel/arch35/apply_rotary_pos_emb_grad_ab.h`。

### 4.1 两输出路径

当 `kDcosFlag=false` 时，只执行：

```text
grad_query = d(query_embed)/d(query)
grad_key   = d(key_embed)/d(key)
```

计算公式在 `HalfGradAlignVF()` 中：

```text
dx[0:D/2] = grad[0:D/2] * cos[0:D/2] + grad[D/2:D] * sin[D/2:D]
dx[D/2:D] = grad[D/2:D] * cos[D/2:D] - grad[0:D/2] * sin[0:D/2]
```

这条路径不分配 `queryInQue_`、`keyInQue_`、`gradCosOutQue_`、`gradSinOutQue_`，
也不会写 workspace 或调用 reduce。

### 4.2 四输出路径

当 `kDcosFlag=true` 时，phase1 在计算 dx 的同时计算部分积：

```text
grad_cos_partial = grad * x
grad_sin_partial = grad * rotate(x)
```

half 模式下：

```text
grad_cos[0:D/2] = grad1 * x1
grad_cos[D/2:D] = grad2 * x2

grad_sin[0:D/2] = -grad1 * x2
grad_sin[D/2:D] =  grad2 * x1
```

AB 对 Q/K 两路采用同一个 `cosPartial/sinPartial` UB buffer：

1. head 数更多的一路先以 SET 模式写入；
2. head 数更少的一路再以 ADD 模式累加；
3. tile 结束后写一次 workspace。

UB 中部分积的行距固定使用 `ubFactorN_`，不是当前路径的 `currQN/currKN`，避免 Q/K
head 数不一致时同一个 tile 内行偏移不一致。

### 4.3 workspace 写入布局

AB phase1 写入两份连续 workspace：

```text
cos partial base = usrWorkSpace
sin partial base = usrWorkSpace + partSize * sizeof(float)
```

其中：

```text
partSize = b * s * max(raw nQ, raw nK) * d
```

AB kernel 内部写入偏移是：

```text
wsOffset = bsAbsIdx * maxN_ * d + absN * d
```

这里的 `maxN_` 是归一化后的 `max(totalNQ,totalNK)`。

逻辑上可理解为：

- `cosb > 1`：workspace 是 `[B*S, max(nQ,nK), D]`。
- `cosb == 1`：workspace 是 `[S, B*max(nQ,nK), D]`。

两者扁平元素数都等于 `b*s*max(nQ,nK)*d`。

## 5. phase2 reduce

入口在 `op_kernel/apply_rotary_pos_emb_grad_apt.cpp`。AB/BAB 且 `DcosFlag=1` 时：

1. phase1 写完两份部分积；
2. `SyncAll()` 保证所有 core 的 workspace 写入完成；
3. 构造两个 `ReduceSch`：
   - `reduceOp0`: `usrWorkSpace -> gradCosOut`
   - `reduceOp1`: `dSinWorkSpace -> gradSinOut`
4. 两次 reduce 复用同一段 reduce 临时区。AB 修复后 partial workspace 按 float 布局：

```text
reduceWorkSpace = usrWorkSpace + 2 * partSize * sizeof(float)
```

reduce DAG 是单输入单输出：

```text
CopyIn<InT> -> Cast<float> -> ReduceSum<float> -> Cast<OutT> -> CopyOut<OutT>
```

也就是说，phase2 只对 phase1 产生的部分积求和，不再做乘法。AB 四输出修复后使用
`InT=float`、`OutT=DTYPE_GRAD_QUERY_EMBED`；BAB 仍保持原来的 `T` partial 路径。

### 5.1 reduce axes 的来源

host 侧 `TilingReduce()` 调：

```cpp
GetInputParam(opInput, GQ_INDEX, COS_INDEX)
```

`GetInputParam()` 做两件事：

- `opInput.shape` 直接取 `grad_query_embed` 的 storage shape；
- `opInput.axes` 取 `cos` 中维度为 1、但 `grad_query_embed` 对应维度不为 1 的轴。

当前修复对 AB reduce 额外做了两点限定修改：

- `opInput.inputDtype = DT_FLOAT`，匹配 AB phase1 写入的 float partial；
- `opInput.shape[N] = max(nQ, nK)`，匹配 AB workspace 的 `maxN` 布局。

因此：

- BSND + cos `(1,S,1,D)`：reduce axes 为 B、N。
- BSND + cos `(B,S,1,D)`：reduce axis 为 N。
- SBND + cos `(S,1,1,D)`：reduce axes 为 B、N。
- SBND + cos `(S,B,1,D)`：reduce axis 为 N。

这与数学目标一致：`grad_cos/grad_sin` 的 shape 和 `cos/sin` 相同，所有广播轴都需要 sum。

## 6. 四输出精度 fail 原因分析

### 6.1 高概率结构性原因：reduce shape 使用 nQ，不覆盖 nK > nQ

当前 reduce tiling 的输入 shape 来自 `grad_query_embed`。如果 Q/K head 数不同：

```text
opInput.shape 的 N = nQ
phase1 workspace 的 N = max(nQ, nK)
```

当 `nK > nQ` 时：

- K 路尾部 head 的 partial 会写到 workspace 的 `[nQ, nK)` 区间；
- reduce 模板只按 `nQ` 大小生成 tiling；
- 结果是 `grad_cos/grad_sin` 少规约 K 路多出来的 head。

这个问题只影响后两个输出，因为前两个输出 `grad_query/grad_key` 分别按自己的
`nQ/nK` 直接写回，不依赖 reduce。

修复方向：reduce tiling 的逻辑输入 shape 不能固定使用 GQ shape。对于 dcos/dsin reduce，
应把被 reduce 的 N 维改为 `max(nQ,nK)`；AB 的 `cosb==1` 场景如果 B 已合并进 N，仍可用
`[S,B,max(nQ,nK),D]` 描述 reduce 输入，因为其扁平布局与 `[S,B*maxN,D]` 等价。

### 6.2 AB 的 `cosS == 1` 广播未被 kernel 布局覆盖

shape 校验中 SBND 分支允许 `cosS == 1` 或 `cosS == gqS`。但 AB kernel 搬入 cos/sin 时总是：

```text
cosGm[bsAbsIdx * d]
sinGm[bsAbsIdx * d]
```

即默认每个 BS 行都有独立 cos/sin。若实际 `cos` 是 `(1,B,1,D)` 或 `(1,1,1,D)`，
kernel 会按 `S*B` 行读取，超出或错读广播后的 cos/sin。

这个问题会同时影响 dx 和 dcos/dsin，因此如果当前“两输出 pass”，说明触发用例大概率不是
`cosS==1`；但它仍是 AB 模板广播支持范围和 kernel 实现之间的一个不一致点。

### 6.3 修改前部分积以 T 类型落 workspace，reduce 前已经截断

修改前 `ComputeGradWithCosSinVF()` 中乘法和加法使用 fp32 register，但写入 `cosPartial/sinPartial`
时调用 `StoreOneTensorForDtypeT<T>`。当 `T=fp16/bf16`：

- Q/K tile 内的部分积在 UB 中以 T 保存；
- 写入 workspace 也是 T；
- phase2 reduce 再从 T cast 到 fp32 做 sum。

这不是“错值”问题，而是“先截断再规约”的精度风险。如果 golden/reference 是 fp32 累加后最后
cast，长 N/B/S reduce 时误差会放大。该风险对 `grad_query/grad_key` 不明显，因为它们本来就是
逐元素输出，没有跨大量 head/batch 的求和。

如果 fail 表现为小幅相对/绝对误差，而不是整段缺失或位置错乱，应优先验证这一点。

当前 AB 修复已经把 `cosPartial/sinPartial` 的 UB tensor、workspace GM tensor、workspace
拷贝字节数以及 phase2 reduce 输入统一改为 float。最终 `grad_cos/grad_sin` 仍按输出 dtype
写回，避免改变算子对外 dtype 约束。

### 6.4 reduce 临时 workspace 边界需要确认

kernel 中 reduce 临时区从两份 partial 后开始：

```text
usrWorkSpace
| cosPartial: partSize * sizeof(float)  # AB 修复后
| sinPartial: partSize * sizeof(float)  # AB 修复后
| reduce temp: ReduceSch 使用
```

host 侧 `SetTilingKeyBlockDim()` 显式增加的是：

```text
b * s * max(nQ,nK) * d * sizeof(float) * 2  # 仅 AB 四输出
```

reduce 模板自身是否已经把 reduce temp 计入 `context->GetWorkspaceSizes(1)`，需要结合运行时
workspace size 或 reduce tiling 日志确认。若没有计入，则 `reduceWorkSpace` 会落在用户申请
workspace 末尾之后，表现可能是后两个输出随机错、不同 shape 下不稳定、甚至覆盖相邻内存。

该点不是从本目录代码单独能完全证明的问题，但它是四输出专属路径，值得作为排查项。

## 7. 建议验证点

1. 构造 AB 四输出、`nK > nQ` 用例，dump phase1 workspace 与 reference partial。
   重点看 workspace 中 `[nQ, nK)` 的 K tail 是否存在，以及 reduce 输出是否少加。
2. 构造 AB 四输出、`nQ == nK` 用例。
   如果仍 fail，优先看部分积 T 落盘精度和 workspace 边界；如果 pass，则 reduce shape 的
   `nQ/maxN` 问题基本坐实。
3. 构造 AB SBND、`cosS == 1` 用例。
   如果前两个输出也 fail，则说明 AB kernel 未实现 S 广播读址。
4. 对 fp16/bf16 用例分别比较：
   - phase1 workspace partial 与 fp32 reference cast-to-T 后的差异；
   - reduce 输出与“先 cast partial 到 T 再 fp32 reduce”的 reference 差异。
   这样可以区分实现错位和数值累加策略差异。

## 8. 修复方向

优先级建议：

1. 修正 `TilingReduce()` 的 AB dcos/dsin reduce 输入 shape，使 N 维使用 `max(nQ,nK)`，
   不再固定继承 `grad_query_embed` 的 N。当前代码已按 AB 限定修改。
2. 若需要支持 SBND 的 `cosS==1`，AB kernel 的 cos/sin GM offset 需要按广播规则计算，
   不能直接用 `bsAbsIdx * d`。
3. 若是纯数值精度 fail，需要评估 workspace partial 使用 fp32 的成本。当前代码已将 AB
   partial 改为 float，避免 BF16 partial 先截断再 reduce。
4. 确认 reduce temp workspace 是否由 reduce tiling 已加入总 workspace；如未加入，需要 host
   侧额外累加 reduce 模板要求的 workspace 大小。
