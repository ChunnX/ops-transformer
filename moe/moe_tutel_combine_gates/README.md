# MoeTutelCombineGates

## 产品支持情况

| 产品                                    | 是否支持 |
| --------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> |    √     |
| <term>Atlas A3系列产品</term>           |    ×     |
| <term>Atlas A2系列产品</term>           |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>  |    ×     |
| <term>Atlas推理系列产品</term>          |    ×     |
| <term>Atlas训练系列产品</term>          |    ×     |

## 功能说明

- 算子功能：计算Tutel MoE门控权重的反向梯度。按路由下标`indices`/`locations`从dispatch缓冲回传梯度`y_grad`中取出槽位行，与该task的token行`x`做hidden维内积，输出逐task的门控梯度`gates_grad`。

- 计算公式：

  $$
  gates\_grad[b, s] = \sum_{h=0}^{H-1} x[s, h] \times y\_grad[e \times C + p,\ h],\quad e = indices[b, s],\ p = locations[b, s]
  $$

- 计算过程：

  1. 由输入shape推出`S = x.shape[0]`、`H = x.shape[1] = y_grad.shape[2]`、`B = y_grad.shape[0] = indices.shape[0]`、`C = y_grad.shape[1]`；`indices`/`locations`/`gates_grad`的元素数均为`S×B`。
  2. 展平task下标`t ∈ [0, S×B)`：token行 = `t % S`，槽位行 = `indices[t] × C + locations[t]`（即`e × C + p`）。
  3. 逐task计算`x`的token行与`y_grad`的槽位行的hidden维内积，写回`gates_grad[t]`。
  4. 非法路由`e < 0`、`e ≥ B`、`p < 0`、`p ≥ C`对应的task结果为精确`0`（不读`y_grad`，不发起越界访存）。其中会真正越出`y_grad`张量范围的`e ≥ B`与`p < 0`由校验层在取值**可被可信读出**时前置拒收：仅当框架以host张量（`kOnHost`/`kFollowing`）提供取值**且该张量带非空数据指针**时，tiling直读命中即拒收、`TilingFunc`返回失败（图模式下以GE/tiling错误码上报）；其余情形一律**不读值、不拒收**——host放置但取不到数据指针、以及设备张量通路（tiling拿不到stride/非连续描述），都记INFO后放行，同样由kernel的四条件掩码`(e ≥ 0) ∧ (e < B) ∧ (p ≥ 0) ∧ (p < C)`置`0`（分层口径见[约束说明](#约束说明)）。

- 计算精度：`x`/`y_grad`为bfloat16或float16时先转换到float32，逐元素乘法在float32内完成，随后做一次**精确的2的幂归一化**（取全部乘积的`max|·|`得到`2^k`，整体乘`2^-k`后规约、写回前再精确乘回`2^k`；数学上与直接求和等价，用于拓宽累加的指数窗口、避免部分和上溢），hidden维规约实现为fp32 **补偿归约**（TwoSum补偿累加，结果即精确乘积之和的正确舍入值），最后按`CAST_ROUND`回写到输出dtype；float32输入不做归一化、直接以fp32完成乘法（含TwoProduct精确乘积误差通道）与补偿归约，累加精度固定为float32，不降为低精度累加。

- 本算子无属性：`C`（capacity）由`y_grad.shape[1]`隐式给出，`S`/`H`/`B`全部由输入shape推出，调用方不能单独配置。

## 参数说明

参数名、顺序与`op_host/moe_tutel_combine_gates_def.cpp`的`Input`/`Output`声明一致，与图模式接口`op_graph/moe_tutel_combine_gates_proto.h`的`REG_OP(MoeTutelCombineGates)`逐项对应。

<table style="table-layout: fixed; width: 1500px">
<colgroup>
<col style="width: 120px">
<col style="width: 150px">
<col style="width: 380px">
<col style="width: 250px">
<col style="width: 100px">
<col style="width: 250px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
<th>维度(shape)</th>
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>输入</td>
<td>按token的隐藏态，公式中的x；dtype与gates_grad相同。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
<td>[S, H]（rank 2）</td>
</tr>
<tr>
<td>y_grad</td>
<td>输入</td>
<td>dispatch缓冲回传梯度，公式中的y_grad；按槽位(e, p)随机读。</td>
<td>与x保持一致</td>
<td>ND</td>
<td>[B, C, H]（rank 3）</td>
</tr>
<tr>
<td>indices</td>
<td>输入</td>
<td>路由专家下标，公式中的e；逐task一一对应，按平坦的S×B数组消费。合法取值域[0, B)：<code>e ≥ B</code>仅在框架以host张量提供取值<strong>且带数据指针</strong>时由校验层拒绝（图模式下以GE/tiling错误码上报），host放置但无数据指针与设备张量通路下均不拒收、对应task输出精确0（见[约束说明](#约束说明)），<code>e &lt; 0</code>不被拒绝、对应task输出精确0。</td>
<td>INT32</td>
<td>ND</td>
<td>[B, S]（rank 2，元素数S×B）</td>
</tr>
<tr>
<td>locations</td>
<td>输入</td>
<td>专家内槽位，公式中的p；与indices同形（唯一例外：<code>S = 0</code>时leading dim无约束，见[约束说明](#约束说明)的「空Tensor」条），按平坦的S×B数组消费。合法取值域[0, C)：<code>p &lt; 0</code>仅在框架以host张量提供取值<strong>且带数据指针</strong>时由校验层拒绝（图模式下以GE/tiling错误码上报），host放置但无数据指针与设备张量通路下均不拒收、对应task输出精确0（见[约束说明](#约束说明)），<code>p ≥ C</code>不被拒绝、对应task输出精确0。</td>
<td>INT32</td>
<td>ND</td>
<td>[B, S]（rank 2，元素数S×B）</td>
</tr>
<tr>
<td>gates_grad</td>
<td>输出</td>
<td>门控权重梯度，公式中的gates_grad；shape等于indices.shape，元素数S×B，dtype与浮点输入相同。空Tensor：<code>S = 0</code>或<code>B = 0</code>时输出为空张量（shape与<code>indices.shape</code>一致，无论<code>H</code>/<code>C</code>是否为0）；<code>S</code>与<code>B</code>均非0时<code>H = 0</code>或<code>C = 0</code>给出<strong>整张量精确<code>0</code></strong>（不是空张量）。输出由框架按推导shape分配、天然连续，见[约束说明](#约束说明)的「非连续Tensor」条。</td>
<td>与x保持一致</td>
<td>ND</td>
<td>[B, S]</td>
</tr>
</tbody>
</table>

- 属性：无。

## 约束说明

> [!NOTE]
> **范围声明**：本节全部约束描述的是**本仓库Ascend 950（`arch35`）实现**的实际行为。其中rank/shape契约、元素数、dtype/format、int32上界、取值域与UB容量等校验，以及非法路由置`0`、空Tensor语义，均为该实现的行为口径。产品级支持情况见上方「产品支持情况」表。

- 可用性：算子定义只注册`ascend950`这一份AICore配置（`op_host/moe_tutel_combine_gates_def.cpp`的`this->AICore().AddConfig("ascend950", aiCoreConfig)`），故仅在<term>Ascend 950PR/Ascend 950DT</term>（`arch35`）上注册与支持。
- 确定性：本算子默认确定性实现。规约轴`H`不跨核切分（分核只沿展平task轴），同一设备、同一输入多次执行结果bitwise一致。
- 输入形态：`x`必须rank 2、`y_grad`必须rank 3、`indices`/`locations`必须rank 2（`gates_grad`为rank 2）；`indices`/`locations`/`gates_grad`的元素数必须等于`S×B`；`gates_grad`的shape必须等于`indices.shape`。不满足时报错，不静默越界。
- 动态shape：本算子声明动态shape/动态rank支持（`DynamicShapeSupportFlag`/`DynamicRankSupportFlag`），InferShape允许**未解析维（负值）**透传，但tiling的数值范围校验对未解析维**直接拒收**（`OP_LOGE_FOR_INVALID_VALUE_WITH_REASON`上报，返回`ge::GRAPH_FAILED`）——未解析维若进入kernel会以约4.29e9的`uint32`计数进入向量指令与`DataCopyPad`的`blockLen`（合法域`[1, 2097151]`）。故调用方必须保证四个维度在执行前**均已解析**，且逐维取值不超过`INT32_MAX`。
- 数据类型：`x`、`y_grad`、`gates_grad`三者dtype一致，取BFLOAT16/FLOAT16/FLOAT之一；`indices`/`locations`固定为INT32。未列出的组合（如索引为INT64、输入为FLOAT64、`x`与`y_grad` dtype不一致、输出dtype与`x`不一致）被拒绝。
- 数据格式：仅支持ND；不支持broadcast，各输入shape必须严格满足`S`/`H`/`B`/`C`的绑定关系。
- 非连续Tensor：算子定义对四个输入与输出都声明了`.AutoContiguous()`，kernel统一按**连续语义**取行/写回；本算子的aclnn调用方式（及其生成桩的连续化视图拷贝通路）已删除，仅保留GE图模式：
  - **输入**：图模式下不存在调用方手工传入非连续张量的入口——`RunGraph`喂入的图输入是「shape+平坦缓冲」，图内由上游算子产出、框架分配的中转张量本身连续；`.AutoContiguous()`作为「进入kernel的输入必须连续」的契约声明保留。
  - **输出`gates_grad`**：由GE按InferShape推导出的shape分配，天然连续；调用方不自行分配输出内存。kernel/tiling层不感知输出stride，按连续语义写回。
  - **`indices`/`locations`的取值域校验**：本校验**只在框架以host张量（`kOnHost`/`kFollowing`）提供取值、且该张量带非空数据指针时才生效**（此时读到的就是逻辑元素序列本身，判据可信）；host放置但取不到数据指针时记INFO后放行。设备张量通路下**不读值、不拒收**——该通路喂给tiling的描述符不带任何非连续信息（stride/offset为空、storage维等于视图维），按线性缓冲读`S×B`个int32无法证明就是逻辑元素序列：**按线性缓冲读值会误拒合法输入**，而误拒合法输入比漏检更严重，故取「宁可放行、不误拒」，越界取值改由kernel四条件掩码置`0`兜底。
- 空Tensor：允许。**多轴同时为0的优先级（本代契约）**：先判`taskNum = S×B`是否为0——`S = 0`或`B = 0`时输出为空张量（shape与`indices.shape`一致），**无论`H`/`C`是否为0**；只有`S > 0 ∧ B > 0`时，`H = 0`（内积为空）或`C = 0`（所有task非法）才给出**整张量精确`0`**。该契约与实现一致：tiling的空Tensor短路`HandleEmptyTensor()`只判`taskNum == 0`（此时不写任何数据、`taskNumPerBlock = 0`、`SetBlockDim(1)`），而`H = 0`/`C = 0`的其他组合进入正常通路，由kernel的`hidden <= 0`短路与四条件掩码写出精确`0`。`S = 0`另带一条shape豁免（infershape与tiling两处同口径）：`locations.shape[0] = B`仅在`S ≠ 0`时判——`S = 0`时`locations`的leading dim无约束（kernel按`S×B`平坦数组消费`indices`/`locations`，`taskNum = 0`时一个元素也不读；元素数契约`S×B = 0`与`locations.shape[1] = S`已把`locations`钉在`(*, 0)`）；`indices.shape[0] = B`则恒判。
- 取值域：`indices ≥ B`与`locations < 0`会越过`y_grad`张量范围。本代实现**保证绝不发生越界访存**（kernel的四条件掩码`(e ≥ 0) ∧ (e < B) ∧ (p ≥ 0) ∧ (p < C)`不成立时`continue`、不发起任何DMA，槽位保持精确`0`），但**校验层的前置拒收只在取值可被可信读出时生效**，判据是「张量放置」与「数据指针」两个条件**同时成立**：
  - **直读拒收（仅一种情形）**：框架以host张量（`kOnHost`/`kFollowing`）提供取值**且该张量带非空数据指针**时（如常量折叠等由框架把取值物化到host的通路），tiling直读取值，命中`indices ≥ B`或`locations < 0`即拒收、返回`ge::GRAPH_FAILED`，以tiling失败对应的GE错误码上报。
  - **放行（记INFO）：host放置但无数据指针**——张量为`kOnHost`/`kFollowing`，但取值指针为空（`GetData<int32_t>() == nullptr`）⇒ 本层不可判，放行；`indices ≥ B`/`locations < 0`由kernel掩码置`0`。
  - **放行（记INFO）：设备张量**——tiling收到的描述符不带任何非连续信息（无stride/offset、storage维等于视图维），按线性缓冲读值无法证明读到的是逻辑元素序列，因此**不读值、不拒收**，同样由kernel掩码置`0`。
    即：**越界访存永不发生；`indices ≥ B`/`locations < 0`的「前置拦截」在上述两类放行路径上改为「由kernel掩码置0」**。`indices < 0`与`locations ≥ C`在任何通路上都不由校验层拒绝，对应task输出精确`0`（整块task非法时同样写回`0`，不保留旧值）。
- 维数上界：`S`/`H`/`B`/`C`由TilingData的int32字段承载，`S×B`（`taskNum`）不得超过`INT32_MAX - 256`（分核粒度取256对齐，故预留该余量）；`S`/`B`/`H`/`C`逐维不得超过`INT32_MAX`（`C`是槽位地址乘子`slot = e×C + p`，截断会算出错误地址）。超界即返回失败，不静默截断、不降档。
- 隐藏维上界：`H`整段驻留UB且不切分，超过UB容量上限时返回失败且不降档。上限由platform接口实时取`ubSize`计算：tiling把每条`H`长UB路由的元素数向上对齐到`max(blockSize, vectorSize) / sizeof(dtype)`、块级小路由各按自己的block对齐后求和，得到`needBytes(H)`，取满足`needBytes(H) ≤ ubSize`的最大`H`（本算子不调用`ReserveLocalMemory()`，故预留量为0）。以`ubSize = 253952` B、UB block `32` B、向量寄存器宽`256` B代入时：float32路径`H ≤ 15808`，FLOAT16/BFLOAT16路径`H ≤ 12608`；该上限随`ubSize`/`blockSize`/`vectorSize`变化，不是固定常量。
- 特殊值：`x`或`y_grad`含NaN时受影响token行的task输出为NaN；含`+inf`时按IEEE-754逐元素传播——内积由`x_h × y_grad_h`逐项相乘再相加，故**同号且不出现`0 × inf`**时结果为`+inf`，出现`0 × inf`（某一`h`上`x`或`y_grad`为0）或异号`inf`相消时为`NaN`；浮点溢出不做饱和（无clamp/饱和路径）；FLOAT16路径回写超过65504的输出为`±inf`（`CAST_ROUND`，不饱和）。
- 设备：仅在Ascend 950PR/Ascend 950DT（`arch35` = `dav-3510`）上注册与支持。

## 调用说明

| 调用方式 | 样例代码                                                                                   | 说明                                                                                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GE图模式 | [test_geir_moe_tutel_combine_gates](examples/arch35/test_geir_moe_tutel_combine_gates.cpp) | 通过[算子IR](op_graph/moe_tutel_combine_gates_proto.h)的`REG_OP(MoeTutelCombineGates)`构图方式调用；本算子无属性。样例覆盖BFLOAT16/FLOAT16/FLOAT三个dtype组合，并与CPU参考实现逐元素比对。 |

> [!NOTE]
> 本算子在本仓库内只提供GE图模式这一种调用方式（aclnn单算子调用方式已删除），无PyTorch/TensorFlow适配通路。

## 测试说明

- 单元测试（host侧tiling/infershape）：`moe/moe_tutel_combine_gates/tests/ut/op_host/`
  ```bash
  bash build.sh --ophost_test --soc=ascend950 --ops=moe_tutel_combine_gates
  ```
- ST（Kernel精度，TTK框架）：用例与TestSpec见[`tests/assets/README.md`](tests/assets/README.md)
  ```bash
  bash scripts/ci/ops_st_test.sh --soc_version=ascend950 --ops=moe_tutel_combine_gates --test_type=kernel
  ```
- GE图模式端到端样例：`examples/arch35/test_geir_moe_tutel_combine_gates.cpp`
  ```bash
  bash build.sh --run_example moe_tutel_combine_gates graph --soc=ascend950
  ```
