# MoeTutelCombineGates

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：计算 Tutel MoE 门控权重的反向梯度。按路由下标 `indices` / `locations` 从 dispatch 缓冲回传梯度 `y_grad` 中取出槽位行，与该 task 的 token 行 `x` 做 hidden 维内积，输出逐 task 的门控梯度 `gates_grad`。

- 计算公式：

  $$
  gates\_grad[b, s] = \sum_{h=0}^{H-1} x[s, h] \times y\_grad[e \times C + p,\ h],\quad e = indices[b, s],\ p = locations[b, s]
  $$

- 计算过程：

  1. 由输入 shape 推出 `S = x.shape[0]`、`H = x.shape[1] = y_grad.shape[2]`、`B = y_grad.shape[0] = indices.shape[0]`、`C = y_grad.shape[1]`；`indices` / `locations` / `gates_grad` 的元素数均为 `S×B`。
  2. 展平 task 下标 `t ∈ [0, S×B)`：token 行 = `t % S`，槽位行 = `indices[t] × C + locations[t]`（即 `e × C + p`）。
  3. 逐 task 计算 `x` 的 token 行与 `y_grad` 的槽位行的 hidden 维内积，写回 `gates_grad[t]`。
  4. 非法路由 `e < 0`、`e ≥ B`、`p < 0`、`p ≥ C` 对应的 task 结果为精确 `0`（不读 `y_grad`，不发起越界访存）。其中会真正越出 `y_grad` 张量范围的 `e ≥ B` 与 `p < 0` 由校验层在取值**可被可信读出**时前置拒收：仅当框架以 host 张量（`kOnHost` / `kFollowing`）提供取值**且该张量带非空数据指针**时，tiling 直读命中即拒收、`TilingFunc` 返回失败（图模式下以 GE / tiling 错误码上报）；其余情形一律**不读值、不拒收**——host 放置但取不到数据指针、以及设备张量通路（tiling 拿不到 stride / 非连续描述），都记 INFO 后放行，同样由 kernel 的四条件掩码 `(e ≥ 0) ∧ (e < B) ∧ (p ≥ 0) ∧ (p < C)` 置 `0`（分层口径见[约束说明](#约束说明)）。

- 计算精度：`x` / `y_grad` 为 bfloat16 或 float16 时先转换到 float32，逐元素乘法在 float32 内完成，随后做一次**精确的 2 的幂归一化**（取全部乘积的 `max|·|` 得到 `2^k`，整体乘 `2^-k` 后规约、写回前再精确乘回 `2^k`；数学上与直接求和等价，用于拓宽累加的指数窗口、避免部分和上溢），hidden 维规约实现为 fp32 **补偿归约**（TwoSum 补偿累加，结果即精确乘积之和的正确舍入值），最后按 `CAST_ROUND` 回写到输出 dtype；float32 输入不做归一化、直接以 fp32 完成乘法（含 TwoProduct 精确乘积误差通道）与补偿归约，累加精度固定为 float32，不降为低精度累加。

- 本算子无属性：`C`（capacity）由 `y_grad.shape[1]` 隐式给出，`S` / `H` / `B` 全部由输入 shape 推出，调用方不能单独配置。

## 参数说明

参数名、顺序与 `op_host/moe_tutel_combine_gates_def.cpp` 的 `Input` / `Output` 声明一致，与图模式接口 `op_graph/moe_tutel_combine_gates_proto.h` 的 `REG_OP(MoeTutelCombineGates)` 逐项对应。

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
<td>按 token 的隐藏态，公式中的 x；dtype 与 gates_grad 相同。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
<td>[S, H]（rank 2）</td>
</tr>
<tr>
<td>y_grad</td>
<td>输入</td>
<td>dispatch 缓冲回传梯度，公式中的 y_grad；按槽位 (e, p) 随机读。</td>
<td>与 x 保持一致</td>
<td>ND</td>
<td>[B, C, H]（rank 3）</td>
</tr>
<tr>
<td>indices</td>
<td>输入</td>
<td>路由专家下标，公式中的 e；逐 task 一一对应，按平坦的 S×B 数组消费。合法取值域 [0, B)：<code>e ≥ B</code> 仅在框架以 host 张量提供取值<strong>且带数据指针</strong>时由校验层拒绝（图模式下以 GE / tiling 错误码上报），host 放置但无数据指针与设备张量通路下均不拒收、对应 task 输出精确 0（见[约束说明](#约束说明)），<code>e &lt; 0</code> 不被拒绝、对应 task 输出精确 0。</td>
<td>INT32</td>
<td>ND</td>
<td>[B, S]（rank 2，元素数 S×B）</td>
</tr>
<tr>
<td>locations</td>
<td>输入</td>
<td>专家内槽位，公式中的 p；与 indices 同形，按平坦的 S×B 数组消费。合法取值域 [0, C)：<code>p &lt; 0</code> 仅在框架以 host 张量提供取值<strong>且带数据指针</strong>时由校验层拒绝（图模式下以 GE / tiling 错误码上报），host 放置但无数据指针与设备张量通路下均不拒收、对应 task 输出精确 0（见[约束说明](#约束说明)），<code>p ≥ C</code> 不被拒绝、对应 task 输出精确 0。</td>
<td>INT32</td>
<td>ND</td>
<td>[B, S]（rank 2，元素数 S×B）</td>
</tr>
<tr>
<td>gates_grad</td>
<td>输出</td>
<td>门控权重梯度，公式中的 gates_grad；shape 等于 indices.shape，元素数 S×B，dtype 与浮点输入相同。空Tensor：<code>S = 0</code>或<code>B = 0</code>时输出为空张量（shape与<code>indices.shape</code>一致，无论<code>H</code> / <code>C</code>是否为0）；<code>S</code>与<code>B</code>均非0时<code>H = 0</code>或<code>C = 0</code>给出<strong>整张量精确<code>0</code></strong>（不是空张量）。输出由框架按推导 shape 分配、天然连续，见[约束说明](#约束说明)的「非连续 Tensor」条。</td>
<td>与 x 保持一致</td>
<td>ND</td>
<td>[B, S]</td>
</tr>
</tbody>
</table>

- 属性：无。

## 约束说明

> **范围声明**：本节全部约束描述的是**本仓库 Ascend 950（`arch35`）实现**的实际行为。其中 rank / shape 契约、元素数、dtype / format、int32 上界、取值域与 UB 容量等校验，以及非法路由置 `0`、空 Tensor 语义，均为该实现的行为口径。产品级支持情况见上方「产品支持情况」表。

- 可用性：算子定义只注册 `ascend950` 这一份 AICore 配置（`op_host/moe_tutel_combine_gates_def.cpp` 的 `this->AICore().AddConfig("ascend950", aiCoreConfig)`），故仅在 <term>Ascend 950PR/Ascend 950DT</term>（`arch35`）上注册与支持。
- 确定性：本算子默认确定性实现。规约轴 `H` 不跨核切分（分核只沿展平 task 轴），同一设备、同一输入多次执行结果 bitwise 一致。
- 输入形态：`x` 必须 rank 2、`y_grad` 必须 rank 3、`indices` / `locations` 必须 rank 2（`gates_grad` 为 rank 2）；`indices` / `locations` / `gates_grad` 的元素数必须等于 `S×B`；`gates_grad` 的 shape 必须等于 `indices.shape`。不满足时报错，不静默越界。
- 动态shape：本算子声明动态shape / 动态rank支持（`DynamicShapeSupportFlag` / `DynamicRankSupportFlag`），InferShape允许**未解析维（负值）**透传，但tiling的数值范围校验对未解析维**直接拒收**（`OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` 上报，返回 `ge::GRAPH_FAILED`）——未解析维若进入kernel会以约4.29e9的`uint32`计数进入向量指令与`DataCopyPad`的`blockLen`（合法域 `[1, 2097151]`）。故调用方必须保证四个维度在执行前**均已解析**，且逐维取值不超过`INT32_MAX`。
- 数据类型：`x`、`y_grad`、`gates_grad` 三者 dtype 一致，取 BFLOAT16 / FLOAT16 / FLOAT 之一；`indices` / `locations` 固定为 INT32。未列出的组合（如索引为 INT64、输入为 FLOAT64、`x` 与 `y_grad` dtype 不一致、输出 dtype 与 `x` 不一致）被拒绝。
- 数据格式：仅支持 ND；不支持 broadcast，各输入 shape 必须严格满足 `S` / `H` / `B` / `C` 的绑定关系。
- 非连续 Tensor：算子定义对四个输入与输出都声明了 `.AutoContiguous()`，kernel 统一按**连续语义**取行 / 写回；本算子的 aclnn 调用方式（及其生成桩的连续化视图拷贝通路）已删除，仅保留 GE 图模式：
  - **输入**：图模式下不存在调用方手工传入非连续张量的入口——`RunGraph` 喂入的图输入是「shape + 平坦缓冲」，图内由上游算子产出、框架分配的中转张量本身连续；`.AutoContiguous()` 作为「进入 kernel 的输入必须连续」的契约声明保留。
  - **输出 `gates_grad`**：由 GE 按 InferShape 推导出的 shape 分配，天然连续；调用方不自行分配输出内存。kernel / tiling 层不感知输出 stride，按连续语义写回。
  - **`indices` / `locations` 的取值域校验**：本校验**只在框架以 host 张量（`kOnHost` / `kFollowing`）提供取值、且该张量带非空数据指针时才生效**（此时读到的就是逻辑元素序列本身，判据可信）；host 放置但取不到数据指针时记 INFO 后放行。设备张量通路下**不读值、不拒收**——该通路喂给 tiling 的描述符不带任何非连续信息（stride / offset 为空、storage 维等于视图维），按线性缓冲读 `S×B` 个 int32 无法证明就是逻辑元素序列：**按线性缓冲读值会误拒合法输入**，而误拒合法输入比漏检更严重，故取「宁可放行、不误拒」，越界取值改由 kernel 四条件掩码置 `0` 兜底。
- 空 Tensor：允许。**多轴同时为 0 的优先级（本代契约）**：先判 `taskNum = S×B` 是否为 0——`S = 0` 或 `B = 0` 时输出为空张量（shape 与 `indices.shape` 一致），**无论 `H` / `C` 是否为 0**；只有 `S > 0 ∧ B > 0` 时，`H = 0`（内积为空）或 `C = 0`（所有 task 非法）才给出**整张量精确 `0`**。该契约与实现一致：tiling 的空 Tensor 短路 `HandleEmptyTensor()` 只判 `taskNum == 0`（此时不写任何数据、`taskNumPerBlock = 0`、`SetBlockDim(1)`），而 `H = 0` / `C = 0` 的其他组合进入正常通路，由 kernel 的 `hidden <= 0` 短路与四条件掩码写出精确 `0`。
- 取值域：`indices ≥ B` 与 `locations < 0` 会越过 `y_grad` 张量范围。本代实现**保证绝不发生越界访存**（kernel 的四条件掩码 `(e ≥ 0) ∧ (e < B) ∧ (p ≥ 0) ∧ (p < C)` 不成立时 `continue`、不发起任何 DMA，槽位保持精确 `0`），但**校验层的前置拒收只在取值可被可信读出时生效**，判据是「张量放置」与「数据指针」两个条件**同时成立**：
  - **直读拒收（仅一种情形）**：框架以 host 张量（`kOnHost` / `kFollowing`）提供取值**且该张量带非空数据指针**时（如常量折叠等由框架把取值物化到 host 的通路），tiling 直读取值，命中 `indices ≥ B` 或 `locations < 0` 即拒收、返回 `ge::GRAPH_FAILED`，以 tiling 失败对应的 GE 错误码上报。
  - **放行（记 INFO）：host 放置但无数据指针**——张量为 `kOnHost` / `kFollowing`，但取值指针为空（`GetData<int32_t>() == nullptr`）⇒ 本层不可判，放行；`indices ≥ B` / `locations < 0` 由 kernel 掩码置 `0`。
  - **放行（记 INFO）：设备张量**——tiling 收到的描述符不带任何非连续信息（无 stride / offset、storage 维等于视图维），按线性缓冲读值无法证明读到的是逻辑元素序列，因此**不读值、不拒收**，同样由 kernel 掩码置 `0`。
  即：**越界访存永不发生；`indices ≥ B` / `locations < 0` 的「前置拦截」在上述两类放行路径上改为「由 kernel 掩码置 0」**。`indices < 0` 与 `locations ≥ C` 在任何通路上都不由校验层拒绝，对应 task 输出精确 `0`（整块 task 非法时同样写回 `0`，不保留旧值）。
- 维数上界：`S` / `H` / `B` / `C` 由 TilingData 的 int32 字段承载，`S×B`（`taskNum`）不得超过 `INT32_MAX - 256`（分核粒度取 256 对齐，故预留该余量）；`S` / `B` / `H` / `C` 逐维不得超过 `INT32_MAX`（`C` 是槽位地址乘子 `slot = e×C + p`，截断会算出错误地址）。超界即返回失败，不静默截断、不降档。
- 隐藏维上界：`H` 整段驻留 UB 且不切分，超过 UB 容量上限时返回失败且不降档。上限由 platform 接口实时取 `ubSize` 计算：tiling 把每条 `H` 长 UB 路由的元素数向上对齐到 `max(blockSize, vectorSize) / sizeof(dtype)`、块级小路由各按自己的 block 对齐后求和，得到 `needBytes(H)`，取满足 `needBytes(H) ≤ ubSize` 的最大 `H`（本算子不调用 `ReserveLocalMemory()`，故预留量为 0）。以 `ubSize = 253952` B、UB block `32` B、向量寄存器宽 `256` B 代入时：float32 路径 `H ≤ 15808`，FLOAT16 / BFLOAT16 路径 `H ≤ 12608`；该上限随 `ubSize` / `blockSize` / `vectorSize` 变化，不是固定常量。
- 特殊值：`x` 或 `y_grad` 含 NaN 时受影响 token 行的 task 输出为 NaN；含 `+inf` 时按 IEEE-754 逐元素传播——内积由 `x_h × y_grad_h` 逐项相乘再相加，故**同号且不出现 `0 × inf`** 时结果为 `+inf`，出现 `0 × inf`（某一 `h` 上 `x` 或 `y_grad` 为 0）或异号 `inf` 相消时为 `NaN`；浮点溢出不做饱和（无 clamp / 饱和路径）；FLOAT16 路径回写超过 65504 的输出为 `±inf`（`CAST_ROUND`，不饱和）。
- 设备：仅在 Ascend 950PR/Ascend 950DT（`arch35` = `dav-3510`）上注册与支持。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_moe_tutel_combine_gates](examples/test_geir_moe_tutel_combine_gates.cpp) | 通过 [算子IR](op_graph/moe_tutel_combine_gates_proto.h) 的 `REG_OP(MoeTutelCombineGates)` 构图方式调用；本算子无属性。样例覆盖 BFLOAT16 / FLOAT16 / FLOAT 三个 dtype 组合，并与 CPU 参考实现逐元素比对。 |

> 本算子在本仓库内只提供 GE 图模式这一种调用方式（aclnn 单算子调用方式已删除），无 PyTorch / TensorFlow 适配通路。

## 测试说明

- 单元测试（host 侧 tiling / infershape）：`moe/moe_tutel_combine_gates/tests/ut/op_host/`
  ```bash
  bash build.sh --ophost_test --soc=ascend950 --ops=moe_tutel_combine_gates
  ```
- ST（Kernel 精度，TTK 框架）：用例与 TestSpec 见 [`tests/assets/README.md`](tests/assets/README.md)
  ```bash
  bash scripts/ci/ops_st_test.sh --soc_version=ascend950 --ops=moe_tutel_combine_gates --test_type=kernel
  ```
- GE 图模式端到端样例：`bash build.sh --run_example moe_tutel_combine_gates graph --soc=ascend950`
