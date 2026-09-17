# aclnnMoeTutelCombineGates

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_tutel_combine_gates)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：计算 Tutel MoE 门控权重的反向梯度。给定 token 的隐藏态 `x`、dispatch 缓冲的回传梯度 `y_grad` 以及每个 (batch, sample) 任务的路由下标 `indices` / `locations`，按路由下标取出 `y_grad` 的槽位行，与该任务的 token 行做 hidden 维内积，得到逐任务的门控梯度 `gates_grad`。本接口为单算子（aclnn）接口，与图模式接口 `MoeTutelCombineGates`（`op_graph/moe_tutel_combine_gates_proto.h`）为同一算子的两种调用方式，本算子无对标 ATen 接口。

- 计算公式：

  - 公共形态：逐 task 的 hidden 维内积。

    $$
    gates\_grad[b, s] = \sum_{h=0}^{H-1} x[s, h] \times y\_grad[e \times C + p,\ h],\quad e = indices[b, s],\ p = locations[b, s]
    $$

  - 非法路由：`e < 0`、`e \ge B`、`p < 0`、`p \ge C` 四类取值对应的 task 结果为精确 `0`（不参与 `y_grad` 读，不发起越界访存）。其中 `e \ge B` 与 `p < 0` 会越过 `y_grad` 张量范围，由校验层在取值**可被可信读出**时前置拒收——**仅当**框架以 host 张量（`kOnHost` / `kFollowing`）提供取值**且该张量带非空数据指针**时命中即拒收（本接口为 aclnn 通路，第一段接口上抛 `561002`）；host 放置但取不到数据指针、设备张量通路（本接口的常规调用：tiling 拿不到 stride / 非连续描述、线性读值不可信）下都不读值、不拒收，由算子按上式输出 `0`（分层口径见[约束说明](#约束说明)）；`e < 0` 与 `p \ge C` 在任何通路上都不被拒绝，由算子按上式输出 `0`。

- 公式变量说明（shape 契约与 `x` / `y_grad` / `indices` / `locations` 的绑定关系）：

  | 符号 | 含义 | 绑定关系 |
  | --- | --- | --- |
  | `S` | samples，按 token 的样本数 | `S = x.shape[0] = indices.shape[1] = locations.shape[1]` |
  | `H` | hidden，规约轴（固定为末维，不可配置） | `H = x.shape[1] = y_grad.shape[2]` |
  | `B` | 路由分支维（batch） | `B = y_grad.shape[0] = indices.shape[0] = locations.shape[0]` |
  | `C` | capacity，专家容量 | `C = y_grad.shape[1]`，由 shape 隐式给出，调用方不能单独配置 |
  | `e` | task 的专家下标 | `e = indices[b, s]` |
  | `p` | task 在专家内的槽位 | `p = locations[b, s]` |

- task 展平下标 `t ∈ [0, S×B)`：token 行 = `t % S`，槽位行 = `indices[t] × C + locations[t]`。`indices` / `locations` / `gates_grad` 的元素数均为 `S×B`。

- 计算精度：`x` / `yGrad` 为 bfloat16 或 float16 时，先转换到 float32，逐元素乘法在 float32 内完成；随后做一次**精确的 2 的幂归一化**（取全部乘积的 `max|·|` 得到 `2^k`，整体乘 `2^-k` 后规约、写回前再精确乘回 `2^k`，数学上与直接求和等价，用于拓宽累加的指数窗口），hidden 维规约实现为 fp32 **补偿归约**（TwoSum 补偿累加，结果为精确乘积之和的正确舍入值），最后按 `CAST_ROUND` 回写到输出 dtype。float32 输入不做归一化，直接以 float32 完成乘法（含 TwoProduct 精确乘积误差通道）与补偿归约。累加精度固定为 float32，禁止降为低精度累加。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMoeTutelCombineGatesGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTutelCombineGates”接口执行计算。

```cpp
aclnnStatus aclnnMoeTutelCombineGatesGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *yGrad,
    const aclTensor *indices,
    const aclTensor *locations,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnMoeTutelCombineGates(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnMoeTutelCombineGatesGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 280px">
  <col style="width: 340px">
  <col style="width: 200px">
  <col style="width: 80px">
  <col style="width: 130px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（const aclTensor *）</td>
      <td>输入</td>
      <td>按 token 的隐藏态，公式中的 x。</td>
      <td><li>支持空Tensor（S = 0 时输出为空张量）。</li><li>末维必须与yGrad的末维相等（H）。</li><li>必须为2维。</li></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[S, H]（rank 2）</td>
      <td>√（自动连续化）</td>
    </tr>
    <tr>
      <td>yGrad（const aclTensor *）</td>
      <td>输入</td>
      <td>dispatch 缓冲回传梯度，公式中的 y_grad；按槽位 (e, p) 取值。</td>
      <td><li>支持空Tensor（B = 0 或 H = 0 时输出为空张量或全 0）。</li><li>必须为3维，末维与x的末维相等。</li><li>数据类型必须与x一致。</li></td>
      <td>与x保持一致</td>
      <td>ND</td>
      <td>[B, C, H]（rank 3）</td>
      <td>√（自动连续化）</td>
    </tr>
    <tr>
      <td>indices（const aclTensor *）</td>
      <td>输入</td>
      <td>路由专家下标（公式中的 e），逐 task 一一对应，按平坦的 S×B 数组消费。</td>
      <td><li>支持空Tensor（元素数为 0 时输出为空张量）。</li><li>合法取值域为 [0, B)：大于或等于 B 的取值仅在框架以 host 张量提供取值<strong>且带数据指针</strong>时由校验层拒绝（本接口第一段返回 `561002`），host 放置但无数据指针与设备张量通路下不拒收、对应 task 输出精确 0（分层口径见[约束说明](#约束说明)）；负值不被拒绝（对应 task 输出 0）。</li><li>元素数必须为 S×B。</li></td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B, S]（rank 2，元素数 S×B）</td>
      <td>√（输入侧自动连续化；该路参与取值域校验，设备张量通路下不读值、不拒收，见约束说明）</td>
    </tr>
    <tr>
      <td>locations（const aclTensor *）</td>
      <td>输入</td>
      <td>专家内槽位（公式中的 p），与indices同形，按平坦的 S×B 数组消费。</td>
      <td><li>支持空Tensor（元素数为 0 时输出为空张量）。</li><li>合法取值域为 [0, C)：负值仅在框架以 host 张量提供取值<strong>且带数据指针</strong>时由校验层拒绝（本接口第一段返回 `561002`），host 放置但无数据指针与设备张量通路下不拒收、对应 task 输出精确 0（分层口径见[约束说明](#约束说明)）；大于或等于 C 的取值不被拒绝，对应 task 输出 0。</li><li>元素数必须为 S×B。</li></td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B, S]（rank 2，元素数 S×B）</td>
      <td>√（输入侧自动连续化；该路参与取值域校验，设备张量通路下不读值、不拒收，见约束说明）</td>
    </tr>
    <tr>
      <td>out（const aclTensor *）</td>
      <td>输出</td>
      <td>门控权重梯度，对应图模式接口的输出 gates_grad。</td>
      <td><li>支持空Tensor（S = 0 或 B = 0 时输出为空张量，无论H/C是否为0）。</li><li>S与B均非0时，H=0或C=0给出<strong>整张量精确0</strong>（不是空张量）。</li><li>shape必须等于indices的shape。</li><li>数据类型必须与x一致。</li><li>必须为连续Tensor，见[约束说明](#约束说明)的「非连续 Tensor」条。</li></td>
      <td>与x保持一致</td>
      <td>ND</td>
      <td>[B, S]</td>
      <td>×（须连续；本版本 op_build aclnn 生成器不做输出侧连续化，非常规 stride 的 out 不受支持）</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t *）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。<b>四个输入全部连续时返回0</b>（算子自身不使用workspace：H 不跨核切分、无跨核规约，tiling 的唯一 workspace 槽恒写 0）；<b>任一输入非连续时返回大于0的值</b>——该值来自桩为非常规 stride 输入插入的连续化视图拷贝（框架侧开销，非算子申请），第二段接口须在其中预留该空间。两种情况的口径详见<a href="#约束说明">约束说明</a>的「workspaceSize」条。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor **）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- 参数表「非连续Tensor」列取值：`√` = 可传非连续 Tensor（由生成桩的输入侧连续化视图拷贝自动连续化，逻辑结果与连续输入逐位一致）；`×` = **必须**为连续 Tensor（本版本的 op_build aclnn 生成器不为纯输出做连续化，非常规 stride 的输出**不受支持**）。各张量的具体口径与限制见[约束说明](#约束说明)的「非连续 Tensor」条。

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    > **校验分层说明**：本接口由 op_build 生成的 aclnn 桩（`aclnn_moe_tutel_combine_gates.cpp`）承载，**桩内只有空指针判空与 dtype / format 支持列表登记两类分支**（支持列表的比对由框架侧在 `NnopbaseRunForWorkspace` 阶段完成），不含 rank / shape / 元素数 / 取值域 / UB 容量校验（非连续输入时另会插入一次连续化视图的 `aclnnContiguousGetWorkspaceSize` 调用）。因此本表所列的 shape 契约、dtype 组合、取值域与 UB 容量类异常**由算子 tiling 层承担**：`op_host/moe_tutel_combine_gates_tiling_arch35.cpp` 的各校验出口以 CANN 9.2.0 的 `OP_LOGE_FOR_INVALID_*_WITH_REASON` 结构化宏上报，并返回 `ge::GRAPH_FAILED`；在 aclnn 通路上该失败经 `NnopbaseRunForWorkspace` 上抛为第一段接口的 `561002`。其中取值域类里 `e ≥ B` / `p < 0` 的前置拒收只在框架以 host 张量提供取值**且带数据指针**时生效，host 无数据指针与设备张量通路下 tiling 不读值、不拒收，改由 kernel 四条件掩码置 0——见[约束说明](#约束说明)。图模式（GE）通路不经过该桩，同类失败以 GE / tiling 错误码上报。

    <table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 280px">
    <col style="width: 120px">
    <col style="width: 600px">
    </colgroup>
    <thead>
      <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>x、yGrad、indices、locations、out或executor中存在空指针。</td>
    </tr>
    <tr>
      <td> ACLNN_ERR_PARAM_INVALID </td>
      <td> 161002 </td>
      <td>输入或输出的数据类型、数据格式组合不在支持范围内（如indices/locations非INT32、data format非ND、x与yGrad的浮点类型不一致、输出dtype与x不一致）。该判定由框架侧比对算子登记的支持列表（`NnopbaseAddSupportList`）完成。</td>
    </tr>
    <tr>
      <td> ACLNN_ERR_INNER_TILING_ERROR </td>
      <td> 561002 </td>
      <td>shape契约违反：x/yGrad/indices/locations的rank不符、S/H/B绑定不一致、indices或locations的元素数不等于S×B、声明的out shape与indices.shape不一致、H超过UB容量上限、以及<strong>框架以host张量提供取值且该张量带数据指针时</strong>读到的 indices ≥ B / locations为负的取值域越界（host 放置但无数据指针、以及设备张量通路下该校验不读值、不拒收，见[约束说明](#约束说明)）；以及平台参数（ubSize / coreNum / blockSize / cacheLineSize / vectorSize等）取值为0一类的tiling侧前置失败。此时第一段接口返回失败。该码由本算子的tiling阶段（op_host/moe_tutel_combine_gates_tiling_arch35.cpp的各校验出口）产生，经NnopbaseRunForWorkspace上抛后由第一段接口对外返回；形状与dtype的推理期校验（op_host/moe_tutel_combine_gates_infershape.cpp）同样以 OP_LOGE_FOR_INVALID_*_WITH_REASON 上报并返回 ge::GRAPH_FAILED。</td>
    </tr>
    <tr>
      <td> 其它框架码 </td>
      <td> 依CANN版本 </td>
      <td>自动生成桩调用NnopbaseCreateExecutorSpace / NnopbaseAddInput / NnopbaseAddOutput / NnopbaseAddTilingId / NnopbaseAddParamName / NnopbaseAddSupportList等框架接口失败时，其返回码原样上抛，具体取值取决于CANN版本。</td>
    </tr>
    </tbody></table>

## aclnnMoeTutelCombineGates

- **参数说明**

    <table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
    </colgroup>
    <thead>
      <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。workspaceSize为0时可不申请。</td></tr>
      <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMoeTutelCombineGatesGetWorkspaceSize获取。</td></tr>
      <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
      <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody>
    </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

> **范围声明**：本节全部约束描述的是**本仓库 Ascend 950（`arch35`）实现**的实际行为。其中 rank / shape 契约、元素数、dtype / format、int32 上界、取值域与 UB 容量等校验，以及非法路由置 `0`、空 Tensor 语义，均为该实现的行为口径。本接口与图模式接口 `MoeTutelCombineGates`（`op_graph/moe_tutel_combine_gates_proto.h`）是同一算子的两种调用方式，两者的 shape / dtype 契约一致；产品级支持情况见上方「产品支持情况」。

- 确定性说明：aclnnMoeTutelCombineGates默认确定性实现。规约轴 H 不跨核切分（分核只沿 task 轴），同一设备、同一输入多次执行结果 bitwise 一致。

- 可用性约束：

  - 仅支持 <term>Ascend 950PR/Ascend 950DT</term>：算子定义只注册 `ascend950` 这一份 AICore 配置（`op_host/moe_tutel_combine_gates_def.cpp` 的 `this->AICore().AddConfig("ascend950", aiCoreConfig)`）。

- 输入形态约束：

  - rank 固定：`x` 必须 rank 2、`yGrad` 必须 rank 3、`indices` / `locations` 必须 rank 2（`out` 为 rank 2）；rank 0/1 或其它 rank 均被拒绝。
  - shape 绑定：`H = x.shape[1] = yGrad.shape[2]`、`B = yGrad.shape[0] = indices.shape[0]`、`S = x.shape[0] = indices.shape[1] = locations.shape[1]`；`indices` / `locations` / `out` 的元素数均为 `S×B`；`out` 的 shape 必须等于 `indices.shape`。
  - 维度必须已解析：本算子声明动态shape / 动态rank支持，InferShape允许**未解析维（负值）**透传，但tiling的数值范围校验对未解析维**直接拒收**（以 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` 上报并返回 `ge::GRAPH_FAILED`，aclnn通路下经`NnopbaseRunForWorkspace`上抛为第一段接口的`561002`）——未解析维若进入kernel会以约4.29e9的`uint32`计数进入向量指令与`DataCopyPad`的`blockLen`（合法域 `[1, 2097151]`）。故四个维度必须均已解析，且逐维取值不超过`INT32_MAX`。
  - 数据类型：`x`、`yGrad`、`out` 三者 dtype 一致，取 FLOAT、FLOAT16、BFLOAT16 之一；`indices` / `locations` 固定为 INT32。
  - 数据格式：全部输入输出仅支持 ND，不支持 NCHW / NHWC / FRACTAL_NZ 等格式。
  - 非连续 Tensor：算子定义对四路输入与输出都声明了 `.AutoContiguous()`，但本 CANN 版本（9.2.0）的 op_build 生成器**只对输入侧生效**。
    - **输入** `x` / `yGrad` / `indices` / `locations`：可传非连续 Tensor。桩会为非常规 stride 的输入插入一次 `aclnnContiguous` 连续化视图拷贝，**逻辑结果与连续输入逐位一致**。代价见下方「`workspaceSize`」条（该拷贝需要第一段接口多返回一段 workspace）。去掉输入上的 `.AutoContiguous()` 会真实改变生成桩（整段连续化通路消失），故该声明必要且有效。
    - **输出** `out`（`gates_grad`）：**必须连续**。本版本的 op_build aclnn 生成器**不做**输出侧连续化——生成器里唯一的输出侧连续化 / 回拷机制（`NnopbaseGetRefUnContiguousTensors` / `aclnnViewCopy*`）只为「输出参数名与某输入同名」的 **in-place（Ref）** 算子发射，本算子不属该类（`gates_grad` 与任何输入都不同名，shape 也与 `x` / `yGrad` 都不同，且不得改成该类），因此对纯（非 `Ref`）输出不会生成任何连续化代码；输出上的 `.AutoContiguous()` 对本版本生成桩是**惰性**的（保留该调用仅为与四输入的声明形式一致，**不代表输出侧非连续已获支持**）。kernel / tiling 层也无法感知输出 stride（aclnn 通路喂给 tiling 的输出描述符不带 stride / offset 信息，storage 维等于视图维）。因此调用方**必须传入连续的 `out`**：传入非常规 stride 的 `out` 属**不支持用法**，算子不会为它做连续化，kernel 按连续语义写回，结果不可预期。
    - **`indices` / `locations` 的取值域校验**：该校验只在框架以 host 张量（`kOnHost` / `kFollowing`）提供取值、**且该张量带非空数据指针**时生效（此时读到的就是逻辑元素序列本身，判据可信）；host 放置但指针为空时记 INFO 后放行（`GetData<int32_t>() == nullptr` 分支）。设备张量通路（本接口的常规调用）下 tiling 拿不到任何非连续信息（描述符为 `kTensorV1`、无 stride / offset、storage 维等于视图维），按线性缓冲读 `S×B` 个 int32 无法证明就是逻辑元素序列，故**不读值、不拒收**（记 INFO 后放行）。两类放行路径上的越界取值改由 kernel 的四条件掩码置 `0` 兜底，仍不发生越界访存；若按线性缓冲读值判拒，会**误拒合法的非连续视图**，而误拒合法输入比漏检更严重，故取「宁可放行、不误拒」。
  - 广播：不支持 broadcast，各输入 shape 必须严格满足上述绑定关系。
  - 空 Tensor：允许。**多轴同时为 0 的优先级（本代契约）**：先判 `taskNum = S×B` 是否为 0——`S = 0` 或 `B = 0` 时输出为空张量（shape 与 `indices.shape` 一致），**无论 `H` / `C` 是否为 0**；只有 `S > 0 ∧ B > 0` 时，`H = 0`（内积为空）或 `C = 0`（所有 task 非法）才给出**整张量精确 `0`**。该契约与实现一致：tiling 的空 Tensor 短路 `HandleEmptyTensor()` 只判 `taskNum == 0`（此时不写任何数据、`taskNumPerBlock = 0`、`SetBlockDim(1)`），而 `H = 0` / `C = 0` 的其他组合进入正常通路，由 kernel 的 `hidden <= 0` 短路与四条件掩码写出精确 `0`。
  - 维数上界：`S` / `H` / `B` / `C` 由 TilingData 的 int32 字段承载，且 `S×B`（`taskNum`）需不超过 `INT32_MAX - 256`（分核粒度取 256 对齐，故预留该余量）；`S` / `B` / `H` / `C` 逐维不得超过 `INT32_MAX`（`C` 是槽位地址乘子 `slot = e×C + p`，截断会算出错误地址）。超界返回失败，不静默截断。
  - 隐藏维上界：`H` 不切分、整段驻留 UB，故 `H` 受 UB 容量限制，超过上限时返回失败且不降档。上限由 platform 接口实时取 `ubSize` 计算：tiling 把每条 `H` 长 UB 路由的元素数向上对齐到 `max(blockSize, vectorSize) / sizeof(dtype)`、块级小路由各按自己的 block 对齐后求和，得到 `needBytes(H)`，取满足 `needBytes(H) ≤ ubSize` 的最大 `H`（本算子不调用 `ReserveLocalMemory()`，故预留量为 0）。以 `ubSize = 253952` B、UB block `32` B、向量寄存器宽 `256` B 代入时：float32 路径 `H ≤ 15808`，FLOAT16 / BFLOAT16 路径 `H ≤ 12608`；该上限随实际 `ubSize`（以及 UB block / 向量寄存器宽度）变化，不是固定常量。

- 取值域约束（`e = indices`，`p = locations`，`C = yGrad.shape[1]`）：

  - `e ≥ B` 与 `p < 0` 会越过 `yGrad` 张量范围，而本代实现的 kernel 四条件掩码 `(e ≥ 0) ∧ (e < B) ∧ (p ≥ 0) ∧ (p < C)` 保证**任何情况下都不发起越界访存**（掩码不成立时 `continue`，槽位保持精确 `0`）。校验层的前置拒收只在取值**可被可信读出**时生效，判据是「张量放置」与「数据指针」两个条件**同时成立**：
    - **直读拒收（仅一种情形）**：框架以 host 张量（`kOnHost` / `kFollowing`）提供取值**且该张量带非空数据指针**时（如常量折叠等由框架把取值物化到 host 的通路），tiling 直读取值，命中 `e ≥ B` 或 `p < 0` 即拒收（以 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` 上报并返回 `ge::GRAPH_FAILED`）；本接口为 aclnn 通路，该失败经 `NnopbaseRunForWorkspace` 上抛为**第一段接口的 `561002`**（该码为 aclnn 专属，非 aclnn 通路不出现）。
    - **放行（记 INFO）：host 放置但无数据指针**：`GetData<int32_t>() == nullptr` ⇒ 本层不可判而放行，`e ≥ B` / `p < 0` 与下面两类一样由 kernel 掩码置 `0`。
    - **放行（记 INFO）：设备张量**（本接口的常规调用）：tiling 收到的描述符不带任何非连续信息（`kTensorV1`、无 stride / offset、storage 维等于视图维），线性读值不可信，故记 INFO 后放行。
    ⇒ **越界访存永不发生；`e ≥ B` / `p < 0` 的「前置拦截」在上面两类放行路径上改为「由 kernel 掩码置 0」**。
  - `e < 0` 与 `p ≥ C` **不**被拒绝：对应 task 的结果为精确 `0`（整块 task 非法时同样写回 `0`，不保留旧值）。

- 特殊值：`x` 或 `yGrad` 含 NaN 时，受影响 token 行对应的 task 输出为 NaN；含 `+inf` 时按 IEEE-754 逐元素传播——内积由 `x_h × yGrad_h` 逐项相乘再相加，故**同号且不出现 `0 × inf`** 时对应内积为 `+inf`，出现 `0 × inf`（某一 `h` 上 `x` 或 `yGrad` 为 0）或异号 `inf` 相消时为 `NaN`；浮点溢出不做饱和（无 clamp / 饱和路径）。FLOAT16 路径回写时超过 65504 的输出为 `±inf`（按 `CAST_ROUND` 舍入，不饱和）。

- 本算子无属性，`C`（capacity）由 `yGrad.shape[1]` 隐式给出。

- `workspaceSize`：**不是恒为 0**，分两种情况，且两者的来源必须分清：

  - **四个输入全部连续（常规调用）：`workspaceSize = 0`**。这里为 0 的是**算子自身**的 workspace——设计与实现口径一致：`H` 不跨核切分、不存在跨核部分和，因此 tiling 的 workspace 槽恒写 `0`（无跨核部分和、无 buffer 中转）。**「无 workspace」指的是这件事，与下面的连续化 workspace 不是一回事。**
  - **任一输入非连续：`workspaceSize > 0`**。该值来自 aclnn 桩为非常规 stride 输入插入的**连续化视图拷贝**（`aclnnContiguousGetWorkspaceSize`），属框架侧开销，**不是算子申请的 workspace**；算子自有那一路仍为 `0`，桩把两者相加后对外返回。第二段接口要求在 `workspaceSize` 内为这段连续化预留空间。
  - 该值**随 CANN 版本与非连续张量的规模变化，不是固定常量**，调用方须按第一段接口返回值申请，不得硬编码。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)；仓库内的完整端到端用例见 `examples/test_aclnn_moe_tutel_combine_gates.cpp`（覆盖 BFLOAT16 / FLOAT16 / FLOAT 三个 dtype 组合，并与 CPU float64 参考实现逐元素比对）。

```Cpp
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_tutel_combine_gates.h"

namespace {

// 以连续的 shape/strides 创建 aclTensor（本示例为常规调用：四路输入与输出都取连续）；
// 非连续输入由 aclnn 桩自动连续化（此时第一段接口的 workspaceSize > 0）；
// 输出必须连续——本版本的 op_build 生成桩不做输出侧连续化，非常规 stride 的 out 不受支持（见约束说明）。
aclTensor *CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype, void *deviceAddr)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            shape[static_cast<size_t>(i) + 1] * strides[static_cast<size_t>(i) + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), deviceAddr);
}

void *MallocDevice(size_t bytes)
{
    void *addr = nullptr;
    aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    return addr;
}

}  // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    // 取 S = 4、H = 8、B = 2、C = 3；x = [S, H]、yGrad = [B, C, H]、
    // indices / locations / out = [B, S]（元素数 = S*B = 8）
    std::vector<int64_t> xShape = {4, 8};
    std::vector<int64_t> yGradShape = {2, 3, 8};
    std::vector<int64_t> taskShape = {2, 4};

    void *xAddr = MallocDevice(4 * 8 * sizeof(float));
    void *yGradAddr = MallocDevice(2 * 3 * 8 * sizeof(float));
    void *indicesAddr = MallocDevice(2 * 4 * sizeof(int32_t));
    void *locationsAddr = MallocDevice(2 * 4 * sizeof(int32_t));
    void *outAddr = MallocDevice(2 * 4 * sizeof(float));

    // 输入数据须由调用方写入 device 内存（此处略去 aclrtMemcpy 上传步骤）
    aclTensor *x = CreateTensor(xShape, ACL_FLOAT, xAddr);
    aclTensor *yGrad = CreateTensor(yGradShape, ACL_FLOAT, yGradAddr);
    aclTensor *indices = CreateTensor(taskShape, ACL_INT32, indicesAddr);
    aclTensor *locations = CreateTensor(taskShape, ACL_INT32, locationsAddr);
    aclTensor *out = CreateTensor(taskShape, ACL_FLOAT, outAddr);

    // 第一段接口：获取 workspace 大小与执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnMoeTutelCombineGatesGetWorkspaceSize(x, yGrad, indices, locations, out,
                                                               &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("aclnnMoeTutelCombineGatesGetWorkspaceSize failed. ERROR: %d\n", ret);
        return 1;
    }
    // 四个输入全连续时 workspaceSize 为 0（算子自身不使用 workspace）；
    // 非连续输入时为桩的连续化视图拷贝多返回一段，故按返回值申请
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = MallocDevice(workspaceSize);
    }

    // 第二段接口：执行计算
    ret = aclnnMoeTutelCombineGates(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("aclnnMoeTutelCombineGates failed. ERROR: %d\n", ret);
        return 1;
    }
    aclrtSynchronizeStream(stream);

    // 取回结果：gates_grad[b, s] = sum_h x[s, h] * yGrad[indices[b, s] * C + locations[b, s], h]
    std::vector<float> hostOut(8, 0.0f);
    aclrtMemcpy(hostOut.data(), hostOut.size() * sizeof(float), outAddr,
                hostOut.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < hostOut.size(); ++i) {
        printf("gates_grad[%zu] = %f\n", i, hostOut[i]);
    }

    aclDestroyTensor(x);
    aclDestroyTensor(yGrad);
    aclDestroyTensor(indices);
    aclDestroyTensor(locations);
    aclDestroyTensor(out);
    aclrtFree(xAddr);
    aclrtFree(yGradAddr);
    aclrtFree(indicesAddr);
    aclrtFree(locationsAddr);
    aclrtFree(outAddr);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
