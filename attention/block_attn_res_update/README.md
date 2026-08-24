# BlockAttnResUpdate

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

产品支持表中的产品名称与代码、构建和测试所使用的内部标识对应如下：

| 标识 | 使用位置 | 含义 |
| :--- | :--- | :--- |
| `Ascend 950PR/Ascend 950DT` | README | 对外产品名称，也是本算子的产品支持范围。 |
| `ascend950` | OpDef及构建配置 | SoC注册键；仓库构建脚本将其映射为编译目标`Ascend950PR_9599`。 |
| `Ascend950PR_9599` | 编译配置及Kernel UT | 编译目标标识。 |
| `Ascend950` | Host/Kernel UT | 测试框架使用的SoC系列标识，对应架构编号`3510`。 |
| `3510`、`NpuArch::DAV_3510` | 测试框架及Host Tiling | Tiling选择使用的架构标识。 |

上述内部标识仅用于注册、构建、Tiling选择和测试，不代表扩大产品支持范围；本算子仅支持上表列出的Ascend 950PR和Ascend 950DT，不应理解为支持所有950型号。

## 功能说明

- 算子功能：完成BlockAttnRes第二阶段更新。算子首先使用增量delta更新局部注意力结果partial_block，再计算更新结果与pseudo_query的归一化相关分数，并结合已有在线Softmax状态numerator、logit_max和exp_sum生成注意力输出h。

- 计算公式：

  对于每个token $t$，首先更新局部注意力结果：

  $$
  p_t = partial\_block_t + \operatorname{float}(delta_t)
  $$

  计算更新结果的RMS和归一化相关分数：

  $$
  rms_t = \sqrt{\frac{1}{D}\sum_{j=0}^{D-1}p_{t,j}^{2} + eps}
  $$

  $$
  score_t = \frac{\sum_{j=0}^{D-1}p_{t,j} \cdot pseudo\_query_j}{rms_t}
  $$

  使用在线Softmax方式融合已有状态和当前结果：

  $$
  max_t = \max(logit\_max_t, score_t)
  $$

  $$
  alpha_t = \exp(logit\_max_t-max_t), \qquad
  beta_t = \exp(score_t-max_t)
  $$

  $$
  denominator_t = exp\_sum_t \cdot alpha_t + beta_t
  $$

  最终输出为：

  $$
  partial\_block\_out_t = p_t
  $$

  $$
  h_t = \operatorname{bfloat16}\left(
  numerator_t \cdot \frac{alpha_t}{denominator_t} +
  p_t \cdot \frac{beta_t}{denominator_t}\right)
  $$

  其中，$T$表示token数，$D$表示隐藏维度，$t \in [0,T)$，$j \in [0,D)$。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| partial_block | 输入 | 待更新的局部注意力结果，公式中的partial_block，shape为$(T, D)$；与同名输出复用存储。 | FLOAT | ND |
| delta | 输入 | 局部注意力结果的增量，公式中的delta，shape为$(T, D)$。 | BFLOAT16 | ND |
| pseudo_query | 输入 | 用于计算归一化相关分数的伪Query向量，公式中的pseudo_query，shape为$(D)$。 | FLOAT | ND |
| numerator | 输入 | 已有在线Softmax状态对应的分子，公式中的numerator，shape为$(T, D)$。 | FLOAT | ND |
| logit_max | 输入 | 已有在线Softmax状态中每个token的最大值，公式中的logit_max，shape为$(T)$。 | FLOAT | ND |
| exp_sum | 输入 | 已有在线Softmax状态中每个token的指数和，公式中的exp_sum，shape为$(T)$。 | FLOAT | ND |
| partial_block | 输出 | 更新后的局部注意力结果，公式中的partial_block_out，shape为$(T, D)$；与同名输入构成引用关系。 | FLOAT | ND |
| h | 输出 | 融合更新后的局部结果与已有在线Softmax状态得到的注意力结果，公式中的h，shape为$(T, D)$。 | BFLOAT16 | ND |
| eps | 可选属性 | RMS计算中的稳定项，必须为有限值且大于0，默认值为$10^{-6}$。 | FLOAT | - |

## 约束说明

- $T$和$D$必须是已知正整数，且$D$的取值范围为$[1, 8192]$。
- partial_block、delta、numerator及输出partial_block、h的shape必须相同，均为$(T, D)$；pseudo_query、logit_max和exp_sum的shape必须分别为$(D)$、$(T)$和$(T)$。
- $T \times D$不能超出有符号64位Kernel GM元素偏移的表示范围；按核切分后的$T$大小不能超出uint32_t的表示范围。
- 一行完整的$D$维数据必须能够以双缓冲方式放入统一缓冲区（Unified Buffer，UB），否则Tiling失败。
- partial_block为原地更新参数。框架调用时由引用关系保证输入、输出复用存储；Kernel从partial_block输入地址读取，并通过对应输出ABI地址写回，不依赖两个ABI地址相同。
- 当前Host Tiling申请的workspace大小为0，Kernel不访问workspace地址。
- RMS计算使用FP32的sqrt和除法，不进行牛顿迭代，也不对输入Tensor中的零值、极值、NaN或Inf做额外处理。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn接口 | - | 接口预留，当前版本尚未提供aclnnBlockAttnResUpdate接口及调用样例。 |
| TTK | [BlockAttnResUpdate Kernel测试](./tests/st/arch35/ttk_kernel_block_attn_res_update_st.csv) | 通过算子自带的TTK配置执行Kernel测试。 |
