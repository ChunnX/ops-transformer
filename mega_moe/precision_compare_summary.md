# SwiGLU-OAI 精度问题定位

## 1. 问题范围

本次只分析以下问题：

- Kernel：D:\hw_git\base_gitcode\ops-transformer\mc2\mega_moe\op_kernel\arch35\blaze\epilogue\gated_activation.h
- Golden：D:\moe_test\moe_test_develop\mega_moe\gen_input_and_golden.py
- 用例表：D:\linshi\minimax.xlsx
- 重点用例：miniMax_003、miniMax_004、miniMax_005、miniMax_006
- 现象：使用 swigluoai 时，部分卡精度 FAIL，但误差较小；改成普通 swiglu 后可以通过。

这不是对整个精度框架的总结，而是针对这组 SwiGLU-OAI 用例的定位结论。

## 2. 用例共同条件

四个用例的共同配置为：

| 参数 | 值 |
| --- | --- |
| bs | 16 |
| k | 4 |
| rank_per_dev | 8 |
| expert_per_rank | 17 |
| local_moe_expert_num | 16 |
| shared_expert_num_per_rank | 1 |
| moe_stage | decode |
| graph_type | 0 |
| quant_mode | 4 |
| quant_dtype | float8_e4m3fn |
| swiglu_quant_dtype | float8_e4m3fn |
| weight_dtype | float8_e4m3fn |
| clamp_limit | 7 |
| activation_type | swigluoai |
| alpha | 1.702 |
| beta | 1.0 |

用例之间的主要差异：

| 用例 | h | n | combine_quant_mode | A8W8_is_NZ | weight_is_tensor_list |
| --- | ---: | ---: | ---: | --- | --- |
| miniMax_003 | 7168 | 5120 | 0 | False | 1 |
| miniMax_004 | 6144 | 6144 | 4 | False | 1 |
| miniMax_005 | 7168 | 5120 | 0 | True | 1 |
| miniMax_006 | 6144 | 6144 | 4 | False | 0 |

因此，四个用例共同经过 OAI 激活，但权重布局、权重容器形式和 Combine 量化路径并不完全相同。

## 3. Kernel 的实际公式

gated_activation.h 中的 RunSwiGLUOai 对每个元素执行以下顺序：

~~~text
gate = first input
up   = second input

gate = min(gate, clamp_limit)
up   = min(max(up, -clamp_limit), clamp_limit)

up = up + beta
glu = gate / (1 + exp(-alpha * gate))
y = glu * up
~~~

对应源码行为：

- RunSwiGLUOai 的注释给出公式：(up + beta) * min(gate, clamp) * sigmoid(alpha * min(gate, clamp))。
- 先对 gate 做上限裁剪，再对 up 做对称裁剪。
- up + beta 在最终乘法之前执行。
- Kernel 使用 exp(-alpha * gate) 和除法实现 sigmoid，而不是直接调用高层 sigmoid API。
- 激活计算结果在写回前由 FP32 转为 BF16。
- 如果启用 TopK 权重预取，Kernel 会在 BF16 转换前乘以 TopK 权重；本组用例 topk_weights_type=0，不经过这个分支。

## 4. Golden 的实际公式

gen_input_and_golden.py 中的 swiglu_oai 为：

~~~python
gate, up = gmm_out.chunk(2, dim=-1)
if clamp_limit is not None:
    gate = torch.clamp(gate, max=clamp_limit)
    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
glu = gate * torch.sigmoid(gate * alpha)
return (up + beta) * glu
~~~

与 Kernel 对照：

| 步骤 | Kernel | Golden | 是否对齐 |
| --- | --- | --- | --- |
| 拆分顺序 | 第一部分 gate，第二部分 up | 第一部分 gate，第二部分 up | 是 |
| gate 裁剪 | min(gate, clamp) | clamp(max=clamp) | 是 |
| up 裁剪 | clamp(-clamp, clamp) | clamp(min=-clamp, max=clamp) | 是 |
| beta 位置 | up + beta 后参与乘法 | up + beta 后参与乘法 | 是 |
| alpha 位置 | sigmoid(alpha * gate) | sigmoid(gate * alpha) | 是 |
| 最终乘法 | glu * (up + beta) | (up + beta) * glu | 是 |

当前源码中没有发现 OAI 数学公式、Clamp 位置、alpha/beta 位置的明显错误。

## 5. 已经对齐的类型转换顺序

Golden 的 GMM1 输出在激活前执行：

~~~python
output1 = output1.to(torch.bfloat16).to(torch.float32)
~~~

这表示先模拟 Kernel 的 BF16 GMM 输出，再用 FP32 计算激活。

激活和 TopK 权重处理后，Golden 再执行：

~~~python
swiglu_out = swiglu_out.to(torch.bfloat16).to(torch.float32)
quant_output, quant_scale_output = mx_quantize(...)
~~~

这对应 Kernel 的：

~~~text
FP32 激活结果 -> BF16 写回 -> MX 量化
~~~

所以从当前 Golden 代码看，BF16 转换并不是完全遗漏；它已经分别放在 GMM1 输出进入激活之前，以及激活输出进入 MX 量化之前。

## 6. 最可疑的不一致点

### 6.1 sigmoid 的实现方式不同

Kernel 明确使用：

~~~text
gate / (1 + exp(-alpha * gate))
~~~

Golden 使用：

~~~python
gate * torch.sigmoid(gate * alpha)
~~~

两者数学上等价，但不保证逐 bit 等价。Kernel 使用 AscendC MicroAPI 的 Exp + Adds + Div，PyTorch 使用其自身的 sigmoid 实现。对于 float8_e4m3fn 的 MX 量化，激活结果只要发生很小的低位差异，就可能改变：

- per-token 或 per-group 的最大绝对值；
- MX scale；
- 量化后的 FP8 值；
- 后续 GMM2 的输入。

这最符合“误差很小但部分卡 FAIL”的现象。

### 6.2 激活后 BF16 舍入正好落在 MX 量化边界

OAI 公式相较普通 SwiGLU 多了两处影响数值范围的因素：

~~~text
gate * sigmoid(alpha * gate)
up + beta
~~~

本组参数为 alpha=1.702、beta=1.0、clamp=7。它会改变激活输出的符号、幅值和分布。尤其在 gate 为负数或接近零时，alpha 会改变 sigmoid 的过渡位置；随后 up + beta 又可能把结果推到 MX E4M3 的舍入边界附近。

因此，普通 SwiGLU 通过、SwiGLU-OAI 失败，并不能直接说明 OAI 公式错误，更可能说明 OAI 输出更容易暴露 Kernel 与 Golden 在 Exp/Sigmoid、BF16 舍入或 MX scale 计算上的细微差异。

### 6.3 小 BS 和分卡会放大统计误差

这些用例 bs=16，并且是 decode 场景。每张卡实际参与计算的样本少，某一张卡只要出现少量边界值，就可能让该卡的最大误差或相对误差指标变差。不同卡的输入、专家路由和权重布局不同，所以会出现“部分卡 FAIL、其他卡 PASS”。

这也解释了为什么大值误差看起来不大，但小 BS 更容易触发失败：不是总体误差突然变大，而是有效样本少，单个量化边界点在统计指标中的权重更高。

## 7. 对 003/004/005/006 的判断

从表格条件看：

1. 003 和 005 具有相同的 h/n、Combine 量化模式和 TensorList 配置，只改变 A8W8_is_NZ。
2. 004 和 006 具有相同的 h/n、Combine 量化模式和非 NZ 配置，只改变 weight_is_tensor_list。
3. 四个用例仍然共享相同的 OAI 激活参数和小 BS decode 条件。
4. 如果 003/004/005/006 都出现相似的轻微 FAIL，问题更像是 OAI 激活输出进入 MX 量化后的共同数值路径，而不是某一个单独的 Combine 配置。
5. 如果 FAIL 只集中在 005，应优先排查 NZ 权重布局转置、GMM1 输出值以及 NZ 路径的输入精度；如果 FAIL 同时覆盖 003/004/006，则优先排查 OAI 激活的 Exp/Div 与 BF16/MX 量化边界。

## 8. 结论

基于当前 Kernel 和 Golden 源码，已经确认以下内容一致：

- gate/up 拆分顺序一致；
- clamp_limit=7 的位置一致，均在激活计算前；
- gate 只做上限裁剪，up 做对称裁剪；
- alpha=1.702、beta=1.0 的作用位置一致；
- up + beta 在最终乘法之前执行；
- 激活结果进入 MX 量化前都经过 BF16 舍入路径。

当前最可疑的不是 OAI 公式本身，而是：

~~~text
Kernel 的 Exp + Div sigmoid
与
Golden 的 torch.sigmoid
~~~

以及两者后面的 BF16 舍入和 MX E4M3 scale/量化边界。这个差异可以解释“普通 SwiGLU 通过、SwiGLU-OAI 轻微 FAIL、且只有部分卡失败”。

## 9. 建议的最小定位顺序

按以下顺序比较同一个 rank、同一个专家、同一个 token：

1. GMM1 输出 gate/up，确认进入激活前的输入一致。
2. Clamp 后的 gate/up，确认边界和裁剪顺序一致。
3. 激活前的 FP32 OAI 结果，Golden 改用 gate / (1 + exp(-alpha * gate)) 对齐 Kernel 的实现形式。
4. 激活结果转 BF16 后的值。
5. MX quant 的 scale 和量化输出。
6. GMM2 输出及最终 Combine 结果。

判断规则：

- 第 1 步就不同：排查 GMM1、权重布局或 NZ/ND 路径；
- 第 3 步开始不同：排查 sigmoid 实现、alpha/beta 或 Clamp；
- 第 4 步相同、第 5 步不同：排查 MX scale/量化实现；
- 第 5 步相同、最终结果不同：再排查 GMM2、Combine 或 ReduceScatter。

目前没有足够证据把问题直接归因于 OAI 公式写错；最有价值的下一步是同时 dump 激活前 FP32、激活后 BF16、MX scale 和 MX 量化输出，而不是只看最终三方误差。
