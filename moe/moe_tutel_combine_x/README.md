# MoeTutelCombineX

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :--- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：完成 Mixture-of-Experts（MoE）模型反向传播中的 Combine 散射计算。算子按路由元数据（专家索引 `indices` 与专家 buffer 内位置 `locations`）从专家输出梯度 `y_grad` 中取出对应行，乘以门控权重 `gates` 后作为原始 token 位置的输入梯度 `x_grad` 输出。支持无 batch（`y_grad` 为 2 维）与带 batch（`y_grad` 为 3 维）两种张量形态。

- 计算公式：无 batch 场景下，`y_grad` 形状为 `[E*C, H]`，`gates`/`indices`/`locations` 形状为 `[S]`，`x_grad` 形状为 `[S, H]`。对每个 token `t`，设 `idx0 = indices[t]`、`idx1 = locations[t]`、`srcRow = idx0 * C + idx1`：

$$
x\_grad[t, h] =
\begin{cases}
gates[t] \times y\_grad[srcRow, h], & \text{if } idx0 \ge 0 \text{ and } idx1 < C \text{ and } srcRow < E \cdot C \\
0, & \text{otherwise}
\end{cases}
$$

  带 batch 场景下，`y_grad` 形状为 `[B, E*C, H]`，`gates`/`indices`/`locations` 形状为 `[B, S]`，`x_grad` 形状为 `[B, S, H]`，对每个 batch `b`、每个 token `t` 按同一规则计算。

  其中 `E` 为路由覆盖的专家数（`E = max(indices) + 1`），`C` 为每个专家 buffer 的容量（`C = eC / E`，`eC` 为 `y_grad` 在无 batch 时的行数或带 batch 时的第二维长度），`H` 为隐藏维度（`y_grad` 最后一维），`S` 为 token 数（`gates` 最后一维），`B` 为 batch 数。

- 计算规则补充说明：`x_grad` 的每个输出行只由对应 token 的一条路由条目决定，有效条目写 `gates[t] * y_grad[srcRow, :]`，无效条目（`indices[t] < 0`、`locations[t] >= C` 或 `srcRow` 越出 `y_grad` 行范围）输出全零行，调用前无需对 `x_grad` 预置初值。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| y_grad | 输入 | 表示专家 buffer 输出的梯度，即公式中的 y_grad。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| gates | 输入 | 表示每个 token 副本的路由权重（门控值），即公式中的 gates。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| indices | 输入 | 表示每个 token 副本被分配到的专家 ID，即公式中的 indices。 | INT32 | ND |
| locations | 输入 | 表示每个 token 副本在其分配专家 buffer 内的位置偏移，即公式中的 locations。 | INT32 | ND |
| x_grad | 输出 | 表示原始 token 位置的输入梯度，即公式中的 x_grad。 | FLOAT16、FLOAT、BFLOAT16 | ND |

## 约束说明

- shape 与 rank 约定：`y_grad` 的 rank 为 2 或 3，`gates`/`indices`/`locations` 的 rank 为 1 或 2，且 `rank(y_grad) = rank(gates) + 1`；`gates`、`indices`、`locations` 三者 shape 一致；`x_grad` 的 shape 等于 `gates` 的 shape 拼接 `y_grad` 最后一维。隐藏维度 `H` 取值范围为 [1, 65535]，token 数 `S` 取值范围为 [1, 9223372036854775807]。
- 数据类型约束：`y_grad`、`gates`、`x_grad` 三者数据类型必须一致（FLOAT16、FLOAT、BFLOAT16 三选一），`indices`、`locations` 固定为 INT32。
- 数据格式约束：仅支持 ND 格式；传入的 Tensor 需为连续存储（strides 与 shape 一致），不支持非连续 Tensor。
- 空 Tensor：shape 的任一允许维度为 0（如 `S=0`、`eC=0`、`H=0`）即视为空 Tensor，算子不执行散射写，`x_grad` 被置零或保持为空。
- 索引语义：`indices[t] < 0`、`locations[t] >= C` 或计算得到的源行越出 `y_grad` 行范围的条目为无效条目，对应输出行置零，不访问越界内存。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn API | - | 参见 [aclnnMoeTutelCombineX API 文档](docs/aclnnMoeTutelCombineX.md)，通过两段式接口 aclnnMoeTutelCombineXGetWorkspaceSize 与 aclnnMoeTutelCombineX 完成调用。 |
| GE图模式 | - | 通过算子 IR 定义 [moe_tutel_combine_x_proto.h](op_graph/moe_tutel_combine_x_proto.h) 构图方式调用，图编译期由 InferShape/InferDataType 完成 shape、dtype 与 format 校验。 |
