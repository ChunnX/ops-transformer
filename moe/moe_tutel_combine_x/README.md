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

- 算子功能：完成MoE（Mixture of Experts）Tutel路由反向combine的数据梯度计算。算子按路由元数据（专家索引indices与专家buffer内位置locations）从专家输出梯度y_grad中取出对应行，乘以门控权重gates后作为原始token位置的输入梯度x_grad输出。支持无batch（y_grad为2维）与带batch（y_grad为3维）两种张量形态。

- 计算公式：无batch场景下，y_grad形状为[EC, H]，gates/indices/locations形状为[S]，x_grad形状为[S, H]。对每个token t，设idx0=indices[t]、idx1=locations[t]、srcRow=idx0*C+idx1：

  $$
  x\_grad[t, h] =
  \begin{cases}
  gates[t] \times y\_grad[srcRow, h], & \text{if } idx0 \ge 0 \text{ and } 0 \le idx1 < C \text{ and } 0 \le srcRow < EC \\
  0, & \text{otherwise}
  \end{cases}
  $$

  带batch场景下，y_grad形状为[B, EC, H]，gates/indices/locations形状为[B, S]，x_grad形状为[B, S, H]，对每个batch b、每个token t按同一规则计算。

  其中EC为y_grad无batch时的行数或带batch时的第二维长度，H为隐藏维度（y_grad最后一维），S为token数（gates最后一维），B为batch数。专家数E=max(indices)+1，每专家容量C=floor(EC/E)不经算子属性传递，由张量维度与路由元数据推导；E=1时C=EC；indices全为-1时E=0，全部条目无效，输出全零。

- 计算规则补充说明：x_grad的每个输出行只由对应token的一条路由条目决定，有效条目写gates[t]*y_grad[srcRow, :]，无效条目（indices[t]<0、locations[t]<0、locations[t]>=C或srcRow越出y_grad行范围）输出全零行，不访问越界内存，调用前无需对x_grad预置初值。

## 参数说明

  <table style="table-layout: fixed; width: 910px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出/属性</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>y_grad</td>
        <td>输入</td>
        <td>表示专家缓冲区输出的梯度，即公式中的y_grad。shape为[EC, H]或[B, EC, H]，rank为2或3。</td>
        <td>FLOAT16、FLOAT、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gates</td>
        <td>输入</td>
        <td>表示每个token副本的路由权重（门控值），即公式中的gates。shape为[S]或[B, S]，与indices、locations一致。</td>
        <td>FLOAT16、FLOAT、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>输入</td>
        <td>表示每个token副本被分配到的专家ID，-1表示容量丢弃的无效条目，即公式中的indices。shape与gates一致。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>locations</td>
        <td>输入</td>
        <td>表示每个token副本在其分配专家缓冲区内的位置偏移，即公式中的locations。shape与gates一致。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x_grad</td>
        <td>输出</td>
        <td>表示原始token位置的输入梯度，即公式中的x_grad。shape为gates的shape拼接y_grad最后一维，即[S, H]或[B, S, H]。</td>
        <td>FLOAT16、FLOAT、BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- shape与rank契约：y_grad的rank为2或3，gates/indices/locations的rank为1或2，且rank(y_grad)=rank(gates)+1；gates、indices、locations三者shape一致；带batch（rank(y_grad)=3）时gates.dim0=y_grad.dim0。
- 动态shape：支持以-1表示未知维（动态shape）。图模式下-1维被接受并传播到输出shape（x_grad对应维为-1），维具体化后须满足上述rank与shape契约；维度值小于-1（如-2）为非法输入，报错退出。不支持动态rank，各输入rank须为上述固定取值。
- 数据类型约束：y_grad、gates、x_grad三者数据类型必须一致（FLOAT16、FLOAT、BFLOAT16三选一），indices、locations固定为INT32。
- 数据格式约束：仅支持ND格式。
- 非连续Tensor：要求传入的Tensor为连续存储（图模式经算子IR构图调用）。
- 空Tensor：S=0、H=0或B=0时输出为对应shape的空Tensor；EC=0且S大于0时C=0，全部条目无效，输出[S, H]全零。
- 索引语义：indices[t]<0、locations[t]<0、locations[t]>=C或计算得到的源行越出y_grad行范围的条目为无效条目，对应输出行置零，不报错。
- 路由元数据来源：gates/indices/locations须由前向MoeTutelDispatch产出并在反向原样重放，不允许反向独立重算路由。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_moe_tutel_combine_x](examples/test_geir_moe_tutel_combine_x.cpp) | 通过[算子IR](op_graph/moe_tutel_combine_x_proto.h)构图方式调用MoeTutelCombineX算子。 |

```bash
# 在ops-transformer仓库根目录执行：先完成算子包构建并安装到运行环境，再编译运行GE图模式样例
bash build.sh --pkg --soc=ascend950 --ops=moe_tutel_combine_x
bash build.sh --run_example moe_tutel_combine_x graph --soc=ascend950
```
