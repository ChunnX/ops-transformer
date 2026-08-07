# ApplyRotaryPosEmbGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- **算子功能**：执行双路旋转位置编码[ApplyRotaryPosEmb](../apply_rotary_pos_emb/README.md)的反向计算。同时计算 **query** 和 **key** 的 rope 反向梯度，融合为一次 kernel 调用，节省 cos/sin 加载和 kernel launch 开销。

- **计算公式**：

    取旋转位置编码的正向计算中，broadcast的轴列表为`dims`，则计算公式可表达如下：

    - <term>Ascend 950PR/Ascend 950DT</term>：

    （1）half模式：
    $$
    grad\_q_1, grad\_q_2 = chunk(grad\_query\_embed, chunks=2, dim=-1)
    $$

    $$
    grad\_k_1, grad\_k_2 = chunk(grad\_key\_embed, chunks=2, dim=-1)
    $$

    $$
    cos_1, cos_2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin_1, sin_2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    query\_rotate = cat((-query_2, query_1), dim=-1)
    $$

    $$
    key\_rotate = cat((-key_2, key_1), dim=-1)
    $$

    $$
    grad\_query = cat(cos_1 * grad\_q_1 + sin_2 * grad\_q_2, cos_2 * grad\_q_2 - sin_1 * grad\_q_1, dim=-1)
    $$

    $$
    grad\_key = cat(cos_1 * grad\_k_1 + sin_2 * grad\_k_2, cos_2 * grad\_k_2 - sin_1 * grad\_k_1, dim=-1)
    $$

    $$
    grad\_cos = sum(grad\_query\_embed * query + grad\_key\_embed * key, dims)
    $$

    $$
    grad\_sin = sum(grad\_query\_embed * query\_rotate + grad\_key\_embed * key\_rotate, dims)
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1200px">
<colgroup>
  <col style="width: 50px">
  <col style="width: 50px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 50px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>grad_query_embed</td>
    <td>输入</td>
    <td>正向输出query的导数，对应公式中$grad\_q_{embed}$。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad_key_embed</td>
    <td>输入</td>
    <td>正向输出key的导数，对应公式中$grad\_k_{embed}$。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>输入</td>
    <td>正向计算输入cos，需与grad_query_embed数据类型一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>输入</td>
    <td>正向计算输入sin，需与grad_query_embed数据类型一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>query</td>
    <td>可选输入</td>
    <td>正向计算输入query。如果为空指针，则不计算grad_cos和grad_sin。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>可选输入</td>
    <td>正向计算输入key。如果为空指针，则不计算grad_cos和grad_sin。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rotary_mode</td>
    <td>属性</td>
    <td>旋转模式，仅支持"half"。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layout</td>
    <td>属性</td>
    <td>输入Tensor的布局格式。1=BSND，2=SBND，4=TND。默认值为1。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>grad_query</td>
    <td>输出</td>
    <td>正向计算输入query的导数，shape与grad_query_embed相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad_key</td>
    <td>输出</td>
    <td>正向计算输入key的导数，shape与grad_key_embed相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad_cos</td>
    <td>可选输出</td>
    <td>正向计算输入cos的导数，仅当query和key非空时有效。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad_sin</td>
    <td>可选输出</td>
    <td>正向计算输入sin的导数，仅当query和key非空时有效。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

  - <term>Ascend 950PR/Ascend 950DT</term>：
    输入张量支持BNSD、BSND、SBND、TND排布。各参数的shape约束可以描述如下：
    - 输入只支持3维或4维。
    - 输入输出Tensor的dtype必须相同。
    - 输入输出Tensor的D轴必须相同，在half模式下必须≤1024且能被2整除。
    - `grad_query_embed`、`grad_query`的shape必须相同，`grad_key_embed`、`grad_key`的shape必须相同。
    - 对于任意`layout`，`grad_query_embed`和`grad_key_embed`除N维度外其它维度必须相同。
    - `cos`、`sin`的N维度必须等于1，`cos`、`sin`的B维度可以等于1，也可以和`grad_query_embed`的B维度一致，其余维度需与`grad_query_embed`一致。
    - `cos`、`sin`、`grad_cos`、`grad_sin`的shape必须相同。
    - `query`维度需与`grad_query_embed`一致，`key`维度需与`grad_key_embed`一致。
    - `rotary_mode`仅支持"half"。
    - `layout`仅支持{1, 2, 4}，对应{BSND, SBND, TND}。3=BNSD为预留，暂不支持。

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_apply_rotary_pos_emb_grad](./examples/test_aclnn_apply_rotary_pos_emb_grad.cpp) | 通过[aclnnApplyRotaryPosEmbGrad](./docs/aclnnApplyRotaryPosEmbGrad.md)接口方式调用ApplyRotaryPosEmbGrad算子。             |

## Tiling 模板

根据cos/sin的广播模式选择不同的分核策略：

| 模板 | 适用场景 | 核间切分 |
|:--|:--|:--|
| BAB | layout=BSND且cos为1S1D，或layout=TND | B轴分核 × S轴分核 |
| AB | layout=BSND且cos为BS1D，或layout=SBND | BS合并轴分核 × N轴分核 |
| A | grad_query_embed、cos、sin的shape完全一致 | B×N×S合并 → 逐元素分核 |

## 文件结构

```
apply_rotary_pos_emb_grad/
├── CMakeLists.txt
├── README.md
├── docs/
│   └── aclnnApplyRotaryPosEmbGrad.md
├── examples/
│   └── test_aclnn_apply_rotary_pos_emb_grad.cpp
├── op_graph/
│   ├── CMakeLists.txt
│   ├── apply_rotary_pos_emb_grad_proto.h
│   └── fusion_pass/.gitkeep
├── op_host/
│   ├── CMakeLists.txt
│   ├── apply_rotary_pos_emb_grad_def.cpp
│   ├── apply_rotary_pos_emb_grad_infershape.cpp
│   ├── apply_rotary_pos_emb_grad_tiling.cpp
│   ├── apply_rotary_pos_emb_grad_tiling.h
│   ├── apply_rotary_pos_emb_grad_tiling_bab.cpp
│   ├── apply_rotary_pos_emb_grad_tiling_ab.cpp
│   ├── apply_rotary_pos_emb_grad_tiling_a_and_b.cpp
│   └── config/ascend950/
│       ├── apply_rotary_pos_emb_grad_binary.json
│       └── apply_rotary_pos_emb_grad_simplified_key.ini
├── op_kernel/
│   ├── apply_rotary_pos_emb_grad_apt.cpp
│   └── arch35/
│       ├── apply_rotary_pos_emb_grad_tiling_key.h
│       ├── apply_rotary_pos_emb_grad_tiling_data.h
│       ├── apply_rotary_pos_emb_grad_common.h
│       ├── apply_rotary_pos_emb_grad_dag.h
│       ├── apply_rotary_pos_emb_grad_rotary_x.h
│       ├── apply_rotary_pos_emb_grad_rotary_x_vf.h
│       ├── apply_rotary_pos_emb_grad_bab.h
│       ├── apply_rotary_pos_emb_grad_ab.h
│       └── apply_rotary_pos_emb_grad_a_and_b.h
└── tests/
    └── ...
```
