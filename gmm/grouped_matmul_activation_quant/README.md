# GroupedMatmulActivationQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：融合GroupedMatmul、activation和quant计算。当前版本仅支持WeightNz路径下的MXFP8量化场景，激活函数仅支持gelu_tanh。

- 计算公式：
  - <term>Ascend 950PR/Ascend 950DT</term>：

    <details>
    <summary>MXFP8量化场景：</summary>

    - 定义：
      - $E$ 表示专家数，$M$ 表示总token数，$K$ 表示输入特征维度，$N$ 表示输出特征维度。
      - $blocksize$ 表示MX量化时共享指数的分组大小，当前仅支持64。
      - **·** 表示矩阵乘法，**⊙** 表示逐元素乘法。

    - 根据groupList确定每个group对应的token范围。

    - 对每个group执行GroupedMatmul和反量化计算，中间结果默认为FLOAT32类型：

      $$
      C_i = (X_i \cdot weight_i) \odot xScale_{i\ Broadcast} \odot weightScale_{i\ Broadcast}
      $$

    - 执行gelu_tanh激活：

      $$
      S_i = GeluTanh(C_i)
      $$

      当前kernel底层实现使用gelu_sigmoid函数对gelu_tanh进行近似计算：

      `GeluTanh(x) = x / (1 + exp(-1.595769121 * (x + 0.044715 * x^3)))`

    - 对激活结果在N轴按$blocksize=64$分组执行MX动态量化，输出量化结果$Y$和量化因子$YScale$。

    </details>

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">x</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">左矩阵，对应公式中的X。</td>
      <td style="white-space: nowrap">FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupList</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">分组信息，对应公式中的groupList。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weight</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">右矩阵dynamic tensorList。当前MXFP8场景tensorList长度仅支持1。</td>
      <td style="white-space: nowrap">FLOAT8_E4M3FN</td>
      <td style="white-space: nowrap">FRACTAL_NZ</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weightScale</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">weight的MX量化因子，dynamic tensorList。当前MXFP8场景tensorList长度仅支持1。</td>
      <td style="white-space: nowrap">FLOAT8_E8M0</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">bias</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">bias dynamic tensorList。当前MXFP8场景必须为空。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">xScaleOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">x的MX量化因子。当前MXFP8场景必须传入。</td>
      <td style="white-space: nowrap">FLOAT8_E8M0</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">activationType</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">激活函数类型，当前仅支持"gelu_tanh"。</td>
      <td style="white-space: nowrap">STRING</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">quantMode</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">量化模式，当前仅支持"mx"。</td>
      <td style="white-space: nowrap">STRING</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">transposeWeight</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示weight是否转置，默认false。</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupListType</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示groupList输入的分组方式，支持0和1，默认0。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">y</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">激活并量化后的输出矩阵。</td>
      <td style="white-space: nowrap">FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">yScale</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">输出y的MX量化因子。</td>
      <td style="white-space: nowrap">FLOAT8_E8M0</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 当前仅支持<term>Ascend 950PR/Ascend 950DT</term>。
- 当前仅支持激活函数为gelu_tanh、量化模式为MXFP8的组合。
- x仅支持非转置输入；weight支持非转置和转置输入，weightScale转置属性需要与weight保持一致。
- weight必须为FRACTAL_NZ格式，viewShape为3维，storageShape为5维。
- N必须为64的整数倍，groupList第一维取值范围为[1, 1024]。
- MXFP8场景下bias必须为空，支持nullptr、空tensorList或长度为1且元素shape为(0)的空tensorList。
- roundMode当前仅支持"rint"，scaleAlg支持0或1。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| aclnn调用 | [aclnnGroupedMatmulActivationQuantWeightNz](docs/aclnnGroupedMatmulActivationQuantWeightNz.md) | 通过两段式aclnn接口调用GroupedMatmulActivationQuant WeightNz算子。 |
| torch调用 | [grouped_matmul_activation_quant](../../torch_extension/cann_ops_transformer/docs/zh/grouped_matmul_activation_quant.md) | 通过torch_extension调用GroupedMatmulActivationQuant算子。 |
