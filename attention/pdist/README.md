# Pdist

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3训练系列产品</term>|      ×     |
|<term>Atlas A2训练系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：Pdist算子计算行向量两两之间的p-范数距离（pairwise p-norm distances），输出为上三角行优先的一维向量。等价于 `torch.nn.functional.pdist`。

- kernel 实现形态：**PyPTO DSL**（`op_kernel/pdist.py`），经 `enable_pypto_kernel` 在 configure 阶段做 codegen 并以 pypto opc 形式交付（MIX 发射，AIC 空转、计算全部位于 Vector 核）。

- 计算公式：

$$
y_{idx}=\left\|x_{i}-x_{j}\right\|_{p}=\left(\sum_{k=0}^{M-1}\left|x_{i,k}-x_{j,k}\right|^{p}\right)^{1/p},\quad idx=\frac{i\cdot(2N-i-1)}{2}+(j-i-1)
$$

$$
0\le i<j\le N-1
$$

p 分支语义：

| p 取值 | 语义 |
|--------|------|
| p = 0 | 统计不相等的维度数（Hamming 距离），NaN 维度计为不相等 |
| p = +inf | 最大绝对差（Chebyshev 距离） |
| 0 < p < +inf | 幂和的 1/p 次方（Minkowski 距离），p=2 为欧氏距离、p=1 为曼哈顿距离 |

NaN 语义：p=0 时 NaN 维度计为不相等；p=+inf 与一般 p（非 0/0.5/1/2）逐对检测 NaN 维度并输出 NaN；p∈{0.5, 1, 2} 通过 IEEE 算术自然传播。

fp16 输入：片上中间累加一律使用 FP32，出口窄化回 FP16。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>x</td>
      <td>输入</td>
      <td>输入张量，shape为(N, M)，N≥2，M≥1。行i与行j（j>i）组成一个行对参与计算。要求行主序连续。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>p</td>
      <td>属性</td>
      <td>范数阶，p≥0或+inf，默认2.0（欧氏距离）。p=0统计不相等维度数，p=+inf取最大绝对差，0<p<inf为幂和的1/p次方。</td>
      <td>FLOAT</td>
      <td>-</td>
     </tr>
     <tr>
      <td>y</td>
      <td>输出</td>
      <td>两两距离向量，shape为(N*(N-1)/2,)，上三角行优先排列，dtype与x一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
     </tr>
     </tbody>
    </table>

## 约束说明

  - x必须为二维(N, M)行主序连续张量，y的dtype必须与x一致。
  - y的shape必须为(N*(N-1)/2,)。
  - fp16输入的中间累加使用FP32，出口窄化回FP16。

  <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
  <col style="width: 125px">
  <col style="width: 182px">
  <col style="width: 602px">
  </colgroup>
  <thead>
  <tr>
    <th>规格项</th>
    <th>规格</th>
    <th>规格说明</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>N</td>
    <td>2~65535</td>
    <td>行数，至少为2（至少一个行对）。</td>
  </tr>
  <tr>
    <td>M</td>
    <td>1~65535</td>
    <td>每行维度数，超过片上内存时自动分块规约。</td>
  </tr>
  <tr>
    <td>p</td>
    <td>[0, +inf)</td>
    <td>p≥0或+inf；p=0为Hamming距离，p=+inf为Chebyshev距离，p<0或NaN拒绝为参数错误。</td>
  </tr>
  <tr>
    <td>dtype</td>
    <td>FLOAT16/FLOAT32</td>
    <td>输入输出同dtype；fp16中间计算使用FP32累加。</td>
  </tr>
  </tbody>
  </table>

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_pdist](examples/test_aclnn_pdist.cpp) | 通过[aclnnPdist](docs/aclnnPdist.md)接口方式调用pdist算子，覆盖fp32全部p分支、多M分块与fp16。 |
| GEIR调用 | [test_geir_pdist](examples/test_geir_pdist.cpp) | 通过GE IR图模式（RunGraph）方式调用pdist算子。图模式推荐的集成路径为torch_extension（底层走aclnn+预编译二进制）。 |
| torch_extension调用 | [torch_extension/pdist.py](torch_extension/pdist.py) | 通过torch.library注册（`cann_ops_transformer::pdist`）+ `torch.compiler.allow_in_graph` 方式调用，支持eager与图模式。 |
