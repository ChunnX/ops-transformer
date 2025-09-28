# aclnnGroupedMatmulAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |


## 功能说明

- 算子功能：实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i]+y_i[m_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应shape。输入输出数据类型均为aclTensor，K轴分组。

  - k轴分组：$k_i$各不相同，但$m_i/n_i$每组相同。
- 计算公式：

  $$
  y_i=x_i\times weight_i + y_i
  $$


## 参数说明

|参数名| 输入/输出/属性   |    描述 |数据类型 |
|-----|---------|------|------|
|x|输入|公式中的输入x|FLOAT16、BFLOAT16|
|weight|输入|公式中的weight|FLOAT16、BFLOAT16|
|groupList|输入|表示输入K轴方向的matmul大小分布的cumsum结果（累积和）|INT64|
|y|输入|表示原地累加的输出矩阵|FLOAT32|
|transposeX|属性|表示x矩阵是否转置|BOOL|
|transposeWeight|属性|表示weight矩阵是否转置|BOOL|
|groupType|属性|表示分组类型|INT64|
|yRef|输出|表示原地累加的输出矩阵|FLOAT32|

## 约束说明

- x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647。
- 支持的输入类型为：
  - x为FLOAT16、weight为FLOAT16、y为FLOAT32。
  - x为BFLOAT16、weight为BFLOAT16、y为FLOAT32。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_matmul_add](examples/test_aclnn_grouped_matmul_add.cpp) | 通过接口方式调用[GroupedMatmulAdd](docs/aclnnGroupedMatmulAdd.md)算子。 |