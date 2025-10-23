# GroupedMatmulSwigluQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 接口功能：融合GroupedMatmul 、dquant、swiglu和quant，详细解释见计算公式。
- 计算公式：
  
  - **定义**：
    
    * **⋅** 表示矩阵乘法。
    * **⊙** 表示逐元素乘法。
    * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
    * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
    * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
  - **输入**：
    
    * $X∈\mathbb{Z_8}^{M \times K}$：输入矩阵（左矩阵），M是总token 数，K是特征维度。
    * $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
    * $bias∈\mathbb{Z_{32}}^{E  \times N}$：矩阵乘计算的偏移值，E是专家个数，N是输出维度。
    * $offset∈\mathbb{R}^{E  \times N}$：per-channel非对称反量化的偏移，E是专家个数，N是输出维度。
    * $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，N是输出维度。
    * $x\_scale∈\mathbb{R}^{M}$：输入矩阵（左矩阵）的逐 token缩放因子，M是总token 数。
    * $grouplist∈\mathbb{N}^{E}$：前缀和的分组索引列表。
  - **输出**：
    
    * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
    * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
    * $Q\_offset∈\mathbb{R}^{M}$：量化偏移因子。
  - **计算过程**
    
    - 1.根据groupList[i]确定当前分组的 token ，$i \in [0,Len(groupList)]$。
    
      >例子：假设groupList=[3,4,4,6]，从0开始计数。
      >
      >第0个右矩阵`W[0,:,:]`，对应索引位置[0,3)的token`x[0:3]`（共3-0=3个token），对应`x_scale[0:3]`、`w_scale[0]`、`bias[0]`、`offset[0]`、`Q[0:3]`、`Q_scale[0:3]`、`Q_offset[0:3]`；
      >
      >第1个右矩阵`W[1,:,:]`，对应索引位置[3,4)的token`x[3:4]`（共4-3=1个token），对应`x_scale[3:4]`、`w_scale[1]`、`bias[1]`、`offset[1]`、`Q[3:4]`、`Q_scale[3:4]`、`Q_offset[3:4]`；
      >
      >第2个右矩阵`W[2,:,:]`，对应索引位置[4,4)的token`x[4:4]`（共4-4=0个token），对应`x_scale[4:4]`、`w_scale[2]`、`bias[2]`、`offset[2]`、`Q[4:4]`、`Q_scale[4:4]`、`Q_offset[4:4]`；
      >
      >第3个右矩阵`W[3,:,:]`，对应索引位置[4,6)的token`x[4:6]`（共6-4=2个token），对应`x_scale[4:6]`、`w_scale[3]`、`bias[3]`、`offset[3]`、`Q[4:6]`、`Q_scale[4:6]`、`Q_offset[4:6]`；
      >
      >请注意：grouplist中未指定的部分将不会参与更新。
      >例如groupList=[12,14,18]，X的shape为[30，:]。
      >
      >则第一个输出Q的shape为[30，:]，其中Q[18:，：]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。
      >
      >同理，第二个输出Q的shape为[30]，其中Q\_scale[18:]的部分不会进行更新或初始化，其中数据为显存空间申请时的原数据。
      >
      >即输出的Q[:grouplist[-1],:]和Q\_scale[:grouplist[-1]]为有效数据部分。

    - 2.根据分组确定的入参进行如下计算：

      $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

      $C_{i,act}, gate_{i} = split(C_{i})$

      $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$

      >注：当前版本不支持$bias_{i}$、$offset_{i}$，未来版本将支持的计算公式如下：
      >$C_{i} =(X_{i}\cdot W_{i} + bias_{i\ BroadCast})\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}+offset_{i\ BroadCast}$

    - 3.确定量化方式
      
      - 当量化方式为对称量化时：

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left \lfloor \frac{S_{i}}{Q\_scale_{i}}\right \rceil $

      - 当量化方式为非对称量化时：(暂不支持)

        $Q\_scale_{i} = \frac{max(S_{i})-min(S_{i})}{255}$

        $Q\_offset_{i} = -128 - \left \lfloor \frac{min(S_{i})}{Q\_scale_{i}}\right \rceil$

        $Q_{i} = \left \lfloor \frac{S_{i}}{ Q\_scale_{i} } + Q\_offset_{i}\right \rceil $

## 参数说明
|参数名| 输入/输出/属性   |    描述 |数据类型 |
|-----|---------|------|------|
|x|输入|左矩阵，公式中的$X$|INT8|
|weight|输入|权重矩阵，公式中的$W$|INT8|
|bias|输入|矩阵乘计算的偏移值，公式中的$bias$|INT32|
|offsetx|输入|per-channel非对称反量化的偏移，公式中的$offset$|FLOAT32|
|weightScale|输入|右矩阵的量化因子，公式中的$w\_scale$|FLOAT、FLOAT16、BFLOAT16|
|xScale|输入|左矩阵的量化因子，公式中的$x\_scale$|FLOAT32|
|groupList|输入|指示每个分组参与计算的Token个数，公式中的$grouplist$|INT64|
|output|输出|输出的量化因子，公式中的$Q\_scale$|FLOAT|
|outputScale|输出|输出的量化因子，公式中的$Q\_scale$|FLOAT|
|outputOffset|输出|输出的非对称量化的偏移，公式中的$Q\_offset$|FLOAT|


## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_matmul_swiglu_quant](examples/test_aclnn_grouped_matmul_swiglu_quant.cpp) | 通过接口方式调用[GroupedMatmulSwigluQuant](docs/aclnnGroupedMatmulSwigluQuant.md)算子。 |