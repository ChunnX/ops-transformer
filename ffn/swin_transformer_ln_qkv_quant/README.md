# SwinTransformerLnQkvQuant

> 注意：
> 本文档仅仅是算子功能的简介，不支持用户直接调用，因为当前不支持kernel直调，等后续支持再完善文档!!!!!!

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |

## 功能说明
- 算子功能：Swin Transformer 网络模型 完成 Q、K、V 的计算。  
- 计算公式：  

  q/k/v = (Quant(Layernorm(x).transpose)  * weight).dequant.transpose.split
  其中，weight 是 Q、K、V 三个矩阵权重的拼接。


## 参数说明
  - x(aclTensor*,计算输入): 表示待进行归一化计算的目标张量，公式中的x， Device侧的aclTensor，数据类型支持FLOAT16。只支持维度为[B,S,H]，其中B为batch size且只支持[1,32],S为原始图像长宽的乘积，H为序列长度和通道数的乘积且小于等于1024，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - gamma(aclTensor*,计算输入): 表示layernorm计算中尺度缩放的大小，维度只支持1维且为[H]，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - beta(aclTensor*,计算输入): 表示layernorm计算中尺度偏移的大小，维度只支持1维且维度为[H]，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - weight(aclTensor*,计算输入): 表示目标张量转换使用的权重矩阵，维度只支持2维且维度为[H, 3 * H],Device侧的aclTensor，数据类型支持INT8，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - bias(aclTensor*,计算输入): 表示目标张量转换使用的偏移矩阵，维度只支持1维且维度为[3 * H]，Device侧的aclTensor，数据类型支持INT32，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - quantScale(aclTensor*,计算输入):  表示目标张量量化使用的缩放参数，维度只支持1维且维度为[H]，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - quantOffset(aclTensor*,计算输入): 表示目标张量量化使用的偏移参数，维度只支持1维且维度为[H]，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - dequantScale(aclTensor*,计算输入): 表示目标张量乘以权重矩阵之后反量化使用的缩放参数，维度只支持1维且维度为[3 * H]，Device侧的aclTensor，数据类型支持UINT64，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - headNum(int，计算输入): 表示转换使用的通道数；支持范围[1,32]。
  - seqLength(int，计算输入): 表示转换使用的通道深度。只支持32/64两种。
  - epsilon(float,计算输入): layernorm 计算除0保护值；为了保证精度，建议小于等于1e-4。
  - oriHeight(int,计算输入): layernorm 中S轴transpose的维度；oriHeight*oriWeight需等于输入x的第二维S的大小，且为hWinSize的整数倍。
  - oriWeight(int,计算输入): layernorm 中S轴transpose的维度；oriHeight*oriWeight需等于输入x的第二维S的大小，且为wWinSize的整数倍。
  - hWinSize(int,计算输入): 使用的特征窗高度大小；支持范围[7,32]。
  - wWinSize(int,计算输入): 使用的特征窗宽度大小；支持范围[7,32]。
  - weightTranspose(bool,计算输入): weight矩阵需要转置，当前不支持不转置场景。
  - queryOutputOut(aclTensor*, 计算输出)：表示转换之后的张量，公式中的Q，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - keyOutputOut(aclTensor*, 计算输出)：表示转换之后的张量，公式中的K，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - valueOutputOut(aclTensor*, 计算输出)：表示转换之后的张量，公式中的V，Device侧的aclTensor，数据类型支持

## 约束说明
- seqLength只支持32/64。
- oriHeight*oriWeight=输入x Tensor的第二维度，且oriHeight为hWinSize的整数倍，oriWeight为wWinSize的整数倍。
- hWinSize和wWinSize范围只支持7~32。
- 输入x Tensor的第一维度B只支持1~32。
- weight需要转置。

