# FFN

> 注意：
> 本文档仅仅是算子功能的简介，不支持用户直接调用，因为当前不支持kernel直调，等后续支持再完善文档!!!!!!

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |

## 功能说明

- 算子功能：该FFN算子提供MoeFFN和FFN的计算功能。在没有专家分组（expertTokens为空）时是FFN，有专家分组时是MoeFFN，统称为FFN，属于Moe结构。MoE（Mixture-of-Experts，混合专家系统）是一种用于训练万亿参数量级模型的技术。MoE将预测建模任务分解为若干子任务，在每个子任务上训练一个专家模型（Expert Model），开发一个门控模型（Gating Model），该模型会根据输入数据分配一个或多个专家，最终综合多个专家计算结果作为预测结果。Mixture-of-Experts结构的模型是将输入数据分配给最相关的一个或者多个专家，综合涉及的所有专家的计算结果来确定最终结果。
- 计算公式：

  - **非量化场景：**
  
	$$
    y=activation(x * W1 + b1) * W2 + b2
	$$

  - **量化场景：**

	$$
    y=((activation((x * W1 + b1) * deqScale1) * scale + offset) * W2 + b2) * deqScale2
	$$

  - **伪量化场景：**

	$$
    y=activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
	$$

- **参数说明：**

  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT16、BFLOAT16、INT8，[数据格式](common/数据格式.md)支持ND，支持输入的维度最少是2维[M, K1]，最多是8维。
  - weight1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，专家的权重数据，公式中的W1，数据类型支持FLOAT16、BFLOAT16、INT8、INT4，[数据格式](common/数据格式.md)支持ND，输入在有/无专家时分别为[E, K1, N1]/[K1, N1]。
  - weight2（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，专家的权重数据，公式中的W2，数据类型支持FLOAT16、BFLOAT16、INT8、INT4，[数据格式](common/数据格式.md)支持ND，输入在有/无专家时分别为[E, K2, N2]/[K2, N2]。

    >**说明：**
    >M表示token个数，对应transform中的BS（B：Batch，表示输入样本批量大小，S：Seq-Length，表示输入样本序列长度）；K1表示第一个matmul的输入通道数，对应transform中的H（Head-Size，表示隐藏层的大小）；N1表示第一个matmul的输出通道数；K2表示第二个matmul的输入通道数；N2表示第二个matmul的输出通道数，对应transform中的H；E表示有专家场景的专家数。
  - expertTokens（aclIntArray\*，计算输入）：可选参数，Host侧的aclIntArray类型，代表各专家的token数，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND，若不为空时可支持的最大长度为256个。
  - bias1（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，权重数据修正值，公式中的b1，数据类型支持FLOAT16、FLOAT32、INT32，[数据格式](common/数据格式.md)支持ND，输入在有/无专家时分别为[E, N1]/[N1]。
  - bias2（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，权重数据修正值，公式中的b2，数据类型支持FLOAT16、FLOAT32、INT32，[数据格式](common/数据格式.md)支持ND，输入在有/无专家时分别为[E, N2]/[N2]。
  - scale（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，量化参数，量化缩放系数，数据类型支持FLOAT32，[数据格式](common/数据格式.md)支持ND，per-tensor下输入在有/无专家时均为一维向量，输入元素个数在有/无专家时分别为[E]/[1]；per-channel下输入在有/无专家时为二维向量/一维向量，输入元素个数在有/无专家时分别为[E, N1]/[N1]。
  - offset（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，量化参数，量化偏移量，数据类型支持FLOAT32，[数据格式](common/数据格式.md)支持ND，一维向量，输入元素个数在有/无专家时分别为[E]/[1]。
  - deqScale1（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，量化参数，第一个matmul的反量化缩放系数，数据类型支持UINT64、INT64、FLOAT32、BFLOAT16，[数据格式](common/数据格式.md)支持ND，输入在有/无专家时分别为[E, N1]/[N1]。
  - deqScale2（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，量化参数，第二个matmul的反量化缩放系数，数据类型支持UINT64、INT64、FLOAT32、BFLOAT16，[数据格式](common/数据格式.md)支持ND，输入在有/无专家时分别为[E, N2]/[N2]。
  - antiquantScale1（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，伪量化参数，第一个matmul的缩放系数，数据类型支持FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，per-channel下输入在有/无专家时分别为[E, N1]/[N1]，per-group下输入在有/无专家时分别为[E, G, N1]/[G, N1]。
  - antiquantScale2（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，伪量化参数，第二个matmul的缩放系数，数据类型支持FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，per-channel下输入在有/无专家时分别为[E, N2]/[N2]，per-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]。
  - antiquantOffset1（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，伪量化参数，第一个matmul的偏移量，数据类型支持FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，per-channel下输入在有/无专家时分别为[E, N1]/[N1]，per-group下输入在有/无专家时分别为[E, G, N1]/[G, N1]。
  - antiquantOffset2（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，伪量化参数，第二个matmul的偏移量，数据类型支持FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，per-channel下输入在有/无专家时分别为[E, N2]/[N2]，per-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]。


## 约束说明

- 有专家时，专家数据的总数需要与x的M保持一致。
- 激活层为geglu/swiglu/reglu时，仅支持无专家分组时的FLOAT16高性能场景（FLOAT16场景指类型为aclTensor的必选参数数据类型都为FLOAT16的场景），且N1=2\*K2。
- 激活层为gelu/fastgelu/relu/silu时，支持有专家或无专家分组的FLOAT16高精度及高性能场景、BFLOAT16场景、量化场景及伪量化场景，且N1=K2。
- 所有场景下需满足K1=N2, K1<65536, K2<65536, M轴在32Byte对齐后小于INT32的最大值。
- 非量化场景不能输入量化参数和伪量化参数，量化场景不能输入伪量化参数，伪量化场景不能输入量化参数。
- 量化场景参数类型：x为INT8、weight为INT8、bias为INT32、scale为FLOAT32、offset为FLOAT32，其余参数类型根据y不同分两种情况：
  - y为FLOAT16，deqScale支持数据类型：UINT64、INT64、FLOAT32。
  - y为BFLOAT16，deqScale支持数据类型：BFLOAT16。
  - 要求deqScale1与deqScale2的数据类型保持一致。
- 量化场景支持scale的per-channel模式参数类型：x为INT8、weight为INT8、bias为INT32、scale为FLOAT32、offset为FLOAT32，其余参数类型根据y不同分两种情况：
  - y为FLOAT16，deqScale支持数据类型：UINT64、INT64。
  - y为BFLOAT16，deqScale支持数据类型：BFLOAT16。
  - 要求deqScale1与deqScale2的数据类型保持一致。
- 伪量化场景支持两种不同参数类型：
  - y为FLOAT16、x为FLOAT16、bias为FLOAT16，antiquantScale为FLOAT16、antiquantOffset为FLOAT16，weight支持数据类型INT8和INT4。
  - y为BFLOAT16、x为BFLOAT16、bias为FLOAT32，antiquantScale为BFLOAT16、antiquantOffset为BFLOAT16，weight支持数据类型INT8和INT4。
- 当weight1/weight2的数据类型为INT4时，其shape最后一维必须为偶数。
- 伪量化场景，per-group下，antiquantScale1和antiquantOffset1中的K1需要能整除组数G，antiquantScale2和antiquantOffset2中的K2需要能整除组数G。
- 伪量化场景，per-group下目前只支持weight是INT4数据类型的场景。
- innerPrecise参数在BFLOAT16非量化场景，只能配置为0；FLOAT16非量化场景，可以配置为0或者1；量化或者伪量化场景，0和1都可配置，但是配置后不生效。

