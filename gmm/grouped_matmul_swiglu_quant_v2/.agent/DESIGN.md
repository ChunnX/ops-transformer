# DESIGN - GMMSQ算子实现设计文档

> 提供明确的开发说明、设计思路、代码实现文档，需要严格按照设计文档进行开发

## 文档检索

gmmsq算子代码路径: gmm/grouped_matmul_swiglu_quant_v2

要实现GMMSQ(grouped_matmul_swiglu_quant_v2)算子，提供了以下设计文档：

- [算子原型设计说明书](raw/design/gmmsq_op.md)：侧重于实现无关的原理说明，和外部调用接口说明，包含算子数学公式说明、算子原型接口的定义、接口参数的说明及约束。
- [算子kernel实现设计说明书](raw/design/gmmsq_kernel_impl.md)：侧重于实现kernel的代码架构和实现方案的具体细节，包括算法原理和伪代码。
- [算子用例设计说明书](raw/design/gmmsq_kernel_impl.md)：侧重于针对该算子如何UT、ST用例，保证功能和泛化性验证通过。
