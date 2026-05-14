# DESIGN - GMMSQ算子实现设计文档

> 提供明确的开发说明、设计思路、代码实现文档，需要严格按照设计文档进行开发

## 关键信息

- gmmsq算子代码路径: gmm/grouped_matmul_swiglu_quant_v2
- GMMSQ 设计文档: `gmm/grouped_matmul_swiglu_quant_v2/.agent/raw/design/`
- 适配工程指南: `~/Code/asc-wiki/ops-dev-wiki/raw/docs/`

## 文档检索

要实现GMMSQ(grouped_matmul_swiglu_quant_v2)算子，提供了以下设计文档：

- [算子原型设计说明书](raw/design/gmmsq_op.md)：侧重于实现无关的原理说明，和外部调用接口说明，包含算子数学公式说明、算子原型接口的定义、接口参数的说明及约束。
- [算子kernel实现设计说明书](raw/design/gmmsq_kernel_impl.md)：侧重于实现kernel的代码架构和实现方案的具体细节，包括算法原理和伪代码。

需求设计文档和用例设计文档：

- [SRS需求开发设计文档](raw/development/srs_gmmsq.md)：侧重于需求方案反串讲，确认需求实现方案的合理性。
    - [SRS需求实现详细编码方案文档](raw/development/srs_coding_design.md)：侧重于对实现的详细编码方案进行开发前确认，保证代码实现的可行性。
- [STC需求实现用例设计文档](raw/development/stc_gmmsq.md)：侧重于需求方案实现需要设计的用例和方案。

以GMMFR算子作为参考的代码实现，代码走读总结文档：
- [GMMFR算子实现文档以及迁移到GMMSQ的更改实现方案](raw/docs/)
    - GMMFR Kernel入口: `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_apt.cpp`
    - GMMFR Controller: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h`