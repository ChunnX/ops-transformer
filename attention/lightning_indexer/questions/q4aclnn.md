# LightningIndexer ACLNN 接口学习问题记录

> **文档用途**：学习 `aclnnLightningIndexer` 接口过程中的问题与回答记录
>
> **参考文档**：`docs/aclnnLightningIndexer.md`
>
> **接口头文件**：`op_host/op_api/aclnn_lightning_indexer.h`

---

## 一、接口总览

### Q1：aclnnLightningIndexer 中的 $W\in\R^{g\times 1}$，为什么入参的shape是 $(B,S1,N1)$ 而不是 $(B,S1,N_1/N_2)$？

> **回答**：当前 $N_2$ 只支持1。如果不是这个原因，那么公式中的 $W$ 并不完全等价于输入中的 weights

---

### Q2：aclnnLightningIndexer 中的 sparseCount 就是 top-k 的 k？

> **回答**：是。从 sparseIndicesOut 的 shape 可以看出。

---

### Q2：aclnnLightningIndexer 中的 weights 支持 fp32，对应的参数类型应该是 fp16/fp16/fp32 和 bf16/bf16/fp32，这里精度损失情况和使用原因是什么？

> **回答**：待回答（计算过程中低精度会先转换成高精度，再做计算）

---