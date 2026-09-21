# Quant Sparse Lightning Indexer

本目录的 torch_extension 为独立 DSL 包提供 Torch 注册与接口转换。pytest、extension
和 DSL 可以位于不同目录，不依赖项目相对路径。

## 调用入口

导入 `cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl` 后使用：

- `torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer`
- `torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer_metadata`
- `cann_ops_transformer.ops.ds41` 下的两个同名入口

扩展通过 `ops.quant_sparse_lightning_indexer_dsl` 和
`ops.quant_sparse_lightning_indexer_metadata_dsl` 导入 DSL。
真实部署应安装匹配的 DSL 包和 cann_ops_transformer 扩展包。源码路径不会自动回退；
可通过 PYTHONPATH 配置独立的临时包目录进行源码验证；这不等同于 wheel/native
二进制验收。覆盖源码后需重新构建并安装匹配的两个包。

## 数据和接口

Q 为 TND、uint8 packed MXFP4，物理形状 `(T1,32,64)`；逻辑 D=128。
K 支持 PA_BBND 和 TND。测试侧负责把 PA 的 K/scale 组合为每8个token一块的
544字节 packed 表示；TND K 为 `(T2,1,64)`，独立 scale 为 `(T2,1,2,2)`。
candidate indices 为 `(T1,1,2048)`，candidate block size 为8。

主算子保留原 Torch 位置参数：
`q, k, w, descale_q, descale_k, candidate_block_indices, candidate_block_length,
topk, quant_mode, candidate_block_size`。
可选参数包括 `cu_seqlens_q/cu_seqlens_k`、`seqused_q/seqused_k`、
`cmp_residual_k`、`block_table`、`output_idx_offset`、`metadata`，
以及 `max_seqlen_q, mask_mode, cmp_ratio, layout_q, layout_kv, return_value`。
Torch 的 `layout_kv` 映射至 DSL `layout_k`，主算子默认 PA_BBND。

metadata 算子以 `candidate_block_length` 为必需输入，其他参数遵循当前 DSL，
布局参数命名为 `layout_kv`、默认 TND，返回连续 `int32[1024]`。
主算子省略 metadata 时通过注册后的 metadata 算子生成；显式提供时直接使用。

输出是两个 Tensor：int32 indices，形状 `(T1,1,topk)`；bfloat16 values，
开启 return_value 时同形状，关闭时为 `(0,)`。
Meta/FakeTensor 仅推导形状、类型、设备，不执行 DSL。

生产实现及分核不在回迁适配的修改范围。pytest 的 golden、输入与比较策略仍独立于 DSL。
测试入口及代表性接口回归运行方法见 [pytest/README.md](pytest/README.md)。
