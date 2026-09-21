# Quant Lightning Indexer DSL

QLI 对 MXFP4 Q/K 计算带权归约分数、mask 和 sparse TopK，可选生成 candidate block 输出。
Q 为 TND，K 支持 PA_BBND 和 TND；N1=32、N2=1、逻辑 D=128。

## 包与调用入口

导入 `cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl` 后，通过
`torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer` 和
`torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata` 调用。
`cann_ops_transformer.ops.ds41` 也导出两个同名入口。

Torch 扩展从独立包的 `ops.quant_lightning_indexer_dsl` 和
`ops.quant_lightning_indexer_metadata_dsl` 导入实现。部署时需重新构建并安装
匹配版本的 DSL 包和扩展包；源码覆盖不会自动更新已安装的 wheel 或 native 二进制。
pytest、扩展和 DSL 不要求位于相邻目录，不自动从源码目录回退加载。

## 输入与接口

| 输入 | dtype | 物理形状 |
|---|---|---|
| q | uint8，packed MXFP4 | (T1,32,64) |
| k，PA | uint8，packed MXFP4 | (P,page_size,1,64) |
| k，TND | uint8，packed MXFP4 | (T2,1,64) |
| w | float32 | (T1,32) |
| q_descale | uint8，E8M0位表示 | (T1,32,2,2) |
| k_descale，PA | uint8，E8M0位表示 | (P,page_size,1,2,2) |
| k_descale，TND | uint8，E8M0位表示 | (T2,1,2,2) |

Torch 主接口的位置参数为 `q,k,w,q_descale,k_descale,topk,quant_mode`。
可选参数包含 cu_seqlens_q、cu_seqlens_k、seqused_q、seqused_k、cmp_residual_k、
block_table、output_idx_offset、metadata，以及 max_seqlen_q、mask_mode、cmp_ratio、
layout_q、layout_kv、return_value、candidate_topk_blocks、candidate_block_size。
完整签名见 [Torch 接口](torch_extension/quant_lightning_indexer.py)。

Torch 层 `layout_kv` 映射至 DSL 的 `layout_k`；主算子默认 PA_BBND，
metadata 默认 TND。metadata 参数遵循对应 DSL，输出连续 int32[1024]。
主算子省略 metadata 时通过已注册的 metadata 入口生成；显式 metadata 原样使用。
合法的 QLI metadata 无 Tensor 调用使用当前 NPU。

输出为 sparse indices、sparse values、candidate indices、candidate length。
indices/length 为 int32，values 为 bfloat16；关闭 return_value 时 values 为 (0,)。
candidate 参数同时为 -1 时关闭，两个 candidate 输出均为空；开启时 block size 为8。
cu_k 描述 TND K 的 batch 边界；output_idx_offset 只平移有效 sparse token index。

## 验证

参见 [pytest/README.md](pytest/README.md)。回迁仅运行代表性精度与接口检查，
不重复全量 Excel 或性能测试。保留 PA 计算和分核实现；本次同步不增加 native
export profile，也不宣称 TND native 二进制覆盖。
