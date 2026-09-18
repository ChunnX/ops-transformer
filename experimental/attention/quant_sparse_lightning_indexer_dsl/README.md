# Quant Sparse Lightning Indexer

QSLI 在一个融合 Kernel 中完成离散 candidate gather、MXFP4 QK、BF16 权重归约、mask、流式 TopK 和可选 LD 归并。metadata 决定 LI/LD 分核；输出索引保持逻辑 token 语义。

## 融合 K 接口

按照最新 `qsli.xlsx`，PA_BBND 的 `k` 同时携带 K 和 E8M0 scale。接口新增可选 `descale_k=None`，仅为后续 K=TND 路径预留；当前 PA_BBND 路径必须保持 None。

| 输入 | dtype | 物理 shape |
|---|---|---|
| q | uint8，打包 MXFP4 | `(T1,32,64)` |
| k | uint8，K/scale 融合 | `(physical_pages, block_size/8, 544)` |
| w | float32 | `(T1,32)` |
| descale_q | float8_e8m0fnu 或 uint8 位表示 | `(T1,32,2,2)` |
| candidate_block_indices | int32 | `(T1,1,2048)` |
| candidate_block_length | int32 | `(T1,1)` |

逻辑 D=128，N1=32，N2=1。每个 candidate block 包含 8 个 token，连续存储 `K[8,64]` 的 512B，随后是 `scale[8,4]` 的 32B。PA block_size 支持 16 的倍数且不超过 1024。融合 K 允许第 0 轴带 padding，内部两轴必须连续。

测试数据的打包示例（在算子调用及性能计时之外完成）：

```python
packed_k = torch.cat((
    key_bytes.reshape(pages, block_size // 8, 512),
    key_scale_bytes.reshape(pages, block_size // 8, 32),
), dim=-1)
```

算子接口内部不会调用额外的打包或解包小算子。V0 读取融合数据，分别以 ND 写入 K 与 scale staging，Cube 从 staging 读取并完成相应格式转换。

## Torch 调用

```python
from cann_ops_transformer.ops.ds41 import (
    quant_sparse_lightning_indexer,
    quant_sparse_lightning_indexer_metadata,
)

metadata = quant_sparse_lightning_indexer_metadata(
    candidate_block_length,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    cmp_residual_k=cmp_residual_k,
    num_heads_q=32, num_heads_k=1, head_dim=128,
    topk=512, quant_mode=1, candidate_block_size=8,
    max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
    mask_mode=3, cmp_ratio=2,
    layout_q="TND", layout_k="PA_BBND",
)
indices, values = quant_sparse_lightning_indexer(
    q, packed_k, weights, descale_q,
    candidate_block_indices, candidate_block_length,
    512, 1, 8,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q, seqused_k=seqused_k,
    cmp_residual_k=cmp_residual_k,
    block_table=block_table, metadata=metadata,
    output_idx_offset=output_idx_offset,
    max_seqlen_q=max_seqlen_q,
    mask_mode=3, cmp_ratio=2,
    layout_q="TND", layout_kv="PA_BBND",
    return_value=False,
)
```

metadata 是 NPU int32 `[1024]`，由 AICPU 算子生成，可在输入长度和调度参数不变时复用。未提供时，DSL wrapper 会生成 metadata。metadata 同时表示 LD 与非 LD 分核，主计算消费这些边界。

可选 Tensor 为 `descale_k`、`cu_seqlens_q`、`cu_seqlens_k`、`seqused_q`、`seqused_k`、`cmp_residual_k`、`block_table`、`output_idx_offset` 和 `metadata`。`B>1` 时需要 cu_seqlens_q 和 block_table；`mask_mode=3` 且 `cmp_ratio!=1` 时需要 cmp_residual_k，其他组合传 None。

属性：topk 在 `[1,8192]`；quant_mode=1；candidate_block_size=8；mask_mode=0/3；cmp_ratio 在 `[1,128]`；max_seqlen_q 为 -1 或非负数。所有 Tensor 应位于同一 NPU。

## 输出及索引

- indices：int32 `(T1,1,topk)`，无效位置为 -1。
- return_value=True 时 values 为 BF16 `(T1,1,topk)`；False 时返回空 Tensor `(0,)`。
- output_idx_offset 仅作用于有效索引。有效 token 不足 topk 时按无效值填充。
- candidate_block_length 指有效前缀长度，范围为 `[0,2048]`。候选 block id 对应连续 8 个逻辑 token；后缀不参与计算。TopK 不承诺相同分数的稳定排序。

PA 地址为 `physical_page * k.stride(0) + block_within_page * 544`；physical_page 由 block_table 映射，无 block_table 时仅支持单 Batch 的恒等映射。

## 编译与测试

T1、B、physical_pages、block_table 宽度和 K 的第 0 轴 stride 为动态量。mask_mode、cmp_ratio、PA block_size、topk 和可选参数存在性参与静态配置。默认导出覆盖 mask_mode=0/3、cmp_ratio=1/2、PA=64/128、output_idx_offset 有/无；其他受支持配置需要相应静态产物或使用环境允许的 JIT。

原始 pytest 的 20 条 Excel 用例通过真实 Torch 接口调用 metadata 和计算算子。新生成 PT 使用融合 K；旧 PT 由测试适配器在 CPU 打包后上传，保留原有独立 golden 及比较器。`test_quant_sparse_lightning_indexer_generalization.py` 补充静态组合、非连续 PA、零候选及尾块检查。

## 2026-09-18 接口更新

计算和 metadata 接口均新增可选 `cu_seqlens_k=None`。该参数在当前 PA_BBND 路径中不参与寻址和长度计算，仍由 `block_table`、`seqused_k` 控制；本次没有增加 K=TND 的实现。

接口表规定 `layout_kv`（计算）/`layout_k`（metadata）默认 `TND`，因此当前可执行调用须显式指定 `PA_BBND`。metadata 默认 `mask_mode=0`、`cmp_ratio=1`；causal 压缩场景须显式传入 `3`、`2`。`max_seqlen_q/max_seqlen_k` 继续默认 `-1`。metadata 的 PrivateUse1、fallback 与 fake 注册保留。


## Batch 一致性

算子读取 `torch_npu.npu._get_deterministic_level()`：等于 `3` 时，V0 按 candidate 原始顺序逐块调用 `mem_copy`，每块搬入 544B（8 个 token 的 K 与内联 scale）。开关作为运行时 int32 参数传入，与普通路径共享静态编译版本，不新增公开 Torch 参数。

普通路径尝试将相邻两个 candidate 块合并为一条 `ascvec.copy_gm2ub`。仅当第二块物理地址位于第一块之后且 DMA 间距合法时合并；倒序、重复地址等情况回退为两条单行搬运。不会按物理地址交换候选顺序。

`pytest/test_quant_sparse_lightning_indexer_generalization.py` 直接使用 level=3；若运行时不支持则明确跳过，不映射或替换算子开关。测试逐字节比较相同样本在 B=1/2/12/32、PA64/128 下的索引与分数，检查两条路径共享编译对象；不以误差阈值代替 batch 一致性检查。

本次 kernel ABI 新增运行时标量，部署时须重新导出匹配本版源码的 native 二进制，不能直接沿用旧包。开关值变化本身不触发新的静态编译。
