# deterministic level 3 两段式完整调用示例

本文给出 SMLA 与 MQSMLA 在 pytest PT 加载后的完整 eager 调用。重点不是手写新接口，而是说明
level 3 的作用域：必须从 metadata 前置算子开始，一直覆盖主算子和同步点，最后恢复旧 level。

## 1. SMLA：metadata + 主算子

```python
import torch
import torch_npu


def optional_npu(value):
    return value.npu() if value is not None else None


def run_smla_level3(input_data):
    params = input_data["params"]
    md = input_data["metadata_input"]
    x = input_data["input"]

    old_level = int(torch_npu.npu._get_deterministic_level())
    try:
        torch_npu.npu.set_deterministic_level(3)
        assert int(torch_npu.npu._get_deterministic_level()) == 3

        q = x["q"].npu()
        ori_kv = optional_npu(x["ori_kv"])
        cmp_kv = optional_npu(x["cmp_kv"])
        ori_sparse_indices = optional_npu(x["ori_sparse_indices"])
        cmp_sparse_indices = optional_npu(x["cmp_sparse_indices"])
        ori_block_table = optional_npu(x["ori_block_table"])
        cmp_block_table = optional_npu(x["cmp_block_table"])
        cu_seqlens_q = optional_npu(x["cu_seqlens_q"])
        cu_seqlens_ori_kv = optional_npu(x["cu_seqlens_ori_kv"])
        cu_seqlens_cmp_kv = optional_npu(x["cu_seqlens_cmp_kv"])
        seqused_q = optional_npu(x["seqused_q"])
        seqused_ori_kv = optional_npu(x["seqused_ori_kv"])
        seqused_cmp_kv = optional_npu(x["seqused_cmp_kv"])
        cmp_residual_kv = optional_npu(x["cmp_residual_kv"])
        sinks = x["sinks"].npu()

        # 第一段：level 3 必须在调用前已生效。
        metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
            num_heads_q=md["N1"],
            num_heads_kv=md["N2"],
            head_dim=md["D"],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=None,
            cmp_topk_length=None,
            batch_size=md["B"],
            max_seqlen_q=md["max_seqlen_q"],
            max_seqlen_ori_kv=md["max_seqlen_ori_kv"],
            max_seqlen_cmp_kv=md["max_seqlen_cmp_kv"],
            ori_topk=md["K"] if ori_sparse_indices is not None else 0,
            cmp_topk=md["K"] if cmp_sparse_indices is not None else 0,
            cmp_ratio=md["cmp_ratio"] if md["cmp_ratio"] is not None else 1,
            ori_mask_mode=x["ori_mask_mode"],
            cmp_mask_mode=x["cmp_mask_mode"] if x["cmp_mask_mode"] is not None else 3,
            ori_win_left=x["ori_win_left"],
            ori_win_right=x["ori_win_right"],
            layout_q=x["layout_q"],
            layout_kv=x["layout_kv"],
            has_ori_kv=ori_kv is not None,
            has_cmp_kv=cmp_kv is not None,
        )

        # 第二段：必须原样传入第一段生成的同一个 metadata tensor。
        output, softmax_lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=ori_sparse_indices,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q if x["layout_q"] == "TND" else None,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=x["softmax_scale"],
            cmp_ratio=x.get("cmp_ratio", md["cmp_ratio"] or 1),
            ori_mask_mode=x["ori_mask_mode"],
            cmp_mask_mode=x["cmp_mask_mode"] if x["cmp_mask_mode"] is not None else 0,
            ori_win_left=x["ori_win_left"],
            ori_win_right=x["ori_win_right"],
            layout_q=x["layout_q"],
            layout_kv=x["layout_kv"],
            topk_value_mode=x.get("topk_value_mode", 1),
            return_softmax_lse=params.get("return_softmax_lse", False),
        )
        torch.npu.synchronize()
        return output, softmax_lse, metadata.detach().cpu()
    finally:
        if int(torch_npu.npu._get_deterministic_level()) != old_level:
            torch_npu.npu.set_deterministic_level(old_level)
```

仓库中的实际入口是
`attention/sparse_flash_mla/tests/pytest_v2/batch/sparse_flash_mla_consistency.py`。它复用
`sparse_flash_mla_process.call_npu`，而 `call_npu` 内部就是上述两段，因此外层 guard 实际覆盖
metadata 与主算子，不是只覆盖第二段。

## 2. MQSMLA：metadata + 主算子

```python
import torch
import torch_npu


def optional_npu(value, empty_if_none=False):
    if value is None:
        return torch.tensor([], dtype=torch.int32).npu() if empty_if_none else None
    return value.npu()


def run_mqsmla_level3(input_data):
    md = input_data["metadata_input"]
    x = input_data["op_input"]

    old_level = int(torch_npu.npu._get_deterministic_level())
    try:
        torch_npu.npu.set_deterministic_level(3)
        assert int(torch_npu.npu._get_deterministic_level()) == 3

        cu_seqlens_q = optional_npu(x["cu_seqlens_q"], empty_if_none=True)
        cu_seqlens_ori_kv = optional_npu(x["cu_seqlens_ori_kv"], empty_if_none=True)
        cu_seqlens_cmp_kv = optional_npu(x["cu_seqlens_cmp_kv"], empty_if_none=True)
        seqused_q = optional_npu(x["seqused_q"], empty_if_none=True)
        seqused_ori_kv = optional_npu(x["seqused_ori_kv"])
        seqused_cmp_kv = optional_npu(x["seqused_cmp_kv"])
        cmp_residual_kv = optional_npu(x["cmp_residual_kv"])

        # 第一段：MQ metadata 比 SMLA 多 quant_mode 和 rope_head_dim。
        metadata = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
            num_heads_q=md["num_heads_q"],
            num_heads_kv=md["num_heads_kv"],
            head_dim=md["head_dim"],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=None,
            cmp_topk_length=None,
            quant_mode=x["quant_mode"],
            batch_size=md["batch_size"],
            max_seqlen_q=md["max_seqlen_q"],
            max_seqlen_ori_kv=md["max_seqlen_ori_kv"],
            max_seqlen_cmp_kv=md["max_seqlen_cmp_kv"],
            ori_topk=0,
            cmp_topk=md["topk"],
            rope_head_dim=x["rope_head_dim"],
            cmp_ratio=md["cmp_ratio"],
            ori_mask_mode=md["ori_mask_mode"],
            cmp_mask_mode=md["cmp_mask_mode"],
            ori_win_left=md["ori_win_left"],
            ori_win_right=md["ori_win_right"],
            layout_q=md["layout_q"],
            layout_kv=md["layout_kv"],
            has_ori_kv=md["has_ori_kv"],
            has_cmp_kv=md["has_cmp_kv"],
        )

        # 第二段：量化 KV、scale/padding 都已包含在 ori_kv/cmp_kv 的打包格式中。
        output, softmax_lse = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
            q=optional_npu(x["q"]),
            ori_kv=optional_npu(x["ori_kv"]),
            cmp_kv=optional_npu(x["cmp_kv"]),
            ori_sparse_indices=None,
            cmp_sparse_indices=optional_npu(x["cmp_sparse_indices"]),
            ori_block_table=optional_npu(x["ori_block_table"]),
            cmp_block_table=optional_npu(x["cmp_block_table"]),
            cu_seqlens_q=(
                cu_seqlens_q if x["layout_q"] == "TND" and x["cu_seqlens_q"] is not None else None
            ),
            cu_seqlens_ori_kv=(
                cu_seqlens_ori_kv if x["cu_seqlens_ori_kv"] is not None else None
            ),
            cu_seqlens_cmp_kv=(
                cu_seqlens_cmp_kv if x["cu_seqlens_cmp_kv"] is not None else None
            ),
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=optional_npu(x["sinks"]),
            metadata=metadata,
            quant_mode=x["quant_mode"],
            rope_head_dim=x["rope_head_dim"],
            softmax_scale=x["softmax_scale"],
            cmp_ratio=x["cmp_ratio"],
            ori_mask_mode=x["ori_mask_mode"],
            cmp_mask_mode=x["cmp_mask_mode"],
            ori_win_left=x["ori_win_left"],
            ori_win_right=x["ori_win_right"],
            layout_q=x["layout_q"],
            layout_kv=x["layout_kv"],
            topk_value_mode=x.get("topk_value_mode", 1),
            return_softmax_lse=x.get("return_softmax_lse", False),
            key_dtype=None,
            value_dtype=None,
        )
        torch.npu.synchronize()
        return output, softmax_lse, metadata.detach().cpu()
    finally:
        if int(torch_npu.npu._get_deterministic_level()) != old_level:
            torch_npu.npu.set_deterministic_level(old_level)
```

实际入口是
`attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/batch/mixed_quant_sparse_flash_mla_consistency.py`；
MQ process 新增的可选 `metadata_sink` 不改变原 `(npu_result, cpu_output)` 返回契约。

## 3. 当前环境的预期结果

本机 setter 后 getter 可以短暂显示 3，但当前 torch_npu/CANN runtime 会在第一条 NPU 操作时
延迟报错：

```text
'torch_npu.npu.set_deterministic_level' currently only supports configuring 0/1/2 !
```

这应报告为 `UNSUPPORTED`，不得降回 level 0 后继续并声称 PASS。`finally` 仍会把旧 level 恢复。
