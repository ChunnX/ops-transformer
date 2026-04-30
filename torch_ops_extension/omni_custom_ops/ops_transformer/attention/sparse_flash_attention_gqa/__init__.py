import torch
import torch_npu
import torchair

from torchair._acl_concrete_graph.acl_graph import _REPLACE_FUNC_MAP, StaticWorkspaceReplaceFunc
if hasattr(torch.ops.custom, "npu_sparse_flash_attention_gqa"):
    _REPLACE_FUNC_MAP.update({torch.ops.custom.npu_sparse_flash_attention_gqa.default: StaticWorkspaceReplaceFunc(
        get_workspace=torch.ops.custom.npu_sparse_flash_attention_gqa_get_max_workspace.default,
        out_operator=torch.ops.custom.npu_sparse_flash_attention_gqa.out,
        workspace_keys=["workspace"],
        output_keys=["attention_out", "softmax_lse"],
        updated_param_keys=["actual_seq_lengths_query", "actual_seq_lengths_kv"],
    )})
