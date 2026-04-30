import torch
import torch_npu
import torchair

from torchair._acl_concrete_graph.acl_graph import _REPLACE_FUNC_MAP, StaticWorkspaceReplaceFunc
if hasattr(torch.ops.custom, "npu_esa_select_topk"):
    _REPLACE_FUNC_MAP.update({torch.ops.custom.npu_esa_select_topk.default: StaticWorkspaceReplaceFunc(
        get_workspace=torch.ops.custom._npu_esa_select_topk_get_max_workspace.default,
        out_operator=torch.ops.custom.npu_esa_select_topk.out,
        workspace_keys=["workspace"],
        output_keys=["topk_indices"],
        updated_param_keys=["actual_seq_q_len_optional", "actual_seq_k_len_optional", "actual_cmp_seq_k_len_optional"],
    )})