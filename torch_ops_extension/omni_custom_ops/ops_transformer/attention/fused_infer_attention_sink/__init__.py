# import torch
# import torch_npu
# import torchair

# from torchair._acl_concrete_graph.acl_graph import _REPLACE_FUNC_MAP, StaticWorkspaceReplaceFunc
# if hasattr(torch.ops.custom, "npu_fused_infer_attention_sink"):
#     _REPLACE_FUNC_MAP.update({torch.ops.custom.npu_fused_infer_attention_sink.default: StaticWorkspaceReplaceFunc(
#         get_workspace=torch.ops.custom._npu_fused_infer_attention_sink_get_max_workspace.default,
#         out_operator=torch.ops.custom.npu_fused_infer_attention_sink.out,
#         workspace_keys=["workspace"],
#         output_keys=["attention_out", "softmax_lse"],
#         updated_param_keys=["actual_seq_qlen", "actual_seq_kvlen"],
#     )})