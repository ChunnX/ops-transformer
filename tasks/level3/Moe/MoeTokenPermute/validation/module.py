import torch
import torch_npu
import kernel_gen_ops


def permute_with_padded_tokens(tokens, indices):
    permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1).long())
    return permuted_tokens, indices


def permute(tokens, indices, num_out_tokens, padded_mode):
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1).long()

    sorted_indices = torch.argsort(flatten_indices, stable=True)
    sorted_indices1 = torch.argsort(sorted_indices, stable=True)

    if num_out_tokens is not None and num_out_tokens != 0:
        sorted_indices = sorted_indices[: int(num_out_tokens)]
    s_k = sorted_indices // topk
    permuted_tokens = tokens.index_select(0, s_k)
    return permuted_tokens, sorted_indices1


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tokens, indices, num_out_tokens, padded_mode=False):
        if hasattr(num_out_tokens, "item"):
            num_out_tokens = int(num_out_tokens.item())
        else:
            num_out_tokens = int(num_out_tokens)
        permuted_tokens, sorted_indices = permute(tokens, indices, num_out_tokens, bool(padded_mode))
        return permuted_tokens, sorted_indices.to(torch.int32)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, tokens, indices, num_out_tokens, padded_mode=False):
        target = tokens.device if tokens.device.type == "npu" else torch.device("npu", torch.npu.current_device())
        tokens = tokens.to(target)
        indices = indices.to(target)
        if hasattr(num_out_tokens, "item"):
            num_out_tokens = int(num_out_tokens.item())
        else:
            num_out_tokens = int(num_out_tokens)
        padded_mode = bool(padded_mode)
        permuted_tokens, sorted_indices = kernel_gen_ops.moe_token_permute(
            tokens, indices, num_out_tokens, padded_mode
        )
        return permuted_tokens, sorted_indices.to(torch.int32)


expert_matcher = {
    "float32": {"rtol": 1.0e-5, "atol": 1.0e-5, "total_atol": 1.0e-4},
    "float16": {"rtol": 1.0e-3, "atol": 1.0e-3, "total_atol": 1.0e-3},
    "bfloat16": {"rtol": 1.0e-2, "atol": 1.0e-2, "total_atol": 1.0e-2},
}


def get_model(data_format="channels_first", **kwargs):
    return Model()


def get_model_new(data_format="channels_first", **kwargs):
    return ModelNew()


def get_matcher(dtype="float32", **kwargs):
    return expert_matcher.get(dtype, expert_matcher["float32"])
