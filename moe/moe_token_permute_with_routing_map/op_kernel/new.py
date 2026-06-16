# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

import torch
from typing import Optional
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

try:
    import torch_npu
except ModuleNotFoundError as e:
    logging.warning('import torch_npu failed')


def permute_with_routing_map(
    tokens, # [num_tokens, hidden]
    routing_map, # [num_tokens, num_experts], bool
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.
    """

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    sorted_indices2 = None
    print("48888888888888888888")
    if drop_and_pad and not (num_out_tokens is None):
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)
        permuted_probs = None
        if probs is not None:
            # [num_tokens, num_experts] -> num_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        permuted_input = tokens.index_select(0, sorted_indices)
        print(permuted_input.shape,"sssssssssssssssssssssss")

        return permuted_input, permuted_probs, sorted_indices
    else:
        # [num_experts, num_tokens]，与 kernel routingMap.T 一致
        routing_map_et = routing_map.bool().T.contiguous()
        token_indices = (
            torch.arange(num_tokens, device=routing_map_et.device).unsqueeze(0).expand(num_experts, -1)
        )
        token_ids_stream = token_indices.masked_select(routing_map_et)
        topk = num_out_tokens // num_tokens if num_tokens > 0 else 0
        target_len = topk * num_tokens

        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map_et)
            permuted_probs = _pad_first_dim_zeros(permuted_probs, target_len)
        else:
            permuted_probs = None

        if token_ids_stream.numel() != target_len:
            # 非对齐：rank(cumsum) + gather + scatter + indexCopy(scatter 写 token)
            sorted_indices_out = _build_scatter_sorted_indices(routing_map_et, num_tokens, topk)
            permuted_input = _permute_tokens_by_stream_scatter(
                tokens, sorted_indices_out, num_tokens, topk)
        else:
            # 对齐：mrgsort + indexCopy(按 stream 顺序)
            sorted_indices_out = torch.sort(token_ids_stream.float(), stable=True)[1].to(torch.int32)
            permuted_input = tokens.index_select(0, token_ids_stream)

        return permuted_input, permuted_probs, sorted_indices_out

def enforce_topk_per_row(routingMap, topK):
    """
    修改routingMap，使每行只保留topK个1/True，其余为0/False。
    :param routingMap: shape (tokens_num, experts_num), dtype为int8或bool
    :param topK: int
    :return: 修改后的routingMap，保持输入的dtype
    """
    tokens_num, experts_num = routingMap.shape
    routingMap = routingMap.clone()
    if routingMap.dtype == torch.int8:
        routingMap_float = routingMap.float()
    elif routingMap.dtype == torch.bool:
        routingMap_float = routingMap.float()
    elif routingMap.dtype == torch.uint8:
        routingMap_float = routingMap.float()
    else:
        raise ValueError("dtype must be int8 or bool")
    topk_idx = torch.topk(routingMap_float, k=topK, dim=1, sorted=False)[1]
    mask = torch.zeros_like(routingMap_float)
    mask.scatter_(1, topk_idx, 1)

    if routingMap.dtype == torch.int8:
        routingMap_result = mask.to(torch.int8)
    elif routingMap.dtype == torch.uint8:
        routingMap_result = mask.to(torch.uint8)
    else:
        routingMap_result = mask.to(torch.bool)
    return routingMap_result


def generate_sparse_routing_map(tokens_num, experts_num, topk, dtype=torch.int8):
    """
    生成 routing_map，使每个 token 随机选中 [1, topk] 个专家（可少于 topk）。
    shape: [tokens_num, experts_num]
    """
    topk = max(1, min(int(topk), experts_num))
    if dtype == torch.bool:
        routing_map = torch.zeros((tokens_num, experts_num), dtype=torch.bool)
        for i in range(tokens_num):
            k = int(torch.randint(1, topk + 1, (1,)).item())
            ind = torch.randperm(experts_num)[:k]
            routing_map[i, ind] = True
    elif dtype == torch.int8:
        routing_map = torch.zeros((tokens_num, experts_num), dtype=torch.int8)
        for i in range(tokens_num):
            k = int(torch.randint(1, topk + 1, (1,)).item())
            ind = torch.randperm(experts_num)[:k]
            routing_map[i, ind] = 1
    elif dtype == torch.uint8:
        routing_map = torch.zeros((tokens_num, experts_num), dtype=torch.uint8)
        for i in range(tokens_num):
            k = int(torch.randint(1, topk + 1, (1,)).item())
            ind = torch.randperm(experts_num)[:k]
            routing_map[i, ind] = 1
    else:
        raise ValueError("dtype must be int8, uint8 or bool")
    return routing_map


def _pad_first_dim_zeros(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """首维不足 target_len 时尾部补 0；已足够则截断到 target_len。"""
    n = tensor.shape[0]
    if n >= target_len:
        return tensor[:target_len]
    pad_shape = (target_len - n,) + tensor.shape[1:]
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=0)


def _build_scatter_sorted_indices(routing_map_et, num_tokens, topk):
    """
    非对齐 cumsum+scatter 路径 sortedIndicesOut（与 kernel scatter 一致）。
    sortPosition[e,t] = t*topk + sum(mask[e',t], e'<e)
    valueStream = masked_select(sortPosition); out[slot] = streamIdx
    """
    print("_build_scatter_sorted_indices")

    mask_int = routing_map_et.to(torch.int32)
    prefix = torch.cumsum(mask_int, dim=0) - mask_int
    token_base = torch.arange(num_tokens, device=routing_map_et.device).unsqueeze(0) * topk
    sort_position = (token_base + prefix).to(torch.int32)
    value_stream = sort_position.masked_select(routing_map_et)

    total = num_tokens * topk
    out = torch.full((total,), -1, dtype=torch.int32, device=routing_map_et.device)
    stream_idx = torch.arange(value_stream.numel(), device=routing_map_et.device, dtype=torch.int32)
    slots = value_stream
    valid = (slots >= 0) & (slots < total)
    out[slots[valid]] = stream_idx[valid]
    print("sortedIndicesOut",out)
    return out


def _permute_tokens_by_stream_scatter(tokens, sorted_indices_out, num_tokens, topk):
    """
    indexCopy(ifNumOutTokens=true): permuteTokensOut[streamIdx] = tokens[tokenId]。
    sorted_indices_out[slot] 为 maskselect 一维流下标，slot 按 token 分组 [t*topk, (t+1)*topk)。
    """
    total = num_tokens * topk
    out = torch.zeros((total, tokens.shape[1]), dtype=tokens.dtype, device=tokens.device)
    slots = torch.arange(total, device=tokens.device, dtype=torch.long)
    token_ids = slots // topk
    valid = sorted_indices_out >= 0
    stream_idx = sorted_indices_out[valid].to(torch.long)
    out[stream_idx] = tokens[token_ids[valid]]
    return out


def build_padded_sorted_indices(routing_map_et, num_tokens, topk):
    """
    非对齐场景 sortedIndicesOut 金标（与 kernel PadOutput / tokencopy 一致）。
    routing_map_et: [num_experts, num_tokens]
    对 maskselect 一维流按 token id stable sort，每组 topk 槽写入排序后的 position，不足填 -1。
    输出 shape: [num_tokens * topk]
    """
    routing_map_et = routing_map_et.bool()
    num_experts = routing_map_et.shape[0]
    token_indices = (
        torch.arange(num_tokens, device=routing_map_et.device).unsqueeze(0).expand(num_experts, -1)
    )
    flat_token_ids = token_indices.masked_select(routing_map_et)
    positions = torch.arange(flat_token_ids.numel(), device=routing_map_et.device)
    _, order = torch.sort(flat_token_ids.float(), stable=True)
    sorted_token_ids = flat_token_ids[order]
    sorted_pos = positions[order]

    out = torch.full((num_tokens * topk,), -1, dtype=torch.int32, device=routing_map_et.device)
    for t in range(num_tokens):
        sel = sorted_pos[sorted_token_ids == t]
        n = min(sel.numel(), topk)
        if n > 0:
            out[t * topk : t * topk + n] = sel[:n].to(torch.int32)
    return out




@register("function_aclnn_moe_token_permute_with_routing_map")
class FunctionMoeTokenPermuteWithRoutingMapApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionMoeTokenPermuteWithRoutingMapApi, self).__init__(task_result)


    def init_by_input_data(self, input_data: InputDataset):
        OpsDataset.seed_everything(2025)

        numOutTokens = input_data.kwargs['numOutTokens']
        num_token =  input_data.kwargs['tokens'].shape[0]
        
        routingMap = input_data.kwargs['routingMap']
        print(input_data.kwargs['routingMap'])

        expertNum =  routingMap.shape[1]

        if routingMap.dtype == torch.int8:
            routingMap = torch.randint(0, 2, routingMap.shape, dtype=torch.int8)
        elif routingMap.dtype == torch.bool:
            routingMap = torch.randint(0, 2, routingMap.shape, dtype=torch.bool)
        elif routingMap.dtype == torch.uint8:
            routingMap = torch.randint(0, 2, routingMap.shape, dtype=torch.uint8)
        else:
            raise ValueError("routingMap dtype 只支持 int8 或 bool 类型。")
        print(input_data.kwargs['routingMap'])

        input_data.kwargs['routingMap'] = routingMap
        print(input_data.kwargs['routingMap'])
        print("22o44ut33333333333333333333333")

        if input_data.kwargs['padded_mode'] == False:
            if num_token != 0:
                divresult = numOutTokens // num_token
            else:
                divresult = 0
            print("16933333333")

            topk = divresult
            if divresult == 0:
                topk = max(divresult, 1)
                input_data.kwargs["numOutTokens"] = topk * num_token
            # 每 token 随机选中 [1, topk] 个专家，用于验证 actualOutTokens < topK * numTokens
            dtype = input_data.kwargs['routingMap'].dtype
            input_data.kwargs['routingMap'] = generate_sparse_routing_map(
                num_token, expertNum, topk, dtype=dtype
            )
        else:
            if numOutTokens // expertNum == 0 :
                cap = max(numOutTokens // expertNum,1)
                input_data.kwargs["numOutTokens"] = cap * expertNum
                print("180 77878")

        print("22out33333333333333333333333")
        print(self.device)
        if self.device == 'pyaclnn' or self.device == 'npu':
            input_data.kwargs['tokens'] = input_data.kwargs['tokens'].npu()
            input_data.kwargs['routingMap'] = input_data.kwargs['routingMap'].npu()
            input_data.kwargs['probsOptional'] = input_data.kwargs['probsOptional'].npu()
        print(input_data.kwargs['routingMap'])
        print("out33333333333333333333333")

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        tokens = input_data.kwargs["tokens"]
        routingMap = input_data.kwargs["routingMap"]
        probsOptional = input_data.kwargs['probsOptional']
        numOutTokens = input_data.kwargs['numOutTokens']
        padded_mode = input_data.kwargs['padded_mode']
        
        permutedTokensOut,permuteProbsOutOptional,sortedIndicesOut = permute_with_routing_map(tokens,routingMap,probsOptional,numOutTokens,padded_mode)

        return permutedTokensOut,permuteProbsOutOptional,sortedIndicesOut.to(dtype=torch.int32)
        
    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(   \
        const aclTensor *tokens,const aclTensor *routingMap, const aclTensor *probsOptional,  int64_t numOutTokens, \
        bool dropAndPad, aclTensor *permuteTokensOut, aclTensor *permuteProbsOutOptional,   \
        aclTensor *sortedIndicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)" 

@register("function_pyaclnn_moe_token_permute_with_routing_map")
class PyAclnnMoeTokenPermuteWithRoutingMap(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)

    def after_call(self, output_packages):
        token,probs,sortedIndices = super().after_call(output_packages)
        maxt = (sortedIndices != -1).sum().item()
        
        token[maxt:,:] = 0
        probs[maxt:] = 0
        print("sortedIndices",sortedIndices)
        print("probs",probs)
        return token,probs,sortedIndices
