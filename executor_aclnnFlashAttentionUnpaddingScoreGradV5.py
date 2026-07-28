#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes
import copy
import logging
import math

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr


def _is_empty_tensor(value):
    return isinstance(value, torch.Tensor) and value.dim() == 0


def _is_present_tensor(value):
    return isinstance(value, torch.Tensor) and value.dim() > 0 and value.numel() > 0


def _null_tensor_ptr():
    acl_tensor_ptr = ctypes.POINTER(AclTensor)
    return ctypes.cast(ctypes.c_void_p(None), acl_tensor_ptr)


def _null_void_ptr():
    return ctypes.c_void_p(None)


def _dtype_name(tensor):
    return str(tensor.dtype) if torch.is_tensor(tensor) else None


def _set_output_info_dtype(output_info, dtype_name):
    if output_info is None or dtype_name is None:
        return output_info
    if isinstance(output_info, dict):
        output_info["dtype"] = dtype_name
        return output_info
    if isinstance(output_info, list):
        for item in output_info:
            _set_output_info_dtype(item, dtype_name)
        return output_info
    if hasattr(output_info, "dtype"):
        output_info.dtype = dtype_name
    return output_info


def _prepare_aclnn_output_info(task_result, input_data):
    output_info_list = copy.deepcopy(task_result.output_info_list)
    dtype_by_output = {
        0: _dtype_name(input_data.kwargs["query"]),
        1: _dtype_name(input_data.kwargs["queryRope"]),
        2: _dtype_name(input_data.kwargs["keyIn"]),
        3: _dtype_name(input_data.kwargs["keyInRope"]),
        4: _dtype_name(input_data.kwargs["value"]),
        5: _dtype_name(input_data.kwargs["pseShiftOptional"]),
        6: _dtype_name(input_data.kwargs["sinkInOptional"]),
    }
    for index, dtype_name in dtype_by_output.items():
        if 0 <= index < len(output_info_list):
            _set_output_info_dtype(output_info_list[index], dtype_name)
    task_result.output_info_list = output_info_list
    logging.info("GradV5 aclnn output dtypes: %s", [getattr(x, "dtype", None) for x in output_info_list])


def _actual_seq_to_cu(actual_seq, total):
    actual_seq = [int(x) for x in actual_seq]
    if not actual_seq:
        return [0, int(total)]
    if actual_seq[-1] == int(total):
        return [0] + actual_seq
    cu_seq = [0]
    for seq_len in actual_seq:
        cu_seq.append(cu_seq[-1] + seq_len)
    return cu_seq


def _scale_value(input_data):
    scale = float(input_data.kwargs["scaleValue"])
    if scale == 0:
        scale = 1.0 / math.sqrt(int(input_data.kwargs["query"].shape[-1]))
        input_data.kwargs["scaleValue"] = scale
    return scale


def _to_compute_dtype(tensor):
    return tensor.to(torch.float64) if torch.is_tensor(tensor) else tensor


def _broadcast_kv(kv, head_num):
    kv_head_num = kv.shape[1]
    if head_num == kv_head_num:
        return kv
    if head_num % kv_head_num != 0:
        raise RuntimeError(f"headNum {head_num} must be divisible by kv head num {kv_head_num}")
    return kv.repeat_interleave(head_num // kv_head_num, dim=1)


def _seq_slices(tensor, q0, q1, k0, k1, sparse_mode):
    sq = q1 - q0
    sk = k1 - k0
    mask_sq = tensor.shape[-2]
    mask_sk = tensor.shape[-1]
    if mask_sq >= q1 and mask_sk >= k1:
        return slice(q0, q1), slice(k0, k1)
    if sparse_mode in (3, 4, 5):
        return slice(mask_sq - sq, mask_sq), slice(mask_sk - sk, mask_sk)
    return slice(0, sq), slice(0, sk)


def _slice_mask(mask, batch_idx, q0, q1, k0, k1, batch_size, sparse_mode):
    if not _is_present_tensor(mask):
        return None
    q_slice, k_slice = _seq_slices(mask, q0, q1, k0, k1, sparse_mode)
    if mask.dim() == 2:
        return mask[q_slice, k_slice].unsqueeze(0)
    if mask.dim() == 3:
        b_idx = batch_idx if mask.shape[0] == batch_size else 0
        return mask[b_idx:b_idx + 1, q_slice, k_slice]
    if mask.dim() == 4:
        b_idx = batch_idx if mask.shape[0] == batch_size else 0
        return mask[b_idx, :, q_slice, k_slice]
    raise RuntimeError(f"not support attenMaskOptional shape {tuple(mask.shape)}")


def _pad_compressed_pse_query_dim(pse_i, sq):
    if pse_i is None or pse_i.shape[-2] == sq:
        return pse_i
    if pse_i.shape[-2] > sq:
        return pse_i[..., -sq:, :]
    padded = torch.zeros((*pse_i.shape[:-2], sq, pse_i.shape[-1]), dtype=pse_i.dtype, device=pse_i.device)
    padded[..., -pse_i.shape[-2]:, :] = pse_i
    return padded


def _slice_pse(pse, batch_idx, q0, q1, k0, k1, batch_size, head_num, sparse_mode, pad_query_dim=True):
    if not _is_present_tensor(pse):
        return None
    sq = q1 - q0
    sk = k1 - k0
    if pse.dim() == 1:
        return pse.reshape(head_num, 1, 1)
    if pse.dim() == 2:
        if pse.shape[0] == batch_size and pse.shape[1] == head_num:
            return pse[batch_idx].reshape(head_num, 1, 1)
        q_slice, k_slice = _seq_slices(pse, q0, q1, k0, k1, sparse_mode)
        return pse[q_slice, k_slice].unsqueeze(0)
    if pse.dim() == 4:
        sk0 = pse.shape[-1] - sk
        sk1 = pse.shape[-1]
        if pse.shape[-2] == 1:
            if pse.shape[0] == batch_size:
                return pse[batch_idx, :, :, sk0:sk1]
            return pse[0, :, :, sk0:sk1]
        q_start = pse.shape[-2] - sq
        q_end = pse.shape[-2]
        if pse.shape[0] == batch_size:
            pse_i = pse[batch_idx, :, q_start:q_end, sk0:sk1]
        else:
            pse_i = pse[0, :, q_start:q_end, sk0:sk1]
        if pad_query_dim:
            pse_i = _pad_compressed_pse_query_dim(pse_i, sq)
        return pse_i
    raise RuntimeError(f"not support pseShiftOptional shape {tuple(pse.shape)}")


def _qk_pointer(input_data):
    cu = [0]
    actual_q = _actual_seq_to_cu(input_data.kwargs["actualSeqQLenOptional"], input_data.kwargs["query"].shape[0])
    actual_k = _actual_seq_to_cu(input_data.kwargs["actualSeqKvLenOptional"], input_data.kwargs["keyIn"].shape[0])
    for i in range(len(actual_q) - 1):
        cu.append(cu[-1] + (actual_q[i + 1] - actual_q[i]) * (actual_k[i + 1] - actual_k[i]))
    return cu


def _slice_drop_mask(drop_mask, batch_idx, q0, q1, k0, k1, batch_size, head_num, sparse_mode, qk_pointer=None):
    if not _is_present_tensor(drop_mask):
        return None
    sq = q1 - q0
    sk = k1 - k0
    if drop_mask.dim() == 1:
        if qk_pointer is None:
            raise RuntimeError("qk_pointer is required for 1D TND dropMaskOptional")
        start = qk_pointer[batch_idx] * head_num
        end = qk_pointer[batch_idx + 1] * head_num
        return drop_mask[start:end].reshape(head_num, sq, sk)
    q_slice, k_slice = _seq_slices(drop_mask, q0, q1, k0, k1, sparse_mode)
    if drop_mask.dim() == 2:
        return drop_mask[q_slice, k_slice].unsqueeze(0)
    if drop_mask.dim() == 3:
        if drop_mask.shape[0] == head_num:
            return drop_mask[:, q_slice, k_slice]
        b_idx = batch_idx if drop_mask.shape[0] == batch_size else 0
        return drop_mask[b_idx:b_idx + 1, q_slice, k_slice]
    if drop_mask.dim() == 4:
        b_idx = batch_idx if drop_mask.shape[0] == batch_size else 0
        return drop_mask[b_idx, :, q_slice, k_slice]
    raise RuntimeError(f"not support dropMaskOptional shape {tuple(drop_mask.shape)}")


def _pad_last2(tensor, target_s1, target_s2):
    if tensor is None or _is_empty_tensor(tensor):
        return tensor
    if tensor.dim() < 2 or tensor.shape[-2] >= target_s1 and tensor.shape[-1] >= target_s2:
        return tensor
    out_shape = (*tensor.shape[:-2], max(tensor.shape[-2], target_s1), max(tensor.shape[-1], target_s2))
    padded = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)
    padded[..., :tensor.shape[-2], :tensor.shape[-1]] = tensor
    return padded


def _standard_upper_mask_like(tensor, target_s1=2048, target_s2=2048):
    dtype = tensor.dtype if torch.is_tensor(tensor) else torch.bool
    device = tensor.device if torch.is_tensor(tensor) else torch.device("cpu")
    return torch.triu(torch.ones((target_s1, target_s2), dtype=dtype, device=device), diagonal=1)


def _make_upper_mask(shape, diagonal, dtype, device):
    return torch.triu(torch.ones(shape, dtype=dtype, device=device), diagonal=diagonal)


def _make_band_mask(shape, upper_diagonal, lower_diagonal, dtype, device):
    upper = torch.triu(torch.ones(shape, dtype=dtype, device=device), diagonal=upper_diagonal)
    lower = torch.tril(torch.ones(shape, dtype=dtype, device=device), diagonal=lower_diagonal)
    return (upper + lower).to(dtype)


def _tnd_max_seq_lens(input_data):
    q = input_data.kwargs["query"]
    k = input_data.kwargs["keyIn"]
    cu_q = _actual_seq_to_cu(input_data.kwargs["actualSeqQLenOptional"], q.shape[0])
    cu_k = _actual_seq_to_cu(input_data.kwargs["actualSeqKvLenOptional"], k.shape[0])
    max_q = max(cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1))
    max_k = max(cu_k[i + 1] - cu_k[i] for i in range(len(cu_k) - 1))
    return int(max_q), int(max_k)


def _logical_mask_shape(old_mask, batch_size, head_num, sq, sk):
    if not _is_present_tensor(old_mask):
        return None
    if old_mask.dim() == 2:
        return (sq, sk)
    if old_mask.dim() == 3:
        first_dim = batch_size if old_mask.shape[0] == batch_size else old_mask.shape[0]
        return (first_dim, sq, sk)
    if old_mask.dim() == 4:
        batch_dim = batch_size if old_mask.shape[0] == batch_size else old_mask.shape[0]
        head_dim = head_num if old_mask.shape[1] == head_num else old_mask.shape[1]
        return (batch_dim, head_dim, sq, sk)
    raise RuntimeError(f"not support attenMaskOptional shape {tuple(old_mask.shape)}")


def _make_tnd_atten_mask_from_sparse_mode(input_data, use_npu_mask=False):
    sparse_mode = int(input_data.kwargs["sparseMode"])
    old_mask = input_data.kwargs.get("attenMaskOptional")
    dtype = old_mask.dtype if torch.is_tensor(old_mask) else torch.bool
    device = old_mask.device if torch.is_tensor(old_mask) else torch.device("cpu")
    sq, sk = _tnd_max_seq_lens(input_data)
    batch_size = len(input_data.kwargs["actualSeqQLenOptional"])
    head_num = int(input_data.kwargs["headNum"])
    shape = _logical_mask_shape(old_mask, batch_size, head_num, sq, sk)

    if shape is None:
        input_data.kwargs["attenMaskOptional"] = torch.tensor(0, dtype=dtype, device=device)
        input_data.kwargs["preTokens"] = 65536
        input_data.kwargs["nextTokens"] = 65536
        return

    if sparse_mode == 0:
        pre_token = int(input_data.kwargs["preTokens"])
        next_token = int(input_data.kwargs["nextTokens"])
        new_mask = _make_band_mask(shape, next_token + 1, -pre_token - 1, dtype, device)
    elif sparse_mode == 1:
        new_mask = torch.zeros(shape, dtype=dtype, device=device)
        pre_token = sq
        next_token = sk
    elif sparse_mode == 2:
        new_mask = _make_upper_mask(shape, 1, dtype, device)
        if use_npu_mask:
            new_mask = _make_upper_mask((2048, 2048), 1, dtype, device)
        pre_token = sq
        next_token = 0
    elif sparse_mode == 3:
        new_mask = _make_upper_mask(shape, sk - sq + 1, dtype, device)
        if use_npu_mask:
            new_mask = _make_upper_mask((2048, 2048), 1, dtype, device)
        pre_token = sq
        next_token = 0
    elif sparse_mode == 4:
        pre_token = int(input_data.kwargs["preTokens"])
        next_token = int(input_data.kwargs["nextTokens"])
        new_mask = _make_band_mask(shape, next_token + 1 + sk - sq, -pre_token - 1 + sk - sq, dtype, device)
        if use_npu_mask:
            new_mask = _make_upper_mask((2048, 2048), 1, dtype, device)
    else:
        return

    input_data.kwargs["attenMaskOptional"] = new_mask
    input_data.kwargs["preTokens"] = pre_token
    input_data.kwargs["nextTokens"] = next_token
    logging.info(
        "GradV5 sparseMode=%s use_%s_mask attenMaskOptional shape=%s dtype=%s pre=%s next=%s",
        sparse_mode,
        "npu" if use_npu_mask else "logical",
        tuple(new_mask.shape),
        new_mask.dtype,
        pre_token,
        next_token,
    )


def _flatten_tnd_pse_for_aclnn(input_data):
    pse = input_data.kwargs["pseShiftOptional"]
    if pse is None or _is_empty_tensor(pse) or pse.dim() != 4:
        return pse

    head_num = int(input_data.kwargs["headNum"])
    batch_size = len(input_data.kwargs["actualSeqQLenOptional"])
    if pse.shape[0] != batch_size or pse.shape[1] not in (1, head_num):
        return pse

    sparse_mode = int(input_data.kwargs["sparseMode"])
    flattened = []
    for batch_idx, q0, q1, k0, k1 in _iter_slices(input_data):
        if q1 <= q0 or k1 <= k0:
            continue
        pse_i = _slice_pse(pse, batch_idx, q0, q1, k0, k1, batch_size, head_num, sparse_mode, pad_query_dim=False)
        if pse_i.shape[0] == 1 and head_num != 1:
            pse_i = pse_i.expand(head_num, -1, -1)
        flattened.append(pse_i.contiguous().reshape(-1))
    if not flattened:
        return torch.empty(0, dtype=pse.dtype, device=pse.device)
    return torch.cat(flattened)


def _prepare_aclnn_inputs(input_data):
    _make_tnd_atten_mask_from_sparse_mode(input_data, use_npu_mask=True)
    input_data.kwargs["pseShiftOptional"] = _flatten_tnd_pse_for_aclnn(input_data)


def _move_tensor_inputs_to_npu(input_data):
    npu_tensors = []
    for name, value in input_data.kwargs.items():
        if torch.is_tensor(value) and value.device.type != "npu":
            value = value.npu()
            input_data.kwargs[name] = value
        if torch.is_tensor(value):
            npu_tensors.append(value)
    return npu_tensors


def _optional_tensor(value):
    return value if _is_present_tensor(value) else None


def _optional_int_array(value, enabled=True):
    if not enabled or value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1 and int(value[0]) == 0:
        return None
    return tuple(int(x) for x in value)


def _log_tensor_stats(prefix, tensor):
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        logging.info("%s: empty", prefix)
        return
    detached = tensor.detach()
    if detached.device.type != "cpu":
        detached = detached.cpu()
    finite = torch.isfinite(detached) if detached.is_floating_point() else torch.ones_like(detached, dtype=torch.bool)
    if detached.is_floating_point() and finite.any():
        logging.info(
            "%s shape=%s dtype=%s nan=%d inf=%d min=%s max=%s mean=%s",
            prefix,
            tuple(detached.shape),
            detached.dtype,
            int(torch.isnan(detached).sum().item()),
            int(torch.isinf(detached).sum().item()),
            detached[finite].min().item(),
            detached[finite].max().item(),
            detached[finite].float().mean().item(),
        )
    else:
        logging.info("%s shape=%s dtype=%s numel=%d", prefix, tuple(detached.shape), detached.dtype, detached.numel())


def _log_qkv_shapes(input_data, prefix="GradV5"):
    for name in ("query", "keyIn", "value"):
        tensor = input_data.kwargs.get(name)
        if torch.is_tensor(tensor):
            logging.info(
                "%s %s shape=%s dtype=%s device=%s",
                prefix,
                name,
                tuple(tensor.shape),
                tensor.dtype,
                tensor.device,
            )
        else:
            logging.info("%s %s is not tensor: %s", prefix, name, type(tensor).__name__)


def _prepare_forward_inputs_with_npu(input_data):
    try:
        import torch_npu
    except ImportError:
        logging.info("GradV5 npu forward prepare skipped: torch_npu is not available")
        return

    q = input_data.kwargs["query"]
    if not torch.is_tensor(q) or q.device.type != "npu":
        logging.info("GradV5 npu forward prepare skipped: query is not on npu")
        return
    _log_qkv_shapes(input_data, "GradV5 npu forward input")

    sparse_mode = int(input_data.kwargs["sparseMode"])
    kwargs = {
        "pse": _optional_tensor(input_data.kwargs["pseShiftOptional"]),
        "padding_mask": _optional_tensor(input_data.kwargs["paddingMaskOptional"]),
        "atten_mask": _optional_tensor(input_data.kwargs["attenMaskOptional"]),
        "query_rope": _optional_tensor(input_data.kwargs["queryRope"]),
        "key_rope": _optional_tensor(input_data.kwargs["keyInRope"]),
        "scale": _scale_value(input_data),
        "keep_prob": float(input_data.kwargs["keepProb"]),
        "pre_tokens": int(input_data.kwargs["preTokens"]),
        "next_tokens": int(input_data.kwargs["nextTokens"]),
        "inner_precise": int(input_data.kwargs["innerPrecise"]),
        "prefix": _optional_int_array(input_data.kwargs["prefixOptional"], sparse_mode in (5, 6)),
        "actual_seq_qlen": tuple(int(x) for x in input_data.kwargs["actualSeqQLenOptional"]),
        "actual_seq_kvlen": tuple(int(x) for x in input_data.kwargs["actualSeqKvLenOptional"]),
        "sparse_mode": sparse_mode,
        "sync": True,
        "pse_type": int(input_data.kwargs["pseType"]),
        "q_start_idx": _optional_int_array(input_data.kwargs["qStartIdxOptional"], sparse_mode in (7, 8)),
        "kv_start_idx": _optional_int_array(input_data.kwargs["kvStartIdxOptional"], sparse_mode in (7, 8)),
        "softmax_layout": "TND",
        "sink": _optional_tensor(input_data.kwargs["sinkInOptional"]),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    try:
        forward_out = torch_npu.npu_fusion_attention_v2(
            q,
            input_data.kwargs["keyIn"],
            input_data.kwargs["value"],
            int(input_data.kwargs["headNum"]),
            input_data.kwargs["inputLayout"],
            **kwargs,
        )
    except Exception as exc:
        logging.warning("GradV5 npu forward prepare failed, fallback to CPU intermediates: %s", exc)
        return

    attention, softmax_max, softmax_sum = forward_out[0], forward_out[1], forward_out[2]
    input_data.kwargs["attentionInOptional"] = attention.to(q.dtype).contiguous()
    input_data.kwargs["softmaxMaxOptional"] = softmax_max.to(torch.float32).contiguous()
    input_data.kwargs["softmaxSumOptional"] = softmax_sum.to(torch.float32).contiguous()
    input_data.kwargs["softmaxInLayout"] = ""
    logging.info("GradV5 npu forward prepare succeeded")
    _log_tensor_stats("GradV5 attentionInOptional", input_data.kwargs["attentionInOptional"])
    _log_tensor_stats("GradV5 softmaxMaxOptional", input_data.kwargs["softmaxMaxOptional"])
    _log_tensor_stats("GradV5 softmaxSumOptional", input_data.kwargs["softmaxSumOptional"])


def _tsoftmax(score, sink=None):
    x_max = torch.max(score, dim=-1, keepdim=True)[0]
    sink_expanded = None
    if _is_present_tensor(sink):
        sink_expanded = sink.reshape(1, score.shape[1], 1, 1).to(score.device, score.dtype)
        sink_expanded = sink_expanded.expand(score.shape[0], score.shape[1], score.shape[2], 1)
        x_max = torch.maximum(x_max, sink_expanded)
    y = torch.exp(score - x_max)
    x_sum = y.sum(dim=-1, keepdim=True)
    if sink_expanded is not None:
        x_sum = x_sum + torch.exp(sink_expanded - x_max)
    return y / x_sum, x_max, x_sum


def _tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdim=True)
    return (dp - muls_r) * softmax_res


def _forward_tnd(q, k, v, atten_mask, pse, drop_mask, sink, scale, keep_prob):
    q = _to_compute_dtype(q)
    k = _to_compute_dtype(k)
    v = _to_compute_dtype(v)
    k = _broadcast_kv(k, q.shape[1])
    v = _broadcast_kv(v, q.shape[1])

    q_bnsd = q.permute(1, 0, 2).unsqueeze(0)
    k_bnsd = k.permute(1, 0, 2).unsqueeze(0)
    v_bnsd = v.permute(1, 0, 2).unsqueeze(0)
    score = torch.matmul(q_bnsd, k_bnsd.permute(0, 1, 3, 2))
    if pse is not None:
        score = score + _to_compute_dtype(pse).to(score.device)
    score = score * scale
    if atten_mask is not None:
        score = score + atten_mask.to(score.device, score.dtype) * (-40000.0)

    softmax_res, softmax_max, softmax_sum = _tsoftmax(score, sink)
    if atten_mask is not None:
        mask = atten_mask.to(torch.bool).to(score.device)
        softmax_res = softmax_res.masked_fill(mask.broadcast_to(softmax_res.shape), 0)
    if drop_mask is not None:
        drop_res = softmax_res * drop_mask.to(score.device, score.dtype).broadcast_to(score.shape)
        drop_res = drop_res * (1.0 / keep_prob)
    else:
        drop_res = softmax_res
    out = torch.matmul(drop_res, v_bnsd)
    return out.squeeze(0).permute(1, 0, 2).contiguous(), softmax_res, softmax_max, softmax_sum, k, v


def _backward_from_forward(dy, q, k, v, softmax_res, drop_mask, scale, keep_prob, sink=None, x_max=None, x_sum=None):
    dy = _to_compute_dtype(dy)
    q = _to_compute_dtype(q)
    k = _to_compute_dtype(k)
    v = _to_compute_dtype(v)
    dy_bnsd = dy.permute(1, 0, 2).unsqueeze(0)
    q_bnsd = q.permute(1, 0, 2).unsqueeze(0)
    k_bnsd = k.permute(1, 0, 2).unsqueeze(0)
    v_bnsd = v.permute(1, 0, 2).unsqueeze(0)
    if drop_mask is None:
        drop_mask = torch.ones_like(softmax_res)
    else:
        drop_mask = drop_mask.to(softmax_res.device, softmax_res.dtype).broadcast_to(softmax_res.shape)
    drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob).permute(0, 1, 3, 2)
    dv = torch.matmul(drop_res, dy_bnsd)
    dp = torch.matmul(dy_bnsd, v_bnsd.permute(0, 1, 3, 2))
    dp_drop = dp * drop_mask * (1.0 / keep_prob)
    softmax_grad_res = _tsoftmax_grad(dp_drop, softmax_res) * scale
    dq = torch.matmul(softmax_grad_res, k_bnsd)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q_bnsd)
    dsink = None
    if _is_present_tensor(sink) and x_max is not None and x_sum is not None:
        sink_prob = torch.exp(sink.reshape(1, softmax_res.shape[1], 1, 1).to(x_max.device, x_max.dtype) - x_max) / x_sum
        dsink = -(softmax_res * dp_drop * sink_prob).sum(dim=(0, 2, 3))
    return (
        dq.squeeze(0).permute(1, 0, 2).contiguous(),
        dk.squeeze(0).permute(1, 0, 2).contiguous(),
        dv.squeeze(0).permute(1, 0, 2).contiguous(),
        dsink,
    )


def _iter_slices(input_data):
    q = input_data.kwargs["query"]
    k = input_data.kwargs["keyIn"]
    input_layout = input_data.kwargs["inputLayout"].upper()
    if input_layout != "TND":
        raise RuntimeError(f"aclnnFlashAttentionUnpaddingScoreGradV5 only supports TND, got {input_layout}")
    cu_q = _actual_seq_to_cu(input_data.kwargs["actualSeqQLenOptional"], q.shape[0])
    cu_k = _actual_seq_to_cu(input_data.kwargs["actualSeqKvLenOptional"], k.shape[0])
    if len(cu_q) != len(cu_k):
        raise RuntimeError("actualSeqQLenOptional and actualSeqKvLenOptional must have same batch count")
    for batch_idx in range(len(cu_q) - 1):
        yield batch_idx, cu_q[batch_idx], cu_q[batch_idx + 1], cu_k[batch_idx], cu_k[batch_idx + 1]


def _prepare_forward_inputs(input_data):
    q = input_data.kwargs["query"]
    k = input_data.kwargs["keyIn"]
    v = input_data.kwargs["value"]
    atten_mask = input_data.kwargs["attenMaskOptional"]
    pse = input_data.kwargs["pseShiftOptional"]
    drop_mask = input_data.kwargs["dropMaskOptional"]
    sink = input_data.kwargs["sinkInOptional"]
    head_num = int(input_data.kwargs["headNum"])
    keep_prob = float(input_data.kwargs["keepProb"])
    scale = _scale_value(input_data)
    batch_size = len(input_data.kwargs["actualSeqQLenOptional"])
    sparse_mode = int(input_data.kwargs["sparseMode"])
    qk_pointer = _qk_pointer(input_data)

    attention = torch.zeros_like(input_data.kwargs["attentionInOptional"], dtype=q.dtype)
    softmax_max = torch.zeros((q.shape[0], head_num, 8), dtype=torch.float32, device=q.device)
    softmax_sum = torch.zeros((q.shape[0], head_num, 8), dtype=torch.float32, device=q.device)

    with torch.no_grad():
        for batch_idx, q0, q1, k0, k1 in _iter_slices(input_data):
            if q1 <= q0 or k1 <= k0:
                continue
            mask_i = _slice_mask(atten_mask, batch_idx, q0, q1, k0, k1, batch_size, sparse_mode)
            pse_i = _slice_pse(pse, batch_idx, q0, q1, k0, k1, batch_size, head_num, sparse_mode)
            drop_i = _slice_drop_mask(
                drop_mask, batch_idx, q0, q1, k0, k1, batch_size, head_num, sparse_mode, qk_pointer
            )
            out_i, _, max_i, sum_i, _, _ = _forward_tnd(
                q[q0:q1], k[k0:k1], v[k0:k1], mask_i, pse_i, drop_i, sink, scale, keep_prob
            )
            attention[q0:q1] = out_i.to(q.dtype)
            softmax_max[q0:q1] = (
                max_i.broadcast_to(1, head_num, q1 - q0, 8)
                .permute(0, 2, 1, 3)
                .squeeze(0)
                .to(torch.float32)
            )
            softmax_sum[q0:q1] = (
                sum_i.broadcast_to(1, head_num, q1 - q0, 8)
                .permute(0, 2, 1, 3)
                .squeeze(0)
                .to(torch.float32)
            )

    input_data.kwargs["attentionInOptional"] = attention
    input_data.kwargs["softmaxMaxOptional"] = softmax_max
    input_data.kwargs["softmaxSumOptional"] = softmax_sum


def _backward_tnd(input_data):
    q = input_data.kwargs["query"]
    k = input_data.kwargs["keyIn"]
    v = input_data.kwargs["value"]
    dy = input_data.kwargs["dy"]
    atten_mask = input_data.kwargs["attenMaskOptional"]
    pse = input_data.kwargs["pseShiftOptional"]
    drop_mask = input_data.kwargs["dropMaskOptional"]
    sink = input_data.kwargs["sinkInOptional"]
    head_num = int(input_data.kwargs["headNum"])
    keep_prob = float(input_data.kwargs["keepProb"])
    scale = _scale_value(input_data)
    batch_size = len(input_data.kwargs["actualSeqQLenOptional"])
    sparse_mode = int(input_data.kwargs["sparseMode"])
    qk_pointer = _qk_pointer(input_data)

    dq = torch.zeros_like(q, dtype=torch.float64)
    dk_accum = torch.zeros((k.shape[0], head_num, k.shape[-1]), dtype=torch.float64, device=k.device)
    dv_accum = torch.zeros((v.shape[0], head_num, v.shape[-1]), dtype=torch.float64, device=v.device)
    dsink = torch.zeros_like(sink, dtype=torch.float64) if _is_present_tensor(sink) else torch.Tensor(0)

    for batch_idx, q0, q1, k0, k1 in _iter_slices(input_data):
        if q1 <= q0 or k1 <= k0:
            continue
        mask_i = _slice_mask(atten_mask, batch_idx, q0, q1, k0, k1, batch_size, sparse_mode)
        pse_i = _slice_pse(pse, batch_idx, q0, q1, k0, k1, batch_size, head_num, sparse_mode)
        drop_i = _slice_drop_mask(
            drop_mask, batch_idx, q0, q1, k0, k1, batch_size, head_num, sparse_mode, qk_pointer
        )

        _, softmax_res, x_max, x_sum, ki, vi = _forward_tnd(
            q[q0:q1], k[k0:k1], v[k0:k1], mask_i, pse_i, drop_i, sink, scale, keep_prob
        )
        dqi, dki, dvi, dsinki = _backward_from_forward(
            dy[q0:q1], q[q0:q1], ki, vi, softmax_res, drop_i, scale, keep_prob, sink, x_max, x_sum
        )
        dq[q0:q1] = dqi
        dk_accum[k0:k1] = dki
        dv_accum[k0:k1] = dvi
        if dsinki is not None:
            dsink = dsink + dsinki

    kv_head_num = k.shape[1]
    if head_num != kv_head_num:
        group = head_num // kv_head_num
        dk = dk_accum.reshape(k.shape[0], kv_head_num, group, k.shape[-1]).sum(dim=2)
        dv = dv_accum.reshape(v.shape[0], kv_head_num, group, v.shape[-1]).sum(dim=2)
    else:
        dk = dk_accum
        dv = dv_accum

    dsink_out = dsink.to(sink.dtype) if _is_present_tensor(sink) else torch.Tensor(0)
    return (
        dq.to(q.dtype),
        torch.Tensor(0),
        dk.to(k.dtype),
        torch.Tensor(0),
        dv.to(v.dtype),
        torch.Tensor(0),
        dsink_out,
    )


@register("executor_flash_attention_unpadding_score_grad_v5")
class FlashAttentionUnpaddingScoreGradV5Api(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FlashAttentionUnpaddingScoreGradV5Api, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        _log_qkv_shapes(input_data, "GradV5 cpu input")
        _make_tnd_atten_mask_from_sparse_mode(input_data, use_npu_mask=False)
        _prepare_forward_inputs(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return _backward_tnd(input_data)


@register("aclnn_flash_attention_unpadding_score_grad_v5")
class AclnnFlashAttentionUnpaddingScoreGradV5Api(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        _log_qkv_shapes(input_data, "GradV5 aclnn input")
        _make_tnd_atten_mask_from_sparse_mode(input_data, use_npu_mask=False)
        _prepare_forward_inputs(input_data)
        _prepare_aclnn_inputs(input_data)
        _prepare_aclnn_output_info(self.task_result, input_data)
        self._npu_input_cache = _move_tensor_inputs_to_npu(input_data)
        _prepare_forward_inputs_with_npu(input_data)
        self._npu_input_cache = _move_tensor_inputs_to_npu(input_data)
        input_args, output_packages = super().init_by_input_data(input_data)
        self._npu_output_cache = list(getattr(self.backend, "output_cache", []))
        null_tensor_ptr = _null_tensor_ptr()

        optional_tensor_names = [
            "queryRope",
            "keyInRope",
            "pseShiftOptional",
            "dropMaskOptional",
            "paddingMaskOptional",
            "attenMaskOptional",
            "softmaxInOptional",
            "sinkInOptional",
        ]
        optional_tensor_indices = [1, 3, 6, 7, 8, 9, 12, 14]
        for name, index in zip(optional_tensor_names, optional_tensor_indices):
            if _is_empty_tensor(input_data.kwargs[name]):
                input_args[index] = null_tensor_ptr

        optional_int_array_indices = {
            "qStartIdxOptional": 18,
            "kvStartIdxOptional": 19,
        }
        if int(input_data.kwargs["sparseMode"]) not in (5, 6):
            optional_int_array_indices["prefixOptional"] = 15
        null_void_ptr = _null_void_ptr()
        for index in optional_int_array_indices.values():
            input_args[index] = null_void_ptr

        self._null_output_indices = set()
        output_start = len(input_args) - len(output_packages)

        def set_output_null(output_index):
            if 0 <= output_index < len(output_packages):
                input_args[output_start + output_index] = TensorPtr()
                self._null_output_indices.add(output_index)

        if _is_empty_tensor(input_data.kwargs["queryRope"]):
            set_output_null(1)
        if _is_empty_tensor(input_data.kwargs["keyInRope"]):
            set_output_null(3)
        set_output_null(5)
        if _is_empty_tensor(input_data.kwargs["sinkInOptional"]):
            set_output_null(6)
        return input_args, output_packages

    def after_call(self, output_packages):
        outputs = super().after_call(output_packages)
        for index in getattr(self, "_null_output_indices", set()):
            if 0 <= index < len(outputs):
                outputs[index] = torch.Tensor(0)
        return outputs
