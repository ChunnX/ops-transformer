#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import json
import random
import argparse
import math
import numpy as np
import tensorflow as tf
import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
import npu_ops_transformer
from utils import (
    get_np_dtype,
    check_result,
    generate_fp4_tensor,
    decode_fp4_tensor,
    new_trans_np_fp4_e2m1_tensor_to_bfloat16,
    trans_tensor_fp8_e8m0_to_bf16,
    get_npu_dtype,
    gen_tensor_fp32_e6m2e1e16,
    trans_tensor_e6m2e1e16_to_bf16,
    new_trans_np_fp4_e1m2_tensor_to_bfloat16,
    pack_int4,
    pack_fp4_indices,
    unpack_int4,
    split_float32_to_uint16,
    combine_uint16_to_float32,
)
import os
import struct
from meta_cpu.mixed_quant_flash_attn_metadata_op import MixedQuantFlashAttnMetadataOp

DEVICE_ID = 0
# torch_npu.npu.set_device(int(DEVICE_ID))
seed = 21
np.random.seed(seed)  # 固定随机种子
np.set_printoptions(threshold=np.inf, linewidth=200, precision=4, suppress=True)
random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--qn", "-n1", type=int, default=8)
    parser.add_argument("--kvn", "-n2", type=int, default=1)
    parser.add_argument("--qs", "-s1", type=int, default=1)
    parser.add_argument("--kvs", "-s2", type=int, default=512)
    parser.add_argument("--dim", "-d", type=int, default=128)
    parser.add_argument(
        "--layout", "-l", type=str, default="BNSD", choices=["BNSD", "BSH", "BSND"]
    )
    parser.add_argument(
        "--k_antiquant_compute_mode", "-km", type=int, default=6, choices=[0, 6]
    )
    parser.add_argument(
        "--v_antiquant_compute_mode", "-vm", type=int, default=7, choices=[0, 7, 8]
    )
    parser.add_argument("--sparse_mode", "-m", type=int, default=0, choices=[0, 3, 4])
    parser.add_argument("--block_size", "-bs", type=int, default=0, choices=[256, 512])
    parser.add_argument(
        "--pa_layout", "-pa", type=str, default="nz", choices=["nz", "nd"]
    )
    parser.add_argument("--actual_seq_lengths", "-qs", type=int, nargs="+")
    parser.add_argument("--actual_seq_lengths_kv", "-kvs", type=int, nargs="+")
    parser.add_argument("--max_seq_q", "-qsmax", type=int, default=-1)
    parser.add_argument("--max_seq_kv", "-kvsmax", type=int, default=-1)
    parser.add_argument("--kv_seq_unequal_flag", action="store_true")
    parser.add_argument("--kv_seq_has_zero_flag", action="store_true")
    parser.add_argument("--softmax_lse_flag", action="store_true")
    parser.add_argument(
        "--q_dtype", "-qd", type=str, default="bf16", choices=["bf16", "fp16"]
    )
    parser.add_argument(
        "--kv_dtype",
        "-kvd",
        type=str,
        default="fp4_e2m1",
        choices=["fp4_e2m1", "fp4_e1m2", "fp8_e4m3"],
    )
    parser.add_argument("--souter", "-so", type=int, default=32)
    parser.add_argument("--sinner", "-si", type=int, default=512)
    parser.add_argument("--json_file", "-j", type=str, default="")
    parser.add_argument("--nocheck_flag", "-nc", action="store_true")
    parser.add_argument("--cpu_only", "-co", action="store_true")
    parser.add_argument("--npu_only", "-no", action="store_true")
    return parser.parse_args()


class MixedFlashAttnGraphNetwork(torch.nn.Module):
    def __init__(self):
        super(MixedFlashAttnGraphNetwork, self).__init__()

    def forward(
        self,
        q,
        k,
        v,
        k_descale,
        v_descale,
        block_table,
        cu_seqlens_q,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        quant_compute_mode,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_attn_out,
        return_softmax_lse,
    ):
        attn_out, softmax_lse = torch.ops.npu_ops_transformer.mixed_quant_flash_attn(
            q,
            k,
            v,
            k_descale,
            v_descale,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            attn_mask=attn_mask,
            metadata=metadata,
            quant_compute_mode=quant_compute_mode,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_attn_out=layout_attn_out,
            return_softmax_lse=return_softmax_lse,
        )
        return attn_out, softmax_lse


def call_npu_aclgraph(
    query,
    key,
    value,
    atten_mask,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_table,
    key_antiquant_scale,
    value_antiquant_scale,
    learnable_sink,
    num_heads,
    num_key_value_heads,
    scale,
    pre_tokens,
    next_tokens,
    input_layout,
    sparse_mode,
    block_size,
    key_antiquant_compute_mode,
    value_antiquant_compute_mode,
    softmax_lse_flag,
    query_dtype,
    key_dtype,
    value_dtype,
    key_antiquant_scale_dtype,
    value_antiquant_scale_dtype,
    out_dtype,
):
    meta_op = MixedQuantFlashAttnMetadataOp()
    metadata = meta_op.npu_mixed_quant_flash_attn_metadata(
        num_heads_q=num_heads,
        num_heads_kv=num_key_value_heads,
        head_dim=128,
        quant_compute_mode=1,
        batch_size=query.shape[0],
        cu_seqlens_q=None,
        seqused_q=None,
        seqused_kv=None,
        max_seqlen_q=4,
        max_seqlen_kv=2048,
        mask_mode=sparse_mode,
        win_left=pre_tokens,
        win_right=next_tokens,
        layout_q=input_layout,
        layout_kv="PA_NZ",
        layout_out=input_layout,
    )
    metadata = torch.tensor(metadata, dtype=torch.int32).to("npu:%s" % DEVICE_ID)
    torch._dynamo.reset()
    net = MixedFlashAttnGraphNetwork().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    graph_net = torch.compile(net, fullgraph=False, backend=npu_backend, dynamic=False)
    attention_out, softmax_lse = graph_net(
        q=query,
        k=key,
        v=value,
        k_descale=key_antiquant_scale,
        v_descale=value_antiquant_scale,
        attn_mask=atten_mask,
        block_table=block_table,
        cu_seqlens_q=None,
        seqused_q=None,
        seqused_kv=None,
        sinks=learnable_sink,
        softmax_scale=scale,
        win_left=pre_tokens,
        win_right=next_tokens,
        max_seqlen_q=4,
        max_seqlen_kv=2048,
        layout_q=input_layout,
        layout_kv="PA_NZ",
        layout_attn_out=input_layout,
        mask_mode=sparse_mode,
        quant_compute_mode=1,
        return_softmax_lse=softmax_lse_flag,
        metadata=metadata,
    )
    return attention_out, softmax_lse


def call_npu(
    query,
    key,
    value,
    atten_mask,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_table,
    key_antiquant_scale,
    value_antiquant_scale,
    learnable_sink,
    num_heads,
    num_key_value_heads,
    scale,
    pre_tokens,
    next_tokens,
    input_layout,
    sparse_mode,
    block_size,
    key_antiquant_compute_mode,
    value_antiquant_compute_mode,
    softmax_lse_flag,
    query_dtype,
    key_dtype,
    value_dtype,
    key_antiquant_scale_dtype,
    value_antiquant_scale_dtype,
    out_dtype,
    max_seq_q,
    max_seq_kv,
):
    print("ww debug")
    # print(f"num_heads_q: {num_heads}")
    # print(f"num_heads_kv: {num_key_value_heads}")
    # print(f"head_dim: {query.shape[-1]}")
    # print(f"quant_compute_mode: {1}")
    # print(f"batch_size: {query.shape[0]}")
    # print(f"cu_seqlens_q: {None}")
    # print(f"seqused_q: {actual_seq_lengths}, {actual_seq_lengths.dtype}")
    # print(f"seqused_kv: {actual_seq_lengths_kv}, {actual_seq_lengths_kv.dtype}")
    # print(f"max_seqlen_q: {max_seq_q}")
    # print(f"max_seqlen_kv: {max_seq_kv}")
    # print(f"mask_mode: {sparse_mode}")
    # print(f"win_left: {pre_tokens}")
    # print(f"win_right: {next_tokens}")
    # print(f"layout_q: {input_layout}")
    # print(f"layout_kv: {'PA_NZ'}")
    # print(f"layout_out: {input_layout}")
    meta_op = MixedQuantFlashAttnMetadataOp()
    metadata = meta_op.npu_mixed_quant_flash_attn_metadata(
        num_heads_q=num_heads,
        num_heads_kv=num_key_value_heads,
        head_dim=int(query.shape[-1]),
        quant_compute_mode=1,
        batch_size=int(query.shape[0]),
        cu_seqlens_q=None,
        seqused_q=actual_seq_lengths.cpu().tolist()
        if actual_seq_lengths is not None
        else None,
        seqused_kv=actual_seq_lengths_kv.cpu().tolist()
        if actual_seq_lengths_kv is not None
        else None,
        max_seqlen_q=max_seq_q,
        max_seqlen_kv=max_seq_kv,
        mask_mode=sparse_mode,
        win_left=pre_tokens,
        win_right=next_tokens,
        layout_q=input_layout,
        layout_kv="PA_NZ",
        layout_out=input_layout,
    )
    metadata = torch.tensor(metadata, dtype=torch.int32).to("npu:%s" % DEVICE_ID)
    # print(f'q shape: {query.shape}, numel: {query.numel()}, addr: {query.data_ptr()}, dtype: {query.dtype}, value: {query.cpu().flatten()[:-10]}')
    # print(f'k shape: {key.shape}, numel: {key.numel()}, addr: {key.data_ptr()}, dtype: {key.dtype}, value: {key.cpu().flatten()[:-10]}')
    # print(f'v shape: {value.shape}, numel: {value.numel()}, addr: {value.data_ptr()}, dtype: {value.dtype}, value: {value.cpu().flatten()[:-10]}')
    # print(f'k_descale shape: {key_antiquant_scale.shape}, numel: {key_antiquant_scale.numel()}, addr: {key_antiquant_scale.data_ptr()}, dtype: {key_antiquant_scale.dtype}, value: {key_antiquant_scale.cpu().flatten()[:-10]}')
    # print(f'v_descale shape: {value_antiquant_scale.shape}, numel: {value_antiquant_scale.numel()}, addr: {value_antiquant_scale.data_ptr()}, dtype: {value_antiquant_scale.dtype}, value: {value_antiquant_scale.cpu().flatten()[:-10]}')
    # print(f'block_table shape: {block_table.shape}, numel: {block_table.numel()}, addr: {block_table.data_ptr()}, dtype: {block_table.dtype}')
    # print(f'cu_seqlens_q: {None}')
    # print(f'seqused_q: {None}')
    # print(f'seqused_kv: {None}')
    # print(f'sinks: {learnable_sink}')
    # print(f'attn_mask shape: {atten_mask.shape}, numel: {atten_mask.numel()}, addr: {atten_mask.data_ptr()}, dtype: {atten_mask.dtype}')
    # print(f'metadata: {metadata}, numel: {metadata.numel()}, addr: {metadata.data_ptr()}')
    # print(f'quant_compute_mode: {1}')
    # print(f'softmax_scale: {scale}')
    # print(f'mask_mode: {sparse_mode}')
    # print(f'win_left: {pre_tokens}')
    # print(f'win_right: {next_tokens}')
    # print(f'max_seqlen_q: {4}')
    # print(f'max_seqlen_kv: {2048}')
    # print(f'layout_q: {input_layout}')
    # print(f'layout_kv: {"PA_NZ"}')
    # print(f'layout_attn_out: {input_layout}')
    # print(f'return_softmax_lse: {softmax_lse_flag}')
    # save_dict = {}
    # save_dict['q'] = query.cpu()
    # save_dict['k'] = key.cpu()
    # save_dict['v'] = value.cpu()
    # save_dict['k_descale'] = key_antiquant_scale.cpu()
    # save_dict['v_descale'] = value_antiquant_scale.cpu()
    # save_dict['block_table'] = block_table.cpu()
    # save_dict['cu_seqlens_q'] = None
    # save_dict['seqused_q'] = None
    # save_dict['seqused_kv'] = None
    # save_dict['sinks'] = learnable_sink
    # save_dict['attn_mask'] = atten_mask.cpu()
    # save_dict['metadata'] = metadata
    # save_dict['quant_compute_mode'] = 1
    # save_dict['softmax_scale'] = scale
    # save_dict['mask_mode'] = sparse_mode
    # save_dict['win_left'] = pre_tokens
    # save_dict['win_right'] = next_tokens
    # save_dict['max_seqlen_q'] = 4
    # save_dict['max_seqlen_kv'] = 2048
    # save_dict['layout_q'] = input_layout
    # save_dict['layout_kv'] = "PA_NZ"
    # save_dict['layout_attn_out'] = input_layout
    # save_dict['return_softmax_lse'] = softmax_lse_flag
    # torch.save(save_dict, 'tmp.pt')
    attention_out, softmax_lse = torch.ops.npu_ops_transformer.mixed_quant_flash_attn(
        q=query,
        k=key,
        v=value,
        k_descale=key_antiquant_scale,
        v_descale=value_antiquant_scale,
        attn_mask=atten_mask,
        block_table=block_table,
        cu_seqlens_q=None,
        seqused_q=actual_seq_lengths,
        seqused_kv=actual_seq_lengths_kv,
        sinks=learnable_sink,
        softmax_scale=scale,
        win_left=pre_tokens,
        win_right=next_tokens,
        max_seqlen_q=max_seq_q,
        max_seqlen_kv=max_seq_kv,
        layout_q=input_layout,
        layout_kv="PA_NZ",
        layout_attn_out=input_layout,
        mask_mode=sparse_mode,
        quant_compute_mode=1,
        return_softmax_lse=softmax_lse_flag,
        metadata=metadata,
    )
    return attention_out, softmax_lse


def antiquant_data(
    data,
    input_dtype,
    output_dtype,
    antiquant_compute_mode,
    antiquant_scale,
    gs_flag=True,
):
    if input_dtype == "fp4_e2m1":
        from ml_dtypes import bfloat16

        out_antiquant_scale = antiquant_scale.astype(bfloat16)
        # out_antiquant_scale = trans_tensor_fp8_e8m0_to_bf16(antiquant_scale)
        out_data = new_trans_np_fp4_e2m1_tensor_to_bfloat16(data)
        if output_dtype == "fp16":
            out_antiquant_scale = out_antiquant_scale.astype(np.float16)
            out_data = out_data.astype(np.float16)
    elif input_dtype == "fp4_e1m2":
        out_antiquant_scale = trans_tensor_e6m2e1e16_to_bf16(antiquant_scale)
        out_data = new_trans_np_fp4_e1m2_tensor_to_bfloat16(data)
        if output_dtype == "fp16":
            out_antiquant_scale = out_antiquant_scale.astype(np.float16)
            out_data = out_data.astype(np.float16)
    elif input_dtype == "fp8_e4m3":
        out_antiquant_scale = antiquant_scale
        out_data = data.astype(antiquant_scale.dtype)
    else:
        raise NotImplementedError(f"unsupported input_dtype: {input_dtype}")
    if antiquant_compute_mode == 0:
        return out_data * out_antiquant_scale
    elif antiquant_compute_mode == 6:
        g_size = 32
        if input_dtype == "fp4_e1m2":
            g_size = 4
        # out = np.zeros_like(out_data)
        for i in range(out_antiquant_scale.shape[-1]):
            if gs_flag:
                out_data[..., i * g_size : (i + 1) * g_size] = (
                    out_data[..., i * g_size : (i + 1) * g_size]
                    * out_antiquant_scale[..., i : i + 1]
                )
            else:
                out_data[..., i * g_size : (i + 1) * g_size] = (
                    out_data[..., i * g_size : (i + 1) * g_size]
                    * out_antiquant_scale[..., i : i + 1]
                )
        return out_data
    elif antiquant_compute_mode in [7, 8]:
        g_size = 32
        if input_dtype == "fp4_e1m2":
            g_size = 4
        # out = np.zeros_like(out_data)
        for i in range(out_antiquant_scale.shape[-1]):
            if gs_flag:
                out_data[..., i * g_size : (i + 1) * g_size, :] = out_data[
                    ..., i * g_size : (i + 1) * g_size, :
                ] * out_antiquant_scale[..., i : i + 1].transpose(0, 1, 2, 4, 3)
            else:
                out_data[:, i * g_size : (i + 1) * g_size, ...] = out_data[
                    :, i * g_size : (i + 1) * g_size, ...
                ] * out_antiquant_scale[..., i : i + 1].transpose(0, 1, 4, 2, 3)
        return out_data
    else:
        raise NotImplementedError(
            f"unsupported antiquant_compute_mode: {antiquant_compute_mode}"
        )


def softmax(x):
    # this func is only used by quant_dequant
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    # ans = ans.astype(np.float16)
    return ans, x_sum, x_max


def softmax_flash(x, block_sum, block_max, block_exp, is_first):
    # this func is only used by quant_dequant
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    pre_max = block_max
    block_max = np.max(np.hstack([block_max, x_max]), axis=1, keepdims=True)
    if is_first:
        block_exp = block_exp
    else:
        block_exp = np.exp(pre_max - block_max)
    x_sub = x - block_max
    mask = np.isneginf(x) & np.isneginf(block_max)
    x_sub = np.where(mask, 0, x_sub)
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    if is_first:
        block_sum = x_sum
    else:
        block_sum *= block_exp
        block_sum += x_sum
    return y, block_sum, block_max, block_exp


def call_cpu_numpy_by_batch(
    batch,
    qn,
    kvn,
    qs,
    kvs,
    d,
    query,
    key,
    value,
    atten_mask,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_table,
    key_antiquant_scale,
    value_antiquant_scale,
    learnable_sink,
    scale,
    pre_tokens,
    next_tokens,
    input_layout,
    sparse_mode,
    block_size,
    key_antiquant_compute_mode,
    value_antiquant_compute_mode,
    softmax_lse_flag,
    souter,
    sinner,
    q_dtype,
    kv_dtype,
    kv_scale_dtype,
):
    if input_layout == "BSH":
        query = query.reshape(query.shape[0], query.shape[1], qn, d).transpose(
            0, 2, 1, 3
        )
        key = key.reshape(key.shape[0], key.shape[1], kvn, d).transpose(0, 2, 1, 3)
        value = value.reshape(value.shape[0], value.shape[1], kvn, d).transpose(
            0, 2, 1, 3
        )
        if key_antiquant_compute_mode == 6:
            key_antiquant_scale = key_antiquant_scale.transpose(0, 1, 3, 2, 4)
        if key_antiquant_compute_mode == 0:
            key_antiquant_scale = key_antiquant_scale.reshape(
                key_antiquant_scale.shape[0], 1, key_antiquant_scale.shape[1]
            )
        if value_antiquant_compute_mode == 0:
            value_antiquant_scale = value_antiquant_scale.reshape(
                value_antiquant_scale.shape[0], 1, value_antiquant_scale.shape[1]
            )
    elif input_layout == "BSND":
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        if key_antiquant_compute_mode == 6:
            key_antiquant_scale = key_antiquant_scale.transpose(0, 1, 3, 2, 4)
        if key_antiquant_compute_mode == 0:
            key_antiquant_scale = key_antiquant_scale.reshape(
                key_antiquant_scale.shape[0], 1, key_antiquant_scale.shape[1]
            )
        if value_antiquant_compute_mode == 0:
            value_antiquant_scale = value_antiquant_scale.reshape(
                value_antiquant_scale.shape[0], 1, value_antiquant_scale.shape[1]
            )
    atten_out = np.zeros_like(query)
    key = antiquant_data(
        key, kv_dtype, q_dtype, key_antiquant_compute_mode, key_antiquant_scale
    )
    value = antiquant_data(
        value, kv_dtype, q_dtype, value_antiquant_compute_mode, value_antiquant_scale
    )
    g = qn // kvn
    for bidx in range(batch):
        kvs_cur_batch = (
            kvs if actual_seq_lengths_kv is None else actual_seq_lengths_kv[bidx]
        )
        qs_cur_batch = qs if actual_seq_lengths is None else actual_seq_lengths[bidx]
        if kvs_cur_batch == 0 or qs_cur_batch == 0:
            continue
        for n2idx in range(kvn):
            isinvalid = (kvs_cur_batch < qs_cur_batch) and sparse_mode != 0
            q_block = query[bidx, n2idx * g : n2idx * g + g, :qs_cur_batch]
            k_block = key[bidx, n2idx][:kvs_cur_batch]
            v_block = value[bidx, n2idx][:kvs_cur_batch]

            mm1 = np.matmul(q_block, k_block.transpose(1, 0), dtype=np.float32)
            if atten_mask is not None:
                if sparse_mode == 3:
                    mask_block = np.zeros(
                        (q_block.shape[0], q_block.shape[1], k_block.shape[0]),
                        dtype=np.uint8,
                    )
                    for s1idx in range(0, qs_cur_batch):
                        mask_num = qs_cur_batch - 1 - s1idx
                        if 0 < mask_num < kvs_cur_batch:
                            mask_block[..., s1idx, -mask_num:] = 1
                else:
                    # TODO
                    raise NotImplementedError(
                        f"unimplemented sparse_mode: {sparse_mode}"
                    )
                mm1 = mm1 + np.where(mask_block.astype(bool), -np.inf, 0.0).astype(
                    np.float32
                )
            p_block, block_sum, block_max = softmax(mm1 * scale)
            mm2 = np.matmul(p_block, v_block, dtype=np.float32)
            if isinvalid:
                for s1idx in range(mm2.shape[1]):
                    if s1idx < (qs_cur_batch - kvs_cur_batch):
                        mm2[:, s1idx] = 0.0
            atten_out[bidx, n2idx * g : n2idx * g + g, :qs_cur_batch] = mm2
    if input_layout == "BSH":
        atten_out = atten_out.transpose(0, 2, 1, 3).reshape(batch, qs, qn * d)
    elif input_layout == "BSND":
        atten_out = atten_out.transpose(0, 2, 1, 3)
    else:
        atten_out = atten_out.reshape(batch, qn, qs, d)
    return atten_out


def call_cpu_numpy(
    batch,
    qn,
    kvn,
    qs,
    kvs,
    d,
    query,
    key,
    value,
    atten_mask,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_table,
    key_antiquant_scale,
    value_antiquant_scale,
    learnable_sink,
    scale,
    pre_tokens,
    next_tokens,
    input_layout,
    sparse_mode,
    block_size,
    key_antiquant_compute_mode,
    value_antiquant_compute_mode,
    softmax_lse_flag,
    souter,
    sinner,
    q_dtype,
    kv_dtype,
    kv_scale_dtype,
):
    # BSH -> BSND
    if input_layout == "BSH":
        query = query.reshape(query.shape[0], query.shape[1], qn, d)
        key = key.reshape(key.shape[0], key.shape[1], kvn, d)
        value = value.reshape(value.shape[0], value.shape[1], kvn, d)

    g = qn // kvn
    gs_flag = True if input_layout == "BNSD" else False
    if gs_flag:
        query = query.reshape(batch, kvn, g, qs, d)
    else:
        query = query.reshape(batch, qs, kvn, g, d)
    atten_out = np.zeros_like(query)
    v0_block = None
    key = antiquant_data(
        key, kv_dtype, q_dtype, key_antiquant_compute_mode, key_antiquant_scale, gs_flag
    )
    value = antiquant_data(
        value,
        kv_dtype,
        q_dtype,
        value_antiquant_compute_mode,
        value_antiquant_scale,
        gs_flag,
    )
    for bidx in range(batch):
        kvs_cur_batch = (
            kvs if actual_seq_lengths_kv is None else actual_seq_lengths_kv[bidx]
        )
        qs_cur_batch = qs if actual_seq_lengths is None else actual_seq_lengths[bidx]
        if kvs_cur_batch == 0:
            continue
        for n2idx in range(kvn):
            isinvalid = (kvs_cur_batch < qs_cur_batch) and sparse_mode != 0
            souter_loop_times = math.ceil(g * qs_cur_batch / souter)
            for souter_idx in range(souter_loop_times):
                if souter_idx == souter_loop_times - 1:
                    souter_actual = g * qs_cur_batch - souter * (souter_loop_times - 1)
                else:
                    souter_actual = souter
                gidx = (
                    souter_idx * souter // qs_cur_batch
                    if gs_flag
                    else souter_idx * souter % g
                )
                s1idx = (
                    souter_idx * souter % qs_cur_batch
                    if gs_flag
                    else souter_idx * souter // g
                )
                block_sum = np.zeros((souter_actual, 1), dtype=np.float32)
                block_max = np.ones((souter_actual, 1), dtype=np.float32) * (
                    -float("inf")
                )
                block_exp = np.ones((souter_actual, 1), dtype=np.float32)
                if gs_flag:
                    q_block = query[bidx, n2idx].reshape(-1, d)[
                        souter_idx * souter : souter_idx * souter + souter_actual
                    ]
                    if key_antiquant_compute_mode == 0:
                        v0_block = q_block * key_antiquant_scale[n2idx].reshape(-1, d)
                else:
                    q_block = query[bidx, :, n2idx].reshape(-1, d)[
                        souter_idx * souter : souter_idx * souter + souter_actual
                    ]
                    if key_antiquant_compute_mode == 0:
                        v0_block = q_block * key_antiquant_scale[n2idx].reshape(-1, d)
                out_block = np.zeros_like(q_block)
                sinner_loop_times = math.ceil(kvs_cur_batch / sinner)
                for sinner_idx in range(sinner_loop_times):
                    if sinner_idx == sinner_loop_times - 1:
                        sinner_actual = kvs_cur_batch - sinner * (sinner_loop_times - 1)
                    else:
                        sinner_actual = sinner
                    # if block_table is not None:
                    #     k_block = concat_pa_array(key, block_table, kvn, bidx, n2idx, sinner_idx * sinner,
                    #                               sinner_idx * sinner + sinner_actual, input_layout)
                    #     v_block = concat_pa_array(value, block_table, kvn, bidx, n2idx, sinner_idx * sinner,
                    #                               sinner_idx * sinner + sinner_actual, input_layout)
                    if gs_flag:
                        k_block = key[bidx, n2idx][
                            sinner_idx * sinner : sinner_idx * sinner + sinner_actual
                        ]
                        v_block = value[bidx, n2idx][
                            sinner_idx * sinner : sinner_idx * sinner + sinner_actual
                        ]
                    else:
                        k_block = key[bidx, :, n2idx][
                            sinner_idx * sinner : sinner_idx * sinner + sinner_actual
                        ]
                        v_block = value[bidx, :, n2idx][
                            sinner_idx * sinner : sinner_idx * sinner + sinner_actual
                        ]
                    mask_block = None
                    if atten_mask is not None:
                        if sparse_mode == 3:
                            mask_block = np.zeros(
                                (q_block.shape[0], k_block.shape[0]), dtype=np.uint8
                            )
                            for m_idx in range(0, souter_actual):
                                s1idx = (
                                    (souter_idx * souter + m_idx) % qs_cur_batch
                                    if gs_flag
                                    else (souter_idx * souter + m_idx) // g
                                )
                                s2idx_start = sinner_idx * sinner
                                s2idx_end = sinner_idx * sinner + sinner_actual
                                if (
                                    (kvs_cur_batch - qs_cur_batch + 1 + s1idx) > 0
                                ) and s2idx_end <= (
                                    kvs_cur_batch - qs_cur_batch + 1 + s1idx
                                ):
                                    continue
                                mask_num = s2idx_end - (
                                    kvs_cur_batch - qs_cur_batch + 1 + s1idx
                                )
                                mask_block[m_idx, -mask_num:] = 1
                        else:
                            # TODO
                            raise NotImplementedError(
                                f"unimplemented sparse_mode: {sparse_mode}"
                            )
                    mm1 = np.matmul(q_block, k_block.transpose(1, 0), dtype=np.float32)
                    print(
                        f"bidx: {bidx}, n2idx: {n2idx}, souter_idx: {souter_idx}, sinner_idx: {sinner_idx}"
                    )
                    # print(f'v0: {v0_block}')
                    # if bidx == 0 and souter_idx == 0 and sinner_idx == 0:
                    #     print(f'mm1 shape: {mm1.shape}, value: {mm1}')
                    if mask_block is not None:
                        mm1 = mm1 + np.where(
                            mask_block.astype(bool), -np.inf, 0.0
                        ).astype(np.float32)
                    # if bidx == 0 and souter_idx == 0 and sinner_idx == 0:
                    #     print(f'mm1 shape: {mm1.shape}, value: {mm1}')
                    p_block, block_sum, block_max, block_exp = softmax_flash(
                        mm1 * scale, block_sum, block_max, block_exp, sinner_idx == 0
                    )
                    mm2 = np.matmul(p_block, v_block, dtype=np.float32)
                    if sinner_idx == 0:
                        out_block = mm2
                    else:
                        out_block = out_block * block_exp + mm2
                    if sinner_idx == sinner_loop_times - 1:
                        out_block /= block_sum
                if isinvalid:
                    for m_idx in range(0, souter_actual):
                        s1idx = (
                            (souter_idx * souter + m_idx) % qs_cur_batch
                            if gs_flag
                            else (souter_idx * souter + m_idx) // g
                        )
                        if s1idx < (qs_cur_batch - kvs_cur_batch):
                            out_block[m_idx] = 0
                if gs_flag:
                    atten_out[bidx, n2idx].reshape(-1, d)[
                        souter_idx * souter : souter_idx * souter + souter_actual
                    ] = out_block
                else:
                    for m_idx in range(
                        souter_idx * souter, souter_idx * souter + souter_actual
                    ):
                        s1idx = m_idx // g
                        gidx = m_idx % g
                        atten_out[bidx, s1idx, n2idx, gidx] = out_block[
                            m_idx - souter_idx * souter
                        ]
    if input_layout == "BSH":
        atten_out = atten_out.reshape(batch, qs, qn * d)
    elif input_layout == "BSND":
        atten_out = atten_out.reshape(batch, qs, qn, d)
    else:
        atten_out = atten_out.reshape(batch, qn, qs, d)
    return atten_out


def concat_pa_array(data, block_table, n, bidx, nidx, sidx_start, sidx_end, layout):
    if layout == "BNSD":
        block_size = data.shape[2]
        d = data.shape[3]
        seq_len = sidx_end - sidx_start
        finish_seq_len = 0
        tmp_data = np.zeros((seq_len, d), dtype=data.dtype)
        # 左闭右开
        while finish_seq_len < seq_len:
            sidx = sidx_start + finish_seq_len
            block_id = block_table[bidx][sidx // block_size]
            block_seq_start = sidx % block_size
            valid_seq_len = min(block_size - block_seq_start, seq_len - finish_seq_len)
            tmp_data[finish_seq_len : finish_seq_len + valid_seq_len] = data[
                block_id, nidx, block_seq_start : block_seq_start + valid_seq_len
            ]
            finish_seq_len += valid_seq_len
    elif layout == "BSH":
        block_size = data.shape[1]
        d = data.shape[2] // n
        seq_len = sidx_end - sidx_start
        finish_seq_len = 0
        tmp_data = np.zeros((seq_len, d), dtype=data.dtype)
        # 左闭右开
        while finish_seq_len < seq_len:
            sidx = sidx_start + finish_seq_len
            block_id = block_table[bidx][sidx // block_size]
            block_seq_start = sidx % block_size
            valid_seq_len = min(block_size - block_seq_start, seq_len - finish_seq_len)
            tmp_data[finish_seq_len : finish_seq_len + valid_seq_len] = data[
                block_id,
                block_seq_start : block_seq_start + valid_seq_len,
                nidx * d : nidx * d + d,
            ]
            finish_seq_len += valid_seq_len
    elif layout == "BSND":
        block_size = data.shape[1]
        d = data.shape[3]
        seq_len = sidx_end - sidx_start
        finish_seq_len = 0
        tmp_data = np.zeros((seq_len, d), dtype=data.dtype)
        # 左闭右开
        while finish_seq_len < seq_len:
            sidx = sidx_start + finish_seq_len
            block_id = block_table[bidx][sidx // block_size]
            block_seq_start = sidx % block_size
            valid_seq_len = min(block_size - block_seq_start, seq_len - finish_seq_len)
            tmp_data[finish_seq_len : finish_seq_len + valid_seq_len] = data[
                block_id, block_seq_start : block_seq_start + valid_seq_len, nidx
            ]
            finish_seq_len += valid_seq_len
    return tmp_data


def valid_idx_range(idx, num, dim):
    return 0 <= idx < dim and num >= 1 and idx + num <= dim


def generate_input_shape(b, n, s, d, layout):
    if layout == "BNSD":
        return (b, n, s, d)
    elif layout == "BSND":
        return (b, s, n, d)
    elif layout == "BSH":
        return (b, s, n * d)
    else:
        raise NotImplementedError(f"unsupported layout: {layout}")


def generate_antiquant_scale_shape(
    antiquant_compute_mode, b, n, s, d, layout, data_dtype
):
    if layout == "BNSD":
        if antiquant_compute_mode == 0:
            return (n, 1, d)
        elif antiquant_compute_mode == 1:
            return (b, s)
        elif antiquant_compute_mode == 6:
            return (
                (1, b, n, s, math.ceil(d / 64) * 2)
                if data_dtype == "fp4_e2m1"
                else (1, b, n, s, math.ceil(d / 64))
            )
        elif antiquant_compute_mode in [7, 8]:
            return (
                (1, b, n, d, math.ceil(s / 64) * 2)
                if data_dtype == "fp4_e2m1"
                else (1, b, n, d, math.ceil(s / 64))
            )
    elif layout in ["BSND", "BSH"]:
        if antiquant_compute_mode == 0:
            return (n, d)
        elif antiquant_compute_mode == 1:
            return (b, s)
        elif antiquant_compute_mode == 6:
            return (
                (1, b, s, n, math.ceil(d / 64) * 2)
                if data_dtype == "fp4_e2m1"
                else (1, b, s, n, math.ceil(d / 64))
            )
        elif antiquant_compute_mode in [7, 8]:
            return (
                (1, b, n, d, math.ceil(s / 64) * 2)
                if data_dtype == "fp4_e2m1"
                else (1, b, n, d, math.ceil(s / 64))
            )
    raise NotImplementedError(
        f"unsupported layout: {layout} or antiquant_compute_mode: {antiquant_compute_mode}"
    )


def dtype_convert(data, dtype, shape, data_range_min, data_range_max):
    if dtype in ["float4_e2m1", "fp4_e2m1"]:
        encoded_tensor, _ = generate_fp4_tensor(shape, "fp4_e2m1")
        data = encoded_tensor.numpy()
    elif dtype in ["float4_e1m2", "fp4_e1m2", "hifloat4"]:
        encoded_tensor, _ = generate_fp4_tensor(shape, "fp4_e1m2")
        data = encoded_tensor.numpy()
    elif dtype == "fp32_e6m2e1e16":
        data = gen_tensor_fp32_e6m2e1e16(shape, [data_range_min, data_range_max])
        # data = data.astype(np.uint32)
    elif dtype == "hifloat8":
        # data = trans_np_float_tensor_to_hifuint8(in_tensor=data, round_mode="hybrid", over_mode=True)
        pass
    elif dtype == "float8_e8m0":
        # data = np.random.uniform(0, 8, shape).astype(np.uint8)
        # data = 2 ** (data)
        # multi_shape = np.prod(data.shape)
        # input_i_data_fp32 = data.astype(np.float32).reshape(multi_shape)
        # in_tensor = np.zeros_like(data).reshape(multi_shape)
        # for i in range(multi_shape):
        #     in_tensor[i] = float_to_fp8e8m0(input_i_data_fp32[i])
        # data = in_tensor.reshape(data.shape)

        # input_i_data = data.view(np.uint32)
        # input_i_data = (input_i_data << 1) >> 24
        # input_i_data = 2 ** (input_i_data.astype(np.float64) - 127)
        # data = input_i_data.astype(float8_e8m0)
        pass
    elif dtype == "fp4_e2m1_as_fp32":
        choices = [0, -0.5, 0.5, -1, 1, -2, 2, -1.5, 1.5, -3, 3, -4, 4, -6, 6]
        data = np.random.choice(choices, size=shape).astype(np.float32)
    return data


def generate_np_data(data_range, shape, dtype):
    data = np.random.uniform(data_range[0], data_range[1], shape).astype(
        get_np_dtype(dtype)
    )
    data = dtype_convert(data, dtype, shape, data_range[0], data_range[1])
    return data


def generate_pa_data(
    key,
    value,
    dtype,
    actual_seq_len,
    layout,
    block_table,
    block_num,
    block_size,
    b,
    n,
    kvs,
    d,
    pa_layout,
):
    if layout in ["BSND", "BSH"]:
        seq_len_list = [kvs] * b
    elif layout == "BNSD":
        seq_len_list = [kvs] * b
    else:
        raise NotImplementedError(f"unsupported layout: {layout}")
    if actual_seq_len is not None:
        seq_len_list = actual_seq_len
    G = 1
    if pa_layout == "nz":
        if dtype in ["float4_e2m1", "fp4_e2m1", "float4_e1m2", "fp4_e1m2", "hifloat4"]:
            key_cache = np.zeros(
                (block_num, n, d // 64, block_size, 64), dtype=key.dtype
            )
            value_cache = np.zeros(
                (block_num, n, d // 64, block_size, 64), dtype=value.dtype
            )
        elif dtype == "fp8_e4m3":
            key_cache = np.zeros(
                (block_num, n, d // 32, block_size, 32), dtype=key.dtype
            )
            value_cache = np.zeros(
                (block_num, n, d // 32, block_size, 32), dtype=value.dtype
            )
        elif dtype == "fp8_e8m0":
            key_cache = np.zeros(
                (block_num, n, block_size // 16, d // 64, 16, 2), dtype=key.dtype
            )
            value_cache = np.zeros(
                (block_num, n, d // 16, block_size // 64, 16, 2), dtype=value.dtype
            )
            G = 32
        elif dtype == "fp32_e6m2e1e16":
            key_cache = np.zeros(
                (block_num, n, block_size // 16, d // 64, 16), dtype=key.dtype
            )
            value_cache = np.zeros(
                (block_num, n, d // 16, block_size // 64, 16), dtype=value.dtype
            )
            G = 64
        else:
            return key, value
        for bIdx in range(b):
            seq_len = seq_len_list[bIdx]
            block_num_cur_batch = math.ceil(seq_len / block_size)
            for blockIdx in range(block_num_cur_batch):
                blockId = block_table[bIdx][blockIdx]
                assert blockId >= 0
                if len(key.shape) < 5:
                    if layout in ["BSND", "BSH"]:
                        key_cur_block = (
                            key[bIdx][
                                blockIdx * block_size : (blockIdx + 1) * block_size
                            ]
                            .reshape(-1, n, d)
                            .transpose(1, 0, 2)
                        )
                        value_cur_block = (
                            value[bIdx][
                                blockIdx * block_size : (blockIdx + 1) * block_size
                            ]
                            .reshape(-1, n, d)
                            .transpose(1, 0, 2)
                        )
                    else:
                        key_cur_block = key[
                            bIdx, :, blockIdx * block_size : (blockIdx + 1) * block_size
                        ].reshape(n, -1, d)
                        value_cur_block = value[
                            bIdx, :, blockIdx * block_size : (blockIdx + 1) * block_size
                        ].reshape(n, -1, d)
                # for mxfp4 and hifp4
                else:
                    if layout in ["BSND", "BSH"]:
                        key_cur_block = (
                            key[0][bIdx][
                                blockIdx * block_size : (blockIdx + 1) * block_size
                            ]
                            .reshape(-1, n, key.shape[-1])
                            .transpose(1, 0, 2)
                        )
                    else:
                        key_cur_block = key[0][
                            bIdx, :, blockIdx * block_size : (blockIdx + 1) * block_size
                        ].reshape(n, -1, key.shape[-1])
                    value_cur_block = value[0][
                        bIdx,
                        :,
                        :,
                        blockIdx * block_size // G : (blockIdx + 1) * block_size // G,
                    ].reshape(n, d, -1)
                if dtype in [
                    "float4_e2m1",
                    "fp4_e2m1",
                    "float4_e1m2",
                    "fp4_e1m2",
                    "hifloat4",
                    "fp8_e4m3",
                ]:
                    key_cache[blockId, :, :, : key_cur_block.shape[1]] = (
                        key_cur_block.reshape(
                            n, -1, key_cache.shape[2], key_cache.shape[4]
                        ).transpose(0, 2, 1, 3)
                    )
                    value_cache[blockId, :, :, : value_cur_block.shape[1]] = (
                        value_cur_block.reshape(
                            n, -1, value_cache.shape[2], value_cache.shape[4]
                        ).transpose(0, 2, 1, 3)
                    )
                elif dtype in ["fp8_e8m0"]:
                    target_s = math.ceil(key_cur_block.shape[1] / 16) * 16
                    key_cur_block_padded = np.ones(
                        (n, target_s, key.shape[-1]), dtype=key_cur_block.dtype
                    )
                    key_cur_block_padded[:, : key_cur_block.shape[1], :] = key_cur_block

                    target_s = math.ceil(value_cur_block.shape[-1] / 2) * 2
                    value_cur_block_padded = np.ones(
                        (n, d, target_s), dtype=value_cur_block.dtype
                    )
                    value_cur_block_padded[:, :, : value_cur_block.shape[-1]] = (
                        value_cur_block
                    )

                    key_cache[blockId, :, : key_cur_block_padded.shape[1] // 16] = (
                        key_cur_block_padded.reshape(
                            n,
                            key_cur_block_padded.shape[1] // 16,
                            16,
                            key_cur_block_padded.shape[2] // 2,
                            2,
                        ).transpose(0, 1, 3, 2, 4)
                    )
                    value_cache[
                        blockId, :, :, : value_cur_block_padded.shape[2] // 2
                    ] = value_cur_block.reshape(
                        n,
                        value_cur_block_padded.shape[1] // 16,
                        16,
                        value_cur_block_padded.shape[2] // 2,
                        2,
                    ).transpose(0, 1, 3, 2, 4)
                elif dtype == "fp32_e6m2e1e16":
                    target_s = math.ceil(key_cur_block.shape[1] / 16) * 16
                    key_cur_block_padded = np.ones(
                        (n, target_s, key.shape[-1]), dtype=key_cur_block.dtype
                    )
                    key_cur_block_padded[:, : key_cur_block.shape[1]] = key_cur_block

                    tmp_key = split_float32_to_uint16(
                        key_cur_block_padded.reshape(
                            n,
                            key_cur_block_padded.shape[1] // 16,
                            16,
                            key_cur_block_padded.shape[2],
                        )
                        .transpose(0, 1, 3, 2)
                        .copy()
                    ).view(np.float32)
                    tmp_value = split_float32_to_uint16(
                        value_cur_block.reshape(
                            n,
                            value_cur_block.shape[1] // 16,
                            16,
                            value_cur_block.shape[2],
                        )
                        .transpose(0, 1, 3, 2)
                        .copy()
                    ).view(np.float32)
                    key_cache[blockId, :, : tmp_key.shape[1]] = tmp_key
                    value_cache[blockId, :, :, : tmp_value.shape[2]] = tmp_value
    else:
        raise NotImplementedError(f"unsupported pa_layout: {pa_layout}")
    return key_cache, value_cache


def numpy2tensor(data, actual_dtype=None):
    if data is None:
        return None
    if data.dtype == tf.bfloat16.as_numpy_dtype:
        return (
            torch.from_numpy(data.astype(np.float32))
            .to(torch.bfloat16)
            .to("npu:%s" % DEVICE_ID)
        )
    if data.dtype == "float8_e4m3fn":
        return (
            torch.from_numpy(data.astype(np.float32))
            .to(torch.float8_e4m3fn)
            .to("npu:%s" % DEVICE_ID)
        )
    if data.dtype == "float8_e8m0fnu":
        return (
            torch.from_numpy(data.astype(np.float32))
            .to(torch.float8_e8m0fnu)
            .to("npu:%s" % DEVICE_ID)
        )
    if actual_dtype in ["fp4_e2m1", "fp4_e1m2"]:
        shape = list(data.shape)
        shape[-1] = shape[-1] >> 1
        data_pool_np = pack_int4(data)
        dump_np_data = np.array(data_pool_np)
        dump_np_data.tofile("./tmp.bin")
        input_data_i = np.fromfile(
            "./tmp.bin", dtype=get_np_dtype(actual_dtype)
        ).reshape(shape)
        # return torch.from_numpy(pack_fp4_indices(data)).reshape(shape).to("npu:%s" % DEVICE_ID)
        # return torch.from_numpy(pack_int4(data)).reshape(shape).to("npu:%s" % DEVICE_ID)
        return torch.from_numpy(input_data_i).to("npu:%s" % DEVICE_ID)
    else:
        return torch.from_numpy(data).to("npu:%s" % DEVICE_ID)


def generate_input(
    batch,
    qn,
    kvn,
    qs,
    kvs,
    d,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    layout,
    block_size,
    pa_layout,
    key_antiquant_compute_mode,
    value_antiquant_compute_mode,
    sparse_mode,
    q_dtype,
    kv_dtype,
    kv_scale_dtype,
    query_data_range,
    kv_data_range,
    kv_scale_data_range,
):
    q_shape = generate_input_shape(batch, qn, qs, d, layout)
    kv_shape = generate_input_shape(batch, kvn, kvs, d, layout)
    k_scale_shape = generate_antiquant_scale_shape(
        key_antiquant_compute_mode, batch, kvn, kvs, d, layout, kv_dtype
    )
    v_scale_shape = generate_antiquant_scale_shape(
        value_antiquant_compute_mode, batch, kvn, kvs, d, layout, kv_dtype
    )

    query = generate_np_data(query_data_range, q_shape, q_dtype)
    key = generate_np_data(kv_data_range, kv_shape, kv_dtype)
    value = generate_np_data(kv_data_range, kv_shape, kv_dtype)
    key_antiquant_scale = generate_np_data(
        kv_scale_data_range, k_scale_shape, kv_scale_dtype
    )
    value_antiquant_scale = generate_np_data(
        kv_scale_data_range, v_scale_shape, kv_scale_dtype
    )

    block_table = None
    key_cache = None
    value_cache = None
    key_antiquant_scale_cache = None
    value_antiquant_scale_cache = None
    if block_size:
        max_block_num_per_batch = (
            math.ceil(kvs / block_size)
            if actual_seq_lengths_kv is None
            else math.ceil(max(actual_seq_lengths_kv) / block_size)
        )
        block_num = max_block_num_per_batch * batch
        block_table = np.ones((batch, max_block_num_per_batch), dtype=np.int32) * -1

        global_block_offset = 0
        for bIdx in range(batch):
            kvs_cur_batch = (
                kvs if actual_seq_lengths_kv is None else actual_seq_lengths_kv[bIdx]
            )
            num_blocks = math.ceil(kvs_cur_batch / block_size)
            start_id = global_block_offset
            end_id = global_block_offset + num_blocks
            block_table[bIdx][:num_blocks] = list(range(start_id, end_id))
            global_block_offset += num_blocks
        key_cache, value_cache = generate_pa_data(
            key,
            value,
            kv_dtype,
            actual_seq_lengths_kv,
            layout,
            block_table,
            block_num,
            block_size,
            batch,
            kvn,
            kvs,
            d,
            pa_layout,
        )
        key_antiquant_scale_cache, value_antiquant_scale_cache = generate_pa_data(
            key_antiquant_scale,
            value_antiquant_scale,
            kv_scale_dtype,
            actual_seq_lengths_kv,
            layout,
            block_table,
            block_num,
            block_size,
            batch,
            kvn,
            kvs,
            d,
            pa_layout,
        )
    if actual_seq_lengths:
        actual_seq_lengths = np.array(actual_seq_lengths).astype(np.int32)
    if actual_seq_lengths_kv:
        actual_seq_lengths_kv = np.array(actual_seq_lengths_kv).astype(np.int32)

    if sparse_mode == 0:
        mask = None
    elif sparse_mode == 3:
        mask = np.triu(np.ones((2048, 2048), dtype=np.int8), k=1)
    else:
        # TODO: band mode mask
        raise NotImplementedError(f"unimplemented sparse_mode: {sparse_mode}")
    print(f"key_cache:{key_cache.shape}, value_cache: {value_cache.shape}")
    return (
        query,
        key,
        value,
        key_cache,
        value_cache,
        key_antiquant_scale,
        value_antiquant_scale,
        key_antiquant_scale_cache,
        value_antiquant_scale_cache,
        block_table,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        mask,
    )


def bin2numpy(info, root_path, is_golden=False):
    if is_golden:
        bin_path = os.path.join(root_path, info["golden_path"])
    else:
        if "bin_path" in info:
            bin_path = os.path.join(root_path, info["bin_path"])
        else:
            bin_path = os.path.join(root_path, info["bin_paths"][0])
    dtype = info["dtype"]
    shape = info["shape"] if "shape" in info else info["shapes"][0]
    if os.path.exists(bin_path):
        if dtype in ["fp4_e2m1", "fp4_e1m2"]:
            input_data_i = np.fromfile(bin_path, dtype=get_np_dtype(dtype))
            shape_product = np.prod(shape)
            input_data_i_unpack = unpack_int4(input_data_i).astype(np.uint8)
            input_data_i = input_data_i_unpack[0:shape_product]
            input_data_i = input_data_i.reshape(shape)
            return input_data_i
            # shape[-1] = shape[-1] >> 1  # 最后一维长度缩短一半
            # return np.fromfile(bin_path, dtype=get_np_dtype(dtype)).reshape(shape)
        else:
            return np.fromfile(bin_path, dtype=get_np_dtype(dtype)).reshape(shape)
    return None


def load_input(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    root_path = os.path.dirname(json_file)
    query = bin2numpy(data["input"][0], root_path)
    key = bin2numpy(data["input"][1], root_path)
    value = bin2numpy(data["input"][2], root_path)
    key_cache = bin2numpy(data["input"][21], root_path)
    value_cache = bin2numpy(data["input"][22], root_path)
    key_antiquant_scale = bin2numpy(data["input"][15], root_path)
    value_antiquant_scale = bin2numpy(data["input"][17], root_path)
    block_table = None
    actual_seq_lengths = None
    actual_seq_lengths_kv = None
    mask = None
    cpu = bin2numpy(data["output"][0], root_path, True)
    npu = bin2numpy(data["output"][0], root_path, False)
    return (
        query,
        key,
        value,
        key_cache,
        value_cache,
        key_antiquant_scale,
        value_antiquant_scale,
        block_table,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        mask,
        cpu,
        npu,
    )


def main():
    args = parse_args()
    batch = args.batch_size
    qn = args.qn
    kvn = args.kvn
    qs = args.qs
    kvs = args.kvs
    d = args.dim
    scale = 1 / (d**0.5)
    layout = args.layout
    key_antiquant_compute_mode = args.k_antiquant_compute_mode
    value_antiquant_compute_mode = args.v_antiquant_compute_mode
    sparse_mode = args.sparse_mode
    block_size = args.block_size
    pa_layout = args.pa_layout
    actual_seq_lengths = args.actual_seq_lengths
    actual_seq_lengths_kv = args.actual_seq_lengths_kv
    max_seq_q = args.max_seq_q
    max_seq_kv = args.max_seq_kv
    kv_seq_unequal_flag = args.kv_seq_unequal_flag
    kv_seq_has_zero_flag = args.kv_seq_has_zero_flag
    softmax_lse_flag = args.softmax_lse_flag
    q_dtype = args.q_dtype
    kv_dtype = args.kv_dtype
    souter = args.souter
    sinner = args.sinner
    json_file = args.json_file
    check_flag = not args.nocheck_flag
    cpu_only = args.cpu_only
    npu_only = args.npu_only

    cmp_dtype = "bfloat16" if q_dtype == "bf16" else "float16"
    learnable_sink = None
    pre_tokens = -1
    next_tokens = -1

    query_data_range = [-10, 10]
    kv_data_range = [-10, 10]
    kv_scale_data_range = []
    if kv_dtype in ["fp4_e1m2"]:
        kv_scale_data_range = [125, 129]
    elif kv_dtype in ["fp4_e2m1"]:
        kv_scale_data_range = [0.01, 10]
    else:
        kv_scale_data_range = [0, 0.2]

    if actual_seq_lengths_kv is None:
        if kv_seq_unequal_flag:
            if key_antiquant_compute_mode == 6 and value_antiquant_compute_mode in [
                7,
                8,
            ]:
                if kv_dtype == "fp4_e2m1":
                    actual_seq_lengths_kv = random.choices(
                        list(range(32, 4096, 32)), k=batch
                    )
                else:
                    actual_seq_lengths_kv = random.choices(
                        list(range(64, 4096, 64)), k=batch
                    )
            else:
                actual_seq_lengths_kv = [random.randint(1, kvs) for _ in range(batch)]
        if kv_seq_has_zero_flag:
            if not actual_seq_lengths_kv:
                actual_seq_lengths_kv = [kvs] * batch
            # indices = random.sample(range(len(actual_seq_lengths_kv)), random.randint(1, len(actual_seq_lengths_kv)))
            indices = random.sample(range(len(actual_seq_lengths_kv)), 1)
            for idx in indices:
                actual_seq_lengths_kv[idx] = 0
    if block_size:
        if actual_seq_lengths_kv is None:
            actual_seq_lengths_kv = [kvs] * batch
    if (
        key_antiquant_compute_mode == 0
        and value_antiquant_compute_mode == 0
        and kv_dtype == "fp8_e4m3"
    ):
        kv_scale_dtype = q_dtype
    elif (
        key_antiquant_compute_mode == 6
        and value_antiquant_compute_mode in [7, 8]
        and kv_dtype == "fp4_e2m1"
    ):
        kv_scale_dtype = "fp8_e8m0"
    elif (
        key_antiquant_compute_mode == 6
        and value_antiquant_compute_mode in [7, 8]
        and kv_dtype == "fp4_e1m2"
    ):
        kv_scale_dtype = "fp32_e6m2e1e16"
    else:
        raise NotImplementedError(
            f"unimplemented k_antiquant_compute_mode: {key_antiquant_compute_mode}, v_antiquant_compute_mode: {value_antiquant_compute_mode}, kv_dtype: {kv_dtype}"
        )
    fuzz_cpu = None
    fuzz_npu = None
    if len(json_file) and os.path.exists(json_file):
        (
            query,
            key,
            value,
            key_cache,
            value_cache,
            key_antiquant_scale,
            value_antiquant_scale,
            block_table,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            mask,
            fuzz_cpu,
            fuzz_npu,
        ) = load_input(json_file)
    else:
        (
            query,
            key,
            value,
            key_cache,
            value_cache,
            key_antiquant_scale,
            value_antiquant_scale,
            key_antiquant_scale_cache,
            value_antiquant_scale_cache,
            block_table,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            mask,
        ) = generate_input(
            batch,
            qn,
            kvn,
            qs,
            kvs,
            d,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            layout,
            block_size,
            pa_layout,
            key_antiquant_compute_mode,
            value_antiquant_compute_mode,
            sparse_mode,
            q_dtype,
            kv_dtype,
            kv_scale_dtype,
            query_data_range,
            kv_data_range,
            kv_scale_data_range,
        )
    if max_seq_q == -1:
        max_seq_q = qs if actual_seq_lengths is None else int(max(actual_seq_lengths))
    if max_seq_kv == -1:
        max_seq_kv = (
            kvs if actual_seq_lengths_kv is None else int(max(actual_seq_lengths_kv))
        )
    if max_seq_q > qs:
        raise ValueError(f"Invalid max_seq_q: {max_seq_q}, which should <= {qs}")
    if max_seq_kv > kvs:
        raise ValueError(f"Invalid max_seq_kv: {max_seq_kv}, which should <= {kvs}")
    print(f"query shape: {query.shape}, dtype: {query.dtype}")
    if key_cache is None:
        print(f"key shape: {key.shape}, dtype: {key.dtype}")
    else:
        print(f"key shape: {key_cache.shape}, dtype: {key_cache.dtype}")
    if value_cache is None:
        print(f"value shape: {value.shape}, dtype: {value.dtype}")
    else:
        print(f"value shape: {value_cache.shape}, dtype: {value_cache.dtype}")
    print(f"key_antiquant_compute_mode: {key_antiquant_compute_mode}")
    if key_antiquant_scale_cache is None:
        print(
            f"key_antiquant_scale shape: {key_antiquant_scale.shape}, dtype: {key_antiquant_scale.dtype}"
        )
    else:
        print(
            f"key_antiquant_scale shape: {key_antiquant_scale_cache.shape}, dtype: {key_antiquant_scale_cache.dtype}"
        )
    print(f"value_antiquant_compute_mode: {value_antiquant_compute_mode}")
    if value_antiquant_scale_cache is None:
        print(
            f"value_antiquant_scale shape: {value_antiquant_scale.shape}, dtype: {value_antiquant_scale.dtype}"
        )
    else:
        print(
            f"value_antiquant_scale shape: {value_antiquant_scale_cache.shape}, dtype: {value_antiquant_scale_cache.dtype}"
        )

    if fuzz_cpu is not None and fuzz_npu is not None:
        print("fuzz cpu vs fuzz npu")
        check_result(
            torch.from_numpy(fuzz_cpu.astype(np.float32)),
            torch.from_numpy(fuzz_npu.astype(np.float32)),
            cmp_dtype,
        )
    atten_out = None
    atten_out_batch = None
    if not npu_only:
        # cpu by batch
        atten_out_batch = call_cpu_numpy_by_batch(
            batch,
            qn,
            kvn,
            qs,
            kvs,
            d,
            query,
            key,
            value,
            mask,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            block_table,
            key_antiquant_scale,
            value_antiquant_scale,
            None,
            scale,
            pre_tokens,
            next_tokens,
            layout,
            sparse_mode,
            block_size,
            key_antiquant_compute_mode,
            value_antiquant_compute_mode,
            softmax_lse_flag,
            souter,
            sinner,
            q_dtype,
            kv_dtype,
            kv_scale_dtype,
        )
        atten_out_batch.tofile("./cpu_batch.bin")

        # cpu by sinner and souter
        atten_out = call_cpu_numpy(
            batch,
            qn,
            kvn,
            qs,
            kvs,
            d,
            query,
            key,
            value,
            mask,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            block_table,
            key_antiquant_scale,
            value_antiquant_scale,
            None,
            scale,
            pre_tokens,
            next_tokens,
            layout,
            sparse_mode,
            block_size,
            key_antiquant_compute_mode,
            value_antiquant_compute_mode,
            softmax_lse_flag,
            souter,
            sinner,
            q_dtype,
            kv_dtype,
            kv_scale_dtype,
        )
        atten_out.tofile("./cpu.bin")

    if atten_out_batch is None and os.path.exists("./cpu_batch.bin"):
        atten_out_batch = np.fromfile("./cpu_batch.bin", dtype=query.dtype).astype(
            np.float32
        )

    if atten_out is None and os.path.exists("./cpu.bin"):
        atten_out = np.fromfile("./cpu.bin", dtype=query.dtype).astype(np.float32)

    if fuzz_cpu is not None:
        print("cpu vs fuzz cpu")
        check_result(
            torch.from_numpy(atten_out.astype(np.float32)),
            torch.from_numpy(fuzz_cpu.astype(np.float32)),
            cmp_dtype,
        )

        print("cpu by batch vs fuzz cpu")
        check_result(
            torch.from_numpy(atten_out_batch.astype(np.float32)),
            torch.from_numpy(fuzz_cpu.astype(np.float32)),
            cmp_dtype,
        )

    print("cpu by batch vs cpu")
    check_result(
        torch.from_numpy(atten_out_batch.astype(np.float32)),
        torch.from_numpy(atten_out.astype(np.float32)),
        cmp_dtype,
    )
    npu_out = None
    if not cpu_only:
        query = numpy2tensor(query)
        key = numpy2tensor(key, kv_dtype)
        value = numpy2tensor(value, kv_dtype)
        key_cache = numpy2tensor(key_cache, kv_dtype)
        value_cache = numpy2tensor(value_cache, kv_dtype)
        mask = numpy2tensor(mask)
        block_table = numpy2tensor(block_table)
        actual_seq_lengths = numpy2tensor(actual_seq_lengths)
        actual_seq_lengths_kv = numpy2tensor(actual_seq_lengths_kv)
        key_antiquant_scale = (
            numpy2tensor(key_antiquant_scale)
            if key_antiquant_scale_cache is None
            else numpy2tensor(key_antiquant_scale_cache)
        )
        value_antiquant_scale = (
            numpy2tensor(value_antiquant_scale)
            if value_antiquant_scale_cache is None
            else numpy2tensor(value_antiquant_scale_cache)
        )

        print(
            f"query shape: {query.shape}, dtype: {query.dtype}, value: {query.cpu().flatten()[:10]}"
        )
        print(f"key shape: {key.shape}, dtype: {key.dtype}")
        print(f"value shape: {value.shape}, dtype: {value.dtype}")
        print(
            f"key_cache shape: {key_cache.shape}, dtype: {key_cache.dtype}, value: {key_cache.cpu().flatten()[:10]}"
        )
        print(
            f"value_cache shape: {value_cache.shape}, dtype: {value_cache.dtype}, value: {value_cache.cpu().flatten()[:10]}"
        )
        print(f"key_antiquant_compute_mode: {key_antiquant_compute_mode}")
        print(
            f"key_antiquant_scale shape: {key_antiquant_scale.shape}, dtype: {key_antiquant_scale.dtype}, value: {key_antiquant_scale.cpu().flatten()[:10]}"
        )
        print(f"value_antiquant_compute_mode: {value_antiquant_compute_mode}")
        print(
            f"value_antiquant_scale shape: {value_antiquant_scale.shape}, dtype: {value_antiquant_scale.dtype}, value: {value_antiquant_scale.cpu().flatten()[:10]}"
        )
        if actual_seq_lengths is not None:
            print(
                f"actual_seq_lengths shape: {actual_seq_lengths.shape}, dtype: {actual_seq_lengths.dtype}"
            )
        if actual_seq_lengths_kv is not None:
            print(
                f"actual_seq_lengths_kv shape: {actual_seq_lengths_kv.shape}, dtype: {actual_seq_lengths_kv.dtype}"
            )

        out_dtype = get_npu_dtype(q_dtype) if q_dtype == "bf16" else None
        print(
            f"query_dtype: {get_npu_dtype(q_dtype)}, key_dtype: {get_npu_dtype(kv_dtype)}, value_dtype: {get_npu_dtype(kv_dtype)}, key_antiquant_scale_dtype: {get_npu_dtype(kv_scale_dtype)}, value_antiquant_scale_dtype: {get_npu_dtype(kv_scale_dtype)}, out_dtype: {'None'}"
        )
        if block_table is None:
            npu_out, _ = call_npu(
                query,
                key,
                value,
                mask,
                actual_seq_lengths,
                actual_seq_lengths_kv,
                block_table,
                key_antiquant_scale,
                value_antiquant_scale,
                learnable_sink,
                qn,
                kvn,
                scale,
                pre_tokens,
                next_tokens,
                layout,
                sparse_mode,
                block_size,
                key_antiquant_compute_mode,
                value_antiquant_compute_mode,
                softmax_lse_flag,
                get_npu_dtype(q_dtype),
                get_npu_dtype(kv_dtype),
                get_npu_dtype(kv_dtype),
                get_npu_dtype(kv_scale_dtype),
                get_npu_dtype(kv_scale_dtype),
                out_dtype,
                max_seq_q,
                max_seq_kv,
            )
        else:
            npu_out, _ = call_npu(
                query,
                key_cache,
                value_cache,
                mask,
                actual_seq_lengths,
                actual_seq_lengths_kv,
                block_table,
                key_antiquant_scale,
                value_antiquant_scale,
                learnable_sink,
                qn,
                kvn,
                scale,
                pre_tokens,
                next_tokens,
                layout,
                sparse_mode,
                block_size,
                key_antiquant_compute_mode,
                value_antiquant_compute_mode,
                softmax_lse_flag,
                get_npu_dtype(q_dtype),
                get_npu_dtype(kv_dtype),
                get_npu_dtype(kv_dtype),
                get_npu_dtype(kv_scale_dtype),
                get_npu_dtype(kv_scale_dtype),
                out_dtype,
                max_seq_q,
                max_seq_kv,
            )
        torch.save(npu_out.cpu(), "./npu.bin")
    if npu_out is None and os.path.exists("./npu.bin"):
        npu_out = torch.load("./npu.bin")
    if fuzz_npu is not None:
        print("fuzz npu vs npu")
        check_result(
            torch.from_numpy(fuzz_npu.astype(np.float32)),
            npu_out.cpu().to(torch.float32),
            cmp_dtype,
        )

    if fuzz_cpu is not None:
        print("fuzz cpu vs npu")
        check_result(
            torch.from_numpy(fuzz_cpu.astype(np.float32)),
            npu_out.cpu().to(torch.float32),
            cmp_dtype,
        )

    # print("cpu by batch vs npu")
    # check_result(torch.from_numpy(atten_out_batch.astype(np.float32)), npu_out.cpu().to(torch.float32), str(npu_out.dtype))

    print("cpu vs npu")
    check_result(
        torch.from_numpy(atten_out.astype(np.float32)),
        npu_out.cpu().to(torch.float32),
        cmp_dtype,
    )


def check_softmax():
    data = np.random.uniform(-10, 10, (3, 1024)).astype(np.float32)
    block_sum = np.zeros((3, 1), dtype=np.float32)
    block_max = np.ones((3, 1), dtype=np.float32) * (-float("inf"))
    block_exp = np.ones((3, 1), dtype=np.float32)
    out = np.zeros_like(data, dtype=np.float32)
    for idx in range(1024 // 128):
        rst, block_sum, block_max, block_exp = softmax_flash(
            data[:, idx * 128 : idx * 128 + 128],
            block_sum,
            block_max,
            block_exp,
            idx == 0,
        )
        out[:, idx * 128 : idx * 128 + 128] = rst
        for i in range(0, idx):
            out[:, i * 128 : i * 128 + 128] *= block_exp
        if idx == 1024 // 128 - 1:
            out /= block_sum
    gold, g_sum, g_max = softmax(data)
    check_result(
        torch.from_numpy(gold).to(torch.float32),
        torch.from_numpy(out).to(torch.float32),
        "float32",
    )


if __name__ == "__main__":
    # check_softmax()
    main()
