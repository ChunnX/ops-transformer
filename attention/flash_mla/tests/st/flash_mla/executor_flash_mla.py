# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FlashMla Torch ST executor with a CPU double-precision reference.

CPU MLA code extracted/adapted from
attention/fused_infer_attention_score/tests/st/aclnnFusedInferAttentionScoreV4/
executor_aclnnFusedInferAttentionScoreV4.py:
  FiaOpPreprocess.concat_rope_tensor, FiaSoftmax, FiaOpForward.compute_once_bnsd,
  _t_indequant_ds_act and _t_indequant_ds_act_golden.
Keep the split RoPE, paged cache, masks, softmax/PV and LSE calculation here.
Adaptations: independent V and K/V scales follow FlashMla's public contract;
FP16/BF16 low precision uses the original FIA kernel's AMLA scaling;
is_benchmark_task selects FP64 golden versus the CPU low precision benchmark.

Accuracy: atk node --backend npu --devices 1 node --backend cpu task
          -c atk_flash_mla.json -p executor_flash_mla.py
          --task accuracy -bd cpu
Every candidate call includes metadata then FlashMla.
"""

import ctypes
import importlib
import math
import os

import torch
from atk.configs.results_config import OutputData
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


def sequence_values(c, query):
    return c["seq_q_values" if query else "seq_kv_values"]


def prepare_mask(c, mask):
    """Materialize FIA mask/window data in the descriptor supplied by ATK."""
    if mask is None:
        return None
    if c["sparse"] == 0:
        # FIA's ordinary mask must contain the window as well as random holes;
        # pre/next also bound the tiles visited by the kernel.
        query = torch.arange(mask.shape[-2], device=mask.device).unsqueeze(1)
        key = torch.arange(mask.shape[-1], device=mask.device).unsqueeze(0)
        window = (key < query - c["pre"]) | (key > query + c["next"])
        return (mask.bool() | window).to(mask.dtype)
    if c["sparse"] in (3, 4):
        return torch.ones_like(mask).triu(1)
    if c["sparse"] != 9:
        return mask
    masks = []
    for length in c["q_lens"]:
        tree = torch.ones(length, length, dtype=mask.dtype)
        for row in range(length):
            parent = row
            while True:
                tree[row, parent] = 0
                if parent == 0:
                    break
                parent = (parent - 1) // 2
        masks.append(tree)
    if c["layout"].startswith("TND"):
        result = torch.cat([tree.flatten() for tree in masks])
    else:
        result = torch.ones(mask.shape, dtype=mask.dtype)
        for batch, tree in enumerate(masks):
            result[batch, : tree.shape[0], : tree.shape[1]] = tree
    if result.shape != mask.shape:
        raise ValueError(
            "Tree mask descriptor does not match the Query sequence lengths"
        )
    return result.to(mask.device)


def dense_rows(x, c, query):
    layout = c["layout"].split("_")[0]
    lengths = c["q_lens"] if query else c["kv_lens"]
    heads = c["n"] if query else 1
    if layout == "TND":
        return [t.transpose(0, 1) for t in x.split(lengths)]
    if layout == "BSH":
        x = x.reshape(c["b"], x.shape[1], heads, -1).transpose(1, 2)
    elif layout == "BSND":
        x = x.transpose(1, 2)
    return [t[:, :length] for t, length in zip(x, lengths)]


def cache_rows(x, c, table):
    if table is None:
        return dense_rows(x, c, False)
    if c["cache"] == "NZ":
        x = x.permute(0, 1, 3, 2, 4).flatten(3)
    elif c["cache"] == "BSH":
        x = x.unsqueeze(1)
    rows = []
    for b, length in enumerate(c["kv_lens"]):
        pages = table[b, : math.ceil(length / c["block"])].long()
        rows.append(x[pages].permute(1, 0, 2, 3).flatten(1, 2)[:, :length])
    return rows


def output_layout(rows, c, lse=False):
    if c["layout"].startswith("TND"):
        x = torch.cat([r.transpose(0, 1) for r in rows])
        return (
            x.transpose(0, 1).contiguous()
            if not lse and c["layout"].endswith("_NTD")
            else x
        )
    dim = 1 if lse else 512
    x = torch.full(
        (c["b"], c["n"], c["sq"], dim), float("inf") if lse else 0, dtype=rows[0].dtype
    )
    for b, row in enumerate(rows):
        x[b, :, : row.shape[1]] = row
    if lse:
        return x
    if c["layout"].endswith("_NBSD"):
        return x.permute(1, 0, 2, 3).contiguous()
    if c["layout"] == "BSND":
        return x.transpose(1, 2).contiguous()
    if c["layout"] == "BSH":
        return x.transpose(1, 2).flatten(2).contiguous()
    return x


def mla_mask(c, f, batch, qlen, klen):
    """FIA V4 MaskGenerator: ordinary, bottom-right causal and band masks."""
    qi = torch.arange(qlen).unsqueeze(1)
    ki = torch.arange(klen).unsqueeze(0)
    if c["sparse"] == 0 and f["atten_mask"] is None:
        return torch.zeros(qlen, klen, dtype=torch.bool)
    if c["sparse"] in (3, 4):
        aligned = qi + klen - qlen
        if c["sparse"] == 3:
            return ki > aligned
        return (ki < aligned - c["pre"]) | (ki > aligned + c["next"])
    if c["sparse"] == 9:
        # FlashMla's tree extension: all prefix tokens followed by the tree mask.
        if c["layout"].startswith("TND"):
            offset = sum(x * x for x in c["q_lens"][:batch])
            tree = (
                f["atten_mask"][offset : offset + qlen * qlen]
                .reshape(qlen, qlen)
                .bool()
            )
        else:
            tree = f["atten_mask"][batch, :qlen, :qlen].bool()
        return torch.cat(
            (torch.zeros(qlen, klen - qlen, dtype=torch.bool), tree), dim=1
        )
    blocked = (ki < qi - c["pre"]) | (ki > qi + c["next"])
    if c["mask"] == "ordinary":
        blocked |= f["atten_mask"][batch, :qlen, :klen].bool()
    return blocked


def fia_mla_float(q, k, v, blocked, scale, output_dtype, golden):
    """FIA MLA CPU reference with AMLA in the FP16/BF16 low precision path.

    Follow FiaBlockVecNonQuantMla::AmlaVecCompute: round the AMLA factor to
    the input dtype, scale the exponential, then round it before BMM2 and
    divide by the equally scaled sum. LSE uses the original sum and maximum.
    The FP64 golden keeps the unscaled softmax and both matmuls in FP64.
    """
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(blocked, -1.7e38)
    row_max = scores.amax(dim=-1, keepdim=True)
    exp = torch.exp(scores - row_max)
    row_sum = exp.sum(dim=-1, keepdim=True)
    if golden:
        probabilities, denominator = exp, row_sum
    else:
        # Match the kernel's FP32 LN2 and RECIP_OF_LN2 constants and ordering.
        ln2 = scores.new_tensor(math.log(2))
        inv_ln2 = ln2.reciprocal()
        log2_max = row_max * inv_ln2
        n = (-log2_max).round()
        factor = torch.exp((log2_max + n) * ln2).to(output_dtype).float()
        probabilities = (exp * factor).to(output_dtype).float()
        denominator = row_sum * factor
    out = (probabilities @ v) / denominator
    lse = row_sum.log() + row_max
    invalid = blocked.all(dim=-1)
    out[:, invalid] = 0
    lse[:, invalid] = float("inf")
    return out, lse


def fia_mla_int8_low(q, k, v, qr, kr, qscale, kscale, vscale, blocked, scale):
    """V4 _t_indequant_ds_act MLA CPU benchmark, retaining its 512-token loop.

    FIA's INT8 path scales the sum of the integer QK and supplied BF16 RoPE
    products. Its ST pre-scales Query RoPE when starting from real-valued RoPE.
    Interpret the tensors supplied to this case using that same contract,
    retaining independent V scaling, softmax quantisation and FP16 accumulation.
    """
    row_max = torch.full((*q.shape[:-1], 1), -float("inf"))
    row_sum = torch.zeros_like(row_max)
    accum = torch.zeros_like(q, dtype=torch.float16)
    for start in range(0, k.shape[-2], 512):
        end = min(start + 512, k.shape[-2])
        # The generated INT8 range makes the FP32 dot product exactly integral.
        scores = q @ k[:, start:end].transpose(-1, -2)
        scores += qr @ kr[:, start:end].transpose(-1, -2)
        scores = scores * qscale * kscale
        scores = (scores * scale).masked_fill(blocked[:, start:end], -1.7e38)
        local_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(row_max, local_max)
        update = torch.exp(row_max - new_max)
        exp = torch.exp(scores - new_max)
        local_factor = torch.exp(local_max - new_max)
        row_sum = update * row_sum + exp.sum(dim=-1, keepdim=True)
        quantized = (exp * (127 / local_factor)).half().round().to(torch.int8)
        partial = (quantized.float() @ v[:, start:end]) / 1024
        partial = partial.half() * local_factor.half()
        accum = accum * update.half() + partial
        row_max = new_max
    out = accum.float() * 1024 / row_sum / 127 * vscale
    lse = row_sum.log() + row_max
    invalid = blocked.all(dim=-1)
    out[:, invalid] = 0
    lse[:, invalid] = float("inf")
    return out, lse


def cpu_mla(inputs, c, f, golden):
    dtype = torch.float64 if golden else torch.float32
    output_dtype = torch.bfloat16 if c["dtype"] in ("bf16", "int8") else torch.float16
    q, qr = [
        dense_rows(inputs[name].to(dtype), c, True) for name in ("query", "query_rope")
    ]
    k, v, kr = [
        cache_rows(inputs[name].to(dtype), c, f["block_table"])
        for name in ("key", "value", "key_rope")
    ]
    qscales = None
    if c["dtype"] == "int8":
        scales = f["dequant_scale_query"].to(dtype).unsqueeze(-1)
        if c["layout"].startswith("TND"):
            qscales = [x.transpose(0, 1) for x in scales.split(c["q_lens"])]
        else:
            qscales = list(scales.transpose(1, 2))
    outputs, lses = [], []
    for b, (qlen, klen) in enumerate(zip(c["q_lens"], c["kv_lens"])):
        if qlen == 0 or klen == 0:
            outputs.append(torch.zeros(c["n"], qlen, 512, dtype=dtype))
            lses.append(torch.full((c["n"], qlen, 1), float("inf"), dtype=dtype))
            continue
        blocked = mla_mask(c, f, b, qlen, klen)
        qb, kb, vb = q[b], k[b], v[b]
        if qscales is not None:
            ks = f["key_antiquant_scale"].to(dtype)
            vs = f["value_antiquant_scale"].to(dtype)
            if not golden:
                out, lse = fia_mla_int8_low(
                    qb, kb, vb, qr[b], kr[b], qscales[b], ks, vs, blocked, c["scale"]
                )
                outputs.append(out)
                lses.append(lse)
                continue
            # The supplied RoPE shares the QK dequantisation factor in FIA.
            # Factor the joint score into concatenated FP64 Q/K products.
            qb, kb, vb = qb * qscales[b], kb * ks, vb * vs
            qr[b], kr[b] = qr[b] * qscales[b], kr[b] * ks
        # FiaOpPreprocess.concat_rope_tensor: keep MLA as concatenated Q/K.
        out, lse = fia_mla_float(
            torch.cat((qb, qr[b]), dim=-1),
            torch.cat((kb, kr[b]), dim=-1),
            vb,
            blocked,
            c["scale"],
            output_dtype,
            golden,
        )
        outputs.append(out)
        lses.append(lse)
    out = output_layout(outputs, c)
    lse = output_layout(lses, c, lse=True)
    if not golden:
        out, lse = out.to(output_dtype), lse.float()
    return (out, lse) if c["lse"] else out


def prepare_inputs(input_data, case_config):
    """Read explicit ATK inputs; prepare only mask templates and requested views."""
    params = input_data.kwargs
    data = {name: params[name] for name in ("query", "query_rope", "key_rope")}
    for name in ("key", "value"):
        if len(params[name]) != 1:
            raise ValueError(f"{name} must contain exactly one tensor")
        data[name] = params[name][0]
    optional_names = (
        "actual_seq_lengths_q",
        "actual_seq_lengths_kv",
        "block_table",
        "atten_mask",
        "dequant_scale_query",
        "key_antiquant_scale",
        "value_antiquant_scale",
    )
    f = {name: params[name] for name in optional_names}
    aliases = dict(
        n="num_heads",
        layout="input_layout",
        cache="kv_cache_layout",
        block="block_size",
        sparse="sparse_mode",
        pre="pre_tokens",
        next="next_tokens",
        inner="inner_precise",
        lse="softmax_lse_flag",
        scale="scale_value",
        shared="shared_kv",
        view="input_view",
        seq="sequence_mode",
    )
    c = {name: params[source] for name, source in aliases.items()}
    c["params"] = params
    # ATK promotes tensors for the FP64 golden; retain the declared input dtype.
    c["dtype"] = case_config.get_input_data_config(name="query").dtype
    tnd = c["layout"].startswith("TND")
    c["b"] = f["actual_seq_lengths_q"].numel() if tnd else data["query"].shape[0]

    def lengths(query):
        seq = f["actual_seq_lengths_q" if query else "actual_seq_lengths_kv"]
        values = None if seq is None else seq.detach().cpu().tolist()
        c["seq_q_values" if query else "seq_kv_values"] = values
        tensor = data["query" if query else "key"]
        if tnd and (query or not c["block"]):
            return [end - start for start, end in zip([0] + values, values)]
        axis = 2 if c["layout"].startswith("BNSD") else 1
        stored = (
            f["block_table"].shape[1] * c["block"]
            if not query and c["block"]
            else tensor.shape[axis]
        )
        if not values or (query and c["dtype"] == "int8"):
            return [stored] * c["b"]
        return values * c["b"] if len(values) == 1 else values[: c["b"]]

    c["q_lens"], c["kv_lens"] = lengths(True), lengths(False)
    axis = 2 if c["layout"].startswith("BNSD") else 1
    c["sq"] = max(c["q_lens"]) if tnd else data["query"].shape[axis]
    c["sk"] = (
        f["block_table"].shape[1] * c["block"]
        if c["block"]
        else (max(c["kv_lens"]) if tnd else data["key"].shape[axis])
    )
    c["mask"] = (
        "none"
        if f["atten_mask"] is None
        else {0: "ordinary", 3: "causal", 4: "causal", 9: "tree"}[c["sparse"]]
    )
    f["atten_mask"] = prepare_mask(c, f["atten_mask"])
    device = data["query"].device
    if c["seq"] == "offset":
        for name in ("actual_seq_lengths_q", "actual_seq_lengths_kv"):
            tensor = f[name]
            f[name] = torch.cat((tensor.new_zeros(1), tensor))[1:]
    if c["view"] == "query":
        for name in ("query", "query_rope"):
            tensor = data[name]
            storage = torch.empty((*tensor.shape, 2), dtype=tensor.dtype, device=device)
            storage[..., 0] = tensor
            data[name] = storage[..., 0]
    elif c["view"] == "cache_dim0":
        for name in ("key", "value", "key_rope"):
            tensor = data[name]
            storage = torch.empty(
                (tensor.shape[0] * 2, *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=device,
            )
            storage[::2] = tensor
            data[name] = storage[::2]
    if c["shared"]:
        data["value"] = data["key"]
    return c, data, f


def output_info(c, query):
    shape = list(query.shape)
    tokens = sum(c["q_lens"])
    if c["layout"].endswith("_NBSD"):
        shape = [c["n"], c["b"], c["sq"], 512]
    elif c["layout"] == "TND_NTD":
        shape = [c["n"], tokens, 512]
    lse_shape = (
        (
            [tokens, c["n"], 1]
            if c["layout"].startswith("TND")
            else [c["b"], c["n"], c["sq"], 1]
        )
        if c["lse"]
        else [0]
    )
    dtype = torch.bfloat16 if c["dtype"] == "int8" else query.dtype
    return [
        OutputData(shape=shape, dtype=str(dtype)),
        OutputData(shape=lse_shape, dtype="torch.float32"),
    ]


@register("executor_flash_mla")
class FlashMlaApi(BaseApi):
    def init_by_input_data(self, input_data):
        self.spec, self.data, self.f = prepare_inputs(
            input_data, self.task_result.case_config
        )
        if self.task_result.backend == "pyaclnn":
            # ATK requires output descriptors even for a performance-only run.
            infos = output_info(self.spec, self.data["query"])
            self.task_result.output_info_list = infos if self.spec["lse"] else infos[:1]
            return
        if self.data["query"].device.type != "cpu":
            self.ops = importlib.import_module(
                os.environ.get("FLASH_MLA_TORCH_MODULE", "cann_ops_transformer")
            )
            getattr(self.ops, "flash_mla_metadata")
            getattr(self.ops, "flash_mla")

    def __call__(self, input_data, with_output=False):
        c, d, f = self.spec, self.data, self.f
        if d["query"].device.type == "cpu":
            return cpu_mla(d, c, f, self.task_result.is_benchmark_task)
        params = c["params"]
        metadata_names = (
            "batch_size",
            "q_seq_size",
            "kv_seq_size",
            "input_layer",
            "block_size",
            "sparse_mode",
            "pre_tokens",
            "next_tokens",
            "has_atten_mask",
            "quant_mode",
        )
        metadata = self.ops.flash_mla_metadata(
            c["n"],
            **{name: params[name] for name in metadata_names},
            actual_seq_lengths_q=f["actual_seq_lengths_q"],
            actual_seq_lengths_kv=f["actual_seq_lengths_kv"],
            device=d["query"].device,
        )
        attribute_names = (
            "scale_value",
            "pre_tokens",
            "next_tokens",
            "input_layout",
            "num_key_value_heads",
            "sparse_mode",
            "inner_precise",
            "block_size",
            "softmax_lse_flag",
            "key_antiquant_mode",
            "value_antiquant_mode",
            "query_quant_mode",
        )
        out, lse = self.ops.flash_mla(
            d["query"],
            [d["key"]],
            [d["value"]],
            d["query_rope"],
            d["key_rope"],
            metadata,
            c["n"],
            **f,
            **{name: params[name] for name in attribute_names},
        )
        if not c["lse"] and lse.numel() != 0:
            raise AssertionError("Disabled LSE must be an empty tensor")
        return (out, lse) if c["lse"] else out


@register("executor_aclnn_fia_v4")
class FiaV4Api(AclnnBaseApi):
    """Original FIA V4, timed by ATK's normal PyAclnnBackend e2e executor."""

    def init_by_input_data(self, input_data):
        from atk.tasks.backends.lib_interface import acl_wrapper as acl

        c, data, f = prepare_inputs(input_data, self.task_result.case_config)
        self.data, self.f = data, f

        def tensor(value):
            return (
                ctypes.cast(None, acl.TensorPtr)
                if value is None
                else self.torch_tensor_to_acl(value.contiguous())
            )

        def lengths(query):
            values = sequence_values(c, query)
            if not values:
                return ctypes.POINTER(acl.AclIntArray)()
            return acl.nnopbase.create_x_list([ctypes.c_int64(x) for x in values])

        t = {name: tensor(value) for name, value in data.items()}
        extra = {
            name: tensor(f[name])
            for name in (
                "atten_mask",
                "block_table",
                "dequant_scale_query",
                "key_antiquant_scale",
                "value_antiquant_scale",
            )
        }
        nil = tensor(None)
        keys = acl.nnopbase.create_x_list([t["key"]])
        values = acl.nnopbase.create_x_list([t["value"]])
        args = [
            t["query"],
            keys,
            values,
            nil,
            extra["atten_mask"],
            lengths(True),
            lengths(False),
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            extra["block_table"],
            nil,
            nil,
            extra["key_antiquant_scale"],
            nil,
            extra["value_antiquant_scale"],
            nil,
            nil,
            nil,
            ctypes.POINTER(acl.AclIntArray)(),
            t["query_rope"],
            t["key_rope"],
            nil,
            extra["dequant_scale_query"],
            nil,
        ]
        args.extend(
            [
                ctypes.c_int64(c["n"]),
                ctypes.c_double(c["scale"]),
                ctypes.c_int64(c["pre"]),
                ctypes.c_int64(c["next"]),
                ctypes.c_char_p(c["layout"].encode()),
                ctypes.c_int64(c["params"]["num_key_value_heads"]),
                ctypes.c_int64(c["sparse"]),
                ctypes.c_int64(c["inner"]),
                ctypes.c_int64(c["block"]),
                ctypes.c_int64(0),
                ctypes.c_bool(c["lse"]),
                ctypes.c_int64(c["params"]["key_antiquant_mode"]),
                ctypes.c_int64(c["params"]["value_antiquant_mode"]),
                ctypes.c_int64(c["params"]["query_quant_mode"]),
            ]
        )
        device = torch.device("npu", self.device_id)
        self.output_tensors = [
            torch.empty(
                info.shape,
                dtype=getattr(torch, info.dtype.removeprefix("torch.")),
                device=device,
            )
            for info in output_info(c, data["query"])
        ]
        outputs = [tensor(x) for x in self.output_tensors]
        args.extend(outputs)
        return args, outputs if c["lse"] else outputs[:1]
