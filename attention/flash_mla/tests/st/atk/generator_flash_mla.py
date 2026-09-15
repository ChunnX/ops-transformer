# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Link YAML parameters by updating their JSON dtype, shape and range_values."""

from itertools import accumulate

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.generate_types.default import DefaultGenerator


def scalar(config):
    value = config.range_values
    return value[0] if isinstance(value, list) else value


def dense_shape(layout, batch, size, heads, dim):
    if layout == "TND":
        return [size, heads, dim]
    if layout == "BSH":
        return [batch, size, heads * dim]
    if layout == "BSND":
        return [batch, size, heads, dim]
    return [batch, heads, size, dim]


def sequence_pattern(config, batch, capacity):
    values = config.range_values
    if not isinstance(values, list):
        values = [values]
    return [min(capacity, max(0, int(values[i % len(values)]))) for i in range(batch)]


def partition(total, weights):
    """Cumulative TND lengths must end at the tensor's stored total."""
    if not sum(weights):
        weights = [1] * len(weights)
    denominator = sum(weights)
    lengths = [total * weight // denominator for weight in weights]
    remainder = total - sum(lengths)
    for index, weight in enumerate(weights):
        if weight and remainder:
            lengths[index] += 1
            remainder -= 1
    return lengths


def link_sequence(config, lengths, cumulative, mode):
    if mode in ("absent", "empty"):
        config.dtype = "int64"
        config.shape = [len(lengths)] if mode == "absent" else [0]
        config.range_values = ["default"] if mode == "absent" else [0]
        return
    values = list(accumulate(lengths)) if cumulative else lengths
    if mode == "broadcast":
        values = values[:1]
    elif mode == "extra":
        values = values + values[-1:]
    # ATK's fixed integer encoding materializes these exact int64 values.
    config.dtype = "int64_fixed"
    config.shape = [len(values)]
    config.range_values = values


@GENERATOR_REGISTRY.register("generate_flash_mla")
class FlashMlaGenerator(DefaultGenerator):
    def after_case_config(self, case_config):
        p = {(item[0] if isinstance(item, list) else item).name:
             (item[0] if isinstance(item, list) else item) for item in case_config.inputs}
        quant = p["query"].dtype == "int8"
        layout = scalar(p["input_layout"])
        cache = scalar(p["kv_cache_layout"])
        sparse = int(scalar(p["sparse_mode"]))
        heads = int(scalar(p["num_heads"]))
        inner = int(scalar(p["inner_precise"]))
        block = int(scalar(p["block_size"]))
        mode = scalar(p["sequence_mode"])

        if quant:
            layout = layout.replace("BNSD", "BSND")
            cache, block = "NZ", 128
            heads = 1 << (heads.bit_length() - 1)
            sparse = 3 if sparse == 4 else sparse
            inner %= 2
        elif cache == "dense":
            block = 0
        else:
            block = max(16, block)
        base_layout = layout.split("_")[0]
        tnd = base_layout == "TND"

        batch_attr = scalar(p["batch_size"])
        q_attr = scalar(p["q_seq_size"])
        kv_attr = scalar(p["kv_seq_size"])
        batch = p["actual_seq_lengths_q"].shape[0] if batch_attr == "default" else int(batch_attr)
        q_size = p["query"].shape[0 if tnd else 1] if q_attr == "default" else int(q_attr)
        kv_size = p["key"].shape[0 if tnd else 1] if kv_attr == "default" else int(kv_attr)
        if quant:
            q_size = min(q_size, 32)
            if not tnd:
                sparse = 0 if q_size == 1 else (9 if sparse == 9 else 3)
        q_lengths = sequence_pattern(p["actual_seq_lengths_q"], batch, q_size)
        if tnd:
            q_lengths = partition(q_size, q_lengths)
        elif quant:
            q_lengths = [q_size] * batch

        # Missing/broadcast/extra sequence descriptors are only legal outside TND;
        # paged KV always needs nonempty lengths for these nonempty-batch cases.
        if tnd and mode not in ("explicit", "offset"):
            mode = "explicit"
        if cache != "dense" and mode in ("absent", "empty"):
            mode = "explicit"
        if mode in ("absent", "empty", "broadcast"):
            q_lengths = [q_size] * batch

        if sparse in (3, 9):
            kv_size = max(kv_size, sum(q_lengths) if tnd and cache == "dense" else max(q_lengths))
        if sparse == 9 and cache == "dense" and not tnd:
            # The nonquant tree checker compares padded storage sizes as well.
            kv_size = max(kv_size, q_size)
        if cache != "dense":
            kv_size = (kv_size + block - 1) // block * block
        kv_lengths = sequence_pattern(p["actual_seq_lengths_kv"], batch, kv_size)
        if tnd and cache == "dense":
            if sparse in (3, 9):
                remaining = partition(kv_size - sum(q_lengths), kv_lengths)
                kv_lengths = [q + k for q, k in zip(q_lengths, remaining)]
            else:
                kv_lengths = partition(kv_size, kv_lengths)
        elif mode in ("absent", "empty", "broadcast"):
            kv_lengths = [kv_size] * batch
        elif sparse in (3, 9):
            kv_lengths = [max(q, k) for q, k in zip(q_lengths, kv_lengths)]

        p["query"].shape = dense_shape(base_layout, batch, q_size, heads, 512)
        p["query_rope"].shape = dense_shape(base_layout, batch, q_size, heads, 64)
        for name in ("key", "value", "query_rope", "key_rope"):
            p[name].dtype = "bf16" if quant and "rope" in name else p["query"].dtype

        for name, dim in (("key", 512), ("value", 512), ("key_rope", 64)):
            if cache == "dense":
                shape = dense_shape(base_layout, batch, kv_size, 1, dim)
            else:
                blocks = batch * (kv_size // block)
                if cache == "BSH":
                    shape = [blocks, block, dim]
                elif cache == "BNSD":
                    shape = [blocks, 1, block, dim]
                else:
                    dim0 = 32 if quant and dim == 512 else 16
                    shape = [blocks, 1, dim // dim0, block, dim0]
            p[name].shape = shape
        table = p["block_table"]
        table.shape = [batch, kv_size // block] if block else [batch, 1]
        table.range_values = [0, p["key"].shape[0]] if block else ["default"]
        link_sequence(p["actual_seq_lengths_q"], q_lengths, tnd, mode)
        link_sequence(p["actual_seq_lengths_kv"], kv_lengths, tnd and not block, mode)

        mask = p["atten_mask"]
        has_mask = sparse != 0 or (not quant and scalar(mask) != "default")
        if sparse in (3, 4):
            mask.shape = [2048, 2048]
        elif sparse == 9:
            mask.shape = ([sum(length * length for length in q_lengths)] if tnd
                          else [batch, q_size, q_size])
        else:
            mask.shape = [batch, max(q_lengths) if tnd else q_size,
                          max(kv_lengths) if tnd and not block else kv_size]
        mask.range_values = [0, 1] if has_mask else ["default"]
        if has_mask and not quant and sparse in (0, 4):
            inner |= 2  # Random/window masks may contain fully masked rows.

        scales = ("dequant_scale_query", "key_antiquant_scale", "value_antiquant_scale")
        p[scales[0]].shape = [q_size, heads] if tnd else [batch, q_size, heads]
        for name in scales[1:]:
            p[name].shape = [1]
        if not quant:
            for name in scales:
                p[name].range_values = ["default"]

        linked_attrs = dict(input_layout=layout, input_layer=layout, kv_cache_layout=cache,
                            num_heads=heads, sparse_mode=sparse, inner_precise=inner,
                            block_size=block, query_quant_mode=3 if quant else 0,
                            quant_mode=int(quant), has_atten_mask=has_mask, sequence_mode=mode)
        if not tnd or batch_attr != "default":
            linked_attrs["batch_size"] = batch
        if not tnd or q_attr != "default":
            linked_attrs["q_seq_size"] = q_size
        if not tnd or block or kv_attr != "default":
            linked_attrs["kv_seq_size"] = kv_size
        if sparse in (3, 9):
            linked_attrs.update(pre_tokens=2147483647, next_tokens=0)
        if scalar(p["input_view"]) == "cache_dim0" and (quant or not block):
            linked_attrs["input_view"] = "none"
        for name, value in linked_attrs.items():
            p[name].range_values = value
        return case_config
