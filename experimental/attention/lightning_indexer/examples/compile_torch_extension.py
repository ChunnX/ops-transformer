"""Compile/load the migrated Torch bridge and check Meta output contracts.

This script does not launch an NPU compute kernel.
"""
import importlib
import os

import torch
import torch_npu  # noqa: F401

package = os.environ.get("CANN_OPS_TRANSFORMER_PACKAGE", "cann_ops_transformer")
adapter = importlib.import_module(f"{package}.ops.attention.lightning_indexer.lightning_indexer")
for q_shape, k_shape, q_layout, k_layout, expected in [
    ((2, 8, 32, 128), (2, 128, 1, 128), "BSND", "BSND", (2, 8, 1, 64)),
    ((16, 32, 128), (256, 1, 128), "TND", "TND", (16, 1, 64)),
    ((16, 32, 128), (8, 128, 1, 128), "TND", "PA_BSND", (16, 1, 64)),
]:
    q = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(k_shape, device="meta", dtype=q.dtype)
    weights = torch.empty(q_shape[:-1], device="meta", dtype=torch.float32)
    indices, values = torch.ops.cann_ops_transformer.lightning_indexer(
        q, k, weights, layout_query=q_layout, layout_key=k_layout,
        sparse_count=64, kv_block_len=4, init_num=4, local_num=32, return_value=True,
    )
    assert tuple(indices.shape) == expected and indices.dtype == torch.int32
    assert tuple(values.shape) == expected and values.dtype == q.dtype
    print(f"META PASS: {q_layout}/{k_layout}, output={expected}", flush=True)

bridge = adapter.lightning_indexer_op_builder.load()
assert callable(bridge.lightning_indexer)
print("TORCH BRIDGE COMPILE/LOAD PASS", flush=True)
