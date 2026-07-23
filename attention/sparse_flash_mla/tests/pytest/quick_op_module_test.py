#!/usr/bin/env python3
"""Minimal op_module call test without importing cann_ops_transformer.ops __init__."""

import importlib.util
import os
import sys
import traceback
import types

TORCH_EXT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../torch_extension"))
sys.path.insert(0, TORCH_EXT)


def bootstrap_cann_ops_transformer():
    pkg = types.ModuleType("cann_ops_transformer")
    pkg.__path__ = [os.path.join(TORCH_EXT, "cann_ops_transformer")]
    pkg.__file__ = os.path.join(TORCH_EXT, "cann_ops_transformer", "__init__.py")
    sys.modules["cann_ops_transformer"] = pkg


def load_sparse_flash_mla_module():
    bootstrap_cann_ops_transformer()
    mod_path = os.path.join(TORCH_EXT, "cann_ops_transformer", "ops", "sparse_flash_mla.py")
    spec = importlib.util.spec_from_file_location(
        "cann_ops_transformer.ops.sparse_flash_mla", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    print("=== load op_module (sparse_flash_mla only) ===")
    smla_mod = load_sparse_flash_mla_module()
    op_module = smla_mod.op_module
    print("op_module:", op_module)
    print("exports:", [x for x in dir(op_module) if "sparse" in x])

    import torch
    import torch_npu

    torch_npu.npu.set_device(0)
    print("device:", torch_npu.npu.get_device_name(0))

    import sparse_flash_mla_golden as g

    params = {
        "testcase_name": "dspark_swa_sparse",
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
        "q_type": __import__("torch").bfloat16,
        "ori_kv_type": __import__("torch").bfloat16,
        "B": 1,
        "S1": 4,
        "T1": 4,
        "T2": None,
        "T3": None,
        "S2": None,
        "N1": 64,
        "N2": 1,
        "D": 512,
        "K": 256,
        "block_num1": 16,
        "block_size1": 128,
        "block_num2": None,
        "block_size2": None,
        "cu_seqlens_q": [0, 4],
        "cu_seqlens_ori_kv": None,
        "cu_seqlens_cmp_kv": None,
        "seqused_q": None,
        "seqused_ori_kv": [1024],
        "seqused_cmp_kv": None,
        "cmp_residual_kv": None,
        "softmax_scale": 0.04419417,
        "cmp_ratio": 1,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": 130,
        "ori_win_right": 3,
        "ori_sparse_indices_mode": "dspark",
        "ori_kv_topk_mode": "no",
        "cmp_sparse_indices_mode": None,
        "cmp_kv_topk_mode": "full",
        "return_softmax_lse": False,
        "q_datarange": [-10, 10],
        "ori_kv_datarange": [-10, 10],
        "cmp_kv_datarange": [-10, 10],
    }
    data = g.gen_data(params)
    inp = data["input"]
    meta_in = data["metadata_input"]

    q = inp["q"].npu()
    ori_kv = inp["ori_kv"].npu()
    ori_sparse = inp["ori_sparse_indices"].npu()
    ori_bt = inp["ori_block_table"].npu()
    cu_q = inp["cu_seqlens_q"].npu()
    seq_ori = inp["seqused_ori_kv"].npu()
    sinks = inp["sinks"].npu()

    print("=== op_module.sparse_flash_mla_metadata ===")
    metadata = op_module.sparse_flash_mla_metadata(
        meta_in["N1"], meta_in["N2"], meta_in["D"],
        cu_q, None, None, None,
        seq_ori, None, None, None, None,
        meta_in["B"], meta_in["max_seqlen_q"], meta_in["max_seqlen_ori_kv"],
        meta_in["max_seqlen_cmp_kv"],
        meta_in["K"], 0, meta_in["cmp_ratio"], meta_in["ori_mask_mode"], 3,
        meta_in["ori_win_left"], meta_in["ori_win_right"],
        meta_in["layout_q"], meta_in["layout_kv"], True, False)
    print("metadata ok:", metadata.shape, metadata.device)

    print("=== op_module.sparse_flash_mla ===")
    out, lse = op_module.sparse_flash_mla(
        q, ori_kv, None,
        ori_sparse, None,
        ori_bt, None,
        cu_q, None, None, None,
        seq_ori, None, None,
        None, None,
        sinks, metadata,
        inp["softmax_scale"], meta_in["cmp_ratio"],
        meta_in["ori_mask_mode"], 3,
        meta_in["ori_win_left"], meta_in["ori_win_right"],
        meta_in["layout_q"], meta_in["layout_kv"],
        1, False)
    torch.npu.synchronize()
    print("out ok:", out.shape, out.dtype, out.device)
    print("lse ok:", lse.shape, lse.dtype)
    print("SUCCESS: op_module.sparse_flash_mla(...) works")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED:", exc)
        traceback.print_exc()
        sys.exit(1)
