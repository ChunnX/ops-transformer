"""A2/A3 ratio=2 regression through Metadata + SparseFlashMla on an NPU."""

import pytest
import torch
import torch_npu

import result_compare_method
import sparse_flash_mla_golden
import utils
from batch import sparse_flash_mla_process


BASE_SEQUENCE_CONFIGS = [
    (min(7, ori_len), ori_len, num_heads_q, cmp_topk)
    for ori_len in (2, 3, 1022, 1023, 1024, 1025, 1026, 1027)
    for num_heads_q in (1, 32, 128)
    for cmp_topk in (512, 1024)
]

# Keep long-sequence coverage affordable by isolating S1/S2 from the N1 sweep.
EXTENDED_SEQUENCE_SHAPES = (
    (1, 16),
    (8, 32),
    (16, 16),
    (16, 64),
    (31, 127),
    (32, 128),
    (64, 256),
    (127, 511),
    (128, 128),
    (128, 512),
    (255, 1023),
    (256, 1024),
    (512, 512),
    (511, 2047),
    (512, 2048),
    (1024, 1024),
    (1023, 4095),
    (1024, 4096),
    (1024, 8191),
    (1024, 8192),
    (1024, 8193),
)
EXTENDED_SEQUENCE_CONFIGS = [
    (q_len, ori_len, 1, cmp_topk)
    for q_len, ori_len in EXTENDED_SEQUENCE_SHAPES
    for cmp_topk in (512, 1024)
]
RATIO2_SEQUENCE_CONFIGS = BASE_SEQUENCE_CONFIGS + EXTENDED_SEQUENCE_CONFIGS


@pytest.mark.parametrize("layout", ["BSND", "TND", "PA_BBND"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "q_len,ori_len,num_heads_q,cmp_topk", RATIO2_SEQUENCE_CONFIGS
)
def test_ratio2(layout, dtype, q_len, ori_len, num_heads_q, cmp_topk):
    # 288 base cases + 252 extended S1/S2 cases = 540 A2/A3 cases.
    # Odd lengths exercise residual=1. 511/512/513 compressed tokens exercise
    # the arch22 S2 boundary.
    mode = "CSA"
    batch = 2
    cmp_len = ori_len // 2
    q_layout = "BSND" if layout == "BSND" else "TND"
    params = {
        "testcase_name": (
            f"ratio2_{mode}_{layout}_{str(dtype).split('.')[-1]}_"
            f"s1{q_len}_s2{ori_len}_"
            f"n{num_heads_q}_k{cmp_topk}"
        ),
        "template_mode": mode,
        "layout_q": q_layout,
        "layout_kv": layout,
        "q_type": dtype,
        "ori_kv_type": dtype,
        "cmp_kv_type": dtype,
        "B": batch,
        "S1": q_len,
        "S2": ori_len,
        "T1": batch * q_len,
        "T2": batch * ori_len,
        "T3": batch * cmp_len,
        "N1": num_heads_q,
        "N2": 1,
        "D": 512,
        "K": cmp_topk,
        "cmp_ratio": 2,
        "seqused_ori_kv": [ori_len] * batch,
        "seqused_cmp_kv": [cmp_len] * batch,
        "cmp_residual_kv": [ori_len % 2] * batch,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "softmax_scale": 512 ** -0.5,
        "return_softmax_lse": True,
        "q_datarange": [-2, 2],
        "ori_kv_datarange": [-2, 2],
        "cmp_kv_datarange": [-2, 2],
    }
    if q_layout == "TND":
        params["cu_seqlens_q"] = [0, q_len, batch * q_len]
    if layout == "TND":
        # Compress each batch separately before accumulating its storage offset.
        params["cu_seqlens_ori_kv"] = [0, ori_len, batch * ori_len]
        params["cu_seqlens_cmp_kv"] = [0, cmp_len, batch * cmp_len]
    if layout == "PA_BBND":
        params.update({
            "block_size1": 128,
            "block_size2": 16,
            "block_num1": batch * ((ori_len + 127) // 128),
            "block_num2": batch * ((cmp_len + 15) // 16),
        })

    torch_npu.npu.set_device(0)
    torch.manual_seed(0)
    # Use the same two-step default expansion as the existing single-case runner.
    combinations = utils.generate_param_combinations([{key: [value] for key, value in params.items()}])
    case = utils.generate_case_with_default_param(combinations[0], "off")
    data = sparse_flash_mla_golden.gen_data(case)
    output, lse = sparse_flash_mla_process.call_npu(data)
    for name, expected, actual in (
        ("attn_out", data["cpu_output"], output),
        ("softmax_lse", data["softmax_lse"], lse),
    ):
        assert actual.shape == expected.shape, f"{name}: {actual.shape} != {expected.shape}"
        result, percent = result_compare_method.check_result(expected, actual)
        assert result == "Pass", f"{name}: {result}, {percent}%"
