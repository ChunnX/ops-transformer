import numpy as np
import pytest
import torch

import kv_quant_sparse_flash_attention_golden as golden


def _make_params(
    testcase_name,
    layout_query="BSND",
    layout_kv="PA_BSND",
    q_type=torch.bfloat16,
    kv_dtype=None,
    B=2,
    S1=4,
    S2=32,
    N1=8,
    N2=1,
    D=256,
    K=16,
    scale_value=1.0 / 24.0,
    key_quant_mode=2,
    value_quant_mode=2,
    sparse_block_size=1,
    tile_size=128,
    rope_head_dim=64,
    sparse_mode=3,
    attention_mode=2,
    quant_scale_repo_mode=1,
    block_size=16,
    block_num=4,
    actual_seq_q=None,
    actual_seq_kv=None,
):
    if actual_seq_q is None:
        actual_seq_q = [3, 4]
    if actual_seq_kv is None:
        actual_seq_kv = [17, 32]
    return (
        testcase_name,
        layout_query,
        layout_kv,
        q_type,
        kv_dtype,
        B,
        S1,
        S2,
        N1,
        N2,
        D,
        K,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        tile_size,
        rope_head_dim,
        sparse_mode,
        attention_mode,
        quant_scale_repo_mode,
        block_size,
        block_num,
        actual_seq_q,
        actual_seq_kv,
    )


def _tensor_signature(tensor):
    if tensor is None:
        return None
    array = tensor.detach().cpu()
    if array.dtype == torch.float8_e4m3fn:
        return (
            tuple(array.shape),
            str(array.dtype),
            array.view(torch.uint8).sum().item(),
        )
    return (
        tuple(array.shape),
        str(array.dtype),
        array.to(torch.float32).sum().item(),
    )


def test_generate_and_save_testdata_bsnd_pa_shapes_match_contract():
    np.random.seed(0)
    test_data = golden.generate_and_save_testdata(
        _make_params("snapshot_bsnd_pa"),
        compute_golden=False,
    )

    input_data = test_data["input"]
    assert tuple(input_data["query"].shape) == (2, 4, 8, 320)
    assert tuple(input_data["key"].shape) == (4, 16, 1, 392)
    assert tuple(input_data["value"].shape) == (4, 16, 1, 256)
    assert tuple(input_data["sparse_indices"].shape) == (2, 4, 1, 16)
    assert tuple(input_data["block_table"].shape) == (2, 2)
    assert input_data["actual_seq_lengths_query"].tolist() == [3, 4]
    assert input_data["actual_seq_lengths_kv"].tolist() == [17, 32]
    assert test_data["cpu_output"] is None


def test_generate_and_save_testdata_tnd_converts_layout_and_prefix_sum():
    np.random.seed(0)
    test_data = golden.generate_and_save_testdata(
        _make_params(
            "snapshot_tnd",
            layout_query="TND",
            layout_kv="TND",
            block_num=None,
            actual_seq_q=[2, 4],
            actual_seq_kv=[5, 7],
        ),
        compute_golden=False,
    )

    input_data = test_data["input"]
    assert tuple(input_data["query"].shape) == (6, 8, 320)
    assert tuple(input_data["key"].shape) == (12, 1, 392)
    assert tuple(input_data["value"].shape) == (12, 1, 256)
    assert tuple(input_data["sparse_indices"].shape) == (6, 1, 16)
    assert input_data["block_table"] is None
    assert input_data["actual_seq_lengths_query"].tolist() == [2, 6]
    assert input_data["actual_seq_lengths_kv"].tolist() == [5, 12]


def test_generate_and_save_testdata_is_repeatable_under_fixed_seed():
    params = _make_params("snapshot_seed_repeat")

    np.random.seed(0)
    first = golden.generate_and_save_testdata(params, compute_golden=False)
    np.random.seed(0)
    second = golden.generate_and_save_testdata(params, compute_golden=False)

    for key in (
        "query",
        "key",
        "value",
        "sparse_indices",
        "key_dequant_scale",
        "value_dequant_scale",
        "block_table",
        "actual_seq_lengths_query",
        "actual_seq_lengths_kv",
    ):
        assert _tensor_signature(first["input"][key]) == _tensor_signature(second["input"][key])


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "matrix_bsnd_bsnd_s2_3",
            "layout_query": "BSND",
            "layout_kv": "BSND",
            "q_type": torch.bfloat16,
            "kv_dtype": None,
            "B": 1,
            "S1": 4,
            "S2": 3,
            "N1": 2,
            "K": 512,
            "block_num": None,
            "actual_seq_q": [4],
            "actual_seq_kv": [3],
        },
        {
            "name": "matrix_tnd_tnd_s2_3",
            "layout_query": "TND",
            "layout_kv": "TND",
            "q_type": torch.bfloat16,
            "kv_dtype": "hifloat8",
            "B": 1,
            "S1": 5,
            "S2": 3,
            "N1": 4,
            "K": 512,
            "block_num": None,
            "actual_seq_q": [5],
            "actual_seq_kv": [3],
        },
        {
            "name": "matrix_bsnd_pa_s2_4096",
            "layout_query": "BSND",
            "layout_kv": "PA_BSND",
            "q_type": torch.float16,
            "kv_dtype": None,
            "B": 1,
            "S1": 128,
            "S2": 4096,
            "N1": 8,
            "K": 2048,
            "block_size": 128,
            "block_num": 32,
            "actual_seq_q": [128],
            "actual_seq_kv": [4096],
        },
        {
            "name": "matrix_bsnd_bsnd_s2_4096",
            "layout_query": "BSND",
            "layout_kv": "BSND",
            "q_type": torch.float16,
            "kv_dtype": None,
            "B": 1,
            "S1": 1,
            "S2": 4096,
            "N1": 16,
            "K": 2048,
            "block_num": None,
            "actual_seq_q": [1],
            "actual_seq_kv": [4096],
        },
        {
            "name": "matrix_tnd_tnd_s2_4096",
            "layout_query": "TND",
            "layout_kv": "TND",
            "q_type": torch.bfloat16,
            "kv_dtype": None,
            "B": 1,
            "S1": 4,
            "S2": 4096,
            "N1": 32,
            "K": 2048,
            "block_num": None,
            "actual_seq_q": [4],
            "actual_seq_kv": [4096],
        },
        {
            "name": "matrix_bsnd_pa_s2_3_hif8",
            "layout_query": "BSND",
            "layout_kv": "PA_BSND",
            "q_type": torch.bfloat16,
            "kv_dtype": "hifloat8",
            "B": 1,
            "S1": 5,
            "S2": 3,
            "N1": 64,
            "K": 512,
            "block_size": 1,
            "block_num": 3,
            "actual_seq_q": [5],
            "actual_seq_kv": [3],
        },
    ],
)
def test_generate_and_save_testdata_case_matrix_contract(case):
    params = _make_params(
        case["name"],
        layout_query=case["layout_query"],
        layout_kv=case["layout_kv"],
        q_type=case["q_type"],
        kv_dtype=case["kv_dtype"],
        B=case["B"],
        S1=case["S1"],
        S2=case["S2"],
        N1=case["N1"],
        D=512,
        K=case["K"],
        block_size=case.get("block_size", 16),
        block_num=case["block_num"],
        actual_seq_q=case["actual_seq_q"],
        actual_seq_kv=case["actual_seq_kv"],
    )

    np.random.seed(0)
    test_data = golden.generate_and_save_testdata(params, compute_golden=False)
    input_data = test_data["input"]

    query_width = 512 + 64
    key_width = 512 + 2 * 64 + 4 * (512 // 128)

    if case["layout_query"] == "TND":
        assert tuple(input_data["query"].shape) == (sum(case["actual_seq_q"]), case["N1"], query_width)
        assert input_data["actual_seq_lengths_query"].tolist() == np.cumsum(case["actual_seq_q"]).tolist()
    else:
        assert tuple(input_data["query"].shape) == (case["B"], case["S1"], case["N1"], query_width)
        assert input_data["actual_seq_lengths_query"].tolist() == case["actual_seq_q"]

    if case["layout_kv"] == "PA_BSND":
        assert tuple(input_data["key"].shape) == (case["block_num"], case["block_size"], 1, key_width)
        assert tuple(input_data["value"].shape) == (case["block_num"], case["block_size"], 1, 512)
        assert tuple(input_data["block_table"].shape) == (case["B"], max(1, case["S2"] // case["block_size"]))
        assert input_data["actual_seq_lengths_kv"].tolist() == case["actual_seq_kv"]
    elif case["layout_kv"] == "TND":
        assert tuple(input_data["key"].shape) == (sum(case["actual_seq_kv"]), 1, key_width)
        assert tuple(input_data["value"].shape) == (sum(case["actual_seq_kv"]), 1, 512)
        assert input_data["block_table"] is None
        assert input_data["actual_seq_lengths_kv"].tolist() == np.cumsum(case["actual_seq_kv"]).tolist()
    else:
        assert tuple(input_data["key"].shape) == (case["B"], case["S2"], 1, key_width)
        assert tuple(input_data["value"].shape) == (case["B"], case["S2"], 1, 512)
        assert input_data["block_table"] is None
        assert input_data["actual_seq_lengths_kv"].tolist() == case["actual_seq_kv"]

    assert input_data["kv_dtype"] == case["kv_dtype"]
    assert test_data["cpu_output"] is None
