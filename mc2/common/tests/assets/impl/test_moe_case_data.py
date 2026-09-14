import importlib.util
import sys
from pathlib import Path

import torch


def _load_module():
    path = Path(__file__).with_name("moe_case_data.py")
    spec = importlib.util.spec_from_file_location("moe_case_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_case():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=8,
        hidden_size=16,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=16,
        shared_expert_num=2,
        shared_expert_rank_num=2,
        rank_num_per_shared_expert=1,
    )
    case = module.generate_case(config)
    rank = module.rank_case(case, config, 1)
    assert rank["x"].shape == (2, 16)
    assert rank["ori_x"].shape == (2, 16)
    assert rank["x_active_mask"].shape == (2,)
    assert module.torch.equal(rank["ori_x"], rank["x"])
    assert rank["expert_ids"].shape == (2, 2)
    assert rank["golden_x"].shape == (2, 16)
    assert len(case["row_mask_list"]) == config.ep_world_size
    assert all(len(items) == config.ep_world_size for items in case["expand_x_list"])
    assert all(len(items) == config.ep_world_size for items in case["ep_recv_count_list"])


def test_generate_case_is_expand():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=8,
        hidden_size=16,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=16,
        shared_expert_num=2,
        shared_expert_rank_num=2,
        rank_num_per_shared_expert=1,
        is_expand=True,
    )
    case = module.generate_case(config)
    rank = module.rank_case(case, config, 1)
    assert rank["x"].shape == (2, 16)
    assert rank["golden_x"].shape == (2, 16)
    assert len(case["row_mask_list"]) == config.ep_world_size


def test_generate_case_keeps_input_mask_for_variable_bs():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=112,
        hidden_size=32,
        top_k=10,
        seed=458,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=480,
        local_moe_expert_num_override=60,
        shared_expert_num=2,
        shared_expert_rank_num=8,
        rank_num_per_shared_expert=4,
        zero_expert_num=23240,
        copy_expert_num=957,
        const_expert_num=14203,
        is_variable_bs=True,
        is_elastic=True,
        elastic_shared_rank_num=2,
        elastic_moe_rank_num=2,
        comm_quant_mode=2,
        is_expand=True,
    )
    case = module.generate_case(config)
    mask = case["x_active_mask_input_list"][0]
    assert mask.shape == (config.global_bs,)
    assert mask[: config.global_bs // config.ep_world_size].tolist() == [
        True
    ] * (config.global_bs // config.ep_world_size)


def test_quantize_dispatch_values_matches_kernel_conversion_order():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=8,
        hidden_size=16,
        top_k=1,
        seed=42,
        ep_world_size=1,
        tp_world_size=1,
        moe_expert_num=1,
        is_quant=True,
    )
    values = torch.tensor(
        [
            [-1.25, -0.5, 0.0, 0.5, 1.25, 2.5, -3.5, 4.5],
            [-7.5, 8.5, -9.5, 10.5, -11.5, 12.5, -13.5, 14.5],
        ],
        dtype=torch.float32,
    )
    quantized, scales = module._quantize_dispatch_values(values, config)
    expected_scale = values.abs().max(dim=-1, keepdim=True).values / 127.0
    expected_codes = (values / expected_scale).round().to(torch.int32)
    assert torch.equal(quantized, expected_codes.to(torch.int8))
    assert torch.allclose(scales, expected_scale)
    assert torch.equal(
        quantized[1],
        torch.tensor([-66, 74, -83, 92, -101, 109, -118, 127], dtype=torch.int8),
    )


def test_generate_case_with_special_experts_keeps_full_output_and_row_masks():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=8,
        hidden_size=16,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=16,
        zero_expert_num=1,
        copy_expert_num=1,
        const_expert_num=1,
    )
    case = module.generate_case(config)
    assert case["x_active_mask_list"][0].shape == (8,)
    assert case["x_active_compute_mask_list"][0].shape == (8, 2)
    assert all(mask.shape == (2,) for mask in case["row_mask_list"])
    assert all(mask.shape == (2, 16) for mask in case["golden_x"])


def test_expand_dispatch_golden_uses_compute_mask_when_mask_is_one_dimensional():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=112,
        hidden_size=32,
        top_k=10,
        seed=458,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=480,
        local_moe_expert_num_override=30,
        shared_expert_num=0,
        shared_expert_rank_num=0,
        rank_num_per_shared_expert=1,
        zero_expert_num=1,
        is_variable_bs=True,
        is_expand=True,
    )
    case = module.generate_case(config)
    assert case["x_active_mask_list"][0].dim() == 1
    assert case["x_active_compute_mask_list"][0].dim() == 2
    expert_ids = torch.cat(case["expert_ids_expand_list"][0]).reshape(-1)
    compute_mask = case["x_active_compute_mask_list"][0].reshape(-1)
    expected_rows = [
        int(
            ((expert_ids >= rank_id * 30)
             & (expert_ids < (rank_id + 1) * 30)
             & compute_mask).sum()
        )
        for rank_id in range(config.ep_world_size)
    ]
    assert [tensor.shape[0] for tensor in case["expand_x_list"][0]] == expected_rows


def test_shared_expert_source_cards_use_rank_num_per_shared_expert():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=16,
        hidden_size=8,
        top_k=2,
        seed=42,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=8,
        shared_expert_num=2,
        shared_expert_rank_num=8,
        rank_num_per_shared_expert=4,
    )
    expected = [
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ] * 2
    actual = [
        list(module._shared_expert_send_card_ids(config, shared_rank_id))
        for shared_rank_id in range(config.shared_expert_rank_num)
    ]
    assert actual == expected


def test_expand_dispatch_golden_uses_original_masks_and_triplet_indices():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=112,
        hidden_size=32,
        top_k=10,
        seed=458,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=480,
        local_moe_expert_num_override=60,
        shared_expert_num=2,
        shared_expert_rank_num=8,
        rank_num_per_shared_expert=4,
        zero_expert_num=23240,
        copy_expert_num=957,
        const_expert_num=14203,
        is_variable_bs=True,
        is_mask=True,
        is_elastic=True,
        elastic_shared_rank_num=2,
        elastic_moe_rank_num=2,
        comm_quant_mode=2,
        is_expand=True,
        is_deepep=True,
        active_mask_dim=2,
    )
    case = module.generate_case(config)
    assert config.available_ranks == [0, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15]
    assert case["x_active_mask_list"][0].shape == (
        config.global_bs,
        config.top_k,
    )
    assert module.torch.equal(
        case["x_active_mask_list"][0],
        case["x_active_mask_input_list"][0],
    )
    assert len(case["initial_expand_x_list"][0]) == config.ep_world_size
    assert all(
        tensor.numel() % 3 == 0
        for tensor in case["initial_expand_idx_list"][0]
        if tensor is not None
    )


def test_expand_elastic_variable_bs_keeps_original_rank_layout():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=112,
        hidden_size=32,
        top_k=7,
        seed=458,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=480,
        local_moe_expert_num_override=60,
        shared_expert_num=2,
        shared_expert_rank_num=8,
        rank_num_per_shared_expert=4,
        zero_expert_num=23240,
        copy_expert_num=957,
        const_expert_num=14203,
        is_variable_bs=True,
        is_elastic=True,
        elastic_shared_rank_num=2,
        elastic_moe_rank_num=2,
        comm_quant_mode=2,
        is_expand=True,
    )
    case = module.generate_case(config)
    mask = case["x_active_mask_elastic_list"][0]
    expected = [7, 0, 1, 7, 0, 2, 1, 4, 6, 0, 3, 5, 0, 4, 5, 1]
    assert mask.view(config.ep_world_size, -1).sum(dim=1).tolist() == expected
    assert all(tensor.numel() == 0 for tensor in case["shared_expert_x_list"][0])


def test_expand_elastic_rank_case_keeps_full_rank_layout():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=112,
        hidden_size=32,
        top_k=10,
        seed=458,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=480,
        local_moe_expert_num_override=60,
        shared_expert_num=2,
        shared_expert_rank_num=8,
        rank_num_per_shared_expert=4,
        zero_expert_num=23240,
        copy_expert_num=957,
        const_expert_num=14203,
        is_variable_bs=True,
        is_elastic=True,
        elastic_shared_rank_num=2,
        elastic_moe_rank_num=2,
        comm_quant_mode=2,
        is_expand=True,
    )
    case = module.generate_case(config)
    assert len(case["initial_expand_x_list"][0]) == config.ep_world_size
    assert len(case["initial_ep_recv_count_list"][0]) == config.ep_world_size
    rank_case = module.rank_case(case, config, 0)
    assert "initial_expand_x" in rank_case
    for local_rank_id, rank_id in enumerate(config.available_ranks):
        assert int(rank_case["elastic_info"][4 + rank_id]) == local_rank_id
    assert all(
        module.rank_case(case, config, rank_id)["initial_expand_x"]
        is case["initial_expand_x_list"][0][rank_id]
        for rank_id in config.available_ranks
    )


def test_expand_elastic_rank_case_maps_initial_golden_by_available_rank():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=112,
        hidden_size=32,
        top_k=10,
        seed=458,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=480,
        local_moe_expert_num_override=60,
        shared_expert_num=2,
        shared_expert_rank_num=8,
        rank_num_per_shared_expert=4,
        zero_expert_num=23240,
        copy_expert_num=957,
        const_expert_num=14203,
        is_variable_bs=True,
        is_elastic=True,
        elastic_shared_rank_num=2,
        elastic_moe_rank_num=2,
        comm_quant_mode=2,
        is_expand=True,
    )
    case = module.generate_case(config)
    initial_tensors = case["initial_expand_x_list"][0]
    final_tensors = case["expand_x_list"][0]
    for rank_id in config.available_ranks:
        rank_case = module.rank_case(case, config, rank_id)
        assert rank_case["initial_expand_x"] is initial_tensors[rank_id]
        assert rank_case["expand_x"] is final_tensors[rank_id]
    for rank_id in range(config.ep_world_size):
        if rank_id not in config.available_ranks:
            rank_case = module.rank_case(case, config, rank_id)
            assert rank_case["initial_expand_x"].numel() == 0


def test_expand_elastic_rank_case_maps_final_expansion_by_rank():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=128,
        hidden_size=32,
        top_k=16,
        seed=117,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=780,
        local_moe_expert_num_override=60,
        shared_expert_num=3,
        shared_expert_rank_num=3,
        rank_num_per_shared_expert=1,
        is_mask=True,
        active_mask_dim=1,
        is_elastic=True,
        elastic_shared_rank_num=1,
        elastic_moe_rank_num=2,
        is_expand=True,
    )
    case = module.generate_case(config)
    expected_shapes = [tensor.shape for tensor in case["expand_x_list"][0]]
    for rank_id in range(config.ep_world_size):
        actual = module.rank_case(case, config, rank_id)["expand_x"]
        assert actual.shape == expected_shapes[rank_id]


def test_expand_combine_golden_uses_full_shared_layout_and_compute_mask():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=112,
        hidden_size=32,
        top_k=10,
        seed=458,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=480,
        local_moe_expert_num_override=60,
        shared_expert_num=2,
        shared_expert_rank_num=8,
        rank_num_per_shared_expert=4,
        zero_expert_num=23240,
        copy_expert_num=957,
        const_expert_num=14203,
        is_variable_bs=True,
        is_elastic=True,
        elastic_shared_rank_num=2,
        elastic_moe_rank_num=2,
        comm_quant_mode=2,
        is_expand=True,
        is_deepep=True,
    )
    case = module.generate_case(config)
    assert all(
        tensor.shape
        == (config.global_bs // config.ep_world_size, config.hidden_size)
        for tensor in case["golden_x"]
    )


def test_expand_dispatch_shared_cards_keep_physical_rank_layout():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=8192,
        hidden_size=32,
        top_k=16,
        seed=80,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=640,
        local_moe_expert_num_override=64,
        shared_expert_num=3,
        shared_expert_rank_num=6,
        rank_num_per_shared_expert=2,
        const_expert_num=1735,
        input_dtype="float16",
        is_elastic=True,
        elastic_shared_rank_num=3,
        elastic_moe_rank_num=6,
        is_expand=True,
        is_deepep=True,
    )
    case = module.generate_case(config)
    assert config.available_ranks == [0, 1, 5, 6, 7, 9, 11]
    original_x = case["x_list"][0]
    original_mask = case["x_active_mask_list"][0]
    per_rank_bs = config.global_bs // config.ep_world_size
    for shared_rank_id in range(config.shared_expert_rank_num):
        tensor = case["expand_x_list"][0][shared_rank_id]
        source_cards = list(
            module._shared_expert_send_card_ids(config, shared_rank_id)
        )
        expected_tensors = []
        for card_id in source_cards:
            card_mask = original_mask[
                card_id * per_rank_bs : (card_id + 1) * per_rank_bs
            ]
            if card_mask.dim() == 2:
                card_mask = card_mask.any(dim=-1)
            expected_tensors.append(
                original_x[
                    card_id * per_rank_bs : (card_id + 1) * per_rank_bs
                ][card_mask]
            )
        expected = torch.cat(expected_tensors)
        assert torch.equal(tensor[:, 0].to(original_x.dtype), expected[:, 0])


def test_two_dimensional_mask_keeps_full_rows_and_expand_rng_order():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=384,
        hidden_size=32,
        top_k=8,
        seed=1,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=16,
        local_moe_expert_num_override=2,
        shared_expert_num=2,
        shared_expert_rank_num=8,
        rank_num_per_shared_expert=4,
        zero_expert_num=6,
        is_mask=True,
        active_mask_dim=2,
    )
    case = module.generate_case(config)
    assert case["x_active_mask_list"][0].shape == (384, 8)
    assert len(case["expert_ids_expand_list"]) == 1
    assert torch.cat(case["expert_ids_expand_list"][0]).shape == (
        config.global_bs,
        config.top_k,
    )


def test_add_rms_norm_combine_golden_uses_original_mask_and_float32():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=16,
        hidden_size=8,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=8,
        zero_expert_num=2,
        copy_expert_num=2,
        const_expert_num=2,
        is_x_random=True,
        is_add_rms_norm=True,
    )
    case = module.generate_case(config)
    assert all(tensor.dtype == module.torch.float32 for tensor in case["golden_x"])
    assert all(tensor.dtype == module.torch.float32 for tensor in case["add_rms_norm_x"])
    assert all(
        module.torch.equal(
            row_mask,
            case["x_active_compute_mask_list"][0]
            .view(config.ep_world_size, config.global_bs // config.ep_world_size, -1)
            .any(dim=-1)[rank],
        )
        for rank, row_mask in enumerate(case["row_mask_list"])
    )
    assert case["residual_x"]
    assert all(tensor.numel() for tensor in case["residual_x"])
    assert all(not tensor.numel() for tensor in case["shared_expert_x_list"][0])

    compute_mask = case["x_active_compute_mask_list"][0]
    original_mask = case["x_active_mask_elastic_list"][0].reshape(-1, 1).repeat(
        1, config.top_k
    )
    assert not module.torch.equal(compute_mask, original_mask)


def test_masked_add_rms_norm_golden_keeps_full_rank_layout():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=16,
        hidden_size=8,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=8,
        copy_expert_num=2,
        is_mask=True,
        active_mask_dim=2,
        is_add_rms_norm=True,
    )
    case = module.generate_case(config)
    for rank, expected_rstd in enumerate(case["golden_rstd_out"]):
        rank_case = module.rank_case(case, config, rank)
        residual = rank_case["residual_x"].reshape(-1, config.hidden_size).float()
        normalized = rank_case["golden_x"].float() + residual
        calculated = normalized.pow(2).mean(dim=-1).add(1e-6).rsqrt().reshape(-1)
        assert module.torch.allclose(calculated, expected_rstd)
        assert calculated.shape == rank_case["golden_x"].shape[:1]


def test_non_add_rms_norm_combine_golden_keeps_input_dtype():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=16,
        hidden_size=8,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=8,
        is_x_random=True,
    )
    case = module.generate_case(config)
    assert all(tensor.dtype == module.torch.bfloat16 for tensor in case["golden_x"])


def test_shared_expert_x_follows_mask_and_uses_zero_to_one_range():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=16,
        hidden_size=8,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=8,
        is_shared_expert_x=True,
        is_mask=True,
        active_mask_dim=1,
    )
    case = module.generate_case(config)
    shared_tensors = case["shared_expert_x_list"][0]
    masks = case["x_active_mask_list"][0].view(config.ep_world_size, -1)
    assert [
        tensor.shape[0] for tensor in shared_tensors
    ] == [int(mask.sum()) for mask in masks]
    assert all(
        float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
        for tensor in shared_tensors
        if tensor.numel()
    )


def test_shared_expert_x_follows_variable_bs_full_mask():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=16,
        hidden_size=8,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=8,
        is_shared_expert_x=True,
        is_mask=True,
        is_variable_bs=True,
        active_mask_dim=1,
    )
    case = module.generate_case(config)
    shared_tensors = case["shared_expert_x_list"][0]
    masks = case["x_active_mask_list"][0].view(config.ep_world_size, -1)
    assert any(bool(mask.all()) for mask in masks)
    assert [
        tensor.shape[0] for tensor in shared_tensors
    ] == [int(mask.sum()) for mask in masks]


def test_shared_expert_golden_adds_only_masked_rows():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=16,
        hidden_size=8,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=8,
        is_shared_expert_x=True,
        is_mask=True,
        active_mask_dim=1,
    )
    baseline_data = module._generate_case(config)
    dispatch_golden = module._generate_dispatch_golden(baseline_data, config)
    baseline_data.update(
        {
            "expert_scales_list": [
                torch.empty(config.global_bs, config.top_k).uniform_(-1, 1)
                for _ in range(config.tp_world_size)
            ],
        }
    )
    masks = baseline_data["x_active_mask_list"][0].view(
        config.ep_world_size, -1
    )

    def with_shared_values(values):
        data = dict(baseline_data)
        data["shared_expert_x_list"] = [
            [
                torch.full(
                    (int(mask.sum()), config.hidden_size),
                    values,
                    dtype=module.torch.bfloat16,
                )
                for mask in masks
            ]
        ]
        return module._generate_combine_golden(data, dispatch_golden, config)

    zero_shared, _ = with_shared_values(0)
    one_shared, _ = with_shared_values(1)
    for rank_id, mask in enumerate(masks):
            expected = module.torch.zeros(
                config.global_bs // config.ep_world_size,
                config.hidden_size,
                dtype=module.torch.float32,
            )
            expected[: int(mask.sum())] = 1
            assert module.torch.allclose(
                (one_shared[rank_id] - zero_shared[rank_id]).float(),
                expected,
                atol=0.02,
            )


def test_add_rms_norm_shared_expert_golden_keeps_full_rank_layout():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=16,
        hidden_size=8,
        top_k=2,
        seed=42,
        ep_world_size=4,
        tp_world_size=1,
        moe_expert_num=8,
        is_shared_expert_x=True,
        is_mask=True,
        active_mask_dim=1,
        is_add_rms_norm=True,
    )
    case = module.generate_case(config)
    assert all(
        tensor.shape == (config.global_bs // config.ep_world_size, config.hidden_size)
        for tensor in case["golden_x"]
    )

def test_moe_expert_num_matches_reference_for_local_expert_cases():
    module = _load_module()
    config = module.MoeCaseConfig(
        global_bs=128,
        hidden_size=8,
        top_k=16,
        seed=117,
        ep_world_size=16,
        tp_world_size=1,
        moe_expert_num=780,
        local_moe_expert_num_override=60,
        shared_expert_num=3,
        shared_expert_rank_num=3,
        rank_num_per_shared_expert=1,
        is_expand=True,
        is_elastic=True,
        elastic_shared_rank_num=1,
        elastic_moe_rank_num=2,
        is_deepep=True,
    )
    assert config.local_moe_expert_num == 60
    case = module.generate_case(config)
    assert case["expert_ids_list"][0].max().item() < 600
    assert torch.cat(case["expert_ids_expand_list"][0]).max().item() < 780
