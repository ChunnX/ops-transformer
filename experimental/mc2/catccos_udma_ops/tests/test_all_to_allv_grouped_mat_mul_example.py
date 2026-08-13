# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLE = PROJECT_ROOT / "examples" / "run_all_to_allv_grouped_mat_mul.py"


def test_example_uses_public_facade_and_regular_reusable_output():
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "udma.init_udma_group(" in source
    assert "udma.build_rank_route_tensors(" in source
    assert "out = torch.zeros(" in source
    assert "udma.all_to_allv_grouped_mat_mul(" in source
    assert "out=out" in source
    assert "torch_npu.npu.synchronize()" in source
    assert "udma.destroy_udma_group()" in source


def test_example_exposes_business_shape_arguments():
    source = EXAMPLE.read_text(encoding="utf-8")
    for option in ("--m", "--k", "--n", "--expert-num", "--iterations"):
        assert option in source
    assert "args.expert_num % world_size" in source
    assert "world_size * args.m" in source
