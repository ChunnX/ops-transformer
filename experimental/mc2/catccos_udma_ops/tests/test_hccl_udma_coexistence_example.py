# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLE = (
    PROJECT_ROOT
    / "examples"
    / "run_hccl_then_all_to_allv_grouped_mat_mul.py"
)


def test_example_runs_hccl_before_udma_and_keeps_group_alive():
    source = EXAMPLE.read_text(encoding="utf-8")
    hccl_init = source.index("dist.init_process_group(")
    all_gather = source.index("run_hccl_all_gather(", hccl_init)
    udma_init = source.index("udma.init_udma_group(", all_gather)
    udma_op = source.index("udma.all_to_allv_grouped_mat_mul(", udma_init)
    udma_destroy = source.index("udma.destroy_udma_group()", udma_op)
    hccl_destroy = source.index("dist.destroy_process_group()", udma_destroy)
    assert hccl_init < all_gather < udma_init < udma_op
    assert udma_destroy < hccl_destroy


def test_example_validates_hccl_and_reuses_regular_udma_output():
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "dist.all_gather(gathered, value)" in source
    assert "torch.equal(peer_value, expected)" in source
    assert "hccl_all_gather_passed=True" in source
    assert "out = torch.empty(" in source
    assert "(received_rows, args.n)" in source
    assert "received_rows=received_rows" in source
    assert "out=out" in source
    assert "udma_operator_passed=True" in source


def test_example_accepts_shape_from_environment_without_launcher_arguments():
    source = EXAMPLE.read_text(encoding="utf-8")
    for name in (
        "A2AV_GMM_M",
        "A2AV_GMM_K",
        "A2AV_GMM_N",
        "A2AV_GMM_EXPERT_NUM",
        "HCCL_ALL_GATHER_ELEMENTS",
    ):
        assert 'os.environ.get("{}"'.format(name) in source
