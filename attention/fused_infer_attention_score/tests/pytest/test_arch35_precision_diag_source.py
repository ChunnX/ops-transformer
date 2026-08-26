from pathlib import Path


KERNEL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "op_kernel"
    / "arch35"
    / "flash_attention_noquant_block_vec_base_fused_infer.h"
)


def test_arch35_precision_diag_covers_reported_fp16_cases_and_stages():
    source = KERNEL_SOURCE.read_text(encoding="utf-8")

    for case_signature in (
        "constInfo.bSize == 8 && constInfo.n2Size == 5 && constInfo.gSize == 3",
        "constInfo.bSize == 2 && constInfo.n2Size == 1 && constInfo.gSize == 8",
    ):
        assert case_signature in source

    for target in (
        "runInfo.boIdx == 4 && runInfo.n2oIdx == 3",
        "runInfo.boIdx == 5 && runInfo.n2oIdx == 4",
        "runInfo.boIdx == 6 && runInfo.n2oIdx == 0",
        "runInfo.boIdx == 1 && runInfo.n2oIdx == 0",
    ):
        assert target in source

    for stage in (
        "[FIA_DIAG][KERNEL][BMM1_SOFTMAX]",
        "[FIA_DIAG][KERNEL][SOFTMAX_STATE]",
        "[FIA_DIAG][KERNEL][BMM2_ACCUM]",
        "[FIA_DIAG][KERNEL][POSTQUANT]",
    ):
        assert stage in source

