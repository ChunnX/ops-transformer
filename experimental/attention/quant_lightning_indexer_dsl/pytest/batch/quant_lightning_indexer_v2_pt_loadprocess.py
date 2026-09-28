"""Execute one serialized QLI case using the ASC batch-call convention."""

import torch

import quant_lightning_indexer_v2_golden


def test_qliv2_process(filepath, device_id=0):
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    cpu_result, npu_result = quant_lightning_indexer_v2_golden.run_qliv2_case_data(
        data, device_id=device_id
    )
    return (
        cpu_result,
        npu_result,
        cpu_result.get("_reference_scores"),
        cpu_result.get("sparse_values"),
        npu_result.get("sparse_values"),
        None,
        data.params,
    )


def test_qliv2_process_graph(filepath, device_id=0):
    raise NotImplementedError("DSL QLI transfer suite does not expose an ACL graph entry")

