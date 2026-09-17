# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository.
import pytest

from device_support import invoke, make_inputs

pytestmark = [pytest.mark.npu, pytest.mark.contract]
INVALID_CASES = (
    "query_rank3", "query_rank5", "empty_batch", "empty_sequence", "empty_head", "empty_dim",
    "scale_rank4", "scale_axes", "scale_groups", "query_noncontiguous", "scale_noncontiguous",
    "out_tnd", "out_other", "quant_mode1", "cu_empty", "cu_values", "used_empty", "used_values",
    "insufficient_qb", "sparse_batch", "sparse_head", "counts_batch", "table_batch", "kv_length_batch",
    "kv_layout", "d64",
)


def mutate(torch, data, fault):
    q, scale = data["query"], data["q_descale"]
    layout = data["layout_q"]
    if fault in ("query_rank3", "query_rank5"):
        data["query"] = q.reshape(-1, *q.shape[2:]) if fault == "query_rank3" else q.unsqueeze(0)
    elif fault.startswith("empty_"):
        axis = {"empty_batch": 0, "empty_sequence": 1 if layout == "BSND" else 2,
                "empty_head": 2 if layout == "BSND" else 1, "empty_dim": 3}[fault]
        shape = list(q.shape)
        shape[axis] = 0
        data["query"] = torch.empty(shape, dtype=q.dtype, device=q.device)
    elif fault == "scale_rank4":
        data["q_descale"] = scale.reshape(-1, *scale.shape[2:])
    elif fault == "scale_axes":
        data["q_descale"] = scale.view(torch.uint8).transpose(1, 2).contiguous().view(scale.dtype)
    elif fault == "scale_groups":
        data["q_descale"] = torch.empty((*scale.shape[:-2], 3, 2), dtype=scale.dtype, device=scale.device)
    elif fault == "query_noncontiguous":
        data["query"] = q.transpose(1, 2)
        data["q_descale"] = scale.transpose(1, 2)
    elif fault == "scale_noncontiguous":
        data["q_descale"] = scale.transpose(-1, -2)
    elif fault.startswith("out_"):
        data["layout_out"] = "TND" if fault == "out_tnd" else ("BNSD" if layout == "BSND" else "BSND")
    elif fault == "quant_mode1":
        data["quant_mode"] = 1
    elif fault in ("cu_empty", "cu_values", "used_empty", "used_values"):
        key = "cu_seqlens_q" if fault.startswith("cu_") else "seqused_q"
        values = [] if fault.endswith("empty") else ([0, 257, 514] if key == "cu_seqlens_q" else [257, 257])
        data[key] = torch.tensor(values, dtype=torch.int32, device=q.device)
    elif fault == "insufficient_qb":
        data["sparse_indices"] = data["sparse_indices"][:, :, :1].contiguous()
        data["sparse_seq_len"] = data["sparse_seq_len"][:, :, :1].contiguous()
    elif fault in ("sparse_batch", "sparse_head", "counts_batch", "table_batch", "kv_length_batch"):
        key = {"sparse_batch": "sparse_indices", "sparse_head": "sparse_indices",
               "counts_batch": "sparse_seq_len", "table_batch": "block_table",
               "kv_length_batch": "seqused_kv"}[fault]
        value = data[key]
        data[key] = (value[:, :3] if fault == "sparse_head" else value[:1]).contiguous()
    elif fault == "kv_layout":
        data["layout_kv"] = "BSND"
    elif fault == "d64":
        data["query"] = q.view(torch.uint8)[..., :64].contiguous().view(q.dtype)
        data["q_descale"] = scale.view(torch.uint8)[..., :1, :].contiguous().view(scale.dtype)
    else:
        raise ValueError(fault)


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("fault", INVALID_CASES)
def test_public_npu_rejects_invalid_input(runtime, layout, fault):
    torch = runtime[0]
    inputs, _ = make_inputs(torch, layout)
    # Prove fixture and installed operator work before expecting a rejection.
    invoke(torch, inputs)
    mutate(torch, inputs, fault)
    with pytest.raises(RuntimeError) as error:
        invoke(torch, inputs)
    assert str(error.value).strip()
