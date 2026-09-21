"""ASC-style paramset-driven single-case entry for QSLI."""

import itertools
import os
from pathlib import Path

import pytest
import torch

from qsli_test_utils import compare_qsli_case, generate_qsli_test_data, run_qsli_case
from test_quant_sparse_lightning_indexer_paramset import ENABLED_PARAMSETS

PARAM_NAMES = (
    "batch_size", "query_length", "sequence_length", "cmp_ratio",
    "storage_mode", "mask_mode", "topk", "candidate_block_length",
    "output_idx_offset", "seed",
)
REQUESTED = {x.strip() for x in os.environ.get("QSLI_CASE_NAMES", "").split(",") if x.strip()}
SAVE_PT_DIR = os.environ.get("QSLI_SINGLE_SAVE_PT_DIR", "").strip()

PARAMS = []
matched = set()
for paramset_name, values in ENABLED_PARAMSETS:
    names = PARAM_NAMES + tuple(name for name in values if name not in PARAM_NAMES)
    combinations = itertools.product(*(values[name] for name in names))
    for index, combination in enumerate(combinations, 1):
        case_name = paramset_name if index == 1 else f"{paramset_name}_{index:03d}"
        if REQUESTED and paramset_name not in REQUESTED and case_name not in REQUESTED:
            continue
        matched.update((paramset_name, case_name))
        PARAMS.append(pytest.param(dict(zip(names, combination)), case_name, id=case_name))
if REQUESTED - matched:
    raise ValueError(f"unknown case(s): {sorted(REQUESTED - matched)}")


@pytest.mark.ci
@pytest.mark.npu
@pytest.mark.parametrize("params,case_name", PARAMS)
def test_qsli_single(params, case_name):
    case_data = generate_qsli_test_data(params)
    if SAVE_PT_DIR:
        target = Path(SAVE_PT_DIR) / f"{case_name}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(case_data, target)
        print(f"saved: {target}")
    actual = run_qsli_case(case_data, int(os.environ.get("QSLI_DEVICE_ID", "0")))
    result, fulfill = compare_qsli_case(case_data, actual)
    print(f"result={result}, fulfill_percent={fulfill}")


@pytest.mark.parametrize('layout', ['PA_BBND', 'TND'])
@pytest.mark.parametrize('return_value', [False, True])
def test_qsli_torch_extension_metadata(layout, return_value):
    import torch_npu
    import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl
    from qsli_test_utils import _to_npu_tensors
    from ops.quant_sparse_lightning_indexer_dsl import quant_sparse_lightning_indexer as direct
    from ops.quant_sparse_lightning_indexer_metadata_dsl import quant_sparse_lightning_indexer_metadata as direct_metadata

    torch.npu.set_device(int(os.environ.get('QSLI_DEVICE_ID', '0')))
    data = generate_qsli_test_data(dict(batch_size=2, query_length=2, sequence_length=512,
        layout_kv=layout, return_value=return_value, cmp_ratio=2, cmp_residual_k=[0, 1],
        candidate_block_length=64, output_idx_offset=True, seed=4202))
    tensors = _to_npu_tensors(data['tensors'])
    params = data['params']
    namespace = torch.ops.cann_ops_transformer.ds41
    meta_args = (tensors['candidate_length'], tensors['cu'], tensors['cu_k'],
                 tensors['used_q'], tensors['used_k'], tensors['residual'])
    meta_options = dict(num_heads_q=32, num_heads_k=1, head_dim=128, topk=512,
                        quant_mode=1, candidate_block_size=8, max_seqlen_q=2,
                        mask_mode=params['mask_mode'], cmp_ratio=2)
    metadata = namespace.quant_sparse_lightning_indexer_metadata(*meta_args, layout_kv=layout, **meta_options)
    expected_metadata = direct_metadata(*meta_args, layout_k=layout, **meta_options)
    assert torch.equal(metadata.cpu(), expected_metadata.cpu())
    positional = (tensors['q'], tensors['k'], tensors['weights'], tensors['descale_q'],
                  tensors['descale_k'], tensors['candidate'], tensors['candidate_length'])
    options = dict(topk=512, quant_mode=1, candidate_block_size=8, max_seqlen_q=2,
                   mask_mode=params['mask_mode'], cmp_ratio=2, cu_seqlens_q=tensors['cu'],
                   cu_seqlens_k=tensors['cu_k'], seqused_q=tensors['used_q'],
                   seqused_k=tensors['used_k'], cmp_residual_k=tensors['residual'],
                   block_table=tensors['block_table'], output_idx_offset=tensors['offset'],
                   return_value=return_value)
    explicit = namespace.quant_sparse_lightning_indexer(*positional, metadata=metadata, layout_kv=layout, **options)
    automatic = namespace.quant_sparse_lightning_indexer(*positional, layout_kv=layout, **options)
    expected = direct(*positional[:4], *positional[5:], descale_k=tensors['descale_k'] if layout == 'TND' else None,
                      metadata=metadata, layout_k=layout, **options)
    for actual, generated, reference in zip(explicit, automatic, expected):
        assert torch.equal(actual.cpu(), reference.cpu())
        assert torch.equal(generated.cpu(), reference.cpu())
    result = dict(zip(('sparse_indices', 'sparse_values'), (value.cpu() for value in explicit)))
    compare_qsli_case(data, result)
