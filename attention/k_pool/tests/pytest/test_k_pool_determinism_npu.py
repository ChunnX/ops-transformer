from __future__ import annotations

import json

import torch
import torch_npu

from k_pool_public_loader import load_k_pool_public_api
from test_k_pool_layout_equivalence_npu import _run_npu
from test_k_pool_public_npu import CASES, _make_case_inputs


def _run_repeated(case_name: str, norm_on: bool, repeats: int = 10):
    case = CASES[case_name]
    (
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        seq_lens,
    ) = _make_case_inputs(case)
    norm_weight = norm_bias = None
    if norm_on:
        norm_weight = torch.randn((wk.size(0),), dtype=torch.float32)
        norm_bias = torch.randn((wk.size(0),), dtype=torch.float32)

    outputs = []
    caches = []
    for _ in range(repeats):
        output, cache = _run_npu(
            hidden_states,
            wk,
            gate_weight,
            ape,
            state_cache.clone(),
            cache_block_table,
            start_pos,
            seq_lens,
            case.cmp_ratio,
            norm_weight,
            norm_bias,
        )
        outputs.append(output)
        caches.append(cache)

    output_equal = all(torch.equal(outputs[0], item) for item in outputs[1:])
    cache_equal = all(torch.equal(caches[0], item) for item in caches[1:])
    return {
        "case": case_name,
        "norm_on": norm_on,
        "repeats": repeats,
        "output_bitwise": output_equal,
        "cache_bitwise": cache_equal,
        "pass": output_equal and cache_equal,
    }


def main():
    load_k_pool_public_api()
    torch_npu.npu.set_device(0)
    results = [
        _run_repeated(case_name, norm_on)
        for case_name in ("bsh_history_r4", "th_full_history_r4")
        for norm_on in (False, True)
    ]
    print(json.dumps(results, indent=2))
    if not all(item["pass"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
