#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TestSpec adapter for LightningIndexer V2 TTK assets."""

import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"li_v2_assets_impl_{stem}_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


class LightningIndexerV2MetadataInputSpec:
    golden = golden_module.cpu_lightning_indexer_v2
    customize_inputs = inputs_module.generate_li_v2_inputs
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, **kwargs):
        return compare_module.compare(
            *outputs,
            scores=golden_module.get_topk_scores(),
            index_offsets=golden_module.get_index_offsets(),
        )


__spec__ = {
    "li_v2_metadata_input_ttk_ops.lightning_indexer_v2": "LightningIndexerV2MetadataInputSpec",
}
