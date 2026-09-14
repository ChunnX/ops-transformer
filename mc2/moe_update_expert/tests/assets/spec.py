#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TestSpec adapter for MoeUpdateExpert TTK assets."""

import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_impl_module(stem):
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"moe_update_expert_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class MoeUpdateExpertTestSpec:
    @staticmethod
    def golden(*args, **kwargs):
        return _load_impl_module("golden").moe_update_expert_golden(*args, **kwargs)

    @staticmethod
    def compare(*outputs, **kwargs):
        return _load_impl_module("compare").compare(*outputs, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _load_impl_module("inputs").customize_inputs(*args, **kwargs)


__spec__ = {
    "aclnnMoeUpdateExpert": "MoeUpdateExpertTestSpec",
    "moe_update_expert": "MoeUpdateExpertTestSpec",
    }
