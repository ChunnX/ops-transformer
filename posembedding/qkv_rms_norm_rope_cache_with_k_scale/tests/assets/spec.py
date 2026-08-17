#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.

"""TTK TestSpec for QkvRmsNormRopeCacheWithKScale."""

import importlib
import importlib.util
from pathlib import Path
import sys


ASSET_DIR = Path(__file__).resolve().parent
IMPL_DIR = ASSET_DIR / "impl"
IMPL_PACKAGE = "qkv_rms_norm_rope_cache_with_k_scale_ttk_impl"

if IMPL_PACKAGE not in sys.modules:
    package_spec = importlib.util.spec_from_file_location(
        IMPL_PACKAGE,
        IMPL_DIR / "__init__.py",
        submodule_search_locations=[str(IMPL_DIR)],
    )
    if package_spec is None or package_spec.loader is None:
        raise ImportError(f"cannot load TTK implementation package from {IMPL_DIR}")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[IMPL_PACKAGE] = package
    package_spec.loader.exec_module(package)

adapters = importlib.import_module(f"{IMPL_PACKAGE}.adapters")
compare = importlib.import_module(f"{IMPL_PACKAGE}.compare")
inputs = importlib.import_module(f"{IMPL_PACKAGE}.inputs")


__spec__ = {
    "qkv_rms_norm_rope_cache_with_k_scale": "QkvRmsNormRopeCacheWithKScaleTestSpec",
    "aclnnQkvRmsNormRopeCacheWithKScale": "AclnnQkvRmsNormRopeCacheWithKScaleTestSpec",
    "qkv_ttk_wrapper.pta_functional": "PtaQkvRmsNormRopeCacheWithKScaleFunctionalTestSpec",
    "qkv_ttk_wrapper.pta_inplace": "PtaQkvRmsNormRopeCacheWithKScaleInplaceTestSpec",
}


TOLERANCE = {
    "bfloat16": {"standard": "stat_rel_err"},
    "float32": {"standard": "stat_rel_err"},
    # Custom compare owns the FP8 exact/dequantized metrics.  Keep the fallback
    # standard valid for TTK's pre-hook tolerance resolution as well.
    "float8_e4m3fn": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
}


class QkvRmsNormRopeCacheWithKScaleTestSpec:
    golden = adapters.kernel_golden
    customize_inputs = inputs.kernel_customize_inputs
    compare = compare.compare_outputs
    tolerance = TOLERANCE

class AclnnQkvRmsNormRopeCacheWithKScaleTestSpec:
    golden = adapters.aclnn_golden
    customize_inputs = inputs.aclnn_customize_inputs
    compare = compare.compare_outputs
    tolerance = TOLERANCE


class PtaQkvRmsNormRopeCacheWithKScaleFunctionalTestSpec:
    golden = adapters.pta_functional_golden
    customize_inputs = inputs.pta_customize_inputs
    compare = compare.compare_outputs
    tolerance = TOLERANCE


class PtaQkvRmsNormRopeCacheWithKScaleInplaceTestSpec:
    golden = adapters.pta_inplace_golden
    customize_inputs = inputs.pta_customize_inputs
    compare = compare.compare_outputs
    tolerance = TOLERANCE
