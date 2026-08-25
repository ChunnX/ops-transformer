#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run the official ST L2 generator with its aclIntArray list check repaired locally."""

import importlib.util
import os
import sys


def main():
    skill_root = os.environ["GLACIER_SKILL_ROOT"]
    scripts = os.path.join(skill_root, "skills", "ascendc-st-design", "scripts")
    sys.path.insert(0, scripts)
    script = os.path.join(scripts, "generate_test_cases.py")
    spec = importlib.util.spec_from_file_location("ascendc_st_generate_test_cases", script)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    original = generator._process_scalar_types

    def process_scalar_types(param_name, param_info, error_case, scalar_dtypes_list, attrs_dict):
        param_type = param_info.get("type", "")
        if param_type not in ("aclIntArray", "aclFloatArray", "aclBoolArray"):
            return original(param_name, param_info, error_case, scalar_dtypes_list, attrs_dict)
        value = error_case.get(f"{param_name}.value", "")
        present = len(value) != 0 if isinstance(value, (list, tuple)) else generator.pd.notna(value) and value != ""
        if present:
            attrs_dict[param_name] = list(value) if isinstance(value, (list, tuple)) else value

    generator._process_scalar_types = process_scalar_types
    generator.main()


if __name__ == "__main__":
    main()
