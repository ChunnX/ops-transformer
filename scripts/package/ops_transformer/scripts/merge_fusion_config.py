#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
merge_fusion_config.py
usage: merge_fusion_config.py <fragment.json> <target.json>
deep merge Switch/Priority entries of fragment into target, fragment wins on conflict.
"""

import json
import os
import sys


def deep_merge(dst, src):
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            deep_merge(dst[key], val)
        else:
            dst[key] = val


def main():
    if len(sys.argv) != 3:
        print(
            "usage: merge_fusion_config.py <fragment.json> <target.json>",
            file=sys.stderr,
        )
        return 1
    frag_path = sys.argv[1]
    tgt_path = sys.argv[2]
    if not os.path.isfile(frag_path):
        print("fragment file not exist: %s" % frag_path, file=sys.stderr)
        return 1
    with open(frag_path, "r") as fh:
        fragment = json.load(fh)
    if os.path.isfile(tgt_path):
        with open(tgt_path, "r") as fh:
            target = json.load(fh)
    else:
        target = {}
    deep_merge(target, fragment)
    tgt_dir = os.path.dirname(tgt_path)
    if tgt_dir:
        os.makedirs(tgt_dir, exist_ok=True)
    tmp_path = tgt_path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(target, fh, indent=4)
        fh.write("\n")
    os.replace(tmp_path, tgt_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
