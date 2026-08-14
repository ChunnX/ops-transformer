#!/usr/bin/python
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

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


class _FakeTensor:
    def __init__(self, numel=1):
        self._numel = numel

    def numel(self):
        return self._numel


def _fake_torch_module():
    module = types.ModuleType("torch")
    module.Tensor = _FakeTensor
    module.nn = types.SimpleNamespace(Module=object)
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBsaTtkPluginActivation(unittest.TestCase):
    def test_fp8_pack_runs_only_after_plugin_activation(self):
        with mock.patch.dict(sys.modules, {"torch": _fake_torch_module()}):
            module = _load_module(
                "bsa_ttk_ops_plugin_activation_test",
                ASSETS_DIR / "bsa_ttk_ops.py",
            )

        pack_calls = []

        def fake_pack(key, value, k_descale):
            pack_calls.append((key, value, k_descale))
            return "packed_key", "packed_value", "packed_k_descale"

        module._fp8_pack_combined_kv = fake_pack
        inputs = {
            "query": object(),
            "key": object(),
            "value": object(),
            "q_descale": object(),
            "k_descale": object(),
            "v_descale": object(),
            "p_scale": None,
            "sparse_indices": object(),
            "sparse_seq_len": object(),
            "atten_mask": object(),
            "cu_seqlens_q": object(),
            "cu_seqlens_kv": object(),
            "seqused_q": object(),
            "seqused_kv": object(),
            "block_table": object(),
            "metadata": _FakeTensor(numel=8),
        }
        options = {
            "quant_mode": 1,
            "softmax_scale": 1.0,
            "mask_mode": 3,
            "sparse_block_size_q": 128,
            "sparse_block_size_kv": 128,
            "layout_q": "TND",
            "layout_kv": "PA_BNBD",
            "layout_sparse_indices": "B_N_Qb_Kb",
        }

        unpacked = module._prepare_operator_inputs(**inputs, **options)
        self.assertIs(unpacked[0], inputs["key"])
        self.assertEqual(pack_calls, [])

        module.mark_ttk_plugin_active()
        packed = module._prepare_operator_inputs(**inputs, **options)
        self.assertEqual(
            packed[:3],
            ("packed_key", "packed_value", "packed_k_descale"),
        )
        self.assertEqual(
            pack_calls,
            [(inputs["key"], inputs["value"], inputs["k_descale"])],
        )

    def test_loading_spec_activates_bsa_ttk_plugin(self):
        activated = []
        fake_bsa = types.ModuleType("bsa_ttk_ops")
        fake_bsa.QuantBlockSparseAttnGraph = type("QuantBlockSparseAttnGraph", (), {})
        fake_bsa.mark_ttk_plugin_active = lambda: activated.append(True)

        with mock.patch.dict(sys.modules, {"bsa_ttk_ops": fake_bsa}):
            _load_module("qbsa_spec_plugin_activation_test", ASSETS_DIR / "spec.py")

        self.assertEqual(activated, [True])


if __name__ == "__main__":
    unittest.main()
