# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
已知缺陷用例的子进程 runner: 单用例独立进程执行, 崩溃(EE9999/aicore
exception)只污染本进程, 由父进程按输出/退出码判定。

用法:
    python _defect_case_runner.py <case_id> [--device N] [--graph]

    --graph: 算子走 aclgraph(NPUGraph 捕获/回放)路径执行(默认 eager);
             崩溃同样由本进程隔离, 输出协议不变。

环境变量:
    PKI_REF_DEVICE=cpu|npu   标杆设备(默认 cpu; npu 为 NPU 组合标杆)

输出协议(父进程解析):
    DEFECT_CASE_RESULT: PASS <case_id>
    DEFECT_CASE_RESULT: FAIL <case_id> <简述>     # 精度/比对失败(进程存活)
    DEFECT_CASE_RESULT: CRASH <case_id> <简述>    # NPU 运行时异常(进程存活, 已捕获)
退出码: 0=PASS, 1=FAIL, 2=CRASH; 进程被信号杀死(如 SIGABRT)由父进程按
负返回码识别为 CRASH。
"""

import os
import sys
import traceback
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: F401  (torch_npu 依赖加载顺序)
import torch_npu

DEVICE_ID = 0
SEED_BASE = 20260826


def build_case_inputs(cfg):
    """根据用例配置构造 CPU 输入(复用 pki_common, 确定性种子)。"""
    from pki_common import build_case_inputs as _build

    cfg = dict(cfg)
    cfg.setdefault("seed", SEED_BASE + (zlib.crc32(cfg["id"].encode()) & 0xFFFF))
    cfg.setdefault("key_non_contig", False)
    cfg.setdefault("pa_block_size", 16)
    return _build(cfg)


def run_case(cfg, device_id, ref_device="cpu", use_graph=False):
    from pki_common import run_golden, run_op_aclgraph, run_op_eager
    from result_compare import compare_values

    inputs = build_case_inputs(cfg)

    # ---- 标杆(cpu 金标 / npu 组合标杆, 由 PKI_REF_DEVICE 指定) ----
    g_indices, g_values = run_golden(inputs, cfg, ref_device=ref_device)

    # ---- NPU 算子(eager / aclgraph NPUGraph 捕获回放) ----
    if use_graph:
        npu_indices, npu_values = run_op_aclgraph(inputs, cfg, device_id=device_id)
    else:
        npu_indices, npu_values = run_op_eager(inputs, cfg, device_id=device_id)

    # ---- 比对(indices 集合门禁 + values 参考日志) ----
    from result_compare import compare_indices

    ok_idx = compare_indices(g_indices, npu_indices)
    if not ok_idx:
        return False, "indices set comparison failed"
    if cfg["return_value"]:
        assert npu_values.shape == g_values.shape, (
            f"values shape mismatch: golden {g_values.shape} vs npu {npu_values.shape}"
        )
        compare_values(g_values, npu_values, cfg["dtype"], label="values")
    else:
        assert npu_values.numel() == 0 or npu_values.shape[-1] == 0, (
            f"expected empty values, got shape {npu_values.shape}"
        )
    return True, ""


def main():
    global DEVICE_ID
    args = sys.argv[1:]
    if not args:
        print("usage: _defect_case_runner.py <case_id> [--device N] [--graph]")
        sys.exit(3)
    case_id = args[0]
    if "--device" in args:
        DEVICE_ID = int(args[args.index("--device") + 1])
    use_graph = "--graph" in args
    ref_device = os.environ.get("PKI_REF_DEVICE", "cpu")

    from known_defect_cases import CASE_BY_ID

    cfg = CASE_BY_ID[case_id]

    torch_npu.npu.set_device(DEVICE_ID)
    print(
        f"[runner] case={case_id} group={cfg['group']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
        f"dtype={cfg['dtype']} q_lens={cfg['q_lens']} k_pools={cfg['k_pools']} n1={cfg['n1']} "
        f"topk={cfg['topk']} ps={cfg['pool_size']} mask={cfg['mask_mode']} "
        f"val={cfg['return_value']} tail={cfg.get('tail')} ref={ref_device} note={cfg.get('note')}",
        flush=True,
    )

    try:
        ok, detail = run_case(cfg, DEVICE_ID, ref_device, use_graph=use_graph)
    except RuntimeError as e:
        # NPU 运行时异常(aicore exception / EE9999 等)
        msg = str(e).replace("\n", " ")[:500]
        print(f"DEFECT_CASE_RESULT: CRASH {case_id} RuntimeError: {msg}", flush=True)
        sys.exit(2)
    except AssertionError as e:
        msg = str(e).replace("\n", " ")[:500]
        print(f"DEFECT_CASE_RESULT: FAIL {case_id} assert: {msg}", flush=True)
        sys.exit(1)
    except Exception:
        msg = traceback.format_exc().replace("\n", " ")[-500:]
        print(f"DEFECT_CASE_RESULT: CRASH {case_id} unexpected: {msg}", flush=True)
        sys.exit(2)

    if ok:
        print(f"DEFECT_CASE_RESULT: PASS {case_id}", flush=True)
        sys.exit(0)
    print(f"DEFECT_CASE_RESULT: FAIL {case_id} {detail}", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
