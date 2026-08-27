# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer 已知缺陷场景测试(缺陷家族 C/D/E, 见 known_defect_cases.py)。

背景: test_pool_key_indexer.py / test_pool_key_indexer_general.py 曾将这些
组合主动排除(2026-08-24, arch22/910B3 实测崩溃 EE9999/507015 且崩溃会
污染进程内后续用例, 故未作为 xfail 纳入)。本套件按缺陷家族重新纳入:
  - 缺陷C: pool_size ∈ [2,7] 及非 8 的倍数(规格声称 [1,128] 任意整数)
  - 缺陷D: topk ∈ {5120, 6144, 7168} shape 相关崩溃
  - 缺陷E: pool_size>=32 且 topk>=6144 确定性崩溃

进程隔离: 每个用例经 _defect_case_runner.py 在独立子进程执行。EE9999
崩溃只污染子进程, 父进程按输出协议(DEFECT_CASE_RESULT)与退出码判定,
崩溃用例以明确的 CRASH 信息失败而非雪崩整个会话。

当前目标平台 arch35(Ascend 950PR); arch22 侧由对应责任人另行处理。

修复状态记录见 docs/build_fix_record.md。
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from known_defect_cases import ALL_DEFECT_CASES, GROUP_PS32_TOPK6K, GROUP_POOL_SIZE, GROUP_TOPK_SPECIAL

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_defect_case_runner.py")
CASE_TIMEOUT_SEC = 600  # 首次 JIT 编译兜底; 扩展缓存命中时单用例 ~10-60s


def _run_defect_case(case_id):
    """在独立子进程中执行单用例, 返回 (verdict, detail)。

    verdict: PASS / FAIL / CRASH / TIMEOUT
    """
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", os.path.dirname(os.path.abspath(__file__)))
    try:
        proc = subprocess.run(
            [sys.executable, RUNNER, case_id],
            capture_output=True, text=True, timeout=CASE_TIMEOUT_SEC, env=env)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"subprocess exceeded {CASE_TIMEOUT_SEC}s"

    # 优先解析子进程输出协议行
    verdict, detail = None, ""
    for line in proc.stdout.splitlines():
        if line.startswith("DEFECT_CASE_RESULT:"):
            tokens = line[len("DEFECT_CASE_RESULT:"):].strip().split(" ", 2)
            verdict = tokens[0]
            detail = tokens[2] if len(tokens) > 2 else ""
            break

    if verdict is None:
        # 协议行缺失: 子进程被信号杀死/解释器级崩溃, 按退出码兜底判定
        if proc.returncode < 0:
            verdict = "CRASH"
            detail = f"killed by signal {-proc.returncode}"
        else:
            verdict = "CRASH"
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            detail = "no result line; last output: " + (tail[-1][:300] if tail else "<empty>")
    return verdict, detail


@pytest.mark.parametrize("case", ALL_DEFECT_CASES, ids=[c["id"] for c in ALL_DEFECT_CASES])
def test_known_defect_case(case):
    """单缺陷用例: 独立子进程执行, 期望修复后 PASS。"""
    verdict, detail = _run_defect_case(case["id"])
    assert verdict == "PASS", (
        f"[{case['id']}] expected PASS, got {verdict}: {detail} "
        f"(note: {case.get('note', '')})")


def test_defect_case_inventory():
    """用例清单完整性: 三家族齐备且 id 唯一。"""
    ids = [c["id"] for c in ALL_DEFECT_CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    assert len(GROUP_POOL_SIZE) >= 10 and len(GROUP_TOPK_SPECIAL) >= 8 and len(GROUP_PS32_TOPK6K) >= 5
    for c in ALL_DEFECT_CASES:
        assert c["topk"] % c["pool_size"] == 0, (
            f"{c['id']}: topk % pool_size must be 0 (host tiling constraint)")
        assert 1 <= c["pool_size"] <= 128


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
