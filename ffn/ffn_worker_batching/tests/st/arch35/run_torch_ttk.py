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

"""Run FfnWorkerBatching Torch eager or npugraph_ex cases through ops-test-kit.

使用方法（在 ops-transformer 任意目录执行；CANN、自定义 OPP、Torch 扩展需已配置，
参见 docs/arch35_verify_guide.md 环境章节和 docs/arch35_ttk_torch.md）：

    python ffn/ffn_worker_batching/tests/st/arch35/run_torch_ttk.py \
        --mode eager --ttk-root /home/chenpengyi/ops-test-kit \
        -o /tmp/ffn_ttk_eager.csv
    python ffn/ffn_worker_batching/tests/st/arch35/run_torch_ttk.py \
        --mode graph --ttk-root /home/chenpengyi/ops-test-kit \
        -o /tmp/ffn_ttk_graph.csv

    --mode eager|graph 二选一：eager 对应 test_torch_ffn_worker_batching_arch35.csv
    （原生 dtype 输出）；graph 对应 test_torch_graph_ffn_worker_batching_arch35.csv
    （mx_output_uint8=True）并自动追加 --aclgraph（固定 shape npugraph_ex）。
    选择性打包的扩展（包名与 dispatcher 命名空间 cann_ops_transformer 不同）
    加 --package cann_ops_transformer_ffn_mx_graph，或设 FFN_TORCH_PACKAGE；
    其余参数（-t 用例名、--run 2、--task-prof false、--validate 等）透传给 TTK。

本脚本存在的意义：改名扩展必须在 TTK 解析 torch.ops schema 前注册成
cann_ops_transformer 别名，而 TTK e2e 的 forkserver worker 不继承父进程 import，
需要在每个 worker 内重新注册（见下方 __mp_main__ 块）；同时自动完成 cd 到
ops-test-kit、按 mode 选 CSV、拼 --plugin 路径。

不使用本脚本、手动在 ops-test-kit 下运行 e2e 的方法（默认包名时可用）：

    cd /home/chenpengyi/ops-test-kit
    source /home/chenpengyi/Ascend/cann/set_env.sh
    source <自定义OPP路径>/vendors/custom_transformer/bin/set_env.bash
    export PYTHONPATH=<Torch扩展安装目录>:$PYTHONPATH   # 默认完整包可不设
    python -m ttk e2e -i /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/st/arch35/test_torch_ffn_worker_batching_arch35.csv \
        --plugin /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/assets
    # 图模式：换 test_torch_graph_ffn_worker_batching_arch35.csv 并追加 --aclgraph

注意事项：
- 必须在 ops-test-kit 仓目录下执行（python -m ttk 依赖 cwd 进入 sys.path）。
- e2e 的 --plugin 传 tests/assets 目录（TTK 扫描目录内的 golden.py），与
  aclnn/kernel 模式直接传 golden.py 文件不同。
- 改名扩展（如 cann_ops_transformer_ffn_mx_graph）手动直跑可能在 schema 解析处
  失败：CSV 的 api_name 是 torch.ops.cann_ops_transformer.ffn_worker_batching，
  包名不同时无法自动导入，需走本脚本（或自行保证 forkserver worker 内注册别名）。
- 本组只覆盖 need_schedule=0（NORM）用例。RECV 会消费描述符，TTK 重复执行缺少
  生产者重新发布，插件会拒绝这类输入避免挂死；同步/异步 RECV、延迟发布、连续
  消费由 tests/st/arch35/verify_torch_*.py 验证。
- 关闭性能采集用 --task-prof false；不要用 --no-prof，当前 TTK 中该选项表示
  只准备输入/golden 而不执行算子。
- TTK 的 aclgraph 分支固定 fullgraph=False，传 --fullgraph 1 无效；完整图、
  断图拒绝和状态 replay 由 verify_torch_graph.py（fullgraph=True）验证。
"""

import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys


def _register_package(name):
    package = importlib.import_module(name)
    # Load this vendor's operator explicitly in both the parent and forkserver
    # workers. Importing only the package root can leave the dispatcher schema
    # unregistered when entry-point discovery finds a stale installed vendor.
    # TTK parses the schema before invoking the operator, so lazy loading at
    # the first call is too late and every CSV case would be API_PARSE_FAIL.
    importlib.import_module(f"{name}.ops.ffn.ffn_worker_batching")
    if name != "cann_ops_transformer":
        sys.modules.setdefault("cann_ops_transformer", package)


# TTK uses forkserver workers, which do not inherit the parent's imports.
if __name__ == "__mp_main__":
    _register_package(os.environ["FFN_TTK_TORCH_PACKAGE"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--ttk-root", type=Path, required=True)
    parser.add_argument(
        "--package", default=os.getenv("FFN_TORCH_PACKAGE", "cann_ops_transformer")
    )
    args, ttk_args = parser.parse_known_args()
    root = args.ttk_root.resolve()
    if not (root / "ttk" / "__main__.py").is_file():
        parser.error(f"Not an ops-test-kit checkout: {root}")
    case_dir = Path(__file__).resolve().parent
    filename = (
        "test_torch_ffn_worker_batching_arch35.csv"
        if args.mode == "eager"
        else "test_torch_graph_ffn_worker_batching_arch35.csv"
    )
    # Selectively built vendors keep the same dispatcher namespace but have a
    # different Python package name, so import before TTK resolves the schema.
    os.environ["FFN_TTK_TORCH_PACKAGE"] = args.package
    _register_package(args.package)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = [
        "ttk",
        "e2e",
        "-i",
        str(case_dir / filename),
        "--plugin",
        str(case_dir.parents[1] / "assets"),
    ]
    if args.mode == "graph":
        sys.argv.append("--aclgraph")
    sys.argv.extend(ttk_args)
    runpy.run_module("ttk", run_name="__main__")


if __name__ == "__main__":
    main()
