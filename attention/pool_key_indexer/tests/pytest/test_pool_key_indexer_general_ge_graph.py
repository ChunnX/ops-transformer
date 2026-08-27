# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer GE 图模式(TorchAir)泛化验证测试。

复用 test_pool_key_indexer_general.py 的全部用例配置(ALL_CASES, 确定性
MASTER_SEED 生成, 随机 40 例 + 定向 20 例), 将执行后端从 eager torch
extension 替换为 torchair GE 图编译(torch.compile + fullgraph), 验证:

  1. TilingParse 注册后图编译期 compile info 解析(原 "compile info not
     contain [_pattern]" 失败已修复);
  2. 图模式下 tiling 值校验跳过路径(actual_seq_q/k/pool_tail_k 为 device
     tensor, 数值编译期不可见; kernel 运行期从 GM 读值);
  3. key_stride0 属性经 converter 下传(PA key 0 轴非连续)。

比对策略与 eager 泛化一致:
  - indices: 逐行有效索引集合比对(>=0 排序集合相等), 唯一精度门禁;
  - values:  逐行有效值排序数值比对, 仅打印参考日志;
  - 输出 dtype/shape 与接口规格的显式断言。

执行效率说明: 每个 case 独立触发一次图编译(~10s), 60 例全量约 10 分钟;
支持环境变量 PKI_GE_GENERAL_MAX 控制子集规模(冒烟), 全量验证不设限。

依赖: torchair(缺失则整文件 skip); NPU 设备(不可用则 skip)。
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
torchair = pytest.importorskip("torchair")

if not torch.npu.is_available():
    pytest.skip("NPU device is not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from result_compare import compare_indices, compare_values  # noqa: E402
from pool_key_indexer_reference import pool_key_indexer_reference  # noqa: E402

# 复用 eager 泛化用例的全部配置与输入构造(确定性, 同 seed 可复现)
from test_pool_key_indexer_general import (  # noqa: E402
    ALL_CASES,
    _build_case_inputs,
    _COVERAGE,
    _update_coverage,
)

# Import torch extension (registers torch.ops.cann_ops_transformer.pool_key_indexer)
from cann_ops_transformer import pool_key_indexer  # noqa: E402,F401

# 可选子集规模控制(冒烟): PKI_GE_GENERAL_MAX=N 只取前 N 例; 未设置取全量
_MAX = int(os.environ.get("PKI_GE_GENERAL_MAX", "0"))
GE_CASES = ALL_CASES[:_MAX] if _MAX > 0 else ALL_CASES


# ---------------------------------------------------------------------------
# GE graph mode helpers
# ---------------------------------------------------------------------------


class _PoolKeyIndexerModule(torch.nn.Module):
    """Wrapper calling the torch.ops path (converter interception point)."""

    def __init__(self, **op_kwargs):
        super().__init__()
        self._op_kwargs = op_kwargs

    def forward(
        self,
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=None,
        actual_seq_k=None,
        block_table=None,
    ):
        return torch.ops.cann_ops_transformer.pool_key_indexer(
            query,
            pool_key,
            weights,
            pool_tail_k,
            actual_seq_q=actual_seq_q,
            actual_seq_k=actual_seq_k,
            block_table=block_table,
            **self._op_kwargs,
        )


def _compile_ge_graph(module):
    """Compile with the torchair GE graph backend (fullgraph, static shape)."""
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    return torch.compile(module, fullgraph=True, backend=npu_backend, dynamic=False)


@pytest.fixture(scope="session", autouse=True)
def _npu_init():
    torch_npu.npu.set_device(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", GE_CASES, ids=[c["id"] for c in GE_CASES])
def test_pool_key_indexer_general_ge_graph(cfg):
    """GE 图模式泛化用例: CPU golden 标杆 vs torchair 编译执行。"""
    inputs = _build_case_inputs(cfg)
    ps = cfg["pool_size"]
    topk = cfg["topk"]
    sparse = topk // ps
    out_len = topk + ps - 1

    print(
        f"\n=== General-GE: {cfg['id']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
        f"dtype={cfg['dtype']} batch={len(cfg['q_lens'])} q_lens={cfg['q_lens']} "
        f"k_pools={cfg['k_pools']} n1={cfg['n1']} topk={topk} ps={ps} "
        f"mask={cfg['mask_mode']} val={cfg['return_value']} "
        f"pa_bs={cfg['pa_block_size']} noncontig={cfg['key_non_contig']} "
        f"tail={cfg.get('tail')} ==="
    )

    # ---- CPU golden ----
    g_indices, g_values = pool_key_indexer_reference(
        inputs["query"],
        inputs["pool_key"],
        inputs["weights"],
        inputs["pool_tail_k"],
        actual_seq_q=inputs["actual_seq_q"],
        actual_seq_k=inputs["actual_seq_k"],
        block_table=inputs["block_table"],
        layout_q=cfg["layout_q"],
        layout_k=cfg["layout_k"],
        topk=topk,
        pool_size=ps,
        mask_mode=cfg["mask_mode"],
        return_value=cfg["return_value"],
    )

    # ---- GE graph mode ----
    dev = "npu:0"
    key_npu = (
        inputs["npu_key_fn"](dev)
        if inputs["npu_key_fn"] is not None
        else inputs["pool_key"].to(dev)
    )
    model = _compile_ge_graph(
        _PoolKeyIndexerModule(
            layout_q=cfg["layout_q"],
            layout_k=cfg["layout_k"],
            topk=topk,
            pool_size=ps,
            mask_mode=cfg["mask_mode"],
            quant_mode=-1,
            return_value=cfg["return_value"],
        )
    )
    npu_indices, npu_values = model(
        inputs["query"].to(dev),
        key_npu,
        inputs["weights"].to(dev),
        inputs["pool_tail_k"].to(dev),
        inputs["actual_seq_q"].to(dev) if inputs["actual_seq_q"] is not None else None,
        inputs["actual_seq_k"].to(dev) if inputs["actual_seq_k"] is not None else None,
        inputs["block_table"].to(dev) if inputs["block_table"] is not None else None,
    )
    torch.npu.synchronize()
    npu_indices = npu_indices.cpu()
    npu_values = npu_values.cpu()

    # ---- 输出规格断言(dtype/shape, 接口文档口径) ----
    assert npu_indices.dtype == torch.int32, (
        f"indices dtype {npu_indices.dtype} != int32"
    )
    if cfg["layout_q"] == "BSND":
        exp_idx = (len(cfg["q_lens"]), cfg["q_lens"][0], out_len)
        exp_val = (len(cfg["q_lens"]), cfg["q_lens"][0], sparse)
    else:
        exp_idx = (sum(cfg["q_lens"]), out_len)
        exp_val = (sum(cfg["q_lens"]), sparse)
    assert tuple(npu_indices.shape) == exp_idx, (
        f"indices shape {tuple(npu_indices.shape)} != {exp_idx}"
    )
    if cfg["return_value"]:
        assert npu_values.dtype == torch.float32, (
            f"values dtype {npu_values.dtype} != float32"
        )
        assert tuple(npu_values.shape) == exp_val, (
            f"values shape {tuple(npu_values.shape)} != {exp_val}"
        )
    else:
        assert npu_values.numel() == 0, (
            f"return_value=False 时 values 应为空, got {tuple(npu_values.shape)}"
        )

    # ---- 比对(集合比对策略) ----
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"[{cfg['id']}] GE graph indices set comparison failed: "
        f"layout=({cfg['layout_q']},{cfg['layout_k']}) ps={ps} topk={topk} "
        f"mask={cfg['mask_mode']}"
    )
    if cfg["return_value"]:
        compare_values(g_values, npu_values, cfg["dtype"], label="ge-values")

    # 复用 eager 泛化的覆盖统计(独立于其进程, 汇总打印见文末)
    _update_coverage(cfg, cfg["k_pools"], cfg["q_lens"])
    print(f"=== GENERAL-GE PASSED: {cfg['id']} ===")


def test_zz_general_ge_coverage_summary():
    """GE 图模式泛化覆盖汇总: 校验各参数轴覆盖桶均被命中(与 eager 口径一致)。"""
    if not GE_CASES:
        pytest.skip("no cases")
    cov = _COVERAGE
    missing = []
    if len(cov["layout"]) < 4:
        missing.append(f"layout buckets {cov['layout']}")
    if len(cov["mask_mode"]) < 2:
        missing.append(f"mask_mode {cov['mask_mode']}")
    if len(cov["return_value"]) < 2:
        missing.append(f"return_value {cov['return_value']}")
    if len(cov["pool_size_bucket"]) < 4:
        missing.append(f"pool_size buckets {cov['pool_size_bucket']}")
    if len(cov["batch_bucket"]) < 3:
        missing.append(f"batch buckets {cov['batch_bucket']}")
    if not cov["s1_eq_1"]:
        missing.append("S1=1")
    if not cov["pools_lt_sparse"]:
        missing.append("pools<sparseCount")
    if not cov["zero_q_batch"]:
        missing.append("zero-length query batch")
    if not cov["key_non_contig"]:
        missing.append("PA key non-contiguous")
    print(
        f"[GE-general coverage] executed={cov['cases_executed']} "
        f"layouts={sorted(cov['layout'])} dtypes={len(cov['dtype_layout'])} "
        f"ps_buckets={sorted(cov['pool_size_bucket'])} "
        f"topk_buckets={sorted(cov['topk_bucket'])} "
        f"n1={sorted(cov['n1'])} pa_bs={sorted(cov['pa_block_size'])} "
        f"tail={sorted(cov['tail_bound'])}"
    )
    assert not missing, f"GE generalization coverage incomplete: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
