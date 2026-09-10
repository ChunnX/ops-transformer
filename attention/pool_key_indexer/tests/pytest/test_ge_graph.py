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
PoolKeyIndexer GE 图模式(TorchAir/aclgraph)测试套件(套件3)。

合并原 test_pool_key_indexer_ge_graph.py(定点场景) 与
test_pool_key_indexer_general_ge_graph.py(泛化场景), 覆盖 GE 图模式专属路径:

  1. tiling 值校验跳过: 图模式下 actual_seq_q / actual_seq_k / pool_tail_k 是
     device tensor, tiling 期数值不可见(GetData 为 null)。host 侧跳过值校验
     (非递减/末项=T1/取值范围等), 由 kernel 运行期从 GM 读值(GetActualSeqLen)。
     覆盖 TND 与 PA_BBND 布局;
  2. PA key 0 轴非连续: converter(graph_convert_pool_key_indexer.py)从转换期
     tensor 元数据读 stride(0) 下传 key_stride0 属性(与 eager 路径 csrc 直读
     at::Tensor::stride(0) 对齐); 属性缺失时回退 tiling context 运行时 stride
     (框架填 TensorV2 view);
  3. 图编译期 compile info 解析(TilingParse 注册, 原 "compile info not
     contain [_pattern]" 失败已修复);
  4. 流水竞争回归: mask=0 且池数 < sparseCount 时 "-1 清理"分支与尾 token
     追加分支同行背靠背执行, 缺 V->S 硬同步时标量尾写被间歇性覆盖
     (GE 图模式曾 96/128 行丢尾 token), 以 REPEAT 次重复调用捕获偶发失败。

用例构成:
  - 定点场景(7 例): 基线/tiling 跳过/PA/非连续/竞争回归;
  - 泛化场景(60 例): 复用精度套件泛化用例配置(确定性同 seed), 后端替换为
    torchair GE 图编译(每例独立编译 ~10s, 全量约 10 分钟)。

比对策略(与精度套件一致):
  - indices: 逐行有效索引(>=0)排序集合比对, 100% 行通过, 唯一精度门禁;
  - values:  逐行有效值排序数值比对, 仅打印参考日志;
  - 输出 dtype/shape 与接口规格的显式断言。

标杆切换: pytest test_ge_graph.py --ref-device npu(默认 cpu 金标)。
子集控制: PKI_GE_GENERAL_MAX=N 只取前 N 例泛化用例(冒烟)。

依赖: torchair(缺失则整文件 skip); NPU 设备(不可用则 skip)。
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("torchair")

if not torch.npu.is_available():
    pytest.skip("NPU device is not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from pki_common import (  # noqa: E402
    PoolKeyIndexerModule,
    _op_tensor_args,
    assert_output_spec,
    build_case_inputs,
    compile_ge_graph,
    npu_device,
    run_golden,
)
from pki_cases import (  # noqa: E402
    ALL_GENERAL_CASES,
    COVERAGE as _COVERAGE,
    update_coverage as _update_coverage,
)
from result_compare import compare_indices, compare_values  # noqa: E402

from cann_ops_transformer import pool_key_indexer  # noqa: E402,F401

# 可选子集规模控制(冒烟): PKI_GE_GENERAL_MAX=N 只取前 N 例; 未设置取全量
_MAX = int(os.environ.get("PKI_GE_GENERAL_MAX", "0"))
GE_CASES = ALL_GENERAL_CASES[:_MAX] if _MAX > 0 else ALL_GENERAL_CASES


# ---------------------------------------------------------------------------
# 运行与比对封装
# ---------------------------------------------------------------------------


def _run_ge_and_compare(cfg, ref_device, tag):
    """GE 图模式执行单用例: 标杆 -> 图编译执行 -> 规格断言 + 集合比对。"""
    inputs = build_case_inputs(cfg)
    args = _op_tensor_args(inputs, npu_device())
    g_indices, g_values = run_golden(inputs, cfg, ref_device=ref_device)

    model = compile_ge_graph(
        PoolKeyIndexerModule(
            layout_q=cfg["layout_q"],
            layout_k=cfg["layout_k"],
            topk=cfg["topk"],
            pool_size=cfg["pool_size"],
            mask_mode=cfg["mask_mode"],
            quant_mode=-1,
            return_value=cfg["return_value"],
        )
    )
    npu_indices, npu_values = model(*args)
    torch.npu.synchronize()
    npu_indices = npu_indices.cpu()
    npu_values = npu_values.cpu()

    assert_output_spec(npu_indices, npu_values, cfg)

    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"[{tag}] GE graph mode indices set comparison failed "
        f"({cfg['layout_q']}/{cfg['layout_k']} ps={cfg['pool_size']} "
        f"mask={cfg['mask_mode']})"
    )
    if cfg["return_value"]:
        compare_values(g_values, npu_values, cfg["dtype"], label=f"{tag}-values")
    print(f"[{tag}] GE graph mode PASSED")


# ---------------------------------------------------------------------------
# 1. 定点场景
# ---------------------------------------------------------------------------


def test_ge_graph_bsnd_baseline(ref_device):
    """BSND + causal + pooling: 图模式基线。"""
    cfg = dict(
        id="ge_bsnd_baseline",
        layout_q="BSND",
        layout_k="BSND",
        dtype=torch.float16,
        q_lens=[64, 64],
        k_pools=[16, 16],
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=42,
    )
    _run_ge_and_compare(cfg, ref_device, "bsnd_baseline")


def test_ge_graph_tnd(ref_device):
    """TND + no mask: actual_seq_q/k 为 device tensor(tiling 值校验跳过路径)。

    此前以 "TND: actual_seq_q data is null, tiling sink not supported" 失败,
    修复后 kernel 运行期从 GM 读前缀和。
    """
    cfg = dict(
        id="ge_tnd",
        layout_q="TND",
        layout_k="TND",
        dtype=torch.bfloat16,
        q_lens=[32, 48],
        k_pools=[8, 12],
        n1=8,
        topk=64,
        pool_size=16,
        mask_mode=0,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=7,
    )
    _run_ge_and_compare(cfg, ref_device, "tnd")


def test_ge_graph_bsnd_pa(ref_device):
    """BSND query + PA_BBND key: actual_seq_k + block_table 为 device tensor。

    此前 tiling 期以 "actual_seq_k data is null, tiling sink not supported"
    失败; 含不等池数(-1 空槽 + 尾块 padding)。
    """
    cfg = dict(
        id="ge_bsnd_pa",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=torch.float16,
        q_lens=[32, 32],
        k_pools=[16, 8],
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=11,
    )
    _run_ge_and_compare(cfg, ref_device, "bsnd_pa")


def test_ge_graph_tnd_pa(ref_device):
    """TND query + PA_BBND key: 双 actual_seq 均为 device tensor, block_size=32。"""
    cfg = dict(
        id="ge_tnd_pa",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=torch.float16,
        q_lens=[16, 16],
        k_pools=[16, 16],
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=32,
        key_non_contig=False,
        tail=None,
        seed=13,
    )
    _run_ge_and_compare(cfg, ref_device, "tnd_pa")


def test_ge_graph_bsnd_pa_key_noncontig(ref_device):
    """BSND + PA_BBND + key 0 轴非连续(key_stride0 属性路径)。

    物理块嵌入 2 倍缓冲偶数行, 视图 stride(0) = 2 * blockSize * N2 * D。
    GE 图模式下 converter 在转换期读 stride(0) 下传 key_stride0 属性
    (属性缺失时回退框架上报的运行时 stride); 标杆直接消费同一 CPU 视图。
    """
    dtype = torch.float16
    batch, s1, n1 = 2, 32, 8
    k_pool_lens = [16, 16]
    topk, pool_size, mask_mode = 128, 16, 0
    pa_block_size = 16

    cfg = dict(
        id="ge_bsnd_pa_key_noncontig",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=dtype,
        q_lens=[s1] * batch,
        k_pools=k_pool_lens,
        n1=n1,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=True,
        pa_block_size=pa_block_size,
        key_non_contig=True,
        tail=None,
        seed=17,
    )
    inputs = build_case_inputs(cfg)
    # 显式复核非连续视图属性(build_case_inputs 已构造 npu_key_fn)
    assert inputs["npu_key_fn"] is not None
    assert not inputs["pool_key"].is_contiguous()
    assert inputs["pool_key"].stride(0) == 2 * pa_block_size * 1 * 128

    g_indices, g_values = run_golden(inputs, cfg, ref_device=ref_device)

    args = _op_tensor_args(inputs, npu_device())
    model = compile_ge_graph(
        PoolKeyIndexerModule(
            layout_q=cfg["layout_q"],
            layout_k=cfg["layout_k"],
            topk=topk,
            pool_size=pool_size,
            mask_mode=mask_mode,
            quant_mode=-1,
            return_value=True,
        )
    )
    npu_indices, npu_values = model(*args)
    torch.npu.synchronize()
    npu_indices, npu_values = npu_indices.cpu(), npu_values.cpu()

    assert_output_spec(npu_indices, npu_values, cfg)
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, "[bsnd_pa_key_noncontig] GE graph indices comparison failed"
    compare_values(g_values, npu_values, dtype, label="noncontig-values")
    print("[bsnd_pa_key_noncontig] GE graph mode PASSED")


def _race_stress(cfg, ref_device, tag):
    """竞争回归公共逻辑: 首次图执行比对 + 编译图 REPEAT 次重复调用比对。"""
    REPEAT = 20
    inputs = build_case_inputs(cfg)
    g_indices, _ = run_golden(inputs, cfg, ref_device=ref_device)

    args = _op_tensor_args(inputs, npu_device())
    model = compile_ge_graph(
        PoolKeyIndexerModule(
            layout_q=cfg["layout_q"],
            layout_k=cfg["layout_k"],
            topk=cfg["topk"],
            pool_size=cfg["pool_size"],
            mask_mode=cfg["mask_mode"],
            quant_mode=-1,
            return_value=cfg["return_value"],
        )
    )
    idx, _ = model(*args)
    torch.npu.synchronize()
    assert compare_indices(g_indices, idx.cpu()), f"[{tag}] first call indices mismatch"

    # 重复执行已编译图, 压力激发行循环流水竞争: 每次调用必须与标杆行集一致
    for rep in range(REPEAT):
        idx, _ = model(*args)
        torch.npu.synchronize()
        assert compare_indices(g_indices, idx.cpu()), (
            f"[{tag}] race regression: repeat {rep}/{REPEAT} indices mismatch "
            f"(mask={cfg['mask_mode']} pools={cfg['k_pools'][0]} < "
            f"sparse={cfg['topk'] // cfg['pool_size']})"
        )
    print(f"[{tag}] race stress: {REPEAT}/{REPEAT} repeats all matched")


def test_ge_graph_bsnd_mask0_pools_lt_sparse_race(ref_device):
    """BSND + mask=0 + 池数 < sparseCount(尾追加流水竞争回归)。

    arch22 ExpandAndAppendIndices 流水竞争: mask_mode=0 且池数 < sparseCount
    (topk // pool_size) 时, "-1 清理"分支(validExpand < topk)与尾 token 追加
    分支同行背靠背执行。尾区 Duplicate(-1) 后缺 V->S 硬同步、下一行覆写共享
    UB 缓冲前缺 MTE3->V 等待时, 标量尾写被间歇性覆盖(GE 图模式曾 96/128 行
    丢尾 token; eager 被下发时序掩盖)。本用例固定失败组合并重复 REPEAT 次
    调用已编译图捕获偶发失败。
    """
    cfg = dict(
        id="ge_bsnd_mask0_pools_lt_sparse",
        layout_q="BSND",
        layout_k="BSND",
        dtype=torch.float16,
        q_lens=[64, 64],
        k_pools=[4, 4],
        n1=8,
        topk=224,
        pool_size=16,
        mask_mode=0,
        return_value=False,
        pa_block_size=16,
        key_non_contig=False,
        tail=[5, 13],  # 确定性非零尾, 防止 tail==0 时假通过
        seed=23,
    )
    _race_stress(cfg, ref_device, "bsnd_mask0_pools_lt_sparse")


def test_ge_graph_tnd_mask0_pools_lt_sparse_race(ref_device):
    """TND + mask=0 + 池数 < sparseCount(竞争回归, TND 形态)。

    与 BSND 形态相同的竞争模式, 额外覆盖 TND 前缀和 actual_seq 为 device
    tensor 的 tiling 值校验跳过路径。
    """
    cfg = dict(
        id="ge_tnd_mask0_pools_lt_sparse",
        layout_q="TND",
        layout_k="TND",
        dtype=torch.bfloat16,
        q_lens=[32, 48],
        k_pools=[4, 2],
        n1=8,
        topk=224,
        pool_size=16,
        mask_mode=0,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=[5, 13],  # 确定性非零尾
        seed=29,
    )
    _race_stress(cfg, ref_device, "tnd_mask0_pools_lt_sparse")


# ---------------------------------------------------------------------------
# 2. 泛化场景(复用精度套件泛化用例配置)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", GE_CASES, ids=[c["id"] for c in GE_CASES])
def test_ge_graph_general(cfg, ref_device):
    """GE 图模式泛化用例: 标杆 vs torchair 编译执行。"""
    print(
        f"\n=== General-GE: {cfg['id']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
        f"dtype={cfg['dtype']} batch={len(cfg['q_lens'])} q_lens={cfg['q_lens']} "
        f"k_pools={cfg['k_pools']} n1={cfg['n1']} topk={cfg['topk']} ps={cfg['pool_size']} "
        f"mask={cfg['mask_mode']} val={cfg['return_value']} "
        f"pa_bs={cfg['pa_block_size']} noncontig={cfg['key_non_contig']} "
        f"tail={cfg.get('tail')} ref={ref_device} ==="
    )
    _run_ge_and_compare(cfg, ref_device, f"general-{cfg['id']}")
    _update_coverage(cfg, cfg["k_pools"], cfg["q_lens"])
    print(f"=== GENERAL-GE PASSED: {cfg['id']} ===")


def test_zz_ge_coverage_summary():
    """GE 图模式泛化覆盖汇总: 校验各参数轴覆盖桶均被命中(与精度套件口径一致)。"""
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
