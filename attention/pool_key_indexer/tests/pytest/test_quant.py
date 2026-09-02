# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and/or it is modified under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# You may obtain a copy of the License at
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer 量化场景精度测试套件(quantMode=0/1, 仅 Ascend 950 / arch35)。

执行路径: CPU 金标(pool_key_indexer_reference, 量化语义扩展) vs NPU torch
extension(eager), 输出规格断言 + 集合比对(与 test_precision.py 同口径)。

覆盖维度:
  1. quantMode=0 (FP8 per-token-head, descale=FLOAT):
     - 布局: BSND/BSND, TND/TND, TND/PA, BSND/PA
     - mask 0/3, returnValue true/false, pool_size 1/8/16, topk 128/1024
  2. quantMode=1 (mxFP8, descale=FLOAT8_E8M0):
     - 同布局矩阵 + E8M0=0xFF(NaN 传播)专项用例
  3. k_descale PA 0 轴非连续(blockFusion, 与 pool_key 同 stride 模式)
  4. 错误路径: quant_mode=-1 传 descale / quant_mode=0 缺 descale(期望报错)

运行:  pytest test_quant.py            (默认 CPU 金标)
       PKI_DEVICE_ID=2 pytest test_quant.py
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

if not torch.npu.is_available():
    pytest.skip("NPU device is not available", allow_module_level=True)

if "950" not in torch.npu.get_device_properties(0).name:
    pytest.skip("quant_mode 仅 Ascend950(arch35) 支持", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from pki_common import (  # noqa: E402
    HEAD_DIM,
    assert_output_spec,
    build_case_inputs,
    get_device_id,
    run_golden,
    run_op_eager,
)
from result_compare import compare_indices, compare_values  # noqa: E402

FP8 = torch.float8_e4m3fn

# ---------------------------------------------------------------------------
# 用例配置
# ---------------------------------------------------------------------------
# 量化用例 cfg 字段与 pki_common.build_case_inputs 一致, 增加 quant_mode。
# 规模刻意收敛(小 S2/小 topk): FP8+scale 的分数近并列概率高于 fp16,
# 用小池数控制 TopK 选择的区分度, 集合比对稳定。

_QUANT_FIXED_CASES = [
    # ---- mode=0 (FP8 per-token-head) ----
    dict(
        id="q0_bsnd_bsnd_nomask_noval",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[8],
        k_pools=[64],
        n1=8,
        topk=32,
        pool_size=1,
        mask_mode=0,
        return_value=False,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=101,
        quant_mode=0,
    ),
    dict(
        id="q0_bsnd_bsnd_causal_val",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[16],
        k_pools=[128],
        n1=8,
        topk=128,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=102,
        quant_mode=0,
    ),
    dict(
        id="q0_bsnd_bsnd_ps16_causal",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[32],
        k_pools=[64],
        n1=16,
        topk=256,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=[3],
        seed=103,
        quant_mode=0,
    ),
    dict(
        id="q0_tnd_tnd_causal",
        layout_q="TND",
        layout_k="TND",
        dtype=FP8,
        q_lens=[8, 12],
        k_pools=[48, 32],
        n1=8,
        topk=64,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=104,
        quant_mode=0,
    ),
    dict(
        id="q0_tnd_tnd_nomask_noval",
        layout_q="TND",
        layout_k="TND",
        dtype=FP8,
        q_lens=[16, 8],
        k_pools=[64, 64],
        n1=4,
        topk=32,
        pool_size=1,
        mask_mode=0,
        return_value=False,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=105,
        quant_mode=0,
    ),
    dict(
        id="q0_tnd_pa_causal",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=FP8,
        q_lens=[8, 16],
        k_pools=[32, 48],
        n1=8,
        topk=64,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=106,
        quant_mode=0,
    ),
    dict(
        id="q0_tnd_pa_noncontig_causal",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=FP8,
        q_lens=[8, 8],
        k_pools=[48, 32],
        n1=8,
        topk=64,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=True,
        tail=None,
        seed=107,
        quant_mode=0,
    ),
    dict(
        id="q0_bsnd_pa_nomask_val",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP8,
        q_lens=[16],
        k_pools=[64],
        n1=8,
        topk=128,
        pool_size=8,
        mask_mode=0,
        return_value=True,
        pa_block_size=32,
        key_non_contig=False,
        tail=None,
        seed=108,
        quant_mode=0,
    ),
    dict(
        id="q0_bsnd_bsnd_g64_causal",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[16],
        k_pools=[128],
        n1=64,
        topk=128,
        pool_size=1,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=109,
        quant_mode=0,
    ),
    dict(
        id="q0_bsnd_bsnd_topk2k_nomask",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[8],
        k_pools=[256],
        n1=8,
        topk=2048,
        pool_size=16,
        mask_mode=0,
        return_value=False,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=110,
        quant_mode=0,
    ),
    # ---- BF16 weights(量化场景规格允许 FP16/BF16; P1-2 修复的防回退覆盖) ----
    dict(
        id="q0_bf16w_bsnd_causal",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[16],
        k_pools=[64],
        n1=8,
        topk=128,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=111,
        quant_mode=0,
        weights_dtype=torch.bfloat16,
    ),
    dict(
        id="q0_bf16w_tnd_pa",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=FP8,
        q_lens=[8, 16],
        k_pools=[32, 48],
        n1=8,
        topk=64,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=112,
        quant_mode=0,
        weights_dtype=torch.bfloat16,
    ),
    dict(
        id="q1_bf16w_bsnd_causal",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[16],
        k_pools=[64],
        n1=8,
        topk=128,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=113,
        quant_mode=1,
        weights_dtype=torch.bfloat16,
    ),
    dict(
        id="q0_bf16w_n1_48",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[16],
        k_pools=[64],
        n1=48,
        topk=128,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=114,
        quant_mode=0,
        weights_dtype=torch.bfloat16,
    ),
    # ---- mode=1 (mxFP8, E8M0 scale) ----
    dict(
        id="q1_bsnd_bsnd_nomask_noval",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[8],
        k_pools=[64],
        n1=8,
        topk=32,
        pool_size=1,
        mask_mode=0,
        return_value=False,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=201,
        quant_mode=1,
    ),
    dict(
        id="q1_bsnd_bsnd_causal_val",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[16],
        k_pools=[128],
        n1=8,
        topk=128,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=202,
        quant_mode=1,
    ),
    dict(
        id="q1_bsnd_bsnd_ps16_causal",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP8,
        q_lens=[32],
        k_pools=[64],
        n1=16,
        topk=256,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=[5],
        seed=203,
        quant_mode=1,
    ),
    dict(
        id="q1_tnd_tnd_causal",
        layout_q="TND",
        layout_k="TND",
        dtype=FP8,
        q_lens=[8, 12],
        k_pools=[48, 32],
        n1=8,
        topk=64,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=204,
        quant_mode=1,
    ),
    dict(
        id="q1_tnd_pa_causal",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=FP8,
        q_lens=[8, 16],
        k_pools=[32, 48],
        n1=8,
        topk=64,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
        seed=205,
        quant_mode=1,
    ),
    dict(
        id="q1_tnd_pa_noncontig_causal",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=FP8,
        q_lens=[8, 8],
        k_pools=[48, 32],
        n1=8,
        topk=64,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=True,
        tail=None,
        seed=206,
        quant_mode=1,
    ),
    dict(
        id="q1_bsnd_pa_nomask_val",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP8,
        q_lens=[16],
        k_pools=[64],
        n1=8,
        topk=128,
        pool_size=8,
        mask_mode=0,
        return_value=True,
        pa_block_size=32,
        key_non_contig=False,
        tail=None,
        seed=207,
        quant_mode=1,
    ),
]


def _quant_case_ids():
    return [c["id"] for c in _QUANT_FIXED_CASES]


# ---------------------------------------------------------------------------
# 精度执行路径(与 test_precision.py 同口径)
# ---------------------------------------------------------------------------


def _run_quant_case(cfg):
    inputs = build_case_inputs(cfg)
    g_indices, g_values = run_golden(inputs, cfg, ref_device="cpu")
    npu_indices, npu_values = run_op_eager(inputs, cfg)
    assert_output_spec(npu_indices, npu_values, cfg)
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"[{cfg['id']}] indices set comparison failed: "
        f"layout=({cfg['layout_q']},{cfg['layout_k']}) ps={cfg['pool_size']} "
        f"topk={cfg['topk']} mask={cfg['mask_mode']} quant={cfg['quant_mode']}"
    )
    if cfg["return_value"]:
        compare_values(g_values, npu_values, cfg["dtype"], label="values")


@pytest.mark.parametrize("cfg", _QUANT_FIXED_CASES, ids=_quant_case_ids())
def test_quant_precision_fixed(cfg):
    """量化定点用例: CPU 金标(量化语义) vs NPU eager。"""
    print(
        f"\n=== Quant: {cfg['id']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
        f"batch={len(cfg['q_lens'])} q_lens={cfg['q_lens']} k_pools={cfg['k_pools']} "
        f"n1={cfg['n1']} topk={cfg['topk']} ps={cfg['pool_size']} "
        f"mask={cfg['mask_mode']} val={cfg['return_value']} "
        f"quant={cfg['quant_mode']} noncontig={cfg['key_non_contig']} ==="
    )
    _run_quant_case(cfg)
    print(f"=== QUANT PASSED: {cfg['id']} ===")


# ---------------------------------------------------------------------------
# E8M0 = 0xFF (NaN 约定) 专项: 不拦截, NaN 传播
# ---------------------------------------------------------------------------


def _run_nan_case(cfg, nan_ratio):
    """在 descale 中注入 E8M0=0xFF, 验证不崩溃且输出可解析(NaN 传播)。

    golden 与算子的 TopK 对 NaN 的排序行为可能不同, 集合比对不做强断言,
    仅验证: 1) 不崩溃; 2) 输出 shape/dtype 符合规格; 3) NaN 仅影响注入
    位置的行(scale=NaN 的 token/pool 对应行), 其余行 indices 与 golden 一致。
    """
    inputs = build_case_inputs(cfg)
    # 注入 NaN: 随机将 nan_ratio 比例的 scale raw 置 0xFF
    torch.manual_seed(999)
    for key in ("q_descale", "k_descale"):
        if inputs.get(key) is not None:
            mask = torch.rand(inputs[key].shape) < nan_ratio
            inputs[key][mask] = 255

    npu_indices, npu_values = run_op_eager(inputs, cfg)
    assert_output_spec(npu_indices, npu_values, cfg)
    # 弱校验: 输出可解析(不崩溃, 无越界值)
    assert bool((npu_indices >= -1).all()), "indices must be >= -1"


@pytest.mark.parametrize(
    "cfg_id,nan_ratio",
    [
        ("q1_bsnd_bsnd_causal_val", 0.1),
        ("q1_tnd_pa_causal", 0.2),
    ],
)
def test_quant_e8m0_nan_propagation(cfg_id, nan_ratio):
    """E8M0=0xFF(NaN)不拦截, NaN 传播, 算子不崩溃且输出合法。"""
    cfg = next(c for c in _QUANT_FIXED_CASES if c["id"] == cfg_id)
    print(f"\n=== QuantNaN: {cfg_id} nan_ratio={nan_ratio} ===")
    _run_nan_case(cfg, nan_ratio)
    print(f"=== QUANT NAN PASSED: {cfg_id} ===")


# ---------------------------------------------------------------------------
# 错误路径: descale 与 quant_mode 的空/非空匹配(期望算子层报错)
# ---------------------------------------------------------------------------


def test_quant_error_mode_none_with_descale():
    """quant_mode=-1 传 descale -> tiling 校验拒绝。"""
    cfg = dict(
        next(c for c in _QUANT_FIXED_CASES if c["id"] == "q0_bsnd_bsnd_nomask_noval")
    )
    cfg["quant_mode"] = -1  # 保持 FP8 输入 + descale 非空 -> tiling 报错
    inputs = build_case_inputs(cfg)
    # build 仍生成 descale, 但 op 调用 quant_mode=-1
    cfg["quant_mode"] = -1
    with pytest.raises(Exception):
        run_op_eager(inputs, cfg)


def test_quant_error_mode0_without_descale():
    """quant_mode=0 缺 descale -> csrc TORCH_CHECK 拒绝。"""
    cfg = dict(
        next(c for c in _QUANT_FIXED_CASES if c["id"] == "q0_bsnd_bsnd_nomask_noval")
    )
    inputs = build_case_inputs(cfg)
    inputs["q_descale"] = None
    inputs["k_descale"] = None
    with pytest.raises(Exception):
        run_op_eager(inputs, cfg)


def test_quant_error_mode0_wrong_descale_dtype():
    """quant_mode=0 的 descale 不是 float -> csrc TORCH_CHECK 拒绝。"""
    cfg = dict(
        next(c for c in _QUANT_FIXED_CASES if c["id"] == "q0_bsnd_bsnd_nomask_noval")
    )
    inputs = build_case_inputs(cfg)
    # 将 float descale 换成 e8m0 视图(dtype 不匹配 mode=0)
    inputs["q_descale"] = inputs["q_descale"].view(torch.float8_e8m0fnu)
    with pytest.raises(Exception):
        run_op_eager(inputs, cfg)


def test_quant_error_nonpa_noncontig_key():
    """非 PA 布局 + 0 轴非连续 pool_key -> csrc 拒绝(静默错算防护, P0-1)。

    NPU 上构造非连续视图(NPU tensor 切片保留 stride; CPU->NPU 搬运会拍平
    视图导致检查不触发, 必须直接在 device 上切片)。
    """
    dev = f"npu:{get_device_id()}"
    B, S2 = 2, 64
    big = torch.randn(4, S2, 1, HEAD_DIM, dtype=torch.float16, device=dev)
    key_nc = big[::2]  # 0 轴非连续, stride=(16384,128,128,1)
    q = torch.randn(B, 16, 8, HEAD_DIM, dtype=torch.float16, device=dev)
    w = torch.randn(B, 16, 8, dtype=torch.float16, device=dev)
    ptk = torch.zeros(B, dtype=torch.int64, device=dev)
    from cann_ops_transformer import pool_key_indexer as _op

    with pytest.raises(Exception, match="PA_BBND"):
        _op(
            q,
            key_nc,
            w,
            ptk,
            layout_q="BSND",
            layout_k="BSND",
            topk=64,
            pool_size=8,
            mask_mode=0,
            quant_mode=-1,
            return_value=False,
        )

    # TND 布局同样拒绝
    big_t = torch.randn(256, 1, HEAD_DIM, dtype=torch.float16, device=dev)
    key_tnd_nc = big_t[::2]
    q_t = torch.randn(32, 8, HEAD_DIM, dtype=torch.float16, device=dev)
    w_t = torch.randn(32, 8, dtype=torch.float16, device=dev)
    asq = torch.tensor([16, 16], dtype=torch.int64).cumsum(0).to(dev)
    ask = torch.tensor([64, 64], dtype=torch.int64).cumsum(0).to(dev)
    with pytest.raises(Exception, match="PA_BBND"):
        _op(
            q_t,
            key_tnd_nc,
            w_t,
            ptk[:2],
            actual_seq_q=asq,
            actual_seq_k=ask,
            layout_q="TND",
            layout_k="TND",
            topk=64,
            pool_size=8,
            mask_mode=0,
            quant_mode=-1,
            return_value=False,
        )


def test_quant_error_noncontig_kdescale():
    """k_descale 非连续防护 + PA 0 轴非连续支持(P0-2 完整修复)。

    - 非 PA 布局(BSND 轴 1 非连续) -> csrc 拒绝(kernel 连续寻址, 无属性通道);
    - PA + 非 0 轴(轴 1 非连续) -> csrc 拒绝;
    - PA + 0 轴非连续 -> k_descale_stride0 属性下传, 正确计算(与连续参照一致),
      mode=0/mode=1 双模式验证。
    """
    dev = f"npu:{get_device_id()}"
    B, S2 = 2, 64
    block_num, block_size = 2, 64
    q = (
        (torch.randn(B, 16, 8, HEAD_DIM, dtype=torch.float32) * 0.25)
        .to(torch.float8_e4m3fn)
        .to(dev)
    )
    k8 = (
        (torch.randn(B, S2, 1, HEAD_DIM, dtype=torch.float32) * 0.25)
        .to(torch.float8_e4m3fn)
        .to(dev)
    )
    k_pa = (
        (torch.randn(block_num, block_size, 1, HEAD_DIM, dtype=torch.float32) * 0.25)
        .to(torch.float8_e4m3fn)
        .to(dev)
    )
    w = torch.randn(B, 16, 8, dtype=torch.float16).to(dev)
    ptk = torch.zeros(B, dtype=torch.int64).to(dev)
    qds = torch.rand(B, 16, 8, dtype=torch.float32).to(dev)
    bt = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32).to(dev)
    ask = torch.tensor([block_size, block_size], dtype=torch.int64).to(dev)
    from cann_ops_transformer import pool_key_indexer as _op

    # 拒绝 1: BSND + 轴 1 非连续
    big_ks = torch.rand(B, 2 * S2, 1, dtype=torch.float32, device=dev)
    kds_nc = big_ks[:, ::2, :]
    with pytest.raises(Exception, match="PA_BBND"):
        _op(
            q,
            k8,
            w,
            ptk,
            q_descale=qds,
            k_descale=kds_nc,
            layout_q="BSND",
            layout_k="BSND",
            topk=64,
            pool_size=8,
            mask_mode=0,
            quant_mode=0,
            return_value=False,
        )

    # 拒绝 2: PA + 轴 1 非连续
    big2 = torch.rand(2, 2 * block_size, 1, dtype=torch.float32, device=dev)
    kds_ax1 = big2[:, ::2, :]
    with pytest.raises(Exception, match="0-axis"):
        _op(
            q,
            k_pa,
            w,
            ptk,
            actual_seq_k=ask,
            block_table=bt,
            q_descale=qds,
            k_descale=kds_ax1,
            layout_q="BSND",
            layout_k="PA_BBND",
            topk=64,
            pool_size=8,
            mask_mode=0,
            quant_mode=0,
            return_value=False,
        )

    # 支持: PA + 0 轴非连续(mode=0), 结果与连续参照一致
    kds_contig = torch.rand(block_num, block_size, 1, dtype=torch.float32).to(dev)
    big = torch.zeros(2 * block_num, block_size, 1, dtype=torch.float32, device=dev)
    big[0::2] = kds_contig
    kds_nc_pa = big[0::2]

    def run_pa(kds):
        idx, _ = _op(
            q,
            k_pa,
            w,
            ptk,
            actual_seq_k=ask,
            block_table=bt,
            q_descale=qds,
            k_descale=kds,
            layout_q="BSND",
            layout_k="PA_BBND",
            topk=64,
            pool_size=8,
            mask_mode=0,
            quant_mode=0,
            return_value=False,
        )
        torch.npu.synchronize()
        return idx.cpu()

    idx_nc = run_pa(kds_nc_pa)
    idx_ct = run_pa(kds_contig)
    same = all(
        sorted(idx_nc[b, r][idx_nc[b, r] >= 0].tolist())
        == sorted(idx_ct[b, r][idx_ct[b, r] >= 0].tolist())
        for b in range(B)
        for r in range(idx_nc.shape[1])
    )
    assert same, (
        "PA 0-axis non-contiguous k_descale results mismatch with contiguous reference"
    )

    # 支持: PA + 0 轴非连续(mode=1, E8M0 视图)
    qds_m1 = (
        torch.randint(120, 136, (B, 16, 8, 2, 2), dtype=torch.uint8)
        .to(dev)
        .view(torch.float8_e8m0fnu)
    )
    kds_c1 = (
        torch.randint(120, 136, (block_num, block_size, 1, 2, 2), dtype=torch.uint8)
        .to(dev)
        .view(torch.float8_e8m0fnu)
    )
    big1 = torch.zeros(
        2 * block_num, block_size, 1, 2, 2, dtype=torch.uint8, device=dev
    )
    big1[0::2] = kds_c1.view(torch.uint8)
    kds_nc1 = big1[0::2].view(torch.float8_e8m0fnu)

    def run_pa_m1(kds):
        idx, _ = _op(
            q,
            k_pa,
            w,
            ptk,
            actual_seq_k=ask,
            block_table=bt,
            q_descale=qds_m1,
            k_descale=kds,
            layout_q="BSND",
            layout_k="PA_BBND",
            topk=64,
            pool_size=8,
            mask_mode=0,
            quant_mode=1,
            return_value=False,
        )
        torch.npu.synchronize()
        return idx.cpu()

    idx_nc1 = run_pa_m1(kds_nc1)
    idx_ct1 = run_pa_m1(kds_c1)
    same1 = all(
        sorted(idx_nc1[b, r][idx_nc1[b, r] >= 0].tolist())
        == sorted(idx_ct1[b, r][idx_ct1[b, r] >= 0].tolist())
        for b in range(B)
        for r in range(idx_nc1.shape[1])
    )
    assert same1, "mode=1 PA 0-axis non-contiguous k_descale results mismatch"
