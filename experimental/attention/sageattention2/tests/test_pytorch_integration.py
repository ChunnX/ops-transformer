# SageAttention2 PyTorch 集成深度测试（M2c-N1/N2：schema 契约 / Meta·FakeTensor / torch.compile / 重复调用逐位一致）
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from sageattention2 import sageattn, sageattn_qk_int8_pv_fp8_asc  # noqa: E402

OP = torch.ops.sageattention2.forward


def _create_tensors(b=1, hq=2, hk=2, sq=64, sk=64, d=64, dtype=torch.float16):
    q = torch.randn(b, hq, sq, d, dtype=dtype)
    k = torch.randn(b, hk, sk, d, dtype=dtype)
    v = torch.randn(b, hk, sk, d, dtype=dtype)
    return q, k, v


def test_schema_contract():
    """N1a: dispatcher schema 契约——算子名、kwonly 参数序与默认值、返回 (Tensor, Tensor?)。"""
    schema = OP.default._schema
    assert str(schema).startswith("sageattention2::forward")
    assert schema.return_type == torch._C.ArgumentType.Tuple if hasattr(schema, "return_type") else True
    # 位置参数：q/k/v
    pos_names = [a.name for a in schema.arguments if not a.kwarg_only]
    assert pos_names == ["q", "k", "v"]
    # kwonly 参数序与默认值逐项对齐任务书 §2.3（与 _asc 签名同构）
    kw = {a.name: a.default_value for a in schema.arguments if a.kwarg_only}
    assert kw == {
        "tensor_layout": "HND",
        "is_causal": False,
        "qk_quant_gran": "per_thread",
        "sm_scale": None,
        "pv_accum_dtype": "fp32+fp16",
        "smooth_k": True,
        "smooth_v": False,
        "return_lse": False,
    }


def test_meta_and_fake_tensor():
    """N1b: Meta 设备与 FakeTensorMode 走 register_fake 实现——校验与 shape/dtype 推导生效。"""
    q, k, v = _create_tensors()

    # Meta 设备直调：shape/dtype 推导（含 LSE）
    out_m, lse_m = OP(
        q.to("meta"), k.to("meta"), v.to("meta"), return_lse=True
    )
    assert out_m.shape == q.shape and out_m.dtype == q.dtype and out_m.device.type == "meta"
    assert lse_m.shape == (1, 2, 64) and lse_m.dtype == torch.float32

    # return_lse=False 时 lse 为 None
    out_m2, lse_m2 = OP(q.to("meta"), k.to("meta"), v.to("meta"))
    assert lse_m2 is None and out_m2.shape == q.shape

    # NHD 布局下 LSE 仍为 [B, Hq, Sq]
    q_n = q.permute(0, 2, 1, 3)
    k_n = k.permute(0, 2, 1, 3)
    v_n = v.permute(0, 2, 1, 3)
    _, lse_n = OP(q_n.to("meta"), k_n.to("meta"), v_n.to("meta"), tensor_layout="NHD", return_lse=True)
    assert lse_n.shape == (1, 2, 64)

    # FakeTensorMode 同构（含校验：非法参数在 fake 域即拦截）
    with FakeTensorMode():
        fq = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        fk = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        fv = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        out_f, lse_f = OP(fq, fk, fv, return_lse=True)
        assert out_f.shape == fq.shape and out_f.dtype == fq.dtype
        assert lse_f.shape == (1, 2, 64) and lse_f.dtype == torch.float32
        with pytest.raises(ValueError, match="tensor_layout"):
            OP(fq, fk, fv, tensor_layout="XYZ")


def test_torch_compile_eager_fullgraph():
    """N1c: torch.compile(backend="eager", fullgraph=True) 全图捕获——算子级与入口级双冒烟。"""
    q, k, v = _create_tensors()

    # 算子级：自定义 op 单节点图可全图捕获
    def _op_fn(q_, k_, v_):
        return OP(q_, k_, v_, return_lse=True)

    compiled_op = torch.compile(_op_fn, backend="eager", fullgraph=True)
    out_c, lse_c = compiled_op(q, k, v)
    out_e, lse_e = _op_fn(q, k, v)
    assert torch.equal(out_c, out_e) and torch.equal(lse_c, lse_e)

    # 入口级：sageattn python 包装层无 graph break 全图捕获
    def _entry_fn(q_, k_, v_):
        return sageattn(q_, k_, v_, return_lse=True)

    compiled_entry = torch.compile(_entry_fn, backend="eager", fullgraph=True)
    out_w, lse_w = compiled_entry(q, k, v)
    out_e2, lse_e2 = sageattn(q, k, v, return_lse=True)
    assert torch.equal(out_w, out_e2) and torch.equal(lse_w, lse_e2)


@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("accum", ["fp32", "fp32+fp16"])
def test_repeat_call_bitwise_determinism(gran, accum):
    """N2: 重复调用逐位一致（D-R8 单写者确定性树 / 无 atomic 的设计契约钉子）。"""
    torch.manual_seed(99)
    q, k, v = _create_tensors(hq=4, sq=128, sk=128)
    kwargs = {"qk_quant_gran": gran, "pv_accum_dtype": accum, "return_lse": True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fp32+fp16 档 smooth_v=False 不触发；防第三方告警干扰
        out1, lse1 = sageattn_qk_int8_pv_fp8_asc(q, k, v, **kwargs)
        out2, lse2 = sageattn_qk_int8_pv_fp8_asc(q, k, v, **kwargs)
    assert torch.equal(out1, out2)
    assert torch.equal(lse1, lse2)
