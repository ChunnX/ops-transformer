from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
import tomllib

import cannbotdsl
import pytest
from cannbotdsl.package.dispatch import synthesize_aot_dispatch
from cannbotdsl.package.stage_source import direct_package_spec
from cannbotdsl.package.variant_manifest import flatten_variant_manifest


_OPERATOR_DIR = Path(__file__).resolve().parents[1] / "attention" / "flash_kda"


def _load_operator():
    source = _OPERATOR_DIR / "flash_kda.py"
    spec = importlib.util.spec_from_file_location("ops_flash_kda_auto", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flash_kda_source_uses_entry_only_auto_dispatch():
    module = _load_operator()
    synthesis = synthesize_aot_dispatch(
        module.flash_kda_l0_entry, target_soc="ascend950pr"
    )

    assert synthesis.entry["schema_version"] == 5
    assert synthesis.entry["binding_plan"]["schema_version"] == 5
    assert "variant_manifest" in synthesis.entry
    assert "dispatch_plan" not in synthesis.entry
    assert synthesis.variant_manifest is synthesis.entry["variant_manifest"]
    flattened = flatten_variant_manifest(
        synthesis.variant_manifest,
        require_content_keys=False,
        public_contract_digest=synthesis.entry["support_digest"],
        operator_identity="FlashKda",
        target_soc="ascend950pr",
    )
    assert len(flattened) == 6

    assert synthesis.entry["public_parameters"] == [
        "K",
        "V",
        "Q",
        "beta",
        "g",
        "A_log",
        "dt_bias",
        "st_in",
        "scale",
        "lower_bound",
        "o",
        "st_out",
    ]
    assert synthesis.entry["output_parameters"] == ["o", "st_out"]
    assert "internal_parameters" not in synthesis.entry
    assert synthesis.entry["generated_internal_parameters"] == [
        "dv_base",
        "group",
    ]
    assert len(synthesis.variants) == 6
    assert {
        (artifact["variant_key"]["group"], artifact["variant_key"]["dv_base"])
        for artifact in synthesis.variant_manifest["artifacts"]
    } == {
        (128, 16),
        (256, 16),
        (64, 64),
        (8, 128),
        (4, 128),
        (2, 128),
    }


def test_flash_kda_support_preserves_named_degenerate_batch_provenance():
    module = _load_operator()

    for case, expected_batch in zip(
        module.FLASH_KDA_SUPPORT.cases,
        (1, 1, 4, 1, 2, 4),
        strict=True,
    ):
        batch_dim = case.inputs["K"].shape[0]
        assert isinstance(batch_dim, cannbotdsl.Dim)
        assert (batch_dim.name, batch_dim.min, batch_dim.max) == (
            "B",
            expected_batch,
            expected_batch,
        )
        for name in ("K", "V", "Q", "beta", "g", "st_in"):
            assert case.inputs[name].shape[0] is batch_dim
        for name in ("o", "st_out"):
            assert case.outputs[name].shape[0] is batch_dim

        # Only the logical batch carries fixed-dimension provenance.  Literal
        # singleton axes, terminal strides, and constants must stay static.
        assert case.inputs["beta"].shape[-1] == 1
        assert type(case.inputs["beta"].shape[-1]) is int
        assert type(case.inputs["beta"].stride[-1]) is int


def test_flash_kda_direct_entry_captures_one_class_kernel_physical_abi():
    module = _load_operator()
    case = module.FLASH_KDA_SUPPORT.cases[0]
    plan = module.flash_kda_l0_entry._build_plan_for(
        case,
        {"B": 1, "S": 8192},
    )

    assert plan.provider_call["family"] == "flash_kda_kernel___call__"
    assert list(plan.provider_call["bindings"]) == [
        "gm_K",
        "gm_V",
        "gm_Q",
        "gm_beta_brc",
        "gm_g",
        "gm_qk_exchange",
        "gm_A_log",
        "gm_dt_bias",
        "gm_k_restored",
        "gm_gamma",
        "gm_q_decayed",
        "gm_w",
        "gm_Mqk",
        "gm_U",
        "gm_O",
        "gm_state_in",
        "gm_state_out",
        "gm_identity16",
        "bn_total",
        "seq_len",
        "scale_value",
        "lower_bound",
    ]
    assert plan.provider_call["specializations"] == {}
    assert list(plan.provider_call["constructor_specializations"]) == [
        "head_dim",
        "group",
        "dv_base",
    ]
    assert plan.provider_call["constructor_specializations"]["head_dim"] == {
        "source": "literal",
        "value": module.SUPPORTED_HEAD_DIM,
    }
    assert plan.provider_call["block_dim"] == {
        "source": "literal",
        "value": module.ASCEND950PR_PACKAGE_BLOCK_NUM,
    }

    synthesis = synthesize_aot_dispatch(
        module.flash_kda_l0_entry, target_soc="ascend950pr"
    )
    assert all(len(variant.args) == 26 for variant in synthesis.variants)
    assert {
        variant.args[-4:] for variant in synthesis.variants
    } == {
        (module.SUPPORTED_HEAD_DIM, 128, 16, 28),
        (module.SUPPORTED_HEAD_DIM, 256, 16, 28),
        (module.SUPPORTED_HEAD_DIM, 64, 64, 28),
        (module.SUPPORTED_HEAD_DIM, 8, 128, 28),
        (module.SUPPORTED_HEAD_DIM, 4, 128, 28),
        (module.SUPPORTED_HEAD_DIM, 2, 128, 28),
    }


def test_flash_kda_module_has_no_duplicate_launch_or_override_authority():
    module = _load_operator()
    source = (_OPERATOR_DIR / "flash_kda.py").read_text(encoding="utf-8")

    for name in (
        "flash_kda_provider",
        "_flash_kda_native",
        "FlashKDA",
        "flash_kda_variants",
    ):
        assert not hasattr(module, name)
    assert "KDA_GROUP" not in source
    assert "KDA_DV_BASE" not in source


class _FakeDevice:
    type = "npu"
    index = 0


class _FakeTensor:
    def __init__(self, shape, dtype, device):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def unsqueeze(self, axis):
        if axis < 0:
            axis += len(self.shape) + 1
        shape = list(self.shape)
        shape.insert(axis, 1)
        return type(self)(shape, self.dtype, self.device)

    def transpose(self, axis0, axis1):
        shape = list(self.shape)
        shape[axis0], shape[axis1] = shape[axis1], shape[axis0]
        return type(self)(shape, self.dtype, self.device)


def _fake_native_inputs(
    module,
    *,
    layout="BNSD",
    batch=1,
    sequence=8192,
    key_heads=3,
    value_heads=3,
    scale=0.125,
    lower_bound=-1.0,
):
    device = _FakeDevice()
    dim = module.SUPPORTED_HEAD_DIM
    if layout == "BNSD":
        qk_shape = (batch, key_heads, sequence, dim)
        value_shape = (batch, value_heads, sequence, dim)
        beta_shape = (batch, value_heads, sequence)
    else:
        qk_shape = (batch, sequence, key_heads, dim)
        value_shape = (batch, sequence, value_heads, dim)
        beta_shape = (batch, sequence, value_heads)
    low = module.LOW_DTYPE
    high = module.HIGH_DTYPE
    return {
        "q": _FakeTensor(qk_shape, low, device),
        "k": _FakeTensor(qk_shape, low, device),
        "v": _FakeTensor(value_shape, low, device),
        "g": _FakeTensor(value_shape, low, device),
        "beta": _FakeTensor(beta_shape, low, device),
        "scale": scale,
        "initial_state": _FakeTensor(
            (batch, value_heads, dim, dim), high, device
        ),
        "A_log": _FakeTensor((value_heads,), high, device),
        "dt_bias": _FakeTensor((value_heads, dim), high, device),
        "lower_bound": lower_bound,
        "layout_qkv": layout,
    }


def test_flash_kda_native_compiles_the_canonical_entry_with_complete_samples(
    monkeypatch,
):
    module = _load_operator()
    native_entry = pytest.importorskip(
        "cannbotdsl.core.compiler.native_entry"
    )
    observed = {}

    def fake_compile(
        entry,
        physical_args,
        *,
        sample_shapes,
        sample_scalars,
    ):
        observed["entry"] = entry
        observed["physical_args"] = physical_args
        observed["sample_shapes"] = sample_shapes
        observed["sample_scalars"] = sample_scalars

        def invoke(*runtime_args):
            observed["runtime_args"] = runtime_args

        return invoke

    def fake_empty(*shape, dtype, device):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _FakeTensor(shape, dtype, device)

    class LegacyNativePath:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("native path used duplicate FlashKDA authority")

    inputs = _fake_native_inputs(module)
    device = inputs["q"].device
    workspace = SimpleNamespace(
        K_restored=_FakeTensor((1,), module.LOW_DTYPE, device),
        gamma_C=_FakeTensor((1,), module.HIGH_DTYPE, device),
        Q_decayed=_FakeTensor((1,), module.LOW_DTYPE, device),
        W=_FakeTensor((1,), module.LOW_DTYPE, device),
        Mqk=_FakeTensor((1,), module.LOW_DTYPE, device),
        U_pre=_FakeTensor((1,), module.LOW_DTYPE, device),
        identity16=_FakeTensor((16, 16), module.torch.float16, device),
    )
    execution = SimpleNamespace(
        workspace=workspace,
        qk_exchange=_FakeTensor((1,), module.HIGH_DTYPE, device),
        launch_lock=threading.Lock(),
    )
    monkeypatch.setattr(native_entry, "compile_native_entry_variant", fake_compile)
    monkeypatch.setattr(module, "FlashKDA", LegacyNativePath, raising=False)
    monkeypatch.setattr(module, "_device_block_num", lambda ref=None: 28)
    monkeypatch.setattr(module.torch, "empty", fake_empty)
    monkeypatch.setattr(
        module._EXECUTION_CACHE,
        "get_or_create",
        lambda **kwargs: execution,
    )
    monkeypatch.setenv("KDA_GROUP", "1")
    monkeypatch.setenv("KDA_DV_BASE", "1")
    module._DYNAMIC_KERNEL_CACHE.clear()

    out, state = module.flash_kda(**inputs)

    assert observed["entry"] is module.flash_kda_l0_entry
    assert len(observed["physical_args"]) == 26
    assert observed["physical_args"][-4:] == (
        module.SUPPORTED_HEAD_DIM,
        128,
        16,
        module.ASCEND950PR_PACKAGE_BLOCK_NUM,
    )
    assert observed["sample_shapes"] == {
        "K": (1, 3, 8192, 128),
        "V": (1, 3, 8192, 128),
        "Q": (1, 3, 8192, 128),
        "beta": (1, 3, 8192, 1),
        "g": (1, 3, 8192, 128),
        "A_log": (3,),
        "dt_bias": (3, 128),
        "st_in": (1, 3, 128, 128),
        "o": (1, 3, 8192, 128),
        "st_out": (1, 3, 128, 128),
    }
    assert observed["sample_scalars"] == {
        "scale": 0.125,
        "lower_bound": -1.0,
    }
    assert len(observed["runtime_args"]) == 22
    assert out.shape == (1, 3, 8192, 128)
    assert state.shape == (1, 3, 128, 128)


@pytest.mark.parametrize(
    ("warm", "unsupported"),
    [
        (
            {"batch": 1, "sequence": 32768, "value_heads": 3},
            {"batch": 1, "sequence": 16384, "value_heads": 3},
        ),
        (
            {"batch": 4, "sequence": 1024, "value_heads": 96},
            {"batch": 3, "sequence": 1024, "value_heads": 96},
        ),
        (
            {
                "batch": 1,
                "sequence": 32768,
                "value_heads": 3,
                "scale": 0.125,
            },
            {
                "batch": 1,
                "sequence": 32768,
                "value_heads": 3,
                "scale": float("nan"),
            },
        ),
    ],
)
def test_flash_kda_warm_cache_revalidates_canonical_public_support(
    monkeypatch,
    warm,
    unsupported,
):
    module = _load_operator()
    native_entry = pytest.importorskip(
        "cannbotdsl.core.compiler.native_entry"
    )
    compile_samples = []
    runtime_samples = []

    def fake_compile(
        entry,
        physical_args,
        *,
        sample_shapes,
        sample_scalars,
    ):
        del entry, physical_args
        compile_samples.append((sample_shapes, sample_scalars))

        def invoke(*runtime_args):
            runtime_samples.append(
                (
                    runtime_args[0].shape[0],
                    runtime_args[0].shape[2],
                    runtime_args[20],
                    runtime_args[21],
                )
            )

        return invoke

    def fake_empty(*shape, dtype, device):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _FakeTensor(shape, dtype, device)

    device = _FakeDevice()
    workspace = SimpleNamespace(
        K_restored=_FakeTensor((1,), module.LOW_DTYPE, device),
        gamma_C=_FakeTensor((1,), module.HIGH_DTYPE, device),
        Q_decayed=_FakeTensor((1,), module.LOW_DTYPE, device),
        W=_FakeTensor((1,), module.LOW_DTYPE, device),
        Mqk=_FakeTensor((1,), module.LOW_DTYPE, device),
        U_pre=_FakeTensor((1,), module.LOW_DTYPE, device),
        identity16=_FakeTensor((16, 16), module.torch.float16, device),
    )
    execution = SimpleNamespace(
        workspace=workspace,
        qk_exchange=_FakeTensor((1,), module.HIGH_DTYPE, device),
        launch_lock=threading.Lock(),
    )
    monkeypatch.setattr(native_entry, "compile_native_entry_variant", fake_compile)
    monkeypatch.setattr(module, "_device_block_num", lambda ref=None: 28)
    monkeypatch.setattr(module.torch, "empty", fake_empty)
    monkeypatch.setattr(
        module._EXECUTION_CACHE,
        "get_or_create",
        lambda **kwargs: execution,
    )
    module._DYNAMIC_KERNEL_CACHE.clear()

    warm["key_heads"] = warm["value_heads"]
    unsupported["key_heads"] = unsupported["value_heads"]
    module.flash_kda(**_fake_native_inputs(module, **warm))
    assert len(compile_samples) == 1
    assert len(runtime_samples) == 1

    with pytest.raises(
        (TypeError, ValueError),
        match="supported public point|unsupported type",
    ):
        module.flash_kda(**_fake_native_inputs(module, **unsupported))

    assert len(compile_samples) == 1
    assert len(runtime_samples) == 1


@pytest.mark.parametrize(
    ("layout", "key_heads", "value_heads", "message"),
    [
        ("BSND", 3, 3, "canonical.*BNSD"),
        ("BNSD", 1, 3, "canonical.*Nk == Nv"),
    ],
)
def test_flash_kda_native_rejects_layout_or_gqa_outside_canonical_support(
    monkeypatch,
    layout,
    key_heads,
    value_heads,
    message,
):
    module = _load_operator()

    class LegacyNativePath:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("unsupported point reached legacy compiler")

    monkeypatch.setattr(module, "FlashKDA", LegacyNativePath, raising=False)
    monkeypatch.setattr(module, "_device_block_num", lambda ref=None: 28)
    module._DYNAMIC_KERNEL_CACHE.clear()
    with pytest.raises(ValueError, match=message):
        module.flash_kda(
            **_fake_native_inputs(
                module,
                layout=layout,
                key_heads=key_heads,
                value_heads=value_heads,
            )
        )


def test_direct_source_entry_has_no_manual_function_or_variant_provider():
    source = _OPERATOR_DIR / "flash_kda.py"
    path, text = direct_package_spec(
        source=source,
        entry="FlashKda",
        operator="ops-transformer::attention/flash_kda",
        package_name="ops-transformer-flash-kda",
        version="0.1.0",
        vendor="ops-transformer",
        target_soc="ascend950pr",
        npu_arch="dav-3510",
    )
    payload = tomllib.loads(text)

    assert path == source
    assert payload["operators"] == [
        {
            "operator": "ops-transformer::attention/flash_kda",
            "entry": "flash_kda:FlashKda",
            "entry_public_name": True,
        }
    ]
    assert "function" not in payload["operators"][0]
    assert "provider" not in payload["operators"][0]
