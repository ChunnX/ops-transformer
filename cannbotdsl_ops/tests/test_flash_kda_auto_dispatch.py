from __future__ import annotations

from copy import deepcopy
import importlib.util
import inspect
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

_CANONICAL_RUNTIME_PARAMETERS = (
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
    "gm_state_in",
    "gm_identity16",
    "gm_O",
    "gm_state_out",
    "bn_total",
    "seq_len",
    "scale_value",
    "lower_bound",
)


def _declare_canonical_runtime_signature(callable_):
    callable_.__signature__ = inspect.Signature(
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in _CANONICAL_RUNTIME_PARAMETERS
    )
    return callable_


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


def _validated_native_samples(
    module,
    native_entry,
    *,
    batch,
    heads,
    sequence,
):
    dim = module.SUPPORTED_HEAD_DIM
    return native_entry.validate_native_entry_samples(
        module.flash_kda_l0_entry,
        sample_shapes={
            "K": (batch, heads, sequence, dim),
            "V": (batch, heads, sequence, dim),
            "Q": (batch, heads, sequence, dim),
            "beta": (batch, heads, sequence, 1),
            "g": (batch, heads, sequence, dim),
            "A_log": (heads,),
            "dt_bias": (heads, dim),
            "st_in": (batch, heads, dim, dim),
            "o": (batch, heads, sequence, dim),
            "st_out": (batch, heads, dim, dim),
        },
        sample_scalars={"scale": 0.125, "lower_bound": -1.0},
    )


def _fake_native_execution(module):
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
    return SimpleNamespace(
        workspace=workspace,
        qk_exchange=_FakeTensor((1,), module.HIGH_DTYPE, device),
        launch_lock=threading.Lock(),
    )


def _patch_fake_native_execution(monkeypatch, module, execution):
    def fake_empty(*shape, dtype, device):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _FakeTensor(shape, dtype, device)

    monkeypatch.setattr(module, "_device_block_num", lambda ref=None: 28)
    monkeypatch.setattr(module.torch, "empty", fake_empty)
    monkeypatch.setattr(
        module._EXECUTION_CACHE,
        "get_or_create",
        lambda **kwargs: execution,
    )


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

        return _declare_canonical_runtime_signature(invoke)

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
    beta_arg = observed["runtime_args"][3]
    assert observed["runtime_args"] == (
        inputs["k"],
        inputs["v"],
        inputs["q"],
        beta_arg,
        inputs["g"],
        execution.qk_exchange,
        inputs["A_log"],
        inputs["dt_bias"],
        workspace.K_restored,
        workspace.gamma_C,
        workspace.Q_decayed,
        workspace.W,
        workspace.Mqk,
        workspace.U_pre,
        inputs["initial_state"],
        workspace.identity16,
        out,
        state,
        3,
        8192,
        0.125,
        -1.0,
    )
    assert out.shape == (1, 3, 8192, 128)
    assert state.shape == (1, 3, 128, 128)


def test_flash_kda_native_consumes_generated_physical_args_for_all_regions(
    monkeypatch,
):
    module = _load_operator()
    native_entry = pytest.importorskip(
        "cannbotdsl.core.compiler.native_entry"
    )
    synthesis = synthesize_aot_dispatch(
        module.flash_kda_l0_entry,
        target_soc="ascend950pr",
    )
    artifacts = {
        item["physical_key"]: item
        for item in synthesis.variant_manifest["artifacts"]
    }
    generated = {
        (
            artifacts[item.physical_key]["variant_key"]["group"],
            artifacts[item.physical_key]["variant_key"]["dv_base"],
        ): item.variant.args
        for item in synthesis.compile_variants
    }
    observed = []

    def fake_compile(
        entry,
        physical_args,
        *,
        sample_shapes,
        sample_scalars,
    ):
        record = {
            "entry": entry,
            "physical_args": physical_args,
            "sample_shapes": sample_shapes,
            "sample_scalars": sample_scalars,
        }
        observed.append(record)

        def invoke(*runtime_args):
            record["runtime_args"] = runtime_args

        return _declare_canonical_runtime_signature(invoke)

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

    regions = (
        (1, 3, 8192, 128, 16),
        (1, 3, 32768, 256, 16),
        (4, 3, 8192, 64, 64),
        (1, 96, 1024, 8, 128),
        (2, 96, 10240, 4, 128),
        (4, 96, 1024, 2, 128),
    )
    for batch, heads, sequence, group, dv_base in regions:
        inputs = _fake_native_inputs(
            module,
            batch=batch,
            sequence=sequence,
            key_heads=heads,
            value_heads=heads,
        )
        out, state = module.flash_kda(**inputs)
        record = observed[-1]

        assert record["entry"] is module.flash_kda_l0_entry
        assert record["physical_args"] == generated[(group, dv_base)]
        state_in, identity, output, state_out = record["physical_args"][14:18]
        assert (
            state_in.dtype,
            identity.dtype,
            output.dtype,
            state_out.dtype,
        ) == (
            module.dtypes.float32,
            module.dtypes.float16,
            module.dtypes.bfloat16,
            module.dtypes.float32,
        )
        assert state_in.shape[1:] == (heads, 128, 128)
        assert identity.shape == (16, 16)
        assert output.shape[1] == heads and output.shape[-1] == 128
        assert state_out.shape[1:] == (heads, 128, 128)
        runtime_args = record["runtime_args"]
        assert len(runtime_args) == 22
        assert runtime_args == (
            inputs["k"],
            inputs["v"],
            inputs["q"],
            runtime_args[3],
            inputs["g"],
            execution.qk_exchange,
            inputs["A_log"],
            inputs["dt_bias"],
            workspace.K_restored,
            workspace.gamma_C,
            workspace.Q_decayed,
            workspace.W,
            workspace.Mqk,
            workspace.U_pre,
            inputs["initial_state"],
            workspace.identity16,
            out,
            state,
            batch * heads,
            sequence,
            0.125,
            -1.0,
        )

    assert len(observed) == 6


def test_generated_native_args_select_validated_case_before_repeated_variant_key():
    module = _load_operator()
    native_entry = pytest.importorskip(
        "cannbotdsl.core.compiler.native_entry"
    )
    synthesis = synthesize_aot_dispatch(
        module.flash_kda_l0_entry,
        target_soc="ascend950pr",
    )
    manifest = deepcopy(synthesis.variant_manifest)
    target_key = {"group": 128, "dv_base": 16}
    target_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["variant_key"] == target_key
    )
    disjoint_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["variant_key"] == {"group": 256, "dv_base": 16}
    )
    disjoint_artifact["variant_key"] = deepcopy(target_key)
    module._NATIVE_DISPATCH_SYNTHESIS = SimpleNamespace(
        variant_manifest=manifest,
        compile_variants=synthesis.compile_variants,
    )
    samples = _validated_native_samples(
        module,
        native_entry,
        batch=1,
        heads=3,
        sequence=8192,
    )
    expected = next(
        item.variant.args
        for item in synthesis.compile_variants
        if item.physical_key == target_artifact["physical_key"]
    )

    selected_physical_key, actual = module._generated_native_physical_args(
        samples,
        group=128,
        dv_base=16,
    )

    assert selected_physical_key == target_artifact["physical_key"]
    assert actual == expected


def test_generated_native_args_reject_manifest_compile_physical_key_drift():
    module = _load_operator()
    native_entry = pytest.importorskip(
        "cannbotdsl.core.compiler.native_entry"
    )
    synthesis = synthesize_aot_dispatch(
        module.flash_kda_l0_entry,
        target_soc="ascend950pr",
    )
    target_key = {"group": 128, "dv_base": 16}
    target_physical_key = next(
        artifact["physical_key"]
        for artifact in synthesis.variant_manifest["artifacts"]
        if artifact["variant_key"] == target_key
    )
    unrelated = next(
        item
        for item in synthesis.compile_variants
        if item.physical_key != target_physical_key
    )
    module._NATIVE_DISPATCH_SYNTHESIS = SimpleNamespace(
        variant_manifest=deepcopy(synthesis.variant_manifest),
        compile_variants=tuple(
            item
            for item in synthesis.compile_variants
            if item.physical_key != unrelated.physical_key
        ),
    )
    samples = _validated_native_samples(
        module,
        native_entry,
        batch=1,
        heads=3,
        sequence=8192,
    )

    with pytest.raises(RuntimeError, match="physical key"):
        module._generated_native_physical_args(
            samples,
            group=128,
            dv_base=16,
        )


@pytest.mark.parametrize(
    "sequence_order",
    (
        (8192, 32768, 8192),
        (32768, 8192),
    ),
)
def test_flash_kda_warm_cache_is_partitioned_by_selected_physical_key(
    monkeypatch,
    sequence_order,
):
    module = _load_operator()
    native_entry = pytest.importorskip(
        "cannbotdsl.core.compiler.native_entry"
    )
    synthesis = synthesize_aot_dispatch(
        module.flash_kda_l0_entry,
        target_soc="ascend950pr",
    )
    manifest = deepcopy(synthesis.variant_manifest)
    shared_variant_key = {"group": 128, "dv_base": 16}
    artifacts_by_sequence = {}
    for artifact in manifest["artifacts"]:
        slot = artifact["cases"][0]["predicate"]["tensors"]["K"]["shape"][2]
        if slot == {"min": 8192, "max": 8192, "multiple_of": 64}:
            artifacts_by_sequence[8192] = artifact
        elif slot == {"min": 32768, "max": 131072, "multiple_of": 64}:
            artifacts_by_sequence[32768] = artifact
    assert set(artifacts_by_sequence) == {8192, 32768}
    for artifact in artifacts_by_sequence.values():
        artifact["variant_key"] = deepcopy(shared_variant_key)
    module._NATIVE_DISPATCH_SYNTHESIS = SimpleNamespace(
        variant_manifest=manifest,
        compile_variants=synthesis.compile_variants,
    )
    expected_args = {
        sequence: next(
            item.variant.args
            for item in synthesis.compile_variants
            if item.physical_key == artifact["physical_key"]
        )
        for sequence, artifact in artifacts_by_sequence.items()
    }
    execution = _fake_native_execution(module)
    _patch_fake_native_execution(monkeypatch, module, execution)
    monkeypatch.setattr(
        module,
        "get_group_config",
        lambda heads, sequence: 128,
    )
    module._DYNAMIC_KERNEL_CACHE.clear()
    compiled_args = []
    invocations = []

    def fake_compile(
        entry,
        physical_args,
        *,
        sample_shapes,
        sample_scalars,
    ):
        del entry, sample_shapes, sample_scalars
        compile_index = len(compiled_args)
        compiled_args.append(physical_args)

        def invoke(*runtime_args):
            del runtime_args
            invocations.append(compile_index)

        return _declare_canonical_runtime_signature(invoke)

    monkeypatch.setattr(native_entry, "compile_native_entry_variant", fake_compile)
    for sequence in sequence_order:
        module.flash_kda(
            **_fake_native_inputs(module, sequence=sequence)
        )

    first_occurrences = tuple(dict.fromkeys(sequence_order))
    assert compiled_args == [expected_args[item] for item in first_occurrences]
    compile_indices = {
        sequence: index for index, sequence in enumerate(first_occurrences)
    }
    assert invocations == [compile_indices[item] for item in sequence_order]


def test_flash_kda_native_cache_serializes_cold_compile_and_recovers_failure(
    monkeypatch,
):
    module = _load_operator()
    native_entry = pytest.importorskip(
        "cannbotdsl.core.compiler.native_entry"
    )
    dispatch = pytest.importorskip("cannbotdsl.package.dispatch")
    synthesis = synthesize_aot_dispatch(
        module.flash_kda_l0_entry,
        target_soc="ascend950pr",
    )
    execution = _fake_native_execution(module)
    _patch_fake_native_execution(monkeypatch, module, execution)
    module._DYNAMIC_KERNEL_CACHE.clear()
    module._NATIVE_DISPATCH_SYNTHESIS = None

    real_validate = native_entry.validate_native_entry_samples
    validation_barrier = threading.Barrier(2)

    def synchronized_validate(*args, **kwargs):
        result = real_validate(*args, **kwargs)
        validation_barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(
        native_entry,
        "validate_native_entry_samples",
        synchronized_validate,
    )
    state_lock = threading.Lock()
    synthesis_calls = 0
    compile_calls = 0
    first_synthesis_entered = threading.Event()
    duplicate_synthesis_entered = threading.Event()
    release_synthesis = threading.Event()

    def counted_synthesis(*args, **kwargs):
        nonlocal synthesis_calls
        del args, kwargs
        with state_lock:
            synthesis_calls += 1
            call_index = synthesis_calls
        if call_index == 1:
            first_synthesis_entered.set()
            assert release_synthesis.wait(timeout=10)
        else:
            duplicate_synthesis_entered.set()
        return synthesis

    def counted_compile(
        entry,
        physical_args,
        *,
        sample_shapes,
        sample_scalars,
    ):
        nonlocal compile_calls
        del entry, physical_args, sample_shapes, sample_scalars
        with state_lock:
            compile_calls += 1

        def invoke(*runtime_args):
            del runtime_args

        return _declare_canonical_runtime_signature(invoke)

    monkeypatch.setattr(dispatch, "synthesize_aot_dispatch", counted_synthesis)
    monkeypatch.setattr(native_entry, "compile_native_entry_variant", counted_compile)
    inputs = _fake_native_inputs(module)
    errors = []

    def run_native():
        try:
            module.flash_kda(**inputs)
        except BaseException as exc:  # noqa: BLE001 - surfaced in main thread
            errors.append(exc)

    workers = [threading.Thread(target=run_native) for _ in range(2)]
    for worker in workers:
        worker.start()
    assert first_synthesis_entered.wait(timeout=10)
    duplicate_observed = duplicate_synthesis_entered.wait(timeout=0.5)
    release_synthesis.set()
    for worker in workers:
        worker.join(timeout=10)

    assert not any(worker.is_alive() for worker in workers)
    assert errors == []
    assert not duplicate_observed
    assert synthesis_calls == 1
    assert compile_calls == 1

    monkeypatch.setattr(
        native_entry,
        "validate_native_entry_samples",
        real_validate,
    )
    module._DYNAMIC_KERNEL_CACHE.clear()
    compile_attempts = 0

    def flaky_compile(
        entry,
        physical_args,
        *,
        sample_shapes,
        sample_scalars,
    ):
        nonlocal compile_attempts
        del entry, physical_args, sample_shapes, sample_scalars
        compile_attempts += 1
        if compile_attempts == 1:
            raise RuntimeError("injected native compile failure")

        def invoke(*runtime_args):
            del runtime_args

        return _declare_canonical_runtime_signature(invoke)

    monkeypatch.setattr(native_entry, "compile_native_entry_variant", flaky_compile)
    with pytest.raises(RuntimeError, match="injected native compile failure"):
        module.flash_kda(**inputs)
    module.flash_kda(**inputs)

    assert compile_attempts == 2


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

        return _declare_canonical_runtime_signature(invoke)

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
