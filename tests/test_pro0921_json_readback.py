"""CPU read-back tests for the Pro0921 JSON accuracy state-cache gate.

The JSON accuracy entry (``_run_one_accuracy``) previously checked
``state_cache`` with the whole-tensor ``_compare`` (>=99.5% of elements close).
For an HCA state that is mostly *unchanged* -- the non-boundary rows must be
left bit-for-bit alone -- a single corrupted or drifting element hides behind
that percentage.  These tests pin the new per-region ``_compare_state`` gate on
both JSON entries (Compressor and QuantCompressor) and prove a state mismatch
reaches ``checks`` -> ``failed`` through ``run_accuracy``, using a CPU stand-in
for the device move and the operator.  No provider import, no ``torch_npu``, no
JIT, no NPU.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTEST_C = ROOT / "attention" / "compressor" / "tests" / "pytest"
PYTEST_Q = ROOT / "attention" / "quant_compressor" / "tests" / "pytest"


def _load(module_name, directory):
    # The two files share a basename but are unrelated JSON entries; distinct
    # module names keep them from clobbering each other in sys.modules.
    spec = importlib.util.spec_from_file_location(
        module_name, directory / "pro0921_grouped_cases.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


COMPRESSOR = _load("pro0921_json_readback_compressor", PYTEST_C)
QUANT = _load("pro0921_json_readback_quant", PYTEST_Q)
CASES = [pytest.param(COMPRESSOR, id="compressor"),
         pytest.param(QUANT, id="quant_compressor")]


# ---------------------------------------------------------------------------
# _compare_state: the per-region gate itself
# ---------------------------------------------------------------------------
def _big_state(seed=20260928):
    """A large state with a small genuinely-updated region on top."""
    rng = np.random.default_rng(seed)
    pristine = rng.standard_normal((2000, 2000)).astype(np.float32)
    expect = pristine.copy()
    expect[:64, :] = (rng.standard_normal((64, 2000)) + 5.0).astype(np.float32)
    assert np.any(expect == pristine) and np.any(expect != pristine)
    return expect, pristine


@pytest.mark.parametrize("mod", CASES)
def test_single_untouched_element_corrupted_fails(mod):
    expect, pristine = _big_state()
    actual = expect.copy()
    r, c = 1500, 1234
    actual[r, c] = np.float32(actual[r, c] + 1.0)  # 1 of 4e6 elements, +1
    # The old whole-tensor gate (>=99.5%) would have passed this outright.
    assert 100.0 * (1.0 - 1.0 / expect.size) >= 99.5
    res = mod._compare_state(expect, actual, pristine)
    assert res["pass"] is False, res
    assert res["untouched_mismatch"] == 1, res
    assert res["updated_mismatch"] == 0, res


@pytest.mark.parametrize("mod", CASES)
def test_single_untouched_element_bitflip_fails(mod):
    expect, pristine = _big_state(seed=7)
    actual = expect.copy()
    idx = (1500, 1234)
    actual[idx] = np.nextafter(actual[idx], np.float32(np.inf))  # 1-ULP drift
    res = mod._compare_state(expect, actual, pristine)
    assert res["pass"] is False, res
    assert res["untouched_mismatch"] == 1, res


@pytest.mark.parametrize("mod", CASES)
def test_missing_update_in_updated_region_fails(mod):
    rng = np.random.default_rng(11)
    pristine = rng.standard_normal((512, 256)).astype(np.float32)
    expect = pristine.copy()
    # a +10 offset is guaranteed well outside rtol/atol for every element
    expect[:64] = (rng.standard_normal((64, 256)) + 10.0).astype(np.float32)
    actual = expect.copy()
    actual[:64] = pristine[:64]  # the write never happened
    res = mod._compare_state(expect, actual, pristine)
    assert res["pass"] is False, res
    assert res["updated_mismatch"] == 64 * 256, res
    assert res["untouched_mismatch"] == 0, res


@pytest.mark.parametrize("mod", CASES)
def test_small_updated_error_within_tolerance_passes(mod):
    rng = np.random.default_rng(13)
    pristine = rng.standard_normal((256, 128)).astype(np.float32)
    expect = pristine.copy()
    expect[:16] = (rng.standard_normal((16, 128)) + 2.0).astype(np.float32)
    actual = expect.copy()
    actual[3, 5] = np.float32(actual[3, 5] + 5e-5)  # inside atol
    res = mod._compare_state(expect, actual, pristine)
    assert res["pass"] is True, res
    assert res["updated_total"] == 16 * 128, res


@pytest.mark.parametrize("mod", CASES)
def test_shape_mismatch_fails(mod):
    expect, pristine = _big_state(seed=3)
    res = mod._compare_state(expect, expect[:100], pristine)
    assert res["pass"] is False and res["reason"] == "shape mismatch", res


@pytest.mark.parametrize("mod", CASES)
def test_nonfinite_fails(mod):
    expect, pristine = _big_state(seed=5)
    nan = expect.copy()
    nan[10, 10] = np.nan
    res = mod._compare_state(expect, nan, pristine)
    assert res["pass"] is False and res["reason"] == "non-finite value present", res
    inf = expect.copy()
    inf[10, 10] = np.inf
    res = mod._compare_state(expect, inf, pristine)
    assert res["pass"] is False and res["reason"] == "non-finite value present", res
    nan_in = pristine.copy()
    nan_in[0, 0] = np.nan
    res = mod._compare_state(expect, expect.copy(), nan_in)
    assert res["pass"] is False and res["reason"] == "non-finite value present", res


@pytest.mark.parametrize("mod", CASES)
def test_boundary_and_updateonly_states_pass(mod):
    # boundary-like: part of the state updated, the rest untouched
    expect, pristine = _big_state(seed=17)
    res = mod._compare_state(expect, expect.copy(), pristine)
    assert res["pass"] is True, res
    assert res["untouched_total"] > 0 and res["updated_total"] > 0, res
    # update-only-like: nothing changes, readback == pristine == golden
    upd = np.zeros((256, 256), dtype=np.float32)
    res = mod._compare_state(upd, upd.copy(), upd.copy())
    assert res["pass"] is True, res
    assert res["updated_total"] == 0 and res["untouched_total"] == upd.size, res


# ---------------------------------------------------------------------------
# CPU simulation: the gate is wired into _run_one_accuracy -> run_accuracy
# ---------------------------------------------------------------------------
def _fixture_states():
    """Reuse the frozen fixture builders for realistic pristine/golden states."""
    build_tiny_cases, _latent_ref = COMPRESSOR._smoke()
    rng = np.random.default_rng(20260928)
    update_only = build_tiny_cases.build_hca_update_only(rng)   # state untouched
    boundary = build_tiny_cases.build_hca_boundary(rng, False)  # boundary update
    return {
        "update_only": (update_only["payload"]["state_cache"].astype(np.float32),
                        update_only["outputs"][1].astype(np.float32)),
        "boundary": (boundary["payload"]["state_cache"].astype(np.float32),
                     boundary["outputs"][1].astype(np.float32)),
    }


def _cpu_torch(torch):
    """A stand-in ``torch`` that remaps the ``npu:<id>`` device move onto CPU."""
    class _Remap:
        __slots__ = ("_t",)

        def __init__(self, tensor):
            object.__setattr__(self, "_t", tensor)

        @staticmethod
        def _dev(x):
            return "cpu" if isinstance(x, str) and x.startswith("npu") else x

        def _wrap(self, result):
            return _Remap(result) if isinstance(result, torch.Tensor) else result

        def to(self, *args, **kwargs):
            args = tuple(_Remap._dev(a) for a in args)
            kwargs = {k: (_Remap._dev(v) if k == "device" else v)
                      for k, v in kwargs.items()}
            return self._wrap(self._t.to(*args, **kwargs))

        def detach(self):
            return self._wrap(self._t.detach())

        def cpu(self):
            return self._wrap(self._t.cpu())

        def numpy(self):
            return self._t.numpy()

        def copy_(self, src):
            self._t.copy_(src._t if isinstance(src, _Remap) else src)
            return self

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_t"), name)

    class _Npu:
        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def set_device(*_a, **_k):
            pass

    shim = types.SimpleNamespace()
    for name in ("bfloat16", "float16", "float32", "int32", "uint8"):
        setattr(shim, name, getattr(torch, name))
    shim.__version__ = torch.__version__
    shim.npu = _Npu()
    shim.from_numpy = lambda arr: _Remap(torch.from_numpy(arr))
    return shim


def _cpu_sim(mod, tmp_path, pristine, golden, device_state, monkeypatch):
    """Run ``_run_one_accuracy`` on CPU with a stand-in operator.

    ``golden`` is written as the on-disk provider golden; ``device_state`` is
    what the stand-in operator leaves in ``state_cache``.
    """
    import torch

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    np.save(case_dir / "x.npy", np.zeros((1, 1, 1), dtype=np.float32))
    np.save(case_dir / "state_cache.npy",
            np.ascontiguousarray(pristine, dtype=np.float32))
    np.save(case_dir / "golden_state.npy",
            np.ascontiguousarray(golden, dtype=np.float32))
    (case_dir / "golden.json").write_text(json.dumps({
        "output_names": ["cmp_kv", "state_cache"],
        "files": [None, "golden_state.npy"]}), encoding="utf-8")
    entry = {"name": "cpu_sim", "dir": str(case_dir), "path": "normal",
             "dtype": "bfloat16", "quant_mode": 1, "cmp_ratio": 128, "coff": 1,
             "cache_mode": 1, "x_cache_update": False}

    def op(_x, _wkv, _wgate, state_cache, _ape=None, **_kwargs):
        state_cache.copy_(torch.from_numpy(np.ascontiguousarray(device_state)))
        return torch.zeros((1, 1), dtype=torch.float32)

    monkeypatch.setattr(mod, "_load_npu_runtime", lambda: (_cpu_torch(torch), op))
    monkeypatch.setattr(mod, "_provider_note",
                        lambda _t: {"module": "cpu-sim"}, raising=False)
    return mod.run_accuracy({"cases": [entry]}, device_id=0)


def _state_check(report):
    checks = report["cases"][0]["checks"]
    return next(c for c in checks if c["check"].endswith(":state_cache"))


@pytest.mark.parametrize("mod", CASES)
def test_update_only_correct_state_passes_through_run_accuracy(mod, tmp_path, monkeypatch):
    pristine, golden = _fixture_states()["update_only"]
    assert np.array_equal(golden, pristine)  # update-only must not change state
    report = _cpu_sim(mod, tmp_path, pristine, golden, golden, monkeypatch)
    assert report["failed"] == [], report
    check = _state_check(report)
    assert check["ok"] is True, check
    assert check["metric"]["updated_total"] == 0, check


@pytest.mark.parametrize("mod", CASES)
def test_update_only_single_corrupted_untouched_element_fails(mod, tmp_path, monkeypatch):
    pristine, golden = _fixture_states()["update_only"]
    corrupted = golden.copy().reshape(-1)
    mid = corrupted.size // 2
    corrupted[mid] = np.float32(corrupted[mid] + 1.0)
    report = _cpu_sim(mod, tmp_path, pristine, golden,
                      corrupted.reshape(golden.shape), monkeypatch)
    assert report["failed"] == ["cpu_sim"], report
    check = _state_check(report)
    assert check["ok"] is False, check
    assert check["metric"]["untouched_mismatch"] == 1, check


@pytest.mark.parametrize("mod", CASES)
def test_boundary_missing_update_fails_through_run_accuracy(mod, tmp_path, monkeypatch):
    pristine, golden = _fixture_states()["boundary"]
    missing = golden.copy()
    updated = np.argwhere(golden != pristine)
    assert updated.size, "boundary fixture must update some state"
    idx = tuple(updated[0])
    missing[idx] = pristine[idx]  # drop one genuine update
    report = _cpu_sim(mod, tmp_path, pristine, golden, missing, monkeypatch)
    assert report["failed"] == ["cpu_sim"], report
    check = _state_check(report)
    assert check["ok"] is False, check
    assert check["metric"]["updated_mismatch"] >= 1, check
