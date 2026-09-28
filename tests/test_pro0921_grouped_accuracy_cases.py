"""CPU tests for the Pro0921 grouped Compressor/QuantCompressor test entry.

These run without an NPU and without importing ``cann_ops_transformer``: they
exercise the case contract, the independent numpy golden, the full-cycle
state/reset semantics, the timing aggregation and the batch ``.pt`` contract.
The NPU accuracy/perf runners are checked for their explicit no-device failure
behaviour instead of fabricating numbers.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTEST_C = ROOT / "attention" / "compressor" / "tests" / "pytest"
PYTEST_Q = ROOT / "attention" / "quant_compressor" / "tests" / "pytest"
for _dir in (str(PYTEST_C), str(PYTEST_Q)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


def _load(name, directory):
    spec = importlib.util.spec_from_file_location(name, directory / "pro0921_grouped_cases.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COMPRESSOR = _load("pro0921_grouped_cases", PYTEST_C)
QUANT = _load("pro0921_grouped_cases_quant", PYTEST_Q)


def _load_accuracy_perf():
    spec = importlib.util.spec_from_file_location(
        "pro0921_accuracy_perf", PYTEST_C / "pro0921_accuracy_perf.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pro0921_accuracy_perf"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------
def test_compressor_contract_rejects_missing_path():
    with pytest.raises(COMPRESSOR.CaseContractError):
        COMPRESSOR.validate_case({"name": "x", "builder": "hca_boundary", "cmp_ratio": 128,
                                  "coff": 1, "cache_mode": 1, "hidden_size": 1024,
                                  "head_dim": 128, "batch": 1, "seq": 1, "dtype": "bfloat16"})


def test_compressor_contract_rejects_grouped_without_update_flag():
    case = {"name": "x", "path": "grouped", "builder": "hca_boundary", "cmp_ratio": 128,
            "coff": 1, "cache_mode": 1, "hidden_size": 1024, "head_dim": 128,
            "batch": 1, "seq": 1, "dtype": "bfloat16"}
    with pytest.raises(COMPRESSOR.CaseContractError):
        COMPRESSOR.validate_case(case)


def test_compressor_contract_rejects_unknown_path_value():
    case = {"name": "x", "path": "grouped-maybe", "builder": "hca_boundary",
            "cmp_ratio": 128, "coff": 1, "cache_mode": 1, "hidden_size": 1024,
            "head_dim": 128, "batch": 1, "seq": 1, "dtype": "bfloat16",
            "x_cache_update": True}
    with pytest.raises(COMPRESSOR.CaseContractError):
        COMPRESSOR.validate_case(case)


def test_quant_contract_rejects_missing_path():
    with pytest.raises(QUANT.CaseContractError):
        QUANT.validate_case({"name": "q", "builder": "quant_boundary", "cmp_ratio": 128,
                             "coff": 1, "cache_mode": 1, "hidden_size": 1024,
                             "head_dim": 128, "batch": 1, "seq": 1})


def test_build_case_rejects_normal_path_with_grouped_builder():
    spec = {"name": "x", "path": "normal", "builder": "hca_boundary", "cmp_ratio": 128,
            "coff": 1, "cache_mode": 1, "hidden_size": 1024, "head_dim": 128,
            "batch": 1, "seq": 1, "dtype": "bfloat16"}
    with pytest.raises(COMPRESSOR.CaseContractError):
        COMPRESSOR.build_case(spec)


def test_build_case_rejects_geometry_mismatch():
    spec = {"name": "x", "path": "grouped", "builder": "hca_boundary", "cmp_ratio": 128,
            "coff": 1, "cache_mode": 1, "hidden_size": 4096, "head_dim": 128,
            "batch": 1, "seq": 1, "dtype": "bfloat16", "x_cache_update": True}
    with pytest.raises(COMPRESSOR.CaseContractError):
        COMPRESSOR.build_case(spec)


def test_load_manifest_rejects_grouped_without_update_flag(tmp_path):
    manifest_path = tmp_path / "cases.json"
    manifest_path.write_text(json.dumps({
        "contract": COMPRESSOR.CONTRACT, "op": COMPRESSOR.OP,
        "cases": [{"name": "g", "path": "grouped", "dir": str(tmp_path)}]}), encoding="utf-8")
    with pytest.raises(COMPRESSOR.CaseContractError):
        COMPRESSOR.load_manifest(manifest_path)


# ---------------------------------------------------------------------------
# Independent numpy golden over the frozen B1 set
# ---------------------------------------------------------------------------
def _materialize(module, tmp_path, name):
    manifest = module.materialize(module.default_case_set(), tmp_path / name)
    assert manifest["cases"], "no cases materialized"
    return manifest


def test_compressor_independent_golden(tmp_path):
    manifest = _materialize(COMPRESSOR, tmp_path, "c")
    failures = []
    checked = 0
    for entry in manifest["cases"]:
        for check, ok, value in COMPRESSOR.independent_contract_check(entry["dir"], entry):
            checked += 1
            if not ok:
                failures.append((check, value))
    assert checked >= 10, checked
    assert not failures, failures


def test_quant_independent_golden(tmp_path):
    manifest = _materialize(QUANT, tmp_path, "q")
    failures = []
    for entry in manifest["cases"]:
        for check, ok, value in QUANT.independent_contract_check(entry["dir"], entry):
            if not ok:
                failures.append((check, value))
    assert not failures, failures


# ---------------------------------------------------------------------------
# Full-cycle state/reset and normal<->grouped equivalence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ratio,coff", [(4, 2), (4, 1), (128, 1), (128, 2)])
def test_compressor_cycle_state_reset(ratio, coff):
    result = COMPRESSOR.cycle_check(ratio=ratio, coff=coff)
    assert result["pass"], result
    assert result["reset_stable"], result
    assert result["grouped_vs_reference_max_abs"] < 1e-4, result
    assert result["normal_vs_reference_max_abs"] < 1e-4, result


def test_normal_grouped_equivalence():
    result = COMPRESSOR.equivalence_check()
    assert result["pass"], result
    assert result["cmp_kv_max_abs"] == 0.0
    assert result["state_max_abs"] == 0.0


# ---------------------------------------------------------------------------
# Timing aggregation
# ---------------------------------------------------------------------------
def test_timing_aggregation_median_min_raw():
    perf = _load_accuracy_perf()
    stats = perf.aggregate_samples([3.0, 1.0, 2.0, 5.0])
    assert stats["min_ms"] == 1.0
    assert stats["median_ms"] == 2.5
    assert stats["max_ms"] == 5.0
    assert stats["count"] == 4
    assert stats["samples_ms"] == [3.0, 1.0, 2.0, 5.0]
    with pytest.raises(ValueError):
        perf.aggregate_samples([])
    with pytest.raises(ValueError):
        perf.aggregate_samples([1.0, 0.0])
    with pytest.raises(ValueError):
        perf.aggregate_samples([1.0, float("inf")])


def test_perf_scenarios_cover_full_matrix():
    perf = _load_accuracy_perf()
    assert len(perf.SCENARIOS) == 12
    combos = {(s["geom"], s["path"], s["kind"]) for s in perf.SCENARIOS}
    assert len(combos) == 12
    for geom in ("csa", "hca"):
        for path in ("normal", "grouped"):
            for kind in ("boundary", "nonboundary", "cycle"):
                assert (geom, path, kind) in combos


def test_scenario_steps_match_scope():
    perf = _load_accuracy_perf()
    for scn in perf.SCENARIOS:
        want = scn["ratio"] if scn["kind"] == "cycle" else 1
        assert scn["steps"] == want, (scn["name"], scn["steps"], want)


def test_boundary_and_nonboundary_are_single_step_on_both_paths():
    perf = _load_accuracy_perf()
    for scn in perf.SCENARIOS:
        fx = perf.build_scenario(scn, seed=20260927)
        want = scn["ratio"] if scn["kind"] == "cycle" else 1
        assert len(fx["steps"]) == want, (scn["name"], len(fx["steps"]), want)
    # a normal/grouped pair of the same geom+kind must run the same call count
    for geom in ("csa", "hca"):
        for kind in ("boundary", "nonboundary", "cycle"):
            steps = {s["steps"] for s in perf.SCENARIOS
                     if s["geom"] == geom and s["kind"] == kind}
            assert len(steps) == 1, (geom, kind, steps)


def test_perf_pair_scope_guard_rejects_mismatch():
    perf = _load_accuracy_perf()
    good = [{"geom": "csa", "kind": "boundary", "path": "normal", "median_ms": 1.0,
             "min_ms": 1.0, "steps_per_trial": 1, "ratio": 4, "coff": 2, "head": 512,
             "hidden": 1024, "seed": 20260927, "scenario": "csa_normal_boundary"},
            {"geom": "csa", "kind": "boundary", "path": "grouped", "median_ms": 2.0,
             "min_ms": 2.0, "steps_per_trial": 1, "ratio": 4, "coff": 2, "head": 512,
             "hidden": 1024, "seed": 20260927, "scenario": "csa_grouped_boundary"}]
    pairs = perf._pair_speedups(good, 20260927)
    assert pairs and pairs[0]["speedup_median"] == 0.5
    # a cycle-length normal paired with a single-call grouped no longer divides
    bad = [dict(good[0], steps_per_trial=4), good[1]]
    with pytest.raises(AssertionError):
        perf._pair_speedups(bad, 20260927)


def test_perf_scenario_fixtures_build_on_cpu():
    perf = _load_accuracy_perf()
    for scn in perf.SCENARIOS:
        fx = perf.build_scenario(scn, seed=20260927)
        assert len(fx["steps"]) == scn["steps"], (scn["name"], len(fx["steps"]))
        assert fx["wkv"].shape[1] == fx["hidden"]
        assert fx["wkv"].shape[0] == fx["coff"] * fx["head"]
        if scn["path"] == "normal":
            assert fx["ring0"] is None and fx["expected"]["ring"] is None
        else:
            assert fx["ring0"] is not None
            assert fx["ring0"].shape[0] == 2  # null block + request-owned block


def test_cpu_self_check_passes_on_shared_fixtures():
    perf = _load_accuracy_perf()
    report = perf.cpu_self_check(seed=20260927)
    assert report["pass"], report
    assert len(report["scenarios"]) == 12


# ---------------------------------------------------------------------------
# Batch .pt contract
# ---------------------------------------------------------------------------
def test_emit_pt_grouped_fields(tmp_path):
    manifest = _materialize(COMPRESSOR, tmp_path, "c")
    entry = next(e for e in manifest["cases"] if e["name"] == "hca_grouped_boundary")
    pt_path = tmp_path / "grouped.pt"
    COMPRESSOR.emit_pt(entry, pt_path)
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert data["path"] == "grouped"
    assert "x_cache" in data and "x_cache_slot_mapping" in data
    assert "expected_x_cache" in data
    assert data["expected_x_cache"].shape == data["x_cache"].shape
    assert data["expected_x_cache"].dtype == data["x_cache"].dtype
    args = COMPRESSOR.pt_grouped_args(data)
    assert args["path"] == "grouped"
    assert "x_cache" in args and args["x_cache_update"] is True
    assert len(data["params"]) == 20


def test_emit_pt_li_expected_ring_is_read_only(tmp_path):
    manifest = _materialize(COMPRESSOR, tmp_path, "c")
    entry = next(e for e in manifest["cases"] if e["name"] == "csa_grouped_li_read_shared")
    pt_path = tmp_path / "li.pt"
    COMPRESSOR.emit_pt(entry, pt_path)
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    # LI (x_cache_update=false) is a read-only shared cache: expected == input
    assert torch.equal(data["expected_x_cache"], data["x_cache"])


def test_emit_pt_normal_has_no_grouped_payload(tmp_path):
    manifest = _materialize(COMPRESSOR, tmp_path, "c")
    entry = next(e for e in manifest["cases"] if e["name"] == "hca_normal_r128")
    pt_path = tmp_path / "normal.pt"
    COMPRESSOR.emit_pt(entry, pt_path)
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert data["path"] == "normal"
    assert "x_cache" not in data
    assert COMPRESSOR.pt_grouped_args(data) == {"path": "normal"}


def test_pt_grouped_args_requires_explicit_path():
    # a pre-grouped .pt without 'path' is a hard error, never a normal default
    with pytest.raises(COMPRESSOR.CaseContractError):
        COMPRESSOR.pt_grouped_args({"cpu_result": 1})
    assert COMPRESSOR.pt_grouped_args({"path": "normal"}) == {"path": "normal"}
    with pytest.raises(COMPRESSOR.CaseContractError):
        COMPRESSOR.pt_grouped_args({"path": "grouped"})
    with pytest.raises(QUANT.CaseContractError):
        QUANT.pt_grouped_args({"cpu_result": 1})
    assert QUANT.pt_grouped_args({"path": "normal"}) == {"path": "normal"}


def test_quant_emit_pt_grouped_expected_ring(tmp_path):
    manifest = _materialize(QUANT, tmp_path, "q")
    entry = next(e for e in manifest["cases"] if e["name"] == "quant_grouped_boundary")
    pt_path = tmp_path / "qg.pt"
    QUANT.emit_pt(entry, pt_path)
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert data["expected_x_cache"].dtype == torch.uint8
    assert torch.equal(data["x_cache"], data["expected_x_cache"]) is False
    assert data["expected_x_cache"].shape == data["x_cache"].shape


def test_quant_emits_grouped_pt(tmp_path):
    manifest = _materialize(QUANT, tmp_path, "q")
    entry = next(e for e in manifest["cases"] if e["name"] == "quant_grouped_boundary")
    pt_path = tmp_path / "qgrouped.pt"
    QUANT.emit_pt(entry, pt_path)
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert data["path"] == "grouped"
    assert data["quant_mode"] == 1
    assert len(data["params"]) == 24
    assert QUANT.pt_grouped_args(data)["path"] == "grouped"


# ---------------------------------------------------------------------------
# No-device behaviour: explicit failure, never fabricated results
# ---------------------------------------------------------------------------
def test_accuracy_runner_fails_without_device(tmp_path):
    out = tmp_path / "accuracy.json"
    result = subprocess.run(
        [sys.executable, "pro0921_accuracy_perf.py", "--mode", "accuracy",
         "--accuracy-out", str(out)],
        cwd=PYTEST_C, capture_output=True, text=True, timeout=300)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRO0921_ACCURACY=FAIL" in result.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["npu"] is None
    # the CPU-side shared-fixture self-check still ran and passed
    assert report["cpu_self_check"]["pass"] is True
    assert len(report["cpu_self_check"]["scenarios"]) == 12


def test_perf_runner_never_writes_without_device(tmp_path):
    out = tmp_path / "perf.json"
    result = subprocess.run(
        [sys.executable, "pro0921_accuracy_perf.py", "--mode", "perf", "--perf-out", str(out)],
        cwd=PYTEST_C, capture_output=True, text=True, timeout=300)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRO0921_PERF=FAIL" in result.stdout
    assert not out.exists(), "a performance file must not be produced without a device"


def test_unsupported_batch_or_seq_rejected():
    for flag, value in (("--batch", "2"), ("--seq", "4")):
        result = subprocess.run(
            [sys.executable, "pro0921_accuracy_perf.py", "--mode", "plan", flag, value],
            cwd=PYTEST_C, capture_output=True, text=True, timeout=120)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "unsupported batch/seq" in result.stdout


def test_grouped_cases_plan_runs_without_device():
    result = subprocess.run(
        [sys.executable, "pro0921_grouped_cases.py", "--plan"],
        cwd=PYTEST_C, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "contract=pro0921-grouped-compressor-v1" in result.stdout


def test_quant_grouped_cases_plan_runs_without_device():
    result = subprocess.run(
        [sys.executable, "pro0921_grouped_cases.py", "--plan"],
        cwd=PYTEST_Q, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "contract=pro0921-grouped-quant-compressor-v1" in result.stdout


# ---------------------------------------------------------------------------
# bfloat16 readback: a raw .numpy() on bf16 is unsupported, the runner path is
# to cast on-device to float32 and compare to the bf16-rounded golden exactly.
# ---------------------------------------------------------------------------
def test_bf16_readback_matches_golden(tmp_path):
    _, latent_ref = COMPRESSOR._smoke()
    import torch

    golden = latent_ref.to_bf16(np.array([0.1, -0.25, 3.5, 1e-8], dtype=np.float32))
    bf16 = torch.from_numpy(golden.astype(np.float32)).to(torch.bfloat16)
    # the runner's cast path: on-device float32, then .numpy()
    readback = bf16.detach().cpu().to(torch.float32).numpy()
    assert np.array_equal(readback, golden), (readback, golden)
    # a raw .numpy() on the bf16 tensor is unsupported and must not be used
    with pytest.raises((TypeError, RuntimeError)):
        bf16.detach().cpu().numpy()


def test_compare_rejects_nonfinite_and_shape_mismatch():
    ok = COMPRESSOR._compare(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert ok["pass"] is True
    nan = COMPRESSOR._compare(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
    assert nan["pass"] is False and nan["reason"] == "non-finite value present"
    inf = COMPRESSOR._compare(np.array([1.0, 2.0]), np.array([1.0, np.inf]))
    assert inf["pass"] is False
    shaped = COMPRESSOR._compare(np.array([1.0, 2.0]), np.array([1.0]))
    assert shaped["pass"] is False and shaped["reason"] == "shape mismatch"


def _latent_ref():
    _, latent_ref = COMPRESSOR._smoke()
    return latent_ref


def _reference_operator(torch, ref, skip_ring_write=False):
    """CPU mock operator: writes state/cache in place like the real kernel."""
    def op(x, wkv, wgate, state_cache, ape, *, cmp_ratio, state_block_table,
           cu_seqlens=None, seqused=None, start_pos=None, coff=1, cache_mode=1,
           x_cache=None, x_cache_slot_mapping=None, x_cache_update=True):
        def np32(t):
            return None if t is None else t.detach().cpu().to(torch.float32).numpy()

        block_table = state_block_table.detach().cpu().numpy().astype(np.int64)
        start = [int(v) for v in start_pos.detach().cpu().numpy().reshape(-1)]
        used = None if seqused is None else [
            int(v) for v in seqused.detach().cpu().numpy().reshape(-1)]
        xn = np32(x)
        if x_cache is None:
            cmp_kv, _, _, out_state = ref.compress_golden(
                xn, np32(wkv), np32(wgate), np32(state_cache), np32(ape), block_table,
                int(cmp_ratio), int(coff), int(cache_mode), start_pos=start, seqused=used)
            state_cache.copy_(torch.from_numpy(out_state))
            return torch.from_numpy(cmp_kv)
        slots = x_cache_slot_mapping.detach().cpu().numpy().reshape(-1)
        out = ref.grouped_case_golden(
            xn, np32(wkv), np32(wgate), np32(state_cache), np32(ape), block_table,
            np32(x_cache), slots, int(cmp_ratio), int(coff), int(cache_mode),
            start_pos=start, seqused=used,
            x_cache_update=bool(x_cache_update) and not skip_ring_write)
        state_cache.copy_(torch.from_numpy(out["state_cache"]))
        x_cache.copy_(torch.from_numpy(out["x_cache"]))
        cmp_kv = out["cmp_kv"]
        if cmp_kv is None:
            cmp_kv = np.zeros((1, 1, np32(wkv).shape[0] // int(coff)), dtype=np.float32)
        return torch.from_numpy(cmp_kv)

    return op


def test_cpu_device_simulation_all_twelve_paths():
    perf = _load_accuracy_perf()
    ref = _latent_ref()
    import torch

    op = _reference_operator(torch, ref)
    seen = set()
    for scn in perf.SCENARIOS:
        prepared = perf._prepare(torch, scn, seed=20260927, device="cpu")
        checks = perf.verify_prepared(torch, op, prepared)
        assert all(c[1]["pass"] for c in checks), (scn["name"], checks)
        # reset then re-verify: the reset must reproduce the pristine state/ring
        perf.reset_prepared(torch, prepared)
        prepared["state"].add_(1.0)
        if prepared["ring"] is not None:
            prepared["ring"].add_(1.0)
        perf.reset_prepared(torch, prepared)
        checks_again = perf.verify_prepared(torch, op, prepared)
        assert all(c[1]["pass"] for c in checks_again), (scn["name"], checks_again)
        # timing aggregation over the CPU mock exercises the timer plumbing
        stats = perf.measure_scenario(torch, op, prepared, scn, warmup=1, repeats=2,
                                      timer=perf._cpu_timer())
        assert stats["count"] == 2 and stats["min_ms"] > 0
        assert len(stats["samples_ms"]) == 2
        seen.add(scn["name"])
    assert len(seen) == 12


def test_cpu_device_simulation_detects_missing_ring_write():
    perf = _load_accuracy_perf()
    ref = _latent_ref()
    import torch

    op = _reference_operator(torch, ref, skip_ring_write=True)
    scn = next(s for s in perf.SCENARIOS
               if s["path"] == "grouped" and s["kind"] == "boundary")
    prepared = perf._prepare(torch, scn, seed=20260927, device="cpu")
    checks = perf.verify_prepared(torch, op, prepared)
    ring_check = next(c for c in checks if c[0] == "x_cache")
    assert ring_check[1]["pass"] is False, checks


def _wrapped_operator(torch, ref, *, skip_ring=False, skip_state=False,
                      corrupt_ring=False, corrupt_state=False):
    """Correct CPU mock operator with one controlled side-effect fault."""
    base = _reference_operator(torch, ref, skip_ring_write=skip_ring)

    def op(x, wkv, wgate, state_cache, ape, **kwargs):
        snapshot = state_cache.detach().clone() if skip_state else None
        out = base(x, wkv, wgate, state_cache, ape, **kwargs)
        if snapshot is not None:
            state_cache.copy_(snapshot)  # undo the update: a missed state write
        if corrupt_ring and kwargs.get("x_cache") is not None:
            ring = kwargs["x_cache"]
            ring.view(-1)[0] = ring.view(-1)[0] + np.float32(1.0)
        if corrupt_state:
            state_cache.view(-1)[0] = state_cache.view(-1)[0] + 1.0
        return out

    return op


def _check(checks, name):
    return next(c for c in checks if c[0] == name)[1]


def _scenario(perf, geom, path, kind):
    return next(s for s in perf.SCENARIOS
                if s["geom"] == geom and s["path"] == path and s["kind"] == kind)


def test_verify_prepared_is_exact_like_the_batch_loader():
    """The on-board runner must reject the same side-effect faults the batch
    loader catches, element-wise, instead of averaging them away with the 99.5%
    output tolerance.  Every fault below previously passed ``verify_prepared``.
    """
    perf = _load_accuracy_perf()
    ref = _latent_ref()
    import torch

    # a whole token row that was never written into the non-boundary x_cache
    p = perf._prepare(torch, _scenario(perf, "hca", "grouped", "nonboundary"),
                      seed=20260927, device="cpu")
    checks = perf.verify_prepared(
        torch, _wrapped_operator(torch, ref, skip_ring=True), p)
    assert _check(checks, "x_cache")["pass"] is False, checks
    assert _check(checks, "x_cache")["mismatch_elements"] >= 1, checks

    # a single corrupted element in the cache
    p = perf._prepare(torch, _scenario(perf, "hca", "grouped", "boundary"),
                      seed=20260927, device="cpu")
    checks = perf.verify_prepared(
        torch, _wrapped_operator(torch, ref, corrupt_ring=True), p)
    assert _check(checks, "x_cache")["pass"] is False, checks
    assert _check(checks, "x_cache")["mismatch_elements"] >= 1, checks

    # a single damaged element in the state region the golden left untouched
    p = perf._prepare(torch, _scenario(perf, "hca", "grouped", "nonboundary"),
                      seed=20260927, device="cpu")
    checks = perf.verify_prepared(
        torch, _wrapped_operator(torch, ref, corrupt_state=True), p)
    assert _check(checks, "state_cache")["pass"] is False, checks
    assert _check(checks, "state_cache")["untouched_mismatch"] >= 1, checks

    # a missed write into the region the golden did update
    p = perf._prepare(torch, _scenario(perf, "hca", "normal", "boundary"),
                      seed=20260927, device="cpu")
    checks = perf.verify_prepared(
        torch, _wrapped_operator(torch, ref, skip_state=True), p)
    assert _check(checks, "state_cache")["pass"] is False, checks
    assert _check(checks, "state_cache")["updated_mismatch"] >= 1, checks


def test_verify_prepared_requires_exact_untouched_state_and_cache():
    """A correct operator passes element-wise; the non-boundary grouped state is
    entirely untouched and the readback is compared per element, never by a
    whole-block percentage.
    """
    perf = _load_accuracy_perf()
    ref = _latent_ref()
    import torch

    op = _reference_operator(torch, ref)
    for scn in perf.SCENARIOS:
        prepared = perf._prepare(torch, scn, seed=20260927, device="cpu")
        checks = perf.verify_prepared(torch, op, prepared)
        assert all(c[1]["pass"] for c in checks), (scn["name"], checks)
        if scn["path"] == "grouped" and scn["kind"] == "nonboundary":
            # the golden leaves the whole state pristine on a non-boundary token
            assert _check(checks, "state_cache")["updated_elements"] == 0, checks
            assert _check(checks, "state_cache")["untouched_mismatch"] == 0, checks
        if prepared["ring"] is not None:
            assert _check(checks, "x_cache")["mismatch_elements"] == 0, checks


def test_run_perf_refuses_timing_when_verification_fails(monkeypatch, tmp_path):
    """A failed side-effect verification raises before any timing runs and never
    writes a performance file.
    """
    perf = _load_accuracy_perf()
    ref = _latent_ref()
    import types
    import torch

    fake_torch = types.SimpleNamespace(
        float32=torch.float32, bfloat16=torch.bfloat16, int32=torch.int32,
        from_numpy=torch.from_numpy,
        npu=types.SimpleNamespace(set_device=lambda *a, **k: None))
    broken = _reference_operator(torch, ref, skip_ring_write=True)
    monkeypatch.setattr(perf.cases, "_load_npu_runtime", lambda: (fake_torch, broken))
    monkeypatch.setattr(perf, "_npu_timer", lambda torch: (lambda fn: 1.0))
    monkeypatch.setattr(perf, "SCENARIOS",
                        (_scenario(perf, "hca", "grouped", "boundary"),))
    real_prepare = perf._prepare
    monkeypatch.setattr(perf, "_prepare",
                        lambda t, s, seed, device: real_prepare(torch, s, seed, "cpu"))
    timed = []
    monkeypatch.setattr(perf, "measure_scenario", lambda *a, **k: timed.append(1))
    out = tmp_path / "perf.json"
    with pytest.raises(RuntimeError, match="refusing to record"):
        perf.run_perf(20260927, warmup=1, repeats=2, out_json=out)
    assert timed == [], "timing must not run after a failed verification"
    assert not out.exists(), "a failed verification must not write a perf file"


def test_boundary_cpu_mock_readback_and_two_sides_equal():
    perf = _load_accuracy_perf()
    ref = _latent_ref()
    import torch

    op = _reference_operator(torch, ref)
    cmps = {}
    for scn in perf.SCENARIOS:
        if scn["kind"] != "boundary":
            continue
        prepared = perf._prepare(torch, scn, seed=20260927, device="cpu")
        # the real CPU mock operator read-back must equal the full-group golden
        checks = perf.verify_prepared(torch, op, prepared)
        assert all(c[1]["pass"] for c in checks), (scn["name"], checks)
        perf.reset_prepared(torch, prepared)
        outs = perf._run_prepared(torch, op, prepared)
        got = perf._to_np(torch, outs[prepared["cmp_step"]])
        assert np.abs(got - prepared["expected"]["cmp_kv"]).max() < 1e-4, scn["name"]
        cmps[(scn["geom"], scn["path"])] = got
    # the normal (state-warmed) and grouped (ring-warmed) single boundary call
    # produce the same group projection
    for geom in ("csa", "hca"):
        diff = np.abs(cmps[(geom, "normal")] - cmps[(geom, "grouped")]).max()
        assert diff < 1e-5, (geom, float(diff))


def test_cpu_simulation_dtype_and_shape_guard():
    perf = _load_accuracy_perf()
    # every fixture's slot bank and state block index must stay in bounds: a
    # 1-block state with block_table[[1]] previously read out of range.
    for scn in perf.SCENARIOS:
        fx = perf.build_scenario(scn, seed=20260927)
        blocks = fx["state0"].shape[0]
        assert 1 < blocks, (scn["name"], blocks)
        for step in fx["steps"]:
            if step["slot"] is not None:
                assert 0 <= step["slot"] // fx["ring0"].shape[1] < fx["ring0"].shape[0]

    # mismatched readback shape must fail the compare, not pass silently
    assert perf.cases._compare(np.zeros((1, 1, 4)), np.zeros((1, 1, 5)))["pass"] is False


# ---------------------------------------------------------------------------
# Batch loaders: internal grouped side-effect guard (x_cache read-back + full
# state), exercised on CPU with real torch tensors standing in for the device
# (.npu() identity) and a fake operator.  The real provider is never imported
# and JIT is never touched.
# ---------------------------------------------------------------------------
BATCH_C = PYTEST_C / "batch"
BATCH_Q = PYTEST_Q / "batch"


def _fake_torch_env(monkeypatch, torch):
    import types

    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu = types.SimpleNamespace(
        set_device=lambda *a, **k: None, synchronize=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    fake_cann = types.ModuleType("cann_ops_transformer")
    fake_ops = types.ModuleType("cann_ops_transformer.ops")
    fake_ops.compressor = lambda *a, **k: None
    fake_cann.ops = fake_ops
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", fake_cann)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", fake_ops)
    # the batch loaders import pandas at module scope but never call it on this path
    if "pandas" not in sys.modules:
        monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))
    # the loaders call .npu(); for the CPU test the tensor stays on the CPU
    monkeypatch.setattr(torch.Tensor, "npu", lambda self, *a, **k: self, raising=False)


def _load_batch_loader(monkeypatch, torch, name, path):
    _fake_torch_env(monkeypatch, torch)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _expected_state(test_data, torch):
    return torch.cat([test_data["cpu_kv_state"], test_data["cpu_score_state"]], dim=2)


def _compressor_fake(test_data, torch, mode):
    """CPU stand-in for the compressor op: reproduces (or corrupts) side effects."""
    def op(x, wkv, wgate, state_cache, ape, **kwargs):
        ring = kwargs.get("x_cache")
        if ring is not None:
            if mode == "no_cache_write":
                pass  # missing in-place ring write
            elif mode == "li_write":
                ring.view(-1)[0] = ring.view(-1)[0] + 1  # spurious write
            else:  # ok / state_corrupt write the ring correctly
                ring.copy_(test_data["expected_x_cache"].to(ring.dtype))
        state_cache.copy_(_expected_state(test_data, torch).to(state_cache.dtype))
        if mode == "state_corrupt":
            state_cache.view(-1)[0] = state_cache.view(-1)[0] + 1.0
        return torch.zeros(1, dtype=torch.bfloat16)

    return op


def test_compressor_batch_loader_verifies_grouped_readback(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    loader = _load_batch_loader(
        monkeypatch, torch, "compressor_pt_loadprocess",
        BATCH_C / "compressor_pt_loadprocess.py")
    manifest = _materialize(COMPRESSOR, tmp_path, "c")
    entries = {e["name"]: e for e in manifest["cases"]}

    def run(entry_name, mode):
        entry = entries[entry_name]
        pt = tmp_path / (entry_name + ".pt")
        COMPRESSOR.emit_pt(entry, pt)
        data = torch.load(pt, map_location="cpu", weights_only=False)
        monkeypatch.setattr(loader, "compressor", _compressor_fake(data, torch, mode))
        return loader.test_compressor_process(str(pt), device_id=0)

    # correct grouped boundary passes and the read-back equals the golden ring
    result = run("hca_grouped_boundary", "ok")
    assert result[2] is not None
    # missing in-place ring write is detected
    with pytest.raises(AssertionError, match="x_cache"):
        run("hca_grouped_boundary", "no_cache_write")
    # a read-only (LI) shared cache that gets written anyway is detected
    run("csa_grouped_li_read_shared", "ok")
    with pytest.raises(AssertionError, match="x_cache"):
        run("csa_grouped_li_read_shared", "li_write")
    # a non-boundary call whose untouched state region is corrupted is detected
    with pytest.raises(AssertionError, match="state_cache"):
        run("hca_grouped_update_only", "state_corrupt")
    # the normal path keeps its existing behaviour (no grouped fields to guard)
    run("hca_normal_r128", "ok")


def test_quant_batch_loader_verifies_grouped_readback(monkeypatch, tmp_path):
    import types

    torch = pytest.importorskip("torch")
    loader = _load_batch_loader(
        monkeypatch, torch, "quant_compressor_pt_loadprocess",
        BATCH_Q / "quant_compressor_pt_loadprocess.py")
    manifest = _materialize(QUANT, tmp_path, "q")
    entries = {e["name"]: e for e in manifest["cases"]}

    def fake(test_data, mode):
        def op(x, wkv, wgate, state_cache, ape, **kwargs):
            ring = kwargs.get("x_cache")
            if ring is not None:
                if mode == "no_cache_write":
                    pass
                elif mode == "li_write":
                    ring.view(-1)[0] = ring.view(-1)[0] + 1
                else:
                    ring.copy_(test_data["expected_x_cache"].to(ring.dtype))
            state_cache.copy_(_expected_state(test_data, torch).to(state_cache.dtype))
            if mode == "state_corrupt":
                state_cache.view(-1)[0] = state_cache.view(-1)[0] + 1.0
            return torch.zeros(1, dtype=torch.bfloat16)

        return op

    def run(pt_path, data, mode):
        monkeypatch.setattr(torch.ops, "cann_ops_transformer",
                            types.SimpleNamespace(quant_compressor=fake(data, mode)),
                            raising=False)
        return loader.test_compressor_process(str(pt_path), device_id=0)

    def emit(entry_name):
        entry = entries[entry_name]
        pt = tmp_path / (entry_name + ".pt")
        QUANT.emit_pt(entry, pt)
        return pt, torch.load(pt, map_location="cpu", weights_only=False)

    pt, data = emit("quant_grouped_boundary")
    assert run(pt, data, "ok")["npu_result"] is not None
    with pytest.raises(AssertionError, match="x_cache"):
        run(pt, data, "no_cache_write")
    # a non-boundary call whose untouched state region is corrupted is detected
    up_pt, up_data = emit("quant_grouped_update_only")
    run(up_pt, up_data, "ok")
    with pytest.raises(AssertionError, match="state_cache"):
        run(up_pt, up_data, "state_corrupt")

    # read-only shared cache: expected == input, so a spurious write is detected
    li_pt, li_data = emit("quant_grouped_boundary")
    li_data["x_cache_update"] = False
    li_data["expected_x_cache"] = li_data["x_cache"].clone()
    torch.save(li_data, li_pt)
    run(li_pt, li_data, "ok")
    with pytest.raises(AssertionError, match="x_cache"):
        run(li_pt, li_data, "li_write")
