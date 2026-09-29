"""Exercise the real PyTorch dispatcher for the stateful Compressor inference
registration, without loading any NPU binary.

Everything runs in a subprocess with ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` so the
real ``torch.library`` machinery is used. Covered:

* schema: the outer ``compressor`` op and the inner ``_compressor_inference``
  custom op both declare ``state_cache`` and ``x_cache`` as writes, and the
  removed pure-function forward / training registrations are gone;
* the inner ``_compressor_inference`` schema keeps every positional argument up
  to and including the last write (``x_cache``, index 10) free of a schema
  default, so the ADInplaceOrView version bump cannot index past the end of the
  trimmed args tuple;
* meta/fake shape inference for both the inner custom op and the outer op;
* behaviour driven through the real dispatcher with a small CPU native
  stand-in: the native binding is reached, the write is visible on the very
  same cache objects, the native call always receives ``grad_enabled=False``,
  plain inference / ``no_grad`` / ``inference_mode`` all run, and an ``x`` or a
  weight that requires grad is rejected before the native call. The outer
  normal path is exercised with ``x_cache=None`` and the default ``coff`` /
  ``cache_mode`` while still writing back ``state_cache``, and the grouped
  ``x_cache`` tensor path keeps its in-place write check.

The CPU kernels registered in the child are test-only stand-ins for the NPU
backend. They prove the dispatcher wiring and the Python guard, not anything
about the NPU kernel.
"""
import os
import subprocess
import sys


def test_registration_meta_and_inference_behaviour():
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run(
        [sys.executable, __file__, "--child"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for token in ("SCHEMA_AND_META_PASS", "INFERENCE_BEHAVIOUR_PASS"):
        assert token in result.stdout, result.stdout + result.stderr


def test_graph_convert_imports_with_only_inference_entry():
    """The real ``graph_convert_compressor`` module must import cleanly once the
    pure-function forward op is gone.

    A child process installs a minimal TorchAir stand-in so the
    ``_TORCHAIR_AVAILABLE`` decorator block really executes, imports the real
    ``graph_convert_compressor.py``, and asserts that only the mutable inference
    entry is registered while the converter call still refuses GE graph mode.
    """
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run(
        [sys.executable, __file__, "--child-graph"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for token in ("GRAPH_CONVERT_IMPORT_PASS", "GRAPH_CONVERT_NEGATIVE_PASS"):
        assert token in result.stdout, result.stdout + result.stderr


if __name__ == "__main__" and "--child-graph" not in sys.argv:
    import importlib.util
    import pathlib
    import types

    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    torch.utils.rename_privateuse1_backend("npu")
    lib = torch.library.Library("cann_ops_transformer", "DEF")

    CMP_KV_SIZE = (1, 1, 512)

    class NativeStandIn:
        """CPU stand-in for the NPU ``op_module.compressor`` binding."""

        def __init__(self):
            self.calls = []

        def compressor(self, *args):
            # Mirror the native ABI: the positional call ends with grad_enabled,
            # and the native side writes into state_cache (index 3) and x_cache
            # (index 10) in place.
            self.calls.append(args)
            args[3].add_(1)
            if args[10] is not None:
                args[10].add_(1)
            return (
                torch.zeros(CMP_KV_SIZE),
                torch.zeros(1),
                torch.zeros(1),
            )

    native = NativeStandIn()

    class Builder:
        def __init__(self, name, category=None):
            self.name = name

        def _ensure_initialized(self):
            lib.define(self.schema())
            self.register_meta()

        def load(self):
            return native

    m = types.ModuleType("cann_ops_transformer.op_builder")
    m.OpBuilder = Builder
    m.get_as_library = lambda: lib
    sys.modules["cann_ops_transformer"] = types.ModuleType("cann_ops_transformer")
    sys.modules[m.__name__] = m

    root = str(pathlib.Path(__file__).resolve().parents[1]) + "/"
    spec = importlib.util.spec_from_file_location(
        "review_compressor",
        root + "attention/compressor/torch_extension/compressor.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location(
        "review_quant_compressor",
        root + "attention/quant_compressor/torch_extension/quant_compressor.py",
    )
    module_quant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_quant)

    # ---------------------------------------------------------------- schema
    x = torch.empty((1, 1, 1024), device="meta", dtype=torch.bfloat16)
    w = torch.empty((512, 1024), device="meta", dtype=torch.bfloat16)
    state = torch.empty((2, 128, 1024), device="meta")
    ape = torch.empty((128, 512), device="meta")
    cache = torch.empty((2, 256, 1024), device="meta", dtype=torch.bfloat16)
    slots = torch.empty((1,), device="meta", dtype=torch.int32)

    inner_out = torch.ops.cann_ops_transformer._compressor_inference(
        x, w, w, state, ape, 128, None, None, None, None, cache,
        x_cache_slot_mapping=slots,
    )
    assert inner_out[0].shape == CMP_KV_SIZE, inner_out[0].shape

    for name in ("compressor", "quant_compressor", "_compressor_inference"):
        schema = getattr(torch.ops.cann_ops_transformer, name).default._schema
        args = {a.name: a for a in schema.arguments}
        assert args["state_cache"].alias_info.is_write, name
        assert args["x_cache"].alias_info.is_write, name

    # ---------------------------------------------- mutation index contract
    # Every schema argument up to and including the last write must be free of a
    # schema default. The ADInplaceOrView kernel of this mutable custom op bumps
    # the version of the mutated args by positional index (state_cache=3,
    # x_cache=10); a defaulted trailing positional would let an older dispatcher
    # drop it and index past the trimmed args tuple.
    inner_schema = (
        torch.ops.cann_ops_transformer._compressor_inference.default._schema
    )
    write_indices = [
        i
        for i, arg in enumerate(inner_schema.arguments)
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    assert write_indices == [3, 10], write_indices
    last_write = write_indices[-1]
    assert inner_schema.arguments[last_write].name == "x_cache", inner_schema
    for arg in inner_schema.arguments[: last_write + 1]:
        assert not arg.has_default_value(), (
            "inner arg {} must not have a schema default so the mutable "
            "version bump cannot index past the args tuple".format(arg.name)
        )

    # The pure-function forward and the now-unreachable training registrations
    # must be gone from this module.
    for removed in ("_compressor_forward", "_compressor_backward"):
        assert not hasattr(
            torch.ops.cann_ops_transformer, removed
        ), "{} should not be registered".format(removed)
    for removed_attr in ("setup_context", "_compressor_backward"):
        assert not hasattr(module, removed_attr), removed_attr

    with torch.no_grad():
        outer_out = torch.ops.cann_ops_transformer.compressor(
            x, w, w, state, ape, 128, x_cache=cache, x_cache_slot_mapping=slots
        )
    assert outer_out.shape == CMP_KV_SIZE, outer_out.shape

    print("SCHEMA_AND_META_PASS")

    # -------------------------------------- real dispatcher on CPU (test only)
    module._compressor_inference.register_kernel(
        "cpu", module._compressor_forward_impl
    )
    lib.impl("compressor", module.compressor, "CPU")

    def make():
        return (
            torch.empty((1, 1, 1024), dtype=torch.bfloat16),
            torch.empty((512, 1024), dtype=torch.bfloat16),
            torch.zeros((2, 128, 1024)),
            torch.empty((128, 512)),
            torch.zeros((2, 256, 1024)),
            torch.empty((1,), dtype=torch.int32),
        )

    def run_outer(x, w, state, ape, cache=None, slots=None):
        return torch.ops.cann_ops_transformer.compressor(
            x,
            w,
            w,
            state,
            ape,
            128,
            x_cache=cache,
            x_cache_slot_mapping=slots,
        )

    # Inner custom op through the real dispatcher: routes to the CPU kernel,
    # writes back into the very same objects, native sees grad_enabled=False.
    x, w, state, ape, cache, slots = make()
    state_id, cache_id = id(state), id(cache)
    native.calls.clear()
    inner = torch.ops.cann_ops_transformer._compressor_inference(
        x, w, w, state, ape, 128, None, None, None, None, cache,
        x_cache_slot_mapping=slots,
    )
    assert len(inner) == 3
    assert len(native.calls) == 1, native.calls
    assert len(native.calls[-1]) == 16, len(native.calls[-1])
    assert native.calls[-1][-1] is False, native.calls[-1][-1]
    assert id(state) == state_id and torch.equal(state, torch.ones_like(state))
    assert id(cache) == cache_id and torch.equal(cache, torch.ones_like(cache))

    # Grouped x_cache tensor path: default grad mode, no requires_grad.
    x, w, state, ape, cache, slots = make()
    state_id, cache_id = id(state), id(cache)
    native.calls.clear()
    out = run_outer(x, w, state, ape, cache, slots)
    assert out.shape == CMP_KV_SIZE, out.shape
    assert len(native.calls) == 1, native.calls
    assert native.calls[-1][-1] is False, native.calls[-1][-1]
    assert id(state) == state_id and torch.equal(state, torch.ones_like(state))
    assert id(cache) == cache_id and torch.equal(cache, torch.ones_like(cache))

    # Normal path: x_cache omitted (None) with the default coff / cache_mode;
    # state_cache is still written back in place and the native ABI stays the
    # full positional call with x_cache present as a legal None.
    x, w, state, ape, cache, slots = make()
    state_id = id(state)
    native.calls.clear()
    out = torch.ops.cann_ops_transformer.compressor(x, w, w, state, ape, 128)
    assert out.shape == CMP_KV_SIZE, out.shape
    assert len(native.calls) == 1, native.calls
    native_args = native.calls[-1]
    assert len(native_args) == 16, len(native_args)
    assert native_args[10] is None, native_args[10]
    assert native_args[11] is None, native_args[11]
    assert native_args[12] == 1 and native_args[13] == 1, native_args[11:]
    assert native_args[14] is True, native_args[14]
    assert native_args[-1] is False, native_args[-1]
    assert id(state) == state_id and torch.equal(state, torch.ones_like(state))

    # no_grad / inference_mode run even when an input requires grad.
    x, w, state, ape, cache, slots = make()
    x.requires_grad_(True)
    native.calls.clear()
    with torch.no_grad():
        run_outer(x, w, state, ape)
    with torch.inference_mode():
        run_outer(x, w, state, ape)
    assert len(native.calls) == 2, len(native.calls)
    assert all(call[-1] is False for call in native.calls), native.calls

    # x requiring grad under grad mode is rejected before the native call.
    x, w, state, ape, cache, slots = make()
    x.requires_grad_(True)
    native.calls.clear()
    try:
        run_outer(x, w, state, ape)
        raise AssertionError("x.requires_grad=True should have been rejected")
    except ValueError as err:
        assert "inference-only" in str(err), str(err)
        assert "torch.no_grad" in str(err), str(err)
    assert len(native.calls) == 0, native.calls

    # A weight requiring grad under grad mode is rejected before the native call.
    x, w, state, ape, cache, slots = make()
    wgrad = w.detach().clone().requires_grad_(True)
    native.calls.clear()
    try:
        torch.ops.cann_ops_transformer.compressor(x, wgrad, w, state, ape, 128)
        raise AssertionError("wkv.requires_grad=True should have been rejected")
    except ValueError as err:
        assert "wkv" in str(err), str(err)
    assert len(native.calls) == 0, native.calls

    print("INFERENCE_BEHAVIOUR_PASS")


def _run_graph_child():
    """Import the real ``graph_convert_compressor.py`` under a TorchAir stand-in.

    Everything here is a test-only stand-in for the NPU/TorchAir stack: the
    point is that the real module body (with ``_TORCHAIR_AVAILABLE`` true) runs
    its decorators against the real ``cann_ops_transformer`` dispatcher.
    """
    import importlib.util
    import pathlib
    import types

    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    torch.utils.rename_privateuse1_backend("npu")
    lib = torch.library.Library("cann_ops_transformer", "DEF")

    class Builder:
        def __init__(self, name, category=None):
            self.name = name

        def _ensure_initialized(self):
            lib.define(self.schema())
            self.register_meta()

        def load(self):
            return None

    m = types.ModuleType("cann_ops_transformer.op_builder")
    m.OpBuilder = Builder
    m.get_as_library = lambda: lib
    sys.modules["cann_ops_transformer"] = types.ModuleType("cann_ops_transformer")
    sys.modules[m.__name__] = m

    root = str(pathlib.Path(__file__).resolve().parents[1]) + "/"
    graph_path = (
        root + "attention/compressor/torch_extension/graph_convert_compressor.py"
    )

    # Register the real inference op so the surviving converter target exists,
    # then confirm the removed pure-forward op is actually gone.
    spec = importlib.util.spec_from_file_location(
        "review_compressor_graph",
        root + "attention/compressor/torch_extension/compressor.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not hasattr(torch.ops.cann_ops_transformer, "_compressor_forward")
    try:
        torch.ops.cann_ops_transformer._compressor_forward
        raise AssertionError("_compressor_forward should not exist")
    except AttributeError:
        pass

    # -------------------- minimal TorchAir stand-in ----------------------
    registered = []

    def register_fx_node_ge_converter(op):
        registered.append(op)

        def _deco(fn):
            return fn

        return _deco

    def _passthrough_decorator(*args, **kwargs):
        def _deco(fn):
            return fn

        return _deco

    def _pkg(name):
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
        return mod

    torchair = _pkg("torchair")
    ge = _pkg("torchair.ge")
    ge.attr = types.ModuleType("torchair.ge.attr")
    ge_graph = _pkg("torchair.ge._ge_graph")
    gc = _pkg("torchair._ge_concrete_graph")
    ge_apis = _pkg("torchair._ge_concrete_graph.ge_apis")
    fx2ge = _pkg("torchair._ge_concrete_graph.fx2ge_converter")
    supported = _pkg("torchair._ge_concrete_graph.supported_declaration")
    ge_ir = _pkg("torchair._ge_concrete_graph.ge_ir_pb2")
    compat = _pkg("torchair._ge_concrete_graph.compat_ir")
    torchair.ge = ge
    torchair._ge_concrete_graph = gc
    gc.ge_apis = ge_apis

    for name in ("Tensor", "TensorSpec", "DataType", "TensorType"):
        setattr(ge_graph, name, type(name, (), {}))
    ge_graph.get_default_ge_graph = lambda: None
    ge_graph.next_unique_name = lambda *a, **k: "name"
    ge_graph.auto_convert_to_tensor = _passthrough_decorator
    ge_graph.compat_as_bytes = lambda v: v
    ge_graph.compat_as_bytes_list = lambda v: v
    ge_graph.trans_to_list_list_int = lambda v: v
    ge_graph.trans_to_list_list_float = lambda v: v
    ge_graph.get_invalid_desc = lambda *a, **k: None

    fx2ge.declare_supported = _passthrough_decorator
    fx2ge.register_fx_node_ge_converter = register_fx_node_ge_converter
    supported.Support = object
    for name in ("GraphDef", "OpDef", "TensorDescriptor", "TensorDef"):
        setattr(ge_ir, name, type(name, (), {}))
    compat.ge_op = object
    compat.IrDef = object

    sys.modules["torch_npu"] = types.ModuleType("torch_npu")

    spec = importlib.util.spec_from_file_location(
        "review_graph_convert", graph_path
    )
    graph_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_mod)

    assert graph_mod._TORCHAIR_AVAILABLE is True, graph_mod._TORCHAIR_AVAILABLE
    inference_op = torch.ops.cann_ops_transformer._compressor_inference.default
    assert registered == [inference_op], [
        str(op) for op in registered
    ]
    for op in registered:
        assert "_compressor_forward" not in str(op), str(op)

    try:
        graph_mod.convert_compressor()
        raise AssertionError("converter should refuse GE graph mode")
    except AssertionError as err:
        assert "not supported in GE graph mode" in str(err), str(err)
        assert "aclgraph" in str(err), str(err)

    print("GRAPH_CONVERT_IMPORT_PASS")

    # -------------------- negative control (non-vacuous) -----------------
    # Re-insert the deleted decorator that referenced the removed op. Importing
    # that source must fail on the missing op, which is exactly the failure the
    # fix removes.
    src = pathlib.Path(graph_path).read_text()
    anchor = "    def convert_compressor("
    negative_src = src.replace(
        anchor,
        "    @register_fx_node_ge_converter(\n"
        "        torch.ops.cann_ops_transformer._compressor_forward.default\n"
        "    )\n" + anchor,
        1,
    )
    assert negative_src != src, "negative-control anchor not found"
    negative_path = "/tmp/pro0921_neg_graph_convert_compressor.py"
    pathlib.Path(negative_path).write_text(negative_src)

    spec = importlib.util.spec_from_file_location(
        "review_graph_convert_negative", negative_path
    )
    negative_mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(negative_mod)
        raise AssertionError("pre-fix source should have failed to import")
    except AttributeError as err:
        assert "_compressor_forward" in str(err), str(err)

    print("GRAPH_CONVERT_NEGATIVE_PASS")


if __name__ == "__main__" and "--child-graph" in sys.argv:
    _run_graph_child()
