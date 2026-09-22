"""Load checkout indexers with an installed or configured Torch base package."""

import hashlib
import importlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
OPERATORS = {"quant_lightning_indexer": "qli", "quant_sparse_lightning_indexer": "qsli"}


class CheckoutIndexerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for operator, short_name in OPERATORS.items():
            prefix = "cann_ops_transformer.ops.attention." + operator + "_dsl"
            directory = ROOT / (operator + "_dsl") / "torch_extension"
            if fullname == prefix:
                return importlib.util.spec_from_file_location(
                    fullname, directory / "__init__.py", submodule_search_locations=[str(directory)]
                )
            if fullname == prefix + "." + operator:
                return importlib.util.spec_from_file_location(fullname, directory / (operator + ".py"))
            for suffix in ("_dsl", "_metadata_dsl"):
                if fullname == "ops." + operator + suffix:
                    directory = dsl_directory(short_name)
                    return importlib.util.spec_from_file_location(fullname, directory / (operator + suffix + ".py"))
        return None


def dsl_directory(short_name):
    external = os.environ.get("CANNBOTDSL_TEST_OPS_DIR")
    if not external:
        raise RuntimeError("Set CANNBOTDSL_TEST_OPS_DIR to the kernel checkout net/ops directory")
    directory = Path(external).resolve()
    if not directory.is_dir():
        raise RuntimeError(f"Missing kernel directory: {directory}")
    return directory


def configure():
    if any(isinstance(finder, CheckoutIndexerFinder) for finder in sys.meta_path):
        return
    base = os.environ.get("OPS_TRANSFORMER_TORCH_EXTENSION_DIR")
    if base:
        package = Path(base).resolve() / "cann_ops_transformer"
    else:
        spec = importlib.util.find_spec("cann_ops_transformer")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("Install cann_ops_transformer or set OPS_TRANSFORMER_TORCH_EXTENSION_DIR to its source package parent")
        package = Path(next(iter(spec.submodule_search_locations)))
    if not (package / "op_builder" / "__init__.py").is_file():
        raise ImportError(f"Missing op_builder in {package}")
    for name, directory in (
        ("cann_ops_transformer", package),
        ("cann_ops_transformer.ops", package / "ops"),
        ("cann_ops_transformer.ops.attention", ROOT),
        ("ops", dsl_directory("qli")),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(directory)]
            module.__package__ = name
            sys.modules[name] = module
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, module)
    sys.meta_path.insert(0, CheckoutIndexerFinder())


def report_sources(short_name):
    operator = next(name for name, short in OPERATORS.items() if short == short_name)
    module = importlib.import_module("cann_ops_transformer.ops.attention." + operator + "_dsl." + operator)
    assert Path(module.__file__).resolve().parent == (ROOT / (operator + "_dsl") / "torch_extension").resolve()
    print("TORCH_INTERFACE", module.__file__, flush=True)
    for suffix in ("_dsl", "_metadata_dsl"):
        module = importlib.import_module("ops." + operator + suffix)
        source = Path(module.__file__).resolve()
        assert source.parent == dsl_directory(short_name).resolve(), str(source)
        print("DSL_SOURCE", source, hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)


def check_graph_sequence(function, initial, updated, backend, check_output, output_dir):
    import json
    import traceback
    import torch
    import torch_npu

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {"backend": backend, "status": "running", "executions": [], "backend_entries": 0,
                "python": sys.executable, "conda_prefix": os.environ.get("CONDA_PREFIX"),
                "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "device": str(torch.npu.get_device_properties(torch.npu.current_device())),
                "fullgraph": backend != "acl_graph", "dynamic": False, "suppress_errors": False}
    tensors = {name: value.npu() if isinstance(value, torch.Tensor) else value
               for name, value in initial.items()}
    identity = {name: (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype))
                for name, value in tensors.items() if isinstance(value, torch.Tensor)}
    evidence["inputs"] = {phase: {name: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                  for name, value in inputs.items() if isinstance(value, torch.Tensor)}
                          for phase, inputs in (("initial", initial), ("updated", updated))}
    original_replay = torch.npu.NPUGraph.replay
    replay_count = 0
    ge_class = None
    ge_run = None
    ge_compile = None
    evidence["ge_runs"] = 0
    evidence["ge_compiles"] = 0

    def record_replay(graph):
        nonlocal replay_count
        replay_count += 1
        return original_replay(graph)

    torch.npu.NPUGraph.replay = record_replay
    try:
        for warmup in range(3):
            function(tensors)
        torch.npu.synchronize()
        if backend == "acl_graph":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = function(tensors)
            evidence["capture_complete"] = True

            def execute(inputs):
                graph.replay()
                return captured
        else:
            torch._dynamo.reset()
            torch._dynamo.config.suppress_errors = False
            if backend == "npugraph_ex":
                from torch._dynamo.backends.registry import lookup_backend
                compiler = lookup_backend("npugraph_ex")
            elif backend == "ge":
                import torchair
                from torchair.configs.compiler_config import CompilerConfig
                from torchair.core._backend import TorchNpuGraph
                ge_class = TorchNpuGraph
                ge_run, ge_compile = ge_class.run, ge_class.compile

                def record_ge_run(graph, *args, **kwargs):
                    result = ge_run(graph, *args, **kwargs)
                    evidence["ge_runs"] += 1
                    return result

                def record_ge_compile(graph, *args, **kwargs):
                    result = ge_compile(graph, *args, **kwargs)
                    evidence["ge_compiles"] += 1
                    return result

                ge_class.run, ge_class.compile = record_ge_run, record_ge_compile
                config = CompilerConfig()
                config.ge_config.aicore_num = str(torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num)
                evidence["ge_aicore_num"] = str(config.ge_config.aicore_num.value)
                config.debug.graph_dump.type = "pbtxt"
                config.debug.graph_dump.path = str(output_dir.resolve())
                compiler = torchair.get_npu_backend(compiler_config=config)
            else:
                raise ValueError(backend)

            def recorded_backend(graph_module, example_inputs):
                evidence["backend_entries"] += 1
                (output_dir / f"fx_{evidence['backend_entries']}.txt").write_text(str(graph_module.graph))
                return compiler(graph_module, example_inputs)

            execute = torch.compile(function, backend=recorded_backend, fullgraph=True, dynamic=False)
            execute(tensors)
            torch.npu.synchronize()
        for phase, inputs in (("initial", initial), ("updated", updated)):
            if phase == "updated":
                for name, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        tensors[name].copy_(value)
                torch.npu.synchronize()
            for iteration in range(3):
                device_start = torch.npu.Event(enable_timing=True)
                device_end = torch.npu.Event(enable_timing=True)
                device_start.record()
                result = execute(tensors)
                device_end.record()
                torch.npu.synchronize()
                device_elapsed_ms = device_start.elapsed_time(device_end)
                assert device_elapsed_ms > 0
                cpu = tuple(value.cpu().clone() for value in result)
                check_output(phase, cpu)
                evidence["executions"].append({"phase": phase, "iteration": iteration, "status": "Pass",
                                               "device_event_elapsed_ms": device_elapsed_ms})
        assert identity == {name: (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype))
                            for name, value in tensors.items() if isinstance(value, torch.Tensor)}
        if backend in ("acl_graph", "npugraph_ex"):
            assert replay_count >= 6, f"only {replay_count} native replays observed"
        if backend != "acl_graph":
            assert evidence["backend_entries"] == 1, "unexpected recompilation"
        if backend == "ge":
            assert evidence["ge_compiles"] > 0 and evidence["ge_runs"] >= 7
        evidence["status"] = "Pass"
    except Exception:
        evidence["status"] = "Fail"
        evidence["error"] = traceback.format_exc()
        raise
    finally:
        torch.npu.NPUGraph.replay = original_replay
        if ge_class is not None:
            ge_class.run, ge_class.compile = ge_run, ge_compile
        evidence["native_replay_count"] = replay_count
        (output_dir / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
