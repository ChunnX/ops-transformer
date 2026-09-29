"""Load only this repository's adapter, verify build identity, then run the smoke test."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
OP = "mlp_sparse_lightning_indexer_grad_kl_loss"


def load_adapter():
    # Avoid eager-importing unrelated installed entry-point/JIT operators.
    package_dir = ROOT / "torch_extension/cann_ops_transformer"
    if "cann_ops_transformer" in sys.modules:
        raise RuntimeError("a fresh process is required")
    for suffix, directory in (("", package_dir), (".ops", package_dir / "ops")):
        module = types.ModuleType("cann_ops_transformer" + suffix)
        module.__path__ = [str(directory)]
        module.__file__ = str(directory / "__init__.py")
        sys.modules[module.__name__] = module
    source = ROOT / "attention" / OP / "torch_extension" / (OP + ".py")
    spec = importlib.util.spec_from_file_location("slig_local_adapter", source)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    setattr(sys.modules["cann_ops_transformer.ops"], OP, getattr(adapter, OP))
    return adapter


def main():
    marker = Path(sys.argv[1]).resolve()
    manifest = json.loads(marker.read_text())
    if manifest.get("owner") != "klloss-source-reproducer-v1" or not manifest.get("compiled"):
        raise RuntimeError("incomplete or unrecognized build; compile first")
    if not manifest.get("disassembly_verified"):
        raise RuntimeError("run audit_binary.py on this manifest with SLIG_OBJDUMP set before executing")
    vendor = Path(manifest["vendor"])
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH") != str(vendor):
        raise RuntimeError("the generated OPP must be the only custom package in this process")
    for relative, expected in manifest["artifacts"].items():
        actual = hashlib.sha256((vendor / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"artifact changed since build: {relative}")
    print(f"[SLIG] VERIFIED variant={manifest['variant']} rounds={manifest['rounds']} "
          f"bf16_sha256={manifest['audit']['sha256']} vendor={vendor}", flush=True)
    runtime_dir = marker.parent / "runtime" / str(os.getpid())
    runtime_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(runtime_dir)  # Keep CANN exception dumps out of the source tree.
    print(f"[SLIG] RUNTIME_DIR={runtime_dir}", flush=True)
    adapter = load_adapter()
    adapter._builder.load(verbose=True)
    if sys.argv[2:] == ["--build-extension"]:
        print("[SLIG] EXTENSION_READY", flush=True)
        return
    script = ROOT / "attention" / OP / "examples" / ("test_" + OP + ".py")
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")
    print("[SLIG] RESULT=PASS", flush=True)


if __name__ == "__main__":
    main()
