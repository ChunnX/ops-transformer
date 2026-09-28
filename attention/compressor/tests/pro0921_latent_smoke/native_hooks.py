# Local TTK device-readback hooks, installed in the worker by plugin.py.
import json
import logging
import os
from pathlib import Path

def install_readback_hook(case_dir: Path):
    """Opt-in local hook: read the declared inputs back from device memory.

    TTK only returns outputs in result.csv, so a mutated *input* (for example
    x_cache, which the grouped path refreshes in place) cannot be observed from
    the standard flow.  `OnlineRtsProfiling._copy_output_from_hbm` runs while
    the input device buffers are still allocated and their addresses are known
    in `_nd_array_maps`, so wrapping it lets this harness copy those buffers to
    disk without touching the shared toolkit.
    """
    spec_path = case_dir / "readback_spec.json"
    if not spec_path.exists():
        return
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not spec:
        return
    out_dir = case_dir / "readback"
    out_dir.mkdir(parents=True, exist_ok=True)

    from ttk.core_modules.npu.op import rts_sequence
    from ttk.utilities import get_dtype_width

    original = rts_sequence.OnlineRtsProfiling._copy_output_from_hbm

    def patched(self):
        result = original(self)
        try:
            flat_inputs = list(self._param.flatten_input_arrays)
            for entry in spec:
                index = int(entry["index"])
                if index >= len(flat_inputs):
                    continue
                array = flat_inputs[index]
                if array is None:
                    continue
                address = self._nd_array_maps.get(id(array))
                if address is None:
                    continue
                size = int(__import__("math").ceil(array.size * get_dtype_width(array.dtype)))
                data = self._device.get_data_from_hbm(address, size)
                (out_dir / (entry["name"] + ".bin")).write_bytes(bytes(data))
                (out_dir / (entry["name"] + ".meta.json")).write_text(json.dumps({
                    "dtype": str(array.dtype),
                    "shape": [int(v) for v in array.shape],
                    "logical_name": entry["name"],
                }), encoding="utf-8")
        except Exception:
            logging.exception("PRO0921 read-back hook failed (case outputs are unaffected)")
        return result

    rts_sequence.OnlineRtsProfiling._copy_output_from_hbm = patched



def install_execution_receipt(case_dir):
    # TTK calls this only after stream synchronization succeeds. CAMODEL's
    # duration parser returns UNKNOWN, so record completion without inventing us.
    from ttk.core_modules.npu.op import rts_sequence
    receipt = case_dir / "native_execution.json"
    original = rts_sequence.OnlineRtsProfiling._copy_output_from_hbm

    def completed(self):
        result = original(self)
        receipt.write_text(json.dumps({
            "stream_synchronized": True,
            "outputs_copied_from_device": True,
            "kernel_name": self._param.kernel_name,
            "tiling_key": int(self._param.tiling_key),
            "block_dim": int(self._param.block_dim),
        }, indent=2), encoding="utf-8")
        return result

    rts_sequence.OnlineRtsProfiling._copy_output_from_hbm = completed



def install():
    from ttk.core_modules.npu.op import rts_sequence
    cls = rts_sequence.OnlineRtsProfiling
    if getattr(cls, "_pro0921_hooks_installed", False):
        return
    case_dir = Path(os.environ["PRO0921_CASE_DIR"])
    if os.environ.get("PRO0921_READBACK_XCACHE") == "1":
        install_readback_hook(case_dir)
    install_execution_receipt(case_dir)
    cls._pro0921_hooks_installed = True
