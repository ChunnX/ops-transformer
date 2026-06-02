"""
Run ONE qgmmd device test inside a `msprof op` wrap.

Produces an OPPROF_* directory in the current working directory (msprof
lands it there) and writes a JSON result file with cosine similarity,
pass/fail, max/mean abs error, and wall-ms for the kernel call.

Typically invoked by run_device.py via:
    msprof op --kernel-name=QuantGroupedMatmulDequant \
        python3 run_one.py \
        --datadir <path> --variant <name> --output-json <path>
"""
import argparse
import json
import os
import sys
import time
import traceback


def cosine_similarity(a, b):
    import numpy as np
    af = a.cpu().numpy().astype(np.float32).flatten()
    bf = b.cpu().numpy().astype(np.float32).flatten()
    ab = float((af * bf).sum())
    aa = float((af * af).sum()) ** 0.5
    bb = float((bf * bf).sum()) ** 0.5
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return ab / (aa * bb)


def load(datadir, name, required=True):
    import torch
    import numpy as np
    pth = os.path.join(datadir, f"{name}.pth")
    npy = os.path.join(datadir, f"{name}.npy")
    if os.path.exists(pth):
        return torch.load(pth, map_location="cpu", weights_only=False)
    if os.path.exists(npy):
        return torch.from_numpy(np.load(npy))
    if required:
        raise FileNotFoundError(f"No {name}.pth or {name}.npy in {datadir}")
    return None


def _ensure_bins(datadir):
    """Idempotent: dump .bin alongside any .pth that doesn't have one yet.
    Used by the aclnn engine to read raw tensor bytes without torch import."""
    import torch
    if not os.path.isdir(datadir):
        return
    for entry in os.listdir(datadir):
        if not entry.endswith(".pth"):
            continue
        pth = os.path.join(datadir, entry)
        bin_path = pth[:-4] + ".bin"
        if os.path.exists(bin_path):
            continue
        try:
            t = torch.load(pth, weights_only=True, map_location="cpu")
        except TypeError:
            t = torch.load(pth, map_location="cpu")
        if not isinstance(t, torch.Tensor):
            continue
        t.contiguous().numpy().tofile(bin_path)


def _run_aclnn(args):
    """Invoke device_tests/aclnn_bench/build/run_aclnn and translate JSON
    output into the same result schema as the torch_npu engine."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    binary = os.path.join(here, "aclnn_bench", "build", "run_aclnn")
    if not os.path.exists(binary):
        json.dump({"cos": 0.0, "pass": False, "wall_ms": 0.0,
                   "error": f"missing binary: {binary}"},
                  open(args.output_json, "w"))
        print(f"[ERR] {binary} not found — build via aclnn_bench/build.sh")
        sys.exit(1)

    _ensure_bins(args.datadir)

    # Derive shape (E, M, K, N) from .pth metadata. Staged data_stage.py
    # writes a stripped config.txt (just the shape name), not E/M/K/N.
    # Pass explicit --E/--M/--K/--N to run_aclnn.
    import torch
    x_pth = torch.load(os.path.join(args.datadir, "x.pth"),
                       map_location="cpu", weights_only=True)
    gl_pth = torch.load(os.path.join(args.datadir, "group_list.pth"),
                        map_location="cpu", weights_only=True)
    g_pth = torch.load(os.path.join(args.datadir, "golden.pth"),
                       map_location="cpu", weights_only=True)
    M_, K_ = x_pth.shape[-2], x_pth.shape[-1]
    E_ = gl_pth.shape[0]
    N_ = g_pth.shape[-1]

    json_tmp = args.output_json + ".tmp"
    cmd = [binary,
           "--datadir", args.datadir,
           "--variant", args.variant,
           "--iterations", str(args.iterations),
           "--warmup", str(args.warmup),
           "--E", str(int(E_)),
           "--M", str(int(M_)),
           "--K", str(int(K_)),
           "--N", str(int(N_)),
           "--output-json", json_tmp]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0 or not os.path.exists(json_tmp):
        json.dump({"cos": 0.0, "pass": False, "wall_ms": wall * 1000.0,
                   "error": f"run_aclnn rc={proc.returncode}"},
                  open(args.output_json, "w"))
        sys.exit(proc.returncode or 1)

    raw = json.load(open(json_tmp))
    os.remove(json_tmp)
    cos = float(raw.get("cos", 0.0))
    t = raw.get("time_us", {})
    out = {
        "cos": cos,
        "pass": cos > 0.9999,
        "wall_ms": wall * 1000.0,
        "task_dur_us": float(t.get("med", 0.0)),
        "task_dur_min_us": float(t.get("min", 0.0)),
        "task_dur_mean_us": float(t.get("mean", 0.0)),
        "task_dur_max_us": float(t.get("max", 0.0)),
        "samples": raw.get("samples", []),
        "engine": "aclnn",
    }
    json.dump(out, open(args.output_json, "w"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", required=True)
    p.add_argument("--variant", required=True,
                   choices=("dynamic", "dynamic_u64", "dynamic_smooth"))
    p.add_argument("--output-json", required=True)
    p.add_argument("--engine", default="torch", choices=("torch", "aclnn"),
                   help="backend: torch_npu (default) or standalone aclnn "
                        "binary at device_tests/aclnn_bench/build/run_aclnn")
    p.add_argument("--iterations", type=int, default=20,
                   help="aclnn-only: timed iterations per call")
    p.add_argument("--warmup", type=int, default=3,
                   help="aclnn-only: warmup iterations")
    args = p.parse_args()

    if args.engine == "aclnn":
        _run_aclnn(args)
        return 0

    import torch
    import torch_npu

    # Pin to logical device 0. With ASCEND_RT_VISIBLE_DEVICES=N (e.g. 1),
    # logical 0 maps to physical chip N. Without this explicit call,
    # torch_npu's lazy-init can still talk to chip 0 even when the env
    # var remaps the visible set, causing a leak onto a shared chip.
    torch_npu.npu.set_device(0)

    use_u64 = args.variant == "dynamic_u64"
    use_smooth = args.variant == "dynamic_smooth"

    # Load inputs (generate_data.py output in talk/qgmmd/test/data/<cfg>/).
    # qgmmd's torch_npu wrapper rejects internal-format (FRACTAL_NZ) tensors:
    #   "Current operator npu_quant_grouped_matmul_dequant do not support
    #    internal format. ERR00007 PTA feature not supported"
    # so we pass ND weight [E, N, K] and rely on the host aclnn TransData.
    # (qbmv3 supports the FRACTAL_NZ upload trick; qgmmd does not.)
    x_fp16 = load(args.datadir, "x")
    weight_int8_nd = load(args.datadir, "weight_int8")     # [E, K, N]
    weight_scale = load(args.datadir,
                        "weight_scale_u64" if use_u64 else "weight_scale")
    group_list = load(args.datadir, "group_list")
    smooth_scale = load(args.datadir, "smooth_scale", required=False) \
        if use_smooth else None
    # The smooth variant needs a golden that was computed WITH smooth applied.
    # generate_data.py (post-2026-04-29) writes golden_smooth.pth alongside
    # golden.pth — keep the dynamic_smooth comparison apples-to-apples.
    # If the smooth golden file is missing (data generated by an older script),
    # fall back to golden.pth and surface a clear error.
    if use_smooth:
        golden = load(args.datadir, "golden_smooth", required=False)
        if golden is None:
            golden = load(args.datadir, "golden")
            print(
                "WARN: golden_smooth.pth not found — falling back to golden.pth. "
                "Re-run generate_data.py to produce golden_smooth.pth so the "
                "dynamic_smooth comparison is meaningful (cos otherwise sits "
                "near 99.3 due to mismatched reference, not a kernel bug)."
            )
    else:
        golden = load(args.datadir, "golden")

    # Move to NPU. ND upload only — operator transposes internally.
    x_n = x_fp16.to(torch.float16).npu()
    qw_n = weight_int8_nd.transpose(-1, -2).contiguous() \
        .to(torch.int8).npu()
    if use_u64:
        # 2026-04-30 — DIAGNOSTIC RESULT: clean kernel (no QGMMD_DEBUG_DEVICE
        # printf in hot path) makes 16/16 LIGHT shapes PASS at cos=100 with
        # int64 weight_scale passed directly. The earlier cos=0.993 plateau
        # was caused by printf debug calls perturbing pipe timing.
        # HEAVY shapes still have:
        #   (B) aclnn rejection on prefill_med/big, ffn_tall, ffn_tall_md
        #       ("expected DT_FLOAT but found DT_INT64") — host-side dispatch
        #       issue, not kernel issue.
        #   (C) ffn_fat cos=91.79 — separate numerical issue.
        # Workaround kept until issues B and C are resolved; LIGHT u64 path
        # now works without it. To re-test int64 kernel path: comment out
        # the import/decode block and just `ws_n = weight_scale.to(torch.int64).npu()`.
        import numpy as np
        ws_i64 = weight_scale.to(torch.int64).numpy().astype(np.int64)
        ws_i32 = ws_i64.view(np.int32).reshape(*ws_i64.shape, 2)
        ws_lower32 = ws_i32[..., 0].copy()
        ws_fp32 = ws_lower32.view(np.float32)
        ws_n = torch.from_numpy(ws_fp32.copy()).to(torch.float32).npu()
    else:
        ws_n = weight_scale.to(torch.float32).npu()
    gl_n = group_list.to(torch.int64).npu()
    ss_n = smooth_scale.to(torch.float16).npu() if smooth_scale is not None \
        else None

    # Warmup (1 pass — msprof captures the second call).
    t0 = time.time()
    try:
        _ = torch_npu.npu_quant_grouped_matmul_dequant(
            x_n, qw_n, ws_n, gl_n,
            bias=None, x_scale=None, x_offset=None, smooth_scale=ss_n,
            quant_mode="pertoken",
        )
        torch.npu.synchronize()
    except Exception as e:
        dt = time.time() - t0
        out = {"error": str(e), "traceback": traceback.format_exc(),
               "cos": 0.0, "pass": False, "wall_ms": dt * 1e3,
               "stage": "warmup"}
        with open(args.output_json, "w") as f:
            json.dump(out, f)
        return 1

    t1 = time.time()
    try:
        result = torch_npu.npu_quant_grouped_matmul_dequant(
            x_n, qw_n, ws_n, gl_n,
            bias=None, x_scale=None, x_offset=None, smooth_scale=ss_n,
            quant_mode="pertoken",
        )
        torch.npu.synchronize()
    except Exception as e:
        dt = time.time() - t1
        out = {"error": str(e), "traceback": traceback.format_exc(),
               "cos": 0.0, "pass": False, "wall_ms": dt * 1e3,
               "stage": "measure"}
        with open(args.output_json, "w") as f:
            json.dump(out, f)
        return 1

    dt_measure = time.time() - t1
    output = result.cpu()

    cos = cosine_similarity(output, golden)
    diff = (output.float() - golden.float()).abs()
    # Optional: save device output for debug (env QGMMD_SAVE_OUTPUT=1).
    # Saves output.pth alongside result.json so row_diff.py / phase_mm_oracle
    # can compare device vs golden per-element when cos < threshold.
    if os.environ.get("QGMMD_SAVE_OUTPUT", "") == "1":
        out_path = os.path.join(os.path.dirname(args.output_json), "output.pth")
        import torch as _t
        _t.save(output, out_path)
    out = {
        "cos": cos,
        "pass": cos > 0.9999,
        "wall_ms": dt_measure * 1e3,
        "max_err": float(diff.max()),
        "mean_err": float(diff.mean()),
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
