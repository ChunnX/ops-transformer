"""
Thin wrapper around talk/qgmmd/test/generate_data.py that emits data for
one config into a staging directory. The existing generate_data.py
already produces every tensor variant run_one.py might load, so we just
invoke it as a subprocess.
"""
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
TEST_GEN = os.path.join(_REPO, "talk", "qgmmd", "test", "generate_data.py")


def data_is_present(datadir, cfg):
    """True if the cached data dir already covers this config's variant needs."""
    name = cfg["name"]
    cfg_txt = os.path.join(datadir, "config.txt")
    if not os.path.isdir(datadir):
        return False
    required = [
        "x.pth", "weight_int8.pth", "weight_scale.pth",
        "weight_scale_u64.pth", "group_list.pth", "golden.pth",
    ]
    if cfg["variant"] == "dynamic_smooth":
        required.append("smooth_scale.pth")
    for fname in required:
        if not os.path.exists(os.path.join(datadir, fname)):
            return False
    if os.path.exists(cfg_txt):
        with open(cfg_txt) as f:
            content = f.read().strip()
        want = f"{name}"
        if want not in content:
            return False
    return True


def generate_one(cfg, datadir, regenerate=False):
    """Invoke talk/qgmmd/test/generate_data.py for this (shape, variant)."""
    os.makedirs(datadir, exist_ok=True)
    name = cfg["name"]
    cmd = [
        sys.executable, TEST_GEN,
        "--config", name,
        "--outdir", datadir,
    ]
    if cfg["variant"] == "dynamic_smooth":
        cmd.append("--smooth")
    env = dict(os.environ)
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise RuntimeError(
            f"generate_data failed for {name}/{cfg['variant']}: "
            f"{res.stdout.decode(errors='replace')[-2000:]}"
        )
    with open(os.path.join(datadir, "config.txt"), "w") as f:
        f.write(f"{name}\n")
