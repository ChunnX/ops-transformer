"""Audit the installed BF16 ELF, including compiler-inserted CrossCore operations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def set_messages(body):
    """Resolve literal Set operands in this tiny straight-line loop body only.

    Unknown register writes invalidate constants. This is deliberately not a
    general control-flow/data-flow engine; unfamiliar kernels must fail closed.
    """
    registers, messages = {}, []
    for offset, op, operands in re.findall(r"(?m)^\s*([0-9a-f]+):\s+[0-9a-f]{8}\s+(\S+)\s*([^\n]*)", body):
        args = [arg.strip() for arg in operands.split(" <")[0].split(",")]
        if op == "MOV" and len(args) == 2 and re.fullmatch(r"X\d+", args[0]):
            registers[args[0]] = int(args[1][1:], 0) if re.fullmatch(r"#(?:0x[0-9a-f]+|\d+)", args[1]) else registers.get(args[1])
        elif op.startswith("SET_CROSS_CORE."):
            message = registers.get(args[0])
            if message is None:
                raise RuntimeError(f"cannot resolve Set message at {offset}; inspect compiler output")
            messages.append(dict(offset=int(offset, 16), pipe=op.split(".")[1], message=message,
                                 mode=(message >> 4) & 3, flag=(message >> 8) & 15))
        elif args and re.fullmatch(r"X\d+", args[0]) and not op.startswith(
                ("CMP", "JUMP", "ST.", "WAIT_FLAG", "DCCI", "PRELOAD")):
            registers[args[0]] = None
    return messages


def audit(case, manifest):
    vendor = Path(manifest["vendor"])
    metadata = []
    for path in vendor.glob("op_impl/ai_core/tbe/kernel/ascend910b/*/*.json"):
        info = json.loads(path.read_text())
        inputs = info.get("supportInfo", {}).get("inputs", [])
        if inputs and inputs[0].get("dtype") == "bfloat16":
            metadata.append((path, info))
    if len(metadata) != 1:
        raise RuntimeError(f"expected one BF16 ELF descriptor, got {len(metadata)}")
    path, info = metadata[0]
    binary = path.with_suffix(".o")
    sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    if sha != info["sha256"]:
        raise RuntimeError("ELF hash does not match compiler metadata")
    objdump = os.environ.get("SLIG_OBJDUMP", str(Path(manifest["cann"]) / "compiler/ccec_compiler/bin/llvm-objdump"))
    result = subprocess.run([objdump, "-d", "--mcpu=dav-c220", str(binary)],
                            capture_output=True, text=True, check=True)
    disasm = result.stdout
    if "WAIT_FLAG_DEV" not in disasm:
        raise RuntimeError("objdump cannot decode this ELF; set SLIG_OBJDUMP to a dav-c220 capable decoder (tested: CANN 9.2)")
    sections = re.split(r"(?m)^([0-9a-f]+) <[^\n]+_mix_(aic|aiv)>:\s*$", disasm)
    if len(sections) != 7:
        raise RuntimeError("unexpected MIX symbols; inspect disassembly before claiming protocol equivalence")
    summary = {"binary": str(binary.relative_to(vendor)), "sha256": sha, "cores": {}}
    for index in (1, 4):
        start, core, body = sections[index:index + 3]
        waits = [{"offset": int(m[0], 16), "operand": m[1].split(" <")[0].strip()}
                 for m in re.findall(r"(?m)^\s*([0-9a-f]+):\s+[0-9a-f]{8}\s+WAIT_FLAG_DEV\s+([^\n]+)", body)]
        sets = re.findall(r"SET_CROSS_CORE\.(\w+)", body)
        summary["cores"][core] = {"symbol_offset": int(start, 16), "waits": waits, "set_pipes": sets}
        if manifest["variant"] != "pipeline":
            rounds = manifest["rounds"]
            if len(waits) not in (1, rounds):
                raise RuntimeError(f"unexpected {core} Wait count: {len(waits)}")
            expected_flag = 0 if core == "aic" else manifest["vv_flag"]
            if any(wait["operand"] != f"#{expected_flag}" for wait in waits):
                raise RuntimeError(f"unexpected {core} Wait flag")
            expected_sets = [] if core == "aic" else ["MTE3", "MTE2"] * len(waits)
            if sets != expected_sets:
                raise RuntimeError(f"unexpected {core} notifications: {sets}")
            messages = set_messages(body)
            expected_messages = [] if core == "aic" else [(2, 0), (0, manifest["vv_flag"])] * len(waits)
            if [(m["mode"], m["flag"]) for m in messages] != expected_messages:
                raise RuntimeError(f"unexpected {core} Set mode/flag encoding: {messages}")
            summary["cores"][core]["set_messages"] = messages
    (case / "kernel.disasm").write_text(disasm)
    (case / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["audit"] = summary
    manifest["disassembly_verified"] = True
    print(f"[AUDIT] BF16={sha} protocol={manifest['variant']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    audit(args.manifest.parent, data)
    args.manifest.write_text(json.dumps(data, indent=2) + "\n")
