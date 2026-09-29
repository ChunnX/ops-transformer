"""Serial hardware regression. A generic failure or host watchdog is NOT a reproduction."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def classify(text, code, audit, expected_timeout):
    sites = []
    for kind, start, current in re.findall(
            r"fftsplus (aicore|aivector) error[^\n]*pc start: (0x[0-9a-f]+), current: (0x[0-9a-f]+)", text):
        core = "aic" if kind == "aicore" else "aiv"
        data = audit["cores"][core]
        offset = int(current, 16) - int(start, 16) + data["symbol_offset"]
        wait = next((w for w in data["waits"] if w["offset"] == offset), None)
        sites.append(dict(core=core, text_offset=hex(offset), wait=wait))
    launched = "[SLIG] starting runs=" in text
    device_timeout = code not in (0, 124, 137) and "507014" in text and launched
    matched_wait = any(site["wait"] is not None for site in sites)
    passed = code == 0 and "[SLIG] RESULT=PASS" in text
    reproduced = device_timeout and matched_wait
    # Single-round same-ID is observational: it has both passed and timed out
    # with the identical ELF. Never call arbitrary failures observations.
    ok = (passed or reproduced) if expected_timeout is None else (reproduced if expected_timeout else passed)
    outcome = "reproduced" if reproduced else "passed" if passed else "unexpected_failure"
    if expected_timeout and passed:
        outcome = "not_reproduced"
    return dict(ok=ok, outcome=outcome, returncode=code, pc_sites=sites)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--extended", action="store_true", help="also test 1 and 1024 rounds")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--op-timeout-s", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1 or args.repeats < 1 or args.device < 0 or not 1 <= args.op_timeout_s <= 1000:
        parser.error("invalid runs/repeats/device/timeout")
    work = Path(os.environ.get("SLIG_WORK_DIR", ROOT / ".klloss_repro")).resolve()
    rounds = [1, 2, 1024] if args.extended else [2]
    if not args.skip_build:
        subprocess.run(["bash", str(ROOT / "scripts/klloss/compile.sh"), "pipeline"], check=True)
        for n in rounds:
            for variant in ("same", "split"):
                subprocess.run(["bash", str(ROOT / "scripts/klloss/compile.sh"), variant, str(n)], check=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    output = work / "results" / stamp
    output.mkdir(parents=True, exist_ok=False)
    cases = [("pipeline", None, True)]
    cases += [(v, n, None if v == "same" and n == 1 else v == "same")
              for n in rounds for v in ("same", "split")]
    summary = {"device": args.device, "runs": args.runs, "timeout_s": args.op_timeout_s,
               "repeats": args.repeats, "completed": False, "extended": args.extended, "results": []}
    for variant, n, expected in cases:
        name = "pipeline" if n is None else f"{variant}_r{n}"
        manifest = json.loads((work / name / "manifest.json").read_text())
        for attempt in range(1, args.repeats + 1):
            command = ["bash", str(ROOT / "scripts/klloss/run.sh"), variant]
            if n is not None:
                command.append(str(n))
            command += ["--runs", str(args.runs), "--op-timeout-s", str(args.op_timeout_s), "--device", str(args.device)]
            policy = "observe pass or matched-Wait timeout" if expected is None else "device timeout at Wait" if expected else "pass"
            print(f"[MATRIX] {name} attempt={attempt} expected={policy}", flush=True)
            log = output / f"{name}_{attempt}.log"
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            text = log.read_text()
            row = classify(text, result.returncode, manifest["audit"], expected)
            runtime_match = re.search(r"\[SLIG\] RUNTIME_DIR=([^\n]+)", text)
            row["runtime_dump_sha256"] = None
            if runtime_match:
                binary_name = Path(manifest["audit"]["binary"]).name
                for dumped in Path(runtime_match[1]).glob(f"extra-info/data-dump/*/{binary_name}"):
                    row["runtime_dump_sha256"] = hashlib.sha256(dumped.read_bytes()).hexdigest()
                    if row["runtime_dump_sha256"] != manifest["audit"]["sha256"]:
                        row.update(ok=False, outcome="binary_identity_mismatch")
            row.update(case=name, attempt=attempt, expected_timeout=expected, log=log.name,
                       log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                       binary_sha256=manifest["audit"]["sha256"])
            summary["results"].append(row)
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"[MATRIX] {name}: {row['outcome']} PC={row['pc_sites']} log={log}", flush=True)
            if not row["ok"]:
                print("[MATRIX] stopped: unexpected outcome; inspect logs, no automatic reset or retry", flush=True)
                return 1
    summary["completed"] = True
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[MATRIX] VERIFIED {len(summary['results'])} independent processes; {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
