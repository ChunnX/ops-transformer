"""合并全量精度扫描和尾块补齐复测；逐条保存原始来源，不改写测得的状态。"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.resolve()
    original = raw / "precision_before_tail_padding"
    retest = raw / "precision_tail_aligned"
    output = raw / "precision"
    assert not output.exists(), "使用新的精度输出目录，保留原始实测文件"
    cases = read(original / "case_manifest.json")
    before = {x["test_name"]: x for x in read(original / "results.json")}
    after = {x["test_name"]: x for x in read(retest / "results.json")}
    changed = {x["test_name"] for x in cases
               if x["dk"] == x["dv"] == "128" and int(x["seqlen"]) % 64}
    assert len(cases) == len(before) == 288 and set(after) == changed and len(changed) == 20
    old_runtime, new_runtime = [read(p / "runtime.json") for p in (original, retest)]
    for key in ("case_source_sha256", "layout_policy", "seed", "per_dim_gate_seed", "state_v_first",
                "state_dtype", "use_gate_in_kernel", "safe_gate", "python_interface", "extension_package"):
        assert old_runtime[key] == new_runtime[key], key
    assert old_runtime["pad_generic_tail"] and new_runtime["pad_sequence_tails"]
    results, mapping = [], []
    for case in cases:
        name = case["test_name"]
        source = retest if name in changed else original
        item = dict((after if name in changed else before)[name])
        if item["status"] != "UNSUPPORTED_SHAPE":
            assert item["precision_checked"] and item["status"] == "PASSED", name
            assert all(x["passed"] for x in item["precision_metrics"]), name
            length = (int(case["seqlen"]) + 63) // 64 * 64
            assert item["effective_lengths"] == [length] * int(case["B"]), name
            if item["sequence_padding"]:
                assert item["padding_reference_checked"], name
        item["precision_source"] = str(source.relative_to(raw) / "results.json")
        mapping.append({"test_name": name, "source": item["precision_source"]})
        results.append(item)
    provenance = {
        "kind": "full_scan_plus_changed_tail_retest", "full_scan_rows": 288, "replaced_rows": 20,
        "rule": "Only 128x128 nonaligned sequence inputs changed; all other measured inputs retained",
        "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for d in (original, retest) for p in (d / "results.json", d / "runtime.json")],
        "case_sources": mapping,
    }
    output.mkdir()
    runtime = dict(new_runtime, case_count=288, precision_collection=provenance["kind"])
    write(output / "runtime.json", runtime)
    write(output / "case_manifest.json", cases)
    write(output / "results.json", results)
    write(raw / "precision_provenance.json", provenance)
    (output / "runner_exit_code.txt").write_text("0\n")
    print(f"Merged {len(results)} rows; {len(changed)} actual retests; all measured cases passed")


if __name__ == "__main__":
    main()
