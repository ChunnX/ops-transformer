"""CPU-only checks for the candidate scanner and hardware-result classifier."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce import classify
from audit_binary import set_messages

spec = importlib.util.spec_from_file_location("scan_flags", ROOT / "skills/ascend-flag-conflict-audit/scripts/scan_flags.py")
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


class ScannerTests(unittest.TestCase):
    def test_cross_mode_array_reuse(self):
        report = scanner.scan({"kernel.h": """
            constexpr uint8_t CV[2] = {0, 1};
            constexpr uint8_t VV[2] = {0, 1};
            CrossCoreSetFlag<2, PIPE_MTE3>(CV[i & 1]);
            CrossCoreSetFlag<0, PIPE_MTE2>(VV[task & 1]);
        """})
        self.assertEqual([row["flag"] for row in report["candidates"]], [0, 1])

    def test_default_wait_is_not_a_second_protocol(self):
        report = scanner.scan({"kernel.h": "CrossCoreSetFlag<2, PIPE_FIX>(4); CrossCoreWaitFlag(4);"})
        self.assertEqual(report["candidates"], [])
        self.assertEqual(report["calls"][1]["mode"], [0])

    def test_configuration_override_and_unknown(self):
        source = """#define VV 0
            CrossCoreSetFlag<2, PIPE_MTE3>(0);
            CrossCoreSetFlag<0, PIPE_MTE2>(VV);
            CrossCoreSetFlag<0, PIPE_MTE2>(runtimeFlag);
            SyncAll<false>();
        """
        report = scanner.scan({"kernel.h": source}, {"VV": 15})
        self.assertEqual(report["candidates"], [])
        self.assertEqual(report["unresolved"], [2])
        self.assertTrue(report["warnings"])

    def test_comments_do_not_create_candidates(self):
        report = scanner.scan({"kernel.h": "// CrossCoreSetFlag<0, PIPE_MTE2>(0);\nCrossCoreSetFlag<2, PIPE_MTE3>(0);"})
        self.assertEqual(len(report["calls"]), 1)

    def test_current_pipeline_contains_zero_one_candidates(self):
        kernel = ROOT / "attention/mlp_sparse_lightning_indexer_grad_kl_loss/op_kernel"
        report = scanner.scan({str(p): p.read_text() for p in kernel.glob("*.h")})
        modes = {row["flag"]: row["modes"] for row in report["candidates"]}
        self.assertTrue({0, 2} <= set(modes[0]))
        self.assertTrue({0, 2} <= set(modes[1]))


class ResultTests(unittest.TestCase):
    audit = {"cores": {"aiv": {"symbol_offset": 0x78, "waits": [{"offset": 0x104, "operand": "#0"}]}}}
    fault = "[SLIG] starting runs=1000\nerror 507014\nfftsplus aivector error, pc start: 0x100078, current: 0x100104"

    def test_exact_wait_timeout(self):
        result = classify(self.fault, 1, self.audit, True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["pc_sites"][0]["text_offset"], "0x104")

    def test_watchdog_is_not_a_reproduction(self):
        self.assertFalse(classify(self.fault, 124, self.audit, True)["ok"])

    def test_arbitrary_failure_is_not_a_reproduction(self):
        self.assertFalse(classify("ImportError: torch", 1, self.audit, True)["ok"])
        self.assertFalse(classify(self.fault.replace("0x100104", "0x100100"), 1, self.audit, True)["ok"])

    def test_control_must_pass(self):
        self.assertTrue(classify("[SLIG] RESULT=PASS", 0, self.audit, False)["ok"])
        self.assertFalse(classify(self.fault, 1, self.audit, False)["ok"])
        self.assertEqual(classify("[SLIG] RESULT=PASS", 0, self.audit, True)["outcome"], "not_reproduced")

    def test_single_round_observation_is_still_strict(self):
        self.assertTrue(classify("[SLIG] RESULT=PASS", 0, self.audit, None)["ok"])
        self.assertTrue(classify(self.fault, 1, self.audit, None)["ok"])
        self.assertFalse(classify("ImportError", 1, self.audit, None)["ok"])
        self.assertFalse(classify(self.fault, 124, self.audit, None)["ok"])

    def test_set_message_encoding_and_unknown_write(self):
        body = """100: 07000021 MOV X0, #33
104: 07020f01 MOV X1, #3841
108: 41f01400 SET_CROSS_CORE.MTE3 X0
10c: 41f01004 SET_CROSS_CORE.MTE2 X1
"""
        self.assertEqual([(m["mode"], m["flag"]) for m in set_messages(body)], [(2, 0), (0, 15)])
        with self.assertRaises(RuntimeError):
            set_messages(body.replace("108:", "106: 03c00000 LD.b64 X0, [X2]\n108:"))


if __name__ == "__main__":
    unittest.main()
