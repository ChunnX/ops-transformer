"""CPU tests of the golden's layout adapters; no custom ops or NPU import required."""
import ast
from pathlib import Path
import unittest
import torch

tree = ast.parse(Path(__file__).with_name("qbsa_mxfp8_golden.py").read_text(encoding="utf-8"))
names = {"pack_padded_query", "unpack_padded_query"}
functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
scope = {"torch": torch}
exec(compile(functions, "qbsa_mxfp8_golden.py", "exec"), scope)
pack, unpack = scope["pack_padded_query"], scope["unpack_padded_query"]

class PaddedLayoutTest(unittest.TestCase):
    def test_roundtrip(self):
        lengths = [129, 65, 0]
        for dtype in (torch.uint8, torch.bfloat16, torch.float32):
            for tail in ((8, 128), (8, 2, 2), (8,)):
                shape = (sum(lengths), *tail)
                count = 1
                for dim in shape:
                    count *= dim
                original = (torch.arange(count) % 127).reshape(shape).to(dtype)
                for layout in ("BSND", "BNSD"):
                    padded = pack(original, lengths, layout, 129)
                    expected_shape = (3, 129, *tail) if layout == "BSND" else (3, tail[0], 129, *tail[1:])
                    self.assertEqual(tuple(padded.shape), expected_shape)
                    self.assertTrue(padded.is_contiguous())
                    torch.testing.assert_close(unpack(padded, lengths, layout), original, rtol=0, atol=0)
                    if layout == "BNSD":
                        torch.testing.assert_close(padded[1, :, :65].transpose(0, 1), original[129:], rtol=0, atol=0)

    def test_capacity_validation(self):
        with self.assertRaises(ValueError):
            pack(torch.zeros((5, 2, 8)), [3, 2], "BSND", 2)

if __name__ == "__main__":
    unittest.main()
