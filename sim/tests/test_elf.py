# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ELF contract: entries use virtual addresses; rodata/data and linked reloc notes survive."""

from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elf import images


def fixture_elf(elf_type=2):
    names = b"\0.text\0.rodata\0.data\0.symtab\0.strtab\0.shstrtab\0.rela.text\0"
    symbols = b"\0MegaMoe_test_730152_mix_aic\0MegaMoe_test_730152_mix_aiv\0"
    payload = bytearray(0x2000)
    payload[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    shoff = 0x1800
    struct.pack_into(
        "<HHIQQQIHHHHHH",
        payload,
        16,
        elf_type,
        0x1029,
        1,
        0,
        0,
        shoff,
        0,
        64,
        0,
        0,
        64,
        8,
        6,
    )
    payload[0x100:0x500] = b"A" * 512 + b"V" * 512
    payload[0x600:0x608] = b"READONLY"
    payload[0x700:0x708] = b"GLOBAL!!"
    payload[0x900 : 0x900 + len(symbols)] = symbols
    payload[0xA00 : 0xA00 + len(names)] = names
    # Distinct file offsets and virtual addresses deliberately catch st_value-as-file-offset bugs.
    sections = [
        (0,) * 10,
        (names.index(b".text"), 1, 6, 0x200, 0x100, 1024, 0, 0, 256, 0),
        (names.index(b".rodata"), 1, 2, 0x900, 0x600, 8, 0, 0, 8, 0),
        (names.index(b".data"), 1, 3, 0xA00, 0x700, 8, 0, 0, 256, 0),
        (names.index(b".symtab"), 2, 0, 0, 0x800, 72, 5, 1, 8, 24),
        (names.index(b".strtab"), 3, 0, 0, 0x900, len(symbols), 0, 0, 1, 0),
        (names.index(b".shstrtab"), 3, 0, 0, 0xA00, len(names), 0, 0, 1, 0),
        (names.index(b".rela.text"), 4, 0, 0, 0xB00, 24, 4, 1, 8, 24),
    ]
    struct.pack_into("<IBBHQQ", payload, 0x818, 1, 0x12, 0, 1, 0x200, 512)
    struct.pack_into(
        "<IBBHQQ",
        payload,
        0x830,
        symbols.index(b"MegaMoe_test_730152_mix_aiv"),
        0x12,
        0,
        1,
        0x400,
        512,
    )
    # The real CANN linker retains relocation records in final ET_EXEC artifacts.
    struct.pack_into("<QQq", payload, 0xB00, 0x220, 0x10000015C, 0)
    for index, section in enumerate(sections):
        struct.pack_into("<IIQQQQIIQQ", payload, shoff + 64 * index, *section)
    return payload


class ElfExtraction(unittest.TestCase):
    def test_two_entries_and_global_data_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel.o"
            path.write_bytes(fixture_elf())
            bins, symbols = images(path, 730152)
            self.assertEqual(bins["aic"][:512], b"A" * 512)
            self.assertEqual(bins["aiv"][:512], b"V" * 512)
            for kind, entry in (("aic", 0x200), ("aiv", 0x400)):
                self.assertEqual(bins[kind][0x900 - entry : 0x908 - entry], b"READONLY")
                self.assertEqual(bins[kind][0xA00 - entry : 0xA08 - entry], b"GLOBAL!!")
                self.assertEqual(symbols[kind]["offset"], entry)
            with self.assertRaisesRegex(ValueError, "requested key"):
                images(path, 664612)
            path.write_bytes(fixture_elf(1))
            with self.assertRaisesRegex(ValueError, "ET_EXEC"):
                images(path, 730152)


if __name__ == "__main__":
    unittest.main()
