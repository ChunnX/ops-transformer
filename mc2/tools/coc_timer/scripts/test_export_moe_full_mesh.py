"""Host-side ABI, capacity failure and export checks (no NPU required)."""
import csv
import struct
import tempfile
import unittest
from pathlib import Path

from export_moe_full_mesh import EVENTS, MAGIC, RECORD_WORDS, decode, export


def fixture(a=50, cores=48, send_cores=40, status=1):
    """ABI v6: 6 serial events, 64-word records; send cores 0..send_cores-1."""
    data = bytearray(a * 512)
    offset = (a * 12 + 127) // 128 * 128
    struct.pack_into("<16Q", data, offset, MAGIC, 6, status, a, cores, RECORD_WORDS, len(EVENTS),
                     3, send_cores, 4608, 4096, offset, 128 + cores * 512, 50, 1, 1)
    if status == 1:
        for core in range(cores):
            init = (500, 1000)
            recv_copy, tail = (1600, 1950), (1950, 1980)
            if core < send_cores:
                cnt_send, payload, cnt_calc = (1050, 1200), (1200, 1500), (0, 0)
            else:
                cnt_send, payload, cnt_calc = (0, 0), (0, 0), (1050, 1400)
            times = (*init, *cnt_send, *payload, *cnt_calc, *recv_copy, *tail)
            struct.pack_into("<64Q", data, offset + 128 + core * 512, MAGIC, core,
                             0, 0, 0, 0, 0, 0, *times, *([0] * (RECORD_WORDS - 8 - 2 * len(EVENTS))))
    return data


class ExportTests(unittest.TestCase):
    def test_decode_minimum_capacity_and_unused_events(self):
        meta, rows = decode(fixture())
        self.assertEqual(meta["timer_version"], 6)
        self.assertEqual(meta["rank"], 3)
        self.assertEqual(meta["send_cores"], 40)
        self.assertEqual(len(rows), 48)
        self.assertEqual(rows[0]["pairs"][1], (1050, 1200))   # CNT_SEND
        self.assertEqual(rows[-1]["pairs"][3], (1050, 1400))  # CNT_CALC

    def test_capacity_failure(self):
        with self.assertRaisesRegex(ValueError, "Timer disabled"):
            decode(fixture(a=12, status=2))

    def test_bad_dump(self):
        for data in (b"", bytes(512), fixture()[:-4]):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                decode(data)

    def test_missing_core(self):
        data = fixture()
        offset = 640 + 128 + 47 * 512
        struct.pack_into("<Q", data, offset, 0)
        with self.assertRaisesRegex(ValueError, "AIV 47"):
            decode(data)

    def test_unpaired_timestamp(self):
        data = fixture()
        struct.pack_into("<Q", data, 640 + 128 + 9 * 8, 0)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            decode(data)

    def test_stale_wait_counters(self):
        data = fixture()
        struct.pack_into("<Q", data, 640 + 128 + 2 * 8, 100)
        with self.assertRaisesRegex(ValueError, "data-wait"):
            decode(data)

    def test_invalid_serial_chains(self):
        base = 640 + 128
        data = fixture()
        struct.pack_into("<2Q", data, base + 12 * 8, 1150, 1200)  # payload starts before cnt-send ends
        with self.assertRaisesRegex(ValueError, "Overlapping serial stages"):
            decode(data)
        data = fixture()
        struct.pack_into("<2Q", data, base + 40 * 512 + 10 * 8, 1050, 1100)  # CNT_SEND on cumsum core
        with self.assertRaisesRegex(ValueError, "must not record send stages"):
            decode(data)
        data = fixture()
        struct.pack_into("<2Q", data, base + 12 * 8, 0, 0)  # send core without payload
        with self.assertRaisesRegex(ValueError, "Missing send stages"):
            decode(data)
        data = fixture()
        struct.pack_into("<2Q", data, base + 40 * 512 + 14 * 8, 0, 0)  # cumsum core without CNT_CALC
        with self.assertRaisesRegex(ValueError, "Missing cumsum stage"):
            decode(data)
        data = fixture()
        struct.pack_into("<2Q", data, base + 16 * 8, 0, 0)  # no recv-copy on send core
        with self.assertRaisesRegex(ValueError, "Missing recv/tail stage"):
            decode(data)
        data = fixture()
        struct.pack_into("<Q", data, base + 9 * 8, 0)  # init.end zeroed
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            decode(data)

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as directory:
            _, raw, durations = export(fixture(), directory)
            with durations.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["send_kind"], "payload_count")
            self.assertEqual(rows[-1]["send_kind"], "receive_only")
            self.assertEqual(float(rows[0]["AIV_CNT_SEND_0_duration(us)"]), 3.0)
            self.assertEqual(float(rows[0]["AIV_PAYLOAD_0_duration(us)"]), 6.0)
            self.assertEqual(float(rows[0]["AIV_CNT_CALC_0_duration(us)"]), 0.0)
            self.assertEqual(float(rows[-1]["AIV_CNT_CALC_0_duration(us)"]), 7.0)
            self.assertEqual(float(rows[0]["AIV_RECV_COPY_0_duration(us)"]), 7.0)
            self.assertEqual(float(rows[0]["AIV_TAIL_0_duration(us)"]), 0.6)
            self.assertEqual(
                float(rows[0]["AIV_INIT_0_duration(us)"])
                + float(rows[0]["AIV_CNT_SEND_0_duration(us)"])
                + float(rows[0]["AIV_PAYLOAD_0_duration(us)"])
                + float(rows[0]["AIV_RECV_COPY_0_duration(us)"])
                + float(rows[0]["AIV_TAIL_0_duration(us)"]),
                26.6,
            )
            with raw.open() as stream:
                raw_rows = list(csv.DictReader(stream))
            self.assertEqual(int(raw_rows[0]["AIV_INIT_0_start_cycle"]), 500)
            self.assertEqual(int(raw_rows[-1]["AIV_TAIL_0_end_cycle"]), 1980)


if __name__ == "__main__":
    unittest.main()
