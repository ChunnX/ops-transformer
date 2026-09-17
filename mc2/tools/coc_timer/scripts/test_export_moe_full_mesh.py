"""Host-side ABI, capacity failure and trace integration checks (no NPU required)."""

import csv
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

from export_moe_full_mesh import EVENTS, MAGIC, decode, export


def fixture(a=13, cores=48, status=1):
    data = bytearray(a * 512)
    offset = (a * 12 + 127) // 128 * 128
    struct.pack_into(
        "<16Q",
        data,
        offset,
        MAGIC,
        2,
        status,
        a,
        cores,
        16,
        3,
        3,
        cores - 1,
        4608,
        4096,
        offset,
        128 + cores * 128,
        50,
        0,
        0,
    )
    if status == 1:
        for core in range(cores):
            times = [
                1000 + core,
                2000 + core,
                1100 + core,
                1500 + core,
                1100 + core,
                2000 + core,
            ]
            struct.pack_into(
                "<16Q",
                data,
                offset + 128 + core * 128,
                MAGIC,
                core,
                100,
                2,
                3,
                0,
                0,
                0,
                *times,
                0,
                0,
            )
    return data


def fixture_v4(a=25, cores=48, send_cores=40, status=1):
    """ABI v4: 9 events, 32-word records; send cores 0..send_cores-1, receive cores after."""
    data = bytearray(a * 512)
    offset = (a * 12 + 127) // 128 * 128
    struct.pack_into(
        "<16Q",
        data,
        offset,
        MAGIC,
        4,
        status,
        a,
        cores,
        32,
        9,
        3,
        send_cores,
        4608,
        4096,
        offset,
        128 + cores * 256,
        50,
        1,
        1,
    )
    if status == 1:
        for core in range(cores):
            stage = (1000, 2000)
            tail = (1900, 2200)
            init = (500, 1000)
            recv_copy = (1600, 2150)
            if core < send_cores:
                send, total = (1100, 1500), (1100, 2000)
                cnt_send, payload, cnt_calc = (1100, 1200), (1200, 1500), (0, 0)
            else:
                send, total = (0, 0), (0, 0)
                cnt_send, payload, cnt_calc = (0, 0), (0, 0), (1100, 1400)
            times = (
                *stage,
                *send,
                *total,
                *tail,
                *init,
                *cnt_send,
                *payload,
                *cnt_calc,
                *recv_copy,
            )
            struct.pack_into(
                "<32Q",
                data,
                offset + 128 + core * 256,
                MAGIC,
                core,
                100,
                2,
                3,
                0,
                0,
                0,
                *times,
                0,
                0,
                0,
                0,
                0,
                0,
            )
    return data


def fixture_v5(a=50, cores=48, send_cores=40, status=1):
    """ABI v5: 14 events, 64-word records; TAIL is the in-stage drain span."""
    data = bytearray(a * 512)
    offset = (a * 12 + 127) // 128 * 128
    struct.pack_into(
        "<16Q",
        data,
        offset,
        MAGIC,
        5,
        status,
        a,
        cores,
        64,
        14,
        3,
        send_cores,
        4608,
        4096,
        offset,
        128 + cores * 512,
        50,
        1,
        1,
    )
    if status == 1:
        for core in range(cores):
            stage, init = (1000, 2000), (500, 1000)
            recv_copy, recv_wait, tail = (1600, 1950), (1600, 1700), (1950, 1980)
            if core < send_cores:
                send, total = (1050, 1500), (1050, 1990)
                cnt_send, payload, cnt_calc = (1050, 1200), (1200, 1500), (0, 0)
                expid, notify = (1050, 1100), (1150, 1200)
                wait_dispatch, cumsum_cal = (0, 0), (0, 0)
            else:
                send, total = (0, 0), (0, 0)
                cnt_send, payload, cnt_calc = (0, 0), (0, 0), (1100, 1400)
                expid, notify = (0, 0), (0, 0)
                wait_dispatch, cumsum_cal = (1100, 1300), (1300, 1400)
            times = (
                *stage,
                *send,
                *total,
                *tail,
                *init,
                *cnt_send,
                *payload,
                *cnt_calc,
                *recv_copy,
                *expid,
                *notify,
                *wait_dispatch,
                *cumsum_cal,
                *recv_wait,
            )
            struct.pack_into(
                "<64Q",
                data,
                offset + 128 + core * 512,
                MAGIC,
                core,
                0,
                0,
                0,
                0,
                0,
                0,
                *times,
                *([0] * 28),
            )
    return data


class ExportTests(unittest.TestCase):
    def test_decode_minimum_capacity_and_unused_events(self):
        meta, rows = decode(fixture())
        self.assertEqual(meta["rank"], 3)
        self.assertEqual(len(rows), 48)
        self.assertEqual(rows[-1]["pairs"][1], (1147, 1547))
        self.assertEqual(rows[0]["wait_cycles"], 100)

    def test_capacity_failure(self):
        with self.assertRaisesRegex(ValueError, "Timer disabled"):
            decode(fixture(a=12, status=2))

    def test_bad_dump(self):
        for data in (b"", bytes(512), fixture()[:-4]):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                decode(data)

    def test_missing_core(self):
        data = fixture()
        offset = 256 + 128 + 47 * 128
        struct.pack_into("<Q", data, offset, 0)
        with self.assertRaisesRegex(ValueError, "AIV 47"):
            decode(data)

    def test_unpaired_timestamp(self):
        data = fixture()
        struct.pack_into("<Q", data, 256 + 128 + 9 * 8, 0)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            decode(data)

    def test_no_send_no_receive(self):
        data = fixture()
        struct.pack_into(
            "<16Q",
            data,
            256 + 128,
            MAGIC,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1000,
            2000,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        _, rows = decode(data)
        self.assertEqual(rows[0]["pairs"][1:], [(0, 0), (0, 0)])
        self.assertEqual(rows[0]["wait_cycles"], 0)

    def test_invalid_wait_and_old_version(self):
        for word, value in ((2, 1001), (3, 4), (3, 0)):
            data = fixture()
            struct.pack_into("<Q", data, 256 + 128 + word * 8, value)
            with self.assertRaisesRegex(ValueError, "data-wait"):
                decode(data)
        data = fixture()
        struct.pack_into("<Q", data, 256 + 8, 1)
        with self.assertRaisesRegex(ValueError, "ABI v2"):
            decode(data)

    def test_combined_sender_layout(self):
        data = fixture()
        struct.pack_into("<Q", data, 256 + 14 * 8, 1)
        struct.pack_into("<Q", data, 256 + 15 * 8, 1)
        struct.pack_into("<4Q", data, 256 + 128 + 47 * 128 + 10 * 8, 0, 0, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            meta, _, durations = export(data, directory)
            with durations.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(meta["send_layout"], 1)
            self.assertTrue(meta["arrival_only"])
            self.assertEqual(rows[0]["send_kind"], "payload_count")
            self.assertEqual(rows[-1]["send_kind"], "receive_only")
            self.assertEqual(float(rows[-1]["AIV_SEND_0_duration(us)"]), 0)

    def test_v3_tail_and_legacy_columns(self):
        data = fixture()
        struct.pack_into("<Q", data, 256 + 8, 3)
        struct.pack_into("<Q", data, 256 + 6 * 8, 4)
        for core in range(48):
            struct.pack_into(
                "<2Q", data, 256 + 128 + core * 128 + 14 * 8, 1900 + core, 2200 + core
            )
        with tempfile.TemporaryDirectory() as directory:
            meta, raw, durations = export(data, directory)
            self.assertEqual(meta["timer_version"], 3)
            with durations.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["AIV_TAIL_0_duration(us)"]), 6)
            with raw.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(int(rows[0]["AIV_TAIL_0_start_cycle"]), 1900)
        struct.pack_into("<Q", data, 256 + 128 + 15 * 8, 1950)
        with self.assertRaisesRegex(ValueError, "tail"):
            decode(data)

    def test_v4_stage_events(self):
        meta, rows = decode(fixture_v4())
        self.assertEqual(meta["timer_version"], 4)
        self.assertEqual(len(rows[0]["pairs"]), 9)
        self.assertEqual(rows[0]["pairs"][4], (500, 1000))  # INIT
        self.assertEqual(rows[39]["pairs"][5], (1100, 1200))  # CNT_SEND
        self.assertEqual(rows[40]["pairs"][7], (1100, 1400))  # CNT_CALC
        with tempfile.TemporaryDirectory() as directory:
            _, raw, durations = export(fixture_v4(), directory)
            with durations.open() as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(float(csv_rows[0]["AIV_INIT_0_duration(us)"]), 10.0)
            self.assertEqual(float(csv_rows[0]["AIV_RECV_COPY_0_duration(us)"]), 11.0)
            self.assertEqual(float(csv_rows[39]["AIV_PAYLOAD_0_duration(us)"]), 6.0)
            self.assertEqual(float(csv_rows[39]["AIV_CNT_CALC_0_duration(us)"]), 0)
            self.assertEqual(float(csv_rows[-1]["AIV_CNT_CALC_0_duration(us)"]), 6.0)
            self.assertEqual(csv_rows[-1]["send_kind"], "receive_only")
            with raw.open() as stream:
                raw_rows = list(csv.DictReader(stream))
            self.assertEqual(int(raw_rows[-1]["AIV_CNT_CALC_0_start_cycle"]), 1100)

    def test_v4_invalid_stage_chains(self):
        base = 384 + 128
        data = fixture_v4()
        struct.pack_into("<Q", data, base + 21 * 8, 0)  # core 0 PAYLOAD.end
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            decode(data)
        data = fixture_v4()
        struct.pack_into("<2Q", data, base + 40 * 256 + 18 * 8, 1100, 1200)
        with self.assertRaisesRegex(ValueError, "must not record send stages"):
            decode(data)
        data = fixture_v4()
        struct.pack_into(
            "<Q", data, base + 17 * 8, 1100
        )  # core 0 INIT.end after STAGE.start
        with self.assertRaisesRegex(ValueError, "Invalid init interval"):
            decode(data)

    def test_v5_sub_stage_events(self):
        meta, rows = decode(fixture_v5())
        self.assertEqual(meta["timer_version"], 5)
        self.assertEqual(len(rows[0]["pairs"]), 14)
        self.assertEqual(rows[39]["pairs"][9], (1050, 1100))  # EXPID_COPY (send core)
        self.assertEqual(rows[39]["pairs"][10], (1150, 1200))  # CNT_NOTIFY
        self.assertEqual(
            rows[40]["pairs"][11], (1100, 1300)
        )  # WAIT_DISPATCH (receive core)
        self.assertEqual(rows[40]["pairs"][12], (1300, 1400))  # CUMSUM_CAL
        self.assertEqual(rows[0]["pairs"][13], (1600, 1700))  # RECV_WAIT (all cores)
        self.assertEqual(rows[0]["pairs"][3], (1950, 1980))  # TAIL drain inside stage
        with tempfile.TemporaryDirectory() as directory:
            _, raw, durations = export(fixture_v5(), directory)
            with durations.open() as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(float(csv_rows[39]["AIV_EXPID_COPY_0_duration(us)"]), 1.0)
            self.assertEqual(float(csv_rows[39]["AIV_CNT_NOTIFY_0_duration(us)"]), 1.0)
            self.assertEqual(
                float(csv_rows[-1]["AIV_WAIT_DISPATCH_0_duration(us)"]), 4.0
            )
            self.assertEqual(float(csv_rows[-1]["AIV_CUMSUM_CAL_0_duration(us)"]), 2.0)
            self.assertEqual(float(csv_rows[0]["AIV_RECV_WAIT_0_duration(us)"]), 2.0)
            self.assertEqual(float(csv_rows[0]["AIV_TAIL_0_duration(us)"]), 0.6)
            with raw.open() as stream:
                raw_rows = list(csv.DictReader(stream))
            self.assertEqual(int(raw_rows[0]["AIV_CNT_NOTIFY_0_start_cycle"]), 1150)

    def test_v5_invalid_sub_stage_chains(self):
        base = 640 + 128
        data = fixture_v5()
        struct.pack_into(
            "<Q", data, base + 14 * 8, 1900
        )  # core 0 TAIL.start before recv-copy end
        with self.assertRaisesRegex(ValueError, "Invalid tail interval"):
            decode(data)
        data = fixture_v5()
        struct.pack_into(
            "<2Q", data, base + 40 * 512 + 18 * 8, 1050, 1100
        )  # CNT_SEND on receive core
        with self.assertRaisesRegex(ValueError, "must not record send stages"):
            decode(data)
        data = fixture_v5()
        struct.pack_into(
            "<2Q", data, base + 40 * 512 + 34 * 8, 0, 0
        )  # drop RECV_WAIT on receive core
        with self.assertRaisesRegex(ValueError, "Missing recv-wait stage"):
            decode(data)
        data = fixture_v5()
        struct.pack_into(
            "<2Q", data, base + 40 * 512 + 30 * 8, 1150, 1450
        )  # WAIT_DISPATCH overlaps CUMSUM_CAL
        with self.assertRaisesRegex(ValueError, "Invalid cumsum sub-stage chain"):
            decode(data)

    def test_csv_and_upstream_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            _, raw, durations = export(fixture(), directory)
            with durations.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["AIV_SEND_0_duration(us)"]), 8.0)
            if importlib.util.find_spec("pandas") is None:
                self.skipTest("Upstream process.py requires pandas")
            from process import generate_timestamp_timeline_json

            trace = Path(directory) / "trace.json"
            generate_timestamp_timeline_json(str(raw), str(trace), "test")
            events = [
                event for event in json.loads(trace.read_text()) if event["ph"] == "X"
            ]
            self.assertEqual(len(events), 48 * 3)
            self.assertEqual({event["name"] for event in events}, set(EVENTS[:3]))
            self.assertTrue(
                all(
                    event["dur"] == 8.0
                    for event in events
                    if event["name"] == "AIV_SEND_0"
                )
            )
            self.assertEqual(float(rows[0]["DATA_WAIT_duration(us)"]), 2.0)
            self.assertEqual(rows[-1]["send_kind"], "count")


if __name__ == "__main__":
    unittest.main()
