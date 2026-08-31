import csv
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from stem_indexer_batch_filter import filter_testcase_files_by_csv


class TestStemIndexerBatchFilter(unittest.TestCase):
    def test_keeps_only_pt_cases_listed_in_csv(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "cases.csv"
            with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["case_id"])
                writer.writeheader()
                writer.writerows(
                    [{"case_id": "SI_001"}, {"case_id": "SI_003"}]
                )

            testcase_files = [
                str(temp_path / "SI_001.pt"),
                str(temp_path / "SI_002.pt"),
                str(temp_path / "SI_003.pt"),
            ]
            case_ids_by_file = {
                "SI_001.pt": "SI_001",
                "SI_002.pt": "SI_002",
                "SI_003.pt": "SI_003",
            }

            selected = filter_testcase_files_by_csv(
                testcase_files,
                str(csv_path),
                lambda filepath: {"case_id": case_ids_by_file[Path(filepath).name]},
            )

        self.assertEqual(selected, [testcase_files[0], testcase_files[2]])

    def test_ignores_unreadable_pt_outside_csv_intersection(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "cases.csv"
            with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["case_id"])
                writer.writeheader()
                writer.writerow({"case_id": "SI_001"})

            valid_file = str(temp_path / "SI_001.pt")
            unreadable_file = str(temp_path / "stale.pt")

            def load_case(filepath):
                if filepath == unreadable_file:
                    raise ValueError("invalid pt payload")
                return {"case_id": "SI_001"}

            try:
                with redirect_stdout(io.StringIO()):
                    selected = filter_testcase_files_by_csv(
                        [valid_file, unreadable_file], str(csv_path), load_case
                    )
            except ValueError as err:
                self.fail(f"unreadable stale PT must be ignored: {err}")

        self.assertEqual(selected, [valid_file])


if __name__ == "__main__":
    unittest.main()
