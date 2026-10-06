# Copyright (c) 2026 Ethan Zou.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise blank XLSX values through the dependency-free public reader."""

import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def write_workbook(path, value_cells):
    parts = {
        "[Content_Types].xml": (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            f'<Relationships xmlns="{PACKAGE_REL}">'
            f'<Relationship Id="rId1" Type="{REL}/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"
        ),
        "xl/workbook.xml": (
            f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>'
            '<sheet name="Cases" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            f'<Relationships xmlns="{PACKAGE_REL}">'
            f'<Relationship Id="rId1" Type="{REL}/worksheet" Target="worksheets/sheet1.xml"/>'
            f'<Relationship Id="rId2" Type="{REL}/sharedStrings" Target="sharedStrings.xml"/>'
            "</Relationships>"
        ),
        "xl/sharedStrings.xml": (
            f'<sst xmlns="{MAIN}" count="1" uniqueCount="1"><si><t>case</t></si></sst>'
        ),
        "xl/worksheets/sheet1.xml": (
            f'<worksheet xmlns="{MAIN}"><sheetData><row r="1">'
            '<c r="A1" t="s"><v>0</v></c>'
            f"{value_cells}</row></sheetData></worksheet>"
        ),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, xml in parts.items():
            archive.writestr(name, xml)


class EmptyCachedValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).with_name("_xlsx_minireader.py")
        spec = importlib.util.spec_from_file_location("xlsx_reader_under_test", source)
        cls.reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reader)

    def check_workbook(self, cells, expected):
        with tempfile.TemporaryDirectory(prefix="xlsx-empty-cache-") as directory:
            path = Path(directory) / "cases.xlsx"
            write_workbook(path, cells)
            # A shared-string workbook must be handled without the optional fallback.
            with patch.dict(sys.modules, {"openpyxl": None}):
                for selector in (1, "Cases"):
                    with self.subTest(sheet=selector):
                        actual = (
                            self.reader.load_sheet1(str(path))
                            if selector == 1
                            else self.reader.load_sheet(str(path), selector)
                        )
                        self.assertEqual(actual, [{0: "case", **expected}])

    def test_formula_with_empty_or_missing_cache(self):
        for cache in ("<v/>", "<v></v>", ""):
            with self.subTest(cache=cache):
                self.check_workbook(f'<c r="B1"><f>1+2</f>{cache}</c>', {1: None})

    def test_numeric_cell_with_empty_value(self):
        for value in ("<v/>", "<v></v>"):
            with self.subTest(value=value):
                self.check_workbook(f'<c r="B1">{value}</c>', {1: None})

    def test_populated_values_unchanged(self):
        cells = (
            '<c r="B1"><f>1+2</f><v>3</v></c>'
            '<c r="C1"><v>0</v></c>'
            '<c r="D1" t="b"><v>0</v></c>'
            '<c r="E1" t="b"><v>1</v></c>'
            '<c r="F1" t="inlineStr"><is><t>中文</t></is></c>'
            '<c r="G1" t="str"><f>"text"</f><v>text</v></c>'
        )
        self.check_workbook(
            cells, {1: "3", 2: "0", 3: "False", 4: "True", 5: "中文", 6: "text"}
        )


if __name__ == "__main__":
    unittest.main()
