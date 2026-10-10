# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "xlsx_reader", Path(__file__).with_name("_xlsx_minireader.py")
)
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)

WORKBOOK = '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Cases" sheetId="1" r:id="rId1"/></sheets></workbook>'
RELATIONSHIPS = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
CONTENT_TYPES = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>{shared}</Types>'
ROOT_RELS = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
SHEET_START = '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1">'
SHEET_END = "</row></sheetData></worksheet>"


class TestOptionalSharedStrings(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workbook = Path(self.directory.name) / "cases.xlsx"

    def write_workbook(self, cells, shared_strings=None):
        with zipfile.ZipFile(self.workbook, "w") as archive:
            shared_type = (
                '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
                if shared_strings is not None
                else ""
            )
            archive.writestr(
                "[Content_Types].xml", CONTENT_TYPES.format(shared=shared_type)
            )
            archive.writestr("_rels/.rels", ROOT_RELS)
            archive.writestr("xl/workbook.xml", WORKBOOK)
            relationships = RELATIONSHIPS
            if shared_strings is not None:
                relationship = '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
                relationships = relationships.replace(
                    "</Relationships>", relationship + "</Relationships>"
                )
            archive.writestr("xl/_rels/workbook.xml.rels", relationships)
            archive.writestr(
                "xl/worksheets/sheet1.xml", SHEET_START + cells + SHEET_END
            )
            if shared_strings is not None:
                archive.writestr("xl/sharedStrings.xml", shared_strings)

    def test_inline_strings_without_optional_dependency(self):
        self.write_workbook(
            '<c r="A1" t="inlineStr"><is><t>用例名</t></is></c><c r="B1" t="b"><v>1</v></c><c r="C1"><v>4096</v></c><c r="D1"/>'
        )
        with mock.patch.dict("sys.modules", {"openpyxl": None}):
            actual = READER.load_sheet1(str(self.workbook))
        self.assertEqual(actual, [{0: "用例名", 1: "True", 2: "4096", 3: None}])

    def test_named_sheet_with_inline_rich_text(self):
        self.write_workbook(
            '<c r="A1" t="inlineStr"><is><r><t>case_</t></r><r><t>one</t></r></is></c>'
        )
        with mock.patch.dict("sys.modules", {"openpyxl": None}):
            actual = READER.load_sheet(str(self.workbook), "Cases")
        self.assertEqual(actual, [{0: "case_one"}])

    def test_numeric_only_sheet_without_shared_strings(self):
        self.write_workbook('<c r="A1"><v>0.125</v></c><c r="B1" t="b"><v>0</v></c>')
        with mock.patch.dict("sys.modules", {"openpyxl": None}):
            actual = READER.load_sheet(str(self.workbook), 1)
        self.assertEqual(actual, [{0: "0.125", 1: "False"}])

    def test_shared_strings_remain_supported(self):
        self.write_workbook(
            '<c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>inline</t></is></c>',
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><r><t>shared_</t></r><r><t>value</t></r></si></sst>',
        )
        with mock.patch.dict("sys.modules", {"openpyxl": None}):
            actual = READER.load_sheet1(str(self.workbook))
        self.assertEqual(actual, [{0: "shared_value", 1: "inline"}])


if __name__ == "__main__":
    unittest.main()
