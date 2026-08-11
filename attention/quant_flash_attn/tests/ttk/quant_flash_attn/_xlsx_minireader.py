#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""极简 xlsx 读取器：zipfile + xml.dom.minidom，无 openPyXL/pandas 依赖。

供 excel_to_csv.py 使用。支持读取 xl/worksheets/sheetN.xml（通过 sheet 名或序号选择）+
xl/sharedStrings.xml，返回 list[dict[col_idx]→text]，空单元格 → None。
"""

import re
import zipfile
import xml.dom.minidom as minidom

_CELL_RE = re.compile(r"^([A-Z]+)(\d+)$")


def col_to_idx(col: str) -> int:
    """列字母 → 0-based 列索引（A=0 ... Z=25, AA=26, AB=27 ...）"""
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n - 1


def _parse_ref(ref: str):
    """把 Excel 单元格引用（如 "AB12"）拆成 (col_idx, row_idx)。"""
    m = _CELL_RE.match(ref)
    if not m:
        raise ValueError(f"bad cell ref: {ref!r}")
    return col_to_idx(m.group(1)), int(m.group(2)) - 1


def _cell_text(cell, shared_strings):
    """取一个 <c> 元素的文本值。返回 None 表示空单元格。

    - t="s" + <v>i</v>      → shared_strings[i]
    - t="inlineStr" + <is><t> → inline string
    - t="b" + <v>0/1</v>    → bool 字符串 "True"/"False"
    - <v>num</v>            → 数值字符串（保留 Excel 原样，避免精度丢失）
    - <f>...</f> + <v>      → 公式缓存值，取 <v>
    - 空                    → None
    """
    t = cell.getAttribute("t")
    v_els = cell.getElementsByTagName("v")
    if t == "s" and v_els:
        idx = int(v_els[0].firstChild.data)
        return shared_strings[idx]
    if t == "b" and v_els:
        return "True" if v_els[0].firstChild.data == "1" else "False"
    if t == "inlineStr":
        is_els = cell.getElementsByTagName("is")
        if not is_els:
            return None
        ts = is_els[0].getElementsByTagName("t")
        return "".join(tt.firstChild.data if tt.firstChild else "" for tt in ts)
    if v_els:
        # 数值或公式缓存值
        return v_els[0].firstChild.data
    return None


def _parse_sheet_rows(sheet_xml: str, shared_strings: list):
    """解析单个 sheet xml，返回 list[dict[col_idx]→text]。"""
    sh = minidom.parseString(sheet_xml)
    rows = sh.getElementsByTagName("row")
    rows = sorted(rows, key=lambda r: int(r.getAttribute("r")))
    parsed_rows = []
    for row in rows:
        cells = {}
        for c in row.getElementsByTagName("c"):
            ref = c.getAttribute("r")
            col_idx, _row_idx = _parse_ref(ref)
            cells[col_idx] = _cell_text(c, shared_strings)
        parsed_rows.append(cells)
    return parsed_rows


def _load_shared_strings(z: zipfile.ZipFile) -> list:
    try:
        ss_xml = z.read("xl/sharedStrings.xml").decode("utf-8")
    except KeyError:
        return []
    ss_doc = minidom.parseString(ss_xml)
    shared_strings = []
    for si in ss_doc.getElementsByTagName("si"):
        ts = si.getElementsByTagName("t")
        shared_strings.append(
            "".join(t.firstChild.data if t.firstChild else "" for t in ts)
        )
    return shared_strings


def list_sheet_names(xlsx_path: str) -> list:
    """列出 xlsx 中所有 sheet 名称（按 workbook.xml 中的顺序）。"""
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            wb_xml = z.read("xl/workbook.xml").decode("utf-8")
    except KeyError:
        return []
    wb = minidom.parseString(wb_xml)
    names = []
    for sheet_el in wb.getElementsByTagName("sheet"):
        name = sheet_el.getAttribute("name")
        if name:
            names.append(name)
    return names


def header_to_name_idx(header_row: dict) -> dict:
    """表头行 (dict[col_idx→列名]) 反转为 dict[列名→col_idx]。

    供 excel_to_csv.py 按列名动态定位列, 避免不同 sheet 列顺序偏移导致错位。
    空列名 (None/空串) 跳过。
    """
    return {name.strip(): idx for idx, name in sorted(header_row.items()) if name}


def _xlsx_sheet_name_to_idx(xlsx_path: str) -> dict:
    """返回 {sheet_name: sheet_index} 映射（1-based, 与 sheetN.xml 对应）。"""
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            wb_xml = z.read("xl/workbook.xml").decode("utf-8")
    except KeyError:
        return {}
    wb = minidom.parseString(wb_xml)
    result = {}
    idx = 1
    for sheet_el in wb.getElementsByTagName("sheet"):
        name = sheet_el.getAttribute("name")
        if name:
            result[name] = idx
            idx += 1
    return result


def load_sheet(xlsx_path: str, sheet_name: str = None, sheet_idx: int = 1):
    """读取指定 sheet，返回 list[dict[col_idx]→text]。

    sheet_name: sheet 名称（如 "mxfp8", "fp8", "hif8"）。优先使用。
    sheet_idx:  1-based sheet 序号（1=sheet1.xml）。sheet_name 为 None 时使用。
    """
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            shared_strings = _load_shared_strings(z)

            if sheet_name is not None:
                name_idx = _xlsx_sheet_name_to_idx(xlsx_path)
                if sheet_name not in name_idx:
                    available = list(name_idx.keys())
                    raise ValueError(
                        f"sheet '{sheet_name}' not found in {xlsx_path}. "
                        f"Available sheets: {available}"
                    )
                sheet_idx = name_idx[sheet_name]

            sheet_path = f"xl/worksheets/sheet{sheet_idx}.xml"
            sheet_xml = z.read(sheet_path).decode("utf-8")
    except KeyError:
        return _load_sheet_openpyxl(xlsx_path, sheet_name, sheet_idx)

    return _parse_sheet_rows(sheet_xml, shared_strings)


def load_sheet1(xlsx_path: str):
    """向后兼容：读取 sheet1。"""
    return load_sheet(xlsx_path, sheet_idx=1)


def _load_sheet_openpyxl(xlsx_path: str, sheet_name: str = None, sheet_idx: int = 1):
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    if sheet_name is not None:
        ws = wb[sheet_name]
    else:
        ws = wb.worksheets[sheet_idx - 1]
    parsed_rows = []
    for row in ws.iter_rows():
        cells = {}
        for cell in row:
            v = cell.value
            if v is None:
                continue
            cells[cell.col_idx - 1] = str(v) if not isinstance(v, str) else v
        parsed_rows.append(cells)
    return parsed_rows
