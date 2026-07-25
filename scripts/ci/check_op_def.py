#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
算子 OpDef 规则静态检查脚本。

检查规则:
  结构规则:
    S1 (ERROR):   文件必须 #include "register/op_def_registry.h"
    S2 (ERROR):   必须在 namespace ops 中定义
    S3 (ERROR):   必须有 class X : public OpDef
    S4 (ERROR):   必须有 OP_ADD(ClassName) 注册
    S5 (ERROR):   禁止 SetInferShape / SetTiling 出现在 def.cpp
  Input/Output 规则:
    I1 (ERROR):   每个 Input/Output 必须含 .ParamType(...)
    I2 (ERROR):   每个 Input/Output 必须含 .DataType(...) 或 .DataTypeList(...)
    I3 (ERROR):   每个 Input/Output 必须含 .Format(...) 或 .FormatList(...)
    I4 (ERROR):   ParamType 取值必须为 REQUIRED/OPTIONAL/DYNAMIC
    I5 (ERROR):   DataType/Format/UnknownShapeFormat 三件套计数必须一致
    I6 (WARNING): DataTypeList 搭配 Format（缺少 FormatList）为可疑用法
  取值校验规则:
    V1 (ERROR):   ge::DT_* 必须为合法数据类型
    V2 (ERROR):   ge::FORMAT_* 必须为合法格式
  AICore 配置规则:
    A1 (ERROR):   必须有至少一个 AddConfig 调用
    A2 (WARNING): AddConfig 的 SOC 名应为已知合法值
  Attr 规则:
    T1 (ERROR):   AttrType 取值必须为 REQUIRED/OPTIONAL
    T2 (WARNING): attr 名不应与保留字冲突

用法:
  python check_op_def.py <file_or_dir> [<file_or_dir> ...]
退出码:
  0 = 通过（可能有 WARNING 但无 ERROR）
  1 = 存在 ERROR
"""

import re
import sys
import os
from pathlib import Path
from dataclasses import dataclass


# ============================================================
# 合法取值常量（来源: parse_ini_to_json.py + 代码库统计）
# ============================================================

VALID_PARAM_TYPES = {"REQUIRED", "OPTIONAL", "DYNAMIC"}

VALID_ATTR_TYPES = {"REQUIRED", "OPTIONAL"}

VALID_DTYPES = {
    "DT_FLOAT", "DT_FLOAT16", "DT_BF16", "DT_INT8", "DT_INT16", "DT_INT32",
    "DT_UINT8", "DT_UINT16", "DT_UINT32", "DT_BOOL", "DT_INT64", "DT_UINT64",
    "DT_QINT8", "DT_QINT16", "DT_QINT32", "DT_QUINT8", "DT_QUINT16",
    "DT_DOUBLE", "DT_COMPLEX32", "DT_COMPLEX64", "DT_COMPLEX128",
    "DT_STRING", "DT_RESOURCE", "DT_DUAL", "DT_DUAL_SUB_INT8",
    "DT_DUAL_SUB_UINT8", "DT_STRING_REF", "DT_INT4", "DT_UINT1",
    "DT_HIFLOAT8", "DT_FLOAT8_E4M3FN", "DT_FLOAT8_E5M2", "DT_FLOAT8_E8M0",
    "DT_FLOAT4_E2M1", "DT_FLOAT4_E1M2", "DT_INT2", "DT_UNDEFINED",
}

VALID_FORMATS = {
    "FORMAT_NCHW", "FORMAT_NHWC", "FORMAT_ND", "FORMAT_NC1HWC0",
    "FORMAT_FRACTAL_Z", "FORMAT_NC1C0HWPAD", "FORMAT_NHWC1C0",
    "FORMAT_FSR_NCHW", "FORMAT_FRACTAL_DECONV", "FORMAT_C1HWNC0",
    "FORMAT_FRACTAL_DECONV_TRANSPOSE",
    "FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS", "FORMAT_NC1HWC0_C04",
    "FORMAT_FRACTAL_Z_C04", "FORMAT_CHWN",
    "FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS", "FORMAT_HWCN",
    "FORMAT_NC1KHKWHWC0", "FORMAT_BN_WEIGHT", "FORMAT_FILTER_HWCK",
    "FORMAT_HASHTABLE_LOOKUP_LOOKUPS", "FORMAT_HASHTABLE_LOOKUP_KEYS",
    "FORMAT_HASHTABLE_LOOKUP_VALUE", "FORMAT_HASHTABLE_LOOKUP_OUTPUT",
    "FORMAT_HASHTABLE_LOOKUP_HITS", "FORMAT_C1HWNCoC0", "FORMAT_MD",
    "FORMAT_NDHWC", "FORMAT_FRACTAL_ZZ", "FORMAT_FRACTAL_NZ",
    "FORMAT_NCDHW", "FORMAT_DHWCN", "FORMAT_NDC1HWC0",
    "FORMAT_FRACTAL_Z_3D", "FORMAT_CN", "FORMAT_NC", "FORMAT_DHWNC",
    "FORMAT_FRACTAL_Z_3D_TRANSPOSE", "FORMAT_FRACTAL_ZN_LSTM",
    "FORMAT_FRACTAL_ZN_RNN", "FORMAT_FRACTAL_Z_G", "FORMAT_NULL",
    "FORMAT_FRACTAL_NZ_C0_2", "FORMAT_FRACTAL_NZ_C0_4",
    "FORMAT_FRACTAL_NZ_C0_16", "FORMAT_FRACTAL_NZ_C0_32",
}

KNOWN_SOCS = {
    "ascend910b", "ascend910_93", "ascend950", "ascend310p", "ascend310b",
    "ascend910", "ascend610lite", "kirinx90", "kirin9030", "mc62",
    "ascend910_55", "bs9sx1a",
}

CONFLICT_KEYWORDS = {
    "and", "as", "assert", "break", "class", "continue", "def", "del",
    "elif", "else", "except", "finally", "for", "from", "global", "if",
    "import", "in", "is", "lambda", "not", "or", "pass", "raise", "return",
    "try", "while", "with", "yield", "False", "None", "True", "nonlocal",
    "arg", "__inputs__", "__outputs__", "options", "bisheng", "bisheng_path",
    "tikcpp_path", "impl_mode", "custom_compile_options",
    "custom_all_compile_options", "soc_version", "soc_short",
    "custom_compile_options_soc", "custom_all_compile_options_soc",
    "origin_func_name", "ascendc_src_dir_ex", "ascendc_src_dir",
    "ascendc_src_file", "src", "op_type", "code_channel", "op_info",
    "compile_op", "get_code_channel", "result", "__attrs__", "isinstance",
    "attr", "get_current_build_config", "_build_args",
    "get_dtype_fmt_options", "shutil", "os", "get_kernel_source",
}

FORBIDDEN_IN_DEF = ["SetInferShape", "SetTiling", "IMPL_OP_OPTILING", "IMPL_OP_INFERSHAPE"]


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Violation:
    file: str
    line: int
    io_type: str
    io_name: str
    severity: str
    rule: str
    message: str


# ============================================================
# 解析函数
# ============================================================

def parse_io_blocks(content):
    """从 def.cpp 内容中解析 Input/Output 块。"""
    blocks = []
    pattern = r'this->(Input|Output)\(\s*"([^"]+)"\s*\)(.*?);'
    for match in re.finditer(pattern, content, re.DOTALL):
        io_type = match.group(1)
        io_name = match.group(2)
        chain = match.group(3)
        line = content[:match.start()].count('\n') + 1
        blocks.append({
            'io_type': io_type,
            'io_name': io_name,
            'chain': chain,
            'line': line,
        })
    return blocks


def extract_api_calls(chain):
    """从链式调用中提取各 API 方法及其参数列表。

    支持两种参数形式:
      - 花括号初始化列表: .DataType({ge::DT_BF16, ge::DT_FLOAT})
      - 变量名: .DataType(qkDtype)
    """
    api_calls = {}
    # 匹配花括号形式: .Method({...})
    pattern = r'\.(DataType|DataTypeList|Format|FormatList|UnknownShapeFormat)\s*\(\s*\{([^}]*)\}\s*\)'
    for match in re.finditer(pattern, chain):
        method = match.group(1)
        args = match.group(2).strip()
        items = [x.strip() for x in args.split(',') if x.strip()] if args else []
        api_calls[method] = items
    # 匹配变量名形式: .DataType(varName) — 无法解析具体值，标记为存在但值为空列表
    pattern_var = r'\.(DataType|DataTypeList|Format|FormatList|UnknownShapeFormat)\s*\(\s*([A-Za-z_]\w*)\s*\)'
    for match in re.finditer(pattern_var, chain):
        method = match.group(1)
        if method not in api_calls:
            api_calls[method] = []
    # ParamType 的参数没有花括号
    pt_match = re.search(r'\.ParamType\(\s*([A-Z]+)\s*\)', chain)
    if pt_match:
        api_calls['ParamType'] = [pt_match.group(1)]
    # Follow 表示继承另一个输入的 DataType/Format，视为存在
    if re.search(r'\.Follow\(', chain):
        api_calls.setdefault('DataType', [])
        api_calls.setdefault('Format', [])
    return api_calls


def count_add_config(content):
    """统计 AddConfig 调用次数。"""
    return len(re.findall(r'\.AddConfig\(', content))


def extract_add_config_socs(content):
    """提取 AddConfig 中的 SOC 名。"""
    socs = []
    pattern = r'\.AddConfig\(\s*"([^"]+)"'
    for match in re.finditer(pattern, content):
        socs.append(match.group(1))
    return socs


def extract_attr_names(content):
    """提取 Attr 名称列表。"""
    names = []
    pattern = r'this->Attr\(\s*"([^"]+)"\s*\)'
    for match in re.finditer(pattern, content):
        names.append(match.group(1))
    return names


def extract_class_name(content):
    """提取 OpDef 子类名。"""
    match = re.search(r'class\s+(\w+)\s*:\s*public\s+OpDef', content)
    return match.group(1) if match else None


def extract_op_add_name(content):
    """提取 OP_ADD 注册的类名。"""
    match = re.search(r'OP_ADD\(\s*(\w+)', content)
    return match.group(1) if match else None


def extract_dtype_values(items):
    """从参数列表中提取 ge::DT_* 值。"""
    values = []
    for item in items:
        match = re.search(r'ge::(DT_\w+)', item)
        if match:
            values.append(match.group(1))
    return values


def extract_format_values(items):
    """从参数列表中提取 ge::FORMAT_* 值。"""
    values = []
    for item in items:
        match = re.search(r'ge::(FORMAT_\w+)', item)
        if match:
            values.append(match.group(1))
    return values


# ============================================================
# 检查函数
# ============================================================

def check_structure(content, file_path):
    """检查结构规则 S1-S5。"""
    violations = []

    # S1: 必须 #include "register/op_def_registry.h"
    if '#include "register/op_def_registry.h"' not in content and \
       "#include <register/op_def_registry.h>" not in content:
        line = 1
        violations.append(Violation(
            file=file_path, line=line, io_type='FILE', io_name='',
            severity='ERROR', rule='S1',
            message='missing #include "register/op_def_registry.h"'
        ))

    # S2: 必须在 namespace ops 中
    if 'namespace ops' not in content:
        violations.append(Violation(
            file=file_path, line=1, io_type='FILE', io_name='',
            severity='ERROR', rule='S2',
            message='must be in namespace ops'
        ))

    # S3: 必须有 class X : public OpDef
    if not re.search(r'class\s+\w+\s*:\s*public\s+OpDef', content):
        violations.append(Violation(
            file=file_path, line=1, io_type='FILE', io_name='',
            severity='ERROR', rule='S3',
            message='must define a class inheriting from OpDef'
        ))

    # S4: 必须有 OP_ADD(ClassName)
    if 'OP_ADD(' not in content:
        violations.append(Violation(
            file=file_path, line=1, io_type='FILE', io_name='',
            severity='ERROR', rule='S4',
            message='missing OP_ADD() registration'
        ))

    # S5: 禁止 SetInferShape / SetTiling 等
    for forbidden in FORBIDDEN_IN_DEF:
        pattern = re.compile(r'\b' + re.escape(forbidden) + r'\b')
        for match in pattern.finditer(content):
            line = content[:match.start()].count('\n') + 1
            violations.append(Violation(
                file=file_path, line=line, io_type='FILE', io_name='',
                severity='ERROR', rule='S5',
                message=f'{forbidden} must not appear in _def.cpp (move to _tiling.cpp or _infershape.cpp)'
            ))

    return violations


def check_io_block(block, file_path, is_aicpu=False):
    """检查单个 Input/Output 块的 I/O 规则。"""
    violations = []
    api_calls = extract_api_calls(block['chain'])
    io_desc = f"{block['io_type']}('{block['io_name']}')"
    line = block['line']

    has_datatype = 'DataType' in api_calls
    has_datatypelist = 'DataTypeList' in api_calls
    has_format = 'Format' in api_calls
    has_formatlist = 'FormatList' in api_calls
    has_unknown = 'UnknownShapeFormat' in api_calls
    has_paramtype = 'ParamType' in api_calls

    # I1: 必须含 ParamType
    if not has_paramtype:
        violations.append(Violation(
            file=file_path, line=line,
            io_type=block['io_type'], io_name=block['io_name'],
            severity='ERROR', rule='I1',
            message=f"{io_desc}: missing .ParamType(...)"
        ))

    # I4: ParamType 取值必须合法
    if has_paramtype:
        pt_value = api_calls['ParamType'][0] if api_calls['ParamType'] else ''
        if pt_value not in VALID_PARAM_TYPES:
            violations.append(Violation(
                file=file_path, line=line,
                io_type=block['io_type'], io_name=block['io_name'],
                severity='ERROR', rule='I4',
                message=f"{io_desc}: ParamType '{pt_value}' not in {sorted(VALID_PARAM_TYPES)}"
            ))

    # I2: 必须含 DataType 或 DataTypeList
    if not has_datatype and not has_datatypelist:
        violations.append(Violation(
            file=file_path, line=line,
            io_type=block['io_type'], io_name=block['io_name'],
            severity='ERROR', rule='I2',
            message=f"{io_desc}: missing .DataType(...) or .DataTypeList(...)"
        ))

    # I3: 必须含 Format 或 FormatList（AICPU 算子放宽，仅要求 DataType）
    if not is_aicpu and not has_format and not has_formatlist:
        violations.append(Violation(
            file=file_path, line=line,
            io_type=block['io_type'], io_name=block['io_name'],
            severity='ERROR', rule='I3',
            message=f"{io_desc}: missing .Format(...) or .FormatList(...)"
        ))

    # 获取各 API 的元素数量
    dt_items = api_calls.get('DataType', [])
    dtl_items = api_calls.get('DataTypeList', [])
    fmt_items = api_calls.get('Format', [])
    fmtl_items = api_calls.get('FormatList', [])
    unk_items = api_calls.get('UnknownShapeFormat', [])

    dt_count = len(dt_items)
    dtl_count = len(dtl_items)
    fmt_count = len(fmt_items)
    fmtl_count = len(fmtl_items)
    unk_count = len(unk_items)

    # 统一的 Format/FormatList 计数（两者互斥使用）
    has_any_format = has_format or has_formatlist
    fmt_api_name = 'Format' if has_format else 'FormatList'
    fmt_effective_count = fmt_count if has_format else fmtl_count

    # 判断是否为变量形式（无法解析具体值，跳过计数校验）
    dt_is_var = has_datatype and dt_count == 0
    fmt_is_var = (has_format and fmt_count == 0) or (has_formatlist and fmtl_count == 0)
    unk_is_var = has_unknown and unk_count == 0

    # I5: 三件套计数一致（DataType/Format/FormatList/UnknownShapeFormat 之间）
    # 计数规则:
    #   Format({A,B}):    逐类型声明格式，计数必须等于 DataType 计数
    #   FormatList({ND}): 计数为 1 时是广播语义（单值适用于所有 DataType）；
    #                     计数 > 1 时是逐类型声明，必须等于 DataType 计数
    #   UnknownShapeFormat: 计数必须等于 Format/FormatList 计数
    # 仅当两者都能解析出具体值时才校验
    fmtl_is_broadcast = has_formatlist and fmtl_count == 1
    if has_datatype and has_format and not dt_is_var and not fmt_is_var and dt_count != fmt_count:
        violations.append(Violation(
            file=file_path, line=line,
            io_type=block['io_type'], io_name=block['io_name'],
            severity='ERROR', rule='I5',
            message=f"{io_desc}: DataType count ({dt_count}) != Format count ({fmt_count})"
        ))
    if has_datatype and has_formatlist and not dt_is_var and not fmt_is_var and not fmtl_is_broadcast and dt_count != fmtl_count:
        violations.append(Violation(
            file=file_path, line=line,
            io_type=block['io_type'], io_name=block['io_name'],
            severity='ERROR', rule='I5',
            message=f"{io_desc}: DataType count ({dt_count}) != FormatList count ({fmtl_count})"
        ))
    if has_datatype and has_unknown and not dt_is_var and not unk_is_var and dt_count != unk_count:
        violations.append(Violation(
            file=file_path, line=line,
            io_type=block['io_type'], io_name=block['io_name'],
            severity='ERROR', rule='I5',
            message=f"{io_desc}: DataType count ({dt_count}) != UnknownShapeFormat count ({unk_count})"
        ))
    if has_any_format and has_unknown and not fmt_is_var and not unk_is_var and fmt_effective_count != unk_count:
        violations.append(Violation(
            file=file_path, line=line,
            io_type=block['io_type'], io_name=block['io_name'],
            severity='ERROR', rule='I5',
            message=f"{io_desc}: {fmt_api_name} count ({fmt_effective_count}) != UnknownShapeFormat count ({unk_count})"
        ))

    # I6: DataTypeList + Format（缺少 FormatList）为可疑用法
    if has_datatypelist and has_format and not has_formatlist:
        violations.append(Violation(
            file=file_path, line=line,
            io_type=block['io_type'], io_name=block['io_name'],
            severity='WARNING', rule='I6',
            message=f"{io_desc}: DataTypeList with Format (no FormatList) - verify this is truly a tensor list, not a misuse of DataType"
        ))

    # V1: 数据类型取值校验
    dtype_values = []
    if has_datatype:
        dtype_values = extract_dtype_values(api_calls['DataType'])
    elif has_datatypelist:
        dtype_values = extract_dtype_values(api_calls['DataTypeList'])
    for dt in dtype_values:
        if dt not in VALID_DTYPES:
            violations.append(Violation(
                file=file_path, line=line,
                io_type=block['io_type'], io_name=block['io_name'],
                severity='ERROR', rule='V1',
                message=f"{io_desc}: invalid dtype 'ge::{dt}'"
            ))

    # V2: 格式取值校验
    fmt_values = []
    if has_format:
        fmt_values = extract_format_values(api_calls['Format'])
    elif has_formatlist:
        fmt_values = extract_format_values(api_calls['FormatList'])
    for fmt in fmt_values:
        if fmt not in VALID_FORMATS:
            violations.append(Violation(
                file=file_path, line=line,
                io_type=block['io_type'], io_name=block['io_name'],
                severity='ERROR', rule='V2',
                message=f"{io_desc}: invalid format 'ge::{fmt}'"
            ))

    # UnknownShapeFormat 取值校验
    if has_unknown:
        unk_values = extract_format_values(api_calls['UnknownShapeFormat'])
        for fmt in unk_values:
            if fmt not in VALID_FORMATS:
                violations.append(Violation(
                    file=file_path, line=line,
                    io_type=block['io_type'], io_name=block['io_name'],
                    severity='ERROR', rule='V2',
                    message=f"{io_desc}: invalid UnknownShapeFormat 'ge::{fmt}'"
                ))

    return violations


def check_aicore(content, file_path):
    """检查 AICore 配置规则 A1-A2。"""
    violations = []
    add_config_count = count_add_config(content)

    # A1: 必须有至少一个 AddConfig
    if add_config_count == 0:
        # AICPU 算子可能用 AICPU() 而非 AICore()
        if 'AICPU(' not in content and 'ApplyTransformerAicpuDefaultCfg' not in content:
            violations.append(Violation(
                file=file_path, line=1, io_type='FILE', io_name='',
                severity='ERROR', rule='A1',
                message='no AddConfig found (and not an AICPU op)'
            ))

    # A2: SOC 名应为已知合法值
    socs = extract_add_config_socs(content)
    for soc in socs:
        if soc not in KNOWN_SOCS:
            violations.append(Violation(
                file=file_path, line=1, io_type='FILE', io_name='',
                severity='WARNING', rule='A2',
                message=f"AddConfig SOC '{soc}' is not in known SOC list: {sorted(KNOWN_SOCS)}"
            ))

    return violations


def check_attrs(content, file_path):
    """检查 Attr 规则 T1-T2。"""
    violations = []

    # T1: AttrType 取值校验
    pattern = r'\.AttrType\(\s*(\w+)\s*\)'
    for match in re.finditer(pattern, content):
        line = content[:match.start()].count('\n') + 1
        value = match.group(1)
        if value not in VALID_ATTR_TYPES:
            violations.append(Violation(
                file=file_path, line=line, io_type='Attr', io_name='',
                severity='ERROR', rule='T1',
                message=f"AttrType '{value}' not in {sorted(VALID_ATTR_TYPES)}"
            ))

    # T2: attr 名不应与保留字冲突
    attr_names = extract_attr_names(content)
    for name in attr_names:
        if name in CONFLICT_KEYWORDS:
            # 找到行号
            pattern_name = re.compile(r'this->Attr\(\s*"' + re.escape(name) + r'"')
            match = pattern_name.search(content)
            line = content[:match.start()].count('\n') + 1 if match else 1
            violations.append(Violation(
                file=file_path, line=line, io_type='Attr', io_name=name,
                severity='WARNING', rule='T2',
                message=f"attr name '{name}' conflicts with built-in keyword"
            ))

    return violations


def check_class_op_add_match(content, file_path):
    """检查类名与 OP_ADD 注册名是否一致。"""
    violations = []
    class_name = extract_class_name(content)
    op_add_name = extract_op_add_name(content)
    if class_name and op_add_name and class_name != op_add_name:
        violations.append(Violation(
            file=file_path, line=1, io_type='FILE', io_name='',
            severity='ERROR', rule='S4',
            message=f"class name '{class_name}' != OP_ADD name '{op_add_name}'"
        ))
    return violations


# ============================================================
# 主检查流程
# ============================================================

def check_file(file_path):
    """检查单个 _def.cpp 文件。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    violations = []

    # 判断是否为 AICPU 算子
    is_aicpu = 'AICPU(' in content or 'ApplyTransformerAicpuDefaultCfg' in content

    # 结构规则
    violations.extend(check_structure(content, file_path))
    violations.extend(check_class_op_add_match(content, file_path))

    # AICore 配置规则
    violations.extend(check_aicore(content, file_path))

    # Input/Output 规则
    blocks = parse_io_blocks(content)
    for block in blocks:
        violations.extend(check_io_block(block, file_path, is_aicpu))

    # Attr 规则
    violations.extend(check_attrs(content, file_path))

    return violations


EXCLUDE_DIRS = {"third_party", ".git", "build", "cmake"}


def find_def_files(path):
    """查找路径下的所有算子 _def.cpp 文件。

    过滤规则:
      - 排除 third_party/.git/build/cmake 目录
      - 文件内容必须含 'OpDef' 或 'OP_ADD'（排除非算子的 _def.cpp）
    """
    if os.path.isfile(path):
        return [path]
    results = []
    for p in Path(path).rglob("*_def.cpp"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'OpDef' in content or 'OP_ADD' in content:
                results.append(str(p))
        except Exception:
            continue
    return results


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file_or_dir> [<file_or_dir> ...]")
        sys.exit(1)

    all_violations = []
    for path in sys.argv[1:]:
        files = find_def_files(path)
        for file_path in files:
            violations = check_file(file_path)
            all_violations.extend(violations)

    errors = [v for v in all_violations if v.severity == 'ERROR']
    warnings = [v for v in all_violations if v.severity == 'WARNING']

    if not all_violations:
        print("[PASS] No violations found.")
        sys.exit(0)

    for v in all_violations:
        print(f"  [{v.severity}] {v.file}:{v.line} [{v.rule}] {v.message}")

    print(f"\nTotal: {len(errors)} ERROR(s), {len(warnings)} WARNING(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
