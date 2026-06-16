#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

# !
#  \file megamoe_dump_analysis_a3.py
#  \brief 解析megamoe自定义adump文件

import os
import sys
import logging
import openpyxl
import subprocess
import argparse
import numpy as np

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )


_xlsx_fields = []


def write_field_to_xlsx(field_name, field_value, sheet_name="default", row_group=0, xlsx_file_path="adump_parse.xlsx"):
    global _xlsx_fields
    _xlsx_fields.append((field_name, field_value, sheet_name, row_group, xlsx_file_path))


def flush_xlsx():
    """
    将收集的字段数据写入Excel文件并清空缓存
    """
    global _xlsx_fields

    # 按文件路径、工作表、行组对字段数据进行分组
    xlsx_groups = {}
    for field_name, field_value, sheet_name, row_group, xlsx_file_path in _xlsx_fields:
        # 使用setdefault创建嵌套字典结构：文件路径 -> 工作表 -> 行组 -> 字段列表
        xlsx_groups.setdefault(xlsx_file_path, {}).setdefault(sheet_name, {}).setdefault(row_group, []).append((field_name, field_value))

    for xlsx_file_path, sheets in xlsx_groups.items(): # 遍历每个Excel文件的分组数据
        wb = openpyxl.Workbook()
        first_sheet = True

        for sheet_name, row_groups in sheets.items(): # 遍历当前文件的所有工作表
            if first_sheet:
                ws = wb.active
                ws.title = sheet_name
                first_sheet = False
            else:
                ws = wb.create_sheet(title=sheet_name)
            sorted_groups = sorted(row_groups.items()) # 对行组进行排序，确保数据顺序一致
            first_group_fields = sorted_groups[0][1] # 提取第一个行组的字段名作为表头
            ws.append([name for name, _ in first_group_fields])
            for _, fields in sorted_groups:
                ws.append([val for _, val in fields])
        wb.save(xlsx_file_path)
    _xlsx_fields = [] # 清空字段缓存


# dump文件解析流程
def parse_input_from_dump_file(dump_folder_path, tool_path):
    # 判断路径下输入、输出、workspace是否已经解析完成，如果未解析，利用$TOOL_PATH/tools/msaicerr/msaicerr.py工具先解析
    dump_filename_prefix = "exception_info"

    # 检查tool_path路径下是否存在exception_info开头且.bin结尾的文件
    has_parsed_file = False
    for file in os.listdir(dump_folder_path):
        if file.startswith(dump_filename_prefix) and file.endswith(".bin"):
            has_parsed_file = True
            logging.info("路径下发现解析后的文件, 跳过解析流程, 文件：%s", os.path.join(dump_folder_path, file))
            return

    # 如果不存在已解析的.bin文件，检查是否有未解析的exception_info开头文件
    if not has_parsed_file:
        unparse_file_path = None
        for file in os.listdir(dump_folder_path):
            if file.startswith(dump_filename_prefix):
                unparse_file_path = os.path.join(dump_folder_path, file)
                break
        if not unparse_file_path:
            logging.warning("dump_folder_path路径下未发现exception_info相关文件, 请检查输入参数")
            return
        else:
            # 利用$TOOL_PATH/tools/msaicerr/msaicerr.py工具解析
            msaicerr_path = os.path.join(tool_path, "tools/msaicerr/msaicerr.py")
            if not os.path.exists(msaicerr_path):
                logging.error("未发现$TOOL_PATH/tools/msaicerr/msaicerr.py解析工具, 请检查环境变量设置")
                sys.exit(1)
            logging.info("路径下发现未解析的文件, 准备解析, 文件：%s", unparse_file_path)
            subprocess.run(["python3", msaicerr_path, "-d", unparse_file_path], check=True)
            logging.info("解析完成, 退出解析流程, 进行后续分析")
            return


def parse_adump_header(data_u32):
    # 按照AdumpHeader的格式解析数据
    class AdumpHeader:
        def __init__(self, data_u32):
            self.magic = data_u32[1] << 32 | data_u32[0]
            self.exec_times = data_u32[3] << 32 | data_u32[2]
            self.version = data_u32[4]
            self.op_type = data_u32[5]
            self.stage_num = data_u32[6]
            self.reserve = data_u32[7]
            self.time_out_cycles = data_u32[9] << 32 | data_u32[8]

    # 解析AdumpHeader
    adump_header = AdumpHeader(data_u32)

    write_field_to_xlsx("magic", adump_header.magic, sheet_name="header")
    write_field_to_xlsx("execTimes", adump_header.exec_times, sheet_name="header")
    write_field_to_xlsx("version", adump_header.version, sheet_name="header")
    write_field_to_xlsx("opType", adump_header.op_type, sheet_name="header")
    write_field_to_xlsx("stageNum", adump_header.stage_num, sheet_name="header")
    write_field_to_xlsx("timeOutCycles", adump_header.time_out_cycles, sheet_name="header")


def parse_adump_shared_field(data_u32) -> int:
    # 按照MegaMoeAdumpSharedField的格式解析数据
    class MegaMoeAdumpSharedField:
        def __init__(self, data_u32):
            self.ccl_buffer_size = data_u32[0]
            self.world_size = data_u32[1]
            self.rank_id = data_u32[2]
            self.aiv_num = data_u32[3]

    adump_shared_field = MegaMoeAdumpSharedField(data_u32)

    write_field_to_xlsx("cclBufferSize", adump_shared_field.ccl_buffer_size, sheet_name="shared field")
    write_field_to_xlsx("worldSize", adump_shared_field.world_size, sheet_name="shared field")
    write_field_to_xlsx("rankId", adump_shared_field.rank_id, sheet_name="shared field")
    write_field_to_xlsx("aivNum", adump_shared_field.aiv_num, sheet_name="shared field")

    return adump_shared_field.aiv_num


# 解析adump中custom字段，每个aiv核有自己的custom field
def parse_adump_custom_field(data_u32, aiv_id):
    class MegaMoeAdumpCustomField:
        def __init__(self, data_u32):
            self.aiv_id = data_u32[0]
            self.current_stage_id = data_u32[1]
            self.start_time_stamp = data_u32[3] << 32 | data_u32[2]

    adump_custom_field = MegaMoeAdumpCustomField(data_u32)

    write_field_to_xlsx("aivId", adump_custom_field.aiv_id, sheet_name="custom field", row_group=aiv_id)
    write_field_to_xlsx("currentStageId", adump_custom_field.current_stage_id, sheet_name="custom field", row_group=aiv_id)
    write_field_to_xlsx("startTimeStamp", adump_custom_field.start_time_stamp, sheet_name="custom field", row_group=aiv_id)


# 解析mc2自定义adump文件
def parse_mc2_from_adump_file(dump_folder_path):
    adump_filename_prefix = "mc2_exception_info_MegaMoe"
    mc2_adump_file = None # 检查路径下是否存在mc2_exception_info_MegaMoe开头的文件
    for file in os.listdir(dump_folder_path):
        if file.startswith(adump_filename_prefix):
            mc2_adump_file = os.path.join(dump_folder_path, file)
            logging.info("路径下发现adump文件, 开始解析: %s", mc2_adump_file)
            break
    if not mc2_adump_file:
        logging.error("路径下未发现adump相关文件, 请检查路径是否正确")
        sys.exit(1)

    # 读取adump文件内容
    with open(mc2_adump_file, "rb") as f:
        data = f.read()
        data_u32 = np.frombuffer(data, dtype=np.uint32)

        parse_adump_header(data_u32) # 解析AdumpHeader
        aiv_num = parse_adump_shared_field(data_u32[128:]) # 解析AdumpSharedField
        for aiv_id in range(aiv_num): # 解析AdumpCustomField
            offset = 128 + 128 + aiv_id * 128 # 这里128是因为kernel侧512B对齐，两者保持一致才能正确解析
            parse_adump_custom_field(data_u32[offset:], aiv_id)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="MegaMoe dump文件分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
      示例:
      python megamoe_dump_analysis_a3.py <dump_folder_path> <tool_path>
      python megamoe_dump_analysis_a3.py ./dumps ./tools
        """
    )
    parser.add_argument("dump_folder_path", help="dump文件夹路径, 包含exception_info或mc2_exception_info_MegaMoe开头的文件")
    parser.add_argument("tool_path", help="工具路径, 包含tools/msaicerr/msaicerr.py等解析工具")
    args = parser.parse_args()
    dump_folder_path = args.dump_folder_path
    tool_path = args.tool_path

    # 验证输入路径
    if not os.path.isdir(dump_folder_path):
        logging.error(f"dump文件夹路径不存在: {dump_folder_path}")
        sys.exit(1)
    if not os.path.isdir(tool_path):
        logging.error(f"工具路径不存在: {tool_path}")
        sys.exit(1)

    # 执行分析流程
    logging.info("开始分析dump文件...")
    logging.info("dump文件夹路径: %s", dump_folder_path)
    logging.info("工具路径: %s", tool_path)

    # 解析原始dump文件
    parse_input_from_dump_file(dump_folder_path, tool_path)

    # 解析MC2格式的adump文件
    parse_mc2_from_adump_file(dump_folder_path)

    # 将解析结果写入Excel并清理缓存
    flush_xlsx()
    logging.info("dump文件分析完成, 结果已保存至Excel文件")
