import sys
import numpy as np
import os
import logging
import openpyxl
import subprocess

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )


# 从参数中获取文件夹路径、rank_size和soc_version等参数
dump_folder_path = sys.argv[1]
rank_size = int(sys.argv[2])
soc_version = sys.argv[3]
rank_id = int(sys.argv[4])
tool_path = sys.argv[5]


_xlsx_fields = []
def write_field_to_xlsx(field_name, field_value, sheet_name="default", row_group=0, xlsx_file_path="adump_parse.xlsx"):
    global _xlsx_fields
    _xlsx_fields.append((field_name, field_value, sheet_name, row_group, xlsx_file_path))


def flush_xlsx():
    """
    将收集的字段数据写入Excel文件并清空缓存
    
    功能说明：
    1. 从全局变量_xlsx_fields中读取所有待写入Excel的字段数据
    2. 按Excel文件路径、工作表名称、行组进行数据分组
    3. 为每个Excel文件创建工作簿并写入数据
    4. 清空_xlsx_fields缓存，准备下次数据收集
    """
    global _xlsx_fields
    
    # 按文件路径、工作表、行组对字段数据进行分组
    xlsx_groups = {}
    for field_name, field_value, sheet_name, row_group, xlsx_file_path in _xlsx_fields:
        # 使用setdefault创建嵌套字典结构：文件路径 -> 工作表 -> 行组 -> 字段列表
        xlsx_groups.setdefault(xlsx_file_path, {}).setdefault(sheet_name, {}).setdefault(row_group, []).append((field_name, field_value))
    
    # 遍历每个Excel文件的分组数据
    for xlsx_file_path, sheets in xlsx_groups.items():
        # 创建新的Excel工作簿
        wb = openpyxl.Workbook()
        first_sheet = True  # 标记是否为第一个工作表（使用默认工作表）
        
        # 遍历当前文件的所有工作表
        for sheet_name, row_groups in sheets.items():
            # 创建或激活工作表
            if first_sheet:
                ws = wb.active  # 使用默认工作表
                ws.title = sheet_name
                first_sheet = False
            else:
                ws = wb.create_sheet(title=sheet_name)  # 创建新工作表
            # 对行组进行排序，确保数据顺序一致
            sorted_groups = sorted(row_groups.items())
            # 提取第一个行组的字段名作为表头
            first_group_fields = sorted_groups[0][1]
            ws.append([name for name, _ in first_group_fields])
            # 写入每个行组的数据
            for _, fields in sorted_groups:
                ws.append([val for _, val in fields])
        # 保存Excel文件
        wb.save(xlsx_file_path)
    # 清空字段缓存
    _xlsx_fields = []


# dump文件解析流程
def parse_input_from_dump_file(dump_folder_path):
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
            self.execTimes = data_u32[3] << 32 | data_u32[2]
            self.version = data_u32[4]
            self.opType = data_u32[5]
            self.stage_num = data_u32[6]
            self.reserve = data_u32[7]
            self.timeOutCycles = data_u32[9] << 32 | data_u32[8]
            # self.reserve = data_u32[9:17]
    # 解析AdumpHeader
    adump_header = AdumpHeader(data_u32)
    # 将AdumpHeader写入csv文件中
    write_field_to_xlsx("magic", adump_header.magic, sheet_name="header")
    write_field_to_xlsx("execTimes", adump_header.execTimes, sheet_name="header")
    write_field_to_xlsx("version", adump_header.version, sheet_name="header")
    write_field_to_xlsx("opType", adump_header.opType, sheet_name="header")
    write_field_to_xlsx("stage_num", adump_header.stage_num, sheet_name="header")
    write_field_to_xlsx("timeOutCycles", adump_header.timeOutCycles, sheet_name="header")
    write_field_to_xlsx("reserve", adump_header.reserve, sheet_name="header")


def parse_adump_shared_field(data_u32) -> int:
    # 按照MegaMoeAdumpSharedField的格式解析数据
    class MegaMoeAdumpSharedField:
        def __init__(self, data_u32):
            self.cclBufferSize = data_u32[0]
            self.worldSize = data_u32[1]
            self.rankId = data_u32[2]
            self.aivNum = data_u32[3]
            self.m = data_u32[4]
            self.k = data_u32[5]
            self.n = data_u32[6]
            self.isTransposeW1 = data_u32[7]
            self.isTransposeW2 = data_u32[8]
            self.hasBias1 = data_u32[9]
            self.hasBias2 = data_u32[10]
            self.expertNumPerRank = data_u32[11]
            self.topK = data_u32[12]
            self.maxRecvTokenNum = data_u32[13]
            self.dispatchQuantMode = data_u32[14]
            self.dispatchQuantOutDtype = data_u32[15]
            self.combineQuantMode = data_u32[16]
            self.commAlgCode = data_u32[17]
            self.numMaxTokensPerRank = data_u32[18]
            self.activationCode = data_u32[19]
            self.activationClamp = data_u32[20]

    adump_shared_field = MegaMoeAdumpSharedField(data_u32)
    # 将shared field写入csv文件中
    write_field_to_xlsx("cclBufferSize", adump_shared_field.cclBufferSize, sheet_name="shared field")
    write_field_to_xlsx("worldSize", adump_shared_field.worldSize, sheet_name="shared field")
    write_field_to_xlsx("rankId", adump_shared_field.rankId, sheet_name="shared field")
    write_field_to_xlsx("aivNum", adump_shared_field.aivNum, sheet_name="shared field")
    write_field_to_xlsx("m", adump_shared_field.m, sheet_name="shared field")
    write_field_to_xlsx("k", adump_shared_field.k, sheet_name="shared field")
    write_field_to_xlsx("n", adump_shared_field.n, sheet_name="shared field")
    write_field_to_xlsx("isTransposeW1", adump_shared_field.isTransposeW1, sheet_name="shared field")
    write_field_to_xlsx("isTransposeW2", adump_shared_field.isTransposeW2, sheet_name="shared field")
    write_field_to_xlsx("hasBias1", adump_shared_field.hasBias1, sheet_name="shared field")
    write_field_to_xlsx("hasBias2", adump_shared_field.hasBias2, sheet_name="shared field")
    write_field_to_xlsx("expertNumPerRank", adump_shared_field.expertNumPerRank, sheet_name="shared field")
    write_field_to_xlsx("topK", adump_shared_field.topK, sheet_name="shared field")
    write_field_to_xlsx("maxRecvTokenNum", adump_shared_field.maxRecvTokenNum, sheet_name="shared field")
    write_field_to_xlsx("dispatchQuantMode", adump_shared_field.dispatchQuantMode, sheet_name="shared field")
    write_field_to_xlsx("dispatchQuantOutDtype", adump_shared_field.dispatchQuantOutDtype, sheet_name="shared field")
    write_field_to_xlsx("combineQuantMode", adump_shared_field.combineQuantMode, sheet_name="shared field")
    write_field_to_xlsx("commAlgCode", adump_shared_field.commAlgCode, sheet_name="shared field")
    write_field_to_xlsx("numMaxTokensPerRank", adump_shared_field.numMaxTokensPerRank, sheet_name="shared field")
    write_field_to_xlsx("activationCode", adump_shared_field.activationCode, sheet_name="shared field")
    write_field_to_xlsx("activationClamp", adump_shared_field.activationClamp, sheet_name="shared field")

    return adump_shared_field.aivNum
    

# 解析adump中custom字段，每个aiv核由自己的custom field
# 第一个aiv核的起始地址为class AdumpHeader的大小(按照512B对齐) + class MegaMoeAdumpCustomField的大小(按照512B对齐)
# 其余aiv和的偏移在第前一个aiv核的基础上增加class MegaMoeAdumpCustomField的大小(按照512B对齐)
def parse_adump_custom_field(data_u32, aiv_id):
    class MegaMoeAdumpCustomField:
        def __init__(self, data_u32):
            self.aivId = data_u32[0]
            self.currentStageId = data_u32[1]
            self.startTimeStampHigh32Bit = data_u32[2]
            self.startTimeStampLow32Bit = data_u32[3]
    adump_custom_field = MegaMoeAdumpCustomField(data_u32)
    # 将custom field写入csv文件中
    write_field_to_xlsx("aivId", adump_custom_field.aivId, sheet_name="custom field", row_group=aiv_id)
    write_field_to_xlsx("currentStageId", adump_custom_field.currentStageId, sheet_name="custom field", row_group=aiv_id)
    write_field_to_xlsx("startTimeStampHigh32Bit", adump_custom_field.startTimeStampHigh32Bit, sheet_name="custom field", row_group=aiv_id)
    write_field_to_xlsx("startTimeStampLow32Bit", adump_custom_field.startTimeStampLow32Bit, sheet_name="custom field", row_group=aiv_id)


# 解析mc2自定义adump文件
def parse_mc2_from_adump_file(dump_folder_path):
    adump_filename_prefix = "mc2_exception_info_MegaMoe"
    # 检查路径下是否存在mc2_exception_info_MegaMoe开头的文件
    mc2_adump_file = None
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
        # 解析AdumpHeader
        parse_adump_header(data_u32)
        # 解析AdumpSharedField
        aiv_num = parse_adump_shared_field(data_u32[128:])
        # 解析AdumpCustomField
        for aiv_id in range(aiv_num):
            offset = 128 + 128 + aiv_id * 128
            parse_adump_custom_field(data_u32[offset:], aiv_id)


# 调用parse_input_from_dump_file函数进行文件解析流程控制
parse_input_from_dump_file(dump_folder_path)
# 调用parse_mc2_from_adump_file函数解析adump文件
parse_mc2_from_adump_file(dump_folder_path)
# 将收集的字段写入csv文件
flush_xlsx()
