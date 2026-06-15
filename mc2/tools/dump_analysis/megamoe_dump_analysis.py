import sys
import numpy as np
import os
import logging

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


# dump文件解析流程
def parse_input_from_dump_file(dump_folder_path):
    # 判断路径下输入、输出、workspace是否已经解析完成，如果未解析，利用$TOOL_PATH/tools/msaicerr/msaicerr.py工具先解析
    dump_filename_prefix = "exception_info"

    # 检查tool_path路径下是否存在exception_info开头且.bin结尾的文件
    has_parsed_file = False
    for file in os.listdir(dump_folder_path):
        if file.startswith(dump_filename_prefix) and file.endswith(".bin"):
            has_parsed_file = True
            logging.info("路径下发现解析后的%s文件, 跳过解析流程", os.path.join(dump_folder_path, file))
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
            logging.info("路径下发现未解析的%s文件, 准备解析", unparse_file_path) 
            os.system(f"python3 {msaicerr_path} -d {unparse_file_path}")
            logging.info("解析完成, 退出解析流程, 进行后续分析") 
            return


def parse_adump_header(data_u32):
    # 按照AdumpHeader的格式解析数据
    class AdumpHeader:
        def __init__(self, data_u32):
            self.magic = data_u32[0] << 32 | data_u32[1]
            self.execTimes = data_u32[2] << 32 | data_u32[3]
            self.version = data_u32[4]
            self.opType = data_u32[5]
            self.stage_num = data_u32[6]
            self.timeOutCycles = data_u32[7] << 32 | data_u32[8]
            self.reserve = data_u32[9:17]
    # 解析AdumpHeader
    adump_header = AdumpHeader(data_u32)
    logging.info("magic: %d", adump_header.magic)
    logging.info("execTimes: %d", adump_header.execTimes)
    logging.info("version: %d", adump_header.version)
    logging.info("opType: %d", adump_header.opType)
    logging.info("stage_num: %d", adump_header.stage_num)
    logging.info("timeOutCycles: %d", adump_header.timeOutCycles)
    logging.info("reserve: %s", adump_header.reserve)


def parse_adump_shared_field(data_u32):
    CCL_BUFFER_SIZE_POS = 0
    WORLD_SIZE_POS = 1
    RANK_ID_POS = 2
    AIV_NUM_POS = 3
    M_POS = 4
    K_POS = 5
    N_POS = 6
    IS_TRANSPOSE_W1_POS = 7
    IS_TRANSPOSE_W2_POS = 8
    HAS_BIAS1_POS = 9
    HAS_BIAS2_POS = 10
    EXPERT_NUM_PER_RANK_POS = 11
    TOPK_POS = 12
    MAX_RECV_TOKEN_NUM_POS = 13
    DISPATCH_QUANT_MODE_POS = 14
    DISPATCH_QUANT_OUT_DTYPE_POS = 15
    COMBINE_QUANT_MODE_POS = 16
    COMM_ALG_CODE_POS = 17
    NUM_MAX_TOKENS_PER_RANK_POS = 18
    ACTIVATION_CODE_POS = 19
    ACTIVATION_CLAMP_POS = 20
    # 按照MegaMoeAdumpSharedField的格式解析数据
    class MegaMoeAdumpSharedField:
        def __init__(self, data_u32):
            self.cclBufferSize = data_u32[CCL_BUFFER_SIZE_POS]
            self.worldSize = data_u32[WORLD_SIZE_POS]
            self.rankId = data_u32[RANK_ID_POS]
            self.aivNum = data_u32[AIV_NUM_POS]
            self.m = data_u32[M_POS]
            self.k = data_u32[K_POS]
            self.n = data_u32[N_POS]
            self.isTransposeW1 = data_u32[IS_TRANSPOSE_W1_POS]
            self.isTransposeW2 = data_u32[IS_TRANSPOSE_W2_POS]
            self.hasBias1 = data_u32[HAS_BIAS1_POS]
            self.hasBias2 = data_u32[HAS_BIAS2_POS]
            self.expertNumPerRank = data_u32[EXPERT_NUM_PER_RANK_POS]
            self.topK = data_u32[TOPK_POS]
            self.maxRecvTokenNum = data_u32[MAX_RECV_TOKEN_NUM_POS]
            self.dispatchQuantMode = data_u32[DISPATCH_QUANT_MODE_POS]
            self.dispatchQuantOutDtype = data_u32[DISPATCH_QUANT_OUT_DTYPE_POS]
            self.combineQuantMode = data_u32[COMBINE_QUANT_MODE_POS]
            self.commAlgCode = data_u32[COMM_ALG_CODE_POS]
            self.numMaxTokensPerRank = data_u32[NUM_MAX_TOKENS_PER_RANK_POS]
            self.activationCode = data_u32[ACTIVATION_CODE_POS]
            self.activationClamp = data_u32[ACTIVATION_CLAMP_POS]

    adump_shared_field = MegaMoeAdumpSharedField(data_u32[128:])
    logging.info("cclBufferSize: %d", adump_shared_field.cclBufferSize)
    logging.info("worldSize: %d", adump_shared_field.worldSize)
    logging.info("rankId: %d", adump_shared_field.rankId)
    logging.info("aivNum: %d", adump_shared_field.aivNum)
    logging.info("m: %d", adump_shared_field.m)
    logging.info("k: %d", adump_shared_field.k)
    logging.info("n: %d", adump_shared_field.n)
    logging.info("isTransposeW1: %d", adump_shared_field.isTransposeW1)
    logging.info("isTransposeW2: %d", adump_shared_field.isTransposeW2)
    logging.info("hasBias1: %d", adump_shared_field.hasBias1)
    logging.info("hasBias2: %d", adump_shared_field.hasBias2)
    logging.info("expertNumPerRank: %d", adump_shared_field.expertNumPerRank)
    logging.info("topK: %d", adump_shared_field.topK)
    logging.info("maxRecvTokenNum: %d", adump_shared_field.maxRecvTokenNum)
    logging.info("dispatchQuantMode: %d", adump_shared_field.dispatchQuantMode)
    logging.info("dispatchQuantOutDtype: %d", adump_shared_field.dispatchQuantOutDtype)
    logging.info("combineQuantMode: %d", adump_shared_field.combineQuantMode)
    logging.info("commAlgCode: %d", adump_shared_field.commAlgCode)
    logging.info("numMaxTokensPerRank: %d", adump_shared_field.numMaxTokensPerRank)
    logging.info("activationCode: %d", adump_shared_field.activationCode)
    logging.info("activationClamp: %d", adump_shared_field.activationClamp)


# 解析mc2自定义adump文件
def parse_mc2_from_adump_file(dump_folder_path):
    adump_filename_prefix = "mc2_exception_info_MegaMoe"
    # 检查路径下是否存在mc2_exception_info_MegaMoe开头的文件
    mc2_adump_file = None
    for file in os.listdir(dump_folder_path):
        if file.startswith(adump_filename_prefix):
            mc2_adump_file = os.path.join(dump_folder_path, file)
            logging.info("路径下发现adump文件", mc2_adump_file)
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
        parse_adump_shared_field(data_u32[128:])


# 调用parse_input_from_dump_file函数进行文件解析流程控制
parse_input_from_dump_file(dump_folder_path)
# 调用parse_mc2_from_adump_file函数解析adump文件
parse_mc2_from_adump_file(dump_folder_path)
