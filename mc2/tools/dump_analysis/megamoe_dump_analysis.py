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
folder_path = sys.argv[1]
rank_size = int(sys.argv[2])
soc_version = sys.argv[3]
rank_id = int(sys.argv[4])
tool_path = sys.argv[5]


# 判断路径下输入、输出、workspace是否已经解析完成，如果未解析，利用$TOOL_PATH/tools/msaicerr/msaicerr.py工具先解析
dump_filename_prefix = "exception_info"
for file_name in os.listdir(folder_path):
    # 检查文件名中是否包含'exception_info.*.bin'
    if file_name.startswith(dump_filename_prefix) and file_name.endswith(".bin"):
        file_path = os.path.join(folder_path, file_name)
        logging.info("路径下发现解析后的%s文件, 跳过解析流程", file_path) 
    elif:
        # 查找解析工具$TOOL_PATH/tools/msaicerr/msaicerr.py是否存在
        msaicerr_path = os.path.join(tool_path, "tools/msaicerr/msaicerr.py")
        if not os.path.exists(msaicerr_path):
            logging.error("未发现$TOOL_PATH/tools/msaicerr/msaicerr.py解析工具, 请检查环境变量设置")
            sys.exit(1)

        # 查找路径下是否由未解析的dump文件
        if file_name.startswith(dump_filename_prefix):
            logging.info("路径下发现未解析的%s文件, 准备解析", file_path) 
            os.system(f"python3 {msaicerr_path} -d {file_path}")
            logging.info("解析完成, 退出解析流程, 进行后续分析") 
        break
        

adump_filename_prefix = "mc2_exception_info_MegaMoe"

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
  

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件名是否以指定前缀开头
    if file_name.startswith(adump_filename_prefix):
        file_path = os.path.join(folder_path, file_name)
        logging.info("处理文件: %s", file_path)
        with open(file_path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.int8)
        
        data_u32 = arr.view(np.uint32)
        dump_header = AdumpHeader(data_u32)
        logging.info("magic: %d", dump_header.magic)
        logging.info("execTimes: %d", dump_header.execTimes)
        logging.info("version: %d", dump_header.version)
        logging.info("opType: %d", dump_header.opType)
        logging.info("stage_num: %d", dump_header.stage_num)
        logging.info("timeOutCycles: %d", dump_header.timeOutCycles)
        logging.info("reserve: %s", dump_header.reserve)
        dump_shared_field = MegaMoeAdumpSharedField(data_u32[128:])
        logging.info("cclBufferSize: %d", dump_shared_field.cclBufferSize)
        logging.info("worldSize: %d", dump_shared_field.worldSize)
        logging.info("rankId: %d", dump_shared_field.rankId)
        logging.info("aivNum: %d", dump_shared_field.aivNum)
        logging.info("m: %d", dump_shared_field.m)
        logging.info("k: %d", dump_shared_field.k)
        logging.info("n: %d", dump_shared_field.n)
        logging.info("isTransposeW1: %d", dump_shared_field.isTransposeW1)
        logging.info("isTransposeW2: %d", dump_shared_field.isTransposeW2)
        logging.info("hasBias1: %d", dump_shared_field.hasBias1)
        logging.info("hasBias2: %d", dump_shared_field.hasBias2)
        logging.info("expertNumPerRank: %d", dump_shared_field.expertNumPerRank)
        logging.info("topK: %d", dump_shared_field.topK)
        logging.info("maxRecvTokenNum: %d", dump_shared_field.maxRecvTokenNum)
        logging.info("dispatchQuantMode: %d", dump_shared_field.dispatchQuantMode)
        logging.info("dispatchQuantOutDtype: %d", dump_shared_field.dispatchQuantOutDtype)
        logging.info("combineQuantMode: %d", dump_shared_field.combineQuantMode)
        logging.info("commAlgCode: %d", dump_shared_field.commAlgCode)
        logging.info("numMaxTokensPerRank: %d", dump_shared_field.numMaxTokensPerRank)
        logging.info("activationCode: %d", dump_shared_field.activationCode)
        logging.info("activationClamp: %d", dump_shared_field.activationClamp)

