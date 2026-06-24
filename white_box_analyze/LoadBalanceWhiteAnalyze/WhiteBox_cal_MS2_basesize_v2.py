import pandas as pd
import json
import pandas as pd
import os
from pathlib import Path
import csv
import re
import traceback
import math
mBaseSize_nums  = [32, 64, 128, 256, 512, 1024, 2048]
s2BaesSize_nums = [32, 64, 128, 256, 512, 1024, 2048]
BATCH_CONTINUOUS = 0
TENSOR_LIST = 1
PA_ATTENTION = 3

splitclasskey  = []
for i in mBaseSize_nums:
    for j in s2BaesSize_nums:
        splitclasskey.append((i,j))

s1g_class_num = 2
batch_class_num = 5
sparsemode_class_num = 5
split_class_num = 4
n2_class_num = 5
g_class_num = 5



encode_key_list = []
split_key_list = []
n2_list = []
g_list = []

def get_s1_s2_baseline(s1Size,s2Size,inputlayout,block_num,kv_cache_mode,block_size,block_table):
    if kv_cache_mode == BATCH_CONTINUOUS or kv_cache_mode == TENSOR_LIST:
        s1_baseline = s1Size
        s2_baseline = s2Size
    elif kv_cache_mode == PA_ATTENTION:
        s1_baseline = s1Size
        s2_baseline = block_table[1] *block_size
    else:
        print("kv_cache_mode error")
        raise ValueError


    return s1_baseline,s2_baseline

def get_mBaseSize_and_s2BaseSize(inputlayout,q_s,kv_s,s1Size,s2Size,kv_cache_mode,block_num,block_size,sparse_mode,attenMaskFlag,g,block_table):
    # 计算sInnerSize
    s1_baseline,s2_baseline = get_s1_s2_baseline(s1Size,s2Size,inputlayout,block_num,kv_cache_mode,block_size,block_table)
    # print("s1_baseline,s2_baseline",s1_baseline,s2_baseline)
    if inputlayout == "TND" or inputlayout == "NTD":
        sInnerSize = 512
    else:
        if s1_baseline <= 16:
            sInnerSize = 8192
            sInnerSize_ = [8192, 4096, 2048]
            sInnerSize = sInnerSize_[min(int(g // 5), 2)]
        else:
            sInnerSize = 512 
    if attenMaskFlag and (sparse_mode == 2 or sparse_mode == 3 or sparse_mode == 4):
        sInnerSize = min(sInnerSize, 1024)
    if kv_cache_mode==PA_ATTENTION and block_size != 0:
        if sInnerSize > block_size and sInnerSize % block_size != 0:
            sInnerSize = sInnerSize // block_size * block_size
    if sInnerSize > s2_baseline:
        sInnerSize = s2_baseline
    sInnerSizeAlign = (sInnerSize + 31) // 32 * 32

    # 计算mBaseSize
    if inputlayout == "TND" or inputlayout == "NTD":
        mBaseSize = 512
    else:
        if s1_baseline <= 16:
            if sInnerSizeAlign <= 512:
                mBaseSize = 512
            elif sInnerSizeAlign <= 1024:
                mBaseSize = 256
            elif sInnerSizeAlign <= 2048:
                mBaseSize = 128
            elif sInnerSizeAlign <= 4096:
                mBaseSize = 64
            else:
                mBaseSize = 32
        else:
            mBaseSize = 512
    return mBaseSize,sInnerSize

def check_actualseq(s):
    if not s:
        return "000"
    n = len(s)
    # 找头部连续零的结束位置（第一个非零的位置）
    head_end = 0
    while head_end < n and s[head_end] == 0:
        head_end += 1

    # 找尾部连续零的开始位置
    tail_start = n - 1
    while tail_start >= 0 and s[tail_start] == 0:
        tail_start -= 1
    tail_start += 1  # 尾部连续零起始索引

    # 头部是否有连续零（至少一个0在开头）
    head_flag = '1' if head_end > 0 else '0'

    # 尾部是否有连续零（至少一个0在末尾）
    tail_flag = '1' if tail_start < n else '0'

    # 检查中间部分（head_end 到 tail_start - 1）是否有0
    mid_flag = '0'
    if head_end < tail_start:
        for i in range(head_end, tail_start):
            if s[i] == 0:
                mid_flag = '1'
                break

    return head_flag + mid_flag + tail_flag


def parse_loop_data(file_path):
    """
    解析文本文件中的循环数据，并将其存储为DataFrame
    """
    loop_data = []
    current_loop = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # 检查是否是新的循环开始
            loop_match = re.match(r'============================ Loop (\d+) ============================', line)
            if loop_match:
                if current_loop:  # 如果当前循环有数据，则保存
                    loop_data.append(current_loop)
                current_loop = {'loop': int(loop_match.group(1))}  # 开始新的循环
                continue

            # 解析键值对
            kv_match = re.match(r'(\w+):\s*(.*)', line)
            if kv_match:
                key = kv_match.group(1)
                value = kv_match.group(2)

                # 处理actualSeqS1Size和actualSeqS2Size的特殊格式
                if key in ['actualSeqS1Size', 'actualSeqS2Size']:
                    if value:  # 如果值不为空
                        # 解析 [ 0: 4466, 1: 3715, 2: 3966, 3: 6031, ... ] 这样的格式
                        values = re.findall(r'(\d+):\s*(\d+)', value)
                        value = [int(v) for k, v in values]  # 转换为列表
                    else:  # 如果值为空
                        value = []  # 返回空列表

                current_loop[key] = value

        # 添加最后一个循环的数据
        if current_loop:
            loop_data.append(current_loop)

    # 将数据转换为DataFrame
    df = pd.DataFrame(loop_data)
    return df

def coverage_encoding_batch(items):
    """
    对一个包含 '1' 到 '7' 字符串的列表进行按位或运算，
    计算三种情况（bit0/bit1/bit2）的覆盖情况。
    
    参数:
        items (list of str): 元素为 '1' 到 '7' 的字符串列表
    
    返回:
        dict: {
            'result': int,          # 按位或的结果（覆盖掩码）
            'binary': str,          # 3位二进制字符串，如 '101'
            'covered_cases': list,  # 被覆盖的情况编号列表，如 [0, 2]
            'coverage_count': int,  # 覆盖了多少种情况（0~3）
            'is_single_case': bool  # 是否恰好只覆盖了一种情况（即结果只有一个1）
        }
    """
    if not items:
        raise ValueError("输入列表不能为空")
    
    result = 0
    for s in items:
        num = int(s)
        if not (1 <= num <= 7):
            raise ValueError(f"元素必须在 '1' 到 '7' 之间，得到: {s}")
        result |= num  # 按位或，累积覆盖

    # 3位二进制表示
    binary_str = format(result, '03b')
    
    # 分析哪些位被覆盖（从低位到高位：bit0=情况0，bit1=情况1，bit2=情况2）
    covered_cases = []
    if result & 1:      # bit0
        covered_cases.append(0)
    if result & 2:      # bit1
        covered_cases.append(1)
    if result & 4:      # bit2
        covered_cases.append(2)
    
    coverage_count = len(covered_cases)
    is_single_case = (coverage_count == 1)

    return {
        'result': result,
        'binary': binary_str,
        'covered_cases': covered_cases,
        'coverage_count': coverage_count,
        'is_single_case': is_single_case
    }

def encode_batch(actualseqlengths,actualseqlengthskv):
    if not actualseqlengths and not actualseqlengthskv:
        batch_code  = "0"
    elif (actualseqlengths and  max(actualseqlengths) == 0) or (actualseqlengthskv and max(actualseqlengthskv) == 0):
        batch_code = "8"
    else:
        actualseqlengths_code = check_actualseq(actualseqlengths)
        actualseqlengthskv_code = check_actualseq(actualseqlengthskv)
        batch_code = str((int(actualseqlengths_code,2) | int(actualseqlengthskv_code,2)))
    return batch_code

def encode_S1g(inputlayout):
    if "_" in inputlayout:
        inputlayout = inputlayout.split("_")[0]
    if inputlayout == "BSND" or inputlayout == "BSH" or inputlayout =="TND":
        S1g = "1"
    else:
        S1g = "0"
    return S1g

def calc_S1andS2(layout,Q_shape,K_shape,kv_cache_mode,actualseqlengths_S,actualseqlengths_KV):
    if "_" in layout:
        layout_ = layout.split("_")[0]
    else:
        layout_ = layout
    # Qs = 0
    # Kvs = 0
    if kv_cache_mode == TENSOR_LIST:
        if layout_ == 'TND' or layout_ == 'NTD':
            batch_size = len(actualseqlengths_S)
            Qs = max(actualseqlengths_S)
            Kvs = max(actualseqlengths_KV)
        else:
            if layout_ == "BNSD":
                batch_size = Q_shape[0]
                Qs =  Q_shape[2]
                Kvs = max(shape[2] for shape in K_shape)
            elif layout_ == "BSND":
                batch_size = Q_shape[0]
                Qs =  Q_shape[1]
                Kvs = max(shape[1] for shape in K_shape)
            elif layout_ == "BSH":
                batch_size = Q_shape[0]
                Qs =  Q_shape[1]
                Kvs = max(shape[1] for shape in K_shape)
            elif layout_ == "NSD":
                batch_size = 1
                Qs =  Q_shape[1]
                Kvs = max(shape[1] for shape in K_shape)
    else:
        if layout_ == 'TND' or layout_ == 'NTD':
            batch_size = len(actualseqlengths_S)
            Qs = max(actualseqlengths_S)
            Kvs = max(actualseqlengths_KV)
        elif layout_ == "BNSD":
            batch_size = Q_shape[0]
            Qs =  Q_shape[2]
            Kvs = K_shape[2]
        elif layout_ == "BSND":
            batch_size = Q_shape[0]
            Qs =  Q_shape[1]
            Kvs = K_shape[1]
        elif layout_ == "BSH":
            batch_size = Q_shape[0]
            Qs =  Q_shape[1]
            Kvs = K_shape[1]
        elif layout_ == "NSD":
            batch_size = 1
            Qs =  Q_shape[1]
            Kvs = K_shape[1]
    # if Qs == 0 and Kvs == 0:
    #     print(layout)
    return Qs,Kvs,batch_size



def encode_from_json(json_file_path):
    # Load the JSON file
    ##返回值
    encode_key = ""
    split_key = []
    other_tuple = None

    ##参数
    actualseqlengths = []
    actualseqlengthskv = []
    sparse_mode = "N"
    inputlayout = None
    s1Size = None
    s2Size = None
    paFlag =  False
    pa_blocksize = 0
    innerprecise = 0
    block_table = None
    kv_cache_shape = None
    kv_cache_mode = BATCH_CONTINUOUS

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data["input"]:
        # batch 对空batch情况编码
        if item["name"] == "inputlayout":
            inputlayout = item["value"]
        if item["name"] == "actualseqlengths":
            actualseqlengths = item["value"]
        if item["name"] == "actualseqlengthskv":
            actualseqlengthskv = item["value"]
        if item["name"] == "sparsemode" or item["name"] == "sparse_mode":
            sparse_mode = item["value"]
        if item["name"] == "input0":
            Q_shape = item.get("shape", item.get("shapes", [None])[0] if item.get("shapes") else None)
            Q_type = item["dtype"]
        if item["name"] == "input1":
            K_shape = item.get("shape", item.get("shapes", [None])[0] if item.get("shapes") else None)  # 假设只有一个shape
            if "shapes" in item and len(item["shapes"]) >1:
                kv_cache_mode = TENSOR_LIST
                K_shape = item["shapes"]
                # print(len(item["shapes"]))
                # print("tensor_list")
                # print(K_shape)
                # print(json_file_path)
        if item["name"] == "numheads":
            n1 = item["value"]
        if item["name"] == "numkeyvalueheads":
            n2 = item["value"]
        if item["name"] == "blocksize":
            pa_blocksize = item["value"]
        if item["name"] == "input4":
            attention_mask_shape = item.get("shape", item.get("shapes", [None])[0] if item.get("shapes") else None)
        if item["name"] == "innerprecise":
            innerprecise = item["value"]
        if item["name"] == "input12":
            block_table = item.get("shape", item.get("shapes", [None])[0] if item.get("shapes") else None)
        if item["name"] == "input21":
            kv_cache_shape = item.get("shape", item.get("shapes", [None])[0] if item.get("shapes") else None)
    if pa_blocksize != 0:
        paFlag = True
        kv_cache_mode = PA_ATTENTION
    if attention_mask_shape:
        attenMaskFlag = True
    else:
        attenMaskFlag = False
    if "_" in inputlayout:
        inputlayout = inputlayout.split("_")[0]
    if inputlayout == 'TND' or inputlayout == 'NTD' and actualseqlengths:
            actualseqlengths = [actualseqlengths[0]] + [actualseqlengths[i] - actualseqlengths[i-1] for i in range(1, len(actualseqlengths))] 
    if inputlayout == 'TND' or inputlayout == 'NTD' and actualseqlengthskv:
            actualseqlengthskv = [actualseqlengthskv[0]] + [actualseqlengthskv[i] - actualseqlengthskv[i-1] for i in range(1, len(actualseqlengthskv))] 
    batch_code = encode_batch(actualseqlengths,actualseqlengthskv)
    sparse_mode_code = int(sparse_mode) if sparse_mode !="N" else sparse_mode
    if sparse_mode_code == "N":
        print(f"该用例没有sparsemode属性,用例json名:{json_file_path}")
    S1g_code = encode_S1g(inputlayout)
    encode_key = str(S1g_code) + str(batch_code) + str(sparse_mode_code) 
    if n2 == 0:
        g = 0
    else:
        g = n1/n2
    s1Size,s2Size,batch_size = calc_S1andS2(inputlayout,Q_shape,K_shape,kv_cache_mode,actualseqlengths,actualseqlengthskv)
    if not actualseqlengths:
        actualseqlengths = [s1Size] * batch_size
    if not actualseqlengthskv:
        actualseqlengthskv = [s2Size] * batch_size
    block_num = 0
    if paFlag:
        # for i in actualseqlengthskv:
        #     block_num += math.ceil(i/pa_blocksize)   
        if kv_cache_shape:  
            block_num = kv_cache_shape[0]
    mBaseSize_,s2BaesSize_ = get_mBaseSize_and_s2BaseSize(inputlayout,actualseqlengths,actualseqlengthskv,s1Size,s2Size,kv_cache_mode,block_num,pa_blocksize,sparse_mode,attenMaskFlag,g,block_table)
    s1splitcode = '0'
    s2splitcode = '0'
    for i in range(len(actualseqlengths)):
        if actualseqlengths[i] % mBaseSize_ != 0:
            s1splitcode = '1'
            break
    for j in range(len(actualseqlengthskv)):
        if actualseqlengthskv[j] % s2BaesSize_ != 0:
            s2splitcode = '1'
            break
    # print("mBaseSize,s2BaesSize",mBaseSize_,s2BaesSize_)
    # print('json_file_path',json_file_path)
    splitcode = ''
    if s1splitcode == '0' and s2splitcode == '0':
        splitcode = '0'
    elif s1splitcode == '0' and s2splitcode == '1':
        splitcode = '1'
    elif s1splitcode == '1' and s2splitcode == '0':
        splitcode = '2'
    elif s1splitcode == '1' and s2splitcode == '1':
        splitcode = '3'
    else:
        print(json_file_path)
        raise KeyError("splitcode coding error")
    encode_key = encode_key + splitcode
    return encode_key,n2,g

def Calculate_coverage(output_path):

    encode_key = encode_key_list
    s1g_list = []
    batch_list = []
    sparsemode_list = []
    splitcode_list = []
    for i in range(len(encode_key)):
        s1g_list.append(encode_key[i][0])
        batch_list.append(encode_key[i][1])
        if encode_key[i][2] != "N":
            sparsemode_list.append(encode_key[i][2])
        splitcode_list.append(encode_key[i][3])
    ## s1g 情况
    s1g_coverage_rate = len(set(s1g_list))/s1g_class_num * 100
    
    ##batch覆盖情况
    batch_class_count = 0
    batch_result  = []
    tmp_list = []
    if "0" in batch_list:
        batch_class_count += 1
        batch_result.append("0")
    if "8" in batch_list:
        batch_class_count += 1
        batch_result.append("8")
    for batch_mode in batch_list:
        if batch_mode == "0" or batch_mode == "8":
            continue
        else:
            tmp_list.append(batch_mode)
    if tmp_list:
        res = coverage_encoding_batch(tmp_list)
        batch_class_count += res["coverage_count"]
        batch_result.append(res["binary"])
    batch_class_count_rate = batch_class_count/batch_class_num * 100
    ##sparemode情况
    sparsemode_coverage_rate = len(set(sparsemode_list))/sparsemode_class_num * 100

    ##split情况
    splitmode_coverage_rate = len(set(splitcode_list))/split_class_num * 100

    
    # 写入 CSV 文件
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["参数名称","可能取值整数","已覆盖取值数","覆盖率(%)","已有取值情况"])
        writer.writerow(["s1g",s1g_class_num,len(set(s1g_list)),s1g_coverage_rate,sorted(list(set(s1g_list)))])
        writer.writerow(["batch_actualseq",batch_class_num,batch_class_count,batch_class_count_rate,sorted(list(set(batch_result)))])
        writer.writerow(["sparemode",sparsemode_class_num,len(set(sparsemode_list)),sparsemode_coverage_rate,sorted(list(set(sparsemode_list)))])
        
        writer.writerow(["(mBaseSize,s2BaseSize)",split_class_num,len(set(splitcode_list)),splitmode_coverage_rate,sorted(list(set(splitcode_list)))])
        writer.writerow(["n2","uncertain",len(set(n2_list)),"uncertain",sorted(list(set(n2_list)))])
        writer.writerow(["g","uncertain",len(set(g_list)),"uncertain",sorted(list(set(g_list)))])
        

def process_json_folder(folder_path,encode_csv_path,coverage_csv_path):
    """
    处理文件夹下所有JSON文件，将结果追加写入CSV表格
    """
    # 获取文件夹下所有JSON文件
    json_files = list(Path(folder_path).glob("*.json"))
    
    if not json_files:
        print(f"文件夹 {folder_path} 中没有找到JSON文件")
        return False
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 按文件名排序处理
    json_files.sort()
    # 准备数据保存到CSV
    data = []

    # 处理每个JSON文件
    for i, json_file in enumerate(json_files):
        try:
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                case_name = os.path.basename(json_file).split(".")[0]
                encode_key,n2, g = encode_from_json(json_file)
                encode_key_list.append(encode_key)
                n2_list.append(int(n2))
                g_list.append(int(g))
                # 将提取的信息添加到数据列表中
                data.append({
                    'casename': case_name,
                    'encode_key': encode_key,
                    'n2': n2,
                    'g': g
                })
        except Exception as e:
            print(f"用例{json_file}出现error,报错如下:")
            traceback.print_exc()
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data)
    # csv_file = os.path.join(folder_path, 'encode_csv_path')
    df.to_csv(encode_csv_path, index=False)
    Calculate_coverage(coverage_csv_path)

def encode_from_excel(excel_file_path):
    pass

def encode_from_txt(txt_file_path,encode_csv_path,coverage_csv_path):
    df = parse_loop_data(txt_file_path)
    case_name = df["loop"].to_list()
    n2 = df["n2Size"].to_list()
    g = df["gSize"].to_list()
    s1Size = df["s1Size"].to_list()
    s2Size = df["s2Size"].to_list()
    sparse_mode = df["sparseMode"].to_list()
    isS1G = df["isS1G"].to_list()
    isAccumSeqS1 = df["isAccumSeqS1"].to_list()
    isAccumSeqS2 = df["isAccumSeqS2"].to_list()
    actualSeqS1Size = df["actualSeqS1Size"].to_list()
    actualSeqS2Size = df["actualSeqS2Size"].to_list()
    encode_key = ""
    s2splitcode = ""
    split_key = []
    global n2_list
    n2_list = n2[:100]
    global g_list
    g_list = g[:100]
    for index in range(len(df[:100])):
        actualseqlengths = actualSeqS1Size[index]
        actualseqlengthskv = actualSeqS2Size[index]
        if isAccumSeqS1[index] and actualseqlengths:
            actualseqlengths = [actualseqlengths[0]] + [actualseqlengths[i] - actualseqlengths[i-1] for i in range(1, len(actualseqlengths))] 
        if isAccumSeqS2[index] and actualseqlengthskv:
            actualseqlengthskv = [actualseqlengthskv[0]] + [actualseqlengthskv[i] - actualseqlengthskv[i-1] for i in range(1, len(actualseqlengthskv))]
        batch_code = encode_batch(actualseqlengths,actualseqlengthskv)
        sparse_mode_code = sparse_mode[index]
        S1g_code = isS1G[index]
        for i in mBaseSize_nums:
            for j in s2BaesSize_nums:
                s1splitcode = "0" if int(s1Size[index]) % i == 0 else "1"
                s2splitcode = "0" if int(s2Size[index]) % j == 0 else "1"
                split_key.append((i,j,s1splitcode,s2splitcode))
        encode_key = str(S1g_code) + str(batch_code) + str(sparse_mode_code) 
        encode_key_list.append(encode_key)
        split_key_list.append(split_key)
    # print(encode_key_list)
    # df = pd.DataFrame({"case_name":case_name[:100],"encode_key":encode_key_list,"split_key":split_key_list,"n2":n2[:100],"g":g[:100]})
    df.to_csv(encode_csv_path,index=False)
    Calculate_coverage(coverage_csv_path)
input_json_path = "//home/z00949295/LoadBalanceWhiteBox/fia_pfa_ifa_noquant_json"
encode_csv_path = "./encode_fia_pfa_ifa_noquant_v2.csv"
cover_csv_path = "./coverage_fia_pfa_ifa_noquant_v2.csv"
process_json_folder(input_json_path,encode_csv_path,cover_csv_path)


# input_json_path = "./cases.txt"
# encode_csv_path = "./encode_case.csv"
# cover_csv_path = "./cover_case.csv"
# encode_from_txt(input_json_path,encode_csv_path,cover_csv_path)