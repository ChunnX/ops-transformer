# -*- coding: utf-8 -*-
"""
-------------------------------------------------
File Name： excel_trans_to_aclnn_SFA.py
Description :
-------------------------------------------------
"""
import argparse

import openpyxl
import pandas as pd
import numpy as np
import ctypes

is_tensor = 1


def read_excel(path, sheet_name):
    return pd.read_excel(path, sheet_name, dtype=str).to_dict(orient='records')


def add_bracket(data):
    if data == '' or str(data) == 'nan':
        return []
    return '[' + str(data) + ']'

def lse_shape_add_bracket(data):
    if data == '' or str(data) == 'nan':
        return '[0]'
    return '[' + str(data) + ']'

def rope_add_bracket(data):
    if data == '' or str(data) == 'nan':
        return '[1]'
    return '[' + str(data) + ']'


def count_vline(data):
    if '|' in data:
        return data.count('|') + 1
    else:
        return 1


def replace_vline(data):
    if str(data) == 'nan' or data == '':
        return '[[1]]'
    else:
        arr = data.split("|")
        new_data = []
        for a in arr:
            new_data.append('[' + str(a) + ']')
        return '[' + ','.join(new_data) + ']'


def replace_mask_none_value(data):
    if data == '' or str(data) == 'nan':
        new_data = '[1]'
    else:
        new_data = '[' + str(data) + ']'
    return new_data


def concat_datarange_by_num(datarange, num):
    _t = []
    lst = f"[{datarange}]"
    for i in range(int(num)):
        _t.append(lst)
    return '[' + ','.join(_t) + ']'


def quant_int_data_range_handle(data):
    if not data and len(str(data)) == 0:
        if data == 0:
            return [int(data), int(data)]
        return "[" + "'null'" + "," + "'null'" + "]"
    elif str(data) == 'nan':
        return "[" + "'null'" + "," + "'null'" + "]"
    else:
        new_data = int(data)
        return "[" + str(new_data) + "," + str(new_data) + "]"


def conv_float_to_u32(data_f):
    fp = ctypes.pointer(ctypes.c_float(data_f))
    cp = ctypes.cast(fp, ctypes.POINTER(ctypes.c_uint))
    data_hex = cp.contents.value
    result = (data_hex // 8192) * 8192
    return result


def trans_19bit(deqscale):
    deqscale = np.array([deqscale])
    res_19bit = np.zeros(deqscale.shape[0], dtype=np.uint64)

    for idx, scale in enumerate(deqscale):
        val = np.bitwise_and(((conv_float_to_u32(scale) >> 13) << 13), 0xffffffff)  # [31:13], M
        val = val.astype(np.uint64)
        res_19bit[idx] = val

    return res_19bit[0]


def quant_data_range_handle_PFA(data):
    if data:
        import re
        if isinstance(data, str):
            data = re.sub(r"[\u202c]", "", data)
            data = float(data)
        else:
            data = float(data)

        new_data = trans_19bit(data)
        return "[" + str(new_data) + "," + str(new_data) + "]"
    if not data or isinstance(data, str):
        if data == 0:
            return "[" + str(data) + "," + str(data) + "]"
        # return [r"null", r"null"]
        return "[" + "'null'" + "," + "'null'" + "]"


def quant_float_data_range_handle(data):
    if not data or isinstance(data, str) or str(data) == 'nan':
        if data == 0:
            return "[" + str(data) + "," + str(data) + "]"
        return "[" + "'null'" + "," + "'null'" + "]"
    else:
        new_data = float(data)
        return "[" + str(new_data) + "," + str(new_data) + "]"


def antiquant_add_bracket(data):
    if data == '' or str(data) == 'nan':
        return "[" + "'null'" + "," + "'null'" + "]"
    return '[' + str(data) + ']'


def antiquant_add_bracket_check(shape, data):
    if not shape or str(shape) == 'nan':
        if shape == 0:
            return '[' + str(data) + ']'
        return "[" + "'null'" + "," + "'null'" + "]"
    else:
        return '[' + str(data) + ']'


def add_single_quotation(data):
    return '\'' + data.lower() + '\''


def add_dobule_quotation(data):
    return '\"' + data + '\"'


def concat_dtype_by_num(dtype, num):
    _t = []
    for i in range(int(num)):
        _t.append("'" + dtype.lower() + "'")
    return '[' + ','.join(_t) + ']'


def set_default_dtype_value(data):
    if data == '' or str(data) == 'nan':
        new_data = "'fp16'"
    else:
        new_data = '\'' + data.lower() + '\''
    return new_data


def concat_format_by_num(_str, num):
    _t = []
    for i in range(int(num)):
        _t.append("'" + _str + "'")
    return '[' + ",".join(_t) + ']'


def set_default_antiquant_mode_value(data):
    if data == '' or str(data) == 'nan':
        new_data = '0'
    else:
        new_data = int(data)
    return new_data


# def set_softmax_lse_flag_value(data):
#     return str(data).lower().capitalize()


def set_default_attr_value(data):
    if data == '':
        new_data = 0
    else:
        new_data = str(data)
    return new_data

def set_softmax_lse_flag_value(data):
    if data == '':
        new_data = str('False')
    else:
        softmax_lse_flag = str(data).lower()
        if softmax_lse_flag == 'false':
            new_data = str('False')
        else:
            new_data = str('True')
    return new_data


def set_fusedfalg_default_attr_value(data):
    if data == '':
        new_data = "yes"
    else:
        new_data = '\'' + str(data) + '\''
    return new_data


def set_default_string_attr_value(data):
    if data == '':
        new_data = '\'BSH\''
    else:
        new_data = '\'' + data + '\''
    return new_data


def set_default_attr_mramdomtype_value(data):
    if data == '':
        new_data = "Normal"
    else:
        new_data = data
    return new_data


def set_default_enable_gpu_quotation(data):
    if data == '':
        new_data = '\'True\''
    else:
        data = str(data)
        new_data = '\'' + data + '\''
    return new_data


def set_default_softmaxLseFlag_value(data):
    if data == '':
        new_data = str('False')
    else:
        softmax_lse_flag = str(data).lower()
        if softmax_lse_flag == 'false':
            new_data = str('False')
        else:
            new_data = str('True')
    return new_data


def set_pse_datarange(shape, range):
    if shape == '[1]':
        return "[" + "'null'" + "," + "'null'" + "]"
    else:
        return '[' + str(range) + ']'


def set_mask_datarange(shape):
    if shape == '[1]':
        return "[" + "'null'" + "," + "'null'" + "]"
    else:
        return '[0,2]'


def set_bt_datarange(shape):
    if shape == '[1]':
        return "[" + "'null'" + "," + "'null'" + "]"
    else:
        return '[0,1]'


def is_param_exist(data):
    if not data or str(data) == 'nan':
        if data == 0:
            return '1'
        return '0'
    else:
        return '1'


def cache_pre_datarange(shape, datarange):
    if not shape or str(shape) == 'nan':
        if shape == 0:
            return pre_datarange(datarange)
        return "'null','null'"
    else:
        return pre_datarange(datarange)


def pre_datarange(datarange):
    if ',' in datarange:
        if "nan" in datarange or "inf" in datarange:
            temp = datarange.split(',')
            try:
                int(temp[0])
            except:
                temp[0] = f"'{temp[0]}'"
            try:
                int(temp[1])
            except:
                temp[1] = f"'{temp[1]}'"
            lst = f"{temp[0]},{temp[1]}"
        else:
            lst = f"{datarange}"
    else:
        lst = f"'{datarange}'"
    return lst


def trans_dict_to_list(data, op_name):
    print('data: ', data)

    res = [['aclnn_name', 'case_name', 'input_tensor_shape', 'input_tensor_range', 'input_tensor_dtype',
            'input_tensor_format', 'input_tensor_type', 'input_tensor_index', 'output_tensor_shape',
            'output_tensor_range', 'output_tensor_dtype', 'output_tensor_format', 'output_tensor_type',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value',
            'attr_name', 'attr_type', 'attr_dtype', 'attr_value'
            ]]
    for d in data:
        temp = [op_name]
        temp.append(d["case_name"])
        temp.append(d["input_tensor_shape"])
        temp.append(d["input_tensor_range"])
        temp.append(d["input_tensor_dtype"])
        temp.append(d["input_tensor_format"])
        temp.append(d["input_tensor_type"])
        temp.append(d["input_tensor_index"])
        temp.append(d["output_tensor_shape"])
        temp.append(d["output_tensor_range"])
        temp.append(d["output_tensor_dtype"])
        temp.append(d["output_tensor_format"])
        temp.append(d["output_tensor_type"])

        attr = ["actualseqlengths", "actualseqlengthskv", "scalevalue", "layout_query", "layout_kv",
                "sparsemode", "sparse_blocksize", "return_softmax", "flaglist"]

        for a in attr:
            temp.append(a)
            actualseqlengths = d[a]
            for i in range(3):
                temp.append(actualseqlengths[i])

        res.append(temp)
    return res


def write_xlsx_excel(sheet, write_data):
    """
    :param sheet:工作表对象
    :param write_data:将要写入表格的数据（二维列表）
    """
    # 遍历行
    for i in range(0, len(write_data)):
        # 遍历列
        for j in range(0, len(write_data[i])):
            # 写入数据
            sheet.cell(row=i + 1, column=j + 1, value=str(write_data[i][j]))


def trans_to_aclnn(op_name, data):
    for key, value in data.items():
        if str(value) == "nan":  # 判断value是否为nan
            data[key] = ''
    res = {"case_name": data["Testcase_Name"]}
    # input_tensor_shape
    q_shape = add_bracket(data["q_shape"])  # q_shape
    k_shape_num = count_vline(data["k_shape"])  # k_shape
    v_shape_num = count_vline(data["v_shape"])  # v_shape
    k_shape = add_bracket(data["k_shape"])
    v_shape = add_bracket(data["v_shape"])

    k_cache_shape = rope_add_bracket(data["k_cache_shape"])  # k_cache_shape
    v_cache_shape = rope_add_bracket(data["v_cache_shape"])  # v_cache_shape

    bt_shape = replace_mask_none_value(data["blockTable_shape"])  # blockTable_shape
    sparse_indices_shape = replace_mask_none_value(data["sparse_indices_shape"])  # sparse_indices_shape
    print("======================sparse_indices_shape===============", sparse_indices_shape)

    q_rope_shape = '[1]'
    k_rope_shape = '[1]'
    k_rope_cache_shape = '[1]'
    if "q_rope_shape" in data.keys():
        q_rope_shape = rope_add_bracket(data["q_rope_shape"])  # q_rope_shape
        k_rope_shape = rope_add_bracket(data["k_rope_shape"])  # k_rope_shape
        k_rope_cache_shape = rope_add_bracket(data["k_rope_cache_shape"])  # k_rope_cache_shape
    input_tensor_shape = add_bracket(','.join([q_shape,
                                               k_shape,
                                               v_shape,
                                               sparse_indices_shape,
                                               bt_shape,
                                               k_cache_shape,
                                               v_cache_shape,
                                               q_rope_shape,
                                               k_rope_shape,
                                               k_rope_cache_shape,
                                               ]))
    print(input_tensor_shape)
    res["input_tensor_shape"] = input_tensor_shape

    # input_tensor_range
    q_datarange = add_bracket(pre_datarange(data["q_datarange"]))  # q_datarange
    if is_tensor:
        k_cache_datarange = add_bracket(cache_pre_datarange(data["k_cache_shape"], data["k_datarange"]))
        v_cache_datarange = add_bracket(cache_pre_datarange(data["v_cache_shape"], data["k_datarange"]))
        k_datarange = add_bracket(pre_datarange(data["k_datarange"]))
        v_datarange = add_bracket(pre_datarange(data["v_datarange"]))
    else:
        k_cache_datarange = concat_datarange_by_num(cache_pre_datarange(data["k_cache_shape"], data["k_datarange"]),1)  # k_cache_shape
        v_cache_datarange = concat_datarange_by_num(cache_pre_datarange(data["v_cache_shape"], data["k_datarange"]),1)  # v_cache_shape
        k_datarange = concat_datarange_by_num(pre_datarange(data["k_datarange"]), k_shape_num)  # k_datarange
        v_datarange = concat_datarange_by_num(pre_datarange(data["v_datarange"]), v_shape_num)  # v_datarange

    sparse_indices_datarange = "[" + "'null'" + "," + "'null'" + "]"
    q_rope_datarange = "[" + "'null'" + "," + "'null'" + "]"  # q_rope_datarange
    k_rope_datarange = "[" + "'null'" + "," + "'null'" + "]"  # k_rope_datarange
    k_rope_cache_datarange = "[" + "'null'" + "," + "'null'" + "]"  #

    if "sparse_indices_shape" in data.keys():
        sparse_indices_datarange = '[-1,0]'  #
    bt_datarange = set_bt_datarange(bt_shape)  # bt_shape
    if "q_rope_shape" in data.keys():
        q_rope_datarange = antiquant_add_bracket_check(data["q_rope_shape"], data["q_rope_datarange"])  # q_rope_shape
        k_rope_datarange = antiquant_add_bracket_check(data["k_rope_shape"], data["k_rope_datarange"])  # k_rope_shape
        k_rope_cache_datarange = antiquant_add_bracket_check(data["k_rope_cache_shape"], data["k_rope_datarange"])
    input_tensor_range = add_bracket(','.join([q_datarange, k_datarange, v_datarange, sparse_indices_datarange,
                                               bt_datarange, k_cache_datarange, v_cache_datarange,
                                               q_rope_datarange, k_rope_datarange, k_rope_cache_datarange]))
    res["input_tensor_range"] = input_tensor_range
    # input_tensor_dtype
    # out_dtype q_dtype q_rope_dtype sparse_indices_dtype k_dtype k_rope_dtype v_dtype bt_dtype
    q_dtype = add_single_quotation(data["q_dtype"])
    if is_tensor:
        k_cache_dtype = add_single_quotation(data["k_dtype"])
        v_cache_dtype = add_single_quotation(data["v_dtype"])
        k_dtype = add_single_quotation(data["k_dtype"])
        v_dtype = add_single_quotation(data["v_dtype"])
    else:
        k_cache_dtype = concat_dtype_by_num(data["k_dtype"], 1)
        v_cache_dtype = concat_dtype_by_num(data["v_dtype"], 1)
        k_dtype = concat_dtype_by_num(data["k_dtype"], k_shape_num)
        v_dtype = concat_dtype_by_num(data["v_dtype"], v_shape_num)
    sparse_indices_dtype = "'int32'"
    bt_dtype = "'int32'"
    q_rope_dtype = "'fp16'"
    k_rope_dtype = "'fp16'"
    if "q_rope_shape" in data.keys():
        q_rope_dtype = set_default_dtype_value(data["q_rope_dtype"])
        k_rope_dtype = set_default_dtype_value(data["k_rope_dtype"])

    input_tensor_dtype = add_bracket(','.join([q_dtype, k_dtype, v_dtype, sparse_indices_dtype,
                                               bt_dtype, k_cache_dtype, v_cache_dtype,
                                               q_rope_dtype, k_rope_dtype, k_rope_dtype]))
    res["input_tensor_dtype"] = input_tensor_dtype

    # input_tensor_format

    q_format = "'ND'"
    if is_tensor:
        k_format = "'ND'"
        v_format = "'ND'"
        k_cache_format = "'ND'"
        v_cache_format = "'ND'"
    else:
        k_format = concat_format_by_num('ND', k_shape_num)
        v_format = concat_format_by_num('ND', v_shape_num)
        k_cache_format = concat_format_by_num('ND', 1)
        v_cache_format = concat_format_by_num('ND', 1)
    # V3新增 antiquantScale/antiquantOffset/blocktable

    bt_format = "'ND'"
    sparse_indices_format = "'ND'"
    q_rope_format = "'ND'"
    k_rope_format = "'ND'"
    input_tensor_format = add_bracket(','.join([q_format, k_format, v_format, sparse_indices_format,
                                                bt_format, k_cache_format, v_cache_format,
                                                q_rope_format, k_rope_format, k_rope_format]))

    res["input_tensor_format"] = input_tensor_format

    input_tensor_type = (r"['tensor','tensor','tensor','tensor','tensor','tensor','tensor','tensor','tensor','tensor']")

    res["input_tensor_type"] = input_tensor_type
    # input_tensor_type < --- > 对应上 input_tensor_index
    input_tensor_index = r"[0,1,2,3,4,5,6,7,8,9]"
    res["input_tensor_index"] = input_tensor_index
    # output
    # output_tensor_shape
    out_shape = add_bracket(data["out_shape"])
    softmax_max_shape = lse_shape_add_bracket(data["softmax_max_shape"])
    softmax_sum_shape = lse_shape_add_bracket(data["softmax_sum_shape"])
    softmax_lse_flag = str(set_default_softmaxLseFlag_value(data["return_softmax"])).lower()
    if softmax_lse_flag=="true":
        output_tensor_shape = add_bracket(','.join([out_shape, softmax_max_shape,
                                                    softmax_sum_shape]))
        # output_tensor_range
        output_tensor_range = add_bracket(add_bracket(data["q_datarange"]) + ',' + '[-1,1]' + ',' + '[-1,1]')
        # output_tensor_dtype
        output_tensor_dtype = add_bracket(add_single_quotation(data["out_dtype"]) + ',' + add_single_quotation(
                                data["softmax_max_dtype"])+ ',' + add_single_quotation(data["softmax_sum_dtype"]))
        # output_tensor_format
        output_tensor_format = add_bracket("'ND','ND','ND'")
    else:
        output_tensor_shape = add_bracket(out_shape)
        # output_tensor_range
        output_tensor_range = add_bracket(add_bracket(data["q_datarange"]) )
        # output_tensor_dtype
        output_tensor_dtype = add_bracket(add_single_quotation(data["out_dtype"]) )
        # output_tensor_format
        output_tensor_format = add_bracket("'ND','ND','ND'")

    res["output_tensor_shape"] = output_tensor_shape
    res["output_tensor_range"] = output_tensor_range
    res["output_tensor_dtype"] = output_tensor_dtype
    res["output_tensor_format"] = output_tensor_format
    # output_tensor_type
    res["output_tensor_type"] = ""

    # import pdb; pdb.set_trace()
    actualseqlengths = ['int_array', '', add_bracket(data["actual_seq_lengths_query"])]
    res["actualseqlengths"] = actualseqlengths

    # actualseqlengthskv[attr_type, attr_dtype, attr_value]
    actualseqlengthskv = ['int_array', '', add_bracket(data["actual_seq_lengths_kv"])]
    res["actualseqlengthskv"] = actualseqlengthskv

    # scalevalue
    scalevalue = ['buildins', 'double', set_default_attr_value(data["scaleValue"])]
    res["scalevalue"] = scalevalue

    # layout_query
    layout_query = ['buildins', 'String', set_default_string_attr_value(data["layout_query"])]
    res["layout_query"] = layout_query

    # layout_kv
    layout_kv = ['buildins', 'String', set_default_string_attr_value(data["layout_kv"])]
    res["layout_kv"] = layout_kv

    # sparsemode
    sparsemode = ['buildins', 'int64_t', set_default_attr_value(data["sparse_mode"])]
    res["sparsemode"] = sparsemode

    # sparse_blocksize
    sparse_blocksize = ['buildins', 'int64_t', set_default_attr_value(data["sparse_blockSize"])]
    res["sparse_blocksize"] = sparse_blocksize

    # return_softmax
    return_softmax = ['buildins', 'bool', set_softmax_lse_flag_value(data["return_softmax"])]
    res["return_softmax"] = return_softmax

    # flaglist
    q_flag = is_param_exist(data["q_shape"])
    k_flag = is_param_exist(data["k_shape"])
    v_flag = is_param_exist(data["v_shape"])
    sparse_indices_flag = is_param_exist(data["sparse_indices_shape"])
    ac_flag = is_param_exist(data["actual_seq_lengths_query"])
    ackv_flag = is_param_exist(data["actual_seq_lengths_kv"])

    out_flag = is_param_exist(data["out_shape"])
    softmax_max_flag = is_param_exist(data["softmax_max_shape"])
    softmax_sum_flag = is_param_exist(data["softmax_sum_shape"])

    bt_flag = is_param_exist(data["blockTable_shape"])

    q_rope_flag = k_rope_flag = 0
    if "q_rope_shape" in data.keys():
        q_rope_flag = is_param_exist(data["q_rope_shape"])
        k_rope_flag = is_param_exist(data["k_rope_shape"])

    flag_list = ['int_array', '', add_bracket(','.join([q_flag,
                                                        k_flag,
                                                        v_flag,
                                                        sparse_indices_flag,
                                                        ac_flag,
                                                        ackv_flag,
                                                        bt_flag,
                                                        out_flag,
                                                        q_rope_flag,
                                                        k_rope_flag,
                                                        softmax_max_flag,
                                                        softmax_sum_flag,
                                                        ]))]

    res["flaglist"] = flag_list
    print('res: ')
    for key, value in res.items():
        print(key, value)
    return res


# python3 excel_trans_to_aclnn_new.py --case_file fuzz_test.xlsx --op_name aclnnFusedInferAttentionScoreV2
def parse_args():
    global sheet_name
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_file', '-f', type=str, help='原始文件名称，支持路径/文件')
    parser.add_argument('--output_path', '-o', type=str, help='转换后的文件路径')
    parser.add_argument('--sheet_name', '-s', type=str, help='指定sheet')
    parser.add_argument('--op_name', '-n', type=str, help='算子名 ,当前支持aclnnSparseFlashAttention')
    args = parser.parse_args()
    if not args.case_file:
        raise Exception("请输入用例文件，使用--case_file 参数传入")
    if not args.sheet_name:
        args.sheet_name = 'all'
    # 1、加载旧表格
    # 获取所有sheet页，判断传入sheet_name是否存在
    df_sheet = pd.read_excel(args.case_file, sheet_name=None, keep_default_na=False)
    sheet_name_list = []
    if args.sheet_name != 'all' and args.sheet_name in df_sheet.keys():
        sheet_name_list.append(args.sheet_name)
    elif args.sheet_name == 'all':
        sheet_name_list = list(df_sheet.keys())
    else:
        raise Exception("请输入存在的sheet页，使用--sheet_name 参数传入")
    if args.op_name in ["aclnnSparseFlashAttention"]:
        op_name = args.op_name
    else:
        raise Exception(
            "请正确输入算子名称!当前仅支持aclnnSparseFlashAttention")
    # 结果文件路径
    if args.output_path:
        output_name = args.output_path + f'design2_aclnn_{op_name}.xlsx'
    else:
        output_name = f'design2_aclnn_{op_name}.xlsx'

    # 创建工作簿对象
    workbook = openpyxl.Workbook()
    # 2、循环读取
    for sheet_name in sheet_name_list:
        # 设置工作表名称
        sheet = workbook.create_sheet(sheet_name)
        result = []
        df = read_excel(args.case_file, sheet_name)
        print('df:', df)
        for line in df:
            # 用fused_flag做个判断，保留老分支
            result.append(trans_to_aclnn(op_name, line))
        print("*****************new********")
        write_xlsx_excel(sheet, trans_dict_to_list(result, op_name), )

    workbook.remove(workbook["Sheet"])
    workbook.save(output_name)


if __name__ == '__main__':
    parse_args()
