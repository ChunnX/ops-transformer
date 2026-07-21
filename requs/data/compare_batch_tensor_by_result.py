import argparse
import datetime
import os
import sys

import numpy as np
import openpyxl
import pandas as pd

def log(text):
    print(text)
    log_file = open('auto_run.txt', 'a')
    log_file.write(str(datetime.datetime.now()) + '  ')
    log_file.write(text)
    log_file.write("\n")
    log_file.close()

def print_log(data=None):
    """
    print log
    :param data: value to print
    :param level: log level, include DEBUG, INFO(default), WARN and ERROR
    :return:
    """
    level = "INFO"
    log_info = ("%s - [%s]-%s:%s - %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], level,
                                          os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                          str(sys._getframe().f_back.f_lineno).zfill(4), data))
    log(log_info)

def read_csv(file):
    csv_info = {"titles": [], "datas": []}
    if os.path.exists(file):
        try:
            result_df = pd.read_csv(file, header=0)
            for index, row in result_df.iterrows():
                csv_info["titles"] = list(row.keys())
                data = {}
                for key in row.keys():
                    data[key] = row[key]
                csv_info["datas"].append(data)
        except Exception as e:
            print(f"[ERROR] read_csv failed: {e}")
    return csv_info

def are_all_ndarrays_close(array_list):
    if not array_list:
        return True
    if len(array_list) == 1:
        return True

    base_array = array_list[0]
    for array in array_list[1:]:
        # 检查形状、类型和值是否在容差范围内一致
        if not np.array_equal(base_array, array):
            return False
    return True

def write_csv(file,head,line):
    if not os.path.exists(file):
        write_head = True
    else:
        write_head = False
    with open(file, "a+") as f:
        if write_head:
            f.write(f"{head}\n")
        f.write(f"{line}\n")

def get_numpy_dtype(str):
    if str.lower() in ["float32", "fp32","uint32","int32"]:
        return np.uint32
    elif str.lower() in ["bfloat16","bf16","fp16","float16","uint16","int16"]:
        return np.uint16
    elif str.lower() in ["uint8","int8"]:
        return np.uint8
    elif str.lower() in ["complex128"]:
        return np.complex128
    elif str.lower() in ["complex64","int64","uint64","fp64","float64"]:
        return np.uint64
    elif str.lower() in ["bool"]:
        return np.bool
    else:
        raise ValueError(f"[ERROR] dtype:{str} is invalid.")

def read_excel(file):
    excel_info = {}
    if os.path.exists(file):
        wb = openpyxl.load_workbook(file)
        for sheet in wb.sheetnames:
            idx = 0
            if sheet not in excel_info.keys():
                excel_info[sheet] = {"titles":[],"datas":[],"ids":idx}
                idx += 1
            ws = wb[sheet]
            nrows = ws.max_row
            column_indexs = {}
            title_ids = {}
            for i in range(1, ws.max_column + 1):
                title = ws.cell(row=1, column=i).value
                if title is None:
                    title = "UNK"
                if title not in title_ids.keys():
                    title_ids[title] = 0
                else:
                    title_ids[title] += 1
                if title_ids[title] > 0:
                    new_title = title+"."+str(title_ids[title])
                else:
                    new_title = title
                excel_info[sheet]["titles"].append(new_title)
                column_indexs[new_title] = i
            #print(f"get case column indexs:{column_indexs}")
            for i in range(2, nrows + 1):
                colum_info = {}
                for key in column_indexs.keys():
                    cell_value = ws.cell(i, column_indexs[key]).value
                    colum_info[key] = cell_value
                excel_info[sheet]["datas"].append(colum_info)
    return excel_info

def read_aclnn_excel(file):
    excel_info = {}
    if os.path.exists(file):
        wb = openpyxl.load_workbook(file)
        for sheet in wb.sheetnames:
            idx = 0
            if sheet not in excel_info.keys():
                excel_info[sheet] = {"titles":[],"datas":[],"ids":idx}
                idx += 1
            ws = wb[sheet]
            nrows = ws.max_row
            column_indexs = {}
            title_ids = {}
            for i in range(1, ws.max_column + 1):
                title = ws.cell(row=1, column=i).value
                if title is None:
                    title = "UNK"
                if title not in title_ids.keys():
                    title_ids[title] = 0
                else:
                    title_ids[title] += 1
                if title_ids[title] > 0:
                    new_title = title+"."+str(title_ids[title])
                else:
                    new_title = title
                excel_info[sheet]["titles"].append(new_title)
                column_indexs[new_title] = i
            #print(f"get case column indexs:{column_indexs}")
            for i in range(2, nrows + 1):
                colum_info = {}
                for key in column_indexs.keys():
                    cell_value = ws.cell(i, column_indexs[key]).value
                    if "attr_type" in  key or "attr_dtype" in key or "attr_value" in key:
                        continue
                    if "attr_name" in key:
                        attr_name = ws.cell(i, column_indexs[key]).value
                        key_type = key.replace("attr_name","attr_type")
                        attr_type = ws.cell(i, column_indexs[key_type]).value
                        key_dtype = key.replace("attr_name", "attr_dtype")
                        attr_dtype = ws.cell(i, column_indexs[key_dtype]).value
                        key_value = key.replace("attr_name", "attr_value")
                        attr_value = ws.cell(i, column_indexs[key_value]).value
                        colum_info[attr_name] = {"type": attr_type, "dtype": attr_dtype, "value": attr_value}
                    else:
                        colum_info[key] = cell_value
                excel_info[sheet]["datas"].append(colum_info)
    return excel_info

def create_input_info(result_csv):
    if not os.path.exists(result_csv):
        print_log(f"[ERROR] {result_csv} does not exist!")
    result_path = str(os.path.dirname(os.path.abspath(result_csv)))
    result_name = str(os.path.basename(result_csv))
    input_result_file = result_name.replace(f"_result","_input_info")
    result_input_info_file = os.path.join(result_path, input_result_file)
    if os.path.exists(result_input_info_file):
        os.remove(result_input_info_file)
    reslut_info = read_csv(result_csv)
    result_head = "op_name,case_name,input_idx,input_shape,dtype,range,slice,batch_shape,seed"
    for reslut in reslut_info["datas"]:
        op_name = reslut["op_name"]
        case_name = reslut["case_name"]
        result = reslut["result"]
        if result not in ["Pass"]:
            continue
        attr_json = reslut["attrs"]
        attrs = eval(attr_json)
        input_shapes = eval(reslut["input_shape"])
        input_dtypes = eval(reslut["input_dtype"])
        if "tensor_input_seed_info" not in attrs.keys():
            continue
        tensor_input_seed_info = attrs["tensor_input_seed_info"]
        print_log(f"tensor_input_seed_info:{tensor_input_seed_info} len:{len(tensor_input_seed_info)}")
        for input_seed_info in tensor_input_seed_info:
            if len(input_seed_info) == 5:
                tensor_index = input_seed_info[0]
                batch_axis = input_seed_info[1]
                batch_index_list = input_seed_info[2]
                batch_range = input_seed_info[3]
                batch_seed = input_seed_info[4]
                pre_shape_index = input_seed_info[5]
                for index in range(len(input_shapes)):
                    shape_input = input_shapes[index]
                    dtype_input = input_dtypes[index]
                    if tensor_index == index:
                        slice_str = "["
                        print_log(f"tensor_index:{tensor_index},batch_axis:{batch_axis},batch_index_list:{batch_index_list},batch_range:{batch_range},batch_seed:{batch_seed},pre_shape_index:{pre_shape_index}")
                        batch_seed_shape = []
                        for i in range(len(shape_input)):
                            if i > batch_axis:
                                batch_seed_shape.append(shape_input[i])
                                slice_str += ":,"
                            elif i == batch_axis:
                                batch_index_str = "["
                                for batch_index in batch_index_list:
                                    if batch_index >= shape_input[i]:
                                        print_log(f"[ERROR] batch_index {batch_index} in batch_index_list:{batch_index_list} for shape_input:{shape_input} batch_axis:{batch_axis} is invalid.")
                                    else:
                                        batch_index_str += str(batch_index) + ","
                                batch_index_str = batch_index_str[:-1]
                                batch_index_str += "]"
                                slice_str += str(batch_index_str) + ","
                            else:
                                if pre_shape_index[i] != ":" and pre_shape_index[i] >= shape_input[i]:
                                    print_log(f"[ERROR] pre_shape_index {pre_shape_index} for shape_input:{shape_input} index:{i} is invalid.")
                                    continue
                                slice_str += str(pre_shape_index[i]) + ","
                        slice_str = slice_str[:-1]
                        slice_str += "]"
                        print_log(f"{shape_input} {slice_str} = {str(batch_seed_shape)}")
                        write_csv(result_input_info_file,result_head,f"{op_name},{case_name},{tensor_index},\"{str(input_shapes)}\",{dtype_input},\"{str(batch_range)}\",\"{str(slice_str)}\",\"{str(batch_seed_shape)}\",{batch_seed}")

def check_batch_bin_by_result(result_csv,bin_path):
    if not os.path.exists(result_csv):
        print_log(f"[ERROR] {result_csv} does not exist!")
        return False
    result_path = str(os.path.dirname(os.path.abspath(result_csv)))
    result_name = str(os.path.basename(result_csv))
    compare_result_name = result_name.replace(f"_result","_compare_info")
    compare_result_file = os.path.join(result_path, compare_result_name)
    if os.path.exists(compare_result_file):
        os.remove(compare_result_file)
    reslut_info = read_csv(result_csv)
    compare_info = {}
    result_head = "check_seed,case_name,tensor_index,batch_axis,dtype,shape,slice,result"
    for reslut in reslut_info["datas"]:
        op_name = reslut["op_name"]
        case_name = reslut["case_name"]
        result = reslut["result"]
        if result not in ["Pass"]:
            continue
        attr_json = reslut["attrs"]
        attrs = eval(attr_json)
        output_shapes = eval(reslut["output_shape"])
        output_dtypes = eval(reslut["output_dtype"])
        if "tensor_output_seed_info" not in attrs.keys():
            continue
        tensor_output_seed_info = attrs["tensor_output_seed_info"]
        print_log(f"tensor_output_seed_info:{tensor_output_seed_info} len:{len(tensor_output_seed_info)}")
        for idx in range(len(tensor_output_seed_info)):
            tensor_index = tensor_output_seed_info[idx][0]
            batch_axis = tensor_output_seed_info[idx][1]
            batch_index_list = tensor_output_seed_info[idx][2]
            batch_seed = tensor_output_seed_info[idx][3]
            pre_shape_index = tensor_output_seed_info[idx][4]
            bin_file = os.path.join(bin_path, op_name,f"{case_name}_output_{tensor_index}.bin")
            if tensor_index >= len(output_shapes):
                print_log(f"[ERROR] check_batch_bin_by_result tensor_index:{tensor_index} is invalid.")
                continue
            check_tensor_shape = output_shapes[tensor_index]
            check_tensor_dtype = output_dtypes[tensor_index]
            pre_slice_str = "["
            for i in range(batch_axis):
                pre_shape = pre_shape_index[i]
                if pre_shape != ":" and check_tensor_shape[i] <= pre_shape:
                    print_log(f"[ERROR] tensor_index:{tensor_index} pre_shape:{pre_shape} is invalid.")
                    continue
                pre_slice_str += str(pre_shape) + ","
            for batch_index in batch_index_list:
                if batch_index >= check_tensor_shape[batch_axis]:
                    print_log(f"[ERROR] batch_index:{batch_index} is invalid.")
                    continue
                batch_slice_str = pre_slice_str + str(batch_index) + ",...]"
                if batch_seed not in compare_info.keys():
                    compare_info[batch_seed] = {}
                if tensor_index not in compare_info[batch_seed].keys():
                    compare_info[batch_seed][tensor_index] = {}
                if check_tensor_dtype not in compare_info[batch_seed][tensor_index].keys():
                    compare_info[batch_seed][tensor_index][check_tensor_dtype] = []
                compare_info[batch_seed][tensor_index][check_tensor_dtype].append({"file":bin_file,"slice":batch_slice_str,"shape":check_tensor_shape,"case_name":case_name,"batch_axis":batch_axis})
                print_log(f"case_name:{case_name},tensor_shape:{check_tensor_shape},tensor_dtype:{check_tensor_dtype},tensor_index:{tensor_index},batch_axis:{batch_axis},batch_index:{batch_index},batch_seed:{batch_seed},pre_shape_index:{pre_shape_index},slice:{batch_slice_str}")
    for batch_seed in compare_info.keys():
        for tensor_index in compare_info[batch_seed].keys():
            for tensor_dtype in compare_info[batch_seed][tensor_index].keys():
                base_tensor = None
                case_names = []
                for tensor_data in compare_info[batch_seed][tensor_index][tensor_dtype]:
                    file = tensor_data["file"]
                    slice = tensor_data["slice"]
                    shape = tensor_data["shape"]
                    batch_axis = tensor_data["batch_axis"]
                    case_name = tensor_data["case_name"]
                    case_names.append(case_name)
                    print_log(f"case_name:{case_name},shape:{shape},slice:{slice}")
                    if os.path.exists(file):
                        numpy_tensor = np.fromfile(file,dtype=get_numpy_dtype(tensor_dtype))
                        numpy_tensor = numpy_tensor.reshape(shape)
                        if base_tensor is None:
                            base_tensor = eval(f"numpy_tensor{slice}")
                            print_log(f"case_name:{case_name},shape:{shape},slice:{slice} is base tensor")
                            status = "base"
                        else:
                            check_tensor = eval(f"numpy_tensor{slice}")
                            if not np.array_equal(base_tensor, check_tensor):
                                print_log(f"case_name:{case_name},shape:{shape},slice:{slice} is not the same as base_tensor")
                                status = "diff"
                            else:
                                status = "same"
                    else:
                        status = "nofile"
                    write_csv(compare_result_file,result_head,f"{batch_seed},{case_name},{tensor_index},{batch_axis},{tensor_dtype},\"{str(shape)}\",\"{str(slice)}\",{status}")

def vailed_input_seed_info_attr(case_xlsx,sheet_name):
    if not os.path.exists(case_xlsx):
        print_log(f"[ERROR] {case_xlsx} does not exist!")
        return False
    xlsx_data = read_aclnn_excel(case_xlsx)
    if sheet_name not in xlsx_data.keys():
        print_log(f"[ERROR] {sheet_name} does not exist!")
        return False
    for reslut in xlsx_data[sheet_name]["datas"]:
        input_shapes = eval(reslut["input_tensor_shape"])
        if "tensor_input_seed_info" not in reslut.keys():
            continue
        tensor_input_seed_info = eval(reslut["tensor_input_seed_info"]["value"])
        print_log(f"tensor_input_seed_info:{tensor_input_seed_info} len:{len(tensor_input_seed_info)}")
        for input_seed_info in tensor_input_seed_info:
            if len(input_seed_info) == 5:
                tensor_index = input_seed_info[0]
                batch_axis = input_seed_info[1]
                batch_index_list = input_seed_info[2]
                batch_range = input_seed_info[3]
                batch_seed = input_seed_info[4]
                pre_shape_index = input_seed_info[5]
                for index in range(len(input_shapes)):
                    shape_input = input_shapes[index]
                    if tensor_index == index:
                        slice_str = "["
                        print_log(f"tensor_index:{tensor_index},batch_axis:{batch_axis},batch_index_list:{batch_index_list},batch_seed:{batch_seed},pre_shape_index:{pre_shape_index}")
                        batch_seed_shape = []
                        for i in range(len(shape_input)):
                            if i > batch_axis:
                                batch_seed_shape.append(shape_input[i])
                                slice_str += ":,"
                            elif i == batch_axis:
                                batch_index_str = "["
                                for batch_index in batch_index_list:
                                    if batch_index >= shape_input[i]:
                                        raise ValueError(f"[ERROR] batch_index {batch_index} in batch_index_list:{batch_index_list} for shape_input:{shape_input} batch_axis:{batch_axis} is invalid.")
                                    else:
                                        batch_index_str += str(batch_index) + ","
                                batch_index_str = batch_index_str[:-1]
                                batch_index_str += "]"
                                slice_str += str(batch_index_str) + ","
                            else:
                                if pre_shape_index[i] != ":" and pre_shape_index[i] >= shape_input[i]:
                                    raise ValueError(f"[ERROR] pre_shape_index {pre_shape_index} for shape_input:{shape_input} index:{i} is invalid.")
                                slice_str += str(pre_shape_index[i]) + ","
                        slice_str = slice_str[:-1]
                        slice_str += "]"
                        print_log(f"{shape_input} {slice_str} = {str(batch_seed_shape)}")

def vailed_output_seed_info_attr(case_xlsx,sheet_name):
    if not os.path.exists(case_xlsx):
        print_log(f"[ERROR] {case_xlsx} does not exist!")
        return False
    xlsx_data = read_aclnn_excel(case_xlsx)
    if sheet_name not in xlsx_data.keys():
        print_log(f"[ERROR] {sheet_name} does not exist!")
        return False
    for reslut in xlsx_data[sheet_name]["datas"]:
        output_shapes = eval(reslut["output_tensor_shape"])
        if "tensor_output_seed_info" not in reslut.keys():
            continue
        tensor_input_seed_info = eval(reslut["tensor_output_seed_info"]["value"])
        print_log(f"tensor_output_seed_info:{tensor_input_seed_info} len:{len(tensor_input_seed_info)}")
        for input_seed_info in tensor_input_seed_info:
            if len(input_seed_info) == 5:
                tensor_index = input_seed_info[0]
                batch_axis = input_seed_info[1]
                batch_index_list = input_seed_info[2]
                output_seed = input_seed_info[3]
                pre_shape_index = input_seed_info[4]
                for index in range(len(output_shapes)):
                    shape_input = output_shapes[index]
                    if tensor_index == index:
                        slice_str = "["
                        print_log(f"tensor_index:{tensor_index},batch_axis:{batch_axis},batch_index_list:{batch_index_list},output_seed:{output_seed},pre_shape_index:{pre_shape_index}")
                        batch_seed_shape = []
                        for i in range(len(shape_input)):
                            if i > batch_axis:
                                batch_seed_shape.append(shape_input[i])
                                slice_str += ":,"
                            elif i == batch_axis:
                                batch_index_str = "["
                                for batch_index in batch_index_list:
                                    if batch_index >= shape_input[i]:
                                        raise ValueError(f"[ERROR] batch_index {batch_index} in batch_index_list:{batch_index_list} for shape_input:{shape_input} batch_axis:{batch_axis} is invalid.")
                                    else:
                                        batch_index_str += str(batch_index) + ","
                                batch_index_str = batch_index_str[:-1]
                                batch_index_str += "]"
                                slice_str += str(batch_index_str) + ","
                            else:
                                if pre_shape_index[i] != ":" and pre_shape_index[i] >= shape_input[i]:
                                    raise ValueError(f"[ERROR] pre_shape_index {pre_shape_index} for shape_input:{shape_input} index:{i} is invalid.")
                                slice_str += str(pre_shape_index[i]) + ","
                        slice_str = slice_str[:-1]
                        slice_str += "]"
                        print_log(f"{shape_input} {slice_str} = {str(batch_seed_shape)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=None, help='result file path')
    parser.add_argument('--bin_path', type=str, default=None, help='result bin path')
    parser.add_argument('--gen_input', type=str, default="true", help='re gen input info')
    parser.add_argument('--case_file', type=str, default=None, help='check case file')
    parser.add_argument('--check_type', type=str, default="input", help='check check type')
    parser.add_argument('--sheet_name', type=str, default=None, help='check case file sheet_name')
    args = parser.parse_args()
    result_file = args.result_file
    bin_path = args.bin_path
    gen_input = args.gen_input
    case_file = args.case_file
    sheet_name = args.sheet_name
    check_type = args.check_type

    if result_file is not None and bin_path is not None:
        check_batch_bin_by_result(result_file,bin_path=bin_path)
        if gen_input.lower() == "true":
            create_input_info(result_file)
    if case_file is not None and sheet_name is not None:
        if check_type == "input":
            vailed_input_seed_info_attr(case_file,sheet_name)
        else:
            vailed_output_seed_info_attr(case_file,sheet_name)




