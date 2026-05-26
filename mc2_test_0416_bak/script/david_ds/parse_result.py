import os
import pandas as pd
import csv
import sys


def readfile(file):
    a = []
    for line in open(file):
        case_name = line.strip("\n").split(",")[0].strip("'")
        a.append(case_name)
    return a


def write_csv(file, data):
    with open(file, mode="w", encoding='utf-8') as f:
        write = csv.writer(f)
        write.writerows(data)


if __name__ == '__main__':
    result_file = sys.argv[1]
    result_list = []
    case_name_list = []
    fail_case_name_list = []
    allcase_list = readfile("case.txt")
    # 读取原始结果表格
    df = pd.read_csv("csv_result/moe_distribute.csv")
    for index, row in df.iterrows():
        case_name, param, precision = row
        if case_name not in case_name_list:
            print(case_name)
            case_name_list.append(case_name)
        if "FAIL" in precision and "ep_recv_counts" not in param:
            if case_name not in fail_case_name_list:
                print("precision_fail: %s" % case_name)
                fail_case_name_list.append(case_name)

    # 统计未写入结果用例
    fail_function_list = []
    for case_name in allcase_list:
        if case_name not in case_name_list:
            print("function_fail: %s" % case_name)
            fail_function_list.append(case_name)
    fail_case_name_list.extend(fail_function_list)

    for case_name in allcase_list:
        if case_name in fail_case_name_list:
            result_list.append(["MC2", case_name, '', 99, "Failed", "None"])
        else:
            result_list.append(["MC2", case_name, '', 100, "Pass", "None"])

    write_csv(result_file, result_list)
