import re
import pandas as pd
import json
import os
import argparse
import sys

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


class CaseItem:
    def __init__(self, file_name, sheet_name, df):
        self.file_name = file_name
        self.sheet_name = sheet_name
        self.df = df

    def __str__(self):
        return "sheet_name: %s" % self.sheet_name


class Utils:
    @classmethod
    def read_json(cls, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    @classmethod
    def get_files(cls, ipt_path, pattern):
        file_list = []
        for root, dirs, files in os.walk(ipt_path):
            for file in files:
                if re.search(pattern, file):
                    file_list.append(os.path.join(root, file))
        file_list.sort()
        return file_list

    @classmethod
    def write_xlsx(cls, case_df, file_path, sheet_name="level0", mode="w"):
        case_df.drop_duplicates(subset=['case_name'], keep='first', inplace=True)
        if not os.path.exists(file_path):
            case_df.to_excel(file_path, sheet_name=sheet_name, index=False, encoding="utf-8-sig")
        else:
            ori_excel_data = pd.read_excel(file_path, sheet_name=None)
            sheets = list(ori_excel_data.keys())
            if sheet_name in sheets and mode == "a":
                print("sheet: %s already exist, append case and remove repeat case" % sheet_name)
                combined_df = pd.concat([ori_excel_data[sheet_name], case_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['case_name'], keep='first', inplace=True)
                ori_excel_data[sheet_name] = combined_df
            else:
                ori_excel_data[sheet_name] = case_df
            with pd.ExcelWriter(file_path) as writer:
                for sheet, df in ori_excel_data.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)
        print("save excel to %s : %s" % (file_path, sheet_name))


def read_excel(file_path, sheet_name=None):
    print("====read excel %s====" % file_path)
    xls = pd.read_excel(file_path, sheet_name=sheet_name)
    xls_list = []
    for sheet_name, df in xls.items():
        base_file_name = os.path.basename(file_path)
        case_item = CaseItem(base_file_name, sheet_name, df)
        print(case_item)
        xls_list.append(case_item)
    print("="*30 + "\n")
    return xls_list


def diff_xlsx(case_item):
    file_name = case_item.file_name
    sheet_name = case_item.sheet_name
    df = case_item.df
    op_name = None
    new_sheet_name = ""
    if file_name == "mc2_910B_allgather_reducescatter_case.xlsx":
        new_sheet_name = "level0"
        if sheet_name == "allgather":
            op_name = "aclnnAllGatherMatmul"
        elif sheet_name == "reducescatter":
            op_name = "aclnnMatmulReduceScatter"

    elif file_name == "mc2_train_fanhua.xlsx":
        new_sheet_name = "level1"
        if sheet_name == "allgather":
            op_name = "aclnnAllGatherMatmul"
        elif sheet_name == "reducescatter":
            op_name = "aclnnMatmulReduceScatter"

    elif file_name in ["A16W8_allreduce.xlsx",]:
        op_name = "aclnnWeightQuantMatmulAllReduce"
        df['quant_mode'] = "a16w8"
        if sheet_name in ["A16W8_allreduce-单算子", "A16W8_allreduce-图模式"]:
            new_sheet_name = "level0"
        elif sheet_name in ["A16W8_allreduce-泛化", ]:
            new_sheet_name = "level1"

    elif file_name in ["BF16_A8W8_allreduce.xlsx", "FP16-A8W8_allreduce.xlsx"]:
        op_name = "aclnnQuantMatmulAllReduce"
        if sheet_name in ["A8W8_BF16-单算子", "A8W8_BF16-图模式", "A8W8-单算子", "A8W8-入图"]:
            new_sheet_name = "level0"
        elif sheet_name in ["A8W8_BF16-泛化", "A8W8-泛化"]:
            new_sheet_name = "level1"

    elif file_name == "INT8_hcom_A8W8_allreduce_case.xlsx":
        op_name = "aclnnQuantMatmulAllReduceV3"
        if sheet_name in ["门槛用例", "allreduce_a8w8",]:
            new_sheet_name = "level0"
        elif sheet_name in ["泛化shape",]:
            new_sheet_name = "level1"

    elif file_name == "Not_Quant_allreduce.xlsx":
        op_name = "aclnnMatmulAllReduce"
        if sheet_name in ["三维-客户shape", "二维-客户shape", "PTA入图-小艺38B"]:
            new_sheet_name = "level0"

    elif file_name == "PERTOKEN_A8W8_torch.xlsx":
        op_name = "aclnnQuantMatmulAllReduceV2"
        if sheet_name in ["单算子", "图模式",]:
            new_sheet_name = "level0"
        elif sheet_name in ["泛化", ]:
            new_sheet_name = "level1"

    elif file_name in ["With_X3_case.xlsx", "weightnz_a8w8_a16w8.xlsx"]:
        new_sheet_name = "level0" if file_name == "With_X3_case.xlsx" else "level1"
        if sheet_name == "allgather":
            op_name = "aclnnMatmulAllReduce"
        elif sheet_name == "allreduce_a8w8":
            op_name = "aclnnQuantMatmulAllReduce"
        elif sheet_name == "allreduce_a16w8":
            op_name = "aclnnWeightQuantMatmulAllReduce"
            df['quant_mode'] = "a16w8"

    return op_name, new_sheet_name


def write_new_xlsx(case_item_list):
    for case_item in case_item_list:
        op_name, new_sheet_name = diff_xlsx(case_item)
        if op_name and new_sheet_name:
            df = case_item.df
            case_file = os.path.join(CUR_DIR, "excel", f"{op_name}.xlsx")
            Utils.write_xlsx(df, case_file, sheet_name=new_sheet_name, mode="a")


if __name__ == '__main__':
    ipt_path = sys.argv[1]
    if not os.path.exists("excel"):
        os.mkdir("excel")
    else:
        os.system("rm -f excel/*")
    xlsx_files = Utils.get_files(ipt_path, ".xlsx")
    # 读取excel表格
    all_case_item_list = []
    for xlsx_file in xlsx_files:
        xlsx_data = read_excel(xlsx_file)
        all_case_item_list.extend(xlsx_data)
    # excel表格分类汇总
    write_new_xlsx(all_case_item_list)
