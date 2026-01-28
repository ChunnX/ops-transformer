#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import pandas as pd
import numpy 

# ===================== 【配置区 必改】 =====================
# 你的原Excel文件路径（绝对路径/相对路径都可以）
INPUT_EXCEL_PATH = "./excel/QLI_L0_new.xlsx"
INPUT_EXCEL_SHEETNAME = "INT8"
# 生成的新Excel文件保存路径
OUTPUT_EXCEL_PATH = "./excel/qli_rdv_A3.xlsx"

# ===================== 【固定配置 无需修改】 =====================
# 原Excel的所有列名
ORIGIN_COLUMNS = [
    "Testcase_Name", "custorm_info", "B", "Q_S", "ORI_KV_S", "cmp_ratio", "CMP_KV_S", "D",
    "q_layout", "k_layout", "out_shape", "out_dtype", "metadata_shape", "metadata_dtype",
    "return_value", "sparse_value_shape", "sparse_value_dtype", "q_shape", "q_dtype", "q_datarange",
    "k_shape", "k_dtype", "k_datarange", "k_cache_shape", "weights_shape", "weights_datarange",
    "q_scale_shape", "actual_seq_lengths_q", "actual_seq_lengths_k", "block_table_shape",
    "block_table_dtype", "selected_count", "score_scale", "sparse_mode", "block_size",
    "query_quant_mode", "key_quant_mode", "pre_tokens", "next_tokens", "numHeads", "numKeyValueHeads",
    "max_seqlen_q", "max_seqlen_k", "bp_flag"
]

# 新Excel需要生成的所有列名
TARGET_COLUMNS = [
    "Testcase_Name", "batch_size", "q_seq", "k_seq", "q_t_size", "k_t_size",
    "q_head_num", "k_head_num", "head_dim", "block_size", "block_num", "qk_dtype",
    "dequant_dtype", "actual_seq_dtype", "act_seq_q", "act_seq_k", "query_quant_mode",
    "key_quant_mode", "layout_query", "layout_key", "sparse_count", "sparse_mode",
    "query_datarange", "key_datarange" ,"weights_datarange","q_scale_datarange","k_scale_datarange","cmp_ratio"
]

def excel_data_transform():
    """核心：读取原Excel → 逐行转换 → 生成新Excel"""
    try:
        # 1. 读取原Excel表格数据
        print(f"正在读取原Excel文件：{INPUT_EXCEL_PATH}")
        df_origin = pd.read_excel(INPUT_EXCEL_PATH, sheet_name = INPUT_EXCEL_SHEETNAME)
        print(f"读取成功！共读取到 {len(df_origin)} 行测试用例数据")
        
        # 2. 初始化新Excel的空数据容器（按目标列名创建）
        df_target = pd.DataFrame(columns=TARGET_COLUMNS)

        # 3. 逐行读取原数据，循环处理每一行
        for index, row in enumerate(df_origin.itertuples(index=False)):
            # 【当前行的原始数据】row.列名 即可获取任意字段的值，示例：
            # row.Testcase_Name  → 当前行的用例名称
            # row.B              → 当前行的B值
            # row.q_layout      → 当前行的查询布局
            # 以此类推，所有原列名都可以用 row.列名 获取
            
            # ========== 初始化当前行的目标数据 ==========
            target_row = {}

            # 入参的转换
            target_row["Testcase_Name"] = row.Testcase_Name  
            target_row["block_size"] = row.block_size        
            target_row["query_quant_mode"] = row.query_quant_mode  
            target_row["key_quant_mode"] = row.key_quant_mode     
            target_row["layout_query"] = row.q_layout        
            target_row["layout_key"] = row.k_layout          
            target_row["cmp_ratio"] = row.cmp_ratio          
            target_row["sparse_mode"] = row.sparse_mode      
            target_row["act_seq_q"] = row.actual_seq_lengths_q  
            target_row["act_seq_k"] = row.actual_seq_lengths_k  
            target_row["batch_size"] = row.B   
            target_row["q_seq"] = row.Q_S        
            target_row["k_seq"] = row.CMP_KV_S 
            #字符串转整数列表
            q_shape_list = [int(x.strip()) for x in row.q_shape.split(',')]      
            target_row["q_t_size"] = q_shape_list[0]     
            target_row["k_t_size"] = q_shape_list[1]    
            target_row["q_head_num"] = row.numHeads   
            target_row["k_head_num"] = row.numKeyValueHeads   
            target_row["head_dim"] = row.D  
            block_table_shape_list = [int(x.strip()) for x in row.block_table_shape.split(',')]
            target_row["block_num"] = block_table_shape_list[0]*block_table_shape_list[1]
            target_row["qk_dtype"] = row.q_dtype     
            target_row["dequant_dtype"] = row.q_scale_dtype
            target_row["actual_seq_dtype"] = row.out_dtype
            target_row["sparse_count"] = row.selected_count
            target_row['query_datarange'] = row.q_datarange
            target_row['key_datarange'] = row.k_datarange
            target_row['weights_datarange'] = row.weights_datarange
            target_row['q_scale_datarange'] = row.q_scale_datarange
            target_row['k_scale_datarange'] = row.k_scale_datarange

            # ========== 【结束】自定义转换逻辑 ==========

            # 将当前行处理完的数据添加到新表中
            df_target.loc[index] = target_row

        # 4. 将处理完成的所有数据写入新Excel
        print(f"\n正在生成新Excel文件：{OUTPUT_EXCEL_PATH}")
        df_target.to_excel(OUTPUT_EXCEL_PATH, index=False, engine="openpyxl")
        print(f"生成成功！共生成 {len(df_target)} 行数据")

    except FileNotFoundError:
        print(f"错误：找不到原Excel文件，请检查路径是否正确 → {INPUT_EXCEL_PATH}")
    except Exception as e:
        print(f"程序执行异常：{str(e)}")

# 程序入口
if __name__ == "__main__":
    excel_data_transform()